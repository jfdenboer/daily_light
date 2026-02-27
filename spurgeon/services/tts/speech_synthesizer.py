"""Speech synthesis pipeline for ElevenLabs output files."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Sequence

from spurgeon.config.settings import Settings
from spurgeon.models import Reading
from spurgeon.services.tts.elevenlabs_tts_client import ElevenLabsTTSClient
from spurgeon.services.tts.utils import OutputFormatInfo, chunk_text_for_v3, parse_output_format

logger = logging.getLogger(__name__)


class SpeechSynthesizer:
    """Genereer audio-bestanden voor een ``Reading`` via ElevenLabs TTS."""

    def __init__(self, settings: Settings, output_dir: Optional[Path] = None) -> None:
        self.settings = settings
        self.tts_client = ElevenLabsTTSClient(settings)
        self.output_dir = Path(output_dir) if output_dir else Path("output/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "SpeechSynthesizer init: output_dir=%s voice_speed=%.2f format=%s",
            self.output_dir,
            self.settings.elevenlabs_voice_speed,
            self.settings.elevenlabs_output_format,
        )

    def synthesize(self, reading: Reading, force: bool = False) -> Path:
        """Genereer een audiobestand voor de aangeleverde ``Reading``."""

        output_extension = self.settings.elevenlabs_audio_extension
        out_path = self.output_dir / f"{reading.slug}{output_extension}"
        tmp_path = out_path.with_suffix(f"{output_extension}.tmp")

        if out_path.exists() and not force:
            logger.info("Sla synthese over, audio-bestand bestaat al: %s", out_path)
            return out_path

        logger.info(
            "Start TTS-synthese voor: %s (%s) – model=%s format=%s",
            reading.slug,
            reading.date,
            self.settings.elevenlabs_model_id,
            self.settings.elevenlabs_output_format,
        )
        snippet = reading.text.strip().replace("\n", " ")[:100]
        logger.debug("Inputtekst (snippet): %s", snippet)

        audio_bytes = self._synthesize_text(reading.text)

        try:
            tmp_path.write_bytes(audio_bytes)
            tmp_path.replace(out_path)
            size_kb = len(audio_bytes) / 1024
            logger.info("Audio opgeslagen (%.2f kB): %s", size_kb, out_path)
        except PermissionError as pe:  # pragma: no cover - passthrough
            logger.error("Kan audio-bestand niet schrijven: %s (%s)", out_path, pe)
            raise
        except Exception as ex:  # pragma: no cover - unexpected I/O errors
            logger.error("Onbekende fout bij schrijven audio-bestand: %s (%s)", out_path, ex)
            raise

        return out_path

    # ------------------------------------------------------------------ #
    # Interne helpers
    # ------------------------------------------------------------------ #
    def _synthesize_text(self, text: str) -> bytes:
        if not text.strip():
            raise ValueError("Lege tekst kan niet gesynthetiseerd worden")

        model_id = self.settings.elevenlabs_model_id
        format_info = parse_output_format(self.settings.elevenlabs_output_format)
        logger.debug(
            "TTS-config: model=%s format=%s sample_rate=%s",
            model_id,
            self.settings.elevenlabs_output_format,
            format_info.sample_rate,
        )

        if (
            self.settings.elevenlabs_enable_v3
            and model_id == "eleven_v3"
            and len(text) > self.settings.elevenlabs_v3_max_chars
        ):
            target = max(1, self.settings.elevenlabs_v3_max_chars - 200)
            chunks = chunk_text_for_v3(
                text,
                hard_limit=self.settings.elevenlabs_v3_max_chars,
                target=target,
            )
            logger.info(
                "Gebruik chunking voor Eleven v3 (%d chunks, totaal %d karakters)",
                len(chunks),
                len(text),
            )
            return self._synthesize_chunks(chunks, format_info)

        logger.debug(
            "Enkelvoudige synthese: model=%s chars=%d",
            model_id,
            len(text),
        )
        return self.tts_client.synthesize(text)

    def _synthesize_chunks(
        self,
        chunks: list[str],
        final_format: OutputFormatInfo,
    ) -> bytes:
        chunk_format_name = self._determine_chunk_format(final_format)
        chunk_format = parse_output_format(chunk_format_name)
        logger.debug(
            "Chunking met format=%s target_format=%s",
            chunk_format_name,
            self.settings.elevenlabs_output_format,
        )

        chunk_audio: list[bytes] = []
        for idx, chunk in enumerate(chunks, start=1):
            logger.debug(
                "Synthese chunk %d/%d (len=%d)", idx, len(chunks), len(chunk)
            )
            data = self.tts_client.synthesize(
                chunk,
                output_format=chunk_format_name,
            )
            chunk_audio.append(data)

        if len(chunk_audio) == 1 and chunk_format_name == self.settings.elevenlabs_output_format:
            return chunk_audio[0]

        if chunk_format.container == "pcm" and final_format.container != "pcm":
            return self._encode_pcm_chunks(chunk_audio, chunk_format, final_format)

        if chunk_format_name == self.settings.elevenlabs_output_format:
            return self._concat_chunks_ffmpeg(chunk_audio, final_format)

        # Fallback: convert chunk format to final format via FFmpeg
        return self._convert_and_concat(chunk_audio, chunk_format, final_format)

    def _determine_chunk_format(self, final_format: OutputFormatInfo) -> str:
        if (
            final_format.container == "mp3"
            and self.settings.elevenlabs_allow_pcm_chunk_join
        ):
            return "pcm_44100"
        return self.settings.elevenlabs_output_format

    def _concat_chunks_ffmpeg(
        self, chunk_audio: Iterable[bytes], final_format: OutputFormatInfo
    ) -> bytes:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            chunk_paths = []
            for idx, data in enumerate(chunk_audio, start=1):
                chunk_path = tmp_dir / f"chunk_{idx}{self.settings.elevenlabs_audio_extension}"
                chunk_path.write_bytes(data)
                chunk_paths.append(chunk_path)

            list_file = tmp_dir / "concat.txt"
            list_content = "\n".join(
                f"file '{path.as_posix()}'" for path in chunk_paths
            )
            list_file.write_text(list_content, encoding="utf-8")

            output_path = tmp_dir / f"joined{self.settings.elevenlabs_audio_extension}"
            cmd = [
                self.settings.ffmpeg_cmd,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_file),
            ]
            if final_format.container == "mp3":
                cmd += ["-c:a", "libmp3lame"]
                if final_format.bitrate:
                    cmd += ["-b:a", f"{final_format.bitrate}k"]
                cmd += ["-ar", str(final_format.sample_rate)]
            else:
                cmd += ["-c", "copy"]
            cmd.append(str(output_path))

            self._run_ffmpeg(cmd)
            return output_path.read_bytes()

    def _encode_pcm_chunks(
        self,
        chunks: Iterable[bytes],
        chunk_format: OutputFormatInfo,
        final_format: OutputFormatInfo,
    ) -> bytes:
        pcm_bytes = b"".join(chunks)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            raw_path = tmp_dir / "audio.raw"
            raw_path.write_bytes(pcm_bytes)
            output_path = tmp_dir / f"output{self.settings.elevenlabs_audio_extension}"

            cmd = [
                self.settings.ffmpeg_cmd,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "s16le",
                "-ar",
                str(chunk_format.sample_rate),
                "-ac",
                "1",
                "-i",
                str(raw_path),
            ]
            cmd += self._ffmpeg_audio_args(final_format)
            cmd.append(str(output_path))

            self._run_ffmpeg(cmd)
            return output_path.read_bytes()

    def _convert_and_concat(
        self,
        chunks: Iterable[bytes],
        chunk_format: OutputFormatInfo,
        final_format: OutputFormatInfo,
    ) -> bytes:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            converted_paths: list[Path] = []
            for idx, data in enumerate(chunks, start=1):
                raw_path = tmp_dir / f"chunk_{idx}.raw"
                raw_path.write_bytes(data)
                out_path = tmp_dir / f"chunk_{idx}{self.settings.elevenlabs_audio_extension}"

                cmd = [
                    self.settings.ffmpeg_cmd,
                    "-y",
                    "-loglevel",
                    "error",
                ]

                if chunk_format.container == "pcm":
                    cmd += [
                        "-f",
                        "s16le",
                        "-ar",
                        str(chunk_format.sample_rate),
                        "-ac",
                        "1",
                        "-i",
                        str(raw_path),
                    ]
                else:
                    cmd += ["-i", str(raw_path)]

                cmd += self._ffmpeg_audio_args(final_format)
                cmd.append(str(out_path))
                self._run_ffmpeg(cmd)
                converted_paths.append(out_path)

            return self._concat_chunks_ffmpeg(
                [path.read_bytes() for path in converted_paths], final_format
            )

    def _ffmpeg_audio_args(self, fmt: OutputFormatInfo) -> list[str]:
        if fmt.container == "mp3":
            args = ["-c:a", "libmp3lame", "-ar", str(fmt.sample_rate)]
            if fmt.bitrate:
                args += ["-b:a", f"{fmt.bitrate}k"]
            return args
        if fmt.container == "pcm":
            return ["-c:a", "pcm_s16le", "-ar", str(fmt.sample_rate)]
        if fmt.container == "ulaw":
            return ["-c:a", "pcm_mulaw", "-ar", str(fmt.sample_rate)]
        if fmt.container == "alaw":
            return ["-c:a", "pcm_alaw", "-ar", str(fmt.sample_rate)]
        if fmt.container == "opus":
            bitrate = fmt.bitrate or 96
            return [
                "-c:a",
                "libopus",
                "-b:a",
                f"{bitrate}k",
                "-ar",
                str(fmt.sample_rate),
            ]
        return ["-c", "copy"]

    def _run_ffmpeg(self, cmd: Sequence[str]) -> None:
        logger.debug("FFmpeg command: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            logger.error("FFmpeg failure: %s", result.stderr.strip())
            raise RuntimeError(
                f"FFmpeg exited with status {result.returncode}: {result.stderr.strip()}"
            )