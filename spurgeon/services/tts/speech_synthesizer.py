"""Speech synthesis pipeline for ElevenLabs output files."""

from __future__ import annotations

import logging
import hashlib
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from spurgeon.config.settings import Settings
from spurgeon.models import Reading
from spurgeon.services.intro.generate_credit_line import generate_credit_line
from spurgeon.services.intro.generate_spoken_hook import SpokenHookValidationError, generate_spoken_hook
from spurgeon.services.tts.elevenlabs_tts_client import ElevenLabsTTSClient
from spurgeon.services.tts.utils import OutputFormatInfo, chunk_text_for_v3, parse_output_format

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynthesisResult:
    final_audio_path: Path
    narration_audio_path: Path
    intro_duration_seconds: float = 0.0
    intro_status: str = "skipped"


class SpeechSynthesizer:
    """Genereer audio-bestanden voor een ``Reading`` via ElevenLabs TTS."""

    def __init__(self, settings: Settings, output_dir: Optional[Path] = None) -> None:
        self.settings = settings
        self.tts_client = ElevenLabsTTSClient(settings)
        self.output_dir = Path(output_dir) if output_dir else Path("output/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, reading: Reading, force: bool = False) -> Path:
        return self.synthesize_for_video(reading, force=force).final_audio_path

    def synthesize_for_video(self, reading: Reading, force: bool = False) -> SynthesisResult:
        output_extension = self.settings.elevenlabs_audio_extension
        narration_path = self.output_dir / f"{reading.slug}_main{output_extension}"
        final_path = self.output_dir / f"{reading.slug}{output_extension}"

        narration_audio = self._synthesize_reading_audio(reading, narration_path, force=force)
        if not getattr(self.settings, "intro_enabled", True):
            self._copy_file(narration_audio, final_path)
            return SynthesisResult(final_path, narration_audio, 0.0, "disabled")

        try:
            intro_path, intro_duration = self._build_intro_audio(reading, force=force)
            self._concat_audio_files([intro_path, narration_audio], final_path)
            return SynthesisResult(final_path, narration_audio, intro_duration, "full_intro")
        except SpokenHookValidationError as exc:
            strategy = getattr(self.settings, "intro_hook_fail_strategy", "credit_only")
            if strategy == "raise":
                raise
            if strategy == "skip_intro":
                logger.warning(
                    "Hook generation failed for %s; skipping intro (%s)",
                    reading.slug,
                    exc,
                )
                self._copy_file(narration_audio, final_path)
                return SynthesisResult(final_path, narration_audio, 0.0, "hook_failed_skip")

            try:
                intro_path, intro_duration = self._build_credit_only_intro_audio(reading, force=force)
                self._concat_audio_files([intro_path, narration_audio], final_path)
                return SynthesisResult(final_path, narration_audio, intro_duration, "credit_only")
            except Exception as fallback_exc:
                if not getattr(self.settings, "intro_fail_open", True):
                    raise
                logger.warning(
                    "Credit-only intro fallback failed for %s; using narration only (%s)",
                    reading.slug,
                    fallback_exc,
                )
                self._copy_file(narration_audio, final_path)
                return SynthesisResult(final_path, narration_audio, 0.0, "hook_failed_open")
        except Exception as exc:
            if not getattr(self.settings, "intro_fail_open", True):
                raise
            logger.warning("Intro generation failed for %s; fallback to narration (%s)", reading.slug, exc)
            self._copy_file(narration_audio, final_path)
            return SynthesisResult(final_path, narration_audio, 0.0, "intro_failed_open")

    def _copy_file(self, source: Path, destination: Path) -> None:
        if source == destination:
            return
        tmp_path = destination.with_suffix(f"{destination.suffix}.tmp")
        tmp_path.write_bytes(source.read_bytes())
        tmp_path.replace(destination)

    def _synthesize_reading_audio(self, reading: Reading, out_path: Path, *, force: bool) -> Path:
        if out_path.exists() and not force:
            return out_path
        tmp_path = out_path.with_suffix(f"{out_path.suffix}.tmp")
        audio_bytes = self._synthesize_text(reading.text)
        tmp_path.write_bytes(audio_bytes)
        tmp_path.replace(out_path)
        return out_path

    def _build_intro_audio(self, reading: Reading, *, force: bool) -> tuple[Path, float]:
        output_extension = self.settings.elevenlabs_audio_extension
        intro_dir = self.output_dir / "intro"
        intro_dir.mkdir(parents=True, exist_ok=True)

        hook_text = generate_spoken_hook(reading.text, self.settings)
        if not hook_text:
            raise SpokenHookValidationError("empty_hook")
        credit_text = generate_credit_line()

        hook_path = intro_dir / f"{reading.slug}_hook{output_extension}"
        credit_path = intro_dir / f"{reading.slug}_credit{output_extension}"
        pause0_path = intro_dir / f"{reading.slug}_pause0{output_extension}"
        pause1_path = intro_dir / f"{reading.slug}_pause1{output_extension}"
        pause2_path = intro_dir / f"{reading.slug}_pause2{output_extension}"
        intro_path = intro_dir / f"{reading.slug}_intro{output_extension}"

        self._synthesize_plain_text(hook_text, hook_path, force=force)
        self._synthesize_plain_text(credit_text, credit_path, force=force)
        self._generate_silence(
            pause0_path,
            duration_ms=int(getattr(self.settings, "intro_pause_pre_intro_ms", 120)),
            force=force,
        )
        self._generate_silence(
            pause1_path,
            duration_ms=int(getattr(self.settings, "intro_pause_between_ms", 550)),
            force=force,
        )
        self._generate_silence(
            pause2_path,
            duration_ms=int(getattr(self.settings, "intro_pause_after_credit_ms", 550)),
            force=force,
        )
        self._concat_audio_files([pause0_path, hook_path, pause1_path, credit_path, pause2_path], intro_path)
        return intro_path, self._probe_duration_seconds(intro_path)


    def _build_credit_only_intro_audio(self, reading: Reading, *, force: bool) -> tuple[Path, float]:
        output_extension = self.settings.elevenlabs_audio_extension
        intro_dir = self.output_dir / "intro"
        intro_dir.mkdir(parents=True, exist_ok=True)

        credit_text = generate_credit_line()
        credit_path = intro_dir / f"{reading.slug}_credit{output_extension}"
        pause0_path = intro_dir / f"{reading.slug}_pause0{output_extension}"
        pause2_path = intro_dir / f"{reading.slug}_pause2{output_extension}"
        intro_path = intro_dir / f"{reading.slug}_intro_credit_only{output_extension}"

        self._synthesize_plain_text(credit_text, credit_path, force=force)
        self._generate_silence(
            pause0_path,
            duration_ms=int(getattr(self.settings, "intro_pause_pre_intro_ms", 120)),
            force=force,
        )
        self._generate_silence(
            pause2_path,
            duration_ms=int(getattr(self.settings, "intro_pause_after_credit_ms", 550)),
            force=force,
        )
        self._concat_audio_files([pause0_path, credit_path, pause2_path], intro_path)
        return intro_path, self._probe_duration_seconds(intro_path)

    def _synthesize_plain_text(self, text: str, out_path: Path, *, force: bool) -> Path:
        cache_enabled = bool(getattr(self.settings, "intro_cache_enabled", True))
        fingerprint = self._text_fingerprint(text)
        fingerprint_path = self._fingerprint_path(out_path)
        if out_path.exists() and not force and cache_enabled and self._is_cache_hit(fingerprint_path, fingerprint):
            return out_path
        tmp_path = out_path.with_suffix(f"{out_path.suffix}.tmp")
        tmp_path.write_bytes(self._synthesize_text(text))
        tmp_path.replace(out_path)
        self._write_fingerprint(fingerprint_path, fingerprint)
        return out_path

    def _text_fingerprint(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _fingerprint_path(self, out_path: Path) -> Path:
        return out_path.with_suffix(f"{out_path.suffix}.sha256")

    def _is_cache_hit(self, fingerprint_path: Path, expected: str) -> bool:
        if not fingerprint_path.exists():
            return False
        stored = fingerprint_path.read_text(encoding="utf-8").strip()
        return stored == expected

    def _write_fingerprint(self, fingerprint_path: Path, fingerprint: str) -> None:
        tmp_path = fingerprint_path.with_suffix(f"{fingerprint_path.suffix}.tmp")
        tmp_path.write_text(f"{fingerprint}\n", encoding="utf-8")
        tmp_path.replace(fingerprint_path)

    def _generate_silence(self, out_path: Path, *, duration_ms: int, force: bool) -> Path:
        cache_enabled = bool(getattr(self.settings, "intro_cache_enabled", True))
        if out_path.exists() and not force and cache_enabled:
            return out_path
        duration = max(duration_ms, 0) / 1000
        format_info = parse_output_format(self.settings.elevenlabs_output_format)
        cmd = [
            self.settings.ffmpeg_cmd,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={format_info.sample_rate}:cl=mono",
            "-t",
            f"{duration:.3f}",
        ]
        cmd += self._ffmpeg_audio_args(format_info)
        cmd.append(str(out_path))
        self._run_ffmpeg(cmd)
        return out_path

    def _concat_audio_files(self, input_paths: Sequence[Path], out_path: Path) -> Path:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            list_file = tmp_dir / "concat.txt"
            list_file.write_text(
                "\n".join(f"file '{p.resolve().as_posix()}'" for p in input_paths),
                encoding="utf-8",
            )
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
                "-c",
                "copy",
                str(out_path),
            ]
            self._run_ffmpeg(cmd)
        return out_path

    def _probe_duration_seconds(self, audio_path: Path) -> float:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Unable to read duration for {audio_path}: {result.stderr.strip()}")
        return float(result.stdout.strip())

    def _synthesize_text(self, text: str) -> bytes:
        if not text.strip():
            raise ValueError("Lege tekst kan niet gesynthetiseerd worden")

        model_id = self.settings.elevenlabs_model_id
        format_info = parse_output_format(self.settings.elevenlabs_output_format)

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
            return self._synthesize_chunks(chunks, format_info)

        return self.tts_client.synthesize(text)

    def _synthesize_chunks(
        self,
        chunks: list[str],
        final_format: OutputFormatInfo,
    ) -> bytes:
        chunk_format_name = self._determine_chunk_format(final_format)
        chunk_format = parse_output_format(chunk_format_name)

        chunk_audio: list[bytes] = []
        for chunk in chunks:
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
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg exited with status {result.returncode}: {result.stderr.strip()}"
            )
