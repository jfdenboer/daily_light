"""
SpeechSynthesizer – genereert een MP3‑bestand van een Reading via ElevenLabs.
Ondersteunt atomische writes (.tmp → .mp3) en respecteert nu de
instelbare "voice_speed" (Settings.elevenlabs_voice_speed).
"""

import logging
from pathlib import Path
from typing import Optional

from spurgeon.models import Reading
from spurgeon.config.settings import Settings
from spurgeon.services.tts.elevenlabs_tts_client import ElevenLabsTTSClient

logger = logging.getLogger(__name__)


class SpeechSynthesizer:
    """Genereert een MP3 van een Reading via ElevenLabs TTS.

    Args:
        settings (Settings): Applicatie‑instellingen; leest o.a. `elevenlabs_voice_speed`.
        output_dir (Optional[Path]): Map waar audio‑bestanden weggeschreven worden.

    Methods:
        synthesize(reading, force=False): Genereert (of overschrijft) MP3 voor de Reading.
    """

    def __init__(self, settings: Settings, output_dir: Optional[Path] = None) -> None:
        self.settings = settings
        self.tts_client = ElevenLabsTTSClient(settings)
        self.output_dir = Path(output_dir) if output_dir else Path("output/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "SpeechSynthesizer init: output_dir=%s voice_speed=%.2f",
            self.output_dir,
            self.settings.elevenlabs_voice_speed,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def synthesize(self, reading: Reading, force: bool = False) -> Path:
        """Genereert een MP3‑bestand voor een Reading.

        Slaat over als het bestand al bestaat, tenzij ``force=True``.
        Schrijft atomisch: eerst naar ``.tmp``, daarna rename naar definitief ``.mp3``.

        Args:
            reading: De Reading waarvan audio gemaakt wordt.
            force: Overschrijf bestaand bestand.

        Returns:
            Pad naar de uiteindelijke MP3.

        Raises:
            PermissionError: Indien het audiobestand niet geschreven kan worden.
            RuntimeError: Indien synthese via TTS faalt.
        """
        out_path = self.output_dir / f"{reading.slug}.mp3"
        tmp_path = out_path.with_suffix(".mp3.tmp")

        if out_path.exists() and not force:
            logger.info("Sla synthese over, audio‑bestand bestaat al: %s", out_path)
            return out_path

        logger.info(
            "Start TTS‑synthese voor: %s (%s) – voice_speed=%.2f",
            reading.slug,
            reading.date,
            self.settings.elevenlabs_voice_speed,
        )
        logger.debug("Inputtekst (snippet): %s", reading.text.strip().replace("\n", " ")[:100])

        # --- TTS‑call ------------------------------------------------------ #
        try:
            audio_bytes = self.tts_client.synthesize(reading.text)
        except Exception as ex:  # noqa: BLE001 – bubbelt na logging
            logger.error("TTS‑synthese mislukt voor %s: %s", reading.slug, ex)
            raise

        # --- Atomic write -------------------------------------------------- #
        try:
            tmp_path.write_bytes(audio_bytes)
            tmp_path.replace(out_path)
            size_kb = len(audio_bytes) / 1024
            logger.info("Audio opgeslagen (%.2f kB): %s", size_kb, out_path)
        except PermissionError as pe:
            logger.error("Kan audio‑bestand niet schrijven: %s (%s)", out_path, pe)
            raise
        except Exception as ex:  # noqa: BLE001 – unexpected I/O errors
            logger.error("Onbekende fout bij schrijven audio‑bestand: %s (%s)", out_path, ex)
            raise

        return out_path
