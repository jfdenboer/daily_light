"""
ElevenLabsTTSClient – wrapper rond de ElevenLabs SDK.
Bevat retry‑logic, geconsolideerde foutafhandeling én ondersteuning voor
instelbare "voice_speed" (0.1 – 4.0) uit Settings.
"""

import logging
import time
import random
from typing import Optional, Tuple, Type, Union

from elevenlabs.client import ElevenLabs  # v2 SDK client

# Compatibele fouttypes uit SDK – afhankelijk van SDK‑versie
try:
    from elevenlabs.api import Error as ElevenLabsError  # >=0.2.40
except ImportError:  # pragma: no cover – oudere SDK
    try:
        from elevenlabs import Error as ElevenLabsError  # <=0.2.36
    except ImportError:  # pragma: no cover – fallback
        ElevenLabsError = Exception  # type: ignore[assignment]

# Transportfouten bij netwerk‑layers (optioneel requests/httpx)
_transport_errors: Tuple[Type[Exception], ...] = ()
try:
    from requests import RequestException  # type: ignore

    _transport_errors = (RequestException,)
except ImportError:  # pragma: no cover – requests niet geïnstalleerd
    pass
try:
    from httpx import HTTPError as HTTPXError  # type: ignore

    _transport_errors += (HTTPXError,)
except ImportError:  # pragma: no cover – httpx niet geïnstalleerd
    pass

from spurgeon.config.settings import Settings

logger = logging.getLogger(__name__)


class TTSClientError(Exception):
    """Algemene fout voor ElevenLabs TTS‑client."""

    pass


class ElevenLabsTTSClient:
    """Wrapper rond de ElevenLabs SDK om tekst naar audio te converteren."""

    def __init__(self, settings: Settings) -> None:  # noqa: D401 – simple description
        # Vereiste settings
        self.api_key: str = settings.elevenlabs_api_key
        self.voice_id: str = settings.elevenlabs_voice_id
        self.model_id: str = settings.elevenlabs_model_id
        self.output_format: str = settings.elevenlabs_output_format
        self.voice_speed: float = settings.elevenlabs_voice_speed  # NEW ⭐️
        self.max_retries: int = settings.elevenlabs_max_retries
        self.backoff: float = settings.elevenlabs_retry_backoff

        if not all(
            [self.api_key, self.voice_id, self.model_id, self.output_format]
        ):
            raise ValueError("Ontbrekende ElevenLabs‑configuratie in settings")

        # Initialiseert één globale SDK‑client
        self.client = ElevenLabs(api_key=self.api_key)

        logger.info(
            "ElevenLabsTTSClient init: voice=%s model=%s format=%s speed=%.2f "
            "retries=%d backoff=%.1fs",
            self.voice_id,
            self.model_id,
            self.output_format,
            self.voice_speed,
            self.max_retries,
            self.backoff,
        )

    # ------------------------------------------------------------------ #
    # Publieke API
    # ------------------------------------------------------------------ #
    def synthesize(self, text: str) -> bytes:
        """Genereer audio‑bytes uit tekst, met retries.

        Args:
            text: De in te spreken string.

        Returns:
            bytes: MP3/WAV‑data afhankelijk van `output_format`.

        Raises:
            RuntimeError: Bij overschrijden van `max_retries`.
            ValueError: Als er een lege tekst wordt aangeleverd.
        """
        if not text or not text.strip():
            raise ValueError(
                "Lege tekst doorgegeven aan ElevenLabsTTSClient.synthesize()"
            )

        snippet = text.strip().replace("\n", " ")[:100]
        logger.debug("Start synthese: '%s...'", snippet)

        last_exc: Optional[Exception] = None
        catch_errors: Tuple[Type[Exception], ...] = (ElevenLabsError,) + _transport_errors

        for attempt in range(1, self.max_retries + 2):  # max_retries pogingen + 1
            try:
                response: Union[bytes, bytearray, Tuple[bytes, ...]] = (
                    self.client.text_to_speech.convert(
                        text=text,
                        voice_id=self.voice_id,
                        model_id=self.model_id,
                        output_format=self.output_format,
                        voice_settings={"speed": self.voice_speed},  # speed 🚀
                    )
                )

                # Response kan bytes of stream/generator zijn
                if isinstance(response, (bytes, bytearray)):
                    audio_bytes: bytes = bytes(response)
                elif hasattr(response, "__iter__"):
                    audio_bytes = b"".join(chunk for chunk in response if chunk)
                else:
                    logger.warning(
                        "Onverwacht response‑type van ElevenLabs: %s", type(response)
                    )
                    audio_bytes = bytes(response)  # type: ignore[arg-type]

                logger.debug(
                    "Synthese geslaagd (%.2f kB) op poging %d",
                    len(audio_bytes) / 1024,
                    attempt,
                )
                return audio_bytes

            except catch_errors as exc:
                last_exc = exc
                logger.warning("TTS poging %d mislukt: %s", attempt, exc)

                if attempt <= self.max_retries:
                    wait = self.backoff * (2 ** (attempt - 1))
                    jitter = random.uniform(0, wait)
                    logger.debug("Wachten %.2fs voor retry %d", jitter, attempt)
                    time.sleep(jitter)

        # Alle pogingen zijn mislukt
        logger.error("Max retries overschreden voor TTS")
        raise RuntimeError("ElevenLabs TTS synthese mislukt") from last_exc
