"""High level ElevenLabs Text-to-Speech client with retries and logging."""

from __future__ import annotations

import logging
import random
import time
from typing import Iterable, Optional, Tuple, Type, Union

import elevenlabs as elevenlabs_pkg
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
from spurgeon.services.tts.utils import normalise_language_code

logger = logging.getLogger(__name__)


class TTSClientError(Exception):
    """Algemene fout voor ElevenLabs TTS‑client."""

    pass


def _clamp_speed(value: float) -> float:
    """Clamp ElevenLabs voice speed to the supported 0.7–1.2 range."""

    if value < 0.7:
        logger.warning("ElevenLabs voice speed %.3f < 0.7; clamping to 0.7", value)
        return 0.7
    if value > 1.2:
        logger.warning("ElevenLabs voice speed %.3f > 1.2; clamping to 1.2", value)
        return 1.2
    return value


class ElevenLabsTTSClient:
    """Wrapper rond de ElevenLabs SDK om tekst naar audio te converteren."""

    def __init__(self, settings: Settings, client: ElevenLabs | None = None) -> None:
        # Vereiste settings
        self.settings = settings
        self.api_key: str = settings.elevenlabs_api_key
        self.voice_id: str = settings.elevenlabs_voice_id
        self.model_id: str = settings.elevenlabs_model_id
        self.output_format: str = settings.elevenlabs_output_format
        self.voice_speed: float = _clamp_speed(settings.elevenlabs_voice_speed)
        self.max_retries: int = settings.elevenlabs_max_retries
        self.backoff: float = settings.elevenlabs_retry_backoff
        self.language_code = settings.elevenlabs_language_code
        self.seed = settings.elevenlabs_dialogue_seed
        self.enable_v3 = settings.elevenlabs_enable_v3
        self.streaming_enabled = settings.elevenlabs_streaming

        if not all([self.api_key, self.voice_id, self.model_id, self.output_format]):
            raise ValueError("Ontbrekende ElevenLabs‑configuratie in settings")

        self.client = client or ElevenLabs(api_key=self.api_key)

        logger.info(
            (
                "ElevenLabsTTSClient init: sdk=%s voice=%s model=%s format=%s "
                "speed=%.2f lang=%s enable_v3=%s"
            ),
            getattr(elevenlabs_pkg, "__version__", "unknown"),
            self.voice_id,
            self.model_id,
            self.output_format,
            self.voice_speed,
            self.language_code,
            self.enable_v3,
        )

    # ------------------------------------------------------------------ #
    # Publieke API
    # ------------------------------------------------------------------ #
    def synthesize(
        self,
        text: str,
        *,
        output_format: Optional[str] = None,
        language_code: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> bytes:
        """Genereer audio‑bytes uit tekst, met retries.

        Args:
            text: De in te spreken string.
            output_format: Optionele override van het audioformaat.
            language_code: Optionele override voor taalcode.
            seed: Optioneel seed voor deterministische uitvoer.

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

        target_format = output_format or self.output_format
        target_language = normalise_language_code(
            language_code or self.language_code or "en",
            self.model_id,
        )
        if target_language:
            logger.debug("Gebruik language_code=%s voor model=%s", target_language, self.model_id)
        request_seed = seed if seed is not None else self.seed

        last_exc: Optional[Exception] = None
        catch_errors: Tuple[Type[Exception], ...] = (ElevenLabsError,) + _transport_errors

        for attempt in range(1, self.max_retries + 2):  # max_retries pogingen + 1
            try:
                voice_settings = self._build_voice_settings()
                response: Union[bytes, bytearray, Tuple[bytes, ...]] = (
                    self.client.text_to_speech.convert(
                        text=text,
                        voice_id=self.voice_id,
                        model_id=self.model_id,
                        output_format=target_format,
                        voice_settings=voice_settings,
                        language_code=target_language,
                        seed=request_seed,
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

    def synthesize_streaming(self, text: str) -> Iterable[bytes]:
        """Stream audio-chunks indien streaming is geactiveerd."""

        if not self.streaming_enabled:
            raise RuntimeError("Streaming is niet ingeschakeld in de settings")
        if not text or not text.strip():
            raise ValueError("Lege tekst doorgegeven aan streaming TTS")

        target_language = normalise_language_code(
            self.language_code or "en",
            self.model_id,
        )
        if target_language:
            logger.debug(
                "Gebruik language_code=%s voor streaming model=%s",
                target_language,
                self.model_id,
            )

        stream = self.client.text_to_speech.stream(
            text=text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_format=self.output_format,
            voice_settings=self._build_voice_settings(),
            language_code=target_language,
            seed=self.seed,
        )

        for chunk in stream:
            if chunk:
                yield chunk

    def _build_voice_settings(self) -> dict[str, float | bool]:
        voice_settings: dict[str, float | bool] = {"speed": self.voice_speed}
        optional_map = {
            "elevenlabs_stability": "stability",
            "elevenlabs_similarity": "similarity_boost",
            "elevenlabs_style": "style",
            "elevenlabs_speaker_boost": "use_speaker_boost",
        }
        for attr, param in optional_map.items():
            value = getattr(self.settings, attr, None)
            if value is not None:
                voice_settings[param] = value  # type: ignore[assignment]
        return voice_settings