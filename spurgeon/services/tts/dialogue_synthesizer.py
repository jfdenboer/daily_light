"""Support for ElevenLabs Text-to-Dialogue (multi-voice) synthesis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from elevenlabs.client import ElevenLabs

from spurgeon.config.settings import Settings
from spurgeon.services.tts.utils import normalise_language_code

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DialogueTurn:
    text: str
    voice_id: str


class DialogueSynthesizer:
    """Facade rond de ElevenLabs Text-to-Dialogue API."""

    def __init__(self, settings: Settings, client: ElevenLabs | None = None) -> None:
        self.settings = settings
        if settings.elevenlabs_model_id != "eleven_v3":
            raise ValueError("Text-to-Dialogue vereist model_id 'eleven_v3'")

        self.client = client or ElevenLabs(api_key=settings.elevenlabs_api_key)
        self.default_format = settings.elevenlabs_output_format
        self.default_seed = settings.elevenlabs_dialogue_seed
        self.language_code = settings.elevenlabs_language_code

        logger.info(
            "DialogueSynthesizer klaar: format=%s seed=%s",
            self.default_format,
            self.default_seed,
        )

    def synthesize_dialogue(
        self,
        turns: list[DialogueTurn],
        *,
        output_format: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> tuple[bytes, dict[str, Any]]:
        if not turns:
            raise ValueError("Er zijn geen dialogue turns aangeleverd")

        payload = []
        total_chars = 0
        for idx, turn in enumerate(turns, start=1):
            if not turn.text.strip():
                raise ValueError(f"Dialogue turn {idx} bevat geen tekst")
            if not turn.voice_id:
                raise ValueError(f"Dialogue turn {idx} ontbreekt voice_id")
            payload.append({"text": turn.text, "voice_id": turn.voice_id})
            total_chars += len(turn.text)

        fmt = output_format or self.default_format
        resolved_seed = seed if seed is not None else self.default_seed
        language = normalise_language_code(self.language_code or "en", "eleven_v3")

        logger.info(
            "Start dialogue synthese: turns=%d chars=%d format=%s seed=%s",
            len(turns),
            total_chars,
            fmt,
            resolved_seed,
        )

        response = self.client.text_to_dialogue.convert(
            inputs=payload,
            output_format=fmt,
            seed=resolved_seed,
            language_code=language,
        )

        audio_bytes = self._to_bytes(response)
        metadata = {
            "seed": resolved_seed,
            "turns": len(turns),
            "chars": total_chars,
            "format": fmt,
        }
        return audio_bytes, metadata

    @staticmethod
    def _to_bytes(response: Any) -> bytes:
        if isinstance(response, (bytes, bytearray)):
            return bytes(response)
        if isinstance(response, str):
            return response.encode()
        if isinstance(response, Iterable):
            return b"".join(chunk for chunk in response if chunk)
        raise TypeError(f"Unexpected dialogue response type: {type(response)!r}")