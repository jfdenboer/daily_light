"""Step 0 source-read model and service for thumbnail pipeline."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from spurgeon.config.settings import Settings
from spurgeon.models import Reading
from spurgeon.utils.retry_utils import retry_with_backoff

from .thumbnail_text_utils import normalize_clip_reading_text

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=8)
def load_thumbnail_source_read_prompt(version: str = "v1") -> str:
    prompt_path = PROMPTS_DIR / f"thumbnail_source_read.{version}.txt"
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValueError(f"Unknown thumbnail source-read prompt version: {version}") from exc


class ThumbnailSourceRead(BaseModel):
    """Compact analytical source representation for later thumbnail steps."""

    primary_tension: str
    secondary_tension: str
    emotional_trajectory: str
    human_posture: str
    dominant_movement: str
    hook_zones: list[str]
    imageable_moments: list[str]
    not_about: list[str]

    @field_validator(
        "primary_tension",
        "secondary_tension",
        "emotional_trajectory",
        "human_posture",
        "dominant_movement",
        mode="before",
    )
    @classmethod
    def _normalize_text_field(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("Field must be a string")
        normalized = " ".join(value.split()).strip()
        return normalized

    @field_validator("hook_zones", "imageable_moments", "not_about", mode="before")
    @classmethod
    def _normalize_list_field(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            raise TypeError("Field must be a list")

        normalized_items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("List items must be strings")
            normalized = " ".join(item.split()).strip()
            if normalized:
                normalized_items.append(normalized)
        return normalized_items

    @model_validator(mode="after")
    def _validate_list_lengths(self) -> "ThumbnailSourceRead":
        if not 3 <= len(self.hook_zones) <= 6:
            raise ValueError("hook_zones must have 3 to 6 items")
        if not 1 <= len(self.imageable_moments) <= 3:
            raise ValueError("imageable_moments must have 1 to 3 items")
        if not 2 <= len(self.not_about) <= 4:
            raise ValueError("not_about must have 2 to 4 items")
        return self


class ThumbnailSourceReadError(RuntimeError):
    """Raised when source-read generation or parsing fails."""


class ThumbnailSourceReader:
    """Read devotional source text into a compact analytical representation."""

    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        self.settings = settings
        self.client = client or OpenAI(api_key=settings.openai_api_key)
        self.model = settings.thumbnail_source_read_model
        self.temperature = settings.thumbnail_source_read_temperature
        self.prompt_version = settings.thumbnail_prompt_version
        self.max_completion_tokens = 280

    def read(self, reading: Reading) -> ThumbnailSourceRead:
        logger.info("thumbnail_pipeline.source_read.start slug=%s", reading.slug)

        try:
            result = retry_with_backoff(
                func=lambda: self._generate(reading),
                max_retries=self.settings.thumbnail_max_retries,
                backoff=self.settings.thumbnail_retry_backoff,
                error_types=(OpenAIError, ThumbnailSourceReadError),
                context=f"thumbnail_source_read_{reading.slug}",
            )
            logger.info("thumbnail_pipeline.source_read.success slug=%s", reading.slug)
            return result
        except (OpenAIError, ThumbnailSourceReadError) as exc:
            logger.warning(
                "thumbnail_pipeline.source_read.fallback slug=%s error_type=%s message=%s",
                reading.slug,
                type(exc).__name__,
                exc,
            )
            return self._fallback(reading)

    def _generate(self, reading: Reading) -> ThumbnailSourceRead:
        cleaned_reading = normalize_clip_reading_text(reading.text, max_chars=3500)
        user_message = (
            f"Devotional type: {reading.reading_type.value}\n"
            "Reading text:\n"
            f"{cleaned_reading}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
                user=reading.slug,
                messages=[
                    {"role": "system", "content": load_thumbnail_source_read_prompt(self.prompt_version)},
                    {"role": "user", "content": user_message},
                ],
            )
        except OpenAIError as exc:
            raise ThumbnailSourceReadError(
                f"OpenAI thumbnail source-read call failed: {getattr(exc, 'message', exc)}"
            ) from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise ThumbnailSourceReadError("Received empty choices in thumbnail source-read response")

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        normalized = self._normalize_message_content(content)
        if not normalized:
            raise ThumbnailSourceReadError("Received empty thumbnail source-read output")

        try:
            payload = json.loads(normalized)
            return ThumbnailSourceRead.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.warning(
                "thumbnail_pipeline.source_read.parse_error slug=%s error_type=%s raw_output=%r",
                reading.slug,
                type(exc).__name__,
                normalized,
            )
            raise ThumbnailSourceReadError(str(exc)) from exc

    def _fallback(self, reading: Reading) -> ThumbnailSourceRead:
        reading_phase = "morning beginning" if reading.reading_type.value == "Morning" else "evening reflection"
        return ThumbnailSourceRead(
            primary_tension="unclear inner strain",
            secondary_tension="limited contextual signal",
            emotional_trajectory="unease to seeking",
            human_posture="attentive waiting",
            dominant_movement=f"from pressure toward {reading_phase}",
            hook_zones=["strain", "waiting", "turning point"],
            imageable_moments=["a quiet pause before response"],
            not_about=["not spectacle", "not instant resolution"],
        )

    @staticmethod
    def _normalize_message_content(content: object) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_fragments: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        text_fragments.append(text.strip())
                    continue

                text = getattr(block, "text", None)
                if isinstance(text, str) and text.strip():
                    text_fragments.append(text.strip())

            return "\n".join(text_fragments).strip()

        return ""


__all__ = [
    "ThumbnailSourceRead",
    "ThumbnailSourceReadError",
    "ThumbnailSourceReader",
    "load_thumbnail_source_read_prompt",
]
