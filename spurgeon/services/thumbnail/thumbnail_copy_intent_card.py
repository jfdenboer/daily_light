"""Step 2B copy-intent-card model and service for thumbnail pipeline."""

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
from .thumbnail_reading_diagnoser import ThumbnailReadingDiagnosis
from .thumbnail_source_reader import ThumbnailSourceRead

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=8)
def load_thumbnail_copy_intent_card_prompt(version: str = "v1") -> str:
    prompt_path = PROMPTS_DIR / f"thumbnail_copy_intent_card.{version}.txt"
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValueError(f"Unknown thumbnail copy intent-card prompt version: {version}") from exc


class ThumbnailCopyIntentCard(BaseModel):
    """Copy-side reasoning card for later phrase generation and selection."""

    core_felt_state: str
    best_hook_axis: str
    preferred_hook_mode: str
    copy_priority: str
    emotional_open_loop: str
    allowed_energy: str
    forbidden_drifts: list[str]
    winner_should_feel_like: str

    @field_validator(
        "core_felt_state",
        "best_hook_axis",
        "preferred_hook_mode",
        "copy_priority",
        "emotional_open_loop",
        "allowed_energy",
        "winner_should_feel_like",
        mode="before",
    )
    @classmethod
    def _normalize_text_field(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("Field must be a string")
        normalized = " ".join(value.split()).strip()
        if not normalized:
            raise ValueError("Field must not be empty")
        return normalized

    @field_validator("forbidden_drifts", mode="before")
    @classmethod
    def _normalize_forbidden_drifts(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            raise TypeError("forbidden_drifts must be a list")

        normalized_items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("forbidden_drifts items must be strings")
            normalized = " ".join(item.split()).strip()
            if normalized:
                normalized_items.append(normalized)
        return normalized_items

    @model_validator(mode="after")
    def _validate_constraints(self) -> "ThumbnailCopyIntentCard":
        if not 4 <= len(self.forbidden_drifts) <= 8:
            raise ValueError("forbidden_drifts must have 4 to 8 items")
        return self


class ThumbnailCopyIntentCardError(RuntimeError):
    """Raised when copy intent-card generation or parsing fails."""


class ThumbnailCopyIntentCardBuilder:
    """Build step-2B copy intent card from reading + source + diagnosis."""

    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        self.settings = settings
        self.client = client or OpenAI(api_key=settings.openai_api_key)
        self.model = settings.thumbnail_copy_intent_card_model
        self.temperature = settings.thumbnail_copy_intent_card_temperature
        self.prompt_version = settings.thumbnail_prompt_version
        self.max_completion_tokens = 340

    def build(
        self,
        reading: Reading,
        source_read: ThumbnailSourceRead,
        diagnosis: ThumbnailReadingDiagnosis,
    ) -> ThumbnailCopyIntentCard:
        logger.info("thumbnail_pipeline.copy_intent_card.start slug=%s", reading.slug)

        try:
            result = retry_with_backoff(
                func=lambda: self._generate(reading, source_read, diagnosis),
                max_retries=self.settings.thumbnail_max_retries,
                backoff=self.settings.thumbnail_retry_backoff,
                error_types=(OpenAIError, ThumbnailCopyIntentCardError),
                context=f"thumbnail_copy_intent_card_{reading.slug}",
            )
            logger.info("thumbnail_pipeline.copy_intent_card.success slug=%s", reading.slug)
            return result
        except (OpenAIError, ThumbnailCopyIntentCardError) as exc:
            logger.warning(
                "thumbnail_pipeline.copy_intent_card.fallback slug=%s error_type=%s message=%s",
                reading.slug,
                type(exc).__name__,
                exc,
            )
            return self._fallback()

    def _generate(
        self,
        reading: Reading,
        source_read: ThumbnailSourceRead,
        diagnosis: ThumbnailReadingDiagnosis,
    ) -> ThumbnailCopyIntentCard:
        cleaned_reading = normalize_clip_reading_text(reading.text, max_chars=3500)
        user_message = (
            f"Devotional type: {reading.reading_type.value}\n"
            "Reading text:\n"
            f"{cleaned_reading}\n\n"
            "Step-0 source representation:\n"
            f"{source_read.model_dump_json(indent=2)}\n\n"
            "Step-1 reading diagnosis:\n"
            f"{diagnosis.model_dump_json(indent=2)}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
                user=reading.slug,
                messages=[
                    {
                        "role": "system",
                        "content": load_thumbnail_copy_intent_card_prompt(self.prompt_version),
                    },
                    {"role": "user", "content": user_message},
                ],
            )
        except OpenAIError as exc:
            raise ThumbnailCopyIntentCardError(
                f"OpenAI thumbnail copy intent-card call failed: {getattr(exc, 'message', exc)}"
            ) from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise ThumbnailCopyIntentCardError(
                "Received empty choices in thumbnail copy intent-card response"
            )

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        normalized = self._normalize_message_content(content)
        if not normalized:
            raise ThumbnailCopyIntentCardError("Received empty thumbnail copy intent-card output")

        try:
            payload = json.loads(normalized)
            return ThumbnailCopyIntentCard.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.warning(
                "thumbnail_pipeline.copy_intent_card.parse_error slug=%s error_type=%s raw_output=%r",
                reading.slug,
                type(exc).__name__,
                normalized,
            )
            raise ThumbnailCopyIntentCardError(str(exc)) from exc

    def _fallback(self) -> ThumbnailCopyIntentCard:
        return ThumbnailCopyIntentCard(
            core_felt_state="human need under pressure",
            best_hook_axis="felt-state over summary",
            preferred_hook_mode="felt-state",
            copy_priority="human immediacy first",
            emotional_open_loop="emotion remains unresolved",
            allowed_energy="restrained",
            forbidden_drifts=[
                "imperative devotional language",
                "fatigue-only phrasing",
                "collapse melodrama",
                "churchy slogan tone",
            ],
            winner_should_feel_like="a human hook, not a sermon line",
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
    "ThumbnailCopyIntentCard",
    "ThumbnailCopyIntentCardBuilder",
    "ThumbnailCopyIntentCardError",
    "load_thumbnail_copy_intent_card_prompt",
]
