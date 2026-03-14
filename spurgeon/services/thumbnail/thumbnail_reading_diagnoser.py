"""Step 1 reading-diagnosis model and service for thumbnail pipeline."""

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
from .thumbnail_source_reader import ThumbnailSourceRead

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=8)
def load_thumbnail_reading_diagnosis_prompt(version: str = "v1") -> str:
    prompt_path = PROMPTS_DIR / f"thumbnail_reading_diagnosis.{version}.txt"
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValueError(f"Unknown thumbnail reading-diagnosis prompt version: {version}") from exc


class ThumbnailReadingDiagnosis(BaseModel):
    """Compact diagnostic map for downstream thumbnail decision steps."""

    primary_axis: str
    secondary_axis: str
    copy_priority: str
    image_priority: str
    best_hook_mode: str
    wrong_but_tempting: list[str]
    failure_mode_flags: list[str]
    language_corridor: list[str]
    intensity_ceiling: str

    @field_validator(
        "primary_axis",
        "secondary_axis",
        "copy_priority",
        "image_priority",
        "best_hook_mode",
        "intensity_ceiling",
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

    @field_validator("wrong_but_tempting", "failure_mode_flags", "language_corridor", mode="before")
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
    def _validate_list_lengths(self) -> "ThumbnailReadingDiagnosis":
        if not 3 <= len(self.wrong_but_tempting) <= 5:
            raise ValueError("wrong_but_tempting must have 3 to 5 items")
        if not 2 <= len(self.failure_mode_flags) <= 6:
            raise ValueError("failure_mode_flags must have 2 to 6 items")
        if not 3 <= len(self.language_corridor) <= 6:
            raise ValueError("language_corridor must have 3 to 6 items")
        return self


class ThumbnailReadingDiagnosisError(RuntimeError):
    """Raised when reading diagnosis generation or parsing fails."""


class ThumbnailReadingDiagnoser:
    """Convert step-0 source signals + reading text into a typed diagnosis object."""

    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        self.settings = settings
        self.client = client or OpenAI(api_key=settings.openai_api_key)
        self.model = settings.thumbnail_reading_diagnosis_model
        self.temperature = settings.thumbnail_reading_diagnosis_temperature
        self.prompt_version = settings.thumbnail_prompt_version
        self.max_completion_tokens = 320

    def diagnose(self, reading: Reading, source_read: ThumbnailSourceRead) -> ThumbnailReadingDiagnosis:
        logger.info("thumbnail_pipeline.reading_diagnosis.start slug=%s", reading.slug)

        try:
            result = retry_with_backoff(
                func=lambda: self._generate(reading, source_read),
                max_retries=self.settings.thumbnail_max_retries,
                backoff=self.settings.thumbnail_retry_backoff,
                error_types=(OpenAIError, ThumbnailReadingDiagnosisError),
                context=f"thumbnail_reading_diagnosis_{reading.slug}",
            )
            logger.info("thumbnail_pipeline.reading_diagnosis.success slug=%s", reading.slug)
            return result
        except (OpenAIError, ThumbnailReadingDiagnosisError) as exc:
            logger.warning(
                "thumbnail_pipeline.reading_diagnosis.fallback slug=%s error_type=%s message=%s",
                reading.slug,
                type(exc).__name__,
                exc,
            )
            return self._fallback()

    def _generate(self, reading: Reading, source_read: ThumbnailSourceRead) -> ThumbnailReadingDiagnosis:
        cleaned_reading = normalize_clip_reading_text(reading.text, max_chars=3500)
        source_payload = source_read.model_dump_json(indent=2)
        user_message = (
            f"Devotional type: {reading.reading_type.value}\n"
            "Reading text:\n"
            f"{cleaned_reading}\n\n"
            "Step-0 source representation:\n"
            f"{source_payload}"
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
                        "content": load_thumbnail_reading_diagnosis_prompt(self.prompt_version),
                    },
                    {"role": "user", "content": user_message},
                ],
            )
        except OpenAIError as exc:
            raise ThumbnailReadingDiagnosisError(
                f"OpenAI thumbnail reading-diagnosis call failed: {getattr(exc, 'message', exc)}"
            ) from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise ThumbnailReadingDiagnosisError("Received empty choices in thumbnail reading-diagnosis response")

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        normalized = self._normalize_message_content(content)
        if not normalized:
            raise ThumbnailReadingDiagnosisError("Received empty thumbnail reading-diagnosis output")

        try:
            payload = json.loads(normalized)
            return ThumbnailReadingDiagnosis.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.warning(
                "thumbnail_pipeline.reading_diagnosis.parse_error slug=%s error_type=%s raw_output=%r",
                reading.slug,
                type(exc).__name__,
                normalized,
            )
            raise ThumbnailReadingDiagnosisError(str(exc)) from exc

    def _fallback(self) -> ThumbnailReadingDiagnosis:
        return ThumbnailReadingDiagnosis(
            primary_axis="human need under pressure",
            secondary_axis="emotion remains unresolved",
            copy_priority="felt-state over summary",
            image_priority="single human focal point",
            best_hook_mode="felt-state",
            wrong_but_tempting=[
                "generic comfort language",
                "sermonic abstraction",
                "collapse melodrama",
            ],
            failure_mode_flags=[
                "summary_over_tension_risk",
                "melodrama_overreach_risk",
                "symbolic_overpull_risk",
            ],
            language_corridor=[
                "human",
                "emotion-first",
                "idiomatic",
                "compact",
            ],
            intensity_ceiling="moderate",
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
    "ThumbnailReadingDiagnosis",
    "ThumbnailReadingDiagnosisError",
    "ThumbnailReadingDiagnoser",
    "load_thumbnail_reading_diagnosis_prompt",
]
