"""Step 3 bucketed candidate generation for thumbnail pipeline v3."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from spurgeon.config.settings import Settings
from spurgeon.models import Reading
from spurgeon.utils.retry_utils import retry_with_backoff

from .thumbnail_copy_intent_card import ThumbnailCopyIntentCard
from .thumbnail_text_utils import normalize_clip_reading_text
from .thumbnail_reading_diagnoser import ThumbnailReadingDiagnosis
from .thumbnail_source_reader import ThumbnailSourceRead

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=8)
def load_thumbnail_candidate_buckets_prompt(version: str = "v1") -> str:
    prompt_path = PROMPTS_DIR / f"thumbnail_candidate_buckets.{version}.txt"
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValueError(f"Unknown thumbnail candidate-buckets prompt version: {version}") from exc


class ThumbnailCandidateBuckets(BaseModel):
    """Step-3 bucketed candidate set for thumbnail text generation."""

    felt_state: list[str] = Field(default_factory=list)
    posture: list[str] = Field(default_factory=list)
    threshold: list[str] = Field(default_factory=list)
    edge_tension: list[str] = Field(default_factory=list)

    @field_validator("felt_state", "posture", "threshold", "edge_tension", mode="before")
    @classmethod
    def _normalize_bucket(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            raise TypeError("Bucket must be a list")

        normalized_items: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise TypeError("Bucket items must be strings")
            normalized = " ".join(item.split()).strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized_items.append(normalized)
        return normalized_items

    @model_validator(mode="after")
    def _validate_bucket_sizes(self) -> "ThumbnailCandidateBuckets":
        for bucket_name in ("felt_state", "posture", "threshold", "edge_tension"):
            bucket = getattr(self, bucket_name)
            if len(bucket) > 4:
                raise ValueError(f"{bucket_name} must have at most 4 items")
        return self


class ThumbnailBucketedCandidateGenerationError(RuntimeError):
    """Raised when bucketed candidate generation or parsing fails."""


class ThumbnailBucketedCandidateGenerator:
    """Generate and clean bucketed thumbnail copy candidates for v3 step 3."""

    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        self.settings = settings
        self.client = client or OpenAI(api_key=settings.openai_api_key)
        self.model = settings.thumbnail_candidate_buckets_model
        self.temperature = settings.thumbnail_candidate_buckets_temperature
        self.prompt_version = settings.thumbnail_prompt_version
        self.max_completion_tokens = 340

    def generate(
        self,
        reading: Reading,
        source_read: ThumbnailSourceRead,
        diagnosis: ThumbnailReadingDiagnosis,
        copy_intent_card: ThumbnailCopyIntentCard,
    ) -> ThumbnailCandidateBuckets:
        logger.info("thumbnail_pipeline.candidate_buckets.start slug=%s", reading.slug)

        try:
            result = retry_with_backoff(
                func=lambda: self._generate(reading, source_read, diagnosis, copy_intent_card),
                max_retries=self.settings.thumbnail_max_retries,
                backoff=self.settings.thumbnail_retry_backoff,
                error_types=(OpenAIError, ThumbnailBucketedCandidateGenerationError),
                context=f"thumbnail_candidate_buckets_{reading.slug}",
            )
            logger.info("thumbnail_pipeline.candidate_buckets.success slug=%s", reading.slug)
            return result
        except (OpenAIError, ThumbnailBucketedCandidateGenerationError) as exc:
            logger.warning(
                "thumbnail_pipeline.candidate_buckets.fallback slug=%s error_type=%s message=%s",
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
        copy_intent_card: ThumbnailCopyIntentCard,
    ) -> ThumbnailCandidateBuckets:
        cleaned_reading = normalize_clip_reading_text(reading.text, max_chars=3500)
        user_message = (
            f"Devotional type: {reading.reading_type.value}\n"
            "Reading text:\n"
            f"{cleaned_reading}\n\n"
            "Step-0 source representation:\n"
            f"{source_read.model_dump_json(indent=2)}\n\n"
            "Step-1 reading diagnosis:\n"
            f"{diagnosis.model_dump_json(indent=2)}\n\n"
            "Step-2B copy intent card:\n"
            f"{copy_intent_card.model_dump_json(indent=2)}"
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
                        "content": load_thumbnail_candidate_buckets_prompt(self.prompt_version),
                    },
                    {"role": "user", "content": user_message},
                ],
            )
        except OpenAIError as exc:
            raise ThumbnailBucketedCandidateGenerationError(
                f"OpenAI thumbnail candidate-buckets call failed: {getattr(exc, 'message', exc)}"
            ) from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise ThumbnailBucketedCandidateGenerationError(
                "Received empty choices in thumbnail candidate-buckets response"
            )

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        normalized = self._normalize_message_content(content)
        if not normalized:
            raise ThumbnailBucketedCandidateGenerationError(
                "Received empty thumbnail candidate-buckets output"
            )

        try:
            payload = json.loads(normalized)
            buckets = ThumbnailCandidateBuckets.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.warning(
                "thumbnail_pipeline.candidate_buckets.parse_error slug=%s error_type=%s raw_output=%r",
                reading.slug,
                type(exc).__name__,
                normalized,
            )
            raise ThumbnailBucketedCandidateGenerationError(str(exc)) from exc

        return self._cleanup_buckets(buckets, slug=reading.slug)

    def _cleanup_buckets(
        self,
        buckets: ThumbnailCandidateBuckets,
        *,
        slug: str,
    ) -> ThumbnailCandidateBuckets:
        bucket_order = ["felt_state", "posture", "threshold", "edge_tension"]
        seen_global: set[str] = set()
        cleaned_payload: dict[str, list[str]] = {name: [] for name in bucket_order}
        removed_empty = 0
        removed_duplicate = 0

        for bucket_name in bucket_order:
            for candidate in getattr(buckets, bucket_name):
                normalized = " ".join(candidate.split()).strip()
                if not normalized:
                    removed_empty += 1
                    continue

                dedupe_key = normalized.lower()
                if dedupe_key in seen_global:
                    removed_duplicate += 1
                    continue

                seen_global.add(dedupe_key)
                cleaned_payload[bucket_name].append(normalized)

        if removed_empty or removed_duplicate:
            logger.info(
                "thumbnail_pipeline.candidate_buckets.cleanup slug=%s removed_empty=%d removed_duplicates=%d",
                slug,
                removed_empty,
                removed_duplicate,
            )

        return ThumbnailCandidateBuckets.model_validate(cleaned_payload)

    def _fallback(self) -> ThumbnailCandidateBuckets:
        logger.info("thumbnail_pipeline.candidate_buckets.fallback_default")
        return ThumbnailCandidateBuckets(
            felt_state=["Near The End"],
            posture=["Still Returning"],
            threshold=["Almost There"],
            edge_tension=["Before It Breaks"],
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
    "ThumbnailCandidateBuckets",
    "ThumbnailBucketedCandidateGenerationError",
    "ThumbnailBucketedCandidateGenerator",
    "load_thumbnail_candidate_buckets_prompt",
]
