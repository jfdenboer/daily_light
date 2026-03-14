"""Step 4 independent critic pass for thumbnail pipeline v3."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from spurgeon.config.settings import Settings
from spurgeon.models import Reading
from spurgeon.utils.retry_utils import retry_with_backoff

from .thumbnail_bucketed_candidate_generator import ThumbnailCandidateBuckets
from .thumbnail_copy_intent_card import ThumbnailCopyIntentCard
from .thumbnail_text_utils import normalize_clip_reading_text
from .thumbnail_reading_diagnoser import ThumbnailReadingDiagnosis
from .thumbnail_source_reader import ThumbnailSourceRead

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
BUCKET_ORDER = ("felt_state", "posture", "threshold", "edge_tension")


@lru_cache(maxsize=8)
def load_thumbnail_candidate_critic_prompt(version: str = "v1") -> str:
    prompt_path = PROMPTS_DIR / f"thumbnail_candidate_critic.{version}.txt"
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValueError(f"Unknown thumbnail candidate-critic prompt version: {version}") from exc


class ThumbnailCandidateCriticAssessment(BaseModel):
    """Per-candidate diagnostic assessment from the independent critic."""

    candidate: str
    bucket: Literal["felt_state", "posture", "threshold", "edge_tension"]
    keep: bool
    strength: str
    risk_flags: list[str] = Field(default_factory=list)
    critic_note: str
    balance_score: Literal["strong", "good", "fragile", "misleading", "weak"]

    @field_validator("balance_score", mode="before")
    @classmethod
    def _normalize_balance_score(cls, value: object) -> str:
        allowed = {"strong", "good", "fragile", "misleading", "weak"}
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in allowed:
                return normalized

            match = re.search(r"\d+(?:\.\d+)?", normalized)
            if match:
                numeric = float(match.group(0))
                if numeric >= 8.5:
                    return "strong"
                if numeric >= 7.5:
                    return "good"
                if numeric >= 6.5:
                    return "fragile"
                if numeric >= 5.5:
                    return "misleading"
                return "weak"

        raise ValueError(
            "balance_score must be one of: strong, good, fragile, misleading, weak"
        )

    @field_validator("candidate", "strength", "critic_note", mode="before")
    @classmethod
    def _normalize_text_field(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("Field must be a string")
        normalized = " ".join(value.split()).strip()
        if not normalized:
            raise ValueError("Field must not be empty")
        return normalized

    @field_validator("risk_flags", mode="before")
    @classmethod
    def _normalize_risk_flags(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            raise TypeError("risk_flags must be a list")

        normalized_items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("risk_flags items must be strings")
            normalized = " ".join(item.split()).strip()
            if normalized:
                normalized_items.append(normalized)
        return normalized_items

    @model_validator(mode="after")
    def _validate_risk_flags_limit(self) -> "ThumbnailCandidateCriticAssessment":
        if len(self.risk_flags) > 4:
            raise ValueError("risk_flags must contain 0 to 4 items")
        return self


class ThumbnailCandidateCriticSummary(BaseModel):
    """Aggregate critic summary for downstream selector stages."""

    best_balance_candidates: list[str]
    misleading_strong_candidates: list[str]
    dominant_failure_risks: list[str]
    selection_advice: str

    @field_validator(
        "best_balance_candidates",
        "misleading_strong_candidates",
        "dominant_failure_risks",
        mode="before",
    )
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

    @field_validator("selection_advice", mode="before")
    @classmethod
    def _normalize_selection_advice(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("selection_advice must be a string")
        normalized = " ".join(value.split()).strip()
        if not normalized:
            raise ValueError("selection_advice must not be empty")
        return normalized

    @model_validator(mode="after")
    def _validate_lengths(self) -> "ThumbnailCandidateCriticSummary":
        if not 2 <= len(self.best_balance_candidates) <= 4:
            raise ValueError("best_balance_candidates must have 2 to 4 items")
        if len(self.misleading_strong_candidates) > 3:
            raise ValueError("misleading_strong_candidates must have 0 to 3 items")
        if not 2 <= len(self.dominant_failure_risks) <= 5:
            raise ValueError("dominant_failure_risks must have 2 to 5 items")
        return self


class ThumbnailCandidateCriticResult(BaseModel):
    """Typed output for step-4 independent critic."""

    assessments: list[ThumbnailCandidateCriticAssessment]
    summary: ThumbnailCandidateCriticSummary


class ThumbnailCandidateCriticError(RuntimeError):
    """Raised when candidate critic generation or validation fails."""


class ThumbnailCandidateCritic:
    """Evaluate bucketed candidates with a diagnostic critic pass."""

    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        self.settings = settings
        self.client = client or OpenAI(api_key=settings.openai_api_key)
        self.model = settings.thumbnail_candidate_critic_model
        self.temperature = settings.thumbnail_candidate_critic_temperature
        self.prompt_version = settings.thumbnail_prompt_version
        # Reasoning models spend part of the completion budget on hidden reasoning.
        # Keep this high enough so the model can still emit the required JSON payload.
        self.max_completion_tokens = 2000

    def critique(
        self,
        reading: Reading,
        source_read: ThumbnailSourceRead,
        diagnosis: ThumbnailReadingDiagnosis,
        copy_intent_card: ThumbnailCopyIntentCard,
        candidate_buckets: ThumbnailCandidateBuckets,
    ) -> ThumbnailCandidateCriticResult:
        logger.info("thumbnail_pipeline.candidate_critic.start slug=%s", reading.slug)

        try:
            result = retry_with_backoff(
                func=lambda: self._generate(
                    reading,
                    source_read,
                    diagnosis,
                    copy_intent_card,
                    candidate_buckets,
                ),
                max_retries=self.settings.thumbnail_max_retries,
                backoff=self.settings.thumbnail_retry_backoff,
                error_types=(OpenAIError, ThumbnailCandidateCriticError),
                context=f"thumbnail_candidate_critic_{reading.slug}",
            )
            logger.info("thumbnail_pipeline.candidate_critic.success slug=%s", reading.slug)
            return result
        except (OpenAIError, ThumbnailCandidateCriticError) as exc:
            logger.warning(
                "thumbnail_pipeline.candidate_critic.fallback slug=%s error_type=%s message=%s",
                reading.slug,
                type(exc).__name__,
                exc,
            )
            return self._fallback(candidate_buckets)

    def _generate(
        self,
        reading: Reading,
        source_read: ThumbnailSourceRead,
        diagnosis: ThumbnailReadingDiagnosis,
        copy_intent_card: ThumbnailCopyIntentCard,
        candidate_buckets: ThumbnailCandidateBuckets,
    ) -> ThumbnailCandidateCriticResult:
        cleaned_reading = normalize_clip_reading_text(reading.text, max_chars=3500)
        flattened_candidates = self._flatten_candidates(candidate_buckets)
        user_message = (
            f"Devotional type: {reading.reading_type.value}\n"
            "Reading text:\n"
            f"{cleaned_reading}\n\n"
            "Step-0 source representation:\n"
            f"{source_read.model_dump_json(indent=2)}\n\n"
            "Step-1 reading diagnosis:\n"
            f"{diagnosis.model_dump_json(indent=2)}\n\n"
            "Step-2B copy intent card:\n"
            f"{copy_intent_card.model_dump_json(indent=2)}\n\n"
            "Step-3 bucketed candidates:\n"
            f"{candidate_buckets.model_dump_json(indent=2)}\n\n"
            "Flattened candidates with bucket mapping (evaluate each exactly once):\n"
            f"{json.dumps(flattened_candidates, indent=2)}"
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
                        "content": load_thumbnail_candidate_critic_prompt(self.prompt_version),
                    },
                    {"role": "user", "content": user_message},
                ],
            )
        except OpenAIError as exc:
            raise ThumbnailCandidateCriticError(
                f"OpenAI thumbnail candidate-critic call failed: {getattr(exc, 'message', exc)}"
            ) from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise ThumbnailCandidateCriticError(
                "Received empty choices in thumbnail candidate-critic response"
            )

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        normalized = self._normalize_message_content(content)
        if not normalized:
            raise ThumbnailCandidateCriticError("Received empty thumbnail candidate-critic output")

        try:
            payload = json.loads(normalized)
            result = ThumbnailCandidateCriticResult.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.warning(
                "thumbnail_pipeline.candidate_critic.parse_error slug=%s error_type=%s raw_output=%r",
                reading.slug,
                type(exc).__name__,
                normalized,
            )
            raise ThumbnailCandidateCriticError(str(exc)) from exc

        self._validate_structure(candidate_buckets, result, slug=reading.slug)
        return result

    def _validate_structure(
        self,
        candidate_buckets: ThumbnailCandidateBuckets,
        result: ThumbnailCandidateCriticResult,
        *,
        slug: str,
    ) -> None:
        expected_pairs = self._flatten_candidates(candidate_buckets)
        expected_by_candidate = {item["candidate"]: item["bucket"] for item in expected_pairs}
        expected_candidates = set(expected_by_candidate)

        seen_candidates: set[str] = set()
        for assessment in result.assessments:
            if assessment.candidate in seen_candidates:
                logger.warning(
                    "thumbnail_pipeline.candidate_critic.structural_mismatch slug=%s reason=duplicate_output_candidate candidate=%r",
                    slug,
                    assessment.candidate,
                )
                raise ThumbnailCandidateCriticError("Duplicate candidate in critic assessments")
            seen_candidates.add(assessment.candidate)

            if assessment.candidate not in expected_candidates:
                logger.warning(
                    "thumbnail_pipeline.candidate_critic.structural_mismatch slug=%s reason=unknown_output_candidate candidate=%r",
                    slug,
                    assessment.candidate,
                )
                raise ThumbnailCandidateCriticError("Unknown candidate in critic assessments")

            expected_bucket = expected_by_candidate[assessment.candidate]
            if assessment.bucket != expected_bucket:
                logger.warning(
                    "thumbnail_pipeline.candidate_critic.structural_mismatch slug=%s reason=bucket_mismatch candidate=%r expected_bucket=%s actual_bucket=%s",
                    slug,
                    assessment.candidate,
                    expected_bucket,
                    assessment.bucket,
                )
                raise ThumbnailCandidateCriticError("Critic output bucket does not match input bucket")

        if seen_candidates != expected_candidates:
            missing = sorted(expected_candidates - seen_candidates)
            logger.warning(
                "thumbnail_pipeline.candidate_critic.structural_mismatch slug=%s reason=missing_candidates missing=%s",
                slug,
                missing,
            )
            raise ThumbnailCandidateCriticError("Critic output must contain every input candidate exactly once")

        summary_candidate_fields = (
            result.summary.best_balance_candidates,
            result.summary.misleading_strong_candidates,
        )
        for candidate_list in summary_candidate_fields:
            for candidate in candidate_list:
                if candidate not in expected_candidates:
                    logger.warning(
                        "thumbnail_pipeline.candidate_critic.structural_mismatch slug=%s reason=unknown_summary_candidate candidate=%r",
                        slug,
                        candidate,
                    )
                    raise ThumbnailCandidateCriticError(
                        "Critic summary references unknown candidate"
                    )

    def _fallback(
        self,
        candidate_buckets: ThumbnailCandidateBuckets,
    ) -> ThumbnailCandidateCriticResult:
        logger.info("thumbnail_pipeline.candidate_critic.fallback_default")

        assessments: list[ThumbnailCandidateCriticAssessment] = []
        for item in self._flatten_candidates(candidate_buckets):
            candidate = item["candidate"]
            is_valid = bool(candidate.strip())
            assessments.append(
                ThumbnailCandidateCriticAssessment(
                    candidate=candidate,
                    bucket=item["bucket"],
                    keep=is_valid,
                    strength="usable hook",
                    risk_flags=[],
                    critic_note="fallback assessment",
                    balance_score="fragile",
                )
            )

        valid_candidates = [entry["candidate"] for entry in self._flatten_candidates(candidate_buckets) if entry["candidate"].strip()]
        best_balance = valid_candidates[:3]
        while len(best_balance) < 2:
            best_balance.append(valid_candidates[0] if valid_candidates else "fallback candidate")

        summary = ThumbnailCandidateCriticSummary(
            best_balance_candidates=best_balance,
            misleading_strong_candidates=[],
            dominant_failure_risks=["selection_uncertain", "critic_fallback_used"],
            selection_advice="prefer the cleanest human hook",
        )
        return ThumbnailCandidateCriticResult(assessments=assessments, summary=summary)

    @staticmethod
    def _flatten_candidates(candidate_buckets: ThumbnailCandidateBuckets) -> list[dict[str, str]]:
        flat: list[dict[str, str]] = []
        for bucket_name in BUCKET_ORDER:
            for candidate in getattr(candidate_buckets, bucket_name):
                flat.append({"bucket": bucket_name, "candidate": candidate})
        return flat

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
    "ThumbnailCandidateCritic",
    "ThumbnailCandidateCriticAssessment",
    "ThumbnailCandidateCriticError",
    "ThumbnailCandidateCriticResult",
    "ThumbnailCandidateCriticSummary",
    "load_thumbnail_candidate_critic_prompt",
]
