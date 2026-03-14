"""Step 6 final selector for thumbnail pipeline v3."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from spurgeon.config.settings import Settings
from spurgeon.models import Reading
from spurgeon.utils.retry_utils import retry_with_backoff

from .thumbnail_bucketed_candidate_generator import ThumbnailCandidateBuckets
from .thumbnail_candidate_critic import ThumbnailCandidateCriticResult
from .thumbnail_copy_intent_card import ThumbnailCopyIntentCard
from .thumbnail_text_utils import normalize_clip_reading_text
from .thumbnail_pairwise_tournament import ThumbnailPairwiseTournamentResult
from .thumbnail_reading_diagnoser import ThumbnailReadingDiagnosis
from .thumbnail_source_reader import ThumbnailSourceRead

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
BUCKET_ORDER = ("felt_state", "posture", "threshold", "edge_tension")


@lru_cache(maxsize=8)
def load_thumbnail_final_selector_prompt(version: str = "v1") -> str:
    prompt_path = PROMPTS_DIR / f"thumbnail_final_selector.{version}.txt"
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValueError(f"Unknown thumbnail final-selector prompt version: {version}") from exc


class ThumbnailFinalSelection(BaseModel):
    """Typed output for the canonical v3 final selection step."""

    winner: str
    confidence: Literal["high", "medium", "fragile"]
    selection_basis: str
    rejected_finalists: list[str] = Field(default_factory=list)
    selector_note: str

    @field_validator("winner", "selection_basis", "selector_note", mode="before")
    @classmethod
    def _normalize_text_field(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("Field must be a string")
        normalized = " ".join(value.split()).strip()
        if not normalized:
            raise ValueError("Field must not be empty")
        return normalized

    @field_validator("rejected_finalists", mode="before")
    @classmethod
    def _normalize_rejected_finalists(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            raise TypeError("rejected_finalists must be a list")
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise TypeError("rejected_finalists items must be strings")
            candidate = " ".join(item.split()).strip()
            if not candidate:
                continue
            lowered = candidate.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(candidate)
        return normalized

    @model_validator(mode="after")
    def _validate_rejected_bounds(self) -> "ThumbnailFinalSelection":
        if len(self.rejected_finalists) > 4:
            raise ValueError("rejected_finalists must contain at most 4 items")
        return self


class ThumbnailFinalSelectorError(RuntimeError):
    """Raised when final selector generation or validation fails."""


class ThumbnailFinalSelector:
    """Select one final winner from v3 candidate context and tournament signals."""

    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        self.settings = settings
        self.client = client or OpenAI(api_key=settings.openai_api_key)
        self.model = settings.thumbnail_final_selector_model
        self.temperature = settings.thumbnail_final_selector_temperature
        self.prompt_version = settings.thumbnail_prompt_version
        self.max_completion_tokens = 260

    def select(
        self,
        reading: Reading,
        source_read: ThumbnailSourceRead,
        diagnosis: ThumbnailReadingDiagnosis,
        copy_intent_card: ThumbnailCopyIntentCard,
        candidate_buckets: ThumbnailCandidateBuckets,
        critic_result: ThumbnailCandidateCriticResult,
        pairwise_result: ThumbnailPairwiseTournamentResult,
    ) -> ThumbnailFinalSelection:
        logger.info("thumbnail_pipeline.final_selector.start slug=%s", reading.slug)
        final_pool, pool_context = self._build_final_pool(
            candidate_buckets,
            critic_result,
            pairwise_result,
        )

        self._log_signal_conflicts(
            slug=reading.slug,
            final_pool=final_pool,
            critic_result=critic_result,
            pairwise_result=pairwise_result,
        )

        try:
            result = retry_with_backoff(
                func=lambda: self._generate(
                    reading,
                    source_read,
                    diagnosis,
                    copy_intent_card,
                    candidate_buckets,
                    critic_result,
                    pairwise_result,
                    final_pool,
                    pool_context,
                ),
                max_retries=self.settings.thumbnail_max_retries,
                backoff=self.settings.thumbnail_retry_backoff,
                error_types=(OpenAIError, ThumbnailFinalSelectorError),
                context=f"thumbnail_final_selector_{reading.slug}",
            )
            logger.info("thumbnail_pipeline.final_selector.success slug=%s", reading.slug)
            return result
        except (OpenAIError, ThumbnailFinalSelectorError) as exc:
            logger.warning(
                "thumbnail_pipeline.final_selector.fallback slug=%s error_type=%s message=%s",
                reading.slug,
                type(exc).__name__,
                exc,
            )
            return self._fallback(final_pool, pool_context)

    def _generate(
        self,
        reading: Reading,
        source_read: ThumbnailSourceRead,
        diagnosis: ThumbnailReadingDiagnosis,
        copy_intent_card: ThumbnailCopyIntentCard,
        candidate_buckets: ThumbnailCandidateBuckets,
        critic_result: ThumbnailCandidateCriticResult,
        pairwise_result: ThumbnailPairwiseTournamentResult,
        final_pool: list[str],
        pool_context: dict[str, list[str]],
    ) -> ThumbnailFinalSelection:
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
            f"{copy_intent_card.model_dump_json(indent=2)}\n\n"
            "Step-3 bucketed candidates:\n"
            f"{candidate_buckets.model_dump_json(indent=2)}\n\n"
            "Step-4 critic result:\n"
            f"{critic_result.model_dump_json(indent=2)}\n\n"
            "Step-5 pairwise tournament:\n"
            f"{pairwise_result.model_dump_json(indent=2)}\n\n"
            "Eligible final pool (winner and rejected_finalists must come from this list unless explicitly unusable):\n"
            f"{json.dumps(final_pool, indent=2)}\n\n"
            "Final pool context:\n"
            f"{json.dumps(pool_context, indent=2)}"
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
                        "content": load_thumbnail_final_selector_prompt(self.prompt_version),
                    },
                    {"role": "user", "content": user_message},
                ],
            )
        except OpenAIError as exc:
            raise ThumbnailFinalSelectorError(
                f"OpenAI thumbnail final-selector call failed: {getattr(exc, 'message', exc)}"
            ) from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise ThumbnailFinalSelectorError("Received empty choices in thumbnail final-selector response")

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        normalized = self._normalize_message_content(content)
        if not normalized:
            raise ThumbnailFinalSelectorError("Received empty thumbnail final-selector output")

        try:
            payload = json.loads(normalized)
            result = ThumbnailFinalSelection.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.warning(
                "thumbnail_pipeline.final_selector.parse_error slug=%s error_type=%s raw_output=%r",
                reading.slug,
                type(exc).__name__,
                normalized,
            )
            raise ThumbnailFinalSelectorError(str(exc)) from exc

        self._validate_structure(result, final_pool, slug=reading.slug)
        self._log_signal_conflicts(
            slug=reading.slug,
            final_pool=final_pool,
            critic_result=critic_result,
            pairwise_result=pairwise_result,
            winner=result.winner,
        )
        return result

    def _build_final_pool(
        self,
        candidate_buckets: ThumbnailCandidateBuckets,
        critic_result: ThumbnailCandidateCriticResult,
        pairwise_result: ThumbnailPairwiseTournamentResult,
    ) -> tuple[list[str], dict[str, list[str]]]:
        bucket_candidates = [entry["candidate"] for entry in self._flatten_candidates(candidate_buckets)]
        bucket_set = set(bucket_candidates)

        keep_candidates = [
            item.candidate
            for item in critic_result.assessments
            if item.keep and item.candidate in bucket_set
        ]
        best_balance = [c for c in critic_result.summary.best_balance_candidates if c in bucket_set]
        top_advancers = [c for c in pairwise_result.summary.top_advancers if c in bucket_set]
        misleading = [c for c in critic_result.summary.misleading_strong_candidates if c in bucket_set]

        ordered_pool: list[str] = []
        for candidate in [*best_balance, *top_advancers, *keep_candidates, *misleading]:
            if candidate not in ordered_pool:
                ordered_pool.append(candidate)

        if not ordered_pool:
            ordered_pool = bucket_candidates[:]

        if not ordered_pool:
            ordered_pool = ["Hold On"]

        return ordered_pool, {
            "best_balance_candidates": best_balance,
            "top_advancers": top_advancers,
            "misleading_strong_candidates": misleading,
            "keep_true_candidates": keep_candidates,
        }

    def _validate_structure(
        self,
        result: ThumbnailFinalSelection,
        final_pool: list[str],
        *,
        slug: str,
    ) -> None:
        final_pool_set = set(final_pool)
        fallback_new_phrase = result.winner not in final_pool_set

        if fallback_new_phrase:
            if final_pool and any(self._is_usable(candidate) for candidate in final_pool):
                logger.warning(
                    "thumbnail_pipeline.final_selector.structural_mismatch slug=%s reason=winner_outside_pool winner=%r",
                    slug,
                    result.winner,
                )
                raise ThumbnailFinalSelectorError("winner must come from eligible final pool")

            logger.warning(
                "thumbnail_pipeline.final_selector.fallback_new_phrase slug=%s winner=%r",
                slug,
                result.winner,
            )

        for candidate in result.rejected_finalists:
            if candidate not in final_pool_set:
                logger.warning(
                    "thumbnail_pipeline.final_selector.structural_mismatch slug=%s reason=rejected_outside_pool candidate=%r",
                    slug,
                    candidate,
                )
                raise ThumbnailFinalSelectorError("rejected_finalists must come from eligible final pool")

        if result.winner in result.rejected_finalists:
            logger.warning(
                "thumbnail_pipeline.final_selector.structural_mismatch slug=%s reason=winner_in_rejected winner=%r",
                slug,
                result.winner,
            )
            raise ThumbnailFinalSelectorError("winner must not appear in rejected_finalists")

    def _fallback(
        self,
        final_pool: list[str],
        pool_context: dict[str, list[str]],
    ) -> ThumbnailFinalSelection:
        logger.info("thumbnail_pipeline.final_selector.fallback_default")

        candidate = self._choose_fallback_candidate(final_pool, pool_context)
        if not candidate:
            logger.warning("thumbnail_pipeline.final_selector.fallback_new_phrase_used")
            candidate = "Hold On"

        rejected = [item for item in final_pool if item != candidate][:4]
        return ThumbnailFinalSelection(
            winner=candidate,
            confidence="fragile",
            selection_basis="fallback best available candidate",
            rejected_finalists=rejected,
            selector_note="fallback selection used",
        )

    def _choose_fallback_candidate(
        self,
        final_pool: list[str],
        pool_context: dict[str, list[str]],
    ) -> str:
        candidates_by_priority = [
            pool_context.get("top_advancers", []),
            pool_context.get("best_balance_candidates", []),
            pool_context.get("keep_true_candidates", []),
            final_pool,
        ]
        for candidate_list in candidates_by_priority:
            for candidate in candidate_list:
                if candidate in final_pool and self._is_usable(candidate):
                    return candidate
        return ""

    def _log_signal_conflicts(
        self,
        *,
        slug: str,
        final_pool: list[str],
        critic_result: ThumbnailCandidateCriticResult,
        pairwise_result: ThumbnailPairwiseTournamentResult,
        winner: str | None = None,
    ) -> None:
        critic_best = set(critic_result.summary.best_balance_candidates)
        top_advancers = set(pairwise_result.summary.top_advancers)
        overlap = critic_best & top_advancers
        if critic_best and top_advancers and not overlap:
            logger.warning(
                "thumbnail_pipeline.final_selector.signal_conflict slug=%s reason=critic_pairwise_disagree critic_best=%s top_advancers=%s",
                slug,
                sorted(critic_best),
                sorted(top_advancers),
            )

        misleading = set(critic_result.summary.misleading_strong_candidates)
        if winner and winner in misleading:
            logger.warning(
                "thumbnail_pipeline.final_selector.signal_conflict slug=%s reason=winner_marked_misleading winner=%r",
                slug,
                winner,
            )

        repeat_winners = set(pairwise_result.summary.repeat_winners)
        if winner and repeat_winners and winner not in repeat_winners:
            logger.warning(
                "thumbnail_pipeline.final_selector.signal_conflict slug=%s reason=winner_not_repeat_winner winner=%r repeat_winners=%s",
                slug,
                winner,
                sorted(repeat_winners),
            )

        if winner and winner not in final_pool:
            logger.warning(
                "thumbnail_pipeline.final_selector.signal_conflict slug=%s reason=winner_outside_pool winner=%r",
                slug,
                winner,
            )

    @staticmethod
    def _flatten_candidates(candidate_buckets: ThumbnailCandidateBuckets) -> list[dict[str, str]]:
        flat: list[dict[str, str]] = []
        for bucket_name in BUCKET_ORDER:
            for candidate in getattr(candidate_buckets, bucket_name):
                flat.append({"bucket": bucket_name, "candidate": candidate})
        return flat

    @staticmethod
    def _is_usable(candidate: str) -> bool:
        return bool(" ".join(candidate.split()).strip())

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
    "ThumbnailFinalSelection",
    "ThumbnailFinalSelector",
    "ThumbnailFinalSelectorError",
    "load_thumbnail_final_selector_prompt",
]
