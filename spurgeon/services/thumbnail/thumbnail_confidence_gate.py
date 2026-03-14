"""Step 8 confidence gate / challenger round for thumbnail pipeline v3."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError, field_validator

from spurgeon.config.settings import Settings
from spurgeon.models import Reading
from spurgeon.utils.retry_utils import retry_with_backoff

from .thumbnail_bucketed_candidate_generator import ThumbnailCandidateBuckets
from .thumbnail_candidate_critic import ThumbnailCandidateCriticResult
from .thumbnail_copy_intent_card import ThumbnailCopyIntentCard
from .thumbnail_final_selector import ThumbnailFinalSelection
from .thumbnail_pairwise_tournament import ThumbnailPairwiseTournamentResult
from .thumbnail_reading_diagnoser import ThumbnailReadingDiagnosis
from .thumbnail_source_reader import ThumbnailSourceRead
from .thumbnail_text_utils import normalize_clip_reading_text

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
BUCKET_ORDER = ("felt_state", "posture", "threshold", "edge_tension")


@lru_cache(maxsize=8)
def load_thumbnail_confidence_gate_prompt(version: str = "v1") -> str:
    prompt_path = PROMPTS_DIR / f"thumbnail_confidence_gate.{version}.txt"
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValueError(f"Unknown thumbnail confidence-gate prompt version: {version}") from exc


class ThumbnailConfidenceGateResult(BaseModel):
    """Typed output for the confidence gate / challenger round."""

    winner_status: Literal["stable", "medium", "fragile"]
    confirmed_winner: str
    challenger_candidate: str | None
    challenger_reason: str
    final_recommendation: Literal["keep_winner", "prefer_challenger", "keep_winner_fragile"]
    gate_note: str

    @field_validator(
        "confirmed_winner",
        "challenger_reason",
        "gate_note",
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

    @field_validator("challenger_candidate", mode="before")
    @classmethod
    def _normalize_optional_candidate(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("challenger_candidate must be a string or null")
        normalized = " ".join(value.split()).strip()
        return normalized or None


class ThumbnailConfidenceGateError(RuntimeError):
    """Raised when confidence-gate generation or validation fails."""


class ThumbnailConfidenceGate:
    """Run counterfactual winner stability checks after final selector."""

    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        self.settings = settings
        self.client = client or OpenAI(api_key=settings.openai_api_key)
        self.model = settings.thumbnail_confidence_gate_model
        self.temperature = settings.thumbnail_confidence_gate_temperature
        self.prompt_version = settings.thumbnail_prompt_version
        self.max_completion_tokens = 300

    def evaluate(
        self,
        reading: Reading,
        source_read: ThumbnailSourceRead,
        diagnosis: ThumbnailReadingDiagnosis,
        copy_intent_card: ThumbnailCopyIntentCard,
        candidate_buckets: ThumbnailCandidateBuckets,
        critic_result: ThumbnailCandidateCriticResult,
        pairwise_result: ThumbnailPairwiseTournamentResult,
        final_selection: ThumbnailFinalSelection,
    ) -> ThumbnailConfidenceGateResult:
        logger.info("thumbnail_pipeline.confidence_gate.start slug=%s", reading.slug)
        candidate_pool, pool_context = self._build_candidate_pool(
            candidate_buckets,
            critic_result,
            pairwise_result,
            final_selection,
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
                    final_selection,
                    candidate_pool,
                    pool_context,
                ),
                max_retries=self.settings.thumbnail_max_retries,
                backoff=self.settings.thumbnail_retry_backoff,
                error_types=(OpenAIError, ThumbnailConfidenceGateError),
                context=f"thumbnail_confidence_gate_{reading.slug}",
            )
            logger.info(
                "thumbnail_pipeline.confidence_gate.success slug=%s winner=%r status=%s challenger=%r recommendation=%s",
                reading.slug,
                result.confirmed_winner,
                result.winner_status,
                result.challenger_candidate,
                result.final_recommendation,
            )
            self._log_decision(result, slug=reading.slug)
            return result
        except (OpenAIError, ThumbnailConfidenceGateError) as exc:
            logger.warning(
                "thumbnail_pipeline.confidence_gate.fallback slug=%s error_type=%s message=%s",
                reading.slug,
                type(exc).__name__,
                exc,
            )
            fallback = self._fallback(final_selection, candidate_pool)
            self._log_decision(fallback, slug=reading.slug)
            return fallback

    def _generate(
        self,
        reading: Reading,
        source_read: ThumbnailSourceRead,
        diagnosis: ThumbnailReadingDiagnosis,
        copy_intent_card: ThumbnailCopyIntentCard,
        candidate_buckets: ThumbnailCandidateBuckets,
        critic_result: ThumbnailCandidateCriticResult,
        pairwise_result: ThumbnailPairwiseTournamentResult,
        final_selection: ThumbnailFinalSelection,
        candidate_pool: list[str],
        pool_context: dict[str, list[str]],
    ) -> ThumbnailConfidenceGateResult:
        cleaned_reading = normalize_clip_reading_text(reading.text, max_chars=3500)
        user_message = (
            f"Devotional type: {reading.reading_type.value}\n"
            "Reading text:\n"
            f"{cleaned_reading}\n\n"
            "Step-0 source representation:\n"
            f"{source_read.model_dump_json(indent=2)}\n\n"
            "Step-1 reading diagnosis:\n"
            f"{diagnosis.model_dump_json(indent=2)}\n\n"
            "Step-3 copy intent card:\n"
            f"{copy_intent_card.model_dump_json(indent=2)}\n\n"
            "Step-4 bucketed candidates:\n"
            f"{candidate_buckets.model_dump_json(indent=2)}\n\n"
            "Step-5 critic result:\n"
            f"{critic_result.model_dump_json(indent=2)}\n\n"
            "Step-6 pairwise tournament:\n"
            f"{pairwise_result.model_dump_json(indent=2)}\n\n"
            "Step-7 final selector result:\n"
            f"{final_selection.model_dump_json(indent=2)}\n\n"
            "Canonical challenger pool (challenger_candidate must come from this list):\n"
            f"{json.dumps(candidate_pool, indent=2)}\n\n"
            "Pool context:\n"
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
                        "content": load_thumbnail_confidence_gate_prompt(self.prompt_version),
                    },
                    {"role": "user", "content": user_message},
                ],
            )
        except OpenAIError as exc:
            raise ThumbnailConfidenceGateError(
                f"OpenAI thumbnail confidence-gate call failed: {getattr(exc, 'message', exc)}"
            ) from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise ThumbnailConfidenceGateError("Received empty choices in thumbnail confidence-gate response")

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        normalized = self._normalize_message_content(content)
        if not normalized:
            raise ThumbnailConfidenceGateError("Received empty thumbnail confidence-gate output")

        try:
            payload = json.loads(normalized)
            result = ThumbnailConfidenceGateResult.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.warning(
                "thumbnail_pipeline.confidence_gate.parse_error slug=%s error_type=%s raw_output=%r",
                reading.slug,
                type(exc).__name__,
                normalized,
            )
            raise ThumbnailConfidenceGateError(str(exc)) from exc

        self._validate_structure(
            result,
            expected_winner=final_selection.winner,
            candidate_pool=candidate_pool,
            slug=reading.slug,
        )
        return result

    def _build_candidate_pool(
        self,
        candidate_buckets: ThumbnailCandidateBuckets,
        critic_result: ThumbnailCandidateCriticResult,
        pairwise_result: ThumbnailPairwiseTournamentResult,
        final_selection: ThumbnailFinalSelection,
    ) -> tuple[list[str], dict[str, list[str]]]:
        bucket_candidates = [entry["candidate"] for entry in self._flatten_candidates(candidate_buckets)]
        best_balance = [c for c in critic_result.summary.best_balance_candidates if c in bucket_candidates]
        top_advancers = [c for c in pairwise_result.summary.top_advancers if c in bucket_candidates]
        rejected = [c for c in final_selection.rejected_finalists if c in bucket_candidates]

        ordered_pool: list[str] = []
        for candidate in [
            final_selection.winner,
            *rejected,
            *top_advancers,
            *best_balance,
            *bucket_candidates,
        ]:
            if candidate and candidate not in ordered_pool:
                ordered_pool.append(candidate)

        return ordered_pool, {
            "best_balance_candidates": best_balance,
            "top_advancers": top_advancers,
            "final_selector_rejected": rejected,
            "final_selector_winner": [final_selection.winner],
        }

    def _validate_structure(
        self,
        result: ThumbnailConfidenceGateResult,
        *,
        expected_winner: str,
        candidate_pool: list[str],
        slug: str,
    ) -> None:
        if result.confirmed_winner != expected_winner:
            logger.warning(
                "thumbnail_pipeline.confidence_gate.structural_mismatch slug=%s reason=confirmed_winner_mismatch expected=%r actual=%r",
                slug,
                expected_winner,
                result.confirmed_winner,
            )
            raise ThumbnailConfidenceGateError("confirmed_winner must match final selector winner exactly")

        if result.challenger_candidate is None:
            return

        if result.challenger_candidate not in set(candidate_pool):
            logger.warning(
                "thumbnail_pipeline.confidence_gate.structural_mismatch slug=%s reason=challenger_outside_pool challenger=%r",
                slug,
                result.challenger_candidate,
            )
            raise ThumbnailConfidenceGateError("challenger_candidate must come from candidate pool")

        if result.challenger_candidate == result.confirmed_winner:
            logger.warning(
                "thumbnail_pipeline.confidence_gate.structural_mismatch slug=%s reason=challenger_equals_winner winner=%r",
                slug,
                result.confirmed_winner,
            )
            raise ThumbnailConfidenceGateError("challenger_candidate must differ from confirmed_winner")

    def _fallback(
        self,
        final_selection: ThumbnailFinalSelection,
        candidate_pool: list[str],
    ) -> ThumbnailConfidenceGateResult:
        logger.info("thumbnail_pipeline.confidence_gate.fallback_default")
        fallback_challenger = next(
            (candidate for candidate in final_selection.rejected_finalists if candidate in candidate_pool),
            None,
        )
        return ThumbnailConfidenceGateResult(
            winner_status="fragile",
            confirmed_winner=final_selection.winner,
            challenger_candidate=fallback_challenger,
            challenger_reason="fallback confidence gate",
            final_recommendation="keep_winner_fragile",
            gate_note="fallback used",
        )

    def _log_decision(self, result: ThumbnailConfidenceGateResult, *, slug: str) -> None:
        if result.challenger_candidate:
            logger.info(
                "thumbnail_pipeline.confidence_gate.challenger_identified slug=%s challenger=%r",
                slug,
                result.challenger_candidate,
            )

        if result.winner_status == "fragile":
            logger.info("thumbnail_pipeline.confidence_gate.winner_marked_fragile slug=%s", slug)
        elif result.winner_status == "stable":
            logger.info("thumbnail_pipeline.confidence_gate.winner_marked_stable slug=%s", slug)

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
    "ThumbnailConfidenceGateResult",
    "ThumbnailConfidenceGateError",
    "ThumbnailConfidenceGate",
    "load_thumbnail_confidence_gate_prompt",
]
