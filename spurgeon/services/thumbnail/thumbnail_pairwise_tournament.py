"""Step 5 pairwise comparison tournament for thumbnail pipeline v3."""

from __future__ import annotations

import json
import logging
from collections import Counter
from functools import lru_cache
from pathlib import Path

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from spurgeon.config.settings import Settings
from spurgeon.models import Reading
from spurgeon.utils.retry_utils import retry_with_backoff

from .thumbnail_bucketed_candidate_generator import ThumbnailCandidateBuckets
from .thumbnail_candidate_critic import ThumbnailCandidateCriticResult
from .thumbnail_copy_intent_card import ThumbnailCopyIntentCard
from .thumbnail_text_utils import normalize_clip_reading_text
from .thumbnail_reading_diagnoser import ThumbnailReadingDiagnosis
from .thumbnail_source_reader import ThumbnailSourceRead

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
BUCKET_ORDER = ("felt_state", "posture", "threshold", "edge_tension")


@lru_cache(maxsize=8)
def load_thumbnail_pairwise_tournament_prompt(version: str = "v1") -> str:
    prompt_path = PROMPTS_DIR / f"thumbnail_pairwise_tournament.{version}.txt"
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValueError(f"Unknown thumbnail pairwise-tournament prompt version: {version}") from exc


class ThumbnailPairwiseMatchup(BaseModel):
    """Single duel between two existing thumbnail candidates."""

    left_candidate: str
    right_candidate: str
    comparison_axis: str
    winner: str
    reason: str
    confidence: str

    @field_validator(
        "left_candidate",
        "right_candidate",
        "comparison_axis",
        "winner",
        "reason",
        "confidence",
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


class ThumbnailPairwiseTournamentSummary(BaseModel):
    """Compact tournament summary for downstream selector use."""

    top_advancers: list[str]
    repeat_winners: list[str] = Field(default_factory=list)
    fragile_matchups: list[str] = Field(default_factory=list)
    selector_guidance: str

    @field_validator("top_advancers", "repeat_winners", "fragile_matchups", mode="before")
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

    @field_validator("selector_guidance", mode="before")
    @classmethod
    def _normalize_selector_guidance(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("selector_guidance must be a string")
        normalized = " ".join(value.split()).strip()
        if not normalized:
            raise ValueError("selector_guidance must not be empty")
        return normalized

    @model_validator(mode="after")
    def _validate_lengths(self) -> "ThumbnailPairwiseTournamentSummary":
        if not 2 <= len(self.top_advancers) <= 4:
            raise ValueError("top_advancers must have 2 to 4 items")
        if len(self.fragile_matchups) > 3:
            raise ValueError("fragile_matchups must have 0 to 3 items")
        return self


class ThumbnailPairwiseTournamentResult(BaseModel):
    """Typed output for step-5 pairwise tournament."""

    matchups: list[ThumbnailPairwiseMatchup]
    summary: ThumbnailPairwiseTournamentSummary

    @model_validator(mode="after")
    def _validate_matchup_count(self) -> "ThumbnailPairwiseTournamentResult":
        if not 3 <= len(self.matchups) <= 8:
            raise ValueError("matchups must have 3 to 8 items")
        return self


class ThumbnailPairwiseTournamentError(RuntimeError):
    """Raised when pairwise tournament generation or validation fails."""


class ThumbnailPairwiseTournament:
    """Run a pairwise tournament over eligible v3 thumbnail candidates."""

    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        self.settings = settings
        self.client = client or OpenAI(api_key=settings.openai_api_key)
        self.model = settings.thumbnail_pairwise_tournament_model
        self.temperature = settings.thumbnail_pairwise_tournament_temperature
        self.prompt_version = settings.thumbnail_prompt_version
        # Reasoning models spend part of the completion budget on hidden reasoning.
        # Keep this high enough so the model can still emit the required JSON payload.
        self.max_completion_tokens = 2000

    def run(
        self,
        reading: Reading,
        source_read: ThumbnailSourceRead,
        diagnosis: ThumbnailReadingDiagnosis,
        copy_intent_card: ThumbnailCopyIntentCard,
        candidate_buckets: ThumbnailCandidateBuckets,
        critic_result: ThumbnailCandidateCriticResult,
    ) -> ThumbnailPairwiseTournamentResult:
        logger.info("thumbnail_pipeline.pairwise_tournament.start slug=%s", reading.slug)
        eligible_pool = self._build_eligible_pool(candidate_buckets, critic_result)

        try:
            result = retry_with_backoff(
                func=lambda: self._generate(
                    reading,
                    source_read,
                    diagnosis,
                    copy_intent_card,
                    candidate_buckets,
                    critic_result,
                    eligible_pool,
                ),
                max_retries=self.settings.thumbnail_max_retries,
                backoff=self.settings.thumbnail_retry_backoff,
                error_types=(OpenAIError, ThumbnailPairwiseTournamentError),
                context=f"thumbnail_pairwise_tournament_{reading.slug}",
            )
            self._run_sanity_checks(result, critic_result, slug=reading.slug)
            logger.info("thumbnail_pipeline.pairwise_tournament.success slug=%s", reading.slug)
            return result
        except (OpenAIError, ThumbnailPairwiseTournamentError) as exc:
            logger.warning(
                "thumbnail_pipeline.pairwise_tournament.fallback slug=%s error_type=%s message=%s",
                reading.slug,
                type(exc).__name__,
                exc,
            )
            result = self._fallback(eligible_pool)
            self._run_sanity_checks(result, critic_result, slug=reading.slug)
            return result

    def _generate(
        self,
        reading: Reading,
        source_read: ThumbnailSourceRead,
        diagnosis: ThumbnailReadingDiagnosis,
        copy_intent_card: ThumbnailCopyIntentCard,
        candidate_buckets: ThumbnailCandidateBuckets,
        critic_result: ThumbnailCandidateCriticResult,
        eligible_pool: list[str],
    ) -> ThumbnailPairwiseTournamentResult:
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
            "Eligible tournament pool (use only these exact candidates):\n"
            f"{json.dumps(eligible_pool, indent=2)}"
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
                        "content": load_thumbnail_pairwise_tournament_prompt(self.prompt_version),
                    },
                    {"role": "user", "content": user_message},
                ],
            )
        except OpenAIError as exc:
            raise ThumbnailPairwiseTournamentError(
                f"OpenAI thumbnail pairwise-tournament call failed: {getattr(exc, 'message', exc)}"
            ) from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise ThumbnailPairwiseTournamentError(
                "Received empty choices in thumbnail pairwise-tournament response"
            )

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        normalized = self._normalize_message_content(content)
        if not normalized:
            raise ThumbnailPairwiseTournamentError("Received empty thumbnail pairwise-tournament output")

        try:
            payload = json.loads(normalized)
            result = ThumbnailPairwiseTournamentResult.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            logger.warning(
                "thumbnail_pipeline.pairwise_tournament.parse_error slug=%s error_type=%s raw_output=%r",
                reading.slug,
                type(exc).__name__,
                normalized,
            )
            raise ThumbnailPairwiseTournamentError(str(exc)) from exc

        self._validate_structure(result, eligible_pool, slug=reading.slug)
        return result

    def _build_eligible_pool(
        self,
        candidate_buckets: ThumbnailCandidateBuckets,
        critic_result: ThumbnailCandidateCriticResult,
    ) -> list[str]:
        flat_candidates = self._flatten_candidates(candidate_buckets)
        bucket_candidates = [entry["candidate"] for entry in flat_candidates]

        keep_candidates = [a.candidate for a in critic_result.assessments if a.keep and a.candidate in bucket_candidates]
        best_balance = [
            c
            for c in critic_result.summary.best_balance_candidates
            if c in bucket_candidates and c in keep_candidates
        ]
        misleading = [
            c
            for c in critic_result.summary.misleading_strong_candidates
            if c in bucket_candidates
        ][:2]
        other_keeps = [c for c in keep_candidates if c not in best_balance]

        ordered_pool: list[str] = []
        for candidate in [*best_balance, *misleading, *other_keeps]:
            if candidate not in ordered_pool:
                ordered_pool.append(candidate)

        if len(ordered_pool) < 2:
            for candidate in bucket_candidates:
                if candidate not in ordered_pool:
                    ordered_pool.append(candidate)
                if len(ordered_pool) >= 2:
                    break

        return ordered_pool

    def _validate_structure(
        self,
        result: ThumbnailPairwiseTournamentResult,
        eligible_pool: list[str],
        *,
        slug: str,
    ) -> None:
        eligible_set = set(eligible_pool)

        for matchup in result.matchups:
            if matchup.left_candidate == matchup.right_candidate:
                logger.warning(
                    "thumbnail_pipeline.pairwise_tournament.structural_mismatch slug=%s reason=same_candidate_duel candidate=%r",
                    slug,
                    matchup.left_candidate,
                )
                raise ThumbnailPairwiseTournamentError("Each matchup must compare two different candidates")

            if matchup.left_candidate not in eligible_set or matchup.right_candidate not in eligible_set:
                logger.warning(
                    "thumbnail_pipeline.pairwise_tournament.structural_mismatch slug=%s reason=unknown_matchup_candidate left=%r right=%r",
                    slug,
                    matchup.left_candidate,
                    matchup.right_candidate,
                )
                raise ThumbnailPairwiseTournamentError("Matchup candidate must be in eligible pool")

            if matchup.winner not in {matchup.left_candidate, matchup.right_candidate}:
                logger.warning(
                    "thumbnail_pipeline.pairwise_tournament.structural_mismatch slug=%s reason=invalid_winner winner=%r left=%r right=%r",
                    slug,
                    matchup.winner,
                    matchup.left_candidate,
                    matchup.right_candidate,
                )
                raise ThumbnailPairwiseTournamentError("Winner must be one of the two candidates")

        for field_name, candidates in (
            ("top_advancers", result.summary.top_advancers),
            ("repeat_winners", result.summary.repeat_winners),
        ):
            for candidate in candidates:
                if candidate not in eligible_set:
                    logger.warning(
                        "thumbnail_pipeline.pairwise_tournament.structural_mismatch slug=%s reason=unknown_summary_candidate field=%s candidate=%r",
                        slug,
                        field_name,
                        candidate,
                    )
                    raise ThumbnailPairwiseTournamentError(
                        "Tournament summary references candidate outside eligible pool"
                    )

    def _run_sanity_checks(
        self,
        result: ThumbnailPairwiseTournamentResult,
        critic_result: ThumbnailCandidateCriticResult,
        *,
        slug: str,
    ) -> None:
        if not result.summary.top_advancers:
            logger.warning(
                "thumbnail_pipeline.pairwise_tournament.structural_mismatch slug=%s reason=empty_top_advancers",
                slug,
            )
            raise ThumbnailPairwiseTournamentError("Tournament summary top_advancers is empty")

        participation_counter: Counter[str] = Counter()
        for matchup in result.matchups:
            participation_counter[matchup.left_candidate] += 1
            participation_counter[matchup.right_candidate] += 1

        if participation_counter and len(participation_counter) == 1:
            logger.warning(
                "thumbnail_pipeline.pairwise_tournament.sanity_warning slug=%s reason=single_candidate_dominates_matchups candidate=%r",
                slug,
                next(iter(participation_counter.keys())),
            )

        best_balance = set(critic_result.summary.best_balance_candidates)
        if best_balance and not any(candidate in best_balance for candidate in result.summary.top_advancers):
            logger.warning(
                "thumbnail_pipeline.pairwise_tournament.sanity_warning slug=%s reason=best_balance_missing_in_top_advancers best_balance=%s top_advancers=%s",
                slug,
                sorted(best_balance),
                result.summary.top_advancers,
            )

    def _fallback(self, eligible_pool: list[str]) -> ThumbnailPairwiseTournamentResult:
        logger.info("thumbnail_pipeline.pairwise_tournament.fallback_default")

        pool = eligible_pool[:]
        if len(pool) < 2:
            pool = ["Fallback Candidate A", "Fallback Candidate B"]

        matchups: list[ThumbnailPairwiseMatchup] = []
        max_matchups = min(3, len(pool) - 1)
        for index in range(max_matchups):
            left_candidate = pool[index]
            right_candidate = pool[index + 1]
            matchups.append(
                ThumbnailPairwiseMatchup(
                    left_candidate=left_candidate,
                    right_candidate=right_candidate,
                    comparison_axis="fallback comparison",
                    winner=left_candidate,
                    reason="fallback tournament result",
                    confidence="fragile",
                )
            )

        if len(matchups) < 3 and len(pool) >= 2:
            while len(matchups) < 3:
                matchups.append(
                    ThumbnailPairwiseMatchup(
                        left_candidate=pool[0],
                        right_candidate=pool[1],
                        comparison_axis="fallback comparison",
                        winner=pool[0],
                        reason="fallback tournament result",
                        confidence="fragile",
                    )
                )

        top_advancers = pool[:2]
        fragile_matchups = [f"{m.left_candidate} vs {m.right_candidate}" for m in matchups[:3]]
        summary = ThumbnailPairwiseTournamentSummary(
            top_advancers=top_advancers,
            repeat_winners=[],
            fragile_matchups=fragile_matchups,
            selector_guidance="prefer the cleanest human hook among the advancers",
        )
        return ThumbnailPairwiseTournamentResult(matchups=matchups, summary=summary)

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
    "ThumbnailPairwiseMatchup",
    "ThumbnailPairwiseTournament",
    "ThumbnailPairwiseTournamentError",
    "ThumbnailPairwiseTournamentResult",
    "ThumbnailPairwiseTournamentSummary",
    "load_thumbnail_pairwise_tournament_prompt",
]
