"""Canonical thumbnail pipeline orchestrator for steps 0-8 and 2A (image intent)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from spurgeon.config.settings import Settings
    from spurgeon.models import Reading

from .thumbnail_bucketed_candidate_generator import (
    ThumbnailBucketedCandidateGenerator,
    ThumbnailCandidateBuckets,
)
from .thumbnail_candidate_critic import ThumbnailCandidateCritic, ThumbnailCandidateCriticResult
from .thumbnail_copy_intent_card import ThumbnailCopyIntentCard, ThumbnailCopyIntentCardBuilder
from .thumbnail_confidence_gate import ThumbnailConfidenceGate, ThumbnailConfidenceGateResult
from .thumbnail_final_selector import ThumbnailFinalSelection, ThumbnailFinalSelector
from .thumbnail_image_intent_card import ThumbnailImageIntentCard, ThumbnailImageIntentCardBuilder
from .thumbnail_pairwise_tournament import ThumbnailPairwiseTournament, ThumbnailPairwiseTournamentResult
from .thumbnail_reading_diagnoser import ThumbnailReadingDiagnoser, ThumbnailReadingDiagnosis
from .thumbnail_source_reader import ThumbnailSourceRead, ThumbnailSourceReader

logger = logging.getLogger(__name__)


class ThumbnailPipelineResult(BaseModel):
    """Typed aggregate result for the canonical thumbnail pipeline."""

    source_read: ThumbnailSourceRead
    reading_diagnosis: ThumbnailReadingDiagnosis
    image_intent_card: ThumbnailImageIntentCard
    copy_intent_card: ThumbnailCopyIntentCard
    candidate_buckets: ThumbnailCandidateBuckets
    critic_result: ThumbnailCandidateCriticResult
    pairwise_result: ThumbnailPairwiseTournamentResult
    final_selection: ThumbnailFinalSelection
    confidence_gate_result: ThumbnailConfidenceGateResult


class ThumbnailPipelineRunnerError(RuntimeError):
    """Raised when thumbnail orchestration fails unexpectedly."""


class ThumbnailPipelineRunner:
    """Run thumbnail steps 0-8 in the canonical production flow."""

    def __init__(
        self,
        settings: "Settings",
        *,
        source_reader: ThumbnailSourceReader | None = None,
        reading_diagnoser: ThumbnailReadingDiagnoser | None = None,
        image_intent_card_builder: ThumbnailImageIntentCardBuilder | None = None,
        copy_intent_card_builder: ThumbnailCopyIntentCardBuilder | None = None,
        bucketed_candidate_generator: ThumbnailBucketedCandidateGenerator | None = None,
        candidate_critic: ThumbnailCandidateCritic | None = None,
        pairwise_tournament: ThumbnailPairwiseTournament | None = None,
        final_selector: ThumbnailFinalSelector | None = None,
        confidence_gate: ThumbnailConfidenceGate | None = None,
    ) -> None:
        self.settings = settings
        self.source_reader = source_reader or ThumbnailSourceReader(settings)
        self.reading_diagnoser = reading_diagnoser or ThumbnailReadingDiagnoser(settings)
        self.image_intent_card_builder = image_intent_card_builder or ThumbnailImageIntentCardBuilder(settings)
        self.copy_intent_card_builder = copy_intent_card_builder or ThumbnailCopyIntentCardBuilder(settings)
        self.bucketed_candidate_generator = (
            bucketed_candidate_generator or ThumbnailBucketedCandidateGenerator(settings)
        )
        self.candidate_critic = candidate_critic or ThumbnailCandidateCritic(settings)
        self.pairwise_tournament = pairwise_tournament or ThumbnailPairwiseTournament(settings)
        self.final_selector = final_selector or ThumbnailFinalSelector(settings)
        self.confidence_gate = confidence_gate or ThumbnailConfidenceGate(settings)

    def run(self, reading: "Reading") -> ThumbnailPipelineResult:
        """Execute thumbnail steps 0-8 and return all typed outputs."""

        logger.info("thumbnail_pipeline.start slug=%s", reading.slug)

        try:
            logger.info("thumbnail_pipeline.step_start slug=%s step=0_source_read", reading.slug)
            source_read = self.source_reader.read(reading)
            logger.info(
                "thumbnail_pipeline.step_success slug=%s step=0_source_read hook_zones=%d",
                reading.slug,
                len(source_read.hook_zones),
            )

            logger.info("thumbnail_pipeline.step_start slug=%s step=1_reading_diagnosis", reading.slug)
            diagnosis = self.reading_diagnoser.diagnose(reading, source_read)
            logger.info(
                "thumbnail_pipeline.step_success slug=%s step=1_reading_diagnosis primary_axis=%s",
                reading.slug,
                diagnosis.primary_axis,
            )

            logger.info("thumbnail_pipeline.step_start slug=%s step=2a_image_intent_card", reading.slug)
            image_intent_card = self.image_intent_card_builder.build(reading, source_read, diagnosis)
            logger.info(
                "thumbnail_pipeline.step_success slug=%s step=2a_image_intent_card anchor=%s",
                reading.slug,
                image_intent_card.dominant_anchor,
            )

            logger.info("thumbnail_pipeline.step_start slug=%s step=2b_copy_intent_card", reading.slug)
            copy_intent_card = self.copy_intent_card_builder.build(reading, source_read, diagnosis)
            logger.info(
                "thumbnail_pipeline.step_success slug=%s step=2b_copy_intent_card best_hook_mode=%s",
                reading.slug,
                copy_intent_card.preferred_hook_mode,
            )

            logger.info("thumbnail_pipeline.step_start slug=%s step=3_candidate_buckets", reading.slug)
            candidate_buckets = self.bucketed_candidate_generator.generate(
                reading,
                source_read,
                diagnosis,
                copy_intent_card,
            )
            logger.info(
                "thumbnail_pipeline.step_success slug=%s step=3_candidate_buckets bucket_counts=felt_state:%d posture:%d threshold:%d edge_tension:%d",
                reading.slug,
                len(candidate_buckets.felt_state),
                len(candidate_buckets.posture),
                len(candidate_buckets.threshold),
                len(candidate_buckets.edge_tension),
            )

            logger.info("thumbnail_pipeline.step_start slug=%s step=4_candidate_critic", reading.slug)
            critic_result = self.candidate_critic.critique(
                reading,
                source_read,
                diagnosis,
                copy_intent_card,
                candidate_buckets,
            )
            logger.info(
                "thumbnail_pipeline.step_success slug=%s step=4_candidate_critic best_balance=%s",
                reading.slug,
                critic_result.summary.best_balance_candidates,
            )

            logger.info("thumbnail_pipeline.step_start slug=%s step=5_pairwise_tournament", reading.slug)
            pairwise_result = self.pairwise_tournament.run(
                reading,
                source_read,
                diagnosis,
                copy_intent_card,
                candidate_buckets,
                critic_result,
            )
            logger.info(
                "thumbnail_pipeline.step_success slug=%s step=5_pairwise_tournament top_advancers=%s",
                reading.slug,
                pairwise_result.summary.top_advancers,
            )

            logger.info("thumbnail_pipeline.step_start slug=%s step=6_final_selector", reading.slug)
            final_selection = self.final_selector.select(
                reading,
                source_read,
                diagnosis,
                copy_intent_card,
                candidate_buckets,
                critic_result,
                pairwise_result,
            )
            logger.info(
                "thumbnail_pipeline.step_success slug=%s step=6_final_selector winner=%s confidence=%s",
                reading.slug,
                final_selection.winner,
                final_selection.confidence,
            )

            logger.info("thumbnail_pipeline.step_start slug=%s step=8_confidence_gate", reading.slug)
            confidence_gate_result = self.confidence_gate.evaluate(
                reading,
                source_read,
                diagnosis,
                copy_intent_card,
                candidate_buckets,
                critic_result,
                pairwise_result,
                final_selection,
            )
            logger.info(
                "thumbnail_pipeline.step_success slug=%s step=8_confidence_gate winner_status=%s recommendation=%s",
                reading.slug,
                confidence_gate_result.winner_status,
                confidence_gate_result.final_recommendation,
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("thumbnail_pipeline.unexpected_error slug=%s error_type=%s", reading.slug, type(exc).__name__)
            raise ThumbnailPipelineRunnerError(str(exc)) from exc

        result = ThumbnailPipelineResult(
            source_read=source_read,
            reading_diagnosis=diagnosis,
            image_intent_card=image_intent_card,
            copy_intent_card=copy_intent_card,
            candidate_buckets=candidate_buckets,
            critic_result=critic_result,
            pairwise_result=pairwise_result,
            final_selection=final_selection,
            confidence_gate_result=confidence_gate_result,
        )
        logger.info("thumbnail_pipeline.complete slug=%s winner=%s", reading.slug, result.final_selection.winner)
        return result


__all__ = [
    "ThumbnailPipelineResult",
    "ThumbnailPipelineRunner",
    "ThumbnailPipelineRunnerError",
]
