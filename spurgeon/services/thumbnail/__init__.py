"""Thumbnail generation service exports.

The package intentionally keeps imports lazy so tooling can import
``spurgeon.services.thumbnail`` without triggering optional runtime dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ThumbnailGenerationError",
    "ThumbnailGenerator",
    "ThumbnailSourceRead",
    "ThumbnailSourceReadError",
    "ThumbnailSourceReader",
    "ThumbnailReadingDiagnosis",
    "ThumbnailReadingDiagnosisError",
    "ThumbnailReadingDiagnoser",
    "ThumbnailImageIntentCard",
    "ThumbnailImageIntentCardError",
    "ThumbnailImageIntentCardBuilder",
    "ThumbnailCopyIntentCard",
    "ThumbnailCopyIntentCardError",
    "ThumbnailCopyIntentCardBuilder",
    "ThumbnailCandidateBuckets",
    "ThumbnailBucketedCandidateGenerationError",
    "ThumbnailBucketedCandidateGenerator",
    "ThumbnailCandidateCriticAssessment",
    "ThumbnailCandidateCriticSummary",
    "ThumbnailCandidateCriticResult",
    "ThumbnailCandidateCriticError",
    "ThumbnailCandidateCritic",
    "ThumbnailPairwiseMatchup",
    "ThumbnailPairwiseTournamentSummary",
    "ThumbnailPairwiseTournamentResult",
    "ThumbnailPairwiseTournamentError",
    "ThumbnailPairwiseTournament",
    "ThumbnailFinalSelection",
    "ThumbnailFinalSelectorError",
    "ThumbnailFinalSelector",
    "ThumbnailConfidenceGateResult",
    "ThumbnailConfidenceGateError",
    "ThumbnailConfidenceGate",
    "ThumbnailPipelineResult",
    "ThumbnailPipelineRunner",
    "ThumbnailPipelineRunnerError",
]


def __getattr__(name: str) -> Any:
    if name in {"ThumbnailGenerator", "ThumbnailGenerationError"}:
        module = import_module("spurgeon.services.thumbnail.thumbnail_generator")
        return getattr(module, name)
    if name in {"ThumbnailSourceRead", "ThumbnailSourceReadError", "ThumbnailSourceReader"}:
        module = import_module("spurgeon.services.thumbnail.thumbnail_source_reader")
        return getattr(module, name)
    if name in {"ThumbnailReadingDiagnosis", "ThumbnailReadingDiagnosisError", "ThumbnailReadingDiagnoser"}:
        module = import_module("spurgeon.services.thumbnail.thumbnail_reading_diagnoser")
        return getattr(module, name)
    if name in {"ThumbnailImageIntentCard", "ThumbnailImageIntentCardError", "ThumbnailImageIntentCardBuilder"}:
        module = import_module("spurgeon.services.thumbnail.thumbnail_image_intent_card")
        return getattr(module, name)
    if name in {"ThumbnailCopyIntentCard", "ThumbnailCopyIntentCardError", "ThumbnailCopyIntentCardBuilder"}:
        module = import_module("spurgeon.services.thumbnail.thumbnail_copy_intent_card")
        return getattr(module, name)
    if name in {
        "ThumbnailCandidateBuckets",
        "ThumbnailBucketedCandidateGenerationError",
        "ThumbnailBucketedCandidateGenerator",
    }:
        module = import_module("spurgeon.services.thumbnail.thumbnail_bucketed_candidate_generator")
        return getattr(module, name)
    if name in {
        "ThumbnailCandidateCriticAssessment",
        "ThumbnailCandidateCriticSummary",
        "ThumbnailCandidateCriticResult",
        "ThumbnailCandidateCriticError",
        "ThumbnailCandidateCritic",
    }:
        module = import_module("spurgeon.services.thumbnail.thumbnail_candidate_critic")
        return getattr(module, name)

    if name in {
        "ThumbnailPairwiseMatchup",
        "ThumbnailPairwiseTournamentSummary",
        "ThumbnailPairwiseTournamentResult",
        "ThumbnailPairwiseTournamentError",
        "ThumbnailPairwiseTournament",
    }:
        module = import_module("spurgeon.services.thumbnail.thumbnail_pairwise_tournament")
        return getattr(module, name)
    if name in {
        "ThumbnailFinalSelection",
        "ThumbnailFinalSelectorError",
        "ThumbnailFinalSelector",
    }:
        module = import_module("spurgeon.services.thumbnail.thumbnail_final_selector")
        return getattr(module, name)
    if name in {
        "ThumbnailConfidenceGateResult",
        "ThumbnailConfidenceGateError",
        "ThumbnailConfidenceGate",
    }:
        module = import_module("spurgeon.services.thumbnail.thumbnail_confidence_gate")
        return getattr(module, name)
    if name in {
        "ThumbnailPipelineResult",
        "ThumbnailPipelineRunner",
        "ThumbnailPipelineRunnerError",
    }:
        module = import_module("spurgeon.services.thumbnail.thumbnail_pipeline_runner")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
