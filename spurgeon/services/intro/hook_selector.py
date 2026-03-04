"""Deterministic selector for scored hook candidates."""

from __future__ import annotations

from dataclasses import dataclass

from spurgeon.services.intro.hook_judge_scorecard import ScoreRow
from spurgeon.services.intro.hook_pipeline.rules import CandidateCheck, WORD_PATTERN


@dataclass(slots=True)
class RankedHookCandidate:
    rank: int
    candidate_idx: int
    candidate: str
    angle: str
    word_count: int
    score: ScoreRow
    weighted_total: int


@dataclass(slots=True)
class HookSelectionResult:
    selected: RankedHookCandidate
    ranked: list[RankedHookCandidate]


def _weighted_total(score: ScoreRow) -> int:
    return (
        score.tension * 3
        + score.open_loop * 3
        + score.viewer * 3
        + score.fluency * 2
        + score.tone
        + score.intent
    )


def _tie_break_key(item: RankedHookCandidate) -> tuple[int, int, int, int, int, int, int]:
    preferred_length = 1 if 11 <= item.word_count <= 13 else 0
    distance_to_twelve = abs(item.word_count - 12)
    return (
        item.weighted_total,
        item.score.open_loop,
        item.score.viewer,
        item.score.tension,
        preferred_length,
        -distance_to_twelve,
        -item.candidate_idx,
    )


def select_by_scorecard(
    candidates: list[CandidateCheck],
    scores: list[ScoreRow],
) -> HookSelectionResult:
    """Select a winner by deterministic weighted ranking and tie-breakers."""

    if len(candidates) != len(scores):
        raise ValueError("candidates_and_scores_length_mismatch")

    ranked_items: list[RankedHookCandidate] = []
    for idx, (candidate, score) in enumerate(zip(candidates, scores, strict=True), start=1):
        if score.candidate_idx != idx:
            raise ValueError("scorecard_index_mismatch")

        ranked_items.append(
            RankedHookCandidate(
                rank=0,
                candidate_idx=idx,
                candidate=candidate.candidate,
                angle=candidate.angle,
                word_count=len(WORD_PATTERN.findall(candidate.candidate)),
                score=score,
                weighted_total=_weighted_total(score),
            )
        )

    ranked_items.sort(key=_tie_break_key, reverse=True)

    for rank, item in enumerate(ranked_items, start=1):
        item.rank = rank

    return HookSelectionResult(selected=ranked_items[0], ranked=ranked_items)


__all__ = ["HookSelectionResult", "RankedHookCandidate", "select_by_scorecard"]
