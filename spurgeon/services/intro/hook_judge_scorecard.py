"""Parser utilities for hook-judge scorecard outputs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_SCORE_ROW_PATTERN = re.compile(
    r"^(?P<idx>\d+)\|t=(?P<t>[0-2])\|o=(?P<o>[0-2])\|v=(?P<v>[0-2])\|f=(?P<f>[0-2])\|s=(?P<s>[0-2])\|i=(?P<i>[0-2])$"
)


@dataclass(slots=True)
class ScoreRow:
    candidate_idx: int
    tension: int
    open_loop: int
    viewer: int
    fluency: int
    tone: int
    intent: int


def parse_hook_judge_scorecard(text: str, *, expected_count: int) -> list[ScoreRow] | None:
    """Parse strict SCORES/END output into score rows.

    Returns ``None`` when the response is malformed.
    """

    stripped = text.strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]

    if len(lines) < 3 or lines[0] != "SCORES" or lines[-1] != "END":
        logger.warning(
            "hook_pipeline.scorecard_parse_failed reason=missing_markers expected_count=%d raw=%r",
            expected_count,
            stripped[:500],
        )
        return None

    body = lines[1:-1]
    if len(body) != expected_count:
        logger.warning(
            "hook_pipeline.scorecard_parse_failed reason=count_mismatch expected_count=%d parsed_count=%d raw=%r",
            expected_count,
            len(body),
            stripped[:500],
        )
        return None

    parsed: list[ScoreRow] = []
    for expected_idx, line in enumerate(body, start=1):
        match = _SCORE_ROW_PATTERN.match(line)
        if not match:
            logger.warning(
                "hook_pipeline.scorecard_parse_failed reason=invalid_row expected_idx=%d row=%r raw=%r",
                expected_idx,
                line,
                stripped[:500],
            )
            return None

        row_idx = int(match.group("idx"))
        if row_idx != expected_idx:
            logger.warning(
                "hook_pipeline.scorecard_parse_failed reason=index_mismatch expected_idx=%d row_idx=%d raw=%r",
                expected_idx,
                row_idx,
                stripped[:500],
            )
            return None

        parsed.append(
            ScoreRow(
                candidate_idx=row_idx,
                tension=int(match.group("t")),
                open_loop=int(match.group("o")),
                viewer=int(match.group("v")),
                fluency=int(match.group("f")),
                tone=int(match.group("s")),
                intent=int(match.group("i")),
            )
        )

    return parsed


__all__ = ["ScoreRow", "parse_hook_judge_scorecard"]
