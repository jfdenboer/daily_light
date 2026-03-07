"""Intent-card models and parser for thumbnail generation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=8)
def load_thumbnail_intent_card_prompt(version: str = "v1") -> str:
    prompt_path = PROMPTS_DIR / f"thumbnail_intent_card.{version}.txt"
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValueError(
            f"Unknown thumbnail intent-card prompt version: {version}"
        ) from exc


THUMBNAIL_INTENT_CARD_DEVMSG = load_thumbnail_intent_card_prompt()


@dataclass(frozen=True)
class ThumbnailIntentCard:
    core_tension: str
    emotional_tone: str
    visual_motif: str
    scene_direction: str
    avoid: str


class IntentCardParseError(RuntimeError):
    """Raised when intent-card output cannot be parsed."""


def parse_thumbnail_intent_card(text: str) -> ThumbnailIntentCard:
    pattern = re.compile(r"^\s*(?:(\d)\)\s*)?([a-z_]+)\s*:\s*(.*?)\s*$")
    expected_order = [
        "core_tension",
        "emotional_tone",
        "visual_motif",
        "scene_direction",
        "avoid",
    ]
    fields: dict[str, str] = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) != 5:
        logger.warning(
            "thumbnail_pipeline.intent_card_parse_error reason=expected_5_lines lines=%d raw_output=%r",
            len(lines),
            text,
        )
        raise IntentCardParseError(
            f"Malformed thumbnail intent-card output: expected 5 lines, got {len(lines)}"
        )

    for index, line in enumerate(lines, start=1):
        match = pattern.match(line)
        if not match:
            logger.warning(
                "thumbnail_pipeline.intent_card_parse_error reason=invalid_line line=%s raw_output=%r",
                line,
                text,
            )
            raise IntentCardParseError(f"Malformed thumbnail intent-card line: {line}")

        ordinal, key, raw_value = match.groups()
        expected_key = expected_order[index - 1]
        parsed_key = key.strip().lower()

        if ordinal is not None and ordinal != str(index):
            logger.warning(
                "thumbnail_pipeline.intent_card_parse_error reason=wrong_ordinal line_index=%d ordinal=%s expected=%d line=%s raw_output=%r",
                index,
                ordinal,
                index,
                line,
                text,
            )
            raise IntentCardParseError(
                f"Malformed thumbnail intent-card line {index}: expected ordinal '{index})', got '{ordinal})'"
            )

        if parsed_key != expected_key:
            logger.warning(
                "thumbnail_pipeline.intent_card_parse_error reason=unexpected_key line_index=%d got=%s expected=%s line=%s raw_output=%r",
                index,
                parsed_key,
                expected_key,
                line,
                text,
            )
            raise IntentCardParseError(
                f"Malformed thumbnail intent-card line {index}: expected key '{expected_key}', got '{key}'"
            )

        if expected_key in fields:
            logger.warning(
                "thumbnail_pipeline.intent_card_parse_error reason=duplicate_key line_index=%d key=%s line=%s raw_output=%r",
                index,
                expected_key,
                line,
                text,
            )
            raise IntentCardParseError(
                f"Malformed thumbnail intent-card output: duplicate field '{expected_key}'"
            )

        value = " ".join(raw_value.split()).strip()
        if not value:
            logger.warning(
                "thumbnail_pipeline.intent_card_parse_error reason=empty_value line_index=%d key=%s line=%s raw_output=%r",
                index,
                expected_key,
                line,
                text,
            )
            raise IntentCardParseError(
                f"Malformed thumbnail intent-card line {index}: value is empty"
            )
        fields[expected_key] = value

    missing = [name for name in expected_order if name not in fields]
    if missing:
        logger.warning(
            "thumbnail_pipeline.intent_card_parse_error reason=missing_fields missing=%s raw_output=%r",
            ",".join(missing),
            text,
        )
        raise IntentCardParseError(
            f"Malformed thumbnail intent-card output: missing fields {missing}"
        )

    return ThumbnailIntentCard(
        core_tension=fields["core_tension"],
        emotional_tone=fields["emotional_tone"],
        visual_motif=fields["visual_motif"],
        scene_direction=fields["scene_direction"],
        avoid=fields["avoid"],
    )


def normalize_clip_reading_text(text: str, *, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    clipped = normalized[:max_chars].rstrip()
    last_space = clipped.rfind(" ")
    return clipped[:last_space] if last_space > 0 else clipped
