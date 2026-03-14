"""Prompt policies and builders for thumbnail image generation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spurgeon.models import Reading

from .thumbnail_image_intent_card import ThumbnailImageIntentCard

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_PROMPT_VERSION = "v1"


@lru_cache(maxsize=16)
def _load_prompt_template(template_name: str, version: str) -> str:
    template_path = PROMPTS_DIR / f"{template_name}.{version}.txt"
    try:
        return template_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValueError(
            f"Unknown thumbnail prompt template '{template_name}' version '{version}'"
        ) from exc


def get_thumbnail_image_prompt_template(version: str = DEFAULT_PROMPT_VERSION) -> str:
    return _load_prompt_template("thumbnail_image", version)


def _image_policy_lines(version: str = DEFAULT_PROMPT_VERSION) -> list[str]:
    template = get_thumbnail_image_prompt_template(version)
    return template.splitlines()[:7]


_IMAGE_POLICY_LINES = _image_policy_lines(DEFAULT_PROMPT_VERSION)
THUMBNAIL_STYLE_LINE = _IMAGE_POLICY_LINES[0]
THUMBNAIL_SUBJECT_LINE = _IMAGE_POLICY_LINES[1]
THUMBNAIL_BACKGROUND_LINE = _IMAGE_POLICY_LINES[2]
THUMBNAIL_COMPOSITION_LINE = _IMAGE_POLICY_LINES[3]
THUMBNAIL_CONSTRAINTS_LINE = _IMAGE_POLICY_LINES[4]
THUMBNAIL_PALETTE_LINE = _IMAGE_POLICY_LINES[5]
THUMBNAIL_LIGHTING_LINE = _IMAGE_POLICY_LINES[6]


def build_thumbnail_prompt(
    reading: "Reading",
    thumbnail_text: str,
    image_intent_card: ThumbnailImageIntentCard,
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> str:
    """Build the image prompt directly from step-2A image intent-card fields."""

    template = get_thumbnail_image_prompt_template(prompt_version)
    return template.format(
        reading_type=reading.reading_type.value,
        thumbnail_text=thumbnail_text,
        visual_tension=image_intent_card.visual_tension,
        emotional_tone=image_intent_card.emotional_tone,
        dominant_anchor=image_intent_card.dominant_anchor,
        scene_direction=image_intent_card.scene_direction,
        subject_priority=image_intent_card.subject_priority,
        visual_open_loop=image_intent_card.visual_open_loop,
        visual_avoid=", ".join(image_intent_card.visual_avoid),
    )
