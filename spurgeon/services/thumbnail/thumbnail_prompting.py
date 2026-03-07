"""Prompt policies and builders for thumbnail image generation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from spurgeon.models import Reading

from .thumbnail_intent_card import ThumbnailIntentCard

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


def get_thumbnail_intent_card_prompt_template(version: str = DEFAULT_PROMPT_VERSION) -> str:
    return _load_prompt_template("thumbnail_intent_card", version)


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
    reading: Reading,
    thumbnail_text: str,
    intent_card: ThumbnailIntentCard,
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> str:
    template = get_thumbnail_image_prompt_template(prompt_version)
    return template.format(
        thumbnail_text=thumbnail_text,
        core_tension=intent_card.core_tension,
        emotional_tone=intent_card.emotional_tone,
        visual_motif=intent_card.visual_motif,
        scene_direction=intent_card.scene_direction,
        avoid=intent_card.avoid,
    )
