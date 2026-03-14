"""Shared text-normalization helpers for thumbnail v3 steps."""

from __future__ import annotations


def normalize_clip_reading_text(text: str, *, max_chars: int) -> str:
    """Normalize whitespace and clip reading text at word boundaries."""

    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized

    clipped = normalized[:max_chars].rstrip()
    last_space = clipped.rfind(" ")
    return clipped[:last_space] if last_space > 0 else clipped
