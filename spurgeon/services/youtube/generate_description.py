# generate_description.py

"""Utilities to build a full YouTube description for a devotional video."""

from __future__ import annotations

from spurgeon.config.settings import Settings
from spurgeon.models import Reading


DESCRIPTION_TEMPLATE = (
    "📖 Devotional: Daily Light on the Daily Path\n\n"
    "📅 Date: {reading_type}, {reading_date}\n\n"
    "➡️ Don’t miss tomorrow’s devotional — subscribe today.\n\n"
    "---\n"
    "© Public Domain Text. "
)

class DescriptionGenerator:
    """Generate user-facing YouTube descriptions for devotional videos."""

    def __init__(self, settings: Settings) -> None:  # noqa: D401 - signature kept for compatibility
        # ``settings`` is accepted for interface compatibility; future customisation may use it.
        self._settings = settings

    def generate(self, reading: Reading, title: str) -> str:  # noqa: D401 - Dutch docstring retained
        """Genereer een volledige YouTube-description voor de video."""

        formatted_date = reading.date.strftime("%B %d, %Y").replace(" 0", " ")

        return DESCRIPTION_TEMPLATE.format(
            reading_type=reading.reading_type.value,
            reading_date=formatted_date,
        )


__all__ = ["DescriptionGenerator"]
