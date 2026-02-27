# caption_models.py

from __future__ import annotations

"""
spurgeon.services.subtitles.caption_models
-----------------------------------------

Structure-only dataclasses & helpers for the subtitle sub-package.
**No** I/O or merge logic lives here so we avoid circular imports.

2025-07-09 **PUNCTUATION FIX**
  * `STRONG_PUNCT` is now a **set** without ':'.
  * `SOFT_PUNCT` is now a **set** and *includes* ':'.
  * Prevents false flushes on whitespace and keeps Bible references
    like “Judges 16: 6” on a single line.
"""

from dataclasses import dataclass
from datetime import timedelta
import re
from typing import List

__all__ = [
    "SubtitleLine",
    "STRONG_PUNCT",
    "SOFT_PUNCT",
    "_format_srt_time",
    "_SPACE_RE",
]

# ---------------------------------------------------------------------------
# Punctuation constants
# ---------------------------------------------------------------------------

# sentence-breaking punctuation → hard split
STRONG_PUNCT: set[str] = {".", "!", "?", ";"}

# softer pause – allowed to overflow max_chars by two before a flush
SOFT_PUNCT: set[str] = {",", "–", "-", ":"}

# RegEx helpers
_SPACE_RE = re.compile(r"\s+")
_PUNCT_NORMALIZE_RE = re.compile(r"\s([.,!?;:])")

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _format_srt_time(td: timedelta) -> str:
    """Convert :class:`datetime.timedelta` → ``HH:MM:SS,mmm`` string."""
    total_ms = int(td.total_seconds() * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SubtitleLine:
    """Single subtitle line after tokenisation and merging."""

    start: timedelta
    end: timedelta
    text: str

    # Post-processing --------------------------------------------------
    def __post_init__(self):  # type: ignore[override] – dataclass hook
        # Squash whitespace & strip leading/trailing spaces
        object.__setattr__(self, "text", _SPACE_RE.sub(" ", self.text.strip()))

    # Convenience properties ------------------------------------------
    @property
    def duration(self) -> float:
        """Line duration in **seconds**."""
        return self.end.total_seconds() - self.start.total_seconds()

    @property
    def start_srt(self) -> str:
        """Start timestamp in SRT notation (``HH:MM:SS,mmm``)."""
        return _format_srt_time(self.start)

    @property
    def end_srt(self) -> str:
        """End timestamp in SRT notation (``HH:MM:SS,mmm``)."""
        return _format_srt_time(self.end)

    # Helper -----------------------------------------------------------
    def visible_len(self) -> int:
        """Number of visible (alpha-numeric) characters in *text*."""
        return len(re.sub(r"[\s\W_]", "", self.text))
