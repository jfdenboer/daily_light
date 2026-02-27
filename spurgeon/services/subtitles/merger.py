# merger.py
from __future__ import annotations

"""
spurgeon.services.subtitles.merger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure *merge*-logica voor ondertitelregels.

Fix-log
-------
2025-07-03  vergelijk `timedelta.total_seconds()`
2025-07-12  leading **soft-punct** repair
2025-07-13  **BUG-FIX** – `_LEADING_SOFT_RE` kon geen *set* verwerken  
            (TypeError in `re.escape`).  We bouwen de karakterklasse nu
            via een join van per-teken escapes.
2025-07-15  **BEHAVIOUR FIX** – micro-line met strong punct → altijd merge naar links
"""

from copy import deepcopy
import re
from typing import List, Sequence

from .caption_models import STRONG_PUNCT, SOFT_PUNCT, SubtitleLine, _SPACE_RE

__all__ = [
    "merge_micro_lines",
    "fix_leading_soft_punct",
    "_visible_len",
    "_ends_with_strong_punct",
]

###############################################################################
# Helper-functies                                                             #
###############################################################################


def _visible_len(text: str) -> int:
    """Aantal *zichtbare* tekens (α-numerics) in *text*."""
    return sum(1 for ch in text if ch.isalnum())


def _ends_with_strong_punct(text: str) -> bool:
    """True als *text* eindigt op een sterk leesteken (. ! ? ;)"""
    return bool(text) and text.rstrip()[-1] in STRONG_PUNCT


###############################################################################
# Post-pass: verplaats leidende zachte interpunctie                           #
###############################################################################

_LEADING_SOFT_RE = re.compile(
    rf"^[{''.join(re.escape(ch) for ch in SOFT_PUNCT)}]\s+"
)


def fix_leading_soft_punct(lines: List[SubtitleLine]) -> List[SubtitleLine]:
    """
    Knipt een leidende ‘zachte’ interpunctie (`, ` of `: ` of `– `)
    uit de *huidige* regel en plakt haar aan de *vorige* regel.
    """
    if len(lines) < 2:
        return lines

    fixed: List[SubtitleLine] = [lines[0]]

    for cur in lines[1:]:
        m = _LEADING_SOFT_RE.match(cur.text)
        if not m:
            fixed.append(cur)
            continue

        lead = m.group(0)
        prev = fixed[-1]

        prev_upd = SubtitleLine(
            start=prev.start,
            end=prev.end,
            text=f"{prev.text.rstrip()}{lead}".rstrip(),
        )
        cur_upd = SubtitleLine(
            start=cur.start,
            end=cur.end,
            text=cur.text[m.end():].lstrip(),
        )

        fixed[-1] = prev_upd
        fixed.append(cur_upd)

    return fixed


###############################################################################
# Merge main                                                                  #
###############################################################################


def merge_micro_lines(
    lines: Sequence[SubtitleLine],
    *,
    min_chars: int = 8,
    min_duration: float = 0.6,
    hard_max_chars: int = 55,
) -> List[SubtitleLine]:
    """Plak *micro-lines* aan buren en retourneer een nieuwe lijst."""
    if len(lines) <= 1:
        return list(deepcopy(lines))

    merged = deepcopy(list(lines))
    i = 0
    while i < len(merged):
        cur = merged[i]
        duration_sec = (cur.end - cur.start).total_seconds()

        if _visible_len(cur.text) >= min_chars or duration_sec >= min_duration:
            i += 1
            continue  # regel is oké

        left_idx = i - 1 if i > 0 else None
        right_idx = i + 1 if i + 1 < len(merged) else None
        if left_idx is None and right_idx is None:
            i += 1
            continue

        def pick_partner() -> int | None:
            if _ends_with_strong_punct(cur.text):
                return left_idx if left_idx is not None else right_idx
            if left_idx is None:
                return right_idx
            if right_idx is None:
                return left_idx
            left, right = merged[left_idx], merged[right_idx]
            if _ends_with_strong_punct(left.text) != _ends_with_strong_punct(right.text):
                return left_idx if not _ends_with_strong_punct(left.text) else right_idx
            return (
                left_idx
                if _visible_len(left.text) <= _visible_len(right.text)
                else right_idx
            )

        partner_idx = pick_partner()
        if partner_idx is None:
            i += 1
            continue

        partner = merged[partner_idx]
        if partner_idx < i:  # merge links
            new_text = f"{partner.text} {cur.text}"
            new_start, new_end = partner.start, cur.end
        else:                # merge rechts
            new_text = f"{cur.text} {partner.text}"
            new_start, new_end = cur.start, partner.end

        new_text = _SPACE_RE.sub(" ", new_text).strip()

        if _visible_len(new_text) > hard_max_chars:
            alt_idx = left_idx if partner_idx == right_idx else right_idx
            if alt_idx is not None and alt_idx != partner_idx:
                partner_idx = alt_idx
                partner = merged[partner_idx]
                if partner_idx < i:
                    new_text = f"{partner.text} {cur.text}"
                    new_start, new_end = partner.start, cur.end
                else:
                    new_text = f"{cur.text} {partner.text}"
                    new_start, new_end = cur.start, partner.end
                new_text = _SPACE_RE.sub(" ", new_text).strip()
            if _visible_len(new_text) > hard_max_chars:
                i += 1
                continue

        merged[partner_idx] = SubtitleLine(start=new_start, end=new_end, text=new_text)
        merged.pop(i)
        i = max(partner_idx - 1, 0)

    return fix_leading_soft_punct(merged)
