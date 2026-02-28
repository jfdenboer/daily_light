# tokenizer.py
from __future__ import annotations

"""
spurgeon.services.subtitles.tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

✓ Zet een Rev.ai-monologue-payload om naar
  (1) losse tokens en (2) **ruwe** :class:`SubtitleLine`-objecten.
✓ Alle esthetische correcties (leidende komma’s e.d.) gebeuren nu
  volledig downstream in ``merger`` – de tokenizer zelf hanteert een
  *harde* char-limiet (38) en een flush op zinsafbrekende punct.

2025-07-12  CLEAN-UP
--------------------
* Logica rond “zachte” punct + overflow **verwijderd**; beperkingen
  worden in `merger.fix_leading_soft_punct` afgehandeld.
* Bible-reference flush (``\d{1,3}:\d{1,3}``) blijft behouden.
"""

from datetime import timedelta
import re
from typing import Dict, Iterable, Iterator, List, Union

from .caption_models import (
    SubtitleLine,
    STRONG_PUNCT,
    _SPACE_RE,
    _PUNCT_NORMALIZE_RE,
)

__all__ = ["iter_tokens", "build_raw_lines"]

# ---------------------------------------------------------------------------
# Types & helpers
# ---------------------------------------------------------------------------

Token = Dict[str, Union[str, float, None]]
_BIBLE_REF_RE = re.compile(r"\b\d{1,3}:\s?\d{1,3}$")  # 3:16  10:5  119:105

# ---------------------------------------------------------------------------
# Token generator
# ---------------------------------------------------------------------------


def iter_tokens(data: dict, *, include_punct: bool = True) -> Iterator[Token]:
    """Yield tokens (optioneel zonder punct) uit een Rev.ai-JSON-boom."""
    for mono in data.get("monologues", []):
        for elem in mono.get("elements", []):
            typ = elem.get("type")
            if not include_punct and typ == "punct":
                continue
            yield {
                "type": typ,
                "value": elem.get("value", ""),
                "ts": elem.get("ts"),
                "end_ts": elem.get("end_ts"),
            }


# ---------------------------------------------------------------------------
# Raw-line builder
# ---------------------------------------------------------------------------


def build_raw_lines(
    tokens: Iterable[Token],
    *,
    max_chars: int = 38,
    strong_punct: set[str] | None = None,
    **_kwargs,  # soft_punct arg blijft toegestaan voor backwards-compat
) -> List[SubtitleLine]:
    """Converteer *tokens* naar voorlopige :class:`SubtitleLine`-objecten."""
    strong_punct = strong_punct or STRONG_PUNCT
    token_list = list(tokens)

    lines: List[SubtitleLine] = []
    buffer = ""
    start_time: float | None = None
    end_time: float | None = None

    def _flush() -> None:
        nonlocal buffer, start_time, end_time
        if buffer and start_time is not None and end_time is not None:
            text = _PUNCT_NORMALIZE_RE.sub(r"\1 ", buffer.strip())
            text = _SPACE_RE.sub(" ", text).rstrip()
            lines.append(
                SubtitleLine(
                    start=timedelta(seconds=float(start_time)),
                    end=timedelta(seconds=float(end_time)),
                    text=text,
                )
            )
        buffer = ""
        start_time = None
        end_time = None

    for idx, tok in enumerate(token_list):
        typ = tok["type"]
        val = str(tok.get("value", ""))
        val_strip = val.strip()

        # --------------------------------------------------------------
        # 1. Harde char-limiet
        # --------------------------------------------------------------
        if val and len(buffer) + len(val) > max_chars:
            _flush()

        # --------------------------------------------------------------
        # 2. Append token + timestamps
        # --------------------------------------------------------------
        if typ == "text":
            if start_time is None and tok.get("ts") is not None:
                start_time = float(tok["ts"])
            if tok.get("end_ts") is not None:
                end_time = float(tok["end_ts"])
            buffer += val

        elif typ == "punct":
            buffer += val
            if start_time is None and tok.get("ts") is not None:
                start_time = float(tok["ts"])
            if tok.get("end_ts") is not None:
                end_time = float(tok["end_ts"])

        else:  # onbekend token-type
            continue

        # --------------------------------------------------------------
        # 2b. Flush na Bijbel-vers-referentie
        # --------------------------------------------------------------
        if _BIBLE_REF_RE.search(buffer):
            _flush()
            continue  # buffer is leeg – ga door met volgend token

        # --------------------------------------------------------------
        # 3. Standaard flush-triggers
        # --------------------------------------------------------------
        next_tok = token_list[idx + 1] if idx + 1 < len(token_list) else None
        next_val = str(next_tok.get("value", "")).strip() if next_tok else ""
        next_is_dot_punct = bool(
            next_tok
            and next_tok.get("type") == "punct"
            and next_val.startswith(".")
        )
        is_ellipsis_prefix = typ == "punct" and val_strip == "." and next_is_dot_punct

        is_multi_dot_punct = bool(val_strip) and set(val_strip) == {"."} and len(val_strip) > 1

        flush_now = (
            len(buffer) >= max_chars
            or (
                typ == "punct"
                and val_strip
                and (
                    (
                        ((val_strip in strong_punct) and not is_ellipsis_prefix)
                        or is_multi_dot_punct
                    )
                    or "\n" in val
                )
            )
        )
        if flush_now:
            _flush()

    _flush()  # trailing buffer
    return lines
