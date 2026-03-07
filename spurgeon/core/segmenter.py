"""Utilities for splitting :class:`~spurgeon.models.Reading` instances into chunks."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import List

from spurgeon.models import Reading, SegmentBlock

logger = logging.getLogger(__name__)

_WORD_PATTERN = re.compile(r"\S+")
_WHITESPACE_RE = re.compile(r"\s+")
_PARAGRAPH_RE = re.compile(r"\r?\n\s*\r?\n")
_TRAILING_STRIP = "\"'”’›»)}]>）］】」》〉"
_STRONG_PUNCT = {".", "!", "?", ";", ":"}
_SOFT_PUNCT = {",", "–", "—", "-"}


def _normalise_ascii(value: object) -> str:
    """Return ``value`` coerced to ASCII-only text for logging."""

    text = str(value)
    normalised = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalised if ord(ch) < 128)


def _log(level: int, message: str, *args: object) -> None:
    """Log with non-ASCII characters removed from *message* and *args*."""

    ascii_message = _normalise_ascii(message)
    ascii_args = tuple(_normalise_ascii(arg) for arg in args)
    logger.log(level, ascii_message, *ascii_args)


@dataclass(slots=True)
class WordSpan:
    """A slice of the original text together with trailing whitespace metadata."""

    start: int
    end: int
    text: str
    gap_after: str = ""

    def counts_as_word(self) -> bool:
        return any(ch.isalpha() or ch.isdigit() for ch in self.text)


def _build_word_spans(text: str) -> List[WordSpan]:
    spans = [WordSpan(start=m.start(), end=m.end(), text=m.group()) for m in _WORD_PATTERN.finditer(text)]
    for idx in range(len(spans) - 1):
        spans[idx].gap_after = text[spans[idx].end : spans[idx + 1].start]
    return spans


def _normalise_segment_text(raw: str) -> str:
    return _WHITESPACE_RE.sub(" ", raw).strip()


def _last_meaningful_char(token: str) -> str:
    stripped = token.rstrip(_TRAILING_STRIP)
    return stripped[-1] if stripped else ""


def _has_paragraph_break(gap: str) -> bool:
    return bool(_PARAGRAPH_RE.search(gap))


def _has_linebreak(gap: str) -> bool:
    return "\n" in gap or "\r" in gap


def _should_break_after(
    span: WordSpan,
    next_span: WordSpan | None,
    word_count: int,
    *,
    soft_limit: int,
    min_words: int,
    break_threshold: int,
) -> bool:
    if word_count < min_words:
        return False

    gap = span.gap_after
    if _has_paragraph_break(gap):
        return True

    if word_count < soft_limit:
        return False

    last_char = _last_meaningful_char(span.text)
    if last_char in _STRONG_PUNCT and word_count >= break_threshold:
        return True
    if last_char in _SOFT_PUNCT and (_has_linebreak(gap) or not gap.strip()):
        return True
    if next_span is None:
        return True
    return False


def _emit_block(
    *,
    blocks: List[SegmentBlock],
    spans: List[WordSpan],
    text: str,
    reading: Reading,
    index: int,
) -> int:
    if not spans:
        return index

    start_offset = spans[0].start
    end_offset = spans[-1].end
    block_text = _normalise_segment_text(text[start_offset:end_offset])
    if not block_text:
        return index

    blocks.append(
        SegmentBlock(
            reading_date=reading.date,
            reading_type=reading.reading_type,
            index=index,
            start=start_offset,
            end=end_offset,
            text=block_text,
        )
    )
    return index + 1


def block_all(reading: Reading, max_words: int, *, min_words: int = 1) -> List[SegmentBlock]:
    """Split ``reading`` into sequential :class:`SegmentBlock` objects.

    The implementation prefers breaking on sentence-ending punctuation or explicit
    paragraph gaps while still honouring the ``max_words`` hard limit.  ``min_words``
    prevents overly short fragments caused by stray punctuation in the source text.
    """

    if max_words <= 0:
        raise ValueError(f"max_words must be > 0, got {max_words}")

    min_words = max(1, min_words)

    text = reading.text
    if not text:
        _log(logging.WARNING, "Empty text in Reading %s, skipping segmentation.", reading)
        return []

    spans = _build_word_spans(text)
    if not spans:
        _log(logging.WARNING, "No words found in Reading %s.", reading)
        return []

    blocks: List[SegmentBlock] = []
    chunk_spans: List[WordSpan] = []
    word_count = 0
    index = 1

    soft_limit = max(min_words, max_words - max(1, max_words // 5))
    break_threshold = max(min_words, min(soft_limit, max(2, max_words // 2)))

    for idx, span in enumerate(spans):
        next_span = spans[idx + 1] if idx + 1 < len(spans) else None
        chunk_spans.append(span)
        if span.counts_as_word():
            word_count += 1

        should_flush = word_count >= max_words or _should_break_after(
            span,
            next_span,
            word_count,
            soft_limit=soft_limit,
            min_words=min_words,
            break_threshold=break_threshold,
        )

        if should_flush:
            index = _emit_block(
                blocks=blocks,
                spans=chunk_spans,
                text=text,
                reading=reading,
                index=index,
            )
            chunk_spans = []
            word_count = 0

    if chunk_spans:
        _emit_block(
            blocks=blocks,
            spans=chunk_spans,
            text=text,
            reading=reading,
            index=index,
        )

    return blocks