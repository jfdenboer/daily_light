# time_chunker.py
"""
Splits word-aligned SRT into time-based video chunks.

Handles
-------
* Timestamp parsing from SRT
* Filtering invalid entries
* Sequential chunking into fixed-length slices
* Merging of too-short segments

Output model: TimeChunk (start, end, duration, text)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _to_ascii(value: object) -> str:
    """Return *value* normalised to ASCII characters only."""

    text = str(value)
    normalised = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalised if ord(ch) < 128)


def _warn(message: str, *args: object) -> None:
    """Emit a warning with non-ASCII characters stripped out."""

    formatted = message
    if args:
        try:
            formatted = message % args
        except Exception:  # pragma: no cover - defensive guard
            joined = " ".join(str(arg) for arg in args)
            formatted = f"{message} {joined}".strip()

    ascii_message = _to_ascii(formatted)
    logger.warning(ascii_message)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _SRTEntry:
    """Lightweight representation of a single SRT cue."""

    seq: int
    start: timedelta
    end: timedelta
    text: str


def parse_timestamp(t: str) -> timedelta:
    """
    Convert 'HH:MM:SS,fff' or 'HH:MM:SS.fff' → timedelta.
    """
    t = t.replace(",", ".")
    hours_str, minutes_str, sec_frac = t.split(":")
    if "." in sec_frac:
        seconds_str, frac_str = sec_frac.split(".", 1)
    else:
        seconds_str, frac_str = sec_frac, "0"
    hours = int(hours_str)
    minutes = int(minutes_str)
    seconds = int(seconds_str)
    milliseconds = int(frac_str.ljust(3, "0")[:3])
    return timedelta(
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        milliseconds=milliseconds,
    )


def _split_blocks(raw: str) -> Iterable[List[str]]:
    """Yield cleaned line-blocks from raw SRT content."""

    for block in re.split(r"\r?\n\r?\n", raw.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines:
            yield lines


def _collapse_text(lines: Iterable[str]) -> str:
    """Join SRT text lines with normalised whitespace."""

    text = " ".join(lines)
    return re.sub(r"\s+", " ", text).strip()


def parse_srt(srt_path: Path) -> List[_SRTEntry]:
    """Read a forced-alignment SRT file into structured cue entries."""

    text = srt_path.read_text(encoding="utf-8-sig")
    entries: List[_SRTEntry] = []

    for lines in _split_blocks(text):
        if len(lines) < 3:
            _warn("Skipping malformed SRT block in %s: %s", srt_path, " | ".join(lines))
            continue

        try:
            seq = int(lines[0])
        except ValueError:
            _warn("Skipping block with non-integer sequence in %s: %s", srt_path, lines[0])
            continue

        try:
            start_raw, end_raw = [part.strip() for part in lines[1].split("-->", 1)]
        except ValueError:
            _warn("Skipping block %d with invalid timestamp row in %s: %s", seq, srt_path, lines[1])
            continue

        try:
            start = parse_timestamp(start_raw)
            end = parse_timestamp(end_raw)
        except ValueError as exc:
            _warn("Skipping block %d due to invalid timestamp (%s)", seq, exc)
            continue

        text_content = _collapse_text(lines[2:])
        if not text_content:
            _warn("Skipping block %d without text in %s", seq, srt_path)
            continue

        if end <= start:
            _warn(
                "Skipping entry %d with non-positive duration (%s - %s)",
                seq,
                start,
                end,
            )
            continue

        entries.append(_SRTEntry(seq=seq, start=start, end=end, text=text_content))

    if not entries:
        _warn("No valid SRT entries found in %s", srt_path)

    return entries


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


class TimeChunk(BaseModel):
    """
    Represents a group of words spanning a time interval.

    index      – sequential chunk number (1-based)
    start_time – chunk start (timedelta)
    end_time   – chunk end   (timedelta)
    duration   – chunk length in seconds (float)
    text       – concatenated words inside the interval
    """

    index: int
    start_time: timedelta
    end_time: timedelta
    duration: float
    text: str


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _flush_chunk(
    *,
    chunks: List[TimeChunk],
    current_entries: List[_SRTEntry],
    chunk_start: Optional[timedelta],
    index: int,
) -> Tuple[List[_SRTEntry], Optional[timedelta], int]:
    """Write the current buffered entries into ``chunks`` if present."""

    if not current_entries:
        return current_entries, chunk_start, index

    start_time = chunk_start if chunk_start is not None else current_entries[0].start
    end_time = current_entries[-1].end
    text = " ".join(entry.text for entry in current_entries).strip()
    duration = (end_time - start_time).total_seconds()

    chunks.append(
        TimeChunk(
            index=index,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            text=text,
        )
    )

    return [], None, index + 1


def build_time_chunks(
    srt_path: Path,
    seconds_per_chunk: float = 15.0,
) -> List[TimeChunk]:
    """
    Sequentially split SRT entries into ``seconds_per_chunk``-long slices.
    """

    if seconds_per_chunk <= 0:
        raise ValueError("seconds_per_chunk must be greater than zero")

    entries = parse_srt(srt_path)
    if not entries:
        return []

    chunks: List[TimeChunk] = []
    current_entries: List[_SRTEntry] = []
    chunk_start: Optional[timedelta] = None
    index = 1

    for entry in entries:
        if not current_entries:
            chunk_start = entry.start
            current_entries.append(entry)
            continue

        assert chunk_start is not None  # for type-checkers
        projected_duration = (entry.end - chunk_start).total_seconds()
        if projected_duration > seconds_per_chunk:
            current_entries, chunk_start, index = _flush_chunk(
                chunks=chunks,
                current_entries=current_entries,
                chunk_start=chunk_start,
                index=index,
            )
            chunk_start = entry.start

        current_entries.append(entry)

    current_entries, _, index = _flush_chunk(
        chunks=chunks,
        current_entries=current_entries,
        chunk_start=chunk_start,
        index=index,
    )

    return chunks


def merge_short_chunks(
    chunks: List[TimeChunk],
    min_duration: float,
) -> List[TimeChunk]:
    """
    Merge any chunk shorter than ``min_duration`` into its neighbour.
    """
    if not chunks:
        return []

    merged: List[TimeChunk] = []
    for chunk in chunks:
        if merged and chunk.duration < min_duration:
            prev = merged[-1]
            new_end = max(prev.end_time, chunk.end_time)
            new_text = f"{prev.text} {chunk.text}".strip()
            new_duration = (new_end - prev.start_time).total_seconds()
            merged[-1] = TimeChunk(
                index=prev.index,
                start_time=prev.start_time,
                end_time=new_end,
                duration=new_duration,
                text=new_text,
            )
        else:
            merged.append(chunk)

    # handle too-short first chunk
    if len(merged) > 1 and merged[0].duration < min_duration:
        first = merged.pop(0)
        second = merged.pop(0)
        new_end = second.end_time
        new_text = f"{first.text} {second.text}".strip()
        new_duration = (new_end - first.start_time).total_seconds()
        merged.insert(
            0,
            TimeChunk(
                index=first.index,
                start_time=first.start_time,
                end_time=new_end,
                duration=new_duration,
                text=new_text,
            ),
        )

    # re-index
    return [
        chunk.model_copy(update={"index": i})
        for i, chunk in enumerate(merged, start=1)
    ]