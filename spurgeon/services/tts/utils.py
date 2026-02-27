"""Utilities for ElevenLabs text-to-speech integrations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

SENTENCE_REGEX = re.compile(r"[^.!?…]+[.!?…]?", re.DOTALL)


def _split_sentences(text: str) -> Iterable[str]:
    for match in SENTENCE_REGEX.finditer(text):
        segment = match.group().strip()
        if segment:
            yield segment


def _split_long_segment(segment: str, hard_limit: int) -> list[str]:
    segments: list[str] = []
    remaining = segment
    while len(remaining) > hard_limit:
        split_pos = _find_split_position(remaining, hard_limit)
        head = remaining[:split_pos].rstrip()
        tail = remaining[split_pos:].lstrip()
        if head:
            segments.append(head)
        remaining = tail
    if remaining:
        segments.append(remaining)
    return segments


def _find_split_position(text: str, hard_limit: int) -> int:
    soft_targets = [" ", ",", ";", ":"]
    max_pos = min(hard_limit, len(text))
    min_pos = max(max_pos - 300, 1)
    for idx in range(max_pos, min_pos - 1, -1):
        if text[idx - 1 : idx] in soft_targets:
            return idx
    return max_pos


def chunk_text_for_v3(text: str, hard_limit: int = 3000, target: int = 2800) -> list[str]:
    """Split a text into Eleven v3 compatible chunks.

    The implementation keeps audio tags intact and prefers sentence boundaries.
    """

    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= hard_limit:
        return [cleaned]

    chunks: list[str] = []
    current = ""

    for segment in _split_sentences(cleaned):
        candidate = f"{current} {segment}".strip() if current else segment
        if len(candidate) <= target:
            current = candidate
            continue

        if current:
            chunks.append(current)
        if len(segment) > hard_limit:
            chunks.extend(_split_long_segment(segment, hard_limit))
            current = ""
        else:
            current = segment

    if current:
        chunks.append(current)

    # Guard against oversize chunks caused by very long tokens
    normalized: list[str] = []
    for chunk in chunks:
        if len(chunk) <= hard_limit:
            normalized.append(chunk)
            continue
        normalized.extend(_split_long_segment(chunk, hard_limit))
    return normalized


def normalise_language_code(code: Optional[str], model_id: str) -> Optional[str]:
    del model_id
    if not code:
        return None
    return code.strip().lower()


@dataclass(frozen=True)
class OutputFormatInfo:
    container: str
    sample_rate: int
    bitrate: int | None = None


def parse_output_format(fmt: str) -> OutputFormatInfo:
    if fmt.startswith("mp3_"):
        _, sample, bitrate = fmt.split("_")
        return OutputFormatInfo("mp3", int(sample), int(bitrate))
    if fmt.startswith("pcm_"):
        _, sample = fmt.split("_")
        return OutputFormatInfo("pcm", int(sample), None)
    if fmt.startswith("ulaw_"):
        _, sample = fmt.split("_")
        return OutputFormatInfo("ulaw", int(sample), None)
    if fmt.startswith("alaw_"):
        _, sample = fmt.split("_")
        return OutputFormatInfo("alaw", int(sample), None)
    if fmt.startswith("opus_"):
        _, sample = fmt.split("_")
        return OutputFormatInfo("opus", int(sample), 96)
    raise ValueError(f"Unsupported ElevenLabs output format: {fmt}")