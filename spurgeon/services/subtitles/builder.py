# builder.py

from __future__ import annotations

"""spurgeon.services.subtitles.builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Façade‑laag* voor het ondertitel‑sub‑package.
Verzamelt alle losse, puur functionele submodules (io, tokenizer,
merger) tot een eenvoudig publiek API dat elders in de pipeline
wordt gebruikt.

Publieke functies
=================
- ``build_subtitles(reading)`` → genereert/overschrijft ``*.srt`` en
  retourneert de :class:`SubtitleLine`‑lijst.
- ``build_image_chunks(reading)`` → lijst
  :class:`ImageChunk` op basis van woordgebaseerde `.words.srt`.

Alle parameters hebben defaults uit
:pyclass:`spurgeon.config.settings.Settings`, maar kunnen worden
overschreven in unit‑tests.
"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

from spurgeon.config.settings import Settings, load_settings
from spurgeon.models import Reading
from spurgeon.services.subtitles.caption_models import SubtitleLine, ImageChunk
from spurgeon.services.subtitles.io import load_rev_json, write_srt_file
from spurgeon.services.subtitles.tokenizer import iter_tokens, build_raw_lines
from spurgeon.services.subtitles.merger import merge_micro_lines
from spurgeon.core.time_chunker import build_time_chunks, merge_short_chunks


__all__ = [
    "SubtitleLine",
    "ImageChunk",
    "build_subtitles",
    "build_image_chunks",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_DEFAULT_SETTINGS: Optional[Settings] = None


def _get_settings(explicit: Optional[Settings] = None) -> Settings:
    """Return an explicit ``Settings`` or lazily load the module default."""

    if explicit is not None:
        return explicit

    global _DEFAULT_SETTINGS
    if _DEFAULT_SETTINGS is None:
        _DEFAULT_SETTINGS = load_settings()
    return _DEFAULT_SETTINGS


def _output_dir(settings: Settings) -> Path:
    return Path(settings.output_dir)


def _subtitles_dir(settings: Settings) -> Path:
    return _output_dir(settings) / "subtitles"


def _json_dir(settings: Settings) -> Path:
    return _subtitles_dir(settings) / "json"


def _words_srt_path(reading: Reading, settings: Settings) -> Path:
    return _subtitles_dir(settings) / "words" / f"{reading.slug}.words.srt"


def _get_json_path(reading: Reading, settings: Settings) -> Path:
    """Pad naar het Rev.ai‑JSON voor *reading* (slug.rev.json)."""
    return _json_dir(settings) / f"{reading.slug}.rev.json"


def _parse_srt_timestamp(value: str) -> timedelta:
    """Parse ``HH:MM:SS,fff`` (met komma of punt) naar :class:`datetime.timedelta`."""

    hours_str, minutes_str, rest = value.split(":")
    if "," in rest:
        seconds_str, frac_str = rest.split(",", 1)
    elif "." in rest:
        seconds_str, frac_str = rest.split(".", 1)
    else:
        seconds_str, frac_str = rest, "0"

    return timedelta(
        hours=int(hours_str),
        minutes=int(minutes_str),
        seconds=int(seconds_str),
        milliseconds=int(frac_str.ljust(3, "0")[:3]),
    )


def _load_existing_srt(path: Path) -> List[SubtitleLine]:
    """Lees reeds bestaande SRT en giet terug naar ``SubtitleLine``'s."""

    if not path.exists():
        raise FileNotFoundError(path)

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    subtitle_lines: List[SubtitleLine] = []
    for block in raw.split("\n\n"):
        if not block.strip():
            continue

        rows = [row.strip() for row in block.splitlines() if row.strip()]
        if len(rows) < 3:
            raise ValueError(f"Ongeldig SRT-blok in {path}: {block!r}")

        try:
            times_row = rows[1]
            start_raw, end_raw = [item.strip() for item in times_row.split("-->", 1)]
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Ongeldige tijdstempel in {path}: {block!r}") from exc

        start = _parse_srt_timestamp(start_raw)
        end = _parse_srt_timestamp(end_raw)
        text = "\n".join(rows[2:])
        subtitle_lines.append(SubtitleLine(start=start, end=end, text=text))

    return subtitle_lines


def build_subtitle_lines_from_rev_json(
    reading: Reading,
    *,
    settings: Optional[Settings] = None,
    max_chars: Optional[int] = None,
    min_chars: Optional[int] = None,
    min_duration: Optional[float] = None,
    hard_max_chars: Optional[int] = None,
) -> List[SubtitleLine]:
    """Parset Rev.ai‑alignment‑output naar *samengevoegde* caption‑regels."""
    cfg = _get_settings(settings)

    resolved_max_chars = max_chars if max_chars is not None else getattr(cfg, "srt_max_chars", 38)
    resolved_min_chars = min_chars if min_chars is not None else getattr(cfg, "min_subtitle_chars", 10)
    resolved_min_duration = (
        min_duration if min_duration is not None else getattr(cfg, "min_subtitle_duration", 1.0)
    )
    resolved_hard_max = (
        hard_max_chars if hard_max_chars is not None else getattr(cfg, "srt_hard_max_chars", 76)
    )

    json_path = _get_json_path(reading, cfg)
    data = load_rev_json(json_path)

    tokens = iter_tokens(data)
    raw_lines = build_raw_lines(tokens, max_chars=resolved_max_chars)
    merged = merge_micro_lines(
        raw_lines,
        min_chars=resolved_min_chars,
        min_duration=resolved_min_duration,
        hard_max_chars=resolved_hard_max,
    )
    return merged


# ---------------------------------------------------------------------------
# Publieke façade‑API
# ---------------------------------------------------------------------------

def build_subtitles(
    reading: Reading,
    *,
    overwrite: bool = False,
    settings: Optional[Settings] = None,
) -> List[SubtitleLine]:
    """Genereer (en schrijf) line‑based SRT voor *reading*.

    Als het doelbestand al bestaat en *overwrite* == False,
    wordt het bestand behouden en worden de bestaande regels teruggegoten.
    """
    cfg = _get_settings(settings)
    srt_path = _subtitles_dir(cfg) / f"{reading.slug}.srt"

    if srt_path.exists() and not overwrite:
        logger.debug("%s: SRT bestaat reeds – laden bestaande regels", srt_path.name)
        return _load_existing_srt(srt_path)

    subs = build_subtitle_lines_from_rev_json(reading, settings=cfg)
    write_srt_file(subs, srt_path)
    logger.info("%s: %d subtitle lines geschreven", srt_path.name, len(subs))
    return subs


def build_image_chunks(
    reading: Reading,
    *,
    settings: Optional[Settings] = None,
) -> List[ImageChunk]:
    """Bouw image‑chunks (±10s) op basis van woordgebaseerde SRT."""
    cfg = _get_settings(settings)
    words_srt = _words_srt_path(reading, cfg)
    if not words_srt.exists():
        raise FileNotFoundError(f"Woordgebaseerde SRT ontbreekt: {words_srt}")

    raw = build_time_chunks(words_srt, seconds_per_chunk=cfg.time_chunk_duration)
    merged = merge_short_chunks(raw, min_duration=cfg.min_chunk_duration)
    return [
        ImageChunk(chunk.index, chunk.start_time, chunk.end_time, chunk.text)
        for chunk in merged
    ]