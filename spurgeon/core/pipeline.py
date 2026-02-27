"""spurgeon/core/pipeline.py – orchestrates full text→video flow."""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional

from spurgeon.config.settings import Settings
from spurgeon.core.parser import load_all_readings
from spurgeon.models import RawAsset, Reading
from spurgeon.services.alignment.rev_aligner import RevAligner
from spurgeon.services.image_gen.image_generator import ImageGenerator
from spurgeon.services.subtitles.builder import build_subtitles
from spurgeon.services.tts.speech_synthesizer import SpeechSynthesizer
from spurgeon.services.thumbnail import (
    ThumbnailGenerationError,
    ThumbnailGenerator,
    ThumbnailTextGenerationError,
    ThumbnailTextGenerator,
)
from spurgeon.services.video_compile.video_compiler import (
    SQUARE_SHORT_VIDEO,
    WIDE_VIDEO,
    VideoCompiler,
)
from spurgeon.services.youtube.generate_description import DescriptionGenerator
from spurgeon.services.youtube.generate_tags import TagsGenerator
from spurgeon.services.youtube.generate_title import TitleGenerator
from spurgeon.services.youtube.scheduler import next_publish_datetime
from spurgeon.services.youtube.uploader import YouTubeUploader
from spurgeon.utils.logging_setup import init_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderArtifacts:
    """Pipeline artifacts needed for publication steps."""

    video_path: Path
    short_video_path: Path
    hero_image: Optional[Path]


@dataclass(frozen=True)
class PublicationPayload:
    """Metadata needed for YouTube publication."""

    title: str
    description: str
    tags: list[str]
    publish_at: Optional[datetime]
    thumbnail_path: Optional[Path]


def _ensure_output_dirs(base: Path) -> None:
    """Create the standard output sub-directories if they do not exist."""

    for sub in ("audio", "images", "subtitles", "videos", "temp"):
        (base / sub).mkdir(parents=True, exist_ok=True)


def _iter_years(start: Optional[date], end: Optional[date]) -> Optional[Iterable[int]]:
    """Return the relevant years that need to be parsed from the source texts."""

    if start and end:
        if end < start:
            raise ValueError("end_date must not be earlier than start_date")
        return range(start.year, end.year + 1)
    if start:
        return range(start.year, start.year + 1)
    return None


def _load_readings_for_range(
    input_dir: Path,
    start: Optional[date],
    end: Optional[date],
) -> List[Reading]:
    """Load and chronologically sort readings within the requested date window."""

    years = _iter_years(start, end)
    if years is None:
        readings = load_all_readings(input_dir)
    else:
        readings = []
        for year in years:
            readings.extend(load_all_readings(input_dir, year=year))

    if start or end:
        readings = [
            reading
            for reading in readings
            if (start is None or reading.date >= start)
            and (end is None or reading.date <= end)
        ]

    readings.sort(key=lambda r: (r.date, r.reading_type.value))

    unique_by_slug: OrderedDict[str, Reading] = OrderedDict()
    for reading in readings:
        unique_by_slug.setdefault(reading.slug, reading)

    return list(unique_by_slug.values())


def _parse_srt_timestamp_to_seconds(value: str) -> float:
    """Parse ``HH:MM:SS,mmm`` or ``HH:MM:SS.mmm`` into seconds."""

    hours_str, minutes_str, rest = value.split(":")
    if "," in rest:
        seconds_str, millis_str = rest.split(",", 1)
    elif "." in rest:
        seconds_str, millis_str = rest.split(".", 1)
    else:
        seconds_str, millis_str = rest, "0"

    millis = int(millis_str.ljust(3, "0")[:3])
    return int(hours_str) * 3600 + int(minutes_str) * 60 + int(seconds_str) + millis / 1000


def _words_srt_duration_seconds(words_srt_path: Path) -> float:
    """Return the end timestamp of the final words-SRT cue in seconds."""

    if not words_srt_path.exists():
        raise FileNotFoundError(words_srt_path)

    last_time_row: str | None = None
    for line in words_srt_path.read_text(encoding="utf-8").splitlines():
        if "-->" in line:
            last_time_row = line

    if not last_time_row:
        raise ValueError(f"No cue timestamps found in words SRT: {words_srt_path}")

    _, end_raw = [item.strip() for item in last_time_row.split("-->", 1)]
    return _parse_srt_timestamp_to_seconds(end_raw)


def _prepare_render_artifacts(
    reading: Reading,
    *,
    settings: Settings,
    out_base: Path,
    tts: SpeechSynthesizer,
    aligner: RevAligner,
    img_gen: ImageGenerator,
    video_comp: VideoCompiler,
) -> Optional[RenderArtifacts]:
    """Create/reuse media artifacts for a reading."""

    wide_video = out_base / "videos" / WIDE_VIDEO.output_filename(reading.slug)
    short_video = out_base / "videos" / SQUARE_SHORT_VIDEO.output_filename(reading.slug)
    wide_exists = wide_video.exists()
    short_exists = short_video.exists()

    video_path = wide_video
    short_video_path = short_video
    hero_image: Optional[Path] = None
    image_ext = settings.image_file_extension

    if wide_exists and short_exists:
        logger.info(
            "Videos %s and %s already exist – reusing cached files for upload",
            wide_video.name,
            short_video.name,
        )
    else:
        audio_path = tts.synthesize(reading)
        words_srt_path = aligner.align(reading.slug, reading.text, audio_path)
        build_subtitles(reading, settings=settings)

        assets_raw: List[RawAsset] = []
        duration = _words_srt_duration_seconds(words_srt_path)
        single_asset = img_gen.generate_single_image_for_reading(reading, duration=duration)
        if single_asset is not None and single_asset[0].exists():
            assets_raw.append(single_asset)
        else:
            logger.warning("%s: single-image generation produced no asset – skipping", reading.slug)
            return None
        if not assets_raw:
            logger.warning("%s: no images generated – skipping", reading.slug)
            return None

        hero_image = assets_raw[0][0]

        if wide_exists:
            logger.info(
                "%s: wide video already exists – skipping render and using cached file",
                wide_video.name,
            )
        else:
            video_path = video_comp.compile(
                reading,
                assets_raw,
                out_base / "videos",
                audio_path=audio_path,
                variant=WIDE_VIDEO,
            )

        if short_exists:
            logger.info(
                "%s: short video already exists – skipping render and using cached file",
                short_video.name,
            )
        else:
            short_video_path = video_comp.compile(
                reading,
                assets_raw,
                out_base / "videos",
                audio_path=audio_path,
                variant=SQUARE_SHORT_VIDEO,
            )

    if hero_image is None:
        single_path = out_base / "images" / f"{reading.slug}.{image_ext}"
        if single_path.exists():
            hero_image = single_path

    return RenderArtifacts(
        video_path=video_path,
        short_video_path=short_video_path,
        hero_image=hero_image,
    )


def _build_publication_payload(
    reading: Reading,
    *,
    hero_image: Optional[Path],
    title_gen: TitleGenerator,
    desc_gen: DescriptionGenerator,
    tags_gen: TagsGenerator,
    thumb_text_gen: ThumbnailTextGenerator,
    thumb_gen: ThumbnailGenerator,
) -> PublicationPayload:
    """Generate metadata and optional thumbnail for publication."""

    publish_at = next_publish_datetime(reading.date, reading.reading_type)

    try:
        title = title_gen.generate(reading)
    except Exception as exc:
        logger.warning(
            "Title generation failed for %s – fallback used (%s)",
            reading.slug,
            exc,
        )
        title = f"{reading.reading_type.value} Devotional – {reading.date.strftime('%B %d, %Y')}"

    try:
        description = desc_gen.generate(reading, title)
    except Exception as exc:
        logger.warning("Description generation failed for %s (%s)", reading.slug, exc)
        description = ""

    thumbnail_path = None
    try:
        thumbnail_text = thumb_text_gen.generate(reading)
    except (ThumbnailTextGenerationError, Exception) as exc:
        logger.warning(
            "Thumbnail text generation failed for %s (%s)",
            reading.slug,
            exc,
        )
        thumbnail_text = thumb_text_gen.fallback(reading, title)

    try:
        thumbnail_path = thumb_gen.generate_thumbnail(
            reading,
            title=title,
            hero_image=hero_image,
            thumbnail_text=thumbnail_text,
        )
    except ThumbnailGenerationError as exc:
        logger.warning("Thumbnail generation failed for %s (%s)", reading.slug, exc)
        thumbnail_path = None

    try:
        tags = tags_gen.generate(reading)
    except Exception as exc:
        logger.warning(
            "Tag generation failed for %s (%s) – falling back to empty list",
            reading.slug,
            exc,
        )
        tags = []

    return PublicationPayload(
        title=title,
        description=description,
        tags=tags,
        publish_at=publish_at,
        thumbnail_path=thumbnail_path,
    )


def _publish_reading(
    reading: Reading,
    *,
    artifacts: RenderArtifacts,
    payload: PublicationPayload,
    youtube_uploader: YouTubeUploader,
) -> None:
    """Publish both wide and short variants to YouTube."""

    video_id = youtube_uploader.upload_video(
        artifacts.video_path,
        title=payload.title,
        description=payload.description,
        tags=payload.tags,
        publish_at=payload.publish_at,
        thumbnail_path=payload.thumbnail_path,
    )
    logger.info(
        "Video scheduled for %s (%s) – YouTube id=%s, publish_at=%s",
        reading.slug,
        reading.date,
        video_id,
        payload.publish_at.isoformat() if payload.publish_at else "immediate",
    )

    short_video_id = youtube_uploader.upload_video(
        artifacts.short_video_path,
        title=payload.title,
        description=payload.description,
        tags=payload.tags,
        publish_at=payload.publish_at,
        thumbnail_path=None,
    )
    logger.info(
        "Short scheduled for %s (%s) – YouTube id=%s, publish_at=%s",
        reading.slug,
        reading.date,
        short_video_id,
        payload.publish_at.isoformat() if payload.publish_at else "immediate",
    )


def run_pipeline(
    settings: Settings,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    uploader: Optional[YouTubeUploader] = None,
) -> None:
    """Execute the full reading → video → YouTube pipeline.

    Parameters
    ----------
    settings:
        Base configuration object.
    start_date / end_date:
        Optional date window for the readings that should be processed.
    uploader:
        Pre-configured :class:`~spurgeon.services.youtube.uploader.YouTubeUploader`
        instance.  Supplying one allows callers (primarily tests) to inject a
        mock without the pipeline re-initialising API clients.
    """
    effective_settings = settings

    init_logging(effective_settings)

    logger.info("Starting pipeline with settings: %s", effective_settings.model_dump())

    if effective_settings.gcs_credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(effective_settings.gcs_credentials_path)

    out_base = Path(effective_settings.output_dir)
    _ensure_output_dirs(out_base)

    input_dir = Path(effective_settings.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Configured input directory {input_dir} does not exist or is not accessible"
        )

    readings = _load_readings_for_range(input_dir, start_date, end_date)
    if not readings:
        logger.warning("No readings to process after filtering – aborting")
        return
    logger.info("Processing %d readings", len(readings))

    tts = SpeechSynthesizer(effective_settings, output_dir=out_base / "audio")
    aligner = RevAligner(effective_settings)
    img_gen = ImageGenerator(effective_settings)
    video_comp = VideoCompiler(effective_settings)
    title_gen = TitleGenerator(effective_settings)
    desc_gen = DescriptionGenerator(effective_settings)
    thumb_text_gen = ThumbnailTextGenerator(effective_settings)
    thumb_gen = ThumbnailGenerator(effective_settings)
    tags_gen = TagsGenerator(effective_settings)

    youtube_uploader = uploader or YouTubeUploader(
        client_secrets_path=Path("client_secrets.json"),
        token_path=Path("token.json"),
        settings=effective_settings,
    )

    for reading in readings:
        try:
            logger.info("→ %s %s", reading.date, reading.reading_type)

            artifacts = _prepare_render_artifacts(
                reading,
                settings=effective_settings,
                out_base=out_base,
                tts=tts,
                aligner=aligner,
                img_gen=img_gen,
                video_comp=video_comp,
            )
            if artifacts is None:
                continue

            payload = _build_publication_payload(
                reading,
                hero_image=artifacts.hero_image,
                title_gen=title_gen,
                desc_gen=desc_gen,
                tags_gen=tags_gen,
                thumb_text_gen=thumb_text_gen,
                thumb_gen=thumb_gen,
            )

            _publish_reading(
                reading,
                artifacts=artifacts,
                payload=payload,
                youtube_uploader=youtube_uploader,
            )

        except Exception as exc:  # pragma: no cover - defensive orchestration guard
            logger.exception("Processing failed for %s: %s", reading.slug, exc)
