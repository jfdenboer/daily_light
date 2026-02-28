"""Video compilation utilities built on top of FFmpeg."""

from __future__ import annotations

import logging
import numbers
import re
import shlex
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

from spurgeon.config.settings import Settings
from spurgeon.models import RawAsset, Reading
from spurgeon.services.subtitles.builder import build_subtitles
from spurgeon.services.subtitles.io import convert_srt_to_ass

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ClipAsset:
    """Light-weight representation of a single still-image clip."""

    image: Path
    duration: float
    idx: int


@dataclass(slots=True, frozen=True)
class VideoVariant:
    """Configuration bundle describing how a video variant should be rendered."""

    name: str
    filter_chain: str
    slug_suffix: str = ""
    filename_suffix: str = ""

    def clip_slug(self, slug: str, idx: int) -> str:
        prefix = f"{slug}_{self.slug_suffix}" if self.slug_suffix else slug
        return f"{prefix}_clip{idx:02d}"

    def temp_slug(self, slug: str) -> str:
        return f"{slug}_{self.slug_suffix}" if self.slug_suffix else slug

    def output_filename(self, slug: str) -> str:
        suffix = self.filename_suffix or ""
        return f"{slug}{suffix}.mp4"


WIDE_VIDEO = VideoVariant(
    name="wide",
    filter_chain=(
        "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080"
    ),
)

SQUARE_SHORT_VIDEO = VideoVariant(
    name="square",
    filter_chain=(
        "scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080"
    ),
    slug_suffix="short",
    filename_suffix="_short",
)


def _normalise_for_log(value: object) -> str:
    """Return *value* collapsed to plain ASCII for logging purposes."""

    text = str(value)
    normalised = unicodedata.normalize("NFKD", text)
    return normalised.encode("ascii", "ignore").decode("ascii")


def _normalise_arg_for_log(value: object) -> object:
    """Normalise a logging argument whilst preserving numeric types."""

    if isinstance(value, numbers.Real):
        return value
    return _normalise_for_log(value)


def _log(level: int, message: str, *args: object, **kwargs: object) -> None:
    """Log ``message`` and ``args`` after removing non-ASCII characters."""

    ascii_message = _normalise_for_log(message)
    ascii_args = tuple(_normalise_arg_for_log(arg) for arg in args)
    logger.log(level, ascii_message, *ascii_args, **kwargs)


def _escape_subtitle_path(path: Path) -> str:
    """Return a path escaped for use inside the FFmpeg ``ass`` filter."""

    try:
        resolved = path.resolve()
    except FileNotFoundError:
        resolved = path.absolute()

    try:
        relative = resolved.relative_to(Path.cwd())
    except ValueError:
        relative = None

    escaped_path = relative if relative is not None else resolved
    escaped = escaped_path.as_posix()
    escaped = escaped.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    escaped = escaped.replace(" ", "\\ ")
    return escaped


_STYLE_KV_RE = re.compile(r"\s*([A-Za-z]+)\s*=\s*([^,]+)\s*")


def _parse_style_extra(extra: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not extra:
        return out
    for part in extra.split(","):
        match = _STYLE_KV_RE.fullmatch(part.strip())
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        out[key] = value
    return out

class VideoCompiler:
    """Compile generated assets into a final MP4 video file."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ffmpeg_cmd = settings.ffmpeg_cmd
        self.fps = settings.video_fps
        self.codec = settings.video_codec
        self.audio_bitrate = settings.audio_bitrate
        configured_workers = getattr(settings, "video_parallel_workers", 4)
        try:
            self.max_workers = int(configured_workers)
        except (TypeError, ValueError) as exc:
            raise ValueError("video_parallel_workers must be an integer") from exc
        if self.max_workers < 1:
            raise ValueError("video_parallel_workers must be at least 1")
        self.ffmpeg_timeout = getattr(settings, "video_timeout", None)
        self.retention_h = settings.temp_retention_hours

        base = Path(settings.output_dir)
        self.clips_dir = base / "clips"
        self.temp_dir = base / "temp"
        self.videos_dir = base / "videos"
        self.images_dir = base / "images"

        for directory in (self.clips_dir, self.temp_dir, self.videos_dir):
            directory.mkdir(parents=True, exist_ok=True)

        _log(
            logging.INFO,
            "VideoCompiler ready - ffmpeg=%s workers=%s retention=%sh",
            self.ffmpeg_cmd,
            self.max_workers,
            self.retention_h,
        )

    @staticmethod
    def _needs_update(target: Path, sources: Iterable[Path]) -> bool:
        """Return ``True`` when *target* is older than any of *sources*."""

        if not target.exists():
            return True
        target_mtime = target.stat().st_mtime
        try:
            return any(path.stat().st_mtime > target_mtime for path in sources)
        except FileNotFoundError:
            return True

    def _run(self, cmd: Sequence[str], timeout: Optional[int] = None) -> None:
        """Execute an FFmpeg command and raise ``RuntimeError`` on failure."""

        cmd_list = list(cmd)
        cmd_str = " ".join(shlex.quote(part) for part in cmd_list)
        _log(logging.DEBUG, "Running command: %s", cmd_str)
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd_list,
                capture_output=True,
                text=True,
                timeout=timeout or self.ffmpeg_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(f"FFmpeg timeout after {exc.timeout}s\n{cmd_str}") from exc

        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            raise RuntimeError(
                "FFmpeg exit %s (%.1fs)\n%s\nstdout:\n%s\n\nstderr:\n%s"
                % (
                    proc.returncode,
                    elapsed,
                    cmd_str,
                    proc.stdout,
                    proc.stderr,
                )
            )
        _log(logging.DEBUG, "FFmpeg ok (%.1fs)", elapsed)

    def _clip_path(self, slug: str) -> Path:
        return self.clips_dir / f"{slug}.mp4"

    def _build_clip_assets(self, slug: str, assets_raw: Sequence[RawAsset]) -> list[ClipAsset]:
        assets: list[ClipAsset] = []
        for idx, raw in enumerate(assets_raw, start=1):
            try:
                image_raw, duration_raw = raw[0], raw[1]
            except (IndexError, TypeError) as exc:
                raise ValueError(f"Invalid raw asset at position {idx} for {slug}: {raw!r}") from exc

            image_path = Path(image_raw)
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image asset for {slug} asset {idx}: {image_path}")

            try:
                duration_val = float(duration_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid duration for {slug} asset {idx}: {duration_raw!r}") from exc

            if duration_val <= 0:
                raise ValueError(f"Duration must be positive for {slug} asset {idx}: {duration_val}")

            assets.append(ClipAsset(image=image_path, duration=duration_val, idx=idx))

        if not assets:
            raise ValueError(f"No clip assets provided for {slug}")

        return assets

    def _single_background_asset(self, slug: str, assets_raw: Sequence[RawAsset]) -> ClipAsset:
        """Validate and return the single visual asset used for the full timeline."""

        assets = self._build_clip_assets(slug, assets_raw)
        if len(assets) != 1:
            raise ValueError(
                f"Single-image video compositing expects exactly 1 asset for {slug}, got {len(assets)}"
            )
        return assets[0]

    def _wide_zoom_filter(self, duration: float) -> str:
        start = float(getattr(self.settings, "video_zoom_wide_start", 1.0))
        end = float(getattr(self.settings, "video_zoom_wide_end", 1.10))
        if duration <= 0:
            raise ValueError("duration must be positive")
        if end < start:
            start, end = end, start

        frame_count = max(int(round(duration * self.fps)), 1)
        frame_span = max(frame_count - 1, 1)

        return (
            "scale=1920:1080:force_original_aspect_ratio=increase,"
            f"zoompan=z='{start:.6f}+({end - start:.6f})*(0.5-0.5*cos(PI*on/{frame_span}))':"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            "d=1:s=1920x1080:fps={fps}"
        ).format(fps=self.fps)

    def _clip_filter_for_variant(self, variant: VideoVariant, duration: float) -> str:
        is_wide = variant.name == WIDE_VIDEO.name
        zoom_enabled = bool(getattr(self.settings, "video_zoom_wide_enabled", True))
        if is_wide and zoom_enabled:
            return self._wide_zoom_filter(duration)
        return variant.filter_chain

    def compile_image_clip(
        self, slug: str, image: Path, duration: float, variant: VideoVariant
    ) -> Path:
        output_path = self._clip_path(slug)
        if not self._needs_update(output_path, [image]):
            return output_path

        clip_filter = self._clip_filter_for_variant(variant, duration)
        cmd = [
            self.ffmpeg_cmd,
            "-y",
            "-loop",
            "1",
            "-i",
            str(image),
            "-vf",
            clip_filter,
            "-c:v",
            self.codec,
            "-tune",
            "stillimage",
            "-r",
            str(self.fps),
            "-pix_fmt",
            "yuv420p",
            "-t",
            f"{duration:.3f}",
            str(output_path),
        ]
        self._run(cmd)
        _log(
            logging.INFO,
            "Generated clip (%s, zoom=%s) - %s",
            variant.name,
            bool(
                variant.name == WIDE_VIDEO.name
                and getattr(self.settings, "video_zoom_wide_enabled", True)
            ),
            output_path,
        )
        return output_path

    def merge_audio_video(self, video: Path, audio: Path, out_path: Path) -> Path:
        if not self._needs_update(out_path, [video, audio]):
            return out_path

        # ``-shortest`` previously truncated the narration by clipping the audio stream
        # to the marginally shorter video stream.  We now map the streams explicitly
        # and let FFmpeg emit the full audio track so the video keeps playing until
        # the narration actually finishes.
        cmd = [
            self.ffmpeg_cmd,
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            self.audio_bitrate,
            str(out_path),
        ]
        self._run(cmd)
        _log(logging.INFO, "Muxed A/V - %s", out_path)
        return out_path

    @staticmethod
    def _ass_resolution_for_variant(variant: Optional[VideoVariant]) -> tuple[int, int]:
        if variant and variant.name == SQUARE_SHORT_VIDEO.name:
            return 1080, 1080
        return 1920, 1080

    @staticmethod
    def _ass_path_for_variant(
        srt_path: Path, variant: Optional[VideoVariant]
    ) -> Path:
        if variant is None:
            return srt_path.with_suffix(".ass")
        return srt_path.with_name(f"{srt_path.stem}_{variant.name}.ass")

    def _style_parameters_for_variant(
        self,
        variant: Optional[VideoVariant],
    ) -> dict[str, object]:
        params: dict[str, object] = {
            "fontname": "Inter",
            "fontsize": 48,
            "primary": "&H00FFFFFF",
            "outline_col": "&HAA000000",
            "border_style": 1,
            "outline": 2.6,
            "shadow": 0,
            "alignment": 5,
            "margin_l": 230,
            "margin_r": 230,
            "margin_v": 40,
        }

        suffix = ""
        if variant:
            if variant.name == WIDE_VIDEO.name:
                suffix = "_wide"
            elif variant.name == SQUARE_SHORT_VIDEO.name:
                suffix = "_short"
                params["margin_l"] = params["margin_r"] = 120

        def _setting_value(attr: str) -> object:
            if suffix:
                specific = getattr(self.settings, f"{attr}{suffix}", None)
                if specific not in (None, ""):
                    return specific
            value = getattr(self.settings, attr, None)
            if value in (None, ""):
                return None
            return value

        font_value = _setting_value("subtitle_font")
        if font_value:
            params["fontname"] = str(font_value)

        size_value = _setting_value("subtitle_size")
        if size_value is not None:
            try:
                params["fontsize"] = int(size_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"subtitle_size{suffix or ''} moet een integer zijn") from exc

        extra_value = _setting_value("subtitle_style_extra")
        extra_pairs = _parse_style_extra(str(extra_value)) if extra_value else {}

        if "PrimaryColour" in extra_pairs:
            params["primary"] = extra_pairs["PrimaryColour"]
        if "OutlineColour" in extra_pairs:
            params["outline_col"] = extra_pairs["OutlineColour"]

        def _assign_int(key: str, target: str) -> None:
            if key not in extra_pairs:
                return
            try:
                params[target] = int(extra_pairs[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} moet een integer zijn") from exc

        def _assign_float(key: str, target: str) -> None:
            if key not in extra_pairs:
                return
            try:
                params[target] = float(extra_pairs[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} moet een float zijn") from exc

        _assign_int("BorderStyle", "border_style")
        _assign_float("Outline", "outline")
        _assign_int("Shadow", "shadow")
        _assign_int("Alignment", "alignment")
        _assign_int("MarginL", "margin_l")
        _assign_int("MarginR", "margin_r")
        _assign_int("MarginV", "margin_v")

        return params

    def _ensure_ass_subtitles(
        self,
        srt_path: Path,
        variant: Optional[VideoVariant],
    ) -> Path:
        ass_path = self._ass_path_for_variant(srt_path, variant)
        play_res_x, play_res_y = self._ass_resolution_for_variant(variant)
        if self._needs_update(ass_path, [srt_path]):
            style_params = self._style_parameters_for_variant(variant)
            _log(
                logging.DEBUG,
                "Converting SRT->ASS (%s) with params %s",
                ass_path,
                style_params,
            )
            convert_srt_to_ass(
                srt_path,
                ass_path,
                play_res_x=play_res_x,
                play_res_y=play_res_y,
                **style_params,
            )
        return ass_path

    def burn_subtitles(
        self,
        av_path: Path,
        srt_path: Path,
        out_path: Path,
        *,
        variant: Optional[VideoVariant] = None,
    ) -> Path:
        if not self._needs_update(out_path, [av_path, srt_path]):
            return out_path

        ass_path = self._ensure_ass_subtitles(srt_path, variant)
        ass_path_escaped = _escape_subtitle_path(ass_path)

        default_line_spacing = int(getattr(self.settings, "subtitle_line_spacing", 6))
        if variant and variant.name == SQUARE_SHORT_VIDEO.name:
            variant_spacing = getattr(self.settings, "subtitle_line_spacing_short", None)
        else:
            variant_spacing = getattr(self.settings, "subtitle_line_spacing_wide", None)

        if variant_spacing in (None, 0):
            line_spacing = default_line_spacing
        else:
            try:
                line_spacing = int(variant_spacing)
            except (TypeError, ValueError) as exc:
                raise ValueError("subtitle_line_spacing values must be integers") from exc

        subtitles_filter = (
            f"subtitles='{ass_path_escaped}':force_style='LineSpacing={line_spacing}'"
        )
        _log(
            logging.INFO,
            "Rendering subtitles from %s with LineSpacing=%spx",
            ass_path,
            line_spacing,
        )

        cmd = [
            self.ffmpeg_cmd,
            "-y",
            "-i",
            str(av_path),
            "-vf",
            subtitles_filter,
            "-c:v",
            self.codec,
            "-c:a",
            "copy",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        self._run(cmd)
        _log(logging.INFO, "Subtitles burned - %s", out_path)
        return out_path

    def cleanup_temp(self) -> None:
        threshold = datetime.now().timestamp() - self.retention_h * 3600
        removed = 0
        for candidate in self.temp_dir.glob("*"):
            try:
                if candidate.stat().st_mtime < threshold:
                    candidate.unlink(missing_ok=True)
                    removed += 1
            except FileNotFoundError:
                continue
        if removed:
            _log(logging.DEBUG, "Temp cleanup - removed %s old files", removed)

    def compile(
        self,
        reading: Reading,
        assets_raw: Sequence[RawAsset],
        videos_base: Path,
        *,
        audio_path: Optional[Path] = None,
        variant: Optional[VideoVariant] = None,
    ) -> Path:
        """Build the final MP4 for ``reading``."""

        start = time.perf_counter()
        slug = reading.slug

        variant = variant or WIDE_VIDEO

        clip_asset = self._single_background_asset(slug, assets_raw)
        temp_slug = variant.temp_slug(slug)
        video_only = self.compile_image_clip(
            variant.clip_slug(slug, clip_asset.idx),
            clip_asset.image,
            clip_asset.duration,
            variant,
        )

        if audio_path is None:
            audio_ext = getattr(self.settings, "elevenlabs_audio_extension", ".mp3")
            audio_path = Path(self.settings.output_dir) / "audio" / f"{slug}{audio_ext}"
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        av_muxed = self.temp_dir / f"{temp_slug}_av.mp4"
        self.merge_audio_video(video_only, audio_path, av_muxed)

        final_srt = Path(self.settings.output_dir) / "subtitles" / f"{slug}.srt"
        if not final_srt.exists():
            build_subtitles(reading)

        output_video = videos_base / variant.output_filename(slug)
        self.burn_subtitles(av_muxed, final_srt, output_video, variant=variant)

        self.cleanup_temp()

        elapsed = float(time.perf_counter() - start)
        _log(
            logging.INFO,
            "Video complete (%s) -> %s (%.1fs)",
            variant.name,
            output_video,
            elapsed,
        )
        return output_video


__all__ = [
    "VideoCompiler",
    "ClipAsset",
    "VideoVariant",
    "WIDE_VIDEO",
    "SQUARE_SHORT_VIDEO",
]
