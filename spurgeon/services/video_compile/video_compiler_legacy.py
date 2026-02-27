"""Video compilation utilities built on top of FFmpeg."""

from __future__ import annotations

import logging
import numbers
import shlex
import subprocess
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from spurgeon.config.settings import Settings
from spurgeon.models import RawAsset, Reading
from spurgeon.services.subtitles.builder import build_subtitles

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

    def chunk_slug(self, slug: str, idx: int) -> str:
        prefix = f"{slug}_{self.slug_suffix}" if self.slug_suffix else slug
        return f"{prefix}_chunk{idx:02d}"

    def temp_slug(self, slug: str) -> str:
        return f"{slug}_{self.slug_suffix}" if self.slug_suffix else slug

    def output_filename(self, slug: str) -> str:
        suffix = self.filename_suffix or ""
        return f"{slug}{suffix}.mp4"


WIDE_VIDEO = VideoVariant(
    name="wide",
    filter_chain=(
        "scale=-2:1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black"
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

def _escape_concat_path(path: Path) -> str:
    """Return a path escaped for safe usage in FFmpeg concat list files."""

    escaped = path.as_posix().replace("\\", "\\\\").replace("'", "\\'")
    return escaped


def _escape_subtitle_path(path: Path) -> str:
    """Return a path escaped for use inside the FFmpeg ``subtitles`` filter."""

    escaped = path.as_posix()
    escaped = escaped.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return escaped


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
        self.chunks_dir = base / "chunks"
        self.temp_dir = base / "temp"
        self.videos_dir = base / "videos"
        self.images_dir = base / "images"

        for directory in (self.chunks_dir, self.temp_dir, self.videos_dir):
            directory.mkdir(parents=True, exist_ok=True)

        font = getattr(settings, "subtitle_font", "Arial")
        size = getattr(settings, "subtitle_size", 24)
        extra = getattr(settings, "subtitle_style_extra", "")
        self.subtitle_style = f"FontName={font},FontSize={size}"
        if extra:
            self.subtitle_style += f",{extra}"

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
        return self.chunks_dir / f"{slug}.mp4"

    def _build_clip_assets(self, slug: str, assets_raw: Sequence[RawAsset]) -> list[ClipAsset]:
        assets: list[ClipAsset] = []
        for idx, raw in enumerate(assets_raw, start=1):
            try:
                image_raw, duration_raw = raw[0], raw[1]
            except (IndexError, TypeError) as exc:
                raise ValueError(f"Invalid raw asset at position {idx} for {slug}: {raw!r}") from exc

            image_path = Path(image_raw)
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image asset for {slug} chunk {idx}: {image_path}")

            try:
                duration_val = float(duration_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid duration for {slug} chunk {idx}: {duration_raw!r}") from exc

            if duration_val <= 0:
                raise ValueError(f"Duration must be positive for {slug} chunk {idx}: {duration_val}")

            assets.append(ClipAsset(image=image_path, duration=duration_val, idx=idx))

        if not assets:
            raise ValueError(f"No clip assets provided for {slug}")

        return assets

    def compile_image_clip(
        self, slug: str, image: Path, duration: float, variant: VideoVariant
    ) -> Path:
        output_path = self._clip_path(slug)
        if not self._needs_update(output_path, [image]):
            return output_path

        cmd = [
            self.ffmpeg_cmd,
            "-y",
            "-loop",
            "1",
            "-i",
            str(image),
            "-vf",
            variant.filter_chain,
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
        _log(logging.INFO, "Generated clip - %s", output_path)
        return output_path

    def _write_concat_manifest(self, clips: Sequence[Path], concat_file: Path) -> None:
        with concat_file.open("w", encoding="utf-8") as handle:
            for clip in clips:
                try:
                    abs_path = clip.resolve(strict=True)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(f"Concat input missing: {clip}") from exc
                handle.write(f"file '{_escape_concat_path(abs_path)}'\n")

    def concat_clips(self, clips: Sequence[Path], out_path: Path) -> Path:
        if not clips:
            raise ValueError("concat_clips requires at least one clip path")
        if not self._needs_update(out_path, clips):
            return out_path

        concat_file = self.temp_dir / f"{out_path.stem}_concat.txt"
        try:
            self._write_concat_manifest(clips, concat_file)
        except Exception:
            concat_file.unlink(missing_ok=True)
            raise

        cmd = [
            self.ffmpeg_cmd,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(out_path),
        ]
        try:
            self._run(cmd)
        finally:
            concat_file.unlink(missing_ok=True)
        _log(logging.INFO, "Concatenated - %s", out_path)
        return out_path

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


    def burn_subtitles(self, av_path: Path, srt_path: Path, out_path: Path) -> Path:
        if not self._needs_update(out_path, [av_path, srt_path]):
            return out_path

        srt_filter = _escape_subtitle_path(srt_path)
        vf = f"subtitles='{srt_filter}':force_style='{self.subtitle_style}'"
        cmd = [
            self.ffmpeg_cmd,
            "-y",
            "-i",
            str(av_path),
            "-vf",
            vf,
            "-c:v",
            self.codec,
            "-c:a",
            "copy",
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

        clip_assets = self._build_clip_assets(slug, assets_raw)

        clip_results: list[Tuple[int, Path]] = []
        worker_count = min(self.max_workers, len(clip_assets))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    self.compile_image_clip,
                    variant.chunk_slug(slug, asset.idx),
                    asset.image,
                    asset.duration,
                    variant,
                ): asset.idx
                for asset in clip_assets
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    clip_path = future.result()
                except Exception as exc:  # pragma: no cover - validated by runtime
                    _log(
                        logging.ERROR,
                        "Clip generation failed for %s chunk %02d: %s",
                        slug,
                        idx,
                        exc,
                    )
                    raise
                clip_results.append((idx, clip_path))

        clip_paths = [path for _, path in sorted(clip_results, key=lambda item: item[0])]

        temp_slug = variant.temp_slug(slug)
        video_only = self.temp_dir / f"{temp_slug}_video_only.mp4"
        self.concat_clips(clip_paths, video_only)

        if audio_path is None:
            audio_path = Path(self.settings.output_dir) / "audio" / f"{slug}.mp3"
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        av_muxed = self.temp_dir / f"{temp_slug}_av.mp4"
        self.merge_audio_video(video_only, audio_path, av_muxed)

        final_srt = Path(self.settings.output_dir) / "subtitles" / f"{slug}.srt"
        if not final_srt.exists():
            build_subtitles(reading)

        output_video = videos_base / variant.output_filename(slug)
        self.burn_subtitles(av_muxed, final_srt, output_video)

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
