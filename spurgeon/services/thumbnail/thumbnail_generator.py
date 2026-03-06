"""Generate YouTube thumbnails with OpenAI Images and local text compositing."""

from __future__ import annotations

import base64
import io
import logging
import textwrap
from pathlib import Path

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from PIL import Image, ImageDraw, ImageFont, ImageOps

from spurgeon.config.settings import Settings
from spurgeon.models import Reading
from spurgeon.utils.retry_utils import retry_with_backoff

logger = logging.getLogger(__name__)

THUMBNAIL_IMAGE_SYSTEM_PROMPT = (
    "Create a cinematic, realistic, high-clarity YouTube thumbnail background for a Christian devotional. "
    "No text, letters, logos, watermarks, or symbols. "
    "Use one clear focal subject with depth and atmosphere, readable on small screens. "
    "Leave clean negative space on the left/center-left so overlaid text remains highly legible. "
    "Avoid clutter, busy textures, extreme blur, fantasy illustration styles, or artificial CGI looks."
)


class ThumbnailGenerationError(RuntimeError):
    """Raised when thumbnail generation fails in a recoverable way."""


class ThumbnailGenerator:
    """Create thumbnails for readings using OpenAI Images and local compositing."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.output_dir = Path(settings.output_dir) / "thumbnails"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = settings.thumbnail_enabled
        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate_thumbnail(
        self,
        reading: Reading,
        *,
        title: str,
        hero_image: Path | None = None,
        thumbnail_text: str | None = None,
    ) -> Path | None:
        if not self.enabled:
            logger.info("Thumbnail generation disabled by configuration")
            return None

        cached = self._find_existing_thumbnail(reading.slug)
        if cached:
            logger.debug("Reusing existing thumbnail %s", cached.name)
            return cached

        text = thumbnail_text or title
        prompt = self._build_prompt(reading, text)

        try:
            image_bytes = retry_with_backoff(
                func=lambda: self._call_openai(prompt, user=reading.slug),
                max_retries=self.settings.thumbnail_max_retries,
                backoff=self.settings.thumbnail_retry_backoff,
                error_types=(
                    APIError,
                    RateLimitError,
                    APIConnectionError,
                    APITimeoutError,
                    OpenAIError,
                    ThumbnailGenerationError,
                ),
                context=f"thumbnail_image_{reading.slug}",
            )
            thumbnail_path = self.output_dir / f"{reading.slug}.jpg"
            self._compose_thumbnail(image_bytes=image_bytes, text=text, destination=thumbnail_path)
            logger.info("Saved thumbnail: %s", thumbnail_path.name)
            return thumbnail_path
        except Exception as exc:
            raise ThumbnailGenerationError(str(exc)) from exc

    def _call_openai(self, prompt: str, *, user: str | None = None) -> bytes:
        try:
            response = self.client.images.generate(
                model=self.settings.thumbnail_image_model,
                prompt=prompt,
                n=1,
                size=self.settings.thumbnail_image_size,
                user=user,
                quality=self.settings.thumbnail_image_quality,
                background=self.settings.thumbnail_image_background,
            )
        except OpenAIError as exc:
            raise ThumbnailGenerationError(
                f"OpenAI thumbnail image call failed: {getattr(exc, 'message', exc)}"
            ) from exc

        data = getattr(response, "data", None)
        if not data:
            raise ThumbnailGenerationError("No thumbnail image data returned from OpenAI")

        b64_payload = getattr(data[0], "b64_json", None)
        if not b64_payload:
            raise ThumbnailGenerationError("No base64 payload returned from OpenAI thumbnail response")

        try:
            return base64.b64decode(b64_payload)
        except (ValueError, TypeError) as exc:
            raise ThumbnailGenerationError("Invalid base64 thumbnail payload") from exc

    def _compose_thumbnail(self, *, image_bytes: bytes, text: str, destination: Path) -> None:
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                canvas = ImageOps.fit(source.convert("RGB"), (1280, 720), method=Image.Resampling.LANCZOS)
        except OSError as exc:
            raise ThumbnailGenerationError("Failed to decode generated thumbnail image") from exc

        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_overlay.rectangle((0, 0, 760, 720), fill=(0, 0, 0, 105))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

        draw = ImageDraw.Draw(canvas)
        wrapped_text = self._wrap_text_for_thumbnail(text)
        font = self._load_font()

        text_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=8, stroke_width=5)
        text_height = text_bbox[3] - text_bbox[1]
        y = max(40, int((720 - text_height) / 2))

        draw.multiline_text(
            (74, y),
            wrapped_text,
            font=font,
            fill="#FFFFFF",
            spacing=8,
            stroke_width=5,
            stroke_fill="#000000",
            align="left",
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination, format="JPEG", quality=95)

    def _build_prompt(self, reading: Reading, thumbnail_text: str) -> str:
        excerpt = " ".join(reading.text.split())[:700]
        return (
            f"{THUMBNAIL_IMAGE_SYSTEM_PROMPT}\n\n"
            f"Devotional type: {reading.reading_type.value}.\n"
            f"Date: {reading.date.isoformat()}.\n"
            f"Thumbnail headline theme: {thumbnail_text}.\n"
            f"Devotional excerpt: {excerpt}"
        )

    @staticmethod
    def _wrap_text_for_thumbnail(text: str) -> str:
        normalised = " ".join(text.replace("\n", " ").split())
        wrapped = textwrap.wrap(normalised, width=14)
        if not wrapped:
            return "Daily Light"
        return "\n".join(wrapped[:3])

    def _load_font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if self.settings.thumbnail_font_path:
            try:
                return ImageFont.truetype(self.settings.thumbnail_font_path, size=118)
            except OSError:
                logger.warning(
                    "Configured THUMBNAIL_FONT_PATH could not be loaded (%s). Falling back to default font.",
                    self.settings.thumbnail_font_path,
                )
        return ImageFont.load_default()

    def _find_existing_thumbnail(self, slug: str) -> Path | None:
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = self.output_dir / f"{slug}{ext}"
            if candidate.exists():
                return candidate
        return None


__all__ = [
    "ThumbnailGenerator",
    "ThumbnailGenerationError",
    "THUMBNAIL_IMAGE_SYSTEM_PROMPT",
]
