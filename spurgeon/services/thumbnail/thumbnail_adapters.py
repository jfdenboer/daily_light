"""Default provider, renderer and repository adapters for thumbnails."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

from openai import OpenAI, OpenAIError
from PIL import Image, ImageDraw, ImageFont, ImageOps

from spurgeon.config.settings import Settings
from spurgeon.models import Reading

from .thumbnail_contracts import (
    ImageProvider,
    IntentCardProvider,
    ThumbnailRenderer,
    ThumbnailRepository,
)
from .thumbnail_errors import ImageProviderError, IntentCardError, RenderError, StorageError
from .thumbnail_intent_card import (
    THUMBNAIL_INTENT_CARD_DEVMSG,
    IntentCardParseError,
    ThumbnailIntentCard,
    normalize_clip_reading_text,
    parse_thumbnail_intent_card,
)
from .thumbnail_layout import (
    THUMBNAIL_TEXT_SHADOW_ALPHA,
    ThumbnailTextLayoutEngine,
    calculate_text_layout_box,
    line_spacing,
    normalize_thumbnail_display_text,
    resolve_text_position,
)

logger = logging.getLogger(__name__)

THUMBNAIL_CANVAS_SIZE = (1280, 720)


class OpenAIIntentCardProvider(IntentCardProvider):
    """OpenAI-backed provider for thumbnail intent cards."""

    def __init__(self, client: OpenAI, settings: Settings) -> None:
        self.client = client
        self.model = settings.thumbnail_intent_card_model
        self.temperature = settings.thumbnail_intent_card_temperature

    def generate(self, reading: Reading, thumbnail_text: str) -> ThumbnailIntentCard:
        cleaned_reading = normalize_clip_reading_text(reading.text, max_chars=2000)
        user_message = (
            f"Devotional type: {reading.reading_type.value}\n"
            f"Thumbnail theme: {thumbnail_text}\n"
            "Reading text:\n"
            f"{cleaned_reading}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_completion_tokens=220,
                user=reading.slug,
                messages=[
                    {"role": "system", "content": THUMBNAIL_INTENT_CARD_DEVMSG},
                    {"role": "user", "content": user_message},
                ],
            )
        except OpenAIError as exc:
            raise IntentCardError(
                f"OpenAI thumbnail intent-card call failed: {getattr(exc, 'message', exc)}"
            ) from exc

        content = response.choices[0].message.content or ""
        normalized = content.strip()
        if not normalized:
            raise IntentCardError("Received empty thumbnail intent-card output")

        logger.debug("thumbnail_pipeline.intent_card_raw_output=%r", normalized)
        try:
            return parse_thumbnail_intent_card(normalized)
        except IntentCardParseError as exc:
            raise IntentCardError(str(exc)) from exc


class OpenAIImageProvider(ImageProvider):
    """OpenAI-backed provider for thumbnail background images."""

    def __init__(self, client: OpenAI, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def generate(self, prompt: str, *, user: str | None = None) -> bytes:
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
            raise ImageProviderError(
                f"OpenAI thumbnail image call failed: {getattr(exc, 'message', exc)}"
            ) from exc

        data = getattr(response, "data", None)
        if not data:
            raise ImageProviderError("No thumbnail image data returned from OpenAI")

        b64_payload = getattr(data[0], "b64_json", None)
        if not b64_payload:
            raise ImageProviderError(
                "No base64 payload returned from OpenAI thumbnail response"
            )

        try:
            return base64.b64decode(b64_payload)
        except (ValueError, TypeError) as exc:
            raise ImageProviderError("Invalid base64 thumbnail payload") from exc


class PillowThumbnailRenderer(ThumbnailRenderer):
    """Pillow renderer that composites thumbnail text over an image."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def render(self, *, image_bytes: bytes, text: str) -> Image.Image:
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                canvas = ImageOps.fit(
                    source.convert("RGB"),
                    THUMBNAIL_CANVAS_SIZE,
                    method=Image.Resampling.LANCZOS,
                )
        except OSError as exc:
            raise RenderError("Failed to decode generated thumbnail image") from exc

        draw = ImageDraw.Draw(canvas, "RGBA")
        display_text = normalize_thumbnail_display_text(text)
        text_box = calculate_text_layout_box(canvas.size)
        layout_engine = ThumbnailTextLayoutEngine(draw, self._load_font)
        layout = layout_engine.select_text_layout(display_text, text_box)
        text_position = resolve_text_position(layout, text_box)

        shadow_color = (0, 0, 0, THUMBNAIL_TEXT_SHADOW_ALPHA)
        draw.multiline_text(
            (
                text_position[0] + layout.shadow_offset[0],
                text_position[1] + layout.shadow_offset[1],
            ),
            layout.text,
            font=self._load_font(layout.font_size),
            fill=shadow_color,
            spacing=line_spacing(layout.font_size),
            stroke_width=0,
            align="left",
        )

        draw.multiline_text(
            text_position,
            layout.text,
            font=self._load_font(layout.font_size),
            fill="#FFFFFF",
            spacing=line_spacing(layout.font_size),
            stroke_width=layout.stroke_width,
            stroke_fill="#000000",
            align="left",
        )

        logger.debug(
            "thumbnail_pipeline.text_layout original=%r rendered=%r layout=%s font_size=%s text_bbox=%s text_box=%s",
            text,
            display_text,
            f"{layout.line_count}-line",
            layout.font_size,
            (
                *text_position,
                text_position[0] + layout.block_size[0],
                text_position[1] + layout.block_size[1],
            ),
            (text_box.x, text_box.y, text_box.width, text_box.height),
        )
        return canvas

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if self.settings.thumbnail_font_path:
            try:
                return ImageFont.truetype(self.settings.thumbnail_font_path, size=size)
            except OSError:
                logger.warning(
                    "Configured THUMBNAIL_FONT_PATH could not be loaded (%s). Falling back to default font.",
                    self.settings.thumbnail_font_path,
                )

        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
        except OSError:
            logger.warning(
                "Fallback font DejaVuSans-Bold.ttf is unavailable. Falling back to Pillow default bitmap font."
            )
        return ImageFont.load_default()


class FilesystemThumbnailRepository(ThumbnailRepository):
    """Filesystem-backed thumbnail repository."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_existing(self, slug: str) -> Path | None:
        try:
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = self.output_dir / f"{slug}{ext}"
                if candidate.exists():
                    return candidate
            return None
        except OSError as exc:
            raise StorageError(f"Failed to inspect thumbnail cache for slug '{slug}'") from exc

    def save(self, slug: str, image: Image.Image) -> Path:
        destination = self.output_dir / f"{slug}.jpg"
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").save(destination, format="JPEG", quality=95)
            return destination
        except OSError as exc:
            raise StorageError(f"Failed to save thumbnail for slug '{slug}'") from exc
