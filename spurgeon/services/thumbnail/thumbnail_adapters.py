"""Default provider, renderer and repository adapters for thumbnails."""

from __future__ import annotations

import base64
import hashlib
import io
import logging
from functools import lru_cache
from pathlib import Path

from openai import OpenAI, OpenAIError
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

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
    IntentCardParseError,
    ThumbnailIntentCard,
    normalize_clip_reading_text,
    parse_thumbnail_intent_card,
)
from .thumbnail_prompting import get_thumbnail_intent_card_prompt_template
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

THUMBNAIL_PREMIUM_FONT_PATHS = (
    "C:/Users/jfden/daily_light/input/CormorantGaramond-SemiBold.ttf",
    "C:/Users/jfden/daily_light/input/CormorantGaramond-Medium.ttf",
)
THUMBNAIL_FALLBACK_FONT_NAME = "DejaVuSans-Bold.ttf"

THUMBNAIL_TEXT_PRIMARY_GOLD = "#D6B24C"
THUMBNAIL_TEXT_ALT_GOLD = "#CFAF5E"
THUMBNAIL_TEXT_STROKE_HEX = "#111111"
THUMBNAIL_TEXT_STROKE_ALPHA = 56
THUMBNAIL_TEXT_STROKE_WIDTH_CAP = 2


class OpenAIIntentCardProvider(IntentCardProvider):
    """OpenAI-backed provider for thumbnail intent cards."""

    def __init__(self, client: OpenAI, settings: Settings) -> None:
        self.client = client
        self.model = settings.thumbnail_intent_card_model
        self.temperature = settings.thumbnail_intent_card_temperature
        self.prompt_version = settings.thumbnail_prompt_version

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
                    {"role": "system", "content": get_thumbnail_intent_card_prompt_template(self.prompt_version)},
                    {"role": "user", "content": user_message},
                ],
            )
        except OpenAIError as exc:
            raise IntentCardError(
                f"OpenAI thumbnail intent-card call failed: {getattr(exc, 'message', exc)}"
            ) from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise IntentCardError("Received empty choices in thumbnail intent-card response")

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        normalized = self._normalize_message_content(content)
        if not normalized:
            raise IntentCardError("Received empty thumbnail intent-card output")

        logger.debug("thumbnail_pipeline.intent_card_raw_output=%r", normalized)
        try:
            return parse_thumbnail_intent_card(normalized)
        except IntentCardParseError as exc:
            raise IntentCardError(str(exc)) from exc

    @staticmethod
    def _normalize_message_content(content: object) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_fragments: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        text_fragments.append(text.strip())
                    continue
                text = getattr(block, "text", None)
                if isinstance(text, str) and text.strip():
                    text_fragments.append(text.strip())

            return "\n".join(text_fragments).strip()

        return ""


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

        font = self._load_font(layout.font_size)
        shadow_color = (0, 0, 0, THUMBNAIL_TEXT_SHADOW_ALPHA)
        stroke_color = (*ImageColor.getrgb(THUMBNAIL_TEXT_STROKE_HEX), THUMBNAIL_TEXT_STROKE_ALPHA)
        stroke_width = min(layout.stroke_width, THUMBNAIL_TEXT_STROKE_WIDTH_CAP)

        self._draw_tracked_multiline_text(
            draw=draw,
            position=(
                text_position[0] + layout.shadow_offset[0],
                text_position[1] + layout.shadow_offset[1],
            ),
            text=layout.text,
            font=font,
            fill=shadow_color,
            tracking=layout.tracking,
            line_spacing_px=line_spacing(layout.font_size),
            stroke_width=0,
            stroke_fill=None,
        )

        self._draw_tracked_multiline_text(
            draw=draw,
            position=text_position,
            text=layout.text,
            font=font,
            fill=THUMBNAIL_TEXT_PRIMARY_GOLD,
            tracking=layout.tracking,
            line_spacing_px=line_spacing(layout.font_size),
            stroke_width=stroke_width,
            stroke_fill=stroke_color,
        )

        logger.debug(
            "thumbnail_pipeline.text_layout original=%r rendered=%r layout=%s font_size=%s tracking=%s text_bbox=%s text_box=%s",
            text,
            display_text,
            f"{layout.line_count}-line",
            layout.font_size,
            layout.tracking,
            (
                *text_position,
                text_position[0] + layout.block_size[0],
                text_position[1] + layout.block_size[1],
            ),
            (text_box.x, text_box.y, text_box.width, text_box.height),
        )
        return canvas

    @staticmethod
    def _draw_tracked_multiline_text(
        *,
        draw: ImageDraw.ImageDraw,
        position: tuple[int, int],
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        fill: str | tuple[int, int, int, int],
        tracking: int,
        line_spacing_px: int,
        stroke_width: int,
        stroke_fill: str | tuple[int, int, int, int] | None,
    ) -> None:
        x, y = position
        current_y = y
        for line in text.split("\n") or [""]:
            current_x = x
            for character in line:
                draw.text(
                    (current_x, current_y),
                    character,
                    font=font,
                    fill=fill,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill,
                )
                bbox = draw.textbbox((0, 0), character, font=font, stroke_width=stroke_width)
                current_x += max(0, bbox[2] - bbox[0]) + tracking

            line_bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
            line_height = max(0, line_bbox[3] - line_bbox[1])
            current_y += line_height + line_spacing_px

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        return self._resolve_font(self.settings.thumbnail_font_path, size)

    @staticmethod
    @lru_cache(maxsize=64)
    def _resolve_font(
        font_path: str | None, size: int
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        font_candidates = [
            *(candidate for candidate in THUMBNAIL_PREMIUM_FONT_PATHS if candidate),
        ]
        if font_path:
            font_candidates.insert(0, font_path)

        for candidate in font_candidates:
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                logger.debug("Thumbnail font candidate unavailable: %s", candidate)

        try:
            return ImageFont.truetype(THUMBNAIL_FALLBACK_FONT_NAME, size=size)
        except OSError:
            logger.warning(
                "Fallback font %s is unavailable. Falling back to Pillow default bitmap font.",
                THUMBNAIL_FALLBACK_FONT_NAME,
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

    def get_by_fingerprint(self, fingerprint: str) -> Path | None:
        fingerprint_dir = self.output_dir / ".fingerprints"
        digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        marker = fingerprint_dir / f"{digest}.txt"
        try:
            if not marker.exists():
                return None
            candidate_name = marker.read_text(encoding="utf-8").strip()
            if not candidate_name:
                return None
            candidate = self.output_dir / candidate_name
            if candidate.exists():
                return candidate
            return None
        except OSError as exc:
            raise StorageError("Failed to inspect thumbnail fingerprint cache") from exc

    def save(self, slug: str, image: Image.Image, *, fingerprint: str | None = None) -> Path:
        destination = self.output_dir / f"{slug}.jpg"
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").save(destination, format="JPEG", quality=95)

            if fingerprint:
                fingerprint_dir = self.output_dir / ".fingerprints"
                fingerprint_dir.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
                marker = fingerprint_dir / f"{digest}.txt"
                marker.write_text(destination.name, encoding="utf-8")

            return destination
        except OSError as exc:
            raise StorageError(f"Failed to save thumbnail for slug '{slug}'") from exc
