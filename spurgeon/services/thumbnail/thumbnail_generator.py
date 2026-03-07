"""Generate YouTube thumbnails with OpenAI Images and local text compositing."""

from __future__ import annotations

import base64
import io
import logging
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
from .thumbnail_prompting import (
    THUMBNAIL_BACKGROUND_LINE,
    THUMBNAIL_COMPOSITION_LINE,
    THUMBNAIL_CONSTRAINTS_LINE,
    THUMBNAIL_LIGHTING_LINE,
    THUMBNAIL_PALETTE_LINE,
    THUMBNAIL_STYLE_LINE,
    THUMBNAIL_SUBJECT_LINE,
    build_thumbnail_prompt,
)

logger = logging.getLogger(__name__)

THUMBNAIL_CANVAS_SIZE = (1280, 720)


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
        self.intent_card_model = settings.thumbnail_intent_card_model
        self.intent_card_temperature = settings.thumbnail_intent_card_temperature

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
        logger.info("thumbnail_pipeline.start slug=%s", reading.slug)

        intent_card = self._generate_thumbnail_intent_card(reading, text)
        logger.info(
            "thumbnail_pipeline.intent_card core_tension=%s emotional_tone=%s visual_motif=%s scene_direction=%s avoid=%s",
            intent_card.core_tension,
            intent_card.emotional_tone,
            intent_card.visual_motif,
            intent_card.scene_direction,
            intent_card.avoid,
        )

        prompt = self._build_prompt(reading, text, intent_card)
        logger.info("thumbnail_pipeline.image_prompt_ready theme=%s", text)

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
            self._compose_thumbnail(
                image_bytes=image_bytes, text=text, destination=thumbnail_path
            )
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
            raise ThumbnailGenerationError(
                "No thumbnail image data returned from OpenAI"
            )

        b64_payload = getattr(data[0], "b64_json", None)
        if not b64_payload:
            raise ThumbnailGenerationError(
                "No base64 payload returned from OpenAI thumbnail response"
            )

        try:
            return base64.b64decode(b64_payload)
        except (ValueError, TypeError) as exc:
            raise ThumbnailGenerationError("Invalid base64 thumbnail payload") from exc

    def _generate_thumbnail_intent_card(
        self,
        reading: Reading,
        thumbnail_text: str,
    ) -> ThumbnailIntentCard:
        cleaned_reading = normalize_clip_reading_text(reading.text, max_chars=2000)

        user_message = (
            f"Devotional type: {reading.reading_type.value}\n"
            f"Thumbnail theme: {thumbnail_text}\n"
            "Reading text:\n"
            f"{cleaned_reading}"
        )

        try:
            raw_output = retry_with_backoff(
                func=lambda: self._call_openai_thumbnail_intent_card(
                    user_message, user=reading.slug
                ),
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
                context=f"thumbnail_intent_card_{reading.slug}",
            )
        except Exception as exc:
            raise ThumbnailGenerationError(str(exc)) from exc

        logger.debug("thumbnail_pipeline.intent_card_raw_output=%r", raw_output)
        try:
            return parse_thumbnail_intent_card(raw_output)
        except IntentCardParseError as exc:
            raise ThumbnailGenerationError(str(exc)) from exc

    def _call_openai_thumbnail_intent_card(
        self, user_message: str, *, user: str | None = None
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.intent_card_model,
                temperature=self.intent_card_temperature,
                max_completion_tokens=220,
                user=user,
                messages=[
                    {"role": "system", "content": THUMBNAIL_INTENT_CARD_DEVMSG},
                    {"role": "user", "content": user_message},
                ],
            )
        except OpenAIError as exc:
            raise ThumbnailGenerationError(
                f"OpenAI thumbnail intent-card call failed: {getattr(exc, 'message', exc)}"
            ) from exc

        content = response.choices[0].message.content or ""
        normalized = content.strip()
        if not normalized:
            raise ThumbnailGenerationError(
                "Received empty thumbnail intent-card output"
            )
        return normalized

    def _compose_thumbnail(
        self, *, image_bytes: bytes, text: str, destination: Path
    ) -> None:
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                canvas = ImageOps.fit(
                    source.convert("RGB"),
                    THUMBNAIL_CANVAS_SIZE,
                    method=Image.Resampling.LANCZOS,
                )
        except OSError as exc:
            raise ThumbnailGenerationError(
                "Failed to decode generated thumbnail image"
            ) from exc

        draw = ImageDraw.Draw(canvas, "RGBA")
        display_text = normalize_thumbnail_display_text(text)
        text_box = calculate_text_layout_box(canvas.size)
        layout_engine = ThumbnailTextLayoutEngine(draw, self._load_font)
        layout = layout_engine.select_text_layout(display_text, text_box)
        text_position = resolve_text_position(layout, text_box)

        shadow_color = (0, 0, 0, THUMBNAIL_TEXT_SHADOW_ALPHA)
        draw.multiline_text(
            (text_position[0] + layout.shadow_offset[0], text_position[1] + layout.shadow_offset[1]),
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
            (*text_position, text_position[0] + layout.block_size[0], text_position[1] + layout.block_size[1]),
            (text_box.x, text_box.y, text_box.width, text_box.height),
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.convert("RGB").save(destination, format="JPEG", quality=95)

    @staticmethod
    def _build_prompt(
        reading: Reading,
        thumbnail_text: str,
        intent_card: ThumbnailIntentCard,
    ) -> str:
        return build_thumbnail_prompt(reading, thumbnail_text, intent_card)

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

    def _find_existing_thumbnail(self, slug: str) -> Path | None:
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = self.output_dir / f"{slug}{ext}"
            if candidate.exists():
                return candidate
        return None


__all__ = [
    "ThumbnailGenerator",
    "ThumbnailGenerationError",
    "ThumbnailIntentCard",
    "THUMBNAIL_STYLE_LINE",
    "THUMBNAIL_SUBJECT_LINE",
    "THUMBNAIL_BACKGROUND_LINE",
    "THUMBNAIL_COMPOSITION_LINE",
    "THUMBNAIL_CONSTRAINTS_LINE",
    "THUMBNAIL_PALETTE_LINE",
    "THUMBNAIL_LIGHTING_LINE",
]
