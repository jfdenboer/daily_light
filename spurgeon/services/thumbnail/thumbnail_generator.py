"""Generate YouTube thumbnails with provider, renderer and repository adapters."""

from __future__ import annotations

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

from spurgeon.config.settings import Settings
from spurgeon.models import Reading
from spurgeon.utils.retry_utils import retry_with_backoff

from .thumbnail_adapters import (
    FilesystemThumbnailRepository,
    OpenAIImageProvider,
    OpenAIIntentCardProvider,
    PillowThumbnailRenderer,
)
from .thumbnail_contracts import (
    ImageProvider,
    IntentCardProvider,
    ThumbnailRenderer,
    ThumbnailRepository,
)
from .thumbnail_intent_card import ThumbnailIntentCard
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


class ThumbnailGenerationError(RuntimeError):
    """Raised when thumbnail generation fails in a recoverable way."""


class ThumbnailGenerator:
    """Create thumbnails for readings using modular provider/renderer/repository adapters."""

    def __init__(
        self,
        settings: Settings,
        *,
        intent_card_provider: IntentCardProvider | None = None,
        image_provider: ImageProvider | None = None,
        renderer: ThumbnailRenderer | None = None,
        repository: ThumbnailRepository | None = None,
    ) -> None:
        self.settings = settings
        self.enabled = settings.thumbnail_enabled

        client = OpenAI(api_key=settings.openai_api_key)
        output_dir = Path(settings.output_dir) / "thumbnails"

        self.intent_card_provider = intent_card_provider or OpenAIIntentCardProvider(
            client, settings
        )
        self.image_provider = image_provider or OpenAIImageProvider(client, settings)
        self.renderer = renderer or PillowThumbnailRenderer(settings)
        self.repository = repository or FilesystemThumbnailRepository(output_dir)

    def generate_thumbnail(
        self,
        reading: Reading,
        *,
        title: str,
        hero_image: Path | None = None,
        thumbnail_text: str | None = None,
    ) -> Path | None:
        del hero_image  # reserved for future use

        if not self.enabled:
            logger.info("Thumbnail generation disabled by configuration")
            return None

        cached = self.repository.get_existing(reading.slug)
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
                func=lambda: self.image_provider.generate(prompt, user=reading.slug),
                max_retries=self.settings.thumbnail_max_retries,
                backoff=self.settings.thumbnail_retry_backoff,
                error_types=(
                    APIError,
                    RateLimitError,
                    APIConnectionError,
                    APITimeoutError,
                    OpenAIError,
                    RuntimeError,
                ),
                context=f"thumbnail_image_{reading.slug}",
            )
            rendered = self.renderer.render(image_bytes=image_bytes, text=text)
            thumbnail_path = self.repository.save(reading.slug, rendered)
            logger.info("Saved thumbnail: %s", thumbnail_path.name)
            return thumbnail_path
        except Exception as exc:
            raise ThumbnailGenerationError(str(exc)) from exc

    def _generate_thumbnail_intent_card(
        self,
        reading: Reading,
        thumbnail_text: str,
    ) -> ThumbnailIntentCard:
        try:
            return retry_with_backoff(
                func=lambda: self.intent_card_provider.generate(reading, thumbnail_text),
                max_retries=self.settings.thumbnail_max_retries,
                backoff=self.settings.thumbnail_retry_backoff,
                error_types=(
                    APIError,
                    RateLimitError,
                    APIConnectionError,
                    APITimeoutError,
                    OpenAIError,
                    RuntimeError,
                ),
                context=f"thumbnail_intent_card_{reading.slug}",
            )
        except Exception as exc:
            raise ThumbnailGenerationError(str(exc)) from exc

    @staticmethod
    def _build_prompt(
        reading: Reading,
        thumbnail_text: str,
        intent_card: ThumbnailIntentCard,
    ) -> str:
        return build_thumbnail_prompt(reading, thumbnail_text, intent_card)


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
