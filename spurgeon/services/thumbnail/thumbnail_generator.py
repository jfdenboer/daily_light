"""Backward-compatible wrapper around :class:`ThumbnailService`."""

from __future__ import annotations

import logging
from pathlib import Path

from openai import OpenAI

from spurgeon.config.settings import Settings
from spurgeon.models import Reading

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
from .thumbnail_errors import ThumbnailGenerationError
from .thumbnail_intent_card import ThumbnailIntentCard
from .thumbnail_prompting import (
    THUMBNAIL_BACKGROUND_LINE,
    THUMBNAIL_COMPOSITION_LINE,
    THUMBNAIL_CONSTRAINTS_LINE,
    THUMBNAIL_LIGHTING_LINE,
    THUMBNAIL_PALETTE_LINE,
    THUMBNAIL_STYLE_LINE,
    THUMBNAIL_SUBJECT_LINE,
)
from .thumbnail_service import ThumbnailService

logger = logging.getLogger(__name__)


class ThumbnailGenerator:
    """Legacy API surface that delegates thumbnail work to :class:`ThumbnailService`."""

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

        provider_client = None
        if intent_card_provider is None or image_provider is None:
            provider_client = OpenAI(api_key=settings.openai_api_key)

        output_dir = Path(settings.output_dir) / "thumbnails"
        service = ThumbnailService(
            settings=settings,
            intent_card_provider=intent_card_provider
            or OpenAIIntentCardProvider(provider_client, settings),
            image_provider=image_provider or OpenAIImageProvider(provider_client, settings),
            renderer=renderer or PillowThumbnailRenderer(settings),
            repository=repository or FilesystemThumbnailRepository(output_dir),
        )
        self.service = service

    def generate_thumbnail(
        self,
        reading: Reading,
        *,
        hero_image: Path | None = None,
        thumbnail_text: str | None = None,
    ) -> Path | None:
        del hero_image  # reserved for future use

        try:
            return self.service.generate(reading, thumbnail_text=thumbnail_text or "daily light")
        except Exception as exc:
            if isinstance(exc, ThumbnailGenerationError):
                raise
            raise ThumbnailGenerationError(str(exc)) from exc


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
