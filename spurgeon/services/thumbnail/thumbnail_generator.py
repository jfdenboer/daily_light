"""Generate YouTube thumbnails with provider, renderer and repository adapters."""

from __future__ import annotations

import hashlib
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
from .thumbnail_errors import (
    ImageProviderError,
    IntentCardError,
    PromptBuildError,
    QualityGateError,
    RenderError,
    StorageError,
    ThumbnailGenerationError,
)
from .thumbnail_intent_card import ThumbnailIntentCard
from .thumbnail_observability import ThumbnailEvent, log_thumbnail_event
from .thumbnail_quality import validate_thumbnail_quality
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

        text = thumbnail_text or title
        fingerprint = self._build_fingerprint(
            reading=reading,
            title=text,
            prompt_version=self.settings.thumbnail_prompt_version,
        )

        if self.settings.thumbnail_cache_by_fingerprint:
            fingerprint_cached = self.repository.get_by_fingerprint(fingerprint)
            if fingerprint_cached:
                log_thumbnail_event(
                    ThumbnailEvent.CACHE_HIT,
                    slug=reading.slug,
                    path=fingerprint_cached.name,
                    cache_key="fingerprint",
                )
                return fingerprint_cached

        cached = self.repository.get_existing(reading.slug)
        if cached:
            log_thumbnail_event(
                ThumbnailEvent.CACHE_HIT,
                slug=reading.slug,
                path=cached.name,
                cache_key="slug",
            )
            return cached

        log_thumbnail_event(
            ThumbnailEvent.START,
            slug=reading.slug,
            title=text,
            reading_type=reading.reading_type.value,
        )

        try:
            intent_card = self._generate_thumbnail_intent_card(reading, text)
            log_thumbnail_event(
                ThumbnailEvent.INTENT_CARD_READY,
                slug=reading.slug,
                core_tension=intent_card.core_tension,
                emotional_tone=intent_card.emotional_tone,
                visual_motif=intent_card.visual_motif,
                scene_direction=intent_card.scene_direction,
                avoid=intent_card.avoid,
            )

            prompt = self._build_prompt(
                reading,
                text,
                intent_card,
                prompt_version=self.settings.thumbnail_prompt_version,
            )
            log_thumbnail_event(
                ThumbnailEvent.PROMPT_READY,
                slug=reading.slug,
                prompt_char_count=len(prompt),
            )

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
                    ImageProviderError,
                ),
                context=f"thumbnail_image_{reading.slug}",
            )
            log_thumbnail_event(
                ThumbnailEvent.IMAGE_READY,
                slug=reading.slug,
                image_byte_count=len(image_bytes),
            )
            rendered = self.renderer.render(image_bytes=image_bytes, text=text)
            log_thumbnail_event(ThumbnailEvent.RENDER_READY, slug=reading.slug)

            validate_thumbnail_quality(
                rendered,
                checks_enabled=self.settings.thumbnail_quality_checks_enabled,
                min_luma_stddev=self.settings.thumbnail_quality_min_luma_stddev,
            )
            log_thumbnail_event(ThumbnailEvent.QUALITY_GATE_PASSED, slug=reading.slug)

            thumbnail_path = self.repository.save(
                reading.slug,
                rendered,
                fingerprint=fingerprint,
            )
            log_thumbnail_event(
                ThumbnailEvent.SAVED,
                slug=reading.slug,
                path=thumbnail_path.name,
            )
            return thumbnail_path
        except (IntentCardError, PromptBuildError, ImageProviderError, RenderError, StorageError, QualityGateError) as exc:
            log_thumbnail_event(
                ThumbnailEvent.FAILED,
                slug=reading.slug,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise ThumbnailGenerationError(str(exc)) from exc
        except Exception as exc:
            log_thumbnail_event(
                ThumbnailEvent.FAILED,
                slug=reading.slug,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise ThumbnailGenerationError(
                f"Unexpected thumbnail pipeline failure: {exc}"
            ) from exc

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
                    IntentCardError,
                ),
                context=f"thumbnail_intent_card_{reading.slug}",
            )
        except Exception as exc:
            raise IntentCardError(str(exc)) from exc

    @staticmethod
    def _build_prompt(
        reading: Reading,
        thumbnail_text: str,
        intent_card: ThumbnailIntentCard,
        *,
        prompt_version: str,
    ) -> str:
        try:
            return build_thumbnail_prompt(
                reading,
                thumbnail_text,
                intent_card,
                prompt_version=prompt_version,
            )
        except Exception as exc:
            raise PromptBuildError(str(exc)) from exc

    @staticmethod
    def _build_fingerprint(
        *,
        reading: Reading,
        title: str,
        prompt_version: str,
    ) -> str:
        """Build a deterministic cache fingerprint for the thumbnail request."""

        reading_text_hash = hashlib.sha256(reading.text.encode("utf-8")).hexdigest()
        payload = "|".join(
            [
                "thumbnail",
                prompt_version,
                reading.slug,
                reading.reading_type.value,
                title.strip(),
                reading_text_hash,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
