"""Application service orchestrating thumbnail generation pipeline steps."""

from __future__ import annotations

import hashlib
import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spurgeon.models import Reading

from spurgeon.utils.retry_utils import retry_with_backoff

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
)
from .thumbnail_intent_card import ThumbnailIntentCard
from .thumbnail_observability import ThumbnailEvent, log_thumbnail_event
from .thumbnail_prompting import build_thumbnail_prompt
from .thumbnail_quality import validate_thumbnail_quality

logger = logging.getLogger(__name__)


def _openai_retry_error_types() -> tuple[type[BaseException], ...]:
    """Return OpenAI transient error classes when the SDK is available."""

    try:
        openai = importlib.import_module("openai")
    except ModuleNotFoundError:
        return ()

    discovered = (
        getattr(openai, "APIError", None),
        getattr(openai, "RateLimitError", None),
        getattr(openai, "APIConnectionError", None),
        getattr(openai, "APITimeoutError", None),
        getattr(openai, "OpenAIError", None),
    )
    return tuple(
        error_type
        for error_type in discovered
        if isinstance(error_type, type) and issubclass(error_type, BaseException)
    )


class ThumbnailService:
    """Generate thumbnails using provider, renderer and repository abstractions."""

    def __init__(
        self,
        *,
        settings,
        intent_card_provider: IntentCardProvider,
        image_provider: ImageProvider,
        renderer: ThumbnailRenderer,
        repository: ThumbnailRepository,
    ) -> None:
        self.settings = settings
        self.intent_card_provider = intent_card_provider
        self.image_provider = image_provider
        self.renderer = renderer
        self.repository = repository

    def generate(self, reading: "Reading", *, title: str, thumbnail_text: str | None = None) -> Path | None:
        if not self.settings.thumbnail_enabled:
            logger.info("Thumbnail generation disabled by configuration")
            return None

        text = thumbnail_text or title
        fingerprint = self.build_fingerprint(
            reading=reading,
            title=text,
            prompt_version=self.settings.thumbnail_prompt_version,
        )

        cached = self._resolve_cached_path(reading.slug, fingerprint)
        if cached:
            return cached

        log_thumbnail_event(
            ThumbnailEvent.START,
            slug=reading.slug,
            title=text,
            reading_type=reading.reading_type.value,
        )

        stage = "intent_card"
        try:
            intent_card = self._generate_intent_card(reading, text)
            stage = "prompt"
            prompt = self._generate_prompt(reading, text, intent_card)
            stage = "image_render"
            rendered = self._generate_rendered_image(reading.slug, prompt, text)
            stage = "quality_gate"
            self._validate_quality(reading.slug, rendered)
            stage = "storage"
            path = self.repository.save(reading.slug, rendered, fingerprint=fingerprint)
            log_thumbnail_event(ThumbnailEvent.SAVED, slug=reading.slug, path=path.name)
            return path
        except Exception as exc:
            log_thumbnail_event(
                ThumbnailEvent.FAILED,
                slug=reading.slug,
                stage=stage,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            raise

    def _resolve_cached_path(self, slug: str, fingerprint: str) -> Path | None:
        if self.settings.thumbnail_cache_by_fingerprint:
            cached = self.repository.get_by_fingerprint(fingerprint)
            if cached:
                log_thumbnail_event(
                    ThumbnailEvent.CACHE_HIT,
                    slug=slug,
                    path=cached.name,
                    cache_key="fingerprint",
                )
                return cached

        cached = self.repository.get_existing(slug)
        if cached:
            log_thumbnail_event(
                ThumbnailEvent.CACHE_HIT,
                slug=slug,
                path=cached.name,
                cache_key="slug",
            )
            return cached
        return None

    def _generate_intent_card(self, reading: "Reading", text: str) -> ThumbnailIntentCard:
        try:
            card = retry_with_backoff(
                func=lambda: self.intent_card_provider.generate(reading, text),
                max_retries=self.settings.thumbnail_max_retries,
                backoff=self.settings.thumbnail_retry_backoff,
                error_types=(*_openai_retry_error_types(), IntentCardError),
                context=f"thumbnail_intent_card_{reading.slug}",
            )
        except Exception as exc:
            raise IntentCardError(str(exc)) from exc

        log_thumbnail_event(
            ThumbnailEvent.INTENT_CARD_READY,
            slug=reading.slug,
            core_tension=card.core_tension,
            emotional_tone=card.emotional_tone,
            dominant_anchor=card.dominant_anchor,
            open_loop=card.open_loop,
            scene_direction=card.scene_direction,
            avoid=card.avoid,
        )
        return card

    def _generate_prompt(self, reading: "Reading", text: str, intent_card: ThumbnailIntentCard) -> str:
        try:
            prompt = build_thumbnail_prompt(
                reading,
                text,
                intent_card,
                prompt_version=self.settings.thumbnail_prompt_version,
            )
        except Exception as exc:
            raise PromptBuildError(str(exc)) from exc

        log_thumbnail_event(
            ThumbnailEvent.PROMPT_READY,
            slug=reading.slug,
            prompt_char_count=len(prompt),
        )
        return prompt

    def _generate_rendered_image(self, slug: str, prompt: str, text: str):
        image_bytes = retry_with_backoff(
            func=lambda: self.image_provider.generate(prompt, user=slug),
            max_retries=self.settings.thumbnail_max_retries,
            backoff=self.settings.thumbnail_retry_backoff,
            error_types=(*_openai_retry_error_types(), ImageProviderError),
            context=f"thumbnail_image_{slug}",
        )
        log_thumbnail_event(ThumbnailEvent.IMAGE_READY, slug=slug, image_byte_count=len(image_bytes))

        rendered = self.renderer.render(image_bytes=image_bytes, text=text)
        log_thumbnail_event(ThumbnailEvent.RENDER_READY, slug=slug)
        return rendered

    def _validate_quality(self, slug: str, rendered) -> None:
        validate_thumbnail_quality(
            rendered,
            checks_enabled=self.settings.thumbnail_quality_checks_enabled,
            min_luma_stddev=self.settings.thumbnail_quality_min_luma_stddev,
        )
        log_thumbnail_event(ThumbnailEvent.QUALITY_GATE_PASSED, slug=slug)

    @staticmethod
    def build_fingerprint(*, reading: "Reading", title: str, prompt_version: str) -> str:
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
