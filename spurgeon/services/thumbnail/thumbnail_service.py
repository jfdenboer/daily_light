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

from .thumbnail_contracts import ImageProvider, ThumbnailRenderer, ThumbnailRepository
from .thumbnail_errors import ImageProviderError, IntentCardError, PromptBuildError
from .thumbnail_image_intent_card import ThumbnailImageIntentCardBuilder
from .thumbnail_observability import ThumbnailEvent, log_thumbnail_event
from .thumbnail_prompting import build_thumbnail_prompt
from .thumbnail_quality import validate_thumbnail_quality
from .thumbnail_reading_diagnoser import ThumbnailReadingDiagnoser
from .thumbnail_source_reader import ThumbnailSourceReader

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
    """Generate thumbnails using the v3 image-side chain and provider abstractions."""

    def __init__(
        self,
        *,
        settings,
        image_provider: ImageProvider,
        renderer: ThumbnailRenderer,
        repository: ThumbnailRepository,
        source_reader: ThumbnailSourceReader,
        reading_diagnoser: ThumbnailReadingDiagnoser,
        image_intent_card_builder: ThumbnailImageIntentCardBuilder,
    ) -> None:
        self.settings = settings
        self.image_provider = image_provider
        self.renderer = renderer
        self.repository = repository
        self.source_reader = source_reader
        self.reading_diagnoser = reading_diagnoser
        self.image_intent_card_builder = image_intent_card_builder

    def generate(self, reading: "Reading", *, thumbnail_text: str) -> Path | None:
        if not self.settings.thumbnail_enabled:
            logger.info("Thumbnail generation disabled by configuration")
            return None

        text = thumbnail_text
        fingerprint = self.build_fingerprint(
            reading=reading,
            thumbnail_text=text,
            prompt_version=self.settings.thumbnail_prompt_version,
        )

        cached = self._resolve_cached_path(reading.slug, fingerprint)
        if cached:
            return cached

        log_thumbnail_event(
            ThumbnailEvent.START,
            slug=reading.slug,
            thumbnail_text=text,
            reading_type=reading.reading_type.value,
        )

        stage = "source_read"
        try:
            source_read = self.source_reader.read(reading)
            stage = "reading_diagnosis"
            diagnosis = self.reading_diagnoser.diagnose(reading, source_read)
            stage = "image_intent_card"
            image_intent_card = self._generate_image_intent_card(reading, source_read, diagnosis)
            stage = "prompt"
            prompt = self._generate_prompt(reading, text, image_intent_card)
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

    def _generate_image_intent_card(self, reading, source_read, diagnosis):
        log_thumbnail_event(ThumbnailEvent.IMAGE_INTENT_CARD_START, slug=reading.slug)
        try:
            card = self.image_intent_card_builder.build(reading, source_read, diagnosis)
        except Exception as exc:
            raise IntentCardError(str(exc)) from exc

        log_thumbnail_event(
            ThumbnailEvent.IMAGE_INTENT_CARD_READY,
            slug=reading.slug,
            visual_tension=card.visual_tension,
            emotional_tone=card.emotional_tone,
            dominant_anchor=card.dominant_anchor,
            scene_direction=card.scene_direction,
            subject_priority=card.subject_priority,
            visual_open_loop=card.visual_open_loop,
            visual_avoid=", ".join(card.visual_avoid),
        )
        return card

    def _generate_prompt(self, reading: "Reading", text: str, image_intent_card) -> str:
        log_thumbnail_event(ThumbnailEvent.PROMPT_FROM_2A_START, slug=reading.slug)
        try:
            prompt = build_thumbnail_prompt(
                reading,
                text,
                image_intent_card,
                prompt_version=self.settings.thumbnail_prompt_version,
            )
        except Exception as exc:
            raise PromptBuildError(str(exc)) from exc

        log_thumbnail_event(
            ThumbnailEvent.PROMPT_FROM_2A_READY,
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
    def build_fingerprint(*, reading: "Reading", thumbnail_text: str, prompt_version: str) -> str:
        """Build a deterministic cache fingerprint for the thumbnail request."""

        reading_text_hash = hashlib.sha256(reading.text.encode("utf-8")).hexdigest()
        payload = "|".join(
            [
                "thumbnail",
                prompt_version,
                reading.slug,
                reading.reading_type.value,
                thumbnail_text.strip(),
                reading_text_hash,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
