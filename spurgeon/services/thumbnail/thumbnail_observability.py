"""Observability helpers for consistent thumbnail pipeline events."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ThumbnailEvent:
    """Event names for thumbnail pipeline logging."""

    START = "thumbnail.pipeline.start"
    CACHE_HIT = "thumbnail.pipeline.cache_hit"
    INTENT_CARD_READY = "thumbnail.pipeline.intent_card.ready"
    PROMPT_READY = "thumbnail.pipeline.prompt.ready"
    IMAGE_READY = "thumbnail.pipeline.image.ready"
    RENDER_READY = "thumbnail.pipeline.render.ready"
    SAVED = "thumbnail.pipeline.saved"
    QUALITY_GATE_PASSED = "thumbnail.pipeline.quality_gate.passed"
    FAILED = "thumbnail.pipeline.failed"


def log_thumbnail_event(event: str, **fields: Any) -> None:
    """Log a normalized thumbnail event line with stable key ordering."""

    payload = " ".join(
        f"{key}={fields[key]!r}" for key in sorted(fields) if fields[key] is not None
    )
    logger.info("%s %s", event, payload)
