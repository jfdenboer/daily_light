"""Interfaces for the thumbnail generation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PIL import Image

from spurgeon.models import Reading

from .thumbnail_intent_card import ThumbnailIntentCard


class IntentCardProvider(Protocol):
    """Provider that generates a structured thumbnail intent card."""

    def generate(self, reading: Reading, thumbnail_text: str) -> ThumbnailIntentCard:
        """Generate an intent card for the provided reading context."""


class ImageProvider(Protocol):
    """Provider that generates a background image for a prompt."""

    def generate(self, prompt: str, *, user: str | None = None) -> bytes:
        """Generate raw image bytes for the given prompt."""


class ThumbnailRenderer(Protocol):
    """Renderer responsible for compositing text over an image."""

    def render(self, *, image_bytes: bytes, text: str) -> Image.Image:
        """Render a final thumbnail image in memory."""


class ThumbnailRepository(Protocol):
    """Repository that handles thumbnail persistence and cache lookup."""

    def get_existing(self, slug: str) -> Path | None:
        """Return existing thumbnail path when present."""

    def get_by_fingerprint(self, fingerprint: str) -> Path | None:
        """Return an existing thumbnail path for the exact fingerprint when present."""

    def save(self, slug: str, image: Image.Image, *, fingerprint: str | None = None) -> Path:
        """Persist the image and return output path."""
