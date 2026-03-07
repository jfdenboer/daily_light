"""Typed errors for the thumbnail generation pipeline."""

from __future__ import annotations


class ThumbnailError(RuntimeError):
    """Base error for thumbnail pipeline failures."""


class ThumbnailGenerationError(ThumbnailError):
    """Top-level generation error raised by the orchestrator."""


class IntentCardError(ThumbnailError):
    """Raised when generating or parsing a thumbnail intent card fails."""


class PromptBuildError(ThumbnailError):
    """Raised when building the image prompt fails."""


class ImageProviderError(ThumbnailError):
    """Raised when generating the image bytes fails."""


class RenderError(ThumbnailError):
    """Raised when rendering a thumbnail image fails."""


class StorageError(ThumbnailError):
    """Raised when reading or writing thumbnail files fails."""

class QualityGateError(ThumbnailError):
    """Raised when a rendered thumbnail fails quality validation checks."""

