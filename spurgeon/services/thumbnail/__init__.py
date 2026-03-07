"""Thumbnail generation services."""

from .generate_thumbnail_text import (
    SYSTEM_PROMPT_THUMBNAIL_GENERATOR,
    SYSTEM_PROMPT_THUMBNAIL_SELECTOR,
    ThumbnailTextGenerationError,
    ThumbnailTextGenerator,
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
    ThumbnailError,
)
from .thumbnail_generator import (
    THUMBNAIL_BACKGROUND_LINE,
    THUMBNAIL_COMPOSITION_LINE,
    THUMBNAIL_CONSTRAINTS_LINE,
    THUMBNAIL_LIGHTING_LINE,
    THUMBNAIL_PALETTE_LINE,
    THUMBNAIL_STYLE_LINE,
    THUMBNAIL_SUBJECT_LINE,
    ThumbnailGenerationError,
    ThumbnailGenerator,
)

__all__ = [
    "ThumbnailGenerator",
    "ThumbnailGenerationError",
    "ThumbnailTextGenerator",
    "ThumbnailTextGenerationError",
    "SYSTEM_PROMPT_THUMBNAIL_GENERATOR",
    "SYSTEM_PROMPT_THUMBNAIL_SELECTOR",
    "THUMBNAIL_STYLE_LINE",
    "THUMBNAIL_SUBJECT_LINE",
    "THUMBNAIL_BACKGROUND_LINE",
    "THUMBNAIL_COMPOSITION_LINE",
    "THUMBNAIL_CONSTRAINTS_LINE",
    "THUMBNAIL_PALETTE_LINE",
    "THUMBNAIL_LIGHTING_LINE",
    "IntentCardProvider",
    "ImageProvider",
    "ThumbnailRenderer",
    "ThumbnailRepository",
    "ThumbnailError",
    "IntentCardError",
    "PromptBuildError",
    "QualityGateError",
    "ImageProviderError",
    "RenderError",
    "StorageError",
]
