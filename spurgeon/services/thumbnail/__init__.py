"""Thumbnail generation service exports.

The package intentionally keeps imports lazy so tooling can import
``spurgeon.services.thumbnail`` without triggering optional runtime dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ThumbnailGenerationError",
    "ThumbnailGenerator",
    "ThumbnailTextGenerationError",
]


def __getattr__(name: str) -> Any:
    if name in {"ThumbnailGenerator", "ThumbnailGenerationError"}:
        module = import_module("spurgeon.services.thumbnail.thumbnail_generator")
        return getattr(module, name)
    if name == "ThumbnailTextGenerationError":
        module = import_module("spurgeon.services.thumbnail.generate_thumbnail_text")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
