# image_generator.py

from __future__ import annotations

"""image_generator.py – genereert één afbeelding per reading (via OpenAI Images API)."""

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from spurgeon.config.settings import Settings
from spurgeon.services.prompt_generation import PromptOrchestrator
from spurgeon.models import Reading, RawAsset
from spurgeon.utils.retry_utils import retry_with_backoff

logger = logging.getLogger(__name__)

STYLE_LINE = (
    "Style: contemporary watercolor in fine line-and-wash; subtle ink or pencil linework "
    "only where needed; transparent washes over light structure; edges allowed to dissolve "
    "into the wash; restrained detail; no heavy outlines; no glossy/3D/rendered look."
)

BACKGROUND_LINE = (
    "Surface: cold-press 300gsm watercolor paper; warm off-white; gentle paper grain visible; "
    "no photographic backdrops; no harsh vignette."
)

COMPOSITION_LINE = (
    "Composition: square 1:1; clear focal subject slightly off-center (figure, object, or architectural/landscape element); "
    "10–15% negative space; atmospheric perspective (distant elements lower contrast); "
    "avoid edge tangents and tight crops; keep linework selective."
)

CONSTRAINTS_LINE = (
    "Constraints: no visible text, lettering, watermarks, logos, letters or numbers; "
    "figures allowed only as distant silhouettes or abstract posture; avoid faces and detailed anatomy; "
    "avoid neon and over-sharpening; preserve watercolor transparency."
)

PALETTE_LINE = (
    "Palette: indigo or neutral tint for linework; raw umber, burnt sienna, sap green for washes; "
    "optional cobalt/ultramarine accents; Payne's gray for shadows; titanium white reserved as paper; "
    "limited harmonized palette; transparent layering (glazing)."
)

LIGHTING_LINE = (
    "Lighting: diffuse daylight from upper-left; subtle paper-white highlights; "
    "atmospheric haze in the distance; shadows in neutral tint/Payne's gray plus local complement."
)




@dataclass(frozen=True)
class PromptStyleConfig:
    """Container for the reusable style instructions prepended to every image prompt."""

    style_line: str = STYLE_LINE
    background_line: str = BACKGROUND_LINE
    palette_line: str = PALETTE_LINE
    lighting_line: str = LIGHTING_LINE
    composition_line: str = COMPOSITION_LINE
    constraints_line: str = CONSTRAINTS_LINE

    def compose_prompt(self, subject_prompt: str) -> str:
        """Return the full prompt string for :mod:`openai` based on ``subject_prompt``."""

        subject = subject_prompt.strip()
        blocks: Iterable[str] = (
            self.style_line,
            self.background_line,
            self.palette_line,
            self.lighting_line,
            f"Subject: {subject}" if subject else "",
            self.composition_line,
            self.constraints_line,
        )
        return "\n\n".join(block for block in blocks if block)

class ImageGenerationError(Exception):
    """Fout tijdens oproep of download bij image‑generation."""


class ImageGenerator:
    """Generate stylised images for :class:`~spurgeon.models.Reading` objects."""

    def __init__(
        self,
        settings: Settings,
        *,
        style_config: PromptStyleConfig | None = None,
    ) -> None:  # noqa: D401
        self.settings = settings
        self.output_dir = Path(settings.output_dir) / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.image_model
        self.size = settings.image_size
        self.quality = getattr(settings, "image_quality", "high")
        self.background = getattr(settings, "image_background", "opaque")
        self.max_retries = settings.image_max_retries
        self.retry_backoff = settings.image_retry_backoff
        self.style_config = style_config or PromptStyleConfig()

        logger.info(
            "ImageGenerator ready (model=%s, size=%s, quality=%s, background=%s)",
            self.model,
            self.size,
            self.quality,
            self.background,
        )
        if self.model == "gpt-image-1":
            logger.warning(
                "Using legacy image model gpt-image-1; consider switching to gpt-image-1.5"
            )

    def generate_single_image_for_reading(
        self,
        reading: Reading,
        *,
        duration: float,
    ) -> RawAsset | None:
        """Generate exactly one image asset for the full reading."""

        prompt_gen = PromptOrchestrator(self.settings)
        try:
            image_path, _, subject_prompt = prompt_gen.build_single_image_asset(
                reading,
                duration=duration,
            )
        except Exception as exc:
            logger.warning("Single-image prompt generation failed for %s: %s", reading.slug, exc)
            return None

        if image_path.exists():
            logger.debug("Skipping existing single image: %s", image_path.name)
            return (image_path, duration, subject_prompt)

        if not subject_prompt.strip():
            logger.warning("Skipping empty single-image subject prompt for %s", reading.slug)
            return None

        full_prompt = self._compose_full_prompt(subject_prompt)

        logger.debug("Generating single image: %s", image_path.name)
        logger.debug("Full single-image prompt for %s:\n%s", image_path.name, full_prompt)
        try:
            image_bytes = retry_with_backoff(
                func=lambda: self._call_openai(full_prompt, user=reading.slug),
                max_retries=self.max_retries,
                backoff=self.retry_backoff,
                error_types=(
                    APIError,
                    RateLimitError,
                    APIConnectionError,
                    APITimeoutError,
                    OpenAIError,
                    ImageGenerationError,
                ),
                context=f"single_image_prompt_{image_path.stem}",
            )
            retry_with_backoff(
                func=lambda: self._write_image(image_bytes, image_path),
                max_retries=self.max_retries,
                backoff=self.retry_backoff,
                error_types=(ImageGenerationError,),
                context=f"single_image_write_{image_path.stem}",
            )
            logger.info("Saved single image: %s", image_path.name)
            return (image_path, duration, subject_prompt)
        except Exception as exc:
            logger.warning("Single image generation failed for %s: %s", reading.slug, exc)
            return None

    def _compose_full_prompt(self, subject_prompt: str) -> str:
        """Combine the configured style instructions with ``subject_prompt``."""

        return self.style_config.compose_prompt(subject_prompt)

    def _call_openai(self, prompt: str, *, user: str | None = None) -> bytes:
        try:
            response = self.client.images.generate(
                model=self.model,
                prompt=prompt,
                n=1,
                size=self.size,
                user=user,
                quality=self.quality,
                background=self.background,
            )
        except OpenAIError as e:
            raise ImageGenerationError(
                f"OpenAI image call failed: {getattr(e, 'message', e)}"
            ) from e

        data = getattr(response, "data", None)
        if not data:
            raise ImageGenerationError("No image data returned from OpenAI")

        image_info = data[0]
        b64_payload = getattr(image_info, "b64_json", None)
        if not b64_payload:
            raise ImageGenerationError("No base64 payload returned from OpenAI")

        try:
            return base64.b64decode(b64_payload)
        except (ValueError, TypeError) as exc:
            raise ImageGenerationError("Invalid base64 image payload from OpenAI") from exc

    @staticmethod
    def _write_image(image_bytes: bytes, dest_path: Path) -> None:
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with dest_path.open("wb") as fh:
                fh.write(image_bytes)
        except OSError as exc:
            raise ImageGenerationError("Writing image to disk failed") from exc


__all__ = ["ImageGenerator", "ImageGenerationError", "PromptStyleConfig"]
