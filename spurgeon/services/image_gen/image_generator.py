# image_generator.py

from __future__ import annotations

"""image_generator.py – genereert één afbeelding per reading (via OpenAI Images API)."""

import base64
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
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
    "Style: modern cinematic stillframe; naturalistic realism with a calm, contemplative mood; "
    "subtle highlight bloom/halation without overall softness; the focal subject remains clear, distinct, and readable; "
    "fine, subtle film-like grain; matte finish; natural materials and believable surfaces with slight imperfections; "
    "favor subject clarity over global sharpness; "
    "avoid hyper-clarity, micro-contrast, crunchy edges, sharpening halos, heavy vignettes, HDR, smeary detail, and heavy diffusion; "
    "no illustration, paper, paint, stylized concept-art, or CGI/3D/glossy rendering."
)

BACKGROUND_LINE = (
    "Setting: believable outdoor environment with depth and atmosphere; uncluttered but not empty; "
    "prefer simple, restrained exterior settings with varied but plausible terrain and context such as paths, fields, meadows, moorland, shorelines, village edges, courtyards, walls, gates, orchards, sparse woodland edges, quiet roads, weathered stone, wood, water, or garden margins; "
    "do not repeatedly default to the same rocky coast, dune, hill, cliffside, or snow-scene look; "
    "subtle haze may be used for distance separation; keep the central band calm, simple, and low-detail; "
    "avoid close foliage, prominent branches framing the scene, dense leafy clutter, decorative flowers, staged studio backdrops, and vintage, sepia, or antique cues."
)

COMPOSITION_LINE = (
    "Composition: 16:9 wide; one clear focal subject placed off-center; strong depth with simple foreground, midground, and background; minimal elements; "
    "reserve a calm, low-contrast horizontal subtitle safe-zone across the exact center, with no key subject, no busy textures, and no high-detail edges; "
    "keep the focal subject clearly above or below the subtitle safe-zone; "
    "avoid tight crops, edge tangents, horizon lines cutting through the subtitle zone, competing focal points, and collage-like storytelling; "
    "the image should read clearly at a glance."
)

CONSTRAINTS_LINE = (
    "Constraints: no visible text, captions, lettering, numbers, pseudo-text, watermarks, logos, readable signage, icons, emblems, frames, or UI overlays; "
    "do not default to a solitary adult man, traveler, pilgrim, hiker, or figure with a backpack, staff, or obvious travel gear; "
    "human figures are optional, not required; prefer non-human or environmental focal subjects when suitable; "
    "if a person appears, vary age and presentation, keep the figure small, distant, non-identifiable, and naturally integrated into the setting, with no visible face or modern gear; "
    "avoid exaggerated shallow depth of field, bokeh balls, cut-out subject separation, neon colors, oversharpening, repeating patterns, banding in skies or gradients, and high-frequency texture behind the subtitle area."
)

PALETTE_LINE = (
    "Color grade: warm-neutral cinematic palette with restrained saturation; natural earth, stone, sky, water, wood, and muted vegetation tones; "
    "gentle contrast with clean highlights and soft shadows; subtle warm highlight accents without orange cast; "
    "avoid overly lush greens, yellowed vintage tones, teal-orange gimmicks, heavy color casts, or overly picturesque postcard color."
)

LIGHTING_LINE = (
    "Lighting: soft overcast daylight or gentle natural directional light, with calm luminous highlights and subtle atmospheric haze; "
    "soft shadows and controlled highlights; gentle subject-background separation without extreme backlight; "
    "maintain gentle shape definition on the focal subject without flattening it; "
    "avoid visible rays or beams, harsh spotlighting, theatrical drama, or sensational god-ray lighting."
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
            logger.warning(
                "Single-image prompt generation failed for %s: %s", reading.slug, exc
            )
            return None

        metadata_path = self._metadata_path_for_image(image_path)

        if image_path.exists():
            logger.debug("Skipping existing single image: %s", image_path.name)
            self._write_generation_metadata(
                metadata_path=metadata_path,
                reading=reading,
                image_path=image_path,
                subject_prompt=subject_prompt,
                full_prompt=self._compose_full_prompt(subject_prompt),
                duration=duration,
                generated=False,
            )
            return (image_path, duration, subject_prompt)

        if not subject_prompt.strip():
            logger.warning(
                "Skipping empty single-image subject prompt for %s", reading.slug
            )
            return None

        full_prompt = self._compose_full_prompt(subject_prompt)

        logger.debug("Generating single image: %s", image_path.name)
        logger.debug(
            "Full single-image prompt for %s:\n%s", image_path.name, full_prompt
        )
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
            self._write_generation_metadata(
                metadata_path=metadata_path,
                reading=reading,
                image_path=image_path,
                subject_prompt=subject_prompt,
                full_prompt=full_prompt,
                duration=duration,
                generated=True,
            )
            logger.info("Saved single image: %s", image_path.name)
            logger.info(
                "single-image request completed for %s (image_requests_per_video=1)",
                reading.slug,
            )
            return (image_path, duration, subject_prompt)
        except Exception as exc:
            logger.warning(
                "Single image generation failed for %s: %s", reading.slug, exc
            )
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
            raise ImageGenerationError(
                "Invalid base64 image payload from OpenAI"
            ) from exc

    @staticmethod
    def _write_image(image_bytes: bytes, dest_path: Path) -> None:
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with dest_path.open("wb") as fh:
                fh.write(image_bytes)
        except OSError as exc:
            raise ImageGenerationError("Writing image to disk failed") from exc

    @staticmethod
    def _metadata_path_for_image(image_path: Path) -> Path:
        return image_path.with_suffix(".metadata.json")

    def _write_generation_metadata(
        self,
        *,
        metadata_path: Path,
        reading: Reading,
        image_path: Path,
        subject_prompt: str,
        full_prompt: str,
        duration: float,
        generated: bool,
    ) -> None:
        payload = {
            "reading_slug": reading.slug,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "image_path": str(image_path),
            "duration_seconds": duration,
            "prompt_subject": subject_prompt,
            "prompt_full": full_prompt,
            "image_model": self.model,
            "image_size": self.size,
            "image_quality": self.quality,
            "image_background": self.background,
            "created_new_image": generated,
        }
        try:
            metadata_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "Could not write image metadata for %s: %s", reading.slug, exc
            )


__all__ = ["ImageGenerator", "ImageGenerationError", "PromptStyleConfig"]
