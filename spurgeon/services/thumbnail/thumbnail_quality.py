"""Quality gates for rendered thumbnails."""

from __future__ import annotations

from PIL import Image, ImageStat

from .thumbnail_errors import QualityGateError

EXPECTED_THUMBNAIL_SIZE = (1280, 720)


def validate_thumbnail_quality(
    image: Image.Image,
    *,
    checks_enabled: bool,
    min_luma_stddev: float,
) -> None:
    """Validate rendered thumbnail output against configured quality gates."""

    if not checks_enabled:
        return

    if image.size != EXPECTED_THUMBNAIL_SIZE:
        raise QualityGateError(
            f"Rendered thumbnail has invalid dimensions {image.size}; "
            f"expected {EXPECTED_THUMBNAIL_SIZE}"
        )

    grayscale = image.convert("L")
    stddev = float(ImageStat.Stat(grayscale).stddev[0])
    if stddev < min_luma_stddev:
        raise QualityGateError(
            f"Rendered thumbnail failed contrast gate: stddev={stddev:.2f} "
            f"< min_luma_stddev={min_luma_stddev:.2f}"
        )
