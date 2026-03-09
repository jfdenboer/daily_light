"""Text layout helpers for thumbnail rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PIL import ImageDraw, ImageFont

THUMBNAIL_TEXT_HORIZONTAL_MARGIN_FRACTION = 0.08
THUMBNAIL_TEXT_VERTICAL_MARGIN_FRACTION = 0.08
THUMBNAIL_TEXT_CENTER_ZONE_HEIGHT_FRACTION = 0.34
THUMBNAIL_TEXT_MAX_LINES = 1
THUMBNAIL_TEXT_LINE_SPACING_RATIO = 0.10
THUMBNAIL_TEXT_FONT_SIZE = 175
THUMBNAIL_TEXT_FALLBACK_FONT_SIZE = 155
THUMBNAIL_TEXT_MIN_SAFE_FONT_SIZE = 138
THUMBNAIL_TEXT_STROKE_WIDTH_RATIO = 0.010
THUMBNAIL_TEXT_STROKE_MIN_WIDTH = 1
THUMBNAIL_TEXT_SHADOW_OFFSET_RATIO = 0.004
THUMBNAIL_TEXT_SHADOW_ALPHA = 32
THUMBNAIL_TEXT_TRACKING_RATIO = 0.015
THUMBNAIL_TEXT_TRACKING_MIN = 2
THUMBNAIL_TEXT_TRACKING_MAX = 10


@dataclass(frozen=True)
class TextLayoutBox:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class TextLayoutChoice:
    text: str
    line_count: int
    font_size: int
    text_bbox: tuple[int, int, int, int]
    block_size: tuple[int, int]
    stroke_width: int
    shadow_offset: tuple[int, int]
    tracking: int


def normalize_thumbnail_display_text(text: str) -> str:
    normalised = " ".join(text.replace("\n", " ").split())
    if not normalised:
        return "daily light"
    return normalised


def calculate_text_layout_box(canvas_size: tuple[int, int]) -> TextLayoutBox:
    width, height = canvas_size
    horizontal_margin = int(width * THUMBNAIL_TEXT_HORIZONTAL_MARGIN_FRACTION)
    vertical_margin = int(height * THUMBNAIL_TEXT_VERTICAL_MARGIN_FRACTION)
    max_text_width = width - (horizontal_margin * 2)
    max_center_zone_height = max(100, height - (vertical_margin * 2))
    center_zone_height = min(
        max(100, int(height * THUMBNAIL_TEXT_CENTER_ZONE_HEIGHT_FRACTION)),
        max_center_zone_height,
    )
    center_zone_y = max(vertical_margin, int((height - center_zone_height) / 2))
    return TextLayoutBox(
        x=horizontal_margin,
        y=center_zone_y,
        width=max(320, max_text_width),
        height=center_zone_height,
    )


class ThumbnailTextLayoutEngine:
    def __init__(
        self,
        draw: ImageDraw.ImageDraw,
        font_loader: Callable[[int], ImageFont.FreeTypeFont | ImageFont.ImageFont],
    ) -> None:
        self.draw = draw
        self.font_loader = font_loader

    def select_text_layout(
        self,
        display_text: str,
        text_box: TextLayoutBox,
    ) -> TextLayoutChoice:
        layout_candidates = [display_text.replace("\n", " ")]

        for candidate in layout_candidates:
            if candidate.count("\n") + 1 > THUMBNAIL_TEXT_MAX_LINES:
                continue

            measured = self.fit_fixed_font_sizes(
                candidate,
                text_box,
                preferred_sizes=(
                    THUMBNAIL_TEXT_FONT_SIZE,
                    THUMBNAIL_TEXT_FALLBACK_FONT_SIZE,
                    THUMBNAIL_TEXT_MIN_SAFE_FONT_SIZE,
                ),
            )
            if measured is not None:
                return measured

        fallback_text = layout_candidates[0]
        fallback_font_size = THUMBNAIL_TEXT_MIN_SAFE_FONT_SIZE
        fallback = self.measure_text_block(fallback_text, fallback_font_size)
        return TextLayoutChoice(
            text=fallback_text,
            line_count=fallback_text.count("\n") + 1,
            font_size=fallback_font_size,
            text_bbox=fallback,
            block_size=(fallback[2] - fallback[0], fallback[3] - fallback[1]),
            stroke_width=stroke_width_for_font_size(fallback_font_size),
            shadow_offset=shadow_offset_for_font_size(fallback_font_size),
            tracking=tracking_for_font_size(fallback_font_size),
        )

    def fit_fixed_font_sizes(
        self,
        layout_text: str,
        text_box: TextLayoutBox,
        *,
        preferred_sizes: tuple[int, ...],
    ) -> TextLayoutChoice | None:
        for font_size in preferred_sizes:
            stroke_width = stroke_width_for_font_size(font_size)
            text_bbox = self.measure_text_block(layout_text, font_size, stroke_width=stroke_width)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            if text_width <= text_box.width and text_height <= text_box.height:
                return TextLayoutChoice(
                    text=layout_text,
                    line_count=layout_text.count("\n") + 1,
                    font_size=font_size,
                    text_bbox=text_bbox,
                    block_size=(text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]),
                    stroke_width=stroke_width,
                    shadow_offset=shadow_offset_for_font_size(font_size),
                    tracking=tracking_for_font_size(font_size),
                )

        return None

    def measure_text_block(
        self,
        text: str,
        font_size: int,
        *,
        stroke_width: int | None = None,
    ) -> tuple[int, int, int, int]:
        if stroke_width is None:
            stroke_width = stroke_width_for_font_size(font_size)
        lines = text.split("\n") or [""]
        font = self.font_loader(font_size)
        tracking = tracking_for_font_size(font_size)
        spacing = line_spacing(font_size)

        line_widths: list[int] = []
        line_heights: list[int] = []
        for line in lines:
            line_widths.append(
                self._measure_tracked_line_width(
                    line,
                    font=font,
                    stroke_width=stroke_width,
                    tracking=tracking,
                )
            )
            line_bbox = self.draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
            line_heights.append(max(0, line_bbox[3] - line_bbox[1]))

        total_height = sum(line_heights)
        if len(lines) > 1:
            total_height += spacing * (len(lines) - 1)

        return (0, 0, max(line_widths, default=0), total_height)

    def _measure_tracked_line_width(
        self,
        line: str,
        *,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        stroke_width: int,
        tracking: int,
    ) -> int:
        if not line:
            return 0

        width = 0
        for index, character in enumerate(line):
            bbox = self.draw.textbbox((0, 0), character, font=font, stroke_width=stroke_width)
            width += max(0, bbox[2] - bbox[0])
            if index < len(line) - 1:
                width += tracking

        return width


def line_spacing(font_size: int) -> int:
    return max(8, int(font_size * THUMBNAIL_TEXT_LINE_SPACING_RATIO))


def stroke_width_for_font_size(font_size: int) -> int:
    return max(THUMBNAIL_TEXT_STROKE_MIN_WIDTH, int(font_size * THUMBNAIL_TEXT_STROKE_WIDTH_RATIO))


def shadow_offset_for_font_size(font_size: int) -> tuple[int, int]:
    offset = max(2, int(font_size * THUMBNAIL_TEXT_SHADOW_OFFSET_RATIO))
    return (offset, offset)


def tracking_for_font_size(font_size: int) -> int:
    tracked = int(font_size * THUMBNAIL_TEXT_TRACKING_RATIO)
    return max(THUMBNAIL_TEXT_TRACKING_MIN, min(THUMBNAIL_TEXT_TRACKING_MAX, tracked))


def resolve_text_position(
    layout: TextLayoutChoice,
    text_box: TextLayoutBox,
) -> tuple[int, int]:
    centered_x = text_box.x + int((text_box.width - layout.block_size[0]) / 2)
    x = max(text_box.x, centered_x)
    centered_y = text_box.y + int((text_box.height - layout.block_size[1]) / 2)
    y = centered_y
    min_y = text_box.y
    max_y = text_box.y + text_box.height - layout.block_size[1]
    return (x, max(min_y, min(y, max_y)))
