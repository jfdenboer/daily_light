"""Text layout helpers for thumbnail rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PIL import ImageDraw, ImageFont

THUMBNAIL_TEXT_HORIZONTAL_MARGIN_FRACTION = 0.05
THUMBNAIL_TEXT_TOP_MARGIN_FRACTION = 0.05
THUMBNAIL_TEXT_TOP_ZONE_HEIGHT_FRACTION = 0.20
THUMBNAIL_TEXT_MAX_LINES = 1
THUMBNAIL_TEXT_LINE_SPACING_RATIO = 0.08
THUMBNAIL_TEXT_MIN_FONT_SIZE = 90
THUMBNAIL_TEXT_MAX_FONT_SIZE = 440
THUMBNAIL_TEXT_EMERGENCY_MIN_FONT_SIZE = 56
THUMBNAIL_TEXT_STROKE_WIDTH_RATIO = 0.024
THUMBNAIL_TEXT_STROKE_MIN_WIDTH = 2
THUMBNAIL_TEXT_SHADOW_OFFSET_RATIO = 0.018
THUMBNAIL_TEXT_SHADOW_ALPHA = 110


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


def normalize_thumbnail_display_text(text: str) -> str:
    normalised = " ".join(text.replace("\n", " ").split())
    if not normalised:
        return "DAILY LIGHT"
    return normalised.upper()


def calculate_text_layout_box(canvas_size: tuple[int, int]) -> TextLayoutBox:
    width, height = canvas_size
    horizontal_margin = int(width * THUMBNAIL_TEXT_HORIZONTAL_MARGIN_FRACTION)
    top_margin = int(height * THUMBNAIL_TEXT_TOP_MARGIN_FRACTION)
    max_text_width = width - (horizontal_margin * 2)
    text_height = int(height * THUMBNAIL_TEXT_TOP_ZONE_HEIGHT_FRACTION)
    return TextLayoutBox(
        x=horizontal_margin,
        y=top_margin,
        width=max(320, max_text_width),
        height=max(100, text_height),
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

        best: TextLayoutChoice | None = None
        for candidate in layout_candidates:
            if candidate.count("\n") + 1 > THUMBNAIL_TEXT_MAX_LINES:
                continue
            measured = self.fit_largest_font(
                candidate,
                text_box,
                min_font_size=THUMBNAIL_TEXT_MIN_FONT_SIZE,
                max_font_size=THUMBNAIL_TEXT_MAX_FONT_SIZE,
            )
            if not measured:
                continue

            if best is None or measured.font_size > best.font_size:
                best = measured
                continue

        if best is not None:
            return best

        for candidate in layout_candidates:
            if candidate.count("\n") + 1 > THUMBNAIL_TEXT_MAX_LINES:
                continue
            measured = self.fit_largest_font(
                candidate,
                text_box,
                min_font_size=THUMBNAIL_TEXT_EMERGENCY_MIN_FONT_SIZE,
                max_font_size=THUMBNAIL_TEXT_MIN_FONT_SIZE - 1,
            )
            if not measured:
                continue
            if best is None or measured.font_size > best.font_size:
                best = measured

        if best is not None:
            return best

        fallback_text = layout_candidates[0]
        fallback_font_size = THUMBNAIL_TEXT_EMERGENCY_MIN_FONT_SIZE
        fallback = self.measure_text_block(fallback_text, fallback_font_size)
        return TextLayoutChoice(
            text=fallback_text,
            line_count=fallback_text.count("\n") + 1,
            font_size=fallback_font_size,
            text_bbox=fallback,
            block_size=(fallback[2] - fallback[0], fallback[3] - fallback[1]),
            stroke_width=stroke_width_for_font_size(fallback_font_size),
            shadow_offset=shadow_offset_for_font_size(fallback_font_size),
        )

    def fit_largest_font(
        self,
        layout_text: str,
        text_box: TextLayoutBox,
        *,
        min_font_size: int,
        max_font_size: int,
    ) -> TextLayoutChoice | None:
        if min_font_size > max_font_size:
            return None

        low = min_font_size
        high = max_font_size
        best_size: int | None = None
        best_bbox: tuple[int, int, int, int] | None = None
        best_stroke = THUMBNAIL_TEXT_STROKE_MIN_WIDTH

        while low <= high:
            mid = (low + high) // 2
            stroke_width = stroke_width_for_font_size(mid)
            text_bbox = self.measure_text_block(layout_text, mid, stroke_width=stroke_width)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            if text_width <= text_box.width and text_height <= text_box.height:
                best_size = mid
                best_bbox = text_bbox
                best_stroke = stroke_width
                low = mid + 1
            else:
                high = mid - 1

        if best_size is None or best_bbox is None:
            return None

        return TextLayoutChoice(
            text=layout_text,
            line_count=layout_text.count("\n") + 1,
            font_size=best_size,
            text_bbox=best_bbox,
            block_size=(best_bbox[2] - best_bbox[0], best_bbox[3] - best_bbox[1]),
            stroke_width=best_stroke,
            shadow_offset=shadow_offset_for_font_size(best_size),
        )

    def measure_text_block(
        self,
        text: str,
        font_size: int,
        *,
        stroke_width: int | None = None,
    ) -> tuple[int, int, int, int]:
        if stroke_width is None:
            stroke_width = stroke_width_for_font_size(font_size)
        return self.draw.multiline_textbbox(
            (0, 0),
            text,
            font=self.font_loader(font_size),
            spacing=line_spacing(font_size),
            stroke_width=stroke_width,
        )


def line_spacing(font_size: int) -> int:
    return max(8, int(font_size * THUMBNAIL_TEXT_LINE_SPACING_RATIO))


def stroke_width_for_font_size(font_size: int) -> int:
    return max(THUMBNAIL_TEXT_STROKE_MIN_WIDTH, int(font_size * THUMBNAIL_TEXT_STROKE_WIDTH_RATIO))


def shadow_offset_for_font_size(font_size: int) -> tuple[int, int]:
    offset = max(2, int(font_size * THUMBNAIL_TEXT_SHADOW_OFFSET_RATIO))
    return (offset, offset)


def resolve_text_position(
    layout: TextLayoutChoice,
    text_box: TextLayoutBox,
) -> tuple[int, int]:
    centered_x = text_box.x + int((text_box.width - layout.block_size[0]) / 2)
    x = max(text_box.x, centered_x)
    y = text_box.y
    min_y = text_box.y
    max_y = text_box.y + text_box.height - layout.block_size[1]
    return (x, max(min_y, min(y, max_y)))
