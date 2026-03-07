"""Text layout helpers for thumbnail rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PIL import ImageDraw, ImageFont

THUMBNAIL_TEXT_MAX_WIDTH_FRACTION = 0.64
THUMBNAIL_TEXT_LEFT_MARGIN_FRACTION = 0.05
THUMBNAIL_TEXT_VERTICAL_MARGIN_FRACTION = 0.08
THUMBNAIL_TEXT_MAX_LINES = 2
THUMBNAIL_TEXT_LINE_SPACING_RATIO = 0.08
THUMBNAIL_TEXT_MIN_FONT_SIZE = 90
THUMBNAIL_TEXT_MAX_FONT_SIZE = 440
THUMBNAIL_TEXT_EMERGENCY_MIN_FONT_SIZE = 56
THUMBNAIL_TEXT_STROKE_WIDTH_RATIO = 0.024
THUMBNAIL_TEXT_STROKE_MIN_WIDTH = 2
THUMBNAIL_TEXT_SHADOW_OFFSET_RATIO = 0.018
THUMBNAIL_TEXT_SHADOW_ALPHA = 110
THUMBNAIL_TEXT_VERTICAL_ANCHOR_RATIO = 0.40


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
    left_margin = int(width * THUMBNAIL_TEXT_LEFT_MARGIN_FRACTION)
    top_margin = int(height * THUMBNAIL_TEXT_VERTICAL_MARGIN_FRACTION)
    bottom_margin = top_margin
    text_right_edge = int(width * THUMBNAIL_TEXT_MAX_WIDTH_FRACTION)
    max_text_width = text_right_edge - left_margin
    text_height = height - top_margin - bottom_margin
    return TextLayoutBox(
        x=left_margin,
        y=top_margin,
        width=max(220, max_text_width),
        height=max(200, text_height),
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
        words = display_text.split()
        layout_candidates = [display_text]
        if len(words) > 1:
            layout_candidates.extend(
                " ".join(words[:split_index]) + "\n" + " ".join(words[split_index:])
                for split_index in range(1, len(words))
            )

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

            is_two_line_tie_break = (
                measured.font_size == best.font_size
                and len(words) >= 3
                and measured.line_count == 2
                and best.line_count == 1
            )
            if is_two_line_tie_break:
                best = measured

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
    x = text_box.x
    anchor_y = text_box.y + int(text_box.height * THUMBNAIL_TEXT_VERTICAL_ANCHOR_RATIO)
    y = anchor_y - int(layout.block_size[1] / 2)
    min_y = text_box.y
    max_y = text_box.y + text_box.height - layout.block_size[1]
    return (x, max(min_y, min(y, max_y)))
