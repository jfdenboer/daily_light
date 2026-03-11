"""Text layout helpers for thumbnail rendering."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

from PIL import ImageDraw, ImageFont

THUMBNAIL_TEXT_HORIZONTAL_MARGIN_FRACTION = 0.07
THUMBNAIL_TEXT_VERTICAL_MARGIN_FRACTION = 0.10
THUMBNAIL_TEXT_COLUMN_WIDTH_FRACTION = 0.41
THUMBNAIL_TEXT_CENTER_ZONE_HEIGHT_FRACTION = 0.18
THUMBNAIL_TEXT_VERTICAL_CENTER_BIAS_FRACTION = 0.03

THUMBNAIL_TEXT_MAX_LINES = 3
THUMBNAIL_TEXT_FONT_SIZE = 104
THUMBNAIL_TEXT_LINE_SPACING_RATIO = 0.10

THUMBNAIL_TEXT_STROKE_WIDTH_RATIO = 0.006
THUMBNAIL_TEXT_STROKE_MIN_WIDTH = 0
THUMBNAIL_TEXT_SHADOW_OFFSET_RATIO = 0.0015
THUMBNAIL_TEXT_SHADOW_MIN_OFFSET = 1
THUMBNAIL_TEXT_SHADOW_ALPHA = 20
THUMBNAIL_TEXT_TRACKING_RATIO = 0.003
THUMBNAIL_TEXT_TRACKING_MIN = 0
THUMBNAIL_TEXT_TRACKING_MAX = 1


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
        return "Daily Light"
    return _to_title_case_preserving_apostrophes(normalised)


def apply_thumbnail_line_break_logic(text: str) -> str:
    words = [word for word in text.replace("\n", " ").split(" ") if word]
    if len(words) <= 2 or THUMBNAIL_TEXT_MAX_LINES <= 1:
        return " ".join(words)

    lines: list[str] = []
    index = 0
    while index < len(words):
        remaining = len(words) - index
        slots_left = THUMBNAIL_TEXT_MAX_LINES - len(lines)
        if slots_left <= 1:
            lines.append(" ".join(words[index:]))
            break

        if remaining <= 2:
            lines.append(" ".join(words[index:]))
            break

        lines.append(" ".join(words[index : index + 2]))
        index += 2

    return "\n".join(lines[:THUMBNAIL_TEXT_MAX_LINES])


def _to_title_case_preserving_apostrophes(text: str) -> str:
    words = text.split(" ")
    titled_words: list[str] = []
    for word in words:
        if not word:
            titled_words.append(word)
            continue

        letters = re.split(r"([A-Za-z]+(?:'[A-Za-z]+)*)", word)
        rebuilt: list[str] = []
        for token in letters:
            if not token:
                continue
            if re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)*", token):
                rebuilt.append(token[0].upper() + token[1:].lower())
            else:
                rebuilt.append(token)
        titled_words.append("".join(rebuilt))

    return " ".join(titled_words)


def calculate_text_layout_box(canvas_size: tuple[int, int]) -> TextLayoutBox:
    width, height = canvas_size
    horizontal_margin = int(width * THUMBNAIL_TEXT_HORIZONTAL_MARGIN_FRACTION)
    vertical_margin = int(height * THUMBNAIL_TEXT_VERTICAL_MARGIN_FRACTION)
    max_text_width = int(width * THUMBNAIL_TEXT_COLUMN_WIDTH_FRACTION)
    max_allowed_text_width = width - (horizontal_margin * 2)
    text_width = min(max_text_width, max_allowed_text_width)
    max_center_zone_height = max(100, height - (vertical_margin * 2))
    center_zone_height = min(
        max(100, int(height * THUMBNAIL_TEXT_CENTER_ZONE_HEIGHT_FRACTION)),
        max_center_zone_height,
    )
    center_zone_y = max(vertical_margin, int((height - center_zone_height) / 2))
    return TextLayoutBox(
        x=horizontal_margin,
        y=center_zone_y,
        width=max(320, text_width),
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

    def select_text_layout(self, display_text: str) -> TextLayoutChoice:
        """Return a fixed-size, max-3-line layout choice.

        Deliberately simple policy:
        - fixed font size
        - one line-break heuristic with 2-word grouping
        - overflow is reported by caller
        """
        layout_text = apply_thumbnail_line_break_logic(display_text)
        font_size = THUMBNAIL_TEXT_FONT_SIZE
        stroke_width = stroke_width_for_font_size(font_size)
        text_bbox = self.measure_text_block(layout_text, font_size, stroke_width=stroke_width)
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
    return max(THUMBNAIL_TEXT_STROKE_MIN_WIDTH, round(font_size * THUMBNAIL_TEXT_STROKE_WIDTH_RATIO))


def shadow_offset_for_font_size(font_size: int) -> tuple[int, int]:
    offset = max(
        THUMBNAIL_TEXT_SHADOW_MIN_OFFSET,
        round(font_size * THUMBNAIL_TEXT_SHADOW_OFFSET_RATIO),
    )
    return (offset, offset)


def tracking_for_font_size(font_size: int) -> int:
    tracked = int(font_size * THUMBNAIL_TEXT_TRACKING_RATIO)
    return max(THUMBNAIL_TEXT_TRACKING_MIN, min(THUMBNAIL_TEXT_TRACKING_MAX, tracked))


def resolve_text_position(
    layout: TextLayoutChoice,
    text_box: TextLayoutBox,
) -> tuple[int, int]:
    x = text_box.x
    centered_y = text_box.y + int((text_box.height - layout.block_size[1]) / 2)
    editorial_bias = int(text_box.height * THUMBNAIL_TEXT_VERTICAL_CENTER_BIAS_FRACTION)
    y = centered_y + editorial_bias
    min_y = text_box.y
    max_y = text_box.y + text_box.height - layout.block_size[1]
    return (x, max(min_y, min(y, max_y)))
