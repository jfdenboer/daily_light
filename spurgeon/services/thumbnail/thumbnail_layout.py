"""Text layout helpers for thumbnail rendering."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

from PIL import ImageDraw, ImageFont

THUMBNAIL_TEXT_HORIZONTAL_MARGIN_FRACTION = 0.10
THUMBNAIL_TEXT_VERTICAL_MARGIN_FRACTION = 0.12
THUMBNAIL_TEXT_COLUMN_WIDTH_FRACTION = 0.41
THUMBNAIL_TEXT_CENTER_ZONE_HEIGHT_FRACTION = 0.42
THUMBNAIL_TEXT_VERTICAL_CENTER_BIAS_FRACTION = 0.04
THUMBNAIL_TEXT_MAX_LINES = 2
THUMBNAIL_TEXT_LINE_SPACING_RATIO = 0.10
THUMBNAIL_TEXT_FONT_SIZE = 160
THUMBNAIL_TEXT_FALLBACK_FONT_SIZE = 140
THUMBNAIL_TEXT_MIN_SAFE_FONT_SIZE = 104
THUMBNAIL_TEXT_STROKE_WIDTH_RATIO = 0.010
THUMBNAIL_TEXT_STROKE_MIN_WIDTH = 0
THUMBNAIL_TEXT_SHADOW_OFFSET_RATIO = 0.008
THUMBNAIL_TEXT_SHADOW_MIN_OFFSET = 1
THUMBNAIL_TEXT_SHADOW_ALPHA = 28
THUMBNAIL_TEXT_TRACKING_RATIO = 0.005
THUMBNAIL_TEXT_TRACKING_MIN = 0
THUMBNAIL_TEXT_TRACKING_MAX = 2
THUMBNAIL_THREE_WORD_ONE_LINE_MAX_COMFORT_RATIO = 0.84
THUMBNAIL_WIDTH_UTILIZATION_IDEAL_RATIO = 0.74


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

    def select_text_layout(
        self,
        display_text: str,
        text_box: TextLayoutBox,
    ) -> TextLayoutChoice:
        base_text = display_text.replace("\n", " ").strip()
        layout_candidates = self._build_layout_candidates(base_text)
        word_count = len(base_text.split())
        measured_layouts: list[TextLayoutChoice] = []

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
                measured_layouts.append(measured)

        if measured_layouts:
            one_line_ratio = self._one_line_widest_ratio(measured_layouts, text_box)
            preferred_line_count = self._preferred_line_count(word_count, one_line_ratio)
            scored_candidates = [
                (self._score_layout_choice(layout, text_box, preferred_line_count=preferred_line_count), layout)
                for layout in measured_layouts
            ]
            return max(scored_candidates, key=lambda item: item[0])[1]

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

    def _build_layout_candidates(self, base_text: str) -> list[str]:
        words = [word for word in base_text.split(" ") if word]
        if len(words) <= 2:
            return [" ".join(words)]

        one_line = " ".join(words)
        split_candidates: list[tuple[tuple[int, int, int], str]] = []
        total_words = len(words)
        for split_index in range(1, total_words):
            first_words = words[:split_index]
            second_words = words[split_index:]

            if total_words >= 4 and min(len(first_words), len(second_words)) == 1:
                orphan = first_words[0] if len(first_words) == 1 else second_words[0]
                if len(orphan) <= 3:
                    continue

            first_line = " ".join(first_words)
            second_line = " ".join(second_words)

            char_balance = abs(len(first_line) - len(second_line))
            word_balance = abs(len(first_words) - len(second_words))
            orphan_penalty = int(len(first_words) == 1 or len(second_words) == 1)
            split_candidates.append(
                ((orphan_penalty, word_balance, char_balance), f"{first_line}\n{second_line}")
            )

        ordered_splits = [candidate for _, candidate in sorted(split_candidates)]
        if not ordered_splits:
            return [one_line]

        preferred_splits = ordered_splits[:3]
        if total_words <= 5:
            return [*preferred_splits, one_line]
        return [one_line, *preferred_splits]

    def _score_layout_choice(
        self,
        layout: TextLayoutChoice,
        text_box: TextLayoutBox,
        *,
        preferred_line_count: int,
    ) -> tuple[float, ...]:
        line_preference = 1.0 if layout.line_count == preferred_line_count else 0.0

        stroke_width = layout.stroke_width
        line_widths = self._line_widths_for_layout_text(layout.text, layout.font_size, stroke_width)
        widest_ratio = (max(line_widths, default=0) / text_box.width) if text_box.width else 0
        width_utilization = self._width_utilization_score(widest_ratio)

        balance_score = 1.0
        if layout.line_count == 2 and len(line_widths) == 2:
            max_width = max(line_widths)
            if max_width > 0:
                balance_score = 1.0 - (abs(line_widths[0] - line_widths[1]) / max_width)

        return (line_preference, width_utilization, balance_score, float(layout.font_size))

    def _preferred_line_count(self, word_count: int, one_line_ratio: float | None) -> int:
        if word_count >= 4:
            return 2
        if word_count == 3:
            if one_line_ratio is None:
                return 2
            return 2 if one_line_ratio > THUMBNAIL_THREE_WORD_ONE_LINE_MAX_COMFORT_RATIO else 1
        return 1

    def _one_line_widest_ratio(
        self,
        measured_layouts: list[TextLayoutChoice],
        text_box: TextLayoutBox,
    ) -> float | None:
        one_line_layouts = [layout for layout in measured_layouts if layout.line_count == 1]
        if not one_line_layouts or not text_box.width:
            return None

        best_one_line = max(one_line_layouts, key=lambda item: item.font_size)
        line_widths = self._line_widths_for_layout_text(
            best_one_line.text,
            best_one_line.font_size,
            best_one_line.stroke_width,
        )
        return max(line_widths, default=0) / text_box.width

    def _width_utilization_score(self, widest_ratio: float) -> float:
        score = 1.0 - abs(THUMBNAIL_WIDTH_UTILIZATION_IDEAL_RATIO - widest_ratio)
        if widest_ratio > 0.90:
            score -= (widest_ratio - 0.90) * 1.5
        return max(0.0, min(1.0, score))

    def _line_widths_for_layout_text(
        self,
        text: str,
        font_size: int,
        stroke_width: int,
    ) -> list[int]:
        font = self.font_loader(font_size)
        tracking = tracking_for_font_size(font_size)
        return [
            self._measure_tracked_line_width(
                line,
                font=font,
                stroke_width=stroke_width,
                tracking=tracking,
            )
            for line in text.split("\n")
        ]

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
