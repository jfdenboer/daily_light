from __future__ import annotations

"""
spurgeon.core.parser

Parse raw devotional text files into :class:`spurgeon.models.Reading`.

Key features:
- Header detection for Daily Light format ("JANUARY 1 MORNING").
- Automatic year-rollover (Dec → Jan).
- Dynamic leap-year mapping for Feb-29.
- Whitespace-normalization (CRLF → LF, configurable blank-line collapse).
- Case-insensitive, recursive `.txt` discovery.
- Improved error context with source filenames.
- Warning when defaulting to current year for reproducibility.
"""

import calendar
import codecs
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Sequence, Union

from spurgeon.models import Reading, ReadingType

logger = logging.getLogger(__name__)

_INVISIBLE_SEPARATOR_PATTERN: re.Pattern[str] = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
_LINEBREAK_PATTERN: re.Pattern[str] = re.compile(r"\r\n|[\r\v\f\x85\u2028\u2029]")
_BODY_SEPARATOR_LINE_PATTERN: re.Pattern[str] = re.compile(r"^\s*_+\s*$")

HEADER_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*(?P<month>[A-Za-z\.]+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?[\s\.,:\-–—]*"
    r"(?P<rtype>Morning|Evening)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)

_HEADER_LIKE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(morning|evening)\b",
    flags=re.IGNORECASE,
)

_MONTH_MAP: dict[str, int] = {
    name.lower(): num
    for num, name in enumerate(calendar.month_name)
    if name
}
_MONTH_MAP.update({
    abbr.lower(): num
    for num, abbr in enumerate(calendar.month_abbr)
    if abbr
})
_MONTH_MAP.update({"sept": 9})


def _next_leap_year(start_year: int) -> int:
    y = start_year
    while not calendar.isleap(y):
        y += 1
    return y


class Parser:
    """Parse devotional text files into :class:`Reading` objects."""

    def __init__(self, header_pattern: Optional[re.Pattern[str]] = None, max_blank_lines: int = 1) -> None:
        self.header_pattern = header_pattern or HEADER_PATTERN
        self.max_blank_lines = max_blank_lines

    # ------------------------------------------------------------------ #
    # I/O helpers
    # ------------------------------------------------------------------ #
    def load_texts(
        self,
        input_dir: Union[str, Path],
        file_extensions: Optional[List[str]] = None,
    ) -> List[str]:
        p = Path(input_dir)
        if not p.is_dir():
            raise FileNotFoundError(f"Input directory not found: {p}")

        exts = [e.lower() for e in (file_extensions or [".txt"])]
        texts: List[str] = []

        for path in sorted(p.rglob("*")):
            if path.is_file() and path.suffix.lower() in exts:
                raw = path.read_text(encoding="utf-8-sig", errors="ignore")
                texts.append(self._normalise(raw))
        logger.debug("Loaded %d files from %s", len(texts), p)
        return texts

    # ------------------------------------------------------------------ #
    # main entry
    # ------------------------------------------------------------------ #
    def parse(
        self,
        raw_text: str,
        year: Optional[int] = None,
        source_name: Optional[str] = None,
    ) -> List[Reading]:
        raw = self._normalise(raw_text)
        matches = list(self.header_pattern.finditer(raw))
        if not matches and "\\n" in raw_text:
            raw = self._normalise(self._decode_escaped_linebreaks(raw_text))
            matches = list(self.header_pattern.finditer(raw))

        if not matches and self._looks_unicode_escaped(raw_text):
            raw = self._normalise(self._decode_unicode_escapes(raw_text))
            matches = list(self.header_pattern.finditer(raw))

        if not matches:
            ctx = f" ({source_name})" if source_name else ""
            first_line = raw.split("\n", 1)[0][:120] if raw else "<empty input>"
            header_hints = self._collect_header_hints(raw)
            hint_msg = (
                f" Header-like lines seen: {header_hints}."
                if header_hints
                else ""
            )
            raise ValueError(
                "No devotional headers found in supplied text"
                f"{ctx}. Expected lines like 'JANUARY 1 MORNING' (year is passed separately). "
                f"First line was: {first_line!r}.{hint_msg}"
            )

        readings: List[Reading] = []
        prev_month: Optional[int] = None
        current_year = self._resolve_start_year(year, source_name)

        for idx, m in enumerate(matches):
            rtype = self._parse_reading_type(m.group("rtype"))
            month_token = m.group("month")
            day_num = int(m.group("day"))

            month_num = self._resolve_month(month_token, source_name)
            current_year = self._adjust_year(current_year, prev_month, month_num)
            prev_month = month_num

            body = self._extract_body(raw, matches, idx)
            actual_date = self._build_date(
                current_year,
                month_num,
                day_num,
                rtype,
                source_name,
            )

            readings.append(
                Reading(
                    date=actual_date,
                    reading_type=rtype,
                    text=body,
                )
            )

        logger.info(
            "Parsed %d/%d headers into Reading objects%s.",
            len(readings),
            len(matches),
            f" ({source_name})" if source_name else "",
        )
        return readings

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _decode_escaped_linebreaks(text: str) -> str:
        """Decode literal escape sequences that are often produced by copied JSON strings."""
        return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")

    @staticmethod
    def _looks_unicode_escaped(text: str) -> bool:
        return any(token in text for token in ("\\u", "\\x", "\\U", "\\N{"))

    @staticmethod
    def _decode_unicode_escapes(text: str) -> str:
        """Best-effort decoding for strings that contain literal unicode escape sequences."""
        try:
            return codecs.decode(text, "unicode_escape")
        except Exception:
            return text

    def _normalise(self, text: str) -> str:
        text = _INVISIBLE_SEPARATOR_PATTERN.sub("", text)
        text = _LINEBREAK_PATTERN.sub("\n", text)
        lines = text.split("\n")
        normalised: List[str] = []
        blank_run = 0

        for line in lines:
            if line.strip():
                blank_run = 0
                normalised.append(line.rstrip())
            else:
                blank_run += 1
                if blank_run <= self.max_blank_lines:
                    normalised.append("")

        return "\n".join(normalised).strip()

    @staticmethod
    def _collect_header_hints(raw: str) -> List[str]:
        """Return a few lines that look like potential headers for faster diagnostics."""
        hints: List[str] = []
        for line in raw.split("\n"):
            candidate = line.strip()
            if not candidate:
                continue
            if _HEADER_LIKE_PATTERN.search(candidate):
                hints.append(candidate[:120])
            if len(hints) >= 3:
                break
        return hints

    @staticmethod
    def _parse_reading_type(raw: str) -> ReadingType:
        try:
            return ReadingType[raw.strip().upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown reading type '{raw.strip()}'.") from exc

    @staticmethod
    def _format_source(source_name: Optional[str]) -> str:
        return f" ({source_name})" if source_name else ""

    def _resolve_start_year(self, year: Optional[int], source_name: Optional[str]) -> int:
        if year is not None:
            return year

        fallback = date.today().year
        logger.warning(
            "No year provided for parsing%s; defaulting to current year %d. "
            "For reproducibility, pass an explicit year.",
            self._format_source(source_name),
            fallback,
        )
        return fallback

    def _resolve_month(self, token: str, source_name: Optional[str]) -> int:
        month_clean = token.strip().lower().rstrip(".")
        month_num = _MONTH_MAP.get(month_clean)
        if month_num is None:
            ctx = f" in {source_name}" if source_name else ""
            raise ValueError(f"Unknown month name '{token}'{ctx}.")
        return month_num

    @staticmethod
    def _adjust_year(current_year: int, prev_month: Optional[int], month_num: int) -> int:
        if prev_month is not None and month_num < prev_month:
            return current_year + 1
        return current_year

    def _build_date(
        self,
        year: int,
        month: int,
        day: int,
        reading_type: ReadingType,
        source_name: Optional[str],
    ) -> date:
        try:
            return date(year, month, day)
        except ValueError as exc:
            if month == 2 and day == 29:
                next_leap = _next_leap_year(year)
                logger.info(
                    "Non-leap year %d: mapping Feb-29 -> %d-02-29 for %s%s",
                    year,
                    next_leap,
                    reading_type,
                    self._format_source(source_name),
                )
                return date(next_leap, 2, 29)

            ctx = self._format_source(source_name)
            raise ValueError(
                "Invalid calendar date for %s – %04d-%02d-%02d%s"
                % (reading_type.value, year, month, day, ctx)
            ) from exc

    @staticmethod
    def _extract_body(
        raw: str, matches: Sequence[re.Match[str]], index: int
    ) -> str:
        match = matches[index]
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        if not body:
            return body

        cleaned_lines = [line for line in body.split("\n") if not _BODY_SEPARATOR_LINE_PATTERN.match(line)]
        return "\n".join(cleaned_lines).strip()


# ---------------------------------------------------------------------------
# Top-level helpers
# ---------------------------------------------------------------------------
def split_readings(raw_text: str, year: Optional[int] = None) -> List[Reading]:
    return Parser().parse(raw_text, year)


def load_all_readings(input_dir: Union[str, Path], year: Optional[int] = None) -> List[Reading]:
    parser = Parser()
    readings: List[Reading] = []
    p = Path(input_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Input directory not found: {p}")

    for txt_path in sorted(p.rglob("*")):
        if not txt_path.is_file() or txt_path.suffix.lower() != ".txt":
            continue
        raw = txt_path.read_text(encoding="utf-8-sig", errors="ignore")
        try:
            readings.extend(parser.parse(raw, year, source_name=str(txt_path)))
        except Exception as exc:
            raise ValueError(f"Error parsing '{txt_path}': {exc}") from exc
    return readings