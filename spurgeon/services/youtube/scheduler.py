"""Compute the next publication datetime for a devotional video."""

from __future__ import annotations

import logging
from datetime import date, datetime, time
from typing import Final, Mapping, Union

import pytz
from pytz import BaseTzInfo, UnknownTimeZoneError


from spurgeon.models import ReadingType

__all__ = ["next_publish_datetime"]

logger = logging.getLogger(__name__)

_PUBLISH_TIME_BY_TYPE: Final[Mapping[ReadingType, time]] = {
    ReadingType.MORNING: time(7, 0),
    ReadingType.EVENING: time(19, 0),
}

TimeZoneLike = Union[str, BaseTzInfo]


def next_publish_datetime(
    reading_date: date,
    reading_type: ReadingType,
    *,
    tz: TimeZoneLike = "Europe/Amsterdam",
    current_date: date | None = None,
) -> datetime:
    """Return the next future publication datetime for the given reading.

    Args:
        reading_date: Month and day of the devotional to schedule.
        reading_type: Determines whether the video goes live in the morning or evening.
        tz: Name of the timezone or pytz timezone instance for publication.
        current_date: Optional override for the reference date (defaults to today).

    Returns:
        A timezone-aware datetime in the requested timezone.

    Raises:
        ValueError: If the timezone string cannot be resolved.
        TypeError: If an unsupported timezone object is provided.
    """

    timezone = _resolve_timezone(tz)
    reference_date = current_date or datetime.now(timezone).date()
    publish_date = _next_calendar_date(reading_date, reference_date)
    publish_time = _PUBLISH_TIME_BY_TYPE.get(reading_type)
    if publish_time is None:
        raise ValueError(f"Unsupported reading type: {reading_type!r}")

    return _localize(datetime.combine(publish_date, publish_time), timezone)


def _resolve_timezone(tz: TimeZoneLike) -> BaseTzInfo:
    if isinstance(tz, BaseTzInfo):
        return tz
    if isinstance(tz, str):
        try:
            return pytz.timezone(tz)
        except UnknownTimeZoneError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unknown timezone '{tz}'") from exc
    raise TypeError("tz must be a timezone name or pytz timezone instance")


def _next_calendar_date(reading_date: date, reference: date) -> date:
    month, day = reading_date.month, reading_date.day
    year = reference.year

    while True:
        try:
            candidate = date(year, month, day)
        except ValueError:
            year += 1
            continue

        if candidate <= reference:
            year += 1
            continue

        return candidate


def _localize(naive_datetime: datetime, timezone: BaseTzInfo) -> datetime:
    localized = timezone.localize(naive_datetime)
    logger.debug("Localized publish time %s in %s", naive_datetime.isoformat(), timezone.zone)
    return localized
