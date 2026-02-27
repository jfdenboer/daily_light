# models.py

from __future__ import annotations

"""
spurgeon.models

Domain-modellen (Pydantic v2) voor de Spurgeon-pipeline.
"""

from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Tuple, TypeAlias

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Type-alias voor ruwe clip-assets
# ---------------------------------------------------------------------------
RawAsset: TypeAlias = Tuple[Path, float, str]
"""Alias voor tuples ``(image_path, duration_sec, subject_prompt)``."""

# ---------------------------------------------------------------------------
# Core domain modellen
# ---------------------------------------------------------------------------
class ReadingType(str, Enum):
    """Morning or Evening devotional."""
    MORNING = "Morning"
    EVENING = "Evening"

    def __str__(self) -> str:
        return self.value


class Reading(BaseModel):
    """Een volledige ochtend- of avondoverdenking."""

    date: Annotated[date, Field(description="Date of the reading (YYYY-MM-DD)")]
    reading_type: Annotated[ReadingType, Field(description="Morning or Evening")]
    text: Annotated[str, Field(description="Full text content")]

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_fields(cls, data: Any):
        if not isinstance(data, dict):
            return data

        data = dict(data)  # Prevent mutating caller-owned input.

        if "reading_type" not in data and "rtype" in data:
            data["reading_type"] = data.pop("rtype")

        if "date" not in data and "md" in data:
            md_val = data.pop("md")
            year_raw = data.pop("year", datetime.now().year)
            try:
                year_int = int(year_raw)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard.
                raise TypeError("Field 'year' must be coercible to int") from exc
            data["date"] = cls._coerce_month_day(md_val, year_int)

        return data

    @staticmethod
    def _coerce_month_day(value: Any, year: int) -> date:
        if isinstance(value, dict):
            try:
                month = int(value["month"])
                day = int(value["day"])
            except (KeyError, TypeError, ValueError) as exc:
                raise TypeError("Field 'md' must contain integer 'month' and 'day'") from exc
            return date(year, month, day)
        raise TypeError("Field 'md' must be a mapping with 'month' and 'day'")

    @field_validator("reading_type", mode="before")
    @classmethod
    def _normalise_reading_type(cls, value: Any) -> ReadingType:
        if isinstance(value, ReadingType):
            return value
        if isinstance(value, str):
            cleaned = value.strip().lower()
            for option in ReadingType:
                if option.value.lower() == cleaned:
                    return option
        raise ValueError("reading_type must be 'Morning' or 'Evening'")

    @field_validator("text", mode="before")
    @classmethod
    def _clean_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError("text must be a string")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text must not be empty")
        return cleaned

    @property
    def slug(self) -> str:
        """Unieke slug: YYYY-MM-DD_Morning/Evening"""
        return f"{self.date.isoformat()}_{self.reading_type.value}"

    def __str__(self) -> str:
        return f"{self.reading_type.value}, {self.date.isoformat()}"

    def __hash__(self) -> int:
        return hash(self.slug)

    @property
    def word_count(self) -> int:
        return len(self.text.split()) if self.text else 0

    model_config = {
        "extra": "forbid",
        "frozen": True,
        "str_strip_whitespace": True,
    }


__all__ = [
    "RawAsset",
    "ReadingType",
    "Reading",
]
