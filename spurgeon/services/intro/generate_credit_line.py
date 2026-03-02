"""Utilities for generating the spoken intro credit line."""

from __future__ import annotations

from typing import Final

CREDIT_LINE: Final[str] = (
    "You’re listening to: Daily Light on the Daily Path, by Samuel Bagster."
)


def generate_credit_line() -> str:
    """Return the fixed intro credit line."""

    return CREDIT_LINE


__all__ = ["generate_credit_line", "CREDIT_LINE"]
