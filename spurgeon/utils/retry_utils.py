"""Utility-functies voor het uitvoeren van retries met exponentiële backoff."""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Iterable, Type, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


def _ensure_text(value: object) -> str:
    """Zet ``value`` om naar ``str`` waarbij non-ASCII behouden blijven."""

    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:  # pragma: no cover - uiterst zeldzaam pad
        return repr(value)


def retry_with_backoff(
    func: Callable[..., T],
    *func_args: Any,
    max_retries: int,
    backoff: float,
    error_types: Iterable[Type[BaseException]],
    context: str = "operation",
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[float, float], float] = random.uniform,
    **func_kwargs: Any,
) -> T:
    """Voer ``func`` uit met retries, exponentiële backoff en jitter.

    Args:
        func: De uit te voeren callabele.
        *func_args: Positionele argumenten voor ``func``.
        max_retries: Hoe vaak we na de eerste poging opnieuw proberen.
        backoff: Basis-backoff in seconden.
        error_types: Exceptions waarop gere‑tried wordt.
        context: Beschrijving voor logberichten.
        sleep: Sleepfunctie, vooral handig om te mocken in tests.
        rng: Random generator voor jitter (default ``random.uniform``).
        **func_kwargs: Keyword-argumenten voor ``func``.

    Returns:
        De waarde die ``func`` retourneert.

    Raises:
        Laatste opgevangen exceptie na overschrijding van ``max_retries``.
    """

    if max_retries < 0:
        raise ValueError("max_retries moet 0 of groter zijn")
    if backoff <= 0:
        raise ValueError("backoff moet groter dan 0 zijn")

    handled_errors = tuple(error_types)
    if not handled_errors:
        raise ValueError("error_types mag niet leeg zijn")

    context_text = _ensure_text(context)

    for attempt in range(max_retries + 1):
        try:
            return func(*func_args, **func_kwargs)
        except handled_errors as exc:
            exc_text = _ensure_text(exc)

            if attempt == max_retries:
                logger.error(
                    "%s definitief mislukt na %d pogingen: %s",
                    context_text,
                    attempt + 1,
                    exc_text,
                    exc_info=exc,
                )
                raise

            base_delay = backoff * (2**attempt)
            delay = rng(0, base_delay)
            logger.warning(
                "%s mislukt (poging %d/%d): %s – retry over %.2fs",
                context_text,
                attempt + 1,
                max_retries,
                exc_text,
                delay,
            )
            sleep(delay)