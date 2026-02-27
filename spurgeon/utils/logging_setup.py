"""Utilities for configuring structured application logging."""

from __future__ import annotations

import logging
import logging.config
import os
from numbers import Integral
from pathlib import Path
from typing import Any

from spurgeon.config.settings import Settings

DEFAULT_LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def _ensure_ascii(value: str) -> str:
    """Return ``value`` stripped of any non-ASCII characters."""

    if not value:
        return value

    if hasattr(value, "isascii"):
        if value.isascii():  # type: ignore[attr-defined]
            return value
    else:  # pragma: no cover - Python >=3.11 supports ``isascii``
        try:
            value.encode("ascii")
        except UnicodeEncodeError:
            pass
        else:
            return value

    return value.encode("ascii", "ignore").decode("ascii")


class AsciiSanitizingFormatter(logging.Formatter):
    """Formatter that strips non-ASCII characters from rendered log records."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - inherited docstring
        formatted = super().format(record)
        return _ensure_ascii(formatted)


def _resolve_log_level(raw_level: object) -> tuple[int, str | None]:
    """Convert ``raw_level`` into a logging level integer with optional warning."""

    if isinstance(raw_level, Integral):
        return int(raw_level), None

    if isinstance(raw_level, str):
        candidate = raw_level.strip()
        if not candidate:
            return logging.INFO, None

        try:
            return int(candidate), None
        except ValueError:
            normalized = candidate.upper()
            mapping = logging.getLevelNamesMapping()
            if normalized in mapping:
                return mapping[normalized], None

        return logging.INFO, f"Invalid log level '{raw_level}'. Defaulting to INFO."

    return logging.INFO, None


def _determine_log_file(settings: Settings) -> Path | None:
    """Return the configured log file path if file logging is enabled."""

    env_value = os.getenv("LOG_FILE")
    if env_value is not None:
        env_value = env_value.strip()
        if not env_value:
            return None
        return Path(env_value).expanduser().resolve()

    log_file = getattr(settings, "log_file", None)
    if not log_file:
        return None

    return Path(log_file).expanduser().resolve()


def init_logging(settings: Settings) -> None:
    """Configure application logging for both console and optional file output."""

    root = logging.getLogger()
    root.handlers.clear()

    raw_level = os.getenv("LOG_LEVEL")
    if raw_level is None:
        raw_level = getattr(settings, "log_level", "INFO")

    level, warning = _resolve_log_level(raw_level)

    fmt = getattr(settings, "log_format", DEFAULT_LOG_FORMAT)
    datefmt = getattr(settings, "log_datefmt", DEFAULT_DATE_FORMAT)

    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "()": AsciiSanitizingFormatter,
                "format": fmt,
                "datefmt": datefmt,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": level,
            "handlers": ["console"],
        },
    }

    log_path = _determine_log_file(settings)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": "standard",
            "filename": str(log_path),
            "maxBytes": int(getattr(settings, "log_file_max_bytes", 10 * 1024 * 1024)),
            "backupCount": int(getattr(settings, "log_file_backup_count", 5)),
            "delay": True,
            "encoding": "utf-8",
        }
        config["root"]["handlers"].append("file")

    logging.config.dictConfig(config)
    logging.captureWarnings(True)

    if warning:
        logging.getLogger(__name__).warning(warning)