from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAIError
from pydantic import ValidationError

from spurgeon.config.settings import Settings

from .prompt_generator import (
    _DEFAULT_ENV_FILE,
    _DEFAULT_STANDALONE_INPUT,
    PromptGenerationError,
    PromptOrchestrator,
)

logger = logging.getLogger(__name__)


_ENV_FILE_ENV_VAR = "SPURGEON_ENV_FILE"


def _resolve_default_input_file() -> Path:
    """Resolve the stand-alone input file with sensible fallbacks."""

    candidates = (
        _DEFAULT_STANDALONE_INPUT,
        Path("input") / "test_input.txt",
        Path("input") / "Spurgeon_clean.txt",
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            logger.debug("Using default stand-alone input file: %s", candidate)
            return candidate

    return _DEFAULT_STANDALONE_INPUT


def _resolve_default_env_file() -> Path:
    configured = os.getenv(_ENV_FILE_ENV_VAR)
    if configured:
        return Path(configured).expanduser()
    return _DEFAULT_ENV_FILE


def _load_default_env_file(env_file: Path | None = None) -> bool:
    resolved_env_file = env_file or _resolve_default_env_file()
    exists = resolved_env_file.exists()
    is_file = resolved_env_file.is_file()
    logger.debug(
        "Controle op standaard .env-bestand (pad=%s, bestaat=%s, is_file=%s)",
        resolved_env_file,
        exists,
        is_file,
    )

    try:
        loaded = load_dotenv(resolved_env_file, override=False)
    except OSError as exc:  # pragma: no cover
        logger.debug("Kon .env-bestand %s niet laden: %s", resolved_env_file, exc)
        return False

    if loaded:
        logger.debug("Standaard .env-bestand geladen vanaf %s", resolved_env_file)
    else:
        logger.debug("Geen standaard .env-bestand gevonden op %s", resolved_env_file)
    return bool(loaded)


def _resolve_log_level(value: str | None) -> int:
    if not value:
        return logging.INFO
    level = getattr(logging, value.upper(), None)
    if isinstance(level, int):
        return level
    raise ValueError(f"Unsupported log level: {value}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate image prompts from text excerpts.")
    parser.add_argument(
        "input",
        nargs="?",
        help=(
            "Pad naar het brontekstbestand."
            " Als dit wordt weggelaten, wordt test_input.txt uit de modulemap gebruikt."
        ),
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        help="Willekeurige logniveau-override (bijv. DEBUG, INFO, WARNING).",
    )
    parser.add_argument(
        "--env-file",
        dest="env_file",
        help=(
            "Optioneel pad naar .env-bestand. "
            f"Fallback-volgorde: --env-file, ${_ENV_FILE_ENV_VAR}, default pad."
        ),
    )

    args = parser.parse_args(argv)

    desired_log_level = args.log_level or "DEBUG"
    try:
        log_level = _resolve_log_level(desired_log_level)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    logger.debug("Stand-alone prompt generator gestart (argv=%s)", argv or sys.argv[1:])

    explicit_env_file = Path(args.env_file).expanduser() if args.env_file else None
    selected_env_file = explicit_env_file or _resolve_default_env_file()
    env_loaded = _load_default_env_file(selected_env_file)
    logger.debug("Standaard .env geladen: %s", env_loaded)

    try:
        settings = Settings()
    except (ValidationError, OSError) as exc:
        print("Configuratiefout: kon Settings niet initialiseren.", file=sys.stderr)
        if not selected_env_file.exists():
            print(
                "Controleer of het .env-bestand aanwezig is op",
                selected_env_file,
                file=sys.stderr,
            )
        print(exc, file=sys.stderr)
        return 1

    input_path = Path(args.input).expanduser() if args.input else _resolve_default_input_file()
    try:
        input_text = input_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error("Bronbestand %s niet gevonden", input_path)
        return 1

    orchestrator = PromptOrchestrator(settings)

    try:
        prompt = orchestrator.build_prompt(input_text)
    except (PromptGenerationError, OpenAIError) as exc:
        logger.error("Promptgeneratie mislukt: %s", exc)
        return 1

    print(prompt)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
