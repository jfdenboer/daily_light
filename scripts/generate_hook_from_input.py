"""Genereer een spoken hook via de Generator→Judge hook-pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import ValidationError

from spurgeon.config.settings import load_settings
from spurgeon.services.intro.generate_spoken_hook import (
    SpokenHookValidationError,
    generate_spoken_hook,
    validate_candidate,
)

DEFAULT_INPUT_PATH = Path("input/hook.txt")
DEFAULT_OUTPUT_PATH = Path("output/generated_hook.txt")


def _read_input(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Inputbestand niet gevonden: {path}. Maak dit bestand eerst aan."
        )

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Inputbestand is leeg: {path}")

    return text


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genereer één gesproken hook op basis van een inputbestand."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Pad naar reading-input (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Pad voor output-hook (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    settings = load_settings()
    reading_text = _read_input(args.input)
    hook = generate_spoken_hook(reading_text, settings)

    violations = validate_candidate(hook)
    if violations:
        raise SpokenHookValidationError(
            "Pipeline gaf een niet-conforme hook terug: " + ", ".join(violations)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"{hook}\n", encoding="utf-8")

    print(hook)
    print(f"Hook opgeslagen in: {args.output}")
    print(
        "Pipeline-config: "
        f"generator={settings.hook_generator_model}@{settings.hook_generator_temperature}, "
        f"judge={settings.hook_judge_model}@{settings.hook_judge_temperature}, "
        f"candidates={settings.hook_num_candidates}"
    )


if __name__ == "__main__":
    try:
        main()
    except (
        FileNotFoundError,
        ValidationError,
        ValueError,
        SpokenHookValidationError,
    ) as error:
        raise SystemExit(str(error))
