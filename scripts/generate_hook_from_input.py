"""Genereer een spoken hook op basis van ``input/hook.txt``."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from spurgeon.config.settings import load_settings
from spurgeon.services.intro.generate_spoken_hook import (
    SpokenHookValidationError,
    generate_spoken_hook,
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


def main() -> None:
    settings = load_settings()
    reading_text = _read_input(DEFAULT_INPUT_PATH)
    hook = generate_spoken_hook(reading_text, settings)

    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_PATH.write_text(f"{hook}\n", encoding="utf-8")

    print(hook)
    print(f"Hook opgeslagen in: {DEFAULT_OUTPUT_PATH}")


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
