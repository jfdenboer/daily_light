"""Genereer thumbnailtekst + intent cards voor alle januari-readings (1-31, morning/evening)."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from openai import OpenAI, OpenAIError
from pydantic import ValidationError

from spurgeon.config.settings import load_settings
from spurgeon.core.parser import Parser
from spurgeon.models import Reading
from spurgeon.services.thumbnail.generate_thumbnail_text import (
    ThumbnailTextGenerationError,
    ThumbnailTextGenerator,
)
from spurgeon.services.thumbnail.thumbnail_adapters import OpenAIIntentCardProvider
from spurgeon.services.thumbnail.thumbnail_errors import IntentCardError
from spurgeon.utils.retry_utils import retry_with_backoff

DEFAULT_INPUT_DIR = Path("input")
DEFAULT_OUTPUT_PATH = Path("output/thumbnailtext_january.txt")
logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Genereer voor januari (1-31, morning/evening) per reading een intent card "
            "en thumbnailtekst op basis van input/*.txt."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Map met bronbestanden (.txt), default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Outputbestand (.txt), default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=date.today().year,
        help="Jaar voor parsing van headers zonder jaar (default: huidig jaar).",
    )
    return parser


def _load_readings(input_dir: Path, *, year: int) -> list[Reading]:
    parser = Parser()
    texts = parser.load_texts(input_dir)

    readings: list[Reading] = []
    for index, raw_text in enumerate(texts, start=1):
        source_name = f"{input_dir}/<file-{index}>"
        readings.extend(parser.parse(raw_text, year=year, source_name=source_name))

    january_readings = [
        reading
        for reading in readings
        if reading.date.month == 1 and 1 <= reading.date.day <= 31
    ]
    january_readings.sort(
        key=lambda reading: (
            reading.date,
            0 if reading.reading_type.value.lower() == "morning" else 1,
        )
    )

    if not january_readings:
        raise ValueError("Geen januari-readings gevonden in de inputmap.")

    return january_readings


def _format_entry(
    reading: Reading,
    intent_card,
    thumbnail_text: str,
    generated_candidates: list[str],
    judged_candidates: list[str],
    selector_candidates: list[str],
    judge_fallback_used: bool,
) -> str:
    reading_type = reading.reading_type.value.lower()
    lines = [
        f"datum: {reading.date.isoformat()}",
        f"type: {reading_type}",
        "intent card:",
        f"1) core_tension: {intent_card.core_tension}",
        f"2) emotional_tone: {intent_card.emotional_tone}",
        f"3) dominant_anchor: {intent_card.dominant_anchor}",
        f"4) scene_direction: {intent_card.scene_direction}",
        f"5) open_loop: {intent_card.open_loop}",
        f"6) avoid: {intent_card.avoid}",
        "thumbnail generator kandidaten:",
        *(f"- {candidate}" for candidate in generated_candidates),
        "thumbnail judge kandidaten:",
        *(f"- {candidate}" for candidate in judged_candidates),
        f"judge fallback gebruikt: {'ja' if judge_fallback_used else 'nee'}",
        "thumbnail selector kandidaten:",
        *(f"- {candidate}" for candidate in selector_candidates),
        f"thumbnail winnaar: {thumbnail_text}",
    ]
    return "\n".join(lines)


def _generate_thumbnail_with_judge_step(
    text_generator: ThumbnailTextGenerator,
    reading: Reading,
) -> tuple[str, list[str], list[str], list[str], bool]:
    def call_openai() -> tuple[str, list[str], list[str], list[str], bool]:
        generated_candidates = text_generator._generate_candidates(reading)
        judged_candidates = text_generator.judge.judge_thumbnail_text_candidates(
            reading, generated_candidates
        )
        judge_fallback_used = not judged_candidates
        selector_candidates = (
            judged_candidates if judged_candidates else generated_candidates
        )
        winner = text_generator._select_candidate(reading, selector_candidates)
        thumbnail_text = text_generator._sanitize_thumbnail_text(winner)
        if not thumbnail_text:
            raise ThumbnailTextGenerationError(
                "Selected thumbnail text sanitized to empty output."
            )
        return (
            thumbnail_text,
            generated_candidates,
            judged_candidates,
            selector_candidates,
            judge_fallback_used,
        )

    try:
        return retry_with_backoff(
            func=call_openai,
            max_retries=3,
            backoff=1.0,
            error_types=(OpenAIError, ThumbnailTextGenerationError),
            context="OpenAI thumbnail text generation",
        )
    except (OpenAIError, ThumbnailTextGenerationError) as exc:
        logger.warning("Thumbnail text pipeline failed, using fallback: %s", exc)
        return "daily light", [], [], [], True


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    parser = _build_arg_parser()
    args = parser.parse_args()

    settings = load_settings()
    readings = _load_readings(args.input_dir, year=args.year)

    client = OpenAI(api_key=settings.openai_api_key)
    intent_provider = OpenAIIntentCardProvider(client=client, settings=settings)
    text_generator = ThumbnailTextGenerator(settings=settings)

    output_chunks: list[str] = []
    for reading in readings:
        (
            thumbnail_text,
            generated_candidates,
            judged_candidates,
            selector_candidates,
            judge_fallback_used,
        ) = _generate_thumbnail_with_judge_step(text_generator, reading)

        logger.info(
            "JUDGE stap %s: generated=%d judged=%d fallback=%s selector=%d winner=%s",
            reading.slug,
            len(generated_candidates),
            len(judged_candidates),
            judge_fallback_used,
            len(selector_candidates),
            thumbnail_text,
        )

        try:
            intent_card = intent_provider.generate(reading, thumbnail_text=thumbnail_text)
        except IntentCardError as exc:
            raise RuntimeError(
                f"Intent card generatie faalde voor {reading.slug}: {exc}"
            ) from exc
        output_chunks.append(
            _format_entry(
                reading,
                intent_card,
                thumbnail_text,
                generated_candidates,
                judged_candidates,
                selector_candidates,
                judge_fallback_used,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n---\n\n".join(output_chunks) + "\n", encoding="utf-8")

    print(f"Klaar. {len(output_chunks)} entries opgeslagen in: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValidationError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error))
