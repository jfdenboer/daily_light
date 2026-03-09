from __future__ import annotations

"""Generate short thumbnail copy using a two-step OpenAI pipeline."""

import logging
import re
from functools import lru_cache
from pathlib import Path

from openai import OpenAI, OpenAIError

from spurgeon.config.settings import Settings
from spurgeon.models import Reading
from spurgeon.utils.retry_utils import retry_with_backoff

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=1)
def _load_thumbnail_generator_prompt() -> str:
    prompt_path = PROMPTS_DIR / "thumbnail_text_generator.v1.txt"
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ThumbnailTextGenerationError(
            "Missing thumbnail generator prompt template: thumbnail_text_generator.v1.txt"
        ) from exc

@lru_cache(maxsize=1)
def _load_thumbnail_selector_prompt() -> str:
    prompt_path = PROMPTS_DIR / "thumbnail_text_selector.v1.txt"
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ThumbnailTextGenerationError(
            "Missing thumbnail selector prompt template: thumbnail_text_selector.v1.txt"
        ) from exc


class ThumbnailTextGenerationError(RuntimeError):
    """Raised when thumbnail text generation fails."""


class ThumbnailTextGenerator:
    """Generate succinct thumbnail copy for a devotional reading."""

    _SMALL_WORDS = {
        "a",
        "an",
        "and",
        "as",
        "at",
        "but",
        "by",
        "for",
        "from",
        "in",
        "into",
        "nor",
        "of",
        "on",
        "or",
        "over",
        "per",
        "the",
        "to",
        "with",
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.generator_model = settings.thumbnail_text_generator_model
        self.selector_model = settings.thumbnail_text_selector_model
        self.generator_temperature = settings.thumbnail_text_generator_temperature
        self.selector_temperature = settings.thumbnail_text_selector_temperature
        self.num_candidates = settings.thumbnail_text_num_candidates
        self.generator_max_tokens = 120
        self.selector_max_tokens = 16


    def generate(self, reading: Reading) -> str:
        """Return one final thumbnail phrase using generate-then-select."""

        winner, _ = self.generate_with_candidates(reading)
        return winner

    def generate_with_candidates(self, reading: Reading) -> tuple[str, list[str]]:
        """Return final thumbnail phrase and parsed candidates."""

        def call_openai() -> tuple[str, list[str]]:
            candidates = self._generate_candidates(reading)
            winner = self._select_candidate(reading, candidates)
            final_text = self._sanitize_thumbnail_text(winner)
            if not final_text:
                raise ThumbnailTextGenerationError("Selected thumbnail text sanitized to empty output.")
            logger.debug("Final thumbnail phrase: %r", final_text)
            return final_text, candidates

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
            return "daily light", []

    def _generate_candidates(self, reading: Reading) -> list[str]:
        user_sections = [f"Devotional Text:\n{reading.text}"]

        response = self.client.chat.completions.create(
            model=self.generator_model,
            temperature=self.generator_temperature,
            max_completion_tokens=self.generator_max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": _load_thumbnail_generator_prompt().format(
                        num_candidates=self.num_candidates
                    ),
                },
                {"role": "user", "content": "\n\n".join(user_sections)},
            ],
        )

        raw_output = response.choices[0].message.content or ""
        logger.debug("Raw thumbnail generator output: %r", raw_output)
        candidates = self._parse_candidates(raw_output)
        logger.debug("Parsed valid thumbnail candidates (%d): %s", len(candidates), candidates)

        if len(candidates) < 3:
            raise ThumbnailTextGenerationError(
                f"Insufficient valid thumbnail candidates after parsing: {len(candidates)}"
            )

        return candidates

    def _select_candidate(
        self,
        reading: Reading,
        candidates: list[str],
    ) -> str:
        user_sections = [f"Devotional Text:\n{reading.text}"]
        user_sections.append("Candidates:\n" + "\n".join(candidates))

        response = self.client.chat.completions.create(
            model=self.selector_model,
            temperature=self.selector_temperature,
            max_completion_tokens=self.selector_max_tokens,
            messages=[
                {"role": "system", "content": _load_thumbnail_selector_prompt()},
                {"role": "user", "content": "\n\n".join(user_sections)},
            ],
        )

        raw_winner = response.choices[0].message.content or ""
        logger.debug("Raw thumbnail selector output: %r", raw_winner)
        winner = self._extract_selector_winner(raw_winner)
        if not winner:
            raise ThumbnailTextGenerationError("Selector output could not be parsed into a winner.")

        winner = self._sanitize_thumbnail_text(winner)
        if winner and winner in candidates:
            logger.debug("Selector winner accepted: %s", winner)
            return winner

        for candidate in candidates:
            normalized = self._sanitize_thumbnail_text(candidate)
            if normalized:
                logger.debug("Selector winner rejected; fallback to best parsed candidate: %s", normalized)
                return normalized

        raise ThumbnailTextGenerationError("Selector winner invalid and no candidates remained.")

    def _extract_selector_winner(self, raw_text: str) -> str:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            return ""
        first_line = lines[0]
        first_line = re.sub(r"^(?:winner\s*:\s*)", "", first_line, flags=re.IGNORECASE)
        return first_line.strip()

    def _parse_candidates(self, raw_text: str) -> list[str]:
        parsed: list[str] = []
        seen: set[str] = set()
        for line in raw_text.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            cleaned = re.sub(r"^[-*•]+\s*", "", cleaned)
            cleaned = re.sub(r"^\d+[\.)]\s*", "", cleaned)
            sanitized = self._sanitize_thumbnail_text(cleaned)
            if not sanitized:
                continue
            key = self._normalize_for_dedup(sanitized)
            if key in seen:
                continue
            seen.add(key)
            parsed.append(sanitized)
        return parsed

    def _normalize_for_dedup(self, text: str) -> str:
        return re.sub(r"[^a-z]", "", text.lower())

    def _sanitize_thumbnail_text(self, raw_text: str) -> str:
        """Normalise *raw_text* to comply with thumbnail constraints."""

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        text = (lines[0] if lines else raw_text).replace("\n", " ")
        text = re.sub(r"^['\"“”‘’`]+|['\"“”‘’`]+$", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        text = text.replace("’", "'").replace("‘", "'")
        text = re.sub(r"[^A-Za-z\s']", " ", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        words = [word for word in text.split() if word]
        if not words:
            return ""

        words = self._refine_words(words, max_words=3)
        if not words:
            return ""

        title_cased_words = [
            self._title_case_word(word, index) for index, word in enumerate(words)
        ]
        candidate = " ".join(title_cased_words)
        candidate = self._shrink_to_char_limit(candidate)

        if not candidate:
            return ""

        return candidate.strip()

    def _shrink_to_char_limit(self, candidate: str, char_limit: int = 24) -> str:
        words = candidate.split()
        while len(" ".join(words)) > char_limit and len(words) > 1:
            scored = [(self._word_priority(word), idx) for idx, word in enumerate(words)]
            _, drop_idx = min(scored, key=lambda item: (item[0], -item[1]))
            del words[drop_idx]
        compact = " ".join(words)
        return "" if len(compact) > char_limit else compact

    def _word_priority(self, word: str) -> int:
        lower = word.lower()
        score = 0
        if lower in self._SMALL_WORDS:
            score -= 2
        if len(lower) <= 2 and lower not in {"he", "me"}:
            score -= 1
        return score

    def _refine_words(self, words: list[str], max_words: int) -> list[str]:
        refined = words[:]
        while len(refined) > max_words:
            scored = [(self._word_priority(word), idx) for idx, word in enumerate(refined)]
            _, drop_idx = min(scored, key=lambda item: (item[0], -item[1]))
            del refined[drop_idx]
        return refined

    def _title_case_word(self, word: str, index: int) -> str:
        lower = word.lower()
        if index != 0 and lower in self._SMALL_WORDS:
            return lower
        return lower.capitalize()


__all__ = [
    "ThumbnailTextGenerator",
    "ThumbnailTextGenerationError",
    "SYSTEM_PROMPT_THUMBNAIL_GENERATOR",
]
