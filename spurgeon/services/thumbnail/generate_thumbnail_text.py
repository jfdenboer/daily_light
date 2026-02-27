from __future__ import annotations

"""Generate short thumbnail copy using the OpenAI API."""

import logging
import re
from typing import Final

from openai import OpenAI, OpenAIError

from spurgeon.config.settings import Settings
from spurgeon.models import Reading
from spurgeon.utils.retry_utils import retry_with_backoff

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_THUMBNAIL: Final[str] = r"""
You are a thumbnail copywriter for the YouTube channel “Voice of Faith: Spurgeon – Morning and Evening”.
You will receive the FULL devotional text (either morning or evening) as the user message.
Your task: produce **one** short key phrase for the **thumbnail image** only.

OUTPUT RULES
- Return ONE line of text only (no quotes, no JSON, no extra words).
- 3–4 words, Title Case (Capitalize Major Words).
- No emojis, no verse numbers, no dates, no author names, no references (e.g., “John 3:16”), no punctuation.
- Aim for ≤ 28 characters if possible; never exceed 4 words.
- If your first idea uses more than four words, revise it by merging or removing filler words until it fits.
- Language: English.

CREATIVE DIRECTION
- Read the devotional and abstract its heart (comfort, repentance, assurance, trust, grace, holiness, hope).
- Prefer concrete devotional nouns/verbs (Light, Mercy, Shepherd, Refuge, Grace, Rest, Trust, Faithful, Delight).
- Avoid repeating long phrases verbatim from the text; distill the essence instead.

RETURN FORMAT
- Plain text only, exactly the key phrase line.
"""


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
        self.model = settings.prompt_model
        self.temperature = settings.prompt_temperature
        self.max_tokens = 25

    def generate(self, reading: Reading) -> str:
        """Return thumbnail copy for *reading* via OpenAI."""

        def call_openai() -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_THUMBNAIL},
                    {"role": "user", "content": reading.text},
                ],
            )

            content = response.choices[0].message.content or ""
            thumbnail_text = self._sanitize_thumbnail_text(content)
            if not thumbnail_text:
                raise ThumbnailTextGenerationError("Ontvangen thumbnailtekst is leeg.")
            return thumbnail_text

        return retry_with_backoff(
            func=call_openai,
            max_retries=3,
            backoff=1.0,
            error_types=(OpenAIError, ThumbnailTextGenerationError),
            context="OpenAI thumbnail text generation",
        )

    def fallback(self, reading: Reading, title: str) -> str:
        """Derive a sensible fallback thumbnail text from the title."""

        base = title.split("|")[0].strip()
        base = re.sub(r"[\-–—]", " ", base)
        base = re.sub(r"[^\w\s]", " ", base)
        base = re.sub(r"\s+", " ", base).strip()
        if not base:
            base = f"{reading.reading_type.value} Devotional Hope"

        fallback_text = self._sanitize_thumbnail_text(base)
        if fallback_text:
            return fallback_text

        logger.debug("Fallback thumbnail text sanitization resulted in empty string.")
        return self._sanitize_thumbnail_text(f"{reading.reading_type.value} Faith Renewal") or (
            f"{reading.reading_type.value} Faith Renewal"
        )

    def _sanitize_thumbnail_text(self, raw_text: str) -> str:
        """Normalise *raw_text* to comply with thumbnail constraints."""

        text = raw_text.replace("\n", " ")
        text = re.sub(r"^['\"“”‘’`]+|['\"“”‘’`]+$", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        text = re.sub(r"[^\w\s]", "", text)
        words = [word for word in text.split() if word]
        if not words:
            return ""

        if len(words) > 4:
            logger.debug("Thumbnail text too long (%d words) – refining", len(words))
            words = self._refine_word_count(words, max_words=4)

        fillers = ["Grace", "Hope", "Faith", "Mercy"]
        filler_index = 0
        while len(words) < 3:
            filler = fillers[filler_index % len(fillers)]
            if not words or words[-1].lower() != filler.lower():
                words.append(filler)
            filler_index += 1

        title_cased_words = [
            self._title_case_word(word, index) for index, word in enumerate(words)
        ]
        candidate = " ".join(title_cased_words)

        if len(candidate) > 28 and len(words) > 3:
            logger.debug("Thumbnail text exceeds 28 characters – trimming")
            while len(candidate) > 28 and len(words) > 3:
                words = words[:-1]
                title_cased_words = [
                    self._title_case_word(word, index) for index, word in enumerate(words)
                ]
                candidate = " ".join(title_cased_words)

        return candidate.strip().replace(" ", "\n")

    def _title_case_word(self, word: str, index: int) -> str:
        lower = word.lower()
        if index != 0 and lower in self._SMALL_WORDS:
            return lower
        return lower.capitalize()

    def _refine_word_count(self, words: list[str], max_words: int) -> list[str]:
        """Reduce *words* to *max_words* while preserving salient terms."""

        refined = words[:]

        def remove_matching(predicate) -> bool:
            for idx in range(len(refined) - 1, 0, -1):
                if predicate(idx):
                    del refined[idx]
                    return True
            return False

        while len(refined) > max_words:
            if remove_matching(lambda idx: refined[idx].lower() in self._SMALL_WORDS):
                continue

            # Remove duplicate words (case-insensitive) while preferring later occurrences.
            lower_seen: set[str] = set()
            duplicate_index = None
            for idx in range(len(refined) - 1, 0, -1):
                lower_word = refined[idx].lower()
                if lower_word in lower_seen:
                    duplicate_index = idx
                    break
                lower_seen.add(lower_word)
            if duplicate_index is not None:
                del refined[duplicate_index]
                continue

            shortest_index = min(
                range(1, len(refined)), key=lambda idx: (len(refined[idx]), idx)
            )
            del refined[shortest_index]

        return refined

__all__ = ["ThumbnailTextGenerator", "ThumbnailTextGenerationError", "SYSTEM_PROMPT_THUMBNAIL"]