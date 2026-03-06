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
You are writing thumbnail text for YouTube browse surfaces, not search.
Your job is to make people feel something and stop scrolling.
Do NOT explain the video, summarize the lesson, or write a sermon heading.

Output rules:
- Exactly one line.
- English only.
- 1-3 words only (prefer 2-3).
- Title Case.
- No punctuation, emojis, dates, verse references, author names, or numbers.
- Keep it short enough for a thumbnail (about <= 24 characters when possible).

Creative direction:
- Favor emotional signal, tension, ache, nearness, weakness, waiting, refuge, return, surrender, rest, mercy, hiddenness, turning point.
- Evoke a moment or feeling.
- Keep it minimal and curious.
- Avoid repeating the provided title.
- Avoid closely copying devotional wording.

Explicitly avoid:
- Search/tutorial phrasing: How To, Why, Guide, Tips, Steps, Best.
- Explanatory phrasing and SEO language.
- Generic devotional stacks like: Daily Light Devotional, Faith Hope Grace, Trust in God.
- Cleaned-up title fragments.

Good examples:
- Still He Holds
- When Strength Fails
- Not Left Alone
- Under His Shadow
- Before the Dawn

Bad examples:
- Daily Light Devotional
- Faith Hope Grace
- Trust in God
- How To Find Peace
- Morning Devotional Hope
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
        self.max_tokens = 16

        self._search_terms = {"how", "why", "guide", "tips", "steps", "best", "tutorial"}
        self._generic_terms = {
            "daily",
            "light",
            "devotional",
            "morning",
            "evening",
            "bible",
            "god",
        }
        self._abstract_terms = {
            "faith",
            "grace",
            "hope",
            "mercy",
            "peace",
            "trust",
            "love",
            "joy",
            "renewal",
            "blessing",
        }
        self._signal_terms = {
            "still",
            "when",
            "not",
            "before",
            "under",
            "near",
            "alone",
            "shadow",
            "rest",
            "waiting",
            "weary",
            "broken",
            "hidden",
            "holds",
            "fails",
            "finds",
            "return",
            "dawn",
            "silence",
            "stays",
        }

    def generate(self, reading: Reading, title: str | None = None) -> str:
        """Return thumbnail copy for *reading* via OpenAI."""

        def call_openai() -> str:
            user_sections = []
            if title:
                user_sections.append(f"Working Video Title:\n{title.strip()}")
            user_sections.append(f"Devotional Text:\n{reading.text}")

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_THUMBNAIL},
                    {"role": "user", "content": "\n\n".join(user_sections)},
                ],
            )

            content = response.choices[0].message.content or ""
            logger.debug("Raw thumbnail model output: %r", content)
            thumbnail_text = self._sanitize_thumbnail_text(content, title=title)
            logger.debug("Sanitized thumbnail candidate: %r", thumbnail_text)
            if not thumbnail_text:
                raise ThumbnailTextGenerationError(
                    "Generated thumbnail text was empty or rejected as weak browse copy."
                )
            return thumbnail_text

        return retry_with_backoff(
            func=call_openai,
            max_retries=3,
            backoff=1.0,
            error_types=(OpenAIError, ThumbnailTextGenerationError),
            context="OpenAI thumbnail text generation",
        )

    def fallback(self, reading: Reading, title: str) -> str:
        """Return a browse-first fallback phrase based on title + devotional signals."""

        corpus = f"{title} {reading.text}".lower()
        keyword_map: list[tuple[set[str], str]] = [
            ({"weary", "burden", "rest", "faint", "tired"}, "When Strength Fails"),
            ({"shepherd", "hold", "holds", "keep", "kept"}, "Still He Holds"),
            ({"near", "presence", "abide", "with", "close"}, "He Stays Near"),
            ({"refuge", "shadow", "shelter", "cover"}, "Under His Shadow"),
            ({"wait", "watch", "hope", "waiting"}, "Still Waiting Here"),
            ({"wander", "return", "stray"}, "When Hearts Wander"),
            ({"night", "dark", "dawn"}, "Before the Dawn"),
            ({"cry", "prayer", "silence", "silent"}, "Held in Silence"),
            ({"mercy", "forgive", "forgiven", "grace"}, "Mercy Finds Me"),
            ({"fear", "storm", "trouble", "afraid"}, "Not Left Alone"),
        ]

        for keywords, phrase in keyword_map:
            if any(keyword in corpus for keyword in keywords):
                logger.debug("Fallback trigger matched (%s): %s", sorted(keywords), phrase)
                return phrase

        logger.debug("Fallback default selected: He Stays Near")
        return "He Stays Near"

    def _sanitize_thumbnail_text(self, raw_text: str, title: str | None = None) -> str:
        """Normalise *raw_text* to comply with thumbnail constraints."""

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        text = (lines[0] if lines else raw_text).replace("\n", " ")
        text = re.sub(r"^['\"“”‘’`]+|['\"“”‘’`]+$", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        text = re.sub(r"[^A-Za-z\s]", " ", text)
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

        is_weak, reason = self._is_weak_browse_phrase(candidate, title=title)
        if is_weak:
            logger.debug("Rejected weak thumbnail phrase (%s): %s", reason, candidate)
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

    def _title_words(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z]+", text.lower())
            if token not in self._SMALL_WORDS and len(token) > 2
        }

    def _is_weak_browse_phrase(
        self,
        candidate: str,
        *,
        title: str | None = None,
    ) -> tuple[bool, str]:
        lower = candidate.lower()
        words = lower.split()

        if not words or len(words) > 3:
            return True, "invalid_length"

        if any(term in words for term in self._search_terms):
            return True, "search_term"
        if lower.startswith("how to") or lower.startswith("why "):
            return True, "tutorial_phrase"
        if "find" in words and any(word in self._abstract_terms for word in words):
            return True, "search_like_find"

        if all(word in self._abstract_terms for word in words) and len(words) >= 2:
            return True, "abstract_stack"

        generic_labels = {
            "trust in god",
            "gods mercy",
            "faith renewal",
            "morning hope",
            "devotional peace",
            "daily light devotional",
            "morning devotional hope",
        }
        if lower in generic_labels:
            return True, "generic_label"

        if len(words) == 1 and words[0] in self._abstract_terms:
            return True, "single_bland_abstract"

        if len(words) >= 2 and all(
            word in self._abstract_terms or word in self._generic_terms for word in words
        ):
            return True, "generic_stack"

        if not any(word in self._signal_terms for word in words) and all(
            word in self._abstract_terms or word in self._generic_terms for word in words
        ):
            return True, "low_signal"

        if title:
            candidate_words = self._title_words(candidate)
            title_words = self._title_words(title)
            if candidate_words and title_words:
                overlap = len(candidate_words & title_words) / len(candidate_words)
                if candidate_words.issubset(title_words) or overlap >= 0.67:
                    return True, "title_overlap"

        return False, ""

    def _word_priority(self, word: str) -> int:
        lower = word.lower()
        score = 0
        if lower in self._search_terms:
            score -= 4
        if lower in self._generic_terms:
            score -= 3
        if lower in self._SMALL_WORDS:
            score -= 2
        if lower in self._abstract_terms:
            score -= 1
        if lower in self._signal_terms:
            score += 3
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

__all__ = ["ThumbnailTextGenerator", "ThumbnailTextGenerationError", "SYSTEM_PROMPT_THUMBNAIL"]
