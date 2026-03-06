from __future__ import annotations

"""Generate short thumbnail copy using a two-step OpenAI pipeline."""

import logging
import re
from typing import Final

from openai import OpenAI, OpenAIError

from spurgeon.config.settings import Settings
from spurgeon.models import Reading
from spurgeon.utils.retry_utils import retry_with_backoff

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_THUMBNAIL_GENERATOR: Final[str] = r"""
You are writing thumbnail text for YouTube browse surfaces, not search.
Generate multiple candidate thumbnail phrases for one video.
Do NOT explain the video, summarize the lesson, or write a sermon heading.

Output rules:
- Output exactly one candidate per line.
- Output exactly {num_candidates} lines.
- English only.
- 1-3 words per line (prefer 2-3 words).
- Title Case.
- No punctuation, emojis, dates, verse references, author names, or numbers.
- Keep each line short enough for thumbnail use (about <= 24 characters when possible).
- No numbering, bullets, prefixes, or commentary.

Creative direction:
- Favor emotional signal, tension, ache, nearness, weakness, waiting, refuge, return, surrender, rest, mercy.
- Evoke a felt moment, not a topic label.
- Keep phrases minimal and browse-first.
- Avoid repeating the provided title.
- Avoid near-duplicates. Explore varied emotional angles.

Explicitly avoid:
- Search/tutorial phrasing: How To, Why, Guide, Tips, Steps, Best.
- Explanatory/SEO language.
- Generic devotional stacks like: Daily Light Devotional, Faith Hope Grace, Trust in God.
- Cleaned-up title fragments.

Strong examples:
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

SYSTEM_PROMPT_THUMBNAIL_SELECTOR: Final[str] = r"""
You are selecting one thumbnail phrase for YouTube browse packaging.
Choose the strongest candidate from the provided list.
Do NOT explain the video.
Do NOT write a new phrase unless every candidate is unusable.

Selection criteria:
- Strong emotional signal and resonance.
- Curiosity/tension for browse CTR packaging.
- Brief and readable for a thumbnail (1-3 words ideal).
- Distinctive and non-generic.
- Complements the title without repeating it.

Reject candidates that are:
- Tutorial/search style.
- Generic devotional abstractions.
- Sermon-heading or category labels.
- Overlapping too much with the title.

Output rules:
- Output exactly one line only: the winning phrase.
- No explanations, labels, numbering, or extra text.
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
        self.generator_model = settings.thumbnail_text_generator_model
        self.selector_model = settings.thumbnail_text_selector_model
        self.generator_temperature = settings.thumbnail_text_generator_temperature
        self.selector_temperature = settings.thumbnail_text_selector_temperature
        self.num_candidates = settings.thumbnail_text_num_candidates
        self.generator_max_tokens = 120
        self.selector_max_tokens = 16

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
            "left",
            "mercy",
            "strength",
        }

    def generate(self, reading: Reading, title: str | None = None) -> str:
        """Return one final thumbnail phrase using generate-then-select."""

        def call_openai() -> str:
            candidates = self._generate_candidates(reading, title=title)
            winner = self._select_candidate(reading, candidates, title=title)
            final_text = self._sanitize_thumbnail_text(winner, title=title)
            if not final_text:
                raise ThumbnailTextGenerationError(
                    "Selected thumbnail text sanitized to empty or weak output."
                )
            logger.debug("Final thumbnail phrase: %r", final_text)
            return final_text

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
            return self.fallback(reading, title or "")

    def _generate_candidates(self, reading: Reading, title: str | None = None) -> list[str]:
        user_sections = []
        if title:
            user_sections.append(f"Working Video Title:\n{title.strip()}")
        user_sections.append(f"Devotional Text:\n{reading.text}")

        response = self.client.chat.completions.create(
            model=self.generator_model,
            temperature=self.generator_temperature,
            max_tokens=self.generator_max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_THUMBNAIL_GENERATOR.format(
                        num_candidates=self.num_candidates
                    ),
                },
                {"role": "user", "content": "\n\n".join(user_sections)},
            ],
        )

        raw_output = response.choices[0].message.content or ""
        logger.debug("Raw thumbnail generator output: %r", raw_output)
        candidates = self._parse_candidates(raw_output, title=title)
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
        *,
        title: str | None = None,
    ) -> str:
        viable_candidates = [c for c in candidates if not self._is_weak_browse_phrase(c, title=title)[0]]
        logger.debug(
            "Candidate count before/after quality prefilter: %d -> %d",
            len(candidates),
            len(viable_candidates),
        )
        if not viable_candidates:
            raise ThumbnailTextGenerationError("All generated candidates were rejected as weak/generic.")

        user_sections = []
        if title:
            user_sections.append(f"Working Video Title:\n{title.strip()}")
        user_sections.append(f"Devotional Text:\n{reading.text}")
        user_sections.append("Candidates:\n" + "\n".join(viable_candidates))

        response = self.client.chat.completions.create(
            model=self.selector_model,
            temperature=self.selector_temperature,
            max_tokens=self.selector_max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_THUMBNAIL_SELECTOR},
                {"role": "user", "content": "\n\n".join(user_sections)},
            ],
        )

        raw_winner = response.choices[0].message.content or ""
        logger.debug("Raw thumbnail selector output: %r", raw_winner)
        winner = self._extract_selector_winner(raw_winner)
        if not winner:
            raise ThumbnailTextGenerationError("Selector output could not be parsed into a winner.")

        winner = self._sanitize_thumbnail_text(winner, title=title)
        if winner and winner in viable_candidates:
            logger.debug("Selector winner accepted: %s", winner)
            return winner

        if winner and not self._is_weak_browse_phrase(winner, title=title)[0]:
            logger.debug("Selector winner accepted after sanitization normalization: %s", winner)
            return winner

        for candidate in viable_candidates:
            normalized = self._sanitize_thumbnail_text(candidate, title=title)
            if normalized:
                logger.debug("Selector winner rejected; fallback to best viable candidate: %s", normalized)
                return normalized

        raise ThumbnailTextGenerationError("Selector winner invalid and no viable candidates remained.")

    def _extract_selector_winner(self, raw_text: str) -> str:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            return ""
        first_line = lines[0]
        first_line = re.sub(r"^(?:winner\s*:\s*)", "", first_line, flags=re.IGNORECASE)
        return first_line.strip()

    def _parse_candidates(self, raw_text: str, title: str | None = None) -> list[str]:
        parsed: list[str] = []
        seen: set[str] = set()
        for line in raw_text.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            cleaned = re.sub(r"^[-*•]+\s*", "", cleaned)
            cleaned = re.sub(r"^\d+[\.)]\s*", "", cleaned)
            sanitized = self._sanitize_thumbnail_text(cleaned, title=title)
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


__all__ = [
    "ThumbnailTextGenerator",
    "ThumbnailTextGenerationError",
    "SYSTEM_PROMPT_THUMBNAIL_GENERATOR",
    "SYSTEM_PROMPT_THUMBNAIL_SELECTOR",
]
