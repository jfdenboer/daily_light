# generate_tags.py

from __future__ import annotations

import logging
import re
from typing import List

from openai import OpenAI, OpenAIError

from spurgeon.config.settings import Settings
from spurgeon.models import Reading
from spurgeon.utils.retry_utils import retry_with_backoff

logger = logging.getLogger(__name__)


class TagsGenerationError(Exception):
    """Fout bij het genereren van YouTube-tags."""


SYSTEM_MESSAGE = (
    "You are a YouTube SEO assistant for a Christian devotional channel. "
    "Create short, search-friendly tags that describe the content, themes, and audience of the video. "
    "Avoid quotes, hashtags, and numbering. Return the tags separated by commas."
)

FIXED_TAGS: List[str] = [
    "Daily Light on the Daily Path",
    "Daily Devotional",
    "Christian Devotional",
]


class TagsGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.prompt_model
        self.temperature = settings.prompt_temperature
        self.max_tokens = 120

    def generate(self, reading: Reading, *, max_tags: int = 12) -> List[str]:
        """Genereer een lijst van YouTube-tags voor de devotional."""

        fixed_tags = FIXED_TAGS.copy()
        max_dynamic = max(0, max_tags - len(fixed_tags))
        dynamic_target = min(5, max_dynamic)

        if dynamic_target == 0:
            return fixed_tags

        user_prompt = (
            "Generate {dynamic_target} concise YouTube tags for the following devotional. "
            "Focus on faith-based keywords, daily devotion habits, the specific passage themes, "
            "and the intended Christian audience. Use comma-separated values.\n\n{content}"
        ).format(dynamic_target=dynamic_target, content=reading.text)

        def call_openai() -> List[str]:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_prompt},
                ],
            )

            content = response.choices[0].message.content.strip()
            tags = self._parse_tags(content, target_count=dynamic_target)
            if not tags:
                raise TagsGenerationError("Ontvangen tags-lijst is leeg.")
            combined = self._merge_with_fixed_tags(tags, fixed_tags, max_tags)
            return combined

        return retry_with_backoff(
            func=call_openai,
            max_retries=3,
            backoff=1.0,
            error_types=(OpenAIError, TagsGenerationError),
            context="OpenAI tags generation",
        )

    def _parse_tags(self, raw_tags: str, *, target_count: int) -> List[str]:
        """Converteer een door AI gegenereerde tekst naar een lijst met tags."""

        # Remove bullet characters and split on commas or newlines.
        cleaned = re.sub(r"[•\-\n]+", ",", raw_tags)
        parts = re.split(r",|;", cleaned)

        tags: List[str] = []
        seen = set()
        for part in parts:
            tag = re.sub(r"^['\"“”‘’#]+|['\"“”‘’]+$", "", part).strip()
            if not tag:
                continue
            normalized = tag.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            tags.append(tag)
            if len(tags) >= target_count:
                break

        return tags

    def _merge_with_fixed_tags(
        self, dynamic_tags: List[str], fixed_tags: List[str], max_tags: int
    ) -> List[str]:
        """Combineer vaste tags met dynamische tags en verwijder duplicaten."""

        result = fixed_tags.copy()
        seen = {tag.lower() for tag in result}

        for tag in dynamic_tags:
            normalized = tag.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            result.append(tag)
            if len(result) >= max_tags:
                break

        return result


__all__ = ["TagsGenerator", "TagsGenerationError"]