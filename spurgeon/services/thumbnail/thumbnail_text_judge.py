from __future__ import annotations

"""Judge/filter stage for thumbnail text candidate quality."""

import logging
import re
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

from spurgeon.config.settings import Settings
from spurgeon.models import Reading

logger = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=1)
def load_thumbnail_judge_prompt() -> str:
    prompt_path = PROMPTS_DIR / "thumbnail_text_judge.v1.txt"
    return prompt_path.read_text(encoding="utf-8").strip()


class ThumbnailTextJudge:
    """Filter raw candidates down to strong, selector-ready options."""

    def __init__(self, settings: Settings, client: OpenAI) -> None:
        self.model = settings.thumbnail_text_judge_model
        self.temperature = settings.thumbnail_text_judge_temperature
        self.max_tokens = 180
        self.client = client

    def judge_thumbnail_text_candidates(self, reading: Reading, candidates: list[str]) -> list[str]:
        if not candidates:
            return []

        user_sections = [
            f"Thumbnail Intent Card:\n{reading.text}",
            "Candidates:\n" + "\n".join(candidates),
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": load_thumbnail_judge_prompt()},
                {"role": "user", "content": "\n\n".join(user_sections)},
            ],
        )

        raw_output = response.choices[0].message.content or ""
        logger.debug("Raw thumbnail judge output: %r", raw_output)
        return self.parse_surviving_candidates(raw_output, candidates)

    def parse_surviving_candidates(self, raw_output: str, candidates: list[str]) -> list[str]:
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if not lines:
            return []
        if len(lines) == 1 and lines[0].upper() == "NONE":
            return []

        normalized_map: dict[str, str] = {}
        for candidate in candidates:
            normalized_map[self._normalize(candidate)] = candidate

        kept: list[str] = []
        seen: set[str] = set()
        for line in lines:
            cleaned = re.sub(r"^[-*•]+\s*", "", line)
            cleaned = re.sub(r"^\d+[\.)]\s*", "", cleaned)
            cleaned = cleaned.strip().strip('"\'')
            if not cleaned:
                continue

            key = self._normalize(cleaned)
            matched = normalized_map.get(key)
            if not matched:
                logger.warning("Judge output contained unknown candidate line: %r", line)
                return []
            if key in seen:
                continue
            seen.add(key)
            kept.append(matched)
        return kept

    def _normalize(self, text: str) -> str:
        squashed = re.sub(r"\s+", " ", text).strip().lower()
        return re.sub(r"[^a-z0-9']", "", squashed)


__all__ = ["ThumbnailTextJudge", "load_thumbnail_judge_prompt"]
