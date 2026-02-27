# generate_title.py

from __future__ import annotations

import re
import logging
from openai import OpenAI, OpenAIError

from spurgeon.models import Reading
from spurgeon.config.settings import Settings
from spurgeon.utils.retry_utils import retry_with_backoff

logger = logging.getLogger(__name__)


class TitleGenerationError(Exception):
    """Fout bij het genereren van een titel via OpenAI."""

SYSTEM_MESSAGE = (
    "You are a YouTube content expert. Generate a short, compelling key phrase "
    "(maximum 4 words) that captures the essence of the provided Christian devotional. "
    "Return only the key phrase — no explanations. Do not use quotation marks, "
    "Bible references, emojis, or excessive punctuation. Focus on clarity, "
    "engagement, and spiritual depth."
)


class TitleGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.prompt_model
        self.temperature = settings.prompt_temperature

    def generate(self, reading: Reading) -> str:
        """Genereer een YouTube-titel op basis van de devotional-tekst."""

        user_content = f"Provide only the key phrase for this devotional text:\n\n{reading.text}"

        def call_openai() -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=60,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_content},
                ],
            )

            content = response.choices[0].message.content.strip()
            key_phrase = self._sanitize_key_phrase(content)
            if not key_phrase:
                raise TitleGenerationError("Ontvangen key phrase is leeg.")
            return self._format_title(reading, key_phrase)

        return retry_with_backoff(
            func=call_openai,
            max_retries=3,
            backoff=1.0,
            error_types=(OpenAIError, TitleGenerationError),
            context="OpenAI title generation",
        )

    def _sanitize_key_phrase(self, key_phrase: str) -> str:
        """Schoon de gegenereerde key phrase op."""
        key_phrase = re.sub(r"^['\"“”‘’]+|['\"“”‘’]+$", "", key_phrase)
        key_phrase = re.sub(r"\s+", " ", key_phrase).strip()
        key_phrase = key_phrase.rstrip("!?.:;,-")
        return key_phrase

    def _format_title(self, reading: Reading, key_phrase: str) -> str:
        """Construeer de definitieve YouTube-titel in het gewenste format."""

        formatted_date = f"{reading.date.strftime('%B')} {reading.date.day}, {reading.date.year}"
        suffix = (
            " | Daily Light on the Daily Path "
            f"({reading.reading_type.value}, {formatted_date})"
        )

        max_total_length = 100
        max_key_phrase_length = max_total_length - len(suffix)
        if max_key_phrase_length <= 0:
            raise TitleGenerationError("Suffix for YouTube title exceeds maximum length")

        truncated_key_phrase = self._truncate_key_phrase(key_phrase, max_key_phrase_length)
        if not truncated_key_phrase:
            raise TitleGenerationError("Key phrase truncated to empty value")

        return f"{truncated_key_phrase}{suffix}"

    def _truncate_key_phrase(self, key_phrase: str, max_length: int) -> str:
        """Beperk de key phrase tot een veilige lengte voor YouTube-titels."""

        if len(key_phrase) <= max_length:
            return key_phrase

        truncated = key_phrase[:max_length].rstrip()
        if " " in truncated:
            truncated = truncated[: truncated.rfind(" ")]
        truncated = truncated.rstrip("-–—,:; ")

        if not truncated:
            truncated = key_phrase[:max_length].rstrip()

        return truncated

__all__ = ["TitleGenerator", "TitleGenerationError"]
