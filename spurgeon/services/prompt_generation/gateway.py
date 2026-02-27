from __future__ import annotations

import logging

from openai import OpenAIError

from spurgeon.core.gpt5_client import GPT5Client
from spurgeon.utils.retry_utils import retry_with_backoff

from .domain import PromptCandidate, PromptGenerationError


logger = logging.getLogger(__name__)


class LLMGenerationGateway:
    """Provider-facing adapter for prompt generation."""

    def __init__(
        self,
        *,
        client: GPT5Client,
        model: str,
        temperature: float,
        max_output_tokens: int,
        seed: int | None,
        max_retries: int,
        retry_backoff: float,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.seed = seed
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    @staticmethod
    def _extract_choice_content(chat_response: object) -> str | None:
        choices = getattr(chat_response, "choices", None)
        if choices is None and hasattr(chat_response, "model_dump"):
            dumped = chat_response.model_dump()  # type: ignore[call-arg]
            if isinstance(dumped, dict):
                choices = dumped.get("choices")

        if not choices:
            return None

        choice = choices[0]
        message = getattr(choice, "message", None) or getattr(choice, "delta", None)
        if hasattr(choice, "model_dump"):
            dumped_choice = choice.model_dump()  # type: ignore[call-arg]
            if isinstance(dumped_choice, dict):
                message = message or dumped_choice.get("message") or dumped_choice.get("delta")

        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text_value = item.get("text") or item.get("content")
                    if isinstance(text_value, str):
                        parts.append(text_value)
            content = "\n".join(part for part in parts if part.strip())

        return content if isinstance(content, str) else None

    def _call_once(self, *, system_prompt: str, user_prompt: str, attempt: int) -> PromptCandidate:
        if self.max_output_tokens <= 0:
            raise PromptGenerationError("max_output_tokens must be greater than zero")

        request_kwargs: dict[str, object] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "model": self.model,
            "temperature": float(self.temperature),
            "max_tokens": int(self.max_output_tokens),
        }
        if self.seed is not None:
            request_kwargs["seed"] = self.seed

        chat = self.client.chat_complete(**request_kwargs)
        text = self._extract_choice_content(chat)
        if not text or not text.strip():
            raise PromptGenerationError("OpenAI returned an empty prompt (chat)")

        return PromptCandidate(text=text.strip(), model=self.model, attempt=attempt)

    def generate(self, *, system_prompt: str, user_prompt: str, attempt: int) -> PromptCandidate:
        return retry_with_backoff(
            func=lambda: self._call_once(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                attempt=attempt,
            ),
            max_retries=self.max_retries,
            backoff=self.retry_backoff,
            error_types=(OpenAIError, PromptGenerationError),
            context="OpenAI prompt generation",
        )
