"""OpenAI transport helpers for spoken hook pipeline."""

from __future__ import annotations

import random
import time
from typing import Final

import openai
from openai import OpenAI

TRANSPORT_MAX_ATTEMPTS: Final[int] = 3
BACKOFF_BASE_SECONDS: Final[float] = 0.75
BACKOFF_CAP_SECONDS: Final[float] = 8.0


def extract_response_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = getattr(response, "output", None)
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                if part_type in ("output_text", "text") and isinstance(text, str):
                    chunks.append(text)

        joined = " ".join(chunks).strip()
        if joined:
            return joined

    return ""


def is_retryable_transport_error(error: Exception) -> bool:
    if isinstance(
        error,
        (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        ),
    ):
        return True

    if isinstance(error, openai.APIStatusError):
        return error.status_code in {408, 409, 429} or error.status_code >= 500

    return False


def create_response_with_transport_retries(
    client: OpenAI,
    *,
    model: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    input: list[dict[str, object]],
) -> object:
    for attempt in range(TRANSPORT_MAX_ATTEMPTS):
        try:
            return client.responses.create(
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                input=input,
            )
        except Exception as error:
            if not is_retryable_transport_error(error) or attempt == TRANSPORT_MAX_ATTEMPTS - 1:
                raise

            sleep_seconds = min(BACKOFF_CAP_SECONDS, BACKOFF_BASE_SECONDS * (2**attempt))
            jitter = random.uniform(0, sleep_seconds * 0.25)
            time.sleep(sleep_seconds + jitter)

    raise RuntimeError("unreachable")


__all__ = ["extract_response_text", "create_response_with_transport_retries"]
