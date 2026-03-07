"""Lightweight wrapper around :class:`openai.OpenAI` for Responses API usage."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from openai import OpenAI


class GPT5Client:
    """Provide convenience helpers for making Responses API calls.

    The class keeps the underlying :class:`~openai.OpenAI` client private while
    offering thin wrappers that accept either fully constructed "input" payloads
    or conversational "messages".  It also exposes :meth:`extract_text` to
    normalise textual output returned from the Responses API, which is reused by
    multiple services.
    """

    def __init__(
        self,
        *,
        api_key: str,
        default_model: str = "gpt-5",
        max_retries: int | None = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key, max_retries=max_retries)
        self.default_model = default_model

    def responses_create(
        self,
        *,
        input: Sequence[Mapping[str, Any]],
        model: str | None = None,
        **kwargs: Any,
    ):
        """Create a response using a pre-built ``input`` payload."""

        payload = dict(kwargs)
        payload["model"] = model or self.default_model
        payload["input"] = input
        return self._client.responses.create(**payload)

    def chat_complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ):
        """Create a chat completion using ``messages`` payloads."""

        payload: dict[str, Any] = {"model": model or self.default_model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_completion_tokens"] = int(max_tokens)
        payload.update(kwargs)
        return self._client.chat.completions.create(**payload)

    def responses_from_messages(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str | None = None,
        **kwargs: Any,
    ):
        """Create a response from legacy chat-style message dictionaries."""

        input_payload = [
            {
                "role": message["role"],
                "content": [
                    {
                        "type": "input_text",
                        "text": message["content"],
                    }
                ],
            }
            for message in messages
        ]

        return self.responses_create(input=input_payload, model=model, **kwargs)

    @staticmethod
    def extract_text(response: Any) -> str:
        """Coerce textual output from a Responses API payload."""

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output = getattr(response, "output", None) or []
        fragments: list[str] = []

        def _as_mapping(value: Any) -> Mapping[str, Any] | None:
            if isinstance(value, Mapping):
                return value
            if hasattr(value, "model_dump"):
                dumped = value.model_dump()  # type: ignore[call-arg]
                if isinstance(dumped, Mapping):
                    return dumped
            return None

        def _coerce_text(value: Any) -> str | None:
            if not value:
                return None
            if isinstance(value, str) and value.strip():
                return value
            mapping = _as_mapping(value)
            if not mapping:
                return None
            nested = mapping.get("value") or mapping.get("text") or mapping.get("content")
            if isinstance(nested, str) and nested.strip():
                return nested
            if isinstance(nested, Mapping):
                inner_value = (
                    nested.get("value")
                    or nested.get("text")
                    or nested.get("content")
                )
                if isinstance(inner_value, str) and inner_value.strip():
                    return inner_value
            return None

        for item in output:
            content_list: Iterable[Any] | None = getattr(item, "content", None)
            if content_list is None:
                mapping = _as_mapping(item)
                if mapping:
                    content_list = mapping.get("content")  # type: ignore[assignment]
            if not isinstance(content_list, Iterable):
                continue
            for content in content_list:
                content_type = getattr(content, "type", None)
                if content_type not in {None, "output_text", "text"}:
                    continue
                text_obj = getattr(content, "text", None)
                if text_obj is None:
                    mapping = _as_mapping(content)
                    if mapping:
                        text_obj = mapping.get("text")
                text = _coerce_text(text_obj)
                if text:
                    fragments.append(text)

        if fragments:
            return "\n".join(part.strip() for part in fragments if part.strip())

        return ""


__all__ = ["GPT5Client"]
