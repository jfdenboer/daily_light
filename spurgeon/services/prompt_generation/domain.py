from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PromptGenerationError(Exception):
    """Raised when the prompt workflow cannot produce a valid subject line."""


class PromptIntent(str, Enum):
    IMAGE_SUBJECT = "image_subject"


@dataclass(frozen=True)
class PromptModelConfig:
    model: str
    temperature: float
    max_input_chars: int
    max_output_tokens: int
    max_retries: int
    retry_backoff: float
    seed: int | None


@dataclass(frozen=True)
class PromptContext:
    excerpt: str
    intent: PromptIntent = PromptIntent.IMAGE_SUBJECT
    reading_slug: str | None = None


@dataclass(frozen=True)
class PromptCandidate:
    text: str
    model: str
    attempt: int


@dataclass(frozen=True)
class PromptViolation:
    code: str
    message: str
    matched_text: str


@dataclass(frozen=True)
class PromptValidationResult:
    is_valid: bool
    violations: tuple[PromptViolation, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FinalPrompt:
    text: str
    candidate: PromptCandidate
    validation: PromptValidationResult
