from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .domain import (
    FinalPrompt,
    PromptContext,
    PromptGenerationError,
)
from .gateway import LLMGenerationGateway
from .policy import PromptPolicyEngine
from .templates import TemplateBundle


logger = logging.getLogger(__name__)


def build_excerpt_user_prompt(excerpt: str) -> str:
    return f'EXCERPT:\n"""\n{excerpt}\n"""'


class PromptRequestNormalizer:
    @staticmethod
    def normalize_excerpt(text: str, *, max_input_chars: int) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned) > max_input_chars:
            truncated = cleaned[:max_input_chars]
            if " " in truncated:
                truncated = truncated[: truncated.rfind(" ")]
            cleaned = truncated

        if not cleaned:
            raise PromptGenerationError("Cannot create a prompt from empty input")

        return cleaned


class PromptPostProcessor:
    @staticmethod
    def clean_candidate_text(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text)
        return cleaned.strip(" .,;:-")


@dataclass
class PromptGenerationOrchestrator:
    template_bundle: TemplateBundle
    policy_engine: PromptPolicyEngine
    gateway: LLMGenerationGateway
    max_attempts: int = 2

    def generate(self, context: PromptContext) -> FinalPrompt:
        user_prompt = build_excerpt_user_prompt(context.excerpt)

        for attempt in range(1, self.max_attempts + 1):
            candidate = self.gateway.generate(
                system_prompt=self.template_bundle.system_template,
                user_prompt=user_prompt,
                attempt=attempt,
            )
            cleaned = PromptPostProcessor.clean_candidate_text(candidate.text)
            if not cleaned:
                logger.warning("Empty candidate after cleaning at attempt %d", attempt)
                continue

            validation = self.policy_engine.validate(cleaned)
            if validation.is_valid:
                return FinalPrompt(
                    text=cleaned,
                    candidate=candidate,
                    validation=validation,
                )

            if attempt < self.max_attempts:
                violations = ", ".join(v.matched_text for v in validation.violations)
                user_prompt = self.template_bundle.repair_template.format(
                    candidate=cleaned,
                    violations=violations,
                    excerpt=context.excerpt,
                )
                logger.info(
                    "Prompt candidate violated policy at attempt %d; retrying with repair prompt",
                    attempt,
                )
                continue

            raise PromptGenerationError(
                "Model output violated guardrails: "
                + ", ".join(f"'{v.matched_text}'" for v in validation.violations)
            )

        raise PromptGenerationError("Prompt generation failed after retries")
