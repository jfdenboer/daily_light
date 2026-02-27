from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from spurgeon.config.settings import Settings
from spurgeon.core.gpt5_client import GPT5Client
from spurgeon.models import RawAsset, Reading

from .domain import (
    PromptContext,
    PromptGenerationError,
    PromptIntent,
    PromptModelConfig,
)
from .gateway import LLMGenerationGateway
from .pipeline import PromptGenerationOrchestrator, PromptRequestNormalizer
from .policy import PromptPolicy, PromptPolicyEngine, PromptPolicyRepository
from .templates import PromptTemplateEngine, PromptTemplateRepository, TemplateBundle

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent
_DEFAULT_STANDALONE_INPUT = _PROMPTS_DIR / "test_input.txt"
_DEFAULT_ENV_FILE = Path(r"C:\Users\jfden\spurgeon\.env")


def _load_template_bundle(prompts_dir: Path, intent: PromptIntent) -> TemplateBundle:
    return PromptTemplateRepository(prompts_dir).resolve(intent)


def _load_policy(prompts_dir: Path) -> PromptPolicy:
    return PromptPolicyRepository(prompts_dir).load()


def _build_gateway(
    *,
    client: GPT5Client,
    model: str,
    temperature: float,
    max_output_tokens: int,
    seed: int | None,
    max_retries: int,
    retry_backoff: float,
) -> LLMGenerationGateway:
    return LLMGenerationGateway(
        client=client,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        seed=seed,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )


def _build_prompt_generation_orchestrator(
    *,
    client: GPT5Client,
    config: PromptModelConfig,
    prompts_dir: Path,
    intent: PromptIntent,
    max_attempts: int,
) -> PromptGenerationOrchestrator:
    template_bundle = _load_template_bundle(prompts_dir, intent)
    policy_engine = PromptPolicyEngine(_load_policy(prompts_dir))
    gateway = _build_gateway(
        client=client,
        model=config.model,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        seed=config.seed,
        max_retries=config.max_retries,
        retry_backoff=config.retry_backoff,
    )

    return PromptGenerationOrchestrator(
        template_bundle=template_bundle,
        template_engine=PromptTemplateEngine(),
        policy_engine=policy_engine,
        gateway=gateway,
        max_attempts=max_attempts,
    )


def stage_prepare_excerpt(text: str, *, max_input_chars: int) -> str:
    excerpt = PromptRequestNormalizer.normalize_excerpt(text, max_input_chars=max_input_chars)
    logger.info(
        "Excerpt prepared (original_len=%d, truncated_len=%d)", len(text), len(excerpt)
    )
    logger.debug("Excerpt content:\n%s", excerpt)
    return excerpt


def stage_generate_subject(
    excerpt: str,
    *,
    client: GPT5Client,
    model: str,
    temperature: float,
    max_output_tokens: int,
    seed: int | None,
) -> str:
    template_bundle = _load_template_bundle(_PROMPTS_DIR, PromptIntent.IMAGE_SUBJECT)
    gateway = _build_gateway(
        client=client,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        seed=seed,
        max_retries=0,
        retry_backoff=0.01,
    )
    user_prompt = PromptTemplateEngine.render(template_bundle.user_template, excerpt=excerpt)
    candidate = gateway.generate(
        system_prompt=template_bundle.system_template,
        user_prompt=user_prompt,
        attempt=1,
    )
    logger.info("Subject draft generated (length=%d)", len(candidate.text))
    logger.debug("Draft content: %s", candidate.text)
    return candidate.text


def stage_finalize_subject(draft: str) -> str:
    cleaned = draft.strip()
    if not cleaned:
        raise PromptGenerationError("Model returned an empty subject prompt")

    validation = PromptPolicyEngine(_load_policy(_PROMPTS_DIR)).validate(cleaned)
    if not validation.is_valid:
        raise PromptGenerationError(
            "Model output violated guardrails: "
            + ", ".join(f"'{v.matched_text}'" for v in validation.violations)
        )

    return cleaned


class PromptOrchestrator:
    """Coordinates the prompt pipeline for building subject prompts."""

    def __init__(self, settings: Settings, *, client: GPT5Client | None = None) -> None:
        self.settings = settings
        self.client = client or GPT5Client(
            api_key=settings.openai_api_key,
            default_model=getattr(settings, "prompt_model", "gpt-5"),
            max_retries=settings.image_max_retries,
        )

        configured_model = (
            getattr(settings, "prompt_generator_model", None) or "gpt-5-chat-latest"
        )
        if configured_model != "gpt-5-chat-latest":
            logger.warning(
                "Prompt orchestrator configured with model '%s'; overriding to 'gpt-5-chat-latest' per policy",
                configured_model,
            )
            configured_model = "gpt-5-chat-latest"

        self.config = PromptModelConfig(
            model=configured_model,
            temperature=settings.prompt_generator_temperature,
            max_input_chars=settings.image_max_input_chars,
            max_output_tokens=settings.prompt_subject_tokens,
            max_retries=settings.image_max_retries,
            retry_backoff=settings.image_retry_backoff,
            seed=getattr(settings, "prompt_seed", None),
        )

        self._images_dir = Path(settings.output_dir) / "images"
        self._image_extension = settings.image_file_extension

        self._generator = _build_prompt_generation_orchestrator(
            client=self.client,
            config=self.config,
            prompts_dir=_PROMPTS_DIR,
            intent=PromptIntent.IMAGE_SUBJECT,
            max_attempts=2,
        )

        logger.info("PromptOrchestrator initialised (model=%s)", self.config.model)

    def build_prompt(self, text: str) -> str:
        excerpt = stage_prepare_excerpt(text, max_input_chars=self.config.max_input_chars)
        final_prompt = self._generator.generate(PromptContext(excerpt=excerpt))
        logger.info("Prompt build completed (result_len=%d)", len(final_prompt.text))
        logger.debug("Prompt build output: %s", final_prompt.text)
        return final_prompt.text

    def build_prompt_for_reading(self, reading: Reading) -> str:
        """Generate one image subject prompt for the full reading text."""

        reading_text = reading.text.strip()
        if not reading_text:
            raise PromptGenerationError("Reading bevat geen tekst voor promptgeneratie")

        return self.build_prompt(reading_text)

    def build_single_image_asset(self, reading: Reading, *, duration: float) -> RawAsset:
        """Create a single image asset tuple for the entire reading."""

        if duration <= 0:
            raise PromptGenerationError("Single image duration must be positive")

        self._images_dir.mkdir(parents=True, exist_ok=True)
        prompt = self.build_prompt_for_reading(reading)
        image_name = f"{reading.slug}.{self._image_extension}"
        image_path = self._images_dir / image_name
        return (image_path, duration, prompt)


PromptGenerator = PromptOrchestrator


def main(argv: List[str] | None = None) -> int:
    """Backward compatible proxy for stand-alone CLI execution."""

    from .cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PromptGenerationError",
    "PromptModelConfig",
    "PromptOrchestrator",
    "PromptGenerator",
    "main",
    "stage_prepare_excerpt",
    "stage_generate_subject",
    "stage_finalize_subject",
]
