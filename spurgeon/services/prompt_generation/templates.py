from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .domain import PromptGenerationError, PromptIntent


@dataclass(frozen=True)
class TemplateBundle:
    system_template: str
    user_template: str
    repair_template: str


class PromptTemplateRepository:
    """Loads prompt templates from the package directory."""

    def __init__(self, prompts_dir: Path) -> None:
        self._dir = prompts_dir

    def _load(self, name: str) -> str:
        path = self._dir / name
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise RuntimeError(f"Missing prompt template: {path}") from exc

    def resolve(self, intent: PromptIntent) -> TemplateBundle:
        if intent is PromptIntent.IMAGE_SUBJECT:
            system_template = self._load("prompt_system.txt")
            user_template = self._load("stage_subject.txt")
            return TemplateBundle(
                system_template=system_template,
                user_template=user_template,
                repair_template=(
                    "Your previous answer violated policy requirements. "
                    "Rewrite to comply fully and output only the corrected subject prompt.\n"
                    "Original candidate: {candidate}\n"
                    "Violations: {violations}\n"
                    "Excerpt: {excerpt}"
                ),
            )
        raise PromptGenerationError(f"Unsupported prompt intent: {intent}")


class PromptTemplateEngine:
    """Renders templates with strict variable validation."""

    @staticmethod
    def render(template: str, **variables: str) -> str:
        try:
            return template.format(**variables)
        except KeyError as exc:
            missing = exc.args[0] if exc.args else "<unknown>"
            raise PromptGenerationError(
                f"Template rendering failed: missing placeholder '{missing}'"
            ) from exc
