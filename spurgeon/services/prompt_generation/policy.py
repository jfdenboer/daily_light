from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .domain import PromptValidationResult, PromptViolation


@dataclass(frozen=True)
class PromptPolicy:
    forbidden_patterns: tuple[str, ...]
    forbidden_style_terms: tuple[str, ...]


class PromptPolicyRepository:
    def __init__(self, prompts_dir: Path) -> None:
        self._dir = prompts_dir

    @staticmethod
    def _load_pattern_file(path: Path, *, error_subject: str) -> tuple[str, ...]:
        try:
            raw_patterns = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError as exc:
            raise RuntimeError(f"Missing {error_subject} file: {path}") from exc

        patterns: list[str] = []
        for line in raw_patterns:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                patterns.append(stripped)

        if not patterns:
            raise RuntimeError(f"{error_subject.capitalize()} file is empty: {path}")

        return tuple(patterns)

    def load(self) -> PromptPolicy:
        banned = self._load_pattern_file(
            self._dir / "banned_patterns.txt", error_subject="banned patterns"
        )
        style = self._load_pattern_file(
            self._dir / "forbidden_style_terms.txt", error_subject="forbidden style terms"
        )
        return PromptPolicy(forbidden_patterns=banned, forbidden_style_terms=style)


class PromptPolicyEngine:
    def __init__(self, policy: PromptPolicy) -> None:
        self._policy = policy
        self._banned_re = re.compile(
            "(" + "|".join(self._policy.forbidden_patterns) + ")", flags=re.IGNORECASE
        )
        self._style_re = re.compile(
            "(" + "|".join(self._policy.forbidden_style_terms) + ")", flags=re.IGNORECASE
        )

    def validate(self, text: str) -> PromptValidationResult:
        violations: list[PromptViolation] = []

        banned_match = self._banned_re.search(text)
        if banned_match:
            violations.append(
                PromptViolation(
                    code="FORBIDDEN_PATTERN",
                    message="Prompt contains a banned pattern",
                    matched_text=banned_match.group(0),
                )
            )

        style_match = self._style_re.search(text)
        if style_match:
            violations.append(
                PromptViolation(
                    code="FORBIDDEN_STYLE_TERM",
                    message="Prompt contains a forbidden style term",
                    matched_text=style_match.group(0),
                )
            )

        return PromptValidationResult(is_valid=not violations, violations=tuple(violations))
