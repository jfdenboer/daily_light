"""Telemetry helpers for intro hook pipeline experiments."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spurgeon.services.intro.hook_pipeline.rules import HookOutcome


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_hook_event(
    path: Path,
    *,
    reading_hash: str,
    intent_card: dict[str, str],
    candidate_stats: list[dict[str, Any]],
    outcome: HookOutcome,
) -> None:
    """Append one structured hook pipeline event as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": _utc_now_iso(),
        "reading_hash": reading_hash,
        "intent_card": intent_card,
        "candidate_stats": candidate_stats,
        "outcome": asdict(outcome),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

