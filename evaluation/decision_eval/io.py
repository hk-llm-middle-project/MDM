"""Testset loading for decision-suite evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path, max_examples: int = 0) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Testset file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    if max_examples > 0:
        rows = rows[:max_examples]
    if not rows:
        raise RuntimeError(f"No examples were loaded from {path}")
    return rows
