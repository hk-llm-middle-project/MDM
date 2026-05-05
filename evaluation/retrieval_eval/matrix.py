"""Matrix preset loading for retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_eval_matrix(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {path}")
    matrix = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(matrix, dict):
        raise ValueError("Matrix file must contain a JSON object.")
    if not isinstance(matrix.get("presets"), dict):
        raise ValueError("Matrix file must contain a presets object.")
    if not isinstance(matrix.get("runs"), list):
        raise ValueError("Matrix file must contain a runs list.")
    return matrix


def resolve_matrix_runs(matrix: dict[str, Any], preset: str | None) -> list[dict[str, str]]:
    preset_name = preset or str(matrix.get("default_preset") or "upstage")
    presets = matrix["presets"]
    if preset_name not in presets:
        available = ", ".join(sorted(presets))
        raise ValueError(f"Unknown matrix preset: {preset_name}. Available presets: {available}")

    runs_by_name = {
        str(run.get("name")): run
        for run in matrix["runs"]
        if isinstance(run, dict) and run.get("name")
    }
    resolved: list[dict[str, str]] = []
    for run_name in presets[preset_name]:
        if run_name not in runs_by_name:
            raise ValueError(f"Preset {preset_name} references unknown run: {run_name}")
        run = runs_by_name[run_name]
        for field in ["name", "loader_strategy", "chunker_strategy", "embedding_provider"]:
            if not isinstance(run.get(field), str) or not run[field]:
                raise ValueError(f"Matrix run {run_name} is missing {field}")
        resolved.append(
            {
                "name": run["name"],
                "loader_strategy": run["loader_strategy"],
                "chunker_strategy": run["chunker_strategy"],
                "embedding_provider": run["embedding_provider"],
            }
        )
    return resolved


def print_matrix(matrix: dict[str, Any]) -> None:
    print("Presets:")
    for preset, run_names in sorted(matrix["presets"].items()):
        print(f"  {preset}: {', '.join(run_names)}")
    print("Runs:")
    for run in matrix["runs"]:
        print(
            "  "
            f"{run['name']}: "
            f"{run['loader_strategy']}/{run['chunker_strategy']}/{run['embedding_provider']}"
        )
