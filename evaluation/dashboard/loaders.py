"""Load local LangSmith evaluation result exports for dashboard views."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.dashboard.transforms import (
    build_example_frame,
    build_metric_frame,
    build_summary_frame,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResultBundle:
    """A summary JSON file and its optional sibling CSV result export."""

    summary_path: Path
    csv_path: Path | None
    summary: dict[str, Any]

    @property
    def result_stem(self) -> str:
        return self.summary_path.name.removesuffix(".summary.json")

    @property
    def run_name(self) -> str:
        value = self.summary.get("run_name")
        return str(value) if value else self.result_stem


@dataclass(frozen=True)
class DashboardResults:
    """Normalized tables used by the dashboard."""

    summary: pd.DataFrame
    examples: pd.DataFrame
    metrics: pd.DataFrame
    warnings: tuple[str, ...] = ()


def _read_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Summary file must contain a JSON object: {path}")
    return payload


def discover_result_bundles(
    result_dir: Path,
    warning_messages: list[str] | None = None,
) -> list[ResultBundle]:
    """Find local summary exports and pair each with its sibling CSV file."""

    if not result_dir.exists():
        return []

    bundles: list[ResultBundle] = []
    for summary_path in sorted(result_dir.glob("*.summary.json")):
        try:
            summary = _read_summary(summary_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            message = f"Skipping invalid summary file {summary_path.name}: {exc}"
            logger.warning(message)
            if warning_messages is not None:
                warning_messages.append(message)
            continue

        csv_path = summary_path.with_name(
            summary_path.name.removesuffix(".summary.json") + ".csv"
        )
        bundles.append(
            ResultBundle(
                summary_path=summary_path,
                csv_path=csv_path if csv_path.exists() else None,
                summary=summary,
            )
        )
    return bundles


def load_results(result_dir: Path) -> DashboardResults:
    """Load all dashboard tables from a local result directory."""

    warnings: list[str] = []
    try:
        bundles = discover_result_bundles(result_dir, warning_messages=warnings)
    except OSError as exc:
        return DashboardResults(
            summary=pd.DataFrame(),
            examples=pd.DataFrame(),
            metrics=pd.DataFrame(),
            warnings=(str(exc),),
        )

    for bundle in bundles:
        if bundle.csv_path is None:
            warnings.append(f"Missing CSV for {bundle.summary_path.name}")

    summary = build_summary_frame(bundles)
    examples = build_example_frame(bundles)
    metrics = build_metric_frame(summary)
    return DashboardResults(
        summary=summary,
        examples=examples,
        metrics=metrics,
        warnings=tuple(warnings),
    )
