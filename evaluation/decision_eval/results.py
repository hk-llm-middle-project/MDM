"""CSV and summary writers for decision-suite evaluation."""

from __future__ import annotations

from datetime import datetime
import json
import re
from pathlib import Path

import pandas as pd


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣._-]+", "-", value).strip("-")
    return cleaned or "decision-eval"


def summarize_feedback_metrics(dataframe: pd.DataFrame) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for column in dataframe.columns:
        if not str(column).startswith("feedback.") or str(column).endswith(".comment"):
            continue
        metric_name = str(column).removeprefix("feedback.")
        numeric = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        metrics[metric_name] = float(numeric.mean()) if len(numeric) else None
    return metrics


def save_suite_dataframe(
    dataframe: pd.DataFrame,
    suite: str,
    output_dir: Path,
    testset_path: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = safe_filename(f"{timestamp}-{suite}-local")
    csv_path = output_dir / f"{base_name}.csv"
    summary_path = output_dir / f"{base_name}.summary.json"
    dataframe.to_csv(csv_path, index=False)
    summary = {
        "experiment_name": f"MDM {suite} eval - local",
        "dataset_name": f"MDM {suite} testset",
        "testset_path": str(testset_path),
        "evaluation_suite": suite,
        "run_name": f"{suite}-local",
        "loader_strategy": None,
        "chunker_strategy": None,
        "embedding_provider": None,
        "retriever_strategy": None,
        "reranker_strategy": None,
        "row_count": int(len(dataframe)),
        "metrics": summarize_feedback_metrics(dataframe),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"csv": csv_path, "summary_json": summary_path}
