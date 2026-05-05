"""Local evaluation runner that mirrors LangSmith result columns."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from evaluation.retrieval_eval.constants import EXAMPLE_METADATA_FIELDS
from evaluation.retrieval_eval.dataset import build_examples


def _csv_cell(value: Any) -> Any:
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False)
    return value


def _flatten_record(prefix: str, values: dict[str, Any], record: dict[str, Any]) -> None:
    for key, value in values.items():
        record[f"{prefix}.{key}"] = _csv_cell(value)


def _evaluate_local_example(
    example: dict[str, Any],
    target,
    evaluators: list,
) -> dict[str, Any]:
    inputs = example["inputs"]
    reference_outputs = example["outputs"]
    record: dict[str, Any] = {}
    started_at = time.perf_counter()

    try:
        outputs = target(inputs)
        error = None
    except Exception as exc:  # noqa: BLE001 - record errors as eval rows.
        outputs = {"error": str(exc), "error_type": type(exc).__name__}
        error = str(exc)

    record["error"] = error
    record["execution_time"] = time.perf_counter() - started_at
    metadata = example.get("metadata", {})
    record["example_id"] = metadata.get("id")
    for field in EXAMPLE_METADATA_FIELDS:
        if metadata.get(field) is not None:
            record[field] = _csv_cell(metadata.get(field))
    _flatten_record("metadata", metadata, record)
    _flatten_record("inputs", inputs, record)
    _flatten_record("outputs", outputs, record)
    _flatten_record("reference", reference_outputs, record)

    for evaluator in evaluators:
        try:
            feedback = evaluator(outputs, reference_outputs)
        except Exception as exc:  # noqa: BLE001 - evaluator failures are scores too.
            feedback = {
                "key": getattr(evaluator, "__name__", "evaluator_error"),
                "score": 0,
                "comment": f"{type(exc).__name__}: {exc}",
            }
        key = feedback["key"]
        record[f"feedback.{key}"] = feedback.get("score")
        if feedback.get("comment") is not None:
            record[f"feedback.{key}.comment"] = feedback.get("comment")

    return record


def evaluate_local_rows(
    rows: list[dict[str, Any]],
    target,
    evaluators: Iterable,
    max_concurrency: int = 1,
) -> pd.DataFrame:
    """Run the retrieval target and evaluators locally without LangSmith traces."""

    examples = build_examples(rows)
    evaluator_list = list(evaluators)
    workers = max(1, int(max_concurrency or 1))
    if workers == 1 or len(examples) <= 1:
        records = [
            _evaluate_local_example(example, target, evaluator_list)
            for example in examples
        ]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(examples))) as executor:
            records = list(
                executor.map(
                    lambda example: _evaluate_local_example(
                        example,
                        target,
                        evaluator_list,
                    ),
                    examples,
                )
            )
    return pd.DataFrame.from_records(records)
