"""Normalize local evaluation exports into dashboard-friendly tables."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import hashlib

import pandas as pd


METRIC_COLUMNS = (
    "diagram_id_hit",
    "location_match",
    "party_type_match",
    "chunk_type_match",
    "keyword_coverage",
    "retrieval_relevance",
    "critical_error",
)

SUMMARY_METADATA_COLUMNS = (
    "experiment_name",
    "dataset_name",
    "testset_path",
    "run_name",
    "loader_strategy",
    "chunker_strategy",
    "embedding_provider",
    "retriever_strategy",
    "reranker_strategy",
    "row_count",
    "summary_path",
    "csv_path",
    "result_stem",
    "combo",
    "run_label",
)


def make_combo(loader: Any, chunker: Any, embedder: Any) -> str:
    return f"{loader or '-'} / {chunker or '-'} / {embedder or '-'}"


def make_run_label(run_name: Any, retriever: Any, reranker: Any) -> str:
    name = str(run_name or "-")
    retriever_name = str(retriever or "")
    reranker_name = str(reranker or "")
    if not retriever_name and not reranker_name:
        return name
    return f"{name} / {retriever_name or '-'} / {reranker_name or '-'}"


def _bundle_value(bundle: Any, key: str, default: Any = None) -> Any:
    return getattr(bundle, "summary", {}).get(key, default)


def build_summary_frame(bundles: Iterable[Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for bundle in bundles:
        summary = getattr(bundle, "summary", {})
        metrics = summary.get("metrics") or {}
        loader = summary.get("loader_strategy")
        chunker = summary.get("chunker_strategy")
        embedder = summary.get("embedding_provider")
        retriever = summary.get("retriever_strategy")
        reranker = summary.get("reranker_strategy")
        run_name = summary.get("run_name") or getattr(bundle, "run_name", None)
        record: dict[str, Any] = {
            "experiment_name": summary.get("experiment_name"),
            "dataset_name": summary.get("dataset_name"),
            "testset_path": summary.get("testset_path"),
            "run_name": run_name,
            "loader_strategy": loader,
            "chunker_strategy": chunker,
            "embedding_provider": embedder,
            "retriever_strategy": retriever,
            "reranker_strategy": reranker,
            "row_count": summary.get("row_count"),
            "summary_path": str(getattr(bundle, "summary_path", "")),
            "csv_path": str(getattr(bundle, "csv_path", "") or ""),
            "result_stem": getattr(bundle, "result_stem", None),
            "combo": make_combo(loader, chunker, embedder),
            "run_label": make_run_label(run_name, retriever, reranker),
        }
        for metric_name in METRIC_COLUMNS:
            record[metric_name] = metrics.get(metric_name)
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return pd.DataFrame(columns=list(SUMMARY_METADATA_COLUMNS) + list(METRIC_COLUMNS))
    for metric_name in METRIC_COLUMNS:
        frame[metric_name] = pd.to_numeric(frame[metric_name], errors="coerce")
    return frame


def _read_bundle_csv(bundle: Any) -> pd.DataFrame | None:
    csv_path = getattr(bundle, "csv_path", None)
    if csv_path is None:
        return None
    try:
        return pd.read_csv(csv_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def _normalized_question(value: Any) -> str:
    return str(value or "").strip()


def _question_case_key(question: str) -> str:
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
    return f"question:{digest}"


def make_case_key(row: pd.Series) -> str:
    """Return a stable dashboard key for one evaluated test case."""

    example_id = row.get("example_id")
    if pd.notna(example_id) and str(example_id).strip():
        return str(example_id).strip()
    return _question_case_key(_normalized_question(row.get("inputs.question", "")))


def reconcile_cross_run_case_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse export-specific IDs when one question appears once per run."""

    if frame.empty or not {"case_key", "inputs.question"}.issubset(frame.columns):
        return frame

    reconciled = frame.copy()
    normalized_question = reconciled["inputs.question"].map(_normalized_question)
    for question, group in reconciled[normalized_question != ""].groupby(
        normalized_question[normalized_question != ""], dropna=False
    ):
        if group["case_key"].nunique(dropna=False) <= 1:
            continue
        if "run_name" in group.columns and group.groupby("run_name", dropna=False).size().max() > 1:
            continue
        reconciled.loc[group.index, "case_key"] = _question_case_key(str(question))
    return reconciled


def build_example_frame(bundles: Iterable[Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for bundle in bundles:
        frame = _read_bundle_csv(bundle)
        if frame is None:
            continue

        summary = getattr(bundle, "summary", {})
        loader = summary.get("loader_strategy")
        chunker = summary.get("chunker_strategy")
        embedder = summary.get("embedding_provider")
        retriever = summary.get("retriever_strategy")
        reranker = summary.get("reranker_strategy")
        run_name = summary.get("run_name") or getattr(bundle, "run_name", None)
        frame = frame.copy()
        frame["run_name"] = run_name
        frame["loader_strategy"] = loader
        frame["chunker_strategy"] = chunker
        frame["embedding_provider"] = embedder
        frame["retriever_strategy"] = retriever
        frame["reranker_strategy"] = reranker
        frame["combo"] = make_combo(loader, chunker, embedder)
        frame["run_label"] = make_run_label(run_name, retriever, reranker)
        frame["summary_path"] = str(getattr(bundle, "summary_path", ""))
        frame["csv_path"] = str(getattr(bundle, "csv_path", "") or "")

        frame["case_key"] = frame.apply(make_case_key, axis=1)
        for metric_name in METRIC_COLUMNS:
            feedback_column = f"feedback.{metric_name}"
            comment_column = f"feedback.{metric_name}.comment"
            if feedback_column in frame.columns:
                frame[metric_name] = pd.to_numeric(frame[feedback_column], errors="coerce")
            elif metric_name in frame.columns:
                frame[metric_name] = pd.to_numeric(frame[metric_name], errors="coerce")
            if comment_column in frame.columns:
                frame[f"{metric_name}_comment"] = frame[comment_column].fillna("")
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return reconcile_cross_run_case_keys(pd.concat(frames, ignore_index=True))


def build_metric_frame(summary_frame: pd.DataFrame) -> pd.DataFrame:
    if summary_frame.empty:
        return pd.DataFrame(
            columns=[
                "run_name",
                "loader_strategy",
                "chunker_strategy",
                "embedding_provider",
                "combo",
                "run_label",
                "metric",
                "score",
            ]
        )

    id_columns = [
        "run_name",
        "loader_strategy",
        "chunker_strategy",
        "embedding_provider",
        "combo",
        "run_label",
    ]
    id_columns = [column for column in id_columns if column in summary_frame.columns]
    value_columns = [
        column for column in METRIC_COLUMNS if column in summary_frame.columns
    ]
    metric_frame = summary_frame.melt(
        id_vars=id_columns,
        value_vars=value_columns,
        var_name="metric",
        value_name="score",
    )
    metric_frame["score"] = pd.to_numeric(metric_frame["score"], errors="coerce")
    return metric_frame.dropna(subset=["score"]).reset_index(drop=True)


def filter_frame(
    frame: pd.DataFrame,
    loader_strategy: list[str],
    chunker_strategy: list[str],
    embedding_provider: list[str],
    retriever_strategy: list[str] | None = None,
    reranker_strategy: list[str] | None = None,
) -> pd.DataFrame:
    filtered = frame
    if loader_strategy and "loader_strategy" in filtered.columns:
        filtered = filtered[filtered["loader_strategy"].isin(loader_strategy)]
    if chunker_strategy and "chunker_strategy" in filtered.columns:
        filtered = filtered[filtered["chunker_strategy"].isin(chunker_strategy)]
    if embedding_provider and "embedding_provider" in filtered.columns:
        filtered = filtered[filtered["embedding_provider"].isin(embedding_provider)]
    if retriever_strategy and "retriever_strategy" in filtered.columns:
        filtered = filtered[filtered["retriever_strategy"].isin(retriever_strategy)]
    if reranker_strategy and "reranker_strategy" in filtered.columns:
        filtered = filtered[filtered["reranker_strategy"].isin(reranker_strategy)]
    return filtered.reset_index(drop=True)


def build_case_metric_matrix(examples: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot example-level scores into one row per stable case key."""

    run_column = "run_label" if "run_label" in examples.columns else "run_name"
    required_columns = {"case_key", "inputs.question", run_column, metric}
    if examples.empty or not required_columns.issubset(examples.columns):
        return pd.DataFrame()
    base = examples[["case_key", "inputs.question", run_column, metric]].copy()
    base[metric] = pd.to_numeric(base[metric], errors="coerce")
    questions = (
        base.groupby("case_key", dropna=False, as_index=False)["inputs.question"]
        .first()
    )
    pivot = base.pivot_table(
        index="case_key",
        columns=run_column,
        values=metric,
        aggfunc="first",
    ).reset_index()
    pivot.columns = [str(column) for column in pivot.columns]
    return questions.merge(pivot, on="case_key", how="right")


def filter_failed_examples(examples: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return example rows that fail the selected metric."""

    if examples.empty or metric not in examples.columns:
        return pd.DataFrame()
    scores = pd.to_numeric(examples[metric], errors="coerce")
    if metric == "critical_error":
        return examples[scores > 0].reset_index(drop=True)
    return examples[scores < 1].reset_index(drop=True)


def rows_for_case(examples: pd.DataFrame, case_key: str) -> pd.DataFrame:
    """Return all run rows for a single stable test-case key."""

    if examples.empty or "case_key" not in examples.columns:
        return pd.DataFrame()
    return examples[examples["case_key"].astype(str) == str(case_key)].reset_index(drop=True)


def compare_runs_for_case(
    examples: pd.DataFrame,
    case_key: str,
    left_run: str,
    right_run: str,
) -> pd.DataFrame:
    """Return selected run rows for one case."""

    rows = rows_for_case(examples, case_key)
    run_column = "run_label" if "run_label" in rows.columns else "run_name"
    if rows.empty or run_column not in rows.columns:
        return pd.DataFrame()
    selected = rows[rows[run_column].isin([left_run, right_run])].copy()
    order = {left_run: 0, right_run: 1}
    selected["_compare_order"] = selected[run_column].map(order)
    return (
        selected.sort_values("_compare_order", kind="stable")
        .drop(columns=["_compare_order"])
        .reset_index(drop=True)
    )


def rank_combinations(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Sort combinations best-first for the selected metric."""

    if summary.empty or metric not in summary.columns:
        return pd.DataFrame()
    ranked = summary.copy()
    ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
    ascending = metric == "critical_error"
    return ranked.sort_values(metric, ascending=ascending).reset_index(drop=True)


def build_failure_breakdown(failed: pd.DataFrame) -> pd.DataFrame:
    """Count failed rows by strategy and run for systematic failure analysis."""

    group_columns = [
        "loader_strategy",
        "chunker_strategy",
        "embedding_provider",
        "retriever_strategy",
        "reranker_strategy",
        "run_name",
        "run_label",
    ]
    available_group_columns = [column for column in group_columns if column in failed.columns]
    output_columns = [*available_group_columns, "failed_count"]
    if failed.empty or not available_group_columns:
        return pd.DataFrame(columns=output_columns)
    return (
        failed.groupby(available_group_columns, dropna=False)
        .size()
        .reset_index(name="failed_count")
        .sort_values("failed_count", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
