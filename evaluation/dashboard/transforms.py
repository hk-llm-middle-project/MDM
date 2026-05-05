"""Normalize local evaluation exports into dashboard-friendly tables."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import hashlib
from pathlib import Path
import re

import pandas as pd

from evaluation.dashboard.metrics import (
    COMPARISON_METRIC_COLUMNS,
    LOWER_IS_BETTER_METRICS,
    METRIC_COLUMNS,
    TIME_METRIC_COLUMNS,
    describe_metric,
)
from evaluation.dashboard.case_tables import (
    build_case_metric_comparison,
    build_case_metric_matrix,
    build_case_value_comparison,
    build_expected_actual_case_table,
    build_expected_actual_case_table_styles,
    case_question,
    compare_runs_for_case,
    rows_for_case,
)
from evaluation.dashboard.formatters import (
    format_display_value as _format_display_value,
    has_display_value as _has_display_value,
)


CASE_METADATA_COLUMNS = (
    "evaluation_suite",
    "suite",
    "case_type_codes",
    "difficulty",
    "case_family",
    "inference_type",
    "query_style",
    "requires_diagram",
    "requires_table",
    "filter_risk",
)

SUMMARY_METADATA_COLUMNS = (
    "experiment_name",
    "dataset_name",
    "testset_path",
    "evaluation_suite",
    "nickname",
    "run_name",
    "loader_strategy",
    "chunker_strategy",
    "embedding_provider",
    "retriever_strategy",
    "reranker_strategy",
    "retrieval_input_mode",
    "ensemble_bm25_weight",
    "ensemble_candidate_k",
    "ensemble_use_chunk_id",
    "retriever_reranker",
    "row_count",
    "execution_time",
    "summary_path",
    "csv_path",
    "result_stem",
    "combo",
    "run_label",
    "testset_label",
)


def make_combo(loader: Any, chunker: Any, embedder: Any) -> str:
    return f"{loader or '-'} / {chunker or '-'} / {embedder or '-'}"


def make_testset_label(summary: dict[str, Any]) -> str:
    """Return a stable short label for selecting a dashboard testset."""

    suite = summary.get("evaluation_suite") or summary.get("suite")
    if _has_display_value(suite):
        return _format_display_value(suite)
    testset_path = summary.get("testset_path")
    if _has_display_value(testset_path):
        return Path(str(testset_path)).stem
    dataset_name = summary.get("dataset_name")
    if _has_display_value(dataset_name):
        return _format_display_value(dataset_name)
    return "unknown"


def _numeric_weight(value: Any) -> float | None:
    if value is None:
        return None
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(weight):
        return None
    return weight


def _format_ensemble_weight_label(retriever: Any, bm25_weight: Any) -> str:
    retriever_name = str(retriever or "")
    weight = _numeric_weight(bm25_weight)
    if "ensemble" not in retriever_name or weight is None:
        return ""

    if abs(weight - (2 / 11)) < 0.001:
        return "BM25:Dense 2:9"

    bm25_ratio = round(weight * 10, 1)
    dense_ratio = round((1 - weight) * 10, 1)
    if float(bm25_ratio).is_integer() and float(dense_ratio).is_integer():
        ratio = f"{int(bm25_ratio)}:{int(dense_ratio)}"
    else:
        ratio = f"{bm25_ratio:g}:{dense_ratio:g}"
    return f"BM25:Dense {ratio}"


def _result_stem_suffix(value: Any) -> str:
    stem = str(value or "").strip()
    if not stem:
        return "-"
    match = re.match(r"(\d{8}-\d{6})", stem)
    if match:
        return match.group(1)
    return stem


def _disambiguate_duplicate_run_labels(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not {"run_label", "result_stem"}.issubset(frame.columns):
        return frame

    disambiguated = frame.copy()
    stems_per_label = disambiguated.groupby("run_label")["result_stem"].transform(
        lambda values: values.dropna().astype(str).nunique()
    )
    duplicate_labels = stems_per_label > 1
    if not duplicate_labels.any():
        return disambiguated

    disambiguated.loc[duplicate_labels, "run_label"] = disambiguated.loc[
        duplicate_labels
    ].apply(
        lambda row: f"{row['run_label']} [{_result_stem_suffix(row.get('result_stem'))}]",
        axis=1,
    )
    return disambiguated


def make_run_label(
    run_name: Any,
    retriever: Any,
    reranker: Any,
    ensemble_bm25_weight: Any = None,
    retrieval_input_mode: Any = None,
) -> str:
    name = str(run_name or "-")
    retriever_name = str(retriever or "")
    reranker_name = str(reranker or "")
    if not retriever_name and not reranker_name:
        return name
    label = f"{name} / {retriever_name or '-'} / {reranker_name or '-'}"
    weight_label = _format_ensemble_weight_label(retriever, ensemble_bm25_weight)
    if weight_label:
        label = f"{label} / {weight_label}"
    mode = str(retrieval_input_mode or "").strip()
    if mode and mode != "raw":
        return f"{label} / {mode}"
    return label


def _bundle_value(bundle: Any, key: str, default: Any = None) -> Any:
    return getattr(bundle, "summary", {}).get(key, default)


def _mean_execution_time(bundle: Any) -> float | None:
    csv_path = getattr(bundle, "csv_path", None)
    if csv_path is None:
        return None
    try:
        frame = pd.read_csv(csv_path, usecols=["execution_time"])
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None

    values = pd.to_numeric(frame["execution_time"], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def make_retriever_reranker(
    retriever: Any,
    reranker: Any,
    ensemble_bm25_weight: Any = None,
    retrieval_input_mode: Any = None,
) -> str:
    retriever_value = str(retriever or "unknown")
    reranker_value = str(reranker or "unknown")
    label = f"{retriever_value} / {reranker_value}"
    weight_label = _format_ensemble_weight_label(retriever, ensemble_bm25_weight)
    if weight_label:
        label = f"{label} / {weight_label}"
    mode = str(retrieval_input_mode or "").strip()
    if mode and mode != "raw":
        return f"{label} / {mode}"
    return label


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
        retrieval_input_mode = summary.get("retrieval_input_mode") or "raw"
        ensemble_bm25_weight = summary.get("ensemble_bm25_weight")
        run_name = summary.get("run_name") or getattr(bundle, "run_name", None)
        testset_label = make_testset_label(summary)
        record: dict[str, Any] = {
            "experiment_name": summary.get("experiment_name"),
            "dataset_name": summary.get("dataset_name"),
            "testset_path": summary.get("testset_path"),
            "evaluation_suite": summary.get("evaluation_suite") or summary.get("suite"),
            "nickname": summary.get("nickname"),
            "run_name": run_name,
            "loader_strategy": loader,
            "chunker_strategy": chunker,
            "embedding_provider": embedder,
            "retriever_strategy": retriever,
            "reranker_strategy": reranker,
            "retrieval_input_mode": retrieval_input_mode,
            "testset_label": testset_label,
            "ensemble_bm25_weight": ensemble_bm25_weight,
            "ensemble_candidate_k": summary.get("ensemble_candidate_k"),
            "ensemble_use_chunk_id": summary.get("ensemble_use_chunk_id"),
            "retriever_reranker": make_retriever_reranker(
                retriever,
                reranker,
                ensemble_bm25_weight,
                retrieval_input_mode,
            ),
            "row_count": summary.get("row_count"),
            "execution_time": _mean_execution_time(bundle),
            "summary_path": str(getattr(bundle, "summary_path", "")),
            "csv_path": str(getattr(bundle, "csv_path", "") or ""),
            "result_stem": getattr(bundle, "result_stem", None),
            "combo": make_combo(loader, chunker, embedder),
            "run_label": make_run_label(
                run_name,
                retriever,
                reranker,
                ensemble_bm25_weight,
                retrieval_input_mode,
            ),
        }
        for metric_name in METRIC_COLUMNS:
            record[metric_name] = metrics.get(metric_name)
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return pd.DataFrame(columns=list(SUMMARY_METADATA_COLUMNS) + list(METRIC_COLUMNS))
    for metric_name in METRIC_COLUMNS:
        frame[metric_name] = pd.to_numeric(frame[metric_name], errors="coerce")
    for metric_name in TIME_METRIC_COLUMNS:
        frame[metric_name] = pd.to_numeric(frame[metric_name], errors="coerce")
    return _disambiguate_duplicate_run_labels(frame)


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
        retrieval_input_mode = summary.get("retrieval_input_mode") or "raw"
        ensemble_bm25_weight = summary.get("ensemble_bm25_weight")
        run_name = summary.get("run_name") or getattr(bundle, "run_name", None)
        testset_label = make_testset_label(summary)
        frame = frame.copy()
        frame["run_name"] = run_name
        frame["nickname"] = summary.get("nickname")
        frame["experiment_name"] = summary.get("experiment_name")
        frame["dataset_name"] = summary.get("dataset_name")
        frame["testset_path"] = summary.get("testset_path")
        frame["testset_label"] = testset_label
        frame["loader_strategy"] = loader
        frame["chunker_strategy"] = chunker
        frame["embedding_provider"] = embedder
        frame["retriever_strategy"] = retriever
        frame["reranker_strategy"] = reranker
        frame["retrieval_input_mode"] = retrieval_input_mode
        frame["ensemble_bm25_weight"] = ensemble_bm25_weight
        frame["ensemble_candidate_k"] = summary.get("ensemble_candidate_k")
        frame["ensemble_use_chunk_id"] = summary.get("ensemble_use_chunk_id")
        frame["retriever_reranker"] = make_retriever_reranker(
            retriever,
            reranker,
            ensemble_bm25_weight,
            retrieval_input_mode,
        )
        frame["combo"] = make_combo(loader, chunker, embedder)
        frame["run_label"] = make_run_label(
            run_name,
            retriever,
            reranker,
            ensemble_bm25_weight,
            retrieval_input_mode,
        )
        frame["summary_path"] = str(getattr(bundle, "summary_path", ""))
        frame["csv_path"] = str(getattr(bundle, "csv_path", "") or "")
        frame["result_stem"] = getattr(bundle, "result_stem", None)
        if "evaluation_suite" not in frame.columns:
            frame["evaluation_suite"] = summary.get("evaluation_suite") or summary.get("suite")
        for column in CASE_METADATA_COLUMNS:
            metadata_column = f"metadata.{column}"
            if column not in frame.columns and metadata_column in frame.columns:
                frame[column] = frame[metadata_column]
        if "evaluation_suite" in frame.columns and "suite" in frame.columns:
            frame["evaluation_suite"] = frame["evaluation_suite"].fillna(frame["suite"])
        elif "evaluation_suite" not in frame.columns and "suite" in frame.columns:
            frame["evaluation_suite"] = frame["suite"]

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
    examples = reconcile_cross_run_case_keys(pd.concat(frames, ignore_index=True))
    return _disambiguate_duplicate_run_labels(examples)


def build_metric_frame(summary_frame: pd.DataFrame) -> pd.DataFrame:
    if summary_frame.empty:
        return pd.DataFrame(
            columns=[
                "run_name",
                "nickname",
                "loader_strategy",
                "chunker_strategy",
                "embedding_provider",
                "retriever_strategy",
                "reranker_strategy",
                "retrieval_input_mode",
                "ensemble_bm25_weight",
                "ensemble_candidate_k",
                "ensemble_use_chunk_id",
                "retriever_reranker",
                "combo",
                "run_label",
                "metric",
                "score",
            ]
        )

    summary_frame = summary_frame.copy()
    if (
        "retriever_reranker" not in summary_frame.columns
        and {"retriever_strategy", "reranker_strategy"}.issubset(summary_frame.columns)
    ):
        summary_frame["retriever_reranker"] = summary_frame.apply(
            lambda row: make_retriever_reranker(
                row.get("retriever_strategy"),
                row.get("reranker_strategy"),
                row.get("ensemble_bm25_weight"),
                row.get("retrieval_input_mode"),
            ),
            axis=1,
        )

    id_columns = [
        "evaluation_suite",
        "testset_label",
        "nickname",
        "run_name",
        "loader_strategy",
        "chunker_strategy",
        "embedding_provider",
        "retriever_strategy",
        "reranker_strategy",
        "retrieval_input_mode",
        "ensemble_bm25_weight",
        "ensemble_candidate_k",
        "ensemble_use_chunk_id",
        "retriever_reranker",
        "combo",
        "run_label",
    ]
    id_columns = [column for column in id_columns if column in summary_frame.columns]
    value_columns = [
        column for column in COMPARISON_METRIC_COLUMNS if column in summary_frame.columns
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
    evaluation_suite: list[str] | None = None,
    difficulty: list[str] | None = None,
    case_family: list[str] | None = None,
) -> pd.DataFrame:
    filtered = frame
    if loader_strategy and "loader_strategy" in filtered.columns:
        filtered = filtered[_matches_filter_or_empty(filtered["loader_strategy"], loader_strategy)]
    if chunker_strategy and "chunker_strategy" in filtered.columns:
        filtered = filtered[_matches_filter_or_empty(filtered["chunker_strategy"], chunker_strategy)]
    if embedding_provider and "embedding_provider" in filtered.columns:
        filtered = filtered[_matches_filter_or_empty(filtered["embedding_provider"], embedding_provider)]
    if retriever_strategy and "retriever_strategy" in filtered.columns:
        filtered = filtered[_matches_filter_or_empty(filtered["retriever_strategy"], retriever_strategy)]
    if reranker_strategy and "reranker_strategy" in filtered.columns:
        filtered = filtered[_matches_filter_or_empty(filtered["reranker_strategy"], reranker_strategy)]
    if evaluation_suite and "evaluation_suite" in filtered.columns:
        filtered = filtered[filtered["evaluation_suite"].isin(evaluation_suite)]
    if difficulty and "difficulty" in filtered.columns:
        if not _selected_covers_all_non_empty_values(filtered["difficulty"], difficulty):
            filtered = filtered[filtered["difficulty"].isin(difficulty)]
    if case_family and "case_family" in filtered.columns:
        if not _selected_covers_all_non_empty_values(filtered["case_family"], case_family):
            filtered = filtered[filtered["case_family"].isin(case_family)]
    return filtered.reset_index(drop=True)


def _matches_filter_or_empty(series: pd.Series, selected: list[str]) -> pd.Series:
    """Keep strategy-less suite rows when parser/retriever filters are active."""

    text_values = series.astype("string")
    return series.isna() | text_values.isin(selected) | (text_values.fillna("") == "")


def _selected_covers_all_non_empty_values(
    series: pd.Series,
    selected: list[str],
) -> bool:
    """Treat selecting every visible metadata value as no metadata filter."""

    available = {
        value
        for value in series.dropna().astype(str)
        if value
    }
    return bool(available) and available.issubset({str(value) for value in selected})



def filter_failed_examples(examples: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return example rows that fail the selected metric."""

    if examples.empty or metric not in examples.columns:
        return pd.DataFrame()
    scores = pd.to_numeric(examples[metric], errors="coerce")
    if metric == "critical_error":
        return examples[scores > 0].reset_index(drop=True)
    return examples[scores < 1].reset_index(drop=True)


def rank_combinations(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Sort combinations best-first for the selected metric."""

    if summary.empty or metric not in summary.columns:
        return pd.DataFrame()
    ranked = summary.copy()
    ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
    ascending = metric in LOWER_IS_BETTER_METRICS
    return ranked.sort_values(metric, ascending=ascending).reset_index(drop=True)


def build_failure_breakdown(failed: pd.DataFrame) -> pd.DataFrame:
    """Count failed rows by strategy and run for systematic failure analysis."""

    group_columns = [
        "evaluation_suite",
        "suite",
        "case_type_codes",
        "difficulty",
        "case_family",
        "inference_type",
        "query_style",
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
