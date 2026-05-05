"""Local result file writing for retrieval evaluation."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.retrieval_eval.constants import (
    DEFAULT_ENSEMBLE_BM25_WEIGHT,
    DEFAULT_ENSEMBLE_CANDIDATE_K,
    DEFAULT_ENSEMBLE_USE_CHUNK_ID,
    DEFAULT_K,
    DEFAULT_RERANKER_STRATEGY,
    DEFAULT_RETRIEVAL_INPUT_MODE,
    DEFAULT_RETRIEVER_STRATEGY,
)
from evaluation.retrieval_eval.dataset import make_dataset_name


def single_run_from_args(args: argparse.Namespace) -> dict[str, str]:
    return {
        "name": (
            f"{args.loader_strategy}-{args.chunker_strategy}-"
            f"{args.embedding_provider}"
        ),
        "loader_strategy": args.loader_strategy,
        "chunker_strategy": args.chunker_strategy,
        "embedding_provider": args.embedding_provider,
    }


def dataset_name_for_run(
    args: argparse.Namespace,
    run: dict[str, str],
    matrix_mode: bool,
) -> str:
    if args.dataset_name:
        return args.dataset_name
    return make_dataset_name(args.testset_path)


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣._-]+", "-", value).strip("-")
    return cleaned or "langsmith-eval"


def summarize_feedback_metrics(dataframe) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for column in dataframe.columns:
        if not str(column).startswith("feedback."):
            continue
        metric_name = str(column).removeprefix("feedback.")
        if metric_name.endswith(".comment"):
            continue
        numeric = dataframe[column]
        try:
            numeric = numeric.astype(float)
        except (TypeError, ValueError):
            numeric = None
        if numeric is None:
            metrics[metric_name] = None
            continue
        numeric = numeric.dropna()
        metrics[metric_name] = float(numeric.mean()) if len(numeric) else None
    return metrics


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _numeric_sum(dataframe, column: str) -> int:
    if column not in dataframe.columns:
        return 0
    try:
        values = dataframe[column].astype(float).dropna()
    except (TypeError, ValueError):
        return 0
    return int(values.sum()) if len(values) else 0


def summarize_embedding_query_cache(dataframe) -> dict[str, Any]:
    """Summarize query embedding cache usage recorded in eval outputs."""

    enabled_column = "outputs.embedding_query_cache_enabled"
    hit_column = "outputs.embedding_query_cache_hit"
    hits_column = "outputs.embedding_query_cache_hits"
    misses_column = "outputs.embedding_query_cache_misses"

    enabled = None
    if enabled_column in dataframe.columns and len(dataframe[enabled_column].dropna()):
        enabled = _parse_bool(dataframe[enabled_column].dropna().iloc[0])

    hit_rows = 0
    if hit_column in dataframe.columns:
        hit_rows = int(dataframe[hit_column].fillna(False).map(_parse_bool).sum())

    return {
        "enabled": enabled,
        "hit_rows": hit_rows,
        "hits": _numeric_sum(dataframe, hits_column),
        "misses": _numeric_sum(dataframe, misses_column),
    }


def save_experiment_dataframe(
    dataframe,
    experiment_name: str,
    output_dir: Path,
    run: dict[str, str],
    dataset_name: str,
    testset_path: Path,
    retriever_strategy: str = DEFAULT_RETRIEVER_STRATEGY,
    reranker_strategy: str = DEFAULT_RERANKER_STRATEGY,
    final_k: int = DEFAULT_K,
    candidate_k: int | None = None,
    ensemble_bm25_weight: float = DEFAULT_ENSEMBLE_BM25_WEIGHT,
    ensemble_candidate_k: int = DEFAULT_ENSEMBLE_CANDIDATE_K,
    ensemble_use_chunk_id: bool = DEFAULT_ENSEMBLE_USE_CHUNK_ID,
    retrieval_input_mode: str = DEFAULT_RETRIEVAL_INPUT_MODE,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = safe_filename(
        f"{timestamp}-{run['name']}-{retriever_strategy}-{reranker_strategy}-{retrieval_input_mode}"
    )
    csv_path = output_dir / f"{base_name}.csv"
    summary_path = output_dir / f"{base_name}.summary.json"

    dataframe.to_csv(csv_path, index=False)
    suite = None
    if "evaluation_suite" in dataframe.columns and len(dataframe["evaluation_suite"].dropna()):
        suite = str(dataframe["evaluation_suite"].dropna().iloc[0])
    elif "suite" in dataframe.columns and len(dataframe["suite"].dropna()):
        suite = str(dataframe["suite"].dropna().iloc[0])
    summary = {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "testset_path": str(testset_path),
        "evaluation_suite": suite,
        "run_name": run["name"],
        "loader_strategy": run["loader_strategy"],
        "chunker_strategy": run["chunker_strategy"],
        "embedding_provider": run["embedding_provider"],
        "retriever_strategy": retriever_strategy,
        "reranker_strategy": reranker_strategy,
        "retrieval_input_mode": retrieval_input_mode,
        "final_k": final_k,
        "candidate_k": candidate_k,
        "ensemble_bm25_weight": ensemble_bm25_weight,
        "ensemble_candidate_k": ensemble_candidate_k,
        "ensemble_use_chunk_id": ensemble_use_chunk_id,
        "embedding_query_cache": summarize_embedding_query_cache(dataframe),
        "row_count": int(len(dataframe)),
        "metrics": summarize_feedback_metrics(dataframe),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"csv": csv_path, "summary_json": summary_path}


def save_experiment_results(
    results,
    output_dir: Path,
    run: dict[str, str],
    dataset_name: str,
    testset_path: Path,
    retriever_strategy: str = DEFAULT_RETRIEVER_STRATEGY,
    reranker_strategy: str = DEFAULT_RERANKER_STRATEGY,
    final_k: int = DEFAULT_K,
    candidate_k: int | None = None,
    ensemble_bm25_weight: float = DEFAULT_ENSEMBLE_BM25_WEIGHT,
    ensemble_candidate_k: int = DEFAULT_ENSEMBLE_CANDIDATE_K,
    ensemble_use_chunk_id: bool = DEFAULT_ENSEMBLE_USE_CHUNK_ID,
    retrieval_input_mode: str = DEFAULT_RETRIEVAL_INPUT_MODE,
) -> dict[str, Path]:
    dataframe = results.to_pandas()
    return save_experiment_dataframe(
        dataframe=dataframe,
        experiment_name=results.experiment_name,
        output_dir=output_dir,
        run=run,
        dataset_name=dataset_name,
        testset_path=testset_path,
        retriever_strategy=retriever_strategy,
        reranker_strategy=reranker_strategy,
        final_k=final_k,
        candidate_k=candidate_k,
        ensemble_bm25_weight=ensemble_bm25_weight,
        ensemble_candidate_k=ensemble_candidate_k,
        ensemble_use_chunk_id=ensemble_use_chunk_id,
        retrieval_input_mode=retrieval_input_mode,
    )
