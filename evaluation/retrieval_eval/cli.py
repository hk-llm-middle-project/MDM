"""CLI parsing and run-option expansion for retrieval evaluation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from evaluation.retrieval_eval.constants import (
    DEFAULT_CHUNKER_STRATEGY,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_ENSEMBLE_BM25_WEIGHT,
    DEFAULT_ENSEMBLE_CANDIDATE_K,
    DEFAULT_ENSEMBLE_USE_CHUNK_ID,
    DEFAULT_K,
    DEFAULT_LOADER_STRATEGY,
    DEFAULT_MATRIX_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RERANKER_STRATEGY,
    DEFAULT_RETRIEVAL_INPUT_MODE,
    DEFAULT_RETRIEVER_STRATEGY,
    DEFAULT_TESTSET_PATH,
    RERANKER_STRATEGY_ALIASES,
    RERANKER_STRATEGY_CHOICES,
    RERANKER_STRATEGY_INPUT_CHOICES,
    RETRIEVAL_INPUT_MODE_CHOICES,
    RETRIEVER_STRATEGY_CHOICES,
)
from config import DEFAULT_RERANKER_CANDIDATE_K, DEFAULT_RERANKER_FINAL_K


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local Chroma retrieval and record retrieval metrics in LangSmith.",
    )
    parser.add_argument(
        "--testset-path",
        type=Path,
        default=DEFAULT_TESTSET_PATH,
        help="JSONL retrieval testset path.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help=(
            "LangSmith dataset name override. By default, the dataset name is fixed "
            "by the testset filename so matrix runs share one comparable dataset."
        ),
    )
    parser.add_argument(
        "--matrix-path",
        type=Path,
        default=DEFAULT_MATRIX_PATH,
        help="JSON file containing valid loader/chunker/embedder evaluation combinations.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Run a named matrix preset, e.g. upstage, parser-baseline, or all.",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run the default matrix preset instead of a single combination.",
    )
    parser.add_argument(
        "--list-matrix",
        action="store_true",
        help="Print configured matrix presets and runs without contacting LangSmith.",
    )
    parser.add_argument(
        "--loader-strategy",
        default=DEFAULT_LOADER_STRATEGY,
        help="Vectorstore loader namespace, e.g. upstage or pdfplumber.",
    )
    parser.add_argument(
        "--embedding-provider",
        default=DEFAULT_EMBEDDING_PROVIDER,
        help="Embedding provider namespace, e.g. openai, bge, google.",
    )
    parser.add_argument(
        "--chunker-strategy",
        default=DEFAULT_CHUNKER_STRATEGY,
        help="Chunker/vectorstore namespace, e.g. fixed, recursive, raw, custom.",
    )
    parser.add_argument(
        "--retriever-strategy",
        default=DEFAULT_RETRIEVER_STRATEGY,
        choices=RETRIEVER_STRATEGY_CHOICES,
        help="Retrieval strategy, e.g. similarity, ensemble, ensemble_parent, parent.",
    )
    parser.add_argument(
        "--reranker-strategy",
        default=DEFAULT_RERANKER_STRATEGY,
        choices=RERANKER_STRATEGY_INPUT_CHOICES,
        help=(
            "Reranker strategy exposed in Streamlit, e.g. none, cross-encoder, "
            "flashrank, llm-score. Underscore aliases are accepted."
        ),
    )
    parser.add_argument(
        "--retriever-strategies",
        default=None,
        help=(
            "Comma-separated retriever strategy list, or 'all'. "
            "Expands each parser/chunker/embedder run across these strategies."
        ),
    )
    parser.add_argument(
        "--reranker-strategies",
        default=None,
        help=(
            "Comma-separated reranker strategy list, or 'all'. "
            "Expands each parser/chunker/embedder run across these strategies."
        ),
    )
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help=(
            "Shorthand for all retriever strategies x all exposed reranker strategies. "
            "Explicit --retriever-strategies/--reranker-strategies values narrow either side."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=(
            "Number of final retrieved documents to evaluate. "
            "Defaults to Streamlit behavior: 5 without reranker, 3 with reranker."
        ),
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=0,
        help=(
            "Number of candidates to retrieve before reranking. "
            "Set to 0 to use the strategy default."
        ),
    )
    parser.add_argument(
        "--ensemble-bm25-weight",
        type=float,
        default=DEFAULT_ENSEMBLE_BM25_WEIGHT,
        help="BM25 weight for ensemble retrievers. Dense weight is 1 - this value.",
    )
    parser.add_argument(
        "--ensemble-candidate-k",
        type=int,
        default=DEFAULT_ENSEMBLE_CANDIDATE_K,
        help="BM25 and dense candidate count for ensemble retrievers.",
    )
    parser.add_argument(
        "--no-ensemble-use-chunk-id",
        action="store_false",
        dest="ensemble_use_chunk_id",
        default=DEFAULT_ENSEMBLE_USE_CHUNK_ID,
        help="Disable chunk_id de-duplication for ensemble retrievers.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Evaluate only the first N examples. Set to 0 for all examples.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of examples to run concurrently.",
    )
    parser.add_argument(
        "--retrieval-input-mode",
        choices=RETRIEVAL_INPUT_MODE_CHOICES,
        default=DEFAULT_RETRIEVAL_INPUT_MODE,
        help=(
            "How to build the retrieval query. 'raw' uses the testset question "
            "directly. 'intake' first runs the Streamlit accident intake flow, "
            "then retrieves with the intake retrieval_query and metadata filters."
        ),
    )
    parser.add_argument(
        "--with-intake",
        action="store_true",
        help="Alias for --retrieval-input-mode intake.",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Only upload the LangSmith dataset; do not run retrieval evaluation.",
    )
    parser.add_argument(
        "--langsmith",
        action="store_true",
        help=(
            "Run through LangSmith evaluate(). By default this script evaluates "
            "locally and writes CSV/JSON only to avoid consuming trace quota."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for local CSV/JSON summaries of LangSmith experiment results.",
    )
    parser.add_argument(
        "--no-local-results",
        action="store_true",
        help="Do not write local CSV/JSON result summaries.",
    )
    parser.add_argument(
        "--fail-on-missing-vectorstore",
        action="store_true",
        help="Fail matrix runs instead of skipping combinations whose vectorstore is not built.",
    )
    args = parser.parse_args()
    args.reranker_strategy = normalize_strategy_value(
        args.reranker_strategy,
        "reranker",
    )
    if args.with_intake:
        args.retrieval_input_mode = "intake"
    return args


def should_run_matrix(args: argparse.Namespace) -> bool:
    return bool(args.matrix or args.preset)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def resolve_strategy_values(
    raw_value: str | None,
    default_value: str,
    choices: tuple[str, ...],
    label: str,
    *,
    all_by_default: bool = False,
) -> list[str]:
    if raw_value is None:
        return list(choices) if all_by_default else [default_value]

    value = raw_value.strip()
    if value == "all":
        return list(choices)

    selected = _dedupe_preserve_order(
        [
            normalize_strategy_value(part.strip(), label)
            for part in value.split(",")
            if part.strip()
        ]
    )
    if not selected:
        return [default_value]

    unknown = [strategy for strategy in selected if strategy not in choices]
    if unknown:
        available = ", ".join(choices)
        raise ValueError(
            f"Unknown {label} strategy: {', '.join(unknown)}. "
            f"Available {label} strategies: {available}"
        )
    return selected


def normalize_strategy_value(value: str, label: str) -> str:
    if label == "reranker":
        return RERANKER_STRATEGY_ALIASES.get(value, value)
    return value


def resolve_strategy_combinations(args: argparse.Namespace) -> list[dict[str, str]]:
    all_strategies = bool(getattr(args, "all_strategies", False))
    retriever_raw = getattr(args, "retriever_strategies", None)
    reranker_raw = getattr(args, "reranker_strategies", None)
    retrievers = resolve_strategy_values(
        retriever_raw,
        getattr(args, "retriever_strategy", DEFAULT_RETRIEVER_STRATEGY),
        RETRIEVER_STRATEGY_CHOICES,
        "retriever",
        all_by_default=all_strategies and retriever_raw is None,
    )
    rerankers = resolve_strategy_values(
        reranker_raw,
        getattr(args, "reranker_strategy", DEFAULT_RERANKER_STRATEGY),
        RERANKER_STRATEGY_CHOICES,
        "reranker",
        all_by_default=all_strategies and reranker_raw is None,
    )
    return [
        {
            "retriever_strategy": retriever,
            "reranker_strategy": reranker,
        }
        for retriever in retrievers
        for reranker in rerankers
    ]


def build_execution_plan(
    args: argparse.Namespace,
    runs: list[dict[str, str]],
) -> list[tuple[dict[str, str], dict[str, str]]]:
    strategy_combinations = resolve_strategy_combinations(args)
    return [
        (run, strategy_combination)
        for run in runs
        for strategy_combination in strategy_combinations
    ]


def args_with_strategy(
    args: argparse.Namespace,
    strategy_combination: dict[str, str],
) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(strategy_combination)
    return argparse.Namespace(**values)


def effective_candidate_k(reranker_strategy: str, candidate_k: int) -> int | None:
    if candidate_k > 0:
        return candidate_k
    if reranker_strategy != "none":
        return DEFAULT_RERANKER_CANDIDATE_K
    return None


def effective_final_k(reranker_strategy: str, k: int | None) -> int:
    if k is not None:
        return k
    if reranker_strategy != "none":
        return DEFAULT_RERANKER_FINAL_K
    return DEFAULT_K


def ensemble_bm25_weight_from_args(args: argparse.Namespace) -> float:
    return float(getattr(args, "ensemble_bm25_weight", DEFAULT_ENSEMBLE_BM25_WEIGHT))


def ensemble_candidate_k_from_args(args: argparse.Namespace) -> int:
    return int(getattr(args, "ensemble_candidate_k", DEFAULT_ENSEMBLE_CANDIDATE_K))


def ensemble_use_chunk_id_from_args(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "ensemble_use_chunk_id", DEFAULT_ENSEMBLE_USE_CHUNK_ID))


def validate_run_args(args: argparse.Namespace) -> None:
    if args.k is not None and args.k <= 0:
        raise ValueError("--k must be greater than 0")
    if args.candidate_k < 0:
        raise ValueError("--candidate-k must be greater than or equal to 0")
    ensemble_bm25_weight = ensemble_bm25_weight_from_args(args)
    if not 0 <= ensemble_bm25_weight <= 1:
        raise ValueError("--ensemble-bm25-weight must be between 0 and 1")
    ensemble_candidate_k = ensemble_candidate_k_from_args(args)
    if ensemble_candidate_k <= 0:
        raise ValueError("--ensemble-candidate-k must be greater than 0")


def configure_tracing(needs_langsmith: bool) -> None:
    """Prevent accidental LangSmith trace creation during local-only evals."""

    if needs_langsmith:
        return
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
