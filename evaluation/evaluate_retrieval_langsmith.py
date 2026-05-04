"""Evaluate local retrieval results on a LangSmith dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langsmith import Client, evaluate
from langsmith.schemas import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BASE_DIR,
    DEFAULT_CHUNKER_STRATEGY as APP_DEFAULT_CHUNKER_STRATEGY,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_ENSEMBLE_BM25_WEIGHT,
    DEFAULT_ENSEMBLE_CANDIDATE_K,
    DEFAULT_ENSEMBLE_USE_CHUNK_ID,
    DEFAULT_LOADER_STRATEGY as APP_DEFAULT_LOADER_STRATEGY,
    DEFAULT_RERANKER_CANDIDATE_K,
    RETRIEVER_K,
    get_vectorstore_dir,
)
from main import build_pipeline_config
from rag.indexer import load_vectorstore, vectorstore_exists
from rag.pipeline.reranker import RERANKER_STRATEGIES
from rag.pipeline.retrieval import run_retrieval_pipeline
from rag.pipeline.retriever import RETRIEVAL_STRATEGIES, build_retrieval_components


DEFAULT_TESTSET_PATH = (
    BASE_DIR / "data" / "testsets" / "langsmith" / "retrieval_eval.jsonl"
)
DEFAULT_MATRIX_PATH = BASE_DIR / "evaluation" / "retrieval_eval_matrix.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "evaluation" / "results" / "langsmith"
DEFAULT_DATASET_PREFIX = "MDM retrieval testset"
DEFAULT_EXPERIMENT_PREFIX = "MDM retrieval eval"
DEFAULT_LOADER_STRATEGY = APP_DEFAULT_LOADER_STRATEGY
DEFAULT_CHUNKER_STRATEGY = APP_DEFAULT_CHUNKER_STRATEGY
DEFAULT_RETRIEVER_STRATEGY = "similarity"
DEFAULT_RERANKER_STRATEGY = "none"
DEFAULT_K = RETRIEVER_K
RETRIEVER_STRATEGY_CHOICES = tuple(RETRIEVAL_STRATEGIES)
RERANKER_STRATEGY_CHOICES = tuple(
    strategy for strategy in ("none", "cross-encoder", "flashrank", "llm-score")
    if strategy in RERANKER_STRATEGIES
)
RERANKER_STRATEGY_ALIASES = {
    "cross_encoder": "cross-encoder",
    "llm_score": "llm-score",
}
RERANKER_STRATEGY_INPUT_CHOICES = tuple(
    [*RERANKER_STRATEGY_CHOICES, *RERANKER_STRATEGY_ALIASES]
)
CONTENT_PREVIEW_CHARS = 500
EXAMPLE_METADATA_FIELDS = (
    "suite",
    "case_type_codes",
    "difficulty",
    "case_family",
    "inference_type",
    "query_style",
    "requires_diagram",
    "requires_table",
    "filter_risk",
    "candidate_k",
    "final_k",
)


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
        default=DEFAULT_K,
        help="Number of final retrieved documents to evaluate.",
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


def ensemble_bm25_weight_from_args(args: argparse.Namespace) -> float:
    return float(getattr(args, "ensemble_bm25_weight", DEFAULT_ENSEMBLE_BM25_WEIGHT))


def ensemble_candidate_k_from_args(args: argparse.Namespace) -> int:
    return int(getattr(args, "ensemble_candidate_k", DEFAULT_ENSEMBLE_CANDIDATE_K))


def ensemble_use_chunk_id_from_args(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "ensemble_use_chunk_id", DEFAULT_ENSEMBLE_USE_CHUNK_ID))


def validate_run_args(args: argparse.Namespace) -> None:
    if args.k <= 0:
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
        raise RuntimeError("No examples were loaded.")
    return rows


def make_dataset_name(path: Path) -> str:
    return f"{DEFAULT_DATASET_PREFIX} - {path.stem}"


def build_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        question = row.get("question")
        if not question:
            continue

        inputs: dict[str, Any] = {"question": question}
        if row.get("candidate_k") is not None:
            inputs["candidate_k"] = row.get("candidate_k")
        if row.get("final_k") is not None:
            inputs["final_k"] = row.get("final_k")

        expected_keywords = row.get("expected_keywords")
        if expected_keywords is None:
            expected_keywords = row.get("expected_evidence_keywords", [])

        metadata = {
            "id": row.get("id"),
            "notes": row.get("notes"),
            "suite": row.get("suite") or infer_suite_from_row(row),
        }
        for field in EXAMPLE_METADATA_FIELDS:
            if field == "suite":
                continue
            if row.get(field) is not None:
                metadata[field] = row.get(field)

        examples.append(
            {
                "inputs": inputs,
                "outputs": {
                    "reference": row.get("reference", ""),
                    "expected_diagram_ids": row.get("expected_diagram_ids", []),
                    "acceptable_diagram_ids": row.get("acceptable_diagram_ids", []),
                    "near_miss_diagram_ids": row.get("near_miss_diagram_ids", []),
                    "expected_party_type": row.get("expected_party_type"),
                    "expected_location": row.get("expected_location"),
                    "expected_chunk_types": row.get("expected_chunk_types", []),
                    "expected_keywords": expected_keywords,
                    "requires_diagram": row.get("requires_diagram"),
                    "requires_table": row.get("requires_table"),
                },
                "metadata": metadata,
            }
        )

    if not examples:
        raise RuntimeError("No valid examples with question were found.")
    return examples


def infer_suite_from_row(row: dict[str, Any]) -> str:
    """Infer a suite label from testset IDs when rows omit an explicit suite."""

    row_id = str(row.get("id") or "")
    prefix_map = {
        "retrieval_": "retrieval",
        "reranker_": "reranker",
        "intake_": "intake",
        "router_": "router",
        "filter_": "metadata_filter",
        "mt_": "multiturn",
        "struct_": "structured_output",
    }
    for prefix, suite in prefix_map.items():
        if row_id.startswith(prefix):
            return suite
    return "retrieval"


def get_existing_dataset(client: Client, dataset_name: str) -> Dataset | None:
    return next(client.list_datasets(dataset_name=dataset_name, limit=1), None)


def dataset_example_count(client: Client, dataset_name: str) -> int:
    return sum(1 for _ in client.list_examples(dataset_name=dataset_name))


def get_or_create_dataset(
    client: Client,
    dataset_name: str,
    rows: list[dict[str, Any]],
) -> Dataset:
    existing_dataset = get_existing_dataset(client, dataset_name)
    if existing_dataset is not None:
        example_count = dataset_example_count(client, dataset_name)
        if example_count > 0:
            print(
                f"[INFO] reusing existing dataset: {dataset_name} "
                f"({example_count} examples)"
            )
            return existing_dataset

        examples = build_examples(rows)
        client.create_examples(dataset_id=existing_dataset.id, examples=examples)
        print(f"[INFO] added examples to empty dataset: {len(examples)}")
        return existing_dataset

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Retrieval evaluation dataset for accident fault-ratio RAG.",
    )

    examples = build_examples(rows)
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"[INFO] uploaded examples: {len(examples)}")
    return dataset


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (AttributeError, TypeError, ValueError):
                pass
        if isinstance(value, str | int | float | bool) or value is None:
            sanitized[key] = value
        else:
            sanitized[key] = json.dumps(value, ensure_ascii=False)
    return sanitized


def serialize_document(document: Document, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "page_content": document.page_content[:CONTENT_PREVIEW_CHARS],
        "metadata": sanitize_metadata(dict(document.metadata)),
    }


def build_retrieval_target(
    loader_strategy: str,
    embedding_provider: str,
    chunker_strategy: str,
    retriever_strategy: str,
    reranker_strategy: str,
    k: int,
    candidate_k: int | None = None,
    ensemble_bm25_weight: float = DEFAULT_ENSEMBLE_BM25_WEIGHT,
    ensemble_candidate_k: int = DEFAULT_ENSEMBLE_CANDIDATE_K,
    ensemble_use_chunk_id: bool = DEFAULT_ENSEMBLE_USE_CHUNK_ID,
):
    vectorstore_dir = get_vectorstore_dir(
        loader_strategy,
        embedding_provider,
        chunker_strategy=chunker_strategy,
    )
    if not vectorstore_exists(vectorstore_dir):
        raise RuntimeError(
            f"Vectorstore does not exist or is empty: {vectorstore_dir}. "
            "Build it before running retrieval evaluation."
        )

    vectorstore = load_vectorstore(vectorstore_dir, embedding_provider=embedding_provider)
    components = build_retrieval_components(vectorstore)

    def retrieval_target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs["question"]
        final_k = positive_int_or_default(inputs.get("final_k"), k)
        effective_candidate = positive_int_or_default(
            inputs.get("candidate_k"),
            candidate_k,
        )
        streamlit_config = build_pipeline_config(
            retriever_strategy=retriever_strategy,
            ensemble_bm25_weight=ensemble_bm25_weight,
            ensemble_candidate_k=ensemble_candidate_k,
            ensemble_use_chunk_id=ensemble_use_chunk_id,
            reranker_strategy=reranker_strategy,
        )
        pipeline_config = replace(
            streamlit_config,
            final_k=final_k,
            candidate_k=(
                effective_candidate
                if effective_candidate is not None
                else streamlit_config.candidate_k
            ),
        )
        documents = run_retrieval_pipeline(
            components=components,
            query=question,
            pipeline_config=pipeline_config,
        )
        retrieved = [
            serialize_document(document, rank)
            for rank, document in enumerate(documents, start=1)
        ]
        return {
            "query": question,
            "loader_strategy": loader_strategy,
            "embedding_provider": embedding_provider,
            "chunker_strategy": chunker_strategy,
            "retriever_strategy": retriever_strategy,
            "reranker_strategy": reranker_strategy,
            "k": final_k,
            "candidate_k": effective_candidate,
            "ensemble_bm25_weight": ensemble_bm25_weight,
            "ensemble_candidate_k": ensemble_candidate_k,
            "ensemble_use_chunk_id": ensemble_use_chunk_id,
            "retrieved": retrieved,
            "retrieved_metadata": [item["metadata"] for item in retrieved],
            "contexts": [item["page_content"] for item in retrieved],
        }

    return retrieval_target


def positive_int_or_default(value: Any, default: int | None) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _metadata_values(outputs: dict[str, Any], key: str) -> list[Any]:
    return [
        metadata.get(key)
        for metadata in outputs.get("retrieved_metadata", [])
        if isinstance(metadata, dict)
    ]


def _expected_list(reference_outputs: dict[str, Any], key: str) -> list[Any]:
    value = reference_outputs.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def diagram_id_hit(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    expected = expected_diagram_candidates(reference_outputs)
    actual_values = _metadata_values(outputs, "diagram_id")
    actual = set(actual_values)
    if not expected:
        hit = bool(outputs.get("retrieved"))
        comment = "General query; non-empty retrieval result is accepted."
    else:
        hit = bool(expected & actual)
        comment = f"expected_or_acceptable={sorted(expected)}, actual_topk={actual_values}"
    return {"key": "diagram_id_hit", "score": int(hit), "comment": comment}


def expected_diagram_candidates(reference_outputs: dict[str, Any]) -> set[Any]:
    expected = set(_expected_list(reference_outputs, "expected_diagram_ids"))
    expected.update(_expected_list(reference_outputs, "acceptable_diagram_ids"))
    return expected


def _first_rank(values: list[Any], candidates: set[Any]) -> int | None:
    for index, value in enumerate(values, start=1):
        if value in candidates:
            return index
    return None


def near_miss_not_above_expected(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    expected = set(_expected_list(reference_outputs, "expected_diagram_ids"))
    near_miss = set(_expected_list(reference_outputs, "near_miss_diagram_ids"))
    actual_values = _metadata_values(outputs, "diagram_id")
    if not expected or not near_miss:
        return {
            "key": "near_miss_not_above_expected",
            "score": None,
            "comment": "No expected or near-miss diagram IDs.",
        }

    expected_rank = _first_rank(actual_values, expected)
    near_miss_rank = _first_rank(actual_values, near_miss)
    if expected_rank is None:
        score = 0
    elif near_miss_rank is None:
        score = 1
    else:
        score = int(expected_rank < near_miss_rank)
    return {
        "key": "near_miss_not_above_expected",
        "score": score,
        "comment": (
            f"expected_rank={expected_rank}, near_miss_rank={near_miss_rank}, "
            f"actual_topk={actual_values}"
        ),
    }


def location_match(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    expected = reference_outputs.get("expected_location")
    if expected is None:
        return {"key": "location_match", "score": 1, "comment": "No expected location."}

    actual = _metadata_values(outputs, "location")
    matched = expected in actual
    return {
        "key": "location_match",
        "score": int(matched),
        "comment": f"expected={expected}, actual_topk={actual}",
    }


def party_type_match(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    expected = reference_outputs.get("expected_party_type")
    if expected is None:
        return {"key": "party_type_match", "score": 1, "comment": "No expected party_type."}

    actual = _metadata_values(outputs, "party_type")
    matched = expected in actual
    return {
        "key": "party_type_match",
        "score": int(matched),
        "comment": f"expected={expected}, actual_topk={actual}",
    }


def chunk_type_match(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    expected = set(_expected_list(reference_outputs, "expected_chunk_types"))
    if not expected:
        return {"key": "chunk_type_match", "score": 1, "comment": "No expected chunk_type."}

    actual = set(_metadata_values(outputs, "chunk_type"))
    matched = bool(expected & actual)
    return {
        "key": "chunk_type_match",
        "score": int(matched),
        "comment": f"expected={sorted(expected)}, actual_topk={list(actual)}",
    }


def keyword_coverage(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    keywords = [str(value) for value in _expected_list(reference_outputs, "expected_keywords")]
    if not keywords:
        return {"key": "keyword_coverage", "score": None, "comment": "No expected keywords."}

    haystack = json.dumps(outputs.get("retrieved", []), ensure_ascii=False)
    matched = [keyword for keyword in keywords if keyword in haystack]
    score = len(matched) / len(keywords)
    return {
        "key": "keyword_coverage",
        "score": score,
        "comment": f"matched={matched}, total={len(keywords)}",
    }


def retrieval_relevance(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    checks = [
        diagram_id_hit(outputs, reference_outputs)["score"],
        location_match(outputs, reference_outputs)["score"],
        party_type_match(outputs, reference_outputs)["score"],
        chunk_type_match(outputs, reference_outputs)["score"],
    ]
    keyword_score = keyword_coverage(outputs, reference_outputs)["score"]
    numeric_scores = [float(score) for score in checks if score is not None]
    if keyword_score is not None:
        numeric_scores.append(float(keyword_score))

    score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else None
    return {
        "key": "retrieval_relevance",
        "score": score,
        "comment": "Average of diagram/location/party/chunk/keyword retrieval checks.",
    }


def critical_error(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    expected_diagram_ids = expected_diagram_candidates(reference_outputs)
    diagram_score = diagram_id_hit(outputs, reference_outputs)["score"]
    party_score = party_type_match(outputs, reference_outputs)["score"]
    location_score = location_match(outputs, reference_outputs)["score"]

    is_error = bool(
        (expected_diagram_ids and not diagram_score)
        or not party_score
        or not location_score
    )
    return {
        "key": "critical_error",
        "score": int(is_error),
        "comment": "1 means a critical retrieval mismatch was detected.",
    }


def build_evaluators() -> list:
    return [
        diagram_id_hit,
        location_match,
        party_type_match,
        chunk_type_match,
        keyword_coverage,
        near_miss_not_above_expected,
        retrieval_relevance,
        critical_error,
    ]


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
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = safe_filename(
        f"{timestamp}-{run['name']}-{retriever_strategy}-{reranker_strategy}"
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
        "final_k": final_k,
        "candidate_k": candidate_k,
        "ensemble_bm25_weight": ensemble_bm25_weight,
        "ensemble_candidate_k": ensemble_candidate_k,
        "ensemble_use_chunk_id": ensemble_use_chunk_id,
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
    )


def run_retrieval_experiment(
    args: argparse.Namespace,
    client: Client | None,
    rows: list[dict[str, Any]],
    run: dict[str, str],
    matrix_mode: bool,
) -> None:
    dataset_name = dataset_name_for_run(args, run, matrix_mode)
    vectorstore_dir = get_vectorstore_dir(
        run["loader_strategy"],
        run["embedding_provider"],
        chunker_strategy=run["chunker_strategy"],
    )
    vectorstore_ready = args.upload_only or vectorstore_exists(vectorstore_dir)
    if not vectorstore_ready:
        if matrix_mode and not args.fail_on_missing_vectorstore:
            print(f"[SKIP] {run['name']}: vectorstore not found: {vectorstore_dir}")
            return
        raise RuntimeError(
            f"Vectorstore does not exist or is empty: {vectorstore_dir}. "
            "Build it before running retrieval evaluation."
        )

    print(f"[INFO] testset: {args.testset_path}")
    print(f"[INFO] examples: {len(rows)}")
    print(f"[INFO] run: {run['name']}")
    print(f"[INFO] vectorstore: {vectorstore_dir}")
    print(f"[INFO] loader_strategy: {run['loader_strategy']}")
    print(f"[INFO] embedding_provider: {run['embedding_provider']}")
    print(f"[INFO] chunker_strategy: {run['chunker_strategy']}")
    print(f"[INFO] retriever_strategy: {args.retriever_strategy}")
    print(f"[INFO] reranker_strategy: {args.reranker_strategy}")
    print(f"[INFO] k: {args.k}")
    ensemble_bm25_weight = ensemble_bm25_weight_from_args(args)
    ensemble_candidate_k = ensemble_candidate_k_from_args(args)
    ensemble_use_chunk_id = ensemble_use_chunk_id_from_args(args)
    print(f"[INFO] ensemble_bm25_weight: {ensemble_bm25_weight}")
    print(f"[INFO] ensemble_candidate_k: {ensemble_candidate_k}")
    print(f"[INFO] ensemble_use_chunk_id: {ensemble_use_chunk_id}")
    candidate_k = effective_candidate_k(args.reranker_strategy, args.candidate_k)
    if candidate_k is not None:
        print(f"[INFO] candidate_k: {candidate_k}")

    if not args.langsmith and not args.upload_only:
        target = build_retrieval_target(
            loader_strategy=run["loader_strategy"],
            embedding_provider=run["embedding_provider"],
            chunker_strategy=run["chunker_strategy"],
            retriever_strategy=args.retriever_strategy,
            reranker_strategy=args.reranker_strategy,
            k=args.k,
            candidate_k=candidate_k,
            ensemble_bm25_weight=ensemble_bm25_weight,
            ensemble_candidate_k=ensemble_candidate_k,
            ensemble_use_chunk_id=ensemble_use_chunk_id,
        )
        dataframe = evaluate_local_rows(
            rows=rows,
            target=target,
            evaluators=build_evaluators(),
            max_concurrency=args.max_concurrency,
        )
        experiment_name = (
            f"{DEFAULT_EXPERIMENT_PREFIX} - "
            f"{run['name']}-{args.retriever_strategy}-{args.reranker_strategy}-local"
        )
        dataset_name = dataset_name_for_run(args, run, matrix_mode)
        saved_paths = save_experiment_dataframe(
            dataframe=dataframe,
            experiment_name=experiment_name,
            output_dir=args.output_dir,
            run=run,
            dataset_name=dataset_name,
            testset_path=args.testset_path,
            retriever_strategy=args.retriever_strategy,
            reranker_strategy=args.reranker_strategy,
            final_k=args.k,
            candidate_k=candidate_k,
            ensemble_bm25_weight=ensemble_bm25_weight,
            ensemble_candidate_k=ensemble_candidate_k,
            ensemble_use_chunk_id=ensemble_use_chunk_id,
        )
        print("[INFO] LangSmith skipped. Local result files were written.")
        print(f"[INFO] local CSV: {saved_paths['csv']}")
        print(f"[INFO] local summary: {saved_paths['summary_json']}")
        return

    if client is None:
        raise RuntimeError("LangSmith client is required for --langsmith or --upload-only.")

    dataset = get_or_create_dataset(client, dataset_name, rows)
    print(f"[INFO] LangSmith dataset: {dataset.name}")

    if args.upload_only:
        print("[INFO] Upload complete. Retrieval evaluation skipped.")
        return

    target = build_retrieval_target(
        loader_strategy=run["loader_strategy"],
        embedding_provider=run["embedding_provider"],
        chunker_strategy=run["chunker_strategy"],
        retriever_strategy=args.retriever_strategy,
        reranker_strategy=args.reranker_strategy,
        k=args.k,
        candidate_k=candidate_k,
        ensemble_bm25_weight=ensemble_bm25_weight,
        ensemble_candidate_k=ensemble_candidate_k,
        ensemble_use_chunk_id=ensemble_use_chunk_id,
    )

    experiment_prefix = (
        f"{DEFAULT_EXPERIMENT_PREFIX} - "
        f"{run['name']}-{args.retriever_strategy}-{args.reranker_strategy}"
    )
    results = evaluate(
        data=dataset.name,
        evaluators=build_evaluators(),
        experiment_prefix=experiment_prefix,
        description=(
            "Local Chroma retrieval evaluation using expected diagram/location/"
            "party/chunk metadata."
        ),
        max_concurrency=args.max_concurrency,
        client=client,
        metadata={
            "matrix_run": run["name"],
            "loader_strategy": run["loader_strategy"],
            "embedding_provider": run["embedding_provider"],
            "chunker_strategy": run["chunker_strategy"],
            "retriever_strategy": args.retriever_strategy,
            "reranker_strategy": args.reranker_strategy,
            "k": args.k,
            "candidate_k": candidate_k,
            "ensemble_bm25_weight": ensemble_bm25_weight,
            "ensemble_candidate_k": ensemble_candidate_k,
            "ensemble_use_chunk_id": ensemble_use_chunk_id,
        },
    )
    print(f"[INFO] LangSmith experiment: {results.experiment_name}")
    if not getattr(args, "no_local_results", False):
        saved_paths = save_experiment_results(
            results=results,
            output_dir=args.output_dir,
            run=run,
            dataset_name=dataset.name,
            testset_path=args.testset_path,
            retriever_strategy=args.retriever_strategy,
            reranker_strategy=args.reranker_strategy,
            final_k=args.k,
            candidate_k=candidate_k,
            ensemble_bm25_weight=ensemble_bm25_weight,
            ensemble_candidate_k=ensemble_candidate_k,
            ensemble_use_chunk_id=ensemble_use_chunk_id,
        )
        print(f"[INFO] local CSV: {saved_paths['csv']}")
        print(f"[INFO] local summary: {saved_paths['summary_json']}")


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.list_matrix:
        print_matrix(load_eval_matrix(args.matrix_path))
        return

    needs_langsmith = bool(args.langsmith or args.upload_only)
    configure_tracing(needs_langsmith)
    if needs_langsmith and not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError("LANGSMITH_API_KEY is required.")
    validate_run_args(args)

    rows = load_jsonl(args.testset_path, args.max_examples)
    client = Client() if needs_langsmith else None
    if should_run_matrix(args):
        matrix = load_eval_matrix(args.matrix_path)
        runs = resolve_matrix_runs(matrix, args.preset)
    else:
        runs = [single_run_from_args(args)]

    execution_plan = build_execution_plan(args, runs)
    matrix_mode = len(execution_plan) > 1

    for index, (run, strategy_combination) in enumerate(execution_plan, start=1):
        run_args = args_with_strategy(args, strategy_combination)
        if matrix_mode:
            print(
                f"[MATRIX] {index}/{len(execution_plan)} "
                f"{run['name']} / {run_args.retriever_strategy} / {run_args.reranker_strategy}"
            )
        run_retrieval_experiment(
            args=run_args,
            client=client,
            rows=rows,
            run=run,
            matrix_mode=matrix_mode,
        )


if __name__ == "__main__":
    main()
