"""Evaluate raw, normalized, and hybrid intake queries for retrieval candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BASE_DIR, DEFAULT_EMBEDDING_PROVIDER, get_vectorstore_dir
from evaluation.evaluate_retrieval_langsmith import (
    build_evaluators,
    build_examples,
    evaluate_local_rows,
    save_experiment_dataframe,
    serialize_document,
)
from rag.indexer import load_vectorstore, vectorstore_exists
from rag.pipeline.retrieval import RetrievalPipelineConfig, run_retrieval_pipeline
from rag.pipeline.retriever import build_retrieval_components
from rag.service.intake.filter_service import build_metadata_filters
from rag.service.intake.intake_service import evaluate_input_sufficiency
from rag.service.intake.query_normalizer import normalize_retrieval_query_terms
from rag.service.intake.schema import QuerySlots, UserSearchMetadata


DEFAULT_TESTSET_PATH = BASE_DIR / "data" / "testsets" / "langsmith" / "retrieval_eval.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "evaluation" / "results" / "intake_query_normalization"
DEFAULT_DECISION_CACHE = DEFAULT_OUTPUT_DIR / "intake_decisions_retrieval_eval.json"
DEFAULT_LOADER_STRATEGY = "llamaparser"
DEFAULT_CHUNKER_STRATEGY = "fixed"
DEFAULT_RETRIEVER_STRATEGY = "vectorstore"
DEFAULT_RERANKER_STRATEGY = "none"
DEFAULT_TOP_K = 30
HYBRID_CONDITION = "hybrid_rrf_with_intake_filter"
CONDITIONS = (
    "raw_no_filter",
    "raw_with_intake_filter",
    "normalized_no_filter",
    "normalized_with_intake_filter",
    HYBRID_CONDITION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate raw vs normalized vs hybrid intake retrieval queries locally.",
    )
    parser.add_argument("--testset-path", type=Path, default=DEFAULT_TESTSET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--decision-cache", type=Path, default=DEFAULT_DECISION_CACHE)
    parser.add_argument("--loader-strategy", default=DEFAULT_LOADER_STRATEGY)
    parser.add_argument("--chunker-strategy", default=DEFAULT_CHUNKER_STRATEGY)
    parser.add_argument("--embedding-provider", default=DEFAULT_EMBEDDING_PROVIDER)
    parser.add_argument("--retriever-strategy", default=DEFAULT_RETRIEVER_STRATEGY)
    parser.add_argument("--reranker-strategy", default=DEFAULT_RERANKER_STRATEGY)
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help=f"Comma-separated conditions to run. Available: {', '.join(CONDITIONS)}",
    )
    parser.add_argument(
        "--refresh-intake",
        action="store_true",
        help="Regenerate intake decisions instead of reusing the decision cache.",
    )
    return parser.parse_args()


def configure_local_tracing() -> None:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"


def load_jsonl(path: Path, max_examples: int = 0) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_examples > 0:
        return rows[:max_examples]
    return rows


def load_decision_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_decision_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def decision_for_row(
    row: dict[str, Any],
    cache: dict[str, Any],
    cache_path: Path,
    *,
    refresh_intake: bool,
) -> dict[str, Any]:
    row_id = str(row["id"])
    if not refresh_intake and row_id in cache:
        return cache[row_id]

    decision = evaluate_input_sufficiency(str(row["question"]))
    serialized = {
        "is_sufficient": decision.is_sufficient,
        "normalized_description": decision.normalized_description,
        "search_metadata": asdict(decision.search_metadata),
        "confidence": decision.confidence,
        "missing_fields": [asdict(field) for field in decision.missing_fields],
        "follow_up_questions": decision.follow_up_questions,
    }
    cache[row_id] = serialized
    save_decision_cache(cache_path, cache)
    return serialized


def none_if_null(value: object) -> Any:
    if value in (None, "", "null", "None"):
        return None
    return value


def metadata_from_decision(decision: dict[str, Any]) -> UserSearchMetadata:
    metadata = decision["search_metadata"]
    slots = metadata.get("query_slots") or {}
    return UserSearchMetadata(
        party_type=none_if_null(metadata.get("party_type")),
        location=none_if_null(metadata.get("location")),
        retrieval_query=none_if_null(metadata.get("retrieval_query")),
        query_slots=QuerySlots(
            **{
                field_name: none_if_null(slots.get(field_name))
                for field_name in QuerySlots.__dataclass_fields__
            }
        ),
    )


def document_key(document: Document) -> tuple[str, Any]:
    metadata = dict(document.metadata)
    if metadata.get("chunk_id") is not None:
        return ("chunk_id", str(metadata["chunk_id"]))
    values = (
        metadata.get("source"),
        metadata.get("page"),
        metadata.get("chunk_index"),
        metadata.get("diagram_id"),
    )
    if any(value is not None for value in values):
        return ("metadata", tuple(str(value) for value in values))
    digest = hashlib.sha1(document.page_content.encode("utf-8", errors="ignore")).hexdigest()
    return ("content", digest)


def rrf_merge(
    ranked_lists: list[list[Document]],
    *,
    k: int,
    rrf_k: int,
) -> list[Document]:
    documents_by_key: dict[tuple[str, Any], Document] = {}
    scores: dict[tuple[str, Any], float] = {}
    best_ranks: dict[tuple[str, Any], int] = {}
    for documents in ranked_lists:
        for rank, document in enumerate(documents, start=1):
            key = document_key(document)
            documents_by_key.setdefault(key, document)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            best_ranks[key] = min(best_ranks.get(key, rank), rank)

    ordered_keys = sorted(scores, key=lambda key: (-scores[key], best_ranks[key]))
    return [documents_by_key[key] for key in ordered_keys[:k]]


def make_retrieval_target(
    *,
    rows: list[dict[str, Any]],
    components,
    pipeline_config: RetrievalPipelineConfig,
    decision_cache: dict[str, Any],
    decision_cache_path: Path,
    condition: str,
    refresh_intake: bool,
    rrf_k: int,
):
    rows_by_question = {str(row["question"]): row for row in rows}

    def retrieve(query: str, filters: dict[str, object] | None) -> list[Document]:
        return run_retrieval_pipeline(
            components,
            query,
            filters=filters,
            pipeline_config=pipeline_config,
        )

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        row = rows_by_question[str(inputs["question"])]
        decision = decision_for_row(
            row,
            decision_cache,
            decision_cache_path,
            refresh_intake=refresh_intake,
        )
        metadata = metadata_from_decision(decision)
        raw_query = str(inputs["question"])
        normalized_query = normalize_retrieval_query_terms(raw_query, metadata) or raw_query
        filters = (
            build_metadata_filters(metadata)
            if condition
            in {
                "raw_with_intake_filter",
                "normalized_with_intake_filter",
                HYBRID_CONDITION,
            }
            else None
        )

        if condition == "raw_no_filter":
            query = raw_query
            documents = retrieve(raw_query, None)
        elif condition == "raw_with_intake_filter":
            query = raw_query
            documents = retrieve(raw_query, filters)
        elif condition == "normalized_no_filter":
            query = normalized_query
            documents = retrieve(normalized_query, None)
        elif condition == "normalized_with_intake_filter":
            query = normalized_query
            documents = retrieve(normalized_query, filters)
        elif condition == HYBRID_CONDITION:
            query = f"{normalized_query} || {raw_query}"
            documents = rrf_merge(
                [retrieve(normalized_query, filters), retrieve(raw_query, filters)],
                k=pipeline_config.final_k,
                rrf_k=rrf_k,
            )
        else:
            raise ValueError(f"Unknown condition: {condition}")

        return {
            "query": query,
            "filters": filters,
            "k": pipeline_config.final_k,
            "retrieved": [
                serialize_document(document, rank)
                for rank, document in enumerate(documents, start=1)
            ],
            "retrieved_metadata": [dict(document.metadata) for document in documents],
            "contexts": [document.page_content for document in documents],
        }

    return target


def resolve_conditions(raw_conditions: str) -> list[str]:
    selected = [condition.strip() for condition in raw_conditions.split(",") if condition.strip()]
    unknown = [condition for condition in selected if condition not in CONDITIONS]
    if unknown:
        raise ValueError(f"Unknown condition(s): {', '.join(unknown)}")
    return selected


def summarize_condition_deltas(summary: dict[str, Any]) -> None:
    def delta(after: str, before: str) -> dict[str, float]:
        return {
            metric: summary[after]["metrics"][metric] - summary[before]["metrics"][metric]
            for metric in summary[after]["metrics"]
            if metric in summary[before]["metrics"]
        }

    if "normalized_with_intake_filter" in summary and "raw_with_intake_filter" in summary:
        summary["delta_normalization_only"] = delta(
            "normalized_with_intake_filter",
            "raw_with_intake_filter",
        )
    if HYBRID_CONDITION in summary and "normalized_with_intake_filter" in summary:
        summary["delta_hybrid_vs_normalized"] = delta(
            HYBRID_CONDITION,
            "normalized_with_intake_filter",
        )
    if HYBRID_CONDITION in summary and "raw_with_intake_filter" in summary:
        summary["delta_hybrid_vs_raw_filter"] = delta(
            HYBRID_CONDITION,
            "raw_with_intake_filter",
        )


def main() -> None:
    load_dotenv()
    configure_local_tracing()
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be greater than 0")
    if args.rrf_k <= 0:
        raise ValueError("--rrf-k must be greater than 0")

    rows = load_jsonl(args.testset_path, args.max_examples)
    conditions = resolve_conditions(args.conditions)
    vectorstore_dir = get_vectorstore_dir(
        args.loader_strategy,
        args.embedding_provider,
        chunker_strategy=args.chunker_strategy,
    )
    if not vectorstore_exists(vectorstore_dir):
        raise RuntimeError(f"Vectorstore does not exist or is empty: {vectorstore_dir}")

    vectorstore = load_vectorstore(vectorstore_dir, embedding_provider=args.embedding_provider)
    components = build_retrieval_components(vectorstore)
    pipeline_config = RetrievalPipelineConfig(
        retriever_strategy=args.retriever_strategy,
        reranker_strategy=args.reranker_strategy,
        final_k=args.k,
    )
    decision_cache = load_decision_cache(args.decision_cache)
    evaluators = build_evaluators()

    frames: list[pd.DataFrame] = []
    for condition in conditions:
        target = make_retrieval_target(
            rows=rows,
            components=components,
            pipeline_config=pipeline_config,
            decision_cache=decision_cache,
            decision_cache_path=args.decision_cache,
            condition=condition,
            refresh_intake=args.refresh_intake,
            rrf_k=args.rrf_k,
        )
        frame = evaluate_local_rows(
            rows=rows,
            target=target,
            evaluators=evaluators,
            max_concurrency=args.max_concurrency,
        )
        frame["condition"] = condition
        frame["top_k"] = args.k
        frames.append(frame)

    dataframe = pd.concat(frames, ignore_index=True)
    run = {
        "name": f"{args.loader_strategy}-{args.chunker_strategy}-{args.embedding_provider}",
        "loader_strategy": args.loader_strategy,
        "chunker_strategy": args.chunker_strategy,
        "embedding_provider": args.embedding_provider,
    }
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    saved = save_experiment_dataframe(
        dataframe=dataframe,
        experiment_name=f"MDM hybrid query eval - {run['name']}-top{args.k}-{timestamp}",
        output_dir=args.output_dir,
        run=run,
        dataset_name=f"MDM retrieval testset - {args.testset_path.stem}",
        testset_path=args.testset_path,
        retriever_strategy=args.retriever_strategy,
        reranker_strategy=args.reranker_strategy,
        final_k=args.k,
        candidate_k=None,
    )

    summary = json.loads(saved["summary_json"].read_text(encoding="utf-8"))
    condition_summaries: dict[str, Any] = {}
    for condition, group in dataframe.groupby("condition"):
        metrics = {}
        for column in group.columns:
            if str(column).startswith("feedback.") and not str(column).endswith(".comment"):
                metrics[str(column).removeprefix("feedback.")] = float(
                    pd.to_numeric(group[column], errors="coerce").mean()
                )
        condition_summaries[condition] = {
            "row_count": int(len(group)),
            "top_k": args.k,
            "metrics": metrics,
        }
    summarize_condition_deltas(condition_summaries)
    summary["conditions"] = condition_summaries
    saved["summary_json"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[INFO] local CSV: {saved['csv']}")
    print(f"[INFO] local summary: {saved['summary_json']}")
    print(json.dumps(condition_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
