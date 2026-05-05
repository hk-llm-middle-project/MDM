"""Evaluate local retrieval results on a LangSmith dataset.

Compatibility entry point for the retrieval evaluation CLI. Most reusable
helpers live under :mod:`evaluation.retrieval_eval`.
"""

from __future__ import annotations

import os
import sys
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ["ANONYMIZED_TELEMETRY"] = "False"

from dotenv import load_dotenv
from langchain_core.documents import Document
from langsmith import Client, evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_vectorstore_dir
from main import build_pipeline_config
from rag.indexer import load_vectorstore, vectorstore_exists
from rag.pipeline.retrieval import run_retrieval_pipeline
from rag.pipeline.retriever import build_retrieval_components
from rag.pipeline.retriever.common import get_embedding_function_from_vectorstore
from rag.service.analysis.answer_schema import AnalysisResult, RetrievedContext
from rag.service.conversation.pipelines.accident_analysis import answer_accident_analysis
from rag.service.intake.filter_service import build_metadata_filters
from rag.service.intake.intake_service import evaluate_input_sufficiency
from rag.service.intake.schema import IntakeState
from rag.service.session.serialization import intake_state_to_dict

from evaluation.retrieval_eval.cli import (
    args_with_strategy,
    build_execution_plan,
    configure_tracing,
    effective_candidate_k,
    ensemble_bm25_weight_from_args,
    ensemble_candidate_k_from_args,
    ensemble_use_chunk_id_from_args,
    normalize_strategy_value,
    parse_args,
    resolve_strategy_combinations,
    resolve_strategy_values,
    should_run_matrix,
    validate_run_args,
)
from evaluation.retrieval_eval.constants import *  # noqa: F403 - compatibility re-export.
from evaluation.retrieval_eval.dataset import (
    build_examples,
    dataset_example_count,
    get_existing_dataset,
    get_or_create_dataset,
    infer_suite_from_row,
    load_jsonl,
    make_dataset_name,
)
from evaluation.retrieval_eval.evaluators import (
    build_evaluators,
    chunk_type_match,
    critical_error,
    diagram_id_hit,
    expected_diagram_candidates,
    keyword_coverage,
    location_match,
    near_miss_not_above_expected,
    party_type_match,
    retrieval_relevance,
)
from evaluation.retrieval_eval.local import evaluate_local_rows
from evaluation.retrieval_eval.matrix import load_eval_matrix, print_matrix, resolve_matrix_runs
from evaluation.retrieval_eval.results import (
    dataset_name_for_run,
    safe_filename,
    save_experiment_dataframe,
    save_experiment_results,
    single_run_from_args,
    summarize_embedding_query_cache,
    summarize_feedback_metrics,
)


logger = logging.getLogger(__name__)


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


def _embedding_cache_counts(embedding_function: Any | None) -> tuple[int, int]:
    if embedding_function is None:
        return (0, 0)
    return (
        int(getattr(embedding_function, "query_cache_hits", 0) or 0),
        int(getattr(embedding_function, "query_cache_misses", 0) or 0),
    )


def _embedding_query_cache_metadata(
    embedding_function: Any | None,
    before: tuple[int, int],
    *,
    embedding_provider: str,
) -> dict[str, Any]:
    hits_before, misses_before = before
    hits_after, misses_after = _embedding_cache_counts(embedding_function)
    cache_hits = hits_after - hits_before
    cache_misses = misses_after - misses_before
    cache_enabled = bool(getattr(embedding_function, "enabled", False))
    cache_hit = cache_hits > 0 or bool(
        getattr(embedding_function, "last_query_cache_hit", False)
    )
    if cache_enabled:
        logger.info(
            "[embedding-query-cache] %s provider=%s hits=%s misses=%s",
            "hit" if cache_hit else "miss",
            embedding_provider,
            cache_hits,
            cache_misses,
        )
    return {
        "embedding_query_cache_enabled": cache_enabled,
        "embedding_query_cache_hit": cache_hit,
        "embedding_query_cache_hits": cache_hits,
        "embedding_query_cache_misses": cache_misses,
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
    retrieval_input_mode: str = DEFAULT_RETRIEVAL_INPUT_MODE,
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
    try:
        embedding_function = get_embedding_function_from_vectorstore(vectorstore)
    except ValueError:
        embedding_function = None

    def pipeline_config_for_inputs(inputs: dict[str, Any]):
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
        return (
            replace(
                streamlit_config,
                final_k=final_k,
                candidate_k=(
                    effective_candidate
                    if effective_candidate is not None
                    else streamlit_config.candidate_k
                ),
            ),
            final_k,
            effective_candidate,
        )

    def raw_retrieval_target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs["question"]
        pipeline_config, final_k, effective_candidate = pipeline_config_for_inputs(inputs)
        cache_counts_before = _embedding_cache_counts(embedding_function)
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
            "retrieval_input_mode": "raw",
            "k": final_k,
            "candidate_k": effective_candidate,
            "ensemble_bm25_weight": ensemble_bm25_weight,
            "ensemble_candidate_k": ensemble_candidate_k,
            "ensemble_use_chunk_id": ensemble_use_chunk_id,
            "retrieved": retrieved,
            "retrieved_metadata": [item["metadata"] for item in retrieved],
            "contexts": [item["page_content"] for item in retrieved],
            **_embedding_query_cache_metadata(
                embedding_function,
                cache_counts_before,
                embedding_provider=embedding_provider,
            ),
        }

    def intake_retrieval_target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = str(inputs["question"])
        pipeline_config, final_k, effective_candidate = pipeline_config_for_inputs(inputs)
        analysis_observation: dict[str, Any] = {}

        def retrieval_only_analyzer(
            analysis_question: str,
            *,
            search_metadata=None,
            pipeline_config=None,
            **_: Any,
        ) -> AnalysisResult:
            filters = build_metadata_filters(search_metadata)
            retrieval_query = (
                search_metadata.retrieval_query
                if search_metadata is not None and search_metadata.retrieval_query
                else analysis_question
            )
            documents = run_retrieval_pipeline(
                components=components,
                query=retrieval_query,
                filters=filters,
                pipeline_config=pipeline_config,
            )
            retrieved_contexts = [
                RetrievedContext(
                    content=document.page_content,
                    metadata=dict(document.metadata),
                )
                for document in documents
            ]
            analysis_observation.update(
                {
                    "query": retrieval_query,
                    "analysis_question": analysis_question,
                    "filters": filters,
                    "search_metadata": (
                        intake_state_to_dict(IntakeState(search_metadata=search_metadata))[
                            "search_metadata"
                        ]
                        if search_metadata is not None
                        else {}
                    ),
                }
            )
            return AnalysisResult(
                response="",
                contexts=[context.content for context in retrieved_contexts],
                retrieved_contexts=retrieved_contexts,
            )

        cache_counts_before = _embedding_cache_counts(embedding_function)
        result = answer_accident_analysis(
            question,
            pipeline_config=pipeline_config,
            intake_state=IntakeState(),
            loader_strategy=loader_strategy,
            chunker_strategy=chunker_strategy,
            embedding_provider=embedding_provider,
            chat_history=None,
            intake_evaluator=evaluate_input_sufficiency,
            analyzer=retrieval_only_analyzer,
        )
        cache_metadata = _embedding_query_cache_metadata(
            embedding_function,
            cache_counts_before,
            embedding_provider=embedding_provider,
        )
        retrieved = [
            {
                "rank": rank,
                "page_content": context.content[:CONTENT_PREVIEW_CHARS],
                "metadata": sanitize_metadata(dict(context.metadata)),
            }
            for rank, context in enumerate(result.retrieved_contexts, start=1)
        ]
        intake_state = intake_state_to_dict(result.intake_state)
        return {
            "query": analysis_observation.get("query") or question,
            "raw_query": question,
            "intake_analysis_question": analysis_observation.get("analysis_question"),
            "intake_filters": analysis_observation.get("filters"),
            "intake_search_metadata": analysis_observation.get("search_metadata")
            or intake_state.get("search_metadata"),
            "intake_state": intake_state,
            "intake_needs_more_input": result.needs_more_input,
            "intake_result_type": getattr(result.result_type, "value", str(result.result_type)),
            "intake_missing_fields": result.intake_state.last_missing_fields,
            "intake_follow_up_questions": result.intake_state.last_follow_up_questions,
            "loader_strategy": loader_strategy,
            "embedding_provider": embedding_provider,
            "chunker_strategy": chunker_strategy,
            "retriever_strategy": retriever_strategy,
            "reranker_strategy": reranker_strategy,
            "retrieval_input_mode": "intake",
            "k": final_k,
            "candidate_k": effective_candidate,
            "ensemble_bm25_weight": ensemble_bm25_weight,
            "ensemble_candidate_k": ensemble_candidate_k,
            "ensemble_use_chunk_id": ensemble_use_chunk_id,
            "retrieved": retrieved,
            "retrieved_metadata": [item["metadata"] for item in retrieved],
            "contexts": [item["page_content"] for item in retrieved],
            **cache_metadata,
        }

    if retrieval_input_mode == "intake":
        return intake_retrieval_target
    return raw_retrieval_target


def positive_int_or_default(value: Any, default: int | None) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


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
    retrieval_input_mode = getattr(args, "retrieval_input_mode", DEFAULT_RETRIEVAL_INPUT_MODE)
    print(f"[INFO] retrieval_input_mode: {retrieval_input_mode}")
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
            retrieval_input_mode=retrieval_input_mode,
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
            retrieval_input_mode=retrieval_input_mode,
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
        retrieval_input_mode=retrieval_input_mode,
    )

    experiment_prefix = (
        f"{DEFAULT_EXPERIMENT_PREFIX} - "
        f"{run['name']}-{args.retriever_strategy}-{args.reranker_strategy}"
    )
    results = evaluate(
        target,
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
            "retrieval_input_mode": retrieval_input_mode,
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
            retrieval_input_mode=retrieval_input_mode,
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
