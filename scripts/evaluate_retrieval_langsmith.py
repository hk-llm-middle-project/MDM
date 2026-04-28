"""Evaluate local retrieval results on a LangSmith dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langsmith import Client, evaluate
from langsmith.schemas import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BASE_DIR, DEFAULT_EMBEDDING_PROVIDER, INDEX_BATCH_SIZE, get_vectorstore_dir
from rag.indexer import load_vectorstore, vectorstore_exists
from rag.pipeline.retrieval import RetrievalPipelineConfig, run_retrieval_pipeline
from rag.pipeline.retriever import build_retrieval_components


DEFAULT_TESTSET_PATH = (
    BASE_DIR / "data" / "testsets" / "upstage_retrieval_v0.1.0_20260428.jsonl"
)
DEFAULT_DATASET_PREFIX = "MDM retrieval testset"
DEFAULT_EXPERIMENT_PREFIX = "MDM retrieval eval"
DEFAULT_LOADER_STRATEGY = "upstage"
DEFAULT_RETRIEVER_STRATEGY = "vectorstore"
DEFAULT_RERANKER_STRATEGY = "none"
DEFAULT_K = 5
CONTENT_PREVIEW_CHARS = 500


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
        help="LangSmith dataset name. Defaults to a stable name based on the testset file.",
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
        "--retriever-strategy",
        default=DEFAULT_RETRIEVER_STRATEGY,
        help="Retrieval strategy, e.g. vectorstore, ensemble, multiquery.",
    )
    parser.add_argument(
        "--reranker-strategy",
        default=DEFAULT_RERANKER_STRATEGY,
        help="Reranker strategy, e.g. none, flashrank, cohere, llm_score.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="Number of final retrieved documents to evaluate.",
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
    return parser.parse_args()


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


def make_dataset_name(path: Path, loader_strategy: str, embedding_provider: str) -> str:
    del loader_strategy, embedding_provider
    return f"{DEFAULT_DATASET_PREFIX} - {path.stem}"


def build_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        question = row.get("question")
        if not question:
            continue

        examples.append(
            {
                "inputs": {"question": question},
                "outputs": {
                    "reference": row.get("reference", ""),
                    "expected_diagram_ids": row.get("expected_diagram_ids", []),
                    "expected_party_type": row.get("expected_party_type"),
                    "expected_location": row.get("expected_location"),
                    "expected_chunk_types": row.get("expected_chunk_types", []),
                    "expected_keywords": row.get("expected_keywords", []),
                },
                "metadata": {
                    "id": row.get("id"),
                    "notes": row.get("notes"),
                },
            }
        )

    if not examples:
        raise RuntimeError("No valid examples with question were found.")
    return examples


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
    retriever_strategy: str,
    reranker_strategy: str,
    k: int,
):
    vectorstore_dir = get_vectorstore_dir(loader_strategy, embedding_provider)
    if not vectorstore_exists(vectorstore_dir):
        raise RuntimeError(
            f"Vectorstore does not exist or is empty: {vectorstore_dir}. "
            "Build it before running retrieval evaluation."
        )

    vectorstore = load_vectorstore(vectorstore_dir, embedding_provider=embedding_provider)
    components = build_retrieval_components(vectorstore)
    pipeline_config = RetrievalPipelineConfig(
        retriever_strategy=retriever_strategy,
        reranker_strategy=reranker_strategy,
        final_k=k,
    )

    def retrieval_target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs["question"]
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
            "retriever_strategy": retriever_strategy,
            "reranker_strategy": reranker_strategy,
            "k": k,
            "retrieved": retrieved,
            "retrieved_metadata": [item["metadata"] for item in retrieved],
            "contexts": [item["page_content"] for item in retrieved],
        }

    return retrieval_target


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
    expected = set(_expected_list(reference_outputs, "expected_diagram_ids"))
    actual = set(_metadata_values(outputs, "diagram_id"))
    if not expected:
        hit = bool(outputs.get("retrieved"))
        comment = "General query; non-empty retrieval result is accepted."
    else:
        hit = bool(expected & actual)
        comment = f"expected={sorted(expected)}, actual_topk={list(actual)}"
    return {"key": "diagram_id_hit", "score": int(hit), "comment": comment}


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
    expected_diagram_ids = _expected_list(reference_outputs, "expected_diagram_ids")
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
        retrieval_relevance,
        critical_error,
    ]


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError("LANGSMITH_API_KEY is required.")
    if args.k <= 0:
        raise ValueError("--k must be greater than 0")

    rows = load_jsonl(args.testset_path, args.max_examples)
    dataset_name = args.dataset_name or make_dataset_name(
        args.testset_path,
        args.loader_strategy,
        args.embedding_provider,
    )
    vectorstore_dir = get_vectorstore_dir(args.loader_strategy, args.embedding_provider)

    print(f"[INFO] testset: {args.testset_path}")
    print(f"[INFO] examples: {len(rows)}")
    print(f"[INFO] vectorstore: {vectorstore_dir}")
    print(f"[INFO] loader_strategy: {args.loader_strategy}")
    print(f"[INFO] embedding_provider: {args.embedding_provider}")
    print(f"[INFO] retriever_strategy: {args.retriever_strategy}")
    print(f"[INFO] reranker_strategy: {args.reranker_strategy}")
    print(f"[INFO] k: {args.k}")

    client = Client()
    dataset = get_or_create_dataset(client, dataset_name, rows)
    print(f"[INFO] LangSmith dataset: {dataset.name}")

    if args.upload_only:
        print("[INFO] Upload complete. Retrieval evaluation skipped.")
        return

    target = build_retrieval_target(
        loader_strategy=args.loader_strategy,
        embedding_provider=args.embedding_provider,
        retriever_strategy=args.retriever_strategy,
        reranker_strategy=args.reranker_strategy,
        k=args.k,
    )

    experiment_prefix = (
        f"{DEFAULT_EXPERIMENT_PREFIX} - "
        f"{args.loader_strategy}-{args.embedding_provider}-"
        f"{args.retriever_strategy}-{args.reranker_strategy}"
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
            "loader_strategy": args.loader_strategy,
            "embedding_provider": args.embedding_provider,
            "retriever_strategy": args.retriever_strategy,
            "reranker_strategy": args.reranker_strategy,
            "k": args.k,
        },
    )
    print(f"[INFO] LangSmith experiment: {results.experiment_name}")


if __name__ == "__main__":
    main()
