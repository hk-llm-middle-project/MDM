"""Debug BM25, dense, and ensemble retrieval without LangSmith or LLM calls.

Run from the project root, for example:

    .venv\Scripts\python.exe tests\debug_ensemble_retrieval.py ^
      --loader llamaparser ^
      --chunker case-boundary ^
      --bm25-weight 0.85 ^
      --dense-weight 0.15

The script prints each retriever's returned metadata and content preview so the
BM25/Dense contribution can be inspected separately from the final ensemble.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.pipeline.retriever.common import kiwi_tokenize  # noqa: E402
from rag.pipeline.retriever.strategies.ensemble import (  # noqa: E402
    DEFAULT_CANDIDATE_K_MULTIPLIER,
    EnsembleRetrieverConfig,
    MIN_ENSEMBLE_CANDIDATE_K,
    _metadata_matches_filter,
    retrieve_with_ensemble,
)
from rag.pipeline.retriever.strategies.ensemble_parent import (  # noqa: E402
    retrieve_with_ensemble_parent_documents,
)
from rag.pipeline.retriever.strategies.parent import (  # noqa: E402
    retrieve_with_parent_documents,
)
from rag.service.vectorstore.vectorstore_service import get_retrieval_components  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print BM25, dense, and ensemble retrieval results.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Search query text. If omitted, the script asks for it interactively.",
    )
    parser.add_argument("--loader", default="llamaparser", help="Loader strategy.")
    parser.add_argument("--chunker", default="case-boundary", help="Chunker strategy.")
    parser.add_argument("--embedding", default="openai", help="Embedding provider.")
    parser.add_argument(
        "--strategy",
        choices=("ensemble", "ensemble_parent", "parent"),
        default="ensemble",
        help="Final retrieval strategy to print after BM25/Dense baselines.",
    )
    parser.add_argument("--k", type=int, default=5, help="Final top-k to print.")
    parser.add_argument("--bm25-k", type=int, default=None, help="BM25 candidate k.")
    parser.add_argument("--dense-k", type=int, default=None, help="Dense candidate k.")
    parser.add_argument("--bm25-weight", type=float, default=0.85, help="BM25 weight.")
    parser.add_argument("--dense-weight", type=float, default=0.15, help="Dense weight.")
    parser.add_argument("--party-type", default="자동차", help="Optional party_type filter.")
    parser.add_argument("--location", default="교차로 사고", help="Optional location filter.")
    parser.add_argument("--target", default="차7-2", help="Expected diagram/chunk marker.")
    parser.add_argument(
        "--noise",
        nargs="*",
        default=["차1-1", "차12-1"],
        help="Unexpected diagram/chunk markers to count.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="Maximum characters to print for each document.",
    )
    args = parser.parse_args()
    if args.query is None:
        args.query = input("query: ").strip()
    if not args.query:
        parser.error("--query is required when no interactive query is entered.")
    return args


def build_filters(args: argparse.Namespace) -> dict[str, object] | None:
    filters: list[dict[str, object]] = []
    if args.party_type:
        filters.append({"party_type": args.party_type})
    if args.location:
        filters.append({"location": args.location})
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def filtered_documents(documents, filters: dict[str, object] | None):
    if not filters:
        return list(documents)
    return [
        document
        for document in documents
        if _metadata_matches_filter(document.metadata, filters)
    ]


def retrieve_bm25(documents, query: str, k: int):
    retriever = BM25Retriever.from_documents(
        documents,
        preprocess_func=kiwi_tokenize,
    )
    retriever.k = k
    return list(retriever.invoke(query))


def retrieve_dense(components, query: str, k: int, filters: dict[str, object] | None):
    search_kwargs: dict[str, object] = {"k": k}
    if filters:
        search_kwargs["filter"] = filters
    retriever = components.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )
    return list(retriever.invoke(query))


def retrieve_selected_strategy(
    components,
    query: str,
    k: int,
    filters: dict[str, object] | None,
    args: argparse.Namespace,
    bm25_k: int,
    dense_k: int,
):
    config = EnsembleRetrieverConfig(
        weights=(args.bm25_weight, args.dense_weight),
        bm25_k=bm25_k,
        dense_k=dense_k,
    )
    if args.strategy == "ensemble_parent":
        return retrieve_with_ensemble_parent_documents(
            components,
            query,
            k=k,
            filters=filters,
            strategy_config=config,
        )
    if args.strategy == "parent":
        return retrieve_with_parent_documents(
            components,
            query,
            k=k,
            filters=filters,
        )
    return retrieve_with_ensemble(
        components,
        query,
        k=k,
        filters=filters,
        strategy_config=config,
    )


def candidate_k(configured_k: int | None, final_k: int) -> int:
    if configured_k is not None:
        return configured_k
    return max(
        final_k * DEFAULT_CANDIDATE_K_MULTIPLIER,
        MIN_ENSEMBLE_CANDIDATE_K,
    )


def document_key(document) -> str:
    metadata = document.metadata
    for key in ("chunk_id", "diagram_id", "source", "page"):
        value = metadata.get(key)
        if value is not None:
            return f"{key}={value}"
    return document.page_content[:80].replace("\n", " ")


def marker_found(document, marker: str) -> bool:
    metadata_text = " ".join(str(value) for value in document.metadata.values())
    return marker in metadata_text or marker in document.page_content


def print_summary(name: str, documents, args: argparse.Namespace, filters) -> None:
    target_hits = sum(1 for document in documents if marker_found(document, args.target))
    noise_hits = {
        marker: sum(1 for document in documents if marker_found(document, marker))
        for marker in args.noise
    }
    filter_violations = (
        sum(
            1
            for document in documents
            if not _metadata_matches_filter(document.metadata, filters)
        )
        if filters
        else 0
    )
    keys = [document_key(document) for document in documents]

    print(f"\n=== {name} ===")
    print(f"returned count: {len(documents)}")
    print(f"target hits ({args.target}): {target_hits}")
    print(f"noise hits: {noise_hits}")
    print(f"filter violations: {filter_violations}")
    print(f"rank keys: {keys}")


def print_document(name: str, index: int, document, args: argparse.Namespace) -> None:
    metadata = dict(document.metadata)
    preview = document.page_content.strip()
    if len(preview) > args.preview_chars:
        preview = preview[: args.preview_chars].rstrip() + "\n...(truncated)"

    print(f"\n{name} [{index}] {document_key(document)}")
    for key in sorted(metadata):
        print(f"  {key}: {metadata[key]}")
    print("  preview:")
    print(preview)


def print_results(name: str, documents, args: argparse.Namespace, filters) -> None:
    print_summary(name, documents, args, filters)
    for index, document in enumerate(documents, start=1):
        print_document(name, index, document, args)


def main() -> None:
    load_dotenv()
    args = parse_args()
    filters = build_filters(args)
    components = get_retrieval_components(
        args.loader,
        args.embedding,
        args.chunker,
    )
    source_documents = components.get_source_documents()
    bm25_documents = filtered_documents(source_documents, filters)

    bm25_k = candidate_k(args.bm25_k, args.k)
    dense_k = candidate_k(args.dense_k, args.k)
    bm25_results = retrieve_bm25(bm25_documents, args.query, bm25_k)
    dense_results = retrieve_dense(components, args.query, dense_k, filters)
    selected_results = retrieve_selected_strategy(
        components,
        args.query,
        args.k,
        filters,
        args,
        bm25_k,
        dense_k,
    )

    strategy_label = args.strategy.replace("_", " ").title()

    print(f"query: {args.query}")
    print(f"loader/chunker/embedding: {args.loader}/{args.chunker}/{args.embedding}")
    print(f"strategy: {args.strategy}")
    print(f"filters: {filters}")
    print(f"weights: BM25={args.bm25_weight}, dense={args.dense_weight}")
    print(f"k: final={args.k}, bm25={bm25_k}, dense={dense_k}")
    print(f"source documents: {len(source_documents)}")
    print(f"BM25 documents after filter: {len(bm25_documents)}")

    print_results("BM25 only", bm25_results, args, filters)
    print_results("Dense only", dense_results, args, filters)
    print_results(strategy_label, selected_results, args, filters)


if __name__ == "__main__":
    main()
