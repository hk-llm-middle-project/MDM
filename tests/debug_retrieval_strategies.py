"""LangSmith와 LLM 호출 없이 검색 전략별 결과를 확인합니다.

프로젝트 루트에서 다음처럼 실행합니다.

    .venv\Scripts\python.exe tests\debug_retrieval_strategies.py ^
      --query "검색 질문" ^
      --loader llamaparser ^
      --chunker case-boundary ^
      --embedding openai ^
      --strategy ensemble_parent ^
      --target "보3" ^
      --party-type "보행자" ^
      --location "횡단보도 내" ^
      --k 3 ^
      --preview-chars 300 ^
      --use-chunk-id ^
      --bm25-weight 0.85 ^
      --dense-weight 0.15

BM25/Dense 기여도와 최종 전략 결과를 분리해서 볼 수 있도록
각 retriever가 반환한 metadata와 content preview를 출력합니다.
최종 전략은 ensemble, parent, ensemble_parent 중에서 선택할 수 있습니다.
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
    DEFAULT_ID_KEY,
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
        description="BM25, Dense, 선택한 최종 검색 전략 결과를 출력합니다.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="검색 질문입니다. 생략하면 실행 중 입력받습니다.",
    )
    parser.add_argument("--loader", default="llamaparser", help="loader 전략입니다.")
    parser.add_argument("--chunker", default="case-boundary", help="chunker 전략입니다.")
    parser.add_argument("--embedding", default="openai", help="embedding 제공자입니다.")
    parser.add_argument(
        "--strategy",
        choices=("ensemble", "ensemble_parent", "parent"),
        default="ensemble",
        help="BM25/Dense baseline 뒤에 출력할 최종 검색 전략입니다.",
    )
    parser.add_argument("--k", type=int, default=5, help="출력할 최종 top-k입니다.")
    parser.add_argument("--bm25-k", type=int, default=None, help="BM25 후보 k입니다.")
    parser.add_argument("--dense-k", type=int, default=None, help="Dense 후보 k입니다.")
    parser.add_argument("--bm25-weight", type=float, default=0.85, help="BM25 가중치입니다.")
    parser.add_argument("--dense-weight", type=float, default=0.15, help="Dense 가중치입니다.")
    chunk_id_group = parser.add_mutually_exclusive_group()
    chunk_id_group.add_argument(
        "--use-chunk-id",
        dest="use_chunk_id",
        action="store_true",
        default=True,
        help="ensemble 문서 식별 기준으로 metadata chunk_id를 사용합니다.",
    )
    chunk_id_group.add_argument(
        "--no-chunk-id",
        dest="use_chunk_id",
        action="store_false",
        help="ensemble 문서 식별 기준으로 page_content를 사용합니다.",
    )
    parser.add_argument("--party-type", default=None, help="party_type metadata filter입니다.")
    parser.add_argument("--location", default=None, help="location metadata filter입니다.")
    parser.add_argument("--target", default=None, help="기대하는 diagram/chunk marker입니다.")
    parser.add_argument(
        "--noise",
        nargs="*",
        default=[],
        help="섞이면 안 되는 diagram/chunk marker를 집계합니다.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="각 문서 preview에 출력할 최대 글자 수입니다.",
    )
    args = parser.parse_args()
    if args.query is None:
        args.query = input("query: ").strip()
    if not args.query:
        parser.error("--query가 없으면 실행 중 질문을 입력해야 합니다.")
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
        id_key=DEFAULT_ID_KEY if args.use_chunk_id else None,
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
    target_hits = (
        sum(1 for document in documents if marker_found(document, args.target))
        if args.target
        else None
    )
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
    if args.target:
        print(f"target hits ({args.target}): {target_hits}")
    if args.noise:
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
    print(f"id_key: {DEFAULT_ID_KEY if args.use_chunk_id else None}")
    print(f"k: final={args.k}, bm25={bm25_k}, dense={dense_k}")
    print(f"source documents: {len(source_documents)}")
    print(f"BM25 documents after filter: {len(bm25_documents)}")

    print_results("BM25 only", bm25_results, args, filters)
    print_results("Dense only", dense_results, args, filters)
    print_results(strategy_label, selected_results, args, filters)


if __name__ == "__main__":
    main()
