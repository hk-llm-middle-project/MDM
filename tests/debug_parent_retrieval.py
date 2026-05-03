"""LangSmith와 LLM 호출 없이 parent 최종 검색 결과를 확인합니다.

프로젝트 루트에서 다음처럼 실행합니다.

    .venv\Scripts\python.exe tests\debug_parent_retrieval.py ^
      --query "검색 질문"

반환 문서 개수와 관계없이 각 문서의 metadata와 content preview를 출력합니다.
BM25/Dense 단계별 비교는 debug_retrieval_strategies.py에서 확인합니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.pipeline.retrieval import RetrievalPipelineConfig, run_retrieval_pipeline  # noqa: E402
from rag.pipeline.retriever.strategies.parent import retrieve_with_parent_documents  # noqa: E402
from rag.service.vectorstore.vectorstore_service import get_retrieval_components  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="parent 검색 전략 또는 retrieval pipeline 반환 문서를 출력합니다.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="검색 질문입니다. 생략하면 실행 중 입력받습니다.",
    )
    parser.add_argument("--loader", default="upstage", help="loader 전략입니다.")
    parser.add_argument("--chunker", default="custom", help="chunker 전략입니다.")
    parser.add_argument("--embedding", default="openai", help="embedding 제공자입니다.")
    parser.add_argument("--k", type=int, default=3, help="요청할 검색 문서 수입니다.")
    parser.add_argument(
        "--mode",
        choices=("parent", "pipeline"),
        default="parent",
        help="parent 전략을 직접 실행할지 retrieval pipeline으로 실행할지 선택합니다.",
    )
    parser.add_argument("--party-type", default=None, help="party_type metadata filter입니다.")
    parser.add_argument("--location", default=None, help="location metadata filter입니다.")
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=900,
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


def print_document(index: int, document, preview_chars: int) -> None:
    metadata = dict(document.metadata)
    preview = document.page_content.strip()
    if len(preview) > preview_chars:
        preview = preview[:preview_chars].rstrip() + "\n...(truncated)"

    print(f"\n[{index}]")
    print("metadata:")
    for key in sorted(metadata):
        print(f"  {key}: {metadata[key]}")
    print("content:")
    print(preview)


def main() -> None:
    load_dotenv()
    args = parse_args()
    filters = build_filters(args)
    components = get_retrieval_components(
        args.loader,
        args.embedding,
        args.chunker,
    )

    if args.mode == "pipeline":
        documents = run_retrieval_pipeline(
            components,
            args.query,
            filters=filters,
            pipeline_config=RetrievalPipelineConfig(
                retriever_strategy="parent",
                reranker_strategy="none",
                final_k=args.k,
            ),
        )
    else:
        documents = retrieve_with_parent_documents(
            components,
            args.query,
            k=args.k,
            filters=filters,
        )

    print(f"query: {args.query}")
    print(f"mode: {args.mode}")
    print(f"loader/chunker/embedding: {args.loader}/{args.chunker}/{args.embedding}")
    print(f"filters: {filters}")
    print(f"requested k: {args.k}")
    print(f"returned count: {len(documents)}")

    for index, document in enumerate(documents, start=1):
        print_document(index, document, args.preview_chars)


if __name__ == "__main__":
    main()
