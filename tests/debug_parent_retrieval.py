"""Debug parent retrieval results without LangSmith or LLM calls.

Run from the project root, for example:

    .venv\Scripts\python.exe tests\debug_parent_retrieval.py ^
      --query "신호기에 의해 교통정리가 이루어지고 있지 않고 한쪽 도로가 일방통행로인 교차로..."

The script prints every returned document's metadata and a content preview,
regardless of how many documents are returned.
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


DEFAULT_QUERY = (
    "신호기에 의해 교통정리가 이루어지고 있지 않고 한쪽 도로가 일방통행로인 교차로에서 "
    "일방통행로가 아닌 도로를 이용하여 교차로에 진입하여 직진 중인 A차량과 "
    "일방통행로를 역주행하여 교차로에 진입한 B차량이 충돌한 사고이다."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print documents returned by the parent retriever or retrieval pipeline.",
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Search query text.")
    parser.add_argument("--loader", default="upstage", help="Loader strategy.")
    parser.add_argument("--chunker", default="custom", help="Chunker strategy.")
    parser.add_argument("--embedding", default="openai", help="Embedding provider.")
    parser.add_argument("--k", type=int, default=3, help="Requested retrieval k.")
    parser.add_argument(
        "--mode",
        choices=("parent", "pipeline"),
        default="parent",
        help="Run parent strategy directly or through the retrieval pipeline.",
    )
    parser.add_argument("--party-type", default="자동차", help="Optional party_type metadata filter.")
    parser.add_argument("--location", default="교차로 사고", help="Optional location metadata filter.")
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=900,
        help="Maximum characters to print for each document.",
    )
    return parser.parse_args()


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
