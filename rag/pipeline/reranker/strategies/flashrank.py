"""FlashRank 리랭커 전략."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from rag.service.tracing import TraceContext


@dataclass(frozen=True)
class FlashrankRerankerConfig:
    """FlashRank 기반 리랭커 설정."""

    model_name: str = "ms-marco-MiniLM-L-12-v2"
    cache_dir: str | None = None


def rerank_with_flashrank(
    query: str,
    documents: list[Document],
    k: int,
    strategy_config: FlashrankRerankerConfig | None = None,
    trace_context: TraceContext | None = None,
) -> list[Document]:
    """FlashRank로 문서를 재정렬한 뒤 상위 k개를 반환합니다."""
    if not documents:
        return []

    try:
        from flashrank import Ranker, RerankRequest
    except ImportError as error:
        raise ImportError(
            "flashrank 리랭커를 사용하려면 `flashrank` 패키지가 설치되어 있어야 합니다."
        ) from error

    del trace_context
    config = strategy_config or FlashrankRerankerConfig()
    ranker_kwargs: dict[str, str] = {"model_name": config.model_name}
    if config.cache_dir:
        ranker_kwargs["cache_dir"] = config.cache_dir

    ranker = Ranker(**ranker_kwargs)
    passages = [
        {
            "id": str(index),
            "text": document.page_content,
            "metadata": document.metadata,
        }
        for index, document in enumerate(documents)
    ]
    request = RerankRequest(query=query, passages=passages)
    ranked_passages = ranker.rerank(request)

    reranked_documents: list[Document] = []
    for ranked_passage in ranked_passages[:k]:
        metadata = dict(ranked_passage.get("metadata") or {})
        if "score" in ranked_passage:
            metadata["rerank_score"] = ranked_passage["score"]
        reranked_documents.append(
            Document(
                page_content=ranked_passage["text"],
                metadata=metadata,
            )
        )

    return reranked_documents
