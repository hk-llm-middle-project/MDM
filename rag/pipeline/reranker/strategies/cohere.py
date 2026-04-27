"""Cohere 리랭커 전략."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from rag.pipeline.reranker.strategies.common import build_scored_document


@dataclass(frozen=True)
class CohereRerankerConfig:
    """Cohere 기반 리랭커 설정."""

    model: str = "rerank-v3.5"
    top_n: int | None = None
    cohere_api_key: str | None = None
    client: Any | None = None


def rerank_with_cohere(
    query: str,
    documents: list[Document],
    k: int,
    strategy_config: CohereRerankerConfig | None = None,
) -> list[Document]:
    """Cohere Rerank로 문서를 재정렬한 뒤 상위 k개를 반환합니다."""
    if not documents:
        return []

    try:
        from langchain_cohere import CohereRerank
    except ImportError as error:
        raise ImportError(
            "cohere 리랭커를 사용하려면 `langchain-cohere` 패키지가 설치되어 있어야 합니다."
        ) from error

    config = strategy_config or CohereRerankerConfig()
    reranker_kwargs: dict[str, Any] = {
        "model": config.model,
        "top_n": config.top_n or k,
    }
    if config.client is not None:
        reranker_kwargs["client"] = config.client
    if config.cohere_api_key:
        reranker_kwargs["cohere_api_key"] = config.cohere_api_key

    reranker = CohereRerank(**reranker_kwargs)
    ranked_documents = reranker.compress_documents(documents=documents, query=query)

    return [
        build_scored_document(document, document.metadata.get("relevance_score"))
        for document in ranked_documents[:k]
    ]
