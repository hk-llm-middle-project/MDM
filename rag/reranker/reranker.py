"""리랭커 전략 라우팅 유틸리티."""

from __future__ import annotations

from langchain_core.documents import Document

from rag.reranker.strategies import (
    CohereRerankerConfig,
    FlashrankRerankerConfig,
    LLMScoreRerankerConfig,
    NoOpRerankerConfig,
    rerank_with_cohere,
    rerank_with_flashrank,
    rerank_with_llm_score,
    rerank_with_none,
)


RERANKER_STRATEGIES = {
    "none": rerank_with_none,
    "flashrank": rerank_with_flashrank,
    "cohere": rerank_with_cohere,
    "llm-score": rerank_with_llm_score,
}

RerankerConfig = (
    NoOpRerankerConfig
    | FlashrankRerankerConfig
    | CohereRerankerConfig
    | LLMScoreRerankerConfig
)


def rerank(
    query: str,
    documents: list[Document],
    k: int,
    strategy: str = "none",
    strategy_config: RerankerConfig | None = None,
) -> list[Document]:
    """선택한 리랭커 전략으로 문서를 재정렬합니다."""
    try:
        reranker_strategy = RERANKER_STRATEGIES[strategy]
    except KeyError as error:
        available = ", ".join(sorted(RERANKER_STRATEGIES))
        raise ValueError(f"알 수 없는 리랭커 전략입니다: {strategy}. 사용 가능 전략: {available}") from error

    return reranker_strategy(query, documents, k, strategy_config)
