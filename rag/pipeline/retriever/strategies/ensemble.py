"""앙상블 검색 전략."""

from __future__ import annotations

from dataclasses import dataclass

from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from rag.pipeline.retriever.components import RetrievalComponents, get_or_create_bm25_retriever


@dataclass(frozen=True)
class EnsembleRetrieverConfig:
    """Kiwi BM25와 dense 검색을 결합하는 앙상블 전략 설정."""

    weights: tuple[float, float] = (0.5, 0.5)
    bm25_k: int | None = None
    dense_k: int | None = None
    search_type: str = "similarity"


def retrieve_with_ensemble(
    components: RetrievalComponents,
    query: str,
    k: int,
    filters: dict[str, object] | None = None,
    strategy_config: EnsembleRetrieverConfig | None = None,
) -> list[Document]:
    """Kiwi BM25와 dense 벡터 검색 결과를 결합합니다."""
    config = strategy_config or EnsembleRetrieverConfig()
    source_documents = components.get_source_documents()
    if not source_documents:
        return []

    bm25_retriever = get_or_create_bm25_retriever(components)
    bm25_retriever.k = config.bm25_k or k

    dense_search_kwargs: dict[str, object] = {"k": config.dense_k or k}
    if filters:
        dense_search_kwargs["filter"] = filters

    dense_retriever = components.vectorstore.as_retriever(
        search_type=config.search_type,
        search_kwargs=dense_search_kwargs,
    )

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=list(config.weights),
    )
    return list(ensemble.invoke(query))[:k]
