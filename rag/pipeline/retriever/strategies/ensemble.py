"""앙상블 검색 전략."""

from __future__ import annotations

from dataclasses import dataclass

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag.pipeline.retriever.common import kiwi_tokenize
from rag.pipeline.retriever.components import RetrievalComponents, get_or_create_bm25_retriever
from rag.pipeline.retriever.filters import filter_documents_by_metadata
from rag.service.tracing import TraceContext


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
    trace_context: TraceContext | None = None,
) -> list[Document]:
    """Kiwi BM25와 dense 벡터 검색 결과를 결합합니다."""
    config = strategy_config or EnsembleRetrieverConfig()
    source_documents = components.get_source_documents()
    if not source_documents:
        return []

    bm25_documents = filter_documents_by_metadata(source_documents, filters)
    if filters and not bm25_documents:
        return []

    if filters:
        bm25_retriever = BM25Retriever.from_documents(
            bm25_documents,
            preprocess_func=kiwi_tokenize,
        )
    else:
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
    config_dict = trace_context.langchain_config("mdm.retrieve.ensemble") if trace_context else None
    results = ensemble.invoke(query, config=config_dict) if config_dict else ensemble.invoke(query)
    return filter_documents_by_metadata(list(results), filters)[:k]
