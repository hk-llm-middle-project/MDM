"""검색 전략 라우팅 유틸리티."""

from langchain_core.documents import Document

from config import RETRIEVER_K
from rag.pipeline.retriever.components import RetrievalComponents
from rag.pipeline.retriever.strategies import (
    EnsembleRetrieverConfig,
    EnsembleParentRetrieverConfig,
    MultiQueryRetrieverConfig,
    ParentDocumentRetrieverConfig,
    VectorStoreRetrieverConfig,
    retrieve_with_ensemble,
    retrieve_with_ensemble_parent_documents,
    retrieve_with_multiquery,
    retrieve_with_parent_documents,
    retrieve_with_vectorstore,
)
from rag.service.tracing import TraceContext


RETRIEVAL_STRATEGIES = {
    "vectorstore": retrieve_with_vectorstore,
    "ensemble": retrieve_with_ensemble,
    "ensemble_parent": retrieve_with_ensemble_parent_documents,
    "parent": retrieve_with_parent_documents,
    "multiquery": retrieve_with_multiquery,
    "similarity": retrieve_with_vectorstore,
}

StrategyConfig = (
    VectorStoreRetrieverConfig
    | EnsembleRetrieverConfig
    | EnsembleParentRetrieverConfig
    | ParentDocumentRetrieverConfig
    | MultiQueryRetrieverConfig
)


def retrieve(
    components: RetrievalComponents,
    query: str,
    k: int = RETRIEVER_K,
    strategy: str = "vectorstore",
    filters: dict[str, object] | None = None,
    strategy_config: StrategyConfig | None = None,
    trace_context: TraceContext | None = None,
) -> list[Document]:
    """선택한 검색 전략으로 문서 청크를 조회합니다."""
    try:
        retrieval_strategy = RETRIEVAL_STRATEGIES[strategy]
    except KeyError as error:
        available = ", ".join(sorted(RETRIEVAL_STRATEGIES))
        raise ValueError(f"알 수 없는 검색 전략입니다: {strategy}. 사용 가능 전략: {available}") from error

    return retrieval_strategy(components, query, k, filters, strategy_config, trace_context)
