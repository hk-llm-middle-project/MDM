"""검색 전략 라우팅 유틸리티."""

from langchain_core.documents import Document

from config import RETRIEVER_K
from rag.retriever.components import RetrievalComponents
from rag.retriever.strategies import (
    EnsembleRetrieverConfig,
    MultiQueryRetrieverConfig,
    ParentDocumentRetrieverConfig,
    SelfQueryRetrieverConfig,
    VectorStoreRetrieverConfig,
    retrieve_with_ensemble,
    retrieve_with_multiquery,
    retrieve_with_parent_documents,
    retrieve_with_self_query,
    retrieve_with_vectorstore,
)


RETRIEVAL_STRATEGIES = {
    "vectorstore": retrieve_with_vectorstore,
    "ensemble": retrieve_with_ensemble,
    "parent": retrieve_with_parent_documents,
    "multiquery": retrieve_with_multiquery,
    "selfquery": retrieve_with_self_query,
    "similarity": retrieve_with_vectorstore,
}

StrategyConfig = (
    VectorStoreRetrieverConfig
    | EnsembleRetrieverConfig
    | ParentDocumentRetrieverConfig
    | MultiQueryRetrieverConfig
    | SelfQueryRetrieverConfig
)


def retrieve(
    components: RetrievalComponents,
    query: str,
    k: int = RETRIEVER_K,
    strategy: str = "vectorstore",
    filters: dict[str, object] | None = None,
    strategy_config: StrategyConfig | None = None,
) -> list[Document]:
    """선택한 검색 전략으로 문서 청크를 조회합니다."""
    try:
        retrieval_strategy = RETRIEVAL_STRATEGIES[strategy]
    except KeyError as error:
        available = ", ".join(sorted(RETRIEVAL_STRATEGIES))
        raise ValueError(f"알 수 없는 검색 전략입니다: {strategy}. 사용 가능 전략: {available}") from error

    return retrieval_strategy(components, query, k, filters, strategy_config)
