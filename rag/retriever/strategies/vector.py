"""VectorStoreRetriever 전략."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from rag.retriever.components import RetrievalComponents


@dataclass(frozen=True)
class VectorStoreRetrieverConfig:
    """벡터스토어 검색 전략 설정."""

    search_type: str = "similarity"


def retrieve_with_vectorstore(
    components: RetrievalComponents,
    query: str,
    k: int,
    filters: dict[str, object] | None = None,
    strategy_config: VectorStoreRetrieverConfig | None = None,
) -> list[Document]:
    """벡터스토어 retriever 인터페이스로 문서를 조회합니다."""
    config = strategy_config or VectorStoreRetrieverConfig()
    search_kwargs: dict[str, object] = {"k": k}
    if filters:
        search_kwargs["filter"] = filters

    retriever = components.vectorstore.as_retriever(
        search_type=config.search_type,
        search_kwargs=search_kwargs,
    )
    return list(retriever.invoke(query))
