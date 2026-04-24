"""부모 문서 기반 검색 전략."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from rag.retriever.components import RetrievalComponents, get_or_create_parent_retriever


@dataclass(frozen=True)
class ParentDocumentRetrieverConfig:
    """부모 문서 기반 검색 전략 설정."""

    source_documents: list[Document] | None = None
    child_chunk_size: int = 300
    child_chunk_overlap: int = 50
    parent_chunk_size: int | None = None
    parent_chunk_overlap: int = 100


def retrieve_with_parent_documents(
    components: RetrievalComponents,
    query: str,
    k: int,
    filters: dict[str, object] | None = None,
    strategy_config: ParentDocumentRetrieverConfig | None = None,
) -> list[Document]:
    """메모리 기반 자식 인덱스를 이용해 부모 문맥까지 확장하여 검색합니다."""
    config = strategy_config or ParentDocumentRetrieverConfig()
    source_documents = config.source_documents or components.get_source_documents()
    if not source_documents:
        return []

    retriever = get_or_create_parent_retriever(components, config, k)
    retriever.search_kwargs = {"k": k}
    results = list(retriever.invoke(query))
    if not filters:
        return results[:k]

    filtered = [
        document
        for document in results
        if all(document.metadata.get(key) == value for key, value in filters.items())
    ]
    return filtered[:k]
