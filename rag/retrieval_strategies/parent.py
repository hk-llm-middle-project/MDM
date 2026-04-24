"""부모 문서 기반 검색 전략."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.retrieval_strategies.common import get_embedding_function, get_vectorstore_documents


@dataclass(frozen=True)
class ParentDocumentRetrieverConfig:
    """부모 문서 기반 검색 전략 설정."""

    source_documents: list[Document] | None = None
    child_chunk_size: int = 300
    child_chunk_overlap: int = 50
    parent_chunk_size: int | None = None
    parent_chunk_overlap: int = 100


def retrieve_with_parent_documents(
    vectorstore: Any,
    query: str,
    k: int,
    filters: dict[str, object] | None = None,
    strategy_config: ParentDocumentRetrieverConfig | None = None,
) -> list[Document]:
    """메모리 기반 자식 인덱스를 이용해 부모 문맥까지 확장하여 검색합니다."""
    config = strategy_config or ParentDocumentRetrieverConfig()
    source_documents = config.source_documents
    if source_documents is None:
        source_documents = get_vectorstore_documents(vectorstore)

    documents = list(source_documents)
    if not documents:
        return []

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.child_chunk_size,
        chunk_overlap=config.child_chunk_overlap,
    )
    parent_splitter = None
    if config.parent_chunk_size is not None:
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.parent_chunk_size,
            chunk_overlap=config.parent_chunk_overlap,
        )

    child_vectorstore = InMemoryVectorStore(get_embedding_function(vectorstore))
    docstore = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=child_vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": k},
    )
    retriever.add_documents(documents)

    results = list(retriever.invoke(query))
    if not filters:
        return results[:k]

    filtered = [
        document
        for document in results
        if all(document.metadata.get(key) == value for key, value in filters.items())
    ]
    return filtered[:k]
