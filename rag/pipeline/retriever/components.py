"""검색에서 재사용할 컴포넌트 묶음."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.pipeline.retriever.common import (
    get_embedding_function_from_vectorstore,
    get_vectorstore_documents,
    kiwi_tokenize,
)

if TYPE_CHECKING:
    from rag.pipeline.retriever.strategies.parent import ParentDocumentRetrieverConfig


@dataclass
class RetrievalComponents:
    """검색 전략이 공유하는 재사용 객체 묶음."""

    vectorstore: Any
    source_documents: list[Document] | None = None
    embedding_function: Any | None = None
    bm25_retriever: Any | None = None
    parent_retrievers: dict[tuple[int, int, int | None, int], ParentDocumentRetriever] = field(
        default_factory=dict
    )

    def get_source_documents(self) -> list[Document]:
        """벡터스토어 원문 문서를 한 번만 읽어 재사용합니다."""
        if self.source_documents is None:
            self.source_documents = get_vectorstore_documents(self.vectorstore)
        return self.source_documents

    def get_embedding_function(self) -> Any:
        """벡터스토어 임베딩 함수를 한 번만 읽어 재사용합니다."""
        if self.embedding_function is None:
            self.embedding_function = get_embedding_function_from_vectorstore(self.vectorstore)
        return self.embedding_function


def build_retrieval_components(
    vectorstore: Any,
    source_documents: list[Document] | None = None,
) -> RetrievalComponents:
    """벡터스토어를 감싼 검색 컴포넌트 묶음을 생성합니다."""
    return RetrievalComponents(
        vectorstore=vectorstore,
        source_documents=source_documents,
    )


def get_or_create_bm25_retriever(components: RetrievalComponents) -> BM25Retriever:
    """BM25 리트리버를 한 번만 생성해 재사용합니다."""
    if components.bm25_retriever is None:
        documents = components.get_source_documents()
        if not documents:
            raise ValueError("BM25 리트리버를 생성할 문서가 없습니다.")
        components.bm25_retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=kiwi_tokenize,
        )
    return components.bm25_retriever


def build_parent_retriever(
    components: RetrievalComponents,
    config: ParentDocumentRetrieverConfig,
    source_documents: list[Document],
    k: int,
) -> ParentDocumentRetriever:
    """부모 문서 리트리버 인스턴스를 생성합니다."""
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

    child_vectorstore = InMemoryVectorStore(components.get_embedding_function())
    docstore = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=child_vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": k},
    )
    retriever.add_documents(source_documents)
    return retriever


def get_or_create_parent_retriever(
    components: RetrievalComponents,
    config: ParentDocumentRetrieverConfig,
    k: int,
) -> ParentDocumentRetriever:
    """기본 문서 집합 기준의 부모 문서 리트리버를 재사용합니다."""
    source_documents = config.source_documents or components.get_source_documents()
    if not source_documents:
        raise ValueError("부모 문서 리트리버를 생성할 문서가 없습니다.")

    if config.source_documents is not None:
        return build_parent_retriever(components, config, list(source_documents), k)

    cache_key = (
        config.child_chunk_size,
        config.child_chunk_overlap,
        config.parent_chunk_size,
        config.parent_chunk_overlap,
    )
    retriever = components.parent_retrievers.get(cache_key)
    if retriever is None:
        retriever = build_parent_retriever(components, config, list(source_documents), k)
        components.parent_retrievers[cache_key] = retriever
    return retriever
