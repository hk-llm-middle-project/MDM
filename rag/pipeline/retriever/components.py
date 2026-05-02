"""Retrieval component bundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag.pipeline.retriever.common import (
    get_vectorstore_documents,
    kiwi_tokenize,
)


@dataclass
class RetrievalComponents:
    """Shared objects used by retrieval strategies."""

    vectorstore: Any
    source_documents: list[Document] | None = None
    bm25_retriever: Any | None = None

    def get_source_documents(self) -> list[Document]:
        """Load source documents from the vectorstore once and reuse them."""
        if self.source_documents is None:
            self.source_documents = get_vectorstore_documents(self.vectorstore)
        return self.source_documents


def build_retrieval_components(
    vectorstore: Any,
    source_documents: list[Document] | None = None,
) -> RetrievalComponents:
    """Create the shared retrieval component bundle."""
    return RetrievalComponents(
        vectorstore=vectorstore,
        source_documents=source_documents,
    )


def get_or_create_bm25_retriever(components: RetrievalComponents) -> BM25Retriever:
    """Create and cache a BM25 retriever."""
    if components.bm25_retriever is None:
        documents = components.get_source_documents()
        if not documents:
            raise ValueError("Cannot create a BM25 retriever without source documents.")
        components.bm25_retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=kiwi_tokenize,
        )
    return components.bm25_retriever
