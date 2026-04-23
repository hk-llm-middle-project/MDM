"""Similarity search and retrieval utilities."""

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from config import RETRIEVER_K


def retrieve(vectorstore: VectorStore, query: str, k: int = RETRIEVER_K) -> list[Document]:
    """Retrieve similar chunks using vector similarity search."""
    return vectorstore.similarity_search(query, k=k)
