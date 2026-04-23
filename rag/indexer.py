"""Embedding and vector store indexing utilities."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import EMBEDDING_MODEL, VECTORSTORE_DIR


def create_embeddings() -> OpenAIEmbeddings:
    """Create the embedding model used for indexing and retrieval."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def build_vectorstore(
    documents: list[Document],
    persist_directory: Path = VECTORSTORE_DIR,
) -> Chroma:
    """Embed documents and save them in a local Chroma vector store."""
    persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=documents,
        embedding=create_embeddings(),
        persist_directory=str(persist_directory),
    )


def load_vectorstore(persist_directory: Path = VECTORSTORE_DIR) -> Chroma:
    """Load the local Chroma vector store."""
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=create_embeddings(),
    )


def vectorstore_exists(persist_directory: Path = VECTORSTORE_DIR) -> bool:
    """Return whether a Chroma store has already been created."""
    return persist_directory.exists() and any(persist_directory.iterdir())
