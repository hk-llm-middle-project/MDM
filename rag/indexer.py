"""Vector store indexing utilities."""

import sqlite3
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import DEFAULT_EMBEDDING_PROVIDER, INDEX_BATCH_SIZE, VECTORSTORE_DIR
from rag.embeddings import create_embeddings


def build_vectorstore(
    documents: list[Document],
    persist_directory: Path = VECTORSTORE_DIR,
    batch_size: int = INDEX_BATCH_SIZE,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
) -> Chroma:
    """Embed documents and save them in a local Chroma vector store."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    persist_directory.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=create_embeddings(embedding_provider),
    )
    for start in range(0, len(documents), batch_size):
        vectorstore.add_documents(documents[start : start + batch_size])

    return vectorstore


def load_vectorstore(
    persist_directory: Path = VECTORSTORE_DIR,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
) -> Chroma:
    """Load the local Chroma vector store."""
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=create_embeddings(embedding_provider),
    )


def vectorstore_exists(persist_directory: Path = VECTORSTORE_DIR) -> bool:
    """Return whether a Chroma store contains embedded documents."""
    database_path = persist_directory / "chroma.sqlite3"
    if not database_path.exists():
        return False

    try:
        with sqlite3.connect(database_path) as connection:
            count = connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    except sqlite3.Error:
        return False

    return count > 0
