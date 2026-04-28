"""Embedding strategy routing utilities."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from config import DEFAULT_EMBEDDING_PROVIDER
from rag.embeddings.strategies import (
    create_bge_embeddings,
    create_google_embeddings,
    create_openai_embeddings,
)


EMBEDDING_STRATEGIES = {
    "openai": create_openai_embeddings,
    "bge": create_bge_embeddings,
    "google": create_google_embeddings,
}


def create_embeddings(provider: str = DEFAULT_EMBEDDING_PROVIDER) -> Embeddings:
    """Create the selected embedding provider."""
    try:
        embedding_strategy = EMBEDDING_STRATEGIES[provider]
    except KeyError as error:
        available = ", ".join(sorted(EMBEDDING_STRATEGIES))
        raise ValueError(
            f"Unknown embedding provider: {provider}. Available providers: {available}"
        ) from error

    return embedding_strategy()
