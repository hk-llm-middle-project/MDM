"""Embedding strategy routing utilities."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from config import (
    DEFAULT_EMBEDDING_PROVIDER,
    GOOGLE_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    get_embedding_query_cache_enabled,
)
from rag.embeddings.cache import CachedQueryEmbeddings


def create_openai_embeddings() -> Embeddings:
    from rag.embeddings.strategies.openai import create_openai_embeddings as factory

    return factory()


def create_bge_embeddings() -> Embeddings:
    from rag.embeddings.strategies.bge import create_bge_embeddings as factory

    return factory()


def create_google_embeddings() -> Embeddings:
    from rag.embeddings.strategies.google import create_google_embeddings as factory

    return factory()


EMBEDDING_STRATEGIES = {
    "openai": create_openai_embeddings,
    "bge": create_bge_embeddings,
    "google": create_google_embeddings,
}

EMBEDDING_MODEL_IDS = {
    "openai": OPENAI_EMBEDDING_MODEL,
    "bge": "bge-m3",
    "google": GOOGLE_EMBEDDING_MODEL,
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

    embeddings = embedding_strategy()
    if not isinstance(embeddings, Embeddings):
        return embeddings

    return CachedQueryEmbeddings(
        embeddings,
        provider=provider,
        model_id=EMBEDDING_MODEL_IDS.get(provider, provider),
        enabled=get_embedding_query_cache_enabled(),
    )
