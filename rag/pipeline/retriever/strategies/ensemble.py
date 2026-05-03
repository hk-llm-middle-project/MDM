"""Ensemble retrieval strategy."""

from __future__ import annotations

from dataclasses import dataclass

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag.pipeline.retriever.common import (
    filter_documents_by_metadata,
    kiwi_tokenize,
    mark_retrieval_fallback,
    metadata_matches_filter,
)
from rag.pipeline.retriever.components import (
    RetrievalComponents,
    get_or_create_bm25_retriever,
)
from rag.service.tracing import TraceContext


MIN_ENSEMBLE_CANDIDATE_K = 20
DEFAULT_CANDIDATE_K_MULTIPLIER = 4
DEFAULT_ID_KEY = "chunk_id"


@dataclass(frozen=True)
class EnsembleRetrieverConfig:
    """Configuration for combining Kiwi BM25 and dense vector retrieval."""

    weights: tuple[float, float] = (0.5, 0.5)
    bm25_k: int | None = None
    dense_k: int | None = None
    search_type: str = "similarity"
    id_key: str | None = DEFAULT_ID_KEY


def retrieve_with_ensemble(
    components: RetrievalComponents,
    query: str,
    k: int,
    filters: dict[str, object] | None = None,
    strategy_config: EnsembleRetrieverConfig | None = None,
    trace_context: TraceContext | None = None,
    run_name: str = "mdm.retrieve.ensemble",
) -> list[Document]:
    """Combine Kiwi BM25 and dense vector search results."""
    config = strategy_config or EnsembleRetrieverConfig()
    source_documents = components.get_source_documents()
    if not source_documents:
        return []

    bm25_documents = filter_documents_by_metadata(source_documents, filters)
    dense_retriever = _build_dense_retriever(components, config, k, filters)

    if not bm25_documents:
        fallback_documents = _invoke_retriever(
            dense_retriever,
            query,
            k,
            trace_context,
            f"{run_name}.dense",
        )
        return mark_retrieval_fallback(
            fallback_documents,
            fallback_from="ensemble",
            fallback_to="dense",
            reason="no BM25 documents after metadata filter",
        )

    bm25_k = _candidate_k(config.bm25_k, k)
    bm25_retriever = (
        _build_bm25_retriever(bm25_documents)
        if filters
        else get_or_create_bm25_retriever(components)
    )
    bm25_retriever.k = bm25_k

    ensemble_kwargs: dict[str, object] = {
        "retrievers": [bm25_retriever, dense_retriever],
        "weights": list(config.weights),
    }
    id_key = _select_id_key(source_documents, config.id_key)
    if id_key is not None:
        ensemble_kwargs["id_key"] = id_key

    ensemble = EnsembleRetriever(**ensemble_kwargs)
    return _invoke_retriever(
        ensemble,
        query,
        k,
        trace_context,
        run_name,
    )


def _build_dense_retriever(
    components: RetrievalComponents,
    config: EnsembleRetrieverConfig,
    k: int,
    filters: dict[str, object] | None,
):
    dense_search_kwargs: dict[str, object] = {
        "k": _candidate_k(config.dense_k, k),
    }
    if filters:
        dense_search_kwargs["filter"] = filters

    return components.vectorstore.as_retriever(
        search_type=config.search_type,
        search_kwargs=dense_search_kwargs,
    )


def _candidate_k(configured_k: int | None, final_k: int) -> int:
    if configured_k is not None:
        return configured_k
    return max(
        final_k * DEFAULT_CANDIDATE_K_MULTIPLIER,
        MIN_ENSEMBLE_CANDIDATE_K,
    )


def _select_id_key(
    documents: list[Document],
    preferred_id_key: str | None,
) -> str | None:
    if preferred_id_key is None:
        return None
    if all(document.metadata.get(preferred_id_key) is not None for document in documents):
        return preferred_id_key
    return None


def _build_bm25_retriever(documents: list[Document]) -> BM25Retriever:
    return BM25Retriever.from_documents(
        documents,
        preprocess_func=kiwi_tokenize,
    )


def _filter_documents(
    documents: list[Document],
    filters: dict[str, object] | None,
) -> list[Document]:
    """Backward-compatible wrapper for debug scripts."""
    return filter_documents_by_metadata(documents, filters)


def _metadata_matches_filter(
    metadata: dict[str, object],
    filters: dict[str, object],
) -> bool:
    """Backward-compatible wrapper for debug scripts."""
    return metadata_matches_filter(metadata, filters)


def _invoke_retriever(
    retriever,
    query: str,
    k: int,
    trace_context: TraceContext | None,
    run_name: str,
) -> list[Document]:
    config_dict = trace_context.langchain_config(run_name) if trace_context else None
    results = (
        retriever.invoke(query, config=config_dict)
        if config_dict
        else retriever.invoke(query)
    )
    return list(results)[:k]
