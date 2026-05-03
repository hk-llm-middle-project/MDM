"""Ensemble child retrieval with parent document expansion."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from rag.pipeline.retriever.common import mark_retrieval_fallback
from rag.pipeline.retriever.components import RetrievalComponents
from rag.pipeline.retriever.strategies.ensemble import (
    DEFAULT_ID_KEY,
    EnsembleRetrieverConfig,
    retrieve_with_ensemble,
)
from rag.pipeline.retriever.strategies.parent import (
    _build_parent_indexes,
    _dedupe_documents,
    _has_metadata_parent_structure,
    _is_accident_situation_child,
    _merge_filter,
    _parent_child_candidate_k,
    _parent_for_child,
)
from rag.service.tracing import TraceContext


@dataclass(frozen=True)
class EnsembleParentRetrieverConfig:
    """Settings for BM25/Dense child search followed by parent expansion."""

    weights: tuple[float, float] = (0.5, 0.5)
    bm25_k: int | None = None
    dense_k: int | None = None
    search_type: str = "similarity"
    id_key: str | None = DEFAULT_ID_KEY
    source_documents: list[Document] | None = None


def _to_ensemble_config(
    config: EnsembleParentRetrieverConfig | EnsembleRetrieverConfig,
) -> EnsembleRetrieverConfig:
    return EnsembleRetrieverConfig(
        weights=config.weights,
        bm25_k=config.bm25_k,
        dense_k=config.dense_k,
        search_type=config.search_type,
        id_key=config.id_key,
    )


def retrieve_with_ensemble_parent_documents(
    components: RetrievalComponents,
    query: str,
    k: int,
    filters: dict[str, object] | None = None,
    strategy_config: EnsembleParentRetrieverConfig | EnsembleRetrieverConfig | None = None,
    trace_context: TraceContext | None = None,
) -> list[Document]:
    """Retrieve child chunks with ensemble search, then return their parents."""
    config = strategy_config or EnsembleParentRetrieverConfig()
    source_documents = (
        config.source_documents
        if isinstance(config, EnsembleParentRetrieverConfig)
        else None
    ) or components.get_source_documents()
    if not source_documents or k <= 0:
        return []

    ensemble_config = _to_ensemble_config(config)
    if not _has_metadata_parent_structure(list(source_documents)):
        fallback_documents = retrieve_with_ensemble(
            components,
            query,
            k,
            filters=filters,
            strategy_config=ensemble_config,
            trace_context=trace_context,
            run_name="mdm.retrieve.ensemble_parent.fallback_ensemble",
        )
        return mark_retrieval_fallback(
            fallback_documents,
            fallback_from="ensemble_parent",
            fallback_to="ensemble",
            reason="missing parent-child metadata",
        )

    child_filter = _merge_filter(filters, {"chunk_type": "child"})
    child_candidates = retrieve_with_ensemble(
        components,
        query,
        k=max(
            _parent_child_candidate_k(k),
            ensemble_config.bm25_k or 0,
            ensemble_config.dense_k or 0,
        ),
        filters=child_filter,
        strategy_config=ensemble_config,
        trace_context=trace_context,
        run_name="mdm.retrieve.ensemble_parent.child_ensemble",
    )
    if not child_candidates:
        return []

    parents_by_chunk_id, parents_by_diagram_id = _build_parent_indexes(
        list(source_documents),
    )
    ordered_children = sorted(
        child_candidates,
        key=lambda document: 0 if _is_accident_situation_child(document) else 1,
    )
    parent_documents = [
        parent
        for child_document in ordered_children
        if (parent := _parent_for_child(child_document, parents_by_chunk_id, parents_by_diagram_id))
        is not None
    ]
    return _dedupe_documents(parent_documents, k)
