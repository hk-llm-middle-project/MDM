"""Parent document retrieval strategy."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from rag.pipeline.retriever.common import mark_retrieval_fallback
from rag.pipeline.retriever.components import RetrievalComponents
from rag.service.tracing import TraceContext


PARENT_CHILD_CANDIDATE_K_MULTIPLIER = 8
PARENT_CHILD_MIN_CANDIDATE_K = 20


@dataclass(frozen=True)
class ParentDocumentRetrieverConfig:
    """Parent document retrieval settings.

    This strategy follows stored chunk metadata instead of rebuilding parent/child
    chunks at runtime.
    """

    source_documents: list[Document] | None = None


def _has_metadata_parent_structure(documents: list[Document]) -> bool:
    has_child = False
    has_parent = False
    for document in documents:
        metadata = document.metadata
        chunk_type = metadata.get("chunk_type")
        if chunk_type == "parent":
            has_parent = True
        elif chunk_type == "child" and (
            metadata.get("parent_id") is not None or metadata.get("diagram_id") is not None
        ):
            has_child = True
        if has_parent and has_child:
            return True
    return False


def _merge_filter(
    base_filter: dict[str, object] | None,
    extra_filter: dict[str, object],
) -> dict[str, object]:
    if not base_filter:
        return extra_filter
    if "$and" in base_filter and isinstance(base_filter["$and"], list):
        return {"$and": [*base_filter["$and"], extra_filter]}
    return {"$and": [base_filter, extra_filter]}


def _is_accident_situation_child(document: Document) -> bool:
    compact_content = "".join(document.page_content.split())
    return "사고상황" in compact_content


def _dedupe_documents(documents: list[Document], k: int) -> list[Document]:
    seen_keys: set[tuple[object, object, str]] = set()
    unique_documents: list[Document] = []
    for document in documents:
        metadata = document.metadata
        key = (
            metadata.get("chunk_id"),
            metadata.get("diagram_id"),
            document.page_content,
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_documents.append(document)
        if len(unique_documents) >= k:
            break
    return unique_documents


def _build_parent_indexes(
    source_documents: list[Document],
) -> tuple[dict[object, Document], dict[str, Document]]:
    parents_by_chunk_id: dict[object, Document] = {}
    parents_by_diagram_id: dict[str, Document] = {}
    for document in source_documents:
        metadata = document.metadata
        if metadata.get("chunk_type") != "parent":
            continue

        chunk_id = metadata.get("chunk_id")
        if chunk_id is not None:
            parents_by_chunk_id.setdefault(chunk_id, document)
            parents_by_chunk_id.setdefault(str(chunk_id), document)

        diagram_id = metadata.get("diagram_id")
        if diagram_id is not None:
            parents_by_diagram_id.setdefault(str(diagram_id), document)
    return parents_by_chunk_id, parents_by_diagram_id


def _parent_for_child(
    child_document: Document,
    parents_by_chunk_id: dict[object, Document],
    parents_by_diagram_id: dict[str, Document],
) -> Document | None:
    metadata = child_document.metadata

    parent_id = metadata.get("parent_id")
    if parent_id is not None:
        parent = parents_by_chunk_id.get(parent_id) or parents_by_chunk_id.get(str(parent_id))
        if parent is not None:
            return parent

    diagram_id = metadata.get("diagram_id")
    if diagram_id is None:
        return None
    return parents_by_diagram_id.get(str(diagram_id))


def _parent_child_candidate_k(final_k: int) -> int:
    return max(
        final_k * PARENT_CHILD_CANDIDATE_K_MULTIPLIER,
        PARENT_CHILD_MIN_CANDIDATE_K,
    )


def _document_with_child_relevance_score(document: Document, score: float) -> Document:
    metadata = dict(document.metadata)
    metadata["similarity_score"] = float(score)
    metadata["score_source"] = "child_vector_relevance"
    return Document(page_content=document.page_content, metadata=metadata)


def _copy_parent_with_child_match(parent: Document, child: Document) -> Document:
    metadata = dict(parent.metadata)
    child_metadata = child.metadata
    similarity_score = child_metadata.get("similarity_score")
    if isinstance(similarity_score, int | float):
        metadata["similarity_score"] = float(similarity_score)
        metadata["score_source"] = child_metadata.get("score_source", "child_vector_relevance")
    child_chunk_id = child_metadata.get("chunk_id")
    if child_chunk_id is not None:
        metadata["matched_child_chunk_id"] = child_chunk_id
    return Document(page_content=parent.page_content, metadata=metadata)


def _retrieve_child_candidates_with_relevance_scores(
    components: RetrievalComponents,
    query: str,
    candidate_k: int,
    child_filter: dict[str, object],
    trace_context: TraceContext | None = None,
) -> list[Document] | None:
    search_with_scores = getattr(
        components.vectorstore,
        "similarity_search_with_relevance_scores",
        None,
    )
    if not callable(search_with_scores):
        return None

    config_dict = (
        trace_context.langchain_config("mdm.retrieve.parent.child.scores")
        if trace_context
        else {}
    )
    try:
        scored_documents = search_with_scores(
            query,
            k=candidate_k,
            filter=child_filter,
            **config_dict,
        )
    except (AttributeError, NotImplementedError, TypeError):
        return None

    return [
        _document_with_child_relevance_score(document, score)
        for document, score in scored_documents
    ]


def _retrieve_child_candidates(
    components: RetrievalComponents,
    query: str,
    k: int,
    filters: dict[str, object] | None,
    trace_context: TraceContext | None,
) -> list[Document]:
    child_filter = _merge_filter(filters, {"chunk_type": "child"})
    candidate_k = _parent_child_candidate_k(k)
    scored_candidates = _retrieve_child_candidates_with_relevance_scores(
        components,
        query,
        candidate_k,
        child_filter,
        trace_context=trace_context,
    )
    if scored_candidates is not None:
        return scored_candidates

    search_kwargs: dict[str, object] = {"k": candidate_k, "filter": child_filter}
    retriever = components.vectorstore.as_retriever(search_kwargs=search_kwargs)
    config_dict = trace_context.langchain_config("mdm.retrieve.parent.child") if trace_context else None
    return list(retriever.invoke(query, config=config_dict) if config_dict else retriever.invoke(query))


def _retrieve_vectorstore_documents(
    components: RetrievalComponents,
    query: str,
    k: int,
    filters: dict[str, object] | None,
    trace_context: TraceContext | None,
) -> list[Document]:
    search_kwargs: dict[str, object] = {"k": k}
    if filters:
        search_kwargs["filter"] = filters
    retriever = components.vectorstore.as_retriever(search_kwargs=search_kwargs)
    config_dict = trace_context.langchain_config("mdm.retrieve.parent.vectorstore") if trace_context else None
    return list(retriever.invoke(query, config=config_dict) if config_dict else retriever.invoke(query))


def retrieve_with_parent_documents(
    components: RetrievalComponents,
    query: str,
    k: int,
    filters: dict[str, object] | None = None,
    strategy_config: ParentDocumentRetrieverConfig | None = None,
    trace_context: TraceContext | None = None,
) -> list[Document]:
    """Retrieve stored parents for the best child metadata matches."""
    config = strategy_config or ParentDocumentRetrieverConfig()
    source_documents = config.source_documents or components.get_source_documents()
    if not source_documents or k <= 0:
        return []

    if not _has_metadata_parent_structure(list(source_documents)):
        fallback_documents = _retrieve_vectorstore_documents(
            components,
            query,
            k,
            filters,
            trace_context,
        )
        return mark_retrieval_fallback(
            fallback_documents,
            fallback_from="parent",
            fallback_to="vectorstore",
            reason="missing parent-child metadata",
        )

    child_candidates = _retrieve_child_candidates(components, query, k, filters, trace_context)
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
        _copy_parent_with_child_match(parent, child_document)
        for child_document in ordered_children
        if (parent := _parent_for_child(child_document, parents_by_chunk_id, parents_by_diagram_id))
        is not None
    ]
    return _dedupe_documents(parent_documents, k)
