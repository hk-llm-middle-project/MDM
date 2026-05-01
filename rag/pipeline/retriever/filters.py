"""Retriever metadata filter helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.documents import Document


def metadata_matches_filter(
    metadata: Mapping[str, Any],
    filters: Mapping[str, Any] | None,
) -> bool:
    """Return whether document metadata satisfies the supported filter syntax."""
    if not filters:
        return True

    and_conditions = filters.get("$and")
    if and_conditions is not None:
        if not isinstance(and_conditions, Sequence) or isinstance(and_conditions, (str, bytes)):
            return False
        return all(
            isinstance(condition, Mapping) and metadata_matches_filter(metadata, condition)
            for condition in and_conditions
        )

    return all(metadata.get(key) == value for key, value in filters.items())


def filter_documents_by_metadata(
    documents: Sequence[Document],
    filters: Mapping[str, Any] | None,
) -> list[Document]:
    """Filter documents with the same metadata filter syntax used by retrievers."""
    if not filters:
        return list(documents)
    return [
        document
        for document in documents
        if metadata_matches_filter(document.metadata, filters)
    ]
