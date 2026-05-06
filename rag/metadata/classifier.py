"""Rule-based page metadata cache and merge helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from rag.service.intake.values import LOCATIONS, PARTY_TYPES


SKIP_CHUNK_TYPES = {"general", "preface"}


@dataclass(frozen=True)
class PageMetadataClassification:
    """Allowed metadata labels for one source page."""

    party_type: str | None = None
    location: str | None = None


def normalize_page_metadata_cache_entry(data: dict[str, object]) -> PageMetadataClassification:
    """Validate cached metadata output against allowed values."""
    party_type = data.get("party_type")
    if party_type not in PARTY_TYPES:
        party_type = None

    location = data.get("location")
    if location not in LOCATIONS:
        location = None

    return PageMetadataClassification(
        party_type=party_type,
        location=location,
    )


def classify_page_metadata(page: int | Document | None) -> PageMetadataClassification:
    """Classify a page number into rule-based metadata."""
    if isinstance(page, Document):
        page = page.metadata.get("page")
    try:
        page_number = int(page) if page is not None else None
    except (TypeError, ValueError):
        page_number = None
    return _classification_for_page(page_number)


def default_page_metadata_classification() -> PageMetadataClassification:
    """Return empty metadata for pages outside the case ranges."""
    return PageMetadataClassification()


def load_page_metadata_cache(cache_path: Path) -> dict[str, dict[str, object]]:
    """Load page metadata classifications keyed by page number."""
    if not cache_path.exists():
        return {}

    with cache_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"Page metadata cache must be a JSON object: {cache_path}")
    return {
        str(page): value
        for page, value in payload.items()
        if isinstance(value, dict)
    }


def save_page_metadata_cache(cache_path: Path, cache: dict[str, dict[str, object]]) -> None:
    """Persist page metadata classifications keyed by page number."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_cache = dict(sorted(cache.items(), key=lambda item: _cache_sort_key(item[0])))
    with cache_path.open("w", encoding="utf-8") as fp:
        json.dump(ordered_cache, fp, ensure_ascii=False, indent=2)


def _cache_sort_key(key: str) -> tuple[int, int | str]:
    """Sort numeric page keys as numbers, followed by non-numeric keys."""
    return (0, int(key)) if key.isdigit() else (1, key)


def _classification_to_cache_entry(classification: PageMetadataClassification) -> dict[str, object]:
    return {
        "party_type": classification.party_type,
        "location": classification.location,
    }


def _page_cache_key(document: Document, fallback_index: int) -> str:
    page = document.metadata.get("page")
    if page is None:
        return str(fallback_index)
    return str(page)


def _merge_classification_metadata(
    document: Document,
    classification: PageMetadataClassification,
) -> Document:
    metadata = dict(document.metadata)

    if classification.party_type is not None:
        metadata["party_type"] = classification.party_type
    if classification.location is not None:
        metadata["location"] = classification.location

    return Document(page_content=document.page_content, metadata=metadata)


def _should_merge_metadata(document: Document) -> bool:
    return document.metadata.get("chunk_type") not in SKIP_CHUNK_TYPES


def build_rule_based_page_metadata_cache(
    documents: list[Document],
) -> dict[str, dict[str, object]]:
    """Build a page-number keyed metadata cache from deterministic rules."""
    cache: dict[str, dict[str, object]] = {}
    for index, document in enumerate(documents, start=1):
        cache_key = _page_cache_key(document, index)
        cache[cache_key] = _classification_to_cache_entry(classify_page_metadata(document))
    return cache


def write_rule_based_page_metadata_cache(
    documents: list[Document],
    cache_path: Path,
) -> dict[str, dict[str, object]]:
    """Generate and persist the deterministic page metadata cache."""
    cache = build_rule_based_page_metadata_cache(documents)
    save_page_metadata_cache(cache_path, cache)
    return cache


def enrich_documents_with_page_metadata(
    documents: list[Document],
    cache_path: Path | None = None,
) -> list[Document]:
    """Merge rule-based page metadata into chunk documents using a cache when provided."""
    cache = load_page_metadata_cache(cache_path) if cache_path is not None else {}
    enriched_documents: list[Document] = []

    for index, document in enumerate(documents, start=1):
        if not _should_merge_metadata(document):
            enriched_documents.append(
                Document(page_content=document.page_content, metadata=dict(document.metadata))
            )
            continue

        cache_key = _page_cache_key(document, index)
        cached_entry = cache.get(cache_key)
        if cached_entry is None:
            enriched_documents.append(
                Document(page_content=document.page_content, metadata=dict(document.metadata))
            )
            continue

        classification = normalize_page_metadata_cache_entry(cached_entry)
        enriched_documents.append(
            _merge_classification_metadata(
                document,
                classification,
            )
        )

    return enriched_documents


def _classification_for_page(page: int | None) -> PageMetadataClassification:
    if page is None:
        return default_page_metadata_classification()

    party_type: str | None = None
    location: str | None = None

    if 39 <= page <= 69:
        party_type, location = "보행자", "횡단보도 내"
    elif 70 <= page <= 89:
        party_type, location = "보행자", "횡단보도 부근"
    elif 90 <= page <= 106:
        party_type, location = "보행자", "횡단보도 없음"
    elif 107 <= page <= 123:
        party_type, location = "보행자", "기타"
    elif 148 <= page <= 327:
        party_type, location = "자동차", "교차로 사고"
    elif 328 <= page <= 351:
        party_type, location = "자동차", "마주보는 방향 진행차량 상호 간의 사고"
    elif 352 <= page <= 425:
        party_type, location = "자동차", "같은 방향 진행차량 상호간의 사고"
    elif 426 <= page <= 488:
        party_type, location = "자동차", "기타"
    elif 501 <= page <= 565:
        party_type, location = "자전거", "교차로 사고"
    elif 566 <= page <= 568:
        party_type, location = "자전거", "마주보는 방향 진행차량 상호 간의 사고"
    elif 569 <= page <= 578:
        party_type, location = "자전거", "같은 방향 진행차량 상호간의 사고"
    elif 579 <= page <= 587:
        party_type, location = "자전거", "기타"

    if party_type is None or location is None:
        return default_page_metadata_classification()

    return PageMetadataClassification(
        party_type=party_type,
        location=location,
    )
