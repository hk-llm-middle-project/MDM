"""LLM-based page metadata classification."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import LLM_MODEL
from rag.service.common.json_utils import extract_json_object
from rag.service.intake.intake_service import clamp_confidence
from rag.service.intake.values import LOCATIONS, PARTY_TYPES


CONFIDENCE_THRESHOLD = 0.75
MAX_PROMPT_CHARS = 6000
MAX_CLASSIFICATION_WORKERS = 8


@dataclass(frozen=True)
class PageMetadataClassification:
    """Allowed metadata labels for one source page."""

    party_type: str | None = None
    location: str | None = None
    confidence: dict[str, float] | None = None


def normalize_page_metadata_response(data: dict[str, Any]) -> PageMetadataClassification:
    """Validate LLM metadata output against allowed values and confidence."""
    confidence_data = data.get("confidence")
    if not isinstance(confidence_data, dict):
        confidence_data = {}

    confidence = {
        "party_type": clamp_confidence(confidence_data.get("party_type")),
        "location": clamp_confidence(confidence_data.get("location")),
    }

    party_type = data.get("party_type")
    if party_type not in PARTY_TYPES or confidence["party_type"] < CONFIDENCE_THRESHOLD:
        party_type = None

    location = data.get("location")
    if location not in LOCATIONS or confidence["location"] < CONFIDENCE_THRESHOLD:
        location = None

    return PageMetadataClassification(
        party_type=party_type,
        location=location,
        confidence=confidence,
    )


def build_page_metadata_prompt(page_content: str) -> str:
    """Build a constrained classification prompt for one page."""
    content = page_content.strip()[:MAX_PROMPT_CHARS]
    party_values = "\n".join(f"- {value}" for value in PARTY_TYPES)
    location_values = "\n".join(f"- {value}" for value in LOCATIONS)
    party_schema_values = " | ".join(f'"{value}"' for value in PARTY_TYPES)
    location_schema_values = " | ".join(f'"{value}"' for value in LOCATIONS)
    return f"""다음 자동차 사고 과실비율 문서 페이지를 읽고 검색용 metadata를 분류하세요.

party_type은 반드시 다음 중 하나 또는 null이어야 합니다:
{party_values}

location은 반드시 다음 중 하나 또는 null이어야 합니다:
{location_values}

확실하지 않으면 null을 사용하세요.
JSON 객체만 반환하세요.

반환 형식:
{{
  "party_type": {party_schema_values} | null,
  "location": {location_schema_values} | null,
  "confidence": {{
    "party_type": 0.0,
    "location": 0.0
  }}
}}

페이지 내용:
{content}
"""


def classify_page_metadata(page_content: str, llm: Any | None = None) -> PageMetadataClassification:
    """Classify one page into allowed search metadata values."""
    if not page_content.strip():
        return PageMetadataClassification(confidence={"party_type": 0.0, "location": 0.0})

    classifier_llm = llm or ChatOpenAI(model=LLM_MODEL, temperature=0)
    response = classifier_llm.invoke(build_page_metadata_prompt(page_content))
    content = getattr(response, "content", response)
    return normalize_page_metadata_response(extract_json_object(str(content)))


def default_page_metadata_classification() -> PageMetadataClassification:
    """Return empty metadata for pages that cannot be classified."""
    return PageMetadataClassification(confidence={"party_type": 0.0, "location": 0.0})


def classify_page_metadata_safely(
    document: Document,
    llm: Any | None = None,
) -> tuple[PageMetadataClassification, bool]:
    """Classify one page, falling back to empty metadata if classification fails."""
    try:
        return classify_page_metadata(document.page_content, llm=llm), False
    except Exception as exc:
        page = document.metadata.get("page", "unknown")
        print(
            "[WARN] page metadata classification failed "
            f"- page={page}, error_type={type(exc).__name__}, error={exc}"
        )
        return default_page_metadata_classification(), True


def load_page_metadata_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
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


def save_page_metadata_cache(cache_path: Path, cache: dict[str, dict[str, Any]]) -> None:
    """Persist page metadata classifications keyed by page number."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_cache = dict(sorted(cache.items(), key=lambda item: _cache_sort_key(item[0])))
    with cache_path.open("w", encoding="utf-8") as fp:
        json.dump(ordered_cache, fp, ensure_ascii=False, indent=2)


def _cache_sort_key(key: str) -> tuple[int, int | str]:
    """Sort numeric page keys as numbers, followed by non-numeric keys."""
    return (0, int(key)) if key.isdigit() else (1, key)


def _classification_to_cache_entry(classification: PageMetadataClassification) -> dict[str, Any]:
    return {
        "party_type": classification.party_type,
        "location": classification.location,
        "confidence": classification.confidence or {
            "party_type": 0.0,
            "location": 0.0,
        },
    }


def _page_cache_key(document: Document, fallback_index: int) -> str:
    page = document.metadata.get("page")
    if page is None:
        return str(fallback_index)
    return str(page)


def _merge_classification_metadata(
    document: Document,
    classification: PageMetadataClassification,
    metadata_source: str,
) -> Document:
    metadata = dict(document.metadata)
    confidence = classification.confidence or {}

    if classification.party_type is not None:
        metadata["party_type"] = classification.party_type
    if classification.location is not None:
        metadata["location"] = classification.location

    metadata["metadata_source"] = metadata_source
    metadata["metadata_confidence_party_type"] = confidence.get("party_type", 0.0)
    metadata["metadata_confidence_location"] = confidence.get("location", 0.0)

    return Document(page_content=document.page_content, metadata=metadata)


def enrich_document_with_llm_metadata(document: Document, llm: Any | None = None) -> Document:
    """Return a copy of a page document with classified metadata merged in."""
    classification = classify_page_metadata(document.page_content, llm=llm)
    return _merge_classification_metadata(document, classification, metadata_source="llm")


def enrich_documents_with_llm_metadata(
    documents: list[Document],
    llm: Any | None = None,
    cache_path: Path | None = None,
) -> list[Document]:
    """Classify and enrich each page document before chunking, reusing a cache if provided."""
    cache = load_page_metadata_cache(cache_path) if cache_path is not None else {}
    cache_changed = False
    enriched_documents: list[Document | None] = [None] * len(documents)
    cache_misses: list[tuple[int, str, Document]] = []
    classifier_llm = llm

    for index, document in enumerate(documents, start=1):
        cache_key = _page_cache_key(document, index)
        cached_entry = cache.get(cache_key)
        if cached_entry is not None:
            classification = normalize_page_metadata_response(cached_entry)
            enriched_documents[index - 1] = _merge_classification_metadata(
                document,
                classification,
                metadata_source="llm_cache",
            )
            continue

        cache_misses.append((index - 1, cache_key, document))

    if classifier_llm is None and any(document.page_content.strip() for _, _, document in cache_misses):
        classifier_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    worker_count = min(MAX_CLASSIFICATION_WORKERS, len(cache_misses))
    if worker_count:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            classification_results = executor.map(
                lambda item: classify_page_metadata_safely(item[2], llm=classifier_llm),
                cache_misses,
            )

            for (document_index, cache_key, document), (classification, had_error) in zip(
                cache_misses,
                classification_results,
                strict=True,
            ):
                if cache_path is not None:
                    cache[cache_key] = _classification_to_cache_entry(classification)
                    cache_changed = True
                enriched_documents[document_index] = _merge_classification_metadata(
                    document,
                    classification,
                    metadata_source="llm_error" if had_error else "llm",
                )

    if cache_path is not None and cache_changed:
        save_page_metadata_cache(cache_path, cache)

    return [document for document in enriched_documents if document is not None]
