"""Chunk cache persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from config import BASE_DIR


CHUNKS_FILENAME = "chunks.json"
PREVIEW_FILENAME = "preview.md"


def chunk_cache_exists(cache_dir: Path) -> bool:
    return (cache_dir / CHUNKS_FILENAME).exists()


def load_chunk_cache(cache_dir: Path, *, source_path: Path | None = None) -> list[Document]:
    payload = json.loads((cache_dir / CHUNKS_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Chunk cache must be a JSON array: {cache_dir / CHUNKS_FILENAME}")
    documents = [_dict_to_document(item) for item in payload if isinstance(item, dict)]
    return _normalize_document_sources(documents, source_path)


def save_chunk_cache(
    documents: list[Document],
    cache_dir: Path,
    *,
    source_path: Path | None = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized_documents = _normalize_document_sources(documents, source_path)
    payload = [_document_to_dict(document) for document in normalized_documents]
    (cache_dir / CHUNKS_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (cache_dir / PREVIEW_FILENAME).write_text(
        build_chunk_preview(normalized_documents),
        encoding="utf-8",
    )


def build_chunk_preview(documents: list[Document]) -> str:
    sections = ["# Chunk Preview", ""]
    for index, document in enumerate(documents, start=1):
        metadata = dict(document.metadata)
        sections.append(f"## Chunk {index}")
        if metadata:
            for key in sorted(metadata):
                sections.append(f"- `{key}`: `{metadata[key]}`")
        else:
            sections.append("- metadata: 없음")
        sections.append("")
        sections.append("```text")
        sections.append(document.page_content.strip())
        sections.append("```")
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def _document_to_dict(document: Document) -> dict[str, object]:
    return {
        "page_content": document.page_content,
        "metadata": dict(document.metadata),
    }


def _dict_to_document(payload: dict[str, object]) -> Document:
    return Document(
        page_content=str(payload.get("page_content", "")),
        metadata=dict(payload.get("metadata") or {}),
    )


def _normalize_document_sources(
    documents: list[Document],
    source_path: Path | None,
) -> list[Document]:
    if source_path is None:
        return documents

    try:
        normalized_source = str(source_path.relative_to(BASE_DIR))
    except ValueError:
        normalized_source = str(source_path)
    normalized_documents: list[Document] = []
    for document in documents:
        metadata = dict(document.metadata)
        metadata["source"] = normalized_source
        normalized_documents.append(
            Document(
                page_content=document.page_content,
                metadata=metadata,
            )
        )
    return normalized_documents
