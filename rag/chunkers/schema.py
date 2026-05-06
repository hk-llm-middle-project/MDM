"""Standard chunk schema and LangChain conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class Chunk:
    chunk_id: int
    text: str
    chunk_type: str
    page: int
    source: str
    diagram_id: str | None
    parent_id: int | None
    location: str | None
    party_type: str | None
    image_path: str | None


def chunk_to_document(chunk: Chunk) -> Document:
    metadata = {
        "chunk_id": chunk.chunk_id,
        "chunk_type": chunk.chunk_type,
        "page": chunk.page,
        "source": chunk.source,
        "diagram_id": chunk.diagram_id,
        "parent_id": chunk.parent_id,
        "location": chunk.location,
        "party_type": chunk.party_type,
        "image_path": chunk.image_path,
    }
    return Document(
        page_content=chunk.text,
        metadata={key: value for key, value in metadata.items() if value is not None},
    )
