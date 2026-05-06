"""Fixed-size chunker implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document

from config import CHUNK_OVERLAP, CHUNK_SIZE
from rag.chunker import chunk_text
from rag.chunkers.base import BaseChunker
from rag.chunkers.schema import Chunk


@dataclass(frozen=True)
class FixedSizeChunker(BaseChunker):
    chunk_size: int = CHUNK_SIZE
    overlap: int = CHUNK_OVERLAP

    def chunk(self, parsed_input: str | Document | Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in self._documents_for(parsed_input):
            for text in chunk_text(
                document.page_content,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            ):
                chunks.append(
                    Chunk(
                        chunk_id=len(chunks),
                        text=text,
                        chunk_type="flat",
                        page=self._page_for(document),
                        source=str(document.metadata.get("source", "")),
                        diagram_id=None,
                        parent_id=None,
                        location=None,
                        party_type=None,
                        image_path=None,
                    )
                )
        return chunks

    def _documents_for(self, parsed_input: str | Document | Iterable[Document]) -> list[Document]:
        if isinstance(parsed_input, str):
            return [Document(page_content=parsed_input, metadata={})]
        if isinstance(parsed_input, Document):
            return [parsed_input]
        return list(parsed_input)

    def _page_for(self, document: Document) -> int:
        page = document.metadata.get("page", 0)
        if isinstance(page, int):
            return page
        if isinstance(page, str) and page.isdigit():
            return int(page)
        return 0
