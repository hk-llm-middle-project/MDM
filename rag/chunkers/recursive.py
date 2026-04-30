"""Recursive character chunker implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE
from rag.chunkers.base import BaseChunker
from rag.chunkers.schema import Chunk


DEFAULT_SEPARATORS = ["\n\n", "\n", ".", " "]


@dataclass(frozen=True)
class RecursiveCharacterChunker(BaseChunker):
    chunk_size: int = CHUNK_SIZE
    overlap: int = CHUNK_OVERLAP
    separators: list[str] = field(default_factory=lambda: list(DEFAULT_SEPARATORS))

    def chunk(self, parsed_input: str | Document | Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=self.separators,
        )
        for document in self._documents_for(parsed_input):
            for text in splitter.split_text(document.page_content):
                if not text.strip():
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=len(chunks),
                        text=text.strip(),
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
