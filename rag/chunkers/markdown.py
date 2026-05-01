"""Markdown structure chunker for LlamaParse markdown."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from langchain_core.documents import Document

from rag.chunkers.base import BaseChunker
from rag.chunkers.schema import Chunk


DIAGRAM_ID_PATTERN = r"[차보거]\d+(?:-\d+)?"
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


@dataclass(frozen=True)
class MarkdownSection:
    level: int
    title: str
    text: str


@dataclass(frozen=True)
class MarkdownStructureChunker(BaseChunker):
    def chunk(self, parsed_input: str | Document | Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        current_parent_id: int | None = None
        current_parent_level: int | None = None
        current_diagram_id: str | None = None

        for document in self._documents_for(parsed_input):
            for section in self._sections(document.page_content):
                next_diagram_id = self._extract_diagram_id(section.title)
                if next_diagram_id is not None:
                    current_parent_id = len(chunks)
                    current_parent_level = section.level
                    current_diagram_id = next_diagram_id
                    chunks.append(
                        self._make_chunk(
                            chunk_id=current_parent_id,
                            text=section.text,
                            chunk_type="parent",
                            document=document,
                            diagram_id=current_diagram_id,
                            parent_id=None,
                        )
                    )
                    continue

                if current_parent_id is None or current_parent_level is None:
                    continue

                if section.level <= current_parent_level:
                    current_parent_id = None
                    current_parent_level = None
                    current_diagram_id = None
                    continue

                if current_diagram_id is not None:
                    chunks.append(
                        self._make_chunk(
                            chunk_id=len(chunks),
                            text=section.text,
                            chunk_type="child",
                            document=document,
                            diagram_id=current_diagram_id,
                            parent_id=current_parent_id,
                        )
                    )
        return chunks

    def _sections(self, markdown: str) -> list[MarkdownSection]:
        lines = markdown.splitlines()
        heading_indexes = [
            index for index, line in enumerate(lines) if HEADING_PATTERN.match(line.strip())
        ]
        sections: list[MarkdownSection] = []

        for position, start in enumerate(heading_indexes):
            match = HEADING_PATTERN.match(lines[start].strip())
            if match is None:
                continue

            end = heading_indexes[position + 1] if position + 1 < len(heading_indexes) else len(lines)
            text = "\n".join(lines[start:end]).strip()
            if not text:
                continue

            sections.append(
                MarkdownSection(
                    level=len(match.group(1)),
                    title=match.group(2).strip(),
                    text=text,
                )
            )
        return sections

    def _make_chunk(
        self,
        chunk_id: int,
        text: str,
        chunk_type: str,
        document: Document,
        diagram_id: str,
        parent_id: int | None,
    ) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            text=text.strip(),
            chunk_type=chunk_type,
            page=self._page_for(document),
            source=str(document.metadata.get("source", "")),
            diagram_id=diagram_id,
            parent_id=parent_id,
            location=None,
            party_type=None,
            image_path=self._extract_image_path(text),
        )

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

    def _extract_diagram_id(self, text: str) -> str | None:
        match = re.search(DIAGRAM_ID_PATTERN, text)
        return match.group(0) if match else None

    def _extract_image_path(self, text: str) -> str | None:
        match = IMAGE_RE.search(text)
        return match.group(1) if match else None
