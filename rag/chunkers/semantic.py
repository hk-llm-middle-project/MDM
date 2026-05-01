"""Semantic chunker that compares adjacent sentences across page boundaries."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from langchain_core.documents import Document

from rag.chunkers.base import BaseChunker
from rag.chunkers.schema import Chunk


EmbeddingFunction = Callable[[list[str]], list[list[float]]]

SENTENCE_RE = re.compile(r"[^.!?。！？]+(?:[.!?。！？]+|$)", re.MULTILINE)
DIAGRAM_ID_RE = re.compile(r"\b(?:차|보|거)\d+(?:-\d+)?(?:\([가-힣]\))?")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


class SupportsEmbedDocuments(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for texts."""


@dataclass(frozen=True)
class SentenceSpan:
    text: str
    page: int
    source: str


@dataclass(frozen=True)
class SemanticChunker(BaseChunker):
    embedding_function: EmbeddingFunction | SupportsEmbedDocuments | None = None
    breakpoint_threshold: float = 0.35
    min_chunk_chars: int = 0
    max_chunk_chars: int | None = 1200

    def chunk(self, parsed_input: str | Document | Iterable[Document]) -> list[Chunk]:
        sentences = self._sentence_spans_for(parsed_input)
        if not sentences:
            return []

        embeddings = self._embed([sentence.text for sentence in sentences])
        if len(embeddings) != len(sentences):
            raise ValueError("Semantic embedding function must return one vector per sentence")

        chunks: list[Chunk] = []
        current: list[SentenceSpan] = [sentences[0]]
        for index in range(1, len(sentences)):
            candidate = sentences[index]
            distance = self._cosine_distance(embeddings[index - 1], embeddings[index])
            should_break = (
                distance > self.breakpoint_threshold
                and self._text_length(current) >= self.min_chunk_chars
            )
            if self.max_chunk_chars is not None:
                should_break = should_break or (
                    self._text_length([*current, candidate]) > self.max_chunk_chars
                    and self._text_length(current) >= self.min_chunk_chars
                )

            if should_break:
                chunks.append(self._chunk_from_spans(len(chunks), current))
                current = [candidate]
            else:
                current.append(candidate)

        if current:
            chunks.append(self._chunk_from_spans(len(chunks), current))
        return chunks

    def _documents_for(self, parsed_input: str | Document | Iterable[Document]) -> list[Document]:
        if isinstance(parsed_input, str):
            return [Document(page_content=parsed_input, metadata={})]
        if isinstance(parsed_input, Document):
            return [parsed_input]
        return list(parsed_input)

    def _sentence_spans_for(
        self,
        parsed_input: str | Document | Iterable[Document],
    ) -> list[SentenceSpan]:
        spans: list[SentenceSpan] = []
        for document in self._documents_for(parsed_input):
            page = self._page_for(document)
            source = str(document.metadata.get("source", ""))
            for text in self._split_sentences(document.page_content):
                spans.append(SentenceSpan(text=text, page=page, source=source))
        return spans

    def _split_sentences(self, text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        protected, images = self._protect_markdown_images(normalized)
        sentences = []
        for match in SENTENCE_RE.finditer(protected):
            sentence = match.group(0).strip()
            if not sentence:
                continue
            sentences.append(self._restore_markdown_images(sentence, images))
        return sentences

    def _protect_markdown_images(self, text: str) -> tuple[str, list[str]]:
        images: list[str] = []

        def replace(match: re.Match[str]) -> str:
            images.append(match.group(0))
            return f"__MD_IMAGE_{len(images) - 1}__"

        return IMAGE_RE.sub(replace, text), images

    def _restore_markdown_images(self, text: str, images: list[str]) -> str:
        restored = text
        for index, image in enumerate(images):
            restored = restored.replace(f"__MD_IMAGE_{index}__", image)
        return restored

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.embedding_function is None:
            return [_lexical_embedding(text) for text in texts]
        if hasattr(self.embedding_function, "embed_documents"):
            return self.embedding_function.embed_documents(texts)
        return self.embedding_function(texts)

    def _chunk_from_spans(self, chunk_id: int, spans: list[SentenceSpan]) -> Chunk:
        text = " ".join(span.text for span in spans).strip()
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            chunk_type="flat",
            page=spans[0].page,
            source=spans[0].source,
            diagram_id=self._first_match(DIAGRAM_ID_RE, text),
            parent_id=None,
            location=None,
            party_type=None,
            image_path=self._first_match(IMAGE_RE, text),
        )

    def _page_for(self, document: Document) -> int:
        page = document.metadata.get("page", 0)
        if isinstance(page, int):
            return page
        if isinstance(page, str) and page.isdigit():
            return int(page)
        return 0

    def _text_length(self, spans: list[SentenceSpan]) -> int:
        return sum(len(span.text) for span in spans)

    def _cosine_distance(self, first: list[float], second: list[float]) -> float:
        first_norm = math.sqrt(sum(value * value for value in first))
        second_norm = math.sqrt(sum(value * value for value in second))
        if first_norm == 0 or second_norm == 0:
            return 1.0
        dot = sum(left * right for left, right in zip(first, second))
        similarity = dot / (first_norm * second_norm)
        return 1.0 - max(-1.0, min(1.0, similarity))

    def _first_match(self, pattern: re.Pattern[str], text: str) -> str | None:
        match = pattern.search(text)
        return match.group(1) if pattern.groups and match else match.group(0) if match else None


def _lexical_embedding(text: str, dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    tokens = re.findall(r"[\w가-힣]+", text.lower())
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        vector[index] += 1.0
    return vector
