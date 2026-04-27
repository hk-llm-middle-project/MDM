"""Fixed-size document chunking utilities."""

from langchain_core.documents import Document

from config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into fixed-size chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    clean_text = text.strip()
    if not clean_text:
        return []

    step = chunk_size - overlap
    return [clean_text[start : start + chunk_size] for start in range(0, len(clean_text), step)]


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into fixed-size chunks while preserving metadata."""
    chunks: list[Document] = []
    for document in documents:
        for chunk in chunk_text(document.page_content, chunk_size=chunk_size, overlap=overlap):
            chunks.append(Document(page_content=chunk, metadata=dict(document.metadata)))
    return chunks
