"""pdfplumber PDF 로더 전략."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document

from rag.loader.strategies.common import build_page_document


@dataclass(frozen=True)
class PdfPlumberLoaderConfig:
    """pdfplumber 로더 설정."""


def load_with_pdfplumber(
    path: Path,
    strategy_config: PdfPlumberLoaderConfig | None = None,
) -> list[Document]:
    del strategy_config
    documents: list[Document] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    build_page_document(
                        path=path,
                        page_content=text,
                        page_number=page.page_number,
                        parser_name="pdfplumber",
                    )
                )
    return documents
