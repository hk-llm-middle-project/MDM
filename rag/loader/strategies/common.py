"""PDF 로더 전략에서 공통으로 사용하는 도우미 함수들."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document


def build_page_document(
    path: Path,
    page_content: str,
    page_number: int,
    parser_name: str,
) -> Document:
    return Document(
        page_content=page_content,
        metadata={
            "source": str(path),
            "page": page_number,
            "parser": parser_name,
        },
    )
