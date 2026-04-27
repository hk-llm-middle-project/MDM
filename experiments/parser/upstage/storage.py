"""Utilities for reading and writing LangChain Document lists as JSON."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document


def load_documents_json(json_path: str | Path) -> list[Document]:
    """Load documents saved as [{"page_content": ..., "metadata": ...}, ...]."""
    path = Path(json_path)

    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    docs = [
        Document(
            page_content=item.get("page_content", ""),
            metadata=item.get("metadata", {}),
        )
        for item in payload
    ]
    print(f"[INFO] load_documents_json - 로드 완료: {path}")
    return docs


def save_documents_json(docs: list[Document], output_path: str | Path) -> Path:
    """Save LangChain documents to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]

    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    print(f"[INFO] save_documents_json - 저장 완료: {path}")
    return path
