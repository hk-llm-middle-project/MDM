"""리랭커 전략에서 공통으로 사용하는 도우미 함수들."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.utils.json import parse_json_markdown


def build_scored_document(document: Document, score: float | None) -> Document:
    """원본 문서 메타데이터를 유지하면서 리랭크 점수를 추가합니다."""
    metadata = dict(document.metadata)
    if score is not None:
        metadata["rerank_score"] = score
    return Document(page_content=document.page_content, metadata=metadata)


def parse_json_response(content: str) -> dict[str, object]:
    """LLM 응답에서 JSON 객체를 파싱합니다."""
    parsed = parse_json_markdown(content.strip())
    if not isinstance(parsed, dict):
        raise ValueError("리랭커 응답이 JSON 객체가 아닙니다.")
    return parsed
