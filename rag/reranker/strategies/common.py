"""리랭커 전략에서 공통으로 사용하는 도우미 함수들."""

from __future__ import annotations

import json

from langchain_core.documents import Document


def build_scored_document(document: Document, score: float | None) -> Document:
    """원본 문서 메타데이터를 유지하면서 리랭크 점수를 추가합니다."""
    metadata = dict(document.metadata)
    if score is not None:
        metadata["rerank_score"] = score
    return Document(page_content=document.page_content, metadata=metadata)


def parse_json_response(content: str) -> dict[str, object]:
    """LLM 응답에서 JSON 객체를 파싱합니다."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("리랭커 응답이 JSON 객체가 아닙니다.")
    return parsed
