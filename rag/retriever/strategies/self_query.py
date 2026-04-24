"""Self-query 검색 전략 자리표시자."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass(frozen=True)
class SelfQueryRetrieverConfig:
    """향후 self-query 검색 전략을 위한 설정 자리표시자."""


def retrieve_with_self_query(
    vectorstore: Any,
    query: str,
    k: int,
    filters: dict[str, object] | None = None,
    strategy_config: SelfQueryRetrieverConfig | None = None,
) -> list[Document]:
    """향후 SelfQueryRetriever 연동을 위한 자리표시자 함수입니다."""
    raise NotImplementedError(
        "SelfQueryRetriever는 아직 연결되지 않았습니다. 메타데이터 필드와 질의 생성기를 준비한 뒤 활성화하세요."
    )
