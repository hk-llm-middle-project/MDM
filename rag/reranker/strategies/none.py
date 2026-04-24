"""동작 없는 리랭커 전략."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class NoOpRerankerConfig:
    """기존 retriever 순서를 그대로 유지하는 리랭커 설정."""


def rerank_with_none(
    query: str,
    documents: list[Document],
    k: int,
    strategy_config: NoOpRerankerConfig | None = None,
) -> list[Document]:
    """입력 순서를 유지한 채 상위 k개만 반환합니다."""
    del query, strategy_config
    return list(documents[:k])
