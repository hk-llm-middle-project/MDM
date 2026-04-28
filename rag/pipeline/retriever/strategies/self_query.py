"""Self-query 검색 전략 자리표시자."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from rag.pipeline.retriever.components import RetrievalComponents
from rag.service.tracing import TraceContext


@dataclass(frozen=True)
class SelfQueryRetrieverConfig:
    """향후 self-query 검색 전략을 위한 설정 자리표시자."""


def retrieve_with_self_query(
    components: RetrievalComponents,
    query: str,
    k: int,
    filters: dict[str, object] | None = None,
    strategy_config: SelfQueryRetrieverConfig | None = None,
    trace_context: TraceContext | None = None,
) -> list[Document]:
    """향후 SelfQueryRetriever 연동을 위한 자리표시자 함수입니다."""
    del components, query, k, filters, strategy_config, trace_context
    raise NotImplementedError(
        "SelfQueryRetriever는 아직 연결되지 않았습니다. 메타데이터 필드와 질의 생성기를 준비한 뒤 활성화하세요."
    )
