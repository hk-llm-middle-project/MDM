"""Retrieval pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from config import RETRIEVER_K
from rag.pipeline.reranker import RerankerConfig, rerank
from rag.pipeline.retriever import RetrievalComponents, StrategyConfig, retrieve
from rag.service.tracing import TraceContext


@dataclass(frozen=True)
class RetrievalPipelineConfig:
    """Retriever와 reranker를 묶어 실행하는 파이프라인 설정."""

    retriever_strategy: str = "vectorstore"
    retriever_config: StrategyConfig | None = None
    reranker_strategy: str = "none"
    reranker_config: RerankerConfig | None = None
    final_k: int = RETRIEVER_K
    candidate_k: int | None = None


def run_retrieval_pipeline(
    components: RetrievalComponents,
    query: str,
    filters: dict[str, object] | None = None,
    pipeline_config: RetrievalPipelineConfig | None = None,
    trace_context: TraceContext | None = None,
) -> list[Document]:
    """Retriever 결과에 선택적 reranker를 적용해 최종 문서를 반환합니다."""
    config = pipeline_config or RetrievalPipelineConfig()
    candidate_k = config.candidate_k or config.final_k
    retrieve_kwargs = {
        "components": components,
        "query": query,
        "k": candidate_k,
        "strategy": config.retriever_strategy,
        "filters": filters,
        "strategy_config": config.retriever_config,
    }
    if trace_context is not None:
        retrieve_kwargs["trace_context"] = trace_context
    candidate_documents = retrieve(**retrieve_kwargs)
    if filters is not None and not candidate_documents:
        candidate_documents = retrieve(**{**retrieve_kwargs, "filters": None})

    rerank_kwargs = {
        "query": query,
        "documents": candidate_documents,
        "k": config.final_k,
        "strategy": config.reranker_strategy,
        "strategy_config": config.reranker_config,
    }
    if trace_context is not None:
        rerank_kwargs["trace_context"] = trace_context
    return rerank(**rerank_kwargs)
