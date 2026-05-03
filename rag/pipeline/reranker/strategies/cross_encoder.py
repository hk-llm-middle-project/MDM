"""HuggingFace CrossEncoder 리랭커 전략."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from rag.pipeline.reranker.strategies.common import build_scored_document
from rag.service.tracing import TraceContext


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
# DEFAULT_CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class CrossEncoderRerankerConfig:
    """HuggingFace CrossEncoder 기반 리랭커 설정."""

    model_name: str = DEFAULT_CROSS_ENCODER_MODEL
    top_n: int | None = None
    model: Any | None = None


@lru_cache(maxsize=2)
def get_cross_encoder_model(model_name: str):
    """CrossEncoder 모델을 프로세스 단위로 캐시합니다."""
    try:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    except ImportError as error:
        raise ImportError(
            "cross-encoder 리랭커를 사용하려면 `langchain-community`와 "
            "`sentence-transformers` 패키지가 설치되어 있어야 합니다."
        ) from error

    return HuggingFaceCrossEncoder(model_name=model_name)


def rerank_with_cross_encoder(
    query: str,
    documents: list[Document],
    k: int,
    strategy_config: CrossEncoderRerankerConfig | None = None,
    trace_context: TraceContext | None = None,
) -> list[Document]:
    """HuggingFace CrossEncoder로 문서를 재정렬한 뒤 상위 k개를 반환합니다."""
    if not documents:
        return []

    try:
        from langchain.retrievers.document_compressors import CrossEncoderReranker
    except ImportError as error:
        raise ImportError(
            "cross-encoder 리랭커를 사용하려면 LangChain document compressor가 필요합니다."
        ) from error

    del trace_context
    config = strategy_config or CrossEncoderRerankerConfig()
    model = config.model or get_cross_encoder_model(config.model_name)
    reranker = CrossEncoderReranker(
        model=model,
        top_n=config.top_n or k,
    )
    ranked_documents = reranker.compress_documents(
        documents=documents,
        query=query,
    )

    return [
        build_scored_document(document, document.metadata.get("relevance_score"))
        for document in list(ranked_documents)[:k]
    ]
