"""LLM 점수화 리랭커 전략."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import RERANKER_LLM_MODEL
from rag.pipeline.reranker.prompts import build_llm_score_reranker_prompt
from rag.pipeline.reranker.strategies.common import build_scored_document, parse_json_response
from rag.service.tracing import TraceContext


@dataclass(frozen=True)
class LLMScoreRerankerConfig:
    """LLM이 문서별 관련도를 점수화하는 리랭커 설정."""

    llm: Any | None = None
    model: str = RERANKER_LLM_MODEL
    temperature: float = 0.0


def rerank_with_llm_score(
    query: str,
    documents: list[Document],
    k: int,
    strategy_config: LLMScoreRerankerConfig | None = None,
    trace_context: TraceContext | None = None,
) -> list[Document]:
    """LLM 점수화를 통해 문서를 재정렬한 뒤 상위 k개를 반환합니다."""
    if not documents:
        return []

    config = strategy_config or LLMScoreRerankerConfig()
    llm = config.llm or ChatOpenAI(model=config.model, temperature=config.temperature)
    prompt = build_llm_score_reranker_prompt(query, documents)
    config_dict = trace_context.langchain_config("mdm.rerank.llm_score") if trace_context else None
    response = llm.invoke(prompt, config=config_dict) if config_dict else llm.invoke(prompt)
    content = getattr(response, "content", response)
    parsed = parse_json_response(str(content))

    raw_results = parsed.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("LLM 리랭커 응답에 results 배열이 없습니다.")

    scored_documents: list[tuple[float, int, Document]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        if "id" not in item or "score" not in item:
            continue
        index = int(item["id"])
        if not 0 <= index < len(documents):
            continue
        score = float(item["score"])
        scored_documents.append((score, index, documents[index]))

    if not scored_documents:
        raise ValueError("LLM 리랭커가 유효한 점수를 반환하지 않았습니다.")

    scored_documents.sort(key=lambda row: (-row[0], row[1]))
    return [
        build_scored_document(document, score)
        for score, _, document in scored_documents[:k]
    ]
