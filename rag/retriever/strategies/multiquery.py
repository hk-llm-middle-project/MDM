"""다중 질의 검색 전략."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import LLM_MODEL


@dataclass(frozen=True)
class MultiQueryRetrieverConfig:
    """다중 질의 검색 전략 설정."""

    llm: Any | None = None
    model: str = LLM_MODEL
    include_original: bool = True
    dense_k: int | None = None
    search_type: str = "similarity"


def retrieve_with_multiquery(
    vectorstore: Any,
    query: str,
    k: int,
    filters: dict[str, object] | None = None,
    strategy_config: MultiQueryRetrieverConfig | None = None,
) -> list[Document]:
    """사용자 질문을 여러 의미적 변형 질의로 확장해 검색합니다."""
    config = strategy_config or MultiQueryRetrieverConfig()
    search_kwargs: dict[str, object] = {"k": config.dense_k or k}
    if filters:
        search_kwargs["filter"] = filters

    base_retriever = vectorstore.as_retriever(
        search_type=config.search_type,
        search_kwargs=search_kwargs,
    )
    llm = config.llm or ChatOpenAI(model=config.model)
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        include_original=config.include_original,
    )
    return list(retriever.invoke(query))[:k]
