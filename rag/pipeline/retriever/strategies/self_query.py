"""Self-query retriever strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_openai import ChatOpenAI

from config import LLM_MODEL
from rag.pipeline.retriever.components import RetrievalComponents
from rag.service.intake.values import LOCATIONS, PARTY_TYPES
from rag.service.tracing import TraceContext


@dataclass(frozen=True)
class SelfQueryRetrieverConfig:
    """Configuration for metadata-aware self-query retrieval."""

    llm_model: str = LLM_MODEL
    search_type: str = "similarity"
    document_contents: str = (
        "자동차 사고 과실비율 인정기준 문서 조각. "
        "각 문서는 사고 당사자 유형과 사고 장소/유형 메타데이터를 가진다."
    )
    metadata_field_info: tuple[AttributeInfo, ...] = field(
        default_factory=lambda: (
            AttributeInfo(
                name="party_type",
                description=(
                    "사고 당사자 유형. 반드시 다음 값 중 하나만 사용한다: "
                    f"{', '.join(PARTY_TYPES)}"
                ),
                type="string",
            ),
            AttributeInfo(
                name="location",
                description=(
                    "사고 장소 또는 사고 유형. 반드시 다음 값 중 하나만 사용한다: "
                    f"{', '.join(LOCATIONS)}"
                ),
                type="string",
            ),
        )
    )
    use_original_query: bool = True
    verbose: bool = False


def _comparison_to_extracted_filter(filter_value: Any) -> dict[str, str]:
    extracted: dict[str, str] = {}
    if not isinstance(filter_value, dict):
        return extracted

    for key, value in filter_value.items():
        if key == "$and" and isinstance(value, list):
            for item in value:
                extracted.update(_comparison_to_extracted_filter(item))
            continue
        if key == "$or" or key not in {"party_type", "location"}:
            continue

        candidate = value.get("$eq") if isinstance(value, dict) else value
        if isinstance(candidate, str):
            extracted[key] = candidate

    return extracted


def _valid_extracted_filters(search_kwargs: dict[str, Any]) -> dict[str, str] | None:
    extracted = _comparison_to_extracted_filter(search_kwargs.get("filter"))
    valid: dict[str, str] = {}

    party_type = extracted.get("party_type")
    if party_type in PARTY_TYPES:
        valid["party_type"] = party_type

    location = extracted.get("location")
    if location in LOCATIONS:
        valid["location"] = location

    return valid or None


def _record_extracted_filters(
    extracted_filters: dict[str, str] | None,
    trace_context: TraceContext | None,
) -> None:
    if trace_context is None:
        return

    payload = {"extracted_filters": extracted_filters}
    recorder = RunnableLambda(lambda value: value)
    recorder.invoke(
        payload,
        config=trace_context.langchain_config("mdm.retrieve.selfquery.extracted_filters"),
    )


def _as_filter_conditions(filter_value: dict[str, Any]) -> list[dict[str, Any]]:
    if set(filter_value) == {"$and"} and isinstance(filter_value["$and"], list):
        return list(filter_value["$and"])
    return [filter_value]


def _merge_filters(
    extracted_filter: dict[str, Any],
    external_filter: dict[str, object] | None,
) -> dict[str, Any]:
    if not external_filter:
        return extracted_filter

    return {
        "$and": [
            *_as_filter_conditions(dict(external_filter)),
            *_as_filter_conditions(extracted_filter),
        ]
    }


def _to_chroma_filter(extracted_filters: dict[str, str]) -> dict[str, Any]:
    conditions = [{key: {"$eq": value}} for key, value in extracted_filters.items()]
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve_with_self_query(
    components: RetrievalComponents,
    query: str,
    k: int,
    filters: dict[str, object] | None = None,
    strategy_config: SelfQueryRetrieverConfig | None = None,
    trace_context: TraceContext | None = None,
) -> list[Document]:
    """Extract metadata filters with self-query and search Chroma with them."""
    config = strategy_config or SelfQueryRetrieverConfig()
    llm = ChatOpenAI(model=config.llm_model, temperature=0)
    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=components.vectorstore,
        document_contents=config.document_contents,
        metadata_field_info=config.metadata_field_info,
        structured_query_translator=ChromaTranslator(),
        use_original_query=config.use_original_query,
        verbose=config.verbose,
        search_type=config.search_type,
    )

    constructor_config = (
        trace_context.langchain_config("mdm.retrieve.selfquery.query_constructor")
        if trace_context
        else None
    )
    structured_query = (
        retriever.query_constructor.invoke({"query": query}, config=constructor_config)
        if constructor_config
        else retriever.query_constructor.invoke({"query": query})
    )
    new_query, search_kwargs = retriever._prepare_query(query, structured_query)
    extracted_filters = _valid_extracted_filters(search_kwargs)
    _record_extracted_filters(extracted_filters, trace_context)

    if extracted_filters is None:
        return []

    search_kwargs["k"] = k
    search_kwargs["filter"] = _merge_filters(_to_chroma_filter(extracted_filters), filters)
    return list(retriever._get_docs_with_query(new_query, search_kwargs))
