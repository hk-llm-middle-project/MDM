"""사고 질의를 분석하고 RAG 답변을 생성합니다."""

import json
import logging

from langchain_openai import ChatOpenAI

from config import (
    DEFAULT_CHUNKER_STRATEGY,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LOADER_STRATEGY,
    LLM_MODEL,
)
from rag.pipeline.retrieval import RetrievalPipelineConfig, run_retrieval_pipeline
from rag.pipeline.retriever.common import get_embedding_function_from_vectorstore
from rag.service.analysis.answer_schema import AnalysisResult, RetrievedContext, parse_structured_answer
from rag.service.analysis.prompt import build_prompt
from rag.service.intake.filter_service import build_metadata_filters
from rag.service.intake.schema import UserSearchMetadata
from rag.service.progress import (
    PROGRESS_ANSWER,
    PROGRESS_RETRIEVAL,
    ProgressCallback,
    report_progress,
    report_progress_detail,
)
from rag.service.session.schema import ChatMessage
from rag.service.tracing import TraceContext
from rag.service.vectorstore.vectorstore_service import get_retrieval_components


logger = logging.getLogger(__name__)


def _format_filter_detail(filters: dict[str, object] | None) -> str:
    if filters is None:
        return "없음"
    return json.dumps(filters, ensure_ascii=False, sort_keys=True)


def _format_document_detail(index: int, context: RetrievedContext) -> str:
    metadata = context.metadata
    parts = [f"상위 근거 {index}"]
    for key in ("diagram_id", "page", "chunk_type", "party_type", "location"):
        value = metadata.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    if len(parts) == 1:
        return f"{parts[0]}: metadata 없음"
    return f"{parts[0]}: {', '.join(parts[1:])}"


def _get_query_cache_stats(components) -> tuple[object | None, int, int]:
    try:
        embedding_function = get_embedding_function_from_vectorstore(components.vectorstore)
    except (AttributeError, ValueError):
        return None, 0, 0
    hits = int(getattr(embedding_function, "query_cache_hits", 0) or 0)
    misses = int(getattr(embedding_function, "query_cache_misses", 0) or 0)
    return embedding_function, hits, misses


def analyze_question(
    question: str,
    search_metadata: UserSearchMetadata | None = None,
    pipeline_config: RetrievalPipelineConfig | None = None,
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    chat_history: list[ChatMessage] | None = None,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    chunker_strategy: str = DEFAULT_CHUNKER_STRATEGY,
    trace_context: TraceContext | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisResult:
    """질문을 검색하고 LLM 답변과 검색 컨텍스트를 반환합니다."""
    components = get_retrieval_components(
        loader_strategy,
        embedding_provider,
        chunker_strategy,
    )
    filters = build_metadata_filters(search_metadata)
    retrieval_query = (
        search_metadata.retrieval_query
        if search_metadata is not None and search_metadata.retrieval_query
        else question
    )
    retrieval_kwargs = {
        "filters": filters,
        "pipeline_config": pipeline_config,
    }
    if trace_context is not None:
        retrieval_kwargs["trace_context"] = trace_context
    if progress_callback is not None:
        retrieval_kwargs["progress_callback"] = progress_callback
    embedding_function, cache_hits_before, cache_misses_before = _get_query_cache_stats(components)
    report_progress(progress_callback, PROGRESS_RETRIEVAL)
    report_progress_detail(progress_callback, f"검색 질의: {retrieval_query}")
    report_progress_detail(
        progress_callback,
        f"적용 필터: {_format_filter_detail(filters)}",
    )
    documents = run_retrieval_pipeline(components, retrieval_query, **retrieval_kwargs)
    if embedding_function is not None:
        cache_hits_after = int(getattr(embedding_function, "query_cache_hits", 0) or 0)
        cache_misses_after = int(getattr(embedding_function, "query_cache_misses", 0) or 0)
        cache_hits = cache_hits_after - cache_hits_before
        cache_misses = cache_misses_after - cache_misses_before
        if cache_hits > 0:
            logger.info(
                "[embedding-query-cache] hit provider=%s query=%s hits=%s misses=%s",
                embedding_provider,
                retrieval_query,
                cache_hits,
                cache_misses,
            )
    retrieved_contexts = [
        RetrievedContext(
            content=document.page_content,
            metadata=dict(document.metadata),
        )
        for document in documents
    ]
    report_progress_detail(progress_callback, f"검색 결과: {len(retrieved_contexts)}개")
    for index, context in enumerate(retrieved_contexts[:3], start=1):
        report_progress_detail(progress_callback, _format_document_detail(index, context))
    contexts = [context.content for context in retrieved_contexts]
    logger.info(f"[retrieved] question={question}")
    if retrieval_query != question:
        logger.info(f"[retrieved] retrieval_query={retrieval_query}")
    for index, context in enumerate(contexts, start=1):
        logger.info(f"[retrieved:{index}] {context}")

    report_progress(progress_callback, PROGRESS_ANSWER)
    prompt = build_prompt(question, "\n\n".join(contexts), chat_history=chat_history)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    config = trace_context.langchain_config("mdm.answer") if trace_context else None
    response = llm.invoke(prompt, config=config) if config else llm.invoke(prompt)
    content = response.content
    structured_answer = parse_structured_answer(str(content))
    if (
        structured_answer.fault_ratio_a is None
        or structured_answer.fault_ratio_b is None
    ):
        report_progress_detail(progress_callback, "과실비율: 판단 보류")
    else:
        report_progress_detail(
            progress_callback,
            (
                f"과실비율: A {structured_answer.fault_ratio_a}% / "
                f"B {structured_answer.fault_ratio_b}%"
            ),
        )
    return AnalysisResult(
        response=structured_answer.response,
        contexts=contexts,
        retrieved_contexts=retrieved_contexts,
        fault_ratio_a=structured_answer.fault_ratio_a,
        fault_ratio_b=structured_answer.fault_ratio_b,
    )
