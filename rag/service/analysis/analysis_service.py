"""사고 질의를 분석하고 RAG 답변을 생성합니다."""

from langchain_openai import ChatOpenAI

from config import (
    DEFAULT_CHUNKER_STRATEGY,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LOADER_STRATEGY,
    LLM_MODEL,
)
from rag.pipeline.retrieval import RetrievalPipelineConfig, run_retrieval_pipeline
from rag.service.analysis.answer_schema import AnalysisResult, RetrievedContext, parse_structured_answer
from rag.service.analysis.prompt import build_prompt
from rag.service.intake.filter_service import build_metadata_filters
from rag.service.intake.schema import UserSearchMetadata
from rag.service.session.schema import ChatMessage
from rag.service.tracing import TraceContext
from rag.service.vectorstore.vectorstore_service import get_retrieval_components


def analyze_question(
    question: str,
    search_metadata: UserSearchMetadata | None = None,
    pipeline_config: RetrievalPipelineConfig | None = None,
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    chat_history: list[ChatMessage] | None = None,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    chunker_strategy: str = DEFAULT_CHUNKER_STRATEGY,
    trace_context: TraceContext | None = None,
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
    documents = run_retrieval_pipeline(components, retrieval_query, **retrieval_kwargs)
    retrieved_contexts = [
        RetrievedContext(
            content=document.page_content,
            metadata=dict(document.metadata),
        )
        for document in documents
    ]
    contexts = [context.content for context in retrieved_contexts]
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"[retrieved] question={question}")
    if retrieval_query != question:
        logger.info(f"[retrieved] retrieval_query={retrieval_query}")
    for index, context in enumerate(contexts, start=1):
        logger.info(f"[retrieved:{index}] {context}")

    prompt = build_prompt(question, "\n\n".join(contexts), chat_history=chat_history)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    config = trace_context.langchain_config("mdm.answer") if trace_context else None
    response = llm.invoke(prompt, config=config) if config else llm.invoke(prompt)
    content = response.content
    structured_answer = parse_structured_answer(str(content))
    return AnalysisResult(
        response=structured_answer.response,
        contexts=contexts,
        retrieved_contexts=retrieved_contexts,
        fault_ratio_a=structured_answer.fault_ratio_a,
        fault_ratio_b=structured_answer.fault_ratio_b,
    )
