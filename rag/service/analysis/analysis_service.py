"""사고 질의를 분석하고 RAG 답변을 생성합니다."""

from langchain_openai import ChatOpenAI

from config import DEFAULT_EMBEDDING_PROVIDER, DEFAULT_LOADER_STRATEGY, LLM_MODEL
from rag.pipeline.retrieval import RetrievalPipelineConfig, run_retrieval_pipeline
from rag.service.analysis.answer_schema import parse_structured_answer
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
    trace_context: TraceContext | None = None,
) -> tuple[str, list[str]]:
    """질문을 검색하고 LLM 답변과 검색 컨텍스트를 반환합니다."""
    components = get_retrieval_components(loader_strategy, embedding_provider)
    filters = build_metadata_filters(search_metadata)
    retrieval_kwargs = {
        "filters": filters,
        "pipeline_config": pipeline_config,
    }
    if trace_context is not None:
        retrieval_kwargs["trace_context"] = trace_context
    documents = run_retrieval_pipeline(components, question, **retrieval_kwargs)
    contexts = [document.page_content for document in documents]
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"[retrieved] question={question}")
    for index, context in enumerate(contexts, start=1):
        logger.info(f"[retrieved:{index}] {context}")

    prompt = build_prompt(question, "\n\n".join(contexts), chat_history=chat_history)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    config = trace_context.langchain_config("mdm.answer") if trace_context else None
    response = llm.invoke(prompt, config=config) if config else llm.invoke(prompt)
    content = response.content
    structured_answer = parse_structured_answer(str(content))
    return structured_answer.response, contexts
