"""사고 질의 분석과 RAG 답변 생성을 담당합니다."""

from langchain_openai import ChatOpenAI

from config import DEFAULT_LOADER_STRATEGY, LLM_MODEL
from rag.pipeline.retrieval import RetrievalPipelineConfig, run_retrieval_pipeline
from rag.service.intake.filter_service import build_metadata_filters
from rag.service.intake.schema import UserSearchMetadata
from rag.service.prompt import build_prompt
from rag.service.vectorstore_service import get_retrieval_components


def analyze_question(
    question: str,
    search_metadata: UserSearchMetadata | None = None,
    pipeline_config: RetrievalPipelineConfig | None = None,
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
) -> tuple[str, list[str]]:
    """질문을 검색하고 LLM 답변과 검색 컨텍스트를 반환합니다."""
    components = get_retrieval_components(loader_strategy)
    filters = build_metadata_filters(search_metadata)
    documents = run_retrieval_pipeline(
        components,
        question,
        filters=filters,
        pipeline_config=pipeline_config,
    )
    contexts = [document.page_content for document in documents]
    print(f"[retrieved] question={question}")
    for index, context in enumerate(contexts, start=1):
        print(f"[retrieved:{index}] {context}")

    prompt = build_prompt(question, "\n\n".join(contexts))
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    answer = llm.invoke(prompt).content
    return answer, contexts
