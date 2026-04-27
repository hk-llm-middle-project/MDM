"""사고 질의 분석과 RAG 답변 생성을 담당합니다."""

from langchain_openai import ChatOpenAI

from config import LLM_MODEL
from rag.pipeline.retrieval import RetrievalPipelineConfig, run_retrieval_pipeline
from rag.service.prompt import build_prompt
from rag.service.vectorstore_service import get_retrieval_components


def analyze_question(
    question: str,
    pipeline_config: RetrievalPipelineConfig | None = None,
) -> tuple[str, list[str]]:
    """질문을 검색하고 LLM 답변과 검색 컨텍스트를 반환합니다."""
    components = get_retrieval_components()
    documents = run_retrieval_pipeline(components, question, pipeline_config=pipeline_config)
    contexts = [document.page_content for document in documents]
    print(f"[retrieved] question={question}")
    for index, context in enumerate(contexts, start=1):
        print(f"[retrieved:{index}] {context}")

    prompt = build_prompt(question, "\n\n".join(contexts))
    llm = ChatOpenAI(model=LLM_MODEL)
    answer = llm.invoke(prompt).content
    return answer, contexts
