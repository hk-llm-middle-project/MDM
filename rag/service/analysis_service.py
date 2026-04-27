"""사고 질의 분석과 RAG 답변 생성을 담당합니다."""

from langchain_openai import ChatOpenAI

from config import LLM_MODEL
from rag.pipeline.retrieval import RetrievalPipelineConfig, run_retrieval_pipeline
from rag.service.vectorstore_service import get_retrieval_components


def build_prompt(question: str, context: str) -> str:
    """검색된 공식 문서 조각과 사용자 질문으로 답변 프롬프트를 만듭니다."""
    return f"""
아래 공식 문서 내용을 참고해서 답변해.

[공식 문서 내용]
{context}

[사용자 질문]
{question}

아래 형식으로 답변해.
1. 의심 사고유형
2. 관련 공식 문서 근거
3. 수정요소 후보
4. 예상 과실비율
5. 설명
""".strip()


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

