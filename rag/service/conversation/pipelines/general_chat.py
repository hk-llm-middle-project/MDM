"""검색 없이 일반 대화와 세션 문맥 질문에 답합니다."""

from collections.abc import Sequence

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from config import LLM_MODEL
from rag.service.analysis.prompt import to_langchain_messages
from rag.service.session.schema import ChatMessage
from rag.service.tracing import TraceContext


def build_general_chat_prompt(
    question: str,
    chat_history: Sequence[ChatMessage] | None,
):
    """일반 대화 답변에 사용할 프롬프트를 만듭니다."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
당신은 자동차 사고 과실비율 RAG 챗봇의 일반 대화 응답 담당입니다.
검색 문서나 새로운 사고 분석 없이, Chat History와 현재 질문만으로 답하세요.

규칙:
- 이전 대화에서 확인되는 내용은 자연스럽게 요약하거나 다시 설명하세요.
- 새 과실비율을 추정하거나 문서를 검색한 것처럼 말하지 마세요.
- 사고 분석에 필요한 새 정보가 들어온 경우 직접 분석하지 말고, 사고 분석 흐름에서 처리될 수 있도록 짧게 안내하세요.
- 확인할 수 없는 내용은 모른다고 말하세요.
""".strip(),
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "# Question\n{question}"),
        ]
    )
    return prompt.invoke(
        {
            "chat_history": to_langchain_messages(chat_history),
            "question": question,
        }
    )


def answer_general_chat(
    question: str,
    chat_history: Sequence[ChatMessage] | None,
    llm=None,
    trace_context: TraceContext | None = None,
) -> str:
    """일반 대화 파이프의 답변을 생성합니다."""
    general_llm = llm or ChatOpenAI(model=LLM_MODEL, temperature=0)
    prompt = build_general_chat_prompt(question, chat_history)
    config = trace_context.langchain_config("mdm.general_chat") if trace_context else None
    response = general_llm.invoke(prompt, config=config) if config else general_llm.invoke(prompt)
    return str(getattr(response, "content", response)).strip()
