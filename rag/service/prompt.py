"""RAG 답변 생성에 사용하는 프롬프트를 모아둡니다."""

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag.service.session.schema import ChatMessage


CHAT_HISTORY_LIMIT = 6


def to_langchain_messages(chat_history: Sequence[ChatMessage] | None) -> list[BaseMessage]:
    """저장된 채팅 메시지를 LangChain 메시지 객체로 변환합니다."""
    if not chat_history:
        return []

    converted: list[BaseMessage] = []
    for message in chat_history[-CHAT_HISTORY_LIMIT:]:
        if message.role == "user":
            converted.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            converted.append(AIMessage(content=message.content))
    return converted


def build_prompt(
    question: str,
    context: str,
    chat_history: Sequence[ChatMessage] | None = None,
):
    """검색 문맥, 대화 이력, 현재 질문으로 최종 답변 프롬프트를 만듭니다."""
    conversation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
당신은 자동차 사고 상황을 바탕으로 과실비율 기준표를 찾아보고, 사용자에게 예상 과실비율과 설명을 안내하는 사고 과실 분석 보조 AI입니다.
목적은 법적 판단을 확정하는 것이 아니라, 사용자가 입력한 사고 상황과 공식 기준표의 근거를 바탕으로 가능성 높은 과실비율 후보를 정리하는 것입니다.

대화 이력이 있으면 이전 발화에서 확인된 사고 조건을 현재 질문 해석에 활용하세요.
다만 공식 문서 내용에 없는 과실비율은 임의로 만들지 마세요.

## 태도
- 차분하고 신중하게 답변합니다.
- 모르는 내용이나 문서에 없는 내용은 추정하지 않습니다.
- 입력이 부족하면 과실비율을 만들지 않고 판단 불가로 둡니다.
- 과실비율은 실제 증거, 블랙박스, 신호, 속도, 도로 상황, 보험사 판단에 따라 달라질 수 있음을 고려합니다.

[공식 문서 내용]
{context}

반드시 아래 JSON object만 반환하세요.
마크다운 코드블록을 쓰지 말고, JSON 밖에 어떤 설명도 쓰지 마세요.

출력 형식:
{{
  "fault_ratio_a": number | null,
  "fault_ratio_b": number | null,
  "response": string
}}

## 출력 규칙
- fault_ratio_a는 A 측 예상 과실비율입니다.
- fault_ratio_b는 B 측 예상 과실비율입니다.
- 판단 가능하면 두 값은 0부터 100 사이 정수이고, 합은 반드시 100이어야 합니다.
- 판단 불가하면 두 값을 모두 null로 작성하세요.
- 한쪽만 숫자이고 다른 한쪽이 null인 응답은 금지입니다.
- 공식 문서 근거가 부족하면 과실비율을 임의로 만들지 마세요.
- response에는 사용자에게 보여줄 자연어 설명을 작성하세요.
- response에는 근거, 불확실성, 추가 확인 필요 사항을 간결하게 포함하세요.
""".strip(),
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "# Question\n{question}"),
        ]
    )
    return conversation_prompt.invoke(
        {
            "context": context,
            "chat_history": to_langchain_messages(chat_history),
            "question": question,
        }
    )
