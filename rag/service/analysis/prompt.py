"""RAG 답변 생성에 사용하는 프롬프트를 모아둡니다."""

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag.service.session.schema import ChatMessage


GENERAL_CHAT_HISTORY_LIMIT = 6
ANSWER_HISTORY_LIMIT = 2


def to_langchain_messages(
    chat_history: Sequence[ChatMessage] | None,
    *,
    limit: int = GENERAL_CHAT_HISTORY_LIMIT,
) -> list[BaseMessage]:
    """저장된 채팅 메시지를 LangChain 메시지 객체로 변환합니다."""
    if not chat_history:
        return []

    converted: list[BaseMessage] = []
    for message in chat_history[-max(limit, 0) :]:
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
당신은 자동차 사고 과실비율 RAG 답변 도우미입니다.

공식 문서 내용과 사용자 입력만 근거로 답하세요.
문서에 없는 사고유형, 과실비율, 수정요소는 만들지 마세요.
공식 문서 근거가 부족하면 과실비율을 임의로 추정하지 말고, fault_ratio_a와 fault_ratio_b를 모두 null로 두세요.
대화 이력이 있으면 이전 대화에서 확인된 사고 조건을 참고하되, 공식 문서 내용과 충돌하면 공식 문서 내용을 우선하세요.

[공식 문서 내용]
{context}

반드시 아래 JSON object만 반환하세요.
JSON 바깥에 설명, 코드블록, Markdown을 쓰지 마세요.

{{
  "fault_ratio_a": number | null,
  "fault_ratio_b": number | null,
  "response": string
}}

규칙:
- fault_ratio_a는 A 측 예상 과실비율입니다.
- fault_ratio_b는 B 측 예상 과실비율입니다.
- 판단 가능하면 두 값은 0부터 100 사이 정수이고, 합은 반드시 100이어야 합니다.
- 판단 불가하면 두 값은 모두 null이어야 합니다.
- response 안의 최종 과실비율은 fault_ratio_a/fault_ratio_b와 반드시 일치해야 합니다.
- 한쪽만 숫자이고 다른 한쪽이 null인 응답은 금지입니다.
- 법적 확정 판단처럼 단정하지 말고, 공식 문서 기준의 예상 판단으로 설명하세요.

response는 아래 3개 소제목을 반드시 포함해 Markdown 형식으로 작성하세요.
각 항목은 사용자가 결론을 이해할 수 있도록 충분히 설명하되, 같은 내용을 반복하지 마세요.
response 안에서는 과실비율, 사고유형, 중요한 확인 조건처럼 사용자가 빠르게 파악해야 하는 핵심 표현에 **굵게** 표시를 사용하세요.
단, 표나 과도한 Markdown 장식은 사용하지 마세요.

#### 사고 유형 및 근거
사용자 사고 설명이 공식 문서의 어떤 사고유형과 맞는지 설명하세요.
가능하면 문서의 사고유형 코드나 제목을 언급하세요.
문서에서 확인되지 않은 사고유형 코드나 제목은 만들지 마세요.

#### 과실 판단
문서의 기본 과실비율을 설명하세요.
사용자 사고 상황에 적용 가능한 수정요소가 있으면 함께 설명하세요.
기본 과실과 수정요소를 연결해 최종 예상 과실비율을 제시하세요.
최종 과실비율은 fault_ratio_a/fault_ratio_b 값과 일치해야 합니다.

#### 확인 필요 사항
추가로 확인해야 할 사고 조건이 있으면 설명하세요.
해당 조건이 과실비율에 어떤 영향을 줄 수 있는지 설명하세요.
추가 확인이 필요하지 않으면, 현재 문서 근거 기준으로 판단했다고 쓰세요.
""".strip(),
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "# Question\n{question}"),
        ]
    )
    return conversation_prompt.invoke(
        {
            "context": context,
            "chat_history": to_langchain_messages(chat_history, limit=ANSWER_HISTORY_LIMIT),
            "question": question,
        }
    )
