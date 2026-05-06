"""대화 라우터에서 사용하는 프롬프트를 모아둡니다."""

from __future__ import annotations

import json
from collections.abc import Sequence

from rag.service.intake.schema import IntakeState
from rag.service.session.schema import ChatMessage


ROUTER_HISTORY_LIMIT = 2


def _format_chat_history(chat_history: Sequence[ChatMessage] | None) -> str:
    if not chat_history:
        return "없음"
    lines: list[str] = []
    for message in chat_history[-ROUTER_HISTORY_LIMIT:]:
        if message.role not in {"user", "assistant"}:
            continue
        lines.append(f"{message.role}: {message.content}")
    return "\n".join(lines) if lines else "없음"


def _format_intake_state(intake_state: IntakeState | None) -> str:
    state = intake_state or IntakeState()
    return json.dumps(
        {
            "attempt_count": state.attempt_count,
            "search_metadata": {
                "party_type": state.search_metadata.party_type,
                "location": state.search_metadata.location,
                "retrieval_query": state.search_metadata.retrieval_query,
                "query_slots": {
                    "road_control": state.search_metadata.query_slots.road_control,
                    "relation": state.search_metadata.query_slots.relation,
                    "a_signal": state.search_metadata.query_slots.a_signal,
                    "b_signal": state.search_metadata.query_slots.b_signal,
                    "a_movement": state.search_metadata.query_slots.a_movement,
                    "b_movement": state.search_metadata.query_slots.b_movement,
                    "road_priority": state.search_metadata.query_slots.road_priority,
                    "special_condition": state.search_metadata.query_slots.special_condition,
                },
            },
            "last_missing_fields": state.last_missing_fields,
            "last_follow_up_questions": state.last_follow_up_questions,
        },
        ensure_ascii=False,
    )


def build_route_prompt(
    question: str,
    chat_history: Sequence[ChatMessage] | None,
    intake_state: IntakeState | None,
) -> str:
    """라우터 LLM에 전달할 분류 프롬프트를 만듭니다."""
    return f"""
당신은 자동차 사고 과실비율 RAG 챗봇의 라우터입니다.
답변을 생성하지 말고, 현재 사용자 입력을 어느 처리 흐름으로 보낼지만 판단하세요.

처리 흐름:

1. general_chat
- 검색 문서나 새 사고 분석 없이 답할 수 있는 일반 대화입니다.
- 예: 인사, 감사, 사용법 질문, 이전 답변을 쉽게 설명해달라는 요청, 대화 내용 요약 요청.

2. accident_analysis
- 자동차 사고 상황을 새로 설명하거나 보충하는 입력입니다.
- 과실비율 판단, 사고 조건 변경, 장소/대상/신호/무단횡단 등 사고 사실 추가가 포함됩니다.
- 이전 follow-up 질문에 대한 짧은 답변도 여기에 속합니다.

판단 규칙:
- 사용자가 사고 사실을 말하거나 과실비율을 묻는다면 accident_analysis입니다.
- intake_state에 last_missing_fields가 있고 현재 입력이 짧은 보충 답변처럼 보이면 accident_analysis입니다.
- 애매하면 accident_analysis로 보내세요.
- 반드시 JSON object만 출력하세요.

출력 형식:
{{
  "route_type": "general_chat | accident_analysis",
  "confidence": 0.0,
  "reason": "짧은 판단 이유"
}}

[최근 대화]
{_format_chat_history(chat_history)}

[현재 intake 상태]
{_format_intake_state(intake_state)}

[현재 사용자 입력]
{question}
""".strip()
