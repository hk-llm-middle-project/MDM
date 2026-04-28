"""사고 입력 메타데이터 추출에 사용하는 프롬프트를 모아둡니다."""

import json
from collections.abc import Sequence

from rag.service.intake.schema import IntakeState
from rag.service.intake.values import LOCATIONS, PARTY_TYPES
from rag.service.session.schema import ChatMessage


INTAKE_HISTORY_LIMIT = 6


def _format_chat_history(chat_history: Sequence[ChatMessage] | None) -> str:
    if not chat_history:
        return "없음"
    lines: list[str] = []
    for message in chat_history[-INTAKE_HISTORY_LIMIT:]:
        if message.role not in {"user", "assistant"}:
            continue
        lines.append(f"{message.role}: {message.content}")
    return "\n".join(lines) if lines else "없음"


def _format_previous_state(previous_state: IntakeState | None) -> str:
    state = previous_state or IntakeState()
    return json.dumps(
        {
            "search_metadata": {
                "party_type": state.search_metadata.party_type,
                "location": state.search_metadata.location,
            },
            "last_missing_fields": state.last_missing_fields,
            "last_follow_up_questions": state.last_follow_up_questions,
        },
        ensure_ascii=False,
    )


def build_intake_prompt(
    user_input: str,
    chat_history: Sequence[ChatMessage] | None = None,
    previous_state: IntakeState | None = None,
) -> str:
    """사용자 사고 설명을 검색 메타데이터 추출 프롬프트로 변환합니다."""
    party_types = "\n".join(f"- {value}" for value in PARTY_TYPES)
    locations = "\n".join(f"- {value}" for value in LOCATIONS)

    return f"""
당신은 자동차 사고 설명에서 검색용 메타데이터를 추출하는 분류기입니다.

사용자의 현재 입력과 필요한 경우 이전 대화 이력을 함께 보고 아래 두 필드만 추출하세요.
현재 입력은 이전 follow-up 질문에 대한 짧은 보충 답변일 수 있습니다.
이전 상태에 이미 추출된 값이 있고 현재 입력에서 반박되지 않았다면 그 값을 유지해도 됩니다.

party_type 허용값:
{party_types}

location 허용값:
{locations}

규칙:
- 반드시 허용값 중 하나 또는 null만 사용하세요.
- 사용자의 입력과 대화 이력에 근거가 없으면 null을 사용하세요.
- 추측하지 마세요.
- JSON 외의 문장은 출력하지 마세요.
- confidence는 0.0부터 1.0 사이 숫자로 작성하세요.
- missing_fields에는 값이 null이거나 confidence가 낮은 필드명을 넣으세요.
- follow_up_questions에는 부족한 정보를 확인하기 위한 질문을 넣으세요.
- 누락된 필드가 있으면 follow_up_questions를 사용자에게 물어볼 자연스러운 한국어 질문으로 작성하세요.
- 질문은 최대 2개만 작성하고, party_type/location 선택지를 반드시 포함하세요.

출력 JSON 형식:
{{
  "party_type": "보행자 | 자동차 | 자전거 | null",
  "location": "허용된 location 값 중 하나 | null",
  "confidence": {{
    "party_type": 0.0,
    "location": 0.0
  }},
  "missing_fields": ["party_type", "location"],
  "follow_up_questions": ["string"]
}}

[이전 대화]
{_format_chat_history(chat_history)}

[이전 intake 상태]
{_format_previous_state(previous_state)}

[현재 사용자 입력]
{user_input}
""".strip()
