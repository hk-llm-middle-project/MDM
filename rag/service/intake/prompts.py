"""사고 입력 메타데이터 추출에 사용하는 프롬프트를 모아둡니다."""

import json
from collections.abc import Sequence

from rag.service.intake.schema import IntakeState
from rag.service.intake.values import LOCATIONS, PARTY_TYPES
from rag.service.session.schema import ChatMessage


INTAKE_HISTORY_LIMIT = 4


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


def build_intake_prompt(
    user_input: str,
    chat_history: Sequence[ChatMessage] | None = None,
    previous_state: IntakeState | None = None,
) -> str:
    """사용자 사고 설명을 검색 메타데이터 추출 프롬프트로 변환합니다."""
    party_types = "\n".join(f"- {value}" for value in PARTY_TYPES)
    locations = "\n".join(f"- {value}" for value in LOCATIONS)

    return f"""
사용자 입력과 이전 intake 상태를 바탕으로 검색용 사고 메타데이터를 추출하세요.
현재 입력이 이전 follow-up의 짧은 답변이면 이전 상태를 보존하고 새 정보만 보완하세요.
확실하지 않은 값은 null로 두세요.

party_type 허용값:
{party_types}

location 허용값:
{locations}

분류 우선순위:
1. 교차로 내 신호, 좌우 도로, 직진/좌회전/우회전 관계가 핵심이면 "교차로 사고".
2. 중앙선 침범, 역주행, 마주 오는 차량 충돌, 반대방향 진행 차량 사고이면 "마주보는 방향 진행차량 상호 간의 사고".
3. 추돌, 안전거리미확보, 진로변경, 차로 변경, 같은 방향, 후행/선행 차량 사고이면 "같은 방향 진행차량 상호간의 사고".
4. 횡단보도 관련 사고이면 횡단보도 내/부근/없음 중 가장 가까운 location.
5. 도로 외 진입, 주차장, 문 열림, 회전교차로, 긴급자동차, 낙하물, 유턴, 정차 후 출발 등 위 분류가 명확하지 않은 자동차 사고는 "기타".

주의:
- party_type과 location은 허용값 또는 null만 사용하세요.
- party_type은 사용자의 차량 종류가 아니라 사고 상대/검색 축이 되는 대상 유형입니다. 이전 상태의 party_type이 "자전거"인데 현재 답변에 "저는 자동차"가 나오면 party_type을 "자동차"로 바꾸지 말고 "자전거"를 유지하세요.
- 단, 사용자가 "상대는 자동차였어요", "사고 상대가 자동차였어요"처럼 사고 상대를 명시적으로 정정하면 그 값을 반영하세요.
- "양쪽 신호등 있는 교차로", "직진 대 직진 사고" 같은 세부 사고유형은 location이 아니라 retrieval_query에 넣으세요.
- "맞은편/반대편"은 마주보는 방향 사고가 아니라 교차로 내 직진/좌회전 관계일 수 있습니다. 교차로 신호나 진입 방향 설명이 있으면 "교차로 사고"를 우선하세요.
- "오른쪽 도로", "왼쪽 도로", "대로", "소로"는 relation이 아니라 road_priority에 보존하세요.
- "오른쪽 도로"와 "왼쪽 도로"처럼 좌우 도로에서 진입하는 교차로 사고는 relation을 "상대차량이 측면에서 진입"으로 처리하세요.
- "비보호 좌회전"은 "비보호좌회전"으로, "차로 변경"/"진로 변경"은 "진로변경"으로 통일하세요.

query_slots 작성 규칙:
- query_slots는 retrieval_query를 안정적으로 조립하기 위한 구조화 사고 단서입니다. 모르면 null입니다.
- road_control: "양쪽 신호등", "한쪽 신호등", "신호등 없음", "점멸신호", "도로", "주차장"
- relation: "상대차량이 측면에서 진입", "상대차량이 맞은편에서 진입", "같은 방향", "도로 외 진입"
- a_signal/b_signal: "녹색", "황색", "적색", "적색점멸", "황색점멸"
- a_movement/b_movement: "직진", "좌회전", "비보호좌회전", "우회전", "진로변경", "추돌", "역주행"
- road_priority: "동일 폭", "대로 소로", "오른쪽 소로", "왼쪽 대로", "오른쪽 도로", "왼쪽 도로"
- special_condition: "중앙선 침범", "추돌사고", "진로변경 사고", "동시 진로변경", "도로가 아닌 장소에서 도로로 진입"

retrieval_query 작성 규칙:
- 원문 요약이 아니라 문서 도표 제목에 가까운 검색어 조합으로 작성하세요.
- 신호, 진행방향, 상대 위치, 도로 우선관계처럼 기준번호를 가르는 표현을 우선하고, 쉼표 기준 2~4개 표현으로 제한하세요.
- 신호와 진행방향 조합은 "녹색직진 대 적색직진"처럼 압축하세요.
- 양쪽 신호등 교차로의 측면 직진 대 직진 사고는 "양쪽 신호등 있는 교차로, 직진 대 직진 사고, 상대차량이 측면에서 진입"을 포함하세요.
- 입력이 짧은 follow-up 답변이라도 이전 상태나 대화 이력으로 사고 설명이 충분하면 retrieval_query를 유지하거나 보완하세요.

confidence/follow-up 규칙:
- confidence는 각 필드가 사용자 입력과 이전 상태로 충분히 뒷받침되는 정도이며 0.0부터 1.0 사이 숫자입니다.
- missing_fields에는 값이 null이거나 confidence가 낮은 필드명을 넣으세요.
- follow_up_questions에는 부족한 정보를 확인하기 위한 자연스러운 한국어 질문을 최대 2개 넣으세요.
- party_type/location이 누락되면 질문에 해당 선택지를 포함하세요.
- JSON 외의 문장은 출력하지 마세요.

출력 JSON 형식:
{{
  "party_type": "보행자 | 자동차 | 자전거 | null",
  "location": "허용된 location 값 중 하나 | null",
  "retrieval_query": "검색용 사고 유형 질의 | null",
  "query_slots": {{
    "road_control": "string | null",
    "relation": "string | null",
    "a_signal": "string | null",
    "b_signal": "string | null",
    "a_movement": "string | null",
    "b_movement": "string | null",
    "road_priority": "string | null",
    "special_condition": "string | null"
  }},
  "confidence": {{
    "party_type": 0.0,
    "location": 0.0,
    "retrieval_query": 0.0
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
