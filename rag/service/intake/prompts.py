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
당신은 자동차 사고 설명에서 검색용 메타데이터를 추출하는 분류기입니다.

사용자의 현재 입력과 필요한 경우 이전 대화 이력을 함께 보고 검색 메타데이터와 검색용 질의를 추출하세요.
현재 입력은 이전 follow-up 질문에 대한 짧은 보충 답변일 수 있습니다.
이전 상태에 이미 추출된 값이 있고 현재 입력에서 반박되지 않았다면 그 값을 유지해도 됩니다.

party_type 허용값:
{party_types}

location 허용값:
{locations}

규칙:
- party_type과 location은 반드시 허용값 중 하나 또는 null만 사용하세요.
- party_type과 location은 사용자의 입력과 대화 이력에 근거가 없으면 null을 사용하세요.
- 추측하지 마세요.
- "양쪽 신호등 있는 교차로", "직진 대 직진 사고" 같은 세부 사고유형 표현은 location 값으로 쓰지 말고 retrieval_query에만 넣으세요.
- 자동차 대 자동차 교차로 세부유형이면 location은 허용값 중 "교차로 사고"를 사용하세요.
- 중앙선 침범, 역주행, 마주 오는 차량과의 충돌, 반대방향 진행 차량 사고는 location을 "마주보는 방향 진행차량 상호 간의 사고"로 분류하세요.
- 단, 신호등 있는/없는 교차로에서 "상대차량이 맞은편에서 진입", "맞은편 좌회전", "직진 대 좌회전"처럼 교차로 내부 진입 방향을 설명하는 경우는 "마주보는 방향 진행차량 상호 간의 사고"가 아니라 "교차로 사고"로 분류하세요.
- 추돌, 안전거리미확보, 진로변경, 차로 변경, 같은 방향, 후행/선행 차량 사고는 location을 "같은 방향 진행차량 상호간의 사고"로 분류하세요.
- 도로가 아닌 장소에서 도로로 진입, 주차장, 문 열림, 회전교차로, 긴급자동차, 낙하물, 유턴, 정차 후 출발 등 교차로/마주보는 방향/같은 방향 진행차량 상호간 분류에 명확히 속하지 않는 자동차 사고는 location을 "기타"로 분류하세요.
- 사용자 입력에 "교차로"라는 단어가 없고 중앙선 침범/역주행/추돌/진로변경/도로 외 진입 같은 더 구체적인 유형 단서가 있으면 location을 "교차로 사고"로 분류하지 마세요.
- 사용자 입력에 "교차로"가 있더라도 중앙선 침범/역주행/추돌/진로변경/도로 외 진입 같은 더 구체적인 비교차로 유형 단서가 명시되면 그 구체 유형의 location을 우선하세요.
- retrieval_query는 retriever에 넣을 사고 유형 검색어입니다.
- query_slots는 retrieval_query를 코드에서 안정적으로 조립하기 위한 구조화된 사고 단서입니다.
- query_slots 값은 모르면 null로 두고, 확실한 값만 채우세요.
- query_slots.road_control 예: "양쪽 신호등", "한쪽 신호등", "신호등 없음", "점멸신호", "도로", "주차장"
- query_slots.relation 예: "상대차량이 측면에서 진입", "상대차량이 맞은편에서 진입", "같은 방향", "도로 외 진입"
- query_slots.a_signal/b_signal 예: "녹색", "황색", "적색", "적색점멸", "황색점멸"
- query_slots.a_movement/b_movement 예: "직진", "좌회전", "비보호좌회전", "우회전", "진로변경", "추돌", "역주행"
- query_slots.road_priority 예: "동일 폭", "대로 소로", "오른쪽 소로", "왼쪽 대로", "오른쪽 도로", "왼쪽 도로"
- query_slots.special_condition 예: "중앙선 침범", "추돌사고", "진로변경 사고", "동시 진로변경", "도로가 아닌 장소에서 도로로 진입"
- query_slots.relation에 "상대차량이 맞은편에서 진입"은 원문에 "맞은편" 또는 "반대편"이 명시된 경우에만 사용하세요.
- 원문에 "오른쪽 도로", "왼쪽 도로", "오른쪽 소로", "왼쪽 대로", "대로", "소로"가 있으면 relation을 "맞은편"으로 쓰지 말고 road_priority에 해당 표현을 넣으세요.
- "오른쪽 도로"와 "왼쪽 도로"처럼 좌우 도로에서 진입하는 교차로 사고는 relation을 "상대차량이 측면에서 진입"으로 처리하세요.
- "대로/소로", "오른쪽 소로", "왼쪽 대로"는 기준번호를 가르는 핵심 단서이므로 반드시 road_priority에 보존하세요.
- "비보호 좌회전"은 a_movement 또는 b_movement에 "비보호좌회전"으로 붙여 쓰세요.
- "차로 변경"과 "진로 변경"은 a_movement/b_movement에 "진로변경"으로 통일하세요.
- retrieval_query에는 원문 문장보다 문서 도표 제목에 가까운 짧은 핵심어를 넣으세요.
- retrieval_query는 가장 중요한 도표 제목형 표현을 앞에 두고, 쉼표 기준 2~4개 표현으로 제한하세요.
- retrieval_query에는 party_type 같은 넓은 표현보다 신호/진행방향/상대 위치/도로 우선관계처럼 기준번호를 가르는 표현을 우선하세요.
- 자동차 대 자동차 교차로 사고도 넓은 목차 표현을 길게 나열하지 말고 도표 제목형 표현을 우선하세요.
- 양쪽 신호등이 있는 교차로에서 서로 다른 도로 또는 측면에서 진입한 직진 대 직진 사고는 반드시 "양쪽 신호등 있는 교차로, 직진 대 직진 사고, 상대차량이 측면에서 진입"을 포함하세요.
- 신호와 진행방향 조합은 문서 도표 제목형으로 압축해 함께 넣으세요. 예: "녹색직진 대 적색직진", "황색직진 대 적색직진", "적색직진 대 적색직진"
- "측면에서 직진", "측면에서 진입", "서로 다른 방향"은 검색어에서 "상대차량이 측면에서 진입"으로 정리하세요.
- 예: "녹색직진 대 적색직진, 직진 대 직진 사고, 상대차량이 측면에서 진입"
- 불명확한 표현은 문서 검색에 방해되지 않도록 정리하세요. 예를 들어 양쪽 신호등이 명시되면 "한쪽 신호등"이라고 쓰지 마세요.
- 입력이 짧은 follow-up 답변이라도 이전 상태나 대화 이력으로 사고 설명이 충분하면 retrieval_query를 유지하거나 보완하세요.
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
