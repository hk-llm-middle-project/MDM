"""사고 입력 메타데이터 추출에 사용할 프롬프트를 모아둡니다."""

from rag.service.intake.values import LOCATIONS, PARTY_TYPES


def build_intake_prompt(user_input: str) -> str:
    """사용자 사고 설명을 검색 메타데이터 추출용 프롬프트로 변환합니다."""
    party_types = "\n".join(f"- {value}" for value in PARTY_TYPES)
    locations = "\n".join(f"- {value}" for value in LOCATIONS)

    return f"""
당신은 자동차 사고 설명에서 검색용 메타데이터를 추출하는 분류기입니다.

사용자의 사고 설명을 읽고 아래 두 필드만 추출하세요.

party_type 허용값:
{party_types}

location 허용값:
{locations}

규칙:
- 반드시 허용값 중 하나 또는 null만 사용하세요.
- 사용자 입력에 근거가 없으면 null을 사용하세요.
- 추측하지 마세요.
- JSON 외의 문장을 출력하지 마세요.
- confidence는 0.0부터 1.0 사이 숫자로 작성하세요.
- missing_fields에는 값이 null이거나 confidence가 낮은 필드명을 넣으세요.
- follow_up_questions에는 부족한 정보를 확인하기 위한 질문을 넣으세요.

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

[사용자 입력]
{user_input}
""".strip()
