"""사고 입력의 분석 가능 여부를 판단합니다."""

from rag.service.intake.schema import IntakeDecision, MissingField


def evaluate_input_sufficiency(user_input: str) -> IntakeDecision:
    """현재는 빈 입력만 걸러내고, 추후 LLM 판단으로 확장합니다."""
    normalized_description = user_input.strip()
    if normalized_description:
        return IntakeDecision(
            is_sufficient=True,
            normalized_description=normalized_description,
        )

    return IntakeDecision(
        is_sufficient=False,
        normalized_description="",
        missing_fields=[
            MissingField(
                name="사고 상황",
                reason="분석할 사고 설명이 입력되지 않았습니다.",
            )
        ],
        follow_up_questions=["사고가 난 장소와 각 차량의 진행 방향을 알려주세요."],
    )

