"""사고 입력의 검색 가능 여부를 판단합니다."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_openai import ChatOpenAI

from config import LLM_MODEL
from rag.service.intake.prompts import build_intake_prompt
from rag.service.intake.schema import IntakeDecision, MissingField, UserSearchMetadata
from rag.service.intake.values import LOCATIONS, PARTY_TYPES


CONFIDENCE_THRESHOLD = 0.75


def extract_json_object(content: str) -> dict[str, Any]:
    """LLM 응답에서 JSON 객체를 추출합니다."""
    # JSON 블록({ ... })을 찾아 추출합니다. 마크다운 코드 블록이나 앞뒤 설명 텍스트가 있어도 대응 가능합니다.
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"응답에서 JSON 객체를 찾을 수 없습니다: {content}")
    return json.loads(match.group())


def clamp_confidence(value: object) -> float:
    """신뢰도 값을 0.0부터 1.0 사이로 보정합니다."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def build_default_follow_up_questions(missing_field_names: list[str]) -> list[str]:
    """누락 필드에 대한 기본 추가 질문을 만듭니다."""
    questions: list[str] = []
    if "party_type" in missing_field_names:
        questions.append("사고 상대는 보행자, 자동차, 자전거 중 무엇인가요?")
    if "location" in missing_field_names:
        questions.append(
            "사고 상황은 횡단보도 내, 횡단보도 부근, 횡단보도 없음, 교차로 사고, "
            "마주보는 방향 차량 간 사고, 같은 방향 차량 간 사고, 자동차 대 이륜차 특수유형, 기타 중 어디에 가까운가요?"
        )
    return questions


def normalize_metadata_response(data: dict[str, Any]) -> IntakeDecision:
    """LLM 추출 결과를 허용값과 신뢰도 기준으로 검증합니다."""
    raw_party_type = data.get("party_type")
    raw_location = data.get("location")
    confidence_data = data.get("confidence")
    if not isinstance(confidence_data, dict):
        confidence_data = {}

    confidence = {
        "party_type": clamp_confidence(confidence_data.get("party_type")),
        "location": clamp_confidence(confidence_data.get("location")),
    }

    party_type = raw_party_type if raw_party_type in PARTY_TYPES else None
    location = raw_location if raw_location in LOCATIONS else None

    missing_field_names: list[str] = []
    missing_fields: list[MissingField] = []
    if party_type is None or confidence["party_type"] < CONFIDENCE_THRESHOLD:
        party_type = None
        missing_field_names.append("party_type")
        missing_fields.append(
            MissingField(
                name="party_type",
                reason="사고 상대 유형이 허용값으로 충분히 특정되지 않았습니다.",
            )
        )
    if location is None or confidence["location"] < CONFIDENCE_THRESHOLD:
        location = None
        missing_field_names.append("location")
        missing_fields.append(
            MissingField(
                name="location",
                reason="사고 장소 또는 상황 유형이 허용값으로 충분히 특정되지 않았습니다.",
            )
        )

    follow_up_questions = [
        question
        for question in data.get("follow_up_questions", [])
        if isinstance(question, str) and question.strip()
    ]
    if missing_field_names and not follow_up_questions:
        follow_up_questions = build_default_follow_up_questions(missing_field_names)

    return IntakeDecision(
        is_sufficient=not missing_fields,
        normalized_description="",
        search_metadata=UserSearchMetadata(
            party_type=party_type,
            location=location,
        ),
        confidence=confidence,
        missing_fields=missing_fields,
        follow_up_questions=follow_up_questions,
    )


def evaluate_input_sufficiency(user_input: str, llm: Any | None = None) -> IntakeDecision:
    """LLM으로 party_type과 location을 추출해 검색 가능 여부를 판단합니다."""
    normalized_description = user_input.strip()
    if not normalized_description:
        return IntakeDecision(
            is_sufficient=False,
            normalized_description="",
            missing_fields=[
                MissingField(
                    name="사고 상황",
                    reason="분석할 사고 설명이 입력되지 않았습니다.",
                )
            ],
            follow_up_questions=["사고 상황을 간단히 입력해주세요."],
        )

    intake_llm = llm or ChatOpenAI(model=LLM_MODEL, temperature=0)
    response = intake_llm.invoke(build_intake_prompt(normalized_description))
    content = getattr(response, "content", response)
    decision = normalize_metadata_response(extract_json_object(str(content)))
    return IntakeDecision(
        is_sufficient=decision.is_sufficient,
        normalized_description=normalized_description,
        search_metadata=decision.search_metadata,
        confidence=decision.confidence,
        missing_fields=decision.missing_fields,
        follow_up_questions=decision.follow_up_questions,
    )
