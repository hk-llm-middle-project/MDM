"""LLM으로 일반 대화와 사고 분석 흐름을 라우팅합니다."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from langchain_openai import ChatOpenAI

from config import LLM_MODEL
from rag.service.conversation.prompts import build_route_prompt
from rag.service.conversation.schema import RouteDecision, RouteType
from rag.service.intake.schema import IntakeState
from rag.service.session.schema import ChatMessage


ROUTER_CONFIDENCE_THRESHOLD = 0.6


def extract_json_object(content: str) -> dict[str, Any]:
    """LLM 응답에서 JSON object를 추출합니다."""
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ValueError(f"라우터 응답에서 JSON을 찾을 수 없습니다: {content}")
    return json.loads(match.group())


def clamp_confidence(value: object) -> float:
    """신뢰도 값을 0.0부터 1.0 사이로 보정합니다."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def fallback_route_conversation_turn(reason: str = "라우터 판단에 실패했습니다.") -> RouteDecision:
    """라우터 실패 시 정보 손실을 줄이기 위해 사고 분석 흐름으로 보냅니다."""
    return RouteDecision(
        route_type=RouteType.ACCIDENT_ANALYSIS,
        confidence=0.0,
        reason=reason,
    )


def parse_route_decision(data: dict[str, Any]) -> RouteDecision:
    """라우터 JSON을 검증된 RouteDecision으로 변환합니다."""
    route_value = data.get("route_type")
    try:
        route_type = RouteType(str(route_value))
    except ValueError:
        return fallback_route_conversation_turn(f"알 수 없는 route_type입니다: {route_value}")

    confidence = clamp_confidence(data.get("confidence"))
    if confidence < ROUTER_CONFIDENCE_THRESHOLD:
        return fallback_route_conversation_turn("라우터 신뢰도가 낮아 사고 분석 흐름으로 보냅니다.")

    reason = data.get("reason")
    return RouteDecision(
        route_type=route_type,
        confidence=confidence,
        reason=reason if isinstance(reason, str) else "",
    )


def route_conversation_turn(
    question: str,
    chat_history: Sequence[ChatMessage] | None,
    intake_state: IntakeState | None = None,
    llm: Any | None = None,
) -> RouteDecision:
    """현재 입력을 일반 대화 또는 사고 분석 흐름으로 분류합니다."""
    router_llm = llm or ChatOpenAI(model=LLM_MODEL, temperature=0)
    try:
        response = router_llm.invoke(build_route_prompt(question, chat_history, intake_state))
        content = getattr(response, "content", response)
        return parse_route_decision(extract_json_object(str(content)))
    except Exception as error:
        return fallback_route_conversation_turn(str(error))
