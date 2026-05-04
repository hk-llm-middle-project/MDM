"""LLM으로 일반 대화와 사고 분석 흐름을 라우팅합니다."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_openai import ChatOpenAI

from config import ROUTER_MODEL
from rag.service.common.json_utils import extract_json_object
from rag.service.conversation.prompts import build_route_prompt
from rag.service.conversation.schema import RouteDecision, RouteType
from rag.service.intake.schema import IntakeState
from rag.service.session.schema import ChatMessage
from rag.service.tracing import TraceContext


ROUTER_CONFIDENCE_THRESHOLD = 0.6


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
    trace_context: TraceContext | None = None,
) -> RouteDecision:
    """현재 입력을 일반 대화 또는 사고 분석 흐름으로 분류합니다."""
    router_llm = llm or ChatOpenAI(model=ROUTER_MODEL, temperature=0)
    try:
        prompt = build_route_prompt(question, chat_history, intake_state)
        config = trace_context.langchain_config("mdm.route") if trace_context else None
        response = router_llm.invoke(prompt, config=config) if config else router_llm.invoke(prompt)
        content = getattr(response, "content", response)
        return parse_route_decision(extract_json_object(str(content)))
    except Exception as error:
        return fallback_route_conversation_turn(str(error))
