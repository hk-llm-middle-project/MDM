"""대화 세션 상태를 JSON으로 변환하는 헬퍼입니다."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from rag.service.intake.schema import IntakeState, UserSearchMetadata
from rag.service.session.schema import ChatMessage, SessionMeta


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def json_loads(value: str) -> dict[str, Any]:
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("직렬화된 세션 값은 JSON 객체여야 합니다.")
    return data


def message_to_dict(message: ChatMessage) -> dict[str, str]:
    return {"role": message.role, "content": message.content}


def message_from_dict(data: dict[str, Any]) -> ChatMessage:
    role = data.get("role")
    content = data.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        raise ValueError("채팅 메시지에는 문자열 role과 content가 필요합니다.")
    return ChatMessage(role=role, content=content)


def session_meta_to_dict(meta: SessionMeta) -> dict[str, str]:
    return asdict(meta)


def session_meta_from_dict(data: dict[str, Any]) -> SessionMeta:
    session_id = data.get("session_id")
    title = data.get("title")
    created_at = data.get("created_at")
    updated_at = data.get("updated_at")
    if not all(isinstance(value, str) for value in (session_id, title, created_at, updated_at)):
        raise ValueError("세션 메타데이터 필드는 모두 문자열이어야 합니다.")
    return SessionMeta(
        session_id=session_id,
        title=title,
        created_at=created_at,
        updated_at=updated_at,
    )


def intake_state_to_dict(state: IntakeState) -> dict[str, Any]:
    return {
        "attempt_count": state.attempt_count,
        "search_metadata": asdict(state.search_metadata),
        "last_missing_fields": list(state.last_missing_fields),
        "last_follow_up_questions": list(state.last_follow_up_questions),
    }


def intake_state_from_dict(data: dict[str, Any]) -> IntakeState:
    metadata_data = data.get("search_metadata")
    if not isinstance(metadata_data, dict):
        metadata_data = {}

    last_missing_fields = data.get("last_missing_fields")
    if not isinstance(last_missing_fields, list):
        last_missing_fields = []

    last_follow_up_questions = data.get("last_follow_up_questions")
    if not isinstance(last_follow_up_questions, list):
        last_follow_up_questions = []

    try:
        attempt_count = int(data.get("attempt_count", 0))
    except (TypeError, ValueError):
        attempt_count = 0

    return IntakeState(
        attempt_count=max(0, attempt_count),
        search_metadata=UserSearchMetadata(
            party_type=metadata_data.get("party_type")
            if isinstance(metadata_data.get("party_type"), str)
            else None,
            location=metadata_data.get("location")
            if isinstance(metadata_data.get("location"), str)
            else None,
        ),
        last_missing_fields=[
            value for value in last_missing_fields if isinstance(value, str)
        ],
        last_follow_up_questions=[
            value for value in last_follow_up_questions if isinstance(value, str)
        ],
    )
