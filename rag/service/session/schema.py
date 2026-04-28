"""저장되는 대화 세션 데이터 구조입니다."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMessage:
    """대화 세션에 저장되는 채팅 메시지 한 건입니다."""

    role: str
    content: str


@dataclass(frozen=True)
class SessionMeta:
    """저장된 대화 세션을 표시할 때 쓰는 메타데이터입니다."""

    session_id: str
    title: str
    created_at: str
    updated_at: str
