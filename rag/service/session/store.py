"""대화 세션 저장소 인터페이스입니다."""

from __future__ import annotations

from typing import Protocol

from rag.service.intake.schema import IntakeState
from rag.service.session.schema import ChatMessage, SessionMeta


class ConversationStore(Protocol):
    """채팅 세션과 intake 상태를 저장하는 경계입니다."""

    def list_sessions(self, user_id: str) -> list[SessionMeta]:
        """사용자의 세션 목록을 표시 순서대로 반환합니다."""

    def get_active_session(self, user_id: str) -> str | None:
        """사용자의 활성 세션 id를 반환합니다."""

    def set_active_session(self, user_id: str, session_id: str) -> None:
        """사용자의 활성 세션 id를 저장합니다."""

    def create_session(self, user_id: str, title: str | None = None) -> SessionMeta:
        """사용자에게 새 세션을 생성합니다."""

    def get_messages(self, user_id: str, session_id: str) -> list[ChatMessage]:
        """한 세션의 메시지 목록을 반환합니다."""

    def append_message(self, user_id: str, session_id: str, role: str, content: str) -> None:
        """세션에 채팅 메시지 한 건을 추가합니다."""

    def get_intake_state(self, user_id: str, session_id: str) -> IntakeState:
        """한 세션에 저장된 intake 상태를 반환합니다."""

    def set_intake_state(self, user_id: str, session_id: str, state: IntakeState) -> None:
        """한 세션의 intake 상태를 저장합니다."""

    def get_loader_strategy(self, user_id: str) -> str | None:
        """사용자가 선택한 문서 로더 전략을 반환합니다."""

    def set_loader_strategy(self, user_id: str, strategy: str) -> None:
        """사용자가 선택한 문서 로더 전략을 저장합니다."""
