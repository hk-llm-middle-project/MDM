"""테스트와 Redis 대체 동작에 사용하는 메모리 대화 저장소입니다."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from config import DEFAULT_LOADER_STRATEGY
from rag.service.intake.schema import IntakeState
from rag.service.session.schema import ChatMessage, SessionMeta


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryConversationStore:
    """프로세스 안에서만 유지되는 대화 저장소 구현입니다."""

    def __init__(self) -> None:
        self._user_sessions: dict[str, list[str]] = {}
        self._active_sessions: dict[str, str] = {}
        self._loader_strategies: dict[str, str] = {}
        self._session_meta: dict[str, SessionMeta] = {}
        self._messages: dict[str, list[ChatMessage]] = {}
        self._intake_states: dict[str, IntakeState] = {}

    def list_sessions(self, user_id: str) -> list[SessionMeta]:
        return [
            self._session_meta[session_id]
            for session_id in self._user_sessions.get(user_id, [])
            if session_id in self._session_meta
        ]

    def get_active_session(self, user_id: str) -> str | None:
        return self._active_sessions.get(user_id)

    def set_active_session(self, user_id: str, session_id: str) -> None:
        self._active_sessions[user_id] = session_id

    def create_session(self, user_id: str, title: str | None = None) -> SessionMeta:
        session_ids = self._user_sessions.setdefault(user_id, [])
        session_id = str(uuid4())
        now = utc_now_iso()
        meta = SessionMeta(
            session_id=session_id,
            title=title or f"세션 {len(session_ids) + 1}",
            created_at=now,
            updated_at=now,
        )
        session_ids.append(session_id)
        self._session_meta[session_id] = meta
        self._messages[session_id] = []
        self._intake_states[session_id] = IntakeState()
        return meta

    def delete_session(self, user_id: str, session_id: str) -> None:
        session_ids = self._user_sessions.get(user_id, [])
        self._user_sessions[user_id] = [
            existing_session_id
            for existing_session_id in session_ids
            if existing_session_id != session_id
        ]
        self._session_meta.pop(session_id, None)
        self._messages.pop(session_id, None)
        self._intake_states.pop(session_id, None)
        if self._active_sessions.get(user_id) == session_id:
            self._active_sessions.pop(user_id, None)

    def get_messages(self, user_id: str, session_id: str) -> list[ChatMessage]:
        del user_id
        return list(self._messages.get(session_id, []))

    def append_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        del user_id
        self._messages.setdefault(session_id, []).append(
            ChatMessage(role=role, content=content, metadata=metadata or {})
        )
        self._touch_session(session_id)

    def get_intake_state(self, user_id: str, session_id: str) -> IntakeState:
        del user_id
        return self._intake_states.get(session_id, IntakeState())

    def set_intake_state(self, user_id: str, session_id: str, state: IntakeState) -> None:
        del user_id
        self._intake_states[session_id] = state
        self._touch_session(session_id)

    def get_loader_strategy(self, user_id: str) -> str | None:
        return self._loader_strategies.get(user_id, DEFAULT_LOADER_STRATEGY)

    def set_loader_strategy(self, user_id: str, strategy: str) -> None:
        self._loader_strategies[user_id] = strategy

    def _touch_session(self, session_id: str) -> None:
        meta = self._session_meta.get(session_id)
        if meta is None:
            return
        self._session_meta[session_id] = SessionMeta(
            session_id=meta.session_id,
            title=meta.title,
            created_at=meta.created_at,
            updated_at=utc_now_iso(),
        )
