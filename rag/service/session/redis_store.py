"""Redis 기반 대화 저장소입니다."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from config import DEFAULT_LOADER_STRATEGY
from rag.service.intake.schema import IntakeState
from rag.service.session.schema import ChatMessage, SessionMeta
from rag.service.session.serialization import (
    intake_state_from_dict,
    intake_state_to_dict,
    json_dumps,
    json_loads,
    message_from_dict,
    message_to_dict,
    session_meta_from_dict,
    session_meta_to_dict,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RedisConversationStore:
    """대화 저장소 인터페이스의 Redis 구현입니다."""

    def __init__(self, redis_url: str, ttl_seconds: int | None = None) -> None:
        import redis

        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._ttl_seconds = ttl_seconds

    def ping(self) -> bool:
        return bool(self._client.ping())

    def list_sessions(self, user_id: str) -> list[SessionMeta]:
        session_ids = self._client.lrange(self._user_sessions_key(user_id), 0, -1)
        sessions: list[SessionMeta] = []
        for session_id in session_ids:
            meta = self._get_session_meta(session_id)
            if meta is not None:
                sessions.append(meta)
        return sessions

    def get_active_session(self, user_id: str) -> str | None:
        return self._client.get(self._active_session_key(user_id))

    def set_active_session(self, user_id: str, session_id: str) -> None:
        key = self._active_session_key(user_id)
        self._client.set(key, session_id)
        self._expire_keys(key)

    def create_session(self, user_id: str, title: str | None = None) -> SessionMeta:
        session_id = str(uuid4())
        session_count = self._client.llen(self._user_sessions_key(user_id))
        now = utc_now_iso()
        meta = SessionMeta(
            session_id=session_id,
            title=title or f"세션 {session_count + 1}",
            created_at=now,
            updated_at=now,
        )
        self._client.rpush(self._user_sessions_key(user_id), session_id)
        self._set_session_meta(meta)
        self.set_intake_state(user_id, session_id, IntakeState())
        self._expire_session(user_id, session_id)
        return meta

    def get_messages(self, user_id: str, session_id: str) -> list[ChatMessage]:
        del user_id
        values = self._client.lrange(self._messages_key(session_id), 0, -1)
        messages: list[ChatMessage] = []
        for value in values:
            messages.append(message_from_dict(json_loads(value)))
        return messages

    def append_message(self, user_id: str, session_id: str, role: str, content: str) -> None:
        message = ChatMessage(role=role, content=content)
        self._client.rpush(self._messages_key(session_id), json_dumps(message_to_dict(message)))
        self._touch_session(session_id)
        self._expire_session(user_id, session_id)

    def get_intake_state(self, user_id: str, session_id: str) -> IntakeState:
        del user_id
        value = self._client.get(self._intake_state_key(session_id))
        if value is None:
            return IntakeState()
        return intake_state_from_dict(json_loads(value))

    def set_intake_state(self, user_id: str, session_id: str, state: IntakeState) -> None:
        self._client.set(
            self._intake_state_key(session_id),
            json_dumps(intake_state_to_dict(state)),
        )
        self._touch_session(session_id)
        self._expire_session(user_id, session_id)

    def get_loader_strategy(self, user_id: str) -> str | None:
        return self._client.get(self._loader_strategy_key(user_id)) or DEFAULT_LOADER_STRATEGY

    def set_loader_strategy(self, user_id: str, strategy: str) -> None:
        key = self._loader_strategy_key(user_id)
        self._client.set(key, strategy)
        self._expire_keys(key)

    def _get_session_meta(self, session_id: str) -> SessionMeta | None:
        data = self._client.hgetall(self._session_meta_key(session_id))
        if not data:
            return None
        return session_meta_from_dict(data)

    def _set_session_meta(self, meta: SessionMeta) -> None:
        key = self._session_meta_key(meta.session_id)
        self._client.hset(key, mapping=session_meta_to_dict(meta))
        self._expire_keys(key)

    def _touch_session(self, session_id: str) -> None:
        meta = self._get_session_meta(session_id)
        if meta is None:
            return
        self._set_session_meta(
            SessionMeta(
                session_id=meta.session_id,
                title=meta.title,
                created_at=meta.created_at,
                updated_at=utc_now_iso(),
            )
        )

    def _expire_session(self, user_id: str, session_id: str) -> None:
        self._expire_keys(
            self._user_sessions_key(user_id),
            self._active_session_key(user_id),
            self._loader_strategy_key(user_id),
            self._messages_key(session_id),
            self._intake_state_key(session_id),
            self._session_meta_key(session_id),
        )

    def _expire_keys(self, *keys: str) -> None:
        if self._ttl_seconds is None:
            return
        for key in keys:
            self._client.expire(key, self._ttl_seconds)

    @staticmethod
    def _user_sessions_key(user_id: str) -> str:
        return f"mdm:user:{user_id}:sessions"

    @staticmethod
    def _active_session_key(user_id: str) -> str:
        return f"mdm:user:{user_id}:active_session"

    @staticmethod
    def _loader_strategy_key(user_id: str) -> str:
        return f"mdm:user:{user_id}:loader_strategy"

    @staticmethod
    def _messages_key(session_id: str) -> str:
        return f"mdm:session:{session_id}:messages"

    @staticmethod
    def _intake_state_key(session_id: str) -> str:
        return f"mdm:session:{session_id}:intake_state"

    @staticmethod
    def _session_meta_key(session_id: str) -> str:
        return f"mdm:session:{session_id}:meta"
