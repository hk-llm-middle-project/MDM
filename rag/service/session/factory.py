"""대화 세션 저장소를 생성하는 팩토리입니다."""

from __future__ import annotations

from functools import lru_cache

from config import (
    get_redis_url,
    get_session_store_backend,
    get_session_store_strict,
    get_session_ttl_seconds,
)
from rag.service.session.memory_store import MemoryConversationStore
from rag.service.session.store import ConversationStore


@lru_cache(maxsize=1)
def get_conversation_store() -> ConversationStore:
    """설정된 대화 저장소를 반환합니다."""
    backend = get_session_store_backend()
    if backend != "redis":
        return MemoryConversationStore()

    redis_url = get_redis_url()
    if not redis_url:
        if get_session_store_strict():
            raise RuntimeError("SESSION_STORE_BACKEND=redis 설정에는 REDIS_URL이 필요합니다.")
        return MemoryConversationStore()

    try:
        from rag.service.session.redis_store import RedisConversationStore

        store = RedisConversationStore(
            redis_url=redis_url,
            ttl_seconds=get_session_ttl_seconds(),
        )
        store.ping()
        return store
    except Exception:
        if get_session_store_strict():
            raise
        return MemoryConversationStore()
