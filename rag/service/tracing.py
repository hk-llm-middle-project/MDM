"""LangSmith 추적 metadata 헬퍼입니다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TraceContext:
    """사용자 대화 하나에서 공유할 LangSmith thread metadata입니다."""

    thread_id: str
    user_id: str | None = None
    session_id: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def langchain_config(self, run_name: str) -> dict[str, Any]:
        """thread metadata를 담은 LangChain RunnableConfig를 만듭니다."""
        metadata = {
            "thread_id": self.thread_id,
            "session_id": self.session_id or self.thread_id,
            **self.metadata,
        }
        if self.user_id is not None:
            metadata["user_id"] = self.user_id

        config: dict[str, Any] = {
            "metadata": metadata,
            "run_name": run_name,
        }
        if self.tags:
            config["tags"] = list(self.tags)
        return config
