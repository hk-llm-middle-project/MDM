"""검색에서 공통으로 사용하는 도우미 함수들."""

from __future__ import annotations

from typing import Any

from kiwipiepy import Kiwi
from langchain_core.documents import Document


_KIWI: Kiwi | None = None


def get_kiwi() -> Kiwi:
    """공유 Kiwi 토크나이저를 지연 생성합니다."""
    global _KIWI
    if _KIWI is None:
        _KIWI = Kiwi()
    return _KIWI


def kiwi_tokenize(text: str) -> list[str]:
    """한국어 텍스트를 BM25에 적합한 토큰으로 분리합니다."""
    return [token.form for token in get_kiwi().tokenize(text)]


def mark_retrieval_fallback(
    documents: list[Document],
    fallback_from: str,
    fallback_to: str,
    reason: str,
) -> list[Document]:
    """Return copied documents annotated with retrieval fallback metadata."""
    fallback_event = {
        "from": fallback_from,
        "to": fallback_to,
        "reason": reason,
    }
    marked_documents: list[Document] = []
    for document in documents:
        metadata = dict(document.metadata)
        existing_events = metadata.get("retrieval_fallbacks")
        fallback_events = (
            [*existing_events, fallback_event]
            if isinstance(existing_events, list)
            else [fallback_event]
        )
        metadata.update(
            {
                "retrieval_fallback": True,
                "fallback_from": fallback_from,
                "fallback_to": fallback_to,
                "fallback_reason": reason,
                "retrieval_fallbacks": fallback_events,
            },
        )
        marked_documents.append(
            Document(page_content=document.page_content, metadata=metadata),
        )
    return marked_documents


def get_vectorstore_documents(vectorstore: Any) -> list[Document]:
    """벡터스토어에 저장된 문서와 메타데이터를 추출합니다."""
    if not hasattr(vectorstore, "get"):
        raise ValueError("이 검색 전략은 get()을 지원하는 벡터스토어가 필요합니다.")

    stored = vectorstore.get(include=["documents", "metadatas"])
    documents = stored.get("documents", [])
    metadatas = stored.get("metadatas", [])

    return [
        Document(page_content=page_content, metadata=metadata or {})
        for page_content, metadata in zip(documents, metadatas, strict=False)
    ]


def get_embedding_function_from_vectorstore(vectorstore: Any) -> Any:
    """가능한 경우 벡터스토어에서 임베딩 함수를 읽어옵니다."""
    embedding_function = getattr(vectorstore, "_embedding_function", None)
    if embedding_function is None:
        embedding_function = getattr(vectorstore, "embedding_function", None)
    if embedding_function is None:
        raise ValueError("이 검색 전략은 벡터스토어의 임베딩 함수에 접근할 수 있어야 합니다.")
    return embedding_function
