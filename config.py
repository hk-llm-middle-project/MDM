"""기본 RAG 앱의 프로젝트 설정입니다."""

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "raw" / "230630_자동차사고 과실비율 인정기준_최종.pdf"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
CHUNK_CACHE_DIR = BASE_DIR / "data" / "chunks"
PAGE_METADATA_DIR = BASE_DIR / "data" / "metadata"
PAGE_METADATA_PATH = PAGE_METADATA_DIR / "main_pdf_page_metadata.json"
LLAMA_MD_DIR = BASE_DIR / "data" / "llama_md"
PDFPLUMBER_OUT_DIR = BASE_DIR / "data" / "pdfplumber_output"
UPSTAGE_OUTPUT_DIR = BASE_DIR / "data" / "upstage_output"
UPSTAGE_MAIN_PDF_OUTPUT_DIR = UPSTAGE_OUTPUT_DIR / "main_pdf"
UPSTAGE_FINAL_DOCUMENTS_PATH = (
    UPSTAGE_MAIN_PDF_OUTPUT_DIR / "final" / "chunked_documents_final_compact.json"
)
UPSTAGE_RAW_DOCUMENTS_PATH = (
    UPSTAGE_MAIN_PDF_OUTPUT_DIR / "raw" / "parsed_documents_raw.json"
)
DEFAULT_LOADER_STRATEGY = "pdfplumber"
DEFAULT_CHUNKER_STRATEGY = "fixed"
LOADER_VECTORSTORE_DIRS = {
    "pdfplumber": VECTORSTORE_DIR / "pdfplumber",
    "llamaparser": VECTORSTORE_DIR / "llamaparser",
    "llama-parse": VECTORSTORE_DIR / "llamaparser",
    "upstage": VECTORSTORE_DIR / "upstage",
}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
INDEX_BATCH_SIZE = 100
RETRIEVER_K = 3

DEFAULT_EMBEDDING_PROVIDER = "bge"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GOOGLE_EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gpt-4o-mini"


def _optional_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is None or value.strip().lower() in {"", "null", "none"}:
        return None
    return value.strip()


def get_redis_url() -> str | None:
    """세션 저장에 사용할 Redis URL을 반환합니다."""
    return _optional_env("REDIS_URL")


def get_session_store_backend() -> str:
    """사용할 세션 저장소 백엔드를 반환합니다."""
    return (_optional_env("SESSION_STORE_BACKEND", "memory") or "memory").lower()


def get_session_store_strict() -> bool:
    """Redis 연결 실패 시 앱을 실패 처리할지 반환합니다."""
    value = _optional_env("SESSION_STORE_STRICT", "false")
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def get_session_ttl_seconds() -> int | None:
    """세션 데이터에 적용할 Redis TTL 초 값을 반환합니다."""
    value = _optional_env("SESSION_TTL_SECONDS")
    if value is None:
        return None
    try:
        ttl = int(value)
    except ValueError:
        return None
    return ttl if ttl > 0 else None


def get_vectorstore_dir(
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    chunker_strategy: str = DEFAULT_CHUNKER_STRATEGY,
) -> Path:
    try:
        loader_vectorstore_dir = LOADER_VECTORSTORE_DIRS[loader_strategy]
    except KeyError as error:
        available = ", ".join(sorted(LOADER_VECTORSTORE_DIRS))
        raise ValueError(
            f"Unknown loader strategy: {loader_strategy}. Available strategies: {available}"
        ) from error

    return loader_vectorstore_dir / chunker_strategy / embedding_provider


def get_chunk_cache_dir(
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    chunker_strategy: str = DEFAULT_CHUNKER_STRATEGY,
) -> Path:
    return CHUNK_CACHE_DIR / loader_strategy / chunker_strategy
