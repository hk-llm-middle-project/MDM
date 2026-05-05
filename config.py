"""기본 RAG 앱의 프로젝트 설정입니다."""

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "raw" / "230630_자동차사고 과실비율 인정기준_최종.pdf"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
CHUNK_CACHE_DIR = BASE_DIR / "data" / "chunks"
EMBEDDING_QUERY_CACHE_DIR = BASE_DIR / "data" / "embedding_cache"
EMBEDDING_QUERY_CACHE_ENABLED = True
PAGE_METADATA_DIR = BASE_DIR / "data" / "metadata"
PAGE_METADATA_PATH = PAGE_METADATA_DIR / "main_pdf_page_metadata.json"
LLAMA_MD_DIR = BASE_DIR / "data" / "llama_md"
PDFPLUMBER_OUT_DIR = BASE_DIR / "data" / "pdfplumber_output"
UPSTAGE_OUTPUT_DIR = BASE_DIR / "data" / "upstage_output"
UPSTAGE_MAIN_PDF_OUTPUT_DIR = UPSTAGE_OUTPUT_DIR / "main_pdf"
UPSTAGE_RAW_DOCUMENTS_PATH = (
    UPSTAGE_MAIN_PDF_OUTPUT_DIR / "raw" / "parsed_documents.json"
)
UPSTAGE_LEGACY_RAW_DOCUMENTS_PATH = (
    UPSTAGE_MAIN_PDF_OUTPUT_DIR / "raw" / "parsed_documents_raw.json"
)
UPSTAGE_CUSTOM_DOCUMENTS_PATH = (
    UPSTAGE_MAIN_PDF_OUTPUT_DIR / "final" / "chunked_documents_final.table_clean.json"
)
UPSTAGE_FINAL_DOCUMENTS_PATH = UPSTAGE_CUSTOM_DOCUMENTS_PATH
DEFAULT_LOADER_STRATEGY = "upstage" # llamaparser
DEFAULT_CHUNKER_STRATEGY = "custom" # case-boundary
LOADER_VECTORSTORE_DIRS = {
    "pdfplumber": VECTORSTORE_DIR / "pdfplumber",
    "llamaparser": VECTORSTORE_DIR / "llamaparser",
    "llama-parse": VECTORSTORE_DIR / "llamaparser",
    "upstage": VECTORSTORE_DIR / "upstage",
}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
INDEX_BATCH_SIZE = 100
RETRIEVER_K = 5
DEFAULT_RETRIEVER_STRATEGY = "ensemble_parent"
DEFAULT_RERANKER_STRATEGY = "none"
DEFAULT_RERANKER_CANDIDATE_K = 30
DEFAULT_RERANKER_FINAL_K = 3

DEFAULT_ENSEMBLE_BM25_WEIGHT = 0.5
ENSEMBLE_RETRIEVER_STRATEGIES = ("ensemble", "ensemble_parent")
ENSEMBLE_CANDIDATE_K_OPTIONS = (5, 10, 20, 30)
DEFAULT_ENSEMBLE_CANDIDATE_K = 20
DEFAULT_ENSEMBLE_USE_CHUNK_ID = True
ENSEMBLE_ID_KEY = "chunk_id"

DEFAULT_EMBEDDING_PROVIDER = "bge"
DEFAULT_MODE = "fast"
MODE_PRESETS = {
    "fast": {
        "loader_strategy": "llamaparser",
        "chunker_strategy": "case-boundary",
        "embedding_provider": DEFAULT_EMBEDDING_PROVIDER,
        "retriever_strategy": "ensemble",
        "reranker_strategy": "none",
        "ensemble_bm25_weight": 0.5,
        "ensemble_candidate_k": DEFAULT_ENSEMBLE_CANDIDATE_K,
        "ensemble_use_chunk_id": DEFAULT_ENSEMBLE_USE_CHUNK_ID,
    },
    "thinking": {
        "loader_strategy": "llamaparser",
        "chunker_strategy": "case-boundary",
        "embedding_provider": DEFAULT_EMBEDDING_PROVIDER,
        "retriever_strategy": "parent",
        "reranker_strategy": "cross-encoder",
        "ensemble_bm25_weight": DEFAULT_ENSEMBLE_BM25_WEIGHT,
        "ensemble_candidate_k": DEFAULT_ENSEMBLE_CANDIDATE_K,
        "ensemble_use_chunk_id": DEFAULT_ENSEMBLE_USE_CHUNK_ID,
    },
    "pro": {
        "loader_strategy": "upstage",
        "chunker_strategy": "custom",
        "embedding_provider": DEFAULT_EMBEDDING_PROVIDER,
        "retriever_strategy": "ensemble_parent",
        "reranker_strategy": "llm-score",
        "ensemble_bm25_weight": DEFAULT_ENSEMBLE_BM25_WEIGHT,
        "ensemble_candidate_k": DEFAULT_ENSEMBLE_CANDIDATE_K,
        "ensemble_use_chunk_id": DEFAULT_ENSEMBLE_USE_CHUNK_ID,
    },
}
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GOOGLE_EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_QUERY_CACHE_SCHEMA_VERSION = "v1"
LLM_MODEL = "gpt-5-mini"
INTAKE_MODEL = "gpt-4o-mini"
ROUTER_MODEL = "gpt-4o-mini"
RERANKER_LLM_MODEL = "gpt-5-mini"


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


def get_embedding_query_cache_enabled() -> bool:
    """검색 질의 임베딩 캐시를 사용할지 반환합니다."""
    default = "true" if EMBEDDING_QUERY_CACHE_ENABLED else "false"
    value = _optional_env("EMBEDDING_QUERY_CACHE_ENABLED", default)
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def get_embedding_query_cache_dir() -> Path:
    """검색 질의 임베딩 캐시 디렉터리를 반환합니다."""
    return EMBEDDING_QUERY_CACHE_DIR


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
