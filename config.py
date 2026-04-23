"""Project configuration for the basic RAG app."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "raw" / "230630_자동차사고 과실비율 인정기준_최종.pdf"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 0
RETRIEVER_K = 3

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
