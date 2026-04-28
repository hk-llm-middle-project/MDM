"""Project configuration for the basic RAG app."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "raw" / "230630_자동차사고 과실비율 인정기준_최종.pdf"
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"
LLAMA_MD_DIR = BASE_DIR / "data" / "llama_md"
UPSTAGE_OUTPUT_DIR = BASE_DIR / "data" / "upstage_output"
UPSTAGE_MAIN_PDF_OUTPUT_DIR = UPSTAGE_OUTPUT_DIR / "main_pdf"
UPSTAGE_FINAL_DOCUMENTS_PATH = (
    UPSTAGE_MAIN_PDF_OUTPUT_DIR / "final" / "chunked_documents_final.json"
)
UPSTAGE_RAW_DOCUMENTS_PATH = (
    UPSTAGE_MAIN_PDF_OUTPUT_DIR / "raw" / "parsed_documents_raw.json"
)
DEFAULT_LOADER_STRATEGY = "pdfplumber"
LOADER_VECTORSTORE_DIRS = {
    "pdfplumber": VECTORSTORE_DIR / "pdfplumber",
    "llamaparser": VECTORSTORE_DIR / "llamaparser",
    "llama-parse": VECTORSTORE_DIR / "llamaparser",
    "upstage": VECTORSTORE_DIR / "upstage",
}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 0
INDEX_BATCH_SIZE = 100
RETRIEVER_K = 3

DEFAULT_EMBEDDING_PROVIDER = "bge"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GOOGLE_EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gpt-4o-mini"


def get_vectorstore_dir(
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
) -> Path:
    try:
        loader_vectorstore_dir = LOADER_VECTORSTORE_DIRS[loader_strategy]
    except KeyError as error:
        available = ", ".join(sorted(LOADER_VECTORSTORE_DIRS))
        raise ValueError(
            f"Unknown loader strategy: {loader_strategy}. Available strategies: {available}"
        ) from error

    return loader_vectorstore_dir / embedding_provider
