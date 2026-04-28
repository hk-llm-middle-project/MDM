"""Build a Chroma vectorstore from Upstage post-processed chunks."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BASE_DIR, INDEX_BATCH_SIZE, get_vectorstore_dir
from rag.indexer import build_vectorstore


INPUT_PATH = (
    BASE_DIR
    / "data"
    / "upstage_output"
    / "main_pdf"
    / "final"
    / "chunked_documents_final.json"
)
VECTORSTORE_STRATEGY = "upstage"
OUTPUT_DIR = get_vectorstore_dir(VECTORSTORE_STRATEGY)
EXCLUDED_CHUNK_TYPES = {"preface", "text"}
INCLUDED_CHUNK_TYPES = {"general", "image", "child"}
IMAGE_CONTENT_METADATA_KEY = "description"
ALLOWED_METADATA_TYPES = (str, int, float, bool)


def load_chunk_payload(input_path: Path = INPUT_PATH) -> list[dict]:
    with input_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def select_embedding_chunks(chunks: list[dict]) -> list[dict]:
    return [
        chunk
        for chunk in chunks
        if chunk.get("metadata", {}).get("chunk_type") in INCLUDED_CHUNK_TYPES
    ]


def build_document(chunk: dict) -> Document | None:
    metadata = sanitize_metadata(dict(chunk.get("metadata", {})))
    chunk_type = metadata.get("chunk_type")

    if chunk_type == "image":
        page_content = str(metadata.get(IMAGE_CONTENT_METADATA_KEY, "")).strip()
    else:
        page_content = str(chunk.get("page_content", "")).strip()

    if not page_content:
        return None

    return Document(page_content=page_content, metadata=metadata)


def sanitize_metadata(metadata: dict) -> dict:
    sanitized: dict = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, ALLOWED_METADATA_TYPES):
            sanitized[key] = value
        else:
            sanitized[key] = json.dumps(value, ensure_ascii=False)
    return sanitized


def build_documents(chunks: list[dict]) -> list[Document]:
    documents: list[Document] = []
    for chunk in chunks:
        document = build_document(chunk)
        if document is not None:
            documents.append(document)
    return documents


def log_batch_plan(total_count: int, batch_size: int = INDEX_BATCH_SIZE) -> None:
    for start in range(0, total_count, batch_size):
        end = min(start + batch_size, total_count)
        print(f"[INFO] 임베딩 배치 예정: {start + 1}-{end}/{total_count}")


def main() -> None:
    load_dotenv(BASE_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "null":
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다. 프로젝트 루트의 .env를 확인해주세요.")

    chunks = load_chunk_payload(INPUT_PATH)
    distribution = Counter(chunk.get("metadata", {}).get("chunk_type") for chunk in chunks)
    selected_chunks = select_embedding_chunks(chunks)
    documents = build_documents(selected_chunks)

    excluded_count = len(chunks) - len(selected_chunks)
    empty_removed_count = len(selected_chunks) - len(documents)

    print(f"[INFO] 전체 청크 수: {len(chunks)}")
    print(f"[INFO] 제외 수: {excluded_count}")
    print(f"[INFO] 빈 page_content 제외 수: {empty_removed_count}")
    print(f"[INFO] 임베딩 대상 수: {len(documents)}")
    print(f"[INFO] chunk_type별 분포: {dict(distribution)}")

    log_batch_plan(len(documents), INDEX_BATCH_SIZE)
    build_vectorstore(
        documents=documents,
        persist_directory=OUTPUT_DIR,
        batch_size=INDEX_BATCH_SIZE,
    )
    print(f"[INFO] 저장 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
