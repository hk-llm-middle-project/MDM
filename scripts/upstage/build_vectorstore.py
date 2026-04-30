"""Build a Chroma vectorstore from the curated Upstage custom chunks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    BASE_DIR,
    DEFAULT_EMBEDDING_PROVIDER,
    INDEX_BATCH_SIZE,
    UPSTAGE_CUSTOM_DOCUMENTS_PATH,
    get_vectorstore_dir,
)
from rag.embeddings import EMBEDDING_STRATEGIES  # noqa: E402


INPUT_PATH = UPSTAGE_CUSTOM_DOCUMENTS_PATH
VECTORSTORE_STRATEGY = "upstage"
DEFAULT_CHUNKER_STRATEGY = "custom"
DEFAULT_EXCLUDED_CHUNK_TYPES = {"preface"}
IMAGE_CONTENT_METADATA_KEY = "description"
ALLOWED_METADATA_TYPES = (str, int, float, bool)
EXCLUDED_METADATA_KEYS = {"section", "subsection"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an Upstage Chroma vectorstore with the selected embedding provider."
    )
    parser.add_argument(
        "--embedding-provider",
        choices=sorted(EMBEDDING_STRATEGIES),
        default=DEFAULT_EMBEDDING_PROVIDER,
        help="Embedding provider to use for indexing.",
    )
    parser.add_argument(
        "--exclude-text",
        action="store_true",
        help="Exclude parent text chunks and embed only general/image/child chunks.",
    )
    return parser.parse_args()


def load_chunk_payload(input_path: Path = INPUT_PATH) -> list[dict]:
    with input_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def select_embedding_chunks(chunks: list[dict], include_text: bool = True) -> list[dict]:
    excluded_chunk_types = set(DEFAULT_EXCLUDED_CHUNK_TYPES)
    if not include_text:
        excluded_chunk_types.add("text")

    return [
        chunk
        for chunk in chunks
        if chunk.get("metadata", {}).get("chunk_type") not in excluded_chunk_types
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
        if key in EXCLUDED_METADATA_KEYS or value is None:
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
        print(f"[INFO] embedding batch planned: {start + 1}-{end}/{total_count}")


def main() -> None:
    args = parse_args()
    embedding_provider = args.embedding_provider
    include_text = not args.exclude_text
    output_dir = get_vectorstore_dir(
        VECTORSTORE_STRATEGY,
        embedding_provider,
        chunker_strategy=DEFAULT_CHUNKER_STRATEGY,
    )

    load_dotenv(BASE_DIR / ".env")
    if (
        embedding_provider == "openai"
        and (not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "null")
    ):
        raise RuntimeError("OPENAI_API_KEY is required for the OpenAI embedding provider.")

    chunks = load_chunk_payload(INPUT_PATH)
    distribution = Counter(chunk.get("metadata", {}).get("chunk_type") for chunk in chunks)
    selected_chunks = select_embedding_chunks(chunks, include_text=include_text)
    documents = build_documents(selected_chunks)

    excluded_count = len(chunks) - len(selected_chunks)
    empty_removed_count = len(selected_chunks) - len(documents)

    print(f"[INFO] total chunks: {len(chunks)}")
    print(f"[INFO] excluded chunks: {excluded_count}")
    print(f"[INFO] empty page_content removed: {empty_removed_count}")
    print(f"[INFO] embedding targets: {len(documents)}")
    print(f"[INFO] chunk_type distribution: {dict(distribution)}")
    print(f"[INFO] excluded chunk_types: {sorted(DEFAULT_EXCLUDED_CHUNK_TYPES)}")
    print(f"[INFO] excluded metadata keys: {sorted(EXCLUDED_METADATA_KEYS)}")
    print(f"[INFO] include_text: {include_text}")
    print(f"[INFO] embedding_provider: {embedding_provider}")
    print(f"[INFO] output_dir: {output_dir}")

    log_batch_plan(len(documents), INDEX_BATCH_SIZE)
    from rag.indexer import build_vectorstore

    build_vectorstore(
        documents=documents,
        persist_directory=output_dir,
        batch_size=INDEX_BATCH_SIZE,
        embedding_provider=embedding_provider,
    )
    print(f"[INFO] saved vectorstore: {output_dir}")


if __name__ == "__main__":
    main()
