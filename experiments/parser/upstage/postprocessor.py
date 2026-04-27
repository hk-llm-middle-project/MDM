"""Post-process chunked documents for vector ingestion."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


MIN_CONTENT_LENGTH = 10
HEADING_PREFIX = "# "
TABLE_SEPARATOR_PATTERN = "\n---\n"
PREFACE_CHUNK_TYPE = "preface"
PARTY_TYPE_BY_PREFIX = {
    "\ubcf4": "\ubcf4\ud589\uc790",
    "\ucc28": "\uc790\ub3d9\ucc28",
    "\uac70": "\uc790\uc804\uac70",
}


DocumentDict = dict[str, Any]


def clean_page_content(text: str) -> str:
    """Normalize chunk text content."""
    cleaned = text.replace(HEADING_PREFIX, "")
    cleaned = cleaned.replace(TABLE_SEPARATOR_PATTERN, "\n")
    return cleaned.strip()


def _party_type(diagram_id: Any) -> str | None:
    if not isinstance(diagram_id, str) or not diagram_id:
        return None

    for prefix, party_type in PARTY_TYPE_BY_PREFIX.items():
        if diagram_id.startswith(prefix):
            return party_type
    return None


def add_metadata(doc: DocumentDict) -> DocumentDict:
    """Add derived metadata fields to a document dict."""
    metadata = doc.setdefault("metadata", {})
    diagram_id = metadata.get("diagram_id")
    metadata["party_type"] = _party_type(diagram_id)
    metadata["location"] = metadata.get("subsection")
    return doc


def should_remove(doc: DocumentDict) -> bool:
    """Return True if a non-preface chunk is too short to keep."""
    metadata = doc.get("metadata", {})
    if metadata.get("chunk_type") == PREFACE_CHUNK_TYPE:
        return False
    return len(str(doc.get("page_content", "")).strip()) < MIN_CONTENT_LENGTH


def _load_json(input_path: str | Path) -> list[DocumentDict]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _save_json(docs: list[DocumentDict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(docs, fp, ensure_ascii=False, indent=2)
    return path


def run_postprocess(input_path: str, output_path: str) -> None:
    """Load, clean, enrich, filter, and save chunk dicts."""
    docs = _load_json(input_path)
    print(f"[INFO] \ucc98\ub9ac \uc2dc\uc791 - \uc804\uccb4 \uccad\ud06c \uc218: {len(docs)}")

    processed_docs: list[DocumentDict] = []
    remove_count = 0

    for doc in docs:
        if should_remove(doc):
            remove_count += 1
            continue

        doc["page_content"] = clean_page_content(str(doc.get("page_content", "")))
        doc = add_metadata(doc)

        if should_remove(doc):
            remove_count += 1
            continue

        processed_docs.append(doc)

    distribution = Counter(doc.get("metadata", {}).get("chunk_type") for doc in processed_docs)
    _save_json(processed_docs, output_path)
    print(
        f"[INFO] \ucc98\ub9ac \uc644\ub8cc - \uc81c\uac70 \uc218: {remove_count}, "
        f"\ucd5c\uc885 \uccad\ud06c \uc218: {len(processed_docs)}, chunk_type\ubcc4 \ubd84\ud3ec: {dict(distribution)}"
    )
    print(f"[INFO] \uc800\uc7a5 \uc644\ub8cc: {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process chunked documents.")
    parser.add_argument("input_path", help="Path to chunked_documents_v2.json")
    parser.add_argument("output_path", help="Path to chunked_documents_final.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_postprocess(args.input_path, args.output_path)
