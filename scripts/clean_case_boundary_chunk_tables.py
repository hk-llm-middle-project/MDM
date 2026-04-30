"""Clean table text after case-boundary chunking.

This postprocessor keeps case-boundary metadata stable. It cleans only child
chunks that contain markdown tables, then rebuilds affected parent chunks from
their cleaned children.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import re
from pathlib import Path
from typing import Any

try:
    from scripts.clean_llamaparser_diagram_tables import clean_markdown
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from clean_llamaparser_diagram_tables import clean_markdown


DocumentDict = dict[str, Any]

TABLE_LINE_PATTERN = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
CHILD_CHUNK_TYPE = "child"
PARENT_CHUNK_TYPE = "parent"


def _metadata(doc: DocumentDict) -> dict[str, Any]:
    metadata = doc.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return {}


def _chunk_id(doc: DocumentDict) -> int | None:
    chunk_id = _metadata(doc).get("chunk_id")
    return chunk_id if isinstance(chunk_id, int) else None


def _parent_id(doc: DocumentDict) -> int | None:
    parent_id = _metadata(doc).get("parent_id")
    return parent_id if isinstance(parent_id, int) else None


def _chunk_type(doc: DocumentDict) -> str | None:
    chunk_type = _metadata(doc).get("chunk_type")
    return chunk_type if isinstance(chunk_type, str) else None


def _is_table_child(doc: DocumentDict) -> bool:
    if _chunk_type(doc) != CHILD_CHUNK_TYPE:
        return False
    return TABLE_LINE_PATTERN.search(str(doc.get("page_content", ""))) is not None


def _children_by_parent(docs: list[DocumentDict]) -> dict[int, list[DocumentDict]]:
    children: dict[int, list[DocumentDict]] = {}
    for doc in docs:
        parent_id = _parent_id(doc)
        if _chunk_type(doc) != CHILD_CHUNK_TYPE or parent_id is None:
            continue
        children.setdefault(parent_id, []).append(doc)

    for parent_children in children.values():
        parent_children.sort(
            key=lambda doc: (
                _chunk_id(doc) is None,
                _chunk_id(doc) if _chunk_id(doc) is not None else 0,
            )
        )
    return children


def clean_case_boundary_tables(docs: list[DocumentDict]) -> list[DocumentDict]:
    """Return docs with cleaned table children and synchronized parents."""
    cleaned_docs = deepcopy(docs)
    changed_parent_ids: set[int] = set()

    for doc in cleaned_docs:
        if not _is_table_child(doc):
            continue

        original_text = str(doc.get("page_content", ""))
        cleaned_text = clean_markdown(original_text)
        if cleaned_text == original_text:
            continue

        doc["page_content"] = cleaned_text
        parent_id = _parent_id(doc)
        if parent_id is not None:
            changed_parent_ids.add(parent_id)

    if not changed_parent_ids:
        return cleaned_docs

    children = _children_by_parent(cleaned_docs)
    for doc in cleaned_docs:
        chunk_id = _chunk_id(doc)
        if _chunk_type(doc) != PARENT_CHUNK_TYPE or chunk_id not in changed_parent_ids:
            continue

        parent_children = children.get(chunk_id, [])
        if not parent_children:
            continue

        doc["page_content"] = "\n\n".join(
            str(child.get("page_content", "")).strip()
            for child in parent_children
            if str(child.get("page_content", "")).strip()
        ).strip()

    return cleaned_docs


def load_documents_json(input_path: str | Path) -> list[DocumentDict]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def save_documents_json(docs: list[DocumentDict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(docs, fp, ensure_ascii=False, indent=2)
    return path


def run_clean_case_boundary_tables(input_path: str | Path, output_path: str | Path) -> None:
    docs = load_documents_json(input_path)
    cleaned_docs = clean_case_boundary_tables(docs)
    save_documents_json(cleaned_docs, output_path)

    changed_docs = sum(
        1
        for before, after in zip(docs, cleaned_docs, strict=True)
        if before.get("page_content") != after.get("page_content")
    )
    distribution = Counter(_chunk_type(doc) for doc in cleaned_docs)
    print(
        "[INFO] cleaned case-boundary chunk tables - "
        f"changed_docs={changed_docs}, total_docs={len(cleaned_docs)}, "
        f"chunk_type_counts={dict(distribution)}"
    )
    print(f"[INFO] saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean LlamaParse diagram tables after case-boundary chunking."
    )
    parser.add_argument("input_path", help="Input chunked documents JSON path.")
    parser.add_argument("output_path", help="Output cleaned documents JSON path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_clean_case_boundary_tables(args.input_path, args.output_path)
