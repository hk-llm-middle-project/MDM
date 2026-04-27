"""Chunk parsed PDF documents by diagram id."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

try:
    from parsing.storage import load_documents_json, save_documents_json
except ModuleNotFoundError:
    from storage import load_documents_json, save_documents_json


DIAGRAM_ID_PATTERN = r"[\ucc28\ubcf4\uac70]\d+(?:-\d+)?"
DIAGRAM_GROUP_PATTERN = r"[\ucc28\ubcf4\uac70]\d+"
INDEX_DIAGRAM_PATTERN = r"[\[\uff3b](" + DIAGRAM_GROUP_PATTERN + r")[\]\uff3d]"
SECTION_PATTERN = r"^[\uac00-\ud7a3]\.\s+(.+)$"
SUBSECTION_PATTERN = r"^\(\d+\)\s+(.+)$"
PAGE_NUMBER_PATTERN = r"\s+\d{2,4}$"
PAREN_CONTENT_PATTERN = r"\([^)]*\)"
NOTE_PREFIX_PATTERN = r"^\s*-?\s*\u203b"

IMAGE_CATEGORIES = {"figure", "chart", "table"}
TEXT_CATEGORIES = {"paragraph", "caption", "list", "heading1"}
SKIP_CATEGORIES = {"header", "footer"}
INDEX_CATEGORY = "index"
TABLE_CATEGORY = "table"

PREFACE_CHUNK_TYPE = "preface"
GENERAL_CHUNK_TYPE = "general"
TEXT_CHUNK_TYPE = "text"
IMAGE_CHUNK_TYPE = "image"
CHILD_CHUNK_TYPE = "child"

NON_TABLE_IMAGE_CATEGORIES = IMAGE_CATEGORIES - {TABLE_CATEGORY}
PREFACE_PAGE_MAX = 9
OUTPUT_FILENAME = "chunked_documents_v2.json"


IndexMapping = dict[str, dict[str, str | None]]


def _category(doc: Document) -> str:
    return str(doc.metadata.get("category", "")).lower()


def _page(doc: Document) -> Any:
    return doc.metadata.get("page")


def _page_number(doc: Document) -> int | None:
    page = _page(doc)
    if isinstance(page, int):
        return page
    if isinstance(page, str) and page.isdigit():
        return int(page)
    return None


def _is_preface_doc(doc: Document) -> bool:
    page = _page_number(doc)
    return _category(doc) == INDEX_CATEGORY or (page is not None and page <= PREFACE_PAGE_MAX)


def _clean_index_title(line: str) -> str:
    title = re.sub(PAGE_NUMBER_PATTERN, "", line).strip()
    title = re.sub(INDEX_DIAGRAM_PATTERN, "", title).strip()
    title = re.sub(SECTION_PATTERN, r"\1", title).strip()
    title = re.sub(SUBSECTION_PATTERN, r"\1", title).strip()
    title = re.sub(r"^\d+\)\s+", "", title).strip()
    title = re.sub(PAREN_CONTENT_PATTERN, "", title).strip()
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _first_markdown_cell(page_content: str) -> str:
    first_line = page_content.strip().splitlines()[0] if page_content.strip() else ""
    if not first_line.startswith("|"):
        return page_content

    cells = [cell.strip() for cell in first_line.strip().strip("|").split("|")]
    return cells[0] if cells else ""


def _metadata_for(diagram_id: str | None, mapping: IndexMapping) -> dict[str, str | None]:
    hierarchy = mapping.get(diagram_id or "", {})
    return {
        "diagram_id": diagram_id,
        "section": hierarchy.get("section"),
        "subsection": hierarchy.get("subsection"),
    }


def _is_note(doc: Document) -> bool:
    return _category(doc) in {"list", "paragraph"} and bool(re.match(NOTE_PREFIX_PATTERN, doc.page_content.strip()))


def _make_single_chunk(doc: Document, chunk_type: str) -> Document | None:
    page_content = doc.page_content.strip()
    if not page_content:
        return None

    metadata: dict[str, Any] = {
        "chunk_type": chunk_type,
        "diagram_id": None,
        "section": None,
        "subsection": None,
        "page": _page(doc),
    }
    for key in ("description", "image_path"):
        if key in doc.metadata:
            metadata[key] = doc.metadata[key]

    return Document(page_content=page_content, metadata=metadata)


def _make_image_chunk(doc: Document, diagram_id: str, mapping: IndexMapping) -> Document:
    metadata: dict[str, Any] = {
        "chunk_type": IMAGE_CHUNK_TYPE,
        **_metadata_for(diagram_id, mapping),
        "page": _page(doc),
    }

    for key in ("description", "image_path"):
        if key in doc.metadata:
            metadata[key] = doc.metadata[key]

    return Document(page_content=doc.page_content, metadata=metadata)


def _flush_general_buffer(buffer: list[Document]) -> list[Document]:
    chunks: list[Document] = []
    for doc in buffer:
        chunk = _make_single_chunk(doc, GENERAL_CHUNK_TYPE)
        if chunk:
            chunks.append(chunk)
    buffer.clear()
    return chunks


def build_index_mapping(docs: list[Document]) -> IndexMapping:
    """Build diagram hierarchy mapping from index documents."""
    group_mapping: IndexMapping = {}
    exact_mapping: IndexMapping = {}
    current_section: str | None = None
    current_subsection: str | None = None

    for doc in docs:
        if _category(doc) != INDEX_CATEGORY:
            continue

        for raw_line in doc.page_content.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            if re.match(SECTION_PATTERN, line):
                current_section = _clean_index_title(line)
                current_subsection = None
                continue

            if re.match(SUBSECTION_PATTERN, line):
                current_subsection = _clean_index_title(line)

            for diagram_id in re.findall(DIAGRAM_ID_PATTERN, line):
                exact_mapping[diagram_id] = {
                    "section": current_section,
                    "subsection": current_subsection,
                }

            for group_id in re.findall(INDEX_DIAGRAM_PATTERN, line):
                group_mapping[group_id] = {
                    "section": current_section,
                    "subsection": current_subsection,
                }

    for doc in docs:
        diagram_id = extract_diagram_id(doc.page_content)
        if not diagram_id or diagram_id in exact_mapping:
            continue

        group_match = re.match(DIAGRAM_GROUP_PATTERN, diagram_id)
        if not group_match:
            continue

        hierarchy = group_mapping.get(group_match.group(0))
        if hierarchy:
            exact_mapping[diagram_id] = hierarchy.copy()

    return exact_mapping


def extract_diagram_id(page_content: str) -> str | None:
    """Extract a diagram id from the first markdown table cell."""
    target = _first_markdown_cell(page_content)
    match = re.search(DIAGRAM_ID_PATTERN, target)
    return match.group(0) if match else None


def flush_text_buffer(
    buffer: list[Document],
    diagram_id: str | None,
    mapping: IndexMapping,
) -> Document | None:
    """Convert accumulated diagram text documents into a single text chunk."""
    if not buffer:
        return None

    page_content = "\n".join(doc.page_content.strip() for doc in buffer if doc.page_content.strip())
    if not page_content:
        buffer.clear()
        return None

    first_page = _page(buffer[0])
    metadata = {
        "chunk_type": TEXT_CHUNK_TYPE,
        **_metadata_for(diagram_id, mapping),
        "page": first_page,
    }
    buffer.clear()
    return Document(page_content=page_content, metadata=metadata)


def _append_note_to_front(buffer: list[Document], doc: Document, note_count: int) -> int:
    buffer.insert(note_count, doc)
    return note_count + 1


def _split_text_for_children(page_content: str) -> list[str]:
    if not page_content.strip():
        return []

    pieces = page_content.split("\n# ")
    child_contents = [pieces[0]]
    child_contents.extend(f"# {piece}" for piece in pieces[1:])
    return [piece.strip() for piece in child_contents if piece.strip()]


def _extract_heading(child_content: str) -> str:
    first_line = child_content.splitlines()[0].strip()
    return first_line[1:].strip() if first_line.startswith("#") else first_line


def create_child_chunks(parent: Document) -> list[Document]:
    """Split a text parent chunk into child chunks by heading."""
    if parent.metadata.get("chunk_type") != TEXT_CHUNK_TYPE:
        return []

    child_chunks: list[Document] = []
    for child_content in _split_text_for_children(parent.page_content):
        child_chunks.append(
            Document(
                page_content=child_content,
                metadata={
                    "chunk_type": CHILD_CHUNK_TYPE,
                    "parent_id": parent.metadata.get("diagram_id"),
                    "diagram_id": parent.metadata.get("diagram_id"),
                    "section": parent.metadata.get("section"),
                    "subsection": parent.metadata.get("subsection"),
                    "page": parent.metadata.get("page"),
                    "heading": _extract_heading(child_content),
                },
            )
        )
    return child_chunks


def append_child_chunks(chunks: list[Document]) -> list[Document]:
    """Append child chunks immediately after each text parent chunk."""
    expanded_chunks: list[Document] = []
    for chunk in chunks:
        expanded_chunks.append(chunk)
        if chunk.metadata.get("chunk_type") == TEXT_CHUNK_TYPE:
            expanded_chunks.extend(create_child_chunks(chunk))
    return expanded_chunks


def chunk_documents(docs: list[Document], mapping: IndexMapping) -> list[Document]:
    """Create chunks by walking parsed documents in order."""
    chunks: list[Document] = []
    text_buffer: list[Document] = []
    pending_next_buffer: list[Document] = []
    current_diagram_id: str | None = None
    note_prefix_count = 0
    after_diagram_table = False

    for index, doc in enumerate(docs, start=1):
        category = _category(doc)
        page = _page(doc)

        try:
            if _is_preface_doc(doc):
                chunks.extend(_flush_general_buffer(pending_next_buffer))
                preface_chunk = _make_single_chunk(doc, PREFACE_CHUNK_TYPE)
                if preface_chunk:
                    chunks.append(preface_chunk)
                continue

            if category in SKIP_CATEGORIES:
                continue

            if category == TABLE_CATEGORY:
                next_diagram_id = extract_diagram_id(doc.page_content)
                if not next_diagram_id:
                    chunks.extend(_flush_general_buffer(pending_next_buffer))
                    preview = doc.page_content.replace("\n", " ")[:50]
                    print(f"[WARN] diagram_id \ucd94\ucd9c \uc2e4\ud328 table - page={page}, page_content={preview}")
                    continue

                flushed = flush_text_buffer(text_buffer, current_diagram_id, mapping)
                if flushed:
                    chunks.append(flushed)

                current_diagram_id = next_diagram_id
                text_buffer.extend(pending_next_buffer)
                pending_next_buffer.clear()
                note_prefix_count = 0
                after_diagram_table = True
                chunks.append(_make_image_chunk(doc, current_diagram_id, mapping))
                continue

            if category in NON_TABLE_IMAGE_CATEGORIES:
                chunks.extend(_flush_general_buffer(pending_next_buffer))
                if current_diagram_id is None:
                    continue

                flushed = flush_text_buffer(text_buffer, current_diagram_id, mapping)
                if flushed:
                    chunks.append(flushed)

                chunks.append(_make_image_chunk(doc, current_diagram_id, mapping))
                after_diagram_table = False
                note_prefix_count = 0
                continue

            if category in TEXT_CATEGORIES:
                if current_diagram_id:
                    if after_diagram_table and _is_note(doc):
                        note_prefix_count = _append_note_to_front(text_buffer, doc, note_prefix_count)
                    else:
                        text_buffer.append(doc)
                        after_diagram_table = False
                    continue

                pending_next_buffer.append(doc)
                continue

        except Exception as exc:
            print(
                f"[ERROR] \uc694\uc18c \ucc98\ub9ac \uc2e4\ud328 - index={index}, category={category}, page={page}, "
                f"\uc5d0\ub7ec \ud0c0\uc785={type(exc).__name__}, \uc5d0\ub7ec \uba54\uc2dc\uc9c0={exc}"
            )
            continue

    flushed = flush_text_buffer(text_buffer, current_diagram_id, mapping)
    if flushed:
        chunks.append(flushed)

    chunks.extend(_flush_general_buffer(pending_next_buffer))
    return chunks


def run_chunking(json_path: str) -> None:
    """Run the full chunking pipeline and save chunked_documents.json."""
    input_path = Path(json_path)
    output_path = input_path.with_name(OUTPUT_FILENAME)

    docs = load_documents_json(input_path)
    mapping = build_index_mapping(docs)
    print(f"[INFO] \ucc98\ub9ac \uc2dc\uc791 - \uc804\uccb4 \uc694\uc18c \uc218: {len(docs)}, index \ub9e4\ud551 \uac74\uc218: {len(mapping)}")

    if output_path.exists():
        print(f"[INFO] \uae30\uc874 {OUTPUT_FILENAME}\uc774 \uc788\uc5b4 \ub36e\uc5b4\uc501\ub2c8\ub2e4: {output_path}")

    parent_chunks = chunk_documents(docs, mapping)
    chunks = append_child_chunks(parent_chunks)
    text_count = sum(1 for doc in chunks if doc.metadata.get("chunk_type") == TEXT_CHUNK_TYPE)
    child_count = sum(1 for doc in chunks if doc.metadata.get("chunk_type") == CHILD_CHUNK_TYPE)
    image_count = sum(1 for doc in chunks if doc.metadata.get("chunk_type") == IMAGE_CHUNK_TYPE)
    general_count = sum(1 for doc in chunks if doc.metadata.get("chunk_type") == GENERAL_CHUNK_TYPE)
    preface_count = sum(1 for doc in chunks if doc.metadata.get("chunk_type") == PREFACE_CHUNK_TYPE)
    orphan_count = sum(
        1
        for doc in chunks
        if doc.metadata.get("diagram_id") is None
        and doc.metadata.get("chunk_type") not in {PREFACE_CHUNK_TYPE, GENERAL_CHUNK_TYPE}
    )

    print(
        f"[INFO] \ucc98\ub9ac \uc644\ub8cc - \uc804\uccb4 \uccad\ud06c \uc218: {len(chunks)}, "
        f"text \ubd80\ubaa8 \uccad\ud06c \uc218: {text_count}, child \uc790\uc2dd \uccad\ud06c \uc218: {child_count}, "
        f"image \uccad\ud06c \uc218: {image_count}, "
        f"general \uccad\ud06c \uc218: {general_count}, preface \uccad\ud06c \uc218: {preface_count}, "
        f"orphan \uccad\ud06c \uc218: {orphan_count}"
    )
    save_documents_json(chunks, output_path)
    print(f"[INFO] \uc800\uc7a5 \uc644\ub8cc: {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk parsed documents by diagram id.")
    parser.add_argument("json_path", help="Path to parsed_documents.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_chunking(args.json_path)
