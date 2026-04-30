"""Build a table-clean copy of the finalized Upstage chunks.

This script does not modify the finalized source file. It reads
``chunked_documents_final.json`` and writes a copy whose markdown table blocks
are normalized for retrieval/debugging:

- remove markdown image placeholders such as ``![image](/image/placeholder)``
- flatten nested HTML ``<table>...</table>`` fragments into plain cell text
- regenerate markdown table separator rows
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/upstage_output/main_pdf/final/chunked_documents_final.json")
DEFAULT_OUTPUT = Path("data/upstage_output/main_pdf/final/chunked_documents_final.table_clean.json")
DEFAULT_REPORT = Path("data/upstage_output/main_pdf/final/chunked_documents_final.table_clean.report.json")
DEFAULT_MARKDOWN_OUTPUT = Path("data/upstage_output/main_pdf/final/chunked_documents_final.table_clean.tables.md")

MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
HTML_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
FIGCAP_RE = re.compile(r"<figcaption\b[^>]*>.*?</figcaption>", re.IGNORECASE | re.DOTALL)
HTML_TAG_RE = re.compile(r"<[^>]+>")
MARKDOWN_SEPARATOR_RE = re.compile(r"^\|[\s:\-|]+\|$")


class HTMLTableTextParser(HTMLParser):
    """Extract readable text rows from a small HTML table fragment."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.current_row: list[str] | None = None
        self.current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"}:
            self.current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.current_cell is not None:
            text = normalize_space("".join(self.current_cell))
            if self.current_row is not None:
                self.current_row.append(text)
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            if any(cell for cell in self.current_row):
                self.rows.append(self.current_row)
            self.current_row = None

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.append(data)


def normalize_space(text: str) -> str:
    return " ".join(html.unescape(text).split())


def flatten_html_table(fragment: str) -> str:
    parser = HTMLTableTextParser()
    parser.feed(fragment)
    row_texts = [" / ".join(cell for cell in row if cell) for row in parser.rows]
    return " ; ".join(row for row in row_texts if row)


def clean_cell(cell: str) -> str:
    """Clean one markdown table cell without inserting markdown table delimiters."""
    cell = MD_IMAGE_RE.sub("", cell)
    cell = FIGCAP_RE.sub("", cell)
    cell = HTML_TABLE_RE.sub(lambda match: flatten_html_table(match.group(0)), cell)
    cell = HTML_TAG_RE.sub("", cell)
    # The pipe character would split markdown cells after rendering.
    cell = cell.replace("|", " ")
    return normalize_space(cell)


def clean_non_table_text(text: str) -> str:
    """Remove image placeholders and flatten stray HTML tables outside markdown tables."""
    text = MD_IMAGE_RE.sub("", text)
    text = FIGCAP_RE.sub("", text)
    text = HTML_TABLE_RE.sub(lambda match: flatten_html_table(match.group(0)), text)
    text = HTML_TAG_RE.sub("", text)
    return text.strip()


def collect_table_rows(table_text: str) -> list[list[str]]:
    """Collect markdown rows while preserving multiline cells from Upstage output."""
    raw_rows: list[list[str]] = []
    current: list[str] = []

    for line in table_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            if current:
                raw_rows.append(current)
            current = [stripped]
        elif current:
            current.append(stripped)

    if current:
        raw_rows.append(current)

    rows: list[list[str]] = []
    for line_group in raw_rows:
        joined = " ".join(part.strip() for part in line_group if part.strip())
        if MARKDOWN_SEPARATOR_RE.match(joined):
            continue
        rows.append([clean_cell(cell) for cell in joined.strip("|").split("|")])

    return rows


def render_markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rendered: list[str] = []
    for index, row in enumerate(rows):
        padded = row + [""] * (width - len(row))
        rendered.append("| " + " | ".join(padded) + " |")
        if index == 0:
            rendered.append("| " + " | ".join(["---"] * width) + " |")
    return "\n".join(rendered)


def split_markdown_table_segments(text: str) -> list[tuple[str, bool]]:
    """Split content into markdown table and non-table segments."""
    segments: list[tuple[str, bool]] = []
    current: list[str] = []
    in_table = False

    for line in text.splitlines():
        stripped = line.strip()
        # Upstage often emits multiline table cells. Once a markdown table has
        # started, non-empty non-pipe lines are still part of the current table
        # row until a blank line closes the table block.
        is_table_line = stripped.startswith("|") or (in_table and bool(stripped))
        if current and is_table_line != in_table:
            segments.append(("\n".join(current), in_table))
            current = []
        current.append(line)
        in_table = is_table_line

    if current:
        segments.append(("\n".join(current), in_table))
    return segments


def clean_table_blocks(content: str) -> tuple[str, int]:
    """Clean markdown table blocks inside a chunk and return changed table count."""
    parts: list[str] = []
    table_count = 0
    for text, is_table in split_markdown_table_segments(content):
        if is_table:
            table_count += 1
            parts.append(render_markdown_table(collect_table_rows(text)))
        else:
            parts.append(clean_non_table_text(text))
    return "\n\n".join(part for part in parts if part.strip()), table_count


def extract_markdown_tables(content: str) -> list[str]:
    """Extract cleaned markdown table blocks from one chunk."""
    tables: list[str] = []
    for text, is_table in split_markdown_table_segments(content):
        if not is_table:
            continue
        table = render_markdown_table(collect_table_rows(text))
        if table.strip():
            tables.append(table)
    return tables


def render_tables_markdown(chunks: list[dict[str, Any]]) -> tuple[str, int]:
    """Render all cleaned table blocks as a real markdown file for preview."""
    sections: list[str] = ["# Cleaned Tables", ""]
    table_count = 0

    for chunk in chunks:
        content = str(chunk.get("page_content", ""))
        if "|" not in content:
            continue

        tables = extract_markdown_tables(content)
        if not tables:
            continue

        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        chunk_id = metadata.get("chunk_id", "-")
        chunk_type = metadata.get("chunk_type", "-")
        diagram_id = metadata.get("diagram_id") or "-"
        parent_id = metadata.get("parent_id")
        page = metadata.get("page", "-")
        image_path = metadata.get("image_path")

        heading = f"## chunk {chunk_id} | {chunk_type} | {diagram_id} | page {page}"
        meta_parts = []
        if parent_id is not None:
            meta_parts.append(f"parent_id: {parent_id}")
        if image_path:
            meta_parts.append(f"image_path: {image_path}")

        sections.append(heading)
        if meta_parts:
            sections.append("")
            sections.append(" / ".join(meta_parts))
        sections.append("")
        sections.append("\n\n".join(tables))
        sections.append("")
        sections.append("---")
        sections.append("")
        table_count += len(tables)

    if sections[-1] == "":
        sections.pop()
    return "\n".join(sections), table_count


def build_table_clean_copy(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = copy.deepcopy(chunks)
    changed_chunks = 0
    table_blocks = 0
    removed_markdown_images = 0
    flattened_html_tables = 0

    for chunk in output:
        content = str(chunk.get("page_content", ""))
        if "|" not in content and "![image]" not in content and "<table" not in content:
            continue

        removed_markdown_images += len(MD_IMAGE_RE.findall(content))
        flattened_html_tables += len(HTML_TABLE_RE.findall(content))
        cleaned, count = clean_table_blocks(content)
        table_blocks += count
        if cleaned != content:
            chunk["page_content"] = cleaned
            changed_chunks += 1

    report = {
        "source_chunks": len(chunks),
        "output_chunks": len(output),
        "changed_chunks": changed_chunks,
        "cleaned_table_blocks": table_blocks,
        "removed_markdown_images": removed_markdown_images,
        "flattened_html_tables": flattened_html_tables,
    }
    return output, report


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a table-clean copy of final Upstage chunks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = read_json(args.input)
    cleaned_chunks, report = build_table_clean_copy(chunks)
    tables_markdown, markdown_table_count = render_tables_markdown(cleaned_chunks)
    report["markdown_output"] = str(args.markdown_output)
    report["markdown_table_blocks"] = markdown_table_count

    write_json(args.output, cleaned_chunks)
    write_text(args.markdown_output, tables_markdown)
    write_json(args.report, report)

    print(f"[INFO] input: {args.input}")
    print(f"[INFO] output: {args.output}")
    print(f"[INFO] markdown output: {args.markdown_output}")
    print(f"[INFO] report: {args.report}")
    print(f"[INFO] changed chunks: {report['changed_chunks']}")
    print(f"[INFO] cleaned table blocks: {report['cleaned_table_blocks']}")
    print(f"[INFO] markdown table blocks: {report['markdown_table_blocks']}")


if __name__ == "__main__":
    main()
