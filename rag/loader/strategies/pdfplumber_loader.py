"""pdfplumber PDF 로더 전략."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber
from langchain_core.documents import Document

from config import PDFPLUMBER_OUT_DIR
from rag.loader.strategies.common import build_page_document


DEFAULT_WORD_SETTINGS = {
    "x_tolerance": 3,
    "y_tolerance": 3,
    "keep_blank_chars": False,
    "use_text_flow": True,
}
DEFAULT_TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 5,
    "join_tolerance": 5,
    "intersection_tolerance": 5,
    "text_tolerance": 3,
}
DEFAULT_CROP_MARGINS = (0, 50, 0, 40)


@dataclass(frozen=True)
class PdfPlumberLoaderConfig:
    """pdfplumber 로더 설정."""

    output_dir: Path = PDFPLUMBER_OUT_DIR
    crop_margins: tuple[float, float, float, float] | None = DEFAULT_CROP_MARGINS
    word_settings: dict[str, Any] | None = None
    table_settings: dict[str, Any] | None = None


def get_document_cache_dir(path: Path, output_dir: Path) -> Path:
    return output_dir / "main_pdf"


def save_page_markdown(output_dir: Path, page_number: int, page_content: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{page_number:03d}.md"
    output_path.write_text(page_content.strip() + "\n", encoding="utf-8")
    return output_path


def load_saved_markdown_documents(path: Path, output_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for markdown_path in sorted(output_dir.glob("*.md")):
        try:
            page_number = int(markdown_path.stem)
        except ValueError:
            continue

        page_content = markdown_path.read_text(encoding="utf-8")
        if page_content.strip():
            documents.append(
                build_page_document(
                    path=path,
                    page_content=page_content,
                    page_number=page_number,
                    parser_name="pdfplumber",
                )
            )
    return documents


def extract_page_markdown(
    page: object,
    config: PdfPlumberLoaderConfig | None = None,
) -> str:
    config = config or PdfPlumberLoaderConfig()
    content_page = _crop_page(page, config.crop_margins)
    word_settings = config.word_settings or DEFAULT_WORD_SETTINGS
    table_settings = config.table_settings or DEFAULT_TABLE_SETTINGS

    tables = _find_tables(content_page, table_settings)
    if not tables:
        extract_text = getattr(content_page, "extract_text", None)
        return extract_text() if callable(extract_text) else ""

    words = _extract_words(content_page, word_settings)
    if not words:
        extract_text = getattr(content_page, "extract_text", None)
        text = extract_text() if callable(extract_text) else ""
        table_text = "\n\n".join(_table_to_markdown(table) for table in tables)
        return "\n\n".join(part for part in [text.strip(), table_text] if part).strip()

    sorted_tables = sorted(tables, key=lambda table: _table_top(table))
    non_table_words = [
        word for word in words if not _word_is_inside_any_table(word, sorted_tables)
    ]

    parts: list[str] = []
    previous_bottom = float("-inf")
    for table in sorted_tables:
        table_top = _table_top(table)
        text_block = _words_to_text(
            word
            for word in non_table_words
            if previous_bottom < _word_top(word) < table_top
        )
        if text_block:
            parts.append(text_block)
        table_markdown = _table_to_markdown(table)
        if table_markdown:
            parts.append(table_markdown)
        previous_bottom = _table_bottom(table)

    trailing_text = _words_to_text(
        word for word in non_table_words if _word_top(word) > previous_bottom
    )
    if trailing_text:
        parts.append(trailing_text)

    return "\n\n".join(parts).strip()


def _crop_page(
    page: object,
    crop_margins: tuple[float, float, float, float] | None,
) -> object:
    if crop_margins is None:
        return page
    crop = getattr(page, "crop", None)
    width = getattr(page, "width", None)
    height = getattr(page, "height", None)
    if not callable(crop) or not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        return page
    left, top, right, bottom = crop_margins
    bbox = (left, top, width - right, height - bottom)
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return page
    return crop(bbox)


def _find_tables(page: object, table_settings: dict[str, Any]) -> list[Any]:
    find_tables = getattr(page, "find_tables", None)
    if not callable(find_tables):
        return []
    try:
        tables = find_tables(table_settings=table_settings)
    except Exception:
        return []
    return tables if isinstance(tables, list) else []


def _extract_words(page: object, word_settings: dict[str, Any]) -> list[dict]:
    extract_words = getattr(page, "extract_words", None)
    if not callable(extract_words):
        return []
    try:
        words = extract_words(**word_settings)
    except Exception:
        return []
    return words if isinstance(words, list) else []


def _table_to_markdown(table: Any) -> str:
    extract = getattr(table, "extract", None)
    if not callable(extract):
        return ""
    rows = extract() or []
    if not rows:
        return ""

    max_cols = max(len(row or []) for row in rows)
    normalized_rows = [_normalize_table_row(row, max_cols) for row in rows]
    header = normalized_rows[0]
    markdown_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * max_cols) + " |",
    ]
    for row in normalized_rows[1:]:
        markdown_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(markdown_lines)


def _normalize_table_row(row: list[Any] | None, max_cols: int) -> list[str]:
    cells = [_normalize_table_cell(cell) for cell in (row or [])]
    cells.extend([""] * (max_cols - len(cells)))
    return cells


def _normalize_table_cell(cell: Any) -> str:
    return str(cell or "").replace("\n", "<br/>").replace("|", "\\|").strip()


def _word_is_inside_any_table(word: dict, tables: list[Any]) -> bool:
    return any(_word_is_inside_table(word, table) for table in tables)


def _word_is_inside_table(word: dict, table: Any) -> bool:
    x0, top, x1, bottom = _table_bbox(table)
    word_x0 = float(word.get("x0", 0))
    word_x1 = float(word.get("x1", word_x0))
    word_top = _word_top(word)
    word_bottom = float(word.get("bottom", word_top))
    word_x_center = (word_x0 + word_x1) / 2
    word_y_center = (word_top + word_bottom) / 2
    return x0 <= word_x_center <= x1 and top <= word_y_center <= bottom


def _words_to_text(words: Any) -> str:
    sorted_words = sorted(words, key=lambda word: (_word_top(word), float(word.get("x0", 0))))
    if not sorted_words:
        return ""

    lines: list[list[dict]] = []
    current_line: list[dict] = []
    current_top: float | None = None
    for word in sorted_words:
        top = _word_top(word)
        if current_top is None or abs(top - current_top) <= 3:
            current_line.append(word)
            current_top = top if current_top is None else current_top
            continue
        lines.append(current_line)
        current_line = [word]
        current_top = top
    if current_line:
        lines.append(current_line)

    return "\n".join(
        " ".join(str(word.get("text", "")).strip() for word in line if word.get("text"))
        for line in lines
    ).strip()


def _table_top(table: Any) -> float:
    return _table_bbox(table)[1]


def _table_bottom(table: Any) -> float:
    return _table_bbox(table)[3]


def _table_bbox(table: Any) -> tuple[float, float, float, float]:
    bbox = getattr(table, "bbox", (0, 0, 0, 0))
    return tuple(float(value) for value in bbox)


def _word_top(word: dict) -> float:
    return float(word.get("top", 0))


def load_with_pdfplumber(
    path: Path,
    strategy_config: PdfPlumberLoaderConfig | None = None,
) -> list[Document]:
    config = strategy_config or PdfPlumberLoaderConfig()
    document_cache_dir = get_document_cache_dir(path, config.output_dir)
    saved_documents = load_saved_markdown_documents(path, document_cache_dir)
    if saved_documents:
        return saved_documents

    documents: list[Document] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = extract_page_markdown(page, config)
            if text.strip():
                save_page_markdown(document_cache_dir, page.page_number, text)
                documents.append(
                    build_page_document(
                        path=path,
                        page_content=text,
                        page_number=page.page_number,
                        parser_name="pdfplumber",
                    )
                )
    return documents
