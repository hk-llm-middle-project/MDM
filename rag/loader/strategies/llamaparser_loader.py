"""LlamaParse PDF 로더 전략."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from config import LLAMA_MD_DIR
from rag.loader.strategies.common import build_page_document

LlamaParse = None

DEFAULT_LLAMAPARSE_OPTIONS = {
    "split_by_page": True,
    "language": "ko",
    "adaptive_long_table": True,
    "disable_ocr": True,
    "disable_image_extraction": False,
    "do_not_unroll_columns": True,
    "extract_layout": True,
    "auto_mode": True,
    "auto_mode_trigger_on_table_in_page": True,
    "auto_mode_trigger_on_image_in_page": True,
}


@dataclass(frozen=True)
class LlamaParserLoaderConfig:
    result_type: str = "markdown"
    output_dir: Path = LLAMA_MD_DIR
    llamaparse_options: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_LLAMAPARSE_OPTIONS)
    )


def select_page_content(page: object, result_type: str) -> str:
    markdown = getattr(page, "md", None) or ""
    text = getattr(page, "text", None) or ""
    if result_type == "markdown":
        return markdown or text
    if result_type == "text":
        return text or markdown
    raise ValueError("result_type must be either 'markdown' or 'text'.")


def build_llamaparse_options(config: LlamaParserLoaderConfig) -> dict[str, Any]:
    return {
        **config.llamaparse_options,
        "result_type": config.result_type,
    }


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
                    parser_name="llamaparser",
                )
            )
    return documents


def create_llamaparse_parser(config: LlamaParserLoaderConfig) -> object:
    global LlamaParse
    if LlamaParse is None:
        from llama_parse import LlamaParse as ImportedLlamaParse

        LlamaParse = ImportedLlamaParse
    return LlamaParse(**build_llamaparse_options(config))


def load_with_llamaparser(
    path: Path,
    strategy_config: LlamaParserLoaderConfig | None = None,
) -> list[Document]:
    config = strategy_config or LlamaParserLoaderConfig()
    saved_documents = load_saved_markdown_documents(path, config.output_dir)
    if saved_documents:
        return saved_documents

    parser = create_llamaparse_parser(config)
    job_result = parser.parse(str(path))

    documents: list[Document] = []
    for page in job_result.pages:
        page_content = select_page_content(page, config.result_type)
        if page_content.strip():
            save_page_markdown(config.output_dir, page.page, page_content)
            documents.append(
                build_page_document(
                    path=path,
                    page_content=page_content,
                    page_number=page.page,
                    parser_name="llamaparser",
                )
            )
    return documents
