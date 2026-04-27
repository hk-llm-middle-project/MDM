"""Upstage Document Parse loader strategy."""

from __future__ import annotations

import base64
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document


IMAGE_CATEGORIES = {"figure", "chart", "table"}
DEFAULT_PAGE_SIZE = 100
DEFAULT_UPSTAGE_OPTIONS = {
    "split": "element",
    "output_format": "markdown",
    "coordinates": True,
    "base64_encoding": ["figure", "chart", "table"],
    "ocr": "auto",
}


@dataclass(frozen=True)
class UpstageLoaderConfig:
    page_size: int = DEFAULT_PAGE_SIZE
    save_images: bool = True
    image_output_dir: Path | None = None
    upstage_options: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_UPSTAGE_OPTIONS)
    )


def split_pdf_for_upstage(path: Path, page_size: int) -> list[dict[str, int | Path]]:
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(path))
    split_files: list[dict[str, int | Path]] = []

    for start in range(0, len(reader.pages), page_size):
        end = min(start + page_size, len(reader.pages))
        writer = PdfWriter()

        for page_index in range(start, end):
            writer.add_page(reader.pages[page_index])

        temp_dir = Path(tempfile.mkdtemp(prefix="upstage_split_"))
        temp_path = temp_dir / f"pages_{start + 1}_{end}.pdf"
        with temp_path.open("wb") as fp:
            writer.write(fp)

        split_files.append({"path": temp_path, "page_offset": start})

    return split_files


def restore_page_metadata(docs: list[Document], source_path: Path, page_offset: int) -> None:
    for doc in docs:
        page = doc.metadata.get("page")
        if isinstance(page, int):
            doc.metadata["page"] = page + page_offset
        elif isinstance(page, str) and page.isdigit():
            doc.metadata["page"] = int(page) + page_offset
        doc.metadata["source"] = str(source_path)
        doc.metadata["parser"] = "upstage"


def create_upstage_loader(path: Path, config: UpstageLoaderConfig) -> Any:
    from langchain_upstage import UpstageDocumentParseLoader

    return UpstageDocumentParseLoader(file_path=str(path), **config.upstage_options)


def save_base64_images(docs: list[Document], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    counters: dict[tuple[int | str, str], int] = defaultdict(int)

    for doc in docs:
        metadata = doc.metadata
        category = str(metadata.get("category", "")).lower()
        encoded = metadata.get("base64_encoding")

        if category not in IMAGE_CATEGORIES or not encoded:
            continue

        page = metadata.get("page", "unknown")
        counters[(page, category)] += 1
        image_path = output_dir / f"page_{page}_{category}_{counters[(page, category)]}.png"
        image_path.write_bytes(base64.b64decode(encoded))

        metadata["image_path"] = str(image_path)
        metadata.pop("base64_encoding", None)
        saved_paths.append(image_path)

    return saved_paths


def load_with_upstage(
    path: Path,
    strategy_config: UpstageLoaderConfig | None = None,
) -> list[Document]:
    load_dotenv()

    config = strategy_config or UpstageLoaderConfig()
    documents: list[Document] = []

    for split_info in split_pdf_for_upstage(path, config.page_size):
        split_path = Path(split_info["path"])
        page_offset = int(split_info["page_offset"])

        loader = create_upstage_loader(split_path, config)
        split_docs = loader.load()
        restore_page_metadata(split_docs, path, page_offset)
        documents.extend(split_docs)

    if config.save_images:
        image_output_dir = config.image_output_dir or path.with_suffix("")
        save_base64_images(documents, image_output_dir)
    else:
        for doc in documents:
            doc.metadata.pop("base64_encoding", None)

    return documents
