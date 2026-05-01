"""Loader for precomputed Upstage document JSON artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.documents import Document

from config import (
    UPSTAGE_CUSTOM_DOCUMENTS_PATH,
    UPSTAGE_LEGACY_RAW_DOCUMENTS_PATH,
    UPSTAGE_RAW_DOCUMENTS_PATH,
)


UpstageVariant = Literal["raw", "custom"]


@dataclass(frozen=True)
class UpstageLoaderConfig:
    variant: UpstageVariant = "raw"
    raw_documents_path: Path = UPSTAGE_RAW_DOCUMENTS_PATH
    custom_documents_path: Path = UPSTAGE_CUSTOM_DOCUMENTS_PATH


def load_documents_json(json_path: str | Path) -> list[Document]:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    return [
        Document(
            page_content=item.get("page_content", ""),
            metadata=item.get("metadata", {}),
        )
        for item in payload
    ]


def normalize_cached_document_metadata(docs: list[Document], source_path: Path) -> None:
    for doc in docs:
        metadata = {
            key: value
            for key, value in doc.metadata.items()
            if value is not None
        }
        metadata["source"] = str(source_path)
        metadata["parser"] = "upstage"
        doc.metadata = metadata


def get_upstage_documents_path(config: UpstageLoaderConfig) -> Path:
    if config.variant == "raw":
        if config.raw_documents_path.is_file():
            return config.raw_documents_path
        if config.raw_documents_path == UPSTAGE_RAW_DOCUMENTS_PATH and UPSTAGE_LEGACY_RAW_DOCUMENTS_PATH.is_file():
            return UPSTAGE_LEGACY_RAW_DOCUMENTS_PATH
        return config.raw_documents_path
    return config.custom_documents_path


def load_with_upstage(
    path: Path,
    strategy_config: UpstageLoaderConfig | None = None,
) -> list[Document]:
    config = strategy_config or UpstageLoaderConfig()
    documents_path = get_upstage_documents_path(config)
    if not documents_path.is_file():
        raise FileNotFoundError(
            f"Upstage source JSON not found for variant '{config.variant}': {documents_path}"
        )

    documents = load_documents_json(documents_path)
    normalize_cached_document_metadata(documents, path)
    return documents
