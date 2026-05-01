"""Rule-based page metadata cache generation helpers."""

from __future__ import annotations

from pathlib import Path

from config import DEFAULT_LOADER_STRATEGY, PAGE_METADATA_PATH, PDF_PATH
from rag.loader import load_pdf
from rag.metadata.classifier import (
    load_page_metadata_cache,
    write_rule_based_page_metadata_cache,
)


def ensure_page_metadata_cache(
    documents,
    cache_path: Path = PAGE_METADATA_PATH,
) -> dict[str, dict[str, object]]:
    """Create the page metadata cache when it is missing or empty."""
    if cache_path.exists():
        cache = load_page_metadata_cache(cache_path)
        if cache:
            return cache
    return write_rule_based_page_metadata_cache(documents, cache_path)


def generate_main_pdf_page_metadata_cache(
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    cache_path: Path = PAGE_METADATA_PATH,
) -> dict[str, dict[str, object]]:
    """Generate the page metadata cache for the main PDF."""
    documents = load_pdf(PDF_PATH, strategy=loader_strategy)
    return write_rule_based_page_metadata_cache(documents, cache_path)
