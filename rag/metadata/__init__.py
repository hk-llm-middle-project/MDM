"""Document metadata enrichment helpers."""

from rag.metadata.classifier import (
    PageMetadataClassification,
    build_rule_based_page_metadata_cache,
    classify_page_metadata,
    enrich_documents_with_page_metadata,
    normalize_page_metadata_cache_entry,
    write_rule_based_page_metadata_cache,
)
from rag.metadata.generator import (
    ensure_page_metadata_cache,
    generate_main_pdf_page_metadata_cache,
)

__all__ = [
    "PageMetadataClassification",
    "build_rule_based_page_metadata_cache",
    "classify_page_metadata",
    "ensure_page_metadata_cache",
    "enrich_documents_with_page_metadata",
    "generate_main_pdf_page_metadata_cache",
    "normalize_page_metadata_cache_entry",
    "write_rule_based_page_metadata_cache",
]
