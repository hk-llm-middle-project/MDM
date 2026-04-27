"""PDF loader strategy exports."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from rag.loader.strategies.llamaparser_loader import (
    LlamaParserLoaderConfig,
    load_with_llamaparser,
)
from rag.loader.strategies.pdfplumber_loader import (
    PdfPlumberLoaderConfig,
    load_with_pdfplumber,
)

if TYPE_CHECKING:
    from rag.loader.strategies.upstage.upstage_loader import UpstageLoaderConfig
else:
    from rag.loader.strategies.upstage.upstage_loader import UpstageLoaderConfig


def load_with_upstage(
    path: Path,
    strategy_config: UpstageLoaderConfig | None = None,
) -> list[Document]:
    """Load with Upstage only when the strategy is actually selected."""
    from rag.loader.strategies.upstage.upstage_loader import (
        load_with_upstage as _load_with_upstage,
    )

    return _load_with_upstage(path, strategy_config)


__all__ = [
    "PdfPlumberLoaderConfig",
    "LlamaParserLoaderConfig",
    "UpstageLoaderConfig",
    "load_with_pdfplumber",
    "load_with_llamaparser",
    "load_with_upstage",
]
