"""PDF 로더 전략 라우팅 유틸리티."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from config import DEFAULT_LOADER_STRATEGY, PDF_PATH
from rag.loader.strategies import (
    LlamaParserLoaderConfig,
    PdfPlumberLoaderConfig,
    load_with_llamaparser,
    load_with_pdfplumber,
)


LOADER_STRATEGIES = {
    "pdfplumber": load_with_pdfplumber,
    "llamaparser": load_with_llamaparser,
    "llama-parse": load_with_llamaparser,
}

LoaderConfig = PdfPlumberLoaderConfig | LlamaParserLoaderConfig


def load_pdf(
    path: Path = PDF_PATH,
    strategy: str = DEFAULT_LOADER_STRATEGY,
    strategy_config: LoaderConfig | None = None,
) -> list[Document]:
    try:
        loader_strategy = LOADER_STRATEGIES[strategy]
    except KeyError as error:
        available = ", ".join(sorted(LOADER_STRATEGIES))
        raise ValueError(
            f"Unknown PDF loader strategy: {strategy}. Available strategies: {available}"
        ) from error

    return loader_strategy(path, strategy_config)
