"""캐시된 벡터스토어와 검색 컴포넌트 서비스입니다."""

from functools import lru_cache

from config import (
    DEFAULT_CHUNKER_STRATEGY,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LOADER_STRATEGY,
    PAGE_METADATA_PATH,
    PDF_PATH,
    get_chunk_cache_dir,
    get_vectorstore_dir,
)
from rag.chunkers import (
    CaseBoundaryChunker,
    FixedSizeChunker,
    MarkdownStructureChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    chunk_to_document,
)
from rag.embeddings import create_embeddings
from rag.indexer import build_vectorstore, load_vectorstore, vectorstore_exists
from rag.loader import load_pdf
from rag.metadata import ensure_page_metadata_cache, enrich_documents_with_page_metadata
from rag.service.chunk_cache import chunk_cache_exists, load_chunk_cache, save_chunk_cache
from rag.pipeline.retriever import RetrievalComponents, build_retrieval_components
from scripts.clean_case_boundary_chunk_tables import clean_case_boundary_tables


PRE_CHUNKED_LOADER_STRATEGIES = {"upstage"}
CASE_BOUNDARY_CHUNKER_STRATEGY = "case-boundary"
CASE_BOUNDARY_LOADER_STRATEGY = "llamaparser"
SEMANTIC_CHUNKER_STRATEGY = "semantic"
RECURSIVE_CHUNKER_STRATEGY = "recursive"
MARKDOWN_CHUNKER_STRATEGY = "markdown"
UPSTAGE_RAW_CHUNKER_STRATEGY = "raw"
UPSTAGE_CUSTOM_CHUNKER_STRATEGY = "custom"
UPSTAGE_LEGACY_CHUNKER_ALIASES = {DEFAULT_CHUNKER_STRATEGY, "native"}


def get_page_metadata_cache_path(loader_strategy: str):
    """Return the cache path for page-level LLM metadata labels."""
    del loader_strategy
    return PAGE_METADATA_PATH


def _should_enrich_page_metadata(loader_strategy: str) -> bool:
    return loader_strategy != "upstage"


def _document_to_dict(document):
    return {
        "page_content": document.page_content,
        "metadata": dict(document.metadata),
    }


def _dict_to_document(doc):
    from langchain_core.documents import Document

    return Document(
        page_content=str(doc.get("page_content", "")),
        metadata=dict(doc.get("metadata") or {}),
    )


def _case_boundary_documents(documents):
    chunks = CaseBoundaryChunker(mode="B").chunk(documents)
    chunk_documents = [chunk_to_document(chunk) for chunk in chunks]
    cleaned_docs = clean_case_boundary_tables(
        [_document_to_dict(document) for document in chunk_documents]
    )
    return [_dict_to_document(doc) for doc in cleaned_docs]


def _semantic_documents(documents, *, embedding_provider: str):
    embeddings = create_embeddings(embedding_provider)
    chunks = SemanticChunker(embedding_function=embeddings).chunk(documents)
    return [chunk_to_document(chunk) for chunk in chunks]


def _standard_chunker_documents(documents, *, chunker_strategy: str):
    if chunker_strategy == DEFAULT_CHUNKER_STRATEGY:
        chunks = FixedSizeChunker().chunk(documents)
    elif chunker_strategy == RECURSIVE_CHUNKER_STRATEGY:
        chunks = RecursiveCharacterChunker().chunk(documents)
    elif chunker_strategy == MARKDOWN_CHUNKER_STRATEGY:
        chunks = MarkdownStructureChunker().chunk(documents)
    else:
        raise ValueError(f"Unsupported standard chunker strategy: {chunker_strategy}")
    return [chunk_to_document(chunk) for chunk in chunks]


def _chunk_documents_for_vectorstore(
    documents,
    *,
    loader_strategy: str,
    chunker_strategy: str,
    embedding_provider: str,
):
    if loader_strategy in PRE_CHUNKED_LOADER_STRATEGIES:
        chunks = documents
    elif chunker_strategy == CASE_BOUNDARY_CHUNKER_STRATEGY:
        chunks = _case_boundary_documents(documents)
    elif chunker_strategy == SEMANTIC_CHUNKER_STRATEGY:
        chunks = _semantic_documents(documents, embedding_provider=embedding_provider)
    elif chunker_strategy in {
        DEFAULT_CHUNKER_STRATEGY,
        RECURSIVE_CHUNKER_STRATEGY,
        MARKDOWN_CHUNKER_STRATEGY,
    }:
        chunks = _standard_chunker_documents(documents, chunker_strategy=chunker_strategy)
    else:
        raise ValueError(f"Unsupported chunker strategy: {chunker_strategy}")
    if not _should_enrich_page_metadata(loader_strategy):
        return chunks
    return enrich_documents_with_page_metadata(
        chunks,
        cache_path=get_page_metadata_cache_path(loader_strategy),
    )


def _get_chunk_cache_dir(loader_strategy: str, chunker_strategy: str):
    return get_chunk_cache_dir(loader_strategy, chunker_strategy)


def _validate_loader_chunker_combination(
    loader_strategy: str,
    chunker_strategy: str,
) -> None:
    if (
        chunker_strategy == CASE_BOUNDARY_CHUNKER_STRATEGY
        and loader_strategy != CASE_BOUNDARY_LOADER_STRATEGY
    ):
        raise ValueError("case-boundary chunker requires llamaparser loader")
    if chunker_strategy == MARKDOWN_CHUNKER_STRATEGY and loader_strategy != "llamaparser":
        raise ValueError("markdown chunker requires llamaparser loader")
    if loader_strategy == "upstage" and chunker_strategy not in {
        UPSTAGE_RAW_CHUNKER_STRATEGY,
        UPSTAGE_CUSTOM_CHUNKER_STRATEGY,
        *UPSTAGE_LEGACY_CHUNKER_ALIASES,
    }:
        raise ValueError("upstage loader requires raw or custom chunker")


def _normalize_chunker_strategy_for_loader(
    loader_strategy: str,
    chunker_strategy: str,
) -> str:
    if loader_strategy == "upstage" and chunker_strategy in UPSTAGE_LEGACY_CHUNKER_ALIASES:
        return UPSTAGE_RAW_CHUNKER_STRATEGY
    return chunker_strategy


@lru_cache(maxsize=32)
def get_vectorstore(
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    chunker_strategy: str = DEFAULT_CHUNKER_STRATEGY,
):
    """기존 벡터스토어를 불러오거나 PDF에서 새로 생성합니다."""
    chunker_strategy = _normalize_chunker_strategy_for_loader(
        loader_strategy,
        chunker_strategy,
    )
    _validate_loader_chunker_combination(loader_strategy, chunker_strategy)
    vectorstore_dir = (
        get_vectorstore_dir(loader_strategy, embedding_provider)
        if chunker_strategy == DEFAULT_CHUNKER_STRATEGY
        else get_vectorstore_dir(
            loader_strategy,
            embedding_provider,
            chunker_strategy=chunker_strategy,
        )
    )
    if vectorstore_exists(vectorstore_dir):
        return load_vectorstore(vectorstore_dir, embedding_provider=embedding_provider)

    chunk_cache_dir = _get_chunk_cache_dir(loader_strategy, chunker_strategy)
    if chunk_cache_exists(chunk_cache_dir):
        chunks = load_chunk_cache(chunk_cache_dir, source_path=PDF_PATH)
        return build_vectorstore(
            chunks,
            vectorstore_dir,
            embedding_provider=embedding_provider,
        )

    if loader_strategy == "upstage":
        raise FileNotFoundError(
            f"Upstage chunk cache is required but missing: {chunk_cache_dir / 'chunks.json'}"
        )

    documents = load_pdf(PDF_PATH, strategy=loader_strategy)
    if _should_enrich_page_metadata(loader_strategy):
        ensure_page_metadata_cache(
            documents,
            get_page_metadata_cache_path(loader_strategy),
        )
    chunks = _chunk_documents_for_vectorstore(
        documents,
        loader_strategy=loader_strategy,
        chunker_strategy=chunker_strategy,
        embedding_provider=embedding_provider,
    )
    save_chunk_cache(chunks, chunk_cache_dir, source_path=PDF_PATH)
    chunks = load_chunk_cache(chunk_cache_dir, source_path=PDF_PATH)
    return build_vectorstore(
        chunks,
        vectorstore_dir,
        embedding_provider=embedding_provider,
    )


@lru_cache(maxsize=32)
def get_retrieval_components(
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    chunker_strategy: str = DEFAULT_CHUNKER_STRATEGY,
) -> RetrievalComponents:
    """앱 프로세스에서 재사용할 검색 컴포넌트를 준비합니다."""
    return build_retrieval_components(
        get_vectorstore(loader_strategy, embedding_provider, chunker_strategy)
    )
