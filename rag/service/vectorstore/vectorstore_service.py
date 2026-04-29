"""캐시된 벡터스토어와 검색 컴포넌트 서비스입니다."""

from functools import lru_cache

from config import (
    DEFAULT_CHUNKER_STRATEGY,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LOADER_STRATEGY,
    PAGE_METADATA_PATH,
    PDF_PATH,
    get_vectorstore_dir,
)
from rag.chunker import split_documents
from rag.chunkers import CaseBoundaryChunker, SemanticChunker, chunk_to_document
from rag.embeddings import create_embeddings
from rag.indexer import build_vectorstore, load_vectorstore, vectorstore_exists
from rag.loader import load_pdf
from rag.metadata import enrich_documents_with_llm_metadata
from rag.pipeline.retriever import RetrievalComponents, build_retrieval_components
from scripts.clean_case_boundary_chunk_tables import clean_case_boundary_tables


PRE_CHUNKED_LOADER_STRATEGIES = {"upstage"}
CASE_BOUNDARY_CHUNKER_STRATEGY = "case-boundary"
CASE_BOUNDARY_LOADER_STRATEGY = "llamaparser"
SEMANTIC_CHUNKER_STRATEGY = "semantic"


def get_page_metadata_cache_path(loader_strategy: str):
    """Return the cache path for page-level LLM metadata labels."""
    del loader_strategy
    return PAGE_METADATA_PATH


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


def _chunk_documents_for_vectorstore(
    documents,
    *,
    loader_strategy: str,
    chunker_strategy: str,
    embedding_provider: str,
):
    if loader_strategy in PRE_CHUNKED_LOADER_STRATEGIES:
        return documents
    if chunker_strategy == CASE_BOUNDARY_CHUNKER_STRATEGY:
        return _case_boundary_documents(documents)
    if chunker_strategy == SEMANTIC_CHUNKER_STRATEGY:
        return _semantic_documents(documents, embedding_provider=embedding_provider)
    return split_documents(
        enrich_documents_with_llm_metadata(
            documents,
            cache_path=get_page_metadata_cache_path(loader_strategy),
        )
    )


def _validate_loader_chunker_combination(
    loader_strategy: str,
    chunker_strategy: str,
) -> None:
    if (
        chunker_strategy == CASE_BOUNDARY_CHUNKER_STRATEGY
        and loader_strategy != CASE_BOUNDARY_LOADER_STRATEGY
    ):
        raise ValueError("case-boundary chunker requires llamaparser loader")


@lru_cache(maxsize=32)
def get_vectorstore(
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    chunker_strategy: str = DEFAULT_CHUNKER_STRATEGY,
):
    """기존 벡터스토어를 불러오거나 PDF에서 새로 생성합니다."""
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

    documents = load_pdf(PDF_PATH, strategy=loader_strategy)
    chunks = _chunk_documents_for_vectorstore(
        documents,
        loader_strategy=loader_strategy,
        chunker_strategy=chunker_strategy,
        embedding_provider=embedding_provider,
    )
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
