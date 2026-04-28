"""캐시된 벡터스토어와 검색 컴포넌트 서비스입니다."""

from functools import lru_cache

from config import (
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LOADER_STRATEGY,
    PAGE_METADATA_PATH,
    PDF_PATH,
    get_vectorstore_dir,
)
from rag.chunker import split_documents
from rag.indexer import build_vectorstore, load_vectorstore, vectorstore_exists
from rag.loader import load_pdf
from rag.metadata import enrich_documents_with_llm_metadata
from rag.pipeline.retriever import RetrievalComponents, build_retrieval_components


PRE_CHUNKED_LOADER_STRATEGIES = {"upstage"}


def get_page_metadata_cache_path(loader_strategy: str):
    """Return the cache path for page-level LLM metadata labels."""
    del loader_strategy
    return PAGE_METADATA_PATH


@lru_cache(maxsize=32)
def get_vectorstore(
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
):
    """기존 벡터스토어를 불러오거나 PDF에서 새로 생성합니다."""
    vectorstore_dir = get_vectorstore_dir(loader_strategy, embedding_provider)
    if vectorstore_exists(vectorstore_dir):
        return load_vectorstore(vectorstore_dir, embedding_provider=embedding_provider)

    documents = load_pdf(PDF_PATH, strategy=loader_strategy)
    chunks = (
        documents
        if loader_strategy in PRE_CHUNKED_LOADER_STRATEGIES
        else split_documents(
            enrich_documents_with_llm_metadata(
                documents,
                cache_path=get_page_metadata_cache_path(loader_strategy),
            )
        )
    )
    return build_vectorstore(
        chunks,
        vectorstore_dir,
        embedding_provider=embedding_provider,
    )


@lru_cache(maxsize=None)
def get_retrieval_components(
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
) -> RetrievalComponents:
    """앱 프로세스에서 재사용할 검색 컴포넌트를 준비합니다."""
    return build_retrieval_components(get_vectorstore(loader_strategy, embedding_provider))
