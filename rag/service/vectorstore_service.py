"""캐시된 벡터스토어와 검색 컴포넌트 서비스입니다."""

from functools import lru_cache

from config import PDF_PATH, VECTORSTORE_DIR
from rag.chunker import split_documents
from rag.indexer import build_vectorstore, load_vectorstore, vectorstore_exists
from rag.loader import load_pdf
from rag.pipeline.retriever import RetrievalComponents, build_retrieval_components


@lru_cache(maxsize=1)
def get_vectorstore():
    """기존 벡터스토어를 불러오거나 PDF에서 새로 생성합니다."""
    if vectorstore_exists(VECTORSTORE_DIR):
        return load_vectorstore(VECTORSTORE_DIR)

    documents = load_pdf(PDF_PATH)
    chunks = split_documents(documents)
    return build_vectorstore(chunks, VECTORSTORE_DIR)


@lru_cache(maxsize=1)
def get_retrieval_components() -> RetrievalComponents:
    """앱 프로세스에서 재사용할 검색 컴포넌트를 준비합니다."""
    return build_retrieval_components(get_vectorstore())
