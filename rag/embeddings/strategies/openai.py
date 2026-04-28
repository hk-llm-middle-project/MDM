"""OpenAI embedding provider."""

from langchain_openai import OpenAIEmbeddings

from config import OPENAI_EMBEDDING_MODEL


def create_openai_embeddings() -> OpenAIEmbeddings:
    """Create an OpenAI embedding provider."""
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
