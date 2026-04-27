"""OpenAI embedding provider."""

from langchain_openai import OpenAIEmbeddings

from config import EMBEDDING_MODEL


def create_openai_embeddings() -> OpenAIEmbeddings:
    """Create an OpenAI embedding provider."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)
