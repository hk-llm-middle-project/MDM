"""Google Gemini embedding provider."""

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import GOOGLE_EMBEDDING_MODEL


class GoogleGeminiEmbeddings(Embeddings):
    """LangChain embeddings adapter for Google Gemini embeddings."""

    def __init__(self, model: str = GOOGLE_EMBEDDING_MODEL):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings.embed_query(text)


def create_google_embeddings() -> GoogleGeminiEmbeddings:
    """Create a Google Gemini embedding provider."""
    return GoogleGeminiEmbeddings()
