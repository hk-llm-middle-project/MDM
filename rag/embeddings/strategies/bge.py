"""BGE embedding provider."""

import os

import requests
from langchain_core.embeddings import Embeddings


BGE_EMBEDDING_BATCH_SIZE = 16


class BGEM3Embeddings(Embeddings):
    """LangChain embeddings adapter for the BGE-M3 embedding API."""

    def __init__(self):
        self.url = os.environ.get("BGE_BASE_URL")
        self.key = os.environ.get("BGE_API_KEY")
        if not self.url or not self.key:
            raise ValueError("BGE_BASE_URL and BGE_API_KEY environment variables must be set.")
        self.session = requests.Session()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for start in range(0, len(texts), BGE_EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + BGE_EMBEDDING_BATCH_SIZE]
            response = requests.post(
                f"{self.url}/v1/embeddings/m3",
                headers={"Authorization": f"Bearer {self.key}"},
                json={"input": batch, "return_dense": True},
                timeout=120,
            )
            response.raise_for_status()
            results.extend(item["dense"] for item in response.json()["data"])
        return results

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def create_bge_embeddings() -> BGEM3Embeddings:
    """Create a BGE-M3 embedding provider."""
    return BGEM3Embeddings()
