"""Persistent query embedding cache helpers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from langchain_core.embeddings import Embeddings

from config import (
    EMBEDDING_QUERY_CACHE_SCHEMA_VERSION,
    get_embedding_query_cache_dir,
)


class CachedQueryEmbeddings(Embeddings):
    """Cache only query embeddings while leaving document embeddings uncached."""

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        provider: str,
        model_id: str,
        cache_dir: Path | None = None,
        enabled: bool = True,
    ):
        self.embeddings = embeddings
        self.provider = provider
        self.model_id = model_id
        self.cache_dir = cache_dir or get_embedding_query_cache_dir()
        self.enabled = enabled
        self.last_query_cache_hit: bool | None = None
        self.query_cache_hits = 0
        self.query_cache_misses = 0
        self._initialized = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Do not cache document embeddings; Chroma persistence owns that layer."""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if not self.enabled:
            self.last_query_cache_hit = None
            return self.embeddings.embed_query(text)

        cache_key = self._cache_key(text)
        cached = self._read(cache_key)
        if cached is not None:
            self.last_query_cache_hit = True
            self.query_cache_hits += 1
            return cached

        self.last_query_cache_hit = False
        self.query_cache_misses += 1
        vector = self.embeddings.embed_query(text)
        self._write(cache_key, text, vector)
        return vector

    def _cache_key(self, text: str) -> str:
        payload = json.dumps(
            {
                "schema": EMBEDDING_QUERY_CACHE_SCHEMA_VERSION,
                "provider": self.provider,
                "model_id": self.model_id,
                "text": text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def _cache_path(self) -> Path:
        return self.cache_dir / "query_embeddings.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._cache_path)

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS query_embeddings (
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed_at REAL NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        self._initialized = True

    def _read(self, cache_key: str) -> list[float] | None:
        try:
            self._ensure_initialized()
            with closing(self._connect()) as connection:
                with connection:
                    row = connection.execute(
                        "SELECT vector_json FROM query_embeddings WHERE cache_key = ?",
                        (cache_key,),
                    ).fetchone()
                    if row is None:
                        return None
                    connection.execute(
                        """
                        UPDATE query_embeddings
                        SET last_accessed_at = ?, hit_count = hit_count + 1
                        WHERE cache_key = ?
                        """,
                        (time.time(), cache_key),
                    )
        except sqlite3.Error:
            return None

        try:
            vector = json.loads(row[0])
        except json.JSONDecodeError:
            return None
        if not isinstance(vector, list):
            return None
        return vector

    def _write(self, cache_key: str, text: str, vector: list[float]) -> None:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        now = time.time()
        try:
            self._ensure_initialized()
            with closing(self._connect()) as connection:
                with connection:
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO query_embeddings (
                            cache_key,
                            provider,
                            model_id,
                            text_hash,
                            vector_json,
                            created_at,
                            last_accessed_at,
                            hit_count
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                        """,
                        (
                            cache_key,
                            self.provider,
                            self.model_id,
                            text_hash,
                            json.dumps(vector),
                            now,
                            now,
                        ),
                    )
        except sqlite3.Error:
            return
