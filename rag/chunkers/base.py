"""Base interface for chunker implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag.chunkers.schema import Chunk


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, parsed_input: Any) -> list[Chunk]:
        """Return standard chunks for parsed input."""
        pass
