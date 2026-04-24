import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag.chunker import chunk_text, split_documents
from rag.indexer import build_vectorstore, vectorstore_exists
from rag.retriever import EnsembleRetrieverConfig
from rag.retriever import RETRIEVAL_STRATEGIES, retrieve


class BasicRagTest(unittest.TestCase):
    def test_chunk_text_uses_fixed_size_without_overlap(self):
        chunks = chunk_text("a" * 1201, chunk_size=500, overlap=0)

        self.assertEqual([len(chunk) for chunk in chunks], [500, 500, 201])

    def test_split_documents_does_not_add_metadata(self):
        documents = [Document(page_content="abcdef", metadata={"page": 1})]

        chunks = split_documents(documents, chunk_size=3, overlap=0)

        self.assertEqual([chunk.page_content for chunk in chunks], ["abc", "def"])
        self.assertEqual([chunk.metadata for chunk in chunks], [{}, {}])

    def test_retrieve_uses_vectorstore_strategy_by_default(self):
        fake_retriever = MagicMock()
        fake_retriever.invoke.return_value = [Document(page_content="result")]
        vectorstore = MagicMock()
        vectorstore.as_retriever.return_value = fake_retriever

        results = retrieve(vectorstore, "query")

        self.assertEqual(results[0].page_content, "result")
        vectorstore.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 3},
        )
        fake_retriever.invoke.assert_called_once_with("query")

    def test_retrieve_routes_strategy_config_to_selected_strategy(self):
        called = {}
        config = EnsembleRetrieverConfig(weights=(0.7, 0.3))

        def fake_strategy(vectorstore, query, k, filters, strategy_config):
            called["args"] = (vectorstore, query, k, filters, strategy_config)
            return [Document(page_content="routed")]

        with patch.dict(RETRIEVAL_STRATEGIES, {"ensemble": fake_strategy}, clear=False):
            vectorstore = MagicMock()
            results = retrieve(
                vectorstore,
                "query",
                strategy="ensemble",
                filters={"page": 3},
                strategy_config=config,
            )

        self.assertEqual(results[0].page_content, "routed")
        self.assertEqual(
            called["args"],
            (vectorstore, "query", 3, {"page": 3}, config),
        )

    def test_retrieve_raises_for_unknown_strategy(self):
        with self.assertRaises(ValueError):
            retrieve(MagicMock(), "query", strategy="unknown")

    def test_selfquery_strategy_placeholder_raises(self):
        with self.assertRaises(NotImplementedError):
            retrieve(MagicMock(), "query", strategy="selfquery")

    def test_vectorstore_exists_ignores_placeholder_files(self):
        self.assertFalse(vectorstore_exists(Path("missing-vectorstore")))

    def test_build_vectorstore_adds_documents_in_batches(self):
        class FakeChroma:
            def __init__(self, persist_directory, embedding_function):
                self.persist_directory = persist_directory
                self.embedding_function = embedding_function
                self.batches = []

            def add_documents(self, documents):
                self.batches.append(list(documents))

        documents = [Document(page_content=str(index)) for index in range(5)]

        with (
            patch("pathlib.Path.mkdir"),
            patch("rag.indexer.Chroma", FakeChroma),
            patch("rag.indexer.create_embeddings", return_value="embeddings"),
        ):
            vectorstore = build_vectorstore(documents, Path("vectorstore"), batch_size=2)

        self.assertEqual([len(batch) for batch in vectorstore.batches], [2, 2, 1])


if __name__ == "__main__":
    unittest.main()
