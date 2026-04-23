import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from rag.chunker import chunk_text, split_documents
from rag.indexer import build_vectorstore, vectorstore_exists
from rag.retriever import retrieve


class BasicRagTest(unittest.TestCase):
    def test_chunk_text_uses_fixed_size_without_overlap(self):
        chunks = chunk_text("a" * 1201, chunk_size=500, overlap=0)

        self.assertEqual([len(chunk) for chunk in chunks], [500, 500, 201])

    def test_split_documents_does_not_add_metadata(self):
        documents = [Document(page_content="abcdef", metadata={"page": 1})]

        chunks = split_documents(documents, chunk_size=3, overlap=0)

        self.assertEqual([chunk.page_content for chunk in chunks], ["abc", "def"])
        self.assertEqual([chunk.metadata for chunk in chunks], [{}, {}])

    def test_retrieve_uses_similarity_search_with_default_k(self):
        class FakeVectorStore:
            def __init__(self):
                self.calls = []

            def similarity_search(self, query, k):
                self.calls.append((query, k))
                return [Document(page_content="result")]

        vectorstore = FakeVectorStore()

        results = retrieve(vectorstore, "질문")

        self.assertEqual(results[0].page_content, "result")
        self.assertEqual(vectorstore.calls, [("질문", 3)])

    def test_vectorstore_exists_ignores_placeholder_files(self):
        with TemporaryDirectory() as temp_dir:
            persist_directory = Path(temp_dir)
            (persist_directory / ".gitkeep").touch()

            self.assertFalse(vectorstore_exists(persist_directory))

    def test_build_vectorstore_adds_documents_in_batches(self):
        class FakeChroma:
            def __init__(self, persist_directory, embedding_function):
                self.persist_directory = persist_directory
                self.embedding_function = embedding_function
                self.batches = []

            def add_documents(self, documents):
                self.batches.append(list(documents))

        documents = [Document(page_content=str(index)) for index in range(5)]

        with TemporaryDirectory() as temp_dir:
            with (
                patch("rag.indexer.Chroma", FakeChroma),
                patch("rag.indexer.create_embeddings", return_value="embeddings"),
            ):
                vectorstore = build_vectorstore(documents, Path(temp_dir), batch_size=2)

        self.assertEqual([len(batch) for batch in vectorstore.batches], [2, 2, 1])


if __name__ == "__main__":
    unittest.main()
