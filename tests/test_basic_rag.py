import unittest

from langchain_core.documents import Document

from rag.chunker import chunk_text, split_documents
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


if __name__ == "__main__":
    unittest.main()
