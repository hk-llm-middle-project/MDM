import unittest

from langchain_core.documents import Document

from rag.chunkers import SemanticChunker


class SemanticChunkerTest(unittest.TestCase):
    def test_semantic_chunker_continues_similarity_across_page_boundary(self):
        documents = [
            Document(
                page_content="교차로에서 차량 A가 직진했습니다. 차량 B도 같은 교차로에 진입했습니다.",
                metadata={"page": 10, "source": "010.md"},
            ),
            Document(
                page_content="두 차량은 신호 교차로에서 충돌했습니다. 주차장 관리 규정은 별도입니다.",
                metadata={"page": 11, "source": "011.md"},
            ),
        ]
        embeddings = [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.0, 1.0],
        ]

        chunker = SemanticChunker(
            embedding_function=lambda sentences: embeddings[: len(sentences)],
            breakpoint_threshold=0.2,
            min_chunk_chars=0,
        )

        chunks = chunker.chunk(documents)

        self.assertEqual([chunk.chunk_id for chunk in chunks], [0, 1])
        self.assertEqual([chunk.chunk_type for chunk in chunks], ["flat", "flat"])
        self.assertEqual(chunks[0].page, 10)
        self.assertEqual(chunks[0].source, "010.md")
        self.assertIn("교차로에서 차량 A가 직진했습니다.", chunks[0].text)
        self.assertIn("두 차량은 신호 교차로에서 충돌했습니다.", chunks[0].text)
        self.assertEqual(chunks[1].page, 11)
        self.assertEqual(chunks[1].source, "011.md")
        self.assertEqual(chunks[1].text, "주차장 관리 규정은 별도입니다.")

    def test_semantic_chunker_splits_on_large_distance_across_page_boundary(self):
        documents = [
            Document(
                page_content="차16-2 교차로 차량 사고입니다.",
                metadata={"page": "175", "source": "175.md"},
            ),
            Document(
                page_content="보험 약관의 일반 조항입니다.",
                metadata={"page": "176", "source": "176.md"},
            ),
        ]

        chunker = SemanticChunker(
            embedding_function=lambda sentences: [[1.0, 0.0], [0.0, 1.0]][: len(sentences)],
            breakpoint_threshold=0.2,
            min_chunk_chars=0,
        )

        chunks = chunker.chunk(documents)

        self.assertEqual([chunk.text for chunk in chunks], ["차16-2 교차로 차량 사고입니다.", "보험 약관의 일반 조항입니다."])
        self.assertEqual([chunk.page for chunk in chunks], [175, 176])
        self.assertEqual(chunks[0].diagram_id, "차16-2")

    def test_semantic_chunker_preserves_first_image_path_in_chunk(self):
        document = Document(
            page_content=(
                "보1 횡단보도 사고입니다.\n"
                "![page_39_table_1](../../upstage_output/main_pdf/final/img/page_39_table_1.png)"
            ),
            metadata={"page": 39, "source": "039.md"},
        )

        chunks = SemanticChunker(
            embedding_function=lambda sentences: [[1.0, 0.0] for _ in sentences],
            min_chunk_chars=0,
        ).chunk(document)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].diagram_id, "보1")
        self.assertEqual(
            chunks[0].image_path,
            "../../upstage_output/main_pdf/final/img/page_39_table_1.png",
        )


if __name__ == "__main__":
    unittest.main()
