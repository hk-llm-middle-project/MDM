import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from config import PDF_PATH


class ChunkCacheTest(unittest.TestCase):
    def test_save_chunk_cache_writes_chunks_json_and_preview(self):
        from rag.service.chunk_cache import save_chunk_cache

        documents = [
            Document(
                page_content="첫 번째 청크 내용",
                metadata={"page": 38, "chunk_type": "child", "diagram_id": "보1"},
            ),
            Document(
                page_content="두 번째 청크 내용",
                metadata={"page": 39, "chunk_type": "child", "diagram_id": "보2"},
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            save_chunk_cache(documents, cache_dir)

            chunks_payload = json.loads((cache_dir / "chunks.json").read_text(encoding="utf-8"))
            preview = (cache_dir / "preview.md").read_text(encoding="utf-8")

        self.assertEqual(len(chunks_payload), 2)
        self.assertEqual(chunks_payload[0]["page_content"], "첫 번째 청크 내용")
        self.assertEqual(chunks_payload[0]["metadata"]["diagram_id"], "보1")
        self.assertIn("보1", preview)
        self.assertIn("첫 번째 청크 내용", preview)

    def test_save_chunk_cache_normalizes_source_to_pdf_path(self):
        from rag.service.chunk_cache import save_chunk_cache

        documents = [
            Document(
                page_content="정규화 테스트",
                metadata={"page": 389, "source": "389.md", "chunk_type": "child"},
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            save_chunk_cache(documents, cache_dir, source_path=PDF_PATH)
            chunks_payload = json.loads((cache_dir / "chunks.json").read_text(encoding="utf-8"))

        self.assertEqual(chunks_payload[0]["metadata"]["source"], "data/raw/230630_자동차사고 과실비율 인정기준_최종.pdf")

    def test_load_chunk_cache_restores_documents(self):
        from rag.service.chunk_cache import load_chunk_cache, save_chunk_cache

        documents = [
            Document(
                page_content="복원 테스트",
                metadata={"page": 147, "chunk_type": "flat"},
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            save_chunk_cache(documents, cache_dir)
            restored = load_chunk_cache(cache_dir)

        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].page_content, "복원 테스트")
        self.assertEqual(restored[0].metadata["page"], 147)
        self.assertEqual(restored[0].metadata["chunk_type"], "flat")

    def test_load_chunk_cache_normalizes_legacy_markdown_source_to_pdf_path(self):
        from rag.service.chunk_cache import CHUNKS_FILENAME, load_chunk_cache

        payload = [
            {
                "page_content": "레거시 캐시",
                "metadata": {"page": 389, "source": "389.md", "chunk_type": "child"},
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            (cache_dir / CHUNKS_FILENAME).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            restored = load_chunk_cache(cache_dir, source_path=PDF_PATH)

        self.assertEqual(restored[0].metadata["source"], "data/raw/230630_자동차사고 과실비율 인정기준_최종.pdf")


if __name__ == "__main__":
    unittest.main()
