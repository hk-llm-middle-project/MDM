import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.clean_case_boundary_chunk_tables import clean_case_boundary_tables


class CleanCaseBoundaryChunkTablesTest(unittest.TestCase):
    def test_cleans_table_child_and_rebuilds_parent_from_cleaned_children(self):
        table_text = (
            "## 차43-7 안전지대 통과 직진 대 선행 진로변경\n"
            "|     | 기본 과실비율 | 기본 과실비율 | 기본 과실비율 | 기본 과실비율 | "
            "(가) A100<br/>(나) A70 | B0<br/>B30 | B0<br/>B30 |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| (가) | 과실비율 조정예시 | A 현저한 과실 | | +10 | | | |\n"
            "|     | | B 중대한 과실 | | | +20 | | |\n"
            "![page_389_table_1](../../upstage_output/main_pdf/final/img/page_389_table_1.png)"
        )
        section_text = "### 사고 상황\n* 안전지대 사고 상황입니다."
        docs = [
            {
                "page_content": f"{table_text}\n\n{section_text}",
                "metadata": {
                    "chunk_id": 0,
                    "chunk_type": "parent",
                    "diagram_id": "차43-7(가)",
                    "image_path": "../../upstage_output/main_pdf/final/img/page_389_table_1.png",
                },
            },
            {
                "page_content": table_text,
                "metadata": {
                    "chunk_id": 1,
                    "chunk_type": "child",
                    "diagram_id": "차43-7(가)",
                    "parent_id": 0,
                    "image_path": "../../upstage_output/main_pdf/final/img/page_389_table_1.png",
                },
            },
            {
                "page_content": section_text,
                "metadata": {
                    "chunk_id": 2,
                    "chunk_type": "child",
                    "diagram_id": "차43-7(가)",
                    "parent_id": 0,
                },
            },
        ]

        cleaned = clean_case_boundary_tables(docs)

        table_child = cleaned[1]
        parent = cleaned[0]
        self.assertIn("### 기본 과실비율", table_child["page_content"])
        self.assertIn("| 유형 | A 과실 | B 과실 |", table_child["page_content"])
        self.assertIn("### 사고 상황", parent["page_content"])
        self.assertIn("### 기본 과실비율", parent["page_content"])
        self.assertNotIn("|     | 기본 과실비율", parent["page_content"])
        self.assertEqual(
            parent["metadata"]["image_path"],
            "../../upstage_output/main_pdf/final/img/page_389_table_1.png",
        )

    def test_leaves_general_and_preface_chunks_unchanged(self):
        docs = [
            {
                "page_content": "| 차1-1 | 참고 |\n| --- | --- |",
                "metadata": {"chunk_id": 0, "chunk_type": "general"},
            },
            {
                "page_content": "# 발간사\n본문",
                "metadata": {"chunk_id": 1, "chunk_type": "preface"},
            },
        ]

        cleaned = clean_case_boundary_tables(docs)

        self.assertEqual(cleaned, docs)


if __name__ == "__main__":
    unittest.main()
