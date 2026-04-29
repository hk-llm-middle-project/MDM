import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.clean_llamaparser_diagram_tables import clean_markdown


class CleanLlamaParserDiagramTablesTest(unittest.TestCase):
    def test_rewrites_diagram_table_to_fault_ratio_table(self):
        markdown = (
            "### 1) 자동차 녹색신호 교차로 통과 후(後) [보1]\n\n"
            "| 보1 | 보1 | 보1 | 보1 | 보행자 적색신호 횡단 개시 | |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| 보행자 및 차량의 교차로 통행 상황도 | 보행자 기본 과실비율 | | 70 | | |\n"
            "| | 과실비율 조정예시 | ① | 야간·기타 시야장애 | +5 | |\n"
            "| | | ② | 간선도로 | +5 | |\n"
            "| | | ③ | 정지·후퇴·ㄹ자 보행 | +5 | |\n"
            "| | | | | 주택·상점가·학교 | -5 |\n"
            "| | | ④ | 어린이·노인·장애인 | -5 | |\n"
            "| | | | | 어린이·노인·장애인보호구역 | -15 |\n"
            "| | | | | 차의 현저한 과실 | -10 |\n"
            "| | | | 차의 중대한 과실 | -20 | |\n"
            "| | | ⑤ | 보행자 급진입 | 비적용 | |\n"
            "| | | | | | 보·차도 구분 없음 |\n"
            "![page_39_table_1](../../upstage_output/main_pdf/final/img/page_39_table_1.png)\n"
        )

        cleaned = clean_markdown(markdown)

        self.assertIn("| 구분 | 항목 | 과실 |", cleaned)
        self.assertIn("| 기본 과실비율 | 보행자 기본 과실비율 | 70 |", cleaned)
        self.assertIn("| ① | 야간·기타 시야장애 | +5 |", cleaned)
        self.assertIn("| ③ | 주택·상점가·학교 | -5 |", cleaned)
        self.assertIn("| ④ | 차의 현저한 과실 | -10 |", cleaned)
        self.assertIn("| ⑤ | 보·차도 구분 없음 | 비적용 |", cleaned)
        self.assertIn("![page_39_table_1]", cleaned)
        self.assertNotIn("보행자 및 차량의 교차로 통행 상황도", cleaned)

    def test_rewrites_ab_fault_ratio_table_to_llm_friendly_sections(self):
        markdown = (
            "## 차43-7 안전지대 통과 직진 대 선행 진로변경\n"
            "**(A) 후행직진**\n"
            "(가) 안전지대 벗어나기 전\n"
            "(나) 안전지대 벗어난 후\n"
            "**(B) 선행진로변경**\n\n"
            "|     | 기본 과실비율 | 기본 과실비율 | 기본 과실비율 | 기본 과실비율 | "
            "(가) A100<br/>(나) A70 | B0<br/>B30 | B0<br/>B30 | B0<br/>B30 | B0<br/>B30 |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| (가) | 과실비율 조정예시 | A 현저한 과실 | | +10 | | | | | |\n"
            "|     | | A 중대한 과실 | | +20 | | | | | |\n"
            "|     | | B진로변경 ① 신호불이행·지연 | | +10 | | | | | |\n"
            "|     | | | | | (나) | B 현저한 과실 | | | +10 |\n"
            "|     | | B 중대한 과실 | | | +20 | | | | |\n"
            "![page_389_table_1](../../upstage_output/main_pdf/final/img/page_389_table_1.png)\n"
        )

        cleaned = clean_markdown(markdown)

        self.assertIn("### 기본 과실비율", cleaned)
        self.assertIn("| 유형 | A 과실 | B 과실 |", cleaned)
        self.assertIn("| (가) | A100 | B0 |", cleaned)
        self.assertIn("| (나) | A70 | B30 |", cleaned)
        self.assertIn("### 과실비율 조정예시", cleaned)
        self.assertIn("| 대상 | 수정요소 | A 조정 | B 조정 |", cleaned)
        self.assertIn("| A | 현저한 과실 | +10 |  |", cleaned)
        self.assertIn("| B | 진로변경 신호불이행·지연 |  | +10 |", cleaned)
        self.assertIn("| B | 현저한 과실 |  | +10 |", cleaned)
        self.assertIn("| B | 중대한 과실 |  | +20 |", cleaned)
        self.assertNotIn("| 유형 | 대상 | 수정요소 | A 조정 | B 조정 |", cleaned)


if __name__ == "__main__":
    unittest.main()
