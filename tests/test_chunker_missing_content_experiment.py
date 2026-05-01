"""실험: case-boundary chunker 가 본문을 누락시키던 경로의 수정 후 동작 검증.

CHUNK_COVERAGE_AUDIT.md 에 기록된 6 페이지 (013/017/129/130/131/344) 중
청커 자체 결함으로 누락되던 두 가지 root cause 를 수정한 뒤,
같은 입력에서 본문이 모두 청크에 포함되는지 확인한다.

수정 내용 (rag/chunkers/case_boundary.py):
1. `GENERAL_SUBSECTION_PATTERN` 이 `**(N)…**` 형태의 강조 마크다운도 매치하도록 확장.
2. `_is_case_category_heading` 도 같은 강조 마크다운을 strip 후 매치.
3. `pending_context_blocks` 가 케이스 표에 한 번도 소비되지 않은 채로 끝나거나
   top-level heading 을 만나면 idle 로 되돌려 누락을 방지. 이미 표에 소비된
   ancestor breadcrumb 은 idle 로 흘리지 않음 (중복 방지).

실행:
    python -m unittest discover -s tests \\
        -p test_chunker_missing_content_experiment.py -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

from langchain_core.documents import Document

from rag.chunkers import CaseBoundaryChunker
from rag.chunkers.case_boundary import (
    GENERAL_SUBSECTION_PATTERN,
    CaseBoundaryChunker as CB,
)


def _all_text(chunks) -> str:
    return "\n\n".join(c.text for c in chunks)


def _contains(chunks, needle: str) -> bool:
    return needle.replace(" ", "") in _all_text(chunks).replace(" ", "")


class Pattern_BoldWrappedSubsectionMatches(unittest.TestCase):
    """수정 후 GENERAL_SUBSECTION_PATTERN 이 ** 강조를 허용한다."""

    def test_bold_wrapped_subsection_matches(self):
        self.assertIsNotNone(
            GENERAL_SUBSECTION_PATTERN.match("### **(1) 현저한 과실 : 10%까지 가산 가능**")
        )
        self.assertIsNotNone(
            GENERAL_SUBSECTION_PATTERN.match("### (2) 중대한 과실 : 20%까지 가산 가능(예외있음)")
        )

    def test_case_category_heading_strips_bold(self):
        chunker = CB(mode="B")
        # bold-wrapped form, level 2 → category
        self.assertTrue(chunker._is_case_category_heading("## **(3) 횡단보도 부근**"))
        # plain form, level 2 → still category
        self.assertTrue(chunker._is_case_category_heading("## (1) 신뢰의 원칙 - 예견가능성"))
        # level 3 → not a category (would be misclassified otherwise)
        self.assertFalse(chunker._is_case_category_heading("### (1) 현저한 과실"))


class Page017_PrefacedSubsectionRecovered(unittest.TestCase):
    """`## (N)` 카테고리 헤딩 다음에 케이스 표가 안 와도 본문이 살아남는다."""

    SOURCE = (
        "# 3. 과실비율 인정기준의 기본원칙\n\n"
        "## (1) 신뢰의 원칙 - 예견가능성\n"
        "자동차의 운전자는 통상 예견되는 사태에 대비하여 회피할 수 있는 정도의 주의의무를 다해야 한다.\n\n"
        "일반적으로 교통규칙을 준수한 운전자는 다른 차량 운전자도 교통규칙을 준수할 것이라고 신뢰하게 된다.\n\n"
        "다만, 신뢰의 원칙은 특별한 사정이 있는 경우에는 그 적용이 배제된다.\n\n"
        "즉, 사고 순간에는 회피가능성이 없으나 사고에 대한 예견가능성이 있으면 미리 대비해야 한다.\n\n"
        "### 1) 자동차 대 자동차\n"
        "① 신호기의 신호\n\n"
        "신호등에 의하여 교통정리가 행하여지고 있는 교차로를 진행신호에 따라 진행하는 차량의 운전자.\n"
    )

    def test_all_paragraphs_present(self):
        doc = Document(page_content=self.SOURCE, metadata={"page": 17, "source": "017.md"})
        chunks = CaseBoundaryChunker(mode="B").chunk(doc)
        for needle in [
            "(1) 신뢰의 원칙 - 예견가능성",
            "통상 예견되는 사태에 대비",
            "신뢰하게 된다",
            "특별한 사정이 있는 경우에는 그 적용이 배제된다",
            "사고 순간에는 회피가능성이 없으나",
            "1) 자동차 대 자동차",
            "신호등에 의하여 교통정리",
        ]:
            self.assertTrue(
                _contains(chunks, needle),
                f"본문이 누락됨: '{needle}'",
            )


class Page129_BoldSubsectionPreservedInIsolation(unittest.TestCase):
    """`### **(1) 현저한 과실**` 본문이 모두 살아남는다 (단일 페이지)."""

    SOURCE = (
        "# **3. 수정요소(인과관계를 감안한 과실비율 조정)의 해설**\n\n"
        "### **(1) 현저한 과실 : 10%까지 가산 가능**\n\n"
        "수정요소가 사고의 발생 또는 손해의 확대에 영향을 끼친 것으로 판단될 때에 5 ~ 10% 범위로 과실을 가산할 수 있다.\n\n"
        "- 현저한 과실 사이에는 중복 적용이 가능하다.\n"
        "- 아래 사유들이 그 예이다.\n"
    )

    def test_bold_subsection_body_present(self):
        doc = Document(page_content=self.SOURCE, metadata={"page": 129, "source": "129.md"})
        chunks = CaseBoundaryChunker(mode="B").chunk(doc)
        for needle in [
            "(1) 현저한 과실 : 10%까지 가산 가능",
            "수정요소가 사고의 발생 또는",
            "현저한 과실 사이에는 중복 적용이 가능하다",
        ]:
            self.assertTrue(_contains(chunks, needle), f"누락: '{needle}'")


class MultiPage_RealCorpus_Pages_129_to_132(unittest.TestCase):
    """실제 chunks.json 환경처럼 다중 페이지 입력에서 누락이 없어졌는지 확인."""

    def test_pages_129_130_131_132_full_recovery(self):
        md_dir = Path("/home/nyong/mdm/data/llama_md/main_pdf")
        documents = [
            Document(
                page_content=(md_dir / f"{p:03d}.md").read_text(encoding="utf-8"),
                metadata={"page": p, "source": f"{p:03d}.md"},
            )
            for p in [129, 130, 131, 132]
        ]

        chunks = CaseBoundaryChunker(mode="B").chunk(documents)

        for needle in [
            "(1) 현저한 과실",
            "현저한 과실 사이에는 중복 적용",
            "③ 시속 20km 미만 제한속도 위반",
            "야간 교통안전에 지장을 주는 차량 유리의 암도",
            "도로교통법 시행령 제28조(자동차 창유리 가시광선 투과율의 기준)",
            "운전자는 자동차등 또는 노면전차의 운전 중에는 휴대용 전화",
            "지리안내 영상 또는 교통정보안내 영상",
            "노면전차 운전자가 운전에 필요한 영상표시장치를 조작하는 경우",
            # (2) 중대한 과실 도 그대로 살아 있어야 함
            "(2) 중대한 과실",
        ]:
            self.assertTrue(
                _contains(chunks, needle),
                f"누락: '{needle}'",
            )


class Page344_ImageDescriptionBullets(unittest.TestCase):
    """case 표 다음의 이미지 설명 불릿이 보존된다 (이미 통과하던 케이스, 회귀 방지)."""

    SOURCE = (
        "## 차25 유턴 사고\n\n"
        "| 차25 | 유턴 사고 |\n"
        "| --- | --- |\n"
        "| (A) 직진 | (B) 유턴 |\n"
        "![page_344_table_1](../img/page_344_table_1.png)\n\n"
        "**[이미지 설명]**\n"
        "*   **(가) 상시유턴구역:** 신호등이 없는 유턴 구역에서 직진 차량 A와 유턴 차량 B가 충돌.\n"
        "*   **(나) 신호유턴:** 좌회전 신호 시 유턴 가능 구역에서 직진 차량 A와 유턴 차량 B가 충돌.\n\n"
        "### 사고 상황\n"
        "* A와 B가 충돌했다.\n"
    )

    def test_bullets_present(self):
        doc = Document(page_content=self.SOURCE, metadata={"page": 344, "source": "344.md"})
        chunks = CaseBoundaryChunker(mode="B").chunk(doc)
        for needle in [
            "(가) 상시유턴구역",
            "신호등이 없는 유턴 구역에서 직진",
            "(나) 신호유턴",
            "좌회전 신호 시 유턴 가능 구역에서",
        ]:
            self.assertTrue(_contains(chunks, needle), f"누락: '{needle}'")


if __name__ == "__main__":
    unittest.main(verbosity=2)
