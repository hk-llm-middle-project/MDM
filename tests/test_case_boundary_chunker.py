import unittest

from langchain_core.documents import Document

from rag.chunkers import CaseBoundaryChunker


class CaseBoundaryChunkerBasicCaseTest(unittest.TestCase):
    def test_emits_parent_and_table_child_with_image_path_for_simple_case(self):
        document = Document(
            page_content=(
                "## 차2-5 녹색 직진 대 녹색화살표\n\n"
                "| 차2-5 | 녹색 직진 |\n"
                "| --- | --- |\n"
                "| (A) | 좌회전 |\n"
                "![page_175_table_1](../img/page_175_table_1.png)\n"
            ),
            metadata={"page": 175, "source": "175.md"},
        )

        chunks = CaseBoundaryChunker(mode="B").chunk(document)

        self.assertEqual([chunk.chunk_type for chunk in chunks], ["parent", "child"])
        self.assertEqual([chunk.diagram_id for chunk in chunks], ["차2-5", "차2-5"])
        self.assertEqual([chunk.parent_id for chunk in chunks], [None, 0])
        self.assertEqual(chunks[0].image_path, "../img/page_175_table_1.png")
        self.assertEqual(chunks[1].image_path, "../img/page_175_table_1.png")
        self.assertIn("| 차2-5 |", chunks[0].text)
        self.assertIn("| 차2-5 |", chunks[1].text)
        self.assertIn("![page_175_table_1]", chunks[1].text)

    def test_splits_case_body_into_section_children(self):
        document = Document(
            page_content=(
                "## 차2-5 녹색 직진 대 녹색화살표\n\n"
                "| 차2-5 | 녹색 직진 |\n"
                "| --- | --- |\n"
                "| (A) | 좌회전 |\n"
                "![page_175_table_1](../img/page_175_table_1.png)\n\n"
                "### 사고 상황\n"
                "* A와 B가 충돌했다.\n\n"
                "### 기본 과실비율 해설\n"
                "* 기본 비율 설명입니다.\n\n"
                "### <u>관련 법규</u>\n"
                "* 도로교통법 제5조.\n"
            ),
            metadata={"page": 175, "source": "175.md"},
        )

        chunks = CaseBoundaryChunker(mode="B").chunk(document)

        self.assertEqual(
            [chunk.chunk_type for chunk in chunks],
            ["parent", "child", "child", "child", "child"],
        )
        self.assertEqual([chunk.diagram_id for chunk in chunks], ["차2-5"] * 5)
        self.assertEqual([chunk.parent_id for chunk in chunks], [None, 0, 0, 0, 0])
        self.assertEqual(chunks[1].image_path, "../img/page_175_table_1.png")
        for child in chunks[2:]:
            self.assertIsNone(child.image_path)
        self.assertIn("### 사고 상황", chunks[2].text)
        self.assertIn("A와 B가 충돌했다", chunks[2].text)
        self.assertIn("### 기본 과실비율 해설", chunks[3].text)
        self.assertIn("기본 비율 설명입니다", chunks[3].text)
        self.assertIn("관련 법규", chunks[4].text)
        self.assertIn("도로교통법 제5조", chunks[4].text)
        self.assertIn("### 사고 상황", chunks[0].text)
        self.assertIn("도로교통법 제5조", chunks[0].text)


class CaseBoundaryChunkerPageContinuationTest(unittest.TestCase):
    def test_continues_case_across_pages_and_uses_actual_page_for_each_child(self):
        documents = [
            Document(
                page_content=(
                    "## 차2-5 신호 교차로 사고\n\n"
                    "| 차2-5 | 직진 |\n"
                    "| --- | --- |\n"
                    "| (A) | 좌회전 |\n"
                    "![page_175_table_1](../img/page_175_table_1.png)\n\n"
                    "### 사고 상황\n"
                    "* A와 B가 충돌했다.\n"
                ),
                metadata={"page": 175, "source": "175.md"},
            ),
            Document(
                page_content=(
                    "### 기본 과실비율 해설\n"
                    "* 다음 페이지에 이어진 해설입니다.\n\n"
                    "### 관련 법규\n"
                    "* 도로교통법 제5조.\n"
                ),
                metadata={"page": 176, "source": "176.md"},
            ),
        ]

        chunks = CaseBoundaryChunker(mode="B").chunk(documents)

        self.assertEqual(
            [chunk.chunk_type for chunk in chunks],
            ["parent", "child", "child", "child", "child"],
        )
        self.assertEqual([chunk.diagram_id for chunk in chunks], ["차2-5"] * 5)
        self.assertEqual([chunk.parent_id for chunk in chunks], [None, 0, 0, 0, 0])
        self.assertEqual([chunk.page for chunk in chunks], [175, 175, 175, 176, 176])
        self.assertEqual(
            [chunk.source for chunk in chunks],
            ["175.md", "175.md", "175.md", "176.md", "176.md"],
        )
        self.assertIn("기본 과실비율 해설", chunks[3].text)
        self.assertIn("관련 법규", chunks[4].text)
        self.assertIn("기본 과실비율 해설", chunks[0].text)
        self.assertIn("관련 법규", chunks[0].text)

    def test_starts_new_parent_when_next_page_begins_with_new_case_table(self):
        documents = [
            Document(
                page_content=(
                    "| 보1 | 횡단 |\n"
                    "| --- | --- |\n"
                    "| (A) | 직진 |\n"
                    "![p1](../img/p1.png)\n\n"
                    "### 사고 상황\n"
                    "* 보1 사고 상황.\n"
                ),
                metadata={"page": 39, "source": "039.md"},
            ),
            Document(
                page_content=(
                    "| 보2 | 다른 횡단 |\n"
                    "| --- | --- |\n"
                    "| (A) | 직진 |\n"
                    "![p2](../img/p2.png)\n\n"
                    "### 사고 상황\n"
                    "* 보2 사고 상황.\n"
                ),
                metadata={"page": 40, "source": "040.md"},
            ),
        ]

        chunks = CaseBoundaryChunker(mode="B").chunk(documents)

        self.assertEqual([chunk.chunk_type for chunk in chunks], ["parent", "child", "child", "parent", "child", "child"])
        self.assertEqual([chunk.diagram_id for chunk in chunks], ["보1", "보1", "보1", "보2", "보2", "보2"])
        self.assertEqual([chunk.parent_id for chunk in chunks], [None, 0, 0, None, 3, 3])
        self.assertEqual([chunk.page for chunk in chunks], [39, 39, 39, 40, 40, 40])

    def test_splits_decorated_child_headings_across_pages(self):
        documents = [
            Document(
                page_content=(
                    "| 차7-2 | 직진 대 일방통행위반 직진 |\n"
                    "| --- | --- |\n"
                    "| (A) 직진 | (B) 직진 |\n"
                    "![p230](../img/p230.png)\n\n"
                    "### 수정요소(인과관계를 감안한 과실비율 조정) 해설\n"
                    "* 수정요소 설명입니다.\n"
                ),
                metadata={"page": 230, "source": "230.md"},
            ),
            Document(
                page_content=(
                    "자동차사고 과실비율 인정기준 | 제3편 사고유형별 과실비율 적용기준 230\n\n"
                    "<u>**활용시 참고 사항**</u>\n\n"
                    "* 활용 참고입니다.\n\n"
                    "<u>**관련 법규**</u>\n\n"
                    "**⊙ 도로교통법 제5조**\n"
                    "법규 본문입니다.\n\n"
                    "<u>**참고 판례**</u>\n\n"
                    "**⊙ 대법원 판례**\n"
                    "판례 본문입니다.\n"
                ),
                metadata={"page": 231, "source": "231.md"},
            ),
        ]

        chunks = CaseBoundaryChunker(mode="B").chunk(documents)

        child_texts = [chunk.text for chunk in chunks if chunk.chunk_type == "child"]
        self.assertTrue(any("### 활용시 참고 사항" in text for text in child_texts))
        self.assertTrue(any("### 관련 법규" in text for text in child_texts))
        self.assertTrue(any("### 참고 판례" in text for text in child_texts))
        modification_child = next(
            text for text in child_texts if "### 수정요소" in text
        )
        self.assertNotIn("활용시 참고 사항", modification_child)

    def test_intermediate_case_group_heading_starts_next_case_context(self):
        documents = [
            Document(
                page_content=(
                    "| 차7-2 | 직진 대 일방통행위반 직진 |\n"
                    "| --- | --- |\n"
                    "| (A) 직진 | (B) 직진 |\n"
                    "![p230](../img/p230.png)\n\n"
                    "### 참고 판례\n"
                    "* 차7-2 판례입니다.\n\n"
                    "# (3) 한쪽 지시표지 있는 교차로\n\n"
                    "## 2) 직진 대 좌회전 [차8]\n\n"
                    "| 차8-1 | 직진 대 일시정지위반 좌회전 |\n"
                    "| --- | --- |\n"
                    "| (A) 직진 | (B) 좌회전 |\n"
                    "![p232](../img/p232.png)\n\n"
                    "### 사고 상황\n"
                    "* 차8-1 사고 상황입니다.\n"
                ),
                metadata={"page": 232, "source": "232.md"},
            )
        ]

        chunks = CaseBoundaryChunker(mode="B").chunk(documents)

        case_7_parent = next(
            chunk for chunk in chunks if chunk.chunk_type == "parent" and chunk.diagram_id == "차7-2"
        )
        case_7_children = [
            chunk
            for chunk in chunks
            if chunk.chunk_type == "child" and chunk.diagram_id == "차7-2"
        ]
        case_8_parent = next(
            chunk for chunk in chunks if chunk.chunk_type == "parent" and chunk.diagram_id == "차8-1"
        )
        self.assertNotIn("## 2) 직진 대 좌회전 [차8]", case_7_parent.text)
        self.assertNotIn("# (3) 한쪽 지시표지 있는 교차로", case_7_parent.text)
        self.assertTrue(
            all(
                "## 2) 직진 대 좌회전 [차8]" not in child.text
                and "# (3) 한쪽 지시표지 있는 교차로" not in child.text
                for child in case_7_children
            )
        )
        self.assertIn("# (3) 한쪽 지시표지 있는 교차로", case_8_parent.text)
        self.assertIn("## 2) 직진 대 좌회전 [차8]", case_8_parent.text)

    def test_level_two_category_heading_starts_next_case_context(self):
        documents = [
            Document(
                page_content=(
                    "<u>참고 판례</u>\n\n"
                    "이전 기준 참고 판례입니다.\n\n"
                    "## (3) 횡단보도 부근(신호등 있음)\n\n"
                    "### 1) 직진 자동차 횡단보도 통과 후 [보14~보16]\n\n"
                    "| 보14 | 횡단보도 부근 사고 |\n"
                    "| --- | --- |\n"
                    "| (보) 횡단 | (차) 직진 |\n"
                    "![p70](../img/p70.png)\n"
                ),
                metadata={"page": 70, "source": "070.md"},
            )
        ]

        chunks = CaseBoundaryChunker(mode="B").chunk(documents)

        general_text = "\n".join(
            chunk.text for chunk in chunks if chunk.chunk_type == "general"
        )
        parent = next(chunk for chunk in chunks if chunk.chunk_type == "parent")
        self.assertNotIn("## (3) 횡단보도 부근(신호등 있음)", general_text)
        self.assertIn("## (3) 횡단보도 부근(신호등 있음)", parent.text)
        self.assertIn("### 1) 직진 자동차 횡단보도 통과 후 [보14~보16]", parent.text)

    def test_consecutive_heading_based_case_tables_share_following_sections(self):
        documents = [
            Document(
                page_content=(
                    "## 2) 우회전 자동차 횡단보도 통과 후 [보17~보18]\n\n"
                    "### 보17 보행자 녹색(적색)신호 횡단 중 사고\n"
                    "| 보행자 기본 과실비율 | (가) 녹색 횡단<br/>(나) 적색 횡단 |\n"
                    "| --- | --- |\n"
                    "| 과실비율 조정 예시 | 보행자 급진입 |\n"
                    "![p78-1](../img/p78-1.png)\n\n"
                    "----\n\n"
                    "### 보18 보행자 적색신호 횡단 중 사고\n"
                    "| 보행자 기본 과실비율 | 60 |\n"
                    "| --- | --- |\n"
                    "| 과실비율 조정 예시 | 보행자 급진입 |\n"
                    "![p78-2](../img/p78-2.png)\n"
                ),
                metadata={"page": 78, "source": "078.md"},
            ),
            Document(
                page_content=(
                    "### 사고 상황\n"
                    "* **보17** 보17 사고 상황입니다.\n"
                    "* **보18** 보18 사고 상황입니다.\n\n"
                    "### 기본 과실비율 해설\n"
                    "* **보17** 보17 해설입니다.\n"
                    "* **보18** 보18 해설입니다.\n"
                ),
                metadata={"page": 79, "source": "079.md"},
            ),
        ]

        chunks = CaseBoundaryChunker(mode="B").chunk(documents)

        parent_ids = {chunk.diagram_id for chunk in chunks if chunk.chunk_type == "parent"}
        self.assertEqual(parent_ids, {"보17(가)", "보17(나)", "보18"})
        for diagram_id in parent_ids:
            parent = next(
                chunk
                for chunk in chunks
                if chunk.chunk_type == "parent" and chunk.diagram_id == diagram_id
            )
            self.assertIn("사고 상황", parent.text)
            self.assertIn("기본 과실비율 해설", parent.text)
            children = [
                chunk
                for chunk in chunks
                if chunk.chunk_type == "child" and chunk.diagram_id == diagram_id
            ]
            self.assertTrue(any("사고 상황" in child.text for child in children))
            self.assertTrue(any("기본 과실비율 해설" in child.text for child in children))

        bo17_child_text = "\n".join(
            chunk.text
            for chunk in chunks
            if chunk.chunk_type == "child" and chunk.diagram_id == "보17(가)"
        )
        bo18_child_text = "\n".join(
            chunk.text
            for chunk in chunks
            if chunk.chunk_type == "child" and chunk.diagram_id == "보18"
        )
        self.assertIn("보17 사고 상황입니다", bo17_child_text)
        self.assertIn("보17 해설입니다", bo17_child_text)
        self.assertNotIn("보18 사고 상황입니다", bo17_child_text)
        self.assertNotIn("보18 해설입니다", bo17_child_text)
        self.assertIn("보18 사고 상황입니다", bo18_child_text)
        self.assertIn("보18 해설입니다", bo18_child_text)
        self.assertNotIn("보17 사고 상황입니다", bo18_child_text)
        self.assertNotIn("보17 해설입니다", bo18_child_text)

    def test_drops_running_header_footer_and_sidebar_navigation_from_case_text(self):
        document = Document(
            page_content=(
                "| 차43-1 | 차로 감소 도로 사고 |\n"
                "| --- | --- |\n"
                "| (A) 본선차 | (B) 합류차 |\n"
                "![p371](../img/p371.png)\n\n"
                "### 활용시 참고 사항\n"
                "* 일반 도로 및 고속도로 사고에 본 도표를 적용한다.\n\n"
                "자동차사고 과실비율 인정기준 | 제3편 사고유형별 과실비율 적용기준 371\n"
                "목차\n"
                "제1장. 자동차와 보행자의 사고\n"
                "**제2장. 자동차와 자동차(이륜차 포함)의 사고**\n"
                "제3장. 자동차와 자전거(농기계 포함)의 사고\n\n"
                "### 관련 법규\n"
                "* 도로교통법 제19조.\n"
            ),
            metadata={"page": 371, "source": "371.md"},
        )

        chunks = CaseBoundaryChunker(mode="B").chunk(document)

        combined_text = "\n".join(chunk.text for chunk in chunks)
        self.assertIn("일반 도로 및 고속도로 사고", combined_text)
        self.assertIn("도로교통법 제19조", combined_text)
        self.assertNotIn("자동차사고 과실비율 인정기준 |", combined_text)
        self.assertNotIn("목차", combined_text)
        self.assertNotIn("제1장. 자동차와 보행자의 사고", combined_text)
        self.assertNotIn("제2장. 자동차와 자동차", combined_text)
        self.assertNotIn("제3장. 자동차와 자전거", combined_text)

    def test_keeps_actual_chapter_title_when_it_is_not_navigation_furniture(self):
        document = Document(
            page_content=(
                "# 제2장. 자동차와 자동차(이륜차 포함)의 사고\n\n"
                "이 장은 자동차 상호 간 사고 기준을 설명한다.\n"
            ),
            metadata={"page": 125, "source": "125.md"},
        )

        chunks = CaseBoundaryChunker(mode="B").chunk(document)

        combined_text = "\n".join(chunk.text for chunk in chunks)
        self.assertIn("# 제2장. 자동차와 자동차(이륜차 포함)의 사고", combined_text)
        self.assertIn("자동차 상호 간 사고 기준", combined_text)


class CaseBoundaryChunkerMultiCaseTest(unittest.TestCase):
    def test_creates_separate_parents_for_two_cases_on_same_page(self):
        document = Document(
            page_content=(
                "| 보20 | 횡단보도 사고 |\n"
                "| --- | --- |\n"
                "| (A) | 직진 |\n"
                "![p20](../img/p20.png)\n\n"
                "| 보21 | 육교 사고 |\n"
                "| --- | --- |\n"
                "| (A) | 직진 |\n"
                "![p21](../img/p21.png)\n"
            ),
            metadata={"page": 85, "source": "085.md"},
        )

        chunks = CaseBoundaryChunker(mode="B").chunk(document)

        self.assertEqual([chunk.chunk_type for chunk in chunks], ["parent", "child", "parent", "child"])
        self.assertEqual([chunk.diagram_id for chunk in chunks], ["보20", "보20", "보21", "보21"])
        self.assertEqual([chunk.parent_id for chunk in chunks], [None, 0, None, 2])
        self.assertEqual(chunks[0].image_path, "../img/p20.png")
        self.assertEqual(chunks[1].image_path, "../img/p20.png")
        self.assertEqual(chunks[2].image_path, "../img/p21.png")
        self.assertEqual(chunks[3].image_path, "../img/p21.png")

    def test_uses_preceding_heading_case_id_when_table_lacks_one(self):
        document = Document(
            page_content=(
                "## 차43-7 안전지대 통과\n\n"
                "|     | 기본 과실비율 | (가) A100 | (나) A70 |\n"
                "| --- | -------- | ------- | ------- |\n"
                "| 1   | a        | b       | c       |\n"
                "![table](../img/p389.png)\n"
            ),
            metadata={"page": 389, "source": "389.md"},
        )

        chunks = CaseBoundaryChunker(mode="B").chunk(document)

        diagram_ids = [chunk.diagram_id for chunk in chunks]
        self.assertIn("차43-7(가)", diagram_ids)
        self.assertIn("차43-7(나)", diagram_ids)

    def test_duplicates_shared_table_into_each_variant_parent(self):
        document = Document(
            page_content=(
                "## 차43-7 안전지대 통과\n\n"
                "| 차43-7 (가) (나) | 직진 |\n"
                "| --- | --- |\n"
                "| (A) | 후행직진 |\n"
                "![table](../img/p389.png)\n\n"
                "### 사고 상황\n"
                "* (가) 안전지대 벗어나기 전 사고.\n"
            ),
            metadata={"page": 389, "source": "389.md"},
        )

        chunks = CaseBoundaryChunker(mode="B").chunk(document)

        diagram_ids = [chunk.diagram_id for chunk in chunks]
        self.assertIn("차43-7(가)", diagram_ids)
        self.assertIn("차43-7(나)", diagram_ids)
        # Each variant gets its own parent and the table/image child duplicated.
        parents = [chunk for chunk in chunks if chunk.chunk_type == "parent"]
        self.assertEqual(len(parents), 2)
        self.assertEqual({parent.diagram_id for parent in parents}, {"차43-7(가)", "차43-7(나)"})
        for parent in parents:
            self.assertEqual(parent.image_path, "../img/p389.png")
        # Every child belongs to one of the two parents.
        children = [chunk for chunk in chunks if chunk.chunk_type == "child"]
        parent_ids = {parent.chunk_id for parent in parents}
        for child in children:
            self.assertIn(child.parent_id, parent_ids)

    def test_duplicates_shared_sections_after_consecutive_case_tables(self):
        document = Document(
            page_content=(
                "| 보20 | 횡단보도 사고 |\n"
                "| --- | --- |\n"
                "| (A) | 직진 |\n"
                "![p20](../img/p20.png)\n\n"
                "| 보21 | 육교 사고 |\n"
                "| --- | --- |\n"
                "| (A) | 직진 |\n"
                "![p21](../img/p21.png)\n\n"
                "### 사고 상황\n"
                "* 공통 사고 상황입니다.\n\n"
                "### 기본 과실비율 해설\n"
                "* 공통 해설입니다.\n"
            ),
            metadata={"page": 85, "source": "085.md"},
        )

        chunks = CaseBoundaryChunker(mode="B").chunk(document)

        shared_children = [
            chunk
            for chunk in chunks
            if chunk.chunk_type == "child" and "공통 사고 상황" in chunk.text
        ]
        self.assertEqual(
            {chunk.diagram_id for chunk in shared_children},
            {"보20", "보21"},
        )
        shared_parents = [
            chunk
            for chunk in chunks
            if chunk.chunk_type == "parent" and "공통 해설" in chunk.text
        ]
        self.assertEqual(
            {chunk.diagram_id for chunk in shared_parents},
            {"보20", "보21"},
        )

    def test_splits_variant_labeled_section_text_per_variant(self):
        document = Document(
            page_content=(
                "## 차43-7 안전지대 통과\n\n"
                "| 차43-7 (가) (나) | 직진 |\n"
                "| --- | --- |\n"
                "| (A) | 후행직진 |\n"
                "![table](../img/p389.png)\n\n"
                "### 사고 상황\n"
                "* (가) 안전지대 벗어나기 전 사고.\n"
                "* (나) 안전지대 벗어난 후 사고.\n\n"
                "### 관련 법규\n"
                "* 공통 법규.\n"
            ),
            metadata={"page": 389, "source": "389.md"},
        )

        chunks = CaseBoundaryChunker(mode="B").chunk(document)

        variant_a_children = [
            chunk
            for chunk in chunks
            if chunk.chunk_type == "child"
            and chunk.diagram_id == "차43-7(가)"
            and "사고 상황" in chunk.text
        ]
        variant_b_children = [
            chunk
            for chunk in chunks
            if chunk.chunk_type == "child"
            and chunk.diagram_id == "차43-7(나)"
            and "사고 상황" in chunk.text
        ]
        self.assertEqual(len(variant_a_children), 1)
        self.assertEqual(len(variant_b_children), 1)
        self.assertIn("벗어나기 전", variant_a_children[0].text)
        self.assertNotIn("벗어난 후", variant_a_children[0].text)
        self.assertIn("벗어난 후", variant_b_children[0].text)
        self.assertNotIn("벗어나기 전", variant_b_children[0].text)

        common_law_children = [
            chunk
            for chunk in chunks
            if chunk.chunk_type == "child" and "공통 법규" in chunk.text
        ]
        self.assertEqual(
            {chunk.diagram_id for chunk in common_law_children},
            {"차43-7(가)", "차43-7(나)"},
        )

    def test_splits_bulleted_variant_labeled_section_text_per_variant(self):
        document = Document(
            page_content=(
                "## 차43-7 안전지대 통과\n\n"
                "| 차43-7 (가) (나) | 직진 |\n"
                "| --- | --- |\n"
                "| (A) | 후행직진 |\n"
                "![table](../img/p389.png)\n\n"
                "### 사고 상황\n"
                "* ◉ **(가)** 안전지대 벗어나기 전 사고.\n"
                "* ◉ **(나)** 안전지대 벗어난 후 사고.\n\n"
                "## <u>관련 법규</u>\n"
                "* 공통 법규.\n"
            ),
            metadata={"page": 390, "source": "390.md"},
        )

        chunks = CaseBoundaryChunker(mode="B").chunk(document)

        variant_a_text = "\n".join(
            chunk.text
            for chunk in chunks
            if chunk.chunk_type == "child" and chunk.diagram_id == "차43-7(가)"
        )
        variant_b_text = "\n".join(
            chunk.text
            for chunk in chunks
            if chunk.chunk_type == "child" and chunk.diagram_id == "차43-7(나)"
        )
        self.assertIn("벗어나기 전", variant_a_text)
        self.assertNotIn("벗어난 후", variant_a_text)
        self.assertIn("벗어난 후", variant_b_text)
        self.assertNotIn("벗어나기 전", variant_b_text)
        self.assertTrue(
            any(
                chunk.chunk_type == "child"
                and chunk.diagram_id == "차43-7(가)"
                and "관련 법규" in chunk.text
                for chunk in chunks
            )
        )


class CaseBoundaryChunkerGeneralAndPrefaceTest(unittest.TestCase):
    def test_emits_general_chunks_for_each_subsection_under_top_heading(self):
        document = Document(
            page_content=(
                "# 1. 과실비율 인정기준의 필요성\n\n"
                "### (1) 신속한 보상처리\n"
                "보상의 신속성에 대한 설명입니다.\n\n"
                "### (2) 분쟁 해결\n"
                "분쟁 해결에 대한 설명입니다.\n"
            ),
            metadata={"page": 13, "source": "013.md"},
        )

        chunks = CaseBoundaryChunker(mode="B").chunk(document)

        self.assertEqual([chunk.chunk_type for chunk in chunks], ["general", "general"])
        self.assertTrue(all(chunk.diagram_id is None for chunk in chunks))
        self.assertTrue(all(chunk.parent_id is None for chunk in chunks))
        self.assertIn("# 1. 과실비율 인정기준의 필요성", chunks[0].text)
        self.assertIn("### (1) 신속한 보상처리", chunks[0].text)
        self.assertIn("보상의 신속성", chunks[0].text)
        self.assertIn("# 1. 과실비율 인정기준의 필요성", chunks[1].text)
        self.assertIn("### (2) 분쟁 해결", chunks[1].text)
        self.assertIn("분쟁 해결", chunks[1].text)

    def test_emits_preface_chunk_for_front_matter(self):
        document = Document(
            page_content=(
                "# 발간사\n\n"
                "이 책은 과실비율 인정기준을 다룹니다.\n"
            ),
            metadata={"page": 3, "source": "003.md"},
        )

        chunks = CaseBoundaryChunker(mode="B").chunk(document)

        self.assertEqual([chunk.chunk_type for chunk in chunks], ["preface"])
        self.assertIsNone(chunks[0].diagram_id)
        self.assertIsNone(chunks[0].parent_id)
        self.assertIsNone(chunks[0].image_path)
        self.assertIn("발간사", chunks[0].text)
        self.assertIn("이 책은", chunks[0].text)

    def test_splits_long_detailed_application_examples_by_circled_number(self):
        document = Document(
            page_content=(
                "# 5. 인적 손해에서의 과실상계 별도적용기준\n\n"
                "### (4) 세부적용 예\n"
                "**① 보호자의 자녀감호 태만**\n"
                "첫 예시입니다.\n\n"
                "**② 안전벨트 미착용**\n"
                "둘째 예시입니다.\n"
            ),
            metadata={"page": 24, "source": "024.md"},
        )

        chunks = CaseBoundaryChunker(mode="B").chunk(document)

        self.assertEqual([chunk.chunk_type for chunk in chunks], ["general", "general"])
        self.assertIn("# 5. 인적 손해에서의 과실상계 별도적용기준", chunks[0].text)
        self.assertIn("### (4) 세부적용 예", chunks[0].text)
        self.assertIn("① 보호자의 자녀감호 태만", chunks[0].text)
        self.assertNotIn("② 안전벨트 미착용", chunks[0].text)
        self.assertIn("# 5. 인적 손해에서의 과실상계 별도적용기준", chunks[1].text)
        self.assertIn("### (4) 세부적용 예", chunks[1].text)
        self.assertIn("② 안전벨트 미착용", chunks[1].text)

    def test_reference_table_with_case_ids_remains_general(self):
        document = Document(
            page_content=(
                "### ※(참고) 12대 중과실 사고 관련\n\n"
                "| 연번 | 내용 | 과실비율 반영 예시<br/>기준 | 과실비율 반영 예시<br/>수정요소 |\n"
                "| -- | -- | -- | -- |\n"
                "| 1 | 신호위반 등 | 차1-1, 차2-1, 차3-1, 차11-1, 2, 4, 5, 차43-7 기준 등<br/>▶ 기본 0:100 | |\n"
                "| 2 | 중앙선 침범 | 차31-1, 2, 3 등<br/>▶ 기본 0:100 등 | |\n"
            ),
            metadata={"page": 133, "source": "133.md"},
        )

        chunks = CaseBoundaryChunker(mode="B").chunk(document)

        self.assertEqual([chunk.chunk_type for chunk in chunks], ["general"])
        self.assertIsNone(chunks[0].diagram_id)
        self.assertIn("12대 중과실 사고 관련", chunks[0].text)
        self.assertIn("차1-1, 차2-1", chunks[0].text)


if __name__ == "__main__":
    unittest.main()
