import unittest

from langchain_core.messages import AIMessage, HumanMessage

from rag.service.analysis.prompt import (
    ANSWER_HISTORY_LIMIT,
    GENERAL_CHAT_HISTORY_LIMIT,
    build_prompt,
    to_langchain_messages,
)
from rag.service.intake.prompts import INTAKE_HISTORY_LIMIT, build_intake_prompt
from rag.service.session.schema import ChatMessage


class PromptTest(unittest.TestCase):
    def test_to_langchain_messages_converts_roles(self):
        messages = to_langchain_messages(
            [
                ChatMessage(role="user", content="첫 질문"),
                ChatMessage(role="assistant", content="첫 답변"),
            ]
        )

        self.assertIsInstance(messages[0], HumanMessage)
        self.assertEqual(messages[0].content, "첫 질문")
        self.assertIsInstance(messages[1], AIMessage)
        self.assertEqual(messages[1].content, "첫 답변")

    def test_to_langchain_messages_limits_recent_history(self):
        history = [
            ChatMessage(role="user", content=f"질문 {index}")
            for index in range(GENERAL_CHAT_HISTORY_LIMIT + 2)
        ]

        messages = to_langchain_messages(history)

        self.assertEqual(len(messages), GENERAL_CHAT_HISTORY_LIMIT)
        self.assertEqual(messages[0].content, "질문 2")

    def test_build_prompt_uses_answer_specific_history_limit(self):
        history = [
            ChatMessage(role="user", content=f"질문 {index}")
            for index in range(ANSWER_HISTORY_LIMIT + 2)
        ]

        prompt_value = build_prompt(
            question="현재 질문",
            context="검색 문맥",
            chat_history=history,
        )
        messages = prompt_value.to_messages()

        history_messages = messages[1:-1]
        self.assertEqual(len(history_messages), ANSWER_HISTORY_LIMIT)
        self.assertEqual(history_messages[0].content, "질문 2")

    def test_build_prompt_places_history_before_current_question(self):
        prompt_value = build_prompt(
            question="현재 질문",
            context="검색 문맥",
            chat_history=[
                ChatMessage(role="user", content="이전 질문"),
                ChatMessage(role="assistant", content="이전 답변"),
            ],
        )
        messages = prompt_value.to_messages()

        self.assertEqual([type(message) for message in messages[1:]], [HumanMessage, AIMessage, HumanMessage])
        self.assertEqual(messages[1].content, "이전 질문")
        self.assertEqual(messages[2].content, "이전 답변")
        self.assertEqual(messages[3].content, "# Question\n현재 질문")

    def test_build_prompt_requires_three_markdown_response_sections(self):
        prompt_value = build_prompt(
            question="현재 질문",
            context="검색 문맥",
        )
        system_prompt = prompt_value.to_messages()[0].content

        self.assertIn("#### 사고 유형 및 근거", system_prompt)
        self.assertIn("#### 과실 판단", system_prompt)
        self.assertIn("#### 확인 필요 사항", system_prompt)
        self.assertIn("**굵게**", system_prompt)
        self.assertIn("표나 과도한 Markdown 장식은 사용하지 마세요", system_prompt)
        self.assertIn('"fault_ratio_a": number | null', system_prompt)
        self.assertIn("JSON 바깥에 설명", system_prompt)
        self.assertNotIn("#### 판단한 사고 상황", system_prompt)
        self.assertNotIn("#### 기본 과실", system_prompt)
        self.assertNotIn("#### 수정요소", system_prompt)
        self.assertNotIn("#### 최종 예상 과실", system_prompt)

    def test_build_intake_prompt_keeps_compressed_rule_sections(self):
        prompt = build_intake_prompt(user_input="양쪽 신호등 교차로에서 녹색 직진 대 적색 직진 사고")

        self.assertIn("사용자 입력과 이전 intake 상태를 바탕으로 검색용 사고 메타데이터를 추출하세요", prompt)
        self.assertIn("분류 우선순위:", prompt)
        self.assertIn('1. 교차로 내 신호, 좌우 도로, 직진/좌회전/우회전 관계가 핵심이면 "교차로 사고".', prompt)
        self.assertIn("query_slots 작성 규칙:", prompt)
        self.assertIn("retrieval_query 작성 규칙:", prompt)
        self.assertIn("confidence/follow-up 규칙:", prompt)
        self.assertIn("원문 요약이 아니라 문서 도표 제목에 가까운 검색어 조합", prompt)
        self.assertIn('"양쪽 신호등 있는 교차로, 직진 대 직진 사고, 상대차량이 측면에서 진입"', prompt)
        self.assertIn("JSON 외의 문장은 출력하지 마세요", prompt)
        self.assertNotIn("당신은 자동차 사고 설명에서 검색용 메타데이터를 추출하는 분류기입니다.", prompt)

    def test_build_intake_prompt_limits_recent_history(self):
        history = [
            ChatMessage(role="user", content=f"추가 설명 {index}")
            for index in range(INTAKE_HISTORY_LIMIT + 2)
        ]

        prompt = build_intake_prompt(
            user_input="현재 설명",
            chat_history=history,
        )

        self.assertNotIn("추가 설명 0", prompt)
        self.assertNotIn("추가 설명 1", prompt)
        self.assertIn("추가 설명 2", prompt)
        self.assertIn("추가 설명 5", prompt)


if __name__ == "__main__":
    unittest.main()
