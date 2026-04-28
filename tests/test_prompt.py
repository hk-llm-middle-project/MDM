import unittest

from langchain_core.messages import AIMessage, HumanMessage

from rag.service.prompt import CHAT_HISTORY_LIMIT, build_prompt, to_langchain_messages
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
            for index in range(CHAT_HISTORY_LIMIT + 2)
        ]

        messages = to_langchain_messages(history)

        self.assertEqual(len(messages), CHAT_HISTORY_LIMIT)
        self.assertEqual(messages[0].content, "질문 2")

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


if __name__ == "__main__":
    unittest.main()
