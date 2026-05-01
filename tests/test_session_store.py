import unittest

from rag.service.intake.schema import IntakeState, UserSearchMetadata
from rag.service.session.memory_store import MemoryConversationStore
from rag.service.session.schema import ChatMessage
from rag.service.session.serialization import (
    intake_state_from_dict,
    intake_state_to_dict,
    json_dumps,
    json_loads,
    message_from_dict,
    message_to_dict,
)


class SessionSerializationTest(unittest.TestCase):
    def test_chat_message_round_trips_korean_text(self):
        message = ChatMessage(role="user", content="무단횡단 사고가 났어요")

        restored = message_from_dict(json_loads(json_dumps(message_to_dict(message))))

        self.assertEqual(restored, message)

    def test_chat_message_round_trips_metadata(self):
        message = ChatMessage(
            role="assistant",
            content="답변",
            metadata={"fault_ratio_a": 30, "fault_ratio_b": 70},
        )

        restored = message_from_dict(json_loads(json_dumps(message_to_dict(message))))

        self.assertEqual(restored, message)

    def test_intake_state_round_trips_nested_metadata(self):
        state = IntakeState(
            attempt_count=2,
            search_metadata=UserSearchMetadata(
                party_type="보행자",
                location="횡단보도 없음",
            ),
            last_missing_fields=["location"],
            last_follow_up_questions=["사고 장소는 어디인가요?"],
        )

        restored = intake_state_from_dict(
            json_loads(json_dumps(intake_state_to_dict(state)))
        )

        self.assertEqual(restored, state)


class MemoryConversationStoreTest(unittest.TestCase):
    def test_create_session_and_set_active_session(self):
        store = MemoryConversationStore()

        session = store.create_session("local", title="테스트 세션")
        store.set_active_session("local", session.session_id)

        self.assertEqual(store.get_active_session("local"), session.session_id)
        self.assertEqual(store.list_sessions("local"), [session])

    def test_append_messages_preserves_order(self):
        store = MemoryConversationStore()
        session = store.create_session("local")

        store.append_message("local", session.session_id, "user", "질문")
        store.append_message(
            "local",
            session.session_id,
            "assistant",
            "답변",
            metadata={"fault_ratio_a": 30, "fault_ratio_b": 70},
        )

        self.assertEqual(
            store.get_messages("local", session.session_id),
            [
                ChatMessage(role="user", content="질문"),
                ChatMessage(
                    role="assistant",
                    content="답변",
                    metadata={"fault_ratio_a": 30, "fault_ratio_b": 70},
                ),
            ],
        )

    def test_intake_state_is_session_scoped(self):
        store = MemoryConversationStore()
        session = store.create_session("local")
        state = IntakeState(
            attempt_count=1,
            search_metadata=UserSearchMetadata(party_type="자동차"),
        )

        store.set_intake_state("local", session.session_id, state)

        self.assertEqual(store.get_intake_state("local", session.session_id), state)

    def test_loader_strategy_defaults_and_updates(self):
        store = MemoryConversationStore()

        self.assertEqual(store.get_loader_strategy("local"), "pdfplumber")
        store.set_loader_strategy("local", "llamaparser")

        self.assertEqual(store.get_loader_strategy("local"), "llamaparser")


if __name__ == "__main__":
    unittest.main()
