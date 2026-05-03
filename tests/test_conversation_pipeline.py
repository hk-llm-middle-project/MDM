import unittest
from unittest.mock import MagicMock

from rag.service.analysis.answer_schema import AnalysisResult
from rag.service.conversation.orchestrator import answer_conversation_turn
from rag.service.conversation.router import route_conversation_turn
from rag.service.conversation.schema import RouteType, TurnResultType
from rag.service.intake.schema import IntakeDecision, IntakeState, QuerySlots, UserSearchMetadata
from rag.service.session.schema import ChatMessage


class FakeLLM:
    def __init__(self, content):
        self.content = content

    def invoke(self, prompt):
        self.prompt = prompt
        return MagicMock(content=self.content)


class ConversationPipelineTest(unittest.TestCase):
    def test_router_uses_llm_decision_for_general_chat(self):
        router_llm = FakeLLM(
            '{"route_type":"general_chat","confidence":0.91,"reason":"일반 설명 요청"}'
        )

        decision = route_conversation_turn(
            "방금 답변 쉽게 다시 말해줘",
            [ChatMessage(role="assistant", content="이전 답변")],
            IntakeState(),
            llm=router_llm,
        )

        self.assertEqual(decision.route_type, RouteType.GENERAL_CHAT)
        self.assertGreater(decision.confidence, 0.9)

    def test_router_falls_back_to_accident_analysis_on_invalid_json(self):
        decision = route_conversation_turn(
            "방금 답변 쉽게 다시 말해줘",
            [],
            IntakeState(),
            llm=FakeLLM("not json"),
        )

        self.assertEqual(decision.route_type, RouteType.ACCIDENT_ANALYSIS)

    def test_general_chat_route_skips_intake_and_analysis(self):
        router_llm = FakeLLM(
            '{"route_type":"general_chat","confidence":0.95,"reason":"일반 대화"}'
        )
        general_llm = FakeLLM("이전 답변을 쉽게 풀어 설명했습니다.")
        intake = MagicMock()
        analyzer = MagicMock()

        result = answer_conversation_turn(
            "방금 답변 쉽게 다시 말해줘",
            chat_history=[ChatMessage(role="assistant", content="기존 답변")],
            intake_evaluator=intake,
            analyzer=analyzer,
            router_llm=router_llm,
            general_chat_llm=general_llm,
        )

        self.assertEqual(result.result_type, TurnResultType.GENERAL_CHAT)
        self.assertEqual(result.answer, "이전 답변을 쉽게 풀어 설명했습니다.")
        intake.assert_not_called()
        analyzer.assert_not_called()

    def test_accident_analysis_follow_up_result_type(self):
        router_llm = FakeLLM(
            '{"route_type":"accident_analysis","confidence":0.97,"reason":"사고 분석"}'
        )
        intake = MagicMock(
            return_value=IntakeDecision(
                is_sufficient=False,
                normalized_description="나 사고냈어",
                follow_up_questions=["사고 대상은 보행자, 자동차, 자전거 중 무엇인가요?"],
            )
        )
        analyzer = MagicMock()

        result = answer_conversation_turn(
            "나 사고냈어",
            chat_history=[],
            intake_evaluator=intake,
            analyzer=analyzer,
            router_llm=router_llm,
        )

        self.assertEqual(result.result_type, TurnResultType.ACCIDENT_FOLLOW_UP)
        self.assertTrue(result.needs_more_input)
        analyzer.assert_not_called()

    def test_accident_analysis_rag_result_type_and_contextual_intake(self):
        router_llm = FakeLLM(
            '{"route_type":"accident_analysis","confidence":0.97,"reason":"보충 정보"}'
        )
        previous_state = IntakeState(
            attempt_count=1,
            search_metadata=UserSearchMetadata(party_type="자동차"),
            last_missing_fields=["location"],
        )
        chat_history = [ChatMessage(role="assistant", content="사고 장소가 어디인가요?")]
        intake = MagicMock(
            return_value=IntakeDecision(
                is_sufficient=True,
                normalized_description="횡단보도 사고",
                search_metadata=UserSearchMetadata(location="횡단보도 내"),
            )
        )
        analyzer = MagicMock(return_value=("answer", ["context"]))

        result = answer_conversation_turn(
            "횡단보도였어",
            chat_history=chat_history,
            intake_state=previous_state,
            intake_evaluator=intake,
            analyzer=analyzer,
            router_llm=router_llm,
        )

        self.assertEqual(result.result_type, TurnResultType.ACCIDENT_RAG)
        self.assertFalse(result.needs_more_input)
        intake.assert_called_once()
        self.assertIs(intake.call_args.kwargs["chat_history"], chat_history)
        self.assertEqual(intake.call_args.kwargs["previous_state"], previous_state)
        analyzer.assert_called_once()
        self.assertEqual(
            analyzer.call_args.kwargs["search_metadata"],
            UserSearchMetadata(party_type="자동차", location="횡단보도 내"),
        )

    def test_accident_analysis_preserves_fault_ratios(self):
        router_llm = FakeLLM(
            '{"route_type":"accident_analysis","confidence":0.97,"reason":"사고 분석"}'
        )
        intake = MagicMock(
            return_value=IntakeDecision(
                is_sufficient=True,
                normalized_description="주차장 사고",
                search_metadata=UserSearchMetadata(
                    party_type="자동차",
                    location="주차장",
                ),
            )
        )
        analyzer = MagicMock(
            return_value=AnalysisResult(
                response="answer",
                contexts=["context"],
                fault_ratio_a=30,
                fault_ratio_b=70,
            )
        )

        result = answer_conversation_turn(
            "주차장에서 부딪혔어",
            chat_history=[],
            intake_evaluator=intake,
            analyzer=analyzer,
            router_llm=router_llm,
        )

        self.assertEqual(result.answer, "answer")
        self.assertEqual(result.contexts, ["context"])
        self.assertEqual(result.fault_ratio_a, 30)
        self.assertEqual(result.fault_ratio_b, 70)

    def test_accident_analysis_asks_for_missing_query_slots(self):
        router_llm = FakeLLM(
            '{"route_type":"accident_analysis","confidence":0.97,"reason":"사고 분석"}'
        )
        intake = MagicMock(
            return_value=IntakeDecision(
                is_sufficient=True,
                normalized_description="교차로에서 사고났어",
                search_metadata=UserSearchMetadata(
                    party_type="자동차",
                    location="교차로 사고",
                    retrieval_query="양쪽 신호등 있는 교차로",
                    query_slots=QuerySlots(road_control="양쪽 신호등"),
                ),
            )
        )
        analyzer = MagicMock()

        result = answer_conversation_turn(
            "교차로에서 사고났어",
            chat_history=[],
            intake_evaluator=intake,
            analyzer=analyzer,
            router_llm=router_llm,
        )

        self.assertEqual(result.result_type, TurnResultType.ACCIDENT_FOLLOW_UP)
        self.assertTrue(result.needs_more_input)
        self.assertIn("두 차량의 신호", result.answer)
        self.assertIn("각각 직진", result.answer)
        self.assertEqual(result.intake_state.last_missing_fields[0], "query_slots.signals")
        analyzer.assert_not_called()

    def test_accident_analysis_runs_when_core_query_slots_are_present(self):
        router_llm = FakeLLM(
            '{"route_type":"accident_analysis","confidence":0.97,"reason":"사고 분석"}'
        )
        metadata = UserSearchMetadata(
            party_type="자동차",
            location="교차로 사고",
            retrieval_query="녹색직진 대 적색직진",
            query_slots=QuerySlots(
                road_control="양쪽 신호등",
                relation="상대차량이 측면에서 진입",
                a_signal="녹색",
                b_signal="적색",
                a_movement="직진",
                b_movement="직진",
            ),
        )
        intake = MagicMock(
            return_value=IntakeDecision(
                is_sufficient=True,
                normalized_description="정규화된 설명",
                search_metadata=metadata,
            )
        )
        analyzer = MagicMock(return_value=("answer", ["context"]))

        result = answer_conversation_turn(
            "교차로에서 A 녹색 직진, B 적색 직진",
            chat_history=[],
            intake_evaluator=intake,
            analyzer=analyzer,
            router_llm=router_llm,
        )

        self.assertEqual(result.result_type, TurnResultType.ACCIDENT_RAG)
        self.assertFalse(result.needs_more_input)
        analyzer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
