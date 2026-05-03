import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.service.analysis.answer_schema import AnalysisResult, RetrievedContext
from rag.service.conversation.orchestrator import AnswerResult
from rag.service.conversation.schema import RouteDecision, RouteType, TurnResultType
from rag.service.intake.schema import IntakeDecision, IntakeState, MissingField, UserSearchMetadata


def load_decision_eval_module():
    module_path = Path(__file__).resolve().parents[1] / "evaluation" / "evaluate_decision_suites.py"
    spec = importlib.util.spec_from_file_location("evaluate_decision_suites", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load evaluate_decision_suites.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DecisionSuiteEvalTest(unittest.TestCase):
    def test_evaluate_intake_rows_scores_metadata_missing_fields_and_forbidden_filters(self):
        module = load_decision_eval_module()

        def evaluator(question, **kwargs):
            return IntakeDecision(
                is_sufficient=False,
                normalized_description=question,
                search_metadata=UserSearchMetadata(party_type="보행자"),
                missing_fields=[MissingField(name="location", reason="missing")],
                follow_up_questions=["횡단보도 안인지, 부근인지 알려주세요."],
            )

        frame = module.evaluate_intake_rows(
            [
                {
                    "id": "intake_001",
                    "question": "보행자랑 사고났어요",
                    "case_type_codes": ["INTAKE_FOLLOWUP"],
                    "expected_party_type": "보행자",
                    "expected_location": None,
                    "expected_is_sufficient": False,
                    "expected_missing_fields": ["location"],
                    "expected_follow_up_questions_contain": ["횡단보도"],
                    "forbidden_filters": [{"location": "횡단보도 내"}],
                    "difficulty": "easy",
                }
            ],
            evaluator=evaluator,
        )

        self.assertEqual(frame.loc[0, "evaluation_suite"], "intake")
        self.assertEqual(frame.loc[0, "feedback.intake_is_sufficient"], 1)
        self.assertEqual(frame.loc[0, "feedback.party_type_match"], 1)
        self.assertEqual(frame.loc[0, "feedback.location_match"], 1)
        self.assertEqual(frame.loc[0, "feedback.missing_fields_match"], 1)
        self.assertEqual(frame.loc[0, "feedback.follow_up_contains"], 1)
        self.assertEqual(frame.loc[0, "feedback.forbidden_filter_absent"], 1)
        self.assertEqual(frame.loc[0, "difficulty"], "easy")

    def test_evaluate_router_rows_scores_route_type(self):
        module = load_decision_eval_module()

        def router(question, chat_history, intake_state):
            return RouteDecision(
                route_type=RouteType.GENERAL_CHAT,
                confidence=0.91,
                reason="rephrase_previous_answer",
            )

        frame = module.evaluate_router_rows(
            [
                {
                    "id": "router_001",
                    "question": "표로 다시 정리해줘",
                    "expected_route_type": "general_chat",
                    "expected_reason_category": "rephrase_previous_answer",
                    "chat_history": [{"role": "assistant", "content": "기존 답변"}],
                    "intake_state": {"search_metadata": {"party_type": "보행자"}},
                }
            ],
            router=router,
        )

        self.assertEqual(frame.loc[0, "evaluation_suite"], "router")
        self.assertEqual(frame.loc[0, "outputs.route_type"], "general_chat")
        self.assertEqual(frame.loc[0, "feedback.route_type_match"], 1)
        self.assertEqual(frame.loc[0, "feedback.reason_category_match"], 1)

    def test_evaluate_metadata_filter_rows_scores_expected_filter_and_forbidden_fields(self):
        module = load_decision_eval_module()

        frame = module.evaluate_metadata_filter_rows(
            [
                {
                    "id": "filter_001",
                    "question": "보행자 사고",
                    "search_metadata": {"party_type": "보행자", "location": None},
                    "expected_filter": {"party_type": "보행자"},
                    "must_not_filter_fields": ["location"],
                    "expected_behavior": "partial_filter",
                }
            ]
        )

        self.assertEqual(frame.loc[0, "evaluation_suite"], "metadata_filter")
        self.assertEqual(frame.loc[0, "feedback.metadata_filter_match"], 1)
        self.assertEqual(frame.loc[0, "feedback.forbidden_filter_absent"], 1)

    def test_evaluate_multiturn_rows_scores_state_sequence_and_ready_turn(self):
        module = load_decision_eval_module()

        def turn_handler(question, **kwargs):
            state = kwargs.get("intake_state") or IntakeState()
            if question == "자전거랑 사고났어요":
                return AnswerResult(
                    answer="사고 장소를 알려주세요.",
                    needs_more_input=True,
                    intake_state=IntakeState(
                        attempt_count=1,
                        search_metadata=UserSearchMetadata(party_type="자전거"),
                        last_missing_fields=["location"],
                        last_follow_up_questions=["사고 장소를 알려주세요."],
                    ),
                    result_type=TurnResultType.ACCIDENT_FOLLOW_UP,
                )
            self.assertEqual(state.search_metadata.party_type, "자전거")
            return AnswerResult(
                answer="분석 준비 완료",
                intake_state=IntakeState(
                    search_metadata=UserSearchMetadata(
                        party_type="자전거",
                        location="교차로 사고",
                    )
                ),
                result_type=TurnResultType.ACCIDENT_RAG,
            )

        frame = module.evaluate_multiturn_rows(
            [
                {
                    "id": "mt_001",
                    "turns": [
                        {"role": "user", "content": "자전거랑 사고났어요"},
                        {"role": "assistant", "content": "사고 장소를 알려주세요."},
                        {"role": "user", "content": "교차로였어요"},
                    ],
                    "expected_state_after_each_turn": [
                        {
                            "search_metadata": {"party_type": "자전거", "location": None},
                            "last_missing_fields": ["location"],
                        },
                        {
                            "search_metadata": {"party_type": "자전거", "location": "교차로 사고"},
                            "last_missing_fields": [],
                        },
                    ],
                    "expected_questions_after_each_turn": [["사고 장소"], []],
                    "expected_final_metadata": {"party_type": "자전거", "location": "교차로 사고"},
                    "expected_final_result_type": "ready_for_analysis",
                    "expected_turns_to_ready": 2,
                }
            ],
            turn_handler=turn_handler,
        )

        self.assertEqual(frame.loc[0, "evaluation_suite"], "multiturn")
        self.assertEqual(frame.loc[0, "feedback.state_sequence_match"], 1)
        self.assertEqual(frame.loc[0, "feedback.followup_questions_match"], 1)
        self.assertEqual(frame.loc[0, "feedback.final_metadata_match"], 1)
        self.assertEqual(frame.loc[0, "feedback.final_result_type_match"], 1)
        self.assertEqual(frame.loc[0, "feedback.turns_to_ready_match"], 1)

    def test_evaluate_structured_output_rows_scores_ratios_evidence_and_diagram_hit(self):
        module = load_decision_eval_module()

        def analyzer(question, **kwargs):
            return AnalysisResult(
                response="차1-4 기준입니다. A50 B50 적색 직진 사고입니다.",
                contexts=["차1-4 적색 직진 A50 B50"],
                retrieved_contexts=[
                    RetrievedContext(content="차1-4", metadata={"diagram_id": "차1-4"})
                ],
                fault_ratio_a=50,
                fault_ratio_b=50,
            )

        frame = module.evaluate_structured_output_rows(
            [
                {
                    "id": "struct_001",
                    "question": "양쪽 적색 직진 사고",
                    "search_metadata": {"party_type": "자동차", "location": "교차로 사고"},
                    "expected_diagram_ids": ["차1-4"],
                    "expected_final_fault_ratio": {"a": 50, "b": 50},
                    "expected_party_roles": {
                        "a": "적색 직진 사고",
                        "b": "적색 직진 사고",
                    },
                    "expected_cannot_determine_reason": None,
                    "required_evidence": ["차1-4", "적색 직진", "A50", "B50"],
                }
            ],
            analyzer=analyzer,
        )

        self.assertEqual(frame.loc[0, "evaluation_suite"], "structured_output")
        self.assertEqual(frame.loc[0, "feedback.final_fault_ratio_match"], 1)
        self.assertEqual(frame.loc[0, "feedback.cannot_determine_match"], 1)
        self.assertEqual(frame.loc[0, "feedback.required_evidence_coverage"], 1.0)
        self.assertEqual(frame.loc[0, "feedback.party_role_coverage"], 1.0)
        self.assertEqual(frame.loc[0, "feedback.reference_diagram_hit"], 1)


if __name__ == "__main__":
    unittest.main()
