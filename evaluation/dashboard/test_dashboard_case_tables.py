"""Tests for dashboard case-level table builders."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dashboard import transforms
from evaluation.dashboard.loaders import discover_result_bundles
from evaluation.dashboard.test_dashboard_support import write_json


class DashboardCaseTableTest(unittest.TestCase):
    def test_build_case_metric_matrix_pivots_cases_by_run(self) -> None:
        self.assertTrue(
            hasattr(transforms, "build_case_metric_matrix"),
            "build_case_metric_matrix must exist",
        )
        examples = pd.DataFrame(
            [
                {
                    "case_key": "retrieval_001",
                    "inputs.question": "q1",
                    "run_name": "upstage-custom-bge",
                    "critical_error": 0,
                },
                {
                    "case_key": "retrieval_001",
                    "inputs.question": "q1",
                    "run_name": "upstage-raw-bge",
                    "critical_error": 1,
                },
            ]
        )

        matrix = transforms.build_case_metric_matrix(examples, metric="critical_error")

        self.assertEqual(matrix.loc[0, "upstage-custom-bge"], 0)
        self.assertEqual(matrix.loc[0, "upstage-raw-bge"], 1)

    def test_case_metric_matrix_uses_run_label_to_avoid_strategy_collisions(self) -> None:
        examples = pd.DataFrame(
            [
                {
                    "case_key": "retrieval_001",
                    "inputs.question": "q1",
                    "run_name": "upstage-custom-bge",
                    "run_label": "upstage-custom-bge / similarity / none",
                    "critical_error": 0,
                },
                {
                    "case_key": "retrieval_001",
                    "inputs.question": "q1",
                    "run_name": "upstage-custom-bge",
                    "run_label": "upstage-custom-bge / ensemble_parent / cross-encoder",
                    "critical_error": 1,
                },
            ]
        )

        matrix = transforms.build_case_metric_matrix(examples, metric="critical_error")

        self.assertEqual(matrix.loc[0, "upstage-custom-bge / similarity / none"], 0)
        self.assertEqual(
            matrix.loc[0, "upstage-custom-bge / ensemble_parent / cross-encoder"],
            1,
        )

    def test_rows_for_case_returns_all_runs_for_one_case(self) -> None:
        self.assertTrue(hasattr(transforms, "rows_for_case"), "rows_for_case must exist")
        examples = pd.DataFrame(
            [
                {"case_key": "retrieval_001", "run_name": "a"},
                {"case_key": "retrieval_001", "run_name": "b"},
                {"case_key": "retrieval_002", "run_name": "c"},
            ]
        )

        rows = transforms.rows_for_case(examples, "retrieval_001")

        self.assertEqual(rows["run_name"].tolist(), ["a", "b"])

    def test_compare_runs_for_case_returns_two_rows_for_selected_runs(self) -> None:
        self.assertTrue(
            hasattr(transforms, "compare_runs_for_case"),
            "compare_runs_for_case must exist",
        )
        examples = pd.DataFrame(
            [
                {"case_key": "retrieval_001", "run_name": "run-a", "critical_error": 0},
                {"case_key": "retrieval_001", "run_name": "run-b", "critical_error": 1},
                {"case_key": "retrieval_001", "run_name": "run-c", "critical_error": 0},
            ]
        )

        rows = transforms.compare_runs_for_case(examples, "retrieval_001", "run-a", "run-b")

        self.assertEqual(rows["run_name"].tolist(), ["run-a", "run-b"])

    def test_compare_runs_for_case_accepts_three_selected_runs(self) -> None:
        examples = pd.DataFrame(
            [
                {"case_key": "retrieval_001", "run_name": "run-a", "critical_error": 0},
                {"case_key": "retrieval_001", "run_name": "run-b", "critical_error": 1},
                {"case_key": "retrieval_001", "run_name": "run-c", "critical_error": 0},
                {"case_key": "retrieval_001", "run_name": "run-d", "critical_error": 1},
            ]
        )

        rows = transforms.compare_runs_for_case(
            examples,
            "retrieval_001",
            "run-c",
            "run-a",
            "run-b",
        )

        self.assertEqual(rows["run_name"].tolist(), ["run-c", "run-a", "run-b"])

    def test_case_value_comparison_shows_expected_and_actual_values_by_run(self) -> None:
        examples = pd.DataFrame(
            [
                {
                    "case_key": "retrieval_001",
                    "run_label": "run-a / ensemble_parent / none",
                    "reference.expected_diagram_ids": '["보1"]',
                    "outputs.retrieved_metadata": '[{"diagram_id":"보1","location":"횡단보도 내"}]',
                },
                {
                    "case_key": "retrieval_001",
                    "run_label": "run-b / ensemble_parent / cross-encoder",
                    "reference.expected_diagram_ids": '["보1"]',
                    "outputs.retrieved_metadata": '[{"diagram_id":"보10","location":"교차로"}]',
                },
            ]
        )

        comparison = transforms.build_case_value_comparison(examples)

        self.assertEqual(
            comparison.columns.tolist(),
            [
                "항목",
                "예상 값",
                "run-a / ensemble_parent / none",
                "run-b / ensemble_parent / cross-encoder",
            ],
        )
        self.assertEqual(comparison.loc[0, "항목"], "기대 diagram")
        self.assertEqual(comparison.loc[0, "예상 값"], "보1")
        self.assertEqual(comparison.loc[0, "run-a / ensemble_parent / none"], "보1")
        self.assertEqual(comparison.loc[0, "run-b / ensemble_parent / cross-encoder"], "보10")

    def test_case_value_comparison_falls_back_when_keyword_comment_is_nan(self) -> None:
        examples = pd.DataFrame(
            [
                {
                    "case_key": "retrieval_001",
                    "run_label": "run-a",
                    "reference.expected_keywords": '["교차로"]',
                    "keyword_coverage_comment": float("nan"),
                    "feedback.keyword_coverage.comment": "fallback keyword detail",
                }
            ]
        )

        comparison = transforms.build_case_value_comparison(examples)

        self.assertEqual(comparison.loc[0, "항목"], "기대 keyword")
        self.assertEqual(comparison.loc[0, "run-a"], "fallback keyword detail")

    def test_case_metric_comparison_shows_scores_by_run(self) -> None:
        examples = pd.DataFrame(
            [
                {
                    "case_key": "retrieval_001",
                    "run_label": "run-a / ensemble_parent / none",
                    "diagram_id_hit": 1,
                    "diagram_id_hit_comment": "expected_or_acceptable=['보1'], actual_topk=['보1']",
                },
                {
                    "case_key": "retrieval_001",
                    "run_label": "run-b / ensemble_parent / cross-encoder",
                    "diagram_id_hit": 0,
                    "diagram_id_hit_comment": "expected_or_acceptable=['보1'], actual_topk=['보10']",
                },
            ]
        )

        comparison = transforms.build_case_metric_comparison(examples)

        self.assertEqual(comparison.loc[0, "metric"], "diagram_id_hit")
        self.assertEqual(comparison.loc[0, "run-a / ensemble_parent / none"], "1")
        self.assertEqual(comparison.loc[0, "run-b / ensemble_parent / cross-encoder"], "0")
        self.assertIn("actual_topk=['보10']", comparison.loc[0, "run-b / ensemble_parent / cross-encoder comment"])

    def test_same_question_with_different_example_ids_stays_one_matrix_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for run_name, example_id, score in [
                ("run-a", "export-a-001", 0),
                ("run-b", "export-b-999", 1),
            ]:
                write_json(
                    root / f"{run_name}.summary.json",
                    {
                        "run_name": run_name,
                        "loader_strategy": "upstage",
                        "chunker_strategy": "custom",
                        "embedding_provider": "bge",
                        "metrics": {"critical_error": score},
                    },
                )
                pd.DataFrame(
                    [
                        {
                            "example_id": example_id,
                            "inputs.question": "same question",
                            "feedback.critical_error": score,
                        }
                    ]
                ).to_csv(root / f"{run_name}.csv", index=False)

            frame = transforms.build_example_frame(discover_result_bundles(root))
            matrix = transforms.build_case_metric_matrix(frame, metric="critical_error")

        self.assertEqual(frame["case_key"].nunique(), 1)
        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(matrix.loc[0, "run-a"], 0)
        self.assertEqual(matrix.loc[0, "run-b"], 1)

    def test_matrix_uses_stable_case_key_even_when_question_text_drifts(self) -> None:
        examples = pd.DataFrame(
            [
                {
                    "case_key": "retrieval_001",
                    "inputs.question": "same question",
                    "run_name": "run-a",
                    "critical_error": 0,
                },
                {
                    "case_key": "retrieval_001",
                    "inputs.question": "same question   ",
                    "run_name": "run-b",
                    "critical_error": 1,
                },
            ]
        )

        matrix = transforms.build_case_metric_matrix(examples, metric="critical_error")

        self.assertEqual(matrix.shape[0], 1)
        self.assertEqual(matrix.loc[0, "case_key"], "retrieval_001")
        self.assertEqual(matrix.loc[0, "run-a"], 0)
        self.assertEqual(matrix.loc[0, "run-b"], 1)

    def test_duplicate_question_with_stable_ids_keeps_distinct_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for run_name in ["run-a", "run-b"]:
                write_json(
                    root / f"{run_name}.summary.json",
                    {
                        "run_name": run_name,
                        "loader_strategy": "upstage",
                        "chunker_strategy": "custom",
                        "embedding_provider": "bge",
                        "metrics": {"critical_error": 0},
                    },
                )
                pd.DataFrame(
                    [
                        {
                            "example_id": "retrieval_001",
                            "inputs.question": "duplicate wording",
                            "feedback.critical_error": 0,
                        },
                        {
                            "example_id": "retrieval_002",
                            "inputs.question": "duplicate wording",
                            "feedback.critical_error": 1,
                        },
                    ]
                ).to_csv(root / f"{run_name}.csv", index=False)

            frame = transforms.build_example_frame(discover_result_bundles(root))
            matrix = transforms.build_case_metric_matrix(frame, metric="critical_error")

        self.assertEqual(set(frame["case_key"]), {"retrieval_001", "retrieval_002"})
        self.assertEqual(matrix.shape[0], 2)

    def test_build_expected_actual_case_table_shows_query_expected_and_actual_for_run(self) -> None:
        self.assertTrue(
            hasattr(transforms, "build_expected_actual_case_table"),
            "build_expected_actual_case_table must exist",
        )
        examples = pd.DataFrame(
            [
                {
                    "testset_label": "intake",
                    "case_key": "intake_001",
                    "inputs.question": "보행자 사고",
                    "run_label": "intake-local",
                    "reference.expected_party_type": "보행자",
                    "reference.expected_location": "횡단보도 내",
                    "outputs.party_type": "보행자",
                    "outputs.location": "횡단보도 부근",
                    "intake_overall": 0.5,
                },
                {
                    "testset_label": "intake",
                    "case_key": "intake_002",
                    "inputs.question": "자동차 사고",
                    "run_label": "other-run",
                    "reference.expected_party_type": "자동차",
                    "outputs.party_type": "자동차",
                    "intake_overall": 1,
                },
            ]
        )

        table = transforms.build_expected_actual_case_table(
            examples,
            testset_label="intake",
            run_label="intake-local",
        )

        self.assertEqual(table.columns[:5].tolist(), ["query", "정답 location", "모델 location", "정답 party_type", "모델 party_type"])
        self.assertEqual(table.shape[0], 1)
        self.assertEqual(table.loc[0, "query"], "보행자 사고")
        self.assertEqual(table.loc[0, "정답 party_type"], "보행자")
        self.assertEqual(table.loc[0, "정답 location"], "횡단보도 내")
        self.assertEqual(table.loc[0, "모델 party_type"], "보행자")
        self.assertEqual(table.loc[0, "모델 location"], "횡단보도 부근")
        self.assertEqual(table.loc[0, "case_key"], "intake_001")
        self.assertEqual(table.loc[0, "intake_overall"], 0.5)

    def test_expected_actual_case_table_pretty_prints_intake_values(self) -> None:
        examples = pd.DataFrame(
            [
                {
                    "testset_label": "intake",
                    "case_key": "intake_001",
                    "inputs.question": "보행자 사고",
                    "run_label": "intake-local",
                    "reference.expected_party_type": "보행자",
                    "reference.expected_location": "횡단보도 내",
                    "reference.expected_is_sufficient": True,
                    "reference.expected_missing_fields": "[]",
                    "outputs.party_type": "보행자",
                    "outputs.location": None,
                    "outputs.is_sufficient": False,
                    "outputs.missing_fields": '["location"]',
                    "intake_overall": 0.5,
                }
            ]
        )

        table = transforms.build_expected_actual_case_table(
            examples,
            testset_label="intake",
            run_label="intake-local",
        )

        self.assertEqual(table.loc[0, "정답 party_type"], "보행자")
        self.assertEqual(table.loc[0, "정답 location"], "횡단보도 내")
        self.assertEqual(table.loc[0, "정답 sufficient"], "예")
        self.assertEqual(table.loc[0, "정답 missing fields"], "없음")
        self.assertEqual(table.loc[0, "모델 location"], "-")
        self.assertEqual(table.loc[0, "모델 sufficient"], "아니오")
        self.assertEqual(table.loc[0, "모델 missing fields"], "location")
        self.assertNotIn("정답", table.loc[0, "정답 party_type"])

    def test_expected_actual_case_table_uses_pipe_separator_for_lists(self) -> None:
        examples = pd.DataFrame(
            [
                {
                    "testset_label": "intake",
                    "case_key": "intake_001",
                    "inputs.question": "자전거 사고",
                    "run_label": "intake-local",
                    "reference.expected_follow_up_questions_contain": '["사고 상황", "교차로"]',
                    "outputs.follow_up_questions": '["사고 장소는 어디인가요?", "교차로였나요?"]',
                }
            ]
        )

        table = transforms.build_expected_actual_case_table(
            examples,
            testset_label="intake",
            run_label="intake-local",
        )

        self.assertEqual(table.loc[0, "정답 follow-up 문구"], "사고 상황 | 교차로")
        self.assertEqual(
            table.loc[0, "모델 follow-up 문구"],
            "사고 장소는 어디인가요? | 교차로였나요?",
        )

    def test_expected_actual_case_table_styles_model_cells_by_match(self) -> None:
        self.assertTrue(
            hasattr(transforms, "build_expected_actual_case_table_styles"),
            "build_expected_actual_case_table_styles must exist",
        )
        table = pd.DataFrame(
            [
                {
                    "query": "q1",
                    "정답 location": "횡단보도 내",
                    "모델 location": "횡단보도 내",
                    "정답 party_type": "보행자",
                    "모델 party_type": "자동차",
                }
            ]
        )

        styles = transforms.build_expected_actual_case_table_styles(table)

        self.assertIn("background-color", styles.loc[0, "모델 location"])
        self.assertIn("#dcfce7", styles.loc[0, "모델 location"])
        self.assertIn("#dcfce7", styles.loc[0, "정답 location"])
        self.assertIn("#fee2e2", styles.loc[0, "모델 party_type"])
        self.assertIn("#fee2e2", styles.loc[0, "정답 party_type"])


if __name__ == "__main__":
    unittest.main()
