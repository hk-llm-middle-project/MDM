"""TDD coverage for local evaluation dashboard transforms.

These tests live under evaluation/dashboard to preserve the session edit boundary.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dashboard.loaders import discover_result_bundles
from evaluation.dashboard import transforms


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class DashboardTransformPlanTest(unittest.TestCase):
    def test_build_example_frame_derives_case_key_from_example_id_or_question(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(
                root / "run-a.summary.json",
                {
                    "run_name": "run-a",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                    "metrics": {"critical_error": 0.0},
                },
            )
            pd.DataFrame(
                [
                    {"example_id": "retrieval_001", "inputs.question": "q1"},
                    {"example_id": "", "inputs.question": "q2"},
                ]
            ).to_csv(root / "run-a.csv", index=False)

            frame = transforms.build_example_frame(discover_result_bundles(root))

        self.assertIn("case_key", frame.columns)
        self.assertEqual(frame.loc[0, "case_key"], "retrieval_001")
        self.assertTrue(frame.loc[1, "case_key"].startswith("question:"))
        self.assertEqual(len(frame.loc[1, "case_key"]), len("question:") + 12)

    def test_build_example_frame_adds_feedback_score_and_comment_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(
                root / "run-a.summary.json",
                {
                    "run_name": "run-a",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                    "metrics": {"critical_error": 1.0},
                },
            )
            pd.DataFrame(
                [
                    {
                        "inputs.question": "q1",
                        "feedback.critical_error": "1",
                        "feedback.critical_error.comment": "1 means critical failure",
                    }
                ]
            ).to_csv(root / "run-a.csv", index=False)

            frame = transforms.build_example_frame(discover_result_bundles(root))

        self.assertEqual(frame.loc[0, "critical_error"], 1)
        self.assertIn("critical_error_comment", frame.columns)
        self.assertEqual(
            frame.loc[0, "critical_error_comment"],
            "1 means critical failure",
        )

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

    def test_build_summary_frame_adds_run_label_with_retriever_and_reranker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(
                root / "run-a.summary.json",
                {
                    "run_name": "upstage-custom-bge",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                    "retriever_strategy": "ensemble_parent",
                    "reranker_strategy": "cross-encoder",
                    "metrics": {"critical_error": 0.0},
                },
            )

            frame = transforms.build_summary_frame(discover_result_bundles(root))

        self.assertEqual(
            frame.loc[0, "run_label"],
            "upstage-custom-bge / ensemble_parent / cross-encoder",
        )

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

    def test_filter_frame_can_filter_retriever_and_reranker_strategy(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "run_name": "a",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                    "retriever_strategy": "similarity",
                    "reranker_strategy": "none",
                },
                {
                    "run_name": "b",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                    "retriever_strategy": "ensemble_parent",
                    "reranker_strategy": "cross-encoder",
                },
            ]
        )

        filtered = transforms.filter_frame(
            frame,
            loader_strategy=["upstage"],
            chunker_strategy=["custom"],
            embedding_provider=["bge"],
            retriever_strategy=["ensemble_parent"],
            reranker_strategy=["cross-encoder"],
        )

        self.assertEqual(filtered["run_name"].tolist(), ["b"])

    def test_filter_failed_examples_treats_critical_error_as_bad_when_score_is_one(self) -> None:
        self.assertTrue(
            hasattr(transforms, "filter_failed_examples"),
            "filter_failed_examples must exist",
        )
        examples = pd.DataFrame(
            [
                {"case_key": "a", "critical_error": 1},
                {"case_key": "b", "critical_error": 0},
            ]
        )

        failed = transforms.filter_failed_examples(examples, "critical_error")

        self.assertEqual(failed["case_key"].tolist(), ["a"])

    def test_filter_failed_examples_treats_hit_metric_as_bad_when_score_below_one(self) -> None:
        self.assertTrue(
            hasattr(transforms, "filter_failed_examples"),
            "filter_failed_examples must exist",
        )
        examples = pd.DataFrame(
            [
                {"case_key": "a", "diagram_id_hit": 0},
                {"case_key": "b", "diagram_id_hit": 1},
            ]
        )

        failed = transforms.filter_failed_examples(examples, "diagram_id_hit")

        self.assertEqual(failed["case_key"].tolist(), ["a"])

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

    def test_rank_combinations_sorts_critical_error_ascending(self) -> None:
        self.assertTrue(
            hasattr(transforms, "rank_combinations"),
            "rank_combinations must exist",
        )
        summary = pd.DataFrame(
            [
                {"run_name": "bad", "critical_error": 1.0},
                {"run_name": "good", "critical_error": 0.0},
            ]
        )

        ranked = transforms.rank_combinations(summary, "critical_error")

        self.assertEqual(ranked["run_name"].tolist(), ["good", "bad"])

    def test_rank_combinations_sorts_hit_metrics_descending(self) -> None:
        self.assertTrue(
            hasattr(transforms, "rank_combinations"),
            "rank_combinations must exist",
        )
        summary = pd.DataFrame(
            [
                {"run_name": "bad", "diagram_id_hit": 0.0},
                {"run_name": "good", "diagram_id_hit": 1.0},
            ]
        )

        ranked = transforms.rank_combinations(summary, "diagram_id_hit")

        self.assertEqual(ranked["run_name"].tolist(), ["good", "bad"])

    def test_metric_comparison_defaults_to_one_selected_metric(self) -> None:
        from evaluation.dashboard.views import metric_comparison

        self.assertEqual(
            metric_comparison.default_metric_selection(
                ["retrieval_relevance", "critical_error", "router_overall"]
            ),
            ["critical_error"],
        )
        self.assertEqual(
            metric_comparison.default_metric_selection(
                ["router_overall", "metadata_filter_overall"]
            ),
            ["router_overall"],
        )

    def test_metric_comparison_caption_includes_metric_description(self) -> None:
        from evaluation.dashboard.views import metric_comparison

        critical_caption = metric_comparison.metric_caption("critical_error")
        router_caption = metric_comparison.metric_caption("router_overall")

        self.assertIn("critical_error", critical_caption)
        self.assertIn("낮을수록", critical_caption)
        self.assertIn("router_overall", router_caption)

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

    def test_build_failure_breakdown_groups_failed_rows_by_strategy_and_run(self) -> None:
        self.assertTrue(
            hasattr(transforms, "build_failure_breakdown"),
            "build_failure_breakdown must exist",
        )
        failed = pd.DataFrame(
            [
                {
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                    "run_name": "run-a",
                },
                {
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                    "run_name": "run-a",
                },
                {
                    "loader_strategy": "raw",
                    "chunker_strategy": "semantic",
                    "embedding_provider": "openai",
                    "run_name": "run-b",
                },
            ]
        )

        breakdown = transforms.build_failure_breakdown(failed)

        self.assertEqual(breakdown.loc[0, "failed_count"], 2)
        self.assertEqual(breakdown.loc[0, "run_name"], "run-a")
        self.assertEqual(breakdown.loc[1, "failed_count"], 1)

    def test_metric_descriptions_cover_failure_metrics(self) -> None:
        for metric in transforms.METRIC_COLUMNS:
            with self.subTest(metric=metric):
                description = transforms.describe_metric(metric)
                self.assertIn(metric, description)
                self.assertIn("0", description)
                self.assertIn("1", description)

    def test_app_tab_labels_match_dashboard_plan(self) -> None:
        from evaluation.dashboard import app

        self.assertEqual(
            app.TAB_LABELS,
            ["Overview", "Metrics", "Matrix", "Test Cases", "Failures", "Case Detail", "Compare"],
        )


if __name__ == "__main__":
    unittest.main()
