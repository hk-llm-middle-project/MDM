"""Tests for dashboard transform table builders and filters."""

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


class DashboardTransformTest(unittest.TestCase):
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

    def test_build_summary_frame_preserves_result_nickname(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(
                root / "run-a.summary.json",
                {
                    "nickname": "Pro",
                    "run_name": "upstage-custom-bge",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                    "retriever_strategy": "ensemble_parent",
                    "reranker_strategy": "llm-score",
                    "metrics": {"critical_error": 0.0},
                },
            )
            pd.DataFrame(
                [{"inputs.question": "q1", "feedback.critical_error": 0}]
            ).to_csv(root / "run-a.csv", index=False)

            bundles = discover_result_bundles(root)
            summary = transforms.build_summary_frame(bundles)
            examples = transforms.build_example_frame(bundles)
            metrics = transforms.build_metric_frame(summary)

        self.assertEqual(summary.loc[0, "nickname"], "Pro")
        self.assertEqual(examples.loc[0, "nickname"], "Pro")
        self.assertEqual(metrics.loc[0, "nickname"], "Pro")

    def test_build_summary_frame_adds_ensemble_weight_to_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(
                root / "run-a.summary.json",
                {
                    "run_name": "upstage-custom-openai",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "openai",
                    "retriever_strategy": "ensemble_parent",
                    "reranker_strategy": "none",
                    "ensemble_bm25_weight": 0.7,
                    "ensemble_candidate_k": 20,
                    "ensemble_use_chunk_id": True,
                    "metrics": {"critical_error": 0.0},
                },
            )

            frame = transforms.build_summary_frame(discover_result_bundles(root))

        self.assertEqual(
            frame.loc[0, "run_label"],
            "upstage-custom-openai / ensemble_parent / none / BM25:Dense 7:3",
        )
        self.assertEqual(
            frame.loc[0, "retriever_reranker"],
            "ensemble_parent / none / BM25:Dense 7:3",
        )
        self.assertEqual(frame.loc[0, "ensemble_bm25_weight"], 0.7)

    def test_build_summary_frame_adds_intake_mode_to_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(
                root / "run-a.summary.json",
                {
                    "run_name": "upstage-custom-openai",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "openai",
                    "retriever_strategy": "ensemble",
                    "reranker_strategy": "none",
                    "retrieval_input_mode": "intake",
                    "ensemble_bm25_weight": 0.5,
                    "metrics": {"critical_error": 0.0},
                },
            )

            frame = transforms.build_summary_frame(discover_result_bundles(root))

        self.assertEqual(
            frame.loc[0, "retriever_reranker"],
            "ensemble / none / BM25:Dense 5:5 / intake",
        )
        self.assertEqual(
            frame.loc[0, "run_label"],
            "upstage-custom-openai / ensemble / none / BM25:Dense 5:5 / intake",
        )

    def test_build_summary_frame_formats_literal_two_to_nine_weight_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(
                root / "run-a.summary.json",
                {
                    "run_name": "upstage-custom-openai",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "openai",
                    "retriever_strategy": "ensemble",
                    "reranker_strategy": "none",
                    "ensemble_bm25_weight": 2 / 11,
                    "metrics": {"critical_error": 0.0},
                },
            )

            frame = transforms.build_summary_frame(discover_result_bundles(root))

        self.assertEqual(
            frame.loc[0, "retriever_reranker"],
            "ensemble / none / BM25:Dense 2:9",
        )

    def test_build_summary_frame_disambiguates_repeated_run_labels_by_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for timestamp in ("20260504-150000", "20260504-151500"):
                write_json(
                    root
                    / f"{timestamp}-upstage-custom-openai-ensemble-none.summary.json",
                    {
                        "run_name": "upstage-custom-openai",
                        "loader_strategy": "upstage",
                        "chunker_strategy": "custom",
                        "embedding_provider": "openai",
                        "retriever_strategy": "ensemble",
                        "reranker_strategy": "none",
                        "ensemble_bm25_weight": 0.7,
                        "metrics": {"critical_error": 0.0},
                    },
                )

            frame = transforms.build_summary_frame(discover_result_bundles(root))

        labels = frame["run_label"].tolist()
        self.assertEqual(len(set(labels)), 2)
        self.assertIn(
            "upstage-custom-openai / ensemble / none / BM25:Dense 7:3 [20260504-150000]",
            labels,
        )
        self.assertIn(
            "upstage-custom-openai / ensemble / none / BM25:Dense 7:3 [20260504-151500]",
            labels,
        )

    def test_build_summary_frame_averages_execution_time_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(
                root / "run-a.summary.json",
                {
                    "run_name": "upstage-custom-bge",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                    "metrics": {"critical_error": 0.0},
                },
            )
            pd.DataFrame(
                [
                    {"inputs.question": "q1", "execution_time": 1.5},
                    {"inputs.question": "q2", "execution_time": 2.5},
                ]
            ).to_csv(root / "run-a.csv", index=False)

            frame = transforms.build_summary_frame(discover_result_bundles(root))

        self.assertIn("execution_time", frame.columns)
        self.assertEqual(frame.loc[0, "execution_time"], 2.0)

    def test_build_metric_frame_includes_execution_time_for_comparison(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "run_label": "run-a / similarity / none",
                    "critical_error": 0.0,
                    "execution_time": 2.25,
                }
            ]
        )

        metrics = transforms.build_metric_frame(summary)

        self.assertIn("execution_time", metrics["metric"].tolist())
        execution_time = metrics[metrics["metric"] == "execution_time"].iloc[0]
        self.assertEqual(execution_time["score"], 2.25)

    def test_build_metric_frame_includes_short_retrieval_group_columns(self) -> None:
        summary = pd.DataFrame(
            [
                {
                    "run_label": "upstage-custom-bge / ensemble_parent / cross-encoder",
                    "retriever_strategy": "ensemble_parent",
                    "reranker_strategy": "cross-encoder",
                    "critical_error": 0.0,
                }
            ]
        )

        metrics = transforms.build_metric_frame(summary)
        row = metrics[metrics["metric"] == "critical_error"].iloc[0]

        self.assertEqual(row["retriever_strategy"], "ensemble_parent")
        self.assertEqual(row["reranker_strategy"], "cross-encoder")
        self.assertEqual(row["retriever_reranker"], "ensemble_parent / cross-encoder")

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

    def test_filter_frame_keeps_missing_case_family_when_all_values_selected(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "evaluation_suite": "intake",
                    "case_family": None,
                },
                {
                    "evaluation_suite": "structured_output",
                    "case_family": "car_intersection",
                },
                {
                    "evaluation_suite": "structured_output",
                    "case_family": "pedestrian_crosswalk",
                },
            ]
        )

        filtered = transforms.filter_frame(
            frame,
            loader_strategy=[],
            chunker_strategy=[],
            embedding_provider=[],
            case_family=["car_intersection", "pedestrian_crosswalk"],
        )

        self.assertEqual(
            filtered["evaluation_suite"].tolist(),
            ["intake", "structured_output", "structured_output"],
        )

    def test_filter_frame_filters_case_family_when_subset_selected(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "evaluation_suite": "intake",
                    "case_family": None,
                },
                {
                    "evaluation_suite": "structured_output",
                    "case_family": "car_intersection",
                },
                {
                    "evaluation_suite": "structured_output",
                    "case_family": "pedestrian_crosswalk",
                },
            ]
        )

        filtered = transforms.filter_frame(
            frame,
            loader_strategy=[],
            chunker_strategy=[],
            embedding_provider=[],
            case_family=["car_intersection"],
        )

        self.assertEqual(filtered["case_family"].tolist(), ["car_intersection"])

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

    def test_build_example_frame_adds_testset_label_from_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(
                root / "intake.summary.json",
                {
                    "run_name": "intake-local",
                    "evaluation_suite": "intake",
                    "dataset_name": "MDM intake testset",
                    "testset_path": "/tmp/intake_eval.jsonl",
                    "metrics": {"intake_overall": 1.0},
                },
            )
            pd.DataFrame(
                [
                    {
                        "example_id": "intake_001",
                        "inputs.question": "보행자 사고",
                        "feedback.intake_overall": 1,
                    }
                ]
            ).to_csv(root / "intake.csv", index=False)

            frame = transforms.build_example_frame(discover_result_bundles(root))

        self.assertEqual(frame.loc[0, "testset_label"], "intake")
        self.assertEqual(frame.loc[0, "dataset_name"], "MDM intake testset")
        self.assertEqual(frame.loc[0, "testset_path"], "/tmp/intake_eval.jsonl")


if __name__ == "__main__":
    unittest.main()
