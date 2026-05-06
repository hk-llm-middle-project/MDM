import json
import tempfile
import unittest
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dashboard.loaders import discover_result_bundles, load_results
from evaluation.dashboard.transforms import (
    build_failure_breakdown,
    build_example_frame,
    build_metric_frame,
    build_summary_frame,
    filter_frame,
)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class EvaluationDashboardTest(unittest.TestCase):
    def test_discovers_summary_files_and_matches_sibling_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(
                root / "20260502-120803-upstage-custom-bge.summary.json",
                {
                    "run_name": "upstage-custom-bge",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                    "row_count": 2,
                    "metrics": {"critical_error": 0.5},
                },
            )
            (root / "20260502-120803-upstage-custom-bge.csv").write_text(
                "inputs.question,feedback.critical_error\nq1,1\nq2,0\n",
                encoding="utf-8",
            )

            bundles = discover_result_bundles(root)

        self.assertEqual(len(bundles), 1)
        self.assertEqual(bundles[0].run_name, "upstage-custom-bge")
        self.assertEqual(bundles[0].csv_path.name, "20260502-120803-upstage-custom-bge.csv")

    def test_discover_result_bundles_skips_invalid_summary_with_warning_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(
                root / "valid.summary.json",
                {
                    "run_name": "valid-run",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                    "metrics": {"critical_error": 1.0},
                },
            )
            (root / "broken.summary.json").write_text("{not-json", encoding="utf-8")

            with self.assertLogs("evaluation.dashboard.loaders", level="WARNING") as logs:
                bundles = discover_result_bundles(root)

        self.assertEqual([bundle.run_name for bundle in bundles], ["valid-run"])
        self.assertTrue(
            any("Skipping invalid summary file broken.summary.json" in message for message in logs.output)
        )

    def test_load_results_warns_and_continues_when_summary_file_is_invalid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(
                root / "valid.summary.json",
                {
                    "run_name": "valid-run",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                    "row_count": 1,
                    "metrics": {"critical_error": 1.0},
                },
            )
            pd.DataFrame([{"inputs.question": "q1", "feedback.critical_error": 1}]).to_csv(
                root / "valid.csv",
                index=False,
            )
            (root / "broken.summary.json").write_text("{not-json", encoding="utf-8")

            with self.assertLogs("evaluation.dashboard.loaders", level="WARNING"):
                results = load_results(root)

        self.assertEqual(results.summary.loc[0, "run_name"], "valid-run")
        self.assertEqual(len(results.examples), 1)
        self.assertTrue(
            any("Skipping invalid summary file broken.summary.json" in warning for warning in results.warnings)
        )

    def test_build_summary_frame_flattens_metrics_and_keeps_combo_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(
                root / "run-a.summary.json",
                {
                    "experiment_name": "exp-a",
                    "dataset_name": "dataset-a",
                    "run_name": "upstage-custom-bge",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                    "row_count": 30,
                    "metrics": {
                        "diagram_id_hit": 0.2,
                        "critical_error": 0.8,
                    },
                },
            )

            frame = build_summary_frame(discover_result_bundles(root))

        self.assertEqual(frame.loc[0, "run_name"], "upstage-custom-bge")
        self.assertEqual(frame.loc[0, "combo"], "upstage / custom / bge")
        self.assertEqual(frame.loc[0, "diagram_id_hit"], 0.2)
        self.assertEqual(frame.loc[0, "critical_error"], 0.8)

    def test_build_example_frame_adds_run_metadata_and_feedback_aliases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(
                root / "run-a.summary.json",
                {
                    "run_name": "upstage-custom-bge",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                    "row_count": 2,
                    "metrics": {"critical_error": 0.5},
                },
            )
            pd.DataFrame(
                [
                    {
                        "inputs.question": "q1",
                        "metadata.suite": "retrieval",
                        "metadata.case_type_codes": json.dumps(["RET_DIAGRAM"]),
                        "metadata.difficulty": "hard",
                        "metadata.case_family": "car_intersection",
                        "feedback.critical_error": 1,
                        "feedback.diagram_id_hit": 0,
                    },
                    {
                        "inputs.question": "q2",
                        "feedback.critical_error": 0,
                        "feedback.diagram_id_hit": 1,
                    },
                ]
            ).to_csv(root / "run-a.csv", index=False)

            frame = build_example_frame(discover_result_bundles(root))

        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.loc[0, "run_name"], "upstage-custom-bge")
        self.assertEqual(frame.loc[0, "evaluation_suite"], "retrieval")
        self.assertEqual(frame.loc[0, "critical_error"], 1)
        self.assertEqual(frame.loc[0, "diagram_id_hit"], 0)
        self.assertEqual(frame.loc[0, "case_type_codes"], json.dumps(["RET_DIAGRAM"]))
        self.assertEqual(frame.loc[0, "difficulty"], "hard")
        self.assertEqual(frame.loc[0, "case_family"], "car_intersection")

    def test_failure_breakdown_includes_case_metadata_when_available(self):
        failed = pd.DataFrame(
            [
                {
                    "run_name": "run-a",
                    "run_label": "run-a / vectorstore / none",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                    "retriever_strategy": "vectorstore",
                    "reranker_strategy": "none",
                    "case_type_codes": json.dumps(["RET_NEAR_MISS"]),
                    "difficulty": "hard",
                    "case_family": "car_intersection",
                },
                {
                    "run_name": "run-a",
                    "run_label": "run-a / vectorstore / none",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                    "retriever_strategy": "vectorstore",
                    "reranker_strategy": "none",
                    "case_type_codes": json.dumps(["RET_NEAR_MISS"]),
                    "difficulty": "hard",
                    "case_family": "car_intersection",
                },
            ]
        )

        breakdown = build_failure_breakdown(failed)

        self.assertIn("case_type_codes", breakdown.columns)
        self.assertIn("difficulty", breakdown.columns)
        self.assertIn("case_family", breakdown.columns)
        self.assertEqual(breakdown.loc[0, "failed_count"], 2)

    def test_build_metric_frame_converts_summary_metrics_to_long_rows(self):
        summary_frame = pd.DataFrame(
            [
                {
                    "evaluation_suite": "retrieval",
                    "run_name": "upstage-custom-bge",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                    "combo": "upstage / custom / bge",
                    "critical_error": 0.8,
                    "diagram_id_hit": 0.2,
                }
            ]
        )

        metric_frame = build_metric_frame(summary_frame)

        self.assertEqual(set(metric_frame["metric"]), {"critical_error", "diagram_id_hit"})
        self.assertEqual(
            metric_frame.loc[metric_frame["metric"] == "critical_error", "evaluation_suite"].iloc[0],
            "retrieval",
        )
        self.assertEqual(
            metric_frame.loc[metric_frame["metric"] == "critical_error", "score"].iloc[0],
            0.8,
        )

    def test_filter_frame_can_filter_suite_and_case_metadata(self):
        frame = pd.DataFrame(
            [
                {
                    "evaluation_suite": "retrieval",
                    "difficulty": "hard",
                    "case_family": "car_intersection",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                },
                {
                    "evaluation_suite": "intake",
                    "difficulty": "easy",
                    "case_family": "general_rule",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                },
            ]
        )

        filtered = filter_frame(
            frame,
            loader_strategy=["upstage"],
            chunker_strategy=["custom"],
            embedding_provider=["bge"],
            evaluation_suite=["retrieval"],
            difficulty=["hard"],
            case_family=["car_intersection"],
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.loc[0, "evaluation_suite"], "retrieval")

    def test_strategy_filters_keep_strategyless_decision_suite_rows(self):
        frame = pd.DataFrame(
            [
                {
                    "evaluation_suite": "retrieval",
                    "loader_strategy": "upstage",
                    "chunker_strategy": "custom",
                    "embedding_provider": "bge",
                },
                {
                    "evaluation_suite": "intake",
                    "loader_strategy": None,
                    "chunker_strategy": None,
                    "embedding_provider": None,
                },
            ]
        )

        filtered = filter_frame(
            frame,
            loader_strategy=["upstage"],
            chunker_strategy=["custom"],
            embedding_provider=["bge"],
        )

        self.assertEqual(set(filtered["evaluation_suite"]), {"retrieval", "intake"})

    def test_load_results_returns_empty_frames_for_missing_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing"

            results = load_results(missing)

        self.assertTrue(results.summary.empty)
        self.assertTrue(results.examples.empty)
        self.assertTrue(results.metrics.empty)


if __name__ == "__main__":
    unittest.main()
