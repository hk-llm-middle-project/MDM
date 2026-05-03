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
    build_example_frame,
    build_metric_frame,
    build_summary_frame,
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
        self.assertEqual(frame.loc[0, "critical_error"], 1)
        self.assertEqual(frame.loc[0, "diagram_id_hit"], 0)

    def test_build_metric_frame_converts_summary_metrics_to_long_rows(self):
        summary_frame = pd.DataFrame(
            [
                {
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
            metric_frame.loc[metric_frame["metric"] == "critical_error", "score"].iloc[0],
            0.8,
        )

    def test_load_results_returns_empty_frames_for_missing_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing"

            results = load_results(missing)

        self.assertTrue(results.summary.empty)
        self.assertTrue(results.examples.empty)
        self.assertTrue(results.metrics.empty)


if __name__ == "__main__":
    unittest.main()
