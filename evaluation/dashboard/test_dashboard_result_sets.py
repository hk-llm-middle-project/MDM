"""Tests for dashboard result-set discovery and app tabs."""

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
from evaluation.dashboard.loaders import discover_result_bundles, discover_result_sets
from evaluation.dashboard.test_dashboard_support import write_json


class DashboardResultSetTest(unittest.TestCase):
    def test_discover_result_sets_returns_single_selectable_folders_with_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "all").mkdir()
            (root / "final").mkdir()
            (root / "experiments" / "k5").mkdir(parents=True)
            write_json(root / "all" / "run-a.summary.json", {"run_name": "run-a", "metrics": {}})
            write_json(root / "final" / "run-b.summary.json", {"run_name": "run-b", "metrics": {}})
            write_json(root / "experiments" / "k5" / "run-c.summary.json", {"run_name": "run-c", "metrics": {}})
            (root / "empty").mkdir()

            result_sets = discover_result_sets(root)

        labels = [item.label for item in result_sets]
        self.assertEqual(labels, ["all", "experiments/k5", "final"])
        self.assertEqual([item.path.name for item in result_sets], ["all", "k5", "final"])

    def test_discover_result_sets_excludes_archive_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "all").mkdir()
            (root / "archive" / "tmp").mkdir(parents=True)
            write_json(root / "all" / "run-a.summary.json", {"run_name": "run-a", "metrics": {}})
            write_json(root / "archive" / "tmp" / "run-b.summary.json", {"run_name": "run-b", "metrics": {}})

            result_sets = discover_result_sets(root)

        self.assertEqual([item.label for item in result_sets], ["all"])

    def test_discover_result_sets_includes_root_when_root_has_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "final").mkdir()
            write_json(root / "run-a.summary.json", {"run_name": "run-a", "metrics": {}})
            write_json(root / "final" / "run-b.summary.json", {"run_name": "run-b", "metrics": {}})

            result_sets = discover_result_sets(root)

        self.assertEqual([item.label for item in result_sets], [".", "final"])


    def test_result_root_input_resolves_relative_to_project_root(self) -> None:
        from evaluation.dashboard import app

        self.assertEqual(
            app.resolve_result_root("evaluation/results"),
            (app.PROJECT_ROOT / "evaluation" / "results").resolve(),
        )
        self.assertEqual(
            app.resolve_result_root(""),
            app.DEFAULT_RESULT_ROOT.resolve(),
        )

    def test_result_root_input_keeps_absolute_paths(self) -> None:
        from evaluation.dashboard import app

        with tempfile.TemporaryDirectory() as temp_dir:
            absolute_root = Path(temp_dir)
            self.assertEqual(app.resolve_result_root(str(absolute_root)), absolute_root)


    def test_result_set_display_path_omits_common_result_root(self) -> None:
        from evaluation.dashboard import app

        result_root = app.PROJECT_ROOT / "evaluation" / "results"

        self.assertEqual(
            app.display_result_set_path(result_root / "all", result_root),
            "all",
        )
        self.assertEqual(
            app.display_result_set_path(result_root / "experiments" / "k5", result_root),
            "experiments/k5",
        )
        self.assertEqual(app.display_result_set_path(result_root, result_root), ".")

    def test_text_input_apply_instruction_is_hidden(self) -> None:
        from evaluation.dashboard import app

        self.assertIn("InputInstructions", app.HIDE_TEXT_INPUT_INSTRUCTIONS_CSS)
        self.assertIn("display: none", app.HIDE_TEXT_INPUT_INSTRUCTIONS_CSS)

    def test_app_tab_labels_match_dashboard_plan(self) -> None:
        from evaluation.dashboard import app

        self.assertEqual(
            app.TAB_LABELS,
            [
                "Overview",
                "Decision Suites",
                "Metrics",
                "Matrix",
                "Test Cases",
                "Failures",
                "Case Detail",
                "Compare",
            ],
        )


if __name__ == "__main__":
    unittest.main()
