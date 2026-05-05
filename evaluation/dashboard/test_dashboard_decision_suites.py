"""Tests for dashboard decision-suite helpers."""

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


class DashboardDecisionSuiteTest(unittest.TestCase):
    def test_decision_suite_helpers_focus_on_non_retrieval_suites(self) -> None:
        from evaluation.dashboard.views import decision_suites

        examples = pd.DataFrame(
            [
                {"evaluation_suite": "retrieval", "case_key": "retrieval_001"},
                {"evaluation_suite": "intake", "case_key": "intake_001"},
                {"evaluation_suite": "router", "case_key": "router_001"},
            ]
        )

        filtered = decision_suites.decision_suite_examples(examples)

        self.assertEqual(filtered["evaluation_suite"].tolist(), ["intake", "router"])
        self.assertEqual(
            decision_suites.available_decision_suites(examples),
            ["intake", "router"],
        )

    def test_decision_suite_helpers_choose_suite_specific_default_metrics(self) -> None:
        from evaluation.dashboard.views import decision_suites

        self.assertEqual(decision_suites.default_metric_for_suite("intake"), "intake_overall")
        self.assertEqual(decision_suites.default_metric_for_suite("router"), "router_overall")
        self.assertEqual(
            decision_suites.available_metrics_for_suite(
                pd.DataFrame(columns=["intake_overall", "router_overall", "critical_error"]),
                "intake",
            ),
            ["intake_overall", "critical_error"],
        )


if __name__ == "__main__":
    unittest.main()
