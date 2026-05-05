"""Compatibility entry point for all dashboard tests.

The concrete tests are split by dashboard responsibility. This file keeps the
historical `python evaluation/dashboard/test_dashboard_transforms.py` command
running the full dashboard suite.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dashboard import test_dashboard_case_tables
from evaluation.dashboard import test_dashboard_decision_suites
from evaluation.dashboard import test_dashboard_metric_views
from evaluation.dashboard import test_dashboard_result_sets
from evaluation.dashboard import test_dashboard_transforms_core


TEST_MODULES = (
    test_dashboard_transforms_core,
    test_dashboard_case_tables,
    test_dashboard_metric_views,
    test_dashboard_decision_suites,
    test_dashboard_result_sets,
)


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    if pattern is not None:
        return tests

    suite = unittest.TestSuite()
    for module in TEST_MODULES:
        suite.addTests(loader.loadTestsFromModule(module))
    return suite


if __name__ == "__main__":
    unittest.main()
