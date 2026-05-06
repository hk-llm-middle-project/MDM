"""CLI parsing for decision-suite evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.decision_eval.constants import DEFAULT_OUTPUT_DIR, DEFAULT_TESTSET_DIR, SUITE_FILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local evaluation for intake/router/filter/multiturn/structured suites.",
    )
    parser.add_argument(
        "--suite",
        choices=tuple(SUITE_FILES) + ("all",),
        default="all",
        help="Suite to evaluate. Use all to run every non-retrieval suite.",
    )
    parser.add_argument(
        "--testset-dir",
        type=Path,
        default=DEFAULT_TESTSET_DIR,
        help="Directory containing LangSmith JSONL testsets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV/summary exports consumed by the dashboard.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Evaluate only the first N examples per suite. Set to 0 for all examples.",
    )
    return parser.parse_args()
