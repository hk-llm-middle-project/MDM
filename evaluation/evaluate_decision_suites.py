"""Evaluate non-retrieval LangSmith-style suites with local project functions.

Compatibility entry point for decision-suite evaluation. Reusable helpers live
under :mod:`evaluation.decision_eval`.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.decision_eval.cli import parse_args
from evaluation.decision_eval.common import (
    _base_record,
    _contains_all,
    _csv_cell,
    _diagram_ids_from_contexts,
    _filter_field_absent,
    _flatten_record,
    _forbidden_filters_absent,
    _intake_state,
    _mean,
    _messages,
    _metadata_dict,
    _missing_field_names,
    _normalize_analysis_result,
    _partial_match,
    _retrieved_metadata,
    _route_value,
    _score,
    _search_metadata,
)
from evaluation.decision_eval.constants import *  # noqa: F403 - compatibility re-export.
from evaluation.decision_eval.io import load_jsonl
from evaluation.decision_eval.results import (
    safe_filename,
    save_suite_dataframe,
    summarize_feedback_metrics,
)
from evaluation.decision_eval.runner import evaluate_suite
from evaluation.decision_eval.suites import (
    _list_coverage,
    _required_evidence_coverage,
    _role_coverage,
    _state_snapshot,
    _text_coverage,
    _turn_result_label,
    evaluate_intake_rows,
    evaluate_metadata_filter_rows,
    evaluate_multiturn_rows,
    evaluate_router_rows,
    evaluate_structured_output_rows,
)


def main() -> None:
    args = parse_args()
    suites = list(SUITE_FILES) if args.suite == "all" else [args.suite]
    for suite in suites:
        testset_path = args.testset_dir / SUITE_FILES[suite]
        rows = load_jsonl(testset_path, args.max_examples)
        dataframe = evaluate_suite(suite, rows)
        paths = save_suite_dataframe(
            dataframe=dataframe,
            suite=suite,
            output_dir=args.output_dir,
            testset_path=testset_path,
        )
        print(f"[INFO] {suite}: wrote {paths['csv']}")
        print(f"[INFO] {suite}: wrote {paths['summary_json']}")


if __name__ == "__main__":
    main()
