"""Constants for decision-suite evaluation."""

from __future__ import annotations

from config import BASE_DIR


DEFAULT_TESTSET_DIR = BASE_DIR / "data" / "testsets" / "langsmith"
DEFAULT_OUTPUT_DIR = BASE_DIR / "evaluation" / "results" / "uncategorized"
SUITE_FILES = {
    "intake": "intake_eval.jsonl",
    "router": "router_eval.jsonl",
    "metadata_filter": "metadata_filter_eval.jsonl",
    "multiturn": "multiturn_eval.jsonl",
    "structured_output": "structured_output_eval.jsonl",
}
CASE_METADATA_FIELDS = (
    "case_type_codes",
    "difficulty",
    "case_family",
    "inference_type",
    "query_style",
    "conversation_phase",
    "filter_risk",
    "modifier_source",
)
