#!/usr/bin/env python3
"""Validate LangSmith evaluation JSONL testsets against the local guide contract.

This harness is intentionally dependency-free so it can run before the LangSmith
runner exists. It validates JSONL shape, required fields, case type codes, row
counts, diagram-id existence, and chunk-derived evidence keywords where diagram
labels are present.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SUITES = {
    "intake": {
        "file": "intake_eval.jsonl",
        "min": 20,
        "fields": {
            "id", "question", "case_type_codes", "expected_party_type", "expected_location",
            "expected_is_sufficient", "expected_missing_fields", "ambiguous_fields",
            "forbidden_filters", "expected_follow_up_questions_contain", "inference_type",
            "chat_history", "previous_state", "difficulty",
        },
        "prefix": "intake_",
        "behavior_prefixes": ("INTAKE_",),
    },
    "router": {
        "file": "router_eval.jsonl",
        "min": 15,
        "fields": {
            "id", "question", "case_type_codes", "expected_route_type", "chat_history",
            "intake_state", "expected_is_follow_up", "expected_reason_category",
            "conversation_phase", "requires_chat_history", "expected_should_preserve_state",
            "difficulty",
        },
        "prefix": "router_",
        "behavior_prefixes": ("ROUTE_",),
    },
    "metadata_filter": {
        "file": "metadata_filter_eval.jsonl",
        "min": 15,
        "fields": {
            "id", "question", "case_type_codes", "search_metadata", "expected_filter",
            "expected_behavior", "must_not_filter_fields", "expected_fallback_required",
            "filter_risk", "expected_unfiltered_diagram_ids", "expected_diagram_ids", "difficulty",
        },
        "prefix": "filter_",
        "behavior_prefixes": ("FILTER_",),
    },
    "retrieval": {
        "file": "retrieval_eval.jsonl",
        "min": 30,
        "fields": {
            "id", "question", "case_type_codes", "expected_diagram_ids",
            "acceptable_diagram_ids", "near_miss_diagram_ids", "expected_evidence_keywords",
            "requires_diagram", "requires_table", "inference_type", "query_style",
            "difficulty", "case_family",
        },
        "prefix": "retrieval_",
        "behavior_prefixes": ("RET_",),
    },
    "reranker": {
        "file": "reranker_eval.jsonl",
        "min": 20,
        "max": 30,
        "fields": {
            "id", "question", "case_type_codes", "expected_diagram_ids", "candidate_k",
            "final_k", "near_miss_diagram_ids", "difficulty", "case_family",
        },
        "prefix": "reranker_",
        "behavior_prefixes": ("RERANK_",),
    },
    "multiturn": {
        "file": "multiturn_eval.jsonl",
        "min": 5,
        "max": 8,
        "fields": {
            "id", "turns", "case_type_codes", "expected_final_metadata",
            "expected_final_result_type", "expected_turns_to_ready",
            "expected_state_after_each_turn", "expected_questions_after_each_turn",
            "expected_reset_turn_index", "expected_preserved_fields", "expected_overwritten_fields",
            "difficulty",
        },
        "prefix": "mt_",
        "behavior_prefixes": ("MT_",),
    },
    "structured_output": {
        "file": "structured_output_eval.jsonl",
        "min": 15,
        "fields": {
            "id", "question", "case_type_codes", "search_metadata", "expected_diagram_ids",
            "near_miss_diagram_ids", "expected_base_fault_ratio", "expected_final_fault_ratio",
            "expected_party_roles", "expected_applicable_modifiers",
            "expected_non_applicable_modifiers", "modifier_source", "expected_reference_diagram_ids",
            "expected_cannot_determine_reason", "required_evidence", "difficulty", "case_family",
        },
        "prefix": "struct_",
        "behavior_prefixes": ("STRUCT_",),
    },
}

CASE_CODES = {
    "CLARITY_EXPLICIT", "CLARITY_IMPLICIT", "CLARITY_SYNONYM", "CLARITY_INCOMPLETE", "CLARITY_DISTRACTOR",
    "META_PARTY", "META_LOCATION", "META_CROSSWALK_IN", "META_CROSSWALK_NEAR", "META_NO_CROSSWALK",
    "META_INTERSECTION", "META_SAME_DIRECTION", "META_OPPOSITE_DIRECTION", "META_MOTORCYCLE_SPECIAL", "META_OTHER",
    "INTAKE_FULL", "INTAKE_PARTIAL", "INTAKE_NONE", "INTAKE_FOLLOWUP",
    "ROUTE_NEW_ACCIDENT", "ROUTE_FOLLOWUP", "ROUTE_GENERAL", "ROUTE_CORRECTION",
    "FILTER_STRICT", "FILTER_PARTIAL", "FILTER_NONE", "FILTER_FALLBACK",
    "RET_DIAGRAM", "RET_TABLE", "RET_NEAR_MISS", "RET_MULTI_ACCEPT",
    "RERANK_PROMOTE", "RERANK_PROTECT",
    "MT_ACCUMULATE", "MT_CORRECTION", "MT_MAX_FOLLOWUP",
    "STRUCT_BASE_RATIO", "STRUCT_ROLE_REVERSAL", "STRUCT_MODIFIER_SAME", "STRUCT_MODIFIER_CROSS_REF",
    "STRUCT_NON_APPLICABLE_MODIFIER", "STRUCT_CANNOT_DETERMINE",
}

DIFFICULTIES = {"easy", "medium", "hard"}
INFERENCE_TYPES = {"explicit_keyword", "implicit_metadata", "synonym", "incomplete"}
QUERY_STYLES = {"formal", "natural", "colloquial", "incomplete"}
PARTY_TYPES = {None, "보행자", "자동차", "자전거"}
LOCATIONS = {
    None,
    "횡단보도 내", "횡단보도 부근", "횡단보도 없음", "교차로 사고",
    "같은 방향 진행차량 상호간의 사고", "마주보는 방향 진행차량 상호 간의 사고",
    "자동차 대 이륜차 특수유형", "기타",
}
LEGACY_STRUCTURED_FIELDS = {"expected_fault_ratio_a", "expected_fault_ratio_b", "expected_modifiers"}
DIAGRAM_FIELDS = (
    "expected_diagram_ids", "acceptable_diagram_ids", "near_miss_diagram_ids",
    "expected_reference_diagram_ids", "expected_unfiltered_diagram_ids",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), action="append", help="Suite to validate. Repeatable.")
    parser.add_argument("--data-dir", default="data/testsets/langsmith")
    parser.add_argument("--chunks", default="data/chunks/upstage/custom/chunks.json")
    parser.add_argument("--allow-short-counts", action="store_true")
    parser.add_argument("--no-evidence-check", action="store_true")
    return parser.parse_args()


def load_chunks(path: Path) -> tuple[set[str], dict[str, str]]:
    if not path.exists():
        raise ValueError(f"chunks file not found: {path}")
    chunks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise ValueError("chunks file must be a JSON list")
    diagrams: set[str] = set()
    text_by_diagram: dict[str, list[str]] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        metadata = chunk.get("metadata") or {}
        did = metadata.get("diagram_id")
        if isinstance(did, str) and did:
            diagrams.add(did)
            text_by_diagram.setdefault(did, []).append(str(chunk.get("page_content") or ""))
    return diagrams, {did: "\n".join(parts) for did, parts in text_by_diagram.items()}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path}:{index}: blank line")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{index}: invalid json: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index}: row must be object")
        rows.append(row)
    return rows


def normalize_text(value: str) -> str:
    normalized = re.sub(r"\s+", "", value)
    return re.sub(r"([AB])O(?=\D|$)", r"\g<1>0", normalized)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def ensure_type(row: dict[str, Any], field: str, expected_type: type | tuple[type, ...], suite: str) -> None:
    ensure(isinstance(row.get(field), expected_type), f"{suite}:{row.get('id')}: {field} must be {expected_type}")


def check_common(suite: str, row: dict[str, Any], spec: dict[str, Any]) -> None:
    row_id = row.get("id")
    ensure(isinstance(row_id, str) and row_id.startswith(spec["prefix"]), f"{suite}: invalid id {row_id!r}")
    keys = set(row)
    ensure(keys == spec["fields"], f"{suite}:{row_id}: schema mismatch missing={sorted(spec['fields']-keys)} extra={sorted(keys-spec['fields'])}")
    ensure(row.get("difficulty") in DIFFICULTIES, f"{suite}:{row_id}: invalid difficulty")
    codes = row.get("case_type_codes")
    ensure(isinstance(codes, list) and codes, f"{suite}:{row_id}: case_type_codes must be non-empty list")
    unknown = [code for code in codes if code not in CASE_CODES]
    ensure(not unknown, f"{suite}:{row_id}: unknown case_type_codes {unknown}")
    ensure(any(str(code).startswith("CLARITY_") for code in codes), f"{suite}:{row_id}: missing CLARITY_* code")
    ensure(any(any(str(code).startswith(prefix) for prefix in spec["behavior_prefixes"]) for code in codes), f"{suite}:{row_id}: missing suite behavior code")
    ensure(not (LEGACY_STRUCTURED_FIELDS & keys), f"{suite}:{row_id}: legacy structured fields are forbidden")


def check_diagrams(suite: str, row: dict[str, Any], diagrams: set[str], text_by_diagram: dict[str, str], evidence_check: bool) -> None:
    row_id = row["id"]
    diagram_values_by_field: dict[str, list[str]] = {}
    for field in DIAGRAM_FIELDS:
        values = row.get(field, []) or []
        ensure(isinstance(values, list), f"{suite}:{row_id}: {field} must be list")
        diagram_values_by_field[field] = values
        for did in values:
            ensure(did in diagrams, f"{suite}:{row_id}: unknown {field} diagram_id={did}")
    if not evidence_check:
        return

    expected_ids = diagram_values_by_field.get("expected_diagram_ids", [])
    if suite in {"retrieval", "reranker", "structured_output"}:
        missing_exact_ids = [
            did for did in expected_ids
            if normalize_text(did) not in normalize_text(text_by_diagram.get(did, ""))
        ]
        ensure(not missing_exact_ids, f"{suite}:{row_id}: expected diagram ids not found in their own chunk page_content: {missing_exact_ids}")

    keywords = row.get("expected_evidence_keywords") or row.get("required_evidence") or []
    evidence_diagrams = list(expected_ids) + diagram_values_by_field.get("acceptable_diagram_ids", []) + diagram_values_by_field.get("expected_reference_diagram_ids", [])
    if keywords and evidence_diagrams:
        ensure(isinstance(keywords, list), f"{suite}:{row_id}: evidence keywords must be list")
        haystack = "\n".join(text_by_diagram.get(did, "") for did in evidence_diagrams)
        normalized_haystack = normalize_text(haystack)
        missing = [kw for kw in keywords if str(kw) not in haystack and normalize_text(str(kw)) not in normalized_haystack]
        ensure(not missing, f"{suite}:{row_id}: evidence keywords not found in expected chunk page_content: {missing}")


def check_suite_specific(suite: str, row: dict[str, Any]) -> None:
    row_id = row["id"]
    if suite == "intake":
        ensure(row["expected_party_type"] in PARTY_TYPES, f"{suite}:{row_id}: invalid expected_party_type")
        ensure(row["expected_location"] in LOCATIONS, f"{suite}:{row_id}: invalid expected_location")
        ensure_type(row, "expected_is_sufficient", bool, suite)
        if (not row["expected_is_sufficient"]) and row.get("expected_follow_up_questions_contain"):
            ensure("INTAKE_FOLLOWUP" in row["case_type_codes"], f"{suite}:{row_id}: insufficient rows with follow-up expectations need INTAKE_FOLLOWUP")
        for field in ["expected_missing_fields", "ambiguous_fields", "forbidden_filters", "expected_follow_up_questions_contain", "chat_history"]:
            ensure_type(row, field, list, suite)
        ensure_type(row, "previous_state", dict, suite)
        ensure(row["inference_type"] in INFERENCE_TYPES, f"{suite}:{row_id}: invalid inference_type")
    elif suite == "router":
        ensure(row["expected_route_type"] in {"accident_analysis", "general_chat"}, f"{suite}:{row_id}: invalid route")
        ensure(row["expected_reason_category"] in {"new_accident_analysis", "accident_follow_up", "rephrase_previous_answer", "general_rule_question", "out_of_domain", "smalltalk", "correction"}, f"{suite}:{row_id}: invalid reason")
        ensure(row["conversation_phase"] in {"new_session", "after_follow_up_question", "after_rag_answer", "after_general_chat", "correction"}, f"{suite}:{row_id}: invalid phase")
        for field in ["chat_history"]:
            ensure_type(row, field, list, suite)
        ensure_type(row, "intake_state", dict, suite)
        for field in ["expected_is_follow_up", "requires_chat_history", "expected_should_preserve_state"]:
            ensure_type(row, field, bool, suite)
    elif suite == "metadata_filter":
        ensure(row["expected_behavior"] in {"filter", "partial_filter", "no_filter"}, f"{suite}:{row_id}: invalid behavior")
        ensure(row["filter_risk"] in {"low", "medium", "high"}, f"{suite}:{row_id}: invalid filter_risk")
        ensure_type(row, "search_metadata", dict, suite)
        ensure(row["expected_filter"] is None or isinstance(row["expected_filter"], dict), f"{suite}:{row_id}: expected_filter must be object or null")
        ensure_type(row, "must_not_filter_fields", list, suite)
        ensure_type(row, "expected_fallback_required", bool, suite)
        if row["expected_fallback_required"] or row["filter_risk"] == "high":
            ensure(row["expected_unfiltered_diagram_ids"], f"{suite}:{row_id}: high/fallback rows need expected_unfiltered_diagram_ids")
    elif suite == "retrieval":
        ensure(row["inference_type"] in INFERENCE_TYPES, f"{suite}:{row_id}: invalid inference_type")
        ensure(row["query_style"] in QUERY_STYLES, f"{suite}:{row_id}: invalid query_style")
        ensure_type(row, "requires_diagram", bool, suite)
        ensure_type(row, "requires_table", bool, suite)
        ensure_type(row, "expected_evidence_keywords", list, suite)
        ensure(2 <= len(row["expected_evidence_keywords"]) <= 5, f"{suite}:{row_id}: expected_evidence_keywords must have 2-5 items")
        if row["case_family"] != "general_rule":
            ensure(row["near_miss_diagram_ids"], f"{suite}:{row_id}: non-general retrieval row needs near_miss_diagram_ids")
        if row["near_miss_diagram_ids"]:
            ensure("RET_NEAR_MISS" in row["case_type_codes"], f"{suite}:{row_id}: rows with near_miss_diagram_ids need RET_NEAR_MISS")
        if row["acceptable_diagram_ids"]:
            ensure("RET_MULTI_ACCEPT" in row["case_type_codes"], f"{suite}:{row_id}: rows with acceptable_diagram_ids need RET_MULTI_ACCEPT")
    elif suite == "reranker":
        ensure(isinstance(row["candidate_k"], int) and isinstance(row["final_k"], int), f"{suite}:{row_id}: k fields must be ints")
        ensure(row["candidate_k"] > row["final_k"], f"{suite}:{row_id}: candidate_k must be > final_k")
        ensure(row["near_miss_diagram_ids"], f"{suite}:{row_id}: reranker near_miss_diagram_ids required")
        ensure("RET_NEAR_MISS" in row["case_type_codes"], f"{suite}:{row_id}: reranker needs RET_NEAR_MISS")
    elif suite == "multiturn":
        ensure_type(row, "turns", list, suite)
        ensure_type(row, "expected_final_metadata", dict, suite)
        ensure(row["expected_final_result_type"] in {"ready_for_analysis", "needs_follow_up", "general_chat", "max_followup_reached"}, f"{suite}:{row_id}: invalid result type")
        ensure(isinstance(row["expected_turns_to_ready"], int) or row["expected_turns_to_ready"] is None, f"{suite}:{row_id}: expected_turns_to_ready must be int or null")
        ensure_type(row, "expected_state_after_each_turn", list, suite)
        ensure_type(row, "expected_questions_after_each_turn", list, suite)
        user_turns = [turn for turn in row["turns"] if isinstance(turn, dict) and turn.get("role") == "user"]
        ensure(len(row["expected_state_after_each_turn"]) == len(user_turns), f"{suite}:{row_id}: state count must equal user turns")
        ensure(len(row["expected_questions_after_each_turn"]) == len(user_turns), f"{suite}:{row_id}: question count must equal user turns")
        for index, state in enumerate(row["expected_state_after_each_turn"], start=1):
            ensure(isinstance(state, dict), f"{suite}:{row_id}: expected_state_after_each_turn[{index}] must be object")
            ensure("last_missing_fields" in state, f"{suite}:{row_id}: expected state must use last_missing_fields")
            ensure("missing_fields" not in state, f"{suite}:{row_id}: expected state must not use stale missing_fields key")
        if row["expected_final_result_type"] == "max_followup_reached":
            ensure(row["expected_questions_after_each_turn"][-1] == [], f"{suite}:{row_id}: max_followup terminal turn must not expect another follow-up question")
        turn_positions = [i for i, turn in enumerate(row["turns"]) if isinstance(turn, dict) and turn.get("role") == "user"]
        for q_index, turn_index in enumerate(turn_positions):
            expected_keywords = row["expected_questions_after_each_turn"][q_index]
            if expected_keywords:
                ensure(turn_index + 1 < len(row["turns"]) and row["turns"][turn_index + 1].get("role") == "assistant", f"{suite}:{row_id}: expected follow-up keywords require a following assistant turn")
            if turn_index + 1 < len(row["turns"]) and row["turns"][turn_index + 1].get("role") == "assistant":
                assistant_text = str(row["turns"][turn_index + 1].get("content") or "")
                missing_keywords = [kw for kw in expected_keywords if str(kw) not in assistant_text]
                ensure(not missing_keywords, f"{suite}:{row_id}: expected follow-up keywords absent from next assistant turn: {missing_keywords}")
    elif suite == "structured_output":
        ensure_type(row, "search_metadata", dict, suite)
        ensure(row["modifier_source"] in {"none", "same_standard", "cross_reference"}, f"{suite}:{row_id}: invalid modifier_source")
        for field in ["expected_base_fault_ratio", "expected_final_fault_ratio"]:
            value = row[field]
            ensure(value is None or (isinstance(value, dict) and set(value) == {"a", "b"}), f"{suite}:{row_id}: {field} must be null or {{a,b}}")
            if isinstance(value, dict):
                ensure(isinstance(value["a"], int) and isinstance(value["b"], int) and value["a"] + value["b"] == 100, f"{suite}:{row_id}: {field} must sum to 100")
        if row["modifier_source"] == "cross_reference":
            ensure(row["expected_reference_diagram_ids"], f"{suite}:{row_id}: cross_reference needs expected_reference_diagram_ids")
        else:
            ensure(row["expected_reference_diagram_ids"] == [], f"{suite}:{row_id}: non-cross-reference must not have reference ids")
        ensure_type(row, "required_evidence", list, suite)
        if row["case_family"] != "general_rule":
            ensure(row["near_miss_diagram_ids"], f"{suite}:{row_id}: non-general structured row needs near_miss_diagram_ids")


def validate_suite(suite: str, data_dir: Path, diagrams: set[str], text_by_diagram: dict[str, str], allow_short: bool, evidence_check: bool) -> int:
    spec = SUITES[suite]
    path = data_dir / spec["file"]
    if not path.exists():
        raise ValueError(f"{suite}: missing suite file: {path}")
    rows = read_jsonl(path)
    if not allow_short:
        ensure(len(rows) >= spec["min"], f"{suite}: expected at least {spec['min']} rows, got {len(rows)}")
        if "max" in spec:
            ensure(len(rows) <= spec["max"], f"{suite}: expected at most {spec['max']} rows, got {len(rows)}")
    seen: set[str] = set()
    for row in rows:
        check_common(suite, row, spec)
        ensure(row["id"] not in seen, f"{suite}:{row['id']}: duplicate id")
        seen.add(row["id"])
        check_suite_specific(suite, row)
        check_diagrams(suite, row, diagrams, text_by_diagram, evidence_check)
    if suite == "intake" and not allow_short:
        pairs = {(row.get("expected_location"), row.get("inference_type")) for row in rows}
        ensure(("같은 방향 진행차량 상호간의 사고", "explicit_keyword") in pairs, "intake: missing explicit same-direction metadata row")
        ensure(("같은 방향 진행차량 상호간의 사고", "implicit_metadata") in pairs, "intake: missing implicit same-direction metadata row")
        ensure(("마주보는 방향 진행차량 상호 간의 사고", "explicit_keyword") in pairs, "intake: missing explicit opposite-direction metadata row")
        ensure(("마주보는 방향 진행차량 상호 간의 사고", "implicit_metadata") in pairs, "intake: missing implicit opposite-direction metadata row")
    if suite == "retrieval" and not allow_short:
        inference_types = {row.get("inference_type") for row in rows}
        missing = sorted(INFERENCE_TYPES - inference_types)
        ensure(not missing, f"retrieval: missing inference_type coverage {missing}")
        ensure(any(row.get("acceptable_diagram_ids") for row in rows), "retrieval: missing multi-acceptable diagram coverage")
    return len(rows)


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    try:
        diagrams, text_by_diagram = load_chunks(Path(args.chunks))
        suites = args.suite or list(SUITES)
        for suite in suites:
            count = validate_suite(suite, data_dir, diagrams, text_by_diagram, args.allow_short_counts, not args.no_evidence_check)
            print(f"{suite}: {count} rows ok")
        print("validation ok")
        return 0
    except ValueError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
