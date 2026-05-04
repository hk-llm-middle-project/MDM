"""Evaluate non-retrieval LangSmith-style suites with local project functions."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict
from datetime import datetime
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BASE_DIR
from rag.service.analysis.analysis_service import analyze_question
from rag.service.analysis.answer_schema import AnalysisResult, RetrievedContext
from rag.service.conversation.orchestrator import AnswerResult, answer_conversation_turn
from rag.service.conversation.schema import RouteDecision, TurnResultType
from rag.service.intake.filter_service import build_metadata_filters
from rag.service.intake.intake_service import evaluate_input_sufficiency
from rag.service.intake.schema import IntakeDecision, IntakeState, UserSearchMetadata
from rag.service.session.schema import ChatMessage
from rag.service.session.serialization import intake_state_from_dict, message_from_dict


DEFAULT_TESTSET_DIR = BASE_DIR / "data" / "testsets" / "langsmith"
DEFAULT_OUTPUT_DIR = BASE_DIR / "evaluation" / "results" / "langsmith"
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


def load_jsonl(path: Path, max_examples: int = 0) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Testset file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    if max_examples > 0:
        rows = rows[:max_examples]
    if not rows:
        raise RuntimeError(f"No examples were loaded from {path}")
    return rows


def _csv_cell(value: Any) -> Any:
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False)
    if hasattr(value, "value"):
        return value.value
    return value


def _flatten_record(prefix: str, values: dict[str, Any], record: dict[str, Any]) -> None:
    for key, value in values.items():
        record[f"{prefix}.{key}"] = _csv_cell(value)


def _base_record(row: dict[str, Any], suite: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "evaluation_suite": suite,
        "example_id": row.get("id"),
    }
    question = row.get("question")
    if question is None and isinstance(row.get("turns"), list):
        question = next(
            (
                turn.get("content")
                for turn in row["turns"]
                if isinstance(turn, dict) and turn.get("role") == "user"
            ),
            None,
        )
    if question is not None:
        record["inputs.question"] = question
    for field in CASE_METADATA_FIELDS:
        if row.get(field) is not None:
            record[field] = _csv_cell(row.get(field))
    return record


def _score(record: dict[str, Any], key: str, score: float | int | None, comment: str = "") -> None:
    record[f"feedback.{key}"] = score
    if comment:
        record[f"feedback.{key}.comment"] = comment


def _mean(values: Iterable[float | int | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _messages(raw_messages: object) -> list[ChatMessage]:
    if not isinstance(raw_messages, list):
        return []
    messages: list[ChatMessage] = []
    for raw in raw_messages:
        if isinstance(raw, dict):
            messages.append(message_from_dict(raw))
    return messages


def _intake_state(raw_state: object) -> IntakeState:
    return intake_state_from_dict(raw_state if isinstance(raw_state, dict) else {})


def _search_metadata(raw_metadata: object) -> UserSearchMetadata:
    return _intake_state({"search_metadata": raw_metadata if isinstance(raw_metadata, dict) else {}}).search_metadata


def _metadata_dict(metadata: UserSearchMetadata) -> dict[str, Any]:
    return asdict(metadata)


def _missing_field_names(decision: IntakeDecision) -> list[str]:
    return [field.name for field in decision.missing_fields]


def _contains_all(actual_values: Sequence[str], expected_substrings: object) -> bool:
    if not isinstance(expected_substrings, list) or not expected_substrings:
        return True
    haystack = "\n".join(str(value) for value in actual_values)
    return all(str(expected) in haystack for expected in expected_substrings)


def _forbidden_filters_absent(metadata: dict[str, Any], forbidden_filters: object) -> bool:
    if not isinstance(forbidden_filters, list):
        return True
    for forbidden_filter in forbidden_filters:
        if not isinstance(forbidden_filter, dict):
            continue
        for key, value in forbidden_filter.items():
            if metadata.get(key) == value:
                return False
    return True


def _filter_field_absent(filters: Any, field_name: str) -> bool:
    if isinstance(filters, dict):
        if field_name in filters:
            return False
        return all(_filter_field_absent(value, field_name) for value in filters.values())
    if isinstance(filters, list):
        return all(_filter_field_absent(value, field_name) for value in filters)
    return True


def _partial_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for key, expected_value in expected.items():
            if not _partial_match(expected_value, actual.get(key)):
                return False
        return True
    if isinstance(expected, list):
        return list(expected) == list(actual or [])
    return expected == actual


def _route_value(decision: RouteDecision) -> str:
    return decision.route_type.value if hasattr(decision.route_type, "value") else str(decision.route_type)


def _retrieved_metadata(contexts: Sequence[RetrievedContext]) -> list[dict[str, Any]]:
    return [dict(context.metadata) for context in contexts]


def _diagram_ids_from_contexts(contexts: Sequence[RetrievedContext]) -> list[Any]:
    return [context.metadata.get("diagram_id") for context in contexts if isinstance(context.metadata, dict)]


def _normalize_analysis_result(result: AnalysisResult | tuple[str, list[str]]) -> AnalysisResult:
    if isinstance(result, AnalysisResult):
        return result
    answer, contexts = result
    return AnalysisResult(response=answer, contexts=contexts)


def evaluate_intake_rows(
    rows: list[dict[str, Any]],
    evaluator: Callable[..., IntakeDecision] = evaluate_input_sufficiency,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        record = _base_record(row, "intake")
        started_at = time.perf_counter()
        try:
            decision = evaluator(
                str(row.get("question") or ""),
                chat_history=_messages(row.get("chat_history")),
                previous_state=_intake_state(row.get("previous_state")),
            )
            metadata = _metadata_dict(decision.search_metadata)
            outputs = {
                "is_sufficient": decision.is_sufficient,
                "party_type": decision.search_metadata.party_type,
                "location": decision.search_metadata.location,
                "retrieval_query": decision.search_metadata.retrieval_query,
                "missing_fields": _missing_field_names(decision),
                "follow_up_questions": decision.follow_up_questions,
                "search_metadata": metadata,
            }
            record["error"] = None
        except Exception as exc:  # noqa: BLE001 - record suite failures as rows.
            metadata = {}
            outputs = {"error": str(exc), "error_type": type(exc).__name__}
            record["error"] = str(exc)

        record["execution_time"] = time.perf_counter() - started_at
        _flatten_record("outputs", outputs, record)
        references = {
            "expected_is_sufficient": row.get("expected_is_sufficient"),
            "expected_party_type": row.get("expected_party_type"),
            "expected_location": row.get("expected_location"),
            "expected_missing_fields": row.get("expected_missing_fields", []),
            "expected_follow_up_questions_contain": row.get("expected_follow_up_questions_contain", []),
            "forbidden_filters": row.get("forbidden_filters", []),
        }
        _flatten_record("reference", references, record)

        if record["error"] is None:
            scores = {
                "intake_is_sufficient": outputs.get("is_sufficient") == row.get("expected_is_sufficient"),
                "party_type_match": outputs.get("party_type") == row.get("expected_party_type"),
                "location_match": outputs.get("location") == row.get("expected_location"),
                "missing_fields_match": set(outputs.get("missing_fields", []))
                == set(row.get("expected_missing_fields", [])),
                "follow_up_contains": _contains_all(
                    outputs.get("follow_up_questions", []),
                    row.get("expected_follow_up_questions_contain", []),
                ),
                "forbidden_filter_absent": _forbidden_filters_absent(
                    metadata,
                    row.get("forbidden_filters", []),
                ),
            }
        else:
            scores = {
                "intake_is_sufficient": False,
                "party_type_match": False,
                "location_match": False,
                "missing_fields_match": False,
                "follow_up_contains": False,
                "forbidden_filter_absent": False,
            }
        for key, matched in scores.items():
            _score(record, key, int(bool(matched)))
        _score(record, "intake_overall", _mean(record[f"feedback.{key}"] for key in scores))
        records.append(record)
    return pd.DataFrame.from_records(records)


def evaluate_router_rows(
    rows: list[dict[str, Any]],
    router: Callable[..., RouteDecision],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        record = _base_record(row, "router")
        started_at = time.perf_counter()
        try:
            decision = router(
                str(row.get("question") or ""),
                _messages(row.get("chat_history")),
                _intake_state(row.get("intake_state")),
            )
            outputs = {
                "route_type": _route_value(decision),
                "confidence": decision.confidence,
                "reason": decision.reason,
            }
            record["error"] = None
        except Exception as exc:  # noqa: BLE001
            outputs = {"error": str(exc), "error_type": type(exc).__name__}
            record["error"] = str(exc)
        record["execution_time"] = time.perf_counter() - started_at
        _flatten_record("outputs", outputs, record)
        references = {
            "expected_route_type": row.get("expected_route_type"),
            "expected_reason_category": row.get("expected_reason_category"),
            "expected_is_follow_up": row.get("expected_is_follow_up"),
        }
        _flatten_record("reference", references, record)

        route_score = int(record["error"] is None and outputs.get("route_type") == row.get("expected_route_type"))
        expected_reason = row.get("expected_reason_category")
        reason_score = None
        if expected_reason and record["error"] is None:
            reason_score = int(str(expected_reason) in str(outputs.get("reason") or ""))
        elif expected_reason:
            reason_score = 0
        _score(record, "route_type_match", route_score)
        _score(record, "reason_category_match", reason_score)
        _score(record, "router_overall", _mean([route_score, reason_score]))
        records.append(record)
    return pd.DataFrame.from_records(records)


def evaluate_metadata_filter_rows(
    rows: list[dict[str, Any]],
    filter_builder: Callable[[UserSearchMetadata | None], dict[str, object] | None] = build_metadata_filters,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        record = _base_record(row, "metadata_filter")
        started_at = time.perf_counter()
        try:
            metadata = _search_metadata(row.get("search_metadata"))
            filters = filter_builder(metadata)
            outputs = {
                "search_metadata": _metadata_dict(metadata),
                "filter": filters,
            }
            record["error"] = None
        except Exception as exc:  # noqa: BLE001
            filters = None
            outputs = {"error": str(exc), "error_type": type(exc).__name__}
            record["error"] = str(exc)
        record["execution_time"] = time.perf_counter() - started_at
        _flatten_record("outputs", outputs, record)
        references = {
            "expected_filter": row.get("expected_filter"),
            "expected_behavior": row.get("expected_behavior"),
            "must_not_filter_fields": row.get("must_not_filter_fields", []),
            "expected_fallback_required": row.get("expected_fallback_required"),
        }
        _flatten_record("reference", references, record)

        filter_score = int(record["error"] is None and filters == row.get("expected_filter"))
        forbidden_score = (
            int(
                all(
                    _filter_field_absent(filters, field)
                    for field in row.get("must_not_filter_fields", [])
                )
            )
            if record["error"] is None
            else 0
        )
        _score(record, "metadata_filter_match", filter_score)
        _score(record, "forbidden_filter_absent", forbidden_score)
        _score(record, "metadata_filter_overall", _mean([filter_score, forbidden_score]))
        records.append(record)
    return pd.DataFrame.from_records(records)


def _turn_result_label(result: AnswerResult) -> str:
    if result.result_type == TurnResultType.ACCIDENT_FOLLOW_UP or result.needs_more_input:
        return "needs_follow_up"
    if result.result_type == TurnResultType.ACCIDENT_RAG:
        metadata = result.intake_state.search_metadata
        if metadata.party_type is None or metadata.location is None:
            return "max_followup_reached"
        return "ready_for_analysis"
    return result.result_type.value if hasattr(result.result_type, "value") else str(result.result_type)


def _state_snapshot(state: IntakeState) -> dict[str, Any]:
    return {
        "search_metadata": {
            "party_type": state.search_metadata.party_type,
            "location": state.search_metadata.location,
            "retrieval_query": state.search_metadata.retrieval_query,
        },
        "last_missing_fields": list(state.last_missing_fields),
        "last_follow_up_questions": list(state.last_follow_up_questions),
    }


def evaluate_multiturn_rows(
    rows: list[dict[str, Any]],
    turn_handler: Callable[..., AnswerResult] = answer_conversation_turn,
    intake_evaluator: Callable[..., IntakeDecision] = evaluate_input_sufficiency,
    analyzer: Callable[..., AnalysisResult] = analyze_question,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        record = _base_record(row, "multiturn")
        started_at = time.perf_counter()
        history: list[ChatMessage] = []
        state = IntakeState()
        snapshots: list[dict[str, Any]] = []
        followup_questions: list[list[str]] = []
        ready_turn: int | None = None
        final_label = "needs_follow_up"
        user_turn_count = 0
        try:
            for turn in row.get("turns", []):
                if not isinstance(turn, dict) or turn.get("role") != "user":
                    continue
                user_turn_count += 1
                question = str(turn.get("content") or "")
                result = turn_handler(
                    question,
                    chat_history=history,
                    intake_state=state,
                    intake_evaluator=intake_evaluator,
                    analyzer=analyzer,
                )
                state = result.intake_state
                snapshots.append(_state_snapshot(state))
                questions = (
                    list(state.last_follow_up_questions)
                    if result.result_type == TurnResultType.ACCIDENT_FOLLOW_UP
                    else []
                )
                followup_questions.append(questions)
                final_label = _turn_result_label(result)
                if ready_turn is None and final_label == "ready_for_analysis":
                    ready_turn = user_turn_count
                history.append(ChatMessage(role="user", content=question))
                history.append(ChatMessage(role="assistant", content=result.answer))
            outputs = {
                "state_after_each_turn": snapshots,
                "questions_after_each_turn": followup_questions,
                "final_metadata": _state_snapshot(state)["search_metadata"],
                "final_result_type": final_label,
                "turns_to_ready": ready_turn,
            }
            record["error"] = None
        except Exception as exc:  # noqa: BLE001
            outputs = {"error": str(exc), "error_type": type(exc).__name__}
            record["error"] = str(exc)

        record["execution_time"] = time.perf_counter() - started_at
        _flatten_record("inputs", {"turns": row.get("turns", [])}, record)
        _flatten_record("outputs", outputs, record)
        references = {
            "expected_state_after_each_turn": row.get("expected_state_after_each_turn", []),
            "expected_questions_after_each_turn": row.get("expected_questions_after_each_turn", []),
            "expected_final_metadata": row.get("expected_final_metadata"),
            "expected_final_result_type": row.get("expected_final_result_type"),
            "expected_turns_to_ready": row.get("expected_turns_to_ready"),
        }
        _flatten_record("reference", references, record)

        if record["error"] is None:
            state_score = int(
                len(outputs.get("state_after_each_turn", []))
                == len(row.get("expected_state_after_each_turn", []))
                and all(
                    _partial_match(expected, actual)
                    for expected, actual in zip(
                        row.get("expected_state_after_each_turn", []),
                        outputs.get("state_after_each_turn", []),
                        strict=True,
                    )
                )
            )
            question_score = int(
                len(outputs.get("questions_after_each_turn", []))
                == len(row.get("expected_questions_after_each_turn", []))
                and all(
                    _contains_all(actual, expected)
                    for expected, actual in zip(
                        row.get("expected_questions_after_each_turn", []),
                        outputs.get("questions_after_each_turn", []),
                        strict=True,
                    )
                )
            )
            final_metadata_score = int(
                _partial_match(
                    row.get("expected_final_metadata"),
                    outputs.get("final_metadata"),
                )
            )
            final_type_score = int(
                outputs.get("final_result_type") == row.get("expected_final_result_type")
            )
            turns_score = int(outputs.get("turns_to_ready") == row.get("expected_turns_to_ready"))
        else:
            state_score = 0
            question_score = 0
            final_metadata_score = 0
            final_type_score = 0
            turns_score = 0
        scores = [
            state_score,
            question_score,
            final_metadata_score,
            final_type_score,
            turns_score,
        ]
        _score(record, "state_sequence_match", state_score)
        _score(record, "followup_questions_match", question_score)
        _score(record, "final_metadata_match", final_metadata_score)
        _score(record, "final_result_type_match", final_type_score)
        _score(record, "turns_to_ready_match", turns_score)
        _score(record, "multiturn_overall", _mean(scores))
        records.append(record)
    return pd.DataFrame.from_records(records)


def _required_evidence_coverage(result: AnalysisResult, required_evidence: object) -> float | None:
    if not isinstance(required_evidence, list) or not required_evidence:
        return None
    return _text_coverage(result, [str(value) for value in required_evidence])


def _text_coverage(result: AnalysisResult, expected_terms: list[str]) -> float | None:
    if not expected_terms:
        return None
    haystack = json.dumps(
        {
            "response": result.response,
            "contexts": result.contexts,
            "retrieved_metadata": _retrieved_metadata(result.retrieved_contexts),
        },
        ensure_ascii=False,
    )
    matched = [value for value in expected_terms if value in haystack]
    return len(matched) / len(expected_terms)


def _role_coverage(result: AnalysisResult, expected_roles: object) -> float | None:
    if not isinstance(expected_roles, dict) or not expected_roles:
        return None
    return _text_coverage(
        result,
        [str(value) for value in expected_roles.values() if value],
    )


def _list_coverage(result: AnalysisResult, expected_terms: object) -> float | None:
    if not isinstance(expected_terms, list) or not expected_terms:
        return None
    return _text_coverage(result, [str(value) for value in expected_terms if value])


def evaluate_structured_output_rows(
    rows: list[dict[str, Any]],
    analyzer: Callable[..., AnalysisResult | tuple[str, list[str]]] = analyze_question,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        record = _base_record(row, "structured_output")
        started_at = time.perf_counter()
        try:
            metadata = _search_metadata(row.get("search_metadata"))
            result = _normalize_analysis_result(
                analyzer(str(row.get("question") or ""), search_metadata=metadata)
            )
            outputs = {
                "fault_ratio_a": result.fault_ratio_a,
                "fault_ratio_b": result.fault_ratio_b,
                "response": result.response,
                "contexts": result.contexts,
                "retrieved_metadata": _retrieved_metadata(result.retrieved_contexts),
            }
            record["error"] = None
        except Exception as exc:  # noqa: BLE001
            result = AnalysisResult(response="")
            outputs = {"error": str(exc), "error_type": type(exc).__name__}
            record["error"] = str(exc)

        record["execution_time"] = time.perf_counter() - started_at
        _flatten_record("outputs", outputs, record)
        references = {
            "expected_diagram_ids": row.get("expected_diagram_ids", []),
            "near_miss_diagram_ids": row.get("near_miss_diagram_ids", []),
            "expected_final_fault_ratio": row.get("expected_final_fault_ratio"),
            "expected_cannot_determine_reason": row.get("expected_cannot_determine_reason"),
            "required_evidence": row.get("required_evidence", []),
            "expected_party_roles": row.get("expected_party_roles"),
            "expected_applicable_modifiers": row.get("expected_applicable_modifiers", []),
            "expected_non_applicable_modifiers": row.get("expected_non_applicable_modifiers", []),
        }
        _flatten_record("reference", references, record)

        if record["error"] is None:
            expected_ratio = row.get("expected_final_fault_ratio")
            if isinstance(expected_ratio, dict):
                ratio_score = int(
                    outputs.get("fault_ratio_a") == expected_ratio.get("a")
                    and outputs.get("fault_ratio_b") == expected_ratio.get("b")
                )
            else:
                ratio_score = int(outputs.get("fault_ratio_a") is None and outputs.get("fault_ratio_b") is None)
            expected_cannot = row.get("expected_cannot_determine_reason") is not None
            cannot_score = int(
                (outputs.get("fault_ratio_a") is None and outputs.get("fault_ratio_b") is None)
                if expected_cannot
                else (outputs.get("fault_ratio_a") is not None and outputs.get("fault_ratio_b") is not None)
            )
            coverage = _required_evidence_coverage(result, row.get("required_evidence", []))
            role_coverage = _role_coverage(result, row.get("expected_party_roles"))
            applicable_modifier_coverage = _list_coverage(
                result,
                row.get("expected_applicable_modifiers", []),
            )
            non_applicable_modifier_coverage = _list_coverage(
                result,
                row.get("expected_non_applicable_modifiers", []),
            )
            expected_diagrams = set(row.get("expected_diagram_ids", []))
            actual_diagrams = set(_diagram_ids_from_contexts(result.retrieved_contexts))
            diagram_score = int(bool(expected_diagrams & actual_diagrams)) if expected_diagrams else 1
        else:
            ratio_score = 0
            cannot_score = 0
            coverage = 0
            role_coverage = 0
            applicable_modifier_coverage = 0
            non_applicable_modifier_coverage = 0
            diagram_score = 0
        _score(record, "final_fault_ratio_match", ratio_score)
        _score(record, "cannot_determine_match", cannot_score)
        _score(record, "required_evidence_coverage", coverage)
        _score(record, "party_role_coverage", role_coverage)
        _score(record, "applicable_modifier_coverage", applicable_modifier_coverage)
        _score(record, "non_applicable_modifier_coverage", non_applicable_modifier_coverage)
        _score(record, "reference_diagram_hit", diagram_score)
        _score(
            record,
            "structured_output_overall",
            _mean(
                [
                    ratio_score,
                    cannot_score,
                    coverage,
                    role_coverage,
                    applicable_modifier_coverage,
                    non_applicable_modifier_coverage,
                    diagram_score,
                ]
            ),
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣._-]+", "-", value).strip("-")
    return cleaned or "decision-eval"


def summarize_feedback_metrics(dataframe: pd.DataFrame) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for column in dataframe.columns:
        if not str(column).startswith("feedback.") or str(column).endswith(".comment"):
            continue
        metric_name = str(column).removeprefix("feedback.")
        numeric = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        metrics[metric_name] = float(numeric.mean()) if len(numeric) else None
    return metrics


def save_suite_dataframe(
    dataframe: pd.DataFrame,
    suite: str,
    output_dir: Path,
    testset_path: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = safe_filename(f"{timestamp}-{suite}-local")
    csv_path = output_dir / f"{base_name}.csv"
    summary_path = output_dir / f"{base_name}.summary.json"
    dataframe.to_csv(csv_path, index=False)
    summary = {
        "experiment_name": f"MDM {suite} eval - local",
        "dataset_name": f"MDM {suite} testset",
        "testset_path": str(testset_path),
        "evaluation_suite": suite,
        "run_name": f"{suite}-local",
        "loader_strategy": None,
        "chunker_strategy": None,
        "embedding_provider": None,
        "retriever_strategy": None,
        "reranker_strategy": None,
        "row_count": int(len(dataframe)),
        "metrics": summarize_feedback_metrics(dataframe),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"csv": csv_path, "summary_json": summary_path}


def evaluate_suite(suite: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    if suite == "intake":
        return evaluate_intake_rows(rows)
    if suite == "router":
        from rag.service.conversation.router import route_conversation_turn

        return evaluate_router_rows(rows, router=route_conversation_turn)
    if suite == "metadata_filter":
        return evaluate_metadata_filter_rows(rows)
    if suite == "multiturn":
        return evaluate_multiturn_rows(rows)
    if suite == "structured_output":
        return evaluate_structured_output_rows(rows)
    raise ValueError(f"Unsupported suite: {suite}")


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
