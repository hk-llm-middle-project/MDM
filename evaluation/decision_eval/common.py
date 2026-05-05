"""Shared record and scoring helpers for decision-suite evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict
import json
from typing import Any

from rag.service.analysis.answer_schema import AnalysisResult, RetrievedContext
from rag.service.conversation.schema import RouteDecision
from rag.service.intake.schema import IntakeDecision, IntakeState, UserSearchMetadata
from rag.service.session.schema import ChatMessage
from rag.service.session.serialization import intake_state_from_dict, message_from_dict

from evaluation.decision_eval.constants import CASE_METADATA_FIELDS


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
