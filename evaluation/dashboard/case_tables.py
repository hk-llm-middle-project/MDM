"""Case-level dashboard table builders."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import json

import pandas as pd

from evaluation.dashboard.formatters import (
    format_display_value as _format_display_value,
    format_score as _format_score,
    has_display_value as _has_display_value,
    is_empty_scalar as _is_empty_scalar,
    parse_jsonish as _parse_jsonish,
)
from evaluation.dashboard.metrics import METRIC_COLUMNS


EXPECTED_VALUE_COLUMNS = (
    "reference.expected_diagram_ids",
    "reference.acceptable_diagram_ids",
    "reference.near_miss_diagram_ids",
    "reference.expected_location",
    "reference.expected_party_type",
    "reference.expected_chunk_types",
    "reference.expected_keywords",
    "reference.expected_filter",
    "reference.expected_route_type",
    "reference.expected_is_sufficient",
    "reference.expected_missing_fields",
    "reference.expected_follow_up_questions_contain",
    "reference.expected_final_fault_ratio",
    "reference.expected_party_roles",
    "reference.expected_applicable_modifiers",
    "reference.expected_non_applicable_modifiers",
    "reference.required_evidence",
    "reference.expected_cannot_determine_reason",
)

EXPECTED_VALUE_LABELS = {
    "reference.expected_diagram_ids": "기대 diagram",
    "reference.acceptable_diagram_ids": "허용 diagram",
    "reference.near_miss_diagram_ids": "near-miss diagram",
    "reference.expected_location": "기대 location",
    "reference.expected_party_type": "기대 party type",
    "reference.expected_chunk_types": "기대 chunk type",
    "reference.expected_keywords": "기대 keyword",
    "reference.expected_filter": "기대 filter",
    "reference.expected_route_type": "기대 route type",
    "reference.expected_is_sufficient": "기대 sufficient",
    "reference.expected_missing_fields": "기대 missing fields",
    "reference.expected_follow_up_questions_contain": "기대 follow-up 문구",
    "reference.expected_final_fault_ratio": "기대 final fault ratio",
    "reference.expected_party_roles": "기대 party roles",
    "reference.expected_applicable_modifiers": "기대 applicable modifiers",
    "reference.expected_non_applicable_modifiers": "기대 non-applicable modifiers",
    "reference.required_evidence": "필수 evidence",
    "reference.expected_cannot_determine_reason": "기대 cannot-determine reason",
}

REFERENCE_ACTUAL_METADATA_KEYS = {
    "reference.expected_diagram_ids": "diagram_id",
    "reference.acceptable_diagram_ids": "diagram_id",
    "reference.near_miss_diagram_ids": "diagram_id",
    "reference.expected_location": "location",
    "reference.expected_party_type": "party_type",
    "reference.expected_chunk_types": "chunk_type",
}


def _row_run_name(row: pd.Series, fallback_index: int) -> str:
    for column in ("run_label", "run_name"):
        value = row.get(column)
        if _has_display_value(value):
            return _format_display_value(value)
    return f"run {fallback_index + 1}"


def _run_names(rows: pd.DataFrame) -> list[str]:
    names: list[str] = []
    seen: dict[str, int] = {}
    for index, row in rows.iterrows():
        name = _row_run_name(row, len(names))
        seen[name] = seen.get(name, 0) + 1
        names.append(name if seen[name] == 1 else f"{name} #{seen[name]}")
    return names


def _reference_columns_with_values(rows: pd.DataFrame) -> list[str]:
    preferred = [column for column in EXPECTED_VALUE_COLUMNS if column in rows.columns]
    extra = [
        column
        for column in rows.columns
        if column.startswith("reference.")
        and column not in preferred
        and column != "reference.reference"
        and _is_expected_reference_column(column)
    ]
    output: list[str] = []
    for column in [*preferred, *sorted(extra)]:
        if rows[column].map(_has_display_value).any():
            output.append(column)
    return output


def _is_expected_reference_column(column: str) -> bool:
    suffix = column.removeprefix("reference.")
    return suffix.startswith(("expected_", "acceptable_", "near_miss_", "required_", "forbidden_"))


def _expected_label(column: str) -> str:
    if column in EXPECTED_VALUE_LABELS:
        return EXPECTED_VALUE_LABELS[column]
    return column.removeprefix("reference.").replace("_", " ")


def _metadata_records(row: pd.Series) -> list[dict[str, Any]]:
    parsed = _parse_jsonish(row.get("outputs.retrieved_metadata"))
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _metadata_values(row: pd.Series, key: str) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for record in _metadata_records(row):
        raw_value = record.get(key)
        if isinstance(raw_value, list):
            candidates = raw_value
        else:
            candidates = [raw_value]
        for candidate in candidates:
            formatted = _format_display_value(candidate)
            if formatted and formatted not in seen:
                values.append(formatted)
                seen.add(formatted)
    return ", ".join(values)


def _first_output_value(row: pd.Series, candidates: Iterable[str]) -> str:
    for column in candidates:
        if column in row.index and _has_display_value(row.get(column)):
            return _format_display_value(row.get(column))
    return ""


def _first_raw_output_value(row: pd.Series, candidates: Iterable[str]) -> Any:
    for column in candidates:
        if column in row.index and _has_display_value(row.get(column)):
            return row.get(column)
    return None


def _keyword_coverage_comment(row: pd.Series) -> Any:
    comment = row.get("keyword_coverage_comment")
    if _is_empty_scalar(comment):
        comment = row.get("feedback.keyword_coverage.comment")
    return comment


def _actual_value_for_reference(row: pd.Series, reference_column: str) -> str:
    metadata_key = REFERENCE_ACTUAL_METADATA_KEYS.get(reference_column)
    if metadata_key:
        metadata_value = _metadata_values(row, metadata_key)
        if metadata_value:
            return metadata_value

    if reference_column == "reference.expected_keywords":
        comment = _keyword_coverage_comment(row)
        if _has_display_value(comment):
            return _format_display_value(comment)

    if reference_column == "reference.expected_follow_up_questions_contain":
        return _first_output_value(
            row,
            [
                "outputs.follow_up_questions",
                "outputs.followup_questions",
                "outputs.follow_up_question",
            ],
        )

    suffix = reference_column.removeprefix("reference.")
    normalized_suffix = suffix.removeprefix("expected_")
    return _first_output_value(
        row,
        [
            f"outputs.{suffix}",
            f"outputs.{normalized_suffix}",
            f"outputs.result.{normalized_suffix}",
            f"outputs.metadata.{normalized_suffix}",
        ],
    )


def _actual_case_table_value_for_reference(row: pd.Series, reference_column: str) -> Any:
    """Return raw-ish actual values so the case table can pretty-print lists."""

    metadata_key = REFERENCE_ACTUAL_METADATA_KEYS.get(reference_column)
    if metadata_key:
        metadata_value = _metadata_values(row, metadata_key)
        if metadata_value:
            return metadata_value

    if reference_column == "reference.expected_keywords":
        comment = _keyword_coverage_comment(row)
        if _has_display_value(comment):
            return comment

    if reference_column == "reference.expected_follow_up_questions_contain":
        return _first_raw_output_value(
            row,
            [
                "outputs.follow_up_questions",
                "outputs.followup_questions",
                "outputs.follow_up_question",
            ],
        )

    suffix = reference_column.removeprefix("reference.")
    normalized_suffix = suffix.removeprefix("expected_")
    return _first_raw_output_value(
        row,
        [
            f"outputs.{suffix}",
            f"outputs.{normalized_suffix}",
            f"outputs.result.{normalized_suffix}",
            f"outputs.metadata.{normalized_suffix}",
        ],
    )


def case_question(rows: pd.DataFrame) -> str:
    if rows.empty or "inputs.question" not in rows.columns:
        return ""
    for value in rows["inputs.question"]:
        if _has_display_value(value):
            return _format_display_value(value)
    return ""


def build_case_value_comparison(rows: pd.DataFrame) -> pd.DataFrame:
    """Show one test case as expected values next to each run's actual values."""

    base_columns = ["항목", "예상 값"]
    if rows.empty:
        return pd.DataFrame(columns=base_columns)

    first = rows.iloc[0]
    run_names = _run_names(rows)
    records: list[dict[str, str]] = []
    for reference_column in _reference_columns_with_values(rows):
        record = {
            "항목": _expected_label(reference_column),
            "예상 값": _format_display_value(first.get(reference_column)),
        }
        for run_name, (_, row) in zip(run_names, rows.iterrows(), strict=True):
            record[run_name] = _actual_value_for_reference(row, reference_column)
        records.append(record)

    if not records:
        return pd.DataFrame(columns=[*base_columns, *run_names])
    return pd.DataFrame.from_records(records)


def build_case_metric_comparison(rows: pd.DataFrame) -> pd.DataFrame:
    """Show metric scores and comments for one test case across runs."""

    if rows.empty:
        return pd.DataFrame(columns=["metric"])

    run_names = _run_names(rows)
    records: list[dict[str, str]] = []
    for metric in [metric for metric in METRIC_COLUMNS if metric in rows.columns]:
        scores = pd.to_numeric(rows[metric], errors="coerce")
        if scores.isna().all():
            continue
        record = {"metric": metric}
        comment_column = f"{metric}_comment"
        for run_name, (_, row) in zip(run_names, rows.iterrows(), strict=True):
            record[run_name] = _format_score(row.get(metric))
            if comment_column in rows.columns and _has_display_value(row.get(comment_column)):
                record[f"{run_name} comment"] = _format_display_value(row.get(comment_column))
        records.append(record)

    if not records:
        return pd.DataFrame(columns=["metric", *run_names])
    return pd.DataFrame.from_records(records)


def build_case_metric_matrix(examples: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot example-level scores into one row per stable case key."""

    run_column = "run_label" if "run_label" in examples.columns else "run_name"
    required_columns = {"case_key", "inputs.question", run_column, metric}
    if examples.empty or not required_columns.issubset(examples.columns):
        return pd.DataFrame()
    base = examples[["case_key", "inputs.question", run_column, metric]].copy()
    base[metric] = pd.to_numeric(base[metric], errors="coerce")
    questions = (
        base.groupby("case_key", dropna=False, as_index=False)["inputs.question"]
        .first()
    )
    pivot = base.pivot_table(
        index="case_key",
        columns=run_column,
        values=metric,
        aggfunc="first",
    ).reset_index()
    pivot.columns = [str(column) for column in pivot.columns]
    return questions.merge(pivot, on="case_key", how="right")


def rows_for_case(examples: pd.DataFrame, case_key: str) -> pd.DataFrame:
    """Return all run rows for a single stable test-case key."""

    if examples.empty or "case_key" not in examples.columns:
        return pd.DataFrame()
    return examples[examples["case_key"].astype(str) == str(case_key)].reset_index(drop=True)


def compare_runs_for_case(
    examples: pd.DataFrame,
    case_key: str,
    left_run: str,
    right_run: str | None = None,
    *additional_runs: str,
) -> pd.DataFrame:
    """Return selected run rows for one case."""

    rows = rows_for_case(examples, case_key)
    run_column = "run_label" if "run_label" in rows.columns else "run_name"
    if rows.empty or run_column not in rows.columns:
        return pd.DataFrame()
    selected_runs = [
        str(run)
        for run in (left_run, right_run, *additional_runs)
        if run is not None and str(run).strip()
    ]
    selected = rows[rows[run_column].isin(selected_runs)].copy()
    order = {run: index for index, run in enumerate(selected_runs)}
    selected["_compare_order"] = selected[run_column].map(order)
    return (
        selected.sort_values("_compare_order", kind="stable")
        .drop(columns=["_compare_order"])
        .reset_index(drop=True)
    )


def _run_mask(frame: pd.DataFrame, run_label: str) -> pd.Series:
    run_text = str(run_label)
    if "run_label" in frame.columns:
        return frame["run_label"].astype(str) == run_text
    if "run_name" in frame.columns:
        return frame["run_name"].astype(str) == run_text
    return pd.Series(False, index=frame.index)


def _testset_mask(frame: pd.DataFrame, testset_label: str) -> pd.Series:
    label_text = str(testset_label)
    if "testset_label" in frame.columns:
        return frame["testset_label"].astype(str) == label_text
    if "evaluation_suite" in frame.columns:
        return frame["evaluation_suite"].astype(str) == label_text
    return pd.Series(False, index=frame.index)



CASE_TABLE_LABELS = {
    "reference.expected_diagram_ids": "diagram",
    "reference.acceptable_diagram_ids": "허용 diagram",
    "reference.near_miss_diagram_ids": "near-miss diagram",
    "reference.expected_location": "location",
    "reference.expected_party_type": "party_type",
    "reference.expected_chunk_types": "chunk_type",
    "reference.expected_keywords": "keyword",
    "reference.expected_filter": "filter",
    "reference.expected_route_type": "route_type",
    "reference.expected_is_sufficient": "sufficient",
    "reference.expected_missing_fields": "missing fields",
    "reference.expected_follow_up_questions_contain": "follow-up 문구",
    "reference.expected_final_fault_ratio": "final fault ratio",
    "reference.expected_party_roles": "party roles",
    "reference.expected_applicable_modifiers": "applicable modifiers",
    "reference.expected_non_applicable_modifiers": "non-applicable modifiers",
    "reference.required_evidence": "required evidence",
    "reference.expected_cannot_determine_reason": "cannot-determine reason",
}


def _case_table_label(reference_column: str) -> str:
    return CASE_TABLE_LABELS.get(
        reference_column,
        _expected_label(reference_column).removeprefix("기대 "),
    )


def _format_case_table_value(value: Any) -> str:
    parsed = _parse_jsonish(value)
    if parsed is None:
        return "-"
    if isinstance(parsed, str):
        normalized = parsed.strip()
        if normalized.lower() in {"true", "false"}:
            return "예" if normalized.lower() == "true" else "아니오"
        return normalized or "-"
    if isinstance(parsed, bool):
        return "예" if parsed else "아니오"
    if isinstance(parsed, dict):
        if not parsed:
            return "없음"
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    if isinstance(parsed, (list, tuple, set)):
        if not parsed:
            return "없음"
        values = [_format_case_table_value(item) for item in parsed]
        return " | ".join(value for value in values if value and value != "-") or "없음"
    if isinstance(parsed, float) and pd.isna(parsed):
        return "-"
    if isinstance(parsed, float) and parsed.is_integer():
        return str(int(parsed))
    return str(parsed)


def _reference_columns_for_case_table(row: pd.Series) -> list[str]:
    columns: list[str] = []
    for column in [
        *[column for column in EXPECTED_VALUE_COLUMNS if column in row.index],
        *[
            column
            for column in sorted(row.index)
            if column.startswith("reference.")
            and column not in EXPECTED_VALUE_COLUMNS
            and not column.startswith("reference.forbidden_")
            and _is_expected_reference_column(column)
        ],
    ]:
        if _has_display_value(row.get(column)):
            columns.append(column)
            continue
        actual_value = _actual_value_for_reference(row, column)
        if _has_display_value(actual_value):
            columns.append(column)
    return columns


def build_expected_actual_case_table(
    examples: pd.DataFrame,
    testset_label: str,
    run_label: str,
) -> pd.DataFrame:
    """Build one-row-per-case expected-vs-actual table for a selected testset/run."""

    base_columns = ["query", "case_key"]
    if examples.empty:
        return pd.DataFrame(columns=base_columns)

    selected = examples[_testset_mask(examples, testset_label) & _run_mask(examples, run_label)]
    if selected.empty:
        return pd.DataFrame(columns=base_columns)

    sort_columns = [column for column in ["case_key", "inputs.question"] if column in selected.columns]
    if sort_columns:
        selected = selected.sort_values(sort_columns, kind="stable")

    records: list[dict[str, Any]] = []
    value_columns: list[str] = []
    for _, row in selected.iterrows():
        record: dict[str, Any] = {
            "query": _format_display_value(row.get("inputs.question")),
            "case_key": _format_display_value(row.get("case_key")),
        }
        for reference_column in _reference_columns_for_case_table(row):
            label = _case_table_label(reference_column)
            expected = _format_case_table_value(row.get(reference_column))
            actual = _format_case_table_value(
                _actual_case_table_value_for_reference(row, reference_column)
            )
            if expected == "-" and actual == "-":
                continue
            expected_column = f"정답 {label}"
            actual_column = f"모델 {label}"
            record[expected_column] = expected
            record[actual_column] = actual
            for column in (expected_column, actual_column):
                if column not in value_columns:
                    value_columns.append(column)
        for metric in METRIC_COLUMNS:
            if metric in row.index and _has_display_value(row.get(metric)):
                record[metric] = pd.to_numeric([row.get(metric)], errors="coerce")[0]
        records.append(record)

    if not records:
        return pd.DataFrame(columns=base_columns)
    frame = pd.DataFrame.from_records(records)
    metric_columns = [metric for metric in METRIC_COLUMNS if metric in frame.columns]
    ordered_value_columns = [column for column in value_columns if column in frame.columns]
    if ordered_value_columns:
        frame[ordered_value_columns] = frame[ordered_value_columns].fillna("-")
    return frame[["query", *ordered_value_columns, *metric_columns, "case_key"]]


CASE_TABLE_MATCH_STYLE = "background-color: #dcfce7; color: #166534;"
CASE_TABLE_MISMATCH_STYLE = "background-color: #fee2e2; color: #991b1b;"
CASE_TABLE_SCORE_METRICS = {
    "diagram": "diagram_id_hit",
    "허용 diagram": "diagram_id_hit",
    "near-miss diagram": "near_miss_not_above_expected",
    "location": "location_match",
    "party_type": "party_type_match",
    "chunk_type": "chunk_type_match",
    "keyword": "keyword_coverage",
}


def _case_table_style_for_pair(row: pd.Series, suffix: str, expected: str, actual: str) -> str:
    metric = CASE_TABLE_SCORE_METRICS.get(suffix)
    if metric is not None and metric in row.index:
        score = pd.to_numeric([row.get(metric)], errors="coerce")[0]
        if not pd.isna(score):
            return CASE_TABLE_MATCH_STYLE if float(score) >= 1 else CASE_TABLE_MISMATCH_STYLE
    return CASE_TABLE_MATCH_STYLE if actual == expected else CASE_TABLE_MISMATCH_STYLE


def build_expected_actual_case_table_styles(table: pd.DataFrame) -> pd.DataFrame:
    """Return cell styles for selected run values in expected/actual case tables."""

    styles = pd.DataFrame("", index=table.index, columns=table.columns)
    for actual_column in [column for column in table.columns if str(column).startswith("모델 ")]:
        suffix = str(actual_column).removeprefix("모델 ")
        expected_column = f"정답 {suffix}"
        if expected_column not in table.columns:
            continue
        for index, row in table.iterrows():
            expected = str(row.get(expected_column)).strip()
            actual = str(row.get(actual_column)).strip()
            if expected == "-" and actual == "-":
                continue
            style = _case_table_style_for_pair(row, suffix, expected, actual)
            styles.loc[index, expected_column] = style
            styles.loc[index, actual_column] = style
    return styles
