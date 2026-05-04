"""Normalize local evaluation exports into dashboard-friendly tables."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import hashlib
import json
import re

import pandas as pd


METRIC_COLUMNS = (
    "diagram_id_hit",
    "location_match",
    "party_type_match",
    "chunk_type_match",
    "keyword_coverage",
    "near_miss_not_above_expected",
    "retrieval_relevance",
    "critical_error",
    "intake_is_sufficient",
    "missing_fields_match",
    "follow_up_contains",
    "forbidden_filter_absent",
    "intake_overall",
    "route_type_match",
    "reason_category_match",
    "router_overall",
    "metadata_filter_match",
    "metadata_filter_overall",
    "state_sequence_match",
    "followup_questions_match",
    "final_metadata_match",
    "final_result_type_match",
    "turns_to_ready_match",
    "multiturn_overall",
    "final_fault_ratio_match",
    "cannot_determine_match",
    "required_evidence_coverage",
    "party_role_coverage",
    "applicable_modifier_coverage",
    "non_applicable_modifier_coverage",
    "reference_diagram_hit",
    "structured_output_overall",
)
TIME_METRIC_COLUMNS = ("execution_time",)
COMPARISON_METRIC_COLUMNS = (*METRIC_COLUMNS, *TIME_METRIC_COLUMNS)
LOWER_IS_BETTER_METRICS = ("critical_error", *TIME_METRIC_COLUMNS)

METRIC_DESCRIPTIONS = {
    "diagram_id_hit": (
        "`diagram_id_hit` - 평가: 기대 diagram_id 또는 허용 diagram_id가 검색 결과 metadata에 "
        "포함됐는지 봅니다. 점수: 0=기대 diagram 미검색, 1=기대 diagram 검색."
    ),
    "location_match": (
        "`location_match` - 평가: 기대 사고 장소가 검색 결과 metadata에 포함됐는지 봅니다. "
        "점수: 0=장소 불일치, 1=장소 일치 또는 기대 장소 없음."
    ),
    "party_type_match": (
        "`party_type_match` - 평가: 기대 당사자 유형이 검색 결과 metadata에 포함됐는지 봅니다. "
        "점수: 0=당사자 유형 불일치, 1=당사자 유형 일치 또는 기대값 없음."
    ),
    "chunk_type_match": (
        "`chunk_type_match` - 평가: 기대 chunk type이 검색 결과 metadata에 포함됐는지 봅니다. "
        "점수: 0=chunk type 불일치, 1=chunk type 일치 또는 기대값 없음."
    ),
    "keyword_coverage": (
        "`keyword_coverage` - 평가: 기대 키워드가 검색 결과 본문에 얼마나 포함됐는지 봅니다. "
        "점수: 0=키워드 미포함, 1=모든 키워드 포함, 중간값=포함 비율."
    ),
    "near_miss_not_above_expected": (
        "`near_miss_not_above_expected` - 평가: near-miss diagram이 기대 diagram보다 위에 "
        "랭크되지 않았는지 봅니다. 점수: 0=near-miss가 우선되었거나 기대 diagram 없음, "
        "1=기대 diagram이 near-miss보다 먼저 나옴."
    ),
    "retrieval_relevance": (
        "`retrieval_relevance` - 평가: diagram/location/party/chunk/keyword 검색 체크의 평균입니다. "
        "점수: 0=관련 검색 신호 없음, 1=모든 검색 신호 충족, 중간값=평균 충족률."
    ),
    "critical_error": (
        "`critical_error` - 평가: 기대 diagram, party_type, location 중 치명적인 retrieval mismatch가 "
        "있는지 봅니다. 점수: 0=치명 오류 없음, 1=치명 오류 있음."
    ),
    "intake_is_sufficient": (
        "`intake_is_sufficient` - 평가: intake가 입력 충분/부족 여부를 기대값대로 판단했는지 봅니다. "
        "점수: 0=판단 불일치, 1=판단 일치."
    ),
    "missing_fields_match": (
        "`missing_fields_match` - 평가: intake가 찾아낸 누락 필드 목록이 기대값과 같은지 봅니다. "
        "점수: 0=누락 필드 불일치, 1=누락 필드 일치."
    ),
    "follow_up_contains": (
        "`follow_up_contains` - 평가: follow-up 질문에 기대 문구가 포함됐는지 봅니다. "
        "점수: 0=기대 문구 누락, 1=기대 문구 포함."
    ),
    "forbidden_filter_absent": (
        "`forbidden_filter_absent` - 평가: 만들면 안 되는 metadata filter가 빠져 있는지 봅니다. "
        "점수: 0=금지 filter 포함, 1=금지 filter 없음."
    ),
    "intake_overall": (
        "`intake_overall` - 평가: intake 관련 세부 점수의 평균입니다. "
        "점수: 0=intake 조건 전부 실패, 1=intake 조건 전부 통과, 중간값=평균 통과율."
    ),
    "route_type_match": (
        "`route_type_match` - 평가: router가 기대 route type을 선택했는지 봅니다. "
        "점수: 0=route 불일치, 1=route 일치."
    ),
    "reason_category_match": (
        "`reason_category_match` - 평가: router 판단 이유가 기대 reason category를 포함하는지 봅니다. "
        "점수: 0=이유 category 불일치, 1=이유 category 일치."
    ),
    "router_overall": (
        "`router_overall` - 평가: router 관련 세부 점수의 평균입니다. "
        "점수: 0=router 조건 전부 실패, 1=router 조건 전부 통과, 중간값=평균 통과율."
    ),
    "metadata_filter_match": (
        "`metadata_filter_match` - 평가: 생성된 metadata filter가 기대 filter와 같은지 봅니다. "
        "점수: 0=filter 불일치, 1=filter 일치."
    ),
    "metadata_filter_overall": (
        "`metadata_filter_overall` - 평가: metadata filter 일치와 금지 filter 부재 점수의 평균입니다. "
        "점수: 0=filter 조건 실패, 1=filter 조건 통과, 중간값=평균 통과율."
    ),
    "state_sequence_match": (
        "`state_sequence_match` - 평가: multiturn 대화의 turn별 intake state 변화가 기대 흐름과 "
        "맞는지 봅니다. 점수: 0=state 흐름 불일치, 1=state 흐름 일치."
    ),
    "followup_questions_match": (
        "`followup_questions_match` - 평가: turn별 follow-up 질문이 기대 내용을 포함하는지 봅니다. "
        "점수: 0=질문 흐름 불일치, 1=질문 흐름 일치."
    ),
    "final_metadata_match": (
        "`final_metadata_match` - 평가: multiturn 종료 시 최종 metadata가 기대값과 맞는지 봅니다. "
        "점수: 0=최종 metadata 불일치, 1=최종 metadata 일치."
    ),
    "final_result_type_match": (
        "`final_result_type_match` - 평가: multiturn 종료 결과 타입이 기대값과 맞는지 봅니다. "
        "점수: 0=최종 result type 불일치, 1=최종 result type 일치."
    ),
    "turns_to_ready_match": (
        "`turns_to_ready_match` - 평가: 분석 준비 상태가 된 turn 번호가 기대값과 맞는지 봅니다. "
        "점수: 0=준비 turn 불일치, 1=준비 turn 일치."
    ),
    "multiturn_overall": (
        "`multiturn_overall` - 평가: multiturn 관련 세부 점수의 평균입니다. "
        "점수: 0=multiturn 조건 전부 실패, 1=multiturn 조건 전부 통과, 중간값=평균 통과율."
    ),
    "final_fault_ratio_match": (
        "`final_fault_ratio_match` - 평가: 최종 과실비율이 기대 비율과 같은지 봅니다. "
        "점수: 0=과실비율 불일치, 1=과실비율 일치."
    ),
    "cannot_determine_match": (
        "`cannot_determine_match` - 평가: 판단 불가 여부가 기대값과 맞는지 봅니다. "
        "점수: 0=판단 가능/불가 상태 불일치, 1=상태 일치."
    ),
    "required_evidence_coverage": (
        "`required_evidence_coverage` - 평가: 답변/근거/metadata에 필수 evidence가 얼마나 포함됐는지 "
        "봅니다. 점수: 0=필수 evidence 없음, 1=필수 evidence 전부 포함, 중간값=포함 비율."
    ),
    "party_role_coverage": (
        "`party_role_coverage` - 평가: 기대 당사자 역할 설명이 답변/근거에 얼마나 포함됐는지 봅니다. "
        "점수: 0=역할 근거 없음, 1=역할 근거 전부 포함, 중간값=포함 비율."
    ),
    "applicable_modifier_coverage": (
        "`applicable_modifier_coverage` - 평가: 적용되어야 하는 수정요소가 답변/근거에 얼마나 "
        "포함됐는지 봅니다. 점수: 0=적용 modifier 없음, 1=전부 포함, 중간값=포함 비율."
    ),
    "non_applicable_modifier_coverage": (
        "`non_applicable_modifier_coverage` - 평가: 적용되지 않아야 하는 수정요소 설명이 답변/근거에 "
        "얼마나 포함됐는지 봅니다. 점수: 0=비적용 modifier 근거 없음, 1=전부 포함, 중간값=포함 비율."
    ),
    "reference_diagram_hit": (
        "`reference_diagram_hit` - 평가: structured output 생성 시 기대 reference diagram이 검색 "
        "context에 포함됐는지 봅니다. 점수: 0=reference diagram 미검색, 1=reference diagram 검색."
    ),
    "structured_output_overall": (
        "`structured_output_overall` - 평가: structured output 관련 세부 점수의 평균입니다. "
        "점수: 0=structured output 조건 전부 실패, 1=조건 전부 통과, 중간값=평균 통과율."
    ),
    "execution_time": (
        "`execution_time` - 평가: 각 run의 row별 실행시간 평균입니다. 단위: 초. "
        "낮을수록 같은 평가를 더 빠르게 처리한 run입니다."
    ),
}

CASE_METADATA_COLUMNS = (
    "evaluation_suite",
    "suite",
    "case_type_codes",
    "difficulty",
    "case_family",
    "inference_type",
    "query_style",
    "requires_diagram",
    "requires_table",
    "filter_risk",
)

SUMMARY_METADATA_COLUMNS = (
    "experiment_name",
    "dataset_name",
    "testset_path",
    "evaluation_suite",
    "run_name",
    "loader_strategy",
    "chunker_strategy",
    "embedding_provider",
    "retriever_strategy",
    "reranker_strategy",
    "ensemble_bm25_weight",
    "ensemble_candidate_k",
    "ensemble_use_chunk_id",
    "retriever_reranker",
    "row_count",
    "execution_time",
    "summary_path",
    "csv_path",
    "result_stem",
    "combo",
    "run_label",
)

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


def describe_metric(metric: str) -> str:
    return METRIC_DESCRIPTIONS.get(
        metric,
        f"`{metric}` - 평가: dashboard metric score입니다. 점수: 0=실패, 1=통과.",
    )


def make_combo(loader: Any, chunker: Any, embedder: Any) -> str:
    return f"{loader or '-'} / {chunker or '-'} / {embedder or '-'}"


def _numeric_weight(value: Any) -> float | None:
    if value is None:
        return None
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(weight):
        return None
    return weight


def _format_ensemble_weight_label(retriever: Any, bm25_weight: Any) -> str:
    retriever_name = str(retriever or "")
    weight = _numeric_weight(bm25_weight)
    if "ensemble" not in retriever_name or weight is None:
        return ""

    bm25_ratio = round(weight * 10, 1)
    dense_ratio = round((1 - weight) * 10, 1)
    if float(bm25_ratio).is_integer() and float(dense_ratio).is_integer():
        ratio = f"{int(bm25_ratio)}:{int(dense_ratio)}"
    else:
        ratio = f"{bm25_ratio:g}:{dense_ratio:g}"
    return f"BM25:Dense {ratio}"


def _result_stem_suffix(value: Any) -> str:
    stem = str(value or "").strip()
    if not stem:
        return "-"
    match = re.match(r"(\d{8}-\d{6})", stem)
    if match:
        return match.group(1)
    return stem


def _disambiguate_duplicate_run_labels(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not {"run_label", "result_stem"}.issubset(frame.columns):
        return frame

    disambiguated = frame.copy()
    stems_per_label = disambiguated.groupby("run_label")["result_stem"].transform(
        lambda values: values.dropna().astype(str).nunique()
    )
    duplicate_labels = stems_per_label > 1
    if not duplicate_labels.any():
        return disambiguated

    disambiguated.loc[duplicate_labels, "run_label"] = disambiguated.loc[
        duplicate_labels
    ].apply(
        lambda row: f"{row['run_label']} [{_result_stem_suffix(row.get('result_stem'))}]",
        axis=1,
    )
    return disambiguated


def make_run_label(
    run_name: Any,
    retriever: Any,
    reranker: Any,
    ensemble_bm25_weight: Any = None,
) -> str:
    name = str(run_name or "-")
    retriever_name = str(retriever or "")
    reranker_name = str(reranker or "")
    if not retriever_name and not reranker_name:
        return name
    label = f"{name} / {retriever_name or '-'} / {reranker_name or '-'}"
    weight_label = _format_ensemble_weight_label(retriever, ensemble_bm25_weight)
    if weight_label:
        return f"{label} / {weight_label}"
    return label


def _is_empty_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null"}
    if isinstance(value, (list, tuple, set, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parse_jsonish(value: Any) -> Any:
    if _is_empty_scalar(value):
        return None
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return None
    if text[0] not in "[{\"" and text.lower() not in {"true", "false", "null"}:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _has_display_value(value: Any) -> bool:
    parsed = _parse_jsonish(value)
    if parsed is None:
        return False
    if isinstance(parsed, (list, tuple, set, dict)):
        return len(parsed) > 0
    return not _is_empty_scalar(parsed)


def _format_display_value(value: Any) -> str:
    parsed = _parse_jsonish(value)
    if parsed is None:
        return ""
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    if isinstance(parsed, (list, tuple, set)):
        values = [_format_display_value(item) for item in parsed]
        return ", ".join(value for value in values if value)
    if isinstance(parsed, float) and parsed.is_integer():
        return str(int(parsed))
    return str(parsed)


def _format_score(value: Any) -> str:
    if not _has_display_value(value):
        return ""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return _format_display_value(value)
    if float(number).is_integer():
        return str(int(number))
    return f"{float(number):.4f}".rstrip("0").rstrip(".")


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


def _actual_value_for_reference(row: pd.Series, reference_column: str) -> str:
    metadata_key = REFERENCE_ACTUAL_METADATA_KEYS.get(reference_column)
    if metadata_key:
        metadata_value = _metadata_values(row, metadata_key)
        if metadata_value:
            return metadata_value

    if reference_column == "reference.expected_keywords":
        comment = row.get("keyword_coverage_comment") or row.get("feedback.keyword_coverage.comment")
        if _has_display_value(comment):
            return _format_display_value(comment)

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


def _bundle_value(bundle: Any, key: str, default: Any = None) -> Any:
    return getattr(bundle, "summary", {}).get(key, default)


def _mean_execution_time(bundle: Any) -> float | None:
    csv_path = getattr(bundle, "csv_path", None)
    if csv_path is None:
        return None
    try:
        frame = pd.read_csv(csv_path, usecols=["execution_time"])
    except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None

    values = pd.to_numeric(frame["execution_time"], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def make_retriever_reranker(
    retriever: Any,
    reranker: Any,
    ensemble_bm25_weight: Any = None,
) -> str:
    retriever_value = str(retriever or "unknown")
    reranker_value = str(reranker or "unknown")
    label = f"{retriever_value} / {reranker_value}"
    weight_label = _format_ensemble_weight_label(retriever, ensemble_bm25_weight)
    if weight_label:
        return f"{label} / {weight_label}"
    return label


def build_summary_frame(bundles: Iterable[Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for bundle in bundles:
        summary = getattr(bundle, "summary", {})
        metrics = summary.get("metrics") or {}
        loader = summary.get("loader_strategy")
        chunker = summary.get("chunker_strategy")
        embedder = summary.get("embedding_provider")
        retriever = summary.get("retriever_strategy")
        reranker = summary.get("reranker_strategy")
        ensemble_bm25_weight = summary.get("ensemble_bm25_weight")
        run_name = summary.get("run_name") or getattr(bundle, "run_name", None)
        record: dict[str, Any] = {
            "experiment_name": summary.get("experiment_name"),
            "dataset_name": summary.get("dataset_name"),
            "testset_path": summary.get("testset_path"),
            "evaluation_suite": summary.get("evaluation_suite") or summary.get("suite"),
            "run_name": run_name,
            "loader_strategy": loader,
            "chunker_strategy": chunker,
            "embedding_provider": embedder,
            "retriever_strategy": retriever,
            "reranker_strategy": reranker,
            "ensemble_bm25_weight": ensemble_bm25_weight,
            "ensemble_candidate_k": summary.get("ensemble_candidate_k"),
            "ensemble_use_chunk_id": summary.get("ensemble_use_chunk_id"),
            "retriever_reranker": make_retriever_reranker(
                retriever,
                reranker,
                ensemble_bm25_weight,
            ),
            "row_count": summary.get("row_count"),
            "execution_time": _mean_execution_time(bundle),
            "summary_path": str(getattr(bundle, "summary_path", "")),
            "csv_path": str(getattr(bundle, "csv_path", "") or ""),
            "result_stem": getattr(bundle, "result_stem", None),
            "combo": make_combo(loader, chunker, embedder),
            "run_label": make_run_label(
                run_name,
                retriever,
                reranker,
                ensemble_bm25_weight,
            ),
        }
        for metric_name in METRIC_COLUMNS:
            record[metric_name] = metrics.get(metric_name)
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return pd.DataFrame(columns=list(SUMMARY_METADATA_COLUMNS) + list(METRIC_COLUMNS))
    for metric_name in METRIC_COLUMNS:
        frame[metric_name] = pd.to_numeric(frame[metric_name], errors="coerce")
    for metric_name in TIME_METRIC_COLUMNS:
        frame[metric_name] = pd.to_numeric(frame[metric_name], errors="coerce")
    return _disambiguate_duplicate_run_labels(frame)


def _read_bundle_csv(bundle: Any) -> pd.DataFrame | None:
    csv_path = getattr(bundle, "csv_path", None)
    if csv_path is None:
        return None
    try:
        return pd.read_csv(csv_path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def _normalized_question(value: Any) -> str:
    return str(value or "").strip()


def _question_case_key(question: str) -> str:
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
    return f"question:{digest}"


def make_case_key(row: pd.Series) -> str:
    """Return a stable dashboard key for one evaluated test case."""

    example_id = row.get("example_id")
    if pd.notna(example_id) and str(example_id).strip():
        return str(example_id).strip()
    return _question_case_key(_normalized_question(row.get("inputs.question", "")))


def reconcile_cross_run_case_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse export-specific IDs when one question appears once per run."""

    if frame.empty or not {"case_key", "inputs.question"}.issubset(frame.columns):
        return frame

    reconciled = frame.copy()
    normalized_question = reconciled["inputs.question"].map(_normalized_question)
    for question, group in reconciled[normalized_question != ""].groupby(
        normalized_question[normalized_question != ""], dropna=False
    ):
        if group["case_key"].nunique(dropna=False) <= 1:
            continue
        if "run_name" in group.columns and group.groupby("run_name", dropna=False).size().max() > 1:
            continue
        reconciled.loc[group.index, "case_key"] = _question_case_key(str(question))
    return reconciled


def build_example_frame(bundles: Iterable[Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for bundle in bundles:
        frame = _read_bundle_csv(bundle)
        if frame is None:
            continue

        summary = getattr(bundle, "summary", {})
        loader = summary.get("loader_strategy")
        chunker = summary.get("chunker_strategy")
        embedder = summary.get("embedding_provider")
        retriever = summary.get("retriever_strategy")
        reranker = summary.get("reranker_strategy")
        ensemble_bm25_weight = summary.get("ensemble_bm25_weight")
        run_name = summary.get("run_name") or getattr(bundle, "run_name", None)
        frame = frame.copy()
        frame["run_name"] = run_name
        frame["loader_strategy"] = loader
        frame["chunker_strategy"] = chunker
        frame["embedding_provider"] = embedder
        frame["retriever_strategy"] = retriever
        frame["reranker_strategy"] = reranker
        frame["ensemble_bm25_weight"] = ensemble_bm25_weight
        frame["ensemble_candidate_k"] = summary.get("ensemble_candidate_k")
        frame["ensemble_use_chunk_id"] = summary.get("ensemble_use_chunk_id")
        frame["retriever_reranker"] = make_retriever_reranker(
            retriever,
            reranker,
            ensemble_bm25_weight,
        )
        frame["combo"] = make_combo(loader, chunker, embedder)
        frame["run_label"] = make_run_label(
            run_name,
            retriever,
            reranker,
            ensemble_bm25_weight,
        )
        frame["summary_path"] = str(getattr(bundle, "summary_path", ""))
        frame["csv_path"] = str(getattr(bundle, "csv_path", "") or "")
        frame["result_stem"] = getattr(bundle, "result_stem", None)
        if "evaluation_suite" not in frame.columns:
            frame["evaluation_suite"] = summary.get("evaluation_suite") or summary.get("suite")
        for column in CASE_METADATA_COLUMNS:
            metadata_column = f"metadata.{column}"
            if column not in frame.columns and metadata_column in frame.columns:
                frame[column] = frame[metadata_column]
        if "evaluation_suite" in frame.columns and "suite" in frame.columns:
            frame["evaluation_suite"] = frame["evaluation_suite"].fillna(frame["suite"])
        elif "evaluation_suite" not in frame.columns and "suite" in frame.columns:
            frame["evaluation_suite"] = frame["suite"]

        frame["case_key"] = frame.apply(make_case_key, axis=1)
        for metric_name in METRIC_COLUMNS:
            feedback_column = f"feedback.{metric_name}"
            comment_column = f"feedback.{metric_name}.comment"
            if feedback_column in frame.columns:
                frame[metric_name] = pd.to_numeric(frame[feedback_column], errors="coerce")
            elif metric_name in frame.columns:
                frame[metric_name] = pd.to_numeric(frame[metric_name], errors="coerce")
            if comment_column in frame.columns:
                frame[f"{metric_name}_comment"] = frame[comment_column].fillna("")
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    examples = reconcile_cross_run_case_keys(pd.concat(frames, ignore_index=True))
    return _disambiguate_duplicate_run_labels(examples)


def build_metric_frame(summary_frame: pd.DataFrame) -> pd.DataFrame:
    if summary_frame.empty:
        return pd.DataFrame(
            columns=[
                "run_name",
                "loader_strategy",
                "chunker_strategy",
                "embedding_provider",
                "retriever_strategy",
                "reranker_strategy",
                "ensemble_bm25_weight",
                "ensemble_candidate_k",
                "ensemble_use_chunk_id",
                "retriever_reranker",
                "combo",
                "run_label",
                "metric",
                "score",
            ]
        )

    summary_frame = summary_frame.copy()
    if (
        "retriever_reranker" not in summary_frame.columns
        and {"retriever_strategy", "reranker_strategy"}.issubset(summary_frame.columns)
    ):
        summary_frame["retriever_reranker"] = summary_frame.apply(
            lambda row: make_retriever_reranker(
                row.get("retriever_strategy"),
                row.get("reranker_strategy"),
                row.get("ensemble_bm25_weight"),
            ),
            axis=1,
        )

    id_columns = [
        "evaluation_suite",
        "run_name",
        "loader_strategy",
        "chunker_strategy",
        "embedding_provider",
        "retriever_strategy",
        "reranker_strategy",
        "ensemble_bm25_weight",
        "ensemble_candidate_k",
        "ensemble_use_chunk_id",
        "retriever_reranker",
        "combo",
        "run_label",
    ]
    id_columns = [column for column in id_columns if column in summary_frame.columns]
    value_columns = [
        column for column in COMPARISON_METRIC_COLUMNS if column in summary_frame.columns
    ]
    metric_frame = summary_frame.melt(
        id_vars=id_columns,
        value_vars=value_columns,
        var_name="metric",
        value_name="score",
    )
    metric_frame["score"] = pd.to_numeric(metric_frame["score"], errors="coerce")
    return metric_frame.dropna(subset=["score"]).reset_index(drop=True)


def filter_frame(
    frame: pd.DataFrame,
    loader_strategy: list[str],
    chunker_strategy: list[str],
    embedding_provider: list[str],
    retriever_strategy: list[str] | None = None,
    reranker_strategy: list[str] | None = None,
    evaluation_suite: list[str] | None = None,
    difficulty: list[str] | None = None,
    case_family: list[str] | None = None,
) -> pd.DataFrame:
    filtered = frame
    if loader_strategy and "loader_strategy" in filtered.columns:
        filtered = filtered[_matches_filter_or_empty(filtered["loader_strategy"], loader_strategy)]
    if chunker_strategy and "chunker_strategy" in filtered.columns:
        filtered = filtered[_matches_filter_or_empty(filtered["chunker_strategy"], chunker_strategy)]
    if embedding_provider and "embedding_provider" in filtered.columns:
        filtered = filtered[_matches_filter_or_empty(filtered["embedding_provider"], embedding_provider)]
    if retriever_strategy and "retriever_strategy" in filtered.columns:
        filtered = filtered[_matches_filter_or_empty(filtered["retriever_strategy"], retriever_strategy)]
    if reranker_strategy and "reranker_strategy" in filtered.columns:
        filtered = filtered[_matches_filter_or_empty(filtered["reranker_strategy"], reranker_strategy)]
    if evaluation_suite and "evaluation_suite" in filtered.columns:
        filtered = filtered[filtered["evaluation_suite"].isin(evaluation_suite)]
    if difficulty and "difficulty" in filtered.columns:
        filtered = filtered[filtered["difficulty"].isin(difficulty)]
    if case_family and "case_family" in filtered.columns:
        filtered = filtered[filtered["case_family"].isin(case_family)]
    return filtered.reset_index(drop=True)


def _matches_filter_or_empty(series: pd.Series, selected: list[str]) -> pd.Series:
    """Keep strategy-less suite rows when parser/retriever filters are active."""

    text_values = series.astype("string")
    return series.isna() | text_values.isin(selected) | (text_values.fillna("") == "")


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


def filter_failed_examples(examples: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return example rows that fail the selected metric."""

    if examples.empty or metric not in examples.columns:
        return pd.DataFrame()
    scores = pd.to_numeric(examples[metric], errors="coerce")
    if metric == "critical_error":
        return examples[scores > 0].reset_index(drop=True)
    return examples[scores < 1].reset_index(drop=True)


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


def rank_combinations(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Sort combinations best-first for the selected metric."""

    if summary.empty or metric not in summary.columns:
        return pd.DataFrame()
    ranked = summary.copy()
    ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
    ascending = metric in LOWER_IS_BETTER_METRICS
    return ranked.sort_values(metric, ascending=ascending).reset_index(drop=True)


def build_failure_breakdown(failed: pd.DataFrame) -> pd.DataFrame:
    """Count failed rows by strategy and run for systematic failure analysis."""

    group_columns = [
        "evaluation_suite",
        "suite",
        "case_type_codes",
        "difficulty",
        "case_family",
        "inference_type",
        "query_style",
        "loader_strategy",
        "chunker_strategy",
        "embedding_provider",
        "retriever_strategy",
        "reranker_strategy",
        "run_name",
        "run_label",
    ]
    available_group_columns = [column for column in group_columns if column in failed.columns]
    output_columns = [*available_group_columns, "failed_count"]
    if failed.empty or not available_group_columns:
        return pd.DataFrame(columns=output_columns)
    return (
        failed.groupby(available_group_columns, dropna=False)
        .size()
        .reset_index(name="failed_count")
        .sort_values("failed_count", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
