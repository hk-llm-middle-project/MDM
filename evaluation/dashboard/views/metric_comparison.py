"""Metric comparison view."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import re

import pandas as pd
import streamlit as st

from evaluation.dashboard.charts import metric_bar_chart
from evaluation.dashboard.metrics import COMPARISON_METRIC_COLUMNS, describe_metric


GROUP_BY_COLUMNS = (
    "nickname",
    "retriever_strategy",
    "reranker_strategy",
    "retriever_reranker",
    "run_label",
    "run_name",
    "loader_strategy",
    "chunker_strategy",
    "embedding_provider",
)
GROUP_BY_LABELS = {
    "nickname": "nickname",
    "retriever_strategy": "retriever",
    "reranker_strategy": "reranker",
    "retriever_reranker": "retriever + reranker",
}
SUITE_DEFAULT_METRICS = {
    "retrieval": (
        "diagram_id_hit",
        "retrieval_relevance",
        "critical_error",
        "keyword_coverage",
        "near_miss_not_above_expected",
        "execution_time",
    ),
    "intake": (
        "intake_is_sufficient",
        "party_type_match",
        "location_match",
        "missing_fields_match",
        "follow_up_contains",
        "forbidden_filter_absent",
        "intake_overall",
    ),
    "router": ("route_type_match", "reason_category_match", "router_overall"),
    "metadata_filter": (
        "metadata_filter_match",
        "forbidden_filter_absent",
        "metadata_filter_overall",
    ),
    "multiturn": (
        "state_sequence_match",
        "followup_questions_match",
        "final_metadata_match",
        "final_result_type_match",
        "turns_to_ready_match",
        "multiturn_overall",
    ),
    "structured_output": (
        "final_fault_ratio_match",
        "cannot_determine_match",
        "required_evidence_coverage",
        "party_role_coverage",
        "applicable_modifier_coverage",
        "non_applicable_modifier_coverage",
        "reference_diagram_hit",
        "structured_output_overall",
    ),
}
SUITE_DEFAULT_GROUP_BY = {
    "retrieval": "nickname",
    "intake": "nickname",
    "router": "nickname",
    "metadata_filter": "nickname",
    "multiturn": "nickname",
    "structured_output": "nickname",
}

MetricConfig = Mapping[str, Any]


def _result_set_profile(result_set_label: str | None) -> str | None:
    if not result_set_label:
        return None
    normalized = result_set_label.strip().lower().replace("-", "_")
    parts = [part for part in re.split(r"[/\\]+", normalized) if part]
    for key in SUITE_DEFAULT_METRICS:
        if normalized == key or key in parts:
            return key
    return None


def _available_metrics(frame: pd.DataFrame) -> list[str]:
    return [metric for metric in COMPARISON_METRIC_COLUMNS if metric in frame.columns]


def available_group_by_options(frame: pd.DataFrame) -> list[str]:
    options: list[str] = []
    for column in GROUP_BY_COLUMNS:
        if column not in frame.columns:
            continue
        if column == "nickname":
            values = frame[column].dropna().astype(str).str.strip()
            if not values[~values.str.lower().isin({"", "nan", "none", "null"})].empty:
                options.append(column)
            continue
        options.append(column)
    return options


def group_by_label(column: str) -> str:
    return GROUP_BY_LABELS.get(column, column)


def _configured_group_by(config: MetricConfig | None) -> str | None:
    value = (config or {}).get("group_by")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _configured_metrics(config: MetricConfig | None) -> list[str]:
    values = (config or {}).get("metrics")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return [value for value in values if isinstance(value, str) and value.strip()]


def default_group_by_index(
    options: list[str],
    result_set_label: str | None = None,
    config: MetricConfig | None = None,
) -> int:
    preferred = _configured_group_by(config)
    if preferred in options:
        return options.index(preferred)
    profile = _result_set_profile(result_set_label)
    preferred = SUITE_DEFAULT_GROUP_BY.get(profile or "")
    if preferred in options:
        return options.index(preferred)
    if "nickname" in options:
        return options.index("nickname")
    if "retriever_reranker" in options:
        return options.index("retriever_reranker")
    if "run_label" in options:
        return options.index("run_label")
    return 0


def default_metric_selection(
    available: list[str],
    result_set_label: str | None = None,
    config: MetricConfig | None = None,
) -> list[str]:
    if not available:
        return []
    configured = [metric for metric in _configured_metrics(config) if metric in available]
    if configured:
        return configured
    profile = _result_set_profile(result_set_label)
    if profile:
        selected = [metric for metric in SUITE_DEFAULT_METRICS[profile] if metric in available]
        if selected:
            return selected
    if "critical_error" in available:
        return ["critical_error"]
    return [available[0]]


def metric_caption(metric: str) -> str:
    caption = describe_metric(metric)
    if metric == "critical_error":
        return f"{caption} critical_error는 낮을수록 좋습니다."
    if metric == "execution_time":
        return f"{caption} execution_time은 낮을수록 좋습니다."
    return caption


def render(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    result_set_label: str | None = None,
    config: MetricConfig | None = None,
) -> None:
    st.subheader("Metric Comparison")
    available = _available_metrics(summary)
    if summary.empty or metrics.empty or not available:
        st.info("No metric data to compare.")
        return

    group_by_options = available_group_by_options(metrics)
    if not group_by_options:
        st.info("No grouping columns found.")
        return

    group_by = st.selectbox(
        "Group by",
        group_by_options,
        index=default_group_by_index(group_by_options, result_set_label, config),
        format_func=group_by_label,
    )
    selected_metrics = st.multiselect(
        "Metrics",
        available,
        default=default_metric_selection(available, result_set_label, config),
    )
    if not selected_metrics:
        st.info("No metrics selected.")
        return

    for metric in selected_metrics:
        st.markdown(f"#### `{metric}`")
        st.caption(metric_caption(metric))
        st.altair_chart(
            metric_bar_chart(
                metrics,
                metric=metric,
                group_by=group_by,
                group_label=group_by_label(group_by),
            ),
            use_container_width=True,
        )
