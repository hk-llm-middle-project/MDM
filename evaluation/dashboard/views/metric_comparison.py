"""Metric comparison view."""

from __future__ import annotations

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


def default_group_by_index(options: list[str]) -> int:
    if "nickname" in options:
        return options.index("nickname")
    if "retriever_reranker" in options:
        return options.index("retriever_reranker")
    if "run_label" in options:
        return options.index("run_label")
    return 0


def default_metric_selection(available: list[str]) -> list[str]:
    if not available:
        return []
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


def default_metric_selection(available: list[str]) -> list[str]:
    if not available:
        return []
    if "critical_error" in available:
        return ["critical_error"]
    return [available[0]]


def metric_caption(metric: str) -> str:
    caption = describe_metric(metric)
    if metric == "critical_error":
        return f"{caption} critical_error는 낮을수록 좋습니다."
    return caption


def render(summary: pd.DataFrame, metrics: pd.DataFrame) -> None:
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
        index=default_group_by_index(group_by_options),
        format_func=group_by_label,
    )
    selected_metrics = st.multiselect(
        "Metrics",
        available,
        default=default_metric_selection(available),
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
