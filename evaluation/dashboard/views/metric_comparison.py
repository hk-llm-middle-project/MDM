"""Metric comparison view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from evaluation.dashboard.charts import metric_bar_chart, parser_chunker_heatmap
from evaluation.dashboard.transforms import METRIC_COLUMNS


def _available_metrics(frame: pd.DataFrame) -> list[str]:
    return [metric for metric in METRIC_COLUMNS if metric in frame.columns]


def render(summary: pd.DataFrame, metrics: pd.DataFrame) -> None:
    st.subheader("Metric Comparison")
    available = _available_metrics(summary)
    if summary.empty or metrics.empty or not available:
        st.info("No metric data to compare.")
        return

    metric = st.selectbox(
        "Metric",
        available,
        index=available.index("critical_error") if "critical_error" in available else 0,
    )
    group_by_options = [
        column
        for column in [
            "run_label",
            "run_name",
            "loader_strategy",
            "chunker_strategy",
            "embedding_provider",
            "retriever_strategy",
            "reranker_strategy",
        ]
        if column in metrics.columns
    ]
    if not group_by_options:
        st.info("No grouping columns found.")
        return

    group_by = st.selectbox("Group by", group_by_options, index=0)
    st.altair_chart(
        metric_bar_chart(metrics, metric=metric, group_by=group_by),
        use_container_width=True,
    )
    if metric == "critical_error":
        st.caption("critical_error는 낮을수록 좋습니다.")


def render_matrix(summary: pd.DataFrame) -> None:
    st.subheader("Parser × Chunker Matrix")
    available = _available_metrics(summary)
    if summary.empty or not available:
        st.info("No summary data to display.")
        return

    metric = st.selectbox(
        "Heatmap metric",
        available,
        index=available.index("critical_error") if "critical_error" in available else 0,
    )
    st.altair_chart(
        parser_chunker_heatmap(summary, metric=metric),
        use_container_width=True,
    )
    if metric == "critical_error":
        st.caption("critical_error는 낮을수록 좋습니다.")
