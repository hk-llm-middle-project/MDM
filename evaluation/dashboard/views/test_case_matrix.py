"""Test-case by run metric matrix view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from evaluation.dashboard.transforms import METRIC_COLUMNS, build_case_metric_matrix


def render(examples: pd.DataFrame) -> None:
    st.subheader("Test Case Matrix")
    if examples.empty:
        st.info("No example-level results found.")
        return

    metrics = [metric for metric in METRIC_COLUMNS if metric in examples.columns]
    if not metrics:
        st.info("No example-level metric columns found.")
        return

    metric = st.selectbox(
        "Matrix metric",
        metrics,
        index=metrics.index("critical_error") if "critical_error" in metrics else 0,
    )
    matrix = build_case_metric_matrix(examples, metric=metric)
    if matrix.empty:
        st.info("No matrix data available for the selected metric.")
        return
    st.dataframe(matrix, use_container_width=True, hide_index=True)
