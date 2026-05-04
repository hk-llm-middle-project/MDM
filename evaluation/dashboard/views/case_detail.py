"""Case detail view across all runs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from evaluation.dashboard.transforms import (
    build_case_metric_comparison,
    build_case_value_comparison,
    case_question,
    rows_for_case,
)


def render(examples: pd.DataFrame) -> None:
    st.subheader("Case Detail")
    if examples.empty or "case_key" not in examples.columns:
        st.info("No cases available.")
        return

    case_options = examples["case_key"].dropna().astype(str).sort_values().unique().tolist()
    if not case_options:
        st.info("No cases available.")
        return

    selected_case = st.selectbox("Test case", case_options)
    rows = rows_for_case(examples, selected_case)
    if rows.empty:
        st.info("No rows found for the selected case.")
        return
    first = rows.iloc[0]

    st.markdown("#### Question")
    st.write(case_question(rows) or first.get("inputs.question", ""))

    value_comparison = build_case_value_comparison(rows)
    if not value_comparison.empty:
        st.markdown("#### Expected vs actual")
        st.caption("왼쪽은 testset의 기대값이고, 각 run 컬럼은 해당 run이 실제로 낸 값입니다.")
        st.dataframe(value_comparison, use_container_width=True, hide_index=True)

    metric_comparison = build_case_metric_comparison(rows)
    if not metric_comparison.empty:
        st.markdown("#### Metric scores")
        st.dataframe(metric_comparison, use_container_width=True, hide_index=True)

    with st.expander("Raw run rows"):
        st.dataframe(rows, use_container_width=True, hide_index=True)
