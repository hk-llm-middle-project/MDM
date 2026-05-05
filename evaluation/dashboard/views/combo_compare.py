"""Run comparison for one test case."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from evaluation.dashboard.case_tables import (
    build_case_metric_comparison,
    build_case_value_comparison,
    case_question,
    compare_runs_for_case,
)


def render(examples: pd.DataFrame) -> None:
    st.subheader("Combo Compare")
    run_column = "run_label" if "run_label" in examples.columns else "run_name"
    if examples.empty or not {"case_key", run_column}.issubset(examples.columns):
        st.info("No example-level results found.")
        return

    case_options = examples["case_key"].dropna().astype(str).sort_values().unique().tolist()
    run_options = examples[run_column].dropna().astype(str).sort_values().unique().tolist()
    if not case_options or not run_options:
        st.info("No comparable cases or runs found.")
        return

    selected_case = st.selectbox("Test case", case_options, key="combo_compare_case")
    default_runs = run_options[: min(3, len(run_options))]
    selected_runs = st.multiselect(
        "Runs to compare",
        run_options,
        default=default_runs,
        max_selections=3,
        key="combo_compare_runs",
    )
    if not selected_runs:
        st.info("Select up to three runs to compare.")
        return

    rows = compare_runs_for_case(examples, selected_case, *selected_runs)
    if rows.empty:
        st.info("No rows found for the selected comparison.")
        return

    st.markdown("#### Query")
    st.write(case_question(rows))

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
