"""Case detail view across all runs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from evaluation.dashboard.case_tables import (
    build_expected_actual_case_table,
    build_expected_actual_case_table_styles,
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

    testset_column = "testset_label" if "testset_label" in examples.columns else "evaluation_suite"
    if testset_column not in examples.columns:
        st.info("No testset labels available.")
        return

    testset_options = (
        examples[testset_column].dropna().astype(str).sort_values().unique().tolist()
    )
    if not testset_options:
        st.info("No testsets available.")
        return

    selected_testset = st.selectbox(
        "Test set",
        testset_options,
        key="case_detail_testset",
    )
    testset_rows = examples[
        examples[testset_column].astype(str) == str(selected_testset)
    ].reset_index(drop=True)

    run_column = "run_label" if "run_label" in testset_rows.columns else "run_name"
    if run_column not in testset_rows.columns:
        st.info("No evaluation runs available.")
        return
    run_options = testset_rows[run_column].dropna().astype(str).sort_values().unique().tolist()
    if not run_options:
        st.info("No evaluation runs available.")
        return

    selected_run = st.selectbox(
        "Evaluation run",
        run_options,
        key="case_detail_run",
    )

    table = build_expected_actual_case_table(
        testset_rows,
        testset_label=str(selected_testset),
        run_label=str(selected_run),
    )
    if table.empty:
        st.info("No expected/actual rows found for this testset and run.")
    else:
        st.markdown("#### Expected vs actual for all cases")
        st.caption("query별로 testset 정답과 선택한 evaluation run의 모델 출력을 바로 비교합니다.")
        styles = build_expected_actual_case_table_styles(table)
        styled_table = table.style.apply(lambda _: styles, axis=None)
        st.dataframe(styled_table, use_container_width=True, hide_index=True)

    st.markdown("#### Single case drill-down")
    case_options = (
        testset_rows["case_key"].dropna().astype(str).sort_values().unique().tolist()
    )
    if not case_options:
        st.info("No cases available.")
        return

    selected_case = st.selectbox("Test case", case_options, key="case_detail_case")
    rows = rows_for_case(testset_rows, selected_case)
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
