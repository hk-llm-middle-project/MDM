"""Pairwise run comparison for one test case."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from evaluation.dashboard.transforms import METRIC_COLUMNS, compare_runs_for_case


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
    col_left, col_right = st.columns(2)
    left_run = col_left.selectbox("Left run", run_options, key="combo_compare_left")
    right_index = 1 if len(run_options) > 1 else 0
    right_run = col_right.selectbox(
        "Right run", run_options, index=right_index, key="combo_compare_right"
    )

    rows = compare_runs_for_case(examples, selected_case, left_run, right_run)
    score_columns = [metric for metric in METRIC_COLUMNS if metric in rows.columns]
    columns = [
        column
        for column in [
            "run_label",
            "run_name",
            "loader_strategy",
            "chunker_strategy",
            "embedding_provider",
            "retriever_strategy",
            "reranker_strategy",
            *score_columns,
            "outputs.query",
            "outputs.retrieved",
            "outputs.retrieved_metadata",
            "outputs.contexts",
            "execution_time",
        ]
        if column in rows.columns
    ]
    st.dataframe(rows[columns] if columns else rows, use_container_width=True, hide_index=True)
