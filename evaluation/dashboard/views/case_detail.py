"""Case detail view across all runs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from evaluation.dashboard.transforms import METRIC_COLUMNS, rows_for_case


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
    st.write(first.get("inputs.question", ""))

    expected_columns = [
        column
        for column in [
            "reference.expected_diagram_ids",
            "reference.expected_location",
            "reference.expected_party_type",
            "reference.expected_chunk_types",
            "reference.expected_keywords",
        ]
        if column in rows.columns
    ]
    if expected_columns:
        st.markdown("#### Expected")
        st.dataframe(rows[expected_columns].head(1), use_container_width=True, hide_index=True)

    score_columns = [metric for metric in METRIC_COLUMNS if metric in rows.columns]
    display_columns = [
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
    st.markdown("#### Runs")
    st.dataframe(
        rows[display_columns] if display_columns else rows,
        use_container_width=True,
        hide_index=True,
    )
