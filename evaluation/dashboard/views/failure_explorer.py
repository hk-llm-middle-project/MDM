"""Failure explorer view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from evaluation.dashboard.transforms import (
    METRIC_COLUMNS,
    build_failure_breakdown,
    filter_failed_examples,
)


def render(examples: pd.DataFrame) -> None:
    st.subheader("Failure Explorer")
    if examples.empty:
        st.info("No example-level results found.")
        return

    metrics = [metric for metric in METRIC_COLUMNS if metric in examples.columns]
    if not metrics:
        st.info("No example-level metric columns found.")
        return

    metric = st.selectbox(
        "Failure metric",
        metrics,
        index=metrics.index("critical_error") if "critical_error" in metrics else 0,
    )
    failed = filter_failed_examples(examples, metric)
    breakdown = build_failure_breakdown(failed)
    if not breakdown.empty:
        st.markdown("#### Failure breakdown")
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

    st.markdown("#### Failed rows")
    comment_column = f"{metric}_comment"
    display_columns = [
        column
        for column in [
            "case_key",
            "run_label",
            "run_name",
            "loader_strategy",
            "chunker_strategy",
            "embedding_provider",
            "retriever_strategy",
            "reranker_strategy",
            "inputs.question",
            metric,
            comment_column,
            "reference.expected_diagram_ids",
            "reference.expected_location",
            "reference.expected_party_type",
            "outputs.retrieved_metadata",
            "execution_time",
        ]
        if column in failed.columns
    ]

    st.caption(f"{len(failed)} failed rows")
    st.dataframe(
        failed[display_columns] if display_columns else failed,
        use_container_width=True,
        hide_index=True,
    )
