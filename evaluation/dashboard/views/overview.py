"""Overview view for local evaluation summaries."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from evaluation.dashboard.transforms import METRIC_COLUMNS, rank_combinations


def _available_metrics(frame: pd.DataFrame) -> list[str]:
    return [metric for metric in METRIC_COLUMNS if metric in frame.columns]


def _display_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in [
            "run_label",
            "run_name",
            "loader_strategy",
            "chunker_strategy",
            "embedding_provider",
            "retriever_strategy",
            "reranker_strategy",
            "row_count",
            *_available_metrics(frame),
        ]
        if column in frame.columns
    ]


def render(summary: pd.DataFrame) -> None:
    st.subheader("Overview")
    if summary.empty:
        st.info("No local evaluation results found.")
        return

    average_relevance = (
        summary["retrieval_relevance"].mean()
        if "retrieval_relevance" in summary.columns
        else None
    )
    average_critical = (
        summary["critical_error"].mean() if "critical_error" in summary.columns else None
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Experiments", len(summary))
    col_b.metric(
        "Combinations",
        summary["run_label"].nunique()
        if "run_label" in summary.columns
        else summary["combo"].nunique() if "combo" in summary.columns else len(summary),
    )
    col_c.metric(
        "Avg relevance",
        "-" if pd.isna(average_relevance) else f"{average_relevance:.3f}",
    )
    col_d.metric(
        "Avg critical error",
        "-" if pd.isna(average_critical) else f"{average_critical:.3f}",
    )

    if "critical_error" in summary.columns:
        st.markdown("#### Best 5 by critical error")
        best_critical = rank_combinations(summary, "critical_error").head(5)
        st.dataframe(
            best_critical[_display_columns(best_critical)],
            use_container_width=True,
            hide_index=True,
        )

    if "retrieval_relevance" in summary.columns:
        st.markdown("#### Best 5 by retrieval relevance")
        best_relevance = rank_combinations(summary, "retrieval_relevance").head(5)
        st.dataframe(
            best_relevance[_display_columns(best_relevance)],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Summary")
    columns = _display_columns(summary)
    st.dataframe(
        summary[columns] if columns else summary,
        use_container_width=True,
        hide_index=True,
    )
