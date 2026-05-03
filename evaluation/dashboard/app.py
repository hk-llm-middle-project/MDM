"""Streamlit dashboard for local LangSmith evaluation exports."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dashboard.loaders import DashboardResults, load_results
from evaluation.dashboard.transforms import filter_frame
from evaluation.dashboard.views import (
    case_detail,
    combo_compare,
    failure_explorer,
    metric_comparison,
    overview,
    test_case_matrix,
)


DEFAULT_RESULT_DIR = PROJECT_ROOT / "evaluation" / "results" / "langsmith"
TAB_LABELS = ["Overview", "Metrics", "Matrix", "Test Cases", "Failures", "Case Detail", "Compare"]


@st.cache_data(show_spinner=False)
def cached_load_results(result_dir: str) -> DashboardResults:
    return load_results(Path(result_dir))


def option_values(frame: pd.DataFrame, column: str) -> list[str]:
    if frame.empty or column not in frame.columns:
        return []
    values = frame[column].dropna().astype(str).unique().tolist()
    return sorted(values)


def main() -> None:
    st.set_page_config(
        page_title="MDM Evaluation Dashboard",
        page_icon=None,
        layout="wide",
    )
    st.title("MDM Evaluation Dashboard")

    with st.sidebar:
        st.header("Filters")
        result_dir = st.text_input("Result directory", str(DEFAULT_RESULT_DIR))
        if st.button("Reload"):
            cached_load_results.clear()

    results = cached_load_results(result_dir)
    for warning in results.warnings:
        st.warning(warning)

    summary = results.summary
    examples = results.examples
    metrics = results.metrics

    with st.sidebar:
        selected_loaders = st.multiselect(
            "Parser",
            option_values(summary, "loader_strategy"),
            default=option_values(summary, "loader_strategy"),
        )
        selected_chunkers = st.multiselect(
            "Chunker",
            option_values(summary, "chunker_strategy"),
            default=option_values(summary, "chunker_strategy"),
        )
        selected_embedders = st.multiselect(
            "Embedder",
            option_values(summary, "embedding_provider"),
            default=option_values(summary, "embedding_provider"),
        )
        selected_retrievers = st.multiselect(
            "Retriever",
            option_values(summary, "retriever_strategy"),
            default=option_values(summary, "retriever_strategy"),
        )
        selected_rerankers = st.multiselect(
            "Reranker",
            option_values(summary, "reranker_strategy"),
            default=option_values(summary, "reranker_strategy"),
        )

    summary = filter_frame(
        summary,
        selected_loaders,
        selected_chunkers,
        selected_embedders,
        selected_retrievers,
        selected_rerankers,
    )
    examples = filter_frame(
        examples,
        selected_loaders,
        selected_chunkers,
        selected_embedders,
        selected_retrievers,
        selected_rerankers,
    )
    metrics = filter_frame(
        metrics,
        selected_loaders,
        selected_chunkers,
        selected_embedders,
        selected_retrievers,
        selected_rerankers,
    )

    tabs = st.tabs(TAB_LABELS)
    with tabs[0]:
        overview.render(summary)
    with tabs[1]:
        metric_comparison.render(summary, metrics)
    with tabs[2]:
        metric_comparison.render_matrix(summary)
    with tabs[3]:
        test_case_matrix.render(examples)
    with tabs[4]:
        failure_explorer.render(examples)
    with tabs[5]:
        case_detail.render(examples)
    with tabs[6]:
        combo_compare.render(examples)


if __name__ == "__main__":
    main()
