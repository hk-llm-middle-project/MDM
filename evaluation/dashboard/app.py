"""Streamlit dashboard for local LangSmith evaluation exports."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dashboard.loaders import DashboardResults, discover_result_sets, load_results
from evaluation.dashboard.transforms import filter_frame
from evaluation.dashboard.views import (
    case_detail,
    case_matrix,
    combo_compare,
    decision_suites,
    failure_explorer,
    metric_comparison,
    metric_matrix,
    overview,
)


DEFAULT_RESULT_ROOT = PROJECT_ROOT / "evaluation" / "results"
DEFAULT_RESULT_ROOT_INPUT = "evaluation/results"
HIDE_TEXT_INPUT_INSTRUCTIONS_CSS = """
<style>
div[data-testid="InputInstructions"] {
    display: none;
}
</style>
"""
TAB_LABELS = [
    "Overview",
    "Decision Suites",
    "Metrics",
    "Matrix",
    "Test Cases",
    "Failures",
    "Case Detail",
    "Compare",
]


@st.cache_data(show_spinner=False)
def cached_load_results(result_dir: str) -> DashboardResults:
    return load_results(Path(result_dir))


def resolve_result_root(result_root_value: str) -> Path:
    """Resolve dashboard result roots relative to the project root."""

    raw_value = (result_root_value or DEFAULT_RESULT_ROOT_INPUT).strip()
    result_root = Path(raw_value).expanduser()
    if result_root.is_absolute():
        return result_root
    return (PROJECT_ROOT / result_root).resolve()


def result_set_options(result_root: Path) -> list[tuple[str, Path]]:
    """Return single-select result folders under the configured root."""

    return [(result_set.label, result_set.path) for result_set in discover_result_sets(result_root)]


def display_result_set_path(result_path: Path, result_root: Path) -> str:
    """Display result-set paths without the common result root prefix."""

    try:
        relative = result_path.resolve().relative_to(result_root.resolve())
    except ValueError:
        return str(result_path)
    if str(relative) == ".":
        return "."
    return relative.as_posix()


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
        st.markdown(HIDE_TEXT_INPUT_INSTRUCTIONS_CSS, unsafe_allow_html=True)
        st.header("Filters")
        result_root_value = st.text_input("Result root", DEFAULT_RESULT_ROOT_INPUT)
        result_root = resolve_result_root(result_root_value)
        available_result_sets = result_set_options(result_root)
        if available_result_sets:
            result_set_labels = [label for label, _ in available_result_sets]
            selected_result_set = st.selectbox("Result set", result_set_labels, index=0)
            selected_result_path = dict(available_result_sets)[selected_result_set]
            st.caption(display_result_set_path(selected_result_path, result_root))
        else:
            selected_result_set = None
            selected_result_path = result_root
            st.info("No result sets found under the selected root.")
        if st.button("Reload"):
            cached_load_results.clear()

    results = cached_load_results(str(selected_result_path))
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
        selected_suites = st.multiselect(
            "Evaluation suite",
            option_values(examples, "evaluation_suite"),
            default=option_values(examples, "evaluation_suite"),
        )
        selected_difficulties = st.multiselect(
            "Difficulty",
            option_values(examples, "difficulty"),
            default=option_values(examples, "difficulty"),
        )
        selected_case_families = st.multiselect(
            "Case family",
            option_values(examples, "case_family"),
            default=option_values(examples, "case_family"),
        )

    summary = filter_frame(
        summary,
        selected_loaders,
        selected_chunkers,
        selected_embedders,
        selected_retrievers,
        selected_rerankers,
        evaluation_suite=selected_suites,
    )
    examples = filter_frame(
        examples,
        selected_loaders,
        selected_chunkers,
        selected_embedders,
        selected_retrievers,
        selected_rerankers,
        evaluation_suite=selected_suites,
        difficulty=selected_difficulties,
        case_family=selected_case_families,
    )
    metrics = filter_frame(
        metrics,
        selected_loaders,
        selected_chunkers,
        selected_embedders,
        selected_retrievers,
        selected_rerankers,
        evaluation_suite=selected_suites,
    )

    tabs = st.tabs(TAB_LABELS)
    with tabs[0]:
        overview.render(summary)
    with tabs[1]:
        decision_suites.render(summary, examples, metrics)
    with tabs[2]:
        metric_comparison.render(summary, metrics)
    with tabs[3]:
        metric_matrix.render(summary)
    with tabs[4]:
        case_matrix.render(examples)
    with tabs[5]:
        failure_explorer.render(examples)
    with tabs[6]:
        case_detail.render(examples)
    with tabs[7]:
        combo_compare.render(examples)


if __name__ == "__main__":
    main()
