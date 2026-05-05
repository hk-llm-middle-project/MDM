"""Decision-suite focused dashboard view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from evaluation.dashboard.charts import metric_bar_chart
from evaluation.dashboard.case_tables import build_case_metric_matrix, rows_for_case
from evaluation.dashboard.metrics import METRIC_COLUMNS, describe_metric
from evaluation.dashboard.transforms import filter_failed_examples


DECISION_SUITE_ORDER = (
    "intake",
    "router",
    "metadata_filter",
    "multiturn",
    "structured_output",
)
DECISION_SUITE_METRICS = {
    "intake": (
        "intake_overall",
        "intake_is_sufficient",
        "party_type_match",
        "location_match",
        "missing_fields_match",
        "follow_up_contains",
        "forbidden_filter_absent",
    ),
    "router": (
        "router_overall",
        "route_type_match",
        "reason_category_match",
    ),
    "metadata_filter": (
        "metadata_filter_overall",
        "metadata_filter_match",
        "forbidden_filter_absent",
    ),
    "multiturn": (
        "multiturn_overall",
        "state_sequence_match",
        "followup_questions_match",
        "final_metadata_match",
        "final_result_type_match",
        "turns_to_ready_match",
    ),
    "structured_output": (
        "structured_output_overall",
        "final_fault_ratio_match",
        "cannot_determine_match",
        "required_evidence_coverage",
        "party_role_coverage",
        "applicable_modifier_coverage",
        "non_applicable_modifier_coverage",
        "reference_diagram_hit",
    ),
}
ALL_DECISION_SUITE_METRICS = {
    metric
    for suite_metrics in DECISION_SUITE_METRICS.values()
    for metric in suite_metrics
}


def decision_suite_examples(examples: pd.DataFrame) -> pd.DataFrame:
    """Return only rows produced by evaluate_decision_suites.py."""

    if examples.empty or "evaluation_suite" not in examples.columns:
        return pd.DataFrame()
    return examples[
        examples["evaluation_suite"].astype(str).isin(DECISION_SUITE_ORDER)
    ].reset_index(drop=True)


def available_decision_suites(examples: pd.DataFrame) -> list[str]:
    """List available decision suites in a stable dashboard order."""

    filtered = decision_suite_examples(examples)
    if filtered.empty:
        return []
    present = set(filtered["evaluation_suite"].dropna().astype(str))
    ordered = [suite for suite in DECISION_SUITE_ORDER if suite in present]
    extras = sorted(present - set(DECISION_SUITE_ORDER))
    return [*ordered, *extras]


def available_metrics_for_suite(frame: pd.DataFrame, suite: str) -> list[str]:
    """Return suite-specific metrics first, followed by any other metric columns."""

    if len(frame.columns) == 0:
        return []
    preferred = [
        metric
        for metric in DECISION_SUITE_METRICS.get(suite, ())
        if metric in frame.columns
    ]
    extras = [
        metric
        for metric in METRIC_COLUMNS
        if metric in frame.columns
        and metric not in preferred
        and metric not in ALL_DECISION_SUITE_METRICS
    ]
    return [*preferred, *extras]


def default_metric_for_suite(suite: str) -> str:
    """Return the primary metric for a decision suite."""

    metrics = DECISION_SUITE_METRICS.get(suite)
    if metrics:
        return metrics[0]
    return "execution_time"


def _summary_for_suite(summary: pd.DataFrame, suite: str) -> pd.DataFrame:
    if summary.empty or "evaluation_suite" not in summary.columns:
        return pd.DataFrame()
    return summary[summary["evaluation_suite"].astype(str) == suite].reset_index(drop=True)


def _examples_for_suite(examples: pd.DataFrame, suite: str) -> pd.DataFrame:
    if examples.empty or "evaluation_suite" not in examples.columns:
        return pd.DataFrame()
    return examples[examples["evaluation_suite"].astype(str) == suite].reset_index(drop=True)


def _metrics_for_suite(metrics: pd.DataFrame, suite: str) -> pd.DataFrame:
    if metrics.empty or "evaluation_suite" not in metrics.columns:
        return pd.DataFrame()
    return metrics[metrics["evaluation_suite"].astype(str) == suite].reset_index(drop=True)


def _select_default_metric(metrics: list[str], suite: str) -> int:
    default = default_metric_for_suite(suite)
    if default in metrics:
        return metrics.index(default)
    return 0


def _display_summary_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in [
            "run_label",
            "run_name",
            "row_count",
            *[metric for metric in METRIC_COLUMNS if metric in frame.columns],
            "execution_time",
            "csv_path",
        ]
        if column in frame.columns
    ]


def _display_failure_columns(frame: pd.DataFrame, metric: str) -> list[str]:
    comment_column = f"{metric}_comment"
    return [
        column
        for column in [
            "case_key",
            "inputs.question",
            "run_label",
            metric,
            comment_column,
            "reference.expected_party_type",
            "reference.expected_location",
            "reference.expected_is_sufficient",
            "reference.expected_missing_fields",
            "outputs.party_type",
            "outputs.location",
            "outputs.is_sufficient",
            "outputs.missing_fields",
            "outputs.follow_up_questions",
            "execution_time",
        ]
        if column in frame.columns
    ]


def _render_suite_section(
    suite: str,
    summary: pd.DataFrame,
    examples: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    suite_summary = _summary_for_suite(summary, suite)
    suite_examples = _examples_for_suite(examples, suite)
    suite_metrics = _metrics_for_suite(metrics, suite)
    available_metrics = available_metrics_for_suite(suite_examples, suite)

    st.markdown(f"### `{suite}`")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Runs", len(suite_summary))
    col_b.metric("Rows", len(suite_examples))
    default_metric = default_metric_for_suite(suite)
    if default_metric in suite_summary.columns:
        score = pd.to_numeric(suite_summary[default_metric], errors="coerce").mean()
        col_c.metric(default_metric, "-" if pd.isna(score) else f"{score:.3f}")
    else:
        col_c.metric("Primary metric", default_metric)

    if suite_summary.empty or not available_metrics:
        st.info("No decision-suite metric data found.")
        return

    selected_metric = st.selectbox(
        "Metric",
        available_metrics,
        index=_select_default_metric(available_metrics, suite),
        key=f"decision_suite_metric_{suite}",
    )
    st.caption(describe_metric(selected_metric))

    if not suite_metrics.empty and {"run_label", "metric", "score"}.issubset(
        suite_metrics.columns
    ):
        st.altair_chart(
            metric_bar_chart(
                suite_metrics,
                metric=selected_metric,
                group_by="run_label",
                group_label="run",
            ),
            use_container_width=True,
        )

    st.markdown("#### Summary")
    summary_columns = _display_summary_columns(suite_summary)
    st.dataframe(
        suite_summary[summary_columns] if summary_columns else suite_summary,
        use_container_width=True,
        hide_index=True,
    )

    if selected_metric in suite_examples.columns:
        st.markdown("#### Test case matrix")
        matrix = build_case_metric_matrix(suite_examples, selected_metric)
        if matrix.empty:
            st.info("No case matrix data for this metric.")
        else:
            st.dataframe(matrix, use_container_width=True, hide_index=True)

        failed = filter_failed_examples(suite_examples, selected_metric)
        st.markdown("#### Failed rows")
        if failed.empty:
            st.success("No failed rows for this metric.")
        else:
            columns = _display_failure_columns(failed, selected_metric)
            st.caption(f"{len(failed)} failed rows")
            st.dataframe(
                failed[columns] if columns else failed,
                use_container_width=True,
                hide_index=True,
            )

    if not suite_examples.empty and "case_key" in suite_examples.columns:
        st.markdown("#### Case detail")
        case_options = (
            suite_examples["case_key"].dropna().astype(str).sort_values().unique().tolist()
        )
        if case_options:
            selected_case = st.selectbox(
                "Test case",
                case_options,
                key=f"decision_suite_case_{suite}",
            )
            rows = rows_for_case(suite_examples, selected_case)
            st.dataframe(rows, use_container_width=True, hide_index=True)


def render(summary: pd.DataFrame, examples: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Render one broad tab for all non-retrieval decision suites."""

    st.subheader("Decision Suites")
    suites = available_decision_suites(examples)
    if not suites:
        st.info("No intake/router/metadata_filter/multiturn/structured_output results found.")
        return

    decision_summary = summary[
        summary["evaluation_suite"].astype(str).isin(suites)
    ].reset_index(drop=True) if "evaluation_suite" in summary.columns else pd.DataFrame()

    if not decision_summary.empty:
        st.markdown("#### All decision-suite runs")
        columns = _display_summary_columns(decision_summary)
        st.dataframe(
            decision_summary[columns] if columns else decision_summary,
            use_container_width=True,
            hide_index=True,
        )

    tabs = st.tabs(suites)
    for suite, tab in zip(suites, tabs, strict=True):
        with tab:
            _render_suite_section(suite, summary, examples, metrics)
