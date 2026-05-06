"""Chart helpers for the evaluation dashboard."""

from __future__ import annotations

import altair as alt
import pandas as pd


LOWER_IS_BETTER_METRICS = {"critical_error", "execution_time"}
GROUP_AXIS_LABEL_LIMIT = 360


def _metric_axis(metric: str) -> alt.X:
    if metric == "execution_time":
        return alt.X("score:Q", title="seconds")
    return alt.X("score:Q", title="score", scale=alt.Scale(domain=[0, 1]))


def _heatmap_scale(metric: str) -> alt.Scale:
    if metric == "execution_time":
        return alt.Scale(scheme="redyellowgreen", reverse=True)
    return alt.Scale(
        scheme="redyellowgreen",
        domain=[0, 1],
        reverse=(metric in LOWER_IS_BETTER_METRICS),
    )


def _y_sort(metric: str) -> str:
    if metric in LOWER_IS_BETTER_METRICS:
        return "x"
    return "-x"


def _group_axis(group_by: str, metric: str, group_label: str | None = None) -> alt.Y:
    return alt.Y(
        f"{group_by}:N",
        title=None,
        sort=_y_sort(metric),
        axis=alt.Axis(labelLimit=GROUP_AXIS_LABEL_LIMIT),
    )


def metric_bar_chart(
    metric_frame: pd.DataFrame,
    metric: str,
    group_by: str,
    group_label: str | None = None,
) -> alt.Chart:
    data = metric_frame[metric_frame["metric"] == metric]
    if data.empty:
        data = pd.DataFrame({group_by: [], "score": []})
    grouped = (
        data.groupby(group_by, dropna=False, as_index=False)["score"]
        .mean()
        .sort_values("score", ascending=(metric in LOWER_IS_BETTER_METRICS))
    )
    color_scheme = "orangered" if metric in LOWER_IS_BETTER_METRICS else "tealblues"
    return (
        alt.Chart(grouped)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=_metric_axis(metric),
            y=_group_axis(group_by, metric, group_label),
            tooltip=[
                alt.Tooltip(f"{group_by}:N", title=group_by),
                alt.Tooltip("score:Q", title=metric, format=".3f"),
            ],
            color=alt.Color("score:Q", scale=alt.Scale(scheme=color_scheme)),
        )
        .properties(height=max(220, 28 * max(len(grouped), 1)))
    )


def parser_chunker_heatmap(
    summary_frame: pd.DataFrame,
    metric: str,
) -> alt.Chart:
    if summary_frame.empty or metric not in summary_frame.columns:
        data = pd.DataFrame(
            {
                "loader_strategy": [],
                "chunker_strategy": [],
                metric: [],
            }
        )
    else:
        data = (
            summary_frame.groupby(
                ["loader_strategy", "chunker_strategy"],
                dropna=False,
                as_index=False,
            )[metric]
            .mean()
            .dropna(subset=[metric])
        )

    return (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X("chunker_strategy:N", title="chunker"),
            y=alt.Y("loader_strategy:N", title="parser"),
            color=alt.Color(
                f"{metric}:Q",
                title=metric,
                scale=_heatmap_scale(metric),
            ),
            tooltip=[
                alt.Tooltip("loader_strategy:N", title="parser"),
                alt.Tooltip("chunker_strategy:N", title="chunker"),
                alt.Tooltip(f"{metric}:Q", title=metric, format=".3f"),
            ],
        )
        .properties(height=260)
    )
