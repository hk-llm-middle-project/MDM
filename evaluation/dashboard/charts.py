"""Chart helpers for the evaluation dashboard."""

from __future__ import annotations

import altair as alt
import pandas as pd


def metric_bar_chart(
    metric_frame: pd.DataFrame,
    metric: str,
    group_by: str,
) -> alt.Chart:
    data = metric_frame[metric_frame["metric"] == metric]
    if data.empty:
        data = pd.DataFrame({group_by: [], "score": []})
    grouped = (
        data.groupby(group_by, dropna=False, as_index=False)["score"]
        .mean()
        .sort_values("score", ascending=(metric == "critical_error"))
    )
    color_scheme = "orangered" if metric == "critical_error" else "tealblues"
    return (
        alt.Chart(grouped)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("score:Q", title="score", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(f"{group_by}:N", title=group_by, sort="-x"),
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
                scale=alt.Scale(
                    scheme="redyellowgreen",
                    domain=[0, 1],
                    reverse=(metric == "critical_error"),
                ),
            ),
            tooltip=[
                alt.Tooltip("loader_strategy:N", title="parser"),
                alt.Tooltip("chunker_strategy:N", title="chunker"),
                alt.Tooltip(f"{metric}:Q", title=metric, format=".3f"),
            ],
        )
        .properties(height=260)
    )
