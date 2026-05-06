"""Parser/chunker metric matrix view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from evaluation.dashboard.charts import parser_chunker_heatmap
from evaluation.dashboard.metrics import COMPARISON_METRIC_COLUMNS


def _available_metrics(frame: pd.DataFrame) -> list[str]:
    return [metric for metric in COMPARISON_METRIC_COLUMNS if metric in frame.columns]


def render(summary: pd.DataFrame) -> None:
    st.subheader("Parser × Chunker Matrix")
    available = _available_metrics(summary)
    if summary.empty or not available:
        st.info("No summary data to display.")
        return

    metric = st.selectbox(
        "Heatmap metric",
        available,
        index=available.index("critical_error") if "critical_error" in available else 0,
    )
    st.altair_chart(
        parser_chunker_heatmap(summary, metric=metric),
        use_container_width=True,
    )
    if metric == "critical_error":
        st.caption("critical_error는 낮을수록 좋습니다.")
