"""Suite dispatch for decision-suite evaluation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from evaluation.decision_eval.suites import (
    evaluate_intake_rows,
    evaluate_metadata_filter_rows,
    evaluate_multiturn_rows,
    evaluate_router_rows,
    evaluate_structured_output_rows,
)


def evaluate_suite(suite: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    if suite == "intake":
        return evaluate_intake_rows(rows)
    if suite == "router":
        from rag.service.conversation.router import route_conversation_turn

        return evaluate_router_rows(rows, router=route_conversation_turn)
    if suite == "metadata_filter":
        return evaluate_metadata_filter_rows(rows)
    if suite == "multiturn":
        return evaluate_multiturn_rows(rows)
    if suite == "structured_output":
        return evaluate_structured_output_rows(rows)
    raise ValueError(f"Unsupported suite: {suite}")
