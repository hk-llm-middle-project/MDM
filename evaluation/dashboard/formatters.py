"""Display-value formatting helpers for dashboard tables."""

from __future__ import annotations

from typing import Any
import json

import pandas as pd


def is_empty_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null"}
    if isinstance(value, (list, tuple, set, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def parse_jsonish(value: Any) -> Any:
    if is_empty_scalar(value):
        return None
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return None
    if text[0] not in '[{"' and text.lower() not in {"true", "false", "null"}:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def has_display_value(value: Any) -> bool:
    parsed = parse_jsonish(value)
    if parsed is None:
        return False
    if isinstance(parsed, (list, tuple, set, dict)):
        return len(parsed) > 0
    return not is_empty_scalar(parsed)


def format_display_value(value: Any) -> str:
    parsed = parse_jsonish(value)
    if parsed is None:
        return ""
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    if isinstance(parsed, (list, tuple, set)):
        values = [format_display_value(item) for item in parsed]
        return ", ".join(value for value in values if value)
    if isinstance(parsed, float) and parsed.is_integer():
        return str(int(parsed))
    return str(parsed)


def format_score(value: Any) -> str:
    if not has_display_value(value):
        return ""
    number = pd.to_numeric([value], errors="coerce")[0]
    if pd.isna(number):
        return format_display_value(value)
    if float(number).is_integer():
        return str(int(number))
    return f"{float(number):.4f}".rstrip("0").rstrip(".")
