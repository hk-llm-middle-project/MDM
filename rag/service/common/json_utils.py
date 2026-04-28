"""JSON parsing helpers shared by service modules."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(content: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response string."""
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ValueError(f"JSON object not found in response: {content}")
    return json.loads(match.group())

