"""JSON parsing helpers shared by service modules."""

from __future__ import annotations

import json
from typing import Any


def _strip_markdown_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _find_first_json_object(content: str) -> str:
    start = content.find("{")
    if start == -1:
        raise ValueError(f"JSON object not found in response: {content}")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1]

    raise ValueError(f"JSON object is incomplete in response: {content}")


def extract_json_object(content: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response string."""
    stripped = _strip_markdown_fence(content)
    return json.loads(_find_first_json_object(stripped), strict=False)
