"""Retrieval quality evaluators."""

from __future__ import annotations

import json
from typing import Any


def _metadata_values(outputs: dict[str, Any], key: str) -> list[Any]:
    return [
        metadata.get(key)
        for metadata in outputs.get("retrieved_metadata", [])
        if isinstance(metadata, dict)
    ]


def _expected_list(reference_outputs: dict[str, Any], key: str) -> list[Any]:
    value = reference_outputs.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def diagram_id_hit(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    expected = expected_diagram_candidates(reference_outputs)
    actual_values = _metadata_values(outputs, "diagram_id")
    actual = set(actual_values)
    if not expected:
        hit = bool(outputs.get("retrieved"))
        comment = "General query; non-empty retrieval result is accepted."
    else:
        hit = bool(expected & actual)
        comment = f"expected_or_acceptable={sorted(expected)}, actual_topk={actual_values}"
    return {"key": "diagram_id_hit", "score": int(hit), "comment": comment}


def expected_diagram_candidates(reference_outputs: dict[str, Any]) -> set[Any]:
    expected = set(_expected_list(reference_outputs, "expected_diagram_ids"))
    expected.update(_expected_list(reference_outputs, "acceptable_diagram_ids"))
    return expected


def _first_rank(values: list[Any], candidates: set[Any]) -> int | None:
    for index, value in enumerate(values, start=1):
        if value in candidates:
            return index
    return None


def near_miss_not_above_expected(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    expected = set(_expected_list(reference_outputs, "expected_diagram_ids"))
    near_miss = set(_expected_list(reference_outputs, "near_miss_diagram_ids"))
    actual_values = _metadata_values(outputs, "diagram_id")
    if not expected or not near_miss:
        return {
            "key": "near_miss_not_above_expected",
            "score": None,
            "comment": "No expected or near-miss diagram IDs.",
        }

    expected_rank = _first_rank(actual_values, expected)
    near_miss_rank = _first_rank(actual_values, near_miss)
    if expected_rank is None:
        score = 0
    elif near_miss_rank is None:
        score = 1
    else:
        score = int(expected_rank < near_miss_rank)
    return {
        "key": "near_miss_not_above_expected",
        "score": score,
        "comment": (
            f"expected_rank={expected_rank}, near_miss_rank={near_miss_rank}, "
            f"actual_topk={actual_values}"
        ),
    }


def location_match(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    expected = reference_outputs.get("expected_location")
    if expected is None:
        return {"key": "location_match", "score": 1, "comment": "No expected location."}

    actual = _metadata_values(outputs, "location")
    matched = expected in actual
    return {
        "key": "location_match",
        "score": int(matched),
        "comment": f"expected={expected}, actual_topk={actual}",
    }


def party_type_match(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    expected = reference_outputs.get("expected_party_type")
    if expected is None:
        return {"key": "party_type_match", "score": 1, "comment": "No expected party_type."}

    actual = _metadata_values(outputs, "party_type")
    matched = expected in actual
    return {
        "key": "party_type_match",
        "score": int(matched),
        "comment": f"expected={expected}, actual_topk={actual}",
    }


def chunk_type_match(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    expected = set(_expected_list(reference_outputs, "expected_chunk_types"))
    if not expected:
        return {"key": "chunk_type_match", "score": 1, "comment": "No expected chunk_type."}

    actual = set(_metadata_values(outputs, "chunk_type"))
    matched = bool(expected & actual)
    return {
        "key": "chunk_type_match",
        "score": int(matched),
        "comment": f"expected={sorted(expected)}, actual_topk={list(actual)}",
    }


def keyword_coverage(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    keywords = [str(value) for value in _expected_list(reference_outputs, "expected_keywords")]
    if not keywords:
        return {"key": "keyword_coverage", "score": None, "comment": "No expected keywords."}

    haystack = json.dumps(outputs.get("retrieved", []), ensure_ascii=False)
    matched = [keyword for keyword in keywords if keyword in haystack]
    score = len(matched) / len(keywords)
    return {
        "key": "keyword_coverage",
        "score": score,
        "comment": f"matched={matched}, total={len(keywords)}",
    }


def retrieval_relevance(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    checks = [
        diagram_id_hit(outputs, reference_outputs)["score"],
        location_match(outputs, reference_outputs)["score"],
        party_type_match(outputs, reference_outputs)["score"],
        chunk_type_match(outputs, reference_outputs)["score"],
    ]
    keyword_score = keyword_coverage(outputs, reference_outputs)["score"]
    numeric_scores = [float(score) for score in checks if score is not None]
    if keyword_score is not None:
        numeric_scores.append(float(keyword_score))

    score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else None
    return {
        "key": "retrieval_relevance",
        "score": score,
        "comment": "Average of diagram/location/party/chunk/keyword retrieval checks.",
    }


def critical_error(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    expected_diagram_ids = expected_diagram_candidates(reference_outputs)
    diagram_score = diagram_id_hit(outputs, reference_outputs)["score"]
    party_score = party_type_match(outputs, reference_outputs)["score"]
    location_score = location_match(outputs, reference_outputs)["score"]

    is_error = bool(
        (expected_diagram_ids and not diagram_score)
        or not party_score
        or not location_score
    )
    return {
        "key": "critical_error",
        "score": int(is_error),
        "comment": "1 means a critical retrieval mismatch was detected.",
    }


def build_evaluators() -> list:
    return [
        diagram_id_hit,
        location_match,
        party_type_match,
        chunk_type_match,
        keyword_coverage,
        near_miss_not_above_expected,
        retrieval_relevance,
        critical_error,
    ]
