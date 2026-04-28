"""RAG 답변 JSON 스키마와 검증 로직입니다."""

from __future__ import annotations

from dataclasses import dataclass

from rag.service.common.json_utils import extract_json_object


@dataclass(frozen=True)
class StructuredAnswer:
    """LLM이 반환해야 하는 최소 구조화 답변입니다."""

    fault_ratio_a: int | None
    fault_ratio_b: int | None
    response: str


def parse_fault_ratio(value: object) -> int | None:
    """과실비율 값을 정수 또는 None으로 검증합니다."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("과실비율은 정수 또는 null이어야 합니다.")
    if not 0 <= value <= 100:
        raise ValueError("과실비율은 0부터 100 사이여야 합니다.")
    return value


def parse_structured_answer(content: str) -> StructuredAnswer:
    """LLM JSON 응답을 검증된 구조화 답변으로 변환합니다."""
    data = extract_json_object(content)
    fault_ratio_a = parse_fault_ratio(data.get("fault_ratio_a"))
    fault_ratio_b = parse_fault_ratio(data.get("fault_ratio_b"))

    if (fault_ratio_a is None) != (fault_ratio_b is None):
        raise ValueError("과실비율은 둘 다 숫자이거나 둘 다 null이어야 합니다.")
    if fault_ratio_a is not None and fault_ratio_b is not None and fault_ratio_a + fault_ratio_b != 100:
        raise ValueError("두 과실비율의 합은 100이어야 합니다.")

    response = data.get("response")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("response는 비어 있지 않은 문자열이어야 합니다.")

    return StructuredAnswer(
        fault_ratio_a=fault_ratio_a,
        fault_ratio_b=fault_ratio_b,
        response=response.strip(),
    )
