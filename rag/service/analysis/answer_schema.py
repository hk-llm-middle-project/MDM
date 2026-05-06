"""RAG 답변 JSON 스키마와 검증 로직입니다."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.service.common.json_utils import extract_json_object


@dataclass(frozen=True)
class StructuredAnswer:
    """LLM이 반환해야 하는 최소 구조화 답변입니다."""

    fault_ratio_a: int | None
    fault_ratio_b: int | None
    response: str


@dataclass(frozen=True)
class RetrievedContext:
    """검색된 문서 조각과 원본 metadata입니다."""

    content: str
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def image_path(self) -> str | None:
        value = self.metadata.get("image_path")
        return value if isinstance(value, str) and value else None


@dataclass(frozen=True)
class AnalysisResult:
    """검증된 분석 답변과 검색 문맥입니다.

    기존 호출부의 ``answer, contexts = analyze_question(...)`` 형태를 유지하기
    위해 반복 시에는 답변 문자열과 문맥만 반환합니다.
    """

    response: str
    contexts: list[str] = field(default_factory=list)
    retrieved_contexts: list[RetrievedContext] = field(default_factory=list)
    fault_ratio_a: int | None = None
    fault_ratio_b: int | None = None

    def __iter__(self):
        yield self.response
        yield self.contexts


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
