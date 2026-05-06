"""User-visible progress labels for long-running analysis steps."""

from __future__ import annotations

from collections.abc import Callable


ProgressCallback = Callable[[str], None]

PROGRESS_INPUT = "사고 설명을 확인하는 중"
PROGRESS_ROUTE = "대화 의도를 판단하는 중"
PROGRESS_INTAKE = "사고 단서를 추출하는 중"
PROGRESS_NEEDS_CHECK = "추가로 필요한 정보가 있는지 확인하는 중"
PROGRESS_RETRIEVAL = "과실비율 기준 문서를 검색하는 중"
PROGRESS_RERANK = "관련 근거를 정렬하는 중"
PROGRESS_ANSWER = "과실비율 판단을 정리하는 중"
PROGRESS_RESULT = "참고 근거를 준비하는 중"


def report_progress(callback: ProgressCallback | None, label: str) -> None:
    """Notify the UI about a long-running step when a reporter exists."""
    if callback is not None:
        callback(label)


def report_progress_detail(callback: ProgressCallback | None, detail: str) -> None:
    """Send optional debug details to reporters that support them."""
    if callback is None:
        return
    detail_callback = getattr(callback, "detail", None)
    if callable(detail_callback):
        detail_callback(detail)
