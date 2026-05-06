"""리랭커에서 사용하는 프롬프트 모음."""

from __future__ import annotations

import json

from langchain_core.documents import Document


LLM_SCORE_DOCUMENT_PREVIEW_CHARS = 1800


def _format_candidate_document(index: int, document: Document) -> str:
    metadata = json.dumps(document.metadata or {}, ensure_ascii=False, sort_keys=True)
    content = document.page_content
    if len(content) > LLM_SCORE_DOCUMENT_PREVIEW_CHARS:
        content = f"{content[:LLM_SCORE_DOCUMENT_PREVIEW_CHARS]}\n...[truncated]"
    return f"[{index}]\nmetadata: {metadata}\ncontent:\n{content}"


def build_llm_score_reranker_prompt(query: str, documents: list[Document]) -> str:
    """LLM 점수화 리랭커용 프롬프트를 생성합니다."""
    candidates = "\n\n".join(
        _format_candidate_document(index, document)
        for index, document in enumerate(documents)
    )
    return f"""
당신은 자동차사고 과실비율 인정기준 문서의 리랭커다.
목표는 사용자 질문의 사고 상황에 적용할 수 있는 기준번호/도표/기본 과실비율 문서를 위로 올리는 것이다.

각 후보 문서를 질문과 비교해 관련도 score를 매겨라.
단순히 같은 단어가 많다는 이유로 높게 주지 말고, 사고유형이 정확히 일치하는지 우선 판단하라.

[평가 기준]
- 0.90~1.00: 질문과 같은 기준번호, 도표, 사고상황, 기본 과실비율을 직접 포함한다.
- 0.75~0.89: 같은 유형군이며 당사자, 장소, 진행방향, 신호/우선관계가 거의 모두 일치한다.
- 0.45~0.74: 일부 조건은 맞지만 핵심 조건이 빠졌거나 다른 세부 유형일 수 있다.
- 0.20~0.44: 같은 장/유형군처럼 보이나 진행방향, 신호, 상대 위치, 차종 등 핵심 조건이 다르다.
- 0.00~0.19: 다른 사고유형이거나 질문 답변에 거의 도움이 되지 않는다.

[반드시 확인할 핵심 조건]
- 당사자 유형: 자동차, 자전거, 보행자, 이륜차 등
- 장소: 신호 교차로, 무신호 교차로, 도로, 주차장, 횡단보도 등
- 진행관계: 직진, 좌회전, 우회전, 진로변경, 추돌, 역주행 등
- 신호/우선관계: 녹색, 황색, 적색, 점멸, 비보호, 대로/소로, 우측/좌측 등
- 상대 위치: 측면 진입, 맞은편, 같은 방향, 도로 외 진입 등
- 문서가 기본 과실비율이나 해당 기준번호를 직접 제공하는지

[주의]
- 참고 판례, 일반 해설, 수정요소만 있는 문서는 기본 과실비율 표/사고상황 문서보다 낮게 평가하라.
- 같은 교차로 사고라도 직진 대 직진, 직진 대 좌회전, 우회전 대 직진은 서로 다른 유형이다.
- 같은 무신호 교차로라도 대로/소로, 우측/좌측, 맞은편/측면, 좌회전/우회전 조건이 다르면 낮게 평가하라.
- metadata의 diagram_id, page, chunk_type은 힌트로 활용하되, 본문과 질문의 사고 조건 일치를 우선하라.

반드시 아래 형식의 유효한 JSON만 반환하라.
{{
  "results": [
    {{"id": 0, "score": 0.97}},
    {{"id": 1, "score": 0.31}}
  ]
}}
score는 0과 1 사이의 실수여야 한다.
문서 id는 입력으로 받은 후보 문서의 번호를 그대로 사용하라.
모든 후보 문서에 대해 results 항목을 하나씩 반환하라.
설명 문장, 코드 블록 마커, 추가 텍스트는 출력하지 마라.

[질문]
{query}

[후보 문서]
{candidates}
""".strip()
