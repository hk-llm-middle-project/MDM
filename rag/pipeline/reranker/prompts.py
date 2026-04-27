"""리랭커에서 사용하는 프롬프트 모음."""

from __future__ import annotations

from langchain_core.documents import Document


def build_llm_score_reranker_prompt(query: str, documents: list[Document]) -> str:
    """LLM 점수화 리랭커용 프롬프트를 생성합니다."""
    candidates = "\n\n".join(
        f"[{index}]\n{document.page_content}"
        for index, document in enumerate(documents)
    )
    return f"""
당신은 검색 파이프라인의 리랭커다.
사용자 질문에 답하는 데 각 후보 문서가 얼마나 도움이 되는지 점수화하라.
반드시 아래 형식의 유효한 JSON만 반환하라.
{{
  "results": [
    {{"id": 0, "score": 0.97}},
    {{"id": 1, "score": 0.31}}
  ]
}}
score는 0과 1 사이의 실수여야 한다.
문서 id는 입력으로 받은 후보 문서의 번호를 그대로 사용하라.
설명 문장, 코드 블록 마커, 추가 텍스트는 출력하지 마라.

[질문]
{query}

[후보 문서]
{candidates}
""".strip()
