"""RAG 앱의 서비스 흐름을 조율합니다."""

from rag.pipeline.retrieval import RetrievalPipelineConfig
from rag.service.analysis_service import analyze_question


def answer_question(
    question: str,
    pipeline_config: RetrievalPipelineConfig | None = None,
) -> tuple[str, list[str]]:
    """사용자 질문에 대한 RAG 답변을 반환합니다."""
    return analyze_question(question, pipeline_config=pipeline_config)
