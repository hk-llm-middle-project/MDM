"""대화형 RAG 서비스의 공개 진입점입니다."""

from config import DEFAULT_CHUNKER_STRATEGY, DEFAULT_EMBEDDING_PROVIDER, DEFAULT_LOADER_STRATEGY
from rag.pipeline.retrieval import RetrievalPipelineConfig
from rag.service.analysis import analyze_question
from rag.service.conversation.orchestrator import AnswerResult, answer_conversation_turn
from rag.service.conversation.pipelines.accident_analysis import (
    MAX_FOLLOW_UP_ATTEMPTS,
    answer_accident_analysis,
    build_fallback_notice,
    build_follow_up_answer,
    get_missing_metadata_fields,
    merge_search_metadata,
)
from rag.service.intake.intake_service import evaluate_input_sufficiency
from rag.service.intake.schema import IntakeState
from rag.service.session.schema import ChatMessage
from rag.service.tracing import TraceContext


def answer_question_with_intake(
    question: str,
    pipeline_config: RetrievalPipelineConfig | None = None,
    intake_state: IntakeState | None = None,
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    chunker_strategy: str = DEFAULT_CHUNKER_STRATEGY,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    chat_history: list[ChatMessage] | None = None,
    trace_context: TraceContext | None = None,
) -> AnswerResult:
    """intake, 일반 대화, RAG 분석을 조율해 답변을 반환합니다."""
    if chat_history is None:
        answer, contexts, needs_more_input, next_state, result_type = answer_accident_analysis(
            question,
            pipeline_config=pipeline_config,
            intake_state=intake_state,
            loader_strategy=loader_strategy,
            chunker_strategy=chunker_strategy,
            embedding_provider=embedding_provider,
            chat_history=chat_history,
            trace_context=trace_context,
            intake_evaluator=evaluate_input_sufficiency,
            analyzer=analyze_question,
        )
        return AnswerResult(
            answer=answer,
            contexts=contexts,
            needs_more_input=needs_more_input,
            intake_state=next_state,
            result_type=result_type,
        )

    return answer_conversation_turn(
        question,
        pipeline_config=pipeline_config,
        intake_state=intake_state,
        loader_strategy=loader_strategy,
        chunker_strategy=chunker_strategy,
        embedding_provider=embedding_provider,
        chat_history=chat_history,
        trace_context=trace_context,
        intake_evaluator=evaluate_input_sufficiency,
        analyzer=analyze_question,
    )


def answer_question(
    question: str,
    pipeline_config: RetrievalPipelineConfig | None = None,
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    chunker_strategy: str = DEFAULT_CHUNKER_STRATEGY,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    chat_history: list[ChatMessage] | None = None,
    trace_context: TraceContext | None = None,
) -> tuple[str, list[str]]:
    """사용자 질문에 대한 답변과 검색 컨텍스트를 반환합니다."""
    result = answer_question_with_intake(
        question,
        pipeline_config=pipeline_config,
        loader_strategy=loader_strategy,
        chunker_strategy=chunker_strategy,
        embedding_provider=embedding_provider,
        chat_history=chat_history,
        trace_context=trace_context,
    )
    return result.answer, result.contexts


def answer_question_without_intake(
    question: str,
    pipeline_config: RetrievalPipelineConfig | None = None,
    chat_history: list[ChatMessage] | None = None,
    trace_context: TraceContext | None = None,
) -> tuple[str, list[str]]:
    """intake 없이 바로 RAG 답변을 반환합니다."""
    return analyze_question(
        question,
        search_metadata=None,
        pipeline_config=pipeline_config,
        chat_history=chat_history,
        trace_context=trace_context,
    )
