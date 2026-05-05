"""대화형 RAG의 한 턴 처리 흐름을 조율합니다."""

from dataclasses import dataclass, field
from typing import Any

from config import DEFAULT_CHUNKER_STRATEGY, DEFAULT_EMBEDDING_PROVIDER, DEFAULT_LOADER_STRATEGY
from rag.service.analysis.answer_schema import RetrievedContext
from rag.pipeline.retrieval import RetrievalPipelineConfig
from rag.service.conversation.pipelines.accident_analysis import answer_accident_analysis
from rag.service.conversation.pipelines.general_chat import answer_general_chat
from rag.service.conversation.router import route_conversation_turn
from rag.service.conversation.schema import RouteType, TurnResultType
from rag.service.intake.schema import IntakeState
from rag.service.progress import (
    PROGRESS_ANSWER,
    PROGRESS_ROUTE,
    ProgressCallback,
    report_progress,
    report_progress_detail,
)
from rag.service.session.schema import ChatMessage
from rag.service.tracing import TraceContext


@dataclass(frozen=True)
class AnswerResult:
    """사용자 입력 처리 결과와 갱신된 intake 상태입니다."""

    answer: str
    contexts: list[str] = field(default_factory=list)
    needs_more_input: bool = False
    intake_state: IntakeState = field(default_factory=IntakeState)
    result_type: TurnResultType = TurnResultType.ACCIDENT_RAG
    fault_ratio_a: int | None = None
    fault_ratio_b: int | None = None
    retrieved_contexts: list[RetrievedContext] = field(default_factory=list)


def answer_conversation_turn(
    question: str,
    *,
    pipeline_config: RetrievalPipelineConfig | None = None,
    intake_state: IntakeState | None = None,
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    chunker_strategy: str = DEFAULT_CHUNKER_STRATEGY,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    chat_history: list[ChatMessage] | None = None,
    trace_context: TraceContext | None = None,
    intake_evaluator,
    analyzer,
    router_llm: Any | None = None,
    general_chat_llm: Any | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AnswerResult:
    """라우팅 판단에 따라 일반 대화 또는 사고 분석 파이프를 실행합니다."""
    current_state = intake_state or IntakeState()
    report_progress(progress_callback, PROGRESS_ROUTE)
    route = route_conversation_turn(
        question,
        chat_history,
        current_state,
        llm=router_llm,
        trace_context=trace_context,
    )
    report_progress_detail(
        progress_callback,
        f"대화 의도: {route.route_type.value}",
    )
    report_progress_detail(
        progress_callback,
        f"라우터 신뢰도: {route.confidence:.2f}",
    )

    if route.route_type == RouteType.GENERAL_CHAT:
        report_progress(progress_callback, PROGRESS_ANSWER)
        return AnswerResult(
            answer=answer_general_chat(
                question,
                chat_history,
                llm=general_chat_llm,
                trace_context=trace_context,
            ),
            contexts=[],
            needs_more_input=False,
            intake_state=current_state,
            result_type=TurnResultType.GENERAL_CHAT,
        )

    pipeline_result = answer_accident_analysis(
        question,
        pipeline_config=pipeline_config,
        intake_state=current_state,
        loader_strategy=loader_strategy,
        chunker_strategy=chunker_strategy,
        embedding_provider=embedding_provider,
        chat_history=chat_history,
        trace_context=trace_context,
        intake_evaluator=intake_evaluator,
        analyzer=analyzer,
        progress_callback=progress_callback,
    )
    return AnswerResult(
        answer=pipeline_result.answer,
        contexts=pipeline_result.contexts,
        needs_more_input=pipeline_result.needs_more_input,
        intake_state=pipeline_result.intake_state,
        result_type=pipeline_result.result_type,
        fault_ratio_a=pipeline_result.fault_ratio_a,
        fault_ratio_b=pipeline_result.fault_ratio_b,
        retrieved_contexts=pipeline_result.retrieved_contexts,
    )
