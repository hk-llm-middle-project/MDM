"""대화형 RAG의 한 턴 처리 흐름을 조율합니다."""

from dataclasses import dataclass, field
from typing import Any

from config import DEFAULT_LOADER_STRATEGY
from rag.pipeline.retrieval import RetrievalPipelineConfig
from rag.service.conversation.pipelines.accident_analysis import answer_accident_analysis
from rag.service.conversation.pipelines.general_chat import answer_general_chat
from rag.service.conversation.router import route_conversation_turn
from rag.service.conversation.schema import RouteType, TurnResultType
from rag.service.intake.schema import IntakeState
from rag.service.session.schema import ChatMessage


@dataclass(frozen=True)
class AnswerResult:
    """사용자 입력 처리 결과와 갱신된 intake 상태입니다."""

    answer: str
    contexts: list[str] = field(default_factory=list)
    needs_more_input: bool = False
    intake_state: IntakeState = field(default_factory=IntakeState)
    result_type: TurnResultType = TurnResultType.ACCIDENT_RAG


def answer_conversation_turn(
    question: str,
    *,
    pipeline_config: RetrievalPipelineConfig | None = None,
    intake_state: IntakeState | None = None,
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    chat_history: list[ChatMessage] | None = None,
    intake_evaluator,
    analyzer,
    router_llm: Any | None = None,
    general_chat_llm: Any | None = None,
) -> AnswerResult:
    """라우팅 판단에 따라 일반 대화 또는 사고 분석 파이프를 실행합니다."""
    current_state = intake_state or IntakeState()
    route = route_conversation_turn(
        question,
        chat_history,
        current_state,
        llm=router_llm,
    )

    if route.route_type == RouteType.GENERAL_CHAT:
        return AnswerResult(
            answer=answer_general_chat(question, chat_history, llm=general_chat_llm),
            contexts=[],
            needs_more_input=False,
            intake_state=current_state,
            result_type=TurnResultType.GENERAL_CHAT,
        )

    answer, contexts, needs_more_input, next_state, result_type = answer_accident_analysis(
        question,
        pipeline_config=pipeline_config,
        intake_state=current_state,
        loader_strategy=loader_strategy,
        chat_history=chat_history,
        intake_evaluator=intake_evaluator,
        analyzer=analyzer,
    )
    return AnswerResult(
        answer=answer,
        contexts=contexts,
        needs_more_input=needs_more_input,
        intake_state=next_state,
        result_type=result_type,
    )
