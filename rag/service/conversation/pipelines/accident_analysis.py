"""사고 분석 파이프라인: intake, follow-up, RAG 답변을 처리합니다."""

from collections.abc import Callable, Sequence

from config import DEFAULT_EMBEDDING_PROVIDER, DEFAULT_LOADER_STRATEGY
from rag.pipeline.retrieval import RetrievalPipelineConfig
from rag.service.conversation.schema import TurnResultType
from rag.service.intake.intake_service import build_default_follow_up_questions
from rag.service.intake.schema import IntakeDecision, IntakeState, UserSearchMetadata
from rag.service.session.schema import ChatMessage


MAX_FOLLOW_UP_ATTEMPTS = 2

IntakeEvaluator = Callable[..., IntakeDecision]
Analyzer = Callable[..., tuple[str, list[str]]]


def merge_search_metadata(
    previous: UserSearchMetadata,
    current: UserSearchMetadata,
) -> UserSearchMetadata:
    """이전 intake metadata와 이번 입력에서 새로 추출한 값을 병합합니다."""
    return UserSearchMetadata(
        party_type=current.party_type or previous.party_type,
        location=current.location or previous.location,
    )


def get_missing_metadata_fields(metadata: UserSearchMetadata) -> list[str]:
    """검색에 필요한 최소 metadata 중 아직 없는 필드를 반환합니다."""
    missing_fields: list[str] = []
    if metadata.party_type is None:
        missing_fields.append("party_type")
    if metadata.location is None:
        missing_fields.append("location")
    return missing_fields


def build_follow_up_answer(follow_up_questions: list[str]) -> str:
    """검색에 필요한 정보가 부족할 때 사용자에게 보여줄 답변을 만듭니다."""
    if not follow_up_questions:
        return "검색에 필요한 사고 정보가 부족합니다. 사고 대상과 사고 상황을 조금 더 알려주세요."

    questions = "\n".join(f"- {question}" for question in follow_up_questions)
    return f"검색에 필요한 사고 정보가 조금 더 필요합니다.\n\n{questions}"


def build_fallback_notice(answer: str) -> str:
    """추가 질문 시도를 넘긴 뒤 best-effort 검색 결과임을 알립니다."""
    return f"입력 정보가 충분하지 않아 정확도가 낮을 수 있습니다. 현재 설명만으로 가능한 범위에서 찾아봤습니다.\n\n{answer}"


def reset_intake_progress(search_metadata: UserSearchMetadata) -> IntakeState:
    """완료된 사고 metadata는 남기고 진행 중인 follow-up 상태만 초기화합니다."""
    return IntakeState(search_metadata=search_metadata)


def evaluate_with_optional_context(
    evaluator: IntakeEvaluator,
    question: str,
    chat_history: Sequence[ChatMessage] | None,
    previous_state: IntakeState,
) -> IntakeDecision:
    """기존 테스트 호환을 위해 문맥이 없으면 예전 호출 형태를 유지합니다."""
    if chat_history is None and previous_state == IntakeState():
        return evaluator(question)
    return evaluator(question, chat_history=chat_history, previous_state=previous_state)


def answer_accident_analysis(
    question: str,
    *,
    pipeline_config: RetrievalPipelineConfig | None = None,
    intake_state: IntakeState | None = None,
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    chat_history: list[ChatMessage] | None = None,
    intake_evaluator: IntakeEvaluator,
    analyzer: Analyzer,
) -> tuple[str, list[str], bool, IntakeState, TurnResultType]:
    """사고 분석 흐름을 처리하고 답변, context, 상태, 결과 타입을 반환합니다."""
    current_state = intake_state or IntakeState()
    intake_decision = evaluate_with_optional_context(
        intake_evaluator,
        question,
        chat_history,
        current_state,
    )
    merged_metadata = merge_search_metadata(
        current_state.search_metadata,
        intake_decision.search_metadata,
    )
    missing_field_names = get_missing_metadata_fields(merged_metadata)

    if missing_field_names:
        if current_state.attempt_count < MAX_FOLLOW_UP_ATTEMPTS:
            follow_up_questions = (
                intake_decision.follow_up_questions
                or build_default_follow_up_questions(missing_field_names)
            )
            next_state = IntakeState(
                attempt_count=current_state.attempt_count + 1,
                search_metadata=merged_metadata,
                last_missing_fields=missing_field_names,
                last_follow_up_questions=follow_up_questions,
            )
            return (
                build_follow_up_answer(follow_up_questions),
                [],
                True,
                next_state,
                TurnResultType.ACCIDENT_FOLLOW_UP,
            )

        analysis_kwargs = {
            "search_metadata": merged_metadata,
            "pipeline_config": pipeline_config,
            "loader_strategy": loader_strategy,
            "embedding_provider": embedding_provider,
        }
        if chat_history is not None:
            analysis_kwargs["chat_history"] = chat_history
        answer, contexts = analyzer(
            intake_decision.normalized_description or question,
            **analysis_kwargs,
        )
        next_state = (
            reset_intake_progress(merged_metadata)
            if chat_history is not None
            else IntakeState()
        )
        return (
            build_fallback_notice(answer),
            contexts,
            False,
            next_state,
            TurnResultType.ACCIDENT_RAG,
        )

    analysis_kwargs = {
        "search_metadata": merged_metadata,
        "pipeline_config": pipeline_config,
        "loader_strategy": loader_strategy,
        "embedding_provider": embedding_provider,
    }
    if chat_history is not None:
        analysis_kwargs["chat_history"] = chat_history
    answer, contexts = analyzer(
        intake_decision.normalized_description,
        **analysis_kwargs,
    )
    next_state = (
        reset_intake_progress(merged_metadata)
        if chat_history is not None
        else IntakeState()
    )
    return answer, contexts, False, next_state, TurnResultType.ACCIDENT_RAG
