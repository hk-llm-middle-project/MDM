"""사고 분석 파이프라인: intake, follow-up, RAG 답변을 처리합니다."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from config import DEFAULT_CHUNKER_STRATEGY, DEFAULT_EMBEDDING_PROVIDER, DEFAULT_LOADER_STRATEGY
from rag.pipeline.retrieval import RetrievalPipelineConfig
from rag.service.analysis.answer_schema import AnalysisResult, RetrievedContext
from rag.service.conversation.schema import TurnResultType
from rag.service.intake.intake_service import build_default_follow_up_questions
from rag.service.intake.schema import IntakeDecision, IntakeState, QuerySlots, UserSearchMetadata
from rag.service.session.schema import ChatMessage
from rag.service.tracing import TraceContext


MAX_FOLLOW_UP_ATTEMPTS = 2

IntakeEvaluator = Callable[..., IntakeDecision]
Analyzer = Callable[..., AnalysisResult | tuple[str, list[str]]]


@dataclass(frozen=True)
class AccidentAnalysisResult:
    """사고 분석 파이프라인 결과입니다.

    기존 호출부의 5개 값 언패킹을 유지하기 위해 반복 시에는 기존 순서만
    노출하고, 새 UI는 fault_ratio_a/b 속성을 읽습니다.
    """

    answer: str
    contexts: list[str] = field(default_factory=list)
    needs_more_input: bool = False
    intake_state: IntakeState = field(default_factory=IntakeState)
    result_type: TurnResultType = TurnResultType.ACCIDENT_RAG
    fault_ratio_a: int | None = None
    fault_ratio_b: int | None = None
    retrieved_contexts: list[RetrievedContext] = field(default_factory=list)

    def __iter__(self):
        yield self.answer
        yield self.contexts
        yield self.needs_more_input
        yield self.intake_state
        yield self.result_type


def normalize_analysis_result(result: AnalysisResult | tuple[str, list[str]]) -> AnalysisResult:
    """테스트 대역처럼 튜플을 반환하는 analyzer도 동일하게 다룹니다."""
    if isinstance(result, AnalysisResult):
        return result
    answer, contexts = result
    return AnalysisResult(response=answer, contexts=contexts)


def merge_search_metadata(
    previous: UserSearchMetadata,
    current: UserSearchMetadata,
) -> UserSearchMetadata:
    """이전 intake metadata와 이번 입력에서 새로 추출한 값을 병합합니다."""
    return UserSearchMetadata(
        party_type=current.party_type or previous.party_type,
        location=current.location or previous.location,
        retrieval_query=current.retrieval_query or previous.retrieval_query,
        query_slots=merge_query_slots(previous.query_slots, current.query_slots),
    )


def merge_query_slots(previous: QuerySlots, current: QuerySlots) -> QuerySlots:
    """이전 query slot과 이번 입력의 query slot을 필드별로 병합합니다."""
    return QuerySlots(
        road_control=current.road_control or previous.road_control,
        relation=current.relation or previous.relation,
        a_signal=current.a_signal or previous.a_signal,
        b_signal=current.b_signal or previous.b_signal,
        a_movement=current.a_movement or previous.a_movement,
        b_movement=current.b_movement or previous.b_movement,
        road_priority=current.road_priority or previous.road_priority,
        special_condition=current.special_condition or previous.special_condition,
    )


def get_missing_metadata_fields(metadata: UserSearchMetadata) -> list[str]:
    """검색에 필요한 최소 metadata 중 아직 없는 필드를 반환합니다."""
    missing_fields: list[str] = []
    if metadata.party_type is None:
        missing_fields.append("party_type")
    if metadata.location is None:
        missing_fields.append("location")
    return missing_fields


def get_missing_query_slot_fields(metadata: UserSearchMetadata) -> list[str]:
    """검색 품질에 중요한 사고 단서 중 추가 확인할 필드를 반환합니다."""
    if not metadata.retrieval_query:
        return []
    if metadata.party_type != "자동차":
        return []

    slots = metadata.query_slots
    missing_fields: list[str] = []
    has_movement_pair = bool(slots.a_movement and slots.b_movement)

    if metadata.location == "교차로 사고":
        if slots.road_control and "신호" in slots.road_control:
            if not (slots.a_signal and slots.b_signal):
                missing_fields.append("query_slots.signals")
        if not has_movement_pair:
            missing_fields.append("query_slots.movements")
        if not (slots.relation or slots.road_priority):
            missing_fields.append("query_slots.relation")
    elif metadata.location == "같은 방향 진행차량 상호간의 사고":
        if not (slots.special_condition or slots.a_movement or slots.b_movement):
            missing_fields.append("query_slots.same_direction_detail")
    elif metadata.location == "마주보는 방향 진행차량 상호 간의 사고":
        if not slots.special_condition:
            missing_fields.append("query_slots.opposite_direction_detail")
    elif metadata.location == "기타":
        if not (slots.special_condition or slots.relation):
            missing_fields.append("query_slots.special_condition")

    return missing_fields[:2]


def get_missing_search_fields(metadata: UserSearchMetadata) -> list[str]:
    """필수 metadata와 검색 품질 보강용 slot 누락을 함께 반환합니다."""
    metadata_fields = get_missing_metadata_fields(metadata)
    if metadata_fields:
        return metadata_fields
    return get_missing_query_slot_fields(metadata)


def build_query_slot_follow_up_questions(missing_field_names: list[str]) -> list[str]:
    """slot 누락에 대해 사용자가 답하기 쉬운 추가 질문을 만듭니다."""
    questions: list[str] = []
    if "query_slots.signals" in missing_field_names:
        questions.append("두 차량의 신호는 각각 무엇이었나요? 예: A 녹색 직진, B 적색 직진")
    if "query_slots.movements" in missing_field_names:
        questions.append("두 차량은 각각 직진, 좌회전, 우회전, 진로변경 중 어떻게 진행했나요?")
    if "query_slots.relation" in missing_field_names:
        questions.append("상대 차량은 맞은편에서 왔나요, 오른쪽/왼쪽 도로처럼 측면에서 진입했나요?")
    if "query_slots.same_direction_detail" in missing_field_names:
        questions.append("같은 방향 사고라면 추돌인지, 진로변경/차로변경 중 충돌인지 알려주세요.")
    if "query_slots.opposite_direction_detail" in missing_field_names:
        questions.append("마주보는 방향 사고라면 중앙선 침범이나 역주행이 있었나요?")
    if "query_slots.special_condition" in missing_field_names:
        questions.append("도로 외 진입, 주차장, 문 열림, 유턴, 정차 후 출발 중 해당되는 특수 상황이 있나요?")
    return questions[:2]


def build_follow_up_questions(
    missing_field_names: list[str],
    intake_decision: IntakeDecision,
) -> list[str]:
    """누락 종류에 따라 LLM 질문 또는 기본 질문을 선택합니다."""
    metadata_missing_fields = [
        field_name
        for field_name in missing_field_names
        if not field_name.startswith("query_slots.")
    ]
    if metadata_missing_fields:
        return (
            intake_decision.follow_up_questions
            or build_default_follow_up_questions(metadata_missing_fields)
        )
    return (
        intake_decision.follow_up_questions
        or build_query_slot_follow_up_questions(missing_field_names)
    )


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
    trace_context: TraceContext | None = None,
) -> IntakeDecision:
    """기존 테스트 호환을 위해 문맥이 없으면 예전 호출 형태를 유지합니다."""
    if chat_history is None and previous_state == IntakeState():
        if trace_context is not None:
            return evaluator(question, trace_context=trace_context)
        return evaluator(question)
    return evaluator(
        question,
        chat_history=chat_history,
        previous_state=previous_state,
        trace_context=trace_context,
    )


def answer_accident_analysis(
    question: str,
    *,
    pipeline_config: RetrievalPipelineConfig | None = None,
    intake_state: IntakeState | None = None,
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    chunker_strategy: str = DEFAULT_CHUNKER_STRATEGY,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    chat_history: list[ChatMessage] | None = None,
    trace_context: TraceContext | None = None,
    intake_evaluator: IntakeEvaluator,
    analyzer: Analyzer,
) -> AccidentAnalysisResult:
    """사고 분석 흐름을 처리하고 답변, context, 상태, 결과 타입을 반환합니다."""
    current_state = intake_state or IntakeState()
    intake_decision = evaluate_with_optional_context(
        intake_evaluator,
        question,
        chat_history,
        current_state,
        trace_context,
    )
    merged_metadata = merge_search_metadata(
        current_state.search_metadata,
        intake_decision.search_metadata,
    )
    missing_field_names = get_missing_search_fields(merged_metadata)

    if missing_field_names:
        if current_state.attempt_count < MAX_FOLLOW_UP_ATTEMPTS:
            follow_up_questions = build_follow_up_questions(
                missing_field_names,
                intake_decision,
            )
            next_state = IntakeState(
                attempt_count=current_state.attempt_count + 1,
                search_metadata=merged_metadata,
                last_missing_fields=missing_field_names,
                last_follow_up_questions=follow_up_questions,
            )
            return AccidentAnalysisResult(
                answer=build_follow_up_answer(follow_up_questions),
                contexts=[],
                needs_more_input=True,
                intake_state=next_state,
                result_type=TurnResultType.ACCIDENT_FOLLOW_UP,
            )

        analysis_kwargs = {
            "search_metadata": merged_metadata,
            "pipeline_config": pipeline_config,
            "loader_strategy": loader_strategy,
            "embedding_provider": embedding_provider,
        }
        if chunker_strategy != DEFAULT_CHUNKER_STRATEGY:
            analysis_kwargs["chunker_strategy"] = chunker_strategy
        if chat_history is not None:
            analysis_kwargs["chat_history"] = chat_history
        if trace_context is not None:
            analysis_kwargs["trace_context"] = trace_context
        analysis_result = normalize_analysis_result(
            analyzer(
                intake_decision.normalized_description or question,
                **analysis_kwargs,
            )
        )
        next_state = (
            reset_intake_progress(merged_metadata)
            if chat_history is not None
            else IntakeState()
        )
        return AccidentAnalysisResult(
            answer=build_fallback_notice(analysis_result.response),
            contexts=analysis_result.contexts,
            needs_more_input=False,
            intake_state=next_state,
            result_type=TurnResultType.ACCIDENT_RAG,
            fault_ratio_a=analysis_result.fault_ratio_a,
            fault_ratio_b=analysis_result.fault_ratio_b,
            retrieved_contexts=analysis_result.retrieved_contexts,
        )

    analysis_kwargs = {
        "search_metadata": merged_metadata,
        "pipeline_config": pipeline_config,
        "loader_strategy": loader_strategy,
        "embedding_provider": embedding_provider,
    }
    if chunker_strategy != DEFAULT_CHUNKER_STRATEGY:
        analysis_kwargs["chunker_strategy"] = chunker_strategy
    if chat_history is not None:
        analysis_kwargs["chat_history"] = chat_history
    if trace_context is not None:
        analysis_kwargs["trace_context"] = trace_context
    analysis_result = normalize_analysis_result(
        analyzer(
            intake_decision.normalized_description,
            **analysis_kwargs,
        )
    )
    next_state = (
        reset_intake_progress(merged_metadata)
        if chat_history is not None
        else IntakeState()
    )
    return AccidentAnalysisResult(
        answer=analysis_result.response,
        contexts=analysis_result.contexts,
        needs_more_input=False,
        intake_state=next_state,
        result_type=TurnResultType.ACCIDENT_RAG,
        fault_ratio_a=analysis_result.fault_ratio_a,
        fault_ratio_b=analysis_result.fault_ratio_b,
        retrieved_contexts=analysis_result.retrieved_contexts,
    )
