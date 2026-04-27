"""RAG 앱의 서비스 흐름을 조율합니다."""

from dataclasses import dataclass, field

from rag.pipeline.retrieval import RetrievalPipelineConfig
from rag.service.analysis_service import analyze_question
from rag.service.intake.intake_service import (
    build_default_follow_up_questions,
    evaluate_input_sufficiency,
)
from rag.service.intake.schema import IntakeState, UserSearchMetadata


MAX_FOLLOW_UP_ATTEMPTS = 2


@dataclass(frozen=True)
class AnswerResult:
    """사용자 입력 처리 결과와 갱신된 intake 상태입니다."""

    answer: str
    contexts: list[str] = field(default_factory=list)
    needs_more_input: bool = False
    intake_state: IntakeState = field(default_factory=IntakeState)


def merge_search_metadata(
    previous: UserSearchMetadata,
    current: UserSearchMetadata,
) -> UserSearchMetadata:
    """이전 intake metadata에 이번 입력에서 새로 추출한 값을 병합합니다."""
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
        return "검색에 필요한 사고 정보가 부족합니다. 사고 상대와 사고 상황을 조금 더 알려주세요."

    questions = "\n".join(f"- {question}" for question in follow_up_questions)
    return f"검색에 필요한 사고 정보가 조금 더 필요합니다.\n\n{questions}"


def build_fallback_notice(answer: str) -> str:
    """추가 질문 한도를 넘긴 뒤 best-effort 검색 결과임을 알립니다."""
    return f"입력 정보가 충분하지 않아 정확도가 낮을 수 있습니다. 현재 설명만으로 가능한 범위에서 찾아봤습니다.\n\n{answer}"


def answer_question_with_intake(
    question: str,
    pipeline_config: RetrievalPipelineConfig | None = None,
    intake_state: IntakeState | None = None,
) -> AnswerResult:
    """intake 결과에 따라 추가 질문 또는 RAG 답변을 반환합니다."""
    current_state = intake_state or IntakeState()
    intake_decision = evaluate_input_sufficiency(question)
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
            return AnswerResult(
                answer=build_follow_up_answer(follow_up_questions),
                contexts=[],
                needs_more_input=True,
                intake_state=next_state,
            )

        # 질문 한도를 넘기면 지금까지 누적된 metadata만 사용해 가능한 범위에서 검색합니다.
        answer, contexts = analyze_question(
            intake_decision.normalized_description or question,
            search_metadata=merged_metadata,
            pipeline_config=pipeline_config,
        )
        return AnswerResult(
            answer=build_fallback_notice(answer),
            contexts=contexts,
            needs_more_input=False,
            intake_state=IntakeState(),
        )

    answer, contexts = analyze_question(
        intake_decision.normalized_description,
        search_metadata=merged_metadata,
        pipeline_config=pipeline_config,
    )
    return AnswerResult(
        answer=answer,
        contexts=contexts,
        needs_more_input=False,
        intake_state=IntakeState(),
    )


def answer_question(
    question: str,
    pipeline_config: RetrievalPipelineConfig | None = None,
) -> tuple[str, list[str]]:
    """사용자 질문에 대한 RAG 답변을 반환합니다."""
    result = answer_question_with_intake(
        question,
        pipeline_config=pipeline_config,
    )
    return result.answer, result.contexts
