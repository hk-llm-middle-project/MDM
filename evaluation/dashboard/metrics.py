"""Metric definitions and descriptions for the evaluation dashboard."""

from __future__ import annotations


METRIC_COLUMNS = (
    "diagram_id_hit",
    "location_match",
    "party_type_match",
    "chunk_type_match",
    "keyword_coverage",
    "near_miss_not_above_expected",
    "retrieval_relevance",
    "critical_error",
    "intake_is_sufficient",
    "missing_fields_match",
    "follow_up_contains",
    "forbidden_filter_absent",
    "intake_overall",
    "route_type_match",
    "reason_category_match",
    "router_overall",
    "metadata_filter_match",
    "metadata_filter_overall",
    "state_sequence_match",
    "followup_questions_match",
    "final_metadata_match",
    "final_result_type_match",
    "turns_to_ready_match",
    "multiturn_overall",
    "final_fault_ratio_match",
    "cannot_determine_match",
    "required_evidence_coverage",
    "party_role_coverage",
    "applicable_modifier_coverage",
    "non_applicable_modifier_coverage",
    "reference_diagram_hit",
    "structured_output_overall",
)
TIME_METRIC_COLUMNS = ("execution_time",)
COMPARISON_METRIC_COLUMNS = (*METRIC_COLUMNS, *TIME_METRIC_COLUMNS)
LOWER_IS_BETTER_METRICS = ("critical_error", *TIME_METRIC_COLUMNS)

METRIC_DESCRIPTIONS = {
    "diagram_id_hit": (
        "`diagram_id_hit` - 평가: 기대 diagram_id 또는 허용 diagram_id가 검색 결과 metadata에 "
        "포함됐는지 봅니다. 점수: 0=기대 diagram 미검색, 1=기대 diagram 검색."
    ),
    "location_match": (
        "`location_match` - 평가: 기대 사고 장소가 검색 결과 metadata에 포함됐는지 봅니다. "
        "점수: 0=장소 불일치, 1=장소 일치 또는 기대 장소 없음."
    ),
    "party_type_match": (
        "`party_type_match` - 평가: 기대 당사자 유형이 검색 결과 metadata에 포함됐는지 봅니다. "
        "점수: 0=당사자 유형 불일치, 1=당사자 유형 일치 또는 기대값 없음."
    ),
    "chunk_type_match": (
        "`chunk_type_match` - 평가: 기대 chunk type이 검색 결과 metadata에 포함됐는지 봅니다. "
        "점수: 0=chunk type 불일치, 1=chunk type 일치 또는 기대값 없음."
    ),
    "keyword_coverage": (
        "`keyword_coverage` - 평가: 기대 키워드가 검색 결과 본문에 얼마나 포함됐는지 봅니다. "
        "점수: 0=키워드 미포함, 1=모든 키워드 포함, 중간값=포함 비율."
    ),
    "near_miss_not_above_expected": (
        "`near_miss_not_above_expected` - 평가: near-miss diagram이 기대 diagram보다 위에 "
        "랭크되지 않았는지 봅니다. 점수: 0=near-miss가 우선되었거나 기대 diagram 없음, "
        "1=기대 diagram이 near-miss보다 먼저 나옴."
    ),
    "retrieval_relevance": (
        "`retrieval_relevance` - 평가: diagram/location/party/chunk/keyword 검색 체크의 평균입니다. "
        "점수: 0=관련 검색 신호 없음, 1=모든 검색 신호 충족, 중간값=평균 충족률."
    ),
    "critical_error": (
        "`critical_error` - 평가: 기대 diagram, party_type, location 중 치명적인 retrieval mismatch가 "
        "있는지 봅니다. 점수: 0=치명 오류 없음, 1=치명 오류 있음."
    ),
    "intake_is_sufficient": (
        "`intake_is_sufficient` - 평가: intake가 입력 충분/부족 여부를 기대값대로 판단했는지 봅니다. "
        "점수: 0=판단 불일치, 1=판단 일치."
    ),
    "missing_fields_match": (
        "`missing_fields_match` - 평가: intake가 찾아낸 누락 필드 목록이 기대값과 같은지 봅니다. "
        "점수: 0=누락 필드 불일치, 1=누락 필드 일치."
    ),
    "follow_up_contains": (
        "`follow_up_contains` - 평가: follow-up 질문에 기대 문구가 포함됐는지 봅니다. "
        "점수: 0=기대 문구 누락, 1=기대 문구 포함."
    ),
    "forbidden_filter_absent": (
        "`forbidden_filter_absent` - 평가: 만들면 안 되는 metadata filter가 빠져 있는지 봅니다. "
        "점수: 0=금지 filter 포함, 1=금지 filter 없음."
    ),
    "intake_overall": (
        "`intake_overall` - 평가: intake 관련 세부 점수의 평균입니다. "
        "점수: 0=intake 조건 전부 실패, 1=intake 조건 전부 통과, 중간값=평균 통과율."
    ),
    "route_type_match": (
        "`route_type_match` - 평가: router가 기대 route type을 선택했는지 봅니다. "
        "점수: 0=route 불일치, 1=route 일치."
    ),
    "reason_category_match": (
        "`reason_category_match` - 평가: router 판단 이유가 기대 reason category를 포함하는지 봅니다. "
        "점수: 0=이유 category 불일치, 1=이유 category 일치."
    ),
    "router_overall": (
        "`router_overall` - 평가: router 관련 세부 점수의 평균입니다. "
        "점수: 0=router 조건 전부 실패, 1=router 조건 전부 통과, 중간값=평균 통과율."
    ),
    "metadata_filter_match": (
        "`metadata_filter_match` - 평가: 생성된 metadata filter가 기대 filter와 같은지 봅니다. "
        "점수: 0=filter 불일치, 1=filter 일치."
    ),
    "metadata_filter_overall": (
        "`metadata_filter_overall` - 평가: metadata filter 일치와 금지 filter 부재 점수의 평균입니다. "
        "점수: 0=filter 조건 실패, 1=filter 조건 통과, 중간값=평균 통과율."
    ),
    "state_sequence_match": (
        "`state_sequence_match` - 평가: multiturn 대화의 turn별 intake state 변화가 기대 흐름과 "
        "맞는지 봅니다. 점수: 0=state 흐름 불일치, 1=state 흐름 일치."
    ),
    "followup_questions_match": (
        "`followup_questions_match` - 평가: turn별 follow-up 질문이 기대 내용을 포함하는지 봅니다. "
        "점수: 0=질문 흐름 불일치, 1=질문 흐름 일치."
    ),
    "final_metadata_match": (
        "`final_metadata_match` - 평가: multiturn 종료 시 최종 metadata가 기대값과 맞는지 봅니다. "
        "점수: 0=최종 metadata 불일치, 1=최종 metadata 일치."
    ),
    "final_result_type_match": (
        "`final_result_type_match` - 평가: multiturn 종료 결과 타입이 기대값과 맞는지 봅니다. "
        "점수: 0=최종 result type 불일치, 1=최종 result type 일치."
    ),
    "turns_to_ready_match": (
        "`turns_to_ready_match` - 평가: 분석 준비 상태가 된 turn 번호가 기대값과 맞는지 봅니다. "
        "점수: 0=준비 turn 불일치, 1=준비 turn 일치."
    ),
    "multiturn_overall": (
        "`multiturn_overall` - 평가: multiturn 관련 세부 점수의 평균입니다. "
        "점수: 0=multiturn 조건 전부 실패, 1=multiturn 조건 전부 통과, 중간값=평균 통과율."
    ),
    "final_fault_ratio_match": (
        "`final_fault_ratio_match` - 평가: 최종 과실비율이 기대 비율과 같은지 봅니다. "
        "점수: 0=과실비율 불일치, 1=과실비율 일치."
    ),
    "cannot_determine_match": (
        "`cannot_determine_match` - 평가: 판단 불가 여부가 기대값과 맞는지 봅니다. "
        "점수: 0=판단 가능/불가 상태 불일치, 1=상태 일치."
    ),
    "required_evidence_coverage": (
        "`required_evidence_coverage` - 평가: 답변/근거/metadata에 필수 evidence가 얼마나 포함됐는지 "
        "봅니다. 점수: 0=필수 evidence 없음, 1=필수 evidence 전부 포함, 중간값=포함 비율."
    ),
    "party_role_coverage": (
        "`party_role_coverage` - 평가: 기대 당사자 역할 설명이 답변/근거에 얼마나 포함됐는지 봅니다. "
        "점수: 0=역할 근거 없음, 1=역할 근거 전부 포함, 중간값=포함 비율."
    ),
    "applicable_modifier_coverage": (
        "`applicable_modifier_coverage` - 평가: 적용되어야 하는 수정요소가 답변/근거에 얼마나 "
        "포함됐는지 봅니다. 점수: 0=적용 modifier 없음, 1=전부 포함, 중간값=포함 비율."
    ),
    "non_applicable_modifier_coverage": (
        "`non_applicable_modifier_coverage` - 평가: 적용되지 않아야 하는 수정요소 설명이 답변/근거에 "
        "얼마나 포함됐는지 봅니다. 점수: 0=비적용 modifier 근거 없음, 1=전부 포함, 중간값=포함 비율."
    ),
    "reference_diagram_hit": (
        "`reference_diagram_hit` - 평가: structured output 생성 시 기대 reference diagram이 검색 "
        "context에 포함됐는지 봅니다. 점수: 0=reference diagram 미검색, 1=reference diagram 검색."
    ),
    "structured_output_overall": (
        "`structured_output_overall` - 평가: structured output 관련 세부 점수의 평균입니다. "
        "점수: 0=structured output 조건 전부 실패, 1=조건 전부 통과, 중간값=평균 통과율."
    ),
    "execution_time": (
        "`execution_time` - 평가: 각 run의 row별 실행시간 평균입니다. 단위: 초. "
        "낮을수록 같은 평가를 더 빠르게 처리한 run입니다."
    ),
}


def describe_metric(metric: str) -> str:
    return METRIC_DESCRIPTIONS.get(
        metric,
        f"`{metric}` - 평가: dashboard metric score입니다. 점수: 0=실패, 1=통과.",
    )
