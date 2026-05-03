"""검색 질의를 문서 taxonomy에 가까운 표현으로 보강합니다."""

from __future__ import annotations

import re

from rag.service.intake.schema import IntakeDecision, QuerySlots, UserSearchMetadata


def enrich_intake_decision(user_input: str, decision: IntakeDecision) -> IntakeDecision:
    """사용자 원문에 명시된 단서를 검색 질의에 보강합니다."""
    retrieval_query = normalize_retrieval_query_terms(
        user_input,
        decision.search_metadata,
    )
    if retrieval_query == decision.search_metadata.retrieval_query:
        return decision

    return IntakeDecision(
        is_sufficient=decision.is_sufficient,
        normalized_description=decision.normalized_description,
        search_metadata=UserSearchMetadata(
            party_type=decision.search_metadata.party_type,
            location=decision.search_metadata.location,
            retrieval_query=retrieval_query,
            query_slots=decision.search_metadata.query_slots,
        ),
        confidence=decision.confidence,
        missing_fields=decision.missing_fields,
        follow_up_questions=decision.follow_up_questions,
    )


def normalize_retrieval_query_terms(
    user_input: str,
    metadata: UserSearchMetadata,
) -> str | None:
    """LLM 검색 질의에 사고 유형 단서를 중복 없이 추가합니다."""
    slot_query = build_retrieval_query_from_slots(metadata.query_slots)
    base_query = metadata.retrieval_query
    if not base_query:
        return slot_query

    remove_terms, add_terms = collect_retrieval_query_terms(user_input, base_query, metadata)
    query = remove_query_terms(base_query, remove_terms)
    query = append_unique_terms(query, add_terms)
    fallback_query = compact_retrieval_query(query)
    if slot_query is None:
        return fallback_query
    return compact_retrieval_query(f"{fallback_query}, {slot_query}")


def build_retrieval_query_from_slots(slots: QuerySlots, max_terms: int = 4) -> str | None:
    """구조화된 사고 단서로 도표 제목형 검색 질의를 만듭니다."""
    terms: list[str] = []
    signal_pair = build_signal_movement_pair(slots)
    movement_pair = build_movement_pair(slots)
    relation = normalize_relation_for_query(slots.relation, slots.road_priority)

    if slots.special_condition:
        append_compact_term(terms, slots.special_condition)
    if signal_pair:
        append_compact_term(terms, signal_pair)
    if movement_pair:
        append_compact_term(terms, movement_pair)
    for road_priority_term in build_road_priority_terms(slots.road_priority):
        append_compact_term(terms, road_priority_term)
    if relation:
        append_compact_term(terms, relation)

    compact_terms = [term for term in terms if not is_broad_query_term(term)]
    if not compact_terms:
        return None
    return ", ".join(compact_terms[:max_terms])


def build_signal_movement_pair(slots: QuerySlots) -> str | None:
    if not (slots.a_signal and slots.a_movement and slots.b_signal and slots.b_movement):
        return None
    return (
        f"{slots.a_signal}{normalize_movement_for_query(slots.a_movement)} 대 "
        f"{slots.b_signal}{normalize_movement_for_query(slots.b_movement)}"
    )


def build_movement_pair(slots: QuerySlots) -> str | None:
    if not (slots.a_movement and slots.b_movement):
        return None
    a_movement = normalize_movement_for_query(slots.a_movement)
    b_movement = normalize_movement_for_query(slots.b_movement)
    if a_movement == "추돌" or b_movement == "추돌":
        return "추돌사고"
    if a_movement == "진로변경" and b_movement == "진로변경":
        return "동시 진로변경 사고"
    return f"{a_movement} 대 {b_movement} 사고"


def normalize_movement_for_query(movement: str) -> str:
    normalized = movement.replace(" ", "")
    if normalized in {"차로변경", "진로변경"}:
        return "진로변경"
    if normalized == "비보호좌회전":
        return "비보호좌회전"
    return normalized


def build_road_priority_terms(road_priority: str | None) -> list[str]:
    if not road_priority:
        return []

    terms: list[str] = []
    normalized = normalize_spacing(road_priority)
    if contains_any(normalized, ("대로", "소로")):
        terms.append("대로 소로 교차로")
    if "동일 폭" in normalized or "동일폭" in normalized:
        terms.append("동일 폭 교차로")
    for term in (
        "오른쪽 소로",
        "왼쪽 소로",
        "오른쪽 대로",
        "왼쪽 대로",
        "오른쪽 도로",
        "왼쪽 도로",
    ):
        if term in normalized:
            terms.append(term)
    if not terms:
        terms.append(normalized)
    return terms


def normalize_relation_for_query(
    relation: str | None,
    road_priority: str | None,
) -> str | None:
    if not relation:
        return None
    normalized = normalize_spacing(relation)
    if (
        normalized == "상대차량이 맞은편에서 진입"
        and road_priority
        and contains_any(
            road_priority,
            ("오른쪽", "왼쪽", "대로", "소로", "동일 폭", "동일폭"),
        )
    ):
        return "상대차량이 측면에서 진입"
    if normalized in {"도로 외 진입", "상대차량이 도로 외 진입"}:
        return "도로 외 진입"
    return normalized


def collect_retrieval_query_terms(
    user_input: str,
    base_query: str,
    metadata: UserSearchMetadata,
) -> tuple[list[str], list[str]]:
    """원문에서 확실한 taxonomy 단서를 수집합니다."""
    user_text = normalize_spacing(user_input)
    text = normalize_spacing(f"{user_input} {base_query}")
    remove_terms: list[str] = []
    terms: list[str] = []

    if "양쪽 신호등" in text:
        terms.append("양쪽 신호등 있는 교차로")
    if contains_any(text, ("신호등이 없는", "신호등 없는")):
        terms.append("신호등 없는 교차로")

    if "같은 방향" in text:
        terms.append("같은 방향 진행차량 상호간의 사고")
    if contains_any(text, ("중앙선", "역주행")):
        terms.append("중앙선 침범 사고")
    if "추돌" in text:
        terms.append("추돌사고")
    if contains_any(text, ("진로변경", "차로를 변경", "차로 변경")):
        terms.append("진로변경 사고")
    if contains_any(text, ("도로가 아닌 장소", "도로로 우회전 진입", "도로로 진입")):
        terms.append("도로가 아닌 장소에서 도로로 진입")

    if "맞은편" in user_input:
        remove_terms.append("상대차량이 측면에서 진입")
        terms.append("상대차량이 맞은편에서 진입")
    elif contains_any(user_input, ("측면", "서로 다른 방향")):
        terms.append("상대차량이 측면에서 진입")
    elif contains_any(user_text, ("오른쪽 도로", "왼쪽 도로", "오른쪽 소로", "왼쪽 소로", "오른쪽 대로", "왼쪽 대로")):
        remove_terms.append("상대차량이 맞은편에서 진입")
        terms.append("상대차량이 측면에서 진입")

    terms.extend(collect_road_terms(user_text))
    terms.extend(collect_movement_terms(user_text))
    terms.extend(collect_signal_terms(user_input))
    terms.extend(collect_specific_case_terms(user_text, metadata))

    return remove_terms, terms


def collect_road_terms(text: str) -> list[str]:
    terms: list[str] = []
    if "동일 폭" in text or "동일폭" in text:
        terms.append("동일 폭 교차로")
    if "대로" in text and "소로" in text:
        terms.append("대로 소로 교차로")
    if "오른쪽 소로" in text:
        terms.append("오른쪽 소로")
    if "왼쪽 소로" in text:
        terms.append("왼쪽 소로")
    if "오른쪽 대로" in text:
        terms.append("오른쪽 대로")
    if "왼쪽 대로" in text:
        terms.append("왼쪽 대로")
    if "오른쪽 도로" in text:
        terms.append("오른쪽 도로")
    if "왼쪽 도로" in text:
        terms.append("왼쪽 도로")
    return terms


def collect_movement_terms(text: str) -> list[str]:
    terms: list[str] = []
    if "비보호 좌회전" in text:
        terms.append("비보호 좌회전 대 직진")
    if "좌회전" in text and "직진" in text:
        terms.append("직진 대 좌회전 사고")
    if "우회전" in text and "직진" in text:
        terms.append("직진 대 우회전 사고")
    if "서로 직진" in text or both_parties_move(text, "직진"):
        terms.append("직진 대 직진 사고")
    return terms


def collect_signal_terms(user_input: str) -> list[str]:
    terms: list[str] = []
    a_signal = detect_party_signal(user_input, "A")
    b_signal = detect_party_signal(user_input, "B")
    a_movement = detect_party_movement(user_input, "A")
    b_movement = detect_party_movement(user_input, "B")

    if a_signal and a_movement:
        terms.append(f"A {a_signal}{a_movement}")
    if b_signal and b_movement:
        terms.append(f"B {b_signal}{b_movement}")
    if a_signal and b_signal and a_movement and b_movement:
        terms.append(f"{a_signal}{a_movement} 대 {b_signal}{b_movement}")
    return terms


def collect_specific_case_terms(text: str, metadata: UserSearchMetadata) -> list[str]:
    terms: list[str] = []
    if metadata.party_type == "자전거":
        terms.append("자전거 대 자동차")
    elif metadata.party_type == "자동차":
        terms.append("자동차 대 자동차")

    if "녹색 또는 적색" in text and "좌회전" in text:
        terms.append("녹색직진 대 녹색(적색)신호위반 좌회전")
    if "녹색신호를 위반" in text and "좌회전 진입 후 황색" in text:
        terms.append("황색직진 대 녹색신호위반 좌회전 진입 후 황색에 충돌")
    if "비보호 좌회전" in text and "맞은편" in text:
        terms.append("녹색 비보호 좌회전 대 맞은편 녹색 직진")
    if "동시에 차로" in text or "좌우에서 동시에" in text:
        terms.append("동시 진로변경 사고")
    return terms


def detect_party_signal(text: str, party: str) -> str | None:
    """A/B 주변 문맥에서 신호 표현을 찾습니다."""
    shared_signal = detect_shared_party_signal(text)
    if shared_signal:
        return shared_signal

    context = get_party_context(text, party)
    if context is None:
        return None
    signal_patterns = (
        ("적색점멸", "적색점멸"),
        ("황색점멸", "황색점멸"),
        ("녹색", "녹색"),
        ("황색", "황색"),
        ("적색", "적색"),
    )
    for needle, signal in signal_patterns:
        if needle in context and "신호" in context:
            return signal
    return None


def detect_party_movement(text: str, party: str) -> str | None:
    """A/B 주변 문맥에서 진행 방향을 찾습니다."""
    shared_movement = detect_shared_party_movement(text)
    if shared_movement:
        return shared_movement

    context = get_party_context(text, party)
    if context is None:
        return None
    movement_patterns = (
        ("비보호 좌회전", "비보호좌회전"),
        ("비보호좌회전", "비보호좌회전"),
        ("좌회전", "좌회전"),
        ("우회전", "우회전"),
        ("직진", "직진"),
    )
    for needle, movement in movement_patterns:
        if needle in context:
            return movement
    return None


def get_party_context(text: str, party: str) -> str | None:
    """해당 당사자 언급부터 다음 당사자 언급 전까지의 문맥을 반환합니다."""
    marker_pattern = re.compile(r"[AB](?:차량|자동차|자전거)?(?:은|는|이|가|과|와)?")
    markers = [
        (match.group(0)[0], match.start(), match.end())
        for match in marker_pattern.finditer(text)
        if match.group(0)[0] in {"A", "B"}
    ]
    for index, (found_party, start, _) in enumerate(markers):
        if found_party != party:
            continue
        end = markers[index + 1][1] if index + 1 < len(markers) else len(text)
        return text[start:end]
    return None


def detect_shared_party_signal(text: str) -> str | None:
    """A와 B가 모두 같은 신호라는 표현을 처리합니다."""
    if not re.search(r"A(?:차량|자동차|자전거)?(?:과|와)?\s*B(?:차량|자동차|자전거)?(?:이|가)?\s*모두", text):
        return None
    for needle, signal in (
        ("적색점멸", "적색점멸"),
        ("황색점멸", "황색점멸"),
        ("녹색", "녹색"),
        ("황색", "황색"),
        ("적색", "적색"),
    ):
        if needle in text and "신호" in text:
            return signal
    return None


def detect_shared_party_movement(text: str) -> str | None:
    """A와 B가 모두 같은 진행방향이라는 표현을 처리합니다."""
    if not re.search(r"A(?:차량|자동차|자전거)?(?:과|와)?\s*B(?:차량|자동차|자전거)?(?:이|가)?\s*모두", text):
        return None
    if "좌회전" in text:
        return "좌회전"
    if "우회전" in text:
        return "우회전"
    if "직진" in text:
        return "직진"
    return None


def both_parties_move(text: str, movement: str) -> bool:
    return detect_party_movement(text, "A") == movement and detect_party_movement(text, "B") == movement


def remove_query_terms(query: str, remove_terms: list[str]) -> str:
    if not remove_terms:
        return query
    parts = [part.strip() for part in query.split(",") if part.strip()]
    filtered_parts = [part for part in parts if part not in remove_terms]
    return ", ".join(filtered_parts)


def append_unique_terms(query: str, terms: list[str]) -> str:
    parts = [part.strip() for part in query.split(",") if part.strip()]
    for term in terms:
        if term and term not in parts and term not in query:
            parts.append(term)
    return ", ".join(parts)


def compact_retrieval_query(query: str, max_terms: int = 4) -> str:
    """검색 질의를 도표 제목형 핵심 표현 위주로 압축합니다."""
    parts = [part.strip() for part in query.split(",") if part.strip()]
    selected: list[str] = []

    for selector in (
        is_specific_case_term,
        is_signal_pair_term,
        is_movement_pair_term,
        is_road_priority_term,
        is_location_relation_term,
        is_accident_family_term,
    ):
        for part in parts:
            if selector(part):
                append_compact_term(selected, part)
                if len(selected) >= max_terms:
                    return ", ".join(selected)

    for part in parts:
        if not is_broad_query_term(part):
            append_compact_term(selected, part)
            if len(selected) >= max_terms:
                return ", ".join(selected)

    return ", ".join(selected or parts[:max_terms])


def append_compact_term(parts: list[str], term: str) -> None:
    if term and term not in parts:
        parts.append(term)


def is_specific_case_term(term: str) -> bool:
    return contains_any(
        term,
        (
            "녹색(적색)신호위반 좌회전",
            "녹색신호위반 좌회전 진입 후 황색",
            "녹색 비보호 좌회전",
            "동시 진로변경",
            "도로가 아닌 장소에서 도로로 진입",
            "중앙선 침범",
        ),
    )


def is_signal_pair_term(term: str) -> bool:
    return " 대 " in term and contains_any(
        term,
        ("녹색", "황색", "적색", "점멸", "비보호"),
    )


def is_movement_pair_term(term: str) -> bool:
    return " 대 " in term and contains_any(
        term,
        ("직진", "좌회전", "우회전", "진로변경", "추돌"),
    )


def is_road_priority_term(term: str) -> bool:
    return contains_any(
        term,
        ("동일 폭", "대로 소로", "오른쪽 소로", "왼쪽 소로", "오른쪽 대로", "왼쪽 대로"),
    )


def is_location_relation_term(term: str) -> bool:
    return contains_any(
        term,
        (
            "상대차량이 측면에서 진입",
            "상대차량이 맞은편에서 진입",
            "오른쪽 도로",
            "왼쪽 도로",
            "도로 외 진입",
        ),
    )


def is_accident_family_term(term: str) -> bool:
    return contains_any(
        term,
        ("추돌사고", "진로변경 사고", "직진 대 직진 사고", "직진 대 좌회전 사고", "직진 대 우회전 사고"),
    )


def is_broad_query_term(term: str) -> bool:
    return term in {
        "자동차 대 자동차",
        "자동차 대 자전거",
        "자전거 대 자동차",
        "양쪽 신호등 있는 교차로",
        "신호등 없는 교차로",
        "교차로 사고",
        "같은 방향 진행차량 상호간의 사고",
        "같은 방향 진행차량 상호간 사고",
    }


def contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def normalize_spacing(text: str) -> str:
    return " ".join(text.split())
