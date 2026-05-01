"""검색 질의를 문서 taxonomy에 가까운 표현으로 보강합니다."""

from __future__ import annotations

import re

from rag.service.intake.schema import IntakeDecision, UserSearchMetadata


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
    base_query = metadata.retrieval_query
    if not base_query:
        return None

    remove_terms, add_terms = collect_retrieval_query_terms(user_input, base_query, metadata)
    query = remove_query_terms(base_query, remove_terms)
    return append_unique_terms(query, add_terms)


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


def contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def normalize_spacing(text: str) -> str:
    return " ".join(text.split())
