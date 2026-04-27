"""사용자 검색 메타데이터를 검색 필터로 변환합니다."""

from rag.service.intake.schema import UserSearchMetadata


def build_metadata_filters(metadata: UserSearchMetadata | None) -> dict[str, object] | None:
    """party_type과 location을 retriever metadata filter로 변환합니다."""
    if metadata is None:
        return None

    conditions: list[dict[str, object]] = []
    if metadata.party_type:
        conditions.append({"party_type": metadata.party_type})
    if metadata.location:
        conditions.append({"location": metadata.location})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
