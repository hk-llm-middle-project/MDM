"""사용자 검색 메타데이터를 검색 필터로 변환합니다."""

from rag.service.intake.schema import UserSearchMetadata


def build_metadata_filters(metadata: UserSearchMetadata | None) -> dict[str, object] | None:
    """party_type과 location을 retriever metadata filter로 변환합니다."""
    if metadata is None:
        return None

    filters: dict[str, object] = {}
    if metadata.party_type:
        filters["party_type"] = metadata.party_type
    if metadata.location:
        filters["location"] = metadata.location

    return filters or None

