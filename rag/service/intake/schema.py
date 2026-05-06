"""사고 입력 판단 결과에 사용하는 데이터 구조입니다."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QuerySlots:
    """검색 질의 생성을 위한 구조화된 사고 단서입니다."""

    road_control: str | None = None
    relation: str | None = None
    a_signal: str | None = None
    b_signal: str | None = None
    a_movement: str | None = None
    b_movement: str | None = None
    road_priority: str | None = None
    special_condition: str | None = None


@dataclass(frozen=True)
class UserSearchMetadata:
    """사용자 입력에서 검색 필터로 사용할 최소 메타데이터입니다."""

    party_type: str | None = None
    location: str | None = None
    retrieval_query: str | None = None
    query_slots: QuerySlots = field(default_factory=QuerySlots)


@dataclass(frozen=True)
class IntakeState:
    """세션 안에서 누적되는 intake 진행 상태입니다."""

    attempt_count: int = 0
    search_metadata: UserSearchMetadata = field(default_factory=UserSearchMetadata)
    last_missing_fields: list[str] = field(default_factory=list)
    last_follow_up_questions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MissingField:
    """분석 전에 더 필요한 정보 한 가지를 나타냅니다."""

    name: str
    reason: str


@dataclass(frozen=True)
class IntakeDecision:
    """사용자 입력이 분석에 충분한지에 대한 판단 결과입니다."""

    is_sufficient: bool
    normalized_description: str
    search_metadata: UserSearchMetadata = field(default_factory=UserSearchMetadata)
    confidence: dict[str, float] = field(default_factory=dict)
    missing_fields: list[MissingField] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
