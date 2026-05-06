"""대화 턴 라우팅과 처리 결과에 사용하는 데이터 구조입니다."""

from dataclasses import dataclass
from enum import Enum


class RouteType(str, Enum):
    """대화 턴을 보낼 큰 처리 흐름입니다."""

    GENERAL_CHAT = "general_chat"
    ACCIDENT_ANALYSIS = "accident_analysis"


class TurnResultType(str, Enum):
    """사용자에게 실제로 반환된 처리 결과 종류입니다."""

    GENERAL_CHAT = "general_chat"
    ACCIDENT_FOLLOW_UP = "accident_follow_up"
    ACCIDENT_RAG = "accident_rag"


@dataclass(frozen=True)
class RouteDecision:
    """현재 사용자 입력을 어느 흐름으로 보낼지에 대한 판단 결과입니다."""

    route_type: RouteType
    confidence: float = 0.0
    reason: str = ""
