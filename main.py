"""새로운 RAG 앱의 Streamlit 채팅 UI입니다."""

from __future__ import annotations

import base64
from collections.abc import Callable
import logging
from html import escape
from io import BytesIO
from pathlib import Path
import re

from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

from config import (
    DEFAULT_CHUNKER_STRATEGY,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_ENSEMBLE_BM25_WEIGHT,
    DEFAULT_ENSEMBLE_CANDIDATE_K,
    DEFAULT_ENSEMBLE_USE_CHUNK_ID,
    DEFAULT_LOADER_STRATEGY,
    DEFAULT_MODE,
    DEFAULT_RERANKER_CANDIDATE_K,
    DEFAULT_RERANKER_FINAL_K,
    DEFAULT_RERANKER_STRATEGY,
    DEFAULT_RETRIEVER_STRATEGY,
    ENSEMBLE_CANDIDATE_K_OPTIONS,
    ENSEMBLE_ID_KEY,
    ENSEMBLE_RETRIEVER_STRATEGIES,
    MODE_PRESETS,
    get_debug_progress_enabled,
)
from rag.embeddings import EMBEDDING_STRATEGIES
from rag.pipeline.reranker import (
    CrossEncoderRerankerConfig,
    FlashrankRerankerConfig,
    LLMScoreRerankerConfig,
)
from rag.pipeline.retrieval import RetrievalPipelineConfig
from rag.pipeline.retriever import EnsembleRetrieverConfig, RETRIEVAL_STRATEGIES
from rag.service.analysis.answer_schema import RetrievedContext
from rag.service.conversation.app_service import answer_question_with_intake
from rag.service.presentation.result_service import truncate_context
from rag.service.progress import PROGRESS_INPUT, PROGRESS_RESULT
from rag.service.session import ConversationStore, get_conversation_store
from rag.service.tracing import TraceContext


SHOW_RETRIEVED_CONTEXTS = True
LOADER_STRATEGY_OPTIONS = ("pdfplumber", "llamaparser", "upstage")
CHUNKER_STRATEGY_OPTIONS_BY_LOADER = {
    "pdfplumber": ("fixed", "recursive", "semantic"),
    "llamaparser": ("fixed", "recursive", "markdown", "case-boundary", "semantic"),
    "upstage": ("raw", "custom"),
}
EMBEDDING_PROVIDER_OPTIONS = tuple(EMBEDDING_STRATEGIES)
RETRIEVER_STRATEGY_OPTIONS = tuple(
    strategy for strategy in RETRIEVAL_STRATEGIES if strategy != "vectorstore"
)
RERANKER_STRATEGY_OPTIONS = ("none", "cross-encoder", "flashrank", "llm-score")
USER_ID = "local"
PROJECT_ROOT = Path(__file__).resolve().parent
MAX_DONUT_PERCENT = 99
MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(?!■\s)(.+)$")
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
logger = logging.getLogger(__name__)


def render_progress_html(label: str, *, loading: bool = True) -> str:
    spinner = '<span class="progress-spinner" aria-hidden="true"></span>' if loading else ""
    return (
        '<div class="progress-line">'
        f"{spinner}"
        f'<span>{escape(label)}</span>'
        "</div>"
    )


class ProgressReporter:
    """Render user progress as plain text or debug details."""

    def __init__(self, *, debug_enabled: bool, streamlit_module=st):
        self.debug_enabled = debug_enabled
        self._streamlit = streamlit_module
        self._last_label = PROGRESS_INPUT
        if debug_enabled:
            self._outer_container = streamlit_module.status(
                "분석 진행 상황",
                expanded=True,
            )
            self._container = self._outer_container.status(PROGRESS_INPUT, expanded=True)
        else:
            self._outer_container = None
            self._container = streamlit_module.empty()
            self._container.markdown(
                render_progress_html(PROGRESS_INPUT),
                unsafe_allow_html=True,
            )

    def __call__(self, label: str) -> None:
        self.update(label)

    def update(self, label: str) -> None:
        if label == self._last_label:
            return
        if self.debug_enabled:
            self._container.update(state="complete", expanded=False)
            self._container = self._outer_container.status(label, expanded=True)
        else:
            self._container.markdown(
                render_progress_html(label),
                unsafe_allow_html=True,
            )
        self._last_label = label

    def detail(self, detail: str) -> None:
        if self.debug_enabled:
            self._container.write(detail)

    def complete(self) -> None:
        already_showing_result = self._last_label == PROGRESS_RESULT
        if self.debug_enabled:
            if not already_showing_result:
                self._container.update(state="complete", expanded=False)
                self._container = self._outer_container.status(
                    PROGRESS_RESULT,
                    expanded=True,
                )
            self._container.update(state="complete", expanded=False)
            self._outer_container.update(state="complete", expanded=False)
        elif not already_showing_result:
            self._container.markdown(
                render_progress_html(PROGRESS_RESULT),
                unsafe_allow_html=True,
            )
        self._last_label = PROGRESS_RESULT

    def error(self) -> None:
        self._last_label = "오류가 발생했습니다"
        if self.debug_enabled:
            self._container.update(
                label="오류가 발생했습니다",
                state="error",
                expanded=True,
            )
            self._outer_container.update(state="error", expanded=True)
        else:
            self._container.markdown(
                render_progress_html("오류가 발생했습니다", loading=False),
                unsafe_allow_html=True,
            )


def create_progress_reporter(
    debug_enabled: bool,
    *,
    streamlit_module=st,
) -> ProgressReporter:
    return ProgressReporter(
        debug_enabled=debug_enabled,
        streamlit_module=streamlit_module,
    )


def ensure_active_session(store: ConversationStore) -> str:
    """활성 세션 id를 반환하고, 없으면 새로 생성합니다."""
    sessions = store.list_sessions(USER_ID)
    if not sessions:
        session = store.create_session(USER_ID, title="Session 1")
        store.set_active_session(USER_ID, session.session_id)
        return session.session_id

    active_session = store.get_active_session(USER_ID)
    session_ids = {session.session_id for session in sessions}
    if active_session in session_ids:
        return active_session

    fallback_session = sessions[0].session_id
    store.set_active_session(USER_ID, fallback_session)
    return fallback_session


def delete_session_and_select_fallback(store: ConversationStore, session_id: str) -> str:
    """세션을 삭제하고 남은 세션 또는 새 세션을 활성화합니다."""
    store.delete_session(USER_ID, session_id)
    sessions = store.list_sessions(USER_ID)
    if not sessions:
        session = store.create_session(USER_ID, title="Session 1")
        store.set_active_session(USER_ID, session.session_id)
        return session.session_id

    active_session = store.get_active_session(USER_ID)
    session_ids = {session.session_id for session in sessions}
    if active_session in session_ids:
        return active_session

    fallback_session_id = sessions[0].session_id
    store.set_active_session(USER_ID, fallback_session_id)
    return fallback_session_id


def init_state(store: ConversationStore) -> None:
    if "active_session" not in st.session_state:
        st.session_state.active_session = ensure_active_session(store)
    if "selected_mode" not in st.session_state:
        apply_mode_preset(DEFAULT_MODE, store)
    if "loader_strategy" not in st.session_state:
        loader_strategy = store.get_loader_strategy(USER_ID) or DEFAULT_LOADER_STRATEGY
        if loader_strategy not in LOADER_STRATEGY_OPTIONS:
            loader_strategy = DEFAULT_LOADER_STRATEGY
        st.session_state.loader_strategy = loader_strategy
    if "embedding_provider" not in st.session_state:
        st.session_state.embedding_provider = DEFAULT_EMBEDDING_PROVIDER
    if "chunker_strategy" not in st.session_state:
        st.session_state.chunker_strategy = normalize_chunker_strategy(
            DEFAULT_CHUNKER_STRATEGY,
            st.session_state.loader_strategy,
        )
    if "retriever_strategy" not in st.session_state:
        st.session_state.retriever_strategy = DEFAULT_RETRIEVER_STRATEGY
    if "reranker_strategy" not in st.session_state:
        st.session_state.reranker_strategy = DEFAULT_RERANKER_STRATEGY
    if "ensemble_bm25_weight" not in st.session_state:
        st.session_state.ensemble_bm25_weight = DEFAULT_ENSEMBLE_BM25_WEIGHT
    if "ensemble_candidate_k" not in st.session_state:
        st.session_state.ensemble_candidate_k = DEFAULT_ENSEMBLE_CANDIDATE_K
    if "ensemble_use_chunk_id" not in st.session_state:
        st.session_state.ensemble_use_chunk_id = DEFAULT_ENSEMBLE_USE_CHUNK_ID
    st.session_state.active_session = ensure_active_session(store)


def get_chunker_strategy_options(loader_strategy: str) -> tuple[str, ...]:
    return CHUNKER_STRATEGY_OPTIONS_BY_LOADER.get(
        loader_strategy,
        CHUNKER_STRATEGY_OPTIONS_BY_LOADER[DEFAULT_LOADER_STRATEGY],
    )


def normalize_chunker_strategy(chunker_strategy: str, loader_strategy: str) -> str:
    options = get_chunker_strategy_options(loader_strategy)
    if chunker_strategy in options:
        return chunker_strategy
    return options[0]


def apply_mode_preset(mode: str, store: ConversationStore | None = None) -> None:
    preset = MODE_PRESETS.get(mode, MODE_PRESETS[DEFAULT_MODE])
    loader_strategy = str(preset["loader_strategy"])
    st.session_state.selected_mode = mode if mode in MODE_PRESETS else DEFAULT_MODE
    st.session_state.loader_strategy = loader_strategy
    st.session_state.chunker_strategy = normalize_chunker_strategy(
        str(preset["chunker_strategy"]),
        loader_strategy,
    )
    st.session_state.embedding_provider = str(preset["embedding_provider"])
    st.session_state.retriever_strategy = str(preset["retriever_strategy"])
    st.session_state.reranker_strategy = str(preset["reranker_strategy"])
    st.session_state.ensemble_bm25_weight = float(preset["ensemble_bm25_weight"])
    st.session_state.ensemble_candidate_k = int(preset["ensemble_candidate_k"])
    st.session_state.ensemble_use_chunk_id = bool(preset["ensemble_use_chunk_id"])
    if store is not None:
        store.set_loader_strategy(USER_ID, loader_strategy)


def render_sidebar(store: ConversationStore) -> tuple[str, str, str, str, float, int, bool, str]:
    st.sidebar.markdown('<div class="sidebar-title">Sessions</div>', unsafe_allow_html=True)
    if st.sidebar.button("New Session", use_container_width=True):
        session_count = len(store.list_sessions(USER_ID))
        session = store.create_session(USER_ID, title=f"Session {session_count + 1}")
        store.set_active_session(USER_ID, session.session_id)
        st.session_state.active_session = session.session_id
        st.rerun()

    active_session_id = st.session_state.get("active_session", store.get_active_session(USER_ID))
    for session in store.list_sessions(USER_ID):
        session_column, delete_column = st.sidebar.columns([0.8, 0.2])
        with session_column:
            if st.button(
                session.title,
                key=f"session-{session.session_id}",
                type="primary" if session.session_id == active_session_id else "secondary",
                use_container_width=True,
            ):
                store.set_active_session(USER_ID, session.session_id)
                st.session_state.active_session = session.session_id
                st.rerun()
        with delete_column:
            if st.button(
                "×",
                key=f"delete-session-{session.session_id}",
                help=f"{session.title} 삭제",
                use_container_width=True,
            ):
                st.session_state.active_session = delete_session_and_select_fallback(
                    store,
                    session.session_id,
                )
                st.rerun()

    st.sidebar.markdown('<div class="mode-title">Mode</div>', unsafe_allow_html=True)
    current_mode = st.session_state.get("selected_mode", DEFAULT_MODE)
    if current_mode not in MODE_PRESETS:
        apply_mode_preset(DEFAULT_MODE, store)
        current_mode = DEFAULT_MODE
    for mode in MODE_PRESETS:
        if st.sidebar.button(
            mode,
            key=f"mode-{mode}",
            type="primary" if mode == current_mode else "secondary",
            use_container_width=True,
        ):
            apply_mode_preset(mode, store)
            st.rerun()

    st.session_state.loader_strategy = st.session_state.get(
        "loader_strategy",
        DEFAULT_LOADER_STRATEGY,
    )
    st.session_state.chunker_strategy = normalize_chunker_strategy(
        st.session_state.get("chunker_strategy", DEFAULT_CHUNKER_STRATEGY),
        st.session_state.loader_strategy,
    )
    st.session_state.embedding_provider = st.session_state.get(
        "embedding_provider",
        DEFAULT_EMBEDDING_PROVIDER,
    )
    st.session_state.retriever_strategy = st.session_state.get(
        "retriever_strategy",
        DEFAULT_RETRIEVER_STRATEGY,
    )
    st.session_state.reranker_strategy = st.session_state.get("reranker_strategy", "none")

    return (
        st.session_state.loader_strategy,
        st.session_state.chunker_strategy,
        st.session_state.embedding_provider,
        st.session_state.retriever_strategy,
        st.session_state.ensemble_bm25_weight,
        st.session_state.ensemble_candidate_k,
        st.session_state.ensemble_use_chunk_id,
        st.session_state.reranker_strategy,
    )


def build_ensemble_weight_label(bm25_weight: float) -> str:
    """앙상블 slider 라벨을 반환합니다."""
    del bm25_weight
    return "앙상블 검색 가중치"


def build_ensemble_weight_caption_html(bm25_weight: float) -> str:
    """BM25/Dense 앙상블 가중치를 slider 아래 좌우 캡션으로 표시합니다."""
    bm25_percent = round(bm25_weight * 100)
    dense_percent = 100 - bm25_percent
    return f"""
<div style="display: flex; justify-content: space-between; gap: 0.5rem; margin-top: -0.65rem; margin-bottom: 0.75rem; font-size: 0.78rem; font-weight: 700;">
  <span style="color: #2563eb;">BM25 {bm25_percent}%</span>
  <span style="color: #059669;">Dense {dense_percent}%</span>
</div>
"""


def build_ensemble_slider_css(bm25_weight: float) -> str:
    """Streamlit slider 트랙을 BM25/Dense 가중치 gradient로 표시합니다."""
    bm25_percent = round(bm25_weight * 100)
    return f"""
<style>
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child {{
  background: linear-gradient(
    90deg,
    #2563eb 0%,
    #2563eb {bm25_percent}%,
    #059669 {bm25_percent}%,
    #059669 100%
  ) !important;
  height: 0.8rem !important;
  border-radius: 999px !important;
}}
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child > div {{
  background: transparent !important;
}}
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child > div > div {{
  background: transparent !important;
}}
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child > div > div > div {{
  background: transparent !important;
}}
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] [role="slider"] {{
  width: 4px !important;
  height: 1rem !important;
  border-radius: 999px !important;
  background: #111827 !important;
  border: 1px solid #ffffff !important;
  box-shadow: none !important;
  opacity: 1 !important;
}}
[data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBar"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stSliderTickBarMax"],
[data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] > div:nth-child(2) {{
  display: none !important;
}}
</style>
"""


def build_pipeline_config(
    retriever_strategy: str,
    ensemble_bm25_weight: float = DEFAULT_ENSEMBLE_BM25_WEIGHT,
    ensemble_candidate_k: int = DEFAULT_ENSEMBLE_CANDIDATE_K,
    ensemble_use_chunk_id: bool = DEFAULT_ENSEMBLE_USE_CHUNK_ID,
    reranker_strategy: str = DEFAULT_RERANKER_STRATEGY,
) -> RetrievalPipelineConfig:
    """Streamlit 선택값을 retrieval pipeline 설정으로 변환합니다."""
    reranker_kwargs: dict[str, object] = {}
    if reranker_strategy == "cross-encoder":
        reranker_kwargs = {
            "candidate_k": DEFAULT_RERANKER_CANDIDATE_K,
            "final_k": DEFAULT_RERANKER_FINAL_K,
            "reranker_strategy": "cross-encoder",
            "reranker_config": CrossEncoderRerankerConfig(),
        }
    elif reranker_strategy == "flashrank":
        reranker_kwargs = {
            "candidate_k": DEFAULT_RERANKER_CANDIDATE_K,
            "final_k": DEFAULT_RERANKER_FINAL_K,
            "reranker_strategy": "flashrank",
            "reranker_config": FlashrankRerankerConfig(),
        }
    elif reranker_strategy == "llm-score":
        reranker_kwargs = {
            "candidate_k": DEFAULT_RERANKER_CANDIDATE_K,
            "final_k": DEFAULT_RERANKER_FINAL_K,
            "reranker_strategy": "llm-score",
            "reranker_config": LLMScoreRerankerConfig(),
        }

    if retriever_strategy not in ENSEMBLE_RETRIEVER_STRATEGIES:
        return RetrievalPipelineConfig(
            retriever_strategy=retriever_strategy,
            **reranker_kwargs,
        )

    bm25_weight = round(ensemble_bm25_weight, 2)
    dense_weight = round(1.0 - bm25_weight, 2)
    candidate_k = int(ensemble_candidate_k)
    id_key = ENSEMBLE_ID_KEY if ensemble_use_chunk_id else None
    return RetrievalPipelineConfig(
        retriever_strategy=retriever_strategy,
        retriever_config=EnsembleRetrieverConfig(
            weights=(bm25_weight, dense_weight),
            bm25_k=candidate_k,
            dense_k=candidate_k,
            id_key=id_key,
        ),
        **reranker_kwargs,
    )


def build_fault_ratio_metadata(
    fault_ratio_a: int | None,
    fault_ratio_b: int | None,
) -> dict[str, object]:
    """저장 가능한 과실비율 metadata를 만듭니다."""
    if fault_ratio_a is None or fault_ratio_b is None:
        return {}
    return {
        "fault_ratio_a": fault_ratio_a,
        "fault_ratio_b": fault_ratio_b,
    }


def estimate_keyword_match_percent(question: str, content: str) -> int:
    question_terms = {
        token.strip(".,!?()[]{}:;\"'").lower()
        for token in question.split()
        if len(token.strip(".,!?()[]{}:;\"'")) >= 2
    }
    if not question_terms:
        return 0

    content_lower = content.lower()
    matched = sum(1 for term in question_terms if term in content_lower)
    return min(MAX_DONUT_PERCENT, round((matched / len(question_terms)) * MAX_DONUT_PERCENT))


def estimate_rerank_match_percent(metadata: dict[str, object]) -> int | None:
    for key in ("rerank_score", "score", "similarity_score"):
        value = metadata.get(key)
        if isinstance(value, int | float):
            if 0 <= value <= 1:
                return min(MAX_DONUT_PERCENT, round(value * MAX_DONUT_PERCENT))
            if 1 < value <= MAX_DONUT_PERCENT:
                return round(value)
            if MAX_DONUT_PERCENT < value <= 100:
                return MAX_DONUT_PERCENT
    return None


def read_rerank_score(metadata: dict[str, object]) -> float | None:
    value = metadata.get("rerank_score")
    if isinstance(value, int | float):
        return float(value)
    return None


def estimate_match_percent(
    question: str,
    content: str,
    metadata: dict[str, object],
    reranker_strategy: str = DEFAULT_RERANKER_STRATEGY,
) -> int:
    """점수 metadata가 없을 때 화면 표시용 일치율을 보수적으로 추정합니다."""
    del reranker_strategy
    score_percent = estimate_rerank_match_percent(metadata)
    if score_percent is not None:
        return score_percent
    return estimate_keyword_match_percent(question, content)


def build_retrieved_context_metadata(
    retrieved_contexts: list[RetrievedContext],
    question: str = "",
    reranker_strategy: str = DEFAULT_RERANKER_STRATEGY,
) -> list[dict[str, object]]:
    """검색 근거 조각과 이미지를 다시 보여줄 최소 metadata를 만듭니다."""
    rendered_contexts: list[dict[str, object]] = []
    for context in retrieved_contexts[:5]:
        rendered_context: dict[str, object] = {
            "content": truncate_context(context.content),
            "match_percent": estimate_match_percent(
                question,
                context.content,
                context.metadata,
                reranker_strategy,
            ),
        }
        rerank_score = read_rerank_score(context.metadata)
        if rerank_score is not None:
            rendered_context["rerank_score"] = rerank_score
        metadata_image_path = context.metadata.get("image_path")
        if isinstance(metadata_image_path, str) and metadata_image_path:
            rendered_context["image_path"] = metadata_image_path
        for key in ("image_path", "source", "page", "diagram_id"):
            value = context.metadata.get(key)
            if isinstance(value, str | int):
                rendered_context[key] = value
        rendered_contexts.append(rendered_context)
    return rendered_contexts


def build_assistant_metadata(
    result,
    question: str = "",
    reranker_strategy: str = DEFAULT_RERANKER_STRATEGY,
) -> dict[str, object]:
    """assistant 메시지 렌더링에 필요한 부가 정보를 저장합니다."""
    metadata = build_fault_ratio_metadata(result.fault_ratio_a, result.fault_ratio_b)
    retrieved_contexts = build_retrieved_context_metadata(
        result.retrieved_contexts,
        question,
        reranker_strategy,
    )
    if retrieved_contexts:
        metadata["retrieved_contexts"] = retrieved_contexts
    metadata["reranker_strategy"] = reranker_strategy
    return metadata


def read_fault_ratio_metadata(metadata: dict[str, object]) -> tuple[int | None, int | None]:
    """메시지 metadata에서 검증된 과실비율을 읽습니다."""
    fault_ratio_a = metadata.get("fault_ratio_a")
    fault_ratio_b = metadata.get("fault_ratio_b")
    if not isinstance(fault_ratio_a, int) or not isinstance(fault_ratio_b, int):
        return None, None
    if not 0 <= fault_ratio_a <= 100 or not 0 <= fault_ratio_b <= 100:
        return None, None
    if fault_ratio_a + fault_ratio_b != 100:
        return None, None
    return fault_ratio_a, fault_ratio_b


def read_retrieved_context_metadata(metadata: dict[str, object]) -> list[dict[str, object]]:
    """메시지 metadata에서 저장된 검색 문서 조각을 읽습니다."""
    contexts = metadata.get("retrieved_contexts")
    if not isinstance(contexts, list):
        return []
    return [context for context in contexts if isinstance(context, dict)]


def strip_markdown_images(markdown_text: str) -> str:
    return MARKDOWN_IMAGE_RE.sub("", markdown_text)


def resolve_image_path(image_path: str, source: object = None) -> Path | None:
    """metadata의 이미지 경로를 실제 로컬 파일 경로로 해석합니다."""
    raw_path = Path(image_path.replace("\\", "/"))
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(PROJECT_ROOT / raw_path)

        parts = raw_path.parts
        if "data" in parts:
            data_index = parts.index("data")
            candidates.append(PROJECT_ROOT / Path(*parts[data_index:]))
        if "upstage_output" in parts:
            upstage_index = parts.index("upstage_output")
            candidates.append(PROJECT_ROOT / "data" / Path(*parts[upstage_index:]))

    if isinstance(source, str) and source:
        source_path = Path(source.replace("\\", "/"))
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        candidates.append(source_path.parent / raw_path)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def latest_assistant_metadata(messages) -> dict[str, object]:
    for message in reversed(messages):
        if message.role == "assistant":
            return message.metadata
    return {}


def add_heading_markers(markdown_text: str) -> str:
    """Markdown 제목 앞에 화면 표시용 사각형 표식을 붙입니다."""
    marked_lines: list[str] = []
    for line in markdown_text.splitlines():
        match = MARKDOWN_HEADING_RE.match(line)
        if match:
            marked_lines.append(f"{match.group(1)} ■ {match.group(2)}")
        else:
            marked_lines.append(line)
    return "\n".join(marked_lines)


def render_app_css() -> None:
    st.markdown(
        """
<style>
:root {
  --mdm-accent: #176b87;
  --mdm-fault-a: #dc2626;
  --mdm-fault-b: #9ca3af;
  --mdm-neutral: #cfd5dc;
}
.sidebar-title {
  margin: 1rem 0 0.8rem;
  color: #dc2626;
  font-size: 1.35rem;
  font-weight: 900;
}
.mode-title {
  margin: 1.35rem 0 0.55rem;
  color: #dc2626;
  font-size: 1.25rem;
  font-weight: 900;
  letter-spacing: 0;
}
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
  border: 1px solid #dc2626 !important;
  background: transparent !important;
  color: inherit !important;
  box-shadow: none !important;
}
[data-testid="stSidebar"] button[kind="primary"] p,
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] p {
  font-weight: 900 !important;
}
.top-title {
  padding: 1.3rem 0 0.75rem;
  margin-bottom: 1rem;
}
.top-title h1 {
  margin: 0;
  color: #dc2626;
  font-size: 3.1rem;
  line-height: 1.24;
  font-weight: 900;
  letter-spacing: 0;
}
.top-title p {
  margin: 0.35rem 0 0;
  color: inherit;
  font-size: 2.05rem;
  line-height: 1.26;
  font-weight: 850;
}
.fault-card {
  margin: 0.25rem 0 1.15rem;
}
.fault-caption {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.4rem;
  font-size: 0.86rem;
  font-weight: 850;
}
.fault-track {
  height: 12px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--mdm-neutral);
}
.fault-fill {
  display: flex;
  width: 100%;
  height: 100%;
}
.fault-a {
  background: var(--mdm-fault-a);
}
.fault-b {
  background: var(--mdm-fault-b);
}
.fault-placeholder {
  width: 100%;
  background: var(--mdm-neutral);
}
.answer-panel {
  padding: 0.15rem 0 1rem;
  font-size: 1rem;
  line-height: 1.78;
}
.answer-panel h1,
.answer-panel h2,
.answer-panel h3,
.answer-panel h4,
.answer-panel h5,
.answer-panel h6 {
  margin-top: 1.1rem;
  margin-bottom: 0.4rem;
}
.empty-answer {
  color: inherit;
  opacity: 0.54;
  margin-top: -0.25rem;
  margin-bottom: 1.35rem;
  padding-top: 0;
  font-weight: 750;
}
.progress-line {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  margin: 0.25rem 0 1rem;
  color: inherit;
  font-size: 0.98rem;
  font-weight: 850;
  opacity: 0.78;
}
.progress-spinner {
  width: 0.9rem;
  height: 0.9rem;
  flex: 0 0 auto;
  border: 2px solid rgba(220, 38, 38, 0.22);
  border-top-color: #dc2626;
  border-radius: 999px;
  animation: mdmSpin 720ms linear infinite;
}
.donut-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(74px, 1fr));
  gap: 1.05rem 0.9rem;
  margin: 0.4rem 0 1.2rem;
}
.donut-item {
  display: grid;
  justify-items: center;
  gap: 0.45rem;
}
.donut {
  --percent: 0;
  width: 76px;
  height: 76px;
  border-radius: 50%;
  background: conic-gradient(var(--mdm-accent) calc(var(--percent) * 1%), var(--mdm-neutral) 0);
  display: grid;
  place-items: center;
  margin: 0 auto;
}
.donut::before {
  content: attr(data-label);
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: Canvas;
  display: grid;
  place-items: center;
  color: CanvasText;
  font-size: 0.92rem;
  font-weight: 900;
}
.rank-label {
  color: inherit;
  font-size: 0.88rem;
  font-weight: 900;
  opacity: 0.82;
}
.right-title {
  margin: 0.25rem 0 0.65rem;
  color: inherit;
  font-size: 1.25rem;
  font-weight: 900;
}
.reference-image {
  display: block;
  margin: 0 0 0.35rem;
  animation: fadeSlideIn 420ms ease both;
  cursor: zoom-in;
}
.reference-rank {
  margin: 0 0 1rem;
  color: inherit;
  font-size: 0.9rem;
  font-weight: 900;
  opacity: 0.82;
  text-align: center;
}
.reference-image img {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 6px;
}
.reference-lightbox-toggle {
  position: absolute;
  opacity: 0;
  pointer-events: none;
}
.reference-lightbox {
  position: fixed;
  inset: 0;
  z-index: 999999;
  display: none;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  background: rgba(0, 0, 0, 0.78);
  cursor: zoom-out;
}
.reference-lightbox-toggle:checked + .reference-image + .reference-rank + .reference-lightbox {
  display: flex;
}
.reference-lightbox img {
  display: block;
  width: auto;
  max-width: min(94vw, 1200px);
  max-height: 90vh;
  height: auto;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 24px 70px rgba(0, 0, 0, 0.38);
}
.reference-lightbox-close {
  position: absolute;
  top: 1rem;
  right: 1.2rem;
  color: #ffffff;
  font-size: 2rem;
  line-height: 1;
  font-weight: 900;
}
.question-history {
  margin: 0.2rem 0 1.2rem;
}
.question-history-title {
  margin-bottom: 0.45rem;
  font-size: 1rem;
  font-weight: 900;
}
.question-item {
  margin: 0 0 0.45rem;
  padding-left: 0.75rem;
  border-left: 3px solid #dc2626;
  font-size: 0.95rem;
  line-height: 1.55;
}
.previous-answers-title {
  margin: 0.35rem 0 0.45rem;
  color: inherit;
  opacity: 0.72;
  font-size: 0.95rem;
  font-weight: 850;
}
@keyframes fadeSlideIn {
  from {
    opacity: 0;
    transform: translateY(12px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
@keyframes mdmSpin {
  to {
    transform: rotate(360deg);
  }
}
[data-testid="stImage"] {
  animation: fadeSlideIn 420ms ease both;
}
[data-testid="stChatInput"] div[data-baseweb="textarea"] {
  border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
  border-radius: 14px !important;
  padding-left: 1rem !important;
  padding-right: 1rem !important;
}
[data-testid="stSpinner"] > div {
  margin-top: 0.35rem;
  color: inherit;
  font-weight: 850;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
<div class="top-title">
  <h1>몇 대 몇? 예상과실 비율 RAG 시스템</h1>
  <p>사고 사례를 바탕으로 과실 비율 해석해주는 도우미</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_fault_ratio(fault_ratio_a: int | None, fault_ratio_b: int | None) -> None:
    if fault_ratio_a is None or fault_ratio_b is None:
        st.markdown(
            """
<div class="fault-card">
  <div class="fault-caption">
    <span>A 측 -</span>
    <span>B 측 -</span>
  </div>
  <div class="fault-track">
    <div class="fault-fill">
      <div class="fault-placeholder"></div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
<div class="fault-card">
  <div class="fault-caption">
    <span>A 측 {fault_ratio_a}%</span>
    <span>B 측 {fault_ratio_b}%</span>
  </div>
  <div class="fault-track">
    <div class="fault-fill">
      <div class="fault-a" style="width: {fault_ratio_a}%;"></div>
      <div class="fault-b" style="width: {fault_ratio_b}%;"></div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_question_history(messages) -> None:
    questions = [message.content for message in messages if message.role == "user"]
    if not questions:
        return

    st.markdown('<div class="question-history">', unsafe_allow_html=True)
    st.markdown('<h4 class="question-history-title">■ 질문 내역</h4>', unsafe_allow_html=True)
    for index, question in enumerate(questions, start=1):
        st.markdown(
            f'<div class="question-item">Q{index}. {escape(question)}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_previous_answers(messages) -> None:
    turns: list[tuple[int, str, object]] = []
    question_index = 0
    latest_question = ""
    for message in messages:
        if message.role == "user":
            question_index += 1
            latest_question = message.content
        elif message.role == "assistant":
            turns.append((question_index, latest_question, message))

    previous_turns = turns[:-1]
    if not previous_turns:
        return

    st.markdown('<h5 class="previous-answers-title">■ 이전 답변</h5>', unsafe_allow_html=True)
    for index, (turn_question_index, question, message) in enumerate(
        previous_turns,
        start=1,
    ):
        question_preview = " ".join(question.split())
        if len(question_preview) > 34:
            question_preview = f"{question_preview[:34]}..."
        expander_label = f"Q{turn_question_index or index}. {question_preview or '이전 답변'}"
        with st.expander(expander_label, expanded=False):
            message_metadata = getattr(message, "metadata", {}) or {}
            fault_ratio_a, fault_ratio_b = read_fault_ratio_metadata(message_metadata)
            if fault_ratio_a is not None and fault_ratio_b is not None:
                render_fault_ratio(fault_ratio_a, fault_ratio_b)
            st.markdown(add_heading_markers(strip_markdown_images(message.content)))


def render_answer_area(
    metadata: dict[str, object],
    answer: str | None,
    messages=None,
    *,
    progress_reporter_factory: Callable[[], ProgressReporter] | None = None,
) -> ProgressReporter | None:
    render_fault_ratio(*read_fault_ratio_metadata(metadata))
    messages = messages or []
    render_question_history(messages)
    progress_reporter = (
        progress_reporter_factory() if progress_reporter_factory is not None else None
    )
    render_previous_answers(messages)
    if answer:
        st.markdown('<div class="answer-panel">', unsafe_allow_html=True)
        st.markdown(add_heading_markers(strip_markdown_images(answer)))
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="empty-answer">■ 사고 상황을 입력하면 과실비율 분석 결과가 여기에 표시됩니다.</div>',
            unsafe_allow_html=True,
        )
    return progress_reporter


def context_match_percent(context: dict[str, object]) -> int:
    value = context.get("match_percent", 0)
    if not isinstance(value, int):
        return 0
    return max(0, min(MAX_DONUT_PERCENT, value))


def context_rerank_score(context: dict[str, object]) -> float | None:
    value = context.get("rerank_score")
    if isinstance(value, int | float):
        return float(value)
    return None


def normalize_reference_scores(
    reference_items: list[tuple[str, dict[str, object], Path]],
) -> None:
    scores = [
        score
        for _, context, _ in reference_items
        if (score := context_rerank_score(context)) is not None
    ]
    if not scores:
        return

    max_score = max(scores)
    if max_score <= 0:
        return

    for _, context, _ in reference_items:
        score = context_rerank_score(context)
        if score is None:
            continue
        context["match_percent"] = max(
            0,
            min(MAX_DONUT_PERCENT, round((score / max_score) * MAX_DONUT_PERCENT)),
        )


def top_ranked_contexts(contexts: list[dict[str, object]], limit: int = 3) -> list[dict[str, object]]:
    return sorted(
        contexts,
        key=context_match_percent,
        reverse=True,
    )[:limit]


def ranked_reference_items(
    contexts: list[dict[str, object]],
    limit: int = 3,
) -> list[tuple[str, dict[str, object], Path]]:
    rank_labels = ("1st", "2nd", "3rd")
    items: list[tuple[str, dict[str, object], Path]] = []
    seen_images: set[str] = set()
    for context in top_ranked_contexts(contexts, limit=limit):
        for image_path in get_context_image_paths(context):
            resolved_image_path = resolve_image_path(image_path, context.get("source"))
            if resolved_image_path is None:
                continue
            resolved_image_key = str(resolved_image_path)
            if resolved_image_key in seen_images:
                continue
            seen_images.add(resolved_image_key)
            rank_label = rank_labels[len(items)]
            items.append((rank_label, context, resolved_image_path))
            break
        if len(items) >= limit:
            break
    return items


def render_donut_panel(reference_items: list[tuple[str, dict[str, object], Path]]) -> None:
    if not reference_items:
        return

    st.markdown('<div class="right-title">● Reference Match</div>', unsafe_allow_html=True)
    donut_html = ['<div class="donut-grid">']
    for rank_label, context, _ in reference_items:
        safe_percent = context_match_percent(context)
        donut_html.append(
            f"""
<div class="donut-item">
  <div class="donut" style="--percent: {safe_percent};" data-label="{safe_percent}%"></div>
  <div class="rank-label">{rank_label}</div>
</div>
"""
        )
    donut_html.append("</div>")
    st.markdown("".join(donut_html), unsafe_allow_html=True)


def get_context_image_paths(context: dict[str, object]) -> list[str]:
    image_path = context.get("image_path")
    if isinstance(image_path, str) and image_path:
        return [image_path]
    return []


@st.cache_data(show_spinner=False)
def read_image_as_png_bytes(image_path: Path) -> bytes | None:
    try:
        with Image.open(image_path) as image:
            rendered_image = ImageOps.exif_transpose(image)
            if rendered_image.mode not in {"RGB", "RGBA"}:
                rendered_image = rendered_image.convert("RGB")
            output = BytesIO()
            rendered_image.save(output, format="PNG")
            return output.getvalue()
    except (OSError, UnidentifiedImageError):
        return None


def render_reference_images(reference_items: list[tuple[str, dict[str, object], Path]]) -> None:
    if not reference_items:
        return

    st.markdown('<div class="right-title">● Reference Images</div>', unsafe_allow_html=True)
    for index, (rank_label, _, image_path) in enumerate(reference_items, start=1):
        image_bytes = read_image_as_png_bytes(image_path)
        if image_bytes is None:
            continue
        image_data = base64.b64encode(image_bytes).decode("ascii")
        lightbox_id = f"reference-lightbox-{index}"
        st.markdown(
            f"""
<input id="{lightbox_id}" class="reference-lightbox-toggle" type="checkbox">
<label class="reference-image" for="{lightbox_id}">
  <img src="data:image/png;base64,{image_data}" alt="Reference image">
</label>
<div class="reference-rank">{rank_label}</div>
<label class="reference-lightbox" for="{lightbox_id}">
  <span class="reference-lightbox-close">×</span>
  <img src="data:image/png;base64,{image_data}" alt="Expanded reference image">
</label>
""",
            unsafe_allow_html=True,
        )


def render_right_panel(metadata: dict[str, object]) -> None:
    contexts = read_retrieved_context_metadata(metadata)
    reference_items = ranked_reference_items(contexts, limit=3)
    if metadata.get("reranker_strategy") == "cross-encoder":
        normalize_reference_scores(reference_items)
    render_donut_panel(reference_items)
    render_reference_images(reference_items)


def render_chat(
    store: ConversationStore,
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    chunker_strategy: str = DEFAULT_CHUNKER_STRATEGY,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    retriever_strategy: str = DEFAULT_RETRIEVER_STRATEGY,
    ensemble_bm25_weight: float = DEFAULT_ENSEMBLE_BM25_WEIGHT,
    ensemble_candidate_k: int = DEFAULT_ENSEMBLE_CANDIDATE_K,
    ensemble_use_chunk_id: bool = DEFAULT_ENSEMBLE_USE_CHUNK_ID,
    reranker_strategy: str = DEFAULT_RERANKER_STRATEGY,
) -> None:
    active_session = st.session_state.active_session
    messages = store.get_messages(USER_ID, active_session)
    assistant_metadata = latest_assistant_metadata(messages)
    last_answer = next(
        (message.content for message in reversed(messages) if message.role == "assistant"),
        None,
    )
    pending_request = st.session_state.get("pending_request")
    has_pending_request = (
        isinstance(pending_request, dict)
        and pending_request.get("session_id") == active_session
    )
    progress_reporter_factory = (
        (lambda: create_progress_reporter(get_debug_progress_enabled()))
        if has_pending_request
        else None
    )
    progress_reporter = None

    left, right = st.columns([0.62, 0.38], gap="large")
    with left:
        progress_reporter = render_answer_area(
            assistant_metadata,
            last_answer,
            messages,
            progress_reporter_factory=progress_reporter_factory,
        )
    with right:
        render_right_panel(assistant_metadata)

    question = st.chat_input("사고 내용을 입력하세요")
    if question:
        store.append_message(USER_ID, active_session, "user", question)
        st.session_state.pending_request = {
            "session_id": active_session,
            "question": question,
            "loader_strategy": loader_strategy,
            "chunker_strategy": chunker_strategy,
            "embedding_provider": embedding_provider,
            "retriever_strategy": retriever_strategy,
            "ensemble_bm25_weight": ensemble_bm25_weight,
            "ensemble_candidate_k": ensemble_candidate_k,
            "ensemble_use_chunk_id": ensemble_use_chunk_id,
            "reranker_strategy": reranker_strategy,
        }
        st.rerun()

    if not isinstance(pending_request, dict):
        return
    if pending_request.get("session_id") != active_session:
        return

    pending_question = str(pending_request.get("question", ""))
    request_loader_strategy = str(pending_request.get("loader_strategy", loader_strategy))
    request_chunker_strategy = str(pending_request.get("chunker_strategy", chunker_strategy))
    request_embedding_provider = str(
        pending_request.get("embedding_provider", embedding_provider)
    )
    request_retriever_strategy = str(
        pending_request.get("retriever_strategy", retriever_strategy)
    )
    request_ensemble_bm25_weight = float(
        pending_request.get("ensemble_bm25_weight", ensemble_bm25_weight)
    )
    request_ensemble_candidate_k = int(
        pending_request.get("ensemble_candidate_k", ensemble_candidate_k)
    )
    request_ensemble_use_chunk_id = bool(
        pending_request.get("ensemble_use_chunk_id", ensemble_use_chunk_id)
    )
    request_reranker_strategy = str(
        pending_request.get("reranker_strategy", reranker_strategy)
    )
    chat_history = list(messages)
    if (
        chat_history
        and chat_history[-1].role == "user"
        and chat_history[-1].content == pending_question
    ):
        chat_history = chat_history[:-1]

    trace_context = TraceContext(
        thread_id=active_session,
        session_id=active_session,
        user_id=USER_ID,
        tags=(
            "streamlit",
            "mdm",
            request_loader_strategy,
            request_chunker_strategy,
            request_embedding_provider,
            request_retriever_strategy,
            request_reranker_strategy,
        ),
        metadata={
            "loader_strategy": request_loader_strategy,
            "chunker_strategy": request_chunker_strategy,
            "embedding_provider": request_embedding_provider,
            "retriever_strategy": request_retriever_strategy,
            "ensemble_bm25_weight": request_ensemble_bm25_weight,
            "ensemble_candidate_k": request_ensemble_candidate_k,
            "ensemble_use_chunk_id": request_ensemble_use_chunk_id,
            "reranker_strategy": request_reranker_strategy,
        },
    )

    if progress_reporter is None:
        progress_reporter = create_progress_reporter(get_debug_progress_enabled())
    try:
        result = answer_question_with_intake(
            pending_question,
            intake_state=store.get_intake_state(
                USER_ID,
                active_session,
            ),
            loader_strategy=request_loader_strategy,
            chunker_strategy=request_chunker_strategy,
            chat_history=chat_history,
            embedding_provider=request_embedding_provider,
            pipeline_config=build_pipeline_config(
                request_retriever_strategy,
                request_ensemble_bm25_weight,
                request_ensemble_candidate_k,
                request_ensemble_use_chunk_id,
                request_reranker_strategy,
            ),
            trace_context=trace_context,
            progress_callback=progress_reporter,
        )
        answer = result.answer
        progress_reporter.update(PROGRESS_RESULT)
        assistant_metadata = build_assistant_metadata(
            result,
            pending_question,
            request_reranker_strategy,
        )
        retrieved_contexts = assistant_metadata.get("retrieved_contexts")
        if isinstance(retrieved_contexts, list):
            progress_reporter.detail(f"참고 근거: {len(retrieved_contexts)}개")
        store.set_intake_state(USER_ID, active_session, result.intake_state)
        progress_reporter.complete()
    except Exception:
        logger.exception("Failed to answer question")
        progress_reporter.error()
        answer = "오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
        assistant_metadata = {}

    store.append_message(
        USER_ID,
        active_session,
        "assistant",
        answer,
        metadata=assistant_metadata,
    )
    st.session_state.pop("pending_request", None)
    st.rerun()


def main():
    load_dotenv()
    st.set_page_config(
        page_title="MDM",
        page_icon="🚗",
        layout="wide",
    )
    render_app_css()
    render_header()
    store = get_conversation_store()
    init_state(store)
    (
        loader_strategy,
        chunker_strategy,
        embedding_provider,
        retriever_strategy,
        ensemble_bm25_weight,
        ensemble_candidate_k,
        ensemble_use_chunk_id,
        reranker_strategy,
    ) = render_sidebar(store)
    render_chat(
        store,
        loader_strategy,
        chunker_strategy,
        embedding_provider,
        retriever_strategy,
        ensemble_bm25_weight,
        ensemble_candidate_k,
        ensemble_use_chunk_id,
        reranker_strategy,
    )


if __name__ == "__main__":
    main()
