"""기본 RAG 앱의 Streamlit 채팅 UI입니다."""

from dotenv import load_dotenv
from pathlib import Path
import streamlit as st

from config import (
    DEFAULT_CHUNKER_STRATEGY,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LOADER_STRATEGY,
)
from rag.embeddings import EMBEDDING_STRATEGIES
from rag.pipeline.retrieval import RetrievalPipelineConfig
from rag.pipeline.reranker import CrossEncoderRerankerConfig, FlashrankRerankerConfig
from rag.pipeline.retriever import EnsembleRetrieverConfig, RETRIEVAL_STRATEGIES
from rag.service.analysis.answer_schema import RetrievedContext
from rag.service.conversation.app_service import answer_question_with_intake
from rag.service.presentation.result_service import truncate_context
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
    strategy
    for strategy in RETRIEVAL_STRATEGIES
    if strategy != "vectorstore"
)
RERANKER_STRATEGY_OPTIONS = ("none", "cross-encoder", "flashrank")
USER_ID = "local"
DEFAULT_ENSEMBLE_BM25_WEIGHT = 0.5
DEFAULT_RERANKER_STRATEGY = "none"
DEFAULT_RERANKER_CANDIDATE_K = 10
DEFAULT_RERANKER_FINAL_K = 3
PROJECT_ROOT = Path(__file__).resolve().parent


def ensure_active_session(store: ConversationStore) -> str:
    """활성 세션 id를 반환하고, 없으면 새로 생성합니다."""
    sessions = store.list_sessions(USER_ID)
    if not sessions:
        session = store.create_session(USER_ID, title="세션 1")
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
        session = store.create_session(USER_ID, title="세션 1")
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
        st.session_state.retriever_strategy = "similarity"
    if "reranker_strategy" not in st.session_state:
        st.session_state.reranker_strategy = DEFAULT_RERANKER_STRATEGY
    if "ensemble_bm25_weight" not in st.session_state:
        st.session_state.ensemble_bm25_weight = DEFAULT_ENSEMBLE_BM25_WEIGHT
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


def render_sidebar(store: ConversationStore) -> tuple[str, str, str, str, float, str]:
    st.sidebar.title("세션 목록")
    if st.sidebar.button("새 세션", use_container_width=True):
        session_count = len(store.list_sessions(USER_ID))
        session = store.create_session(USER_ID, title=f"세션 {session_count + 1}")
        store.set_active_session(USER_ID, session.session_id)
        st.session_state.active_session = session.session_id

    st.sidebar.divider()

    for session in store.list_sessions(USER_ID):
        session_column, delete_column = st.sidebar.columns([0.82, 0.18])
        with session_column:
            if st.button(
                session.title,
                key=f"session-{session.session_id}",
                use_container_width=True,
            ):
                store.set_active_session(USER_ID, session.session_id)
                st.session_state.active_session = session.session_id
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

    st.sidebar.divider()
    current_loader_strategy = st.session_state.get("loader_strategy", DEFAULT_LOADER_STRATEGY)
    if current_loader_strategy not in LOADER_STRATEGY_OPTIONS:
        current_loader_strategy = DEFAULT_LOADER_STRATEGY
    selected_loader_strategy = st.sidebar.selectbox(
        "문서 파서",
        LOADER_STRATEGY_OPTIONS,
        index=LOADER_STRATEGY_OPTIONS.index(current_loader_strategy),
    )
    if selected_loader_strategy != current_loader_strategy:
        store.set_loader_strategy(USER_ID, selected_loader_strategy)
    st.session_state.loader_strategy = selected_loader_strategy

    chunker_options = get_chunker_strategy_options(selected_loader_strategy)
    current_chunker_strategy = normalize_chunker_strategy(
        st.session_state.get("chunker_strategy", DEFAULT_CHUNKER_STRATEGY),
        selected_loader_strategy,
    )
    st.session_state.chunker_strategy = st.sidebar.selectbox(
        "청커",
        chunker_options,
        index=chunker_options.index(current_chunker_strategy),
    )

    current_embedding_provider = st.session_state.get(
        "embedding_provider",
        DEFAULT_EMBEDDING_PROVIDER,
    )
    if current_embedding_provider not in EMBEDDING_PROVIDER_OPTIONS:
        current_embedding_provider = DEFAULT_EMBEDDING_PROVIDER
    st.session_state.embedding_provider = st.sidebar.selectbox(
        "임베딩 모델",
        EMBEDDING_PROVIDER_OPTIONS,
        index=EMBEDDING_PROVIDER_OPTIONS.index(current_embedding_provider),
    )

    current_retriever_strategy = st.session_state.get("retriever_strategy", "similarity")
    if current_retriever_strategy not in RETRIEVER_STRATEGY_OPTIONS:
        current_retriever_strategy = "similarity"
    st.session_state.retriever_strategy = st.sidebar.selectbox(
        "검색 전략",
        RETRIEVER_STRATEGY_OPTIONS,
        index=RETRIEVER_STRATEGY_OPTIONS.index(current_retriever_strategy),
    )

    current_reranker_strategy = st.session_state.get(
        "reranker_strategy",
        DEFAULT_RERANKER_STRATEGY,
    )
    if current_reranker_strategy not in RERANKER_STRATEGY_OPTIONS:
        current_reranker_strategy = DEFAULT_RERANKER_STRATEGY
    st.session_state.reranker_strategy = st.sidebar.selectbox(
        "리랭커",
        RERANKER_STRATEGY_OPTIONS,
        index=RERANKER_STRATEGY_OPTIONS.index(current_reranker_strategy),
    )

    current_ensemble_bm25_weight = float(
        st.session_state.get("ensemble_bm25_weight", DEFAULT_ENSEMBLE_BM25_WEIGHT)
    )
    if st.session_state.retriever_strategy == "ensemble":
        current_ensemble_bm25_percent = int(
            st.session_state.get(
                "ensemble_bm25_percent",
                round(current_ensemble_bm25_weight * 100),
            )
        )
        current_ensemble_bm25_weight = current_ensemble_bm25_percent / 100
        st.sidebar.markdown(
            build_ensemble_slider_css(current_ensemble_bm25_weight),
            unsafe_allow_html=True,
        )
        slider_kwargs = {
            "min_value": 0,
            "max_value": 100,
            "step": 5,
            "format": "%d%%",
            "key": "ensemble_bm25_percent",
        }
        if "ensemble_bm25_percent" not in st.session_state:
            slider_kwargs["value"] = current_ensemble_bm25_percent
        bm25_percent = st.sidebar.slider(
            build_ensemble_weight_label(current_ensemble_bm25_weight),
            **slider_kwargs,
        )
        st.session_state.ensemble_bm25_weight = bm25_percent / 100
        st.sidebar.markdown(
            build_ensemble_weight_caption_html(st.session_state.ensemble_bm25_weight),
            unsafe_allow_html=True,
        )
    return (
        selected_loader_strategy,
        st.session_state.chunker_strategy,
        st.session_state.embedding_provider,
        st.session_state.retriever_strategy,
        st.session_state.ensemble_bm25_weight,
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

    if retriever_strategy != "ensemble":
        return RetrievalPipelineConfig(
            retriever_strategy=retriever_strategy,
            **reranker_kwargs,
        )

    bm25_weight = round(ensemble_bm25_weight, 2)
    dense_weight = round(1.0 - bm25_weight, 2)
    return RetrievalPipelineConfig(
        retriever_strategy=retriever_strategy,
        retriever_config=EnsembleRetrieverConfig(weights=(bm25_weight, dense_weight)),
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


def build_retrieved_context_metadata(
    retrieved_contexts: list[RetrievedContext],
) -> list[dict[str, object]]:
    """새로고침 후 근거 조각과 이미지를 다시 보여줄 최소 metadata를 만듭니다."""
    rendered_contexts: list[dict[str, object]] = []
    for context in retrieved_contexts:
        rendered_context: dict[str, object] = {
            "content": truncate_context(context.content),
        }
        for key in ("image_path", "source", "page", "diagram_id"):
            value = context.metadata.get(key)
            if isinstance(value, str | int):
                rendered_context[key] = value
        rendered_contexts.append(rendered_context)
    return rendered_contexts


def build_assistant_metadata(result) -> dict[str, object]:
    """assistant 메시지 렌더링에 필요한 부가 정보를 저장합니다."""
    metadata = build_fault_ratio_metadata(result.fault_ratio_a, result.fault_ratio_b)
    retrieved_contexts = build_retrieved_context_metadata(result.retrieved_contexts)
    if retrieved_contexts:
        metadata["retrieved_contexts"] = retrieved_contexts
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


def render_retrieved_contexts(metadata: dict[str, object]) -> None:
    """저장된 검색 문서 조각과 연결 이미지를 표시합니다."""
    contexts = read_retrieved_context_metadata(metadata)
    if not contexts:
        return

    with st.expander("검색된 문서 조각"):
        for index, context in enumerate(contexts, start=1):
            content = context.get("content")
            if isinstance(content, str) and content:
                st.markdown(f"[{index}] {content}")

            image_path = context.get("image_path")
            if isinstance(image_path, str) and image_path:
                resolved_image_path = resolve_image_path(image_path, context.get("source"))
                if resolved_image_path is not None:
                    st.image(str(resolved_image_path), caption=f"[{index}] 참고 이미지")

            if index < len(contexts):
                st.markdown("---")


def render_fault_ratio(fault_ratio_a: int | None, fault_ratio_b: int | None) -> None:
    """과실비율을 채팅 답변 상단에 누적 막대로 표시합니다."""
    if fault_ratio_a is None or fault_ratio_b is None:
        return

    st.markdown(
        f"""
<div style="margin: 0 0 0.85rem 0;">
  <div style="display: flex; justify-content: space-between; gap: 0.75rem; margin-bottom: 0.35rem; font-size: 0.92rem; font-weight: 700;">
    <span>A 측 {fault_ratio_a}%</span>
    <span>B 측 {fault_ratio_b}%</span>
  </div>
  <div style="height: 14px; width: 100%; overflow: hidden; border-radius: 7px; background: #e5e7eb;">
    <div style="display: flex; height: 100%; width: 100%;">
      <div style="width: {fault_ratio_a}%; background: #2563eb;"></div>
      <div style="width: {fault_ratio_b}%; background: #f97316;"></div>
    </div>
  </div>
</div>
""".strip(),
        unsafe_allow_html=True,
    )


def render_chat(
    store: ConversationStore,
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    chunker_strategy: str = DEFAULT_CHUNKER_STRATEGY,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    retriever_strategy: str = "similarity",
    ensemble_bm25_weight: float = DEFAULT_ENSEMBLE_BM25_WEIGHT,
    reranker_strategy: str = DEFAULT_RERANKER_STRATEGY,
) -> None:
    active_session = st.session_state.active_session
    messages = store.get_messages(USER_ID, active_session)
    trace_context = TraceContext(
        thread_id=active_session,
        session_id=active_session,
        user_id=USER_ID,
        tags=(
            "streamlit",
            "mdm",
            loader_strategy,
            chunker_strategy,
            embedding_provider,
            retriever_strategy,
            reranker_strategy,
        ),
        metadata={
            "loader_strategy": loader_strategy,
            "chunker_strategy": chunker_strategy,
            "embedding_provider": embedding_provider,
            "retriever_strategy": retriever_strategy,
            "ensemble_bm25_weight": ensemble_bm25_weight,
            "reranker_strategy": reranker_strategy,
        },
    )

    st.title("MDM")
    st.markdown(
        '<p style="color: #555; font-size: 1.05rem;">몇 대 몇: 자동차 사고 과실비율 RAG 시스템</p>',
        unsafe_allow_html=True,
    )
    for message in messages:
        with st.chat_message(message.role):
            if message.role == "assistant":
                render_fault_ratio(*read_fault_ratio_metadata(message.metadata))
            st.markdown(message.content)
            if message.role == "assistant":
                render_retrieved_contexts(message.metadata)

    question = st.chat_input("사고 내용을 입력하세요")
    if not question:
        return

    store.append_message(USER_ID, active_session, "user", question)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        assistant_metadata: dict[str, object] = {}
        with st.spinner("검색하고 답변 중..."):
            try:
                result = answer_question_with_intake(
                    question,
                    intake_state=store.get_intake_state(
                        USER_ID,
                        active_session,
                    ),
                    loader_strategy=loader_strategy,
                    chunker_strategy=chunker_strategy,
                    chat_history=messages,
                    embedding_provider=embedding_provider,
                    pipeline_config=build_pipeline_config(
                        retriever_strategy,
                        ensemble_bm25_weight,
                        reranker_strategy,
                    ),
                    trace_context=trace_context,
                )
                answer = result.answer
                assistant_metadata = build_assistant_metadata(result)
                store.set_intake_state(USER_ID, active_session, result.intake_state)
                render_fault_ratio(result.fault_ratio_a, result.fault_ratio_b)
                st.markdown(answer)
                if SHOW_RETRIEVED_CONTEXTS:
                    render_retrieved_contexts(assistant_metadata)
            except Exception as exc:
                answer = f"오류가 발생했습니다: {exc}"
                st.markdown(answer)

    store.append_message(
        USER_ID,
        active_session,
        "assistant",
        answer,
        metadata=assistant_metadata,
    )


def main():
    load_dotenv()
    st.set_page_config(page_title="MDM Basic RAG")
    store = get_conversation_store()
    init_state(store)
    (
        loader_strategy,
        chunker_strategy,
        embedding_provider,
        retriever_strategy,
        ensemble_bm25_weight,
        reranker_strategy,
    ) = render_sidebar(store)
    render_chat(
        store,
        loader_strategy,
        chunker_strategy,
        embedding_provider,
        retriever_strategy,
        ensemble_bm25_weight,
        reranker_strategy,
    )


if __name__ == "__main__":
    main()
