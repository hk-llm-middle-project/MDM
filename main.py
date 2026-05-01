"""기본 RAG 앱의 Streamlit 채팅 UI입니다."""

from dotenv import load_dotenv
import streamlit as st

from config import (
    DEFAULT_CHUNKER_STRATEGY,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LOADER_STRATEGY,
)
from rag.embeddings import EMBEDDING_STRATEGIES
from rag.pipeline.retrieval import RetrievalPipelineConfig
from rag.pipeline.retriever import EnsembleRetrieverConfig, RETRIEVAL_STRATEGIES
from rag.service.conversation.app_service import answer_question_with_intake
from rag.service.presentation.result_service import format_context_preview
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
USER_ID = "local"
DEFAULT_ENSEMBLE_BM25_WEIGHT = 0.5


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


def render_sidebar(store: ConversationStore) -> tuple[str, str, str, str, float]:
    st.sidebar.title("세션 목록")
    if st.sidebar.button("새 세션", use_container_width=True):
        session_count = len(store.list_sessions(USER_ID))
        session = store.create_session(USER_ID, title=f"세션 {session_count + 1}")
        store.set_active_session(USER_ID, session.session_id)
        st.session_state.active_session = session.session_id

    st.sidebar.divider()

    for session in store.list_sessions(USER_ID):
        if st.sidebar.button(
            session.title,
            key=f"session-{session.session_id}",
            use_container_width=True,
        ):
            store.set_active_session(USER_ID, session.session_id)
            st.session_state.active_session = session.session_id

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
) -> RetrievalPipelineConfig:
    """Streamlit 선택값을 retrieval pipeline 설정으로 변환합니다."""
    if retriever_strategy != "ensemble":
        return RetrievalPipelineConfig(retriever_strategy=retriever_strategy)

    bm25_weight = round(ensemble_bm25_weight, 2)
    dense_weight = round(1.0 - bm25_weight, 2)
    return RetrievalPipelineConfig(
        retriever_strategy=retriever_strategy,
        retriever_config=EnsembleRetrieverConfig(weights=(bm25_weight, dense_weight)),
    )


def render_chat(
    store: ConversationStore,
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
    chunker_strategy: str = DEFAULT_CHUNKER_STRATEGY,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    retriever_strategy: str = "similarity",
    ensemble_bm25_weight: float = DEFAULT_ENSEMBLE_BM25_WEIGHT,
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
        ),
        metadata={
            "loader_strategy": loader_strategy,
            "chunker_strategy": chunker_strategy,
            "embedding_provider": embedding_provider,
            "retriever_strategy": retriever_strategy,
            "ensemble_bm25_weight": ensemble_bm25_weight,
        },
    )

    st.title("MDM")
    st.markdown(
        '<p style="color: #555; font-size: 1.05rem;">몇 대 몇: 자동차 사고 과실비율 RAG 시스템</p>',
        unsafe_allow_html=True,
    )
    for message in messages:
        with st.chat_message(message.role):
            st.markdown(message.content)

    question = st.chat_input("사고 내용을 입력하세요")
    if not question:
        return

    store.append_message(USER_ID, active_session, "user", question)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
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
                    ),
                    trace_context=trace_context,
                )
                answer = result.answer
                contexts = result.contexts
                store.set_intake_state(USER_ID, active_session, result.intake_state)
                st.markdown(answer)
                context_preview = format_context_preview(contexts)
                if SHOW_RETRIEVED_CONTEXTS and context_preview:
                    with st.expander("검색된 문서 조각"):
                        st.markdown(context_preview)
            except Exception as exc:
                answer = f"오류가 발생했습니다: {exc}"
                st.markdown(answer)

    store.append_message(USER_ID, active_session, "assistant", answer)


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
    ) = render_sidebar(store)
    render_chat(
        store,
        loader_strategy,
        chunker_strategy,
        embedding_provider,
        retriever_strategy,
        ensemble_bm25_weight,
    )


if __name__ == "__main__":
    main()
