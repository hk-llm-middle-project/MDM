"""기본 RAG 앱의 Streamlit 채팅 UI입니다."""

from dotenv import load_dotenv
import streamlit as st

from config import DEFAULT_LOADER_STRATEGY
from rag.service.conversation.app_service import answer_question_with_intake
from rag.service.presentation.result_service import format_context_preview
from rag.service.session import ConversationStore, get_conversation_store


SHOW_RETRIEVED_CONTEXTS = True
LOADER_STRATEGY_OPTIONS = ("pdfplumber", "llamaparser")
USER_ID = "local"


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
    st.session_state.active_session = ensure_active_session(store)


def render_sidebar(store: ConversationStore) -> str:
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
    return selected_loader_strategy


def render_chat(
    store: ConversationStore,
    loader_strategy: str = DEFAULT_LOADER_STRATEGY,
) -> None:
    active_session = st.session_state.active_session
    messages = store.get_messages(USER_ID, active_session)

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
                    chat_history=messages,
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
    loader_strategy = render_sidebar(store)
    render_chat(store, loader_strategy)


if __name__ == "__main__":
    main()
