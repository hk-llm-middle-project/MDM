"""Streamlit chat UI for the basic RAG app."""

from dotenv import load_dotenv
import streamlit as st

from rag.service.app_service import answer_question
from rag.service.result_service import format_context_preview


SHOW_RETRIEVED_CONTEXTS = True


def init_state() -> None:
    if "sessions" not in st.session_state:
        st.session_state.sessions = {"세션 1": []}
    if "active_session" not in st.session_state:
        st.session_state.active_session = "세션 1"


def render_sidebar() -> None:
    st.sidebar.title("세션 목록")
    if st.sidebar.button("새 세션", use_container_width=True):
        name = f"세션 {len(st.session_state.sessions) + 1}"
        st.session_state.sessions[name] = []
        st.session_state.active_session = name

    st.sidebar.divider()

    for name in st.session_state.sessions:
        if st.sidebar.button(name, key=f"session-{name}", use_container_width=True):
            st.session_state.active_session = name


def render_chat() -> None:
    messages = st.session_state.sessions[st.session_state.active_session]

    st.title("MDM")
    st.markdown(
        '<p style="color: #555; font-size: 1.05rem;">몇 대 몇: 자동차 사고 과실비율 RAG 시스템</p>',
        unsafe_allow_html=True,
    )
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("사고 내용을 입력하세요")
    if not question:
        return

    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("검색하고 답변 중..."):
            try:
                answer, contexts = answer_question(question)
                st.markdown(answer)
                context_preview = format_context_preview(contexts)
                if SHOW_RETRIEVED_CONTEXTS and context_preview:
                    with st.expander("검색된 문서 조각"):
                        st.markdown(context_preview)
            except Exception as exc:
                answer = f"오류가 발생했습니다: {exc}"
                st.markdown(answer)

    messages.append({"role": "assistant", "content": answer})


def main():
    load_dotenv()
    st.set_page_config(page_title="MDM Basic RAG")
    init_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
