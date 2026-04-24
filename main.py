"""Streamlit chat UI for the basic RAG app."""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st

from config import (
    LLM_MODEL,
    PDF_PATH,
    VECTORSTORE_DIR,
)
from rag.chunker import split_documents
from rag.indexer import build_vectorstore, load_vectorstore, vectorstore_exists
from rag.loader import load_pdf
from rag.retriever_pipeline import RetrievalPipelineConfig, run_retrieval_pipeline


def get_vectorstore():
    """Load an existing vector store or build one from the PDF."""
    if vectorstore_exists(VECTORSTORE_DIR):
        return load_vectorstore(VECTORSTORE_DIR)

    documents = load_pdf(PDF_PATH)
    chunks = split_documents(documents)
    return build_vectorstore(chunks, VECTORSTORE_DIR)


def build_prompt(question: str, context: str) -> str:
    return f"""
아래 공식 문서 내용을 참고해서 답변해.

[공식 문서 내용]
{context}

[사용자 질문]
{question}

아래 형식으로 답변해.
1. 의심 사고유형
2. 관련 공식 문서 근거
3. 수정요소 후보
4. 예상 과실비율
5. 설명
""".strip()


def answer_question(
    question: str,
    pipeline_config: RetrievalPipelineConfig | None = None,
) -> tuple[str, list[str]]:
    vectorstore = get_vectorstore()
    documents = run_retrieval_pipeline(vectorstore, question, pipeline_config=pipeline_config)
    contexts = [document.page_content for document in documents]
    print(f"[retrieved] question={question}")
    for index, context in enumerate(contexts, start=1):
        print(f"[retrieved:{index}] {context}")
    prompt = build_prompt(question, "\n\n".join(contexts))
    llm = ChatOpenAI(model=LLM_MODEL)
    answer = llm.invoke(prompt).content
    return answer, contexts


def init_state() -> None:
    if "sessions" not in st.session_state:
        st.session_state.sessions = {"새션 1": []}
    if "active_session" not in st.session_state:
        st.session_state.active_session = "새션 1"


def render_sidebar() -> None:
    st.sidebar.title("새션 목록")
    if st.sidebar.button("새 새션", use_container_width=True):
        name = f"새션 {len(st.session_state.sessions) + 1}"
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
                if contexts:
                    answer = f"{answer}\n\n---\n\n검색된 문서 조각\n\n" + "\n\n---\n\n".join(contexts)
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
