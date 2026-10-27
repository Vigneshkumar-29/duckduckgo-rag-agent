# main.py

import streamlit as st

from rag import create_rag
from agent import create_agent, run_agent

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="centered"
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 7rem;
        }
        [data-testid="stChatInput"] {
            padding-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! Ask me anything, or upload a TXT document with the "
                "**+** button. I use DuckDuckGo when a document is not "
                "available or does not have the answer."
            )
        }
    ]

if "agent" not in st.session_state:
    st.session_state.agent = create_agent(None)

st.title("🤖 AI Document Assistant")
st.caption("Ask about your document or let me search the web.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Ask a question or attach a TXT document...",
    accept_file=True,
    file_type=["txt"]
)

if prompt:
    uploaded_file = prompt.files[0] if prompt.files else None
    question = prompt.text.strip()

    if uploaded_file:
        upload_message = f"📄 Uploaded **{uploaded_file.name}**"

        with st.chat_message("user"):
            st.markdown(upload_message)

        try:
            with st.spinner("Reading and indexing your document..."):
                data = uploaded_file.getvalue().decode("utf-8")
                vectordb = create_rag(data)
                st.session_state.agent = create_agent(vectordb)

            response = (
                f"Your document **{uploaded_file.name}** is ready. "
                "What would you like to know?"
            )

        except Exception as e:
            response = f"I could not process that document: {e}"

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append(
            {"role": "user", "content": upload_message}
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    response = run_agent(
                        st.session_state.agent,
                        question
                    )

                st.markdown(response)

            except Exception as e:
                response = f"I could not answer that question: {e}"
                st.error(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
