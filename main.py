# main.py

import streamlit as st

from rag import create_rag
from agent import create_agent, run_agent
from voice_to_text import is_api_key_configured, transcribe_audio

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
                "Hi! Ask me anything by typing or using the voice button, "
                "or upload a TXT document with the **+** button. I use DuckDuckGo "
                "when a document is not available or does not have the answer."
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
    "Ask a question, record voice, or attach a TXT document...",
    accept_file=True,
    file_type=["txt"],
    accept_audio=True
)

if prompt:
    uploaded_file = prompt.files[0] if prompt.files else None
    audio_recording = getattr(prompt, "audio", None)
    question = prompt.text.strip() if prompt.text else ""

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

    # Transcribe voice input from the query box using AssemblyAI
    if audio_recording:
        if not is_api_key_configured():
            st.error(
                "AssemblyAI API key not found. "
                "Please add `ASSEMBLYAI_API_KEY=your_key` to your `.env` file."
            )
        else:
            with st.spinner("Transcribing your voice with AssemblyAI..."):
                try:
                    transcribed_text = transcribe_audio(audio_recording)
                    if transcribed_text and transcribed_text.strip():
                        question = transcribed_text.strip()
                    else:
                        st.warning("No speech detected in the audio recording.")
                except Exception as e:
                    st.error(f"Voice transcription failed: {e}")

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
