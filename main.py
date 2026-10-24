# main.py

import streamlit as st

from rag import create_rag
from agent import create_agent, run_agent


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Agent RAG",
    page_icon="🤖",
    layout="centered"
)


# ==========================================
# TITLE
# ==========================================

st.title(
    "🤖 AI Agent RAG System"
)

st.write(
    "Upload a document and ask questions "
    "using RAG or web search."
)


# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["txt"]
)


# ==========================================
# PROCESS DOCUMENT
# ==========================================

if uploaded_file is not None:

    st.success(
        "Document uploaded successfully! ✅"
    )


    # --------------------------------------
    # Process button
    # --------------------------------------

    if st.button(
        "Process Document",
        type="primary"
    ):

        try:

            with st.spinner(
                "Processing document..."
            ):

                # Read uploaded file
                data = uploaded_file.read().decode(
                    "utf-8"
                )


                # Create RAG
                vectordb = create_rag(
                    data
                )


                # Create Agent
                agent = create_agent(
                    vectordb
                )


                # Store agent
                st.session_state[
                    "agent"
                ] = agent


            st.success(
                "Document processed successfully! 🚀"
            )


        except Exception as e:

            st.error(
                f"Error while processing document: {e}"
            )


# ==========================================
# ASK QUESTION
# ==========================================

if "agent" in st.session_state:

    st.divider()

    st.subheader(
        "Ask a Question"
    )


    # --------------------------------------
    # Question input
    # --------------------------------------

    query = st.text_input(
        "Enter your question:"
    )


    # --------------------------------------
    # Ask button
    # --------------------------------------

    if st.button(
        "Ask",
        type="primary"
    ):

        if not query.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                with st.spinner(
                    "Thinking..."
                ):

                    answer = run_agent(
                        st.session_state[
                            "agent"
                        ],
                        query
                    )


                # ----------------------------------
                # Display answer
                # ----------------------------------

                st.subheader(
                    "Answer"
                )

                st.write(
                    answer
                )


            except Exception as e:

                st.error(
                    f"Error while answering: {e}"
                )


# ==========================================
# INFORMATION
# ==========================================

else:

    st.info(
        "Upload a TXT document and click "
        "'Process Document' to start."
    )