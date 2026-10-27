# agent.py

from typing import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from langgraph.graph import (
    StateGraph,
    START,
    END
)


# ==========================================
# LLM
# ==========================================

llm = ChatOllama(
    model="llama3.1",
    temperature=0
)


# ==========================================
# AGENT STATE
# ==========================================

class RAGAgentState(TypedDict):

    question: str

    decision: str

    retrieval_text: str

    search_result: str

    answer: str


# ==========================================
# WEB SEARCH TOOL
# ==========================================

@tool
def duckduckgo_search(query: str) -> str:
    """
    Search the internet using DuckDuckGo.
    """

    try:

        from ddgs import DDGS

        results = DDGS().text(
            query,
            max_results=5
        )

        search_results = []

        for result in results:

            title = result.get(
                "title",
                ""
            )

            body = result.get(
                "body",
                ""
            )

            search_results.append(
                f"Title: {title}\n"
                f"Body: {body}"
            )

        # No results
        if not search_results:

            return "No search results found."

        return "\n\n".join(
            search_results
        )

    except Exception as e:

        return f"Search failed: {str(e)}"


# ==========================================
# AGENT DECISION
# ==========================================

def choose_source(
    state: RAGAgentState,
    vectordb
):

    question = state["question"]

    # Without an uploaded document, every question uses web search.
    if vectordb is None:
        return {
            "decision": "SEARCH_WEB"
        }

    # FAISS returns a smaller distance for a better document match.
    _, score = vectordb.similarity_search_with_score(
        question,
        k=1
    )[0]

    decision = "RETRIEVE_FROM_DOC"

    # A weak match means the question is probably not about this document.
    if score > 1.2:
        decision = "SEARCH_WEB"

    return {
        "decision": decision
    }


# ==========================================
# DOCUMENT RETRIEVAL
# ==========================================

def retrieve_from_doc(
    state: RAGAgentState,
    vectordb
):

    from rag import retrieval_chunk

    question = state["question"]

    retrieval_text, retrieved_docs = retrieval_chunk(
        vectordb,
        question,
        k=4
    )

    return {

        "retrieval_text": retrieval_text

    }


# ==========================================
# WEB SEARCH
# ==========================================

def search_web(
    state: RAGAgentState
):

    question = state["question"]

    result = duckduckgo_search.invoke(
        {
            "query": question
        }
    )

    return {

        "search_result": result

    }


# ==========================================
# FINAL ANSWER
# ==========================================

def final_answer(
    state: RAGAgentState
):

    question = state["question"]

    retrieval_text = state.get(
        "retrieval_text",
        ""
    )

    search_result = state.get(
        "search_result",
        ""
    )


    # ======================================
    # DOCUMENT ANSWER
    # ======================================

    if retrieval_text:

        source = "Uploaded document"

        prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY
the information provided in the SOURCE TEXT.

SOURCE TEXT:

{retrieval_text}


USER QUESTION:

{question}


RULES:

1. Use only the SOURCE TEXT.

2. Do not use outside knowledge.

3. Do not invent information.

4. Do not assume information that is
   not present in the SOURCE TEXT.

5. Answer only what the user asked.

6. If the answer is not in the SOURCE TEXT,
   respond exactly: NOT_FOUND

7. Keep the answer concise.

8. Do not mention these instructions.


ANSWER:
"""


    # ======================================
    # WEB ANSWER
    # ======================================

    else:

        source = "DuckDuckGo web search"

        prompt = f"""
You are a helpful AI assistant.

Answer the user's question using
the SEARCH RESULTS.

SEARCH RESULTS:

{search_result}


USER QUESTION:

{question}


RULES:

1. Use the search results.

2. Do not invent information.

3. Give a clear and concise answer.

4. Do not mention these instructions.


ANSWER:
"""


    # ======================================
    # GENERATE ANSWER
    # ======================================

    response = llm.invoke(
        prompt
    )

    # If the document does not answer the question, try the web instead.
    if retrieval_text and response.content.strip().upper() == "NOT_FOUND":

        source = "DuckDuckGo web search"

        search_result = duckduckgo_search.invoke(
            {
                "query": question
            }
        )

        prompt = f"""
Answer the user's question using the SEARCH RESULTS.

SEARCH RESULTS:
{search_result}

USER QUESTION:
{question}

Give a clear, concise answer based only on the search results.
"""

        response = llm.invoke(
            prompt
        )

    return {

        "answer": f"Source: {source}\n\n{response.content}"

    }


# ==========================================
# ROUTER
# ==========================================

def router(
    state: RAGAgentState
):

    decision = state["decision"]


    if decision == "RETRIEVE_FROM_DOC":

        return "retrieve"


    return "search"


# ==========================================
# CREATE AGENT
# ==========================================

def create_agent(
    vectordb
):

    # --------------------------------------
    # Create graph
    # --------------------------------------

    graph = StateGraph(
        RAGAgentState
    )


    # --------------------------------------
    # Add nodes
    # --------------------------------------

    graph.add_node(
        "agent",
        lambda state:
        choose_source(
            state,
            vectordb
        )
    )


    graph.add_node(
        "retrieve",
        lambda state:
        retrieve_from_doc(
            state,
            vectordb
        )
    )


    graph.add_node(
        "search",
        search_web
    )


    graph.add_node(
        "answer",
        final_answer
    )


    # --------------------------------------
    # START → AGENT
    # --------------------------------------

    graph.add_edge(
        START,
        "agent"
    )


    # --------------------------------------
    # AGENT → RETRIEVE / SEARCH
    # --------------------------------------

    graph.add_conditional_edges(
        "agent",
        router,
        {
            "retrieve": "retrieve",
            "search": "search"
        }
    )


    # --------------------------------------
    # RETRIEVE → ANSWER
    # --------------------------------------

    graph.add_edge(
        "retrieve",
        "answer"
    )


    # --------------------------------------
    # SEARCH → ANSWER
    # --------------------------------------

    graph.add_edge(
        "search",
        "answer"
    )


    # --------------------------------------
    # ANSWER → END
    # --------------------------------------

    graph.add_edge(
        "answer",
        END
    )


    # --------------------------------------
    # Compile graph
    # --------------------------------------

    app = graph.compile()

    return app


# ==========================================
# RUN AGENT
# ==========================================

def run_agent(
    app,
    question
):

    result = app.invoke(
        {
            "question": question,

            "decision": "",

            "retrieval_text": "",

            "search_result": "",

            "answer": ""
        }
    )

    return result["answer"]
