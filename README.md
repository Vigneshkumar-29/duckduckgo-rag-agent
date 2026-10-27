# DuckDuckGo AI RAG Agent with Voice-to-Text

This project is an AI application that combines:

- a document-based retrieval pipeline using FAISS and embeddings
- document-match routing that chooses the uploaded file or the web
- a web search tool using DuckDuckGo
- **AssemblyAI voice-to-text conversion** for spoken voice questions
- a ChatGPT-style chat interface with a built-in voice button and **+** button for TXT uploads

It is built around a LangGraph workflow, with the logic split across the main app, the agent orchestration, the RAG pipeline, and the voice-to-text service.

## Project structure

- `main.py` – Streamlit frontend with query box supporting text, voice recording, and TXT attachments
- `voice_to_text.py` – AssemblyAI integration for transcribing voice queries and audio files
- `agent.py` – LangGraph agent, decision routing, web search tool, final answer generation
- `rag.py` – document preprocessing, chunking, embeddings, FAISS vector store, retrieval logic
- `.env` / `.env.example` – environment file for your AssemblyAI API key

## How it works

1. **Document Upload**: The user uploads a `.txt` document using the **+** button in the query box.
2. **Indexing**: The document is split into chunks, embedded, and stored in a FAISS vector database.
3. **Asking Questions**: Users can ask questions by typing or by clicking the voice recording button in the query box.
4. **Voice Transcription**: Spoken audio is transcribed to text using AssemblyAI.
5. **Smart Routing**: When a question is asked, the agent checks how closely it matches the uploaded document.
6. A close match is answered from the document. Otherwise, DuckDuckGo searches the web.
7. Each answer displays whether it came from the uploaded document or DuckDuckGo web search.

## Features

- **Voice to Text**: Speak directly into the chat query box using the microphone button (powered by AssemblyAI)
- Upload and process local text documents (.txt) with the **+** attachment button
- Clean chat interface with voice button built directly into the query box
- Query the document using semantic retrieval
- Answer from document chunks when they contain the answer
- Automatically fall back to DuckDuckGo when the document does not answer the question
- Show the source of every answer
- Use a LangGraph workflow for routing between tasks
- Run through a simple Streamlit UI

## Requirements

Install the Python packages:

```bash
pip install -r requirements.txt
```

You will also need:

- Ollama installed and running locally
- an Ollama model available such as `llama3.1`
- the spaCy English model downloaded:

```bash
python -m spacy download en_core_web_sm
```

- An AssemblyAI API key:
  - Add your key to the `.env` file in the project root:
    ```env
    ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
    ```

Hugging Face may download the embedding model the first time the app processes a document.

## Run the app

From the project directory:

```bash
streamlit run main.py
```

Then:
1. Upload a text file with the **+** button in the query box
2. Type a question or click the microphone button in the query box to speak
