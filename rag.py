# rag.py

import re
import contractions
import spacy

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ==========================================
# LOAD SPACY MODEL
# ==========================================

nlp = spacy.load("en_core_web_sm")


# ==========================================
# TEXT NORMALIZATION
# ==========================================

def normalize_text(data):

    # Convert text to lowercase
    data = data.lower()

    # Expand contractions
    # Example:
    # don't → do not
    # can't → cannot
    data = contractions.fix(data)

    # Remove numbers
    data = re.sub(r"\b\d+\b", " ", data)

    # Remove special characters
    data = re.sub(r"[^a-zA-Z0-9\s]", " ", data)

    # Remove extra spaces
    data = re.sub(r"\s+", " ", data)

    return data.strip()


# ==========================================
# LEMMATIZATION
# ==========================================

def lemmatize_text(data):

    doc = nlp(data)

    tokens = []

    for token in doc:

        # Remove stop words
        if token.is_stop:
            continue

        # Remove spaces
        if token.is_space:
            continue

        # Add lemma
        tokens.append(token.lemma_)

    return " ".join(tokens).strip()


# ==========================================
# CHUNKING
# ==========================================

def create_chunks(data):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.create_documents(
        [data]
    )

    return chunks


# ==========================================
# EMBEDDING MODEL
# ==========================================

def create_embedding_model():

    embedding_model = HuggingFaceEmbeddings(
        model_name="google/embeddinggemma-300m",
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embedding_model


# ==========================================
# VECTOR DATABASE
# ==========================================

def create_vector_db(
    chunks,
    embedding_model
):

    vectordb = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vectordb


# ==========================================
# RETRIEVAL
# ==========================================

def retrieval_chunk(
    vectordb,
    query,
    k=3
):

    retrieved_docs = vectordb.similarity_search(
        query,
        k=k
    )

    retrieved_chunks = []

    for doc in retrieved_docs:

        retrieved_chunks.append(
            doc.page_content
        )

    retrieval_text = "\n".join(
        retrieved_chunks
    )

    return retrieval_text, retrieved_chunks


# ==========================================
# CREATE RAG
# ==========================================

def create_rag(data):

    # Keep the original document text
    chunks = create_chunks(data)

    print(f"Total chunks created: {len(chunks)}")

    embedding_model = create_embedding_model()

    vectordb = create_vector_db(
        chunks,
        embedding_model
    )

    return vectordb
