import os
from dotenv import load_dotenv

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found.")

VECTOR_DB_PATH = "faiss_index"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Conversation Memory
# -----------------------------

import streamlit as st

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Create Initial Vector DB
# -----------------------------


def create_vector_db():

    loader = CSVLoader(
        file_path="dataset.csv",
        source_column="prompt"
    )

    documents = loader.load()

    vectordb = FAISS.from_documents(
        documents,
        embeddings
    )

    vectordb.save_local(VECTOR_DB_PATH)

    print("Knowledge Base Created")


# -----------------------------
# Load Retriever
# -----------------------------


def get_retriever():

    vectordb = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectordb.as_retriever(
        search_kwargs={"k":3}
    )


# -----------------------------
# MultiModal Reasoning
# -----------------------------


def get_multimodal_response(question, image_context=""):

    retriever = get_retriever()

    docs = retriever.invoke(question)

    retrieved_context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    history = get_history()

    prompt = ChatPromptTemplate.from_template(
"""
You are an intelligent AI Customer Support Assistant.

You have FOUR information sources:

1. Retrieved Knowledge Base
2. Image Analysis
3. Previous Conversation
4. Current User Question

Rules:

• Always answer using the retrieved knowledge first.

• Use image analysis only if it is relevant.

• Use previous conversation if necessary.

• If image and retrieved knowledge disagree,
say so.

• If information is missing,
reply:

'I don't know based on the available evidence.'

Retrieved Knowledge:
{context}

Image Analysis:
{image_context}

Conversation History:
{history}

Current Question:
{question}

Provide one evidence-based answer.
"""
    )

    chain = prompt | llm

    response = chain.invoke({

        "context": retrieved_context,

        "image_context": image_context,

        "history": history,

        "question": question

    })

    add_to_history("user", question)
    add_to_history("assistant", response.content)
    return response.content

def add_to_history(role, message):
    st.session_state.chat_history.append({
        "role": role,
        "content": message
    })
def get_history():

    history = ""

    for chat in st.session_state.chat_history:

        history += f"{chat['role'].capitalize()}: {chat['content']}\n"

    return history

def clear_history():
    st.session_state.chat_history = []