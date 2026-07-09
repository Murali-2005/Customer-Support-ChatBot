# import os
# from dotenv import load_dotenv


# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain_core.prompts import ChatPromptTemplate
# from entity_extractor import extract_entities

# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# VECTOR_DB = "medical_faiss"

# #Gemini
# llm=ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key=GOOGLE_API_KEY,
#     temperature=0
# )

# #Embeddings
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# def get_medical_response(question):

#     vectordb=FAISS.load_local(
#         VECTOR_DB,
#         embeddings,
#         allow_dangerous_deserialization=True
#     )

#     retriever = vectordb.as_retriever(
#         search_kwargs={"k": 3}
#     )

#     docs = retriever.invoke(question)

#     context = "\n\n".join(
#         [doc.page_content for doc in docs]
#     )

#     prompt = ChatPromptTemplate.from_template(
#     """
#     You are a medical question answering assistant.
#     Answer ONLY using the retrieved medical evidence.
#     If the answer cannot be found in the context, reply:
#     I don't know based on the available medical evidence.

#     Retrieved Context:
#     {context}
#     Question:
#     {question}
#     """
#     )

#     chain = prompt | llm

#     response = chain.invoke({
#         "context": context,
#         "question": question
#     })

#     return response.content


import os
from dotenv import load_dotenv

import streamlit as st
from spellchecker import SpellChecker

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

VECTOR_DB = "medical_faiss"

# How many past turns (user+assistant pairs) to keep in the prompt/search
# query. Keeps token usage and search-query length bounded as the
# conversation grows.
MAX_HISTORY_TURNS = 6


# ------------------------------------
# Cached resources
# ------------------------------------
# st.cache_resource ensures these are built ONCE per app process and
# reused across every Streamlit rerun (every button click / text input
# triggers a full script rerun in Streamlit, so without this you'd
# reload the FAISS index and reconnect to Gemini on every interaction).
# It's also hot-reload safe, unlike plain module-level globals.

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_spellchecker():
    return SpellChecker()


def correct_spelling(text):
    """
    Fix obvious typos (e.g. "diabietes" -> "diabetes") before embedding
    the query. Sentence-embedding models are somewhat typo-tolerant but
    not fully — a single misspelled key term can push the query far
    enough from the correct FAISS cluster that retrieval returns
    irrelevant docs, which then makes the LLM (correctly, given its
    instructions) fall back to "I don't know based on the available
    medical evidence."

    This only touches words the spellchecker doesn't recognize, and
    leaves the word unchanged if no correction is found — so it won't
    mangle valid medical terms/abbreviations it simply hasn't seen.
    """
    spell = get_spellchecker()
    words = text.split()

    corrected_words = []
    for word in words:
        # Strip trailing punctuation so "diabietes?" is checked as
        # "diabietes", then re-attach the punctuation after correction.
        stripped = word.strip(".,?!;:")
        suffix = word[len(stripped):]

        if stripped and stripped.lower() not in spell:
            correction = spell.correction(stripped)
            if correction:
                # Preserve original capitalization style.
                if stripped[0].isupper():
                    correction = correction.capitalize()
                corrected_words.append(correction + suffix)
                continue

        corrected_words.append(word)

    return " ".join(corrected_words)


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )


@st.cache_resource
def get_vectordb():
    return FAISS.load_local(
        VECTOR_DB,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )


@st.cache_resource
def get_retriever():
    return get_vectordb().as_retriever(
        search_kwargs={"k": 5}
    )


@st.cache_resource
def get_prompt():
    return ChatPromptTemplate.from_template(
"""
You are an intelligent Medical AI Assistant.

You have two information sources:

1. Previous Conversation
2. Retrieved Medical Knowledge

Conversation History:

{history}

Retrieved Medical Context:

{context}

Current Question:

{question}

Instructions:

- Use the conversation history to resolve follow-up questions such as
  "its", "they", "that disease", etc.

- Use the retrieved medical evidence as the PRIMARY source.

- If the retrieved context partially answers the question,
  combine it with your medical reasoning.

- Keep your answer SHORT: no more than 5 lines total. Prefer 2-3
  concise sentences or a few short bullet points over long explanations.
  Do not list every sub-type, cause, or detail — give only the most
  important information needed to answer the question.

- If the answer cannot be determined,
  reply exactly:

I don't know based on the available medical evidence.

- Never fabricate medical facts.
"""
    )


@st.cache_resource
def get_chain():
    return get_prompt() | get_llm()


# ------------------------------------
# Conversation History
# ------------------------------------
# NOTE: A plain module-level list is fine only if you run one
# Streamlit session per process. For a real multi-user deployment,
# move chat_history into st.session_state in app.py instead — a plain
# list here is shared across ALL users hitting the same server
# process, which will mix up conversations.

chat_history = []


def add_to_history(role, message):
    chat_history.append({
        "role": role,
        "content": message
    })


def get_history(max_turns=MAX_HISTORY_TURNS):
    """
    Build the history string using a bounded window of recent turns.

    Using ''.join(...) instead of repeated += avoids O(n^2) string
    rebuilding as history grows, and capping the window keeps the
    search query and prompt context from growing unbounded over a
    long conversation.
    """
    windowed = chat_history[-(max_turns * 2):] if max_turns else chat_history

    return "".join(
        f"{chat['role']}: {chat['content']}\n" for chat in windowed
    )


def clear_history():
    chat_history.clear()


# ------------------------------------
# Medical Chatbot
# ------------------------------------

def get_medical_response(question):

    # Correct obvious typos before retrieval/answering — see
    # correct_spelling() docstring for why this matters for retrieval
    # accuracy specifically.
    question = correct_spelling(question)

    history = get_history()

    if history.strip():
        search_query = history + "\nCurrent Question: " + question
    else:
        search_query = question

    docs = get_retriever().invoke(search_query)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    response = get_chain().invoke({
        "history": history,
        "context": context,
        "question": question
    })

    add_to_history("User", question)
    add_to_history("Assistant", response.content)

    return response.content
