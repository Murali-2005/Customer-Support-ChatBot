import os
from dotenv import load_dotenv

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

VECTOR_DB_PATH = "faiss_index"

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
)

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


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

    print("✅ Vector Database Created Successfully")


def get_qa_chain():

    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError(
            "Vector database not found. Click 'Create Knowledge Base' first."
        )

    vectordb = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful customer service chatbot.

Use ONLY the provided context to answer the user's question.

If the answer cannot be found in the context, reply exactly:

I don't know.

Context:
{context}

Question:
{input}
"""
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain


if __name__ == "__main__":
    create_vector_db()