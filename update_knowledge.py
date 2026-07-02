import os
import shutil

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTOR_DB_PATH = "faiss_index"
UPDATES_FOLDER = "updates"
PROCESSED_FOLDER = "processed"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def update_vector_db():

    os.makedirs(UPDATES_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    files = [
        f for f in os.listdir(UPDATES_FOLDER)
        if f.endswith(".csv")
    ]

    if len(files) == 0:
        return "No new CSV files found."

    vectordb = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    total = 0

    for file in files:

        path = os.path.join(UPDATES_FOLDER, file)

        loader = CSVLoader(
            file_path=path,
            source_column="prompt"
        )

        docs = loader.load()

        vectordb.add_documents(docs)

        total += len(docs)

        shutil.move(
            path,
            os.path.join(PROCESSED_FOLDER, file)
        )

    vectordb.save_local(VECTOR_DB_PATH)

    return f"{total} new documents added successfully."