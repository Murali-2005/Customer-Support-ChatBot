from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATASET = "medical_dataset.csv"
VECTOR_DB = "medical_faiss"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading Medical Dataset...")

loader = CSVLoader(
    file_path=DATASET,
    source_column="prompt",
    encoding="utf-8"
)

documents = loader.load()

print(f"{len(documents)} questions loaded.")

print("Creating FAISS Index...")

vectordb = FAISS.from_documents(
    documents,
    embeddings
)

vectordb.save_local(VECTOR_DB)

print("====================================")
print("Medical FAISS Created Successfully")
print("Location:", VECTOR_DB)
print("====================================")