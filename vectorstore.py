from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os
import hashlib

BASE_PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")

def pdf_hash(file):
    """Create a unique hash for each PDF"""
    file.seek(0)
    pdf_bytes = file.read()
    file.seek(0)
    return hashlib.md5(pdf_bytes).hexdigest()

def create_or_load_vectorstore(texts, pdf_id):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_dir = os.path.join(BASE_PERSIST_DIR, pdf_id)
    os.makedirs(persist_dir, exist_ok=True)

    # Load existing DB if exists
    if os.path.exists(os.path.join(persist_dir, "index")):
        return Chroma(
            collection_name="pdf_documents",
            embedding_function=embeddings,
            persist_directory=persist_dir
        )

    vectorstore = Chroma.from_texts(
        texts,
        embedding=embeddings,
        collection_name="pdf_documents",
        persist_directory=persist_dir
    )
    vectorstore.persist()
    return vectorstore
