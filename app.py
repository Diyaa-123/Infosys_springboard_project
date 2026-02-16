import os
import shutil
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    TextLoader,
    DirectoryLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

# -----------------------------
# Startup Cleanup
# -----------------------------
def startup_cleanup():
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")

# -----------------------------
# Metadata Cleaning Function
# -----------------------------
def clean_metadata(documents):
    for doc in documents:
        doc.metadata = {k: str(v) for k, v in doc.metadata.items()}
    return documents

# -----------------------------
# Load Documents
# -----------------------------
def load_documents():
    pdf_loader = DirectoryLoader(
        "data",
        glob="**/*.pdf",
        loader_cls=UnstructuredPDFLoader
    )

    text_loader = DirectoryLoader(
        "data",
        glob="**/*.txt",
        loader_cls=TextLoader
    )

    pdf_docs = pdf_loader.load()
    text_docs = text_loader.load()

    return pdf_docs + text_docs

# -----------------------------
# Chunk Documents
# -----------------------------
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)

# -----------------------------
# Create Vector Store
# -----------------------------
def create_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="document_collection"
    )

    vectorstore.persist()

    return vectorstore


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Document Ingestion & Indexing")

if st.button("Build Vector Database"):

    st.write("Cleaning old database...")
    startup_cleanup()

    st.write("Loading documents...")
    documents = load_documents()

    st.write(f"Loaded {len(documents)} documents")

    documents = clean_metadata(documents)

    st.write("Splitting into chunks...")
    chunks = split_documents(documents)

    st.write(f"Total Chunks: {len(chunks)}")

    st.write("Creating vector database...")
    create_vector_store(chunks)

    st.success("Vector Database Built Successfully!")
