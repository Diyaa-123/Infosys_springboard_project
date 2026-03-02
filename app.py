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
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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
# Setup QA Chain (RAG Pipeline)
# -----------------------------
def get_qa_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="document_collection"
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Initialize Groq LLM
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        api_key=os.environ.get("Groq_API_KEY")
    )
    
    prompt_template = """
    Use the following pieces of context to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context:
    {context}
    
    Question: {input}
    
    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "input"]
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


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

st.divider()

st.header("💬 Query Documents (RAG Pipeline)")

user_query = st.text_input("Ask a question about the documents:", placeholder="e.g., What is the main topic of the reports?")

if user_query:
    if not os.path.exists("./chroma_db"):
        st.error("Vector database not found. Please build it first by clicking the button above.")
    else:
        with st.spinner("Retrieving documents and generating answer..."):
            qa_chain = get_qa_chain()
            response = qa_chain.invoke({"input": user_query})
            
            st.subheader("Answer:")
            st.write(response["answer"])
            
            st.subheader("Sources:")
            for idx, doc in enumerate(response["context"]):
                source = doc.metadata.get("source", "Unknown Source")
                st.markdown(f"**Source {idx + 1}:** `{source}`")
                with st.expander(f"View Snippet matching query"):
                    st.write(doc.page_content)
