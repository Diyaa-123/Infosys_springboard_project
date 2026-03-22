from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from config import FAISS_INDEX_FILE, FAISS_METADATA_FILE, VECTORSTORE_DIR


def create_vectorstore(chunks: Iterable[Document], embedding_model: Embeddings) -> FAISS:
    """Build a FAISS vector store from document chunks."""
    chunk_list = list(chunks)
    if not chunk_list:
        raise ValueError("Cannot create a vector store without document chunks.")

    return FAISS.from_documents(chunk_list, embedding_model)


def save_vectorstore(db: FAISS, path: str | Path = VECTORSTORE_DIR) -> None:
    """Persist the FAISS index to disk for later reuse."""
    save_path = Path(path)
    save_path.mkdir(parents=True, exist_ok=True)
    db.save_local(str(save_path))


def load_vectorstore(
    embedding_model: Embeddings,
    path: str | Path = VECTORSTORE_DIR,
) -> FAISS:
    """Load a previously persisted FAISS index."""
    load_path = Path(path)
    return FAISS.load_local(
        str(load_path),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )


def vectorstore_exists(path: str | Path = VECTORSTORE_DIR) -> bool:
    """Check whether the FAISS persistence artifacts already exist."""
    target_path = Path(path)
    return (target_path / FAISS_INDEX_FILE).exists() and (
        target_path / FAISS_METADATA_FILE
    ).exists()
