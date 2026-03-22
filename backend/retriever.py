from __future__ import annotations

from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def get_retriever(db: FAISS, k: int = 3):
    """Create a similarity-based retriever from the persisted vector store."""
    return db.as_retriever(search_type="similarity", search_kwargs={"k": k})


def retrieve_with_scores(db: FAISS, query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Return retrieved chunks with similarity scores and metadata."""
    results = db.similarity_search_with_score(query, k=k)
    return [
        {
            "content": document.page_content,
            "score": float(score),
            "metadata": document.metadata,
            "document": document,
        }
        for document, score in results
    ]


def get_relevant_documents(db: FAISS, query: str, k: int = 3) -> List[Document]:
    """Convenience wrapper for standard similarity retrieval."""
    retriever = get_retriever(db, k=k)
    return retriever.invoke(query)
