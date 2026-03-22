from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.embeddings import get_embedding_model
from backend.generator import generate_answer
from backend.retriever import retrieve_with_scores
from backend.utils import extract_sources, format_context, rewrite_query
from backend.vectorstore import load_vectorstore, vectorstore_exists
from config import MAX_CONTEXT_CHUNKS, RETRIEVER_K, SIMILARITY_SCORE_THRESHOLD, VECTORSTORE_DIR


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _load_db():
    if not vectorstore_exists(VECTORSTORE_DIR):
        raise FileNotFoundError(
            "Vector store not found. Run the ingestion pipeline before starting RAG."
        )

    embedding_model = get_embedding_model()
    return load_vectorstore(embedding_model, VECTORSTORE_DIR)


def _should_retry(results: List[Dict[str, Any]]) -> bool:
    if not results:
        return True

    best_score = min(item["score"] for item in results)
    return best_score > SIMILARITY_SCORE_THRESHOLD


def _has_relevant_results(results: List[Dict[str, Any]]) -> bool:
    if not results:
        return False

    best_score = min(item["score"] for item in results)
    return best_score <= SIMILARITY_SCORE_THRESHOLD


def rag_pipeline(query: str) -> Dict[str, Any]:
    """Run retrieval-augmented generation over the existing FAISS store."""
    if not query or not query.strip():
        return {
            "answer": "Query cannot be empty.",
            "sources": [],
            "context": [],
        }

    try:
        db = _load_db()
    except FileNotFoundError as exc:
        return {"answer": str(exc), "sources": [], "context": []}
    except Exception as exc:
        logger.exception("Failed to load vector store: %s", exc)
        return {"answer": "Failed to load the vector database.", "sources": [], "context": []}

    try:
        retrieval_query = query
        scored_results = retrieve_with_scores(db, retrieval_query, k=RETRIEVER_K)
        if _should_retry(scored_results):
            rewritten_query = rewrite_query(query)
            if rewritten_query and rewritten_query != query:
                logger.info("Retrying retrieval with rewritten query.")
                retrieval_query = rewritten_query
                scored_results = retrieve_with_scores(db, retrieval_query, k=RETRIEVER_K)
    except Exception as exc:
        logger.exception("Retrieval failed: %s", exc)
        return {"answer": "Failed to retrieve relevant context.", "sources": [], "context": []}

    if not _has_relevant_results(scored_results):
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": [],
            "context": [],
        }

    documents = [item["document"] for item in scored_results[:MAX_CONTEXT_CHUNKS]]
    if not documents:
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": [],
            "context": [],
        }

    try:
        answer = generate_answer(query, documents)
    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        return {
            "answer": "The system could not generate an answer at this time.",
            "sources": extract_sources(documents),
            "context": format_context(documents),
        }

    return {
        "answer": answer,
        "sources": extract_sources(documents),
        "context": format_context(documents),
    }


if __name__ == "__main__":
    user_query = input("Enter your question: ").strip()
    result = rag_pipeline(user_query)
    print(f"Answer: {result['answer']}")
    print("Sources:")
    if result["sources"]:
        for source in result["sources"]:
            print(f"- {source}")
    else:
        print("- None")
