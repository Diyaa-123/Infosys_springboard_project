from __future__ import annotations

import logging
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.embeddings import get_embedding_model
from backend.loader import load_documents
from backend.splitter import split_documents
from backend.vectorstore import (
    create_vectorstore,
    load_vectorstore,
    save_vectorstore,
    vectorstore_exists,
)
from config import DOCUMENTS_DIR, VECTORSTORE_DIR


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def run_ingestion_pipeline():
    """Execute the full ingestion pipeline and persist the vector store."""
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        raise ValueError(
            "No valid documents were loaded. Add PDF or TXT files to the documents directory."
        )

    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("Document loading succeeded, but no chunks were produced.")

    embedding_model = get_embedding_model()
    db = create_vectorstore(chunks, embedding_model)
    save_vectorstore(db, VECTORSTORE_DIR)

    logger.info("Created vector store with %s chunks.", len(chunks))
    return db


def load_existing_db_or_create():
    """Reuse a persisted vector store when present, otherwise create it."""
    if vectorstore_exists(VECTORSTORE_DIR):
        logger.info("Existing vector store found. Loading from disk.")
        embedding_model = get_embedding_model()
        return load_vectorstore(embedding_model, VECTORSTORE_DIR)

    logger.info("No vector store found. Running ingestion pipeline.")
    return run_ingestion_pipeline()


if __name__ == "__main__":
    db = load_existing_db_or_create()
    print("Vector DB ready")
