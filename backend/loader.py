from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from config import DOCUMENTS_DIR, SUPPORTED_EXTENSIONS


logger = logging.getLogger(__name__)


def _is_effectively_empty(documents: List[Document]) -> bool:
    return not any(document.page_content.strip() for document in documents)


def _load_pdf(file_path: Path) -> List[Document]:
    try:
        documents = PyPDFLoader(str(file_path)).load()
    except Exception as exc:
        logger.exception("Failed to read PDF '%s': %s", file_path.name, exc)
        return []

    if _is_effectively_empty(documents):
        logger.warning("Skipping empty PDF '%s'.", file_path.name)
        return []

    for document in documents:
        document.metadata["source"] = file_path.name

    return documents


def _load_txt(file_path: Path) -> List[Document]:
    try:
        documents = TextLoader(str(file_path), autodetect_encoding=True).load()
    except Exception as exc:
        logger.exception("Failed to read TXT '%s': %s", file_path.name, exc)
        return []

    if _is_effectively_empty(documents):
        logger.warning("Skipping empty TXT '%s'.", file_path.name)
        return []

    for document in documents:
        document.metadata["source"] = file_path.name
        document.metadata.setdefault("page", 0)

    return documents


def load_documents(documents_dir: Path | None = None) -> List[Document]:
    """Load supported documents from the configured directory."""
    target_dir = documents_dir or DOCUMENTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    all_documents: List[Document] = []

    for file_path in sorted(target_dir.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if file_path.suffix.lower() == ".pdf":
            all_documents.extend(_load_pdf(file_path))
        elif file_path.suffix.lower() == ".txt":
            all_documents.extend(_load_txt(file_path))

    logger.info("Loaded %s document sections from %s.", len(all_documents), target_dir)
    return all_documents
