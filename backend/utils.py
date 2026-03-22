from __future__ import annotations

import re
from typing import Iterable, List

from langchain_core.documents import Document


def extract_sources(documents: Iterable[Document]) -> List[str]:
    """Extract de-duplicated source citations from retrieved documents."""
    seen = set()
    sources: List[str] = []

    for document in documents:
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page")
        label = f"{source} - page {page}" if page is not None else source
        if label not in seen:
            seen.add(label)
            sources.append(label)

    return sources


def rewrite_query(query: str) -> str:
    """Lightweight corrective rewrite for retry retrieval."""
    normalized = re.sub(r"\s+", " ", query).strip()
    normalized = re.sub(r"[^\w\s?-]", "", normalized)
    return normalized


def format_context(documents: Iterable[Document]) -> List[str]:
    """Return retrieved chunk text for debugging or API responses."""
    return [document.page_content for document in documents]
