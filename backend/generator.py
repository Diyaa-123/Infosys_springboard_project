from __future__ import annotations

import os
from typing import Iterable

from langchain_core.documents import Document
from openai import OpenAI

from config import (
    GROQ_API_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    OPENAI_API_BASE_URL,
    RAG_SYSTEM_PROMPT,
)


def _get_llm_client() -> OpenAI:
    if LLM_PROVIDER == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq.")

        return OpenAI(api_key=api_key, base_url=GROQ_API_BASE_URL)

    if LLM_PROVIDER == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")

        return OpenAI(api_key=api_key, base_url=OPENAI_API_BASE_URL)

    raise ValueError("Unsupported LLM_PROVIDER. Use 'groq' or 'openai'.")


def _build_context(documents: Iterable[Document]) -> str:
    chunks = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page")
        page_label = f", page {page}" if page is not None else ""
        chunks.append(
            f"[Chunk {index} | source: {source}{page_label}]\n{document.page_content}"
        )
    return "\n\n".join(chunks)


def generate_answer(query: str, documents: Iterable[Document]) -> str:
    """Generate a grounded answer using retrieved context only."""
    document_list = list(documents)
    if not document_list:
        return "I don't know based on the provided documents."

    context = _build_context(document_list)
    client = _get_llm_client()

    response = client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    f"Context:\n{context}\n\n"
                    "Answer using only the context above."
                ),
            },
        ],
    )

    answer = getattr(response, "output_text", "") or ""
    return answer.strip() or "I don't know based on the provided documents."
