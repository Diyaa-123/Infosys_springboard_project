from __future__ import annotations

import logging
import os
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

from config import (
    EMBEDDING_PROVIDER,
    HUGGINGFACE_EMBEDDING_MODEL,
    OPENAI_API_BASE_URL,
    OPENAI_EMBEDDING_MODEL,
)


logger = logging.getLogger(__name__)


class OpenAICompatibleEmbeddings(Embeddings):
    """LangChain-compatible wrapper for OpenAI-compatible embeddings APIs."""

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding


def get_embedding_model() -> Embeddings:
    """Return an embedding model configured from environment variables."""
    if EMBEDDING_PROVIDER == "huggingface":
        try:
            return HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL)
        except Exception as exc:
            logger.warning(
                "Falling back to local Hugging Face cache for '%s': %s",
                HUGGINGFACE_EMBEDDING_MODEL,
                exc,
            )
            return HuggingFaceEmbeddings(
                model_name=HUGGINGFACE_EMBEDDING_MODEL,
                model_kwargs={"local_files_only": True},
            )

    if EMBEDDING_PROVIDER == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai.")

        return OpenAICompatibleEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=api_key,
            base_url=OPENAI_API_BASE_URL,
        )

    raise ValueError(
        "Unsupported EMBEDDING_PROVIDER. Use 'huggingface' or 'openai'."
    )
