from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.rag import rag_pipeline
from state.memory import build_contextual_query, trim_history


class ChatState:
    """Stateful conversational wrapper around the existing RAG pipeline."""

    def __init__(self) -> None:
        self.question: str = ""
        self.answer: str = ""
        self.history: List[Dict[str, Any]] = []

    def ask(self, question: str) -> Dict[str, Any]:
        """Process a user question while preserving bounded conversation history."""
        if not question or not question.strip():
            return {
                "question": "",
                "answer": "Please enter a non-empty question.",
                "sources": [],
            }

        self.question = question.strip()
        contextual_query = build_contextual_query(self.question, self.history)

        try:
            response = rag_pipeline(contextual_query)
        except Exception:
            response = {
                "answer": "The system could not process your request at this time.",
                "sources": [],
            }

        self.answer = response.get("answer") or "I don't know based on the provided documents."
        result = {
            "question": self.question,
            "answer": self.answer,
            "sources": response.get("sources", []),
        }

        self.history.append(result)
        self.history = trim_history(self.history, max_length=5)
        return result


if __name__ == "__main__":
    chat = ChatState()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        response = chat.ask(user_input)
        print(f"Bot: {response['answer']}")
        print(f"Sources: {response['sources']}")
