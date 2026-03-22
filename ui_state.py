from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import reflex as rx

from state.chat_state import ChatState


@dataclass
class ConversationTurn:
    question: str
    answer: str
    sources: list[str] = field(default_factory=list)


class AppState(rx.State):
    """Reflex UI state that delegates chat logic to the existing ChatState."""

    question: str = ""
    history: list[ConversationTurn] = []
    upload_status: str = "Upload PDF or TXT files, then run the ingestion pipeline to index them."
    chat_status: str = "Ask a question about your indexed documents."
    is_processing: bool = False
    _chat_engine: ChatState = ChatState()

    def sync_history(self) -> None:
        """Mirror the existing chat engine state into the Reflex state."""
        self.history = self._serialize_history()

    def set_question(self, value: str) -> None:
        """Explicit setter for the chat input."""
        self.question = value

    def submit_question(self) -> None:
        """Send the current question through the existing conversational backend."""
        question = self.question.strip()
        if not question:
            self.chat_status = "Please enter a non-empty question."
            return

        self.is_processing = True
        self.chat_status = "Searching documents and generating an answer..."
        response = self._chat_engine.ask(question)
        self.history = self._serialize_history()
        self.question = ""
        self.is_processing = False

        answer = response.get("answer", "").strip()
        if not answer:
            self.chat_status = "No answer was returned."
            return

        self.chat_status = "Answer ready."

    async def handle_upload(self, files: list[rx.UploadFile]) -> None:
        """Save uploaded documents into the existing documents directory."""
        if not files:
            self.upload_status = "No files selected."
            return

        documents_dir = Path("documents")
        documents_dir.mkdir(parents=True, exist_ok=True)

        saved_files: list[str] = []
        skipped_files: list[str] = []

        for file in files:
            filename = file.filename or "uploaded_file"
            suffix = Path(filename).suffix.lower()
            if suffix not in {".pdf", ".txt"}:
                skipped_files.append(filename)
                continue

            target_path = documents_dir / filename
            target_path.write_bytes(await file.read())
            saved_files.append(filename)

        messages: list[str] = []
        if saved_files:
            messages.append(
                f"Saved {len(saved_files)} file(s): {', '.join(saved_files)}. Run the ingestion pipeline to index them."
            )
        if skipped_files:
            messages.append(
                f"Skipped unsupported file(s): {', '.join(skipped_files)}."
            )

        self.upload_status = " ".join(messages) or "No supported files were uploaded."

    def _serialize_history(self) -> list[ConversationTurn]:
        return [
            ConversationTurn(
                question=str(turn.get("question", "")),
                answer=str(turn.get("answer", "")),
                sources=list(turn.get("sources", [])),
            )
            for turn in self._chat_engine.history
        ]
