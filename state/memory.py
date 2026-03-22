from __future__ import annotations

from typing import Dict, List


HistoryEntry = Dict[str, object]


def trim_history(history: List[HistoryEntry], max_length: int = 5) -> List[HistoryEntry]:
    """Keep only the most recent conversation turns."""
    if max_length <= 0:
        return []
    return history[-max_length:]


def build_contextual_query(current_question: str, history: List[HistoryEntry]) -> str:
    """Build a lightweight contextual query from recent conversation turns."""
    cleaned_question = current_question.strip()
    if not history:
        return cleaned_question

    recent_history = trim_history(history, max_length=3)
    context_lines = ["Based on the previous conversation:"]

    for turn in recent_history:
        prior_question = str(turn.get("question", "")).strip()
        prior_answer = str(turn.get("answer", "")).strip()
        if prior_question:
            context_lines.append(f"Q: {prior_question}")
        if prior_answer:
            context_lines.append(f"A: {prior_answer}")

    context_lines.append(f"Now answer: {cleaned_question}")
    return "\n".join(context_lines)
