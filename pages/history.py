from __future__ import annotations

import reflex as rx

from components.chat_message import chat_message
from components.navbar import navbar
from ui_state import AppState


def history() -> rx.Component:
    return rx.box(
        rx.container(
            navbar(),
            rx.vstack(
                rx.heading("Conversation History", size="7", color="#0f172a"),
                rx.text(
                    "Review the most recent chat turns captured by the existing conversational state layer.",
                    color="#486581",
                    font_size="1rem",
                ),
                rx.cond(
                    AppState.history.length() > 0,
                    rx.vstack(
                        rx.foreach(
                            AppState.history,
                            lambda turn: chat_message(
                                turn.question,
                                turn.answer,
                                turn.sources,
                            ),
                        ),
                        spacing="4",
                        width="100%",
                        align="stretch",
                    ),
                    rx.box(
                        rx.text(
                            "No history yet. Visit the chat page to start a conversation.",
                            color="#627d98",
                        ),
                        width="100%",
                        bg="white",
                        border="1px solid #d9e2ec",
                        border_radius="18px",
                        px="1rem",
                        py="1rem",
                    ),
                ),
                spacing="5",
                py="2rem",
                width="100%",
                align="stretch",
            ),
            max_width="1000px",
        ),
        min_height="100vh",
        bg="linear-gradient(180deg, #f9fbff 0%, #eef5ef 100%)",
        px="1rem",
    )
