from __future__ import annotations

import reflex as rx

from components.chat_message import chat_message
from components.navbar import navbar
from ui_state import AppState


def chat() -> rx.Component:
    return rx.box(
        rx.container(
            navbar(),
            rx.vstack(
                rx.hstack(
                    rx.vstack(
                        rx.heading("Chat With Your Documents", size="7", color="#0f172a"),
                        rx.text(
                            "Ask grounded questions, inspect returned sources, and continue the conversation across follow-up turns.",
                            color="#486581",
                            font_size="1rem",
                        ),
                        align="start",
                        spacing="2",
                    ),
                    rx.badge(
                        AppState.chat_status,
                        color_scheme="cyan",
                        variant="soft",
                        border_radius="999px",
                        px="12px",
                        py="6px",
                    ),
                    justify="between",
                    width="100%",
                    align="center",
                ),
                rx.box(
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
                        rx.center(
                            rx.vstack(
                                rx.heading(
                                    "No conversation yet",
                                    size="5",
                                    color="#102a43",
                                ),
                                rx.text(
                                    "Start with a question about one of your indexed documents.",
                                    color="#627d98",
                                ),
                                spacing="2",
                            ),
                            min_height="320px",
                        ),
                    ),
                    width="100%",
                    bg="rgba(255, 255, 255, 0.72)",
                    border="1px solid #d9e2ec",
                    border_radius="24px",
                    px="1rem",
                    py="1rem",
                    box_shadow="0 24px 80px rgba(15, 23, 42, 0.08)",
                ),
                rx.hstack(
                    rx.input(
                        value=AppState.question,
                        on_change=AppState.set_question,
                        placeholder="Ask a question about your documents...",
                        size="3",
                        radius="large",
                        width="100%",
                        bg="white",
                    ),
                    rx.button(
                        rx.cond(AppState.is_processing, "Working...", "Send"),
                        on_click=AppState.submit_question,
                        bg="#1f4b99",
                        color="white",
                        border_radius="14px",
                        min_width="120px",
                        _hover={"bg": "#173b7a"},
                    ),
                    width="100%",
                    align="center",
                    spacing="4",
                ),
                spacing="6",
                py="2rem",
                width="100%",
                align="stretch",
            ),
            max_width="1000px",
        ),
        min_height="100vh",
        bg="radial-gradient(circle at top, #e6f2ff 0%, #f8fbff 35%, #eef5ef 100%)",
        px="1rem",
    )
