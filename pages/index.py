from __future__ import annotations

import reflex as rx

from components.navbar import navbar


def index() -> rx.Component:
    return rx.box(
        rx.container(
            navbar(),
            rx.flex(
                rx.vstack(
                    rx.badge(
                        "Industry-grade RAG workspace",
                        color_scheme="cyan",
                        variant="soft",
                        px="12px",
                        py="6px",
                        border_radius="999px",
                    ),
                    rx.heading(
                        "AI-Based Document Search and Knowledge Retrieval with Conversational Interface",
                        size="8",
                        color="#0f172a",
                        line_height="1.15",
                    ),
                    rx.text(
                        "Upload your documents, index them through the existing ingestion pipeline, and chat with a grounded assistant that answers using your knowledge base.",
                        font_size="1.1rem",
                        color="#334e68",
                        max_width="720px",
                        line_height="1.8",
                    ),
                    rx.hstack(
                        rx.link(
                            rx.button(
                                "Go to Chat",
                                size="4",
                                bg="#1f4b99",
                                color="white",
                                border_radius="14px",
                                _hover={"bg": "#173b7a"},
                            ),
                            href="/chat",
                        ),
                        rx.link(
                            rx.button(
                                "Upload Documents",
                                size="4",
                                variant="outline",
                                border_radius="14px",
                            ),
                            href="/upload",
                        ),
                        spacing="4",
                    ),
                    align="start",
                    spacing="6",
                    width="100%",
                ),
                min_height="72vh",
                align="center",
            ),
            max_width="1100px",
            py="1.5rem",
        ),
        min_height="100vh",
        bg="radial-gradient(circle at top left, #d9f3ff 0%, #f7fbff 45%, #eef6f5 100%)",
        px="1rem",
    )
