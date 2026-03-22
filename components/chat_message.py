from __future__ import annotations

import reflex as rx


def _source_badge(source: str) -> rx.Component:
    return rx.badge(
        source,
        color_scheme="cyan",
        variant="soft",
        border_radius="999px",
        px="10px",
        py="4px",
    )


def chat_message(question, answer, sources) -> rx.Component:
    """Reusable chat message card showing question, answer, and sources."""
    return rx.box(
        rx.vstack(
            rx.box(
                rx.text(
                    "You",
                    font_size="0.8rem",
                    font_weight="700",
                    text_transform="uppercase",
                    letter_spacing="0.08em",
                    color="#1f4b99",
                ),
                rx.text(
                    question,
                    font_size="1rem",
                    color="#0f172a",
                    line_height="1.7",
                ),
                bg="#e8f1ff",
                border="1px solid #c5d9ff",
                border_radius="18px",
                px="1rem",
                py="0.9rem",
                width="100%",
            ),
            rx.box(
                rx.text(
                    "Assistant",
                    font_size="0.8rem",
                    font_weight="700",
                    text_transform="uppercase",
                    letter_spacing="0.08em",
                    color="#0f766e",
                ),
                rx.text(
                    answer,
                    white_space="pre-wrap",
                    font_size="1rem",
                    color="#102a43",
                    line_height="1.8",
                ),
                rx.cond(
                    sources.length() > 0,
                    rx.vstack(
                        rx.text(
                            "Sources",
                            font_size="0.82rem",
                            font_weight="700",
                            color="#486581",
                        ),
                        rx.flex(
                            rx.foreach(sources, _source_badge),
                            wrap="wrap",
                            spacing="2",
                            width="100%",
                        ),
                        align="start",
                        spacing="2",
                        width="100%",
                    ),
                    rx.text(
                        "No sources returned.",
                        font_size="0.85rem",
                        color="#7b8794",
                    ),
                ),
                bg="white",
                border="1px solid #d9e2ec",
                border_radius="18px",
                px="1rem",
                py="1rem",
                width="100%",
                box_shadow="0 12px 35px rgba(15, 23, 42, 0.06)",
            ),
            spacing="3",
            width="100%",
            align="stretch",
        ),
        width="100%",
    )
