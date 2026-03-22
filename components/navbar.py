from __future__ import annotations

import reflex as rx


def navbar() -> rx.Component:
    """Shared navigation bar for the document assistant app."""
    nav_link_style = {
        "color": "#dce7ff",
        "font_weight": "600",
        "text_decoration": "none",
        "_hover": {"color": "#ffffff"},
    }

    return rx.box(
        rx.flex(
            rx.link(
                rx.hstack(
                    rx.box(
                        "AI",
                        bg="#4fd1c5",
                        color="#09203f",
                        font_weight="800",
                        border_radius="12px",
                        px="10px",
                        py="4px",
                    ),
                    rx.text(
                        "Knowledge Retrieval Studio",
                        font_size="1.05rem",
                        font_weight="700",
                        color="white",
                    ),
                    spacing="3",
                    align="center",
                ),
                href="/",
            ),
            rx.hstack(
                rx.link("Home", href="/", style=nav_link_style),
                rx.link("Upload", href="/upload", style=nav_link_style),
                rx.link("Chat", href="/chat", style=nav_link_style),
                rx.link("History", href="/history", style=nav_link_style),
                spacing="6",
                align="center",
            ),
            justify="between",
            align="center",
            width="100%",
        ),
        bg="linear-gradient(135deg, #09203f 0%, #1f4b99 55%, #4fd1c5 140%)",
        px=["1rem", "2rem"],
        py="1rem",
        border_radius="22px",
        box_shadow="0 18px 60px rgba(9, 32, 63, 0.22)",
        position="sticky",
        top="1rem",
        z_index="10",
        backdrop_filter="blur(10px)",
    )
