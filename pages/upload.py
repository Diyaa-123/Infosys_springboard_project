from __future__ import annotations

import reflex as rx

from components.navbar import navbar
from ui_state import AppState


UPLOAD_ID = "document_upload"


def upload() -> rx.Component:
    return rx.box(
        rx.container(
            navbar(),
            rx.vstack(
                rx.heading("Upload Documents", size="7", color="#0f172a"),
                rx.text(
                    "Save PDF and TXT files into the existing documents directory. After uploading, run the ingestion pipeline to index the new content.",
                    color="#486581",
                    font_size="1rem",
                    max_width="760px",
                    line_height="1.8",
                ),
                rx.box(
                    rx.upload(
                        rx.vstack(
                            rx.text(
                                "Drag and drop PDF or TXT files here",
                                font_weight="700",
                                color="#0f172a",
                            ),
                            rx.text(
                                "or click to choose files",
                                color="#486581",
                            ),
                            spacing="2",
                        ),
                        id=UPLOAD_ID,
                        multiple=True,
                        accept={
                            "application/pdf": [".pdf"],
                            "text/plain": [".txt"],
                        },
                        border="2px dashed #9fb3c8",
                        border_radius="22px",
                        padding="2.5rem",
                        width="100%",
                        bg="rgba(255, 255, 255, 0.75)",
                    ),
                    width="100%",
                ),
                rx.hstack(
                    rx.button(
                        "Save to Documents",
                        on_click=[
                            AppState.handle_upload(rx.upload_files(upload_id=UPLOAD_ID)),
                            rx.clear_selected_files(UPLOAD_ID),
                        ],
                        bg="#1f4b99",
                        color="white",
                        border_radius="14px",
                        _hover={"bg": "#173b7a"},
                    ),
                    rx.text(
                        AppState.upload_status,
                        color="#334e68",
                        font_size="0.95rem",
                    ),
                    width="100%",
                    align="center",
                    spacing="4",
                ),
                rx.cond(
                    rx.selected_files(UPLOAD_ID).length() > 0,
                    rx.box(
                        rx.text(
                            "Selected files",
                            font_weight="700",
                            color="#102a43",
                            mb="0.5rem",
                        ),
                        rx.vstack(
                            rx.foreach(
                                rx.selected_files(UPLOAD_ID),
                                lambda filename: rx.text(filename, color="#486581"),
                            ),
                            align="start",
                            spacing="2",
                        ),
                        width="100%",
                        bg="white",
                        border="1px solid #d9e2ec",
                        border_radius="18px",
                        px="1rem",
                        py="1rem",
                    ),
                    rx.box(),
                ),
                spacing="6",
                align="start",
                width="100%",
                py="2rem",
            ),
            max_width="980px",
        ),
        min_height="100vh",
        bg="linear-gradient(180deg, #f7fbff 0%, #edf7f4 100%)",
        px="1rem",
    )
