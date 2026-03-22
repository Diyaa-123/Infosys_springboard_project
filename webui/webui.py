from __future__ import annotations

import reflex as rx

from pages.chat import chat
from pages.history import history
from pages.index import index
from pages.upload import upload


app = rx.App(
    style={
        "font_family": "'IBM Plex Sans', 'Segoe UI', sans-serif",
        "background_color": "#f8fbff",
        "color": "#102a43",
    }
)

app.add_page(index, route="/", title="Knowledge Retrieval Studio")
app.add_page(upload, route="/upload", title="Upload Documents")
app.add_page(chat, route="/chat", title="Chat")
app.add_page(history, route="/history", title="History")
