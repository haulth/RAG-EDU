"""Compatibility launcher for the unified API-backed chat UI.

This module used to boot a standalone Gradio runtime. The project now uses a
single supported interaction path:

    web UI -> api_server -> runtime_service.run_query

Keeping ``main.py`` as a thin launcher avoids breaking older commands such as
``python main.py`` while removing the duplicate Gradio-only semantics.
"""

from __future__ import annotations

import uvicorn


def main() -> None:
    print("=" * 70)
    print("KHOI DONG CHAT UI THONG NHAT")
    print("=" * 70)
    print("Gradio da duoc loai bo khoi duong chay chinh.")
    print("Dang khoi dong FastAPI + web chat UI tai http://localhost:8000")
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
