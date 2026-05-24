"""CLI entrypoints for the unified API-backed chat workflow."""

from __future__ import annotations

import runpy

import uvicorn


def _run_api_ui() -> None:
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)


def main_api() -> None:
    _run_api_ui()


def main_ui() -> None:
    _run_api_ui()


def main_gradio() -> None:
    print("Lenh cu 'chatbotedu-gradio' nay se khoi dong UI thong nhat qua FastAPI.")
    _run_api_ui()


def main_test() -> None:
    from run_random_faq_system_test import main

    raise SystemExit(main())


def main_download_models() -> None:
    runpy.run_module("download_models", run_name="__main__")
