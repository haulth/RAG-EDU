"""FastAPI service and standalone chat UI for the RAG system."""

from __future__ import annotations

from pathlib import Path
import threading
import traceback
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app_settings import load_app_settings, save_app_settings
from runtime_service import (
    clear_processed_data_cache,
    PROJECT_ROOT,
    get_provider_defaults,
    get_runtime,
    get_runtime_state,
    get_shared_settings,
    release_runtime,
    run_query,
)
from ui_test_logging import UITestLogger


WEB_DIR = PROJECT_ROOT / "web"


class BootstrapRequest(BaseModel):
    force_reload: bool = False


class RechunkRequest(BaseModel):
    llm_provider: Optional[str] = None
    force_reload: bool = True


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    retrieval_mode: str = "dual"
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranking: bool = True
    use_mmr: bool = True
    use_hierarchical_expansion: bool = True
    use_evidence_selection: bool = True
    use_semantic_highlighting: bool = True
    llm_provider: str = "local"
    remote_model: Optional[str] = None
    ollama_base_url: Optional[str] = None
    groq_api_key: Optional[str] = None


class SettingsRequest(BaseModel):
    llm_provider: str = "local"
    retrieval_mode: str = "dual"
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranking: bool = True
    use_mmr: bool = True
    use_hierarchical_expansion: bool = True
    use_evidence_selection: bool = True
    use_semantic_highlighting: bool = True
    remote_model: Optional[str] = None
    ollama_base_url: Optional[str] = None
    groq_api_key: Optional[str] = None


def resolve_ollama_test_preset(model_name: Optional[str]) -> str:
    normalized = str(model_name or "").strip().lower()
    if normalized == "gemma4:e4b":
        return "ollama_gemma4_e4b"
    if normalized == "gemma4:31b":
        return "ollama_gemma4_31b"
    if normalized == "glm-4.7-flash:latest":
        return "ollama_glm_4_7_flash"
    return "ollama_20b"


def create_app() -> FastAPI:
    app = FastAPI(
        title="ChatBot Quy Che Dao Tao API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if WEB_DIR.exists():
        app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

    def _warm_runtime() -> None:
        try:
            get_runtime()
        except Exception:
            # Health endpoint will expose the bootstrap error from runtime state.
            pass

    @app.on_event("startup")
    def preload_runtime_on_startup() -> None:
        app.state.ui_test_logger = UITestLogger(PROJECT_ROOT)
        state = get_runtime_state()
        if state.get("loaded") or state.get("loading"):
            return
        threading.Thread(target=_warm_runtime, daemon=True).start()

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        index_path = WEB_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Chat UI not found.")
        return FileResponse(index_path)

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        state = get_runtime_state()
        state["status"] = "ok"
        state["providers"] = get_provider_defaults()
        ui_test_logger = getattr(app.state, "ui_test_logger", None)
        if ui_test_logger is not None:
            state["ui_test_session_id"] = ui_test_logger.session_id
            state["ui_test_log_dir"] = str(ui_test_logger.session_dir)
        return state

    @app.get("/api/providers")
    def providers() -> Dict[str, Any]:
        return get_provider_defaults()

    @app.get("/api/settings")
    def settings() -> Dict[str, Any]:
        return get_shared_settings()

    @app.post("/api/settings")
    def save_settings(payload: SettingsRequest) -> Dict[str, Any]:
        settings = load_app_settings(PROJECT_ROOT)
        provider_name = str(payload.llm_provider or "local").strip().lower()
        existing_chat_defaults = dict(settings.get("chat_defaults", {}) or {})

        settings["providers"]["default_provider"] = provider_name
        settings["chat_defaults"] = {
            **existing_chat_defaults,
            "llm_provider": provider_name,
            "retrieval_mode": payload.retrieval_mode,
            "top_k": int(payload.top_k),
            "use_reranking": bool(payload.use_reranking),
            "use_mmr": bool(payload.use_mmr),
            "use_hierarchical_expansion": bool(payload.use_hierarchical_expansion),
            "use_evidence_selection": bool(payload.use_evidence_selection),
            "use_semantic_highlighting": bool(payload.use_semantic_highlighting),
        }

        if provider_name == "groq":
            if payload.remote_model:
                settings["providers"]["groq"]["model"] = payload.remote_model.strip()
            if payload.groq_api_key and payload.groq_api_key.strip():
                settings["providers"]["groq"]["api_key"] = payload.groq_api_key.strip()
            settings["test_defaults"]["provider_preset"] = "groq_70b"
        elif provider_name == "ollama":
            if payload.remote_model:
                settings["providers"]["ollama"]["model"] = payload.remote_model.strip()
            if payload.ollama_base_url:
                settings["providers"]["ollama"]["base_url"] = payload.ollama_base_url.strip()
            settings["test_defaults"]["provider_preset"] = resolve_ollama_test_preset(
                settings["providers"]["ollama"].get("model")
            )
        else:
            settings["test_defaults"]["provider_preset"] = "local"

        save_app_settings(PROJECT_ROOT, settings)

        release_runtime()

        def _warm_selected_runtime() -> None:
            try:
                get_runtime(provider_name=provider_name, force_reload=True)
            except Exception:
                pass

        threading.Thread(target=_warm_selected_runtime, daemon=True).start()
        return get_shared_settings()

    @app.post("/api/bootstrap")
    def bootstrap(payload: Optional[BootstrapRequest] = None) -> JSONResponse:
        force_reload = bool(payload.force_reload) if payload else False
        try:
            get_runtime(force_reload=force_reload)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
        return JSONResponse(get_runtime_state())

    @app.post("/api/rechunk")
    def rechunk(payload: Optional[RechunkRequest] = None) -> JSONResponse:
        provider_name = str((payload.llm_provider if payload else "") or "").strip().lower() or None
        force_reload = bool(payload.force_reload) if payload else True

        state_before = get_runtime_state()
        resolved_provider = provider_name or state_before.get("provider") or None

        release_runtime()
        cache_reset = clear_processed_data_cache()

        try:
            get_runtime(force_reload=force_reload, provider_name=resolved_provider)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

        response_payload = get_runtime_state()
        response_payload["action"] = "rechunk"
        response_payload["cache_reset"] = cache_reset
        return JSONResponse(response_payload)

    @app.post("/api/reload-knowledge")
    def reload_knowledge_compat(payload: Optional[RechunkRequest] = None) -> JSONResponse:
        return rechunk(payload)

    @app.post("/api/chat")
    def chat(payload: ChatRequest) -> Dict[str, Any]:
        request_payload = payload.model_dump()
        request_payload["debug_trace"] = True
        shared_settings = get_shared_settings()
        ui_test_logger = getattr(app.state, "ui_test_logger", None)
        if ui_test_logger is None:
            ui_test_logger = UITestLogger(PROJECT_ROOT)
            app.state.ui_test_logger = ui_test_logger
        try:
            result = run_query(
                message=payload.message,
                retrieval_mode=payload.retrieval_mode,
                top_k=payload.top_k,
                use_reranking=payload.use_reranking,
                use_mmr=payload.use_mmr,
                use_hierarchical_expansion=payload.use_hierarchical_expansion,
                use_evidence_selection=payload.use_evidence_selection,
                use_semantic_highlighting=payload.use_semantic_highlighting,
                llm_provider=payload.llm_provider,
                remote_model=payload.remote_model,
                ollama_base_url=payload.ollama_base_url,
                groq_api_key=payload.groq_api_key,
                debug_trace=True,
            )
        except Exception as exc:
            if ui_test_logger is not None:
                try:
                    ui_test_logger.log_chat(
                        request_payload=request_payload,
                        response_payload={},
                        runtime_state=get_runtime_state(),
                        shared_settings=shared_settings,
                        error_message=f"{type(exc).__name__}: {exc}",
                        error_traceback=traceback.format_exc(),
                    )
                except Exception:
                    pass
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
        if ui_test_logger is not None:
            try:
                result["ui_test_log"] = ui_test_logger.log_chat(
                    request_payload=request_payload,
                    response_payload=result,
                    runtime_state=get_runtime_state(),
                    shared_settings=shared_settings,
                )
            except Exception as log_exc:
                result["ui_test_log"] = {
                    "error": f"{type(log_exc).__name__}: {log_exc}",
                }
        return result

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
