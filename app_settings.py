"""Shared local app settings for UI, API, and terminal test flows."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse


DEFAULT_APP_SETTINGS: Dict[str, Any] = {
    "providers": {
        "default_provider": "local",
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": "",
            "model": "llama-3.3-70b-versatile",
            "timeout_seconds": 120,
        },
        "ollama": {
            "base_url": "http://127.0.0.1:11434",
            "model": "qwen2.5:7b-instruct",
            "timeout_seconds": 600,
        },
    },
    "chat_defaults": {
        "llm_provider": "local",
        "retrieval_mode": "dual",
        "top_k": 5,
        "use_reranking": True,
        "use_mmr": True,
        "use_hierarchical_expansion": True,
        "use_evidence_selection": True,
        "use_semantic_highlighting": True,
    },
    "test_defaults": {
        "provider_preset": "ollama_20b",
        "sample_size": 250,
        "seed": 42,
        "execution_path": "service",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged


def normalize_provider_name(provider: Optional[str]) -> str:
    value = str(provider or "local").strip().lower()
    aliases = {
        "local": "local",
        "current_local": "local",
        "local_current": "local",
        "groq": "groq",
        "groq_api": "groq",
        "api": "groq",
        "ollama": "ollama",
    }
    return aliases.get(value, "local")


def normalize_ollama_base_url(raw_value: Optional[str], default: str) -> str:
    value = str(raw_value or default or "").strip()
    if not value:
        value = "http://127.0.0.1:11434"

    if "://" not in value:
        value = f"http://{value}"

    parsed = urlparse(value)
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc or parsed.path
    path = parsed.path if parsed.netloc else ""

    if ":" not in netloc:
        netloc = f"{netloc}:11434"

    return f"{scheme}://{netloc}{path}".rstrip("/")


def get_app_settings_path(project_root: Path) -> Path:
    return project_root / "local_app_settings.json"


def get_legacy_settings_path(project_root: Path) -> Path:
    return project_root / "local_inference_settings.json"


def _migrate_legacy_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "providers" in payload or "chat_defaults" in payload or "test_defaults" in payload:
        return payload

    migrated = deepcopy(DEFAULT_APP_SETTINGS)
    migrated["providers"] = _deep_merge(migrated["providers"], payload if isinstance(payload, dict) else {})
    default_provider = normalize_provider_name(migrated["providers"].get("default_provider"))
    migrated["providers"]["default_provider"] = default_provider
    migrated["chat_defaults"]["llm_provider"] = default_provider
    if default_provider == "ollama":
        migrated["test_defaults"]["provider_preset"] = "ollama_20b"
    elif default_provider == "groq":
        migrated["test_defaults"]["provider_preset"] = "groq_70b"
    else:
        migrated["test_defaults"]["provider_preset"] = "local"
    return migrated


def _normalize_loaded_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _deep_merge(DEFAULT_APP_SETTINGS, settings if isinstance(settings, dict) else {})
    normalized["providers"]["default_provider"] = normalize_provider_name(
        normalized["providers"].get("default_provider")
    )
    normalized["chat_defaults"]["llm_provider"] = normalize_provider_name(
        normalized["chat_defaults"].get("llm_provider") or normalized["providers"].get("default_provider")
    )
    normalized["providers"]["ollama"]["base_url"] = normalize_ollama_base_url(
        normalized["providers"].get("ollama", {}).get("base_url"),
        DEFAULT_APP_SETTINGS["providers"]["ollama"]["base_url"],
    )
    normalized["chat_defaults"].pop("use_pdf_fallback", None)
    return normalized


def load_app_settings(project_root: Path) -> Dict[str, Any]:
    app_path = get_app_settings_path(project_root)
    legacy_path = get_legacy_settings_path(project_root)
    payload: Dict[str, Any] = {}

    if app_path.exists():
        try:
            payload = json.loads(app_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    elif legacy_path.exists():
        try:
            legacy_payload = json.loads(legacy_path.read_text(encoding="utf-8"))
        except Exception:
            legacy_payload = {}
        payload = _migrate_legacy_settings(legacy_payload if isinstance(legacy_payload, dict) else {})

    return _normalize_loaded_settings(payload)


def save_app_settings(project_root: Path, settings: Dict[str, Any]) -> Path:
    normalized = _normalize_loaded_settings(settings)
    path = get_app_settings_path(project_root)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def build_public_app_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_loaded_settings(settings)
    providers = normalized["providers"]
    chat_defaults = normalized["chat_defaults"]
    llm_provider = normalize_provider_name(chat_defaults.get("llm_provider") or providers.get("default_provider"))

    public = {
        "providers": {
            "default_provider": normalize_provider_name(providers.get("default_provider")),
            "groq": {
                "base_url": str(providers.get("groq", {}).get("base_url", "")),
                "model": str(providers.get("groq", {}).get("model", "")),
                "timeout_seconds": int(providers.get("groq", {}).get("timeout_seconds") or 120),
                "configured": bool(str(providers.get("groq", {}).get("api_key", "")).strip()),
            },
            "ollama": {
                "base_url": normalize_ollama_base_url(
                    providers.get("ollama", {}).get("base_url"),
                    DEFAULT_APP_SETTINGS["providers"]["ollama"]["base_url"],
                ),
                "model": str(providers.get("ollama", {}).get("model", "")),
                "timeout_seconds": int(providers.get("ollama", {}).get("timeout_seconds") or 600),
            },
        },
        "chat_defaults": {
            "llm_provider": llm_provider,
            "remote_model": (
                str(providers.get("groq", {}).get("model", ""))
                if llm_provider == "groq"
                else str(providers.get("ollama", {}).get("model", ""))
                if llm_provider == "ollama"
                else ""
            ),
            "ollama_base_url": normalize_ollama_base_url(
                providers.get("ollama", {}).get("base_url"),
                DEFAULT_APP_SETTINGS["providers"]["ollama"]["base_url"],
            ),
            "retrieval_mode": str(chat_defaults.get("retrieval_mode", "dual")),
            "top_k": int(chat_defaults.get("top_k") or 5),
            "use_reranking": bool(chat_defaults.get("use_reranking", True)),
            "use_mmr": bool(chat_defaults.get("use_mmr", True)),
            "use_hierarchical_expansion": bool(chat_defaults.get("use_hierarchical_expansion", True)),
            "use_evidence_selection": bool(chat_defaults.get("use_evidence_selection", True)),
            "use_semantic_highlighting": bool(chat_defaults.get("use_semantic_highlighting", True)),
        },
        "test_defaults": {
            "provider_preset": str(normalized.get("test_defaults", {}).get("provider_preset", "ollama_20b")),
            "sample_size": int(normalized.get("test_defaults", {}).get("sample_size") or 250),
            "seed": int(normalized.get("test_defaults", {}).get("seed") or 42),
            "execution_path": str(normalized.get("test_defaults", {}).get("execution_path", "service")),
        },
    }
    return public
