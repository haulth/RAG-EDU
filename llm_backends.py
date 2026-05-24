"""Inference backends for local runtime, Groq API, and Ollama."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Dict, Optional

import requests

from app_settings import (
    DEFAULT_APP_SETTINGS,
    get_app_settings_path,
    load_app_settings,
    normalize_ollama_base_url,
    normalize_provider_name,
)


DEFAULT_SETTINGS: Dict[str, Any] = DEFAULT_APP_SETTINGS["providers"]
OLLAMA_CONNECT_RETRY_ATTEMPTS = 3
OLLAMA_EMPTY_CONTENT_RETRY_ATTEMPTS = 3
OLLAMA_THINKING_EMPTY_CONTENT_RETRY_ATTEMPTS = 6
OLLAMA_RETRY_BACKOFF_SECONDS = 1.0


def _flatten_ollama_text(value: Any) -> str:
    if isinstance(value, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in value
        ).strip()
    return str(value or "").strip()


def _is_gpt_oss_model(model: str) -> bool:
    return "gpt-oss" in str(model or "").strip().lower()


def _is_gemma4_e4b_model(model: str) -> bool:
    return str(model or "").strip().lower() == "gemma4:e4b"


def _is_think_enabled(think_mode: Optional[Any]) -> bool:
    if think_mode is None:
        return False
    if isinstance(think_mode, bool):
        return bool(think_mode)
    normalized = str(think_mode or "").strip().lower()
    return normalized not in {"", "none", "off", "false", "0", "no"}


def _parse_think_mode_override() -> Optional[Any]:
    raw = str(os.environ.get("OLLAMA_THINK_MODE", "") or "").strip().lower()
    if not raw or raw == "auto":
        return None
    if raw in {"off", "false", "0", "none", "no"}:
        return False
    if raw in {"on", "true", "1", "yes"}:
        return True
    if raw in {"low", "medium", "high"}:
        return raw
    return raw


def _default_ollama_think_mode(model: str) -> Optional[Any]:
    override = _parse_think_mode_override()
    if override is not None:
        return override
    if _is_gpt_oss_model(model) or _is_gemma4_e4b_model(model):
        return "medium"
    return None


def _build_ollama_num_predict_schedule(max_new_tokens: int, *, think_mode: Optional[Any] = None) -> list[int]:
    base = max(1, int(max_new_tokens))
    schedule: list[int] = [base]
    think_enabled = _is_think_enabled(think_mode)
    attempts = OLLAMA_THINKING_EMPTY_CONTENT_RETRY_ATTEMPTS if think_enabled else OLLAMA_EMPTY_CONTENT_RETRY_ATTEMPTS
    hard_cap = max(4096, base * 8) if think_enabled else max(2048, base * 4)
    growth_step = 384 if think_enabled else 256
    for step in range(1, attempts):
        candidate = max(base * (step + 1), base + (growth_step * step))
        candidate = min(candidate, hard_cap)
        value = int(candidate)
        if value not in schedule:
            schedule.append(value)
    return schedule


def get_provider_settings_path(project_root: Path) -> Path:
    return get_app_settings_path(project_root)


def load_provider_settings(project_root: Path) -> Dict[str, Any]:
    return dict(load_app_settings(project_root).get("providers", DEFAULT_SETTINGS))


def build_messages(prompt: str, system_prompt: Optional[str] = None) -> list[Dict[str, str]]:
    messages: list[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def groq_generate(
    prompt: str,
    *,
    api_key: str,
    model: str,
    base_url: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    system_prompt: Optional[str] = None,
    timeout_seconds: int = 120,
) -> str:
    if not api_key:
        raise RuntimeError("Groq API key chưa được cấu hình.")

    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": build_messages(prompt, system_prompt),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": int(max_new_tokens),
            "stream": False,
        },
        timeout=(15, timeout_seconds),
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("Groq API không trả về choices.")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "").strip()


def ollama_generate(
    prompt: str,
    *,
    base_url: str,
    model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    system_prompt: Optional[str] = None,
    timeout_seconds: int = 600,
) -> str:
    base_url = base_url.rstrip("/")
    messages = build_messages(prompt, system_prompt)
    think_mode = _default_ollama_think_mode(model)
    last_reason = ""
    partial_content = ""
    think_enabled = _is_think_enabled(think_mode)
    num_predict_schedule = _build_ollama_num_predict_schedule(max_new_tokens, think_mode=think_mode)

    for schedule_index, num_predict in enumerate(num_predict_schedule):
        for attempt in range(1, OLLAMA_CONNECT_RETRY_ATTEMPTS + 1):
            try:
                payload: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_predict": int(num_predict),
                    },
                }
                if think_mode is not None:
                    payload["think"] = think_mode

                response = requests.post(
                    f"{base_url}/api/chat",
                    json=payload,
                    timeout=(15, timeout_seconds),
                )
                response.raise_for_status()
                body = response.json()
                message = body.get("message") or {}
                content = _flatten_ollama_text(message.get("content", ""))
                thinking = _flatten_ollama_text(message.get("thinking", ""))
                done_reason = str(body.get("done_reason") or "").strip().lower()

                if content:
                    if think_enabled and done_reason == "length" and schedule_index < (len(num_predict_schedule) - 1):
                        partial_content = content
                        last_reason = (
                            "partial content while thinking not finished "
                            f"(done_reason=length, num_predict={num_predict}, content_chars={len(content)})"
                        )
                        break
                    return content

                last_reason = (
                    f"empty content (done_reason={done_reason or 'unknown'}, "
                    f"thinking_chars={len(thinking)}, num_predict={num_predict})"
                )

                # Thinking-capable models such as gpt-oss may spend tokens on
                # reasoning first and only emit final answer content afterwards.
                if thinking and done_reason == "length":
                    break

                if thinking:
                    break

                raise RuntimeError(
                    "Ollama tra ve response rong. "
                    f"Chi tiet: {last_reason}"
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_reason = f"{type(exc).__name__}: {exc}"
                if attempt >= OLLAMA_CONNECT_RETRY_ATTEMPTS:
                    break
                time.sleep(OLLAMA_RETRY_BACKOFF_SECONDS * attempt)

    if partial_content:
        return partial_content

    raise RuntimeError(
        "Ollama khong tra ve final content hop le sau khi retry. "
        f"Chi tiet cuoi: {last_reason or 'khong ro nguyen nhan'}"
    )
