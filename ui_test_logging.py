"""UI/API test logging helpers with per-startup sessions."""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app_settings import build_public_app_settings, load_app_settings
from run_random_faq_system_test import (
    append_pipeline_step_outputs,
    build_step_outputs,
    normalize_unit_score,
    safe_float,
)


def _trim_text(value: Any, limit: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


class UITestLogger:
    """Persist UI/API chat runs into per-startup log folders."""

    def __init__(self, project_root: Path, startup_time: Optional[datetime] = None):
        self.project_root = Path(project_root).resolve()
        self.startup_time = startup_time or datetime.now()
        self.session_id = f"ui_session_{self.startup_time.strftime('%Y%m%d_%H%M%S_%f')}"
        self.session_dir = self.project_root / "test_logs" / "ui_test" / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._counter = 0
        self._write_session_info(request_count=0)

    def _next_entry(self) -> Tuple[int, datetime, str]:
        with self._lock:
            self._counter += 1
            index = self._counter
        created_at = datetime.now()
        stem = f"ui_test_{index:04d}_{created_at.strftime('%Y%m%d_%H%M%S_%f')}"
        return index, created_at, stem

    def _write_session_info(self, request_count: int) -> None:
        settings = build_public_app_settings(load_app_settings(self.project_root))
        payload = {
            "log_type": "ui_test_session",
            "session_id": self.session_id,
            "startup_time": self.startup_time.isoformat(),
            "session_dir": str(self.session_dir),
            "request_count": int(request_count),
            "shared_settings": settings,
        }
        path = self.session_dir / "session_info.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def log_chat(
        self,
        *,
        request_payload: Dict[str, Any],
        response_payload: Optional[Dict[str, Any]] = None,
        runtime_state: Optional[Dict[str, Any]] = None,
        shared_settings: Optional[Dict[str, Any]] = None,
        error_message: str = "",
        error_traceback: str = "",
    ) -> Dict[str, str]:
        entry_index, created_at, stem = self._next_entry()
        normalized_request = self._normalize_request_payload(request_payload, shared_settings)
        normalized_result = self._normalize_response_payload(
            response_payload=response_payload,
            request_payload=normalized_request,
            error_message=error_message,
            error_traceback=error_traceback,
        )
        normalized_result["step_outputs"] = build_step_outputs(normalized_result)

        log_payload = {
            "log_type": "ui_test_entry",
            "session_id": self.session_id,
            "entry_index": entry_index,
            "created_at": created_at.isoformat(),
            "startup_time": self.startup_time.isoformat(),
            "runtime_state": dict(runtime_state or {}),
            "shared_settings": dict(shared_settings or {}),
            "request": normalized_request,
            "result": normalized_result,
        }

        json_path = self.session_dir / f"{stem}.json"
        md_path = self.session_dir / f"{stem}.md"
        json_path.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(self._build_markdown(log_payload), encoding="utf-8")
        self._write_session_info(request_count=entry_index)

        return {
            "session_id": self.session_id,
            "json_path": str(json_path),
            "md_path": str(md_path),
        }

    def _normalize_request_payload(self, request_payload: Dict[str, Any], shared_settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        request = dict(request_payload or {})
        chat_defaults = dict((shared_settings or {}).get("chat_defaults", {}) or {})
        providers = dict((shared_settings or {}).get("providers", {}) or {})
        ollama_settings = dict(providers.get("ollama", {}) or {})

        request["message"] = str(request.get("message", "") or "")
        request["retrieval_mode"] = str(request.get("retrieval_mode", chat_defaults.get("retrieval_mode", "dual")) or "dual")
        request["top_k"] = int(safe_float(request.get("top_k", chat_defaults.get("top_k", 5))) or 5)
        request["use_reranking"] = bool(request.get("use_reranking", chat_defaults.get("use_reranking", True)))
        request["use_mmr"] = bool(request.get("use_mmr", chat_defaults.get("use_mmr", True)))
        request["use_hierarchical_expansion"] = bool(request.get("use_hierarchical_expansion", chat_defaults.get("use_hierarchical_expansion", True)))
        request["use_evidence_selection"] = bool(request.get("use_evidence_selection", chat_defaults.get("use_evidence_selection", True)))
        request["use_semantic_highlighting"] = bool(request.get("use_semantic_highlighting", chat_defaults.get("use_semantic_highlighting", True)))
        request["llm_provider"] = str(request.get("llm_provider", chat_defaults.get("llm_provider", "local")) or "local")
        request["remote_model"] = str(request.get("remote_model", chat_defaults.get("remote_model", "")) or "")
        request["ollama_base_url"] = str(request.get("ollama_base_url", chat_defaults.get("ollama_base_url", ollama_settings.get("base_url", ""))) or "")
        request["debug_trace"] = bool(request.get("debug_trace", True))
        request["groq_api_key_present"] = bool(str(request.get("groq_api_key", "") or "").strip())
        request.pop("groq_api_key", None)
        return request

    def _normalize_response_payload(
        self,
        *,
        response_payload: Optional[Dict[str, Any]],
        request_payload: Dict[str, Any],
        error_message: str,
        error_traceback: str,
    ) -> Dict[str, Any]:
        payload = deepcopy(response_payload or {})
        actual_answer = str(payload.get("actual_answer", "") or payload.get("answer", "") or "")
        transport_success = bool(payload.get("transport_success", payload.get("success", not error_message)))

        normalized = {
            **payload,
            "request": dict(payload.get("request", {}) or request_payload or {}),
            "answer": str(payload.get("answer", "") or actual_answer),
            "actual_answer": actual_answer,
            "transport_success": transport_success and not bool(error_message),
            "success": bool(payload.get("success", transport_success and not bool(error_message))),
            "confidence": normalize_unit_score(payload.get("confidence", 0.0)),
            "groundedness_score": normalize_unit_score(payload.get("groundedness_score", 0.0)),
            "provenance_score": normalize_unit_score(payload.get("provenance_score", 0.0)),
            "quality_metrics": dict(payload.get("quality_metrics", {}) or {}),
            "citations": list(payload.get("citations", []) or payload.get("legal_references", []) or []),
            "legal_references": list(payload.get("legal_references", []) or []),
            "selected_evidence": list(payload.get("selected_evidence", []) or []),
            "evidence_spans": list(payload.get("evidence_spans", []) or []),
            "retrieved_chunks": list(payload.get("retrieved_chunks", []) or []),
            "query_plan": dict(payload.get("query_plan", {}) or {}),
            "query_router": dict(payload.get("query_router", {}) or {}),
            "llm_backend": dict(payload.get("llm_backend", {}) or {}),
            "retrieval_metrics": dict(payload.get("retrieval_metrics", {}) or {}),
            "response_type": str(payload.get("response_type", "rag") or "rag"),
            "revision_applied": bool(payload.get("revision_applied", False)),
            "total_time": safe_float(payload.get("total_time", 0.0)),
            "error_message": str(payload.get("error_message", "") or error_message or ""),
            "error_traceback": str(payload.get("error_traceback", "") or error_traceback or ""),
        }
        return normalized

    def _build_markdown(self, log_payload: Dict[str, Any]) -> str:
        request = dict(log_payload.get("request", {}) or {})
        result = dict(log_payload.get("result", {}) or {})
        runtime_state = dict(log_payload.get("runtime_state", {}) or {})
        shared_settings = dict(log_payload.get("shared_settings", {}) or {})
        quality_metrics = dict(result.get("quality_metrics", {}) or {})

        lines = [
            "# UI Test Log",
            "",
            "## Run Summary",
            "",
            f"- Session id: `{log_payload.get('session_id', '')}`",
            f"- Entry index: `{int(log_payload.get('entry_index', 0) or 0)}`",
            f"- Startup time: `{log_payload.get('startup_time', '')}`",
            f"- Created at: `{log_payload.get('created_at', '')}`",
            f"- Runtime loaded: `{'yes' if runtime_state.get('loaded') else 'no'}`",
            f"- Runtime loading: `{'yes' if runtime_state.get('loading') else 'no'}`",
            f"- Runtime loaded at: `{runtime_state.get('loaded_at', '')}`",
            f"- Runtime load seconds: `{safe_float(runtime_state.get('load_seconds', 0.0)):.2f}s`",
            f"- LLM provider: `{request.get('llm_provider', '')}`",
            f"- Remote model: `{request.get('remote_model', '')}`",
            f"- Ollama URL: `{request.get('ollama_base_url', '')}`",
            f"- Retrieval mode: `{request.get('retrieval_mode', '')}`",
            f"- Top-K: `{int(safe_float(request.get('top_k', 0)))}`",
            f"- Reranking: `{'on' if request.get('use_reranking', True) else 'off'}`",
            f"- MMR: `{'on' if request.get('use_mmr', True) else 'off'}`",
            f"- Hierarchical: `{'on' if request.get('use_hierarchical_expansion', True) else 'off'}`",
            f"- Evidence selection: `{'on' if request.get('use_evidence_selection', True) else 'off'}`",
            f"- Semantic highlighting: `{'on' if request.get('use_semantic_highlighting', True) else 'off'}`",
            f"- Transport success: `{'yes' if result.get('transport_success') else 'no'}`",
            f"- Response type: `{result.get('response_type', 'rag')}`",
            f"- Confidence: `{normalize_unit_score(result.get('confidence', 0.0)):.2%}`",
            f"- Groundedness: `{normalize_unit_score(result.get('groundedness_score', 0.0)):.2%}`",
            f"- Provenance: `{normalize_unit_score(result.get('provenance_score', 0.0)):.2%}`",
            f"- Faithfulness: `{normalize_unit_score(quality_metrics.get('faithfulness_score', 0.0)):.2%}`",
            f"- Answer relevance: `{normalize_unit_score(quality_metrics.get('answer_relevance_score', 0.0)):.2%}`",
            f"- Citation support: `{normalize_unit_score(quality_metrics.get('citation_support_score', 0.0)):.2%}`",
            f"- Total time: `{safe_float(result.get('total_time', 0.0)):.2f}s`",
            "",
            "## Question",
            "",
            request.get("message", "") or "_Empty query._",
            "",
            "## Actual Answer",
            "",
            result.get("actual_answer", "") or "_No answer returned._",
            "",
            "## Legal References",
            "",
        ]

        legal_references = list(result.get("legal_references", []) or [])
        if legal_references:
            for item in legal_references:
                lines.append(f"- {item}")
        else:
            lines.append("- _No legal references_")
        lines.extend([
            "",
            "## Query Router",
            "",
            "```json",
            json.dumps(result.get("query_router", {}) or {}, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Retrieval Plan",
            "",
            "```json",
            json.dumps(result.get("query_plan", {}) or {}, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Pipeline Step Outputs",
            "",
        ])
        append_pipeline_step_outputs(lines, result)

        lines.extend([
            "",
            "## Selected Evidence",
            "",
        ])
        selected_evidence = list(result.get("selected_evidence", []) or [])
        if selected_evidence:
            for item in selected_evidence[:5]:
                lines.append(f"- `{item.get('evidence_id', '')}` | `{item.get('citation', '')}`")
                lines.append(f"  Score: `{safe_float(item.get('selector_score', 0.0)):.2%}`")
                lines.append(f"  Span: _{_trim_text(item.get('text', ''))}_")
        else:
            lines.append("- _No selected evidence_")

        lines.extend([
            "",
            "## Evidence Spans",
            "",
        ])
        evidence_spans = list(result.get("evidence_spans", []) or [])
        if evidence_spans:
            for item in evidence_spans[:5]:
                lines.append(f"- Claim: {_trim_text(item.get('claim', ''), limit=260)}")
                lines.append(f"  Label: `{item.get('label', '')}`")
                lines.append(f"  Citation: `{item.get('citation', '')}`")
                lines.append(f"  Provenance: `{safe_float(item.get('provenance_score', 0.0)):.2%}`")
                lines.append(f"  Evidence: _{_trim_text(item.get('evidence', ''))}_")
        else:
            lines.append("- _No evidence spans_")

        retrieval_metrics = dict(result.get("retrieval_metrics", {}) or {})
        lines.extend([
            "",
            "## Retrieval Metrics",
            "",
        ])
        if retrieval_metrics:
            for key in (
                "retrieval_method",
                "hybrid_count",
                "hyde_count",
                "merged_count",
                "stage2_count",
                "hierarchical_anchored_results",
                "mmr_count",
                "final_count",
                "retrieval_time",
                "rerank_time",
                "hierarchical_time",
                "mmr_time",
                "total_time",
            ):
                if key in retrieval_metrics:
                    value = retrieval_metrics.get(key)
                    if key.endswith("_time") or key == "total_time":
                        lines.append(f"- {key}: `{safe_float(value):.3f}s`")
                    else:
                        lines.append(f"- {key}: `{value}`")
        else:
            lines.append("- _No retrieval metrics_")

        debug_trace = dict(retrieval_metrics.get("debug_trace", {}) or {})
        lines.extend([
            "",
            "## Retrieval Trace",
            "",
        ])
        trace_stage_order = [
            "initial_hybrid",
            "initial_hyde",
            "merged_candidates",
            "pre_rerank",
            "post_viranker",
            "post_hierarchical_expansion",
            "post_mmr",
            "final_retrieval_results",
        ]
        has_trace = False
        for stage_key in trace_stage_order:
            stage = dict(debug_trace.get(stage_key, {}) or {})
            items = list(stage.get("items", []) or [])
            if not stage and not items:
                continue
            has_trace = True
            lines.append(f"- {stage.get('label', stage_key)}: `{stage.get('count', len(items))}`")
            if stage.get("note"):
                lines.append(f"  Note: {stage.get('note', '')}")
            for item in items[:5]:
                lines.append(
                    f"  - `#{item.get('rank', '')}` | `{item.get('citation', '')}` | "
                    f"score `{safe_float(item.get('score', 0.0)):.4f}` | method `{item.get('method', '')}`"
                )
                preview = item.get("text_preview", "") or item.get("rerank_text_preview", "")
                if preview:
                    lines.append(f"    Text: _{_trim_text(preview)}_")
        if not has_trace:
            lines.append("- _No retrieval trace_")

        llm_input = dict(debug_trace.get("llm_input", {}) or {})
        llm_context = llm_input.get("context") or result.get("context_used", "") or ""
        lines.extend([
            "",
            "## LLM Input",
            "",
        ])
        if llm_input or llm_context:
            lines.append(f"- Query sent to LLM: {llm_input.get('query') or request.get('message', '')}")
            lines.append(f"- Selected evidence count: `{int(llm_input.get('selected_evidence_count', len(result.get('selected_evidence', []) or [])) or 0)}`")
            if llm_input.get("selected_evidence_ids"):
                lines.append(f"- Selected evidence ids: `{', '.join(str(item) for item in llm_input.get('selected_evidence_ids', []) if item)}`")
            if llm_input.get("selection_method"):
                lines.append(f"- Selection method: `{llm_input.get('selection_method', '')}`")
            lines.append(f"- Context length: `{len(llm_context)}`")
            prompt = llm_input.get("prompt", "") or ""
            if prompt:
                lines.append(f"- Prompt length: `{len(prompt)}`")
            lines.extend([
                "",
                "Context sent to LLM:",
                "",
                "```text",
                llm_context if llm_context else "_Empty context_",
                "```",
            ])
            if prompt:
                lines.extend([
                    "",
                    "Prompt sent to LLM:",
                    "",
                    "```text",
                    prompt,
                    "```",
                ])
        else:
            lines.append("- _No LLM input trace_")

        synthesis_trace = dict(debug_trace.get("synthesis", {}) or {})
        lines.extend([
            "",
            "## Synthesis Trace",
            "",
        ])
        if synthesis_trace:
            lines.extend([
                "```json",
                json.dumps(synthesis_trace, ensure_ascii=False, indent=2),
                "```",
            ])
        else:
            lines.append("- _No synthesis trace_")

        lines.extend([
            "",
            "## Shared Settings Snapshot",
            "",
            "```json",
            json.dumps(shared_settings, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Raw Response",
            "",
            "```json",
            json.dumps(result, ensure_ascii=False, indent=2),
            "```",
        ])
        return "\n".join(lines) + "\n"
