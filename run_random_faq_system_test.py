"""
Run representative end-to-end tests for the current RAG system using questions
from data/test FAQ.xlsx.

Features:
- Pure-stdlib XLSX reader (no pandas/openpyxl required)
- Representative random sampling by topic (default: 250 questions)
- Boots the current RAG runtime by executing CELL_3, CELL_4, CELL_5, CELL_6, CELL_8
- Saves both JSON and Markdown logs under test_logs/YYYY-MM-DD/
- Supports --dry-run to validate sampling and logging without loading models
# Usage:
- python run_random_faq_system_test.py --execution-path service --sample-size 10 --seed 42 --provider-preset groq_70b
- python run_random_faq_system_test.py --execution-path service --sample-size 10 --seed 42 --provider-preset ollama_20b
- python run_random_faq_system_test.py --execution-path service --sample-size 10 --seed 42 --provider-preset ollama_gemma4_31b
- python run_random_faq_system_test.py --execution-path service --sample-size 10 --seed 42 --provider-preset ollama_gemma4_e4b
- python run_random_faq_system_test.py --execution-path service --sample-size 10 --seed 42 --provider-preset ollama_glm_4_7_flash

"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import traceback
import unicodedata
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app_settings import build_public_app_settings, load_app_settings
from rag_quality_metrics import (
    combine_reference_answer_similarity,
    compute_benchmark_quality_metrics,
    compute_reference_text_metrics,
    empty_quality_metrics,
    empty_reference_metrics,
    has_annotation_data,
    legal_reference_similarity as shared_legal_reference_similarity,
)
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_QUESTION_FILE = PROJECT_ROOT / "data" / "Data250.xlsx"
DEFAULT_LOG_DIR = PROJECT_ROOT / "test_logs"
OPTIONAL_MODULE_CHOICES = ("hierarchical", "evidence", "reranking", "mmr", "semantic")


def load_cli_defaults() -> Dict[str, object]:
    shared_settings = build_public_app_settings(load_app_settings(PROJECT_ROOT))
    providers = shared_settings.get("providers", {})
    test_defaults = shared_settings.get("test_defaults", {})
    chat_defaults = shared_settings.get("chat_defaults", {})
    default_provider = str(chat_defaults.get("llm_provider") or providers.get("default_provider") or "local")
    return {
        "provider_preset": str(test_defaults.get("provider_preset") or "ollama_20b"),
        "sample_size": int(test_defaults.get("sample_size") or 250),
        "seed": int(test_defaults.get("seed") or 42),
        "execution_path": "service",
        "llm_provider": default_provider,
        "remote_model": str(chat_defaults.get("remote_model") or ""),
        "ollama_base_url": str(chat_defaults.get("ollama_base_url") or providers.get("ollama", {}).get("base_url") or ""),
        "groq_model": str(providers.get("groq", {}).get("model") or "llama-3.3-70b-versatile"),
        "ollama_model": str(providers.get("ollama", {}).get("model") or "qwen2.5:7b-instruct"),
        "use_reranking": bool(chat_defaults.get("use_reranking", True)),
        "use_mmr": bool(chat_defaults.get("use_mmr", True)),
        "use_hierarchical_expansion": bool(chat_defaults.get("use_hierarchical_expansion", True)),
        "use_evidence_selection": bool(chat_defaults.get("use_evidence_selection", True)),
        "use_semantic_highlighting": bool(chat_defaults.get("use_semantic_highlighting", True)),
    }


def build_provider_presets(defaults: Dict[str, object]) -> Dict[str, Dict[str, Optional[str]]]:
    return {
        "local": {
            "llm_provider": "local",
            "remote_model": None,
            "ollama_base_url": None,
        },
        "ollama_20b": {
            "llm_provider": "ollama",
            "remote_model": str(defaults.get("ollama_model") or ""),
            "ollama_base_url": str(defaults.get("ollama_base_url") or ""),
        },
        "ollama_gemma4_31b": {
            "llm_provider": "ollama",
            "remote_model": "gemma4:31b",
            "ollama_base_url": str(defaults.get("ollama_base_url") or ""),
        },
        "ollama_gemma4_e4b": {
            "llm_provider": "ollama",
            "remote_model": "gemma4:e4b",
            "ollama_base_url": str(defaults.get("ollama_base_url") or ""),
        },
        "ollama_glm_4_7_flash": {
            "llm_provider": "ollama",
            "remote_model": "glm-4.7-flash:latest",
            "ollama_base_url": str(defaults.get("ollama_base_url") or ""),
        },
        "groq_70b": {
            "llm_provider": "groq",
            "remote_model": str(defaults.get("groq_model") or ""),
            "ollama_base_url": None,
        },
    }


def build_default_disabled_modules(defaults: Dict[str, object]) -> List[str]:
    disabled = []
    if not bool(defaults.get("use_reranking", True)):
        disabled.append("reranking")
    if not bool(defaults.get("use_mmr", True)):
        disabled.append("mmr")
    if not bool(defaults.get("use_hierarchical_expansion", True)):
        disabled.append("hierarchical")
    if not bool(defaults.get("use_evidence_selection", True)):
        disabled.append("evidence")
    if not bool(defaults.get("use_semantic_highlighting", True)):
        disabled.append("semantic")
    return disabled


def normalize_optional_module_name(name: object) -> str:
    value = str(name or "").strip().lower()
    aliases = {
        "semantic-highlighting": "semantic",
        "semantic_highlighting": "semantic",
    }
    return aliases.get(value, value)

XML_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


@dataclass
class FAQCase:
    case_id: str
    sheet_name: str
    row_number: int
    topic: str
    question: str
    expected_answer: str
    expected_source: str
    expected_response_type: str = "rag"
    annotations: Dict = field(default_factory=dict)


EXPECTED_RESPONSE_TYPE_HEADERS = (
    "loại phản hồi mong đợi",
    "expected response type",
    "expected route",
    "response type",
)

TOPIC_HEADERS = (
    "chủ đề",
    "chủ đề trao đổi",
)

QUESTION_HEADERS = (
    "nội dung câu hỏi",
    "câu hỏi",
)

ANSWER_HEADERS = (
    "nội dung câu trả lời",
    "nội dung câu trả lời của llm",
    "câu trả lời",
)

SOURCE_HEADERS = (
    "nguồn (căn cứ pháp lý)",
    "nguồn",
)

LEGAL_REFERENCE_STOPWORDS = {
    "chuong",
    "dieu",
    "khoan",
    "diem",
    "trang",
    "muc",
    "phan",
    "quy",
    "che",
    "dao",
    "tao",
    "van",
    "ban",
    "hop",
    "nhat",
    "so",
    "nguon",
    "can",
    "cu",
    "phap",
    "ly",
}

ABSTENTION_MARKERS = (
    "khong co quy dinh",
    "khong co quy dinh truc tiep",
    "chua tim thay can cu",
    "khong tim thay can cu",
    "tai lieu dang nap chua du can cu",
    "lien he lai phong dao tao",
)


def parse_args() -> argparse.Namespace:
    defaults = load_cli_defaults()
    provider_presets = build_provider_presets(defaults)
    default_disabled_modules = build_default_disabled_modules(defaults)
    parser = argparse.ArgumentParser(description="Run representative FAQ tests for the current RAG system.")
    parser.add_argument("--questions-file", default=str(DEFAULT_QUESTION_FILE), help="Path to the FAQ Excel file.")
    parser.add_argument("--sample-size", type=int, default=int(defaults["sample_size"]), help="Number of representative random questions to test.")
    parser.add_argument("--seed", type=int, default=int(defaults["seed"]), help="Random seed for reproducible sampling.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory to store test logs.")
    parser.add_argument("--retrieval-mode", default="dual", choices=["dual", "hybrid", "hyde", "vector", "bm25"], help="RAG retrieval mode.")
    parser.add_argument("--top-k", type=int, default=5, help="Final top-k chunks.")
    parser.add_argument("--disable-reranking", action="store_true", help="Disable reranking during tests.")
    parser.add_argument("--disable-mmr", action="store_true", help="Disable MMR during tests.")
    parser.add_argument(
        "--disable-module",
        action="append",
        default=list(default_disabled_modules),
        choices=list(OPTIONAL_MODULE_CHOICES),
        help="Disable an optional module. Repeatable: hierarchical, evidence, reranking, mmr, semantic.",
    )
    parser.add_argument(
        "--enable-module",
        action="append",
        default=[],
        choices=list(OPTIONAL_MODULE_CHOICES),
        help="Enable an optional module even if it was disabled by defaults or other presets. Repeatable. This overrides disable-module for the same module.",
    )
    parser.add_argument(
        "--execution-path",
        default=str(defaults["execution_path"]),
        choices=["service", "pipeline"],
        help="Unified execution path for UI/API/tests. 'pipeline' is kept only as a deprecated alias and will be mapped to 'service'.",
    )
    parser.add_argument(
        "--provider-preset",
        default=str(defaults["provider_preset"]),
        choices=list(provider_presets.keys()),
        help="Preset nhanh cho backend LLM khi test hàng loạt.",
    )
    parser.add_argument("--llm-provider", default=str(defaults["llm_provider"]), choices=["local", "groq", "ollama"], help="LLM backend to use when execution-path=service.")
    parser.add_argument("--remote-model", default=str(defaults["remote_model"]), help="Remote model name for Groq/Ollama when execution-path=service.")
    parser.add_argument("--ollama-base-url", default=str(defaults["ollama_base_url"]), help="Ollama base URL when execution-path=service.")
    parser.add_argument("--groq-api-key", default=None, help="Groq API key override when execution-path=service.")
    parser.add_argument("--dry-run", action="store_true", help="Sample cases and write logs without loading the RAG runtime.")
    args = parser.parse_args()
    requested_execution_path = str(args.execution_path or "service")
    setattr(args, "requested_execution_path", requested_execution_path)
    if requested_execution_path != "service":
        print("[WARN] --execution-path pipeline da bi loai bo. Tu dong dung luong service de dong bo voi UI/API.")
        args.execution_path = "service"
    setattr(args, "_provider_presets", provider_presets)
    setattr(args, "_cli_defaults", defaults)
    return args


def resolve_module_switches(args: argparse.Namespace) -> Dict[str, object]:
    disabled_raw = {normalize_optional_module_name(item) for item in (getattr(args, "disable_module", []) or [])}
    enabled_raw = {normalize_optional_module_name(item) for item in (getattr(args, "enable_module", []) or [])}
    disabled = {item for item in disabled_raw if item}
    enabled = {item for item in enabled_raw if item}
    resolution_notes: List[str] = []

    if getattr(args, "disable_reranking", False):
        disabled.add("reranking")
        resolution_notes.append("legacy flag `--disable-reranking` requested")
    if getattr(args, "disable_mmr", False):
        disabled.add("mmr")
        resolution_notes.append("legacy flag `--disable-mmr` requested")

    requested_disabled = sorted(disabled)
    requested_enabled = sorted(enabled)
    conflicting_requested = sorted(disabled & enabled)
    if conflicting_requested:
        resolution_notes.append(
            "explicit enable overrides disable for: " + ", ".join(conflicting_requested)
        )

    disabled -= enabled

    use_reranking = bool("reranking" not in disabled)
    use_mmr = bool("mmr" not in disabled)
    use_hierarchical_expansion = bool("hierarchical" not in disabled)
    use_evidence_selection = bool("evidence" not in disabled)
    use_semantic_highlighting = bool("semantic" not in disabled)

    enabled_modules = []
    if use_reranking:
        enabled_modules.append("reranking")
    if use_mmr:
        enabled_modules.append("mmr")
    if use_hierarchical_expansion:
        enabled_modules.append("hierarchical")
    if use_evidence_selection:
        enabled_modules.append("evidence")
    if use_semantic_highlighting:
        enabled_modules.append("semantic")

    return {
        "use_reranking": use_reranking,
        "use_mmr": use_mmr,
        "use_hierarchical_expansion": use_hierarchical_expansion,
        "use_evidence_selection": use_evidence_selection,
        "use_semantic_highlighting": use_semantic_highlighting,
        "requested_disabled_modules": requested_disabled,
        "disabled_modules": sorted(disabled),
        "conflicting_requested_modules": conflicting_requested,
        "requested_enabled_modules": sorted(enabled),
        "enabled_modules": enabled_modules,
        "resolution_notes": resolution_notes,
    }


def apply_provider_preset(args: argparse.Namespace) -> argparse.Namespace:
    provider_presets = getattr(args, "_provider_presets", None) or build_provider_presets(load_cli_defaults())
    defaults = getattr(args, "_cli_defaults", None) or load_cli_defaults()
    preset = provider_presets.get(getattr(args, "provider_preset", ""), {})
    if not preset:
        return args

    if not getattr(args, "llm_provider", None) or args.llm_provider == defaults["llm_provider"]:
        args.llm_provider = preset.get("llm_provider", args.llm_provider)

    if not getattr(args, "remote_model", None) or args.remote_model == defaults["remote_model"]:
        args.remote_model = preset.get("remote_model", args.remote_model)

    if args.llm_provider == "ollama":
        if not getattr(args, "ollama_base_url", None) or args.ollama_base_url == defaults["ollama_base_url"]:
            args.ollama_base_url = preset.get("ollama_base_url", args.ollama_base_url)
    elif args.ollama_base_url == defaults["ollama_base_url"]:
        args.ollama_base_url = preset.get("ollama_base_url")

    return args


def _to_bool_text(value: bool) -> str:
    return "yes" if bool(value) else "no"


def _http_get_json(
    url: str,
    *,
    timeout_seconds: float,
    attempts: int = 1,
) -> Tuple[bool, Optional[Dict], str]:
    last_error = ""
    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
                body = response.read().decode("utf-8", errors="ignore")
            payload = json.loads(body or "{}")
            if isinstance(payload, dict):
                return True, payload, ""
            return True, {}, ""
        except Exception as exc:
            last_error = str(exc)
            if attempt < max(1, int(attempts)):
                time.sleep(min(1.0, 0.25 * attempt))
    return False, None, last_error


def probe_ollama_server(base_url: str, model_name: str = "", timeout_seconds: float = 4.0) -> Tuple[bool, str]:
    endpoint = str(base_url or "").strip().rstrip("/")
    if not endpoint:
        return False, "missing_ollama_base_url"

    version_ok, version_payload, version_error = _http_get_json(
        f"{endpoint}/api/version",
        timeout_seconds=max(2.5, min(timeout_seconds, 6.0)),
        attempts=2,
    )
    tags_ok, tags_payload, tags_error = _http_get_json(
        f"{endpoint}/api/tags",
        timeout_seconds=max(6.0, timeout_seconds),
        attempts=2,
    )

    if not version_ok and not tags_ok:
        detail = version_error or tags_error or "unknown"
        return False, f"probe_failed: {detail}"

    if not tags_ok:
        version_value = str((version_payload or {}).get("version") or "").strip()
        version_suffix = f"_{version_value}" if version_value else ""
        tag_error_text = (tags_error or "").strip().lower()
        if "timed out" in tag_error_text:
            return True, f"reachable_version_only_tags_timeout{version_suffix}"
        return True, f"reachable_version_only_tags_unavailable{version_suffix}"

    try:
        models = [
            str((item or {}).get("name") or "").strip()
            for item in list((tags_payload or {}).get("models", []) or [])
            if str((item or {}).get("name") or "").strip()
        ]

        requested = str(model_name or "").strip().lower()
        if not requested:
            return True, "reachable"

        aliases = {requested}
        if ":" not in requested:
            aliases.add(f"{requested}:latest")

        model_exists = any(str(model).strip().lower() in aliases for model in models)
        if model_exists:
            return True, "reachable_model_found"
        return True, "reachable_model_not_listed"
    except urllib.error.URLError as exc:
        return False, f"url_error: {exc}"
    except Exception as exc:
        return False, f"probe_failed: {exc}"


def probe_local_cuda() -> Tuple[bool, str]:
    try:
        import torch
    except Exception as exc:
        return False, f"torch_import_failed: {exc}"

    try:
        if torch.cuda.is_available():
            device_name = str(torch.cuda.get_device_name(0) or "cuda")
            return True, f"cuda_available: {device_name}"
        return False, "cuda_unavailable"
    except Exception as exc:
        return False, f"cuda_probe_failed: {exc}"


def auto_select_runtime_backend(args: argparse.Namespace) -> argparse.Namespace:
    defaults = getattr(args, "_cli_defaults", None) or load_cli_defaults()

    ollama_model = str(
        args.remote_model
        or defaults.get("ollama_model")
        or defaults.get("remote_model")
        or ""
    ).strip()
    ollama_base_url = str(
        args.ollama_base_url
        or defaults.get("ollama_base_url")
        or ""
    ).strip()

    ollama_ok, ollama_status = probe_ollama_server(ollama_base_url, ollama_model)
    if ollama_ok:
        args.llm_provider = "ollama"
        args.remote_model = ollama_model
        args.ollama_base_url = ollama_base_url
        print(
            "[BACKEND] provider=ollama "
            f"(reachable={_to_bool_text(ollama_ok)}, status={ollama_status}, "
            f"model={args.remote_model or 'default'}, base_url={args.ollama_base_url})"
        )
        if ollama_status == "reachable_model_not_listed":
            print(
                "[BACKEND][WARN] Ollama reachable but requested model not listed in /api/tags. "
                "Still trying Ollama as requested."
            )
        return args
    if "timed out" in str(ollama_status or "").lower():
        args.llm_provider = "ollama"
        args.remote_model = ollama_model
        args.ollama_base_url = ollama_base_url
        print(
            "[BACKEND][WARN] Ollama probe timed out but keeping provider=ollama "
            f"(status={ollama_status}, model={args.remote_model or 'default'}, base_url={args.ollama_base_url})"
        )
        return args

    cuda_ok, cuda_status = probe_local_cuda()
    args.llm_provider = "local"
    args.remote_model = ""
    print(
        "[BACKEND][WARN] Ollama unavailable -> fallback provider=local "
        f"(ollama_status={ollama_status})"
    )
    print(f"[BACKEND] local_cuda={_to_bool_text(cuda_ok)} ({cuda_status})")
    if not cuda_ok:
        print("[BACKEND][WARN] Local runtime will run on CPU, throughput may be slow.")

    return args


def build_dated_log_dir(base_log_dir: Path, run_time: datetime) -> Path:
    return base_log_dir / run_time.strftime("%Y-%m-%d")


def ensure_log_dir(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)


def annotation_sidecar_path(question_path: Path) -> Path:
    return question_path.with_suffix(".annotations.json")


def load_annotation_sidecar(question_path: Path) -> Dict[str, Dict]:
    sidecar_path = annotation_sidecar_path(question_path)
    if not sidecar_path.exists():
        return {}
    with open(sidecar_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected annotation sidecar format: {sidecar_path}")
    return {
        str(case_id): dict(annotation or {})
        for case_id, annotation in payload.items()
        if isinstance(annotation, dict)
    }


def col_to_index(col: str) -> int:
    value = 0
    for ch in col:
        if ch.isalpha():
            value = value * 26 + (ord(ch.upper()) - 64)
    return value - 1


def load_xlsx_cases(path: Path) -> List[FAQCase]:
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")

    annotation_map = load_annotation_sidecar(path)

    with zipfile.ZipFile(path) as workbook:
        shared_strings = load_shared_strings(workbook)
        sheet_map = load_sheet_map(workbook)

        cases: List[FAQCase] = []
        skipped_sheets: List[str] = []
        for sheet_name, sheet_xml_path in sheet_map:
            rows = read_sheet_rows(workbook, sheet_xml_path, shared_strings)
            if not rows:
                continue

            header = rows[0]
            header_map = {normalize_header(header[i]): i for i in range(len(header))}

            topic_idx = find_first_header_index(header_map, TOPIC_HEADERS)
            question_idx = find_first_header_index(header_map, QUESTION_HEADERS)
            answer_idx = find_first_header_index(header_map, ANSWER_HEADERS)
            source_idx = find_first_header_index(header_map, SOURCE_HEADERS)
            response_type_idx = find_first_header_index(header_map, EXPECTED_RESPONSE_TYPE_HEADERS)

            if None in {topic_idx, question_idx, answer_idx, source_idx}:
                skipped_sheets.append(sheet_name)
                continue

            for excel_row_number, row in enumerate(rows[1:], start=2):
                topic = get_row_value(row, topic_idx)
                question = get_row_value(row, question_idx)
                expected_answer = get_row_value(row, answer_idx)
                expected_source = get_row_value(row, source_idx)
                expected_response_type = normalize_expected_response_type(
                    get_row_value(row, response_type_idx) if response_type_idx is not None else ""
                )

                if not question.strip():
                    continue

                case_id = f"{sheet_name}-row-{excel_row_number}"
                cases.append(
                    FAQCase(
                        case_id=case_id,
                        sheet_name=sheet_name,
                        row_number=excel_row_number,
                        topic=topic.strip(),
                        question=question.strip(),
                        expected_answer=expected_answer.strip(),
                        expected_source=expected_source.strip(),
                        expected_response_type=expected_response_type,
                        annotations=dict(annotation_map.get(case_id, {}) or {}),
                    )
                )

    if not cases:
        if skipped_sheets:
            raise ValueError(
                f"No valid FAQ sheets found in '{path.name}'. Skipped sheets with unexpected headers: {', '.join(skipped_sheets)}"
            )
        raise ValueError(f"No test cases found in '{path.name}'.")

    if skipped_sheets:
        print(f"[WARN] Skipped sheets with unexpected headers: {', '.join(skipped_sheets)}")

    return cases


def load_shared_strings(workbook: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []

    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    shared_strings: List[str] = []
    for item in root.findall("a:si", XML_NS):
        text_parts = []
        for text_node in item.iterfind(".//a:t", XML_NS):
            text_parts.append(text_node.text or "")
        shared_strings.append("".join(text_parts))
    return shared_strings


def load_sheet_map(workbook: zipfile.ZipFile) -> List[Tuple[str, str]]:
    workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
    rel_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rel_root}

    sheet_map: List[Tuple[str, str]] = []
    sheets_node = workbook_root.find("a:sheets", XML_NS)
    if sheets_node is None:
        return sheet_map

    for sheet in sheets_node.findall("a:sheet", XML_NS):
        name = sheet.attrib.get("name", "Sheet")
        rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        if not rel_id or rel_id not in rel_map:
            continue
        target = rel_map[rel_id]
        if not target.startswith("worksheets/"):
            target = f"worksheets/{os.path.basename(target)}"
        sheet_map.append((name, f"xl/{target}"))
    return sheet_map


def read_sheet_rows(workbook: zipfile.ZipFile, sheet_xml_path: str, shared_strings: List[str]) -> List[List[str]]:
    root = ET.fromstring(workbook.read(sheet_xml_path))
    rows: List[List[str]] = []

    for row_node in root.findall(".//a:sheetData/a:row", XML_NS):
        cells: Dict[int, str] = {}
        for cell in row_node.findall("a:c", XML_NS):
            ref = cell.attrib.get("r", "")
            match = re.match(r"([A-Z]+)(\d+)", ref)
            if not match:
                continue
            col_idx = col_to_index(match.group(1))
            cells[col_idx] = extract_cell_value(cell, shared_strings)

        if cells:
            max_col = max(cells)
            row = [cells.get(i, "") for i in range(max_col + 1)]
        else:
            row = []
        rows.append(row)

    return rows


def extract_cell_value(cell: ET.Element, shared_strings: List[str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("a:v", XML_NS)

    if cell_type == "s" and value_node is not None:
        idx = int(value_node.text)
        return shared_strings[idx] if 0 <= idx < len(shared_strings) else ""

    if cell_type == "inlineStr":
        inline_node = cell.find("a:is", XML_NS)
        if inline_node is None:
            return ""
        return "".join((text_node.text or "") for text_node in inline_node.iterfind(".//a:t", XML_NS))

    return value_node.text if value_node is not None and value_node.text is not None else ""


def get_row_value(row: List[str], idx: int) -> str:
    return row[idx] if idx < len(row) else ""


def normalize_header(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def normalize_reference_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", str(text or "").strip().lower())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = normalized.replace("đ", "d")
    return re.sub(r"\s+", " ", normalized)


def find_first_header_index(header_map: Dict[str, int], candidates: Tuple[str, ...]) -> Optional[int]:
    return next((header_map.get(normalize_header(name)) for name in candidates if header_map.get(normalize_header(name)) is not None), None)


def probe_bertscore_runtime() -> Dict[str, object]:
    try:
        import bert_score  # noqa: F401

        return {
            "available": True,
            "error": "",
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def extract_source_aliases(text: str) -> set[str]:
    normalized = normalize_reference_text(Path(str(text or "")).stem.replace("_", " "))
    aliases: set[str] = set()
    numeric_tokens = re.findall(r"\b\d{3,4}\b", normalized)
    if not numeric_tokens:
        return aliases

    primary_number = numeric_tokens[-1]
    aliases.add(f"doc:{primary_number}")
    if "ctsv" in normalized:
        aliases.add(f"ctsv:{primary_number}")
    if re.search(r"\bqd\b", normalized) or "quyet dinh" in normalized:
        aliases.add(f"qd:{primary_number}")
    if "dhsp" in normalized:
        aliases.add(f"dhsp:{primary_number}")
    return aliases


def preferred_source_alias(aliases: set[str]) -> str:
    for prefix in ("ctsv:", "qd:", "dhsp:", "doc:"):
        for alias in sorted(aliases):
            if alias.startswith(prefix):
                return alias
    return sorted(aliases)[0] if aliases else ""


def classify_expected_source_coverage(expected_source: str, corpus_source_aliases: set[str]) -> Dict[str, object]:
    expected_aliases = extract_source_aliases(expected_source)
    matched_aliases = sorted(expected_aliases & set(corpus_source_aliases or set()))
    primary_alias = preferred_source_alias(expected_aliases)
    return {
        "expected_source_in_corpus": bool(matched_aliases),
        "expected_source_aliases": sorted(expected_aliases),
        "expected_source_primary_alias": primary_alias,
        "expected_source_matched_corpus_key": matched_aliases[0] if matched_aliases else "",
    }


def collect_corpus_source_aliases(documents_dir: Path) -> Dict[str, object]:
    document_files = sorted(path.name for path in Path(documents_dir).rglob("*.pdf"))
    aliases: set[str] = set()
    for filename in document_files:
        aliases.update(extract_source_aliases(filename))
    return {
        "document_files": document_files,
        "document_file_count": len(document_files),
        "corpus_source_aliases": sorted(aliases),
    }


def audit_case_source_coverage(cases: List[FAQCase], documents_dir: Path) -> Dict[str, object]:
    corpus_info = collect_corpus_source_aliases(documents_dir)
    corpus_aliases = set(corpus_info["corpus_source_aliases"])
    out_of_corpus_groups: Counter[str] = Counter()
    out_of_corpus_topics: Counter[str] = Counter()
    in_corpus_count = 0

    for case in cases:
        coverage = classify_expected_source_coverage(case.expected_source, corpus_aliases)
        if coverage["expected_source_in_corpus"]:
            in_corpus_count += 1
        else:
            out_of_corpus_groups[str(coverage["expected_source_primary_alias"] or "unknown")] += 1
            out_of_corpus_topics[str(case.topic or "unknown")] += 1

    total_cases = len(cases)
    out_of_corpus_count = max(0, total_cases - in_corpus_count)
    return {
        **corpus_info,
        "case_count": total_cases,
        "in_corpus_case_count": in_corpus_count,
        "out_of_corpus_case_count": out_of_corpus_count,
        "in_corpus_case_ratio": (in_corpus_count / total_cases) if total_cases else 0.0,
        "out_of_corpus_case_ratio": (out_of_corpus_count / total_cases) if total_cases else 0.0,
        "out_of_corpus_source_groups": dict(out_of_corpus_groups),
        "out_of_corpus_topic_counts": dict(out_of_corpus_topics),
    }


def normalize_expected_response_type(value: str) -> str:
    raw = re.sub(r"[\s\-]+", "_", normalize_reference_text(value))
    aliases = {
        "": "rag",
        "rag": "rag",
        "rag_direct": "rag",
        "rag_rewrite": "rag",
        "rag_duration_rule": "rag",
        "clarify": "clarify",
        "clarification": "clarify",
        "router_clarify": "clarify",
        "no_rag": "no_rag",
        "basic": "basic_intent",
        "basic_intent": "basic_intent",
    }
    return aliases.get(raw, "rag")


def response_type_bucket(value: str) -> str:
    raw = normalize_expected_response_type(value)
    if raw.startswith("rag"):
        return "rag"
    return raw


def sample_cases(cases: List[FAQCase], sample_size: int, seed: Optional[int]) -> List[FAQCase]:
    if sample_size <= 0:
        return []

    rng = random.Random(seed)
    grouped: Dict[str, List[FAQCase]] = defaultdict(list)
    for case in cases:
        grouped[case.topic or "Không rõ chủ đề"].append(case)

    topic_names = list(grouped.keys())
    rng.shuffle(topic_names)

    selected: List[FAQCase] = []
    used_ids = set()

    for topic in topic_names:
        chosen = rng.choice(grouped[topic])
        selected.append(chosen)
        used_ids.add(chosen.case_id)
        if len(selected) >= sample_size:
            return selected

    if len(selected) < sample_size:
        remaining = [case for case in cases if case.case_id not in used_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: sample_size - len(selected)])

    return selected


def run_case(case: FAQCase, args: argparse.Namespace) -> Dict:
    module_switches = resolve_module_switches(args)
    from runtime_service import run_query

    start = time.time()
    payload = run_query(
        message=case.question,
        retrieval_mode=args.retrieval_mode,
        top_k=args.top_k,
        use_reranking=bool(module_switches["use_reranking"]),
        use_mmr=bool(module_switches["use_mmr"]),
        use_hierarchical_expansion=bool(module_switches["use_hierarchical_expansion"]),
        use_evidence_selection=bool(module_switches["use_evidence_selection"]),
        use_semantic_highlighting=bool(module_switches["use_semantic_highlighting"]),
        llm_provider=args.llm_provider,
        remote_model=args.remote_model,
        ollama_base_url=args.ollama_base_url,
        groq_api_key=args.groq_api_key,
        debug_trace=True,
    )
    elapsed = time.time() - start
    result_payload = {
        "transport_success": bool(payload.get("success", False)),
        "error_message": payload.get("error_message"),
        "actual_answer": payload.get("answer", ""),
        "citations": list(payload.get("citations", []) or []),
        "confidence": normalize_unit_score(payload.get("confidence", 0.0)),
        "groundedness_score": normalize_unit_score(payload.get("groundedness_score", 0.0)),
        "provenance_score": normalize_unit_score(payload.get("provenance_score", 0.0)),
        "revision_applied": bool(payload.get("revision_applied", False)),
        "retrieval_time": safe_float(payload.get("retrieval_time", 0.0)),
        "synthesis_time": safe_float(payload.get("synthesis_time", 0.0)),
        "total_time": safe_float(payload.get("total_time", elapsed)),
        "retrieval_metrics": payload.get("retrieval_metrics", {}) or {},
        "retrieved_chunks": list(payload.get("retrieved_chunks", []) or []),
        "selected_evidence": list(payload.get("selected_evidence", []) or []),
        "evidence_spans": list(payload.get("evidence_spans", []) or []),
        "claim_analyses": list(payload.get("claim_analyses", []) or []),
        "quality_metrics": normalize_quality_metrics(
            payload.get("quality_metrics", {}) or {},
            groundedness_score=payload.get("groundedness_score", 0.0),
            provenance_score=payload.get("provenance_score", 0.0),
        ),
        "legal_references": list(payload.get("legal_references", []) or []),
        "query_router": payload.get("query_router", {}) or {},
        "query_plan": payload.get("query_plan", {}) or {},
        "llm_backend": payload.get("llm_backend", {}) or {},
        "request": payload.get("request", {}) or {},
        "context_used": payload.get("context_used", "") or "",
        "response_type": payload.get("response_type", "rag"),
    }

    result_payload["request"] = {
        **(result_payload.get("request", {}) or {}),
        "use_reranking": bool(module_switches["use_reranking"]),
        "use_mmr": bool(module_switches["use_mmr"]),
        "use_hierarchical_expansion": bool(module_switches["use_hierarchical_expansion"]),
        "use_evidence_selection": bool(module_switches["use_evidence_selection"]),
        "use_semantic_highlighting": bool(module_switches["use_semantic_highlighting"]),
        "module_resolution": {
            "requested_disabled_modules": list(module_switches["requested_disabled_modules"]),
            "requested_enabled_modules": list(module_switches["requested_enabled_modules"]),
            "conflicting_requested_modules": list(module_switches["conflicting_requested_modules"]),
            "disabled_modules": list(module_switches["disabled_modules"]),
            "enabled_modules": list(module_switches["enabled_modules"]),
            "resolution_notes": list(module_switches["resolution_notes"]),
        },
    }

    answer_overlap = lexical_overlap(case.expected_answer, result_payload.get("actual_answer", ""))
    reference_metrics = compute_reference_text_metrics(
        case.expected_answer,
        result_payload.get("actual_answer", ""),
        include_bertscore=True,
    )
    reference_answer_score = combine_reference_answer_similarity(
        answer_overlap,
        reference_metrics,
    )
    source_overlap_score = source_overlap(case.expected_source, list(result_payload.get("citations", []) or []))
    legacy_answer_correctness = score_answer_correctness(
        case,
        result_payload,
        expected_answer_overlap=answer_overlap,
        reference_answer_score=reference_answer_score,
        expected_source_overlap=source_overlap_score,
    )
    benchmark_result = score_benchmark_composite(
        case,
        result_payload,
        expected_answer_overlap=answer_overlap,
        expected_source_overlap=source_overlap_score,
        top_k=args.top_k,
    )

    payload = {
        "case_id": case.case_id,
        "sheet_name": case.sheet_name,
        "row_number": case.row_number,
        "topic": case.topic,
        "question": case.question,
        "expected_answer": case.expected_answer,
        "expected_source": case.expected_source,
        "expected_response_type": case.expected_response_type,
        "annotations": dict(case.annotations or {}),
        **result_payload,
        "success": bool(benchmark_result["is_pass"]),
        "benchmark_pass": bool(benchmark_result["is_pass"]),
        "benchmark_composite_score": benchmark_result["score"],
        "benchmark_veto_reason": benchmark_result["veto_reason"],
        "benchmark_reason": benchmark_result["reason"],
        "benchmark_metrics": benchmark_result.get("metrics", {}) or {},
        "annotation_coverage_state": (benchmark_result.get("metrics", {}) or {}).get("annotation_coverage_state", "proxy_only"),
        "answer_correct": bool(legacy_answer_correctness["is_correct"]),
        "answer_correctness": legacy_answer_correctness["score"],
        "response_type_match": bool(legacy_answer_correctness["response_type_match"]),
        "evaluation_reason": benchmark_result["reason"],
        "legacy_evaluation_reason": legacy_answer_correctness["reason"],
        "expected_answer_overlap": answer_overlap,
        "reference_answer_score": reference_answer_score,
        "reference_metrics": reference_metrics,
        "expected_source_overlap": source_overlap_score,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    payload["step_outputs"] = build_step_outputs(payload)

    return payload


def build_runtime_exception_result(case: FAQCase, exc: Exception) -> Dict:
    annotation_coverage_state = "annotation_augmented" if has_annotation_data(case.annotations) else "proxy_only"
    return {
        "case_id": case.case_id,
        "sheet_name": case.sheet_name,
        "row_number": case.row_number,
        "topic": case.topic,
        "question": case.question,
        "expected_answer": case.expected_answer,
        "expected_source": case.expected_source,
        "expected_response_type": case.expected_response_type,
        "annotations": dict(case.annotations or {}),
        "transport_success": False,
        "success": False,
        "benchmark_pass": False,
        "benchmark_composite_score": 0.0,
        "benchmark_veto_reason": "runtime_exception",
        "benchmark_reason": "runtime_exception",
        "benchmark_metrics": {
            "annotation_coverage_state": annotation_coverage_state,
            "retrieval_dimension_score": 0.0,
            "citation_dimension_score": 0.0,
            "benchmark_composite_score": 0.0,
        },
        "annotation_coverage_state": annotation_coverage_state,
        "answer_correct": False,
        "answer_correctness": 0.0,
        "response_type_match": False,
        "evaluation_reason": "runtime_exception",
        "legacy_evaluation_reason": "runtime_exception",
        "error_message": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        "error_traceback": traceback.format_exc(),
        "actual_answer": "",
        "citations": [],
        "confidence": 0.0,
        "groundedness_score": 0.0,
        "provenance_score": 0.0,
        "quality_metrics": empty_quality_metrics(),
        "reference_answer_score": 0.0,
        "reference_metrics": empty_reference_metrics(),
        "revision_applied": False,
        "retrieval_time": 0.0,
        "synthesis_time": 0.0,
        "total_time": 0.0,
        "retrieval_metrics": {},
        "retrieved_chunks": [],
        "selected_evidence": [],
        "evidence_spans": [],
        "claim_analyses": [],
        "expected_answer_overlap": 0.0,
        "expected_source_overlap": 0.0,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "step_outputs": {},
    }


def run_case_safely(case: FAQCase, args: argparse.Namespace) -> Dict:
    try:
        return run_case(case, args)
    except Exception as exc:
        return build_runtime_exception_result(case, exc)


def is_llm_no_response_result(result: Dict) -> bool:
    if not bool(result.get("transport_success", False)):
        return True
    actual_answer = str(result.get("actual_answer", "") or "").strip()
    return not bool(actual_answer)


def print_case_progress(tag: str, index: int, total: int, case: FAQCase) -> None:
    progress_pct = ((index / total) * 100.0) if total else 100.0
    topic = (case.topic or "Khong ro chu de").strip()
    print(f"[{tag}] [{index}/{total} | {progress_pct:6.2f}%] {case.case_id} | {topic}")


def print_case_status(result: Dict) -> None:
    transport_status = "ok" if bool(result.get("transport_success")) else "fail"
    benchmark_status = "pass" if bool(result.get("benchmark_pass")) else "fail"
    no_response_status = "yes" if is_llm_no_response_result(result) else "no"
    total_time = safe_float(result.get("total_time", 0.0))
    print(
        "    -> "
        f"transport={transport_status}, "
        f"benchmark={benchmark_status}, "
        f"llm_no_response={no_response_status}, "
        f"total={total_time:.2f}s"
    )
    error_message = str(result.get("error_message", "") or "").strip()
    if error_message:
        print(f"       error={error_message}")


def lexical_overlap(left: str, right: str) -> float:
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens))


def source_overlap(expected_source: str, citations: List[str]) -> float:
    if not expected_source.strip():
        return 0.0
    return max(
        (shared_legal_reference_similarity(expected_source, item) for item in citations if str(item or "").strip()),
        default=0.0,
    )


def tokenize(text: str) -> set:
    return {tok for tok in re.findall(r"\w+", (text or "").lower()) if len(tok) > 2}


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def normalize_unit_score(value) -> float:
    score = safe_float(value)
    if score <= 0.0:
        return 0.0
    if score <= 1.0:
        return score
    if score <= 100.0:
        return max(0.0, min(score / 100.0, 1.0))
    return 1.0


def normalize_quality_metrics(raw_metrics: Dict, *, groundedness_score: float = 0.0, provenance_score: float = 0.0) -> Dict:
    metrics = dict(empty_quality_metrics())
    raw_metrics = dict(raw_metrics or {})
    metrics.update(raw_metrics)
    for key in (
        "faithfulness_score",
        "answer_relevance_score",
        "citation_support_score",
        "claim_citation_coverage",
        "entailed_claim_citation_coverage",
        "entailed_claim_provenance_score",
        "hallucination_rate",
        "contradiction_rate",
        "generic_rate",
        "off_topic_rate",
    ):
        metrics[key] = normalize_unit_score(metrics.get(key, 0.0))
    if "faithfulness_score" not in raw_metrics:
        metrics["faithfulness_score"] = normalize_unit_score(groundedness_score)
    if "citation_support_score" not in raw_metrics:
        metrics["citation_support_score"] = normalize_unit_score(provenance_score)
    for key in (
        "entailed_claims",
        "hallucinated_claims",
        "generic_claims",
        "off_topic_claims",
        "contradictory_claims",
    ):
        metrics[key] = int(safe_float(metrics.get(key, 0.0)))
    return metrics


STEP_OUTPUT_STAGE_SPECS = [
    ("hybrid", "initial_hybrid", "Hybrid retrieval"),
    ("hyde", "initial_hyde", "HyDE retrieval"),
    ("merge_dedup", "merged_candidates", "Merge and deduplicate"),
    ("rerank_stage2", "post_viranker", "ViRanker rerank"),
    ("hierarchical", "post_hierarchical_expansion", "Hierarchical expansion"),
    ("mmr", "post_mmr", "MMR"),
    ("final_retrieval", "final_retrieval_results", "Final retrieval results"),
]


def normalize_log_text(text: object) -> str:
    return str(text or "")


def first_nonempty_text(*values: object) -> str:
    for value in values:
        text = normalize_log_text(value)
        if text.strip():
            return text
    return ""


def unique_nonempty_strings(items: List[object]) -> List[str]:
    unique_items: List[str] = []
    seen = set()
    for item in list(items or []):
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique_items.append(text)
    return unique_items


def summarize_trace_items(items: List[Dict]) -> List[Dict]:
    summarized = []
    for item in list(items or []):
        metadata = dict(item.get("metadata", {}) or {})
        summarized.append({
            "rank": int(safe_float(item.get("rank", 0))),
            "chunk_id": str(item.get("chunk_id", "") or ""),
            "citation": str(item.get("citation", "") or ""),
            "score": safe_float(item.get("score", 0.0)),
            "method": str(item.get("method", "") or ""),
            "article": str(metadata.get("article", "") or ""),
            "hierarchical_path": str(metadata.get("hierarchical_path", "") or ""),
            "text": first_nonempty_text(
                item.get("text"),
                item.get("text_preview"),
                item.get("rerank_text_preview"),
                item.get("contextualized_text_preview"),
                item.get("raw_text_preview"),
            ),
        })
    return summarized


def summarize_selected_evidence(items: List[Dict]) -> List[Dict]:
    summarized = []
    for item in list(items or []):
        summarized.append({
            "evidence_id": str(item.get("evidence_id", "") or ""),
            "citation": str(item.get("citation", "") or ""),
            "selector_score": safe_float(item.get("selector_score", 0.0)),
            "text": first_nonempty_text(
                item.get("text"),
                item.get("text_preview"),
                item.get("llm_text"),
                item.get("llm_text_preview"),
            ),
            "hierarchical_path": str(item.get("hierarchical_path", "") or (item.get("metadata", {}) or {}).get("hierarchical_path", "") or ""),
        })
    return summarized


def summarize_evidence_spans(items: List[Dict]) -> List[Dict]:
    summarized = []
    for item in list(items or []):
        summarized.append({
            "claim": first_nonempty_text(item.get("claim")),
            "label": str(item.get("label", "") or ""),
            "citation": str(item.get("citation", "") or ""),
            "provenance_score": safe_float(item.get("provenance_score", 0.0)),
            "evidence": first_nonempty_text(item.get("evidence"), item.get("evidence_preview")),
        })
    return summarized


def build_trace_stage_output(debug_trace: Dict, stage_key: str, default_label: str) -> Dict:
    stage = dict(debug_trace.get(stage_key, {}) or {})
    items = list(stage.get("items", []) or [])
    if not stage and not items:
        return {}
    return {
        "label": str(stage.get("label", default_label) or default_label),
        "count": int(safe_float(stage.get("count", len(items)))),
        "note": str(stage.get("note", "") or ""),
        "items": summarize_trace_items(items),
    }


def build_evidence_step_output(result: Dict, synthesis_trace: Dict) -> Dict:
    selection_stage = dict((synthesis_trace or {}).get("evidence_selection", {}) or {})
    grounding_stage = dict((synthesis_trace or {}).get("evidence_grounding", {}) or {})
    selected_evidence = list(result.get("selected_evidence", []) or [])
    evidence_spans = list(result.get("evidence_spans", []) or [])
    return {
        "label": "Evidence",
        "module_enabled": bool(selection_stage.get("module_enabled", bool(selected_evidence))),
        "query_type": str(selection_stage.get("query_type", "") or ""),
        "candidate_count": int(safe_float(selection_stage.get("candidate_count", 0))),
        "selected_count": int(safe_float(selection_stage.get("selected_count", len(selected_evidence)))),
        "selection_method": str(selection_stage.get("selection_method", "") or ""),
        "max_selector_score": safe_float(selection_stage.get("max_selector_score", 0.0)),
        "structural_evidence_count": int(safe_float(selection_stage.get("structural_evidence_count", 0))),
        "selected_evidence_ids": unique_nonempty_strings(selection_stage.get("selected_evidence_ids", []) or []),
        "items": summarize_selected_evidence(selection_stage.get("items", []) or selected_evidence),
        "grounded_span_count": int(safe_float(grounding_stage.get("span_count", len(evidence_spans)))),
        "grounded_spans": summarize_evidence_spans(grounding_stage.get("items", []) or evidence_spans),
    }


def build_context_pruning_step_output(synthesis_trace: Dict) -> Dict:
    pruning_stage = dict((synthesis_trace or {}).get("context_pruning", {}) or {})
    evidence_stage = dict((synthesis_trace or {}).get("evidence_selection", {}) or {})
    if not pruning_stage and not evidence_stage:
        return {}
    return {
        "label": "Context pruning and semantic highlighting",
        "method": str(pruning_stage.get("method", "") or "unknown"),
        "context_source": str(pruning_stage.get("context_source", "") or ""),
        "context_length": int(safe_float(pruning_stage.get("context_length", 0))),
        "sentences_before": int(safe_float(pruning_stage.get("sentences_before", 0))),
        "sentences_after": int(safe_float(pruning_stage.get("sentences_after", 0))),
        "reduction_ratio": safe_float(pruning_stage.get("reduction_ratio", 0.0)),
        "highlight_threshold": safe_float(pruning_stage.get("highlight_threshold", 0.0)),
        "max_sentences_per_result": int(safe_float(pruning_stage.get("max_sentences_per_result", 0))),
        "builtin_blocks": int(safe_float(pruning_stage.get("builtin_blocks", 0))),
        "fallback_blocks": int(safe_float(pruning_stage.get("fallback_blocks", 0))),
        "unsupported_language_blocks": int(safe_float(pruning_stage.get("unsupported_language_blocks", 0))),
        "semantic_highlighted_sources": int(safe_float(evidence_stage.get("semantic_highlighted_sources", 0))),
        "items": list(pruning_stage.get("items", []) or []),
        "semantic_trace_items": list(pruning_stage.get("semantic_trace_items", []) or evidence_stage.get("semantic_trace_items", []) or []),
    }


def build_verification_step_output(synthesis_trace: Dict) -> Dict:
    verification_stage = dict((synthesis_trace or {}).get("verification", {}) or {})
    if not verification_stage:
        return {}
    claim_counts = dict(verification_stage.get("claim_counts", {}) or {})
    return {
        "label": "Verification",
        "ran": bool(verification_stage.get("ran", False)),
        "groundedness_score": safe_float(verification_stage.get("groundedness_score", 0.0)),
        "provenance_score": safe_float(verification_stage.get("provenance_score", 0.0)),
        "revision_applied": bool(verification_stage.get("revision_applied", False)),
        "answer_mode": str(verification_stage.get("answer_mode", "") or ""),
        "claim_counts": {
            "entailment": int(safe_float(claim_counts.get("entailment", 0))),
            "hallucination": int(safe_float(claim_counts.get("hallucination", 0))),
            "generic": int(safe_float(claim_counts.get("generic", 0))),
            "off_topic": int(safe_float(claim_counts.get("off_topic", 0))),
            "contradiction": int(safe_float(claim_counts.get("contradiction", 0))),
        },
    }


def build_citation_extraction_step_output(synthesis_trace: Dict, result: Dict) -> Dict:
    citation_stage = dict((synthesis_trace or {}).get("citation_extraction", {}) or {})
    citations = unique_nonempty_strings(citation_stage.get("citations", []) or result.get("citations", []) or [])
    if not citation_stage and not citations:
        return {}
    return {
        "label": "Citation extraction",
        "citation_count": int(safe_float(citation_stage.get("citation_count", len(citations)))),
    }


def build_step_outputs(result: Dict) -> Dict:
    retrieval_metrics = dict(result.get("retrieval_metrics", {}) or {})
    debug_trace = dict(retrieval_metrics.get("debug_trace", {}) or {})
    synthesis_trace = dict(debug_trace.get("synthesis", {}) or {})

    outputs = {}
    for stage_name, stage_key, default_label in STEP_OUTPUT_STAGE_SPECS:
        stage_output = build_trace_stage_output(debug_trace, stage_key, default_label)
        if stage_output:
            outputs[stage_name] = stage_output

    outputs["evidence"] = build_evidence_step_output(result, synthesis_trace)
    context_pruning_output = build_context_pruning_step_output(synthesis_trace)
    if context_pruning_output:
        outputs["context_pruning"] = context_pruning_output
    verification_output = build_verification_step_output(synthesis_trace)
    if verification_output:
        outputs["verification"] = verification_output

    llm_answer_stage = dict((synthesis_trace or {}).get("llm_answer", {}) or {})
    outputs["llm_answer"] = {
        "label": "LLM answer draft",
        "answer_length": int(safe_float(llm_answer_stage.get("answer_length", len(result.get("actual_answer", "") or "")))),
    }

    citation_extraction_output = build_citation_extraction_step_output(synthesis_trace, result)
    if citation_extraction_output:
        outputs["citation_extraction"] = citation_extraction_output

    confidence_stage = dict((synthesis_trace or {}).get("confidence_scoring", {}) or {})
    quality_metrics = dict(result.get("quality_metrics", {}) or {})
    citations = unique_nonempty_strings(result.get("citations", []) or [])
    outputs["citation_confidence"] = {
        "label": "Citation and confidence",
        "citation_count": len(citations),
        "confidence": normalize_unit_score(confidence_stage.get("confidence", result.get("confidence", 0.0))),
        "groundedness_score": normalize_unit_score(confidence_stage.get("groundedness_score", result.get("groundedness_score", 0.0))),
        "provenance_score": normalize_unit_score(confidence_stage.get("provenance_score", result.get("provenance_score", 0.0))),
        "faithfulness_score": normalize_unit_score(confidence_stage.get("faithfulness_score", quality_metrics.get("faithfulness_score", 0.0))),
        "citation_support_score": normalize_unit_score(confidence_stage.get("citation_support_score", quality_metrics.get("citation_support_score", 0.0))),
        "answer_relevance_score": normalize_unit_score(confidence_stage.get("answer_relevance_score", quality_metrics.get("answer_relevance_score", 0.0))),
        "hallucination_rate": normalize_unit_score(confidence_stage.get("hallucination_rate", quality_metrics.get("hallucination_rate", 0.0))),
        "contradiction_rate": normalize_unit_score(confidence_stage.get("contradiction_rate", quality_metrics.get("contradiction_rate", 0.0))),
    }

    final_stage = dict((synthesis_trace or {}).get("final_answer", {}) or {})
    outputs["final"] = {
        "label": "Final answer",
        "answer_mode": str(final_stage.get("answer_mode", "") or ("abstain" if answer_contains_abstention(result.get("actual_answer", "")) else "direct")),
        "answer_length": int(safe_float(final_stage.get("answer_length", len(result.get("actual_answer", "") or "")))),
        "revision_applied": bool(final_stage.get("revision_applied", result.get("revision_applied", False))),
        "citation_count": len(citations),
        "response_type": str(result.get("response_type", "rag") or "rag"),
    }

    return outputs


def normalize_eval_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", (text or "").strip().lower())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def answer_contains_abstention(answer: str) -> bool:
    answer_norm = normalize_eval_text(answer)
    if not answer_norm:
        return False
    return any(marker in answer_norm for marker in ABSTENTION_MARKERS)


def count_claim_labels(claims: List[Dict], labels: set[str]) -> int:
    return sum(
        1
        for item in list(claims or [])
        if str(item.get("label", "") or "").strip().lower() in labels
    )


def parse_legal_reference_signature(text: str) -> Dict[str, object]:
    normalized = normalize_reference_text(text)
    document_ref_match = re.search(r"\b(\d{3,4}\s*/\s*[a-z0-9-]+(?:\s*/\s*[a-z0-9-]+)*)\b", normalized)
    chapter_match = re.search(r"\bchuong\s+([ivxlcdm0-9]+)\b", normalized)
    article_match = re.search(r"\bdieu\s+(\d+)\b", normalized)
    section_match = re.search(r"\bkhoan\s+(\d+)\b", normalized)
    point_match = re.search(r"\bdiem\s+([a-z])\b", normalized)

    keyword_tokens = {
        token
        for token in re.findall(r"\w+", normalized)
        if len(token) > 2 and token not in LEGAL_REFERENCE_STOPWORDS
    }

    return {
        "document_ref": document_ref_match.group(1).replace(" ", "") if document_ref_match else "",
        "chapter": chapter_match.group(1).upper() if chapter_match else "",
        "article": article_match.group(1) if article_match else "",
        "section": section_match.group(1) if section_match else "",
        "point": point_match.group(1) if point_match else "",
        "keywords": keyword_tokens,
    }


def legal_reference_similarity(expected_source: str, citation: str) -> float:
    return shared_legal_reference_similarity(expected_source, citation)


def score_answer_correctness(
    case: FAQCase,
    result_payload: Dict,
    *,
    expected_answer_overlap: float,
    reference_answer_score: float,
    expected_source_overlap: float,
) -> Dict[str, object]:
    actual_bucket = response_type_bucket(result_payload.get("response_type", "rag"))
    expected_bucket = response_type_bucket(case.expected_response_type)
    transport_success = bool(result_payload.get("transport_success", False))
    response_type_match = actual_bucket == expected_bucket

    if not transport_success:
        return {
            "score": 0.0,
            "is_correct": False,
            "response_type_match": response_type_match,
            "reason": "transport_failed",
        }

    if not response_type_match:
        return {
            "score": 0.0,
            "is_correct": False,
            "response_type_match": False,
            "reason": "response_type_mismatch",
        }

    if expected_bucket == "clarify":
        return {
            "score": 1.0,
            "is_correct": True,
            "response_type_match": True,
            "reason": "expected_clarify",
        }

    has_expected_answer = bool(case.expected_answer.strip())
    has_expected_source = bool(case.expected_source.strip())

    if has_expected_answer and has_expected_source:
        score = 0.35 * reference_answer_score + 0.65 * expected_source_overlap
        threshold = 0.40
    elif has_expected_source:
        score = expected_source_overlap
        threshold = 0.35
    elif has_expected_answer:
        score = reference_answer_score
        threshold = 0.30
    else:
        score = 1.0
        threshold = 1.0

    actual_answer = str(result_payload.get("actual_answer", "") or "").strip()
    groundedness_score = normalize_unit_score(result_payload.get("groundedness_score", 0.0))
    claim_analyses = list(result_payload.get("claim_analyses", []) or [])
    hallucinated_claims = count_claim_labels(claim_analyses, {"hallucination", "contradiction"})

    if (
        answer_contains_abstention(actual_answer)
        and has_expected_source
        and expected_source_overlap >= 0.60
    ):
        return {
            "score": min(score, 0.15),
            "is_correct": False,
            "response_type_match": True,
            "reason": "unsupported_abstention",
        }

    if hallucinated_claims > 0 and groundedness_score < 0.35:
        return {
            "score": min(score, 0.20),
            "is_correct": False,
            "response_type_match": True,
            "reason": "groundedness_failed",
        }

    return {
        "score": score,
        "is_correct": score >= threshold,
        "response_type_match": True,
        "reason": "score_threshold",
    }


def score_benchmark_composite(
    case: FAQCase,
    result_payload: Dict,
    *,
    expected_answer_overlap: float,
    expected_source_overlap: float,
    top_k: int,
) -> Dict[str, object]:
    actual_bucket = response_type_bucket(result_payload.get("response_type", "rag"))
    expected_bucket = response_type_bucket(case.expected_response_type)
    transport_success = bool(result_payload.get("transport_success", False))
    response_type_match = actual_bucket == expected_bucket
    annotation_coverage_state = "annotation_augmented" if has_annotation_data(case.annotations) else "proxy_only"

    if not transport_success:
        return {
            "score": 0.0,
            "is_pass": False,
            "reason": "transport_failed",
            "veto_reason": "transport_failed",
            "metrics": {
                "annotation_coverage_state": annotation_coverage_state,
                "retrieval_dimension_score": 0.0,
                "citation_dimension_score": 0.0,
                "benchmark_composite_score": 0.0,
            },
        }

    if not response_type_match:
        return {
            "score": 0.0,
            "is_pass": False,
            "reason": "response_type_mismatch",
            "veto_reason": "response_type_mismatch",
            "metrics": {
                "annotation_coverage_state": annotation_coverage_state,
                "retrieval_dimension_score": 0.0,
                "citation_dimension_score": 0.0,
                "benchmark_composite_score": 0.0,
            },
        }

    if expected_bucket == "clarify":
        return {
            "score": 1.0,
            "is_pass": True,
            "reason": "expected_clarify",
            "veto_reason": "",
            "metrics": {
                "annotation_coverage_state": annotation_coverage_state,
                "citation_dimension_score": expected_source_overlap,
                "retrieval_dimension_score": 1.0,
                "benchmark_composite_score": 1.0,
            },
        }

    quality_metrics = dict(result_payload.get("quality_metrics", {}) or {})
    benchmark_metrics = compute_benchmark_quality_metrics(
        expected_source=case.expected_source,
        citations=result_payload.get("citations", []) or [],
        retrieved_chunks=result_payload.get("retrieved_chunks", []) or [],
        runtime_quality_metrics=quality_metrics,
        annotation=case.annotations,
        top_k=top_k,
    )

    contradiction_rate = normalize_unit_score(quality_metrics.get("contradiction_rate", 0.0))
    citation_dimension_score = normalize_unit_score(benchmark_metrics.get("citation_dimension_score", expected_source_overlap))
    score = normalize_unit_score(benchmark_metrics.get("benchmark_composite_score", 0.0))
    actual_answer = str(result_payload.get("actual_answer", "") or "").strip()

    veto_reason = ""
    if contradiction_rate > 0.0:
        veto_reason = "contradiction_veto"
    elif case.expected_source.strip() and citation_dimension_score < 0.35:
        veto_reason = "citation_miss_veto"
    elif answer_contains_abstention(actual_answer) and citation_dimension_score >= 0.60:
        veto_reason = "unsupported_abstention_veto"

    return {
        "score": score,
        "is_pass": score >= 0.55 and not veto_reason,
        "reason": veto_reason or "benchmark_composite_threshold",
        "veto_reason": veto_reason,
        "metrics": benchmark_metrics,
    }


def build_run_summary(
    cases: List[FAQCase],
    results: List[Dict],
    args: argparse.Namespace,
    mode: str,
    error: Optional[str] = None,
    error_traceback: Optional[str] = None,
    all_cases: Optional[List[FAQCase]] = None,
) -> Dict:
    module_switches = resolve_module_switches(args)
    question_bank_cases = list(all_cases or cases)
    question_bank_audit = audit_case_source_coverage(question_bank_cases, PROJECT_ROOT / "documents")
    selected_case_audit = audit_case_source_coverage(cases, PROJECT_ROOT / "documents")
    bertscore_probe = probe_bertscore_runtime()
    annotation_sidecar = annotation_sidecar_path(Path(args.questions_file).resolve())
    bertscore_available_count = sum(
        1 for item in results if (item.get("reference_metrics", {}) or {}).get("bertscore_available")
    )
    initial_no_response_count = sum(1 for item in results if item.get("initial_no_response"))
    retry_attempted_count = sum(1 for item in results if item.get("retry_attempted"))
    retry_resolved_count = sum(
        1 for item in results if item.get("retry_attempted") and not is_llm_no_response_result(item)
    )
    retry_unresolved_count = sum(
        1 for item in results if item.get("retry_attempted") and is_llm_no_response_result(item)
    )
    final_no_response_count = sum(1 for item in results if is_llm_no_response_result(item))
    return {
        "run_mode": mode,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "execution_path": args.execution_path,
        "requested_execution_path": getattr(args, "requested_execution_path", args.execution_path),
        "provider_preset": getattr(args, "provider_preset", ""),
        "retrieval_mode": args.retrieval_mode,
        "top_k": args.top_k,
        "use_reranking": bool(module_switches["use_reranking"]),
        "use_mmr": bool(module_switches["use_mmr"]),
        "use_hierarchical_expansion": bool(module_switches["use_hierarchical_expansion"]),
        "use_evidence_selection": bool(module_switches["use_evidence_selection"]),
        "use_semantic_highlighting": bool(module_switches["use_semantic_highlighting"]),
        "requested_disabled_modules": list(module_switches["requested_disabled_modules"]),
        "disabled_modules": list(module_switches["disabled_modules"]),
        "conflicting_requested_modules": list(module_switches["conflicting_requested_modules"]),
        "requested_enabled_modules": list(module_switches["requested_enabled_modules"]),
        "enabled_modules": list(module_switches["enabled_modules"]),
        "module_resolution_notes": list(module_switches["resolution_notes"]),
        "llm_provider": args.llm_provider,
        "remote_model": getattr(args, "remote_model", ""),
        "ollama_base_url": getattr(args, "ollama_base_url", ""),
        "question_file": str(Path(args.questions_file).resolve()),
        "annotation_sidecar": str(annotation_sidecar),
        "annotation_sidecar_exists": annotation_sidecar.exists(),
        "document_file_count": int(question_bank_audit["document_file_count"]),
        "document_files": list(question_bank_audit["document_files"]),
        "corpus_source_aliases": list(question_bank_audit["corpus_source_aliases"]),
        "question_bank_case_count": int(question_bank_audit["case_count"]),
        "question_bank_in_corpus_case_count": int(question_bank_audit["in_corpus_case_count"]),
        "question_bank_out_of_corpus_case_count": int(question_bank_audit["out_of_corpus_case_count"]),
        "question_bank_in_corpus_case_ratio": safe_float(question_bank_audit["in_corpus_case_ratio"]),
        "question_bank_out_of_corpus_case_ratio": safe_float(question_bank_audit["out_of_corpus_case_ratio"]),
        "question_bank_out_of_corpus_source_groups": dict(question_bank_audit["out_of_corpus_source_groups"]),
        "question_bank_out_of_corpus_topic_counts": dict(question_bank_audit["out_of_corpus_topic_counts"]),
        "selected_in_corpus_case_count": int(selected_case_audit["in_corpus_case_count"]),
        "selected_out_of_corpus_case_count": int(selected_case_audit["out_of_corpus_case_count"]),
        "selected_topics": [case.topic for case in cases],
        "selected_case_ids": [case.case_id for case in cases],
        "result_count": len(results),
        "avg_answer_correctness": average(results, "answer_correctness"),
        "avg_benchmark_composite": average(results, "benchmark_composite_score"),
        "avg_confidence": average(results, "confidence"),
        "avg_groundedness": average(results, "groundedness_score"),
        "avg_provenance": average(results, "provenance_score"),
        "avg_faithfulness": average_quality_metric(results, "faithfulness_score"),
        "avg_answer_relevance": average_quality_metric(results, "answer_relevance_score"),
        "avg_citation_support": average_quality_metric(results, "citation_support_score"),
        "avg_hallucination_rate": average_quality_metric(results, "hallucination_rate"),
        "avg_contradiction_rate": average_quality_metric(results, "contradiction_rate"),
        "avg_reference_answer_score": average(results, "reference_answer_score"),
        "avg_bleu_1": average_reference_metric(results, "bleu_1"),
        "avg_bleu_4": average_reference_metric(results, "bleu_4"),
        "avg_rouge_1": average_reference_metric(results, "rouge_1"),
        "avg_rouge_2": average_reference_metric(results, "rouge_2"),
        "avg_rouge_l": average_reference_metric(results, "rouge_l"),
        "avg_bertscore_f1": average_reference_metric(results, "bertscore_f1", availability_key="bertscore_available"),
        "avg_total_time": average(results, "total_time"),
        "transport_success_count": sum(1 for item in results if item.get("transport_success")),
        "transport_failure_count": sum(1 for item in results if not item.get("transport_success")),
        "initial_no_response_count": initial_no_response_count,
        "retry_attempted_count": retry_attempted_count,
        "retry_resolved_count": retry_resolved_count,
        "retry_unresolved_count": retry_unresolved_count,
        "final_no_response_count": final_no_response_count,
        "answer_correct_count": sum(1 for item in results if item.get("answer_correct")),
        "answer_incorrect_count": sum(1 for item in results if not item.get("answer_correct")),
        "benchmark_pass_count": sum(1 for item in results if item.get("benchmark_pass")),
        "benchmark_fail_count": sum(1 for item in results if not item.get("benchmark_pass")),
        "success_count": sum(1 for item in results if item.get("success")),
        "failure_count": sum(1 for item in results if not item.get("success")),
        "bertscore_runtime_available": bool(bertscore_probe["available"]),
        "bertscore_runtime_error": str(bertscore_probe["error"] or ""),
        "bertscore_available_count": bertscore_available_count,
        "bertscore_unavailable_count": max(0, len(results) - bertscore_available_count),
        "annotation_augmented_count": sum(1 for item in results if item.get("annotation_coverage_state") == "annotation_augmented"),
        "proxy_only_count": sum(1 for item in results if item.get("annotation_coverage_state") != "annotation_augmented"),
        "error": error,
        "error_traceback": error_traceback,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def average(items: List[Dict], key: str) -> float:
    values = [safe_float(item.get(key, 0.0)) for item in items]
    return sum(values) / len(values) if values else 0.0


def average_quality_metric(items: List[Dict], key: str) -> float:
    values = [
        safe_float((item.get("quality_metrics", {}) or {}).get(key, 0.0))
        for item in items
    ]
    return sum(values) / len(values) if values else 0.0


def average_reference_metric(items: List[Dict], key: str, *, availability_key: Optional[str] = None) -> Optional[float]:
    values = [
        safe_float((item.get("reference_metrics", {}) or {}).get(key, 0.0))
        for item in items
        if not availability_key or bool((item.get("reference_metrics", {}) or {}).get(availability_key))
    ]
    return (sum(values) / len(values)) if values else None


def format_optional_percentage(value) -> str:
    if value is None:
        return "N/A"
    return f"{safe_float(value):.2%}"


def compact_log_payload(value: object) -> Optional[object]:
    if value is None:
        return None
    if isinstance(value, dict):
        compacted = {}
        for key, item in value.items():
            compacted_item = compact_log_payload(item)
            if compacted_item is None:
                continue
            compacted[str(key)] = compacted_item
        return compacted or None
    if isinstance(value, list):
        compacted_list = []
        for item in value:
            compacted_item = compact_log_payload(item)
            if compacted_item is None:
                continue
            compacted_list.append(compacted_item)
        return compacted_list or None
    if isinstance(value, str):
        return value if value.strip() else None
    return value


def build_stage_group(step_outputs: Dict[str, Dict], stage_keys: List[str]) -> Dict[str, Dict]:
    return {
        stage_key: dict(step_outputs.get(stage_key, {}) or {})
        for stage_key in stage_keys
        if dict(step_outputs.get(stage_key, {}) or {})
    }


def build_request_config_log(result: Dict) -> Dict:
    return dict(compact_log_payload(dict(result.get("request", {}) or {})) or {})


def build_retrieval_flow_log(result: Dict) -> Dict:
    step_outputs = dict(result.get("step_outputs", {}) or {})
    retrieval_metrics = dict(result.get("retrieval_metrics", {}) or {})
    retrieval_metrics.pop("debug_trace", None)
    payload = {
        "query_plan": dict(result.get("query_plan", {}) or {}),
        "stages": build_stage_group(step_outputs, [
            "hybrid",
            "hyde",
            "merge_dedup",
            "rerank_stage2",
            "hierarchical",
            "mmr",
            "final_retrieval",
        ]),
        "metrics": retrieval_metrics,
    }
    return dict(compact_log_payload(payload) or {})


def build_synthesis_flow_log(result: Dict) -> Dict:
    step_outputs = dict(result.get("step_outputs", {}) or {})
    payload = {
        "stages": build_stage_group(step_outputs, [
            "evidence",
            "context_pruning",
            "llm_answer",
            "verification",
            "citation_extraction",
            "citation_confidence",
            "final",
        ]),
    }
    return dict(compact_log_payload(payload) or {})


def build_llm_input_log(result: Dict) -> Dict:
    debug_trace = dict(((result.get("retrieval_metrics", {}) or {}).get("debug_trace", {}) or {}))
    llm_input = dict(debug_trace.get("llm_input", {}) or {})
    if not llm_input:
        llm_input = {
            "query": ((result.get("request", {}) or {}).get("effective_query") or result.get("question", "")),
            "context": result.get("context_used", "") or "",
        }
    return dict(compact_log_payload(llm_input) or {})


def append_json_log_block(lines: List[str], title: str, payload: Dict, empty_message: str) -> None:
    lines.append(f"**{title}**")
    lines.append("")
    if payload:
        lines.append("```json")
        lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
        lines.append("```")
    else:
        lines.append(f"- {empty_message}")
    lines.append("")


def append_flow_logs(lines: List[str], result: Dict) -> None:
    append_json_log_block(lines, "Request Config", build_request_config_log(result), "_No request config_")
    append_json_log_block(lines, "Retrieval Flow", build_retrieval_flow_log(result), "_No retrieval flow_")
    append_json_log_block(lines, "Synthesis Flow", build_synthesis_flow_log(result), "_No synthesis flow_")
    append_json_log_block(lines, "LLM Input", build_llm_input_log(result), "_No LLM input trace_")


def write_logs(log_dir: Path, run_id: str, summary: Dict, selected_cases: List[FAQCase], results: List[Dict]) -> Tuple[Path, Path]:
    json_path = log_dir / f"{run_id}.json"
    md_path = log_dir / f"{run_id}.md"

    payload = {
        "summary": summary,
        "selected_cases": [asdict(case) for case in selected_cases],
        "results": results,
    }

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(summary, selected_cases, results))

    return json_path, md_path


def render_markdown(summary: Dict, selected_cases: List[FAQCase], results: List[Dict]) -> str:
    lines: List[str] = []
    lines.append("# FAQ System Test Log")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append(f"- Mode: `{summary['run_mode']}`")
    lines.append(f"- Created at: `{summary['created_at']}`")
    lines.append(f"- Sample size: `{summary['sample_size']}`")
    lines.append(f"- Execution path: `{summary.get('execution_path', 'service')}`")
    if summary.get("requested_execution_path") and summary.get("requested_execution_path") != summary.get("execution_path"):
        lines.append(f"- Requested execution path: `{summary['requested_execution_path']}` (deprecated alias, mapped to `service`)")
    if summary.get("provider_preset"):
        lines.append(f"- Provider preset: `{summary['provider_preset']}`")
    lines.append(f"- Retrieval mode: `{summary['retrieval_mode']}`")
    lines.append(f"- Top-k: `{summary['top_k']}`")
    lines.append(f"- Reranking: `{'on' if summary['use_reranking'] else 'off'}`")
    lines.append(f"- MMR: `{'on' if summary['use_mmr'] else 'off'}`")
    lines.append(f"- Hierarchical: `{'on' if summary.get('use_hierarchical_expansion') else 'off'}`")
    lines.append(f"- Evidence selection: `{'on' if summary.get('use_evidence_selection') else 'off'}`")
    lines.append(f"- Semantic highlighting: `{'on' if summary.get('use_semantic_highlighting') else 'off'}`")
    if summary.get("requested_disabled_modules"):
        lines.append(f"- Requested disabled modules: `{', '.join(str(item) for item in summary.get('requested_disabled_modules', []) if item)}`")
    if summary.get("disabled_modules"):
        lines.append(f"- Disabled modules: `{', '.join(str(item) for item in summary.get('disabled_modules', []) if item)}`")
    if summary.get("conflicting_requested_modules"):
        lines.append(f"- Conflict-resolved modules: `{', '.join(str(item) for item in summary.get('conflicting_requested_modules', []) if item)}`")
    if summary.get("requested_enabled_modules"):
        lines.append(f"- Explicitly enabled modules: `{', '.join(str(item) for item in summary.get('requested_enabled_modules', []) if item)}`")
    if summary.get("module_resolution_notes"):
        for note in summary.get("module_resolution_notes", []):
            lines.append(f"- Module resolution note: `{note}`")
    lines.append(f"- LLM provider: `{summary.get('llm_provider', 'local')}`")
    if summary.get("remote_model"):
        lines.append(f"- Remote model: `{summary['remote_model']}`")
    if summary.get("llm_provider") == "ollama" and summary.get("ollama_base_url"):
        lines.append(f"- Ollama URL: `{summary['ollama_base_url']}`")
    lines.append(f"- Annotation sidecar: `{summary.get('annotation_sidecar', '')}`")
    lines.append(f"- Transport success count: `{summary['transport_success_count']}`")
    lines.append(f"- Transport failure count: `{summary['transport_failure_count']}`")
    lines.append(f"- Initial LLM no-response count: `{summary.get('initial_no_response_count', 0)}`")
    lines.append(f"- Retry attempted count: `{summary.get('retry_attempted_count', 0)}`")
    lines.append(f"- Retry resolved count: `{summary.get('retry_resolved_count', 0)}`")
    lines.append(f"- Retry unresolved count: `{summary.get('retry_unresolved_count', 0)}`")
    lines.append(f"- Final LLM no-response count: `{summary.get('final_no_response_count', 0)}`")
    lines.append(f"- Benchmark pass count: `{summary['benchmark_pass_count']}`")
    lines.append(f"- Benchmark fail count: `{summary['benchmark_fail_count']}`")
    lines.append(f"- Avg benchmark composite: `{summary['avg_benchmark_composite']:.2%}`")
    lines.append(f"- Legacy answer correct count: `{summary['answer_correct_count']}`")
    lines.append(f"- Legacy answer incorrect count: `{summary['answer_incorrect_count']}`")
    lines.append(f"- Avg legacy answer correctness: `{summary['avg_answer_correctness']:.2%}`")
    lines.append(f"- Avg confidence: `{summary['avg_confidence']:.2%}`")
    lines.append(f"- Avg groundedness: `{summary['avg_groundedness']:.2%}`")
    lines.append(f"- Avg provenance: `{summary['avg_provenance']:.2%}`")
    lines.append(f"- Avg faithfulness: `{summary['avg_faithfulness']:.2%}`")
    lines.append(f"- Avg answer relevance: `{summary['avg_answer_relevance']:.2%}`")
    lines.append(f"- Avg citation support: `{summary['avg_citation_support']:.2%}`")
    lines.append(f"- Avg hallucination rate: `{summary['avg_hallucination_rate']:.2%}`")
    lines.append(f"- Avg contradiction rate: `{summary['avg_contradiction_rate']:.2%}`")
    lines.append(f"- Avg reference-answer score: `{summary['avg_reference_answer_score']:.2%}`")
    lines.append(f"- Avg BLEU-1: `{summary['avg_bleu_1']:.2%}`")
    lines.append(f"- Avg BLEU-4: `{summary['avg_bleu_4']:.2%}`")
    lines.append(f"- Avg ROUGE-1: `{summary['avg_rouge_1']:.2%}`")
    lines.append(f"- Avg ROUGE-2: `{summary['avg_rouge_2']:.2%}`")
    lines.append(f"- Avg ROUGE-L: `{summary['avg_rouge_l']:.2%}`")
    lines.append(f"- Avg BERTScore-F1: `{format_optional_percentage(summary.get('avg_bertscore_f1'))}`")
    lines.append(f"- BERTScore available count: `{summary['bertscore_available_count']}`")
    lines.append(f"- Annotation augmented count: `{summary['annotation_augmented_count']}`")
    lines.append(f"- Proxy-only count: `{summary['proxy_only_count']}`")
    lines.append(f"- Avg total time: `{summary['avg_total_time']:.2f}s`")
    if summary.get("seed") is not None:
        lines.append(f"- Seed: `{summary['seed']}`")
    if summary.get("error"):
        lines.append(f"- Bootstrap / run error: `{summary['error']}`")
    if summary.get("error_traceback"):
        lines.append("")
        lines.append("### Bootstrap Traceback")
        lines.append("")
        lines.append("```text")
        lines.append(summary["error_traceback"].rstrip())
        lines.append("```")
    lines.append("")

    lines.append("## Selected Questions")
    lines.append("")
    for case in selected_cases:
        lines.append(f"- `{case.case_id}` | **{case.topic}** | {case.question}")
    lines.append("")

    lines.append("## Detailed Results")
    lines.append("")
    for idx, result in enumerate(results, start=1):
        lines.append(f"### Test {idx}: {result['case_id']}")
        lines.append("")
        lines.append(f"- Topic: **{result['topic']}**")
        lines.append(f"- Question: {result['question']}")
        lines.append(f"- Expected source: `{result['expected_source']}`")
        lines.append(f"- Expected response type: `{result.get('expected_response_type', 'rag')}`")
        lines.append(f"- Transport success: `{'yes' if result.get('transport_success') else 'no'}`")
        lines.append(f"- Benchmark pass: `{'yes' if result.get('benchmark_pass') else 'no'}`")
        lines.append(f"- Benchmark composite: `{safe_float(result.get('benchmark_composite_score', 0.0)):.2%}`")
        lines.append(f"- Legacy answer correct: `{'yes' if result.get('answer_correct') else 'no'}`")
        lines.append(f"- Legacy answer correctness: `{safe_float(result.get('answer_correctness', 0.0)):.2%}`")
        lines.append(f"- Response type match: `{'yes' if result.get('response_type_match') else 'no'}`")
        lines.append(f"- Response type: `{result.get('response_type', 'rag')}`")
        lines.append(f"- Attempt: `{int(safe_float(result.get('attempt', 1)))}`")
        lines.append(f"- Retry attempted: `{'yes' if result.get('retry_attempted') else 'no'}`")
        if result.get("retry_reason"):
            lines.append(f"- Retry reason: `{result.get('retry_reason', '')}`")
        lines.append(f"- LLM no response (final): `{'yes' if is_llm_no_response_result(result) else 'no'}`")
        retry_history = list(result.get("retry_history", []) or [])
        if retry_history:
            first_attempt = dict(retry_history[0] or {})
            lines.append(f"- First attempt transport success: `{'yes' if first_attempt.get('transport_success') else 'no'}`")
            if first_attempt.get("error_message"):
                lines.append(f"- First attempt error: `{first_attempt.get('error_message', '')}`")
        request_config = result.get("request", {}) or {}
        lines.append(f"- Hierarchical: `{'on' if request_config.get('use_hierarchical_expansion', True) else 'off'}`")
        lines.append(f"- Evidence selection: `{'on' if request_config.get('use_evidence_selection', True) else 'off'}`")
        lines.append(f"- Semantic highlighting: `{'on' if request_config.get('use_semantic_highlighting', True) else 'off'}`")
        module_resolution = request_config.get("module_resolution", {}) or {}
        if module_resolution.get("requested_disabled_modules"):
            lines.append(f"- Requested disabled modules: `{', '.join(str(item) for item in module_resolution.get('requested_disabled_modules', []) if item)}`")
        if module_resolution.get("requested_enabled_modules"):
            lines.append(f"- Requested enabled modules: `{', '.join(str(item) for item in module_resolution.get('requested_enabled_modules', []) if item)}`")
        if module_resolution.get("conflicting_requested_modules"):
            lines.append(f"- Conflict-resolved modules: `{', '.join(str(item) for item in module_resolution.get('conflicting_requested_modules', []) if item)}`")
        for note in module_resolution.get("resolution_notes", []) or []:
            lines.append(f"- Module resolution note: `{note}`")
        lines.append(f"- Confidence: `{safe_float(result['confidence']):.2%}`")
        lines.append(f"- Groundedness: `{safe_float(result['groundedness_score']):.2%}`")
        lines.append(f"- Provenance: `{safe_float(result['provenance_score']):.2%}`")
        lines.append(f"- Faithfulness: `{safe_float((result.get('quality_metrics', {}) or {}).get('faithfulness_score', 0.0)):.2%}`")
        lines.append(f"- Answer relevance: `{safe_float((result.get('quality_metrics', {}) or {}).get('answer_relevance_score', 0.0)):.2%}`")
        lines.append(f"- Citation support: `{safe_float((result.get('quality_metrics', {}) or {}).get('citation_support_score', 0.0)):.2%}`")
        lines.append(f"- Hallucination rate: `{safe_float((result.get('quality_metrics', {}) or {}).get('hallucination_rate', 0.0)):.2%}`")
        lines.append(f"- Contradiction rate: `{safe_float((result.get('quality_metrics', {}) or {}).get('contradiction_rate', 0.0)):.2%}`")
        lines.append(f"- Reference-answer score: `{safe_float(result.get('reference_answer_score', 0.0)):.2%}`")
        lines.append(f"- BLEU-1: `{safe_float((result.get('reference_metrics', {}) or {}).get('bleu_1', 0.0)):.2%}`")
        lines.append(f"- BLEU-4: `{safe_float((result.get('reference_metrics', {}) or {}).get('bleu_4', 0.0)):.2%}`")
        lines.append(f"- ROUGE-1: `{safe_float((result.get('reference_metrics', {}) or {}).get('rouge_1', 0.0)):.2%}`")
        lines.append(f"- ROUGE-2: `{safe_float((result.get('reference_metrics', {}) or {}).get('rouge_2', 0.0)):.2%}`")
        lines.append(f"- ROUGE-L: `{safe_float((result.get('reference_metrics', {}) or {}).get('rouge_l', 0.0)):.2%}`")
        lines.append(f"- BERTScore-F1: `{safe_float((result.get('reference_metrics', {}) or {}).get('bertscore_f1', 0.0)):.2%}`")
        lines.append(f"- Retrieval dimension: `{safe_float((result.get('benchmark_metrics', {}) or {}).get('retrieval_dimension_score', 0.0)):.2%}`")
        lines.append(f"- Citation dimension: `{safe_float((result.get('benchmark_metrics', {}) or {}).get('citation_dimension_score', 0.0)):.2%}`")
        lines.append(f"- Revision applied: `{'yes' if result['revision_applied'] else 'no'}`")
        lines.append(f"- Total time: `{safe_float(result['total_time']):.2f}s`")
        lines.append(f"- Expected-answer overlap: `{safe_float(result['expected_answer_overlap']):.2%}`")
        lines.append(f"- Expected-source overlap: `{safe_float(result['expected_source_overlap']):.2%}`")
        lines.append(f"- Annotation coverage: `{result.get('annotation_coverage_state', 'proxy_only')}`")
        if result.get("evaluation_reason"):
            lines.append(f"- Benchmark reason: `{result['evaluation_reason']}`")
        if result.get("benchmark_veto_reason"):
            lines.append(f"- Safety veto: `{result['benchmark_veto_reason']}`")
        if result.get("legacy_evaluation_reason"):
            lines.append(f"- Legacy evaluation reason: `{result['legacy_evaluation_reason']}`")
        if (result.get("reference_metrics", {}) or {}).get("bertscore_error"):
            lines.append(f"- BERTScore status: `{(result.get('reference_metrics', {}) or {}).get('bertscore_error', '')}`")
        if result.get("error_message"):
            lines.append(f"- Error: `{result['error_message']}`")
        if result.get("error_traceback"):
            lines.append("")
            lines.append("```text")
            lines.append(result["error_traceback"].rstrip())
            lines.append("```")
        lines.append("")
        lines.append("**Actual answer**")
        lines.append("")
        actual_answer = result.get("actual_answer", "") or ""
        if actual_answer:
            lines.append("```text")
            lines.append(actual_answer)
            lines.append("```")
        else:
            lines.append("_No answer returned._")
        lines.append("")

        citations = unique_nonempty_strings(result.get("citations", []) or [])
        lines.append("**Citations**")
        lines.append("")
        if citations:
            for citation in citations:
                lines.append(f"- {citation}")
        else:
            lines.append("- _No citations_")
        lines.append("")

        legal_references = unique_nonempty_strings(result.get("legal_references", []) or [])
        lines.append("**Legal references**")
        lines.append("")
        if legal_references:
            for item in legal_references:
                lines.append(f"- {item}")
        else:
            lines.append("- _No legal references_")
        lines.append("")

        lines.append("**Flow Logs**")
        lines.append("")
        append_flow_logs(lines, result)

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    args = apply_provider_preset(args)
    if not args.dry_run:
        args = auto_select_runtime_backend(args)
    question_file = Path(args.questions_file).resolve()
    cases = load_xlsx_cases(question_file)
    selected_cases = sample_cases(cases, args.sample_size, args.seed)

    run_mode = "dry-run" if args.dry_run else "full-run"
    run_time = datetime.now()
    run_id = f"faq_system_test_{run_mode}_{run_time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = build_dated_log_dir(Path(args.log_dir).resolve(), run_time)
    ensure_log_dir(log_dir)

    if args.dry_run:
        results = []
        for case in selected_cases:
            annotation_coverage_state = "annotation_augmented" if has_annotation_data(case.annotations) else "proxy_only"
            results.append(
                {
                    "case_id": case.case_id,
                    "sheet_name": case.sheet_name,
                    "row_number": case.row_number,
                    "topic": case.topic,
                    "question": case.question,
                    "expected_answer": case.expected_answer,
                    "expected_source": case.expected_source,
                    "expected_response_type": case.expected_response_type,
                    "annotations": dict(case.annotations or {}),
                    "transport_success": False,
                    "success": False,
                    "benchmark_pass": False,
                    "benchmark_composite_score": 0.0,
                    "benchmark_veto_reason": "dry_run",
                    "benchmark_reason": "dry_run",
                    "benchmark_metrics": {
                        "annotation_coverage_state": annotation_coverage_state,
                        "retrieval_dimension_score": 0.0,
                        "citation_dimension_score": 0.0,
                        "benchmark_composite_score": 0.0,
                    },
                    "annotation_coverage_state": annotation_coverage_state,
                    "answer_correct": False,
                    "answer_correctness": 0.0,
                    "response_type_match": False,
                    "evaluation_reason": "dry_run",
                    "legacy_evaluation_reason": "dry_run",
                    "error_message": "Dry run only - runtime not bootstrapped.",
                    "actual_answer": "",
                    "citations": [],
                    "confidence": 0.0,
                    "groundedness_score": 0.0,
                    "provenance_score": 0.0,
                    "quality_metrics": empty_quality_metrics(),
                    "reference_answer_score": 0.0,
                    "reference_metrics": empty_reference_metrics(),
                    "revision_applied": False,
                    "retrieval_time": 0.0,
                    "synthesis_time": 0.0,
                    "total_time": 0.0,
                    "retrieval_metrics": {},
                    "retrieved_chunks": [],
                    "selected_evidence": [],
                    "evidence_spans": [],
                    "claim_analyses": [],
                    "query_plan": {},
                    "context_used": "",
                    "expected_answer_overlap": 0.0,
                    "expected_source_overlap": 0.0,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "attempt": 1,
                    "retry_attempted": False,
                    "retry_reason": "",
                    "initial_no_response": False,
                    "retry_history": [],
                    "step_outputs": {},
                }
            )

        summary = build_run_summary(selected_cases, results, args, run_mode, all_cases=cases)
        json_path, md_path = write_logs(log_dir, run_id, summary, selected_cases, results)
        print(f"Dry-run log saved to:\n- {json_path}\n- {md_path}")
        return 0

    results = []
    retry_queue: List[Tuple[int, FAQCase]] = []
    total_cases = len(selected_cases)
    print(f"[RUN] Starting FAQ system test for {total_cases} case(s).")

    for index, case in enumerate(selected_cases, start=1):
        print_case_progress("RUN", index, total_cases, case)
        result = run_case_safely(case, args)
        result["attempt"] = 1
        result["retry_attempted"] = False
        result["retry_reason"] = ""
        result["initial_no_response"] = False
        result["retry_history"] = []
        if is_llm_no_response_result(result):
            result["initial_no_response"] = True
            retry_queue.append((index - 1, case))
            print("    -> marked_for_retry=yes (reason=llm_no_response)")
        print_case_status(result)
        results.append(result)

    if retry_queue:
        print(f"[RETRY] Running {len(retry_queue)} queued case(s) after first pass.")
    else:
        print("[RETRY] No llm_no_response case detected in first pass.")

    for retry_index, (result_index, case) in enumerate(retry_queue, start=1):
        print_case_progress("RETRY", retry_index, len(retry_queue), case)
        first_attempt = dict(results[result_index] or {})
        retry_result = run_case_safely(case, args)
        retry_result["attempt"] = 2
        retry_result["retry_attempted"] = True
        retry_result["retry_reason"] = "llm_no_response"
        retry_result["initial_no_response"] = True
        retry_result["retry_history"] = [
            {
                "attempt": int(safe_float(first_attempt.get("attempt", 1))),
                "transport_success": bool(first_attempt.get("transport_success", False)),
                "error_message": first_attempt.get("error_message"),
                "actual_answer": first_attempt.get("actual_answer", ""),
                "total_time": safe_float(first_attempt.get("total_time", 0.0)),
                "timestamp": first_attempt.get("timestamp", ""),
            }
        ]
        results[result_index] = retry_result
        if is_llm_no_response_result(retry_result):
            print("    -> retry_result=no_response")
        else:
            print("    -> retry_result=answered")
        print_case_status(retry_result)

    if retry_queue:
        retry_resolved_count = sum(
            1 for item in results if item.get("retry_attempted") and not is_llm_no_response_result(item)
        )
        retry_unresolved_count = sum(
            1 for item in results if item.get("retry_attempted") and is_llm_no_response_result(item)
        )
        print(f"[RETRY] Completed: resolved={retry_resolved_count}, unresolved={retry_unresolved_count}")

    summary = build_run_summary(
        selected_cases,
        results,
        args,
        run_mode,
        all_cases=cases,
    )
    json_path, md_path = write_logs(log_dir, run_id, summary, selected_cases, results)
    print(f"Test log saved to:\n- {json_path}\n- {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
