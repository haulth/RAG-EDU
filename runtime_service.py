"""Shared runtime/bootstrap helpers for API and alternate UIs."""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import unicodedata
import gc
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from app_settings import build_public_app_settings, load_app_settings
from llm_backends import (
    groq_generate,
    load_provider_settings,
    normalize_ollama_base_url,
    normalize_provider_name,
    ollama_generate,
)
from rag_quality_metrics import empty_quality_metrics
from runtime_bootstrap import bootstrap_runtime


PROJECT_ROOT = Path(__file__).resolve().parent

_RUNTIME_CACHE: Dict[str, Dict[str, Any]] = {}
_RUNTIME_LOCK = threading.Lock()
_QUERY_LOCK = threading.Lock()
_RUNTIME_STATE: Dict[str, Any] = {
    "loaded": False,
    "loading": False,
    "error": None,
    "loaded_at": None,
    "load_seconds": None,
}

_RUNTIME_MODULE_NAMES = [
    "rag_test_runtime",
    "CELL_3_LOAD_ALL_MODELS",
    "CELL_4_ULTIMATE_COMPLETE_METADATA",
    "CELL_5_HYBRID_RETRIEVAL_ENHANCED",
    "CELL_6_LLM_SYNTHESIS_WITH_PRUNING",
    "CELL_8_END_TO_END_PIPELINE",
]

LEGAL_HINTS = (
    "hoc bong",
    "hoc phi",
    "hoc phan",
    "tin chi",
    "quy che",
    "dao tao",
    "dieu ",
    "khoan ",
    "diem ",
    "thi",
    "vang thi",
    "phuc khao",
    "buoc thoi hoc",
    "canh bao hoc tap",
    "tot nghiep",
    "xep loai",
    "hoc vu",
    "hoc ky",
    "nam hoc",
    "dang ky hoc",
    "sinh vien",
    "ctsv",
    "chinh quy",
    "vua lam vua hoc",
)

COURSE_PROGRESS_HINTS = (
    "rot mon",
    "mon bat buoc",
    "mon hoc",
    "hoc phan",
    "dang ky mon",
    "dang ky hoc",
    "hoc lai",
    "hoc bu",
    "hoc cai thien",
    "hoc lai mon",
    "truot mon",
    "no mon",
)

GREETING_PHRASES = {
    "xin chao",
    "xin chao ban",
    "chao",
    "chao ban",
    "hello",
    "hi",
    "hey",
    "alo",
}

THANKS_PHRASES = {
    "cam on",
    "cam on ban",
    "thanks",
    "thank you",
}

GOODBYE_PHRASES = {
    "tam biet",
    "bye",
    "bye bye",
    "hen gap lai",
    "chao tam biet",
}

TRAINING_OFFICE_CONTACT_NOTE = "Để biết thêm chi tiết, sinh viên vui lòng liên hệ lại Phòng Đào tạo để được trao đổi thêm."
ABSTENTION_MARKERS = (
    "khong co quy dinh",
    "khong co quy dinh truc tiep",
    "chua co can cu",
    "chua tim thay can cu",
    "khong tim thay can cu",
    "lien he lai phong dao tao",
)

IDENTITY_PHRASES = {
    "ban la ai",
    "ban la gi",
    "ban co the lam gi",
    "ban giup duoc gi",
    "ban ho tro duoc gi",
    "huong dan su dung",
    "cach su dung",
}

ROUTER_SYSTEM_PROMPT = """Bạn là bộ định tuyến và lập kế hoạch truy xuất cho chatbot Quy chế đào tạo.
Nhiệm vụ của bạn là GIỮ ĐÚNG Ý ĐỊNH CÂU HỎI, không được trả lời nội dung cuối cùng.

Bạn chỉ được trả JSON object đúng format:
{
  "route": "no_rag | rag_direct | rag_rewrite | clarify",
  "intent": "small_talk | legal_lookup | duration_lookup | score_lookup | procedure | unknown",
  "rewritten_query": "",
  "clarification_question": "",
  "reason": "",
  "normalized_query": "",
  "abstract_query": "",
  "pseudo_document": "",
  "query_type": "binary_legal | exact_numeric | duration_lookup | multi_hop | procedure | definition | legal_lookup",
  "semantic_anchors": [],
  "must_include": [],
  "must_avoid": [],
  "required_hops": 1,
  "answer_policy": "direct | partial | abstain | clarify",
  "planner_confidence": 0.0
}

Luật bắt buộc:
- no_rag: chỉ dùng cho chào hỏi/xã giao/hỏi chatbot làm gì.
- rag_direct: câu hỏi đã đủ rõ và gần ngôn ngữ quy chế.
- rag_rewrite: câu hỏi đời thường nhưng rõ là đang hỏi quy chế/học vụ; phải rewrite sang ngôn ngữ pháp lý.
- clarify: thiếu dữ kiện để truy xuất an toàn.
- normalized_query phải thân thiện với lexical retrieval.
- abstract_query phải là truy vấn step-back ở mức khái niệm.
- pseudo_document là đoạn HyDE/Query2doc 2-4 câu, mô tả căn cứ lý tưởng cần tìm; không được bịa kết luận cuối.
- semantic_anchors, must_include, must_avoid phải ở mức ý nghĩa/chủ đề, không chỉ lặp lại keyword bề mặt.
- Không trả lời nội dung của quy chế.
- Không thêm chữ nào ngoài JSON."""

ROUTER_PROMPT_TEMPLATE = """QUERY:
{query}

Hãy phân loại thật chặt và đồng thời chuẩn bị retrieval seed.
Gợi ý:
- "K50", "tuyển 2024", "hạn chót ra trường", "ngành 4 năm" là dạng đời thường, thường cần rag_rewrite.
- Nếu câu có chữ "ra trường" nhưng đang hỏi về rớt môn, môn bắt buộc, học phần, đăng ký học, học bù hoặc học lại thì KHÔNG được suy ra sang "thời gian học tập tối đa".
- Nếu câu hỏi đã có từ như "Điều", "Khoản", "thời gian học tập tối đa", "buộc thôi học", "cảnh báo học tập", "học phần", "tín chỉ" thì thường là rag_direct.
- Nếu là câu chào hỏi hoặc cảm ơn thì no_rag.
- Nếu thiếu dữ kiện quan trọng thì clarify.
- Nếu route là rag_direct hoặc rag_rewrite thì phải điền cả retrieval seed ở các field planner."""

PLANNER_SYSTEM_PROMPT = """Bạn là retrieval planner cho một hệ RAG pháp lý.
Nhiệm vụ của bạn là GIỮ ĐÚNG Ý ĐỊNH CÂU HỎI và tạo truy vấn/giả định truy xuất tốt hơn. Bạn không được trả lời nội dung cuối cùng.

Bạn chỉ được trả về đúng một JSON object theo schema:
{
  "normalized_query": "",
  "abstract_query": "",
  "pseudo_document": "",
  "query_type": "binary_legal | exact_numeric | duration_lookup | multi_hop | procedure | definition | legal_lookup",
  "semantic_anchors": [],
  "must_include": [],
  "must_avoid": [],
  "required_hops": 1,
  "answer_policy": "direct | partial | abstain | clarify",
  "reason": "",
  "confidence": 0.0
}

Nguyên tắc:
- Không được trả lời nội dung của quy chế.
- normalized_query: phiên bản truy vấn rõ ràng, trung thành với ý định ban đầu, thân thiện với lexical retrieval.
- abstract_query: một truy vấn step-back ở mức khái niệm/pháp lý cao hơn để semantic retrieval hiểu đúng chủ đề.
- pseudo_document: 2-4 câu kiểu Query2doc/HyDE, mô tả đoạn tài liệu lý tưởng cần tìm, KHÔNG được bịa quy định cụ thể.
- semantic_anchors: 2-6 khái niệm ở cấp ý nghĩa, không chỉ lặp lại từ khóa bề mặt.
- must_include: loại căn cứ cần có để trả lời đúng, ví dụ quy định chính, điều kiện, ngoại lệ, ngưỡng, hệ quả, định nghĩa.
- must_avoid: các chủ đề gần nghĩa nhưng dễ làm lệch ý định.
- required_hops: số loại căn cứ cần kết hợp để trả lời đúng.
- answer_policy chỉ là gợi ý cho pipeline, không phải câu trả lời cuối.
- Nếu chưa chắc, giữ confidence thấp thay vì bịa thêm.
- Không thêm chữ nào ngoài JSON."""

PLANNER_PROMPT_TEMPLATE = """ORIGINAL_QUERY:
{original_query}

NORMALIZED_INPUT:
{effective_query}

ROUTER_RESULT:
{router_result}

Hãy lập retrieval plan thật chặt.
Lưu ý:
- Luôn giữ đúng intent ban đầu của người hỏi, không được suy diễn sang một chủ đề khác chỉ vì có vài từ gần nghĩa.
- Nếu câu hỏi đời thường, hãy chuyển nó sang ngôn ngữ quy chế nhưng vẫn giữ nguyên yêu cầu cần trả lời.
- Nếu câu hỏi cần kết hợp nhiều căn cứ mới kết luận được, required_hops phải >= 2.
- must_include và must_avoid phải ở mức khái niệm/chủ đề, không phải chỉ sao chép nguyên câu hỏi.
- pseudo_document phải hữu ích cho HyDE/vector retrieval nhưng không được khẳng định điều chưa được chứng minh."""


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_user_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", (text or "").strip().lower())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_ascii_text(text: str) -> str:
    return normalize_user_text(text)


def ensure_quality_metrics(payload: Optional[Dict[str, Any]], *, visible: bool = True) -> Dict[str, Any]:
    payload = payload or {}
    quality_metrics = dict(empty_quality_metrics())
    quality_metrics.update(payload.get("quality_metrics", {}) or {})
    payload["quality_metrics"] = quality_metrics
    if not visible:
        return payload

    payload["quality_metrics"]["faithfulness_score"] = max(
        float(payload["quality_metrics"].get("faithfulness_score", 0.0) or 0.0),
        safe_float(payload.get("groundedness_score", 0.0)),
    )
    payload["quality_metrics"]["citation_support_score"] = max(
        float(payload["quality_metrics"].get("citation_support_score", 0.0) or 0.0),
        safe_float(payload.get("provenance_score", 0.0)),
    )
    return payload


def has_course_progress_hint(normalized_text: str) -> bool:
    return any(hint in (normalized_text or "") for hint in COURSE_PROGRESS_HINTS)


def is_duration_lookup_query(query_norm: str) -> bool:
    query_norm = query_norm or ""
    if not query_norm:
        return False

    if has_course_progress_hint(query_norm):
        return False

    strong_markers = [
        "han chot",
        "thoi gian toi da",
        "toi da bao nhieu nam",
        "bao lau moi ra truong",
        "tot nghiep khi nao",
        "may nam",
    ]
    if any(marker in query_norm for marker in strong_markers):
        return True

    if "ra truong" in query_norm:
        supporting_markers = [
            "k50",
            "k51",
            "k52",
            "tuyen 20",
            "nganh",
            "bao lau",
            "nam",
            "toi da",
            "han chot",
        ]
        return any(marker in query_norm for marker in supporting_markers)

    return False


def strip_leading_greeting(normalized_text: str) -> str:
    return re.sub(
        r"^(xin chao|chao|hello|hi|hey|alo)( ban| chatbot| he thong| nhe| nha| a| em)?\s*",
        "",
        normalized_text or "",
    ).strip()


def has_legal_hint(normalized_text: str) -> bool:
    return any(hint in (normalized_text or "") for hint in LEGAL_HINTS)


def detect_query_topic_focus(normalized_text: str) -> str:
    normalized_text = normalized_text or ""
    if any(marker in normalized_text for marker in [
        "vang thi",
        "bo thi",
        "vang mat trong buoi thi",
        "vang mat buoi thi",
        "khong du thi",
        "ngu quen",
    ]):
        return "exam_absence"
    if any(marker in normalized_text for marker in [
        "hoc bong",
        "xet hoc bong",
        "diem xet hoc bong",
        "dong diem",
        "quy hoc bong",
    ]):
        return "scholarship"
    if any(marker in normalized_text for marker in [
        "facebook",
        "mang xa hoi",
        "chui boi",
        "binh luan dung tuc",
        "xuc pham",
        "xia xoi truong",
        "ren luyen",
    ]):
        return "student_conduct"
    return ""


def is_generic_router_clarify(clarification_question: str) -> bool:
    question_norm = normalize_ascii_text(clarification_question)
    if not question_norm:
        return False
    generic_patterns = (
        "ten chinh xac cua quy che",
        "ten chinh xac cua quy che hoac van ban",
        "van ban quy pham phap luat",
        "ban dang tim hieu",
        "vi du",
        "quy che hoc bong",
        "quy che tuyen sinh",
    )
    return any(pattern in question_norm for pattern in generic_patterns)


def matches_basic_phrase(normalized_text: str, phrases: set[str]) -> bool:
    return any(
        normalized_text == phrase or normalized_text.startswith(f"{phrase} ")
        for phrase in phrases
    )


def detect_basic_intent(message: str) -> Optional[str]:
    normalized = normalize_user_text(message)
    if not normalized:
        return "empty"

    remainder = strip_leading_greeting(normalized)
    candidate = remainder or normalized

    if normalized in GREETING_PHRASES or (remainder == "" and normalized.split(" ")[0] in {"xin", "chao", "hello", "hi", "hey", "alo"}):
        return "greeting"

    if matches_basic_phrase(normalized, THANKS_PHRASES) or matches_basic_phrase(candidate, THANKS_PHRASES):
        return "thanks"

    if matches_basic_phrase(normalized, GOODBYE_PHRASES) or matches_basic_phrase(candidate, GOODBYE_PHRASES):
        return "goodbye"

    if has_legal_hint(candidate):
        return None

    if matches_basic_phrase(normalized, IDENTITY_PHRASES) or matches_basic_phrase(candidate, IDENTITY_PHRASES):
        return "help"

    return None


def build_basic_response(message: str, intent: str) -> Dict[str, Any]:
    if intent == "greeting":
        answer = (
            "Chào bạn. Mình là trợ lý tra cứu Quy chế đào tạo. "
            "Bạn có thể hỏi mình về điểm số, học phần, thi cử, cảnh báo học tập, buộc thôi học hoặc tốt nghiệp."
        )
    elif intent == "thanks":
        answer = "Mình luôn sẵn sàng hỗ trợ. Khi cần, bạn cứ gửi câu hỏi cụ thể về Quy chế đào tạo nhé."
    elif intent == "goodbye":
        answer = "Chào bạn. Khi cần tra cứu thêm về Quy chế đào tạo, bạn cứ nhắn mình nhé."
    elif intent == "help":
        answer = (
            "Mình có thể hỗ trợ tra cứu các nội dung trong Quy chế đào tạo như điểm số, học vụ, đăng ký học phần, "
            "cảnh báo học tập, buộc thôi học và tốt nghiệp. Bạn cứ gửi câu hỏi cụ thể, mình sẽ tìm theo tài liệu đang nạp."
        )
    else:
        answer = "Bạn có thể gửi câu hỏi cụ thể về Quy chế đào tạo, mình sẽ hỗ trợ tra cứu cho bạn."

    return {
        "success": True,
        "error_message": None,
        "answer": answer,
        "citations": [],
        "confidence": 100.0,
        "groundedness_score": 100.0,
        "provenance_score": 100.0,
        "revision_applied": False,
        "retrieval_time": 0.0,
        "synthesis_time": 0.0,
        "total_time": 0.0,
        "retrieval_metrics": {
            "route": "basic_intent",
            "intent": intent,
        },
        "retrieved_chunks": [],
        "selected_evidence": [],
        "evidence_spans": [],
        "claim_analyses": [],
        "quality_metrics": empty_quality_metrics(),
        "legal_references": [],
        "response_type": "basic_intent",
        "show_metrics": False,
        "show_references": False,
        "request": {
            "message": message,
        },
    }


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def extract_query_slots(query: str) -> Dict[str, Any]:
    query_norm = normalize_ascii_text(query)
    cohort_year = None
    year_match = re.search(r"\b(20\d{2})\b", query_norm)
    if year_match:
        cohort_year = int(year_match.group(1))

    duration_years = None
    duration_match = re.search(r"\b([1-9]|10)\s*nam\b", query_norm)
    if duration_match:
        duration_years = int(duration_match.group(1))

    program_scope = ""
    if "vua lam vua hoc" in query_norm:
        program_scope = "vlvh"
    elif "chinh quy" in query_norm or "k" in query_norm or "tuyen" in query_norm:
        program_scope = "chinh_quy"

    target_fact = ""
    if is_duration_lookup_query(query_norm):
        target_fact = "thoi_gian_hoc_tap_toi_da"

    topic_focus = detect_query_topic_focus(query_norm)

    return {
        "cohort_year": cohort_year,
        "program_duration_years": duration_years,
        "program_scope": program_scope,
        "target_fact": target_fact,
        "topic_focus": topic_focus,
    }


def build_duration_rewrite(query: str, slots: Dict[str, Any]) -> str:
    duration_years = slots.get("program_duration_years")
    cohort_year = slots.get("cohort_year")
    scope = slots.get("program_scope") or "chinh_quy"
    scope_text = "theo hình thức chính quy" if scope == "chinh_quy" else "theo hình thức vừa làm vừa học"

    question = "Theo Quy chế đào tạo, thời gian học tập tối đa"
    if duration_years:
        question += f" đối với chương trình đào tạo {duration_years} năm {scope_text}"
    else:
        question += f" {scope_text}"
    question += " là bao nhiêu năm?"
    if cohort_year:
        question += f" Nếu tuyển sinh năm {cohort_year} thì hạn tối đa dự kiến là năm nào?"
    return question


def should_invoke_query_router_ai(query: str) -> bool:
    query_norm = normalize_ascii_text(query)
    if not query_norm:
        return False

    if has_legal_hint(query_norm) and len(query_norm.split()) <= 18:
        return False

    return True


def heuristic_query_route(query: str) -> Dict[str, Any]:
    query_norm = normalize_ascii_text(query)
    slots = extract_query_slots(query)
    intent = "duration_lookup" if slots.get("target_fact") == "thoi_gian_hoc_tap_toi_da" else "legal_lookup"
    route = "rag_direct" if has_legal_hint(query_norm) else "rag_rewrite"

    return {
        "route": route,
        "intent": intent,
        "rewritten_query": query,
        "clarification_question": "",
        "reason": "Generic fallback routing based on whether the query is already phrased in regulation-like language.",
        "slots": slots,
        "planner_seed": None,
    }


def extract_planner_seed(payload: Dict[str, Any], fallback_query: str = "") -> Optional[Dict[str, Any]]:
    payload = dict(payload or {})
    candidate = {
        "normalized_query": str(payload.get("normalized_query", "") or payload.get("rewritten_query", "") or fallback_query).strip(),
        "abstract_query": str(payload.get("abstract_query", "") or "").strip(),
        "pseudo_document": str(payload.get("pseudo_document", "") or "").strip(),
        "query_type": str(payload.get("query_type", "") or "").strip().lower(),
        "semantic_anchors": payload.get("semantic_anchors", []),
        "must_include": payload.get("must_include", []),
        "must_avoid": payload.get("must_avoid", []),
        "required_hops": payload.get("required_hops", 1),
        "answer_policy": str(payload.get("answer_policy", "") or "").strip().lower(),
        "reason": str(payload.get("reason", "") or "").strip(),
        "confidence": payload.get("planner_confidence", payload.get("confidence", 0.0)),
    }
    if not any([
        candidate["normalized_query"],
        candidate["abstract_query"],
        candidate["pseudo_document"],
        candidate["semantic_anchors"],
        candidate["must_include"],
        candidate["must_avoid"],
    ]):
        return None
    return candidate


def route_query_before_retrieval(query: str, llm_generate: Callable[..., str]) -> Dict[str, Any]:
    heuristic = heuristic_query_route(query)
    if not should_invoke_query_router_ai(query):
        return heuristic

    prompt = ROUTER_PROMPT_TEMPLATE.format(query=query.strip())
    try:
        with _QUERY_LOCK:
            response = llm_generate(
                prompt,
                max_new_tokens=220,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                system_prompt=ROUTER_SYSTEM_PROMPT,
            )
        payload = extract_json_object(response) or {}
    except Exception:
        payload = {}

    route = str(payload.get("route", "") or "").strip().lower()
    if route not in {"no_rag", "rag_direct", "rag_rewrite", "clarify"}:
        return heuristic

    slots = extract_query_slots(query)
    rewritten_query = str(payload.get("rewritten_query", "") or "").strip()
    if route == "rag_rewrite" and not rewritten_query:
        rewritten_query = heuristic.get("rewritten_query", query)

    if route == "no_rag" and has_legal_hint(normalize_user_text(query)):
        return heuristic

    query_norm = normalize_ascii_text(query)
    if (
        route == "clarify"
        and has_legal_hint(query_norm)
        and is_generic_router_clarify(str(payload.get("clarification_question", "") or ""))
    ):
        heuristic["reason"] = "Router clarify was too generic for a legal query; fallback to heuristic retrieval."
        heuristic["slots"] = slots
        return heuristic

    planner_seed = extract_planner_seed(payload, fallback_query=rewritten_query or query)

    return {
        "route": route,
        "intent": str(payload.get("intent", heuristic.get("intent", "unknown")) or "unknown"),
        "rewritten_query": rewritten_query or query,
        "clarification_question": str(payload.get("clarification_question", "") or "").strip(),
        "reason": str(payload.get("reason", "") or "").strip(),
        "slots": slots,
        "planner_seed": planner_seed,
    }


def _dedupe_text_list(items: Any, max_items: int = 8) -> list[str]:
    values: list[str] = []
    seen = set()
    for item in list(items or []):
        text = re.sub(r"\s+", " ", str(item or "").strip())
        if not text:
            continue
        key = normalize_ascii_text(text)
        if key in seen:
            continue
        seen.add(key)
        values.append(text[:220])
        if len(values) >= max_items:
            break
    return values


def _infer_planner_query_type(query: str, router_result: Dict[str, Any]) -> str:
    query_norm = normalize_ascii_text(query)
    slots = dict((router_result or {}).get("slots", {}) or {})
    if str(slots.get("target_fact", "") or "").strip() == "thoi_gian_hoc_tap_toi_da":
        return "duration_lookup"
    if any(marker in query_norm for marker in ["hay khong", "co bi", "co duoc", "co phai", "duoc khong", "co con", "co the"]):
        return "binary_legal"
    if any(marker in query_norm for marker in ["bao nhieu", "may", "muc diem", "he 4", "so tin chi", "bao lau", "bao nhieu nam"]):
        return "exact_numeric"
    if any(marker in query_norm for marker in ["nhu the nao", "quy trinh", "thu tuc"]):
        return "procedure"
    if any(marker in query_norm for marker in ["vi sao", "neu", "roi thi", "xep loai", "bang gi", "sau do", "truong hop", "dong thoi"]):
        return "multi_hop"
    if any(marker in query_norm for marker in ["la gi", "nghia la gi", "duoc hieu la"]):
        return "definition"
    return "legal_lookup"


def _infer_required_hops(query_type: str, query: str) -> int:
    query_norm = normalize_ascii_text(query)
    hops = 1
    if query_type in {"multi_hop", "procedure"}:
        hops = 2
    if any(marker in query_norm for marker in ["neu", "sau do", "dong thoi", "truong hop", "ngoai tru"]):
        hops = max(hops, 2)
    if any(marker in query_norm for marker in ["ngoai le", "giam mot muc", "so voi", "co con"]):
        hops = max(hops, 3)
    return max(1, min(hops, 3))


def _extract_generic_semantic_anchors(original_query: str, effective_query: str) -> list[str]:
    raw = " ; ".join(part for part in [effective_query, original_query] if str(part or "").strip())
    fragments = re.split(r"[?;,]|\s+(?:va|hoac|neu|roi|thi)\s+", raw, flags=re.IGNORECASE)
    candidates = []
    for fragment in fragments:
        cleaned = re.sub(r"\s+", " ", str(fragment or "").strip(" .:-"))
        if len(cleaned.split()) < 2 or len(cleaned) < 8:
            continue
        candidates.append(cleaned)
    return _dedupe_text_list(candidates, max_items=6)


def _topic_specific_retrieval_guidance(original_query: str, effective_query: str, router_result: Dict[str, Any]) -> Dict[str, Any]:
    slots = dict((router_result or {}).get("slots", {}) or {})
    query_norm = normalize_ascii_text(original_query or effective_query)
    topic_focus = str(slots.get("topic_focus", "") or detect_query_topic_focus(query_norm)).strip().lower()

    if topic_focus == "exam_absence":
        return {
            "normalized_query": "Sinh viên vắng mặt buổi thi cuối kỳ hoặc đánh giá không có lý do chính đáng bị xử lý thế nào?",
            "abstract_query": "Quy định trực tiếp về sinh viên vắng mặt trong buổi thi hoặc đánh giá không có lý do chính đáng, phải nhận điểm 0 và trường hợp được thi bổ sung.",
            "pseudo_document": (
                "Đoạn tài liệu phù hợp cần nêu trực tiếp trường hợp sinh viên vắng mặt trong buổi thi hoặc đánh giá, "
                "không có lý do chính đáng thì phải nhận điểm 0; đồng thời nếu có lý do chính đáng thì có thể nộp minh chứng "
                "để được xem xét thi bổ sung."
            ),
            "query_type": "legal_lookup",
            "semantic_anchors": [
                "vắng mặt trong buổi thi",
                "không có lý do chính đáng",
                "phải nhận điểm 0",
                "thi bổ sung",
                "bỏ thi cuối kỳ",
            ],
            "must_include": [
                "vắng mặt trong buổi thi",
                "không có lý do chính đáng",
                "phải nhận điểm 0",
            ],
            "must_avoid": [
                "gian lận trong thi",
                "thi hộ",
                "xử lý kỷ luật",
                "buộc thôi học",
            ],
            "required_hops": 1,
            "answer_policy": "direct",
            "reason": "Topic-aware retrieval guidance for exam absence queries.",
            "confidence": 0.72,
        }

    if topic_focus == "scholarship":
        return {
            "abstract_query": "Quy định trực tiếp về tiêu chí xét học bổng, trường hợp đồng điểm và thứ tự ưu tiên khi quỹ học bổng có hạn.",
            "query_type": "legal_lookup",
            "semantic_anchors": [
                "xét học bổng",
                "đồng điểm",
                "thứ tự ưu tiên",
                "quỹ học bổng",
            ],
            "must_include": [
                "trường hợp bằng điểm",
                "thứ tự ưu tiên hoặc tiêu chí phụ",
            ],
            "must_avoid": [
                "miễn giảm học phí",
                "học phần",
                "thi cử",
            ],
            "required_hops": 1,
            "answer_policy": "direct",
            "reason": "Topic-aware retrieval guidance for scholarship tie-break queries.",
            "confidence": 0.62,
        }

    if topic_focus == "student_conduct":
        return {
            "abstract_query": "Quy định trực tiếp về hành vi phát ngôn xúc phạm, bình luận dung tục hoặc vi phạm chuẩn mực ứng xử của sinh viên trên mạng xã hội.",
            "query_type": "legal_lookup",
            "semantic_anchors": [
                "mạng xã hội",
                "facebook",
                "bình luận dung tục",
                "xúc phạm nhà trường",
                "hành vi vi phạm",
            ],
            "must_include": [
                "hành vi vi phạm",
                "chuẩn mực ứng xử hoặc kỷ luật",
            ],
            "must_avoid": [
                "thi cử",
                "điểm số",
                "học phần",
            ],
            "required_hops": 1,
            "answer_policy": "direct",
            "reason": "Topic-aware retrieval guidance for student conduct queries.",
            "confidence": 0.60,
        }

    return {}


def _build_generic_abstract_query(effective_query: str, query_type: str, required_hops: int) -> str:
    if query_type == "definition":
        prefix = "Khái niệm hoặc định nghĩa trong quy chế liên quan đến"
    elif query_type == "procedure":
        prefix = "Quy định, điều kiện và trình tự áp dụng liên quan đến"
    elif query_type == "duration_lookup":
        prefix = "Quy định về thời gian, giới hạn hoặc mốc áp dụng liên quan đến"
    elif query_type == "exact_numeric":
        prefix = "Quy định về ngưỡng, mốc định lượng hoặc điều kiện số liệu liên quan đến"
    elif required_hops >= 2:
        prefix = "Quy định, điều kiện, ngoại lệ và hệ quả liên quan đến"
    else:
        prefix = "Quy định trực tiếp liên quan đến"
    return f"{prefix}: {effective_query}".strip()[:500]


def _default_evidence_requirements(query_type: str, required_hops: int) -> list[str]:
    if query_type == "definition":
        items = ["định nghĩa hoặc phạm vi áp dụng"]
    elif query_type == "procedure":
        items = ["quy định chính", "điều kiện hoặc trình tự thực hiện"]
    elif query_type in {"duration_lookup", "exact_numeric"}:
        items = ["quy định chính", "ngưỡng hoặc mốc định lượng"]
    elif required_hops >= 2:
        items = ["quy định chính", "điều kiện hoặc ngoại lệ", "hệ quả hoặc kết luận áp dụng"]
    else:
        items = ["quy định chính"]
    return _dedupe_text_list(items, max_items=5)


def heuristic_retrieval_plan(original_query: str, effective_query: str, router_result: Dict[str, Any]) -> Dict[str, Any]:
    query_type = _infer_planner_query_type(original_query, router_result)
    required_hops = _infer_required_hops(query_type, original_query)
    answer_policy = "partial" if required_hops >= 2 else "direct"
    abstract_query = _build_generic_abstract_query(effective_query, query_type, required_hops)
    topic_guidance = _topic_specific_retrieval_guidance(original_query, effective_query, router_result)
    if topic_guidance:
        query_type = str(topic_guidance.get("query_type", query_type) or query_type)
        try:
            required_hops = int(topic_guidance.get("required_hops", required_hops))
        except Exception:
            required_hops = required_hops
        required_hops = max(1, min(required_hops, 3))
        answer_policy = str(topic_guidance.get("answer_policy", answer_policy) or answer_policy)
        abstract_query = str(topic_guidance.get("abstract_query", abstract_query) or abstract_query)
    pseudo_document = (
        "Đoạn tài liệu phù hợp cần nêu trực tiếp quy định liên quan đến câu hỏi, đồng thời thể hiện "
        "điều kiện áp dụng, ngoại lệ, hệ quả hoặc ngưỡng định lượng nếu có. "
        f"Trọng tâm truy xuất là: {abstract_query}"
    ).strip()[:800]
    if topic_guidance.get("pseudo_document"):
        pseudo_document = str(topic_guidance.get("pseudo_document", "") or pseudo_document).strip()[:800]

    semantic_anchors = _extract_generic_semantic_anchors(original_query, effective_query)
    must_include = _default_evidence_requirements(query_type, required_hops)
    must_avoid = []
    if topic_guidance:
        semantic_anchors = _dedupe_text_list(
            list(topic_guidance.get("semantic_anchors", []) or []) + semantic_anchors,
            max_items=6,
        )
        must_include = _dedupe_text_list(
            list(topic_guidance.get("must_include", []) or []) + must_include,
            max_items=6,
        )
        must_avoid = _dedupe_text_list(topic_guidance.get("must_avoid", []), max_items=6)

    return {
        "normalized_query": str(topic_guidance.get("normalized_query", "") or effective_query or original_query).strip()[:500],
        "abstract_query": abstract_query[:500],
        "pseudo_document": pseudo_document,
        "query_type": query_type,
        "semantic_anchors": semantic_anchors,
        "must_include": must_include,
        "must_avoid": must_avoid,
        "required_hops": required_hops,
        "answer_policy": answer_policy,
        "reason": str(topic_guidance.get("reason", "") or "Generic fallback retrieval plan."),
        "confidence": safe_float(topic_guidance.get("confidence", 0.35), 0.35),
    }


def validate_retrieval_plan(plan: Dict[str, Any], original_query: str, effective_query: str, router_result: Dict[str, Any]) -> Dict[str, Any]:
    fallback = heuristic_retrieval_plan(original_query, effective_query, router_result)
    topic_guidance = _topic_specific_retrieval_guidance(original_query, effective_query, router_result)
    plan = dict(plan or {})

    normalized_query = str(
        topic_guidance.get("normalized_query", "") or plan.get("normalized_query", "") or fallback["normalized_query"]
    ).strip()
    abstract_query = str(
        topic_guidance.get("abstract_query", "") or plan.get("abstract_query", "") or fallback["abstract_query"]
    ).strip()
    pseudo_document = str(
        topic_guidance.get("pseudo_document", "") or plan.get("pseudo_document", "") or fallback["pseudo_document"]
    ).strip()
    query_type = str(topic_guidance.get("query_type", "") or plan.get("query_type", "") or "").strip().lower() or fallback["query_type"]
    if query_type not in {"binary_legal", "exact_numeric", "duration_lookup", "multi_hop", "procedure", "definition", "legal_lookup"}:
        query_type = fallback["query_type"]

    answer_policy = str(topic_guidance.get("answer_policy", "") or plan.get("answer_policy", "") or "").strip().lower() or fallback["answer_policy"]
    if answer_policy not in {"direct", "partial", "abstain", "clarify"}:
        answer_policy = fallback["answer_policy"]

    try:
        required_hops = int(topic_guidance.get("required_hops", plan.get("required_hops", fallback["required_hops"])))
    except Exception:
        required_hops = fallback["required_hops"]
    required_hops = max(1, min(required_hops, 3))

    confidence = safe_float(plan.get("confidence", fallback["confidence"]), fallback["confidence"])
    confidence = max(0.0, min(confidence, 1.0))

    validated = {
        "normalized_query": normalized_query[:500],
        "abstract_query": abstract_query[:500],
        "pseudo_document": pseudo_document[:800],
        "query_type": query_type,
        "semantic_anchors": _dedupe_text_list(
            list(topic_guidance.get("semantic_anchors", []) or []) +
            list(plan.get("semantic_anchors", []) or []) +
            list(fallback["semantic_anchors"] or []),
            max_items=6,
        ),
        "must_include": _dedupe_text_list(
            list(topic_guidance.get("must_include", []) or []) +
            list(plan.get("must_include", []) or []) +
            list(fallback["must_include"] or []),
            max_items=6,
        ),
        "must_avoid": _dedupe_text_list(
            list(topic_guidance.get("must_avoid", []) or []) +
            list(plan.get("must_avoid", []) or []) +
            list(fallback["must_avoid"] or []),
            max_items=6,
        ),
        "required_hops": required_hops,
        "answer_policy": answer_policy,
        "reason": str(topic_guidance.get("reason", "") or plan.get("reason", "") or fallback["reason"]).strip()[:300],
        "confidence": confidence,
    }

    return validated


def build_retrieval_plan(original_query: str, effective_query: str, router_result: Dict[str, Any], llm_generate: Callable[..., str]) -> Dict[str, Any]:
    heuristic_plan = heuristic_retrieval_plan(original_query, effective_query, router_result)
    planner_seed = router_result.get("planner_seed")
    if isinstance(planner_seed, dict) and planner_seed:
        return validate_retrieval_plan(planner_seed, original_query, effective_query, router_result)

    prompt = PLANNER_PROMPT_TEMPLATE.format(
        original_query=original_query.strip(),
        effective_query=effective_query.strip(),
        router_result=json.dumps({
            "route": router_result.get("route"),
            "reason": router_result.get("reason"),
            "slots": router_result.get("slots", {}),
        }, ensure_ascii=False)
    )

    try:
        with _QUERY_LOCK:
            response = llm_generate(
                prompt,
                max_new_tokens=320,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                system_prompt=PLANNER_SYSTEM_PROMPT,
            )
        payload = extract_json_object(response) or {}
    except Exception:
        payload = {}

    if not payload:
        return heuristic_plan

    return validate_retrieval_plan(payload, original_query, effective_query, router_result)


def build_router_no_rag_response(message: str, router_result: Dict[str, Any]) -> Dict[str, Any]:
    answer = "Mình không cần tra cứu quy chế cho câu này. Bạn có thể hỏi trực tiếp một nội dung học vụ hoặc quy chế cụ thể."
    payload = build_basic_response(message, "help")
    payload["answer"] = answer
    payload["retrieval_metrics"]["route"] = "router_no_rag"
    payload["query_router"] = router_result
    return payload


def build_router_clarify_response(message: str, router_result: Dict[str, Any]) -> Dict[str, Any]:
    clarify = router_result.get("clarification_question") or "Bạn cho mình biết thêm hệ đào tạo hoặc khóa tuyển sinh cụ thể để mình tra đúng quy chế."
    payload = {
        "success": True,
        "error_message": None,
        "answer": clarify,
        "citations": [],
        "confidence": 100.0,
        "groundedness_score": 100.0,
        "provenance_score": 100.0,
        "revision_applied": False,
        "retrieval_time": 0.0,
        "synthesis_time": 0.0,
        "total_time": 0.0,
        "retrieval_metrics": {
            "route": "router_clarify",
            "intent": router_result.get("intent", "unknown"),
        },
        "retrieved_chunks": [],
        "selected_evidence": [],
        "evidence_spans": [],
        "claim_analyses": [],
        "quality_metrics": empty_quality_metrics(),
        "legal_references": [],
        "response_type": "clarify",
        "show_metrics": False,
        "show_references": False,
        "request": {
            "message": message,
        },
        "query_router": router_result,
    }
    return payload


def get_provider_defaults() -> Dict[str, Any]:
    settings = load_app_settings(PROJECT_ROOT)
    return build_public_app_settings(settings).get("providers", {})


def get_shared_settings() -> Dict[str, Any]:
    settings = load_app_settings(PROJECT_ROOT)
    return build_public_app_settings(settings)


def build_legal_reference(metadata: Dict[str, Any]) -> str:
    document_title = str(metadata.get("document_title", "") or "").strip()
    decision_number = str(metadata.get("decision_number", "") or "").strip()
    decision_code = str(metadata.get("decision_code", "") or "").strip()
    chapter = str(metadata.get("chapter", "") or "").strip()
    article = str(metadata.get("article", "") or "").strip()
    section = str(metadata.get("section", "") or "").strip()
    point = str(metadata.get("point", "") or "").strip()

    parts = []
    if document_title:
        if decision_number and decision_code:
            parts.append(f"{document_title} số {decision_number}/{decision_code}")
        else:
            parts.append(document_title)
    elif decision_number and decision_code:
        parts.append(f"Số {decision_number}/{decision_code}")
    if chapter:
        parts.append(chapter)
    if article:
        parts.append(article)
    if section:
        parts.append(section)
    if point:
        parts.append(point)
    return " | ".join(parts)


def collect_legal_references(payload: Dict[str, Any]) -> list[str]:
    references: list[str] = []

    def _append_ref(ref: str) -> None:
        if ref and ref not in references:
            references.append(ref)

    def _append(metadata: Dict[str, Any]) -> None:
        _append_ref(build_legal_reference(metadata or {}))

    duration_rule = ((payload.get("retrieval_metrics") or {}).get("duration_deadline_rule") or {})
    _append_ref(str(duration_rule.get("source_reference", "") or "").strip())
    if references:
        return references[:1]

    for citation in payload.get("citations", []) or []:
        citation_text = str(citation or "").strip()
        if citation_text and ("Điều" in citation_text or "Khoản" in citation_text or "Điểm" in citation_text):
            _append_ref(citation_text)

    for item in payload.get("evidence_spans", []) or []:
        _append(item.get("metadata", {}) or {})

    if references:
        return references[:5]

    for item in (payload.get("selected_evidence", []) or [])[:3]:
        _append(item.get("metadata", {}) or {})

    return references[:5]


def build_regulation_reference_from_metadata(metadata: Dict[str, Any]) -> str:
    metadata = metadata or {}
    document_title = str(metadata.get("document_title", "") or "").strip() or "Quy chế đào tạo"
    decision_number = str(metadata.get("decision_number", "") or "").strip()
    decision_code = str(metadata.get("decision_code", "") or "").strip()
    year = str(metadata.get("year", "") or "").strip()

    if decision_number and decision_code:
        return f"{document_title} số {decision_number}/{decision_code}"
    if year.isdigit():
        return f"{document_title} năm {year}"
    return document_title


def answer_has_abstention_signal(answer: str) -> bool:
    answer_norm = normalize_ascii_text(answer)
    return any(marker in answer_norm for marker in ABSTENTION_MARKERS)


def extract_duration_years(text: str) -> Optional[int]:
    normalized = normalize_ascii_text(text)
    match = re.search(r"\b(1?\d)\s*nam\b", normalized)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def extract_duration_candidate_from_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    metadata = item.get("metadata", {}) or {}
    text = str(item.get("text") or item.get("evidence") or item.get("claim") or "").strip()
    if not metadata and not text:
        return None

    table_kind = str(metadata.get("table_kind", "") or "").strip().lower()
    article = str(metadata.get("article", "") or "").strip()
    section = str(metadata.get("section", "") or "").strip()
    point = str(metadata.get("point", "") or "").strip()
    standard_duration_text = str(metadata.get("standard_duration", "") or "").strip()
    max_duration_text = str(metadata.get("max_duration", "") or "").strip()
    standard_years = extract_duration_years(standard_duration_text or text)
    max_years = extract_duration_years(max_duration_text or text)

    if max_years is None:
        return None

    return {
        "text": text,
        "metadata": metadata,
        "table_kind": table_kind,
        "article": article,
        "section": section,
        "point": point,
        "standard_years": standard_years,
        "max_years": max_years,
        "program_scope": str(metadata.get("program_scope", "") or "").strip().lower(),
        "score": safe_float(item.get("score", item.get("retrieval_score", item.get("composite_score", 0.0)))),
    }


def select_duration_deadline_candidate(
    payload: Dict[str, Any],
    slots: Dict[str, Any],
    runtime: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    requested_duration = slots.get("program_duration_years")
    requested_scope = str(slots.get("program_scope", "") or "").strip().lower() or "chinh_quy"

    raw_candidates = []
    for item in payload.get("selected_evidence", []) or []:
        raw_candidates.append(item)
    for item in payload.get("evidence_spans", []) or []:
        raw_candidates.append(item)
    for item in payload.get("retrieved_chunks", []) or []:
        raw_candidates.append(item)
    if runtime:
        for chunk in runtime.get("chunks", []) or []:
            metadata = getattr(chunk, "metadata", {}) or {}
            if str(metadata.get("table_kind", "") or "").strip().lower() != "duration_limit":
                continue
            raw_candidates.append({
                "text": getattr(chunk, "text_raw", "") or getattr(chunk, "text", "") or getattr(chunk, "text_contextualized", ""),
                "metadata": metadata,
                "score": 0.0,
            })

    best_candidate = None
    best_score = float("-inf")

    for raw_item in raw_candidates:
        candidate = extract_duration_candidate_from_item(raw_item or {})
        if not candidate:
            continue

        score = candidate["score"]
        if candidate["table_kind"] == "duration_limit":
            score += 2.0
        elif candidate["table_kind"] == "duration_limit_intro":
            score -= 0.5

        if candidate["article"] == "Điều 3":
            score += 0.6
        if candidate["section"] == "Khoản 6":
            score += 0.4
        if candidate["point"] in {"Điểm a", "Điểm b"}:
            score += 0.2

        if requested_scope:
            if candidate["program_scope"] == requested_scope:
                score += 0.5
            elif candidate["program_scope"] and candidate["program_scope"] != requested_scope:
                score -= 1.25

        if requested_duration:
            if candidate["standard_years"] == requested_duration:
                score += 1.2
            elif candidate["standard_years"] is not None and candidate["standard_years"] != requested_duration:
                score -= 1.5

        if "thoi gian hoc tap toi da" in normalize_ascii_text(candidate["text"]):
            score += 0.4

        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


def apply_duration_deadline_postprocess(
    payload: Dict[str, Any],
    router_result: Dict[str, Any],
    runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    slots = (router_result or {}).get("slots", {}) or {}
    target_fact = str(slots.get("target_fact", "") or "").strip()
    if target_fact != "thoi_gian_hoc_tap_toi_da":
        return payload

    candidate = select_duration_deadline_candidate(payload, slots, runtime=runtime)
    if not candidate:
        return payload

    max_years = candidate.get("max_years")
    if not max_years:
        return payload

    current_answer = str(payload.get("answer", "") or "").strip()
    current_confidence = safe_float(payload.get("confidence", 0.0))
    should_override = answer_has_abstention_signal(current_answer) or current_confidence < 65.0
    if not should_override:
        return payload

    metadata = candidate.get("metadata", {}) or {}
    regulation_ref = build_regulation_reference_from_metadata(metadata)
    requested_duration = slots.get("program_duration_years")
    cohort_year = slots.get("cohort_year")
    scope = candidate.get("program_scope") or str(slots.get("program_scope", "") or "").strip().lower() or "chinh_quy"
    scope_text = "theo hình thức chính quy" if scope == "chinh_quy" else "theo hình thức vừa làm vừa học"

    if requested_duration:
        prefix = f"Theo {regulation_ref}, đối với chương trình đào tạo {requested_duration} năm {scope_text}, thời gian học tập tối đa là {max_years} năm."
    else:
        prefix = f"Theo {regulation_ref}, thời gian học tập tối đa {scope_text} là {max_years} năm."

    if cohort_year:
        deadline_year = int(cohort_year) + int(max_years)
        answer = f"{prefix} Nếu tuyển sinh năm {cohort_year} thì hạn tối đa dự kiến là năm {deadline_year}."
    else:
        answer = prefix

    payload["answer"] = answer
    payload["response_type"] = "rag_duration_rule"
    payload["revision_applied"] = True
    payload["confidence"] = max(current_confidence, 0.78)
    payload["groundedness_score"] = max(safe_float(payload.get("groundedness_score", 0.0)), 0.78)
    payload["provenance_score"] = max(safe_float(payload.get("provenance_score", 0.0)), 0.78)
    payload = ensure_quality_metrics(payload)
    payload["quality_metrics"]["faithfulness_score"] = max(
        safe_float(payload["quality_metrics"].get("faithfulness_score", 0.0)),
        0.78,
    )
    payload["quality_metrics"]["citation_support_score"] = max(
        safe_float(payload["quality_metrics"].get("citation_support_score", 0.0)),
        0.78,
    )
    canonical_evidence = {
        "text": candidate.get("text", ""),
        "metadata": metadata,
        "score": 1.0,
        "selector_score": 1.0,
        "citation": build_legal_reference(metadata),
    }
    selected_evidence = list(payload.get("selected_evidence", []) or [])
    if not selected_evidence or (selected_evidence[0].get("metadata", {}) or {}).get("row_label") != metadata.get("row_label"):
        payload["selected_evidence"] = [canonical_evidence, *selected_evidence][:5]
    payload.setdefault("retrieval_metrics", {})
    payload["retrieval_metrics"]["duration_deadline_rule"] = {
        "applied": True,
        "max_years": int(max_years),
        "cohort_year": cohort_year,
        "deadline_year": int(cohort_year) + int(max_years) if cohort_year else None,
        "source_reference": build_legal_reference(metadata),
    }
    citation = build_legal_reference(metadata)
    if citation:
        existing_citations = list(payload.get("citations", []) or [])
        payload["citations"] = [citation, *[item for item in existing_citations if item != citation]][:5]
    return ensure_quality_metrics(payload)


def get_runtime_state() -> Dict[str, Any]:
    state = dict(_RUNTIME_STATE)
    state["cached_profiles"] = sorted(_RUNTIME_CACHE.keys())
    return state


def _contains_cuda_failure(text: Optional[str]) -> bool:
    normalized = str(text or "").lower()
    return "cuda" in normalized or "device-side assert" in normalized


def _cleanup_runtime_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def clear_processed_data_cache() -> Dict[str, Any]:
    from config import PROCESSED_DATA_PATH

    processed_root = Path(PROCESSED_DATA_PATH).resolve()
    project_root = PROJECT_ROOT.resolve()
    if processed_root == project_root or project_root not in processed_root.parents:
        raise RuntimeError(f"Refusing to clear unexpected processed_data path: {processed_root}")

    removed_files = []
    removed_dirs = []

    if not processed_root.exists():
        return {
            "processed_data_path": str(processed_root),
            "removed_files": removed_files,
            "removed_dirs": removed_dirs,
        }

    removable_file_names = {"registry.json"}
    removable_file_prefixes = ("faiss_index_",)
    removable_dir_markers = {"chunks.json", "document.json", "chapters.json", "articles.json", "sections.json"}

    for item in sorted(processed_root.iterdir(), key=lambda path: path.name.lower()):
        resolved_item = item.resolve()
        if processed_root not in resolved_item.parents and resolved_item != processed_root:
            continue

        if item.is_file():
            if item.name in removable_file_names or any(item.name.startswith(prefix) for prefix in removable_file_prefixes):
                item.unlink(missing_ok=True)
                removed_files.append(str(item))
            continue

        if item.is_dir():
            child_names = {child.name for child in item.iterdir() if child.is_file()}
            if child_names & removable_dir_markers:
                shutil.rmtree(item)
                removed_dirs.append(str(item))

    return {
        "processed_data_path": str(processed_root),
        "removed_files": removed_files,
        "removed_dirs": removed_dirs,
    }


def release_runtime(profile: Optional[str] = None) -> Dict[str, Any]:
    global _RUNTIME_CACHE

    with _RUNTIME_LOCK:
        profiles = [profile] if profile else list(_RUNTIME_CACHE.keys())
        profiles = [item for item in profiles if item in _RUNTIME_CACHE]
        if not profiles:
            state = dict(_RUNTIME_STATE)
            state["cached_profiles"] = sorted(_RUNTIME_CACHE.keys())
            return state

        for item in profiles:
            runtime = _RUNTIME_CACHE.pop(item, None)
            if isinstance(runtime, dict):
                heavy_keys = [
                    "rag_pipeline",
                    "answer_synthesizer",
                    "groundedness_evaluator",
                    "confidence_scorer",
                    "provenance_scorer",
                    "evidence_selector",
                    "context_pruner",
                    "embedder",
                    "llm_model",
                    "llm_tokenizer",
                    "reranker_stage2_model",
                    "reranker_fallback_model",
                    "semantic_highlight_model",
                    "semantic_highlight_tokenizer",
                    "mmr_qatc_model",
                    "mmr_qatc_tokenizer",
                    "hyde_retriever",
                    "hybrid_retriever",
                    "vector_retriever",
                    "bm25_retriever",
                    "bm25_raw_retriever",
                    "bm25_contextualized_retriever",
                    "vi_reranker",
                    "two_stage_reranker",
                    "mmr_selector",
                    "hierarchical_expander",
                    "chunks",
                    "faiss_index_a",
                    "faiss_index_b",
                    "bm25_index",
                    "bm25_index_raw",
                    "bm25_index_contextualized",
                ]
                for key in heavy_keys:
                    if key in runtime:
                        runtime[key] = None

        for module_name in _RUNTIME_MODULE_NAMES:
            sys.modules.pop(module_name, None)

        _cleanup_runtime_memory()

        if not _RUNTIME_CACHE:
            _RUNTIME_STATE.update({
                "loaded": False,
                "loading": False,
                "loaded_at": None,
                "load_seconds": None,
            })

        state = dict(_RUNTIME_STATE)
        state["cached_profiles"] = sorted(_RUNTIME_CACHE.keys())
        return state


def get_runtime(force_reload: bool = False, provider_name: Optional[str] = None) -> Dict[str, Any]:
    global _RUNTIME_CACHE

    shared_settings = get_shared_settings()
    resolved_provider = normalize_provider_name(
        provider_name or
        shared_settings.get("chat_defaults", {}).get("llm_provider") or
        shared_settings.get("providers", {}).get("default_provider") or
        "local"
    )
    runtime_profile = "local" if resolved_provider == "local" else "remote"
    remote_only = runtime_profile == "remote"

    with _RUNTIME_LOCK:
        stale_profiles = [item for item in _RUNTIME_CACHE.keys() if item != runtime_profile]
        if stale_profiles:
            for stale_profile in stale_profiles:
                runtime = _RUNTIME_CACHE.pop(stale_profile, None)
                if isinstance(runtime, dict):
                    runtime.clear()
            for module_name in _RUNTIME_MODULE_NAMES:
                sys.modules.pop(module_name, None)
            _cleanup_runtime_memory()

        if runtime_profile in _RUNTIME_CACHE and not force_reload:
            return _RUNTIME_CACHE[runtime_profile]

        if force_reload and runtime_profile in _RUNTIME_CACHE:
            _RUNTIME_CACHE.pop(runtime_profile, None)
            for module_name in _RUNTIME_MODULE_NAMES:
                sys.modules.pop(module_name, None)
            _cleanup_runtime_memory()

        _RUNTIME_STATE.update({
            "loaded": False,
            "loading": True,
            "error": None,
            "provider": resolved_provider,
            "runtime_profile": runtime_profile,
        })

        start = time.time()
        try:
            runtime = bootstrap_runtime(PROJECT_ROOT, remote_only=remote_only)
        except Exception as exc:
            _RUNTIME_CACHE.pop(runtime_profile, None)
            _RUNTIME_STATE.update({
                "loaded": False,
                "loading": False,
                "error": f"{type(exc).__name__}: {exc}",
                "loaded_at": None,
                "load_seconds": None,
                "provider": resolved_provider,
                "runtime_profile": runtime_profile,
            })
            raise

        _RUNTIME_CACHE[runtime_profile] = runtime
        _RUNTIME_STATE.update({
            "loaded": True,
            "loading": False,
            "error": None,
            "loaded_at": datetime.now().isoformat(timespec="seconds"),
            "load_seconds": round(time.time() - start, 2),
            "provider": resolved_provider,
            "runtime_profile": runtime_profile,
        })
        return _RUNTIME_CACHE[runtime_profile]


def serialize_result(result: Any, elapsed: float, debug_trace: bool = False) -> Dict[str, Any]:
    payload = {
        "success": bool(getattr(result, "success", False)),
        "error_message": getattr(result, "error_message", None),
        "answer": getattr(result, "answer", ""),
        "citations": list(getattr(result, "citations", []) or []),
        "confidence": safe_float(getattr(result, "confidence", 0.0)),
        "groundedness_score": safe_float(getattr(result, "groundedness_score", 0.0)),
        "provenance_score": safe_float(getattr(result, "provenance_score", 0.0)),
        "revision_applied": bool(getattr(result, "revision_applied", False)),
        "retrieval_time": safe_float(getattr(result, "retrieval_time", 0.0)),
        "synthesis_time": safe_float(getattr(result, "synthesis_time", 0.0)),
        "total_time": safe_float(getattr(result, "total_time", elapsed)),
        "retrieval_metrics": getattr(result, "retrieval_metrics", {}) or {},
        "retrieved_chunks": list(getattr(result, "retrieved_chunks", []) or []),
        "selected_evidence": list(getattr(result, "selected_evidence", []) or []),
        "evidence_spans": list(getattr(result, "evidence_spans", []) or []),
        "claim_analyses": list(getattr(result, "claim_analyses", []) or []),
        "quality_metrics": dict(getattr(result, "quality_metrics", {}) or {}),
        "response_type": "rag",
        "show_metrics": True,
        "show_references": True,
    }
    if debug_trace:
        payload["context_used"] = getattr(result, "context_used", "") or ""
    return ensure_quality_metrics(payload)


def build_llm_backend(
    runtime: Dict[str, Any],
    provider: str,
    remote_model: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    groq_api_key: Optional[str] = None,
) -> tuple[Callable[..., str], Dict[str, Any]]:
    provider_name = normalize_provider_name(provider)
    settings = load_provider_settings(PROJECT_ROOT)
    local_generate = runtime.get("generate_text") or runtime["rag_pipeline"].llm_generate

    if provider_name == "local":
        return local_generate, {
            "provider": "local",
            "label": "Local runtime",
        }

    if provider_name == "groq":
        groq_settings = settings.get("groq", {}) or {}
        model_name = str(remote_model or groq_settings.get("model") or "llama-3.3-70b-versatile").strip()
        api_key = str(groq_api_key or groq_settings.get("api_key") or "").strip()
        base_url = str(groq_settings.get("base_url") or "https://api.groq.com/openai/v1").strip()
        timeout_seconds = int(groq_settings.get("timeout_seconds") or 120)

        def _generate(
            prompt: str,
            max_new_tokens: int = 512,
            temperature: float = 0.3,
            top_p: float = 0.9,
            do_sample: bool = True,
            system_prompt: Optional[str] = None,
        ) -> str:
            return groq_generate(
                prompt,
                api_key=api_key,
                model=model_name,
                base_url=base_url,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                system_prompt=system_prompt,
                timeout_seconds=timeout_seconds,
            )

        return _generate, {
            "provider": "groq",
            "label": "Groq API",
            "model": model_name,
        }

    ollama_settings = settings.get("ollama", {}) or {}
    model_name = str(remote_model or ollama_settings.get("model") or "qwen2.5:7b-instruct").strip()
    base_url = normalize_ollama_base_url(
        ollama_base_url or ollama_settings.get("base_url"),
        "http://127.0.0.1:11434",
    )
    timeout_seconds = int(ollama_settings.get("timeout_seconds") or 600)

    def _generate(
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        do_sample: bool = True,
        system_prompt: Optional[str] = None,
    ) -> str:
        return ollama_generate(
            prompt,
            base_url=base_url,
            model=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
            timeout_seconds=timeout_seconds,
        )

    return _generate, {
        "provider": "ollama",
        "label": "Ollama",
        "model": model_name,
        "base_url": base_url,
    }


@contextmanager
def override_runtime_llm(runtime: Dict[str, Any], llm_generate: Callable[..., str]):
    pipeline = runtime["rag_pipeline"]
    previous_pipeline_generate = pipeline.llm_generate
    previous_answer_generate = pipeline.answer_synthesizer.llm_generate
    previous_groundedness_generate = pipeline.groundedness_evaluator.llm_generate
    hyde_retriever = runtime.get("hyde_retriever")
    previous_hyde_generate = getattr(hyde_retriever, "llm_generate", None) if hyde_retriever is not None else None

    pipeline.llm_generate = llm_generate
    pipeline.answer_synthesizer.llm_generate = llm_generate
    pipeline.groundedness_evaluator.llm_generate = llm_generate
    if hyde_retriever is not None:
        hyde_retriever.llm_generate = llm_generate
    try:
        yield
    finally:
        pipeline.llm_generate = previous_pipeline_generate
        pipeline.answer_synthesizer.llm_generate = previous_answer_generate
        pipeline.groundedness_evaluator.llm_generate = previous_groundedness_generate
        if hyde_retriever is not None:
            hyde_retriever.llm_generate = previous_hyde_generate


def run_query(
    message: str,
    retrieval_mode: str = "dual",
    top_k: int = 5,
    use_reranking: bool = True,
    use_mmr: bool = True,
    use_hierarchical_expansion: bool = True,
    use_evidence_selection: bool = True,
    use_semantic_highlighting: bool = True,
    llm_provider: str = "local",
    remote_model: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    debug_trace: bool = False,
) -> Dict[str, Any]:
    provider_name = normalize_provider_name(llm_provider)
    runtime_profile = "local" if provider_name == "local" else "remote"
    basic_intent = detect_basic_intent(message)
    if basic_intent:
        payload = build_basic_response(message, basic_intent)
        payload["request"].update({
            "llm_provider": provider_name,
            "remote_model": remote_model,
            "ollama_base_url": ollama_base_url,
            "use_semantic_highlighting": bool(use_semantic_highlighting),
        })
        payload["llm_backend"] = {
            "provider": provider_name,
        }
        return payload

    runtime = get_runtime(provider_name=provider_name)
    rag_pipeline = runtime["rag_pipeline"]
    RAGConfig = runtime["RAGConfig"]
    llm_generate, backend_meta = build_llm_backend(
        runtime,
        provider=llm_provider,
        remote_model=remote_model,
        ollama_base_url=ollama_base_url,
        groq_api_key=groq_api_key,
    )
    router_result = route_query_before_retrieval(message, llm_generate)

    if router_result.get("route") == "no_rag":
        payload = build_router_no_rag_response(message, router_result)
        payload["request"].update({
            "llm_provider": normalize_provider_name(llm_provider),
            "remote_model": remote_model,
            "ollama_base_url": ollama_base_url,
            "use_semantic_highlighting": bool(use_semantic_highlighting),
        })
        payload["llm_backend"] = backend_meta
        return payload

    if router_result.get("route") == "clarify":
        payload = build_router_clarify_response(message, router_result)
        payload["request"].update({
            "llm_provider": normalize_provider_name(llm_provider),
            "remote_model": remote_model,
            "ollama_base_url": ollama_base_url,
            "use_semantic_highlighting": bool(use_semantic_highlighting),
        })
        payload["llm_backend"] = backend_meta
        return payload

    effective_query = router_result.get("rewritten_query") or message
    retrieval_plan = build_retrieval_plan(message, effective_query, router_result, llm_generate)

    config = RAGConfig(
        retrieval_mode=(retrieval_mode or "dual").lower(),
        top_k_final=int(top_k),
        use_reranking=bool(use_reranking),
        use_mmr=bool(use_mmr),
        use_hierarchical_expansion=bool(use_hierarchical_expansion),
        use_evidence_selection=bool(use_evidence_selection),
        use_semantic_highlighting=bool(use_semantic_highlighting),
        debug_trace=bool(debug_trace),
        query_plan=retrieval_plan,
        verification_mode="selective" if provider_name == "local" else "full",
        groundedness_max_new_tokens=320 if provider_name == "local" else 900,
        max_revision_attempts=0 if provider_name == "local" else 1,
        enable_direct_answer_rewrite=False if provider_name == "local" else True,
        max_new_tokens=260 if provider_name == "local" else 500,
    )

    with _QUERY_LOCK:
        start = time.time()
        with override_runtime_llm(runtime, llm_generate):
            result = rag_pipeline.run(effective_query, config)
        elapsed = time.time() - start

    payload = serialize_result(result, elapsed, debug_trace=debug_trace)
    payload = apply_duration_deadline_postprocess(payload, router_result, runtime=runtime)
    payload["legal_references"] = collect_legal_references(payload)
    payload["llm_backend"] = backend_meta
    payload["query_router"] = router_result
    payload["query_plan"] = retrieval_plan
    payload["request"] = {
        "message": message,
        "effective_query": effective_query,
        "retrieval_mode": retrieval_mode,
        "top_k": int(top_k),
        "use_reranking": bool(use_reranking),
        "use_mmr": bool(use_mmr),
        "use_hierarchical_expansion": bool(use_hierarchical_expansion),
        "use_evidence_selection": bool(use_evidence_selection),
        "use_semantic_highlighting": bool(use_semantic_highlighting),
        "llm_provider": provider_name,
        "remote_model": remote_model,
        "ollama_base_url": ollama_base_url,
        "debug_trace": bool(debug_trace),
    }
    if _contains_cuda_failure(payload.get("error_message")):
        release_runtime(profile=runtime_profile)
        payload["runtime_recovery"] = {
            "released_after_cuda_error": True,
            "runtime_profile": runtime_profile,
        }
    return payload
