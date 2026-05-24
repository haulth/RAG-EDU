# ==============================================================================
# @title CELL 5 (ENHANCED): HYBRID RETRIEVAL WITH HYDE & RERANKING
# ==============================================================================

"""
CELL 5 - Enhanced Hybrid Retrieval with ViRanker + QATC MMR

TÍNH NĂNG:
 Hybrid Search (BM25 + Vector) - Kết hợp keyword và semantic
 HyDE (Hypothetical Document Embeddings) - Tăng recall
 ViRanker Reranking - Vietnamese cross-encoder reranking
 QATC MMR - Diversity penalty from SemViQA evidence extraction
 Metadata Filtering - Filter theo doc_type, year, category
 Logging & Monitoring - Track performance

IMPROVEMENTS:
- +15-25% recall với HyDE
- +20-30% precision với reranking
- +10-15% diversity với MMR
- Metadata-aware retrieval

PIPELINE:
Query → Hybrid (BM25+Vector) → Top 20
      → ViRanker Reranking → Top 10
      → Hierarchical section/article expansion
      → QATC MMR Diversity → Top 5
"""

print("="*70)
print(" CELL 5: HYBRID RETRIEVAL WITH VIRANKER + QATC MMR")
print("="*70)

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
from datetime import datetime
import re
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Retrieval parameters - UPDATED FOR OPTIMAL PIPELINE
USE_HYBRID = True        # Enable Hybrid search (BM25 + Vector) 
USE_HYDE = True          # Enable HyDE by default; query_plan pseudo_document is preferred when available
TOP_K_BM25 = 20          # Top K from BM25
TOP_K_VECTOR = 20        # Top K from Vector
TOP_K_HYBRID = 20        # Top K from Hybrid (increased for MMR)
TOP_K_HYDE = 20          # Top K from HyDE (if enabled)

# Reranking parameters - RUNS BEFORE HIERARCHICAL + MMR
ENABLE_RERANKING = True                    # Enable reranking

# ViRanker (Vietnamese-specific refinement)
RERANK_STAGE2_MODEL = "itdainb/vietnamese-cross-encoder"  # PhoRanker as fallback
RERANK_STAGE2_TOP_K = 5                    # Final reranked pool before answer synthesis
RERANK_MAX_LENGTH = 512                    # Unified requested pair length; each reranker clamps to its safe supported max

USE_FALLBACK = True                        # Use similarity fallback if models fail

# MMR parameters - RUNS AFTER RERANKING + HIERARCHICAL EXPANSION
USE_MMR = True           # Enable MMR diversity
MMR_LAMBDA = 0.7         # Balance relevance vs diversity (0-1)
TOP_K_MMR = 15           # Candidate pool preserved after reranking + hierarchical expansion, before MMR selection
MMR_QATC_MAX_CONTEXT_CHARS = 1800
MMR_QATC_MAX_ANSWER_TOKENS = 64
MMR_QATC_TOP_RATIONALE_TOKENS = 24
MMR_QATC_MIN_RATIONALE_SCORE = 0.12

# HyDE prompt template
HYDE_PROMPT_TEMPLATE = """Dựa trên câu hỏi sau, hãy viết một đoạn văn bản mô tả câu trả lời lý tưởng:

Câu hỏi: {query}

Đoạn văn mô tả câu trả lời:"""

# Fusion parameters
BM25_WEIGHT = 0.4        # Weight for BM25 scores
VECTOR_WEIGHT = 0.6      # Weight for Vector scores
BM25_RAW_SHARE = 0.45    # Share of BM25 weight for raw chunks
BM25_CONTEXT_SHARE = 0.55  # Share of BM25 weight for contextualized chunks

# Metadata filtering
ENABLE_METADATA_FILTER = False  # Enable metadata filtering
METADATA_FILTERS = {
    # 'doc_type': 'quy_che',
    # 'year': 2024,
    # 'category': 'Quy chế, Quy định đào tạo'
}

# Hierarchical retrieval
ENABLE_METADATA_PATH_BOOST = False      # Cleared to keep the main flow simpler
METADATA_PATH_BOOST_WEIGHT = 0.15
ENABLE_SCOPE_CONSISTENCY_PENALTY = False
SCOPE_MISMATCH_PENALTY = 0.18
ENABLE_HIERARCHICAL_EXPANSION = True    # Expand child hits to parent Khoản / Điều context
HIERARCHICAL_MAX_RESULTS = 5
DEBUG_TRACE_STAGE_LIMIT = 20
DEBUG_TRACE_TEXT_LIMIT = 1400

print(f"\n Configuration (OPTIMIZED PIPELINE):")
print(f"   • Hybrid Search: {'Enabled' if USE_HYBRID else 'Disabled'}")
print(f"   • HyDE: {'Enabled' if USE_HYDE else 'Disabled'}")
print(f"   • BM25 Top-K: {TOP_K_BM25}")
print(f"   • Vector Top-K: {TOP_K_VECTOR}")
print(f"   • Hybrid Top-K: {TOP_K_HYBRID}")
print(f"   • HyDE Top-K: {TOP_K_HYDE}")
print(f"   • MMR candidate pool (AFTER reranking + hierarchical): {TOP_K_MMR}")
print(f"   • ViRanker Top-K: {RERANK_STAGE2_TOP_K}")
print(f"   • MMR: {'Enabled' if USE_MMR else 'Disabled'}")
print(f"   • Metadata Filter: {'Enabled' if ENABLE_METADATA_FILTER else 'Disabled'}")
print(f"   • Metadata Path Boost: {'Enabled' if ENABLE_METADATA_PATH_BOOST else 'Disabled'}")
print(f"   • Scope Consistency Penalty: {'Enabled' if ENABLE_SCOPE_CONSISTENCY_PENALTY else 'Disabled'}")
print(f"   • Hierarchical Expansion: {'Enabled' if ENABLE_HIERARCHICAL_EXPANSION else 'Disabled'}")
print(f"   • BM25 Raw Weight: {BM25_WEIGHT * BM25_RAW_SHARE:.2f}")
print(f"   • BM25 Contextualized Weight: {BM25_WEIGHT * BM25_CONTEXT_SHARE:.2f}")
print(f"   • Vector Contextualized Weight: {VECTOR_WEIGHT:.2f}")


def _is_cuda_runtime_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    markers = (
        "cuda",
        "device-side assert",
        "cublas",
        "cudnn",
        "cuda error",
    )
    return any(marker in message for marker in markers)


def _best_effort_clear_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass

# Auto-detect mode
if USE_HYBRID and USE_HYDE:
    print(f"\n Mode: DUAL (Hybrid + HyDE)")
    print(f"    Pipeline: Hybrid+HyDE → Merge → ViRanker → Hierarchical → QATC MMR → Final")
elif not USE_HYBRID and USE_HYDE:
    print(f"\n Mode: HyDE ONLY")
    print(f"    Pipeline: HyDE → ViRanker → Hierarchical → QATC MMR → Final")
elif USE_HYBRID and not USE_HYDE:
    print(f"\n Mode: HYBRID ONLY")
    print(f"    Pipeline: Hybrid → ViRanker → Hierarchical → QATC MMR → Final")
else:
    print(f"\n  WARNING: Both Hybrid and HyDE are disabled!")
    print(f"   Please enable at least one retrieval method")


# ==============================================================================
# RETRIEVAL CLASSES
# ==============================================================================

@dataclass
class RetrievalResult:
    """Result from retrieval"""
    chunk_id: str
    text: str
    score: float
    metadata: Dict
    method: str  # 'bm25', 'vector', 'hybrid', 'reranked'
    raw_text: str = ""
    contextualized_text: str = ""
    parent_section_text: str = ""
    parent_article_text: str = ""


POINT_ORDER = {
    "a": 1, "b": 2, "c": 3, "d": 4, "đ": 5, "e": 6, "g": 7, "h": 8,
    "i": 9, "k": 10, "l": 11, "m": 12, "n": 13, "o": 14, "p": 15,
    "q": 16, "r": 17, "s": 18, "t": 19, "u": 20, "v": 21, "x": 22, "y": 23
}

ARTICLE_EXPANSION_MARKERS = [
    "xếp loại",
    "điểm tích lũy",
    "các mức",
    "bao nhiêu mức",
    "điều kiện",
    "bao gồm",
    "quy trình",
    "thủ tục"
]


def _safe_score(score: float) -> float:
    """Normalize retrieval / rerank scores to a comparable 0..1 range."""
    try:
        score = float(score)
    except Exception:
        return 0.0

    if 0.0 <= score <= 1.0:
        return score

    clipped = max(min(score, 12.0), -12.0)
    return float(1.0 / (1.0 + np.exp(-clipped)))


def _tokenize_for_overlap(text: str) -> set:
    """Lightweight tokenizer for lexical overlap scoring."""
    tokens = re.findall(r"\w+", (text or "").lower())
    return {tok for tok in tokens if len(tok) > 1}


def _tokenize_for_weighted_overlap(text: str) -> List[str]:
    """Tokenize text for weighted semantic-overlap scoring."""
    return [
        token
        for token in re.findall(r"\w+", (text or "").lower())
        if len(token) > 1 and not token.isdigit()
    ]


def _split_mmr_sentences(text: str) -> List[str]:
    """Best-effort sentence split when semantic-highlight output is sparse."""
    if not text:
        return []

    parts = re.split(r"(?<=[\.\!\?;:\n])\s+", str(text).strip())
    sentences = [part.strip() for part in parts if str(part or "").strip()]
    return sentences or [str(text).strip()]


def _flatten_scalar(value: Any) -> float:
    """Extract a single float from nested payloads."""
    while isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        value = value[0]

    try:
        return float(value)
    except Exception:
        return 0.0


def _clamp_probability(value: Any) -> float:
    """Clamp semantic probabilities to 0..1."""
    try:
        return float(max(0.0, min(1.0, float(value))))
    except Exception:
        return 0.0


def _weighted_jaccard_similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
    """Weighted overlap used as the MMR redundancy penalty."""
    if not left or not right:
        return 0.0

    keys = set(left) | set(right)
    overlap = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    union = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
    if union <= 0:
        return 0.0
    return float(max(0.0, min(1.0, overlap / union)))


class SemViQAMMRBackend:
    """Build query-conditioned evidence signatures for MMR via SemViQA QATC."""

    def __init__(
        self,
        model=None,
        tokenizer=None,
        device: Optional[str] = None,
        max_context_chars: int = MMR_QATC_MAX_CONTEXT_CHARS,
        max_answer_tokens: int = MMR_QATC_MAX_ANSWER_TOKENS,
        top_rationale_tokens: int = MMR_QATC_TOP_RATIONALE_TOKENS,
        min_rationale_score: float = MMR_QATC_MIN_RATIONALE_SCORE,
    ):
        self.model = model if model is not None else globals().get("mmr_qatc_model")
        self.tokenizer = tokenizer if tokenizer is not None else globals().get("mmr_qatc_tokenizer")
        self.max_context_chars = max_context_chars
        self.max_answer_tokens = max_answer_tokens
        self.top_rationale_tokens = top_rationale_tokens
        self.min_rationale_score = min_rationale_score
        self.is_available = self.model is not None and self.tokenizer is not None
        self.backend_name = (
            "semviqa_qatc_infoxlm_viwikifc"
            if self.is_available
            else "lexical_signature_fallback"
        )
        if device is not None:
            self.device = str(device)
        elif self.model is not None:
            try:
                self.device = str(next(self.model.parameters()).device)
            except Exception:
                self.device = "cpu"
        else:
            self.device = "cpu"
        self._cache = {}

    def _get_context_indices(self, encoding) -> List[int]:
        try:
            sequence_ids = encoding.sequence_ids(0)
        except Exception:
            return []
        return [idx for idx, sequence_id in enumerate(sequence_ids) if sequence_id == 1]

    def _normalize_offsets(self, offsets: Any) -> List[Tuple[int, int]]:
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        normalized = []
        for start, end in list(offsets or []):
            normalized.append((int(start), int(end)))
        return normalized

    def _extract_context_text(
        self,
        context_text: str,
        offsets: List[Tuple[int, int]],
        token_indices: List[int],
    ) -> str:
        start_char = None
        end_char = None
        for token_idx in token_indices:
            if token_idx < 0 or token_idx >= len(offsets):
                continue
            token_start, token_end = offsets[token_idx]
            if token_end <= token_start:
                continue
            start_char = token_start if start_char is None else min(start_char, token_start)
            end_char = token_end if end_char is None else max(end_char, token_end)
        if start_char is None or end_char is None:
            return ""
        return str(context_text or "")[start_char:end_char].strip()

    def _best_span(
        self,
        start_logits,
        end_logits,
        context_indices: List[int],
    ) -> Tuple[int, int]:
        if not context_indices:
            return 0, 0

        ranked_starts = sorted(
            context_indices,
            key=lambda idx: float(start_logits[idx].item()),
            reverse=True,
        )[:8]
        ranked_ends = sorted(
            context_indices,
            key=lambda idx: float(end_logits[idx].item()),
            reverse=True,
        )[:8]

        best_score = None
        best_pair = (context_indices[0], context_indices[0])
        for start_idx in ranked_starts:
            for end_idx in ranked_ends:
                if end_idx < start_idx:
                    continue
                if end_idx - start_idx + 1 > self.max_answer_tokens:
                    continue
                pair_score = float(start_logits[start_idx].item() + end_logits[end_idx].item())
                if best_score is None or pair_score > best_score:
                    best_score = pair_score
                    best_pair = (start_idx, end_idx)

        if best_score is not None:
            return best_pair

        fallback_start = max(context_indices, key=lambda idx: float(start_logits[idx].item()))
        fallback_end_candidates = [
            idx for idx in context_indices
            if idx >= fallback_start and idx - fallback_start + 1 <= self.max_answer_tokens
        ]
        if not fallback_end_candidates:
            return fallback_start, fallback_start
        fallback_end = max(
            fallback_end_candidates,
            key=lambda idx: float(end_logits[idx].item()),
        )
        return fallback_start, fallback_end

    def build_signature(self, query: str, text: str) -> Dict[str, Any]:
        truncated_text = str(text or "").strip()[:self.max_context_chars]
        cache_key = (str(query or "").strip(), truncated_text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if not truncated_text:
            empty_signature = {
                "token_weights": {},
                "model_score": 0.0,
                "highlighted_sentence_count": 0,
                "backend": self.backend_name,
                "evidence_text": "",
                "rationale_token_count": 0,
            }
            self._cache[cache_key] = empty_signature
            return empty_signature

        if not self.is_available:
            lexical_signature = {
                "token_weights": {
                    token: 0.1 for token in _tokenize_for_weighted_overlap(truncated_text)
                },
                "model_score": 0.0,
                "highlighted_sentence_count": 0,
                "backend": "lexical_signature_fallback",
                "evidence_text": "",
                "rationale_token_count": 0,
            }
            self._cache[cache_key] = lexical_signature
            return lexical_signature

        try:
            encoding = self.tokenizer(
                query,
                truncated_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                return_offsets_mapping=True,
            )
            context_indices = self._get_context_indices(encoding)
            offsets = self._normalize_offsets(encoding["offset_mapping"][0])
            model_inputs = {
                key: value.to(self.device)
                for key, value in encoding.items()
                if key != "offset_mapping"
            }

            import torch

            with torch.no_grad():
                outputs = self.model(**model_inputs)

            start_logits = outputs.start_logits[0].detach().cpu()
            end_logits = outputs.end_logits[0].detach().cpu()
            rational_scores = outputs.rational_tag_logits[0].detach().cpu().squeeze(-1)

            best_start, best_end = self._best_span(start_logits, end_logits, context_indices)
            evidence_text = self._extract_context_text(
                truncated_text,
                offsets,
                list(range(best_start, best_end + 1)),
            )

            if context_indices:
                context_start_logits = start_logits[context_indices]
                context_end_logits = end_logits[context_indices]
                start_probs = torch.softmax(context_start_logits, dim=0)
                end_probs = torch.softmax(context_end_logits, dim=0)
                start_local_idx = context_indices.index(best_start)
                end_local_idx = context_indices.index(best_end)
                answer_confidence = float(
                    ((start_probs[start_local_idx] + end_probs[end_local_idx]) / 2.0).item()
                )
            else:
                answer_confidence = 0.0

            token_weights: Dict[str, float] = {}
            if evidence_text:
                for token in _tokenize_for_weighted_overlap(evidence_text):
                    token_weights[token] = max(token_weights.get(token, 0.0), answer_confidence)

            ranked_rationales = sorted(
                (
                    (
                        token_idx,
                        _clamp_probability(rational_scores[token_idx].item()),
                    )
                    for token_idx in context_indices
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            kept_rationales = [
                (token_idx, score)
                for token_idx, score in ranked_rationales
                if score >= self.min_rationale_score
            ][: self.top_rationale_tokens]
            if not kept_rationales:
                kept_rationales = ranked_rationales[: self.top_rationale_tokens]

            for token_idx, score in kept_rationales:
                token_text = self._extract_context_text(truncated_text, offsets, [token_idx])
                for token in _tokenize_for_weighted_overlap(token_text):
                    token_weights[token] = max(token_weights.get(token, 0.0), float(score))

            if not token_weights:
                for token in _tokenize_for_weighted_overlap(truncated_text):
                    token_weights[token] = max(token_weights.get(token, 0.0), 0.1)

            signature = {
                "token_weights": token_weights,
                "model_score": answer_confidence,
                "highlighted_sentence_count": 1 if evidence_text else 0,
                "backend": self.backend_name,
                "evidence_text": evidence_text,
                "rationale_token_count": len(kept_rationales),
            }
            self._cache[cache_key] = signature
            return signature
        except Exception as exc:
            print(f" QATC MMR analysis failed, using lexical signature fallback: {exc}")
            fallback_signature = {
                "token_weights": {
                    token: 0.1 for token in _tokenize_for_weighted_overlap(truncated_text)
                },
                "model_score": 0.0,
                "highlighted_sentence_count": 0,
                "backend": "lexical_signature_fallback",
                "evidence_text": "",
                "rationale_token_count": 0,
            }
            self._cache[cache_key] = fallback_signature
            return fallback_signature


def token_overlap(left: str, right: str) -> float:
    """Soft lexical overlap used only for planner include/avoid guidance."""
    left_tokens = {tok for tok in re.findall(r"\w+", (left or "").lower()) if len(tok) > 2}
    right_tokens = {tok for tok in re.findall(r"\w+", (right or "").lower()) if len(tok) > 2}

    if not left_tokens or not right_tokens:
        return 0.0

    overlap = len(left_tokens & right_tokens)
    return overlap / max(1, min(len(left_tokens), len(right_tokens)))


def _is_cpa_threshold_query(query: str) -> bool:
    query_lower = (query or "").lower()
    cpa_terms = [
        "đtb", "dtb", "cpa",
        "điểm trung bình", "diem trung binh",
        "tích lũy", "tich luy"
    ]
    threshold_terms = [
        "bao nhiêu", "bao nhieu",
        "tối thiểu", "toi thieu",
        "ít nhất", "it nhat",
        "mấy", "ngưỡng", "nguong", "mức", "muc"
    ]
    return any(term in query_lower for term in cpa_terms) and any(term in query_lower for term in threshold_terms)


def _rerank_query_bonus(query: str, rerank_text: str) -> float:
    """Apply narrow lexical bonuses when the query asks for a score threshold."""
    if not _is_cpa_threshold_query(query):
        return 0.0

    text_lower = (rerank_text or "").lower()
    has_score_terms = any(term in text_lower for term in [
        "điểm trung bình tích lũy",
        "diem trung binh tich luy",
        "tích lũy",
        "tich luy",
        "cpa"
    ])
    has_threshold_terms = any(term in text_lower for term in [
        "từ ",
        "tu ",
        "trở lên",
        "tro len",
        "dưới",
        "duoi",
        "ít nhất",
        "it nhat",
        "tối thiểu",
        "toi thieu"
    ])

    bonus = 0.6 * token_overlap(query, rerank_text)
    bonus += 2.0 if has_score_terms else -0.8
    if has_threshold_terms:
        bonus += 0.8
    return bonus


def _tokenize_for_bm25(text: str) -> List[str]:
    """Tokenizer shared by raw/contextualized BM25 retrievers."""
    return [tok for tok in re.findall(r"\w+", (text or "").lower()) if tok]


def _chunk_raw_text(chunk) -> str:
    """Return the canonical raw chunk text."""
    return getattr(chunk, "text_raw", "") or getattr(chunk, "text", "")


def _chunk_contextualized_text(chunk) -> str:
    """Return contextualized chunk text when available."""
    contextualized = getattr(chunk, "text_contextualized", "") or ""
    if contextualized:
        return contextualized
    return build_contextualized_text(_chunk_raw_text(chunk), getattr(chunk, "metadata", {}))


def _chunk_parent_section_text(chunk) -> str:
    return getattr(chunk, "parent_section_text", "") or ""


def _chunk_parent_article_text(chunk) -> str:
    return getattr(chunk, "parent_article_text", "") or ""


def _trim_debug_text(text: str, limit: int = DEBUG_TRACE_TEXT_LIMIT) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _build_debug_citation(metadata: Dict) -> str:
    metadata = metadata or {}
    title = str(metadata.get("document_title", "") or "").strip()
    decision_number = str(metadata.get("decision_number", "") or "").strip()
    decision_code = str(metadata.get("decision_code", "") or "").strip()
    filename = str(metadata.get("filename", "") or "").strip()

    if title and decision_number and decision_code:
        head = f"{title} số {decision_number}/{decision_code}"
    elif title:
        head = title
    elif filename:
        head = filename
    else:
        head = "Nguồn trích dẫn"

    trail = []
    for key, prefix in (("page", "trang "), ("chapter", ""), ("article", ""), ("section", ""), ("point", "")):
        value = str(metadata.get(key, "") or "").strip()
        if value:
            trail.append(f"{prefix}{value}" if prefix else value)

    return " | ".join([head, *trail] if trail else [head])


def _serialize_debug_result(result: RetrievalResult, rank: int) -> Dict:
    metadata = dict(getattr(result, "metadata", {}) or {})
    text = getattr(result, "text", "") or ""
    raw_text = getattr(result, "raw_text", "") or ""
    contextualized_text = getattr(result, "contextualized_text", "") or ""
    rerank_view = build_compact_rerank_text(result)

    payload = {
        "rank": int(rank),
        "chunk_id": str(getattr(result, "chunk_id", "") or ""),
        "score": float(getattr(result, "score", 0.0) or 0.0),
        "method": str(getattr(result, "method", "") or ""),
        "citation": _build_debug_citation(metadata),
        "metadata": metadata,
        "text_preview": _trim_debug_text(text),
    }

    if raw_text and raw_text.strip() != text.strip():
        payload["raw_text_preview"] = _trim_debug_text(raw_text)
    if contextualized_text and contextualized_text.strip() != text.strip():
        payload["contextualized_text_preview"] = _trim_debug_text(contextualized_text)
    if (
        rerank_view and
        rerank_view.strip() != text.strip() and
        rerank_view.strip() != contextualized_text.strip()
    ):
        payload["rerank_text_preview"] = _trim_debug_text(rerank_view)

    return payload


def _record_debug_stage(metrics: Dict, stage_key: str, label: str, results: List[RetrievalResult], note: str = "") -> None:
    debug_trace = metrics.setdefault("debug_trace", {})
    stage_results = list(results or [])
    debug_trace[stage_key] = {
        "label": label,
        "count": len(stage_results),
        "items": [
            _serialize_debug_result(result, rank=index + 1)
            for index, result in enumerate(stage_results)
        ],
    }
    if note:
        debug_trace[stage_key]["note"] = note


def _metadata_path_text(metadata: Dict) -> str:
    """Build searchable structural text from metadata."""
    if not metadata:
        return ""

    parts = [
        metadata.get("chapter", ""),
        metadata.get("article", ""),
        metadata.get("article_title", ""),
        metadata.get("section", ""),
        metadata.get("point", ""),
        metadata.get("hierarchical_path", "")
    ]
    return " ".join(part for part in parts if part)


def _metadata_match_score(query: str, metadata: Dict) -> float:
    """Score how well query tokens align with legal structure metadata."""
    query_tokens = _tokenize_for_overlap(query)
    metadata_tokens = _tokenize_for_overlap(_metadata_path_text(metadata))

    if not query_tokens or not metadata_tokens:
        return 0.0

    overlap = len(query_tokens & metadata_tokens) / max(1, len(query_tokens))
    bonus = overlap

    query_lower = (query or "").lower()
    title_lower = (metadata.get("article_title", "") or "").lower()

    if "điểm tích lũy" in query_lower and "xếp loại" in query_lower and "xếp loại" in title_lower:
        bonus += 0.2
    if metadata.get("point") and metadata.get("point", "").lower() in query_lower:
        bonus += 0.1
    if metadata.get("section") and metadata.get("section", "").lower() in query_lower:
        bonus += 0.1
    if metadata.get("article") and metadata.get("article", "").lower() in query_lower:
        bonus += 0.1

    return min(bonus, 1.0)


def build_contextualized_text(text: str, metadata: Dict, anchor_text: str = "") -> str:
    """Attach legal path/title to text so rerankers see the surrounding structure."""
    metadata = metadata or {}

    header_parts = []
    if metadata.get("chapter"):
        header_parts.append(metadata["chapter"])
    if metadata.get("hierarchical_path"):
        header_parts.append(metadata["hierarchical_path"])
    elif metadata.get("article"):
        header_parts.append(metadata["article"])

    header = "\n".join(part for part in header_parts if part).strip()
    anchor_block = ""
    if anchor_text:
        anchor_block = f"[ANCHOR]\n{anchor_text.strip()}\n"

    body = (text or "").strip()
    blocks = [block for block in [header, anchor_block.strip(), body] if block]
    return "\n\n".join(blocks).strip()


def build_contextualized_result_text(result: RetrievalResult) -> str:
    """Contextualized text view for reranking / MMR."""
    if getattr(result, "contextualized_text", ""):
        return result.contextualized_text
    base_text = getattr(result, "raw_text", "") or result.text
    return build_contextualized_text(base_text, result.metadata)


def build_compact_rerank_text(result: RetrievalResult) -> str:
    """Compact anchor-first text view for cross-encoder reranking."""
    metadata = getattr(result, "metadata", {}) or {}
    anchor_text = (getattr(result, "raw_text", "") or getattr(result, "text", "") or "").strip()

    if not anchor_text:
        fallback_text = getattr(result, "contextualized_text", "") or ""
        return fallback_text.strip()

    return build_contextualized_text(anchor_text, metadata)


def _normalize_supported_max_length(value) -> int:
    try:
        limit = int(value)
    except Exception:
        return 0

    if limit <= 0:
        return 0

    # Hugging Face often uses a huge sentinel when the tokenizer has no hard limit.
    if limit >= 100000:
        return 0

    return limit


def resolve_safe_rerank_max_length(tokenizer, model, requested_length: int = RERANK_MAX_LENGTH) -> int:
    """Clamp the requested rerank length to the actual supported pair length."""
    candidates = [_normalize_supported_max_length(requested_length) or 512]

    tokenizer_limit = _normalize_supported_max_length(getattr(tokenizer, "model_max_length", None))
    if tokenizer_limit:
        candidates.append(tokenizer_limit)

    config = getattr(model, "config", None)
    position_limit = _normalize_supported_max_length(getattr(config, "max_position_embeddings", None))
    if position_limit:
        special_tokens = 0
        try:
            special_tokens = int(tokenizer.num_special_tokens_to_add(pair=True))
        except Exception:
            special_tokens = 0
        position_limit = max(32, position_limit - max(0, special_tokens))
        candidates.append(position_limit)

    return max(32, min(candidates))


def apply_metadata_path_boost(query: str, results: List[RetrievalResult], weight: float = METADATA_PATH_BOOST_WEIGHT) -> List[RetrievalResult]:
    """Slightly boost chunks whose legal path/title aligns with the query."""
    if not results:
        return results

    for result in results:
        bonus = _metadata_match_score(query, result.metadata)
        if bonus <= 0:
            continue

        base = _safe_score(result.score)
        result.score = min(1.0, base + weight * bonus)
        result.metadata["path_match_score"] = bonus

    results.sort(key=lambda item: item.score, reverse=True)
    return results


def _infer_program_scope_from_query(query: str) -> str:
    query_lower = (query or "").lower()
    if "vừa làm vừa học" in query_lower or "vua lam vua hoc" in query_lower:
        return "vlvh"
    if "chính quy" in query_lower or "chinh quy" in query_lower:
        return "chinh_quy"
    return "default_chinh_quy"


def _infer_program_scope_from_metadata(metadata: Dict) -> str:
    explicit_scope = str(metadata.get("program_scope", "") or "").strip().lower()
    if explicit_scope in {"chinh_quy", "vlvh"}:
        return explicit_scope

    title_lower = (metadata.get("article_title", "") or "").lower()
    if "vừa làm vừa học" in title_lower or "vua lam vua hoc" in title_lower:
        return "vlvh"
    if "chính quy" in title_lower or "chinh quy" in title_lower:
        return "chinh_quy"
    return "neutral"


def apply_scope_consistency_penalty(
    query: str,
    results: List[RetrievalResult],
    penalty: float = SCOPE_MISMATCH_PENALTY
) -> List[RetrievalResult]:
    if not results:
        return results

    query_scope = _infer_program_scope_from_query(query)
    for result in results:
        metadata = result.metadata or {}
        metadata_scope = _infer_program_scope_from_metadata(metadata)
        base = _safe_score(result.score)

        if query_scope in {"default_chinh_quy", "chinh_quy"} and metadata_scope == "vlvh":
            result.score = max(0.0, base - penalty)
            metadata["scope_penalty"] = penalty
            metadata["scope_query_scope"] = query_scope
            metadata["scope_metadata_scope"] = metadata_scope
        elif query_scope == "vlvh" and metadata_scope == "chinh_quy":
            result.score = max(0.0, base - penalty)
            metadata["scope_penalty"] = penalty
            metadata["scope_query_scope"] = query_scope
            metadata["scope_metadata_scope"] = metadata_scope

    results.sort(key=lambda item: item.score, reverse=True)
    return results


def _query_plan_blueprint(query: str, query_plan: Optional[Dict]) -> str:
    query_plan = query_plan or {}
    parts = [
        str(query_plan.get("normalized_query", "") or "").strip(),
        str(query_plan.get("abstract_query", "") or "").strip(),
        str(query_plan.get("pseudo_document", "") or "").strip(),
        " ; ".join(str(item or "").strip() for item in list(query_plan.get("semantic_anchors", []) or []) if str(item or "").strip()),
        " ; ".join(str(item or "").strip() for item in list(query_plan.get("must_include", []) or []) if str(item or "").strip()),
    ]
    fallback = (query or "").strip()
    blueprint = "\n".join(part for part in parts if part).strip()
    return blueprint or fallback


def _query_plan_avoid_blueprint(query_plan: Optional[Dict]) -> str:
    query_plan = query_plan or {}
    parts = [
        str(item or "").strip()
        for item in list(query_plan.get("must_avoid", []) or [])
        if str(item or "").strip()
    ]
    return "\n".join(parts).strip()


def apply_query_plan_guidance(
    query: str,
    results: List[RetrievalResult],
    query_plan: Optional[Dict],
    weight: float = 0.18
) -> List[RetrievalResult]:
    if not results or not query_plan:
        return results

    focus_blueprint = _query_plan_blueprint(query, query_plan)
    avoid_blueprint = _query_plan_avoid_blueprint(query_plan)

    focus_emb = None
    avoid_emb = None
    if 'embedder' in globals():
        try:
            focus_emb = embedder.encode(focus_blueprint[:1400], convert_to_tensor=True).cpu().numpy()
        except Exception:
            focus_emb = None
        if avoid_blueprint:
            try:
                avoid_emb = embedder.encode(avoid_blueprint[:1200], convert_to_tensor=True).cpu().numpy()
            except Exception:
                avoid_emb = None

    for result in results:
        candidate_text = build_contextualized_result_text(result)
        metadata_text = _metadata_path_text(result.metadata or {})
        combined_text = f"{metadata_text}\n{candidate_text}".strip()

        focus_alignment = 0.0
        avoid_alignment = 0.0
        if focus_emb is not None and combined_text:
            try:
                candidate_emb = embedder.encode(combined_text[:1200], convert_to_tensor=True).cpu().numpy()
                focus_alignment = float(np.dot(focus_emb, candidate_emb) / (
                    np.linalg.norm(focus_emb) * np.linalg.norm(candidate_emb)
                ))
                focus_alignment = max(0.0, min(1.0, focus_alignment))
                if avoid_emb is not None:
                    avoid_alignment = float(np.dot(avoid_emb, candidate_emb) / (
                        np.linalg.norm(avoid_emb) * np.linalg.norm(candidate_emb)
                    ))
                    avoid_alignment = max(0.0, min(1.0, avoid_alignment))
            except Exception:
                focus_alignment = 0.0
                avoid_alignment = 0.0

        base = _safe_score(result.score)
        planner_alignment = max(0.0, min(1.0, 0.85 * focus_alignment - 0.35 * avoid_alignment))
        adjusted = base + weight * planner_alignment - 0.08 * avoid_alignment
        result.score = max(0.0, min(1.0, adjusted))
        result.metadata["planner_alignment"] = planner_alignment
        result.metadata["planner_focus_alignment"] = focus_alignment
        result.metadata["planner_avoid_penalty"] = avoid_alignment
        result.metadata["planner_query_type"] = str(query_plan.get("query_type", "") or "")
        result.metadata["planner_required_hops"] = int(query_plan.get("required_hops", 1) or 1)

    results.sort(key=lambda item: item.score, reverse=True)
    return results


class HierarchicalContextExpander:
    """Expand top child hits into their parent Khoản / Điều context."""

    def __init__(self, chunks: List, article_full_text_map: Dict = None):
        self.chunks = chunks
        self.article_full_text_map = article_full_text_map or {}
        self.section_context_map = {}
        self.article_context_map = {}
        self.section_metadata_map = {}
        self.article_metadata_map = {}
        self._build_context_maps()

    def _build_context_maps(self) -> None:
        for chunk in self.chunks:
            metadata = chunk.metadata
            article_key = self._article_key(metadata)
            section_key = self._section_key(metadata)

            self.article_context_map.setdefault(article_key, []).append(chunk)
            self.article_metadata_map.setdefault(article_key, metadata)

            if section_key:
                self.section_context_map.setdefault(section_key, []).append(chunk)
                self.section_metadata_map.setdefault(section_key, metadata)

    def expand(self, query: str, results: List[RetrievalResult], top_k: int = HIERARCHICAL_MAX_RESULTS) -> Tuple[List[RetrievalResult], Dict]:
        if not results:
            return [], {
                "hierarchical_expansion_applied": False,
                "hierarchical_anchored_results": 0,
                "hierarchical_max_score": 0.0,
                "hierarchical_scope_counts": {}
            }

        expanded_results = []
        seen_group_keys = set()
        scope_counts = {"section": 0, "article": 0, "chunk": 0}

        for result in results:
            scope = self._infer_scope(query, result.metadata)
            group_key = self._group_key(result.metadata, scope)
            if group_key in seen_group_keys:
                continue

            expanded = self._expand_result(query, result, scope)
            expanded_results.append(expanded)
            seen_group_keys.add(group_key)
            scope_counts[scope] = scope_counts.get(scope, 0) + 1

        expanded_results.sort(key=lambda item: item.score, reverse=True)
        expanded_results = expanded_results[:top_k]

        max_score = max((_safe_score(item.score) for item in expanded_results), default=0.0)
        return expanded_results, {
            "hierarchical_expansion_applied": True,
            "hierarchical_anchored_results": len(expanded_results),
            "hierarchical_max_score": max_score,
            "hierarchical_scope_counts": scope_counts
        }

    def _expand_result(self, query: str, result: RetrievalResult, scope: str) -> RetrievalResult:
        metadata = dict(result.metadata or {})
        anchor_text = getattr(result, "raw_text", "") or result.text

        if scope == "section":
            parent_text = self._section_text(metadata)
        elif scope == "article":
            parent_text = self._article_text(metadata)
        else:
            parent_text = anchor_text

        expanded_text = build_contextualized_text(parent_text, metadata, anchor_text=anchor_text)
        hierarchy_score = 0.7 * _safe_score(result.score) + 0.3 * _metadata_match_score(query, metadata)

        metadata.update({
            "expanded_scope": scope,
            "anchor_chunk_id": result.chunk_id,
            "anchor_path": metadata.get("hierarchical_path", ""),
            "expansion_applied": scope != "chunk",
            "hierarchy_score": hierarchy_score
        })

        return RetrievalResult(
            chunk_id=f"{result.chunk_id}__{scope}",
            text=expanded_text,
            score=hierarchy_score,
            metadata=metadata,
            method=f"hierarchical_{scope}",
            raw_text=anchor_text,
            contextualized_text=expanded_text,
            parent_section_text=result.parent_section_text or (parent_text if scope == "section" else ""),
            parent_article_text=result.parent_article_text or (parent_text if scope == "article" else "")
        )

    def _infer_scope(self, query: str, metadata: Dict) -> str:
        query_lower = (query or "").lower()
        wants_article_scope = any(marker in query_lower for marker in ARTICLE_EXPANSION_MARKERS)

        if metadata.get("point"):
            if wants_article_scope:
                return "article"
            return "section"
        if metadata.get("section"):
            if wants_article_scope:
                return "article"
            return "section"
        if wants_article_scope:
            return "article"
        return "chunk"

    def _group_key(self, metadata: Dict, scope: str) -> Tuple:
        if scope == "article":
            return self._article_key(metadata)
        if scope == "section":
            return self._section_key(metadata) or self._article_key(metadata)
        return (
            metadata.get("filename", ""),
            metadata.get("article_key", metadata.get("article", "")),
            metadata.get("section", ""),
            metadata.get("point", "")
        )

    def _article_key(self, metadata: Dict) -> Tuple:
        if metadata.get("article_id"):
            return metadata.get("article_id")
        return (
            metadata.get("filename", ""),
            metadata.get("article_key", metadata.get("article", ""))
        )

    def _section_key(self, metadata: Dict) -> Tuple:
        if metadata.get("section_id"):
            return metadata.get("section_id")
        section = metadata.get("section", "")
        if not section:
            return tuple()
        return (
            metadata.get("filename", ""),
            metadata.get("article_key", metadata.get("article", "")),
            section
        )

    def _section_text(self, metadata: Dict) -> str:
        section_key = self._section_key(metadata)
        if not section_key or section_key not in self.section_context_map:
            return self._article_text(metadata)

        first_chunk = self.section_context_map[section_key][0]
        if getattr(first_chunk, "parent_section_text", ""):
            return first_chunk.parent_section_text

        section_chunks = sorted(
            self.section_context_map[section_key],
            key=lambda item: (item.metadata.get("page", 0), self._point_order(item.metadata.get("point", "")))
        )
        return "\n".join(_chunk_raw_text(chunk) for chunk in section_chunks if _chunk_raw_text(chunk))

    def _article_text(self, metadata: Dict) -> str:
        article_key = self._article_key(metadata)
        article_id = metadata.get("article_id", "")
        article_name = metadata.get("article_key", metadata.get("article", ""))

        if article_id and isinstance(self.article_full_text_map.get(article_id), str):
            return self.article_full_text_map[article_id]

        if isinstance(self.article_full_text_map.get(article_name), str):
            return self.article_full_text_map[article_name]

        if article_key not in self.article_context_map:
            return ""

        first_chunk = self.article_context_map[article_key][0]
        if getattr(first_chunk, "parent_article_text", ""):
            return first_chunk.parent_article_text

        article_chunks = sorted(
            self.article_context_map[article_key],
            key=lambda item: (
                item.metadata.get("page", 0),
                self._section_order(item.metadata.get("section", "")),
                self._point_order(item.metadata.get("point", ""))
            )
        )
        return "\n".join(_chunk_raw_text(chunk) for chunk in article_chunks if _chunk_raw_text(chunk))

    def _section_order(self, section: str) -> int:
        match = re.search(r"(\d+)", section or "")
        return int(match.group(1)) if match else 0

    def _point_order(self, point: str) -> int:
        match = re.search(r"([a-zđ])", (point or "").lower())
        if not match:
            return 0
        return POINT_ORDER.get(match.group(1), 999)

class HyDERetriever:
    """HyDE (Hypothetical Document Embeddings) Retriever"""
    
    def __init__(self, llm_generate_func, embedder):
        self.llm_generate = llm_generate_func
        self.embedder = embedder
    
    def generate_hypothetical_document(self, query: str) -> str:
        """Generate hypothetical document for query"""
        prompt = HYDE_PROMPT_TEMPLATE.format(query=query)
        
        try:
            hyde_doc = self.llm_generate(prompt, max_new_tokens=200)
            return hyde_doc
        except Exception as e:
            print(f" HyDE generation failed: {e}")
            return query  # Fallback to original query
    
    def retrieve(self, query: str, faiss_index, chunks, top_k: int = 20, query_plan: Optional[Dict] = None) -> List[RetrievalResult]:
        """Retrieve using HyDE"""
        # Generate hypothetical document
        hyde_doc = str((query_plan or {}).get("pseudo_document", "") or "").strip()
        if not hyde_doc:
            hyde_doc = self.generate_hypothetical_document(query)
        
        try:
            # Embed hypothetical document
            query_vec = self.embedder.encode(hyde_doc, convert_to_tensor=True).cpu().numpy()
            query_vec = query_vec.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_vec)
            
            # Search
            distances, indices = faiss_index.search(query_vec, top_k)
        except Exception as exc:
            print(f" HyDE retrieval failed, skipping HyDE branch: {exc}")
            return []
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(chunks):
                chunk = chunks[idx]
                similarity = float(max(0.0, min(1.0, (float(dist) + 1.0) / 2.0)))
                results.append(RetrievalResult(
                    chunk_id=chunk.chunk_id if hasattr(chunk, 'chunk_id') else f"chunk_{idx}",
                    text=_chunk_raw_text(chunk),
                    score=similarity,
                    metadata=chunk.metadata,
                    method='hyde',
                    raw_text=_chunk_raw_text(chunk),
                    contextualized_text=_chunk_contextualized_text(chunk),
                    parent_section_text=_chunk_parent_section_text(chunk),
                    parent_article_text=_chunk_parent_article_text(chunk)
                ))
        
        return results

class BM25Retriever:
    """BM25 Keyword Retriever"""
    
    def __init__(self, bm25_index, chunks, retrieval_field: str = "text_raw", method_name: str = "bm25"):
        self.bm25_index = bm25_index
        self.chunks = chunks
        self.retrieval_field = retrieval_field
        self.method_name = method_name
    
    def retrieve(self, query: str, top_k: int = 20) -> List[RetrievalResult]:
        """Retrieve using BM25"""
        # Tokenize query
        query_tokens = _tokenize_for_bm25(query)
        if not query_tokens:
            query_tokens = [query.lower()]
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top K
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append(RetrievalResult(
                    chunk_id=chunk.chunk_id if hasattr(chunk, 'chunk_id') else f"chunk_{idx}",
                    text=_chunk_raw_text(chunk),
                    score=float(scores[idx]),
                    metadata=chunk.metadata,
                    method=self.method_name,
                    raw_text=_chunk_raw_text(chunk),
                    contextualized_text=_chunk_contextualized_text(chunk),
                    parent_section_text=_chunk_parent_section_text(chunk),
                    parent_article_text=_chunk_parent_article_text(chunk)
                ))
        
        return results

class VectorRetriever:
    """Dense Vector Retriever"""
    
    def __init__(self, embedder, faiss_index, chunks, method_name: str = "vector"):
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.method_name = method_name
    
    def retrieve(self, query: str, top_k: int = 20) -> List[RetrievalResult]:
        """Retrieve using vector similarity"""
        try:
            # Embed query
            query_vec = self.embedder.encode(query, convert_to_tensor=True).cpu().numpy()
            query_vec = query_vec.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_vec)
            
            # Search
            distances, indices = self.faiss_index.search(query_vec, top_k)
        except Exception as exc:
            print(f" Vector retrieval failed, falling back to lexical-only branch: {exc}")
            return []
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                similarity = float(max(0.0, min(1.0, (float(dist) + 1.0) / 2.0)))
                results.append(RetrievalResult(
                    chunk_id=chunk.chunk_id if hasattr(chunk, 'chunk_id') else f"chunk_{idx}",
                    text=_chunk_raw_text(chunk),
                    score=similarity,
                    metadata=chunk.metadata,
                    method=self.method_name,
                    raw_text=_chunk_raw_text(chunk),
                    contextualized_text=_chunk_contextualized_text(chunk),
                    parent_section_text=_chunk_parent_section_text(chunk),
                    parent_article_text=_chunk_parent_article_text(chunk)
                ))
        
        return results


class HybridRetriever:
    """Hybrid Retriever combining BM25 and Vector"""
    
    def __init__(self,
                 bm25_raw_retriever,
                 bm25_context_retriever,
                 vector_retriever,
                 bm25_raw_weight: float = BM25_WEIGHT * BM25_RAW_SHARE,
                 bm25_context_weight: float = BM25_WEIGHT * BM25_CONTEXT_SHARE,
                 vector_weight: float = VECTOR_WEIGHT):
        self.bm25_raw_retriever = bm25_raw_retriever
        self.bm25_context_retriever = bm25_context_retriever
        self.vector_retriever = vector_retriever
        self.bm25_raw_weight = bm25_raw_weight
        self.bm25_context_weight = bm25_context_weight
        self.vector_weight = vector_weight
    
    def retrieve(self, query: str, top_k: int = 15, query_plan: Optional[Dict] = None) -> List[RetrievalResult]:
        """Hybrid retrieval with score fusion"""
        query_plan = query_plan or {}
        bm25_query = str(query_plan.get("normalized_query", "") or "").strip() or query
        vector_query = str(query_plan.get("abstract_query", "") or "").strip() or bm25_query

        # Get results from all retrievers
        bm25_raw_results = self.bm25_raw_retriever.retrieve(bm25_query, top_k=TOP_K_BM25)
        bm25_context_results = self.bm25_context_retriever.retrieve(bm25_query, top_k=TOP_K_BM25)
        vector_results = self.vector_retriever.retrieve(vector_query, top_k=TOP_K_VECTOR)
        
        # Normalize scores
        bm25_raw_results = self._normalize_scores(bm25_raw_results)
        bm25_context_results = self._normalize_scores(bm25_context_results)
        vector_results = self._normalize_scores(vector_results)
        
        # Merge results
        merged_scores = {}
        
        for result in bm25_raw_results:
            merged_scores[result.chunk_id] = {
                'text': result.text,
                'raw_text': result.raw_text or result.text,
                'contextualized_text': result.contextualized_text,
                'parent_section_text': result.parent_section_text,
                'parent_article_text': result.parent_article_text,
                'metadata': result.metadata,
                'score': result.score * self.bm25_raw_weight,
                'bm25_raw_score': result.score,
                'bm25_context_score': 0.0,
                'vector_score': 0.0
            }
        
        for result in bm25_context_results:
            if result.chunk_id in merged_scores:
                merged_scores[result.chunk_id]['score'] += result.score * self.bm25_context_weight
                merged_scores[result.chunk_id]['bm25_context_score'] = result.score
                merged_scores[result.chunk_id]['contextualized_text'] = result.contextualized_text or merged_scores[result.chunk_id]['contextualized_text']
                merged_scores[result.chunk_id]['parent_section_text'] = result.parent_section_text or merged_scores[result.chunk_id]['parent_section_text']
                merged_scores[result.chunk_id]['parent_article_text'] = result.parent_article_text or merged_scores[result.chunk_id]['parent_article_text']
            else:
                merged_scores[result.chunk_id] = {
                    'text': result.text,
                    'raw_text': result.raw_text or result.text,
                    'contextualized_text': result.contextualized_text,
                    'parent_section_text': result.parent_section_text,
                    'parent_article_text': result.parent_article_text,
                    'metadata': result.metadata,
                    'score': result.score * self.bm25_context_weight,
                    'bm25_raw_score': 0.0,
                    'bm25_context_score': result.score,
                    'vector_score': 0.0
                }

        for result in vector_results:
            if result.chunk_id in merged_scores:
                merged_scores[result.chunk_id]['score'] += result.score * self.vector_weight
                merged_scores[result.chunk_id]['vector_score'] = result.score
                merged_scores[result.chunk_id]['contextualized_text'] = result.contextualized_text or merged_scores[result.chunk_id]['contextualized_text']
                merged_scores[result.chunk_id]['parent_section_text'] = result.parent_section_text or merged_scores[result.chunk_id]['parent_section_text']
                merged_scores[result.chunk_id]['parent_article_text'] = result.parent_article_text or merged_scores[result.chunk_id]['parent_article_text']
            else:
                merged_scores[result.chunk_id] = {
                    'text': result.text,
                    'raw_text': result.raw_text or result.text,
                    'contextualized_text': result.contextualized_text,
                    'parent_section_text': result.parent_section_text,
                    'parent_article_text': result.parent_article_text,
                    'metadata': result.metadata,
                    'score': result.score * self.vector_weight,
                    'bm25_raw_score': 0.0,
                    'bm25_context_score': 0.0,
                    'vector_score': result.score
                }
        
        # Sort by merged score
        sorted_results = sorted(
            merged_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:top_k]
        
        # Convert to RetrievalResult
        results = []
        for chunk_id, data in sorted_results:
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                text=data['text'],
                score=data['score'],
                metadata=data['metadata'],
                method='hybrid',
                raw_text=data.get('raw_text', data['text']),
                contextualized_text=data.get('contextualized_text', ""),
                parent_section_text=data.get('parent_section_text', ""),
                parent_article_text=data.get('parent_article_text', "")
            ))
        
        return results
    
    def _normalize_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Normalize scores to [0, 1]"""
        if not results:
            return results
        
        scores = [r.score for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return results
        
        for result in results:
            result.score = (result.score - min_score) / (max_score - min_score)
        
        return results


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def merge_and_deduplicate(results1: List[RetrievalResult], 
                          results2: List[RetrievalResult]) -> List[RetrievalResult]:
    """
    Merge two result lists and remove duplicates
    
    Strategy:
    - Interleave results from both lists
    - Remove duplicates by chunk_id
    - Preserve order from both retrievers
    
    Args:
        results1: First list of results (e.g., from Hybrid)
        results2: Second list of results (e.g., from HyDE)
    
    Returns:
        Merged list without duplicates
    """
    seen_ids = set()
    merged = []
    
    # Interleave results
    max_len = max(len(results1), len(results2))
    for i in range(max_len):
        # Add from results1
        if i < len(results1):
            result = results1[i]
            if result.chunk_id not in seen_ids:
                merged.append(result)
                seen_ids.add(result.chunk_id)
        
        # Add from results2
        if i < len(results2):
            result = results2[i]
            if result.chunk_id not in seen_ids:
                merged.append(result)
                seen_ids.add(result.chunk_id)
    
    return merged


class MMRSelector:
    """Maximal Marginal Relevance with a SemViQA-QATC redundancy backend."""

    def __init__(self, embedder, lambda_param=0.7, mmr_qatc_model=None, mmr_qatc_tokenizer=None):
        self.embedder = embedder
        self.lambda_param = lambda_param
        self.mmr_backend = SemViQAMMRBackend(
            model=mmr_qatc_model,
            tokenizer=mmr_qatc_tokenizer,
        )
        self.backend_name = (
            self.mmr_backend.backend_name
            if self.mmr_backend.is_available
            else "embedding_cosine_fallback"
        )
        self.last_backend_used = "uninitialized"

    def _select_with_embeddings(
        self,
        results: List[RetrievalResult],
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        if len(results) <= top_k:
            return results

        texts = [build_contextualized_result_text(r) for r in results]
        embeddings = self.embedder.encode(texts, convert_to_tensor=True).cpu().numpy()

        selected_indices = []
        remaining_indices = list(range(len(results)))
        selected_indices.append(0)
        remaining_indices.remove(0)

        while len(selected_indices) < top_k and remaining_indices:
            mmr_scores = []

            for idx in remaining_indices:
                relevance = _safe_score(results[idx].score)
                similarities = []
                for selected_idx in selected_indices:
                    sim = np.dot(embeddings[idx], embeddings[selected_idx])
                    sim = sim / (np.linalg.norm(embeddings[idx]) * np.linalg.norm(embeddings[selected_idx]))
                    similarities.append(sim)

                max_sim = max(similarities) if similarities else 0
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                mmr_scores.append((idx, mmr_score))

            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [results[idx] for idx in selected_indices]

    def _select_with_qatc(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        if len(results) <= top_k:
            return results

        semantic_views = [build_contextualized_result_text(result) for result in results]
        semantic_signatures = [
            self.mmr_backend.build_signature(query, text)
            for text in semantic_views
        ]

        for result, signature in zip(results, semantic_signatures):
            result.metadata["mmr_signature_backend"] = signature["backend"]
            result.metadata["mmr_signature_terms"] = len(signature["token_weights"])
            result.metadata["mmr_evidence_score"] = float(signature["model_score"])
            result.metadata["mmr_evidence_sentences"] = int(signature["highlighted_sentence_count"])
            result.metadata["mmr_evidence_text"] = str(signature.get("evidence_text", "") or "")
            result.metadata["mmr_rationale_tokens"] = int(signature.get("rationale_token_count", 0) or 0)

        selected_indices = [0]
        remaining_indices = list(range(1, len(results)))
        first_relevance = _safe_score(results[0].score)
        results[0].metadata["mmr_backend"] = semantic_signatures[0]["backend"]
        results[0].metadata["mmr_score"] = float(self.lambda_param * first_relevance)
        results[0].metadata["mmr_similarity_penalty"] = 0.0

        while len(selected_indices) < top_k and remaining_indices:
            mmr_scores = []

            for idx in remaining_indices:
                relevance = _safe_score(results[idx].score)
                similarities = [
                    _weighted_jaccard_similarity(
                        semantic_signatures[idx]["token_weights"],
                        semantic_signatures[selected_idx]["token_weights"],
                    )
                    for selected_idx in selected_indices
                ]
                max_sim = max(similarities) if similarities else 0.0
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                mmr_scores.append((idx, mmr_score, relevance, max_sim))

            best_idx, best_score, best_relevance, best_penalty = max(
                mmr_scores,
                key=lambda item: item[1],
            )
            results[best_idx].metadata["mmr_backend"] = semantic_signatures[best_idx]["backend"]
            results[best_idx].metadata["mmr_score"] = float(best_score)
            results[best_idx].metadata["mmr_relevance"] = float(best_relevance)
            results[best_idx].metadata["mmr_similarity_penalty"] = float(best_penalty)
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [results[idx] for idx in selected_indices]

    def select(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 10,
    ) -> List[RetrievalResult]:
        """Select diverse results using SemViQA-QATC MMR, with cosine fallback."""
        if len(results) <= top_k:
            self.last_backend_used = self.backend_name
            return results

        if query and self.mmr_backend.is_available:
            try:
                selected = self._select_with_qatc(query, results, top_k=top_k)
                used_backends = sorted({
                    str(result.metadata.get("mmr_backend", "") or "")
                    for result in selected
                    if str(result.metadata.get("mmr_backend", "") or "").strip()
                })
                if len(used_backends) == 1:
                    self.last_backend_used = used_backends[0]
                elif used_backends:
                    self.last_backend_used = ",".join(used_backends)
                else:
                    self.last_backend_used = self.mmr_backend.backend_name
                return selected
            except Exception as exc:
                print(f" QATC MMR failed, falling back to embedding cosine MMR: {exc}")

        try:
            selected = self._select_with_embeddings(results, top_k=top_k)
            self.last_backend_used = "embedding_cosine_fallback"
            return selected
        except Exception as exc:
            print(f" MMR embedding failed, returning reranked results without diversity pass: {exc}")
            self.last_backend_used = "disabled_after_failure"
            return results[:top_k]


class ViReranker:
    """Single-step Vietnamese reranker with ViRanker and similarity fallback."""

    def __init__(self, model_name: str = "itdainb/vietnamese-cross-encoder"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.enabled = False
        self.max_length = RERANK_MAX_LENGTH

        try:
            import torch
            requested_device = str(os.environ.get("RERANKER_DEVICE", "") or "").strip().lower()
            if requested_device == "cuda" and not torch.cuda.is_available():
                print("⚠️  RERANKER_DEVICE=cuda nhưng CUDA không sẵn sàng. Fallback sang CPU.")
                requested_device = "cpu"
            self.device = torch.device(requested_device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        except Exception:
            self.device = None

        try:
            preloaded_model = globals().get("reranker_stage2_model")
            preloaded_tokenizer = globals().get("reranker_stage2_tokenizer")

            if preloaded_model is not None and preloaded_tokenizer is not None:
                print("ℹ Using preloaded ViRanker from CELL 3")
                self.model = preloaded_model
                self.tokenizer = preloaded_tokenizer
            else:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                print(f"⏳ Loading ViRanker: {model_name}")
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.model.eval()
            if self.device is not None:
                self.model.to(self.device)

            self.max_length = resolve_safe_rerank_max_length(
                self.tokenizer,
                self.model,
                RERANK_MAX_LENGTH
            )
            self.enabled = True
            print(f" ViRanker loaded: {model_name}")
            print(f"   • ViRanker max_length: request={RERANK_MAX_LENGTH}, effective={self.max_length}")
        except Exception as e:
            print(f" ViRanker loading failed: {e}")
            print("   Will use fallback reranking")
            self.enabled = False

    def _disable_cuda_reranker(self, exc: Exception) -> None:
        if not _is_cuda_runtime_error(exc):
            return
        print(" CUDA reranker failure detected. Disabling GPU reranker for this runtime.")
        self.enabled = False
        try:
            import torch
            self.device = torch.device("cpu")
        except Exception:
            self.device = None
        _best_effort_clear_cuda()

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5,
        debug_trace: bool = False,
    ) -> Tuple[List[RetrievalResult], Dict]:
        """Single-step reranking with ViRanker."""
        start_time = time.time()
        metrics = {
            'method': 'none',
            'input_count': len(results),
            'output_count': 0,
            'time': 0.0,
            'total_time': 0.0,
        }
        if debug_trace:
            metrics['debug_trace'] = {}

        if len(results) <= top_k:
            metrics['output_count'] = len(results)
            metrics['total_time'] = time.time() - start_time
            if debug_trace:
                _record_debug_stage(metrics, 'post_viranker', 'After ViRanker', results, note='ViRanker was not applied because candidate count was already within final top-k.')
            return results, metrics

        rerank_start = time.time()
        if self.enabled:
            try:
                results = self._rerank(query, results, top_k)
                metrics['method'] = 'viranker'
            except Exception as e:
                print(f" ViRanker reranking failed: {e}")
                self._disable_cuda_reranker(e)
                results = self._rerank_fallback(query, results, top_k)
                metrics['method'] = 'fallback'
        else:
            results = self._rerank_fallback(query, results, top_k)
            metrics['method'] = 'fallback'

        metrics['output_count'] = len(results)
        metrics['time'] = time.time() - rerank_start
        metrics['total_time'] = time.time() - start_time
        if debug_trace:
            _record_debug_stage(metrics, 'post_viranker', 'After ViRanker', results, note=f"Single-step rerank method: {metrics['method']}")
        return results, metrics

    def _rerank(self, query: str, results: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        """ViRanker (Vietnamese-specific)."""
        import torch
        
        rerank_texts = [build_compact_rerank_text(r) for r in results]
        pairs = [[query, text] for text in rerank_texts]
        
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation='only_second',
            return_tensors='pt',
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            scores = self.model(**inputs, return_dict=True).logits.view(-1,).float()

        adjusted_scores = scores.clone()
        bonuses = []
        for idx, rerank_text in enumerate(rerank_texts):
            bonus = _rerank_query_bonus(query, rerank_text)
            bonuses.append(float(bonus))
            adjusted_scores[idx] = adjusted_scores[idx] + bonus
        
        sorted_indices = adjusted_scores.argsort(descending=True)[:top_k]
        
        reranked = []
        for idx in sorted_indices:
            result = results[idx.item()]
            result.metadata["viranker_model_score"] = float(scores[idx].item())
            result.metadata["viranker_bonus"] = bonuses[idx.item()]
            result.score = float(adjusted_scores[idx].item())
            result.method = 'viranker_reranked'
            reranked.append(result)
        
        return reranked
    
    def _rerank_fallback(self, query: str, results: List[RetrievalResult], 
                        top_k: int) -> List[RetrievalResult]:
        """Fallback: Similarity-based reranking"""
        
        # Check if embedder is available
        if 'embedder' not in globals():
            print(" Embedder not available. Returning top-K by original score.")
            return results[:top_k]
        
        try:
            # Embed query
            query_emb = embedder.encode(query, convert_to_tensor=True).cpu().numpy()
            
            # Embed all result texts
            texts = [build_contextualized_result_text(r) for r in results]
            text_embs = embedder.encode(texts, convert_to_tensor=True).cpu().numpy()
            
            # Calculate similarities
            similarities = []
            for text_emb in text_embs:
                sim = np.dot(query_emb, text_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(text_emb)
                )
                similarities.append(float(sim))
            
            # Combine with original scores (weighted average)
            combined_scores = []
            for i, result in enumerate(results):
                combined = 0.6 * similarities[i] + 0.4 * _safe_score(result.score)
                combined_scores.append((i, combined))
            
            # Sort by combined score
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top K
            reranked_results = []
            for idx, score in combined_scores[:top_k]:
                result = results[idx]
                result.score = score
                result.method = 'fallback_reranked'
                reranked_results.append(result)
            
            return reranked_results
            
        except Exception as e:
            print(f" Fallback reranking failed: {e}")
            print(f"   Returning top-K by original score.")
            return results[:top_k]


class MetadataFilter:
    """Filter results by metadata"""
    
    def __init__(self, filters: Dict):
        self.filters = filters
    
    def filter(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Filter results by metadata"""
        if not self.filters:
            return results
        
        filtered = []
        for result in results:
            match = True
            for key, value in self.filters.items():
                if key not in result.metadata or result.metadata[key] != value:
                    match = False
                    break
            
            if match:
                filtered.append(result)
        
        return filtered

# ==============================================================================
# INITIALIZE RETRIEVERS
# ==============================================================================

# Initialize global variables for retrievers
bm25_retriever = None
bm25_raw_retriever = None
bm25_contextualized_retriever = None
vector_retriever = None
hybrid_retriever = None
hyde_retriever = None
mmr_selector = None
vi_reranker = None
metadata_filter = None
hierarchical_expander = None

print("\n" + "="*70)
print(" Initializing Retrievers")
print("="*70)

# Check required variables from Cell 3 and Cell 4
required_vars = {
    'embedder': 'Sentence Transformer (from Cell 3)',
    'chunks': 'Chunks list (from Cell 4)',
    'faiss_index_b': 'FAISS Index (from Cell 4)',
    'bm25_index': 'BM25 Index (from Cell 4)'
}

missing = []
for var in required_vars:
    if var not in globals():
        missing.append(var)
        print(f" THIEU: {var} - {required_vars[var]}")

if missing:
    print(f"\n ERROR: Thieu {len(missing)} bien can thiet!")
    print(" Vui long chay Cell 3 va Cell 4 truoc!")
    print("\n CANH BAO: Cac retriever se khong hoat dong cho den khi cac bien nay duoc khoi tao!")
    # Don't initialize retrievers if required variables are missing
    print("\n SKIPPING retriever initialization due to missing variables")
elif 'chunks' in globals() and 'faiss_index_b' in globals() and 'bm25_index' in globals():
    print(" Tat ca cac bien can thiet da san sang")
    
    # Initialize retrievers
    print("\n Dang khoi tao retrievers...")
    
    try:
        # BM25 Retriever
        bm25_index_raw_source = globals().get('bm25_index_raw', bm25_index)
        bm25_index_context_source = globals().get('bm25_index_contextualized', bm25_index)

        bm25_raw_retriever = BM25Retriever(
            bm25_index_raw_source,
            chunks,
            retrieval_field="text_raw",
            method_name='bm25_raw'
        )
        bm25_contextualized_retriever = BM25Retriever(
            bm25_index_context_source,
            chunks,
            retrieval_field="text_contextualized",
            method_name='bm25_contextualized'
        )
        bm25_retriever = bm25_contextualized_retriever
        print(" BM25 Raw Retriever initialized")
        print(" BM25 Contextualized Retriever initialized")
        
        # Vector Retriever
        vector_retriever = VectorRetriever(embedder, faiss_index_b, chunks, method_name='vector_contextualized')
        print(" Vector Contextualized Retriever initialized")
        
        # HyDE Retriever (optional)
        if USE_HYDE:
            hyde_generate = globals().get("generate_text")
            if hyde_generate is None:
                def hyde_generate(*args, **kwargs):
                    raise RuntimeError("HyDE generate function chưa sẵn sàng")
            hyde_retriever = HyDERetriever(hyde_generate, embedder)
            print(" HyDE Retriever initialized")
        
        # Hybrid Retriever
        hybrid_retriever = HybridRetriever(
            bm25_raw_retriever,
            bm25_contextualized_retriever,
            vector_retriever,
            bm25_raw_weight=BM25_WEIGHT * BM25_RAW_SHARE,
            bm25_context_weight=BM25_WEIGHT * BM25_CONTEXT_SHARE,
            vector_weight=VECTOR_WEIGHT
        )
        print(" Hybrid Retriever initialized")
        
        # MMR Selector (optional)
        if USE_MMR:
            mmr_selector = MMRSelector(
                embedder,
                lambda_param=MMR_LAMBDA,
                mmr_qatc_model=globals().get("mmr_qatc_model"),
                mmr_qatc_tokenizer=globals().get("mmr_qatc_tokenizer"),
            )
            print(f" MMR Selector initialized ({mmr_selector.backend_name})")
        
        # ViRanker reranker (optional)
        if ENABLE_RERANKING:
            vi_reranker = ViReranker(model_name=RERANK_STAGE2_MODEL)
            print(" ViRanker reranker initialized")
        
        # Metadata Filter (optional)
        if ENABLE_METADATA_FILTER:
            metadata_filter = MetadataFilter(METADATA_FILTERS)
            print(" Metadata Filter initialized")

        if ENABLE_HIERARCHICAL_EXPANSION:
            hierarchical_expander = HierarchicalContextExpander(
                chunks,
                article_full_text_map if 'article_full_text_map' in globals() else {}
            )
            print(" Hierarchical Context Expander initialized")
        
        print("\n TAT CA RETRIEVERS DA DUOC KHOI TAO THANH CONG!")
        
    except Exception as e:
        print(f"\n LOI khi khoi tao retrievers: {e}")
        print(f" Chi tiet: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("\n CANH BAO: Cac retriever se khong hoat dong!")



# ==============================================================================
# MAIN RETRIEVAL FUNCTION
# ==============================================================================

def retrieve_enhanced(query: str, 
                     retrieval_mode: str = "hybrid",
                     use_hybrid: bool = USE_HYBRID,
                     use_hyde: bool = USE_HYDE,
                     use_rerank: bool = ENABLE_RERANKING,
                     use_mmr: bool = USE_MMR,
                     use_hierarchical_expansion: bool = ENABLE_HIERARCHICAL_EXPANSION,
                     query_plan: Optional[Dict] = None,
                     debug_trace: bool = False,
                     verbose: bool = True) -> Tuple[List[RetrievalResult], Dict]:
    """
    Enhanced retrieval with OPTIMIZED PIPELINE
    
    Args:
        query: Search query
        use_hybrid: Use hybrid retrieval
        use_hyde: Use HyDE retrieval
        use_rerank: Use reranking
        use_mmr: Use MMR for diversity
        use_hierarchical_expansion: Expand child hits to parent context
        verbose: Print detailed progress (default: True)
    
    Returns:
        (results, metrics)
    """
    global hybrid_retriever, hyde_retriever, mmr_selector, vi_reranker, hierarchical_expander
    global bm25_contextualized_retriever, vector_retriever
    
    start_time = time.time()
    metrics = {}
    retrieval_mode = (retrieval_mode or "hybrid").lower()
    if debug_trace:
        metrics['debug_trace'] = {
            'query': str(query or ""),
            'retrieval_mode': retrieval_mode,
        }
    
    # Check if retrievers are initialized
    if use_hybrid and hybrid_retriever is None:
        raise RuntimeError("hybrid_retriever chưa được khởi tạo. Vui lòng chạy Cell 3 và Cell 4 trước!")
    if not use_hybrid and not use_hyde and retrieval_mode == "vector" and vector_retriever is None:
        raise RuntimeError("vector_retriever chưa được khởi tạo. Vui lòng chạy Cell 3 và Cell 4 trước!")
    if not use_hybrid and not use_hyde and retrieval_mode == "bm25" and bm25_contextualized_retriever is None:
        raise RuntimeError("bm25_contextualized_retriever chưa được khởi tạo. Vui lòng chạy Cell 3 và Cell 4 trước!")
    # Nếu chọn Dual nhưng HyDE chưa được khởi tạo → dùng Hybrid only thay vì báo lỗi
    if use_hyde and hyde_retriever is None:
        use_hyde = False
        if verbose:
            print(" (HyDE chưa khởi tạo, dùng Hybrid only)")
    if use_rerank and vi_reranker is None:
        raise RuntimeError("vi_reranker chưa được khởi tạo. Vui lòng bật ENABLE_RERANKING=True và chạy lại phần khởi tạo!")
    if use_mmr and mmr_selector is None:
        raise RuntimeError("mmr_selector chưa được khởi tạo. Vui lòng bật USE_MMR=True và chạy lại phần khởi tạo!")
    
    if verbose:
        print("\n" + "="*70)
        print(" ENHANCED RETRIEVAL PIPELINE")
        print("="*70)
        print(f"\n Query: {query}")
        print(f"  Config: Hybrid={use_hybrid}, HyDE={use_hyde}, Rerank={use_rerank}, MMR={use_mmr}, Hierarchical={use_hierarchical_expansion}")
        
        # Show mode
        if use_hybrid and use_hyde:
            print(f" Mode: DUAL (Hybrid + HyDE)")
        elif not use_hybrid and use_hyde:
            print(f" Mode: HyDE ONLY")
        elif use_hybrid and not use_hyde:
            print(f" Mode: HYBRID ONLY")
        elif retrieval_mode == "vector":
            print(f" Mode: VECTOR ONLY")
        elif retrieval_mode == "bm25":
            print(f" Mode: BM25 ONLY")
        else:
            print(f"  WARNING: Both methods disabled!")
    
    # Step 1: Retrieval
    if verbose:
        print(f"\n{''*70}")
        print(f" STEP 1: RETRIEVAL")
        print(f"{''*70}")
    
    all_results = []
    hybrid_results = []
    hyde_results = []
    base_method = "none"
    
    # 1a. Hybrid Retrieval (if enabled)
    if use_hybrid:
        hybrid_start = time.time()
        hybrid_results = hybrid_retriever.retrieve(query, top_k=TOP_K_HYBRID, query_plan=query_plan)
        hybrid_time = time.time() - hybrid_start
        base_method = "hybrid"
        
        if verbose:
            print(f"\n[1a] Hybrid Search (BM25 + Vector):")
            print(f"   ⏱  Time: {hybrid_time:.3f}s")
            print(f"    Results: {len(hybrid_results)} chunks")
            if len(hybrid_results) > 0:
                print(f"    Top score: {hybrid_results[0].score:.4f}")
                print(f"    Top chunk: {hybrid_results[0].text[:100]}...")
        
        metrics['hybrid_count'] = len(hybrid_results)
        metrics['hybrid_time'] = hybrid_time
        if debug_trace:
            _record_debug_stage(metrics, 'initial_hybrid', 'Initial hybrid retrieval', hybrid_results)
        all_results.extend(hybrid_results)
    elif retrieval_mode == "vector":
        vector_start = time.time()
        hybrid_results = vector_retriever.retrieve(query, top_k=TOP_K_VECTOR)
        hybrid_time = time.time() - vector_start
        base_method = "vector"

        if verbose:
            print(f"\n[1a] Vector Search (contextualized FAISS):")
            print(f"   ⏱  Time: {hybrid_time:.3f}s")
            print(f"    Results: {len(hybrid_results)} chunks")
            if len(hybrid_results) > 0:
                print(f"    Top score: {hybrid_results[0].score:.4f}")
                print(f"    Top chunk: {hybrid_results[0].text[:100]}...")

        metrics['hybrid_count'] = len(hybrid_results)
        metrics['hybrid_time'] = hybrid_time
        all_results.extend(hybrid_results)
    elif retrieval_mode == "bm25":
        bm25_start = time.time()
        hybrid_results = bm25_contextualized_retriever.retrieve(query, top_k=TOP_K_BM25)
        hybrid_time = time.time() - bm25_start
        base_method = "bm25"

        if verbose:
            print(f"\n[1a] BM25 Search (contextualized):")
            print(f"   ⏱  Time: {hybrid_time:.3f}s")
            print(f"    Results: {len(hybrid_results)} chunks")
            if len(hybrid_results) > 0:
                print(f"    Top score: {hybrid_results[0].score:.4f}")
                print(f"    Top chunk: {hybrid_results[0].text[:100]}...")

        metrics['hybrid_count'] = len(hybrid_results)
        metrics['hybrid_time'] = hybrid_time
        all_results.extend(hybrid_results)
    else:
        if verbose:
            print(f"\n[1a] Base Retrieval: DISABLED ")
        metrics['hybrid_count'] = 0
        metrics['hybrid_time'] = 0
    
    # 1b. HyDE Retrieval (if enabled)
    if use_hyde:
        hyde_start = time.time()
        if verbose:
            print(f"\n[1b] HyDE Search (Hypothetical Document):")
            print(f"   ⏳ Generating hypothetical document...")
        
        hyde_results = hyde_retriever.retrieve(query, faiss_index_b, chunks, top_k=TOP_K_HYDE, query_plan=query_plan)
        hyde_time = time.time() - hyde_start
        
        if verbose:
            print(f"   ⏱  Time: {hyde_time:.3f}s")
            print(f"    Results: {len(hyde_results)} chunks")
            if len(hyde_results) > 0:
                print(f"    Top score: {hyde_results[0].score:.4f}")
        
        metrics['hyde_count'] = len(hyde_results)
        metrics['hyde_time'] = hyde_time
        metrics['retrieval_method'] = 'dual' if base_method != 'none' else 'hyde'
        if debug_trace:
            _record_debug_stage(metrics, 'initial_hyde', 'Initial HyDE retrieval', hyde_results)
    else:
        hyde_results = []
        metrics['hyde_count'] = 0
        metrics['hyde_time'] = 0
        metrics['retrieval_method'] = base_method
        if verbose and use_hyde:
            print(f"\n[1b] HyDE Search: ⏭  Skipped (disabled)")
    
    metrics['retrieval_time'] = time.time() - start_time
    
    # Step 2: Merge & Deduplicate (if we have results from multiple sources)
    if len(all_results) == 0:
        if verbose:
            print(f"\n  No results from any retrieval method!")
        return [], metrics
    
    if hyde_results and use_hybrid:
        # Both Hybrid and HyDE results - need to merge
        if verbose:
            print(f"\n{''*70}")
            print(f" STEP 2: MERGE & DEDUPLICATE")
            print(f"{''*70}")
        
        merge_start = time.time()
        results = merge_and_deduplicate(hybrid_results, hyde_results)
        merge_time = time.time() - merge_start
        
        if verbose:
            print(f"\n    Input: {len(hybrid_results)} (Hybrid) + {len(hyde_results)} (HyDE) = {len(hybrid_results) + len(hyde_results)} total")
            print(f"    Deduplicating...")
            print(f"    Output: {len(results)} unique chunks")
            print(f"   ⏱  Time: {merge_time:.3f}s")
            print(f"    Removed: {len(hybrid_results) + len(hyde_results) - len(results)} duplicates")
        
        metrics['merged_count'] = len(results)
        metrics['merge_time'] = merge_time
        metrics['duplicates_removed'] = len(hybrid_results) + len(hyde_results) - len(results)
    else:
        # Only one source - use directly
        results = all_results
        metrics['merged_count'] = len(results)
        metrics['merge_time'] = 0
        metrics['duplicates_removed'] = 0
        
        if verbose:
            source = "HyDE" if hyde_results else (base_method.upper() if base_method != "none" else "None")
            print(f"\n{''*70}")
            print(f" STEP 2: MERGE & DEDUPLICATE")
            print(f"{''*70}")
            print(f"\n    Input: {len(results)} chunks from {source} only")
            print(f"   ⏭  Skipping deduplication (single source)")
            print(f"    Output: {len(results)} chunks")
    if debug_trace:
        _record_debug_stage(metrics, 'merged_candidates', 'Merged candidates', results)
    
    # Step 2.5: Metadata Filtering (optional)
    if ENABLE_METADATA_FILTER:
        if verbose:
            print(f"\n{''*70}")
            print(f" STEP 2.5: METADATA FILTERING")
            print(f"{''*70}")
        
        before_count = len(results)
        results = metadata_filter.filter(results)
        
        if verbose:
            print(f"\n    Input: {before_count} chunks")
            print(f"    Filters: {METADATA_FILTERS}")
            print(f"    Output: {len(results)} chunks")
            print(f"    Filtered out: {before_count - len(results)} chunks")
        
        metrics['filtered_count'] = len(results)

    # Step 2.6: Structural metadata boost
    if ENABLE_METADATA_PATH_BOOST and results:
        results = apply_metadata_path_boost(query, results)
        metrics['path_boost_applied'] = True
        metrics['path_boost_top_score'] = results[0].metadata.get('path_match_score', 0.0) if results else 0.0
    else:
        metrics['path_boost_applied'] = False
        metrics['path_boost_top_score'] = 0.0

    if ENABLE_SCOPE_CONSISTENCY_PENALTY and results:
        results = apply_scope_consistency_penalty(query, results)
        metrics['scope_penalty_applied'] = True
    else:
        metrics['scope_penalty_applied'] = False

    if query_plan and results:
        results = apply_query_plan_guidance(query, results, query_plan)
        metrics['query_plan_guidance_applied'] = True
        metrics['query_plan_type'] = str(query_plan.get("query_type", "") or "")
        metrics['query_plan_hops'] = int(query_plan.get("required_hops", 1) or 1)
    else:
        metrics['query_plan_guidance_applied'] = False
    if debug_trace:
        _record_debug_stage(metrics, 'pre_rerank', 'Candidates before reranking', results)
    
    # Step 3: Vietnamese reranking (BEFORE HIERARCHICAL + MMR)
    rerank_target_k = TOP_K_MMR if use_mmr and USE_MMR else RERANK_STAGE2_TOP_K
    if use_rerank and ENABLE_RERANKING and vi_reranker is not None:
        if verbose:
            print(f"\n{''*70}")
            print(f" STEP 3: VIRANKER RERANKING (BEFORE HIERARCHICAL + MMR) ")
            print(f"{''*70}")

        if verbose:
            print(f"\n    Input: {len(results)} candidate chunks")
            print(f"    ViRanker target: Top {rerank_target_k}")

        results, rerank_metrics = vi_reranker.rerank(
            query,
            results,
            top_k=rerank_target_k,
            debug_trace=debug_trace,
        )

        if verbose:
            print(f"\n   [ViRanker] {rerank_metrics['method']}:")
            print(f"       Input: {rerank_metrics['input_count']} chunks")
            print(f"       Output: {rerank_metrics['output_count']} chunks")
            print(f"      ⏱  Time: {rerank_metrics['time']:.3f}s")
            print(f"\n    Total reranking time: {rerank_metrics['total_time']:.3f}s")

        metrics['rerank_stage2_method'] = rerank_metrics['method']
        metrics['stage2_count'] = rerank_metrics['output_count']
        metrics['stage2_time'] = rerank_metrics['time']
        metrics['rerank_time'] = rerank_metrics['total_time']
        metrics['rerank_mode'] = 'single_stage_viranker'
        metrics['rerank_backend'] = rerank_metrics['method']
        if debug_trace:
            metrics['debug_trace'].update(rerank_metrics.get('debug_trace', {}))
    else:
        if use_rerank and verbose:
            print(f"\n{''*70}")
            print(f"⏭  STEP 3: VIRANKER SKIPPED (not available)")
            print(f"{''*70}")
        results = results[:rerank_target_k]
        metrics['rerank_stage2_method'] = 'none'
        metrics['stage2_count'] = len(results)
        metrics['stage2_time'] = 0
        metrics['rerank_time'] = 0
        metrics['rerank_mode'] = 'disabled'
        metrics['rerank_backend'] = 'none'
        if debug_trace:
            _record_debug_stage(metrics, 'post_viranker', 'After ViRanker', results, note='ViRanker skipped; using sliced candidate pool directly.')

    # Step 4: Hierarchical expansion (child -> parent context)
    hierarchy_target_k = rerank_target_k if use_mmr and USE_MMR else HIERARCHICAL_MAX_RESULTS
    if use_hierarchical_expansion and ENABLE_HIERARCHICAL_EXPANSION and hierarchical_expander is not None:
        if verbose:
            print(f"\n{''*70}")
            print(f" STEP 4: HIERARCHICAL EXPANSION ")
            print(f"{''*70}")

        hierarchy_start = time.time()
        results, hierarchy_metrics = hierarchical_expander.expand(query, results, top_k=hierarchy_target_k)
        hierarchy_time = time.time() - hierarchy_start

        metrics.update(hierarchy_metrics)
        metrics['hierarchical_time'] = hierarchy_time
        if debug_trace:
            _record_debug_stage(metrics, 'post_hierarchical_expansion', 'After hierarchical expansion', results)

        if verbose:
            print(f"\n    Input: {metrics.get('stage2_count', len(results))} reranked chunks")
            print(f"    Target: Top {hierarchy_target_k} hierarchical candidates")
            print(f"    Anchored results: {metrics.get('hierarchical_anchored_results', 0)}")
            print(f"    Max hierarchy score: {metrics.get('hierarchical_max_score', 0.0):.4f}")
            print(f"   ⏱  Time: {hierarchy_time:.3f}s")
    else:
        metrics['hierarchical_expansion_applied'] = False
        metrics['hierarchical_anchored_results'] = 0
        metrics['hierarchical_max_score'] = 0.0
        metrics['hierarchical_scope_counts'] = {}
        metrics['hierarchical_time'] = 0.0
        if debug_trace:
            _record_debug_stage(metrics, 'post_hierarchical_expansion', 'After hierarchical expansion', results, note='Hierarchical expansion disabled or unavailable.')

    # Step 5: MMR Diversity Selection (AFTER HIERARCHICAL EXPANSION)
    if use_mmr and USE_MMR:
        final_mmr_k = min(RERANK_STAGE2_TOP_K, len(results))
        if verbose:
            print(f"\n{''*70}")
            print(f" STEP 5: QATC MMR DIVERSITY SELECTION ")
            print(f"{''*70}")

        mmr_start = time.time()
        before_count = len(results)

        if verbose:
            print(f"\n    Input: {before_count} candidate chunks")
            print(f"     Lambda: {MMR_LAMBDA} (relevance vs diversity)")
            print(f"    Target: Top {final_mmr_k} diverse chunks")
            print(f"    Backend: {mmr_selector.backend_name}")
            print(f"   ⏳ Selecting...")

        results = mmr_selector.select(query, results, top_k=final_mmr_k)
        mmr_time = time.time() - mmr_start

        if verbose:
            print(f"    Output: {len(results)} diverse chunks")
            print(f"   ⏱  Time: {mmr_time:.3f}s")
            print(f"    Backend used: {mmr_selector.last_backend_used}")
            print(f"    Diversity ensured!")

        metrics['mmr_count'] = len(results)
        metrics['mmr_time'] = mmr_time
        metrics['mmr_backend'] = getattr(mmr_selector, 'last_backend_used', 'unknown')
        if debug_trace:
            _record_debug_stage(metrics, 'post_mmr', 'After MMR', results)
    else:
        results = results[:RERANK_STAGE2_TOP_K]
        if verbose:
            print(f"\n{''*70}")
            print(f"⏭  STEP 5: MMR SKIPPED - Taking top {RERANK_STAGE2_TOP_K}")
            print(f"{''*70}")
        metrics['mmr_count'] = len(results)
        metrics['mmr_time'] = 0
        metrics['mmr_backend'] = 'disabled'
        if debug_trace:
            _record_debug_stage(metrics, 'post_mmr', 'After MMR', results, note='MMR disabled; using current candidate pool directly.')

    # Final
    metrics['final_count'] = len(results)
    metrics['total_time'] = time.time() - start_time
    if debug_trace:
        _record_debug_stage(metrics, 'final_retrieval_results', 'Final retrieval results', results)
    
    if verbose:
        hierarchical_applied = bool(metrics.get('hierarchical_expansion_applied', False))
        hierarchical_count = (
            metrics.get('hierarchical_anchored_results', 0)
            if hierarchical_applied
            else metrics.get('stage2_count', metrics['merged_count'])
        )
        print(f"\n{'='*70}")
        print(f" RETRIEVAL COMPLETE")
        print(f"{'='*70}")
        print(f"\n PIPELINE SUMMARY:")
        print(f"   Step 1: {metrics.get('retrieval_method', 'hybrid').upper()} base ({metrics['hybrid_count']}) + HyDE ({metrics['hyde_count']})")
        print(f"   Step 2: Merged ({metrics['merged_count']}) - Removed {metrics.get('duplicates_removed', 0)} duplicates")
        print(f"   Step 3: ViRanker pool ({metrics.get('stage2_count', metrics['merged_count'])})")
        print(f"   Step 4: Hierarchical ({hierarchical_count}){' - skipped' if not hierarchical_applied else ''}")
        print(f"   Step 5: MMR ({metrics['mmr_count']}) - Diversity via {metrics.get('mmr_backend', 'unknown')}")
        print(f"\n⏱  TIMING BREAKDOWN:")
        print(f"   Retrieval: {metrics['retrieval_time']:.3f}s")
        if metrics.get('merge_time', 0) > 0:
            print(f"   Merge: {metrics['merge_time']:.3f}s")
        if metrics.get('rerank_time', 0) > 0:
            print(f"   Reranking: {metrics['rerank_time']:.3f}s")
        if metrics.get('hierarchical_time', 0) > 0:
            print(f"   Hierarchical: {metrics['hierarchical_time']:.3f}s")
        if metrics.get('mmr_time', 0) > 0:
            print(f"   MMR: {metrics['mmr_time']:.3f}s")
        print(f"   TOTAL: {metrics['total_time']:.3f}s")
        
        print(f"\n FINAL RESULTS: {len(results)} chunks")
        if len(results) > 0:
            print(f"   Top score: {results[0].score:.4f}")
            print(f"   Method: {results[0].method}")
    
    return results, metrics

# ==============================================================================
# TEST RETRIEVAL
# ==============================================================================

# Only run test when executed directly, not when imported
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" TEST RETRIEVAL")
    print("="*70)

    # # Check if retrievers are initialized
    # if 'hybrid_retriever' not in globals() or hybrid_retriever is None:
    #     print("\n Error: Retrievers not initialized!")
    #     print(" Please ensure Cell 3 and Cell 4 have been run successfully.")
    #     print("   Required variables: generate_text, embedder, chunks, faiss_index_b, bm25_index")
    # else:
    #     # Test query
    #     test_query = "Điều kiện tốt nghiệp đại học là gì?"

    #     print(f"\n Test Query: {test_query}")
    #     print(f"⏳ Running retrieval pipeline...")

    #     try:
    #         results, metrics = retrieve_enhanced(test_query, verbose=True)
            
    #         # Display detailed results
    #         print(f"\n" + "="*70)
    #         print(f" DETAILED METRICS")
    #         print(f"="*70)
            
    #         print(f"\n Retrieval Metrics:")
    #         print(f"   • Method: {metrics['retrieval_method']}")
    #         print(f"   • Hybrid results: {metrics['hybrid_count']}")
    #         print(f"   • HyDE results: {metrics['hyde_count']}")
    #         print(f"   • Merged results: {metrics['merged_count']}")
    #         if metrics.get('duplicates_removed', 0) > 0:
    #             print(f"   • Duplicates removed: {metrics['duplicates_removed']}")
            
    #         print(f"\n Diversity Metrics:")
    #         print(f"   • MMR selected: {metrics['mmr_count']}")
    #         print(f"   • MMR time: {metrics['mmr_time']:.3f}s")
            
    #         print(f"\n Reranking Metrics:")
    #         print(f"   • ViRanker method: {metrics['rerank_stage2_method']}")
    #         print(f"   • ViRanker results: {metrics.get('stage2_count', 'N/A')}")
    #         print(f"   • Reranking time: {metrics.get('rerank_time', 0):.3f}s")
            
    #         print(f"\n⏱  Timing Breakdown:")
    #         print(f"   • Retrieval: {metrics['retrieval_time']:.3f}s ({metrics['retrieval_time']/metrics['total_time']*100:.1f}%)")
    #         if metrics.get('merge_time', 0) > 0:
    #             print(f"   • Merge: {metrics['merge_time']:.3f}s ({metrics['merge_time']/metrics['total_time']*100:.1f}%)")
    #         if metrics.get('mmr_time', 0) > 0:
    #             print(f"   • MMR: {metrics['mmr_time']:.3f}s ({metrics['mmr_time']/metrics['total_time']*100:.1f}%)")
    #         if metrics.get('rerank_time', 0) > 0:
    #             print(f"   • Reranking: {metrics['rerank_time']:.3f}s ({metrics['rerank_time']/metrics['total_time']*100:.1f}%)")
    #         print(f"   • TOTAL: {metrics['total_time']:.3f}s")
            
    #         print(f"\n Final Results:")
    #         print(f"   • Total chunks: {metrics['final_count']}")
    #         print(f"   • Average score: {np.mean([r.score for r in results]):.4f}")
    #         print(f"   • Score range: {min([r.score for r in results]):.4f} - {max([r.score for r in results]):.4f}")
            
    #         # Display top results
    #         print(f"\n" + "="*70)
    #         print(f" TOP {len(results)} RESULTS")
    #         print(f"="*70)
            
    #         for i, result in enumerate(results, 1):
    #             print(f"\n{''*70}")
    #             print(f" RESULT #{i}")
    #             print(f"{''*70}")
    #             print(f" Score: {result.score:.4f}")
    #             print(f" Method: {result.method}")
                
    #             print(f"\n Text Preview:")
    #             # Show first 300 chars
    #             text_preview = result.text[:300]
    #             if len(result.text) > 300:
    #                 text_preview += "..."
    #             print(f"   {text_preview}")
                
    #             print(f"\n Metadata:")
    #             # File-level metadata
    #             if 'filename' in result.metadata:
    #                 print(f"    File: {result.metadata['filename']}")
    #             if 'doc_type' in result.metadata:
    #                 print(f"    Type: {result.metadata['doc_type']}")
    #             if 'year' in result.metadata:
    #                 print(f"    Year: {result.metadata['year']}")
                
    #             # Content-level metadata
    #             if 'chapter' in result.metadata and result.metadata['chapter']:
    #                 print(f"    Chapter: {result.metadata['chapter']}")
    #             if 'article' in result.metadata and result.metadata['article']:
    #                 print(f"    Article: {result.metadata['article']}")
    #             if 'article_title' in result.metadata and result.metadata['article_title']:
    #                 print(f"    Title: {result.metadata['article_title']}")
    #             if 'hierarchical_path' in result.metadata and result.metadata['hierarchical_path']:
    #                 print(f"     Path: {result.metadata['hierarchical_path']}")
    #             if 'page' in result.metadata:
    #                 print(f"    Page: {result.metadata['page']}")
            
    #         print(f"\n" + "="*70)
    #         print(" RETRIEVAL TEST COMPLETED SUCCESSFULLY!")
    #         print("="*70)
            
    #     except Exception as e:
    #         print(f"\n" + "="*70)
    #         print(f" ERROR DURING RETRIEVAL")
    #         print(f"="*70)
    #         print(f"\n Error: {e}")
    #         print(f"\n Traceback:")
    #         import traceback
    #         traceback.print_exc()

# Always print completion message
print("\n" + "="*70)
print(" CELL 5 COMPLETE - RETRIEVAL READY!")
print("="*70)

print("\n Exported Functions:")
print("   • retrieve_enhanced(query, verbose=True) - Main retrieval function")
print("   • bm25_raw_retriever - BM25 retriever on raw chunk text")
print("   • bm25_contextualized_retriever - BM25 retriever on contextualized chunk text")
print("   • bm25_retriever - Default BM25 alias (contextualized)")
print("   • vector_retriever - Vector retriever on contextualized FAISS")
print("   • hybrid_retriever - Hybrid retriever")
if USE_HYDE:
    print("   • hyde_retriever - HyDE retriever")
if USE_MMR:
    print("   • mmr_selector - SemViQA QATC MMR selector")
if ENABLE_RERANKING:
    print("   • vi_reranker - ViRanker reranker")
if ENABLE_HIERARCHICAL_EXPANSION:
    print("   • hierarchical_expander - Child-to-parent context expander")

print("\n Usage:")
print("   results, metrics = retrieve_enhanced('your query here')")
print("   results, metrics = retrieve_enhanced('your query', verbose=False)  # Silent mode")
print("   • vector_retriever - Vector-only retrieval")
print("   • hybrid_retriever - Hybrid raw/context/vector retrieval")
if USE_HYDE:
    print("   • hyde_retriever - HyDE retriever")
if ENABLE_RERANKING:
    print("   • vi_reranker - ViRanker reranker")
if USE_MMR:
    print("   • mmr_selector - SemViQA QATC MMR selector")
if ENABLE_HIERARCHICAL_EXPANSION:
    print("   • hierarchical_expander - Child-to-parent context expander")

print("\n Next: Run Cell 6 for LLM Synthesis")
