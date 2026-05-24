"""Shared quality metrics for runtime scoring and offline benchmarking."""

from __future__ import annotations

from collections import Counter
import os
import re
import unicodedata
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

try:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except Exception:
    SmoothingFunction = None
    sentence_bleu = None


SUPPORTED_CLAIM_LABELS = {"entailment", "contradiction"}
FAITHFULNESS_CLAIM_LABELS = {"entailment", "hallucination", "contradiction"}
ENTAILED_CLAIM_LABELS = {"entailment"}
QUALITY_LABELS = ("entailment", "hallucination", "generic", "off_topic", "contradiction")
DEFAULT_BERTSCORE_MODEL = os.environ.get("RAG_BERTSCORE_MODEL", "xlm-roberta-base")

_BERT_SCORER_CACHE: Dict[str, Any] = {}
_BERT_SCORER_ERRORS: Dict[str, str] = {}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp_unit(value: Any) -> float:
    score = safe_float(value, 0.0)
    if score <= 0.0:
        return 0.0
    if score >= 1.0:
        return 1.0
    return score


def normalize_ascii_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", str(text or "").strip().lower())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    normalized = normalized.replace("đ", "d")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(text: str, min_len: int = 3) -> set[str]:
    return {
        tok
        for tok in re.findall(r"\w+", normalize_ascii_text(text))
        if len(tok) >= min_len
    }


def token_overlap(left: str, right: str, min_len: int = 3) -> float:
    left_tokens = tokenize(left, min_len=min_len)
    right_tokens = tokenize(right, min_len=min_len)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens))


def reference_tokens(text: str) -> List[str]:
    return re.findall(r"\w+", str(text or "").strip().lower(), flags=re.UNICODE)


def empty_reference_metrics() -> Dict[str, Any]:
    return {
        "bleu_1": 0.0,
        "bleu_4": 0.0,
        "rouge_1": 0.0,
        "rouge_2": 0.0,
        "rouge_l": 0.0,
        "bertscore_precision": 0.0,
        "bertscore_recall": 0.0,
        "bertscore_f1": 0.0,
        "bertscore_available": False,
        "bertscore_model": DEFAULT_BERTSCORE_MODEL,
        "bertscore_error": "",
    }


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[idx:idx + n]) for idx in range(max(0, len(tokens) - n + 1)))


def bleu_score(reference: str, hypothesis: str, n: int = 4) -> float:
    ref_tokens = reference_tokens(reference)
    hyp_tokens = reference_tokens(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0

    if sentence_bleu is not None and SmoothingFunction is not None:
        if n <= 1:
            weights = (1.0, 0.0, 0.0, 0.0)
        elif n == 2:
            weights = (0.5, 0.5, 0.0, 0.0)
        elif n == 3:
            weights = (1 / 3, 1 / 3, 1 / 3, 0.0)
        else:
            weights = (0.25, 0.25, 0.25, 0.25)
        try:
            return clamp_unit(
                sentence_bleu(
                    [ref_tokens],
                    hyp_tokens,
                    weights=weights,
                    smoothing_function=SmoothingFunction().method1,
                )
            )
        except Exception:
            pass

    if n <= 1:
        ref_counts = Counter(ref_tokens)
        hyp_counts = Counter(hyp_tokens)
        overlap = sum(min(count, ref_counts[token]) for token, count in hyp_counts.items())
        return clamp_unit(overlap / max(1, len(hyp_tokens)))

    hyp_counts = _ngram_counts(hyp_tokens, n)
    ref_counts = _ngram_counts(ref_tokens, n)
    if not hyp_counts:
        return 0.0
    overlap = sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
    return clamp_unit(overlap / max(1, sum(hyp_counts.values())))


def _f1_score(precision: float, recall: float) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def rouge_n_f1(reference: str, hypothesis: str, n: int = 1) -> float:
    ref_tokens = reference_tokens(reference)
    hyp_tokens = reference_tokens(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0

    ref_counts = _ngram_counts(ref_tokens, n)
    hyp_counts = _ngram_counts(hyp_tokens, n)
    if not ref_counts or not hyp_counts:
        return 0.0

    overlap = sum(min(count, hyp_counts[gram]) for gram, count in ref_counts.items())
    precision = overlap / max(1, sum(hyp_counts.values()))
    recall = overlap / max(1, sum(ref_counts.values()))
    return clamp_unit(_f1_score(precision, recall))


def _lcs_length(left: List[str], right: List[str]) -> int:
    if not left or not right:
        return 0
    rows = len(left) + 1
    cols = len(right) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            if left[i - 1] == right[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    ref_tokens = reference_tokens(reference)
    hyp_tokens = reference_tokens(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0

    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / max(1, len(hyp_tokens))
    recall = lcs / max(1, len(ref_tokens))
    return clamp_unit(_f1_score(precision, recall))


def _get_bert_scorer(model_type: str = DEFAULT_BERTSCORE_MODEL):
    if model_type in _BERT_SCORER_CACHE:
        return _BERT_SCORER_CACHE[model_type]
    if model_type in _BERT_SCORER_ERRORS:
        raise RuntimeError(_BERT_SCORER_ERRORS[model_type])

    try:
        import torch
        from bert_score import BERTScorer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        scorer = BERTScorer(
            model_type=model_type,
            device=device,
            rescale_with_baseline=False,
        )
        _BERT_SCORER_CACHE[model_type] = scorer
        return scorer
    except Exception as exc:
        _BERT_SCORER_ERRORS[model_type] = str(exc)
        raise


def bert_score_metrics(
    reference: str,
    hypothesis: str,
    model_type: str = DEFAULT_BERTSCORE_MODEL,
) -> Dict[str, Any]:
    metrics = {
        "bertscore_precision": 0.0,
        "bertscore_recall": 0.0,
        "bertscore_f1": 0.0,
        "bertscore_available": False,
        "bertscore_model": model_type,
        "bertscore_error": "",
    }
    if not reference or not hypothesis:
        return metrics

    try:
        scorer = _get_bert_scorer(model_type=model_type)
        precision, recall, f1 = scorer.score([hypothesis], [reference])
        metrics["bertscore_precision"] = clamp_unit(float(precision[0].item()))
        metrics["bertscore_recall"] = clamp_unit(float(recall[0].item()))
        metrics["bertscore_f1"] = clamp_unit(float(f1[0].item()))
        metrics["bertscore_available"] = True
        return metrics
    except Exception as exc:
        metrics["bertscore_error"] = str(exc)
        return metrics


def compute_reference_text_metrics(
    reference: str,
    hypothesis: str,
    *,
    include_bertscore: bool = True,
    bertscore_model: str = DEFAULT_BERTSCORE_MODEL,
) -> Dict[str, Any]:
    metrics = dict(empty_reference_metrics())
    if not reference or not hypothesis:
        if include_bertscore:
            metrics["bertscore_model"] = bertscore_model
        return metrics

    metrics["bleu_1"] = bleu_score(reference, hypothesis, n=1)
    metrics["bleu_4"] = bleu_score(reference, hypothesis, n=4)
    metrics["rouge_1"] = rouge_n_f1(reference, hypothesis, n=1)
    metrics["rouge_2"] = rouge_n_f1(reference, hypothesis, n=2)
    metrics["rouge_l"] = rouge_l_f1(reference, hypothesis)

    if include_bertscore:
        metrics.update(bert_score_metrics(reference, hypothesis, model_type=bertscore_model))
    else:
        metrics["bertscore_model"] = bertscore_model

    return metrics


def combine_reference_answer_similarity(
    lexical_overlap_score: float,
    reference_metrics: Optional[Dict[str, Any]] = None,
) -> float:
    reference_metrics = reference_metrics or {}
    weighted_components = [
        ("lexical_overlap", clamp_unit(lexical_overlap_score), 0.10, True),
        ("bleu_1", clamp_unit(reference_metrics.get("bleu_1", 0.0)), 0.15, True),
        ("bleu_4", clamp_unit(reference_metrics.get("bleu_4", 0.0)), 0.15, True),
        ("rouge_l", clamp_unit(reference_metrics.get("rouge_l", 0.0)), 0.25, True),
        (
            "bertscore_f1",
            clamp_unit(reference_metrics.get("bertscore_f1", 0.0)),
            0.35,
            bool(reference_metrics.get("bertscore_available", False)),
        ),
    ]

    total_weight = sum(weight for _, _, weight, available in weighted_components if available)
    if total_weight <= 0.0:
        return clamp_unit(lexical_overlap_score)

    blended = sum(score * weight for _, score, weight, available in weighted_components if available)
    return clamp_unit(blended / total_weight)


def cosine_similarity(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.size == 0 or right_arr.size == 0:
        return 0.0
    denom = np.linalg.norm(left_arr) * np.linalg.norm(right_arr)
    if denom <= 0.0:
        return 0.0
    return clamp_unit(np.dot(left_arr, right_arr) / denom)


def embedding_similarity(
    left: str,
    right: str,
    embed_fn: Optional[Callable[[str], Any]] = None,
) -> float:
    if not left or not right:
        return 0.0
    if embed_fn is not None:
        try:
            return cosine_similarity(embed_fn(left), embed_fn(right))
        except Exception:
            pass
    return token_overlap(left, right)


def build_legal_citation(metadata: Dict[str, Any]) -> str:
    metadata = metadata or {}
    document_title = str(metadata.get("document_title", "") or "").strip()
    decision_number = str(metadata.get("decision_number", "") or "").strip()
    decision_code = str(metadata.get("decision_code", "") or "").strip()
    parts: List[str] = []

    if document_title:
        if decision_number and decision_code:
            parts.append(f"{document_title} số {decision_number}/{decision_code}")
        else:
            parts.append(document_title)
    elif metadata.get("filename"):
        parts.append(str(metadata.get("filename")).strip())

    for key, prefix in (
        ("page", "trang"),
        ("chapter", ""),
        ("article", ""),
        ("section", ""),
        ("point", ""),
    ):
        value = str(metadata.get(key, "") or "").strip()
        if not value:
            continue
        parts.append(f"{prefix} {value}".strip())

    return " | ".join(part for part in parts if part)


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


def parse_legal_reference_signature(text: str) -> Dict[str, Any]:
    normalized = normalize_ascii_text(text)
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


def legal_reference_similarity(expected_source: str, candidate_source: str) -> float:
    expected = parse_legal_reference_signature(expected_source)
    candidate = parse_legal_reference_signature(candidate_source)

    weighted_matches = 0.0
    weighted_total = 0.0
    field_weights = (
        ("document_ref", 0.30),
        ("chapter", 0.10),
        ("article", 0.30),
        ("section", 0.20),
        ("point", 0.10),
    )

    for field, weight in field_weights:
        expected_value = str(expected.get(field, "") or "").strip()
        if not expected_value:
            continue
        weighted_total += weight
        if str(candidate.get(field, "") or "").strip() == expected_value:
            weighted_matches += weight

    expected_keywords = set(expected.get("keywords", set()) or set())
    candidate_keywords = set(candidate.get("keywords", set()) or set())
    if not str(expected.get("document_ref", "") or "").strip() and expected_keywords:
        weighted_total += 0.40
        weighted_matches += 0.40 * (len(expected_keywords & candidate_keywords) / max(1, len(expected_keywords)))

    if weighted_total > 0.0:
        return weighted_matches / weighted_total
    return token_overlap(expected_source, candidate_source)


def claim_label_counts(claims: Optional[Iterable[Dict[str, Any]]]) -> Dict[str, int]:
    counts = {label: 0 for label in QUALITY_LABELS}
    for claim in list(claims or []):
        label = str(claim.get("label", "") or "").strip().lower()
        if label in counts:
            counts[label] += 1
    return counts


def claim_label_rates(claims: Optional[Iterable[Dict[str, Any]]]) -> Dict[str, float]:
    counts = claim_label_counts(claims)
    total = sum(counts.values())
    if total <= 0:
        return {f"{label}_rate": 0.0 for label in QUALITY_LABELS}
    return {
        f"{label}_rate": counts[label] / total
        for label in QUALITY_LABELS
    }


def claim_citation_coverage(
    claims: Optional[Iterable[Dict[str, Any]]] = None,
    evidence_spans: Optional[Iterable[Dict[str, Any]]] = None,
    labels: Optional[set[str]] = None,
) -> float:
    labels = set(labels or SUPPORTED_CLAIM_LABELS)
    supported_claims = [
        claim
        for claim in list(claims or [])
        if str(claim.get("label", "") or "").strip().lower() in labels
    ]
    if supported_claims:
        covered = 0
        for claim in supported_claims:
            if str(claim.get("citation", "") or "").strip() or str(claim.get("evidence", "") or "").strip():
                covered += 1
        return covered / len(supported_claims)

    supported_spans = [
        span
        for span in list(evidence_spans or [])
        if str(span.get("label", "") or "").strip().lower() in labels
    ]
    if not supported_spans:
        return 0.0
    covered_spans = sum(
        1
        for span in supported_spans
        if str(span.get("citation", "") or "").strip() or str(span.get("evidence", "") or "").strip()
    )
    return covered_spans / len(supported_spans)


def mean_claim_provenance(
    claims: Optional[Iterable[Dict[str, Any]]] = None,
    *,
    labels: Optional[set[str]] = None,
) -> float:
    labels = set(labels or ENTAILED_CLAIM_LABELS)
    values = [
        clamp_unit(claim.get("provenance_score", 0.0))
        for claim in list(claims or [])
        if str(claim.get("label", "") or "").strip().lower() in labels
    ]
    return float(np.mean(values)) if values else 0.0


def compute_faithfulness_score(
    claims: Optional[Iterable[Dict[str, Any]]] = None,
    *,
    groundedness_score: float = 0.0,
) -> float:
    considered_claims = [
        claim
        for claim in list(claims or [])
        if str(claim.get("label", "") or "").strip().lower() in FAITHFULNESS_CLAIM_LABELS
    ]
    if not considered_claims:
        return clamp_unit(groundedness_score)

    entailed_claims = sum(
        1
        for claim in considered_claims
        if str(claim.get("label", "") or "").strip().lower() == "entailment"
    )
    return clamp_unit(entailed_claims / len(considered_claims))


def compute_citation_support_score(
    claims: Optional[Iterable[Dict[str, Any]]] = None,
    evidence_spans: Optional[Iterable[Dict[str, Any]]] = None,
    *,
    provenance_score: float = 0.0,
) -> Dict[str, float]:
    claims = list(claims or [])
    entailed_claims = [
        claim
        for claim in claims
        if str(claim.get("label", "") or "").strip().lower() in ENTAILED_CLAIM_LABELS
    ]
    entailed_citation_coverage = claim_citation_coverage(
        claims=claims,
        evidence_spans=evidence_spans,
        labels=ENTAILED_CLAIM_LABELS,
    )
    entailed_provenance_score = mean_claim_provenance(claims, labels=ENTAILED_CLAIM_LABELS)

    if entailed_claims:
        citation_support_score = clamp_unit(
            0.65 * entailed_citation_coverage + 0.35 * entailed_provenance_score
        )
    else:
        citation_support_score = clamp_unit(
            0.5 * clamp_unit(provenance_score) +
            0.5 * claim_citation_coverage(claims=claims, evidence_spans=evidence_spans)
        ) if not claims else 0.0

    return {
        "citation_support_score": citation_support_score,
        "entailed_claim_citation_coverage": entailed_citation_coverage,
        "entailed_claim_provenance_score": entailed_provenance_score,
    }


def compute_answer_relevance_score(
    query: str,
    answer: str,
    claims: Optional[Iterable[Dict[str, Any]]] = None,
    embed_fn: Optional[Callable[[str], Any]] = None,
) -> float:
    base_similarity = embedding_similarity(query, answer, embed_fn=embed_fn)
    rates = claim_label_rates(claims)
    score = (
        base_similarity
        - 0.15 * rates.get("off_topic_rate", 0.0)
        - 0.05 * rates.get("generic_rate", 0.0)
    )
    return clamp_unit(score)


def compute_runtime_quality_metrics(
    query: str,
    answer: str,
    claims: Optional[Iterable[Dict[str, Any]]],
    evidence_spans: Optional[Iterable[Dict[str, Any]]],
    groundedness_score: float,
    provenance_score: float,
    embed_fn: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    counts = claim_label_counts(claims)
    rates = claim_label_rates(claims)
    faithfulness_score = compute_faithfulness_score(
        claims=claims,
        groundedness_score=groundedness_score,
    )
    citation_coverage = claim_citation_coverage(claims=claims, evidence_spans=evidence_spans)
    citation_support_metrics = compute_citation_support_score(
        claims=claims,
        evidence_spans=evidence_spans,
        provenance_score=provenance_score,
    )
    answer_relevance_score = compute_answer_relevance_score(
        query,
        answer,
        claims=claims,
        embed_fn=embed_fn,
    )
    return {
        "faithfulness_score": faithfulness_score,
        "answer_relevance_score": answer_relevance_score,
        "citation_support_score": citation_support_metrics["citation_support_score"],
        "claim_citation_coverage": citation_coverage,
        "entailed_claim_citation_coverage": citation_support_metrics["entailed_claim_citation_coverage"],
        "entailed_claim_provenance_score": citation_support_metrics["entailed_claim_provenance_score"],
        "hallucination_rate": rates.get("hallucination_rate", 0.0),
        "contradiction_rate": rates.get("contradiction_rate", 0.0),
        "generic_rate": rates.get("generic_rate", 0.0),
        "off_topic_rate": rates.get("off_topic_rate", 0.0),
        "entailed_claims": counts.get("entailment", 0),
        "hallucinated_claims": counts.get("hallucination", 0),
        "generic_claims": counts.get("generic", 0),
        "off_topic_claims": counts.get("off_topic", 0),
        "contradictory_claims": counts.get("contradiction", 0),
    }


def empty_quality_metrics() -> Dict[str, Any]:
    return {
        "faithfulness_score": 0.0,
        "answer_relevance_score": 0.0,
        "citation_support_score": 0.0,
        "claim_citation_coverage": 0.0,
        "entailed_claim_citation_coverage": 0.0,
        "entailed_claim_provenance_score": 0.0,
        "hallucination_rate": 0.0,
        "contradiction_rate": 0.0,
        "generic_rate": 0.0,
        "off_topic_rate": 0.0,
        "entailed_claims": 0,
        "hallucinated_claims": 0,
        "generic_claims": 0,
        "off_topic_claims": 0,
        "contradictory_claims": 0,
    }


def extract_reference_candidates(
    citations: Optional[Iterable[str]],
    retrieved_chunks: Optional[Iterable[Dict[str, Any]]],
    top_k: int = 5,
) -> List[str]:
    candidates: List[str] = []
    seen = set()

    for citation in list(citations or []):
        item = str(citation or "").strip()
        normalized = normalize_ascii_text(item)
        if not item or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(item)

    for chunk in list(retrieved_chunks or [])[: max(0, int(top_k))]:
        citation = str(chunk.get("citation", "") or "").strip()
        if not citation:
            citation = build_legal_citation(chunk.get("metadata", {}) or {})
        normalized = normalize_ascii_text(citation)
        if not citation or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(citation)

    return candidates


def has_annotation_data(annotation: Optional[Dict[str, Any]]) -> bool:
    annotation = annotation or {}
    return any(bool(annotation.get(key)) for key in ("relevant_legal_units", "relevant_chunk_ids", "required_claims", "acceptable_citations"))


def compute_retrieval_proxy_metrics(
    expected_source: str,
    citations: Optional[Iterable[str]],
    retrieved_chunks: Optional[Iterable[Dict[str, Any]]],
    annotation: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
    match_threshold: float = 0.60,
) -> Dict[str, Any]:
    annotation = annotation or {}
    candidates = extract_reference_candidates(citations, retrieved_chunks, top_k=top_k)
    relevant_chunk_ids = [
        str(item or "").strip()
        for item in list(annotation.get("relevant_chunk_ids", []) or [])
        if str(item or "").strip()
    ]
    relevant_units = [
        str(item or "").strip()
        for item in list(annotation.get("relevant_legal_units", []) or [])
        if str(item or "").strip()
    ]
    coverage_state = "annotation_augmented" if has_annotation_data(annotation) else "proxy_only"
    metric_basis = "legal_reference_proxy"

    if relevant_chunk_ids:
        metric_basis = "chunk_id_exact"
        candidate_chunk_ids = []
        seen_chunk_ids = set()
        for chunk in list(retrieved_chunks or [])[: max(0, int(top_k))]:
            chunk_id = str(
                chunk.get("chunk_id", "") or
                chunk.get("source_chunk_id", "") or
                (chunk.get("metadata", {}) or {}).get("chunk_id", "")
            ).strip()
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            candidate_chunk_ids.append(chunk_id)
        relevant_chunk_id_set = set(relevant_chunk_ids)
        hit_count = sum(1 for chunk_id in candidate_chunk_ids if chunk_id in relevant_chunk_id_set)
        recall_proxy = hit_count / len(relevant_chunk_id_set) if relevant_chunk_id_set else 0.0
        precision_proxy = hit_count / len(candidate_chunk_ids) if candidate_chunk_ids else 0.0
    elif relevant_units:
        metric_basis = "legal_unit_annotation_proxy"
        recall_hits = 0
        for unit in relevant_units:
            best_match = max((legal_reference_similarity(unit, candidate) for candidate in candidates), default=0.0)
            if best_match >= match_threshold:
                recall_hits += 1
        recall_proxy = recall_hits / len(relevant_units)
        precision_hits = sum(
            1
            for candidate in candidates
            if max((legal_reference_similarity(unit, candidate) for unit in relevant_units), default=0.0) >= match_threshold
        )
        precision_proxy = precision_hits / len(candidates) if candidates else 0.0
    else:
        best_match = max((legal_reference_similarity(expected_source, candidate) for candidate in candidates), default=0.0)
        recall_proxy = best_match if str(expected_source or "").strip() else 0.0
        precision_hits = sum(
            1
            for candidate in candidates
            if legal_reference_similarity(expected_source, candidate) >= match_threshold
        )
        precision_proxy = precision_hits / len(candidates) if candidates else 0.0

    return {
        "legal_unit_recall_proxy_at_k": clamp_unit(recall_proxy),
        "legal_unit_precision_proxy_at_k": clamp_unit(precision_proxy),
        "retrieval_dimension_score": clamp_unit(0.5 * recall_proxy + 0.5 * precision_proxy),
        "retrieval_candidate_count": len(candidates),
        "annotation_coverage_state": coverage_state,
        "retrieval_metric_basis": metric_basis,
    }


def compute_citation_dimension_score(
    expected_source: str,
    citations: Optional[Iterable[str]],
    annotation: Optional[Dict[str, Any]] = None,
) -> float:
    annotation = annotation or {}
    actual_citations = [str(item or "").strip() for item in list(citations or []) if str(item or "").strip()]
    if not actual_citations:
        return 0.0

    acceptable_citations = [
        str(item or "").strip()
        for item in list(annotation.get("acceptable_citations", []) or [])
        if str(item or "").strip()
    ]
    expected_candidates = acceptable_citations or ([str(expected_source or "").strip()] if str(expected_source or "").strip() else [])
    if not expected_candidates:
        return 0.0

    per_expected = [
        max((legal_reference_similarity(expected_candidate, actual) for actual in actual_citations), default=0.0)
        for expected_candidate in expected_candidates
    ]
    return clamp_unit(sum(per_expected) / len(per_expected))


def compute_benchmark_quality_metrics(
    expected_source: str,
    citations: Optional[Iterable[str]],
    retrieved_chunks: Optional[Iterable[Dict[str, Any]]],
    runtime_quality_metrics: Optional[Dict[str, Any]] = None,
    annotation: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    runtime_quality_metrics = runtime_quality_metrics or {}
    retrieval_metrics = compute_retrieval_proxy_metrics(
        expected_source=expected_source,
        citations=citations,
        retrieved_chunks=retrieved_chunks,
        annotation=annotation,
        top_k=top_k,
    )
    citation_dimension_score = compute_citation_dimension_score(
        expected_source=expected_source,
        citations=citations,
        annotation=annotation,
    )
    faithfulness_score = clamp_unit(runtime_quality_metrics.get("faithfulness_score", 0.0))
    answer_relevance_score = clamp_unit(runtime_quality_metrics.get("answer_relevance_score", 0.0))
    benchmark_composite_score = clamp_unit(
        0.25 * retrieval_metrics["retrieval_dimension_score"]
        + 0.25 * faithfulness_score
        + 0.25 * answer_relevance_score
        + 0.25 * citation_dimension_score
    )
    return {
        **retrieval_metrics,
        "citation_dimension_score": citation_dimension_score,
        "benchmark_composite_score": benchmark_composite_score,
    }
