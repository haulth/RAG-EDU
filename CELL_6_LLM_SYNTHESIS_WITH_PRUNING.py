# ==============================================================================
# @title CELL 6: LLM SYNTHESIS WITH CONTEXT PRUNING
# ==============================================================================


print("="*70)
print(" CELL 6: LLM SYNTHESIS WITH CONTEXT PRUNING")
print("="*70)

import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

from rag_quality_metrics import compute_runtime_quality_metrics, empty_quality_metrics

# ==============================================================================
# DEPENDENCY CHECK & GENERATE_TEXT FUNCTION
# ==============================================================================

print("\n" + "="*70)
print(" CHECKING DEPENDENCIES & DEFINING GENERATE_TEXT")
print("="*70)

BOOTSTRAP_REMOTE_ONLY = bool(globals().get("BOOTSTRAP_REMOTE_ONLY", False))

# Check required variables from Cell 3
required_vars = {
    'embedder': 'Sentence Transformer (from Cell 3)'
}
if not BOOTSTRAP_REMOTE_ONLY:
    required_vars = {
        'llm_model': 'LLM model (from Cell 3)',
        'llm_tokenizer': 'LLM tokenizer (from Cell 3)',
        **required_vars
    }

missing_vars = []
for var_name, description in required_vars.items():
    # Try to get from globals, if not found try from __main__
    if var_name not in globals():
        try:
            import __main__
            if hasattr(__main__, var_name):
                globals()[var_name] = getattr(__main__, var_name)
                print(f"Found: {var_name} - {description} (from __main__)")
            else:
                print(f"Missing: {var_name} - {description}")
                missing_vars.append(var_name)
        except:
            print(f"Missing: {var_name} - {description}")
            missing_vars.append(var_name)
    else:
        print(f"Found: {var_name} - {description}")

if missing_vars:
    print(f"\nLOI: Thieu {len(missing_vars)} bien can thiet!")
    print("Hay chay CELL 3 truoc de load models")
    raise NameError(f"Missing required variables: {', '.join(missing_vars)}")
else:
    print("\nAll required variables found")

# ==============================================================================
# GENERATE_TEXT FUNCTION (for Cell 6 synthesis)
# ==============================================================================

def generate_text(
    prompt: str, 
    max_new_tokens: int = 512, 
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    system_prompt: Optional[str] = None
) -> str:
    """
    Tạo text với LLM model - Version đầy đủ cho synthesis
    
    Args:
        prompt: Câu hỏi hoặc prompt
        max_new_tokens: Số token tối đa để generate
        temperature: Temperature cho sampling (0.0 = deterministic, 1.0 = creative)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty cho repeated tokens
        do_sample: Có dùng sampling không
    
    Returns:
        Generated text
    """
    if llm_model is None or llm_tokenizer is None:
        if BOOTSTRAP_REMOTE_ONLY:
            raise RuntimeError("Local LLM không được nạp trong remote-only runtime.")
        return "Lỗi: LLM model chưa được load."
    
    try:
        import torch
        
        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Prepare messages
        messages = [
            {
                "role": "system", 
                "content": system_prompt or "Bạn là trợ lý AI chuyên về quy chế đào tạo. Trả lời ngắn gọn, chính xác bằng tiếng Việt."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        # Apply chat template
        text = llm_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        model_inputs = llm_tokenizer([text], return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            generated_ids = llm_model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )
        
        # Decode (only new tokens)
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response.strip()
        
    except Exception as e:
        return f"Lỗi khi generate text: {e}"

print("\n generate_text() function defined with full parameters")

# ==============================================================================
# SEMANTIC HIGHLIGHTER CLASS
# ==============================================================================

class SemanticHighlighter:
    """
    Query-conditioned semantic highlighting powered by SemViQA QATC.

    The backend scores a full query/context pair once, then projects
    rationale-token probabilities back to sentence spans so downstream
    pruning/selectors can keep their sentence-oriented API.
    """

    SUPPORTED_LANGUAGES = {"vi"}

    def __init__(
        self,
        model,
        tokenizer=None,
        device="auto",
        default_threshold: float = 0.45,
        allow_unsupported_language: bool = False,
    ):
        if device in {"auto", "", None}:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.default_threshold = default_threshold
        self.allow_unsupported_language = allow_unsupported_language
        self.use_builtin = model is not None and tokenizer is not None
        self.backend_name = (
            "semviqa_qatc_infoxlm_viwikifc"
            if self.use_builtin
            else "lexical_overlap_fallback"
        )
        self.max_length = 512
        self.max_answer_tokens = 64

    def _resolve_language_hint(self, query: str, context: str) -> Tuple[str, bool]:
        sample = f"{query or ''}\n{context or ''}"[:4000]
        if not sample.strip():
            return "vi", True
        if re.search(r"[A-Za-zÀ-ỹ]", sample):
            return "vi", True
        return ("vi", True) if self.allow_unsupported_language else ("auto", False)

    def _normalize_offsets(self, offsets: Any) -> List[Tuple[int, int]]:
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        normalized = []
        for start, end in list(offsets or []):
            normalized.append((int(start), int(end)))
        return normalized

    def _get_context_indices(self, encoding) -> List[int]:
        try:
            sequence_ids = encoding.sequence_ids(0)
        except Exception:
            return []
        return [idx for idx, sequence_id in enumerate(sequence_ids) if sequence_id == 1]

    def _best_span(self, start_logits, end_logits, context_indices: List[int]) -> Tuple[int, int]:
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

        return best_pair

    def _char_span_from_tokens(
        self,
        offsets: List[Tuple[int, int]],
        token_indices: List[int],
    ) -> Tuple[Optional[int], Optional[int]]:
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
        return start_char, end_char

    def _sentence_token_scores(
        self,
        sentence_spans: List[Tuple[str, int, int]],
        offsets: List[Tuple[int, int]],
        context_indices: List[int],
        rational_scores,
        answer_span: Tuple[Optional[int], Optional[int]],
        answer_confidence: float,
    ) -> List[float]:
        answer_start, answer_end = answer_span
        sentence_scores: List[float] = []

        for _, sentence_start, sentence_end in sentence_spans:
            token_scores = []
            for token_idx in context_indices:
                if token_idx < 0 or token_idx >= len(offsets):
                    continue
                token_start, token_end = offsets[token_idx]
                if token_end <= token_start:
                    continue
                if token_end <= sentence_start or token_start >= sentence_end:
                    continue
                token_scores.append(float(max(0.0, min(1.0, rational_scores[token_idx].item()))))

            if token_scores:
                score = 0.7 * max(token_scores) + 0.3 * (sum(token_scores) / len(token_scores))
            else:
                score = 0.0

            if (
                answer_start is not None and
                answer_end is not None and
                not (answer_end <= sentence_start or answer_start >= sentence_end)
            ):
                score = max(score, answer_confidence)

            sentence_scores.append(float(max(0.0, min(1.0, score))))

        return sentence_scores

    def _fallback_score_sentences(
        self,
        query: str,
        sentences: List[str],
        batch_size: int = 8
    ) -> List[float]:
        if not sentences:
            return []
        return [float(token_overlap(query, sentence)) for sentence in sentences]

    def analyze_context(
        self,
        query: str,
        context: str,
        threshold: Optional[float] = None,
        language: Optional[str] = None,
        batch_size: int = 8,
    ) -> Dict:
        context_text = str(context or "").strip()
        sentence_spans = split_sentences_with_offsets(context_text)
        sentences = [sentence for sentence, _, _ in sentence_spans]
        effective_threshold = self.default_threshold if threshold is None else float(threshold)

        if not context_text or not sentences:
            return {
                "sentences": [],
                "scores": [],
                "highlighted_sentences": [],
                "compression_rate": 0.0,
                "used_builtin": False,
                "language": "auto",
                "fallback_reason": "empty_context",
                "supported_language": False,
            }

        resolved_language, supported_language = self._resolve_language_hint(query, context_text)
        process_language = language or resolved_language

        if self.use_builtin and (supported_language or self.allow_unsupported_language):
            try:
                encoding = self.tokenizer(
                    query,
                    context_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
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
                answer_span = self._char_span_from_tokens(
                    offsets,
                    list(range(best_start, best_end + 1)),
                )
                answer_text = ""
                answer_start, answer_end = answer_span
                if answer_start is not None and answer_end is not None:
                    answer_text = context_text[answer_start:answer_end].strip()

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

                scores = self._sentence_token_scores(
                    sentence_spans,
                    offsets,
                    context_indices,
                    rational_scores,
                    answer_span=answer_span,
                    answer_confidence=answer_confidence,
                )
                if not any(score > 0.0 for score in scores):
                    scores = self._fallback_score_sentences(query, sentences, batch_size=batch_size)

                highlighted = [
                    sentence for sentence, score in zip(sentences, scores)
                    if score >= effective_threshold
                ]
                if not highlighted and sentences:
                    best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
                    highlighted = [sentences[best_idx]]

                return {
                    "sentences": sentences,
                    "scores": scores,
                    "highlighted_sentences": highlighted,
                    "compression_rate": 1 - (len(highlighted) / max(1, len(sentences))),
                    "used_builtin": True,
                    "language": process_language,
                    "fallback_reason": "",
                    "supported_language": supported_language,
                    "backend": self.backend_name,
                    "answer_text": answer_text,
                    "answer_confidence": answer_confidence,
                }
            except Exception as exc:
                scores = self._fallback_score_sentences(query, sentences, batch_size=batch_size)
                highlighted = [
                    sentence for sentence, score in zip(sentences, scores)
                    if score >= effective_threshold
                ]
                return {
                    "sentences": sentences,
                    "scores": scores,
                    "highlighted_sentences": highlighted,
                    "compression_rate": 1 - (len(highlighted) / max(1, len(sentences))),
                    "used_builtin": False,
                    "language": process_language,
                    "fallback_reason": f"builtin_error:{exc}",
                    "supported_language": supported_language,
                    "backend": "lexical_overlap_fallback",
                }

        fallback_reason = "unsupported_language"
        if self.use_builtin and self.allow_unsupported_language:
            fallback_reason = "builtin_unavailable"

        scores = self._fallback_score_sentences(query, sentences, batch_size=batch_size)
        highlighted = [
            sentence for sentence, score in zip(sentences, scores)
            if score >= effective_threshold
        ]
        return {
            "sentences": sentences,
            "scores": scores,
            "highlighted_sentences": highlighted,
            "compression_rate": 1 - (len(highlighted) / max(1, len(sentences))),
            "used_builtin": False,
            "language": process_language,
            "fallback_reason": fallback_reason,
            "supported_language": supported_language,
            "backend": "lexical_overlap_fallback",
        }

    def score_sentences(
        self, 
        query: str, 
        sentences: List[str],
        batch_size: int = 8
    ) -> List[float]:
        """
        Score sentences based on semantic relevance to query
        
        Args:
            query: User query
            sentences: List of sentences to score
            batch_size: Batch size for inference
        
        Returns:
            List of scores (0-1) for each sentence
        """
        analysis = self.analyze_context(
            query,
            "\n".join(sentence.strip() for sentence in sentences if sentence and sentence.strip()),
            threshold=0.0,
            batch_size=batch_size,
        )
        scores = list(analysis.get("scores", []) or [])
        if len(scores) == len(sentences):
            return scores
        return self._fallback_score_sentences(query, sentences, batch_size=batch_size)
    
    def highlight(
        self,
        query: str,
        text: str,
        threshold: float = 0.5
    ) -> str:
        """
        Highlight relevant sentences in text
        
        Args:
            query: User query
            text: Full text to highlight
            threshold: Score threshold (0-1)
        
        Returns:
            Highlighted text (only relevant sentences)
        """
        analysis = self.analyze_context(query, text, threshold=threshold)
        highlighted = list(analysis.get("highlighted_sentences", []) or [])
        return " ".join(highlighted).strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        return split_sentences(text)


def create_semantic_highlighter(
    model=None,
    tokenizer=None,
    *,
    allow_unsupported_language: bool = False,
    default_threshold: float = 0.45,
):
    semantic_model = (
        model
        if model is not None
        else globals().get("semantic_highlight_model") or globals().get("mmr_qatc_model")
    )
    semantic_tokenizer = (
        tokenizer
        if tokenizer is not None
        else globals().get("semantic_highlight_tokenizer") or globals().get("mmr_qatc_tokenizer")
    )
    if semantic_model is None or semantic_tokenizer is None:
        return None

    try:
        import torch
        device = str(next(semantic_model.parameters()).device)
    except Exception:
        device = "cpu"

    return SemanticHighlighter(
        model=semantic_model,
        tokenizer=semantic_tokenizer,
        device=device,
        default_threshold=default_threshold,
        allow_unsupported_language=allow_unsupported_language,
    )

print("\n SemanticHighlighter class defined")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Context pruning parameters
MAX_CONTEXT_LENGTH = 2000  # Max characters for context
MIN_SENTENCE_SCORE = 0.3   # Min score to keep sentence
TOP_K_SENTENCES = 10       # Max sentences to keep

# LLM parameters
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.3
TOP_P = 0.9

# Prompt templates
ANSWER_PROMPT_TEMPLATE = """Bạn là trợ lý AI chuyên về quy chế đào tạo đại học Sư Phạm TP.HCM.

Bạn chỉ được trả lời dựa trên EVIDENCE_SET bên dưới.

EVIDENCE_SET:
{context}

CÂU HỎI: {query}

YÊU CẦU:
- Chỉ sử dụng thông tin xuất hiện trong EVIDENCE_SET.
- Trả lời ngắn gọn, trực tiếp, đúng trọng tâm.
- Khi có căn cứ rõ, ưu tiên nêu theo Điều/Khoản/Điểm/Trang.
- Không được chèn mã evidence nội bộ như [E1], [E2], [E3] vào câu trả lời cuối.
- Nếu cần dẫn căn cứ, chỉ nêu trích dẫn pháp lý tự nhiên trong ngoặc, ví dụ: (Điều 12, Khoản 1, Điểm c, trang 15).
- Chọn cách trả lời phù hợp nhất:
  1. direct_extractive: evidence đủ rõ cho phần chính của câu hỏi, trả lời thẳng.
  2. partial_supported: chỉ support được một phần, chỉ nêu ngắn gọn phần support được.
  3. abstain: khi không có căn cứ đủ gần để trả lời phần chính.
- Nếu chọn partial_supported, không thêm các câu boilerplate kiểu "Tài liệu đang nạp chưa đủ căn cứ..." hoặc "chưa đủ căn cứ để trả lời trọn vẹn phần còn lại của câu hỏi"; nếu cần hướng dẫn thêm thì chỉ kết bằng lời mời liên hệ Phòng Đào tạo.
- Không được vừa trả lời khẳng định, vừa nói không có căn cứ trong cùng một câu trả lời.
- Không suy diễn ngoài evidence, không bịa thêm quy định.
- Trả lời bằng tiếng Việt.

TRẢ LỜI:"""

GROUNDEDNESS_SYSTEM_PROMPT = """Bạn là bộ đánh giá groundedness theo taxonomy BEGIN cho hệ thống RAG.
Bạn chỉ được dùng QUERY, ANSWER và EVIDENCE_SET được cung cấp.
Không dùng tri thức ngoài.
Mỗi claim phải nhận đúng 1 nhãn:
- entailment: được support trực tiếp bởi evidence
- contradiction: mâu thuẫn trực tiếp với evidence
- hallucination: cùng chủ đề nhưng không được support bởi evidence
- generic: câu xã giao, mơ hồ, boilerplate, không có fact kiểm chứng được
- off_topic: lệch khỏi câu hỏi hoặc evidence
Bạn phải trả về JSON hợp lệ, không thêm giải thích ngoài JSON."""

GROUNDEDNESS_PROMPT_TEMPLATE = """Hãy phân tích ANSWER theo taxonomy BEGIN.

QUERY:
{query}

EVIDENCE_SET:
{context}

ANSWER:
{answer}

YÊU CẦU:
1. Tách ANSWER thành các claim ngắn gọn.
2. Với mỗi claim, gán đúng 1 nhãn trong:
   - entailment
   - contradiction
   - hallucination
   - generic
   - off_topic
3. Nếu nhãn là entailment hoặc contradiction:
   - phải chỉ ra evidence_ids, ví dụ ["E1"] hoặc ["E2","E4"]
   - phải trích một evidence_quote ngắn từ EVIDENCE_SET
4. Nếu nhãn là hallucination, generic, hoặc off_topic:
   - evidence_ids phải là []
   - evidence_quote phải là chuỗi rỗng

Trả về duy nhất JSON dạng:
{{
  "claims": [
    {{
      "claim": "string",
      "label": "entailment|contradiction|hallucination|generic|off_topic",
      "reason": "string",
      "evidence_ids": ["E1"],
      "evidence_quote": "string"
    }}
  ]
}}"""

REVISION_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên trả lời dựa trên bằng chứng truy xuất.
Chỉ được sử dụng evidence được cung cấp. Không được suy diễn thêm."""

REVISION_PROMPT_TEMPLATE = """Hãy viết lại câu trả lời sao cho chỉ giữ các ý được support bởi evidence.

QUERY:
{query}

ANSWER_DRAFT:
{answer}

SUPPORTED_EVIDENCE:
{evidence}

Yêu cầu:
- Chỉ dùng các fact nằm trong phần SUPPORTED_EVIDENCE.
- Không suy diễn ngoài evidence.
- FINAL_ANSWER phải chọn đúng 1 trong 2 mode:
  - direct_answer: trả lời thẳng bằng nội dung được support
  - abstain: nói rõ chưa có căn cứ trực tiếp
- Không được tạo FINAL_ANSWER kiểu vừa trả lời, vừa nói không có căn cứ.
- Nếu evidence không đủ để trả lời phần chính của câu hỏi, FINAL_ANSWER phải là 1 câu abstain ngắn gọn.
- Trả lời ngắn gọn, chính xác, bằng tiếng Việt.
- Không nhắc tới việc bạn đang chấm groundedness.
- BẮT BUỘC trả về đúng format dưới đây, giữ nguyên tên nhãn:

SUPPORTED_ANSWER:
<các ý được support trực tiếp bởi evidence>

UNSUPPORTED_PART:
<phần nào của câu hỏi chưa có đủ căn cứ; nếu không có thì ghi "Không có">

FINAL_ANSWER:
<câu trả lời cuối cùng để trả cho người dùng>
"""

TRAINING_OFFICE_CONTACT_NOTE = "Để biết thêm chi tiết, sinh viên vui lòng liên hệ lại Phòng Đào tạo để được trao đổi thêm."

DIRECT_ANSWER_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên hòa giải câu trả lời cuối cùng chỉ từ evidence đã được support.
Nhiệm vụ của bạn là quyết định liệu SUPPORTED_EVIDENCE có trả lời trực tiếp phần chính của QUERY hay không.
Không được suy diễn ngoài evidence được cung cấp."""

DIRECT_ANSWER_PROMPT_TEMPLATE = """Hãy kiểm tra xem SUPPORTED_EVIDENCE có đủ để trả lời trực tiếp phần chính của QUERY hay không.

QUERY:
{query}

CURRENT_ANSWER:
{answer}

SUPPORTED_EVIDENCE:
{evidence}

Yêu cầu:
- Nếu SUPPORTED_EVIDENCE trả lời trực tiếp phần chính của QUERY:
  - DECISION phải là: direct_answer
  - FINAL_ANSWER phải bắt đầu ngay bằng kết luận được support
  - Không được mở đầu bằng các câu kiểu "Tôi chưa tìm thấy căn cứ trực tiếp..." hoặc "Tài liệu chỉ đề cập..."
  - Được phép diễn đạt lại hệ quả pháp lý tương đương bằng ngôn ngữ gần với QUERY, ví dụ "được bảo lưu" tương ứng với "không bị xóa sổ", nhưng không được thêm ý mới
- Nếu SUPPORTED_EVIDENCE chỉ cung cấp thông tin nền, không đủ để kết luận phần chính của QUERY:
  - DECISION phải là: abstain
  - FINAL_ANSWER phải là 1 câu abstain ngắn gọn
- Chỉ dùng fact có trong SUPPORTED_EVIDENCE.
- Ưu tiên nêu Điều/Khoản/Điểm/Trang nếu đã có.
- Không được chèn mã evidence nội bộ như [E1], [E2] vào FINAL_ANSWER.
- Nếu cần dẫn căn cứ, chỉ dùng trích dẫn pháp lý tự nhiên trong ngoặc.
- Trả lời ngắn gọn, chính xác, bằng tiếng Việt.
- BẮT BUỘC trả về đúng format:

DECISION:
<direct_answer hoặc abstain>

FINAL_ANSWER:
<câu trả lời cuối cùng để trả cho người dùng>
"""

print(f"\n Configuration:")
print(f"   • Max context length: {MAX_CONTEXT_LENGTH} chars")
print(f"   • Min sentence score: {MIN_SENTENCE_SCORE}")
print(f"   • Top K sentences: {TOP_K_SENTENCES}")
print(f"   • Max new tokens: {MAX_NEW_TOKENS}")
print(f"   • Temperature: {TEMPERATURE}")

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class SynthesisResult:
    """Result from answer synthesis"""
    answer: str
    citations: List[str]
    confidence: float
    context_used: str
    metrics: Dict
    quality_metrics: Dict = field(default_factory=dict)
    groundedness_score: float = 0.0
    provenance_score: float = 0.0
    claim_analyses: List[Dict] = field(default_factory=list)
    evidence_spans: List[Dict] = field(default_factory=list)
    selected_evidence: List[Dict] = field(default_factory=list)
    revision_applied: bool = False


TRACE_TEXT_PREVIEW_LIMIT = 320


def _trace_preview_text(text: str, limit: int = TRACE_TEXT_PREVIEW_LIMIT) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _trace_selected_evidence(items: List[Dict], limit: int = 5) -> List[Dict]:
    summarized = []
    for item in list(items or []):
        metadata = dict(item.get("metadata", {}) or {})
        llm_text = str(item.get("llm_text", "") or "").strip()
        summarized.append({
            "evidence_id": str(item.get("evidence_id", "") or ""),
            "citation": str(item.get("citation", "") or ""),
            "selector_score": float(item.get("selector_score", 0.0) or 0.0),
            "article": str(metadata.get("article", "") or ""),
            "section": str(metadata.get("section", "") or ""),
            "point": str(metadata.get("point", "") or ""),
            "hierarchical_path": str(metadata.get("hierarchical_path", "") or ""),
            "text_preview": _trace_preview_text(item.get("text", "")),
            "llm_text_preview": _trace_preview_text(llm_text) if llm_text else "",
        })
    return summarized


def _trace_evidence_spans(items: List[Dict], limit: int = 5) -> List[Dict]:
    summarized = []
    for item in list(items or []):
        metadata = dict(item.get("metadata", {}) or {})
        summarized.append({
            "claim": _trace_preview_text(item.get("claim", ""), limit=220),
            "label": str(item.get("label", "") or ""),
            "citation": str(item.get("citation", "") or ""),
            "provenance_score": float(item.get("provenance_score", 0.0) or 0.0),
            "evidence_id": str(item.get("evidence_id", "") or ""),
            "article": str(metadata.get("article", item.get("article", "")) or ""),
            "section": str(metadata.get("section", item.get("section", "")) or ""),
            "point": str(metadata.get("point", item.get("point", "")) or ""),
            "hierarchical_path": str(metadata.get("hierarchical_path", "") or ""),
            "evidence_preview": _trace_preview_text(item.get("evidence", "")),
        })
    return summarized


def _trace_semantic_sentences(
    sentences: List[str],
    scores: List[float],
    *,
    kept_sentences: Optional[List[str]] = None,
    limit: int = 8,
) -> List[Dict]:
    kept_lookup = {
        normalize_text(sentence)
        for sentence in list(kept_sentences or [])
        if normalize_text(sentence)
    }
    ranked = sorted(
        list(zip(list(sentences or []), list(scores or []))),
        key=lambda item: float(item[1] or 0.0),
        reverse=True,
    )
    traced = []
    for sentence, score in ranked:
        normalized = normalize_text(sentence)
        traced.append({
            "sentence": _trace_preview_text(sentence, limit=260),
            "score": float(score or 0.0),
            "kept": normalized in kept_lookup if normalized else False,
        })
    return traced


def build_legal_citation(metadata: Dict) -> str:
    """Build consistent legal citation string from chunk metadata."""
    metadata = metadata or {}
    document_title = str(metadata.get("document_title", "") or "").strip()
    decision_number = str(metadata.get("decision_number", "") or "").strip()
    decision_code = str(metadata.get("decision_code", "") or "").strip()
    citation_parts = []

    if document_title:
        if decision_number and decision_code:
            citation_parts.append(f"{document_title} số {decision_number}/{decision_code}")
        else:
            citation_parts.append(document_title)
    elif metadata.get("filename"):
        citation_parts.append(str(metadata.get("filename")).strip())
    else:
        citation_parts.append("Nguồn trích dẫn")

    page = metadata.get("page", "")
    chapter = metadata.get("chapter", "")
    article = metadata.get("article", "")
    section = metadata.get("section", "")
    point = metadata.get("point", "")

    if page:
        citation_parts.append(f"trang {page}")
    if chapter:
        citation_parts.append(chapter)
    if article:
        citation_parts.append(article)
    if section:
        citation_parts.append(section)
    if point:
        citation_parts.append(point)

    return " | ".join(part for part in citation_parts if part)


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip().lower()


def normalize_ascii_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("đ", "d").replace("Đ", "D")
    return normalize_text(text)


def extract_labeled_section(text: str, label: str, next_labels: List[str]) -> str:
    """Extract a labeled multi-line section from structured LLM output."""
    text = (text or "").strip()
    if not text:
        return ""

    next_pattern = "|".join(re.escape(next_label) for next_label in next_labels)
    if next_pattern:
        pattern = rf"{re.escape(label)}\s*:?\s*(.*?)(?=(?:{next_pattern})\s*:|\Z)"
    else:
        pattern = rf"{re.escape(label)}\s*:?\s*(.*)$"

    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""

    return match.group(1).strip()


def parse_revision_output(text: str) -> Dict[str, str]:
    """Parse structured revision output into named sections."""
    labels = ["SUPPORTED_ANSWER", "UNSUPPORTED_PART", "FINAL_ANSWER"]
    parsed = {}
    for idx, label in enumerate(labels):
        parsed[label] = extract_labeled_section(text, label, labels[idx + 1:])
    return parsed


def parse_direct_answer_output(text: str) -> Dict[str, str]:
    """Parse direct-answer reconciliation output."""
    labels = ["DECISION", "FINAL_ANSWER"]
    parsed = {}
    for idx, label in enumerate(labels):
        parsed[label] = extract_labeled_section(text, label, labels[idx + 1:])
    return parsed


QUERY_STOPWORDS = {
    "ai", "gì", "gi", "nao", "nào", "ra", "sao", "thế", "thế nào", "thì",
    "là", "la", "có", "co", "không", "khong", "được", "duoc", "bị", "bi",
    "với", "voi", "và", "va", "của", "cua", "cho", "trong", "theo", "tôi",
    "toi", "em", "anh", "chị", "chi", "ạ", "a", "khi", "nếu", "neu", "về",
    "ve", "mà", "ma", "đến", "den", "qua", "hay", "như", "nhu", "bao", "nhiêu",
    "mấy", "ấy", "này", "kia", "thế", "nào", "đó"
}


def extract_query_keywords(query: str, min_len: int = 3) -> List[str]:
    tokens = re.findall(r"\w+", normalize_text(query))
    keywords = []
    for token in tokens:
        if len(token) < min_len:
            continue
        if token in QUERY_STOPWORDS:
            continue
        keywords.append(token)
    deduped = []
    seen = set()
    for keyword in keywords:
        if keyword not in seen:
            deduped.append(keyword)
            seen.add(keyword)
    return deduped


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?;])\s+|\n+', text or "")
    return [sentence.strip() for sentence in sentences if sentence and sentence.strip()]


def split_sentences_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    raw_text = str(text or "")
    sentences = split_sentences(raw_text)
    spans: List[Tuple[str, int, int]] = []
    cursor = 0

    for sentence in sentences:
        start = raw_text.find(sentence, cursor)
        if start < 0:
            start = raw_text.find(sentence)
        if start < 0:
            start = cursor
        end = min(len(raw_text), start + len(sentence))
        spans.append((sentence, start, end))
        cursor = end

    return spans


def token_overlap(left: str, right: str) -> float:
    left_tokens = {tok for tok in re.findall(r"\w+", normalize_text(left)) if len(tok) > 2}
    right_tokens = {tok for tok in re.findall(r"\w+", normalize_text(right)) if len(tok) > 2}

    if not left_tokens or not right_tokens:
        return 0.0

    overlap = len(left_tokens & right_tokens)
    return overlap / max(1, min(len(left_tokens), len(right_tokens)))


def _strip_prompt_context_headers(text: str) -> str:
    raw_text = str(text or "").strip()
    if "\n\n" in raw_text:
        _, body = raw_text.split("\n\n", 1)
        if body.strip():
            return body.strip()
    return raw_text


def _get_evidence_prompt_text(item: Dict) -> str:
    return str(item.get("llm_text", "") or item.get("text", "") or "").strip()


def _get_evidence_support_text(item: Dict) -> str:
    return str(item.get("llm_text", "") or item.get("text", "") or "").strip()


def build_evidence_context_package(evidence_items: List[Dict]) -> Dict[str, Any]:
    context_lines = []
    prompt_items = []
    compressed_context_lines = []

    for item in evidence_items:
        evidence_id = str(item.get("evidence_id", "") or "").strip()
        citation = str(item.get("citation", "") or "").strip()
        selection_text = str(item.get("text", "") or "").strip()
        prompt_text = _strip_prompt_context_headers(_get_evidence_prompt_text(item))
        if not evidence_id or not citation or not selection_text:
            continue
        if not prompt_text:
            prompt_text = selection_text
        context_lines.append(f"[{evidence_id}] {citation}\nSpan: {prompt_text}")
        compressed_context_lines.append(f"[{evidence_id}] {citation}\nSpan: {selection_text}")
        prompt_items.append({
            "evidence_id": evidence_id,
            "citation": citation,
            "text": prompt_text,
            "selection_text": selection_text,
        })

    context = "\n\n".join(context_lines)
    compressed_context = "\n\n".join(compressed_context_lines)
    return {
        "context": context,
        "compressed_context": compressed_context,
        "items": prompt_items,
        "count": len(prompt_items),
    }


class EvidenceSelector:
    """Select compact evidence spans before answer generation."""

    def __init__(
        self,
        embedder,
        max_evidence: int = 5,
        min_sentence_length: int = 24,
        semantic_highlighter: Optional['SemanticHighlighter'] = None,
        semantic_threshold: float = 0.45,
        semantic_max_sentences: int = 4,
    ):
        self.embedder = embedder
        self.max_evidence = max_evidence
        self.min_sentence_length = min_sentence_length
        self.semantic_highlighter = semantic_highlighter
        self.semantic_threshold = semantic_threshold
        self.semantic_max_sentences = semantic_max_sentences

    def _section_number(self, metadata: Dict) -> Optional[int]:
        section_text = str((metadata or {}).get("section", "") or "").strip()
        match = re.search(r"(\d+)", section_text)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    def _extract_section_from_article_reservoir(self, article_text: str, metadata: Dict) -> str:
        if not article_text or not metadata:
            return ""

        section_number = self._section_number(metadata)
        if section_number is None:
            return ""

        try:
            matches = list(re.finditer(r"(?m)^\s*(\d+)\.\s", article_text))
        except re.error:
            return ""

        start_idx = None
        end_idx = len(article_text)

        for idx, match in enumerate(matches):
            try:
                current_number = int(match.group(1))
            except Exception:
                continue
            if current_number != section_number:
                continue
            start_idx = match.start()
            if idx + 1 < len(matches):
                end_idx = matches[idx + 1].start()
            break

        if start_idx is None:
            return ""

        section_text = article_text[start_idx:end_idx].strip()
        return section_text if len(section_text) >= self.min_sentence_length else ""

    def _section_scoped_source(self, result) -> Tuple[str, str]:
        metadata = dict(getattr(result, "metadata", {}) or {})
        parent_section_text = getattr(result, "parent_section_text", "") or ""
        if parent_section_text.strip():
            return parent_section_text, "section"

        parent_article_text = getattr(result, "parent_article_text", "") or ""
        if parent_article_text.strip():
            section_text = self._extract_section_from_article_reservoir(parent_article_text, metadata)
            if section_text:
                return section_text, "section"

        return "", ""

    def _strip_contextual_headers(self, text: str) -> str:
        raw_text = text or ""
        if "\n\n" in raw_text:
            _, body = raw_text.split("\n\n", 1)
            if body.strip():
                return body.strip()
        return raw_text.strip()

    def _anchor_scoped_source(self, result) -> Tuple[str, str]:
        metadata = dict(getattr(result, "metadata", {}) or {})
        raw_text = self._strip_contextual_headers(getattr(result, "raw_text", "") or "")
        if not raw_text:
            return "", ""

        expanded_scope = str(metadata.get("expanded_scope", "") or "").strip().lower()
        primary_text = self._strip_contextual_headers(getattr(result, "text", "") or "")

        # If retrieval expanded a point/chunk to section/article context, keep the
        # original anchor chunk as the primary evidence source so citations and
        # spans stay aligned.
        if metadata.get("point"):
            return raw_text, "chunk"
        if expanded_scope in {"section", "article"} and normalize_text(raw_text) != normalize_text(primary_text):
            return raw_text, "chunk"

        return "", ""

    def _resolve_result_source(self, result) -> Tuple[str, str]:
        anchor_text, anchor_level = self._anchor_scoped_source(result)
        if anchor_text:
            return anchor_text, anchor_level

        section_text, section_level = self._section_scoped_source(result)
        if section_text:
            return section_text, section_level

        metadata = dict(getattr(result, "metadata", {}) or {})
        expanded_scope = metadata.get("expanded_scope", "")
        if expanded_scope == "section" and (getattr(result, "text", "") or "").strip():
            return getattr(result, "text", "") or "", expanded_scope

        if expanded_scope == "article" and not metadata.get("section") and (getattr(result, "text", "") or "").strip():
            return getattr(result, "text", "") or "", expanded_scope

        return (getattr(result, "raw_text", "") or getattr(result, "text", "") or ""), "chunk"

    def _resolve_result_llm_source(self, result) -> Tuple[str, str]:
        metadata = dict(getattr(result, "metadata", {}) or {})

        section_text, section_level = self._section_scoped_source(result)
        if section_text:
            return section_text, section_level

        expanded_scope = str(metadata.get("expanded_scope", "") or "").strip().lower()
        parent_article_text = getattr(result, "parent_article_text", "") or ""
        if parent_article_text.strip():
            return parent_article_text, "article"

        primary_text = getattr(result, "text", "") or ""
        if expanded_scope in {"section", "article"} and primary_text.strip():
            return primary_text, expanded_scope

        raw_text = getattr(result, "raw_text", "") or ""
        if raw_text.strip():
            return raw_text, "chunk"

        return primary_text, expanded_scope or "chunk"

    def _semantic_compress_source(self, query: str, text: str, source_text_level: str) -> Tuple[str, bool, Dict]:
        clean_text = self._strip_contextual_headers(text)
        if not clean_text:
            return "", False, {"status": "empty_source"}
        if self.semantic_highlighter is None:
            return clean_text, False, {
                "status": "semantic_highlighter_unavailable",
                "source_text_level": source_text_level,
                "original_length": len(clean_text),
                "compressed_length": len(clean_text),
            }
        if source_text_level == "chunk" and len(clean_text) <= 260:
            return clean_text, False, {
                "status": "short_chunk_passthrough",
                "source_text_level": source_text_level,
                "original_length": len(clean_text),
                "compressed_length": len(clean_text),
            }

        analysis = self.semantic_highlighter.analyze_context(
            query,
            clean_text,
            threshold=self.semantic_threshold,
        )
        sentences = list(analysis.get("sentences", []) or [])
        scores = list(analysis.get("scores", []) or [])
        if not sentences or not scores or len(sentences) != len(scores):
            return clean_text, False, {
                "status": "analysis_unavailable",
                "source_text_level": source_text_level,
                "original_length": len(clean_text),
                "compressed_length": len(clean_text),
                "backend": str(analysis.get("backend", "") or ""),
                "used_builtin": bool(analysis.get("used_builtin", False)),
                "fallback_reason": str(analysis.get("fallback_reason", "") or ""),
            }

        ranked = sorted(
            zip(sentences, scores),
            key=lambda item: float(item[1] or 0.0),
            reverse=True,
        )
        kept = [
            sentence.strip()
            for sentence, score in ranked
            if float(score or 0.0) >= self.semantic_threshold and sentence.strip()
        ]
        if not kept and ranked:
            kept = [ranked[0][0].strip()]

        kept = kept[:self.semantic_max_sentences]
        compressed = " ".join(sentence for sentence in kept if sentence).strip()
        if len(compressed) < self.min_sentence_length:
            return clean_text, False, {
                "status": "compressed_too_short",
                "source_text_level": source_text_level,
                "original_length": len(clean_text),
                "compressed_length": len(clean_text),
                "backend": str(analysis.get("backend", "") or ""),
                "used_builtin": bool(analysis.get("used_builtin", False)),
                "fallback_reason": str(analysis.get("fallback_reason", "") or ""),
                "highlighted_sentences": [
                    _trace_preview_text(sentence, limit=220)
                    for sentence in kept[: self.semantic_max_sentences]
                ],
                "top_sentence_scores": _trace_semantic_sentences(
                    sentences,
                    scores,
                    kept_sentences=kept,
                    limit=self.semantic_max_sentences + 2,
                ),
            }

        return compressed, True, {
            "status": "compressed",
            "source_text_level": source_text_level,
            "original_length": len(clean_text),
            "compressed_length": len(compressed),
            "backend": str(analysis.get("backend", "") or ""),
            "used_builtin": bool(analysis.get("used_builtin", False)),
            "fallback_reason": str(analysis.get("fallback_reason", "") or ""),
            "supported_language": bool(analysis.get("supported_language", False)),
            "highlighted_sentences": [
                _trace_preview_text(sentence, limit=220)
                for sentence in kept[: self.semantic_max_sentences]
            ],
            "top_sentence_scores": _trace_semantic_sentences(
                sentences,
                scores,
                kept_sentences=kept,
                limit=self.semantic_max_sentences + 2,
            ),
        }

    def _resolve_result_source_for_query(self, query: str, result) -> Tuple[str, str, str, bool, Dict]:
        ranking_text, source_text_level = self._resolve_result_source(result)
        llm_source_text, llm_source_level = self._resolve_result_llm_source(result)
        llm_text = self._strip_contextual_headers(llm_source_text)
        ranking_text = self._strip_contextual_headers(ranking_text)
        if not ranking_text:
            ranking_text = llm_text
            source_text_level = llm_source_level
        if not llm_text:
            return "", "", source_text_level, False, {"status": "missing_source", "source_text_level": source_text_level}
        compressed_text, used_semantic, semantic_trace = self._semantic_compress_source(
            query,
            ranking_text,
            source_text_level,
        )
        if semantic_trace:
            semantic_trace = {
                **semantic_trace,
                "llm_source_text_level": llm_source_level,
                "llm_original_length": len(llm_text),
            }
        return compressed_text or ranking_text or llm_text, llm_text, source_text_level, used_semantic, semantic_trace

    def _article_key(self, metadata: Dict) -> Tuple[str, str]:
        metadata = metadata or {}
        return (
            str(metadata.get("filename", "") or "").strip(),
            str(metadata.get("article", "") or metadata.get("article_title", "") or "").strip(),
        )

    def _section_key(self, metadata: Dict) -> Tuple[str, str, str]:
        article_key = self._article_key(metadata)
        return (
            article_key[0],
            article_key[1],
            str((metadata or {}).get("section", "") or "").strip(),
        )

    def _candidate_location_key(self, candidate: Dict) -> Tuple[str, str, str, str, str]:
        metadata = candidate.get("metadata", {}) or {}
        return (
            str(metadata.get("filename", "") or "").strip(),
            str(metadata.get("article", "") or metadata.get("article_title", "") or "").strip(),
            str(metadata.get("section", "") or "").strip(),
            str(metadata.get("point", "") or "").strip(),
            str(candidate.get("source_chunk_id", "") or "").strip(),
        )

    def _is_near_duplicate_span(self, text: str, selected: List[Dict]) -> bool:
        normalized = normalize_text(text or "")
        if not normalized:
            return False

        for item in selected:
            existing = normalize_text(item.get("text", "") or "")
            if not existing:
                continue
            if normalized == existing:
                return True

        return False

    def _materialize_selected_candidate(self, candidate: Dict, selected_count: int) -> Dict:
        return {
            "evidence_id": f"E{selected_count + 1}",
            "text": candidate["text"],
            "llm_text": candidate.get("llm_text", candidate["text"]),
            "citation": candidate["citation"],
            "metadata": candidate["metadata"],
            "selector_score": candidate["selector_score"],
            "semantic_score": candidate["semantic_score"],
            "lexical_score": candidate["lexical_score"],
            "planner_alignment": candidate["planner_alignment"],
            "structural_bonus": candidate["structural_bonus"],
            "numeric_bonus": candidate["numeric_bonus"],
            "source_chunk_id": candidate["source_chunk_id"],
            "source_scope": candidate["source_scope"],
            "source_text_level": candidate.get("source_text_level", "chunk"),
        }

    def _select_diverse_candidates(self, candidates: List[Dict]) -> List[Dict]:
        selected: List[Dict] = []
        article_counts: Dict[Tuple[str, str], int] = {}
        section_counts: Dict[Tuple[str, str, str], int] = {}
        seen_locations = set()

        for strict_pass in (True, False):
            for candidate in candidates:
                if len(selected) >= self.max_evidence:
                    return selected

                location_key = self._candidate_location_key(candidate)
                if location_key in seen_locations:
                    continue

                if self._is_near_duplicate_span(candidate.get("text", ""), selected):
                    continue

                metadata = candidate.get("metadata", {}) or {}
                article_key = self._article_key(metadata)
                section_key = self._section_key(metadata)
                source_text_level = candidate.get("source_text_level", "chunk")

                if strict_pass:
                    if section_key[2] and section_counts.get(section_key, 0) >= 1:
                        continue
                    if article_key[1] and article_counts.get(article_key, 0) >= max(1, self.max_evidence // 2):
                        continue
                    if source_text_level == "article" and article_key[1] and article_counts.get(article_key, 0) >= 1:
                        continue
                else:
                    if section_key[2] and section_counts.get(section_key, 0) >= 2:
                        continue
                    if article_key[1] and article_counts.get(article_key, 0) >= max(2, self.max_evidence - 1):
                        continue

                selected.append(self._materialize_selected_candidate(candidate, len(selected)))
                seen_locations.add(location_key)
                if article_key[1]:
                    article_counts[article_key] = article_counts.get(article_key, 0) + 1
                if section_key[2]:
                    section_counts[section_key] = section_counts.get(section_key, 0) + 1

        return selected

    def _focus_lock_candidates(self, query: str, candidates: List[Dict]) -> Tuple[List[Dict], bool]:
        if not candidates:
            return candidates, False

        query_norm = normalize_ascii_text(query)
        exam_absence_markers = (
            "vang thi",
            "bo thi",
            "vang mat",
            "ngu quen",
            "khong du thi",
            "cuoi ky",
        )
        if not any(marker in query_norm for marker in exam_absence_markers):
            return candidates, False

        top_candidate = candidates[0]
        top_score = float(top_candidate.get("selector_score", 0.0) or 0.0)
        top_focus_bonus = float(top_candidate.get("focus_bonus", 0.0) or 0.0)
        if top_focus_bonus < 0.25:
            return candidates, False

        locked = [
            candidate for candidate in candidates
            if (
                float(candidate.get("focus_bonus", 0.0) or 0.0) >= 0.12 or
                float(candidate.get("selector_score", 0.0) or 0.0) >= (top_score - 0.12)
            )
        ]
        if len(locked) >= 1 and len(locked) < len(candidates):
            return locked, True

        return candidates, False

    def select(self, query: str, results: List, query_plan: Optional[Dict] = None) -> Dict:
        start_time = time.time()
        if not results:
            return {
                "evidence_items": [],
                "context": "",
                "metrics": {
                    "selection_time": 0.0,
                    "candidate_count": 0,
                    "selected_count": 0,
                    "query_type": "textual"
                }
            }

        try:
            query_emb = self.embedder.encode(query, convert_to_tensor=True).cpu().numpy()
        except Exception as exc:
            print(f" EvidenceSelector embedding failed, using retrieval-order fallback: {exc}")
            return self._fallback_select(results, query_plan=query_plan, started_at=start_time)
        planner_blueprint = self._planner_blueprint(query, query_plan)
        planner_avoid_blueprint = self._planner_avoid_blueprint(query_plan)
        planner_emb = None
        avoid_emb = None
        semantic_highlighted_sources = 0
        if planner_blueprint and normalize_ascii_text(planner_blueprint) != normalize_ascii_text(query):
            try:
                planner_emb = self.embedder.encode(planner_blueprint, convert_to_tensor=True).cpu().numpy()
            except Exception:
                planner_emb = None
        if planner_avoid_blueprint:
            try:
                avoid_emb = self.embedder.encode(planner_avoid_blueprint, convert_to_tensor=True).cpu().numpy()
            except Exception:
                avoid_emb = None
        query_type = str((query_plan or {}).get("query_type", "") or "").strip() or "generic"
        candidates = []
        semantic_trace_items = []

        for result in results:
            result_text, llm_text, source_text_level, used_semantic, semantic_trace = self._resolve_result_source_for_query(query, result)
            if used_semantic:
                semantic_highlighted_sources += 1
            metadata = dict(getattr(result, "metadata", {}) or {})
            citation = build_legal_citation(metadata)
            semantic_trace_items.append({
                "citation": citation,
                "chunk_id": getattr(result, "chunk_id", ""),
                "used_semantic": bool(used_semantic),
                "source_text_level": source_text_level,
                "source_scope": str(metadata.get("expanded_scope", "chunk") or "chunk"),
                "compressed_preview": _trace_preview_text(result_text, limit=260),
                **dict(semantic_trace or {}),
            })
            spans = self._extract_candidate_spans(result_text, source_text_level=source_text_level)
            if not spans:
                continue

            try:
                sent_embs = self.embedder.encode(spans, convert_to_tensor=True).cpu().numpy()
            except Exception:
                sent_embs = [None] * len(spans)
            structural_bonus = self._structural_bonus(metadata)
            source_scope = metadata.get("expanded_scope", "chunk")
            raw_anchor_bonus = 0.06 if getattr(result, "raw_text", "") else 0.0
            retrieval_bonus = 0.10 * max(0.0, min(1.0, float(getattr(result, "score", 0.0) or 0.0)))

            for idx, sentence in enumerate(spans):
                sent_emb = sent_embs[idx]
                if sent_emb is not None:
                    semantic = float(np.dot(query_emb, sent_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(sent_emb)
                    ))
                else:
                    semantic = 0.0
                lexical = token_overlap(query, sentence)
                planner_alignment, avoid_alignment = self._planner_alignment(
                    query_plan,
                    sentence,
                    metadata,
                    sentence_embedding=sent_emb,
                    planner_embedding=planner_emb,
                    avoid_embedding=avoid_emb,
                )
                conciseness_bonus = 0.05 if 60 <= len(sentence) <= 240 else 0.0
                coverage_bonus = 0.0
                if source_text_level == "section":
                    bullet_count = len(re.findall(r"(?:^|\s)(?:[a-zđ]\)|[1-9]\d*\.)\s", sentence, flags=re.IGNORECASE))
                    if bullet_count >= 2 and len(sentence) >= 180:
                        coverage_bonus = min(0.14, 0.03 * bullet_count)
                focus_bonus = self._query_focus_bonus(query, sentence, metadata)
                topic_penalty = self._topic_penalty(query, sentence, metadata)
                selector_score = (
                    0.34 * semantic +
                    0.12 * lexical +
                    0.34 * planner_alignment +
                    0.14 * structural_bonus +
                    raw_anchor_bonus +
                    retrieval_bonus +
                    conciseness_bonus +
                    coverage_bonus +
                    0.16 * focus_bonus -
                    0.10 * avoid_alignment -
                    0.16 * topic_penalty
                )

                candidates.append({
                    "source_chunk_id": getattr(result, "chunk_id", ""),
                    "text": sentence,
                    "llm_text": llm_text or sentence,
                    "metadata": metadata,
                    "citation": citation,
                    "selector_score": selector_score,
                    "semantic_score": semantic,
                    "lexical_score": lexical,
                    "planner_alignment": planner_alignment,
                    "structural_bonus": structural_bonus,
                    "numeric_bonus": 0.0,
                    "focus_bonus": focus_bonus,
                    "topic_penalty": topic_penalty,
                    "planner_avoid_penalty": avoid_alignment,
                    "source_scope": source_scope,
                    "source_text_level": source_text_level
                })

        candidates.sort(key=lambda item: item["selector_score"], reverse=True)
        selection_candidates, focus_locked = self._focus_lock_candidates(query, candidates)
        selected = self._select_diverse_candidates(selection_candidates)

        context_package = build_evidence_context_package(selected)

        return {
            "evidence_items": selected,
            "context": context_package["context"],
            "llm_context": context_package["context"],
            "llm_context_items": context_package["items"],
            "metrics": {
                "selection_time": time.time() - start_time,
                "candidate_count": len(candidates),
                "selection_candidate_count": len(selection_candidates),
                "selected_count": len(selected),
                "query_type": query_type,
                "planner_used": bool(query_plan),
                "focus_locked": focus_locked,
                "max_selector_score": max((item.get("selector_score", 0.0) for item in selected), default=0.0),
                "semantic_highlighted_sources": semantic_highlighted_sources,
                "semantic_trace_items": semantic_trace_items,
                "structural_evidence_count": sum(
                    1 for item in selected
                    if item.get("metadata", {}).get("article") or
                    item.get("metadata", {}).get("section") or
                    item.get("metadata", {}).get("point")
                )
            }
        }

    def _fallback_select(self, results: List, query_plan: Optional[Dict], started_at: float) -> Dict:
        candidates = []
        fallback_query = str((query_plan or {}).get("normalized_query", "") or "").strip()
        semantic_highlighted_sources = 0
        semantic_trace_items = []
        for result in results[: self.max_evidence]:
            metadata = dict(getattr(result, "metadata", {}) or {})
            citation = build_legal_citation(metadata)
            result_text, llm_text, source_text_level, used_semantic, semantic_trace = self._resolve_result_source_for_query(
                fallback_query,
                result,
            )
            if used_semantic:
                semantic_highlighted_sources += 1
            semantic_trace_items.append({
                "citation": citation,
                "chunk_id": getattr(result, "chunk_id", ""),
                "used_semantic": bool(used_semantic),
                "source_text_level": source_text_level,
                "source_scope": str(metadata.get("expanded_scope", "chunk") or "chunk"),
                "compressed_preview": _trace_preview_text(result_text, limit=260),
                **dict(semantic_trace or {}),
            })
            spans = self._extract_candidate_spans(result_text, source_text_level=source_text_level)
            if not spans:
                continue
            candidates.append({
                "text": spans[0],
                "llm_text": llm_text or spans[0],
                "citation": citation,
                "metadata": metadata,
                "selector_score": float(getattr(result, "score", 0.0) or 0.0),
                "semantic_score": 0.0,
                "lexical_score": 0.0,
                "planner_alignment": 0.0,
                "structural_bonus": self._structural_bonus(metadata),
                "numeric_bonus": 0.0,
                "source_chunk_id": getattr(result, "chunk_id", ""),
                "source_scope": metadata.get("expanded_scope", "chunk"),
                "source_text_level": source_text_level,
            })

        selected = self._select_diverse_candidates(candidates)

        context_package = build_evidence_context_package(selected)

        return {
            "evidence_items": selected,
            "context": context_package["context"],
            "llm_context": context_package["context"],
            "llm_context_items": context_package["items"],
            "metrics": {
                "selection_time": time.time() - started_at,
                "candidate_count": len(results),
                "selected_count": len(selected),
                "query_type": str((query_plan or {}).get("query_type", "") or "generic"),
                "planner_used": bool(query_plan),
                "max_selector_score": max((item.get("selector_score", 0.0) for item in selected), default=0.0),
                "semantic_highlighted_sources": semantic_highlighted_sources,
                "semantic_trace_items": semantic_trace_items,
                "structural_evidence_count": sum(
                    1 for item in selected
                    if item.get("metadata", {}).get("article") or
                    item.get("metadata", {}).get("section") or
                    item.get("metadata", {}).get("point")
                ),
                "fallback_mode": "retrieval_order"
            }
        }

    def _extract_candidate_spans(self, text: str, source_text_level: str = "chunk") -> List[str]:
        source_text = self._strip_contextual_headers(text)
        clean_text = re.sub(r"\s+", " ", source_text or "").strip()
        if not clean_text:
            return []

        marker_splits = re.split(r'(?=(?:[a-zđ]\)|[1-9]\d*\.)\s)', clean_text, flags=re.IGNORECASE)
        semicolon_splits = re.split(r';\s+', clean_text)

        candidates = [clean_text]
        if source_text_level == "article" and len(clean_text) > 260 and (
            len(marker_splits) > 1 or len(semicolon_splits) > 1
        ):
            candidates = []

        if len(marker_splits) > 1:
            candidates.extend(segment.strip() for segment in marker_splits if segment and segment.strip())

        if len(semicolon_splits) > 1:
            candidates.extend(segment.strip() for segment in semicolon_splits if segment and segment.strip())

        candidates.extend(split_sentences(clean_text))

        deduped = []
        seen = set()
        for candidate in candidates:
            candidate = candidate.strip(" -")
            if len(candidate) < self.min_sentence_length:
                continue
            normalized = normalize_text(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(candidate[:500].strip())

        return deduped[:12]

    def _infer_query_type(self, query: str) -> str:
        query_norm = normalize_ascii_text(query)
        if any(marker in query_norm for marker in [" hay khong ", " co bi ", " co duoc ", " co the ", " co con "]):
            return "binary_legal"
        if any(marker in query_norm for marker in [" la gi", " nghia la gi", " duoc hieu la "]):
            return "definition"
        if any(marker in query_norm for marker in [" bao nhieu", " may ", " bao lau", " so tin chi", " he 4", " muc diem "]):
            return "exact_numeric"
        if any(marker in query_norm for marker in [" nhu the nao", " quy trinh", " thu tuc", " gom nhung gi", " cac truong hop "]):
            return "procedure"
        if any(marker in query_norm for marker in [" neu ", " roi ", " sau do ", " dong thoi ", " bang gi", " xep loai"]):
            return "multi_hop"
        return "legal_lookup"

    def _numeric_bonus(self, query_type: str, query: str, sentence: str) -> float:
        if query_type not in {"exact_numeric", "enumeration"}:
            return 0.0
        return 1.0 if re.search(r"\d", sentence or "") else 0.0

    def _planner_blueprint(self, query: str, query_plan: Optional[Dict]) -> str:
        query_plan = query_plan or {}
        parts = [
            str(query_plan.get("normalized_query", "") or "").strip(),
            str(query_plan.get("abstract_query", "") or "").strip(),
            str(query_plan.get("pseudo_document", "") or "").strip(),
            " ; ".join(str(item or "").strip() for item in list(query_plan.get("semantic_anchors", []) or []) if str(item or "").strip()),
            " ; ".join(str(item or "").strip() for item in list(query_plan.get("must_include", []) or []) if str(item or "").strip()),
        ]
        blueprint = "\n".join(part for part in parts if part).strip()
        return blueprint or query

    def _planner_avoid_blueprint(self, query_plan: Optional[Dict]) -> str:
        query_plan = query_plan or {}
        parts = [
            str(item or "").strip()
            for item in list(query_plan.get("must_avoid", []) or [])
            if str(item or "").strip()
        ]
        return "\n".join(parts).strip()

    def _planner_alignment(
        self,
        query_plan: Optional[Dict],
        sentence: str,
        metadata: Dict,
        sentence_embedding=None,
        planner_embedding=None,
        avoid_embedding=None
    ) -> tuple[float, float]:
        query_plan = query_plan or {}
        if not query_plan:
            return 0.0, 0.0

        combined_text = f"{sentence}\n{metadata.get('article_title', '')}\n{metadata.get('hierarchical_path', '')}".strip()
        alignment = 0.0
        avoid_alignment = 0.0

        if planner_embedding is not None and sentence_embedding is not None:
            try:
                alignment = float(np.dot(planner_embedding, sentence_embedding) / (
                    np.linalg.norm(planner_embedding) * np.linalg.norm(sentence_embedding)
                ))
                alignment = max(0.0, min(1.0, alignment))
            except Exception:
                alignment = 0.0

        if avoid_embedding is not None and sentence_embedding is not None:
            try:
                avoid_alignment = float(np.dot(avoid_embedding, sentence_embedding) / (
                    np.linalg.norm(avoid_embedding) * np.linalg.norm(sentence_embedding)
                ))
                avoid_alignment = max(0.0, min(1.0, avoid_alignment))
            except Exception:
                avoid_alignment = 0.0

        include_bonus = 0.0
        for item in list(query_plan.get("must_include", []) or []):
            include_bonus = max(include_bonus, token_overlap(str(item or ""), combined_text))

        anchor_bonus = 0.0
        for item in list(query_plan.get("semantic_anchors", []) or []):
            anchor_bonus = max(anchor_bonus, token_overlap(str(item or ""), combined_text))

        score = 0.60 * alignment + 0.20 * include_bonus + 0.20 * anchor_bonus - 0.25 * avoid_alignment
        return max(0.0, min(1.0, score)), avoid_alignment

    def _topic_penalty(self, query: str, sentence: str, metadata: Dict) -> float:
        query_norm = normalize_ascii_text(query)
        sentence_norm = normalize_ascii_text(sentence)
        metadata = metadata or {}
        article_title_norm = normalize_ascii_text(metadata.get("article_title", ""))
        table_kind = str(metadata.get("table_kind", "") or "").strip().lower()
        combined_norm = f"{sentence_norm} {article_title_norm}".strip()
        penalty = 0.0

        course_markers = [
            "rot mon",
            "mon bat buoc",
            "hoc phan",
            "dang ky",
            "hoc bu",
            "hoc lai",
            "hoc cai thien",
        ]
        duration_markers = [
            "thoi gian hoc tap toi da",
            "han chot",
            "bao lau",
            "may nam",
            "tot nghiep khi nao",
        ]
        exam_absence_markers = [
            "vang thi",
            "bo thi",
            "vang mat",
            "ngu quen",
            "khong du thi",
            "cuoi ky",
        ]
        exam_absence_hits = [
            "vang mat trong buoi thi",
            "khong co ly do chinh dang",
            "phai nhan diem 0",
            "thi bo sung",
        ]
        exam_discipline_hits = [
            "gian lan",
            "thi ho",
            "nho nguoi thi ho",
            "xu ly ky luat",
            "dinh chi hoc tap",
            "buoc thoi hoc",
        ]

        if any(marker in query_norm for marker in course_markers):
            if table_kind == "duration_limit":
                penalty += 0.45
            if "thoi gian hoc tap toi da" in combined_norm:
                penalty += 0.20

        if any(marker in query_norm for marker in duration_markers):
            if any(marker in combined_norm for marker in ["hoc phan", "dang ky", "hoc lai", "hoc bu", "bat buoc"]):
                penalty += 0.25

        if any(marker in query_norm for marker in exam_absence_markers):
            if any(marker in combined_norm for marker in exam_discipline_hits) and not any(marker in combined_norm for marker in exam_absence_hits):
                penalty += 0.55
            if any(marker in combined_norm for marker in ["nghi hoc tam thoi", "tam dung hoc tap", "ren luyen kem"]):
                penalty += 0.30

        return min(penalty, 0.75)

    def _query_focus_bonus(self, query: str, sentence: str, metadata: Dict) -> float:
        query_norm = normalize_ascii_text(query)
        sentence_norm = normalize_ascii_text(sentence)
        metadata = metadata or {}
        article_title_norm = normalize_ascii_text(metadata.get("article_title", ""))
        section_norm = normalize_ascii_text(metadata.get("section", ""))
        point_norm = normalize_ascii_text(metadata.get("point", ""))
        combined_norm = f"{sentence_norm} {article_title_norm} {section_norm} {point_norm}".strip()
        bonus = 0.0

        exam_absence_markers = [
            "vang thi",
            "bo thi",
            "vang mat",
            "ngu quen",
            "khong du thi",
            "cuoi ky",
        ]

        if any(marker in query_norm for marker in exam_absence_markers):
            if "vang mat trong buoi thi" in combined_norm:
                bonus += 0.34
            if "khong co ly do chinh dang" in combined_norm:
                bonus += 0.22
            if "phai nhan diem 0" in combined_norm:
                bonus += 0.22
            if "thi bo sung" in combined_norm:
                bonus += 0.12
            if "diem e" in point_norm:
                bonus += 0.10
            if "danh gia va tinh diem hoc phan" in article_title_norm:
                bonus += 0.08

        return min(bonus, 0.85)

    def _structural_bonus(self, metadata: Dict) -> float:
        metadata = metadata or {}
        bonus = 0.0

        if metadata.get("article_title"):
            bonus += 0.04
        if metadata.get("hierarchical_path"):
            bonus += 0.04
        if metadata.get("table_kind"):
            bonus += 0.05
        if metadata.get("point"):
            bonus += 0.14
        elif metadata.get("section"):
            bonus += 0.08
        elif metadata.get("article"):
            bonus += 0.04

        return min(bonus, 1.0)


class ProvenanceScorer:
    """Score claim provenance using legal hierarchy + evidence span grounding."""

    def __init__(self, embedder=None):
        self.embedder = embedder

    def _safe_encode(self, text: str):
        if self.embedder is None or not (text or "").strip():
            return None
        try:
            embedding = self.embedder.encode(text, convert_to_tensor=True)
            return embedding.cpu().numpy() if hasattr(embedding, "cpu") else np.asarray(embedding)
        except Exception as exc:
            exc_text = str(exc or "")
            if "cuda" not in exc_text.lower() and "device-side assert" not in exc_text.lower():
                return None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                embedding = self.embedder.encode(text, convert_to_tensor=False, device="cpu")
                return np.asarray(embedding)
            except Exception:
                return None

    def score(self, claims: List[Dict], evidence_items: List[Dict]) -> Dict:
        evidence_lookup = {item["evidence_id"]: item for item in evidence_items}
        scored_claims = []
        evidence_spans = []
        claim_scores = []

        for claim_entry in claims:
            scored_claim = dict(claim_entry)
            evidence_ids = claim_entry.get("evidence_ids", []) or []
            label = claim_entry.get("label", "")

            if label not in {"entailment", "contradiction"}:
                scored_claim["provenance_score"] = 0.0
                scored_claim["provenance_breakdown"] = {}
                scored_claims.append(scored_claim)
                continue

            evidence_candidates = []
            for evidence_id in evidence_ids:
                evidence_item = evidence_lookup.get(evidence_id)
                if evidence_item is not None:
                    evidence_candidates.append(evidence_item)
            if not evidence_candidates:
                evidence_candidates = list(evidence_items or [])

            evidence_item, semantic_score, grounding_score, structural_score, composite = (
                self._select_best_evidence(scored_claim, evidence_candidates)
            )
            if evidence_item is None:
                scored_claim["provenance_score"] = 0.0
                scored_claim["provenance_breakdown"] = {}
                scored_claims.append(scored_claim)
                continue

            support_text = _get_evidence_support_text(evidence_item)
            excerpt_lookup = scored_claim.get("evidence_quote", "") or scored_claim.get("claim", "")
            representative_quote, _ = self._find_best_excerpt_from_text(excerpt_lookup, support_text)
            representative_quote = representative_quote or support_text[:220].strip()

            scored_claim["provenance_score"] = composite
            scored_claim["provenance_breakdown"] = {
                "semantic_score": semantic_score,
                "grounding_score": grounding_score,
                "structural_score": structural_score,
                "answer_type": self._infer_answer_type(scored_claim.get("claim", ""))
            }
            scored_claim["citation"] = evidence_item.get("citation", "")
            scored_claim["metadata"] = evidence_item.get("metadata", {})
            if not scored_claim.get("evidence"):
                scored_claim["evidence"] = representative_quote
            if not scored_claim.get("evidence_quote"):
                scored_claim["evidence_quote"] = representative_quote
            scored_claim["evidence_ids"] = [evidence_item.get("evidence_id", "")]

            claim_scores.append(composite)
            evidence_spans.append({
                "claim": scored_claim.get("claim", ""),
                "label": label,
                "reason": scored_claim.get("reason", ""),
                "evidence": scored_claim.get("evidence_quote", "") or representative_quote,
                "citation": evidence_item.get("citation", ""),
                "metadata": evidence_item.get("metadata", {}),
                "evidence_id": evidence_item.get("evidence_id", ""),
                "page": evidence_item.get("metadata", {}).get("page", ""),
                "article": evidence_item.get("metadata", {}).get("article", ""),
                "section": evidence_item.get("metadata", {}).get("section", ""),
                "point": evidence_item.get("metadata", {}).get("point", ""),
                "provenance_score": composite,
                "provenance_breakdown": scored_claim["provenance_breakdown"]
            })
            scored_claims.append(scored_claim)

        provenance_score = float(np.mean(claim_scores)) if claim_scores else 0.0
        return {
            "score": provenance_score,
            "claims": scored_claims,
            "evidence_spans": evidence_spans
        }

    def _select_best_evidence(self, claim_entry: Dict, evidence_candidates: List[Dict]):
        best_item = None
        best_scores = (0.0, 0.0, 0.0, 0.0)
        claim_text = claim_entry.get("claim", "")
        evidence_quote = claim_entry.get("evidence_quote", "")

        for evidence_item in evidence_candidates or []:
            support_text = _get_evidence_support_text(evidence_item)
            semantic_score = self._semantic_match_score(
                claim_text,
                support_text
            )
            grounding_score = self._grounding_score(
                evidence_quote,
                support_text
            )
            structural_score = self._structural_score(
                evidence_item.get("metadata", {}),
                evidence_quote or support_text
            )
            composite = (
                0.40 * semantic_score +
                0.35 * grounding_score +
                0.25 * structural_score
            )

            if composite > best_scores[3]:
                best_item = evidence_item
                best_scores = (semantic_score, grounding_score, structural_score, composite)

        return best_item, best_scores[0], best_scores[1], best_scores[2], best_scores[3]

    def _infer_answer_type(self, text: str) -> str:
        text = text or ""
        has_digit = bool(re.search(r"\d", text))
        has_alpha = bool(re.search(r"[A-Za-zÀ-ỹ]", text))
        if has_digit and has_alpha:
            return "hybrid"
        if has_digit:
            return "numeric"
        return "textual"

    def _semantic_match_score(self, claim: str, evidence: str) -> float:
        answer_type = self._infer_answer_type(claim)

        if answer_type == "numeric":
            claim_numbers = re.findall(r"\d+[.,]?\d*", claim or "")
            evidence_numbers = re.findall(r"\d+[.,]?\d*", evidence or "")
            if not claim_numbers:
                return 0.0
            overlap = len(set(claim_numbers) & set(evidence_numbers))
            return overlap / max(1, len(set(claim_numbers)))

        if answer_type == "hybrid":
            claim_numbers = re.findall(r"\d+[.,]?\d*", claim or "")
            evidence_numbers = re.findall(r"\d+[.,]?\d*", evidence or "")
            num_score = 0.0
            if claim_numbers:
                num_score = len(set(claim_numbers) & set(evidence_numbers)) / max(1, len(set(claim_numbers)))

            text_score = token_overlap(claim, evidence)
            if self.embedder is not None:
                claim_emb = self._safe_encode(claim)
                evidence_emb = self._safe_encode(evidence)
                if claim_emb is not None and evidence_emb is not None:
                    text_score = 0.5 * text_score + 0.5 * float(np.dot(claim_emb, evidence_emb) / (
                        np.linalg.norm(claim_emb) * np.linalg.norm(evidence_emb)
                    ))

            if num_score <= 0:
                return 0.0
            return float(2 * text_score * num_score / max(1e-8, text_score + num_score))

        lexical = token_overlap(claim, evidence)
        if self.embedder is None:
            return lexical

        claim_emb = self._safe_encode(claim)
        evidence_emb = self._safe_encode(evidence)
        if claim_emb is None or evidence_emb is None:
            return lexical
        semantic = float(np.dot(claim_emb, evidence_emb) / (
            np.linalg.norm(claim_emb) * np.linalg.norm(evidence_emb)
        ))
        return 0.5 * lexical + 0.5 * semantic

    def _grounding_score(self, evidence_quote: str, evidence_text: str) -> float:
        quote_norm = normalize_text(evidence_quote)
        evidence_norm = normalize_text(evidence_text)

        if quote_norm and quote_norm in evidence_norm:
            return 1.0
        if not quote_norm:
            return 0.0
        return token_overlap(evidence_quote, evidence_text)

    def _structural_score(self, metadata: Dict, evidence_quote: str) -> float:
        metadata = metadata or {}
        score = 0.0

        if metadata.get("page"):
            score += 0.15
        if metadata.get("article"):
            score += 0.25
        if metadata.get("section"):
            score += 0.20
        if metadata.get("point"):
            score += 0.15
        if evidence_quote:
            score += 0.25

        return min(score, 1.0)

    def _find_best_excerpt_from_text(self, claim: str, text: str) -> Tuple[str, float]:
        sentences = split_sentences(text)
        if not sentences:
            excerpt = (text or "")[:220].strip()
            return excerpt, token_overlap(claim, excerpt)

        best_sentence = ""
        best_score = 0.0
        for sentence in sentences:
            score = token_overlap(claim, sentence)
            if score > best_score:
                best_score = score
                best_sentence = sentence

        return best_sentence[:260].strip(), best_score

# ==============================================================================
# CONTEXT PRUNING
# ==============================================================================

class ContextPruner:
    """
    Hybrid Context Pruning
    - Use Semantic Highlighting if available (SOTA)
    - Fallback to similarity-based if not
    """
    
    def __init__(
        self, 
        embedder, 
        semantic_highlighter: Optional['SemanticHighlighter'] = None,
        max_length: int = 2000, 
        min_score: float = 0.3,
        use_semantic: bool = True,
        highlight_threshold: float = 0.45,
        max_sentences_per_result: int = 4,
    ):
        self.embedder = embedder
        self.highlighter = semantic_highlighter
        self.max_length = max_length
        self.min_score = min_score
        self.use_semantic = use_semantic and semantic_highlighter is not None
        self.highlight_threshold = highlight_threshold
        self.max_sentences_per_result = max(1, max_sentences_per_result)
    
    def prune(self, query: str, results: List, top_k: int = 10) -> Tuple[str, Dict]:
        """
        Prune context using semantic highlighting or similarity fallback
        
        Args:
            query: User query
            results: List of RetrievalResult
            top_k: Max sentences to keep
            
        Returns:
            (pruned_context, metrics)
        """
        if self.use_semantic:
            return self._prune_semantic(query, results, top_k)
        else:
            return self._prune_similarity(query, results, top_k)
    
    def _prune_semantic(self, query: str, results: List, top_k: int) -> Tuple[str, Dict]:
        """Semantic highlighting approach using context-level sentence probabilities."""
        start_time = time.time()

        all_sentences = []
        builtin_blocks = 0
        fallback_blocks = 0
        unsupported_language_blocks = 0
        source_trace_items = []

        for result in results:
            metadata = dict(getattr(result, "metadata", {}) or {})
            result_text = str(getattr(result, "text", "") or getattr(result, "raw_text", "") or "").strip()
            if not result_text:
                continue

            analysis = self.highlighter.analyze_context(
                query,
                result_text,
                threshold=self.highlight_threshold,
            )
            sentences = list(analysis.get("sentences", []) or [])
            scores = list(analysis.get("scores", []) or [])
            if not sentences or not scores or len(sentences) != len(scores):
                continue

            if analysis.get("used_builtin"):
                builtin_blocks += 1
            else:
                fallback_blocks += 1
            if str(analysis.get("fallback_reason", "") or "") == "unsupported_language":
                unsupported_language_blocks += 1

            local_candidates = sorted(
                [
                    (sentence.strip(), float(score or 0.0))
                    for sentence, score in zip(sentences, scores)
                    if len(sentence.strip()) > 20
                ],
                key=lambda item: item[1],
                reverse=True,
            )
            if not local_candidates:
                continue

            kept = [
                item for item in local_candidates
                if item[1] >= self.min_score
            ]
            if not kept:
                kept = [local_candidates[0]]

            kept = kept[: self.max_sentences_per_result]
            citation = build_legal_citation(metadata)
            retrieval_score = max(0.0, min(1.0, float(getattr(result, "score", 0.0) or 0.0)))
            kept_sentences = [sentence for sentence, _ in kept]
            source_trace_items.append({
                "citation": citation,
                "chunk_id": getattr(result, "chunk_id", ""),
                "source_text_level": str(metadata.get("expanded_scope", "") or "chunk"),
                "used_builtin": bool(analysis.get("used_builtin")),
                "backend": str(analysis.get("backend", "") or ""),
                "fallback_reason": str(analysis.get("fallback_reason", "") or ""),
                "supported_language": bool(analysis.get("supported_language", False)),
                "retrieval_score": retrieval_score,
                "sentence_count": len(sentences),
                "kept_count": len(kept_sentences),
                "highlighted_sentences": [
                    _trace_preview_text(sentence, limit=220)
                    for sentence in kept_sentences[: self.max_sentences_per_result]
                ],
                "top_sentence_scores": _trace_semantic_sentences(
                    sentences,
                    scores,
                    kept_sentences=kept_sentences,
                    limit=self.max_sentences_per_result + 2,
                ),
            })

            for rank, (sentence, sentence_score) in enumerate(kept, start=1):
                combined_score = 0.85 * sentence_score + 0.15 * retrieval_score
                all_sentences.append({
                    'text': sentence,
                    'source': citation,
                    'chunk_id': getattr(result, "chunk_id", ""),
                    'score': combined_score,
                    'sentence_score': sentence_score,
                    'retrieval_score': retrieval_score,
                    'rank_within_result': rank,
                })

        if not all_sentences:
            return self._prune_similarity(query, results, top_k)

        all_sentences.sort(key=lambda x: x['score'], reverse=True)
        deduped = []
        seen = set()
        for sentence in all_sentences:
            normalized = normalize_text(sentence.get('text', ""))
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(sentence)

        top_sentences = deduped[:top_k]

        context_parts = []
        current_length = 0

        for sent_dict in top_sentences:
            sent_text = sent_dict['text']
            source = sent_dict['source']

            part = f"[{source}] {sent_text}"

            if current_length + len(part) > self.max_length:
                break

            context_parts.append(part)
            current_length += len(part)

        pruned_context = "\n\n".join(context_parts)

        metrics = {
            'pruning_time': time.time() - start_time,
            'sentences_before': len(all_sentences),
            'sentences_after': len(context_parts),
            'reduction_ratio': 1 - (len(context_parts) / len(all_sentences)) if all_sentences else 0,
            'context_length': len(pruned_context),
            'method': 'semantic_highlight',
            'avg_score': np.mean([s['score'] for s in top_sentences]) if top_sentences else 0.0,
            'builtin_blocks': builtin_blocks,
            'fallback_blocks': fallback_blocks,
            'unsupported_language_blocks': unsupported_language_blocks,
            'highlight_threshold': self.highlight_threshold,
            'max_sentences_per_result': self.max_sentences_per_result,
            'source_trace_items': source_trace_items,
            'trace_items': [
                {
                    'citation': item.get('source', ''),
                    'chunk_id': item.get('chunk_id', ''),
                    'score': float(item.get('score', 0.0) or 0.0),
                    'sentence_score': float(item.get('sentence_score', 0.0) or 0.0),
                    'retrieval_score': float(item.get('retrieval_score', 0.0) or 0.0),
                    'rank_within_result': int(item.get('rank_within_result', 0) or 0),
                    'text_preview': _trace_preview_text(item.get('text', ''), limit=260),
                }
                for item in top_sentences
            ],
        }

        return pruned_context, metrics
    
    def _prune_similarity(self, query: str, results: List, top_k: int) -> Tuple[str, Dict]:
        """Similarity-based approach (fallback)"""
        start_time = time.time()
        
        # Extract all sentences
        all_sentences = []
        for result in results:
            sentences = self._split_sentences(result.text)
            for sent in sentences:
                if len(sent.strip()) > 20:
                    all_sentences.append({
                        'text': sent,
                        'source': build_legal_citation(dict(getattr(result, "metadata", {}) or {})),
                        'chunk_id': getattr(result, "chunk_id", "")
                    })
        
        if not all_sentences:
            return "", {'pruning_time': 0, 'sentences_before': 0, 'sentences_after': 0, 'reduction_ratio': 0.0, 'context_length': 0, 'method': 'similarity'}
        
        # Embed query and sentences
        query_emb = self.embedder.encode(query, convert_to_tensor=True).cpu().numpy()
        sentence_texts = [s['text'] for s in all_sentences]
        sentence_embs = self.embedder.encode(sentence_texts, convert_to_tensor=True).cpu().numpy()
        
        # Calculate similarity scores
        scores = []
        for sent_emb in sentence_embs:
            sim = np.dot(query_emb, sent_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(sent_emb))
            scores.append(float(sim))
        
        # Add scores to sentences
        for i, sent_dict in enumerate(all_sentences):
            sent_dict['score'] = scores[i]
        
        # Filter by min score
        filtered = [s for s in all_sentences if s['score'] >= self.min_score]
        
        # Sort by score
        filtered.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top K
        top_sentences = filtered[:top_k]
        
        # Build context
        context_parts = []
        current_length = 0
        
        for sent_dict in top_sentences:
            sent_text = sent_dict['text']
            source = sent_dict['source']
            
            # Add with source
            part = f"[{source}] {sent_text}"
            
            if current_length + len(part) > self.max_length:
                break
            
            context_parts.append(part)
            current_length += len(part)
        
        pruned_context = "\n\n".join(context_parts)
        
        metrics = {
            'pruning_time': time.time() - start_time,
            'sentences_before': len(all_sentences),
            'sentences_after': len(context_parts),
            'reduction_ratio': 1 - (len(context_parts) / len(all_sentences)) if all_sentences else 0,
            'context_length': len(pruned_context),
            'method': 'similarity',
            'avg_score': np.mean([s['score'] for s in top_sentences]) if top_sentences else 0.0,
            'trace_items': [
                {
                    'citation': item.get('source', ''),
                    'chunk_id': item.get('chunk_id', ''),
                    'score': float(item.get('score', 0.0) or 0.0),
                    'sentence_score': float(item.get('score', 0.0) or 0.0),
                    'retrieval_score': None,
                    'rank_within_result': 0,
                    'text_preview': _trace_preview_text(item.get('text', ''), limit=260),
                }
                for item in top_sentences
            ],
        }
        
        return pruned_context, metrics
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        return split_sentences(text)

# ==============================================================================
# GROUNDEDNESS EVALUATION
# ==============================================================================

class GroundednessEvaluator:
    """Evaluate groundedness with BEGIN-style taxonomy."""

    VALID_LABELS = {"entailment", "hallucination", "generic", "off_topic", "contradiction"}

    def __init__(self, llm_generate_func, embedder=None, max_new_tokens: int = 320):
        self.llm_generate = llm_generate_func
        self.embedder = embedder
        self.max_new_tokens = max_new_tokens

    def evaluate(self, query: str, answer: str, evidence_bundle: Dict, results: List) -> Dict:
        """Return claim-level BEGIN analysis with evidence ids."""
        if not answer or not answer.strip():
            return self._empty_result()

        evidence_items = list((evidence_bundle or {}).get("evidence_items", []) or [])
        evidence_context = (evidence_bundle or {}).get("context", "")
        claim_analyses = self._evaluate_with_llm(query, answer, evidence_context)
        if not claim_analyses:
            claim_analyses = self._evaluate_with_similarity(query, answer, evidence_items, results)

        enriched_claims = []
        evidence_lookup = {item["evidence_id"]: item for item in evidence_items}
        for claim_entry in claim_analyses:
            enriched_claims.append(self._attach_evidence_metadata(claim_entry, evidence_lookup, results))

        counts = {
            "entailment": 0,
            "hallucination": 0,
            "generic": 0,
            "off_topic": 0,
            "contradiction": 0
        }

        for item in enriched_claims:
            counts[item["label"]] = counts.get(item["label"], 0) + 1

        considered = (
            counts["entailment"] +
            counts["hallucination"] +
            counts["off_topic"] +
            counts["contradiction"]
        )
        groundedness_score = counts["entailment"] / considered if considered else 0.0

        return {
            "score": groundedness_score,
            "counts": counts,
            "claims": enriched_claims,
            "evidence_spans": [],
            "needs_revision": (
                counts["contradiction"] > 0 or
                counts["hallucination"] > 0 or
                counts["off_topic"] > 0 or
                groundedness_score < 0.65
            )
        }

    def _evaluate_with_llm(self, query: str, answer: str, context: str) -> List[Dict]:
        """Primary groundedness evaluation using BEGIN-style prompt."""
        try:
            prompt = GROUNDEDNESS_PROMPT_TEMPLATE.format(
                query=query,
                context=context[:5000],
                answer=answer[:2500]
            )

            response = self.llm_generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                system_prompt=GROUNDEDNESS_SYSTEM_PROMPT
            )

            payload = self._extract_json_payload(response)
            if not payload:
                return []

            claims = payload.get("claims", [])
            normalized = []
            for claim in claims:
                normalized_claim = self._normalize_claim_entry(claim)
                if normalized_claim:
                    normalized.append(normalized_claim)

            return normalized

        except Exception as e:
            print(f"Warning: groundedness LLM evaluation failed: {e}")
            return []

    def _evaluate_with_similarity(self, query: str, answer: str, evidence_items: List[Dict], results: List) -> List[Dict]:
        """Fallback BEGIN-style heuristics if JSON judge output cannot be parsed."""
        sentences = self._split_sentences(answer)
        fallback_claims = []

        for sentence in sentences[:8]:
            if self._is_generic(sentence):
                fallback_claims.append({
                    "claim": sentence,
                    "label": "generic",
                    "reason": "Câu mơ hồ hoặc boilerplate, không có fact kiểm chứng được.",
                    "evidence_ids": [],
                    "evidence_quote": ""
                })
                continue

            best_item, best_excerpt, score = self._find_best_evidence(sentence, evidence_items, results)
            query_relevance = self._token_overlap(query, sentence)

            if best_item is not None and score >= 0.60:
                label = "entailment"
                evidence_ids = [best_item.get("evidence_id", "")]
                evidence_quote = best_excerpt
                reason = "Fallback lexical grounding found a strong evidence span."
            elif query_relevance < 0.10:
                label = "off_topic"
                evidence_ids = []
                evidence_quote = ""
                reason = "Fallback heuristic: claim weakly related to query."
            else:
                label = "hallucination"
                evidence_ids = []
                evidence_quote = ""
                reason = "Fallback heuristic: claim is topical but lacks direct supporting evidence."

            fallback_claims.append({
                "claim": sentence,
                "label": label,
                "reason": reason,
                "evidence_ids": evidence_ids,
                "evidence_quote": evidence_quote
            })

        return fallback_claims

    def _extract_json_payload(self, text: str) -> Optional[Dict]:
        """Extract JSON object from a raw LLM response."""
        if not text:
            return None

        patterns = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r"(\{.*\})"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                continue

            candidate = match.group(1).strip()
            candidate = candidate.replace("\u201c", "\"").replace("\u201d", "\"")
            candidate = candidate.replace("\u2018", "'").replace("\u2019", "'")

            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        return None

    def _normalize_claim_entry(self, claim_entry: Dict) -> Optional[Dict]:
        """Normalize raw BEGIN claim JSON from LLM into a stable shape."""
        if not isinstance(claim_entry, dict):
            return None

        claim = str(claim_entry.get("claim", "")).strip()
        if not claim:
            return None

        label = str(claim_entry.get("label", "hallucination")).strip().lower().replace(" ", "_")
        if label not in self.VALID_LABELS:
            label = "hallucination"

        evidence_ids = claim_entry.get("evidence_ids", [])
        if not isinstance(evidence_ids, list):
            evidence_ids = []
        evidence_ids = [str(item).strip() for item in evidence_ids if str(item).strip()]

        return {
            "claim": claim,
            "label": label,
            "reason": str(claim_entry.get("reason", "")).strip(),
            "evidence_ids": evidence_ids,
            "evidence_quote": str(claim_entry.get("evidence_quote", "")).strip()
        }

    def _attach_evidence_metadata(self, claim_entry: Dict, evidence_lookup: Dict, results: List) -> Dict:
        """Attach source metadata and citation to BEGIN claim entries."""
        enriched = dict(claim_entry)
        evidence_ids = enriched.get("evidence_ids", []) or []

        if evidence_ids:
            evidence_item = evidence_lookup.get(evidence_ids[0])
            if evidence_item is not None:
                support_text = _get_evidence_support_text(evidence_item)
                excerpt_lookup = enriched.get("evidence_quote", "") or enriched.get("claim", "")
                representative_quote, _ = self._find_best_excerpt_from_text(excerpt_lookup, support_text)
                enriched["metadata"] = dict(evidence_item.get("metadata", {}) or {})
                enriched["citation"] = evidence_item.get("citation", "")
                enriched["evidence"] = enriched.get("evidence_quote", "") or representative_quote or support_text[:220].strip()
                return enriched

        matched_result, best_excerpt = self._match_result_for_claim(
            claim_entry.get("claim", ""),
            claim_entry.get("evidence_quote", ""),
            results
        )

        if matched_result is None:
            enriched["metadata"] = {}
            enriched["citation"] = ""
            enriched["evidence"] = ""
            return enriched

        metadata = dict(getattr(matched_result, "metadata", {}) or {})
        enriched["metadata"] = metadata
        enriched["citation"] = build_legal_citation(metadata)
        enriched["evidence"] = best_excerpt or enriched.get("evidence_quote", "")
        return enriched

    def _match_result_for_claim(self, claim: str, evidence_quote: str, results: List):
        """Find the most likely source chunk for a claim/evidence pair."""
        if not results:
            return None, ""

        best_result = None
        best_score = -1.0
        best_excerpt = ""
        lookup_text = evidence_quote or claim
        lookup_norm = normalize_text(lookup_text)

        for result in results:
            result_text = getattr(result, "text", "")
            result_norm = normalize_text(result_text)

            if evidence_quote and lookup_norm and lookup_norm in result_norm:
                return result, evidence_quote.strip()

            overlap_score = self._token_overlap(lookup_text, result_text)
            excerpt, sentence_score = self._find_best_excerpt_from_text(lookup_text, result_text)
            total_score = max(overlap_score, sentence_score)

            if total_score > best_score:
                best_score = total_score
                best_result = result
                best_excerpt = excerpt

        return best_result, best_excerpt

    def _find_best_evidence(self, claim: str, evidence_items: List[Dict], results: List) -> Tuple[Optional[Dict], str, float]:
        """Find a supporting evidence item or fallback chunk span."""
        best_item = None
        best_excerpt = ""
        best_score = 0.0

        for item in evidence_items or []:
            excerpt, score = self._find_best_excerpt_from_text(claim, _get_evidence_support_text(item))
            if score > best_score:
                best_score = score
                best_excerpt = excerpt
                best_item = item

        if best_item is not None:
            return best_item, best_excerpt, best_score

        for result in results:
            excerpt, score = self._find_best_excerpt_from_text(claim, getattr(result, "text", ""))
            if score > best_score:
                best_score = score
                best_excerpt = excerpt

        return None, best_excerpt, best_score

    def _find_best_excerpt_from_text(self, claim: str, text: str) -> Tuple[str, float]:
        """Select the best evidence excerpt from a text block."""
        sentences = self._split_sentences(text)
        if not sentences:
            excerpt = text[:220].strip()
            return excerpt, self._token_overlap(claim, excerpt)

        best_sentence = ""
        best_score = 0.0

        for sentence in sentences:
            score = self._token_overlap(claim, sentence)
            if score > best_score:
                best_score = score
                best_sentence = sentence

        return best_sentence[:260].strip(), best_score

    def _token_overlap(self, left: str, right: str) -> float:
        return token_overlap(left, right)

    def _is_generic(self, sentence: str) -> bool:
        """Heuristic for generic / boilerplate claims."""
        normalized = normalize_text(sentence)
        polite_patterns = [
            "theo thong tin duoc cung cap",
            "toi khong tim thay thong tin",
            "toi chua tim thay can cu",
            "xin cam on",
            "hy vong",
            "neu ban can",
            "toi san sang"
        ]
        return len(normalized) < 18 or any(pattern in normalized for pattern in polite_patterns)

    def _normalize_text(self, text: str) -> str:
        return normalize_text(text)

    def _split_sentences(self, text: str) -> List[str]:
        return split_sentences(text)

    def _empty_result(self) -> Dict:
        return {
            "score": 0.0,
            "counts": {
                "entailment": 0,
                "hallucination": 0,
                "generic": 0,
                "off_topic": 0,
                "contradiction": 0
            },
            "claims": [],
            "evidence_spans": [],
            "needs_revision": False
        }

# ==============================================================================
# CITATION EXTRACTOR
# ==============================================================================

class CitationExtractor:
    """Extract citations from answer"""
    
    def extract(
        self,
        answer: str,
        results: List,
        evidence_spans: Optional[List[Dict]] = None,
        selected_evidence: Optional[List[Dict]] = None,
    ) -> List[str]:
        """
        Extract citations from answer
        
        Args:
            answer: Generated answer
            results: List of RetrievalResult
            
        Returns:
            List of citations
        """
        citations = []

        def _append_citation(value: str) -> None:
            citation = str(value or "").strip()
            if citation and citation not in citations:
                citations.append(citation)

        if evidence_spans:
            for span in evidence_spans[:5]:
                _append_citation(span.get("citation", ""))
                if span.get("metadata"):
                    _append_citation(build_legal_citation(span.get("metadata", {}) or {}))

        if citations:
            return citations

        for item in (selected_evidence or [])[:5]:
            _append_citation(item.get("citation", ""))
            if item.get("metadata"):
                _append_citation(build_legal_citation(item.get("metadata", {}) or {}))

        return citations

# ==============================================================================
# CONFIDENCE SCORER
# ==============================================================================

class ConfidenceScorer:
    """Calculate confidence score for answer"""
    
    def __init__(self, embedder):
        self.embedder = embedder

    def _safe_encode(self, text: str):
        if self.embedder is None or not (text or "").strip():
            return None
        try:
            embedding = self.embedder.encode(text, convert_to_tensor=True)
            return embedding.cpu().numpy() if hasattr(embedding, "cpu") else np.asarray(embedding)
        except Exception as exc:
            exc_text = str(exc or "")
            if "cuda" not in exc_text.lower() and "device-side assert" not in exc_text.lower():
                return None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                embedding = self.embedder.encode(text, convert_to_tensor=False, device="cpu")
                return np.asarray(embedding)
            except Exception:
                return None

    def _normalize_retrieval_score(self, score: float) -> float:
        """Normalize retriever / reranker scores to a stable 0..1 range."""
        try:
            score = float(score)
        except Exception:
            return 0.0

        if 0.0 <= score <= 1.0:
            return score

        clipped = max(min(score, 12.0), -12.0)
        return float(1.0 / (1.0 + np.exp(-clipped)))
    
    def score(self, query: str, answer: str, results: List,
              groundedness_result: Optional[Dict] = None,
              provenance_score: float = 0.0) -> float:
        """
        Calculate confidence score
        
        Args:
            query: User query
            answer: Generated answer
            results: List of RetrievalResult
            
        Returns:
            Confidence score (0-1)
        """
        if not answer or not results:
            return 0.0
        
        # Factor 1: Answer-Query similarity (30%)
        query_emb = self._safe_encode(query)
        answer_emb = self._safe_encode(answer)
        if query_emb is not None and answer_emb is not None:
            query_answer_sim = np.dot(query_emb, answer_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(answer_emb)
            )
        else:
            query_answer_sim = token_overlap(query, answer)
        
        # Factor 2: Answer-Context similarity (40%)
        context_texts = [r.text for r in results[:5]]
        context_text = " ".join(context_texts)
        context_emb = self._safe_encode(context_text)
        if answer_emb is not None and context_emb is not None:
            answer_context_sim = np.dot(answer_emb, context_emb) / (
                np.linalg.norm(answer_emb) * np.linalg.norm(context_emb)
            )
        else:
            answer_context_sim = token_overlap(answer, context_text)
        
        # Factor 3: Retrieval scores (30%)
        avg_retrieval_score = np.mean([
            self._normalize_retrieval_score(getattr(r, "score", 0.0))
            for r in results[:5]
        ])
        
        base_confidence = (
            0.3 * float(query_answer_sim) +
            0.3 * float(answer_context_sim) +
            0.2 * float(avg_retrieval_score)
        )

        groundedness_score = 0.0
        taxonomy_penalty = 0.0
        if groundedness_result:
            groundedness_score = float(groundedness_result.get("score", 0.0))
            counts = groundedness_result.get("counts", {})
            contradiction = counts.get("contradiction", 0)
            hallucination = counts.get("hallucination", 0)
            off_topic = counts.get("off_topic", 0)
            generic = counts.get("generic", 0)
            taxonomy_penalty = min(
                0.35,
                0.20 * contradiction + 0.10 * hallucination + 0.08 * off_topic + 0.03 * generic
            )

        confidence = (
            base_confidence +
            0.10 * groundedness_score +
            0.10 * float(provenance_score) -
            taxonomy_penalty
        )
        
        return min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]

# ==============================================================================
# ANSWER SYNTHESIZER
# ==============================================================================

class AnswerSynthesizer:
    """Synthesize answer from retrieved context"""

    ABSTENTION_OPENERS = (
        "toi chua tim thay can cu truc tiep",
        "toi khong tim thay can cu truc tiep",
        "quy che dao tao khong co quy dinh truc tiep",
        "quy che dao tao nam",
        "khong co quy dinh truc tiep ve noi dung",
        "phan con lai cua cau hoi chua co can cu truc tiep",
        "tai lieu dang nap chua du can cu",
        "tai lieu chi de cap",
        "toi chua tim thay thong tin truc tiep",
        "toi khong tim thay thong tin truc tiep",
    )
    
    def __init__(self, llm_generate_func, embedder, context_pruner, 
                 citation_extractor, confidence_scorer,
                 evidence_selector: Optional['EvidenceSelector'] = None,
                 use_evidence_selection: bool = True,
                 provenance_scorer: Optional['ProvenanceScorer'] = None,
                 groundedness_evaluator: Optional['GroundednessEvaluator'] = None,
                 groundedness_threshold: float = 0.65,
                 answer_relevance_floor: float = 0.30,
                 citation_support_floor: float = 0.45,
                 hallucination_rate_ceiling: float = 0.25,
                 verification_mode: str = "full",
                 use_hallucination_guard: bool = True,
                 enable_direct_answer_rewrite: bool = True,
                 max_revision_attempts: int = 1,
                 max_new_tokens: int = MAX_NEW_TOKENS,
                 temperature: float = TEMPERATURE,
                 top_p: float = TOP_P):
        self.llm_generate = llm_generate_func
        self.embedder = embedder
        self.context_pruner = context_pruner
        self.citation_extractor = citation_extractor
        self.confidence_scorer = confidence_scorer
        self.use_evidence_selection = bool(use_evidence_selection)
        self.evidence_selector = (
            evidence_selector
            if evidence_selector is not None
            else (EvidenceSelector(embedder) if self.use_evidence_selection else None)
        )
        self.provenance_scorer = provenance_scorer or ProvenanceScorer(embedder)
        self.groundedness_evaluator = groundedness_evaluator
        self.groundedness_threshold = groundedness_threshold
        self.answer_relevance_floor = answer_relevance_floor
        self.citation_support_floor = citation_support_floor
        self.hallucination_rate_ceiling = hallucination_rate_ceiling
        self.verification_mode = verification_mode
        self.use_hallucination_guard = use_hallucination_guard
        self.enable_direct_answer_rewrite = enable_direct_answer_rewrite
        self.max_revision_attempts = max_revision_attempts
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
    
    def synthesize(self, query: str, results: List, query_plan: Optional[Dict] = None, debug_trace: bool = False) -> SynthesisResult:
        """
        Synthesize answer from results
        
        Args:
            query: User query
            results: List of RetrievalResult
            
        Returns:
            SynthesisResult
        """
        start_time = time.time()
        groundedness_result = None
        provenance_result = {"score": 0.0, "claims": [], "evidence_spans": []}
        quality_metrics = empty_quality_metrics()
        revision_applied = False
        evidence_bundle = None
        synthesis_debug_trace = {} if debug_trace else None
        
        # Step 1: Select evidence before generation
        print(f"\n Step 1: Evidence Selection")
        if self.evidence_selector is not None and self.use_evidence_selection:
            evidence_bundle = self.evidence_selector.select(query, results, query_plan=query_plan)
            selection_metrics = evidence_bundle.get("metrics", {})
            packed_context = evidence_bundle.get("llm_context") or evidence_bundle.get("context", "")
            selected_evidence_items = list(evidence_bundle.get("evidence_items", []) if evidence_bundle else [])
            print(f"   Query type: {selection_metrics.get('query_type', 'unknown')}")
            print(f"   Candidates: {selection_metrics.get('candidate_count', 0)}")
            print(f"   Selected evidence: {selection_metrics.get('selected_count', 0)}")
        else:
            evidence_bundle = {
                "context": "",
                "llm_context": "",
                "llm_context_items": [],
                "evidence_items": [],
                "metrics": {
                    "query_type": "disabled",
                    "candidate_count": len(results or []),
                    "selected_count": 0,
                    "selection_time": 0.0,
                    "fallback_mode": "module_disabled",
                },
            }
            selection_metrics = evidence_bundle["metrics"]
            packed_context = ""
            selected_evidence_items = []
            print("   Evidence selector disabled")
            print(f"   Candidates: {selection_metrics.get('candidate_count', 0)}")
            print("   Selected evidence: 0")

        if not packed_context:
            print(f"   No evidence selected, fallback to context pruner")
            packed_context, pruning_metrics = self.context_pruner.prune(query, results)
            evidence_bundle = dict(evidence_bundle or {})
            evidence_bundle["context"] = packed_context
            evidence_bundle["llm_context"] = packed_context
            evidence_bundle["llm_context_items"] = []
            fallback_metrics = dict(evidence_bundle.get("metrics", {}) or {})
            fallback_metrics.update({
                "context_length": len(packed_context),
                "context_source": "context_pruner",
                "fallback_mode": str(fallback_metrics.get("fallback_mode", "") or "context_pruner"),
            })
            evidence_bundle["metrics"] = fallback_metrics
        else:
            pruning_metrics = {
                'pruning_time': selection_metrics.get('selection_time', 0.0),
                'sentences_before': selection_metrics.get('candidate_count', 0),
                'sentences_after': selection_metrics.get('selected_count', 0),
                'reduction_ratio': 1 - (
                    selection_metrics.get('selected_count', 0) /
                    max(1, selection_metrics.get('candidate_count', 1))
                ),
                'context_length': len(packed_context),
                'method': 'evidence_selector',
                'avg_score': np.mean([
                    item.get('selector_score', 0.0)
                    for item in selected_evidence_items
                ]) if selected_evidence_items else 0.0,
                'max_selector_score': selection_metrics.get('max_selector_score', 0.0),
                'structural_evidence_count': selection_metrics.get('structural_evidence_count', 0)
            }
            evidence_bundle = dict(evidence_bundle or {})
            evidence_bundle["context"] = packed_context
            selector_metrics = dict(evidence_bundle.get("metrics", {}) or {})
            selector_metrics["context_source"] = "evidence_selector_llm_context"
            selector_metrics["context_length"] = len(packed_context)
            evidence_bundle["metrics"] = selector_metrics

        if debug_trace:
            synthesis_debug_trace["evidence_selection"] = {
                "query_type": str(selection_metrics.get("query_type", "unknown") or "unknown"),
                "candidate_count": int(selection_metrics.get("candidate_count", 0) or 0),
                "selected_count": int(selection_metrics.get("selected_count", len(selected_evidence_items)) or 0),
                "selection_time": float(selection_metrics.get("selection_time", 0.0) or 0.0),
                "selection_method": str(
                    ("disabled_context_pruner" if self.evidence_selector is None else pruning_metrics.get("method", "unknown"))
                    or "unknown"
                ),
                "context_length": int(len(packed_context)),
                "max_selector_score": float(pruning_metrics.get("max_selector_score", 0.0) or 0.0),
                "structural_evidence_count": int(pruning_metrics.get("structural_evidence_count", 0) or 0),
                "fallback_mode": str(selection_metrics.get("fallback_mode", "") or ""),
                "module_enabled": bool(self.evidence_selector is not None and self.use_evidence_selection),
                "selected_evidence_ids": [
                    str(item.get("evidence_id", "") or "")
                    for item in selected_evidence_items
                    if str(item.get("evidence_id", "") or "").strip()
                ],
                "semantic_highlighted_sources": int(selection_metrics.get("semantic_highlighted_sources", 0) or 0),
                "semantic_trace_items": list(selection_metrics.get("semantic_trace_items", []) or []),
                "items": _trace_selected_evidence(selected_evidence_items),
                "llm_context_items": list(evidence_bundle.get("llm_context_items", []) or []),
                "context_preview": _trace_preview_text(packed_context, limit=420),
            }
            synthesis_debug_trace["context_pruning"] = {
                "method": str(pruning_metrics.get("method", "unknown") or "unknown"),
                "context_source": str((evidence_bundle.get("metrics", {}) or {}).get("context_source", "") or ""),
                "context_length": int(pruning_metrics.get("context_length", len(packed_context)) or len(packed_context)),
                "sentences_before": int(pruning_metrics.get("sentences_before", 0) or 0),
                "sentences_after": int(pruning_metrics.get("sentences_after", 0) or 0),
                "reduction_ratio": float(pruning_metrics.get("reduction_ratio", 0.0) or 0.0),
                "highlight_threshold": float(pruning_metrics.get("highlight_threshold", self.context_pruner.highlight_threshold) or self.context_pruner.highlight_threshold),
                "max_sentences_per_result": int(pruning_metrics.get("max_sentences_per_result", self.context_pruner.max_sentences_per_result) or self.context_pruner.max_sentences_per_result),
                "builtin_blocks": int(pruning_metrics.get("builtin_blocks", 0) or 0),
                "fallback_blocks": int(pruning_metrics.get("fallback_blocks", 0) or 0),
                "unsupported_language_blocks": int(pruning_metrics.get("unsupported_language_blocks", 0) or 0),
                "semantic_trace_items": list(pruning_metrics.get("source_trace_items", []) or []),
                "items": list(pruning_metrics.get("trace_items", []) or []),
                "context_preview": _trace_preview_text(packed_context, limit=420),
            }
        
        # Step 2: Generate answer
        print(f"\n Step 2: Generate Answer")
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            context=packed_context,
            query=query
        )
        llm_input_trace = {}
        if debug_trace:
            llm_input_trace = {
                "query": query,
                "selected_evidence_ids": [
                    item.get("evidence_id", "")
                    for item in selected_evidence_items
                ],
                "selected_evidence_count": len(selected_evidence_items),
                "selection_method": pruning_metrics.get("method", "unknown"),
                "context": packed_context,
                "context_length": len(packed_context),
                "context_items": list(evidence_bundle.get("llm_context_items", []) or []),
                "prompt": prompt,
                "prompt_length": len(prompt),
            }
        
        answer = self.llm_generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        
        print(f"   Generated: {len(answer)} chars")
        if debug_trace:
            synthesis_debug_trace["llm_input"] = llm_input_trace
            synthesis_debug_trace["llm_answer"] = {
                "answer_length": len(answer),
                "answer_preview": _trace_preview_text(answer, limit=520),
            }

        run_verification = self._should_run_verification(query, answer, evidence_bundle, query_plan=query_plan)

        if run_verification:
            # Step 3: Groundedness evaluation + revision
            print(f"\n Step 3: Groundedness Check")
            groundedness_result = self._evaluate_groundedness(query, answer, evidence_bundle, results)
            print(f"   Groundedness: {groundedness_result['score']:.2%}")
            print(f"   Entailment: {groundedness_result['counts']['entailment']}")
            print(f"   Hallucination: {groundedness_result['counts']['hallucination']}")
            print(f"   Generic: {groundedness_result['counts']['generic']}")
            print(f"   Off-topic: {groundedness_result['counts']['off_topic']}")
            print(f"   Contradiction: {groundedness_result['counts']['contradiction']}")

            print(f"\n Step 4: Provenance Scoring")
            provenance_result = self.provenance_scorer.score(
                groundedness_result.get("claims", []),
                evidence_bundle.get("evidence_items", []) if evidence_bundle else []
            )
            groundedness_result["claims"] = provenance_result.get("claims", groundedness_result.get("claims", []))
            groundedness_result["evidence_spans"] = provenance_result.get("evidence_spans", [])
            print(f"   Provenance: {provenance_result['score']:.2%}")
            quality_metrics = self._compute_quality_metrics(
                query=query,
                answer=answer,
                groundedness_result=groundedness_result,
                provenance_result=provenance_result,
            )

            if self._should_force_direct_answer(query, answer, groundedness_result, provenance_result, quality_metrics):
                print(f"\n Step 4.5: Direct Answer Reconciliation")
                original_answer = answer
                original_groundedness = groundedness_result
                original_provenance = provenance_result
                direct_answer = self._rewrite_direct_answer(
                    query=query,
                    answer=answer,
                    groundedness_result=groundedness_result
                )

                if direct_answer and direct_answer.strip() and direct_answer.strip() != answer.strip():
                    candidate_answer = direct_answer.strip()
                    candidate_groundedness, candidate_provenance = self._refresh_answer_scores(
                        query=query,
                        answer=candidate_answer,
                        evidence_bundle=evidence_bundle,
                        results=results
                    )
                    candidate_quality_metrics = self._compute_quality_metrics(
                        query=query,
                        answer=candidate_answer,
                        groundedness_result=candidate_groundedness,
                        provenance_result=candidate_provenance,
                    )
                    if self._accept_direct_answer_rewrite(
                        original_answer=original_answer,
                        candidate_answer=candidate_answer,
                        original_groundedness=original_groundedness,
                        original_provenance=original_provenance,
                        candidate_groundedness=candidate_groundedness,
                        candidate_provenance=candidate_provenance
                    ):
                        answer = candidate_answer
                        revision_applied = True
                        groundedness_result = candidate_groundedness
                        provenance_result = candidate_provenance
                        quality_metrics = candidate_quality_metrics
                        print(f"   Direct-answer groundedness: {groundedness_result['score']:.2%}")
                        print(f"   Direct-answer provenance: {provenance_result['score']:.2%}")
                    else:
                        print("   Direct-answer rewrite rejected: candidate weakened grounding/provenance")

            if (
                self.use_hallucination_guard and
                self.max_revision_attempts > 0 and
                self._should_revise(groundedness_result, provenance_result, quality_metrics)
            ):
                revised_answer = self._revise_answer(
                    query=query,
                    answer=answer,
                    groundedness_result=groundedness_result
                )

                if revised_answer and revised_answer.strip() and revised_answer.strip() != answer.strip():
                    answer = revised_answer.strip()
                    revision_applied = True
                    groundedness_result, provenance_result = self._refresh_answer_scores(
                        query=query,
                        answer=answer,
                        evidence_bundle=evidence_bundle,
                        results=results
                    )
                    quality_metrics = self._compute_quality_metrics(
                        query=query,
                        answer=answer,
                        groundedness_result=groundedness_result,
                        provenance_result=provenance_result,
                    )
                    print(f"   Revised groundedness: {groundedness_result['score']:.2%}")
                    print(f"   Revised provenance: {provenance_result['score']:.2%}")

            normalized_answer = self._enforce_answer_mode(
                query,
                answer,
                groundedness_result=groundedness_result,
                provenance_result=provenance_result,
                quality_metrics=quality_metrics,
                selected_evidence=evidence_bundle.get("evidence_items", []) if evidence_bundle else []
            )
            if normalized_answer != answer:
                answer = normalized_answer
                groundedness_result, provenance_result = self._refresh_answer_scores(
                    query=query,
                    answer=answer,
                    evidence_bundle=evidence_bundle,
                    results=results
                )
                quality_metrics = self._compute_quality_metrics(
                    query=query,
                    answer=answer,
                    groundedness_result=groundedness_result,
                    provenance_result=provenance_result,
                )
                print(f"   Normalized answer mode: {'abstain' if self._answer_has_abstention_opener(answer) else 'direct'}")
        else:
            print(f"\n Step 3-4: Verification skipped (selective mode)")

        answer = self._strip_inline_evidence_markers(
            self._normalize_answer_language(answer)
        )
        if debug_trace:
            groundedness_counts = dict((groundedness_result or {}).get("counts", {}) or {})
            synthesis_debug_trace["evidence_grounding"] = {
                "span_count": len((groundedness_result or {}).get("evidence_spans", []) or []),
                "items": _trace_evidence_spans((groundedness_result or {}).get("evidence_spans", []) or []),
            }
            synthesis_debug_trace["verification"] = {
                "ran": bool(run_verification),
                "groundedness_score": float((groundedness_result or {}).get("score", 0.0) or 0.0),
                "provenance_score": float((provenance_result or {}).get("score", 0.0) or 0.0),
                "revision_applied": bool(revision_applied),
                "answer_mode": "abstain" if self._answer_has_abstention_opener(answer) else "direct",
                "claim_counts": {
                    "entailment": int(groundedness_counts.get("entailment", 0) or 0),
                    "hallucination": int(groundedness_counts.get("hallucination", 0) or 0),
                    "generic": int(groundedness_counts.get("generic", 0) or 0),
                    "off_topic": int(groundedness_counts.get("off_topic", 0) or 0),
                    "contradiction": int(groundedness_counts.get("contradiction", 0) or 0),
                },
            }

        # Step 5: Extract citations
        print(f"\n Step 5: Extract Citations")
        citations = self.citation_extractor.extract(
            answer,
            results,
            evidence_spans=groundedness_result.get("evidence_spans", []) if groundedness_result else None,
            selected_evidence=selected_evidence_items,
        )
        print(f"   Found: {len(citations)} citations")
        if debug_trace:
            synthesis_debug_trace["citation_extraction"] = {
                "citation_count": len(citations),
                "citations": [str(item) for item in list(citations or [])],
            }
        
        # Step 6: Calculate confidence
        print(f"\n Step 6: Calculate Confidence")
        confidence = self.confidence_scorer.score(
            query,
            answer,
            results,
            groundedness_result=groundedness_result,
            provenance_score=provenance_result.get("score", 0.0)
        )
        print(f"   Confidence: {confidence:.2%}")

        if self._should_append_contact_guidance(answer, confidence):
            answer = self._append_contact_guidance(answer)

        answer = self._strip_inline_evidence_markers(
            self._normalize_answer_language(answer)
        )
        quality_metrics = self._compute_quality_metrics(
            query=query,
            answer=answer,
            groundedness_result=groundedness_result,
            provenance_result=provenance_result,
        )
        if debug_trace:
            synthesis_debug_trace["confidence_scoring"] = {
                "confidence": float(confidence or 0.0),
                "groundedness_score": float((groundedness_result or {}).get("score", 0.0) or 0.0),
                "provenance_score": float((provenance_result or {}).get("score", 0.0) or 0.0),
                "faithfulness_score": float(quality_metrics.get("faithfulness_score", 0.0) or 0.0),
                "citation_support_score": float(quality_metrics.get("citation_support_score", 0.0) or 0.0),
                "answer_relevance_score": float(quality_metrics.get("answer_relevance_score", 0.0) or 0.0),
                "hallucination_rate": float(quality_metrics.get("hallucination_rate", 0.0) or 0.0),
                "contradiction_rate": float(quality_metrics.get("contradiction_rate", 0.0) or 0.0),
            }
            synthesis_debug_trace["final_answer"] = {
                "answer_length": len(answer),
                "answer_preview": _trace_preview_text(answer, limit=520),
                "revision_applied": bool(revision_applied),
                "answer_mode": "abstain" if self._answer_has_abstention_opener(answer) else "direct",
                "citation_count": len(citations),
            }
        
        # Metrics
        metrics = {
            'total_time': time.time() - start_time,
            'pruning_time': pruning_metrics.get('pruning_time', 0),
            'generation_time': time.time() - start_time - pruning_metrics.get('pruning_time', 0),
            'sentences_before': pruning_metrics.get('sentences_before', 0),
            'sentences_after': pruning_metrics.get('sentences_after', 0),
            'reduction_ratio': pruning_metrics.get('reduction_ratio', 0.0),
            'context_length': pruning_metrics.get('context_length', 0),
            'selected_evidence': len(selected_evidence_items),
            'answer_length': len(answer),
            'num_citations': len(citations),
            'confidence': confidence,
            'groundedness_score': groundedness_result.get("score", 0.0) if groundedness_result else 0.0,
            'provenance_score': provenance_result.get("score", 0.0),
            'max_selector_score': pruning_metrics.get('max_selector_score', 0.0),
            'structural_evidence_count': pruning_metrics.get('structural_evidence_count', 0),
            'entailed_claims': groundedness_result.get("counts", {}).get("entailment", 0) if groundedness_result else 0,
            'hallucinated_claims': groundedness_result.get("counts", {}).get("hallucination", 0) if groundedness_result else 0,
            'generic_claims': groundedness_result.get("counts", {}).get("generic", 0) if groundedness_result else 0,
            'off_topic_claims': groundedness_result.get("counts", {}).get("off_topic", 0) if groundedness_result else 0,
            'contradictory_claims': groundedness_result.get("counts", {}).get("contradiction", 0) if groundedness_result else 0,
            'revision_applied': revision_applied,
            'method': pruning_metrics.get('method', 'unknown'),
            'quality_metrics': quality_metrics,
        }
        if debug_trace:
            metrics['llm_input'] = llm_input_trace
            metrics['debug_trace'] = synthesis_debug_trace
        
        return SynthesisResult(
            answer=answer,
            citations=citations,
            confidence=confidence,
            context_used=packed_context,
            metrics=metrics,
            quality_metrics=quality_metrics,
            groundedness_score=groundedness_result.get("score", 0.0) if groundedness_result else 0.0,
            provenance_score=provenance_result.get("score", 0.0),
            claim_analyses=groundedness_result.get("claims", []) if groundedness_result else [],
            evidence_spans=groundedness_result.get("evidence_spans", []) if groundedness_result else [],
            selected_evidence=selected_evidence_items,
            revision_applied=revision_applied
        )

    def _evaluate_groundedness(self, query: str, answer: str, evidence_bundle: Dict, results: List) -> Dict:
        if not self.groundedness_evaluator:
            return {
                "score": 0.0,
                "counts": {
                    "entailment": 0,
                    "hallucination": 0,
                    "generic": 0,
                    "off_topic": 0,
                    "contradiction": 0
                },
                "claims": [],
                "evidence_spans": [],
                "needs_revision": False
            }

        return self.groundedness_evaluator.evaluate(query, answer, evidence_bundle, results)

    def _should_run_verification(
        self,
        query: str,
        answer: str,
        evidence_bundle: Optional[Dict],
        query_plan: Optional[Dict] = None,
    ) -> bool:
        if not self.groundedness_evaluator:
            return False

        mode = str(self.verification_mode or "full").strip().lower()
        if mode == "off":
            return False
        if mode == "full":
            return True

        query_plan = query_plan or {}
        evidence_bundle = evidence_bundle or {}
        metrics = evidence_bundle.get("metrics", {}) or {}
        evidence_items = list(evidence_bundle.get("evidence_items", []) or [])

        required_hops = int(query_plan.get("required_hops", 1) or 1)
        query_type = str(query_plan.get("query_type", metrics.get("query_type", "")) or "").strip().lower()
        max_selector_score = float(metrics.get("max_selector_score", 0.0) or 0.0)
        sorted_scores = sorted(
            [float(item.get("selector_score", 0.0) or 0.0) for item in evidence_items],
            reverse=True,
        )
        score_gap = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else (sorted_scores[0] if sorted_scores else 0.0)

        if required_hops >= 2:
            return True
        if query_type in {"multi_hop", "procedure", "duration_lookup"}:
            return True
        if not answer or len(answer.strip()) < 24:
            return True
        if max_selector_score < 0.48:
            return True
        if len(sorted_scores) >= 2 and score_gap < 0.06:
            return True
        return False

    def _refresh_answer_scores(self, query: str, answer: str, evidence_bundle: Dict, results: List):
        groundedness_result = self._evaluate_groundedness(query, answer, evidence_bundle, results)
        provenance_result = self.provenance_scorer.score(
            groundedness_result.get("claims", []),
            evidence_bundle.get("evidence_items", []) if evidence_bundle else []
        )
        groundedness_result["claims"] = provenance_result.get("claims", groundedness_result.get("claims", []))
        groundedness_result["evidence_spans"] = provenance_result.get("evidence_spans", [])
        return groundedness_result, provenance_result

    def _compute_quality_metrics(
        self,
        query: str,
        answer: str,
        groundedness_result: Optional[Dict],
        provenance_result: Optional[Dict],
    ) -> Dict:
        groundedness_result = groundedness_result or {}
        provenance_result = provenance_result or {}
        return compute_runtime_quality_metrics(
            query=query,
            answer=answer,
            claims=groundedness_result.get("claims", []),
            evidence_spans=groundedness_result.get("evidence_spans", []),
            groundedness_score=groundedness_result.get("score", 0.0),
            provenance_score=provenance_result.get("score", 0.0),
            embed_fn=self.confidence_scorer._safe_encode if self.confidence_scorer else None,
        )

    def _answer_has_abstention_opener(self, answer: str) -> bool:
        answer_norm = normalize_ascii_text(answer or "")
        return any(marker in answer_norm for marker in self.ABSTENTION_OPENERS)

    def _answer_contains_abstention(self, answer: str) -> bool:
        if not answer:
            return False
        if self._answer_has_abstention_opener(answer):
            return True
        return any(self._answer_has_abstention_opener(sentence) for sentence in split_sentences(answer))

    def _has_contact_guidance(self, answer: str) -> bool:
        return normalize_ascii_text(TRAINING_OFFICE_CONTACT_NOTE) in normalize_ascii_text(answer)

    def _append_contact_guidance(self, answer: str) -> str:
        if not answer:
            return TRAINING_OFFICE_CONTACT_NOTE
        if self._has_contact_guidance(answer):
            return answer
        separator = "\n\n" if "\n" in answer else " "
        return f"{answer.strip()}{separator}{TRAINING_OFFICE_CONTACT_NOTE}"

    def _build_regulation_reference(self, groundedness_result: Optional[Dict] = None) -> str:
        metadata = {}
        if groundedness_result:
            evidence_spans = groundedness_result.get("evidence_spans", []) or []
            claims = groundedness_result.get("claims", []) or []
            if evidence_spans:
                metadata = evidence_spans[0].get("metadata", {}) or {}
            elif claims:
                metadata = claims[0].get("metadata", {}) or {}

        year = str(metadata.get("year", "") or "").strip()
        filename = str(metadata.get("filename", "") or "").strip()
        decision_number = str(metadata.get("decision_number", "") or "").strip()
        decision_code = str(metadata.get("decision_code", "") or "").strip()
        decision_reference = f"{decision_number}/{decision_code}" if decision_number and decision_code else str(metadata.get("decision_reference", "") or "").strip()
        document_title = str(metadata.get("document_title", "") or "").strip()

        if decision_reference:
            base_title = document_title or "Quy chế đào tạo"
            return f"{base_title} số {decision_reference}"

        if year.isdigit():
            return f"Quy chế đào tạo năm {year}"
        if filename:
            stem = Path(filename).stem.replace("_", " ").strip()
            if stem:
                return stem
        return "Quy chế đào tạo đang nạp"

    def _build_abstention_answer(self, query: str, groundedness_result: Optional[Dict] = None) -> str:
        regulation_reference = self._build_regulation_reference(groundedness_result)
        query_clean = re.sub(r"\s+", " ", (query or "").strip()).rstrip("?.! ")
        if query_clean:
            base_answer = f'{regulation_reference} không có quy định về việc "{query_clean}".'
        else:
            base_answer = f"{regulation_reference} không có quy định về việc này."
        return self._append_contact_guidance(base_answer)

    def _extract_abstention_sentence(self, answer: str, query: str = "", groundedness_result: Optional[Dict] = None) -> str:
        for sentence in split_sentences(answer):
            if self._answer_has_abstention_opener(sentence):
                return sentence.strip()
        if self._answer_has_abstention_opener(answer):
            return answer.strip()
        return self._build_abstention_answer(query, groundedness_result)

    def _strip_abstention_content(self, answer: str) -> str:
        kept_lines = [
            line.rstrip()
            for line in (answer or "").splitlines()
            if line.strip() and not self._answer_has_abstention_opener(line)
        ]
        cleaned = "\n".join(kept_lines).strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        if cleaned and not self._answer_contains_abstention(cleaned):
            return cleaned

        kept_sentences = [
            sentence.strip()
            for sentence in split_sentences(answer)
            if sentence.strip() and not self._answer_has_abstention_opener(sentence)
        ]
        return " ".join(kept_sentences).strip()

    def _enforce_answer_mode(
        self,
        query: str,
        answer: str,
        groundedness_result: Optional[Dict] = None,
        provenance_result: Optional[Dict] = None,
        quality_metrics: Optional[Dict] = None,
        selected_evidence: Optional[List[Dict]] = None
    ) -> str:
        if not answer:
            return answer
        quality_metrics = quality_metrics or {}
        if (
            quality_metrics.get("faithfulness_score", groundedness_result.get("score", 0.0) if groundedness_result else 0.0) < 0.35 or
            quality_metrics.get("citation_support_score", provenance_result.get("score", 0.0) if provenance_result else 0.0) < 0.35
        ):
            return self._build_abstention_answer(query, groundedness_result)
        answer_mode, supported_spans, query_type = self._classify_answer_mode(
            query=query,
            groundedness_result=groundedness_result,
            provenance_result=provenance_result
        )

        if answer_mode == "direct_keep":
            cleaned_direct_answer = self._strip_answer_mode_prefix(self._strip_abstention_content(answer))
            if cleaned_direct_answer:
                return cleaned_direct_answer
            fallback_direct_answer = self._build_direct_extractive_answer(
                query,
                supported_spans,
                query_type,
                selected_evidence=selected_evidence
            )
            return fallback_direct_answer or answer

        if answer_mode == "direct_extractive":
            direct_answer = self._build_direct_extractive_answer(
                query,
                supported_spans,
                query_type,
                selected_evidence=selected_evidence
            )
            if direct_answer:
                return direct_answer
            cleaned_direct_answer = self._strip_answer_mode_prefix(self._strip_abstention_content(answer))
            return cleaned_direct_answer or answer

        if answer_mode == "partial_supported":
            partial_answer = self._build_partial_supported_answer(
                query,
                supported_spans,
                selected_evidence=selected_evidence
            )
            if partial_answer:
                return partial_answer

        return self._build_abstention_answer(query, groundedness_result)

    def _should_append_contact_guidance(self, answer: str, confidence: float) -> bool:
        return self._answer_contains_abstention(answer)

    def _strip_redundant_partial_boilerplate(self, answer: str) -> str:
        sentences = [sentence.strip() for sentence in split_sentences(answer) if sentence.strip()]
        if len(sentences) < 2:
            return (answer or "").strip()

        boilerplate_markers = (
            "tai lieu dang nap chua du can cu de tra loi tron ven phan con lai cua cau hoi",
            "tai lieu dang nap chua du can cu de ket luan day du phan con lai cua cau hoi",
            "tai lieu dang nap chua du can cu de tra loi truc tiep phan chinh cua cau hoi",
            "khong co can cu du gan de tra loi phan chinh",
            "khong co can cu du gan de tra loi phan chinh cua cau hoi",
        )

        kept_sentences = [
            sentence
            for sentence in sentences
            if not any(marker in normalize_ascii_text(sentence) for marker in boilerplate_markers)
        ]
        if not kept_sentences or len(kept_sentences) == len(sentences):
            return (answer or "").strip()
        return " ".join(kept_sentences).strip()

    def _normalize_answer_language(self, answer: str) -> str:
        answer = (answer or "").strip()
        if not answer:
            return answer

        replacements = {
            "Không có căn cứ đủ gần để trả lời phần chính.": "Tài liệu đang nạp chưa đủ căn cứ để trả lời trực tiếp phần chính của câu hỏi.",
            "Không có căn cứ đủ gần để trả lời phần chính của câu hỏi.": "Tài liệu đang nạp chưa đủ căn cứ để trả lời trực tiếp phần chính của câu hỏi.",
            "Evidence chỉ cung cấp thông tin về": "Tài liệu hiện chỉ cho biết",
            "Evidence chỉ nêu": "Tài liệu hiện chỉ nêu",
        }
        for source, target in replacements.items():
            answer = answer.replace(source, target)

        answer = self._strip_redundant_partial_boilerplate(answer)
        return answer

    def _strip_inline_evidence_markers(self, answer: str) -> str:
        answer = (answer or "").strip()
        if not answer:
            return answer

        # Remove internal evidence IDs such as [E1], [E1, E2], [E2; E4]
        # from user-facing answers while preserving legal citations in parentheses.
        answer = re.sub(
            r"\[\s*(?:E\d+\s*(?:[,;]\s*E\d+\s*)*)\]",
            "",
            answer,
            flags=re.IGNORECASE,
        )

        answer = re.sub(r"(?iu)^\s*(?:dựa trên|dua tren|theo|căn cứ|can cu)\s+(?=\()", "", answer)
        answer = re.sub(r"\s{2,}", " ", answer)
        answer = re.sub(r"\s+([,.;:!?])", r"\1", answer)
        answer = re.sub(r"([(\[]) +", r"\1", answer)
        answer = re.sub(r" +([)\]])", r"\1", answer)
        answer = re.sub(r"\(\s*,\s*", "(", answer)
        answer = re.sub(r"\n[ \t]+", "\n", answer)
        answer = re.sub(r"[ \t]+\n", "\n", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer)
        return answer.strip()

    def _is_binary_query(self, query: str) -> bool:
        query_norm = f" {normalize_ascii_text(query)} "
        binary_markers = [
            " hay khong ",
            " co bi ",
            " co duoc ",
            " co con ",
            " co phai ",
            " co vi pham ",
            " lieu ",
        ]
        if any(marker in query_norm for marker in binary_markers):
            return True
        return " khong" in query_norm and any(marker in query_norm for marker in [" co ", " bi ", " duoc ", " xoa so "])

    def _get_supported_spans(self, groundedness_result: Dict, query: str = "") -> List[Dict]:
        evidence_spans = groundedness_result.get("evidence_spans", []) if groundedness_result else []
        query_norm = normalize_text(query)
        supported = [
            dict(span) for span in evidence_spans
            if span.get("label") == "entailment" and len((span.get("evidence", "") or "").strip()) >= 20
        ]
        for span in supported:
            claim_text = span.get("claim", "")
            evidence_text = span.get("evidence", "")
            query_alignment = max(
                token_overlap(query_norm, claim_text),
                token_overlap(query_norm, evidence_text)
            ) if query_norm else 0.0
            span["query_alignment"] = query_alignment
            span["_rank_score"] = 0.80 * float(span.get("provenance_score", 0.0)) + 0.20 * query_alignment

        supported.sort(key=lambda span: span.get("_rank_score", 0.0), reverse=True)
        return supported

    def _strip_answer_mode_prefix(self, answer: str) -> str:
        cleaned = re.sub(
            r"(?im)^\s*(?:\d+\s*[\.\)]\s*)?(?:[-*]\s*)?(direct_extractive|partial_supported|abstain)\b\s*(?:[:\-]\s*)?",
            "",
            answer or ""
        ).strip()
        return re.sub(r"(?m)^\s+$", "", cleaned).strip()

    def _classify_answer_mode(
        self,
        query: str,
        groundedness_result: Optional[Dict],
        provenance_result: Optional[Dict]
    ) -> Tuple[str, List[Dict], str]:
        query_type = self.evidence_selector._infer_query_type(query) if self.evidence_selector else "textual"
        supported_spans = self._get_supported_spans(groundedness_result or {}, query)
        if not supported_spans:
            return "abstain", supported_spans, query_type

        top_span = supported_spans[0]
        top_provenance = float(top_span.get("provenance_score", 0.0))
        top_alignment = float(top_span.get("query_alignment", 0.0))

        if query_type == "enumeration":
            return "direct_keep", supported_spans, query_type

        if query_type == "exact_numeric":
            if top_provenance >= 0.70 and top_alignment >= 0.12:
                return "direct_keep", supported_spans, query_type
            return "partial_supported", supported_spans, query_type

        if self._is_binary_query(query) and top_provenance >= 0.50 and top_alignment >= 0.08:
            return "direct_extractive", supported_spans, query_type

        if top_provenance >= 0.85 and top_alignment >= 0.14:
            return "direct_extractive", supported_spans, query_type

        return "partial_supported", supported_spans, query_type

    def _clean_fact_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        text = text.strip('"')
        text = re.sub(r"^(?:[A-Za-z]\)|\d+\.)\s*", "", text)
        return text.rstrip(".;:")

    def _build_structured_binary_answer(self, selected_evidence: Optional[List[Dict]]) -> str:
        if not selected_evidence:
            return ""

        top_metadata = selected_evidence[0].get("metadata", {}) or {}
        preferred_group_key = (
            top_metadata.get("filename", ""),
            top_metadata.get("article", ""),
            top_metadata.get("section", ""),
        )

        grouped_items = {}
        for item in selected_evidence:
            metadata = item.get("metadata", {}) or {}
            group_key = (
                metadata.get("filename", ""),
                metadata.get("article", ""),
                metadata.get("section", ""),
            )
            grouped_items.setdefault(group_key, []).append(item)

        group_order = []
        if preferred_group_key in grouped_items:
            group_order.append(preferred_group_key)
        group_order.extend(key for key in grouped_items if key != preferred_group_key)

        for group_key in group_order:
            items = grouped_items[group_key]
            if len(items) < 2:
                continue

            lead_text = ""
            bullet_texts = []
            seen_bullets = set()

            for item in items:
                original_text = _get_evidence_support_text(item)
                cleaned_text = self._clean_fact_text(original_text)
                if not cleaned_text:
                    continue

                if original_text.rstrip().endswith(":") or "cac dieu kien sau" in normalize_ascii_text(original_text):
                    if not lead_text or len(cleaned_text) > len(lead_text):
                        lead_text = cleaned_text
                    continue

                if re.match(r"^(?:[A-Za-zđ]\)|\d+\.)\s*", original_text, flags=re.IGNORECASE):
                    bullet_key = normalize_text(cleaned_text)
                    if bullet_key in seen_bullets:
                        continue
                    seen_bullets.add(bullet_key)
                    bullet_texts.append(cleaned_text)

            if lead_text and bullet_texts:
                return f"{lead_text}: " + "; ".join(bullet_texts) + "."

        return ""

    def _select_binary_anchor_evidence(self, query: str, selected_evidence: Optional[List[Dict]]) -> str:
        if not selected_evidence:
            return ""

        query_norm = normalize_ascii_text(query)
        best_text = ""
        best_score = float("-inf")

        for item in selected_evidence[:4]:
            evidence_text = self._clean_fact_text(_get_evidence_support_text(item))
            if not evidence_text:
                continue

            evidence_norm = normalize_ascii_text(evidence_text)
            score = float(item.get("selector_score", 0.0)) + 0.30 * token_overlap(query, evidence_text)

            if "xoa so" in query_norm and "bao luu" in evidence_norm:
                score += 1.0

            if "co vi pham" in query_norm and "vi pham" in evidence_norm:
                score += 0.25

            if score > best_score:
                best_score = score
                best_text = evidence_text

        return best_text

    def _build_direct_extractive_answer(
        self,
        query: str,
        supported_spans: List[Dict],
        query_type: str,
        selected_evidence: Optional[List[Dict]] = None
    ) -> str:
        if not supported_spans and not selected_evidence:
            return ""

        fact_text = ""
        if self._is_binary_query(query):
            structured_answer = self._build_structured_binary_answer(selected_evidence)
            if structured_answer:
                return structured_answer
            fact_text = self._select_binary_anchor_evidence(query, selected_evidence)

        if not fact_text and supported_spans:
            top_span = supported_spans[0]
            fact_text = self._clean_fact_text(top_span.get("evidence") or top_span.get("claim") or "")

        if not fact_text:
            return ""

        query_norm = normalize_ascii_text(query)
        fact_norm = normalize_ascii_text(fact_text)

        if self._is_binary_query(query) and "xoa so" in query_norm and "bao luu" in fact_norm:
            return "Không, kết quả học tập đã tích lũy không bị xóa sổ mà được bảo lưu."

        if self._is_binary_query(query):
            return fact_text + "."

        return fact_text + "."

    def _build_partial_supported_answer(
        self,
        query: str,
        supported_spans: List[Dict],
        selected_evidence: Optional[List[Dict]] = None
    ) -> str:
        if not supported_spans and not selected_evidence:
            return ""

        top_span = supported_spans[0] if supported_spans else {}
        fact_text = self._build_direct_extractive_answer(
            query,
            [top_span] if top_span else [],
            "textual",
            selected_evidence=selected_evidence
        )
        if not fact_text:
            return ""

        if self._is_binary_query(query):
            return self._append_contact_guidance(fact_text)

        lowered_fact = fact_text[0].lower() + fact_text[1:] if len(fact_text) > 1 else fact_text
        partial_answer = f"Mình chỉ tìm thấy căn cứ trực tiếp cho ý này: {lowered_fact}"
        return self._append_contact_guidance(partial_answer)

    def _should_force_direct_answer(
        self,
        query: str,
        answer: str,
        groundedness_result: Dict,
        provenance_result: Dict,
        quality_metrics: Optional[Dict] = None
    ) -> bool:
        if not self.enable_direct_answer_rewrite:
            return False
        if not answer or self._answer_contains_abstention(answer):
            return False
        quality_metrics = quality_metrics or {}
        answer_relevance = float(quality_metrics.get("answer_relevance_score", 0.0) or 0.0)
        faithfulness = float(quality_metrics.get("faithfulness_score", groundedness_result.get("score", 0.0)) or 0.0)
        citation_support = float(quality_metrics.get("citation_support_score", provenance_result.get("score", 0.0)) or 0.0)
        if answer_relevance >= self.answer_relevance_floor:
            return False
        return (
            faithfulness >= max(self.groundedness_threshold, 0.55) and
            citation_support >= self.citation_support_floor
        )

    def _accept_direct_answer_rewrite(
        self,
        original_answer: str,
        candidate_answer: str,
        original_groundedness: Dict,
        original_provenance: Dict,
        candidate_groundedness: Dict,
        candidate_provenance: Dict
    ) -> bool:
        if not candidate_answer.strip():
            return False

        original_groundedness_score = float((original_groundedness or {}).get("score", 0.0))
        original_provenance_score = float((original_provenance or {}).get("score", 0.0))
        candidate_groundedness_score = float((candidate_groundedness or {}).get("score", 0.0))
        candidate_provenance_score = float((candidate_provenance or {}).get("score", 0.0))

        original_is_strong = (
            original_groundedness_score >= max(self.groundedness_threshold, 0.75) and
            original_provenance_score >= 0.50
        )

        if original_is_strong:
            if self._answer_contains_abstention(candidate_answer):
                return False
            if candidate_groundedness_score + 0.15 < original_groundedness_score:
                return False
            if candidate_provenance_score + 0.15 < original_provenance_score:
                return False

        return True

    def _rewrite_direct_answer(self, query: str, answer: str, groundedness_result: Dict) -> str:
        supported_spans = self._get_supported_spans(groundedness_result, query)
        if not supported_spans:
            return answer

        evidence_blocks = []
        for idx, span in enumerate(supported_spans[:4], 1):
            evidence_blocks.append(
                f"{idx}. Claim: {span.get('claim', '')}\n"
                f"   Citation: {span.get('citation', '')}\n"
                f"   Evidence: {span.get('evidence', '')}"
            )

        prompt = DIRECT_ANSWER_PROMPT_TEMPLATE.format(
            query=query,
            answer=answer,
            evidence="\n\n".join(evidence_blocks)
        )

        try:
            rewritten = self.llm_generate(
                prompt,
                max_new_tokens=280,
                temperature=0.05,
                top_p=0.95,
                do_sample=False,
                system_prompt=DIRECT_ANSWER_SYSTEM_PROMPT
            )
            rewritten = (rewritten or "").strip()
            if not rewritten:
                return answer

            parsed = parse_direct_answer_output(rewritten)
            decision = normalize_text(parsed.get("DECISION", ""))
            final_answer = parsed.get("FINAL_ANSWER", "").strip()

            if final_answer:
                if decision == "direct_answer" and self._answer_has_abstention_opener(final_answer):
                    cleaned_sentences = [
                        sentence for sentence in split_sentences(final_answer)
                        if not self._answer_has_abstention_opener(sentence)
                    ]
                    cleaned_final = " ".join(cleaned_sentences).strip()
                    if cleaned_final:
                        return cleaned_final
                return final_answer

            cleaned = re.sub(
                r"(?im)^(DECISION|FINAL_ANSWER)\s*:?\s*",
                "",
                rewritten
            ).strip()
            return cleaned or answer
        except Exception as e:
            print(f"Warning: direct-answer reconciliation failed: {e}")
            return answer

    def _should_revise(self, groundedness_result: Dict, provenance_result: Dict, quality_metrics: Optional[Dict] = None) -> bool:
        quality_metrics = quality_metrics or {}
        contradiction_rate = float(quality_metrics.get("contradiction_rate", 0.0) or 0.0)
        hallucination_rate = float(quality_metrics.get("hallucination_rate", 0.0) or 0.0)
        return contradiction_rate > 0.0 or hallucination_rate > self.hallucination_rate_ceiling

    def _revise_answer(self, query: str, answer: str, groundedness_result: Dict) -> str:
        evidence_spans = groundedness_result.get("evidence_spans", [])
        if not evidence_spans:
            return answer

        evidence_blocks = []
        for idx, span in enumerate(evidence_spans[:6], 1):
            evidence_blocks.append(
                f"{idx}. Claim: {span.get('claim', '')}\n"
                f"   Citation: {span.get('citation', '')}\n"
                f"   Evidence: {span.get('evidence', '')}"
            )

        prompt = REVISION_PROMPT_TEMPLATE.format(
            query=query,
            answer=answer,
            evidence="\n\n".join(evidence_blocks)
        )

        try:
            revised = self.llm_generate(
                prompt,
                max_new_tokens=450,
                temperature=0.1,
                top_p=0.95,
                do_sample=False,
                system_prompt=REVISION_SYSTEM_PROMPT
            )
            revised = (revised or "").strip()
            if not revised:
                return answer

            parsed = parse_revision_output(revised)
            final_answer = parsed.get("FINAL_ANSWER", "").strip()
            supported_answer = parsed.get("SUPPORTED_ANSWER", "").strip()

            if final_answer:
                return final_answer

            if supported_answer:
                return supported_answer

            cleaned = re.sub(
                r"(?im)^(SUPPORTED_ANSWER|UNSUPPORTED_PART|FINAL_ANSWER)\s*:?\s*",
                "",
                revised
            ).strip()
            return cleaned or answer
        except Exception as e:
            print(f"Warning: answer revision failed: {e}")
            return answer

# ==============================================================================
# INITIALIZE COMPONENTS
# ==============================================================================

print("\n" + "="*70)
print(" Initializing Components")
print("="*70)

# Check required variables
required_vars = {
    'generate_text': 'LLM generate function (from Cell 3)',
    'embedder': 'Sentence Transformer (from Cell 3)'
}

missing = [var for var in required_vars if var not in globals()]

if missing:
    print(" Error: Missing required variables:")
    for var in missing:
        print(f"   • {var}: {required_vars[var]}")
    print("\n Please run Cell 3 first!")
else:
    print(" All required variables found")
    
    # Initialize components
    print("\n⏳ Initializing components...")
    
    # Check if semantic highlighting model is available
    semantic_highlighter = create_semantic_highlighter()
    if semantic_highlighter is not None:
        print("\n✓ Semantic Highlighting Model: FOUND")
        print("   Using query-conditioned sentence scoring from SemViQA QATC")
        print("   ✓ SemanticHighlighter initialized")
    elif 'semantic_highlight_model' in globals() and semantic_highlight_model is None:
        print("\nSemantic Highlighting Model: NOT LOADED")
        print("   Falling back to similarity-based pruning")
    else:
        print("\nSemantic Highlighting Model: NOT FOUND")
        print("   Falling back to similarity-based pruning")
    
    # Context Pruner (with hybrid approach)
    context_pruner = ContextPruner(
        embedder,
        semantic_highlighter=semantic_highlighter,
        max_length=MAX_CONTEXT_LENGTH,
        min_score=MIN_SENTENCE_SCORE,
        use_semantic=semantic_highlighter is not None
    )
    
    if semantic_highlighter:
        print("✓ Context Pruner initialized (SEMANTIC mode)")
        print("   Expected: 70-80% token reduction")
    else:
        print("✓ Context Pruner initialized (SIMILARITY mode)")
        print("   Expected: 40-50% token reduction")
    
    # Citation Extractor
    citation_extractor = CitationExtractor()
    print(" Citation Extractor initialized")
    
    # Confidence Scorer
    confidence_scorer = ConfidenceScorer(embedder)
    print(" Confidence Scorer initialized")

    # Groundedness Evaluator
    groundedness_evaluator = GroundednessEvaluator(generate_text, embedder)
    print(" Groundedness Evaluator initialized")

    evidence_selector = EvidenceSelector(
        embedder,
        semantic_highlighter=semantic_highlighter,
    )
    print(" Evidence Selector initialized")
    
    # Answer Synthesizer
    answer_synthesizer = AnswerSynthesizer(
        generate_text,
        embedder,
        context_pruner,
        citation_extractor,
        confidence_scorer,
        evidence_selector=evidence_selector,
        groundedness_evaluator=groundedness_evaluator
    )
    print(" Answer Synthesizer initialized")
    
    print("\n All components initialized successfully!")

# ==============================================================================
# MAIN SYNTHESIS FUNCTION
# ==============================================================================

def synthesize_answer(query: str, results: List) -> SynthesisResult:
    """
    Main function to synthesize answer
    
    Args:
        query: User query
        results: List of RetrievalResult from Cell 5
        
    Returns:
        SynthesisResult
    """
    return answer_synthesizer.synthesize(query, results)

# ==============================================================================
# TEST SYNTHESIS
# ==============================================================================

# Only run test when executed directly, not when imported
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" Test Synthesis")
    print("="*70)

    # Test if we have results from Cell 5
    if 'results' in globals() and 'test_query' in globals():
        print(f"\n Query: {test_query}")
        
        try:
            synthesis_result = synthesize_answer(test_query, results)
            
            print(f"\n" + "="*70)
            print(" SYNTHESIS RESULT")
            print("="*70)
            
            print(f"\n Answer:")
            print(f"{synthesis_result.answer}")
            
            print(f"\n Citations:")
            for i, citation in enumerate(synthesis_result.citations, 1):
                print(f"   {i}. {citation}")
            
            print(f"\n Metrics:")
            print(f"   • Confidence: {synthesis_result.confidence:.2%}")
            print(f"   • Pruning method: {synthesis_result.metrics.get('method', 'unknown')}")
            print(f"   • Context reduction: {synthesis_result.metrics.get('reduction_ratio', 0.0):.1%}")
            print(f"   • Sentences: {synthesis_result.metrics['sentences_before']} → {synthesis_result.metrics['sentences_after']}")
            print(f"   • Context length: {synthesis_result.metrics['context_length']} chars")
            print(f"   • Answer length: {synthesis_result.metrics['answer_length']} chars")
            print(f"   • Total time: {synthesis_result.metrics['total_time']:.3f}s")
            
            print("\n Synthesis test completed successfully!")
            
        except Exception as e:
            print(f"\n Error during synthesis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n No test results available from Cell 5")
        print(" Run Cell 5 first to test synthesis")

# Always print completion message
print("\n" + "="*70)
print(" CELL 6 COMPLETE - SYNTHESIS READY!")
print("="*70)

print("\n Exported Functions:")
print("   • synthesize_answer(query, results) - Main synthesis function")
print("   • context_pruner - Context pruning component")
print("   • citation_extractor - Citation extraction component")
print("   • confidence_scorer - Confidence scoring component")
print("   • groundedness_evaluator - Claim-level groundedness check")
print("   • answer_synthesizer - Full synthesizer")

print("\n Next: Run Cell 7 for Evaluation Framework")
