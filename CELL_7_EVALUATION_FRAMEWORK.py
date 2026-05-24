# ==============================================================================
# @title CELL 7: EVALUATION FRAMEWORK
# ==============================================================================

"""
CELL 7 - Evaluation Framework cho RAG System

CHỨC NĂNG:
 RAGAS Metrics - Context Precision, Recall, Faithfulness
 Automatic Metrics - BLEU, ROUGE, BERTScore
 LLM-Judged Metrics - Correctness, Relevance, Completeness
 Retrieval Metrics - MRR, NDCG, Hit Rate
 Test Dataset Management - Load, save, manage test cases
 Baseline Comparison - Compare với baseline

BENEFITS:
- Measurable quality metrics
- A/B testing support
- Performance tracking
- Continuous improvement
"""

print("="*70)
print(" CELL 7: EVALUATION FRAMEWORK")
print("="*70)

import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Test dataset path
TEST_DATASET_PATH = "test_dataset.json"

# Evaluation parameters
EVAL_TOP_K = 5  # Top K for retrieval metrics

print(f"\n Configuration:")
print(f"   • Test dataset: {TEST_DATASET_PATH}")
print(f"   • Eval Top-K: {EVAL_TOP_K}")

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class TestCase:
    """Single test case"""
    id: str
    query: str
    ground_truth_answer: str
    relevant_docs: List[str]  # List of relevant document IDs
    metadata: Dict = None

@dataclass
class EvaluationResult:
    """Result from evaluation"""
    test_case_id: str
    query: str
    
    # Generated outputs
    retrieved_docs: List[str]
    generated_answer: str
    
    # Retrieval metrics
    retrieval_precision: float
    retrieval_recall: float
    retrieval_f1: float
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain
    hit_rate: float
    
    # Answer quality metrics
    answer_relevance: float
    answer_faithfulness: float
    answer_correctness: float
    
    # Automatic metrics
    bleu_score: float
    rouge_l_score: float
    
    # Other metrics
    confidence: float
    latency: float
    
    # Metadata
    timestamp: str
    metadata: Dict = None

# ==============================================================================
# TEST DATASET MANAGER
# ==============================================================================

class TestDatasetManager:
    """Manage test dataset"""
    
    def __init__(self, dataset_path: str = TEST_DATASET_PATH):
        self.dataset_path = dataset_path
        self.test_cases = []
    
    def load(self) -> List[TestCase]:
        """Load test dataset from file"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.test_cases = [
                TestCase(**case) for case in data
            ]
            
            print(f" Loaded {len(self.test_cases)} test cases")
            return self.test_cases
            
        except FileNotFoundError:
            print(f" Test dataset not found: {self.dataset_path}")
            print(" Creating sample dataset...")
            self._create_sample_dataset()
            return self.test_cases
    
    def save(self, test_cases: List[TestCase]):
        """Save test dataset to file"""
        data = [asdict(tc) for tc in test_cases]
        
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f" Saved {len(test_cases)} test cases to {self.dataset_path}")
    
    def add_test_case(self, test_case: TestCase):
        """Add a test case"""
        self.test_cases.append(test_case)
    
    def _create_sample_dataset(self):
        """Create sample test dataset"""
        sample_cases = [
            TestCase(
                id="test_001",
                query="Điều kiện tốt nghiệp đại học là gì?",
                ground_truth_answer="Sinh viên phải tích lũy đủ số tín chỉ theo chương trình, đạt điểm trung bình tích lũy từ 2.0 trở lên, hoàn thành khóa luận/đồ án tốt nghiệp, và đạt chuẩn đầu ra ngoại ngữ.",
                relevant_docs=["1410.pdf", "quy_che_dao_tao.pdf"],
                metadata={"category": "graduation", "difficulty": "easy"}
            ),
            TestCase(
                id="test_002",
                query="Chuẩn đầu ra ngoại ngữ là gì?",
                ground_truth_answer="Sinh viên phải đạt trình độ ngoại ngữ tương đương bậc 3/6 theo Khung năng lực ngoại ngữ Việt Nam hoặc tương đương IELTS 4.0, TOEIC 450.",
                relevant_docs=["chuan_dau_ra_ngoai_ngu.pdf"],
                metadata={"category": "language", "difficulty": "medium"}
            ),
            TestCase(
                id="test_003",
                query="Sinh viên có thể học lại môn bao nhiêu lần?",
                ground_truth_answer="Sinh viên được học lại tối đa 3 lần cho mỗi học phần.",
                relevant_docs=["1410.pdf"],
                metadata={"category": "retake", "difficulty": "easy"}
            ),
        ]
        
        self.test_cases = sample_cases
        self.save(sample_cases)
        print(f" Created sample dataset with {len(sample_cases)} test cases")

# ==============================================================================
# RETRIEVAL METRICS
# ==============================================================================

class RetrievalMetrics:
    """Calculate retrieval metrics"""
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """Precision@K"""
        if not retrieved or not relevant:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant))
        
        return relevant_retrieved / len(retrieved_k)
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """Recall@K"""
        if not retrieved or not relevant:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_k) & set(relevant))
        
        return relevant_retrieved / len(relevant)
    
    @staticmethod
    def f1_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """F1@K"""
        precision = RetrievalMetrics.precision_at_k(retrieved, relevant, k)
        recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def mrr(retrieved: List[str], relevant: List[str]) -> float:
        """Mean Reciprocal Rank"""
        if not retrieved or not relevant:
            return 0.0
        
        for i, doc in enumerate(retrieved, 1):
            if doc in relevant:
                return 1.0 / i
        
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """Normalized Discounted Cumulative Gain@K"""
        if not retrieved or not relevant:
            return 0.0
        
        retrieved_k = retrieved[:k]
        
        # DCG
        dcg = 0.0
        for i, doc in enumerate(retrieved_k, 1):
            if doc in relevant:
                dcg += 1.0 / np.log2(i + 1)
        
        # IDCG (ideal DCG)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def hit_rate_at_k(retrieved: List[str], relevant: List[str], k: int = 5) -> float:
        """Hit Rate@K (binary: 1 if any relevant doc in top K, else 0)"""
        if not retrieved or not relevant:
            return 0.0
        
        retrieved_k = retrieved[:k]
        return 1.0 if any(doc in relevant for doc in retrieved_k) else 0.0

# ==============================================================================
# ANSWER QUALITY METRICS
# ==============================================================================

class AnswerQualityMetrics:
    """Calculate answer quality metrics"""
    
    def __init__(self, embedder, llm_generate_func=None):
        self.embedder = embedder
        self.llm_generate = llm_generate_func
    
    def relevance(self, query: str, answer: str) -> float:
        """Answer relevance to query (semantic similarity)"""
        if not query or not answer:
            return 0.0
        
        query_emb = self.embedder.encode(query, convert_to_tensor=True).cpu().numpy()
        answer_emb = self.embedder.encode(answer, convert_to_tensor=True).cpu().numpy()
        
        sim = np.dot(query_emb, answer_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(answer_emb)
        )
        
        return float(sim)
    
    def faithfulness(self, answer: str, context: str) -> float:
        """Answer faithfulness to context (semantic similarity)"""
        if not answer or not context:
            return 0.0
        
        answer_emb = self.embedder.encode(answer, convert_to_tensor=True).cpu().numpy()
        context_emb = self.embedder.encode(context, convert_to_tensor=True).cpu().numpy()
        
        sim = np.dot(answer_emb, context_emb) / (
            np.linalg.norm(answer_emb) * np.linalg.norm(context_emb)
        )
        
        return float(sim)
    
    def correctness(self, answer: str, ground_truth: str) -> float:
        """Answer correctness vs ground truth (semantic similarity)"""
        if not answer or not ground_truth:
            return 0.0
        
        answer_emb = self.embedder.encode(answer, convert_to_tensor=True).cpu().numpy()
        gt_emb = self.embedder.encode(ground_truth, convert_to_tensor=True).cpu().numpy()
        
        sim = np.dot(answer_emb, gt_emb) / (
            np.linalg.norm(answer_emb) * np.linalg.norm(gt_emb)
        )
        
        return float(sim)

# ==============================================================================
# AUTOMATIC METRICS
# ==============================================================================

class AutomaticMetrics:
    """Calculate automatic metrics (BLEU, ROUGE)"""
    
    @staticmethod
    def bleu_score(reference: str, hypothesis: str) -> float:
        """Simple BLEU-1 score (unigram precision)"""
        if not reference or not hypothesis:
            return 0.0
        
        ref_tokens = set(reference.lower().split())
        hyp_tokens = hypothesis.lower().split()
        
        if not hyp_tokens:
            return 0.0
        
        matches = sum(1 for token in hyp_tokens if token in ref_tokens)
        
        return matches / len(hyp_tokens)
    
    @staticmethod
    def rouge_l_score(reference: str, hypothesis: str) -> float:
        """Simple ROUGE-L score (longest common subsequence)"""
        if not reference or not hypothesis:
            return 0.0
        
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        # LCS length
        lcs_length = AutomaticMetrics._lcs_length(ref_tokens, hyp_tokens)
        
        if not hyp_tokens or not ref_tokens:
            return 0.0
        
        # F1 score
        precision = lcs_length / len(hyp_tokens)
        recall = lcs_length / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]

# ==============================================================================
# EVALUATOR
# ==============================================================================

class RAGEvaluator:
    """Main evaluator for RAG system"""
    
    def __init__(self, embedder, retrieve_func, synthesize_func):
        """
        Args:
            embedder: Sentence Transformer
            retrieve_func: Function(query) -> List[RetrievalResult]
            synthesize_func: Function(query, results) -> SynthesisResult
        """
        self.embedder = embedder
        self.retrieve_func = retrieve_func
        self.synthesize_func = synthesize_func
        
        self.retrieval_metrics = RetrievalMetrics()
        self.answer_metrics = AnswerQualityMetrics(embedder)
        self.automatic_metrics = AutomaticMetrics()
    
    def evaluate_single(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate single test case"""
        start_time = time.time()
        
        # Retrieve
        results, _ = self.retrieve_func(test_case.query)
        
        # Synthesize
        synthesis_result = self.synthesize_func(test_case.query, results)
        
        # Extract retrieved doc IDs
        retrieved_docs = [r.metadata.get('filename', 'Unknown') for r in results]
        
        # Calculate retrieval metrics
        retrieval_precision = self.retrieval_metrics.precision_at_k(
            retrieved_docs, test_case.relevant_docs, k=EVAL_TOP_K
        )
        retrieval_recall = self.retrieval_metrics.recall_at_k(
            retrieved_docs, test_case.relevant_docs, k=EVAL_TOP_K
        )
        retrieval_f1 = self.retrieval_metrics.f1_at_k(
            retrieved_docs, test_case.relevant_docs, k=EVAL_TOP_K
        )
        mrr = self.retrieval_metrics.mrr(retrieved_docs, test_case.relevant_docs)
        ndcg = self.retrieval_metrics.ndcg_at_k(
            retrieved_docs, test_case.relevant_docs, k=EVAL_TOP_K
        )
        hit_rate = self.retrieval_metrics.hit_rate_at_k(
            retrieved_docs, test_case.relevant_docs, k=EVAL_TOP_K
        )
        
        # Calculate answer quality metrics
        answer_relevance = self.answer_metrics.relevance(
            test_case.query, synthesis_result.answer
        )
        answer_faithfulness = self.answer_metrics.faithfulness(
            synthesis_result.answer, synthesis_result.context_used
        )
        answer_correctness = self.answer_metrics.correctness(
            synthesis_result.answer, test_case.ground_truth_answer
        )
        
        # Calculate automatic metrics
        bleu = self.automatic_metrics.bleu_score(
            test_case.ground_truth_answer, synthesis_result.answer
        )
        rouge_l = self.automatic_metrics.rouge_l_score(
            test_case.ground_truth_answer, synthesis_result.answer
        )
        
        latency = time.time() - start_time
        
        return EvaluationResult(
            test_case_id=test_case.id,
            query=test_case.query,
            retrieved_docs=retrieved_docs[:EVAL_TOP_K],
            generated_answer=synthesis_result.answer,
            retrieval_precision=retrieval_precision,
            retrieval_recall=retrieval_recall,
            retrieval_f1=retrieval_f1,
            mrr=mrr,
            ndcg=ndcg,
            hit_rate=hit_rate,
            answer_relevance=answer_relevance,
            answer_faithfulness=answer_faithfulness,
            answer_correctness=answer_correctness,
            bleu_score=bleu,
            rouge_l_score=rouge_l,
            confidence=synthesis_result.confidence,
            latency=latency,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            metadata=test_case.metadata
        )
    
    def evaluate_dataset(self, test_cases: List[TestCase]) -> List[EvaluationResult]:
        """Evaluate entire dataset"""
        results = []
        
        print(f"\n{'='*70}")
        print(f" Evaluating {len(test_cases)} test cases")
        print(f"{'='*70}")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] Evaluating: {test_case.query[:50]}...")
            
            try:
                result = self.evaluate_single(test_case)
                results.append(result)
                
                print(f"    Precision: {result.retrieval_precision:.2%}")
                print(f"    Correctness: {result.answer_correctness:.2%}")
                print(f"    Latency: {result.latency:.2f}s")
                
            except Exception as e:
                print(f"    Error: {e}")
        
        return results
    
    def print_summary(self, results: List[EvaluationResult]):
        """Print evaluation summary"""
        if not results:
            print("No results to summarize")
            return
        
        print(f"\n{'='*70}")
        print(" EVALUATION SUMMARY")
        print(f"{'='*70}")
        
        # Aggregate metrics
        metrics = {
            'Retrieval Precision@5': np.mean([r.retrieval_precision for r in results]),
            'Retrieval Recall@5': np.mean([r.retrieval_recall for r in results]),
            'Retrieval F1@5': np.mean([r.retrieval_f1 for r in results]),
            'MRR': np.mean([r.mrr for r in results]),
            'NDCG@5': np.mean([r.ndcg for r in results]),
            'Hit Rate@5': np.mean([r.hit_rate for r in results]),
            'Answer Relevance': np.mean([r.answer_relevance for r in results]),
            'Answer Faithfulness': np.mean([r.answer_faithfulness for r in results]),
            'Answer Correctness': np.mean([r.answer_correctness for r in results]),
            'BLEU': np.mean([r.bleu_score for r in results]),
            'ROUGE-L': np.mean([r.rouge_l_score for r in results]),
            'Confidence': np.mean([r.confidence for r in results]),
            'Latency (s)': np.mean([r.latency for r in results]),
        }
        
        print(f"\n Retrieval Metrics:")
        print(f"   • Precision@5: {metrics['Retrieval Precision@5']:.2%}")
        print(f"   • Recall@5: {metrics['Retrieval Recall@5']:.2%}")
        print(f"   • F1@5: {metrics['Retrieval F1@5']:.2%}")
        print(f"   • MRR: {metrics['MRR']:.3f}")
        print(f"   • NDCG@5: {metrics['NDCG@5']:.3f}")
        print(f"   • Hit Rate@5: {metrics['Hit Rate@5']:.2%}")
        
        print(f"\n Answer Quality Metrics:")
        print(f"   • Relevance: {metrics['Answer Relevance']:.2%}")
        print(f"   • Faithfulness: {metrics['Answer Faithfulness']:.2%}")
        print(f"   • Correctness: {metrics['Answer Correctness']:.2%}")
        
        print(f"\n Automatic Metrics:")
        print(f"   • BLEU: {metrics['BLEU']:.3f}")
        print(f"   • ROUGE-L: {metrics['ROUGE-L']:.3f}")
        
        print(f"\n Performance:")
        print(f"   • Confidence: {metrics['Confidence']:.2%}")
        print(f"   • Latency: {metrics['Latency (s)']:.2f}s")
        
        print(f"\n{'='*70}")

# ==============================================================================
# INITIALIZE EVALUATOR
# ==============================================================================

print("\n" + "="*70)
print(" Initializing Evaluator")
print("="*70)

# Check required variables
required_vars = {
    'embedder': 'Sentence Transformer (from Cell 3)',
    'retrieve_enhanced': 'Retrieve function (from Cell 5)',
    'synthesize_answer': 'Synthesize function (from Cell 6)'
}

missing = [var for var in required_vars if var not in globals()]

if missing:
    print(" Error: Missing required variables:")
    for var in missing:
        print(f"   • {var}: {required_vars[var]}")
    print("\n Please run Cell 3, 5, and 6 first!")
else:
    print(" All required variables found")
    
    # Initialize evaluator
    print("\n⏳ Initializing evaluator...")
    
    evaluator = RAGEvaluator(
        embedder=embedder,
        retrieve_func=retrieve_enhanced,
        synthesize_func=synthesize_answer
    )
    print(" RAG Evaluator initialized")
    
    # Initialize test dataset manager
    dataset_manager = TestDatasetManager(TEST_DATASET_PATH)
    print(" Test Dataset Manager initialized")
    
    print("\n Evaluator ready!")

# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

print("\n" + "="*70)
print(" Example Usage")
print("="*70)

print("""
# Load test dataset
test_cases = dataset_manager.load()

# Evaluate entire dataset
eval_results = evaluator.evaluate_dataset(test_cases)

# Print summary
evaluator.print_summary(eval_results)

# Save results
with open('evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump([asdict(r) for r in eval_results], f, ensure_ascii=False, indent=2)
""")

print("\n" + "="*70)
print(" CELL 7 COMPLETE - EVALUATION READY!")
print("="*70)

print("\n Exported Objects:")
print("   • evaluator - RAG Evaluator")
print("   • dataset_manager - Test Dataset Manager")
print("   • RetrievalMetrics - Retrieval metrics calculator")
print("   • AnswerQualityMetrics - Answer quality metrics")
print("   • AutomaticMetrics - Automatic metrics (BLEU, ROUGE)")

print("\n Next Steps:")
print("   1. Create/load test dataset")
print("   2. Run evaluation")
print("   3. Analyze results")
print("   4. Iterate and improve")
