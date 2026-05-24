# ==============================================================================
# @title CELL 8: END-TO-END RAG PIPELINE
# ==============================================================================

"""
CELL 8 - Complete RAG Pipeline Integration

CHỨC NĂNG:
 End-to-End Pipeline - Từ query đến answer
 Configuration Management - Dễ dàng config
 Multiple Retrieval Modes - Vector, BM25, Hybrid, HyDE
 Performance Tracking - Monitor metrics
 Error Handling - Robust error handling
 Logging - Detailed logging

BENEFITS:
- Single function call for complete RAG
- Easy to use and configure
- Production-ready
- Monitoring and logging
"""

print("="*70)
print(" CELL 8: END-TO-END RAG PIPELINE")
print("="*70)

import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from rag_quality_metrics import empty_quality_metrics

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class RetrievalMode(Enum):
    """Retrieval modes"""
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYDE = "hyde"
    DUAL = "dual"  # Hybrid + HyDE

@dataclass
class RAGConfig:
    """RAG Pipeline Configuration"""
    # Retrieval config
    retrieval_mode: str = "dual"  # vector, bm25, hybrid, hyde, dual
    use_hybrid: bool = True       # Enable Hybrid (BM25 + Vector)
    use_hyde: bool = True         # Enable HyDE
    top_k_retrieval: int = 20
    top_k_final: int = 5
    
    # Hybrid config
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    
    # Reranking config
    use_reranking: bool = True
    
    # MMR config
    use_mmr: bool = True
    mmr_lambda: float = 0.7

    # Hierarchical retrieval config
    use_hierarchical_expansion: bool = True
    hierarchical_fallback_guard_threshold: float = 0.58

    # Evidence selection config
    use_evidence_selection: bool = True
    
    # Context pruning config
    use_context_pruning: bool = True
    use_semantic_highlighting: bool = True
    max_context_length: int = 2000
    min_sentence_score: float = 0.3
    semantic_highlight_threshold: float = 0.45
    semantic_highlight_top_sentences_per_result: int = 4
    semantic_highlight_allow_unsupported_language: bool = False

    # Groundedness / hallucination guard
    use_groundedness_check: bool = True
    debug_trace: bool = False
    groundedness_threshold: float = 0.65
    answer_relevance_floor: float = 0.30
    citation_support_floor: float = 0.45
    hallucination_rate_ceiling: float = 0.25
    benchmark_composite_pass_threshold: float = 0.55
    verification_mode: str = "full"   # full, selective, off
    groundedness_max_new_tokens: int = 900
    use_hallucination_guard: bool = True
    enable_direct_answer_rewrite: bool = True
    max_revision_attempts: int = 1
    
    # LLM config
    max_new_tokens: int = 500
    temperature: float = 0.3
    top_p: float = 0.9
    
    # Metadata filter
    metadata_filters: Dict = None
    query_plan: Dict = None
    
    def __post_init__(self):
        if self.metadata_filters is None:
            self.metadata_filters = {}
        if self.query_plan is None:
            self.query_plan = {}
        
        # Auto-set use_hybrid and use_hyde based on retrieval_mode
        if self.retrieval_mode == "dual":
            self.use_hybrid = True
            self.use_hyde = True
        elif self.retrieval_mode == "hybrid":
            self.use_hybrid = True
            self.use_hyde = False
        elif self.retrieval_mode == "hyde":
            self.use_hybrid = False
            self.use_hyde = True
        elif self.retrieval_mode == "vector" or self.retrieval_mode == "bm25":
            self.use_hybrid = False
            self.use_hyde = False

# Default config - DUAL mode (best quality)
DEFAULT_CONFIG = RAGConfig()

print(f"\n Default Configuration:")
print(f"   • Retrieval mode: {DEFAULT_CONFIG.retrieval_mode}")
print(f"   • Use Hybrid: {DEFAULT_CONFIG.use_hybrid}")
print(f"   • Use HyDE: {DEFAULT_CONFIG.use_hyde}")
print(f"   • Top-K retrieval: {DEFAULT_CONFIG.top_k_retrieval}")
print(f"   • Top-K final: {DEFAULT_CONFIG.top_k_final}")
print(f"   • Use Reranking: {DEFAULT_CONFIG.use_reranking}")
print(f"   • Use MMR: {DEFAULT_CONFIG.use_mmr}")
print(f"   • Use hierarchical expansion: {DEFAULT_CONFIG.use_hierarchical_expansion}")
print(f"   • Use evidence selection: {DEFAULT_CONFIG.use_evidence_selection}")
print(f"   • Use context pruning: {DEFAULT_CONFIG.use_context_pruning}")
print(f"   • Use semantic highlighting: {DEFAULT_CONFIG.use_semantic_highlighting}")
print(f"   • Use groundedness check: {DEFAULT_CONFIG.use_groundedness_check}")
print(f"   • Groundedness threshold: {DEFAULT_CONFIG.groundedness_threshold}")

# ==============================================================================
# PIPELINE RESULT
# ==============================================================================

@dataclass
class RAGPipelineResult:
    """Result from RAG pipeline"""
    # Input
    query: str
    config: Dict
    
    # Retrieval
    retrieved_chunks: List[Dict]
    retrieval_metrics: Dict
    
    # Synthesis
    answer: str
    citations: List[str]
    confidence: float
    context_used: str
    groundedness_score: float
    provenance_score: float
    quality_metrics: Dict
    evidence_spans: List[Dict]
    selected_evidence: List[Dict]
    claim_analyses: List[Dict]
    revision_applied: bool
    
    # Performance
    retrieval_time: float
    synthesis_time: float
    total_time: float
    
    # Metadata
    timestamp: str
    success: bool
    error_message: Optional[str] = None

# ==============================================================================
# RAG PIPELINE
# ==============================================================================

class RAGPipeline:
    """Complete RAG Pipeline"""
    
    def __init__(self, 
                 embedder,
                 faiss_index,
                 bm25_index,
                 chunks,
                 llm_generate_func,
                 retrieve_func,
                 config: RAGConfig = None):
        """
        Initialize RAG Pipeline
        
        Args:
            embedder: Sentence Transformer
            faiss_index: FAISS index
            bm25_index: BM25 index
            chunks: List of chunks
            llm_generate_func: LLM generate function
            retrieve_func: retrieve_enhanced from Cell 5
            config: RAG configuration
        """
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.chunks = chunks
        self.llm_generate = llm_generate_func
        self.retrieve_func = retrieve_func
        self.config = config or DEFAULT_CONFIG
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize pipeline components - Simplified for Cell 5 integration"""
        from CELL_6_LLM_SYNTHESIS_WITH_PRUNING import (
            ContextPruner, CitationExtractor, ConfidenceScorer,
            GroundednessEvaluator, AnswerSynthesizer,
            EvidenceSelector, ProvenanceScorer, create_semantic_highlighter
        )

        semantic_highlighter = None
        if self.config.use_semantic_highlighting:
            semantic_highlight_model = (
                globals().get("semantic_highlight_model")
                or globals().get("mmr_qatc_model")
            )
            if semantic_highlight_model is None:
                try:
                    import __main__
                    semantic_highlight_model = (
                        getattr(__main__, "semantic_highlight_model", None)
                        or getattr(__main__, "mmr_qatc_model", None)
                    )
                except Exception:
                    semantic_highlight_model = None
            semantic_highlighter = create_semantic_highlighter(
                model=semantic_highlight_model,
                allow_unsupported_language=self.config.semantic_highlight_allow_unsupported_language,
                default_threshold=self.config.semantic_highlight_threshold,
            )
        
        # Synthesizer components (Cell 6)
        self.context_pruner = ContextPruner(
            self.embedder,
            semantic_highlighter=semantic_highlighter,
            max_length=self.config.max_context_length,
            min_score=self.config.min_sentence_score,
            use_semantic=self.config.use_semantic_highlighting and semantic_highlighter is not None,
            highlight_threshold=self.config.semantic_highlight_threshold,
            max_sentences_per_result=self.config.semantic_highlight_top_sentences_per_result,
        )
        self.citation_extractor = CitationExtractor()
        self.confidence_scorer = ConfidenceScorer(self.embedder)
        self.evidence_selector = (
            EvidenceSelector(
                self.embedder,
                semantic_highlighter=semantic_highlighter,
                semantic_threshold=self.config.semantic_highlight_threshold,
                semantic_max_sentences=self.config.semantic_highlight_top_sentences_per_result,
            )
            if self.config.use_evidence_selection else None
        )
        self.provenance_scorer = ProvenanceScorer(self.embedder)
        self.groundedness_evaluator = GroundednessEvaluator(
            self.llm_generate,
            self.embedder,
            max_new_tokens=self.config.groundedness_max_new_tokens
        )
        self.answer_synthesizer = AnswerSynthesizer(
            self.llm_generate,
            self.embedder,
            self.context_pruner,
            self.citation_extractor,
            self.confidence_scorer,
            evidence_selector=self.evidence_selector,
            use_evidence_selection=self.config.use_evidence_selection,
            provenance_scorer=self.provenance_scorer,
            groundedness_evaluator=self.groundedness_evaluator if self.config.use_groundedness_check else None,
            groundedness_threshold=self.config.groundedness_threshold,
            answer_relevance_floor=self.config.answer_relevance_floor,
            citation_support_floor=self.config.citation_support_floor,
            hallucination_rate_ceiling=self.config.hallucination_rate_ceiling,
            verification_mode=self.config.verification_mode,
            use_hallucination_guard=self.config.use_hallucination_guard,
            enable_direct_answer_rewrite=self.config.enable_direct_answer_rewrite,
            max_revision_attempts=self.config.max_revision_attempts,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        
        # Note: Retrieval components are handled by Cell 5's retrieve_enhanced()
    
    def retrieve(self, query: str) -> Tuple[List, Dict]:
        """
        Retrieve relevant chunks using Cell 5's retrieve_enhanced (passed at init).
        Uses the same retrieve_enhanced from exec() context so hybrid_retriever is initialized.
        """
        results, metrics = self.retrieve_func(
            query=query,
            retrieval_mode=self.config.retrieval_mode,
            use_hybrid=self.config.use_hybrid,
            use_hyde=self.config.use_hyde,
            use_rerank=self.config.use_reranking,
            use_mmr=self.config.use_mmr,
            use_hierarchical_expansion=self.config.use_hierarchical_expansion,
            query_plan=self.config.query_plan,
            debug_trace=self.config.debug_trace,
            verbose=False  # Disable verbose in pipeline
        )
        
        # Limit to top_k_final
        results = results[:self.config.top_k_final]
        
        metrics['num_results'] = len(results)
        metrics['retrieval_mode'] = self.config.retrieval_mode
        
        return results, metrics
    
    def synthesize(self, query: str, results: List):
        """
        Synthesize answer from results
        
        Args:
            query: User query
            results: Retrieved results
            
        Returns:
            SynthesisResult
        """
        return self.answer_synthesizer.synthesize(
            query,
            results,
            query_plan=self.config.query_plan,
            debug_trace=self.config.debug_trace,
        )
    
    def run(self, query: str, config: RAGConfig = None) -> RAGPipelineResult:
        """
        Run complete RAG pipeline
        
        Args:
            query: User query
            config: Optional config override
            
        Returns:
            RAGPipelineResult
        """
        start_time = time.time()
        old_config = None
        
        # Use provided config or default
        if config:
            old_config = self.config
            self.config = config
            self._init_components()
        
        try:
            # Step 1: Retrieve from chunks
            print(f"\n Step 1: Retrieval ({self.config.retrieval_mode})")
            results, retrieval_metrics = self.retrieve(query)
            print(f"   Retrieved: {len(results)} chunks")
            retrieval_time = time.time() - start_time
            
            # Step 2: Synthesize
            print(f"\n Step 2: Answer Synthesis")
            synthesis_result = self.synthesize(query, results)
            synthesis_time = time.time() - start_time - retrieval_time
            print(f"   Generated: {len(synthesis_result.answer)} chars")
            print(f"   Confidence: {synthesis_result.confidence:.2%}")
            print(f"   Groundedness: {synthesis_result.groundedness_score:.2%}")
            print(f"   Provenance: {synthesis_result.provenance_score:.2%}")

            if self.config.debug_trace:
                debug_trace = retrieval_metrics.setdefault("debug_trace", {})
                synthesis_debug_trace = (synthesis_result.metrics or {}).get("debug_trace", {})
                if synthesis_debug_trace:
                    debug_trace["synthesis"] = synthesis_debug_trace
                llm_input = (synthesis_result.metrics or {}).get("llm_input", {})
                if llm_input:
                    debug_trace["llm_input"] = llm_input
            
            # Build result
            pipeline_result = RAGPipelineResult(
                query=query,
                config=asdict(self.config),
                retrieved_chunks=[
                    {
                        'chunk_id': r.chunk_id,
                        'text': r.text[:200] + "...",
                        'score': r.score,
                        'metadata': r.metadata
                    }
                    for r in results
                ],
                retrieval_metrics=retrieval_metrics,
                answer=synthesis_result.answer,
                citations=synthesis_result.citations,
                confidence=synthesis_result.confidence,
                context_used=synthesis_result.context_used,
                groundedness_score=synthesis_result.groundedness_score,
                provenance_score=synthesis_result.provenance_score,
                quality_metrics=synthesis_result.quality_metrics,
                evidence_spans=synthesis_result.evidence_spans,
                selected_evidence=synthesis_result.selected_evidence,
                claim_analyses=synthesis_result.claim_analyses,
                revision_applied=synthesis_result.revision_applied,
                retrieval_time=retrieval_time,
                synthesis_time=synthesis_time,
                total_time=time.time() - start_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                success=True
            )

            return pipeline_result
            
        except Exception as e:
            print(f"\n Error: {e}")
            import traceback
            traceback.print_exc()
            
            return RAGPipelineResult(
                query=query,
                config=asdict(self.config),
                retrieved_chunks=[],
                retrieval_metrics={},
                answer="",
                citations=[],
                confidence=0.0,
                context_used="",
                groundedness_score=0.0,
                provenance_score=0.0,
                quality_metrics=empty_quality_metrics(),
                evidence_spans=[],
                selected_evidence=[],
                claim_analyses=[],
                revision_applied=False,
                retrieval_time=0.0,
                synthesis_time=0.0,
                total_time=time.time() - start_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                success=False,
                error_message=str(e)
            )
        
        finally:
            # Restore config if overridden
            if config and old_config is not None:
                self.config = old_config
                self._init_components()
    
    def print_result(self, result: RAGPipelineResult):
        """Print pipeline result"""
        print(f"\n{'='*70}")
        print(" RAG PIPELINE RESULT")
        print(f"{'='*70}")
        
        print(f"\n Query: {result.query}")
        
        print(f"\n Answer:")
        print(f"{result.answer}")
        
        print(f"\n Citations ({len(result.citations)}):")
        for i, citation in enumerate(result.citations, 1):
            print(f"   {i}. {citation}")
        
        print(f"\n Metrics:")
        print(f"   • Confidence: {result.confidence:.2%}")
        print(f"   • Groundedness: {result.groundedness_score:.2%}")
        print(f"   • Provenance: {result.provenance_score:.2%}")
        print(f"   • Retrieved chunks: {len(result.retrieved_chunks)}")
        print(f"   • Selected evidence: {len(result.selected_evidence)}")
        print(f"   • Retrieval time: {result.retrieval_time:.3f}s")
        print(f"   • Synthesis time: {result.synthesis_time:.3f}s")
        print(f"   • Total time: {result.total_time:.3f}s")
        print(f"   • Revision applied: {result.revision_applied}")
        
        print(f"\n Config:")
        print(f"   • Retrieval mode: {result.config['retrieval_mode']}")
        print(f"   • Top-K: {result.config['top_k_final']}")
        print(f"   • Use HyDE: {result.config['use_hyde']}")
        print(f"   • Use MMR: {result.config['use_mmr']}")
        
        print(f"\n{'='*70}")

# ==============================================================================
# INITIALIZE PIPELINE
# ==============================================================================

print("\n" + "="*70)
print(" Initializing Pipeline")
print("="*70)

# Check required variables (retrieve_enhanced from Cell 5 must be in same globals)
required_vars = {
    'embedder': 'Sentence Transformer (from Cell 3)',
    'faiss_index_b': 'FAISS Index (from Cell 4)',
    'bm25_index': 'BM25 Index (from Cell 4)',
    'chunks': 'Chunks (from Cell 4)',
    'generate_text': 'LLM generate function (from Cell 3)',
    'retrieve_enhanced': 'Retrieval function (from Cell 5, same exec context)'
}

missing = [var for var in required_vars if var not in globals()]

if missing:
    print(" Error: Missing required variables:")
    for var in missing:
        print(f"   • {var}: {required_vars[var]}")
    print("\n Please run Cell 3 and Cell 4 first!")
else:
    print(" All required variables found")
    
    print("\n⏳ Initializing RAG pipeline...")
    
    rag_pipeline = RAGPipeline(
        embedder=embedder,
        faiss_index=faiss_index_b,
        bm25_index=bm25_index,
        chunks=chunks,
        llm_generate_func=generate_text,
        retrieve_func=retrieve_enhanced,
        config=DEFAULT_CONFIG,
    )
    
    print(" RAG Pipeline initialized successfully!")

# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def ask(query: str, config: RAGConfig = None) -> RAGPipelineResult:
    """
    Convenience function to ask a question
    
    Args:
        query: User query
        config: Optional config override
        
    Returns:
        RAGPipelineResult
    """
    result = rag_pipeline.run(query, config)
    rag_pipeline.print_result(result)
    return result

def ask_simple(query: str) -> str:
    """
    Simple ask function that returns just the answer
    
    Args:
        query: User query
        
    Returns:
        Answer string
    """
    result = rag_pipeline.run(query)
    return result.answer

# ==============================================================================
# TEST PIPELINE
# ==============================================================================

print("\n" + "="*70)
print(" Test Pipeline")
print("="*70)

# # Test query
# test_query = "Điều kiện tốt nghiệp đại học là gì?"

# print(f"\n Testing with query: {test_query}")

# try:
#     result = ask(test_query)
    
#     print("\n Pipeline test completed successfully!")
    
# except Exception as e:
#     print(f"\n Error during pipeline test: {e}")
#     import traceback
#     traceback.print_exc()

# ==============================================================================
# EXAMPLE CONFIGURATIONS
# ==============================================================================

print("\n" + "="*70)
print(" Example Configurations")
print("="*70)

print("""
# Example 1: DUAL mode (Hybrid + HyDE) - Best quality 
dual_config = RAGConfig(
    retrieval_mode="dual",  # Auto-enables both Hybrid and HyDE
    use_reranking=True,
    use_mmr=True,
    top_k_final=5
)
result = ask("Your question?", config=dual_config)

# Example 2: HyDE ONLY mode
hyde_config = RAGConfig(
    retrieval_mode="hyde",  # Only HyDE, no Hybrid
    use_reranking=True,
    use_mmr=True,
    top_k_final=5
)
result = ask("Your question?", config=hyde_config)

# Example 3: HYBRID ONLY mode
hybrid_config = RAGConfig(
    retrieval_mode="hybrid",  # Only Hybrid, no HyDE
    use_reranking=True,
    use_mmr=True,
    top_k_final=5
)
result = ask("Your question?", config=hybrid_config)

# Example 4: Fast mode (no reranking, no MMR)
fast_config = RAGConfig(
    retrieval_mode="hybrid",
    use_reranking=False,
    use_mmr=False,
    use_context_pruning=False,
    top_k_final=3
)
result = ask("Your question?", config=fast_config)

# Example 5: With metadata filter
filtered_config = RAGConfig(
    retrieval_mode="dual",
    metadata_filters={'doc_type': 'quy_che', 'year': 2024}
)
result = ask("Your question?", config=filtered_config)

# Simple usage
answer = ask_simple("Your question?")
print(answer)
""")

print("\n" + "="*70)
print(" CELL 8 COMPLETE - PIPELINE READY!")
print("="*70)

print("\n Exported Objects:")
print("   • rag_pipeline - Complete RAG Pipeline")
print("   • ask(query, config) - Ask with full result")
print("   • ask_simple(query) - Ask with answer only")
print("   • RAGConfig - Configuration class")
print("   • RAGPipelineResult - Result class")

print("\n Usage:")
print("   result = ask('Your question?')")
print("   answer = ask_simple('Your question?')")

print("\n Tips:")
print("   • Use ask() for detailed results")
print("   • Use ask_simple() for quick answers")
print("   • Customize with RAGConfig")
print("   • Save results to JSON for analysis")
