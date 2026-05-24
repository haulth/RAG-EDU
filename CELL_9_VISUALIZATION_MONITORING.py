# ==============================================================================
# @title CELL 9: VISUALIZATION & MONITORING
# ==============================================================================

"""
CELL 9 - Visualization & Monitoring

CHỨC NĂNG:
 Metadata Statistics - Thống kê metadata
 Vector Visualization - t-SNE, PCA visualization
 Retrieval Analysis - Analyze retrieval quality
 Performance Monitoring - Track performance metrics
 Query Analysis - Analyze query patterns
 Interactive Dashboards - Plotly interactive charts

BENEFITS:
- Visual insights into data
- Monitor system performance
- Identify issues quickly
- Understand query patterns
"""

print("="*70)
print(" CELL 9: VISUALIZATION & MONITORING")
print("="*70)

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import List, Dict
import json

# Try to import optional visualization libraries
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    print(" scikit-learn not available. Vector visualization disabled.")
    SKLEARN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print(" plotly not available. Interactive charts disabled.")
    PLOTLY_AVAILABLE = False

# ==============================================================================
# METADATA STATISTICS
# ==============================================================================

class MetadataAnalyzer:
    """Analyze metadata statistics"""
    
    def __init__(self, chunks: List):
        self.chunks = chunks
    
    def analyze(self) -> Dict:
        """Analyze metadata"""
        stats = {
            'total_chunks': len(self.chunks),
            'by_doc_type': Counter(),
            'by_year': Counter(),
            'by_category': Counter(),
            'by_filename': Counter(),
            'chunk_lengths': [],
        }
        
        for chunk in self.chunks:
            metadata = chunk.metadata if hasattr(chunk, 'metadata') else chunk.get('metadata', {})
            text = chunk.text if hasattr(chunk, 'text') else chunk.get('text', '')
            
            # Count by metadata fields
            stats['by_doc_type'][metadata.get('doc_type', 'unknown')] += 1
            stats['by_year'][metadata.get('year', 'unknown')] += 1
            stats['by_category'][metadata.get('category', 'unknown')] += 1
            stats['by_filename'][metadata.get('filename', 'unknown')] += 1
            
            # Chunk length
            stats['chunk_lengths'].append(len(text))
        
        # Calculate length statistics
        lengths = stats['chunk_lengths']
        stats['length_stats'] = {
            'min': min(lengths) if lengths else 0,
            'max': max(lengths) if lengths else 0,
            'mean': np.mean(lengths) if lengths else 0,
            'median': np.median(lengths) if lengths else 0,
            'std': np.std(lengths) if lengths else 0
        }
        
        return stats
    
    def print_stats(self, stats: Dict = None):
        """Print statistics"""
        if stats is None:
            stats = self.analyze()
        
        print(f"\n{'='*70}")
        print(" METADATA STATISTICS")
        print(f"{'='*70}")
        
        print(f"\n Total Chunks: {stats['total_chunks']}")
        
        print(f"\n By Document Type:")
        for doc_type, count in stats['by_doc_type'].most_common():
            pct = count / stats['total_chunks'] * 100
            print(f"   • {doc_type:20s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\n By Year:")
        for year, count in sorted(stats['by_year'].items(), key=lambda x: (x[0] == 'unknown', x[0])):
            pct = count / stats['total_chunks'] * 100
            print(f"   • {str(year):20s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\n By Category:")
        for category, count in stats['by_category'].most_common(10):
            pct = count / stats['total_chunks'] * 100
            print(f"   • {category[:30]:30s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\n By Filename (Top 10):")
        for filename, count in stats['by_filename'].most_common(10):
            pct = count / stats['total_chunks'] * 100
            print(f"   • {filename[:30]:30s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\n Chunk Length Statistics:")
        length_stats = stats['length_stats']
        print(f"   • Min: {length_stats['min']:.0f} chars")
        print(f"   • Max: {length_stats['max']:.0f} chars")
        print(f"   • Mean: {length_stats['mean']:.0f} chars")
        print(f"   • Median: {length_stats['median']:.0f} chars")
        print(f"   • Std Dev: {length_stats['std']:.0f} chars")
    
    def plot_distributions(self, stats: Dict = None):
        """Plot metadata distributions"""
        if stats is None:
            stats = self.analyze()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Metadata Distributions', fontsize=16)
        
        # Doc type distribution
        ax = axes[0, 0]
        doc_types = stats['by_doc_type']
        ax.bar(doc_types.keys(), doc_types.values())
        ax.set_title('Distribution by Document Type')
        ax.set_xlabel('Document Type')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        
        # Year distribution
        ax = axes[0, 1]
        years = dict(sorted(stats['by_year'].items()))
        ax.bar([str(y) for y in years.keys()], years.values())
        ax.set_title('Distribution by Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        
        # Chunk length distribution
        ax = axes[1, 0]
        ax.hist(stats['chunk_lengths'], bins=50, edgecolor='black')
        ax.set_title('Chunk Length Distribution')
        ax.set_xlabel('Length (characters)')
        ax.set_ylabel('Frequency')
        ax.axvline(stats['length_stats']['mean'], color='r', linestyle='--', label='Mean')
        ax.axvline(stats['length_stats']['median'], color='g', linestyle='--', label='Median')
        ax.legend()
        
        # Top categories
        ax = axes[1, 1]
        top_categories = dict(stats['by_category'].most_common(10))
        ax.barh(list(top_categories.keys()), list(top_categories.values()))
        ax.set_title('Top 10 Categories')
        ax.set_xlabel('Count')
        ax.set_ylabel('Category')
        
        plt.tight_layout()
        plt.show()

# ==============================================================================
# VECTOR VISUALIZATION
# ==============================================================================

class VectorVisualizer:
    """Visualize vector embeddings"""
    
    def __init__(self, embedder, chunks: List):
        self.embedder = embedder
        self.chunks = chunks
    
    def visualize_tsne(self, n_samples: int = 500, perplexity: int = 30):
        """Visualize embeddings using t-SNE"""
        if not SKLEARN_AVAILABLE:
            print(" scikit-learn not available")
            return
        
        print(f"\n⏳ Computing t-SNE visualization...")
        
        # Sample chunks
        sample_chunks = self.chunks[:n_samples] if len(self.chunks) > n_samples else self.chunks
        
        # Get embeddings
        texts = [c.text if hasattr(c, 'text') else c['text'] for c in sample_chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Get colors by doc_type
        doc_types = [
            (c.metadata if hasattr(c, 'metadata') else c.get('metadata', {})).get('doc_type', 'unknown')
            for c in sample_chunks
        ]
        unique_types = list(set(doc_types))
        color_map = {dt: i for i, dt in enumerate(unique_types)}
        colors = [color_map[dt] for dt in doc_types]
        
        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=colors,
            cmap='tab10',
            alpha=0.6,
            s=50
        )
        
        # Legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=scatter.cmap(scatter.norm(color_map[dt])), 
                             markersize=10, label=dt)
                  for dt in unique_types]
        plt.legend(handles=handles, title='Document Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title(f't-SNE Visualization of Chunk Embeddings (n={len(sample_chunks)})')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        plt.show()
        
        print(" t-SNE visualization complete")
    
    def visualize_pca(self, n_samples: int = 500):
        """Visualize embeddings using PCA"""
        if not SKLEARN_AVAILABLE:
            print(" scikit-learn not available")
            return
        
        print(f"\n⏳ Computing PCA visualization...")
        
        # Sample chunks
        sample_chunks = self.chunks[:n_samples] if len(self.chunks) > n_samples else self.chunks
        
        # Get embeddings
        texts = [c.text if hasattr(c, 'text') else c['text'] for c in sample_chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Get colors by doc_type
        doc_types = [
            (c.metadata if hasattr(c, 'metadata') else c.get('metadata', {})).get('doc_type', 'unknown')
            for c in sample_chunks
        ]
        unique_types = list(set(doc_types))
        color_map = {dt: i for i, dt in enumerate(unique_types)}
        colors = [color_map[dt] for dt in doc_types]
        
        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=colors,
            cmap='tab10',
            alpha=0.6,
            s=50
        )
        
        # Legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=scatter.cmap(scatter.norm(color_map[dt])), 
                             markersize=10, label=dt)
                  for dt in unique_types]
        plt.legend(handles=handles, title='Document Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title(f'PCA Visualization of Chunk Embeddings (n={len(sample_chunks)})')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.tight_layout()
        plt.show()
        
        print(" PCA visualization complete")

# ==============================================================================
# PERFORMANCE MONITOR
# ==============================================================================

class PerformanceMonitor:
    """Monitor RAG system performance"""
    
    def __init__(self):
        self.query_log = []
    
    def log_query(self, result):
        """Log query result"""
        self.query_log.append({
            'timestamp': result.timestamp,
            'query': result.query,
            'retrieval_time': result.retrieval_time,
            'synthesis_time': result.synthesis_time,
            'total_time': result.total_time,
            'confidence': result.confidence,
            'num_chunks': len(result.retrieved_chunks),
            'answer_length': len(result.answer),
            'success': result.success
        })
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.query_log:
            return {}
        
        stats = {
            'total_queries': len(self.query_log),
            'successful_queries': sum(1 for q in self.query_log if q['success']),
            'failed_queries': sum(1 for q in self.query_log if not q['success']),
            'avg_retrieval_time': np.mean([q['retrieval_time'] for q in self.query_log]),
            'avg_synthesis_time': np.mean([q['synthesis_time'] for q in self.query_log]),
            'avg_total_time': np.mean([q['total_time'] for q in self.query_log]),
            'avg_confidence': np.mean([q['confidence'] for q in self.query_log]),
            'avg_chunks_retrieved': np.mean([q['num_chunks'] for q in self.query_log]),
            'avg_answer_length': np.mean([q['answer_length'] for q in self.query_log]),
        }
        
        return stats
    
    def print_stats(self):
        """Print performance statistics"""
        stats = self.get_stats()
        
        if not stats:
            print("No queries logged yet")
            return
        
        print(f"\n{'='*70}")
        print(" PERFORMANCE STATISTICS")
        print(f"{'='*70}")
        
        print(f"\n Query Statistics:")
        print(f"   • Total queries: {stats['total_queries']}")
        print(f"   • Successful: {stats['successful_queries']}")
        print(f"   • Failed: {stats['failed_queries']}")
        print(f"   • Success rate: {stats['successful_queries']/stats['total_queries']:.1%}")
        
        print(f"\n⏱ Timing Statistics:")
        print(f"   • Avg retrieval time: {stats['avg_retrieval_time']:.3f}s")
        print(f"   • Avg synthesis time: {stats['avg_synthesis_time']:.3f}s")
        print(f"   • Avg total time: {stats['avg_total_time']:.3f}s")
        
        print(f"\n Quality Statistics:")
        print(f"   • Avg confidence: {stats['avg_confidence']:.2%}")
        print(f"   • Avg chunks retrieved: {stats['avg_chunks_retrieved']:.1f}")
        print(f"   • Avg answer length: {stats['avg_answer_length']:.0f} chars")
    
    def plot_performance(self):
        """Plot performance over time"""
        if not self.query_log:
            print("No queries logged yet")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Over Time', fontsize=16)
        
        indices = list(range(len(self.query_log)))
        
        # Total time
        ax = axes[0, 0]
        total_times = [q['total_time'] for q in self.query_log]
        ax.plot(indices, total_times, marker='o')
        ax.set_title('Total Response Time')
        ax.set_xlabel('Query Index')
        ax.set_ylabel('Time (seconds)')
        ax.axhline(np.mean(total_times), color='r', linestyle='--', label='Mean')
        ax.legend()
        
        # Confidence
        ax = axes[0, 1]
        confidences = [q['confidence'] for q in self.query_log]
        ax.plot(indices, confidences, marker='o', color='green')
        ax.set_title('Confidence Score')
        ax.set_xlabel('Query Index')
        ax.set_ylabel('Confidence')
        ax.axhline(np.mean(confidences), color='r', linestyle='--', label='Mean')
        ax.legend()
        
        # Time breakdown
        ax = axes[1, 0]
        retrieval_times = [q['retrieval_time'] for q in self.query_log]
        synthesis_times = [q['synthesis_time'] for q in self.query_log]
        ax.bar(indices, retrieval_times, label='Retrieval', alpha=0.7)
        ax.bar(indices, synthesis_times, bottom=retrieval_times, label='Synthesis', alpha=0.7)
        ax.set_title('Time Breakdown')
        ax.set_xlabel('Query Index')
        ax.set_ylabel('Time (seconds)')
        ax.legend()
        
        # Answer length
        ax = axes[1, 1]
        answer_lengths = [q['answer_length'] for q in self.query_log]
        ax.plot(indices, answer_lengths, marker='o', color='purple')
        ax.set_title('Answer Length')
        ax.set_xlabel('Query Index')
        ax.set_ylabel('Characters')
        ax.axhline(np.mean(answer_lengths), color='r', linestyle='--', label='Mean')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_log(self, filepath: str = "query_log.json"):
        """Save query log to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.query_log, f, ensure_ascii=False, indent=2)
        print(f" Saved query log to {filepath}")
    
    def load_log(self, filepath: str = "query_log.json"):
        """Load query log from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.query_log = json.load(f)
            print(f" Loaded {len(self.query_log)} queries from {filepath}")
        except FileNotFoundError:
            print(f" File not found: {filepath}")

# ==============================================================================
# INITIALIZE COMPONENTS
# ==============================================================================

print("\n" + "="*70)
print(" Initializing Visualization Components")
print("="*70)

# Check required variables
required_vars = {
    'chunks': 'Chunks (from Cell 4)',
    'embedder': 'Sentence Transformer (from Cell 3)'
}

missing = [var for var in required_vars if var not in globals()]

if missing:
    print(" Error: Missing required variables:")
    for var in missing:
        print(f"   • {var}: {required_vars[var]}")
    print("\n Please run Cell 3 and Cell 4 first!")
else:
    print(" All required variables found")
    
    # Initialize components
    print("\n⏳ Initializing components...")
    
    # Metadata Analyzer
    metadata_analyzer = MetadataAnalyzer(chunks)
    print(" Metadata Analyzer initialized")
    
    # Vector Visualizer
    if SKLEARN_AVAILABLE:
        vector_visualizer = VectorVisualizer(embedder, chunks)
        print(" Vector Visualizer initialized")
    else:
        print(" Vector Visualizer disabled (scikit-learn not available)")
    
    # Performance Monitor
    performance_monitor = PerformanceMonitor()
    print(" Performance Monitor initialized")
    
    print("\n All components initialized!")

# ==============================================================================
# QUICK ANALYSIS
# ==============================================================================

print("\n" + "="*70)
print(" Quick Analysis")
print("="*70)

# Analyze metadata
metadata_analyzer.print_stats()

print("\n" + "="*70)
print(" CELL 9 COMPLETE - VISUALIZATION READY!")
print("="*70)

print("\n Exported Objects:")
print("   • metadata_analyzer - Metadata statistics")
print("   • vector_visualizer - Vector visualization (if available)")
print("   • performance_monitor - Performance monitoring")

print("\n Usage:")
print("   # Metadata analysis")
print("   metadata_analyzer.print_stats()")
print("   metadata_analyzer.plot_distributions()")
print("")
print("   # Vector visualization")
print("   vector_visualizer.visualize_tsne()")
print("   vector_visualizer.visualize_pca()")
print("")
print("   # Performance monitoring")
print("   performance_monitor.log_query(result)")
print("   performance_monitor.print_stats()")
print("   performance_monitor.plot_performance()")
