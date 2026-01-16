"""
RAG Comprehensive Analysis Module
=================================

Comprehensive RAG (Retrieval-Augmented Generation) analysis framework covering
chunking, embeddings, vector databases, retrieval, generation, and caching.

Categories:
1. Chunking Analysis - Chunk size, overlap, semantic coherence
2. Embedding Analysis - Embedding quality, dimensionality, coverage
3. Vector Database Analysis - Index efficiency, query performance
4. Retrieval Analysis - Precision, recall, relevance scoring
5. Generation Analysis - Faithfulness, groundedness, attribution
6. Context Window Analysis - Context utilization, truncation
7. Caching Analysis - Cache hit rates, freshness
8. Pipeline Analysis - End-to-end latency, throughput
9. Quality Analysis - Answer accuracy, hallucination detection
10. Cost Analysis - Token usage, API costs
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import math


# =============================================================================
# ENUMS
# =============================================================================

class ChunkingStrategy(Enum):
    """Chunking strategies for documents."""
    FIXED_SIZE = auto()
    SENTENCE = auto()
    PARAGRAPH = auto()
    SEMANTIC = auto()
    RECURSIVE = auto()
    TOKEN_BASED = auto()
    HIERARCHICAL = auto()
    SLIDING_WINDOW = auto()


class EmbeddingModel(Enum):
    """Types of embedding models."""
    OPENAI_ADA = auto()
    OPENAI_SMALL = auto()
    OPENAI_LARGE = auto()
    COHERE = auto()
    SENTENCE_TRANSFORMERS = auto()
    HUGGINGFACE = auto()
    CUSTOM = auto()


class VectorDBType(Enum):
    """Types of vector databases."""
    PINECONE = auto()
    WEAVIATE = auto()
    MILVUS = auto()
    QDRANT = auto()
    CHROMA = auto()
    FAISS = auto()
    PGVECTOR = auto()
    REDIS = auto()


class RetrievalMethod(Enum):
    """Retrieval methods."""
    DENSE = auto()  # Semantic search
    SPARSE = auto()  # BM25, TF-IDF
    HYBRID = auto()  # Dense + Sparse
    RERANKING = auto()
    MULTI_QUERY = auto()
    HyDE = auto()  # Hypothetical Document Embeddings


class IndexType(Enum):
    """Vector index types."""
    FLAT = auto()
    IVF = auto()
    HNSW = auto()
    ANNOY = auto()
    SCANN = auto()
    PQ = auto()  # Product Quantization


class CacheStrategy(Enum):
    """Caching strategies."""
    NONE = auto()
    EXACT_MATCH = auto()
    SEMANTIC = auto()
    LRU = auto()
    TTL = auto()
    HYBRID = auto()


class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = auto()
    GOOD = auto()
    ACCEPTABLE = auto()
    POOR = auto()
    FAILED = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ChunkInfo:
    """Information about a document chunk."""
    chunk_id: str
    document_id: str
    text: str
    start_idx: int
    end_idx: int
    token_count: int
    char_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkingMetrics:
    """Metrics about chunking quality."""
    strategy: ChunkingStrategy
    total_chunks: int
    avg_chunk_size: float
    std_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    overlap_ratio: float
    semantic_coherence_score: float = 0.0
    boundary_quality_score: float = 0.0


@dataclass
class EmbeddingMetrics:
    """Metrics about embedding quality."""
    model: EmbeddingModel
    dimensionality: int
    avg_norm: float
    coverage_score: float
    semantic_quality_score: float
    encoding_time_ms: float = 0.0
    vocabulary_coverage: float = 0.0


@dataclass
class VectorDBMetrics:
    """Metrics about vector database performance."""
    db_type: VectorDBType
    index_type: IndexType
    total_vectors: int
    index_size_mb: float
    avg_query_time_ms: float
    p99_query_time_ms: float
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    build_time_seconds: float = 0.0


@dataclass
class RetrievalResult:
    """Result from retrieval analysis."""
    query: str
    retrieved_chunks: List[ChunkInfo]
    scores: List[float]
    retrieval_time_ms: float
    method: RetrievalMethod


@dataclass
class RetrievalMetrics:
    """Metrics about retrieval quality."""
    method: RetrievalMethod
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    avg_latency_ms: float = 0.0


@dataclass
class GenerationMetrics:
    """Metrics about generation quality in RAG."""
    faithfulness_score: float
    groundedness_score: float
    relevance_score: float
    coherence_score: float
    attribution_accuracy: float
    hallucination_rate: float
    answer_completeness: float


@dataclass
class ContextWindowMetrics:
    """Metrics about context window utilization."""
    max_context_tokens: int
    used_context_tokens: int
    utilization_ratio: float
    num_chunks_used: int
    truncation_occurred: bool
    relevant_content_ratio: float


@dataclass
class CacheMetrics:
    """Metrics about caching performance."""
    strategy: CacheStrategy
    total_queries: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_cache_lookup_ms: float
    cache_size_mb: float
    freshness_score: float = 0.0


@dataclass
class PipelineLatencyBreakdown:
    """Breakdown of RAG pipeline latency."""
    total_latency_ms: float
    embedding_latency_ms: float
    retrieval_latency_ms: float
    reranking_latency_ms: float
    generation_latency_ms: float
    cache_lookup_ms: float = 0.0
    preprocessing_ms: float = 0.0


@dataclass
class CostMetrics:
    """Cost analysis metrics."""
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    embedding_tokens: int
    estimated_cost_usd: float
    cost_per_query: float
    tokens_per_query: float


@dataclass
class RAGAssessment:
    """Comprehensive RAG system assessment."""
    assessment_id: str
    timestamp: datetime
    chunking_quality_score: float
    embedding_quality_score: float
    retrieval_quality_score: float
    generation_quality_score: float
    latency_score: float
    cost_efficiency_score: float
    overall_quality_score: float
    critical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# ANALYZERS - CHUNKING
# =============================================================================

class ChunkingAnalyzer:
    """Analyzer for document chunking quality."""

    def analyze_chunks(
        self,
        chunks: List[ChunkInfo],
        strategy: ChunkingStrategy
    ) -> ChunkingMetrics:
        """Analyze chunking quality."""
        if not chunks:
            return ChunkingMetrics(
                strategy=strategy,
                total_chunks=0,
                avg_chunk_size=0,
                std_chunk_size=0,
                min_chunk_size=0,
                max_chunk_size=0,
                overlap_ratio=0,
            )

        sizes = [c.token_count for c in chunks]
        avg_size = sum(sizes) / len(sizes)
        variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
        std_size = variance ** 0.5

        # Calculate overlap ratio
        overlap_ratio = self._calculate_overlap_ratio(chunks)

        # Semantic coherence (simplified - based on size consistency)
        coherence = 1 / (1 + std_size / avg_size) if avg_size > 0 else 0

        # Boundary quality (simplified)
        boundary_quality = self._assess_boundary_quality(chunks)

        return ChunkingMetrics(
            strategy=strategy,
            total_chunks=len(chunks),
            avg_chunk_size=avg_size,
            std_chunk_size=std_size,
            min_chunk_size=min(sizes),
            max_chunk_size=max(sizes),
            overlap_ratio=overlap_ratio,
            semantic_coherence_score=coherence,
            boundary_quality_score=boundary_quality,
        )

    def _calculate_overlap_ratio(self, chunks: List[ChunkInfo]) -> float:
        """Calculate chunk overlap ratio."""
        if len(chunks) < 2:
            return 0.0

        total_overlap = 0
        for i in range(len(chunks) - 1):
            current = chunks[i]
            next_chunk = chunks[i + 1]

            # Check for text overlap
            if current.document_id == next_chunk.document_id:
                if current.end_idx > next_chunk.start_idx:
                    overlap = current.end_idx - next_chunk.start_idx
                    total_overlap += overlap

        total_chars = sum(c.char_count for c in chunks)
        return total_overlap / total_chars if total_chars > 0 else 0

    def _assess_boundary_quality(self, chunks: List[ChunkInfo]) -> float:
        """Assess chunk boundary quality."""
        good_boundaries = 0
        total_boundaries = max(len(chunks) - 1, 1)

        for chunk in chunks:
            # Good boundary: ends with sentence-ending punctuation
            text = chunk.text.strip()
            if text and text[-1] in '.!?':
                good_boundaries += 1

        return good_boundaries / total_boundaries

    def recommend_chunk_size(
        self,
        document_lengths: List[int],
        embedding_model_limit: int = 512
    ) -> Dict[str, Any]:
        """Recommend optimal chunk size."""
        avg_doc_length = sum(document_lengths) / len(document_lengths) if document_lengths else 0

        # Heuristic recommendations
        if avg_doc_length < 1000:
            recommended_size = min(200, embedding_model_limit)
            overlap = 20
        elif avg_doc_length < 5000:
            recommended_size = min(400, embedding_model_limit)
            overlap = 50
        else:
            recommended_size = min(512, embedding_model_limit)
            overlap = 100

        return {
            "recommended_chunk_size": recommended_size,
            "recommended_overlap": overlap,
            "overlap_percentage": overlap / recommended_size * 100,
            "avg_document_length": avg_doc_length,
        }


# =============================================================================
# ANALYZERS - EMBEDDINGS
# =============================================================================

class EmbeddingAnalyzer:
    """Analyzer for embedding quality."""

    def analyze_embeddings(
        self,
        embeddings: List[List[float]],
        model: EmbeddingModel,
        encoding_time_ms: float = 0.0
    ) -> EmbeddingMetrics:
        """Analyze embedding quality."""
        if not embeddings or not embeddings[0]:
            return EmbeddingMetrics(
                model=model,
                dimensionality=0,
                avg_norm=0,
                coverage_score=0,
                semantic_quality_score=0,
            )

        dimensionality = len(embeddings[0])

        # Calculate norms
        norms = []
        for emb in embeddings:
            norm = sum(e ** 2 for e in emb) ** 0.5
            norms.append(norm)

        avg_norm = sum(norms) / len(norms)

        # Norm consistency (should be similar for good embeddings)
        norm_std = (sum((n - avg_norm) ** 2 for n in norms) / len(norms)) ** 0.5
        norm_consistency = 1 / (1 + norm_std / avg_norm) if avg_norm > 0 else 0

        # Coverage score (how spread out embeddings are)
        coverage = self._calculate_embedding_coverage(embeddings)

        # Semantic quality (based on coverage and consistency)
        semantic_quality = (coverage + norm_consistency) / 2

        return EmbeddingMetrics(
            model=model,
            dimensionality=dimensionality,
            avg_norm=avg_norm,
            coverage_score=coverage,
            semantic_quality_score=semantic_quality,
            encoding_time_ms=encoding_time_ms,
        )

    def _calculate_embedding_coverage(self, embeddings: List[List[float]]) -> float:
        """Calculate how well embeddings cover the embedding space."""
        if len(embeddings) < 2:
            return 1.0

        # Calculate average pairwise distance (sample for efficiency)
        sample_size = min(100, len(embeddings))
        sample = embeddings[:sample_size]

        total_distance = 0
        count = 0

        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                distance = sum((a - b) ** 2 for a, b in zip(sample[i], sample[j])) ** 0.5
                total_distance += distance
                count += 1

        avg_distance = total_distance / count if count > 0 else 0

        # Normalize to 0-1 (higher distance = better coverage)
        # Assuming embeddings are normalized, max distance is ~sqrt(2)
        coverage = min(avg_distance / 1.4, 1.0)

        return coverage

    def compare_embedding_models(
        self,
        results: Dict[str, EmbeddingMetrics]
    ) -> Dict[str, Any]:
        """Compare different embedding models."""
        if not results:
            return {}

        comparison = {}
        for model_name, metrics in results.items():
            comparison[model_name] = {
                "dimensionality": metrics.dimensionality,
                "coverage_score": metrics.coverage_score,
                "semantic_quality_score": metrics.semantic_quality_score,
                "encoding_time_ms": metrics.encoding_time_ms,
                "overall_score": (metrics.coverage_score + metrics.semantic_quality_score) / 2,
            }

        ranked = sorted(comparison.items(), key=lambda x: x[1]["overall_score"], reverse=True)

        return {
            "comparison": comparison,
            "ranking": [name for name, _ in ranked],
            "recommended": ranked[0][0] if ranked else None,
        }


# =============================================================================
# ANALYZERS - VECTOR DATABASE
# =============================================================================

class VectorDBAnalyzer:
    """Analyzer for vector database performance."""

    def analyze_query_performance(
        self,
        query_times_ms: List[float],
        db_type: VectorDBType,
        index_type: IndexType,
        total_vectors: int
    ) -> VectorDBMetrics:
        """Analyze vector database query performance."""
        if not query_times_ms:
            return VectorDBMetrics(
                db_type=db_type,
                index_type=index_type,
                total_vectors=total_vectors,
                index_size_mb=0,
                avg_query_time_ms=0,
                p99_query_time_ms=0,
            )

        sorted_times = sorted(query_times_ms)
        avg_time = sum(query_times_ms) / len(query_times_ms)
        p99_idx = int(len(sorted_times) * 0.99)
        p99_time = sorted_times[min(p99_idx, len(sorted_times) - 1)]

        return VectorDBMetrics(
            db_type=db_type,
            index_type=index_type,
            total_vectors=total_vectors,
            index_size_mb=total_vectors * 768 * 4 / (1024 * 1024),  # Estimate
            avg_query_time_ms=avg_time,
            p99_query_time_ms=p99_time,
        )

    def analyze_recall(
        self,
        retrieved_ids: List[List[str]],
        relevant_ids: List[List[str]],
        k_values: List[int] = [1, 5, 10, 20]
    ) -> Dict[int, float]:
        """Analyze recall at different k values."""
        recall_at_k = {}

        for k in k_values:
            recalls = []
            for retrieved, relevant in zip(retrieved_ids, relevant_ids):
                retrieved_set = set(retrieved[:k])
                relevant_set = set(relevant)

                recall = len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0
                recalls.append(recall)

            recall_at_k[k] = sum(recalls) / len(recalls) if recalls else 0

        return recall_at_k


# =============================================================================
# ANALYZERS - RETRIEVAL
# =============================================================================

class RetrievalAnalyzer:
    """Analyzer for retrieval quality."""

    def analyze_retrieval(
        self,
        results: List[RetrievalResult],
        ground_truth: List[List[str]],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> RetrievalMetrics:
        """Analyze retrieval quality."""
        if not results or not ground_truth:
            return RetrievalMetrics(method=RetrievalMethod.DENSE)

        # Calculate metrics
        precision_at_k = {}
        recall_at_k = {}
        ndcg_at_k = {}

        for k in k_values:
            precisions = []
            recalls = []
            ndcgs = []

            for result, relevant in zip(results, ground_truth):
                retrieved_ids = [c.chunk_id for c in result.retrieved_chunks[:k]]
                relevant_set = set(relevant)

                # Precision@k
                relevant_in_retrieved = len(set(retrieved_ids) & relevant_set)
                precision = relevant_in_retrieved / k
                precisions.append(precision)

                # Recall@k
                recall = relevant_in_retrieved / len(relevant_set) if relevant_set else 0
                recalls.append(recall)

                # NDCG@k
                ndcg = self._calculate_ndcg(retrieved_ids, relevant_set, k)
                ndcgs.append(ndcg)

            precision_at_k[k] = sum(precisions) / len(precisions)
            recall_at_k[k] = sum(recalls) / len(recalls)
            ndcg_at_k[k] = sum(ndcgs) / len(ndcgs)

        # MRR
        mrr = self._calculate_mrr(results, ground_truth)

        # Average latency
        avg_latency = sum(r.retrieval_time_ms for r in results) / len(results)

        return RetrievalMetrics(
            method=results[0].method if results else RetrievalMethod.DENSE,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            mrr=mrr,
            avg_latency_ms=avg_latency,
        )

    def _calculate_ndcg(
        self,
        retrieved_ids: List[str],
        relevant_set: Set[str],
        k: int
    ) -> float:
        """Calculate NDCG@k."""
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = 1 if doc_id in relevant_set else 0
            dcg += rel / math.log2(i + 2)

        # Ideal DCG
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant_set), k)))

        return dcg / idcg if idcg > 0 else 0

    def _calculate_mrr(
        self,
        results: List[RetrievalResult],
        ground_truth: List[List[str]]
    ) -> float:
        """Calculate Mean Reciprocal Rank."""
        reciprocal_ranks = []

        for result, relevant in zip(results, ground_truth):
            relevant_set = set(relevant)
            rr = 0.0

            for i, chunk in enumerate(result.retrieved_chunks):
                if chunk.chunk_id in relevant_set:
                    rr = 1 / (i + 1)
                    break

            reciprocal_ranks.append(rr)

        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0


# =============================================================================
# ANALYZERS - GENERATION
# =============================================================================

class RAGGenerationAnalyzer:
    """Analyzer for RAG generation quality."""

    def analyze_generation(
        self,
        query: str,
        context: List[str],
        generated_response: str,
        reference_answer: Optional[str] = None
    ) -> GenerationMetrics:
        """Analyze generation quality in RAG context."""
        # Faithfulness: how much of response is supported by context
        faithfulness = self._calculate_faithfulness(generated_response, context)

        # Groundedness: overlap with context
        groundedness = self._calculate_groundedness(generated_response, context)

        # Relevance to query
        relevance = self._calculate_relevance(query, generated_response)

        # Coherence
        coherence = self._assess_coherence(generated_response)

        # Attribution accuracy (simplified)
        attribution = groundedness  # In practice, would check specific attributions

        # Hallucination rate
        hallucination_rate = 1 - faithfulness

        # Completeness (compared to reference if available)
        completeness = self._assess_completeness(generated_response, reference_answer)

        return GenerationMetrics(
            faithfulness_score=faithfulness,
            groundedness_score=groundedness,
            relevance_score=relevance,
            coherence_score=coherence,
            attribution_accuracy=attribution,
            hallucination_rate=hallucination_rate,
            answer_completeness=completeness,
        )

    def _calculate_faithfulness(self, response: str, context: List[str]) -> float:
        """Calculate faithfulness to context."""
        response_words = set(response.lower().split())
        context_text = " ".join(context)
        context_words = set(context_text.lower().split())

        overlap = len(response_words & context_words)
        faithfulness = overlap / len(response_words) if response_words else 0

        return min(faithfulness * 1.5, 1.0)  # Scale up

    def _calculate_groundedness(self, response: str, context: List[str]) -> float:
        """Calculate groundedness in context."""
        # Similar to faithfulness but sentence-level
        response_sentences = [s.strip() for s in response.split('.') if s.strip()]
        context_text = " ".join(context).lower()

        grounded_count = 0
        for sent in response_sentences:
            sent_words = set(sent.lower().split())
            context_words = set(context_text.split())

            overlap = len(sent_words & context_words) / len(sent_words) if sent_words else 0
            if overlap > 0.5:
                grounded_count += 1

        return grounded_count / len(response_sentences) if response_sentences else 0

    def _calculate_relevance(self, query: str, response: str) -> float:
        """Calculate response relevance to query."""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        overlap = len(query_words & response_words)
        relevance = overlap / len(query_words) if query_words else 0

        return min(relevance * 2, 1.0)

    def _assess_coherence(self, text: str) -> float:
        """Assess text coherence."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if len(sentences) < 2:
            return 1.0

        # Simple coherence: check for transition words and consistent topics
        transition_words = ['however', 'therefore', 'additionally', 'moreover',
                          'furthermore', 'thus', 'consequently']

        has_transitions = any(
            any(tw in sent.lower() for tw in transition_words)
            for sent in sentences
        )

        return 0.8 if has_transitions else 0.6

    def _assess_completeness(
        self,
        response: str,
        reference: Optional[str]
    ) -> float:
        """Assess answer completeness."""
        if not reference:
            return 0.7  # Default when no reference

        ref_words = set(reference.lower().split())
        resp_words = set(response.lower().split())

        coverage = len(ref_words & resp_words) / len(ref_words) if ref_words else 0
        return min(coverage * 1.5, 1.0)


# =============================================================================
# ANALYZERS - CONTEXT WINDOW
# =============================================================================

class ContextWindowAnalyzer:
    """Analyzer for context window utilization."""

    def analyze_context_usage(
        self,
        chunks: List[ChunkInfo],
        max_tokens: int,
        relevance_scores: Optional[List[float]] = None
    ) -> ContextWindowMetrics:
        """Analyze context window utilization."""
        total_tokens = sum(c.token_count for c in chunks)
        utilization = min(total_tokens / max_tokens, 1.0)
        truncated = total_tokens > max_tokens

        # Relevant content ratio
        if relevance_scores and chunks:
            # Weight tokens by relevance
            weighted_tokens = sum(
                c.token_count * s for c, s in zip(chunks, relevance_scores)
            )
            relevant_ratio = weighted_tokens / total_tokens if total_tokens > 0 else 0
        else:
            relevant_ratio = 0.8  # Default

        return ContextWindowMetrics(
            max_context_tokens=max_tokens,
            used_context_tokens=min(total_tokens, max_tokens),
            utilization_ratio=utilization,
            num_chunks_used=len(chunks),
            truncation_occurred=truncated,
            relevant_content_ratio=relevant_ratio,
        )


# =============================================================================
# ANALYZERS - CACHING
# =============================================================================

class CacheAnalyzer:
    """Analyzer for RAG caching performance."""

    def analyze_cache_performance(
        self,
        cache_hits: int,
        cache_misses: int,
        lookup_times_ms: List[float],
        strategy: CacheStrategy,
        cache_size_mb: float = 0.0
    ) -> CacheMetrics:
        """Analyze cache performance."""
        total = cache_hits + cache_misses
        hit_rate = cache_hits / total if total > 0 else 0
        avg_lookup = sum(lookup_times_ms) / len(lookup_times_ms) if lookup_times_ms else 0

        return CacheMetrics(
            strategy=strategy,
            total_queries=total,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            hit_rate=hit_rate,
            avg_cache_lookup_ms=avg_lookup,
            cache_size_mb=cache_size_mb,
        )

    def recommend_cache_strategy(
        self,
        query_patterns: List[str],
        update_frequency: str = "daily"
    ) -> Dict[str, Any]:
        """Recommend caching strategy."""
        unique_queries = len(set(query_patterns))
        total_queries = len(query_patterns)
        repeat_rate = 1 - (unique_queries / total_queries) if total_queries > 0 else 0

        if repeat_rate > 0.5:
            recommended = CacheStrategy.EXACT_MATCH
            reason = "High query repetition rate suggests exact match caching"
        elif repeat_rate > 0.2:
            recommended = CacheStrategy.SEMANTIC
            reason = "Moderate variation suggests semantic caching"
        else:
            recommended = CacheStrategy.NONE
            reason = "Low repetition - caching may not be beneficial"

        return {
            "recommended_strategy": recommended.name,
            "reason": reason,
            "repeat_rate": repeat_rate,
            "unique_queries": unique_queries,
            "total_queries": total_queries,
        }


# =============================================================================
# ANALYZERS - PIPELINE
# =============================================================================

class RAGPipelineAnalyzer:
    """Analyzer for end-to-end RAG pipeline."""

    def analyze_latency(
        self,
        embedding_time_ms: float,
        retrieval_time_ms: float,
        reranking_time_ms: float,
        generation_time_ms: float,
        cache_time_ms: float = 0.0,
        preprocessing_ms: float = 0.0
    ) -> PipelineLatencyBreakdown:
        """Analyze pipeline latency breakdown."""
        total = (
            embedding_time_ms +
            retrieval_time_ms +
            reranking_time_ms +
            generation_time_ms +
            cache_time_ms +
            preprocessing_ms
        )

        return PipelineLatencyBreakdown(
            total_latency_ms=total,
            embedding_latency_ms=embedding_time_ms,
            retrieval_latency_ms=retrieval_time_ms,
            reranking_latency_ms=reranking_time_ms,
            generation_latency_ms=generation_time_ms,
            cache_lookup_ms=cache_time_ms,
            preprocessing_ms=preprocessing_ms,
        )

    def identify_bottlenecks(
        self,
        breakdown: PipelineLatencyBreakdown
    ) -> List[str]:
        """Identify pipeline bottlenecks."""
        bottlenecks = []
        total = breakdown.total_latency_ms

        if breakdown.generation_latency_ms / total > 0.6:
            bottlenecks.append("Generation is the primary bottleneck - consider smaller model or optimization")
        if breakdown.retrieval_latency_ms / total > 0.3:
            bottlenecks.append("Retrieval is slow - consider indexing optimization")
        if breakdown.embedding_latency_ms / total > 0.2:
            bottlenecks.append("Embedding is slow - consider batch processing or caching")
        if breakdown.reranking_latency_ms / total > 0.2:
            bottlenecks.append("Reranking adds significant latency - evaluate if necessary")

        return bottlenecks


# =============================================================================
# ANALYZERS - COST
# =============================================================================

class CostAnalyzer:
    """Analyzer for RAG cost analysis."""

    def __init__(self):
        # Default pricing (USD per 1K tokens)
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "embedding": {"input": 0.0001},
        }

    def analyze_costs(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        embedding_tokens: int,
        num_queries: int,
        model: str = "gpt-3.5-turbo"
    ) -> CostMetrics:
        """Analyze RAG costs."""
        pricing = self.pricing.get(model, self.pricing["gpt-3.5-turbo"])

        input_cost = prompt_tokens / 1000 * pricing["input"]
        output_cost = completion_tokens / 1000 * pricing["output"]
        embedding_cost = embedding_tokens / 1000 * self.pricing["embedding"]["input"]

        total_cost = input_cost + output_cost + embedding_cost
        total_tokens = prompt_tokens + completion_tokens + embedding_tokens

        return CostMetrics(
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            embedding_tokens=embedding_tokens,
            estimated_cost_usd=total_cost,
            cost_per_query=total_cost / num_queries if num_queries > 0 else 0,
            tokens_per_query=total_tokens / num_queries if num_queries > 0 else 0,
        )


# =============================================================================
# COMPREHENSIVE ANALYZER
# =============================================================================

class RAGComprehensiveAnalyzer:
    """Comprehensive RAG system analyzer."""

    def __init__(self):
        self.chunking_analyzer = ChunkingAnalyzer()
        self.embedding_analyzer = EmbeddingAnalyzer()
        self.vectordb_analyzer = VectorDBAnalyzer()
        self.retrieval_analyzer = RetrievalAnalyzer()
        self.generation_analyzer = RAGGenerationAnalyzer()
        self.context_analyzer = ContextWindowAnalyzer()
        self.cache_analyzer = CacheAnalyzer()
        self.pipeline_analyzer = RAGPipelineAnalyzer()
        self.cost_analyzer = CostAnalyzer()

    def comprehensive_assessment(
        self,
        chunking_metrics: ChunkingMetrics,
        embedding_metrics: EmbeddingMetrics,
        retrieval_metrics: RetrievalMetrics,
        generation_metrics: GenerationMetrics,
        latency_breakdown: PipelineLatencyBreakdown,
        cost_metrics: CostMetrics
    ) -> RAGAssessment:
        """Perform comprehensive RAG assessment."""
        assessment_id = f"RAG-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        critical_issues = []
        recommendations = []

        # Chunking quality
        chunking_score = chunking_metrics.semantic_coherence_score

        # Embedding quality
        embedding_score = embedding_metrics.semantic_quality_score

        # Retrieval quality
        retrieval_score = (
            retrieval_metrics.precision_at_k.get(5, 0) +
            retrieval_metrics.recall_at_k.get(5, 0) +
            retrieval_metrics.mrr
        ) / 3

        if retrieval_score < 0.5:
            critical_issues.append("Low retrieval quality - check embedding model or chunking strategy")

        # Generation quality
        generation_score = (
            generation_metrics.faithfulness_score +
            generation_metrics.groundedness_score +
            generation_metrics.relevance_score
        ) / 3

        if generation_metrics.hallucination_rate > 0.2:
            critical_issues.append(f"High hallucination rate: {generation_metrics.hallucination_rate:.1%}")

        # Latency score (lower is better)
        target_latency = 2000  # 2 seconds
        latency_score = min(target_latency / latency_breakdown.total_latency_ms, 1.0)

        if latency_breakdown.total_latency_ms > 5000:
            recommendations.append("Total latency exceeds 5s - optimize pipeline")

        # Cost efficiency
        target_cost_per_query = 0.01  # $0.01 per query
        cost_efficiency = min(target_cost_per_query / max(cost_metrics.cost_per_query, 0.001), 1.0)

        # Overall score
        overall = (
            chunking_score * 0.1 +
            embedding_score * 0.15 +
            retrieval_score * 0.25 +
            generation_score * 0.3 +
            latency_score * 0.1 +
            cost_efficiency * 0.1
        )

        # Add recommendations based on scores
        if chunking_score < 0.7:
            recommendations.append("Consider different chunking strategy for better semantic coherence")
        if embedding_score < 0.7:
            recommendations.append("Evaluate alternative embedding models")
        if retrieval_score < 0.6:
            recommendations.append("Improve retrieval with hybrid search or better indexing")

        return RAGAssessment(
            assessment_id=assessment_id,
            timestamp=datetime.now(),
            chunking_quality_score=chunking_score,
            embedding_quality_score=embedding_score,
            retrieval_quality_score=retrieval_score,
            generation_quality_score=generation_score,
            latency_score=latency_score,
            cost_efficiency_score=cost_efficiency,
            overall_quality_score=overall,
            critical_issues=critical_issues,
            recommendations=recommendations,
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def analyze_chunking(
    chunks: List[ChunkInfo],
    strategy: ChunkingStrategy
) -> ChunkingMetrics:
    """Analyze document chunking quality."""
    analyzer = ChunkingAnalyzer()
    return analyzer.analyze_chunks(chunks, strategy)


def analyze_embeddings(
    embeddings: List[List[float]],
    model: EmbeddingModel
) -> EmbeddingMetrics:
    """Analyze embedding quality."""
    analyzer = EmbeddingAnalyzer()
    return analyzer.analyze_embeddings(embeddings, model)


def analyze_retrieval(
    results: List[RetrievalResult],
    ground_truth: List[List[str]]
) -> RetrievalMetrics:
    """Analyze retrieval quality."""
    analyzer = RetrievalAnalyzer()
    return analyzer.analyze_retrieval(results, ground_truth)


def analyze_rag_generation(
    query: str,
    context: List[str],
    response: str
) -> GenerationMetrics:
    """Analyze RAG generation quality."""
    analyzer = RAGGenerationAnalyzer()
    return analyzer.analyze_generation(query, context, response)


def analyze_pipeline_latency(
    embedding_ms: float,
    retrieval_ms: float,
    reranking_ms: float,
    generation_ms: float
) -> PipelineLatencyBreakdown:
    """Analyze RAG pipeline latency."""
    analyzer = RAGPipelineAnalyzer()
    return analyzer.analyze_latency(embedding_ms, retrieval_ms, reranking_ms, generation_ms)


def comprehensive_rag_assessment(
    chunking_metrics: ChunkingMetrics,
    embedding_metrics: EmbeddingMetrics,
    retrieval_metrics: RetrievalMetrics,
    generation_metrics: GenerationMetrics,
    latency_breakdown: PipelineLatencyBreakdown,
    cost_metrics: CostMetrics
) -> RAGAssessment:
    """Perform comprehensive RAG assessment."""
    analyzer = RAGComprehensiveAnalyzer()
    return analyzer.comprehensive_assessment(
        chunking_metrics, embedding_metrics, retrieval_metrics,
        generation_metrics, latency_breakdown, cost_metrics
    )
