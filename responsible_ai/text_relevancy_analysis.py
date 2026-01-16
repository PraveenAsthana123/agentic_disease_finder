"""
Text Relevancy Analysis Module for GenAI Evaluation.

This module provides comprehensive text relevancy scoring with 27 dimensions,
including positive relevancy, negative relevancy, and uncertainty handling
as specified in AI/GenAI evaluation frameworks.

Classes:
    RelevancyDimensionAnalyzer: Base analyzer for relevancy dimensions
    SemanticRelevancyAnalyzer: Semantic similarity and meaning alignment
    FactualRelevancyAnalyzer: Factual accuracy and consistency scoring
    ContextualRelevancyAnalyzer: Context-aware relevancy analysis
    NegativeRelevancyAnalyzer: Detection of harmful/incorrect content
    UncertaintyRelevancyAnalyzer: Uncertainty quantification in outputs
    TemporalRelevancyAnalyzer: Time-sensitive relevancy analysis
    ComprehensiveRelevancyScorer: Multi-dimensional relevancy scoring (27 dimensions)

Author: AgenticFinder Research Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
import warnings
from collections import defaultdict


class RelevancyType(Enum):
    """Types of relevancy assessment."""
    SEMANTIC = "semantic"
    FACTUAL = "factual"
    CONTEXTUAL = "contextual"
    TOPICAL = "topical"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    NEGATIVE = "negative"
    UNCERTAINTY = "uncertainty"


class RelevancyLevel(Enum):
    """Relevancy scoring levels."""
    HIGHLY_RELEVANT = 5
    RELEVANT = 4
    PARTIALLY_RELEVANT = 3
    MARGINALLY_RELEVANT = 2
    NOT_RELEVANT = 1
    CONTRADICTORY = 0
    HARMFUL = -1


@dataclass
class RelevancyScore:
    """Container for relevancy scoring results."""
    dimension: str
    score: float
    confidence: float
    evidence: List[str] = field(default_factory=list)
    sub_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'dimension': self.dimension,
            'score': self.score,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'sub_scores': self.sub_scores,
            'metadata': self.metadata
        }


@dataclass
class RelevancyProfile:
    """Complete relevancy profile across all dimensions."""
    text: str
    reference: Optional[str]
    scores: Dict[str, RelevancyScore] = field(default_factory=dict)
    aggregate_score: float = 0.0
    weighted_score: float = 0.0
    negative_flags: List[str] = field(default_factory=list)
    uncertainty_indicators: List[str] = field(default_factory=list)

    def get_radar_data(self) -> Dict[str, float]:
        """Get data formatted for radar chart visualization."""
        return {dim: score.score for dim, score in self.scores.items()}


class RelevancyDimensionAnalyzer(ABC):
    """
    Abstract base class for relevancy dimension analyzers.

    Provides common interface for all relevancy scoring components.
    """

    def __init__(self, dimension_name: str, weight: float = 1.0):
        """
        Initialize relevancy dimension analyzer.

        Args:
            dimension_name: Name of the relevancy dimension
            weight: Weight for this dimension in aggregate scoring
        """
        self.dimension_name = dimension_name
        self.weight = weight
        self._cache = {}

    @abstractmethod
    def analyze(self, text: str, reference: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None) -> RelevancyScore:
        """
        Analyze text for this relevancy dimension.

        Args:
            text: Text to analyze
            reference: Optional reference text for comparison
            context: Optional context information

        Returns:
            RelevancyScore for this dimension
        """
        pass

    def _normalize_score(self, raw_score: float,
                         min_val: float = 0.0,
                         max_val: float = 1.0) -> float:
        """Normalize score to [0, 1] range."""
        if max_val == min_val:
            return 0.5
        return np.clip((raw_score - min_val) / (max_val - min_val), 0.0, 1.0)

    def _compute_confidence(self, scores: List[float]) -> float:
        """Compute confidence based on score consistency."""
        if len(scores) < 2:
            return 0.5
        std = np.std(scores)
        return 1.0 - min(std, 0.5) * 2


class SemanticRelevancyAnalyzer(RelevancyDimensionAnalyzer):
    """
    Analyzer for semantic relevancy between texts.

    Evaluates meaning alignment, semantic similarity, and
    conceptual correspondence between generated and reference text.
    """

    def __init__(self, weight: float = 1.0):
        """Initialize semantic relevancy analyzer."""
        super().__init__("semantic_relevancy", weight)
        self.sub_dimensions = [
            'meaning_preservation',
            'concept_alignment',
            'entity_consistency',
            'relationship_preservation',
            'semantic_completeness'
        ]

    def analyze(self, text: str, reference: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None) -> RelevancyScore:
        """
        Analyze semantic relevancy of text.

        Args:
            text: Generated text to evaluate
            reference: Reference text for comparison
            context: Optional context with domain information

        Returns:
            SemanticRelevancyScore with sub-dimension scores
        """
        sub_scores = {}
        evidence = []

        # Meaning preservation analysis
        meaning_score = self._analyze_meaning_preservation(text, reference)
        sub_scores['meaning_preservation'] = meaning_score

        # Concept alignment
        concept_score = self._analyze_concept_alignment(text, reference, context)
        sub_scores['concept_alignment'] = concept_score

        # Entity consistency
        entity_score = self._analyze_entity_consistency(text, reference)
        sub_scores['entity_consistency'] = entity_score

        # Relationship preservation
        relationship_score = self._analyze_relationship_preservation(text, reference)
        sub_scores['relationship_preservation'] = relationship_score

        # Semantic completeness
        completeness_score = self._analyze_semantic_completeness(text, reference, context)
        sub_scores['semantic_completeness'] = completeness_score

        # Aggregate score
        aggregate = np.mean(list(sub_scores.values()))
        confidence = self._compute_confidence(list(sub_scores.values()))

        # Generate evidence
        if meaning_score < 0.5:
            evidence.append("Low meaning preservation detected")
        if entity_score < 0.5:
            evidence.append("Entity inconsistencies found")

        return RelevancyScore(
            dimension=self.dimension_name,
            score=aggregate,
            confidence=confidence,
            evidence=evidence,
            sub_scores=sub_scores,
            metadata={'analyzer': 'semantic', 'text_length': len(text)}
        )

    def _analyze_meaning_preservation(self, text: str,
                                       reference: Optional[str]) -> float:
        """Analyze how well meaning is preserved."""
        if not reference:
            return 0.7  # Default for no reference

        # Simplified lexical overlap as proxy for meaning preservation
        text_tokens = set(text.lower().split())
        ref_tokens = set(reference.lower().split())

        if not ref_tokens:
            return 0.5

        overlap = len(text_tokens & ref_tokens) / len(ref_tokens)
        return self._normalize_score(overlap, 0.0, 1.0)

    def _analyze_concept_alignment(self, text: str,
                                    reference: Optional[str],
                                    context: Optional[Dict[str, Any]]) -> float:
        """Analyze concept-level alignment."""
        # Extract key concepts (simplified: longer words as concepts)
        text_concepts = {w for w in text.lower().split() if len(w) > 5}

        if reference:
            ref_concepts = {w for w in reference.lower().split() if len(w) > 5}
            if ref_concepts:
                return len(text_concepts & ref_concepts) / len(ref_concepts)

        if context and 'expected_concepts' in context:
            expected = set(context['expected_concepts'])
            return len(text_concepts & expected) / len(expected) if expected else 0.5

        return 0.6

    def _analyze_entity_consistency(self, text: str,
                                     reference: Optional[str]) -> float:
        """Analyze entity consistency between texts."""
        # Simplified: check capitalized words as entities
        text_entities = {w for w in text.split() if w and w[0].isupper()}

        if reference:
            ref_entities = {w for w in reference.split() if w and w[0].isupper()}
            if ref_entities:
                precision = len(text_entities & ref_entities) / len(text_entities) if text_entities else 0
                recall = len(text_entities & ref_entities) / len(ref_entities)
                return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return 0.7

    def _analyze_relationship_preservation(self, text: str,
                                            reference: Optional[str]) -> float:
        """Analyze preservation of relationships between entities."""
        # Simplified: verb preservation as proxy
        common_verbs = {'is', 'are', 'was', 'were', 'has', 'have', 'had',
                       'does', 'do', 'did', 'makes', 'uses', 'provides'}

        text_verbs = {w.lower() for w in text.split()} & common_verbs

        if reference:
            ref_verbs = {w.lower() for w in reference.split()} & common_verbs
            if ref_verbs:
                return len(text_verbs & ref_verbs) / len(ref_verbs)

        return 0.6

    def _analyze_semantic_completeness(self, text: str,
                                        reference: Optional[str],
                                        context: Optional[Dict[str, Any]]) -> float:
        """Analyze semantic completeness of the text."""
        if reference:
            ref_length = len(reference.split())
            text_length = len(text.split())
            if ref_length > 0:
                ratio = text_length / ref_length
                # Penalize both too short and too long
                return 1.0 - abs(1.0 - ratio) * 0.5

        if context and 'min_length' in context:
            text_length = len(text.split())
            if text_length >= context['min_length']:
                return 0.8
            return text_length / context['min_length']

        return 0.7


class FactualRelevancyAnalyzer(RelevancyDimensionAnalyzer):
    """
    Analyzer for factual relevancy and accuracy.

    Evaluates factual consistency, claim verification,
    and factual coverage between generated and reference text.
    """

    def __init__(self, weight: float = 1.0):
        """Initialize factual relevancy analyzer."""
        super().__init__("factual_relevancy", weight)
        self.sub_dimensions = [
            'claim_accuracy',
            'fact_consistency',
            'source_alignment',
            'temporal_accuracy',
            'numerical_precision'
        ]

    def analyze(self, text: str, reference: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None) -> RelevancyScore:
        """
        Analyze factual relevancy of text.

        Args:
            text: Generated text to evaluate
            reference: Reference text containing facts
            context: Optional context with fact database

        Returns:
            FactualRelevancyScore with verification results
        """
        sub_scores = {}
        evidence = []

        # Claim accuracy analysis
        claim_score = self._analyze_claim_accuracy(text, reference, context)
        sub_scores['claim_accuracy'] = claim_score

        # Fact consistency
        consistency_score = self._analyze_fact_consistency(text, reference)
        sub_scores['fact_consistency'] = consistency_score

        # Source alignment
        source_score = self._analyze_source_alignment(text, context)
        sub_scores['source_alignment'] = source_score

        # Temporal accuracy
        temporal_score = self._analyze_temporal_accuracy(text, context)
        sub_scores['temporal_accuracy'] = temporal_score

        # Numerical precision
        numerical_score = self._analyze_numerical_precision(text, reference)
        sub_scores['numerical_precision'] = numerical_score

        # Aggregate score
        aggregate = np.mean(list(sub_scores.values()))
        confidence = self._compute_confidence(list(sub_scores.values()))

        # Generate evidence
        if claim_score < 0.5:
            evidence.append("Potential factual inaccuracies detected")
        if numerical_score < 0.5:
            evidence.append("Numerical inconsistencies found")

        return RelevancyScore(
            dimension=self.dimension_name,
            score=aggregate,
            confidence=confidence,
            evidence=evidence,
            sub_scores=sub_scores,
            metadata={'analyzer': 'factual', 'claim_count': self._count_claims(text)}
        )

    def _analyze_claim_accuracy(self, text: str,
                                 reference: Optional[str],
                                 context: Optional[Dict[str, Any]]) -> float:
        """Analyze accuracy of claims in text."""
        claims = self._extract_claims(text)

        if not claims:
            return 0.7  # No verifiable claims

        verified_count = 0

        if reference:
            ref_lower = reference.lower()
            for claim in claims:
                if claim.lower() in ref_lower:
                    verified_count += 1

        if context and 'facts' in context:
            for claim in claims:
                for fact in context['facts']:
                    if self._claim_matches_fact(claim, fact):
                        verified_count += 1
                        break

        return verified_count / len(claims) if claims else 0.7

    def _analyze_fact_consistency(self, text: str,
                                   reference: Optional[str]) -> float:
        """Analyze internal fact consistency."""
        # Check for contradictions (simplified)
        sentences = text.split('.')

        if len(sentences) < 2:
            return 0.8

        # Simple contradiction detection (negation patterns)
        negations = {'not', 'no', 'never', 'none', "n't", 'neither'}

        contradiction_score = 0.0
        for i, sent1 in enumerate(sentences):
            for sent2 in sentences[i+1:]:
                words1 = set(sent1.lower().split())
                words2 = set(sent2.lower().split())

                # Check if same topic with opposite negation
                common = words1 & words2
                neg1 = bool(words1 & negations)
                neg2 = bool(words2 & negations)

                if len(common) > 3 and neg1 != neg2:
                    contradiction_score += 0.1

        return max(0.0, 1.0 - contradiction_score)

    def _analyze_source_alignment(self, text: str,
                                   context: Optional[Dict[str, Any]]) -> float:
        """Analyze alignment with source materials."""
        if not context or 'sources' not in context:
            return 0.6

        sources = context['sources']
        alignment_scores = []

        text_lower = text.lower()
        for source in sources:
            source_lower = source.lower()
            # Simple overlap measure
            text_words = set(text_lower.split())
            source_words = set(source_lower.split())

            if source_words:
                overlap = len(text_words & source_words) / len(source_words)
                alignment_scores.append(overlap)

        return np.mean(alignment_scores) if alignment_scores else 0.6

    def _analyze_temporal_accuracy(self, text: str,
                                    context: Optional[Dict[str, Any]]) -> float:
        """Analyze temporal accuracy of statements."""
        import re

        # Extract dates/years
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, text)

        if not years:
            return 0.7

        if context and 'valid_time_range' in context:
            start, end = context['valid_time_range']
            valid_years = sum(1 for y in years if start <= int(y) <= end)
            return valid_years / len(years)

        return 0.7

    def _analyze_numerical_precision(self, text: str,
                                      reference: Optional[str]) -> float:
        """Analyze numerical precision and consistency."""
        import re

        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        text_numbers = set(re.findall(number_pattern, text))

        if not text_numbers:
            return 0.8

        if reference:
            ref_numbers = set(re.findall(number_pattern, reference))
            if ref_numbers:
                precision = len(text_numbers & ref_numbers) / len(text_numbers) if text_numbers else 0
                recall = len(text_numbers & ref_numbers) / len(ref_numbers)
                return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return 0.7

    def _extract_claims(self, text: str) -> List[str]:
        """Extract verifiable claims from text."""
        # Simplified: sentences with specific patterns
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        claims = []

        claim_indicators = {'is', 'are', 'was', 'were', 'has', 'have',
                          'provides', 'contains', 'includes', 'equals'}

        for sent in sentences:
            words = set(sent.lower().split())
            if words & claim_indicators:
                claims.append(sent)

        return claims

    def _claim_matches_fact(self, claim: str, fact: str) -> bool:
        """Check if claim matches a fact."""
        claim_words = set(claim.lower().split())
        fact_words = set(fact.lower().split())

        overlap = len(claim_words & fact_words)
        return overlap >= min(len(fact_words) * 0.6, 5)

    def _count_claims(self, text: str) -> int:
        """Count verifiable claims in text."""
        return len(self._extract_claims(text))


class ContextualRelevancyAnalyzer(RelevancyDimensionAnalyzer):
    """
    Analyzer for contextual relevancy.

    Evaluates how well generated text fits the given context,
    conversation history, and situational requirements.
    """

    def __init__(self, weight: float = 1.0):
        """Initialize contextual relevancy analyzer."""
        super().__init__("contextual_relevancy", weight)
        self.sub_dimensions = [
            'context_adherence',
            'topic_consistency',
            'tone_appropriateness',
            'audience_fit',
            'situational_relevance'
        ]

    def analyze(self, text: str, reference: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None) -> RelevancyScore:
        """
        Analyze contextual relevancy of text.

        Args:
            text: Generated text to evaluate
            reference: Reference text or previous context
            context: Context information including topic, tone, audience

        Returns:
            ContextualRelevancyScore with context fitness results
        """
        sub_scores = {}
        evidence = []

        # Context adherence
        adherence_score = self._analyze_context_adherence(text, context)
        sub_scores['context_adherence'] = adherence_score

        # Topic consistency
        topic_score = self._analyze_topic_consistency(text, reference, context)
        sub_scores['topic_consistency'] = topic_score

        # Tone appropriateness
        tone_score = self._analyze_tone_appropriateness(text, context)
        sub_scores['tone_appropriateness'] = tone_score

        # Audience fit
        audience_score = self._analyze_audience_fit(text, context)
        sub_scores['audience_fit'] = audience_score

        # Situational relevance
        situational_score = self._analyze_situational_relevance(text, context)
        sub_scores['situational_relevance'] = situational_score

        # Aggregate score
        aggregate = np.mean(list(sub_scores.values()))
        confidence = self._compute_confidence(list(sub_scores.values()))

        # Generate evidence
        if adherence_score < 0.5:
            evidence.append("Text deviates from provided context")
        if tone_score < 0.5:
            evidence.append("Tone may not match expected style")

        return RelevancyScore(
            dimension=self.dimension_name,
            score=aggregate,
            confidence=confidence,
            evidence=evidence,
            sub_scores=sub_scores,
            metadata={'analyzer': 'contextual'}
        )

    def _analyze_context_adherence(self, text: str,
                                    context: Optional[Dict[str, Any]]) -> float:
        """Analyze how well text adheres to context."""
        if not context:
            return 0.6

        adherence_scores = []

        # Check keyword adherence
        if 'keywords' in context:
            text_lower = text.lower()
            keyword_hits = sum(1 for kw in context['keywords']
                              if kw.lower() in text_lower)
            adherence_scores.append(keyword_hits / len(context['keywords']))

        # Check constraint adherence
        if 'constraints' in context:
            constraints = context['constraints']
            met = 0
            if 'max_length' in constraints:
                if len(text.split()) <= constraints['max_length']:
                    met += 1
            if 'min_length' in constraints:
                if len(text.split()) >= constraints['min_length']:
                    met += 1
            if constraints:
                adherence_scores.append(met / len(constraints))

        return np.mean(adherence_scores) if adherence_scores else 0.6

    def _analyze_topic_consistency(self, text: str,
                                    reference: Optional[str],
                                    context: Optional[Dict[str, Any]]) -> float:
        """Analyze topic consistency with context."""
        if context and 'topic' in context:
            topic_words = set(context['topic'].lower().split())
            text_words = set(text.lower().split())
            overlap = len(topic_words & text_words) / len(topic_words) if topic_words else 0.5
            return min(1.0, overlap * 2)

        if reference:
            ref_words = set(reference.lower().split())
            text_words = set(text.lower().split())
            return len(ref_words & text_words) / len(ref_words) if ref_words else 0.5

        return 0.6

    def _analyze_tone_appropriateness(self, text: str,
                                       context: Optional[Dict[str, Any]]) -> float:
        """Analyze tone appropriateness."""
        if not context or 'expected_tone' not in context:
            return 0.7

        expected_tone = context['expected_tone']

        # Tone indicators
        formal_indicators = {'therefore', 'furthermore', 'consequently', 'hereby',
                           'accordingly', 'nevertheless', 'moreover'}
        informal_indicators = {'hey', 'yeah', 'gonna', 'wanna', 'cool', 'awesome',
                             'stuff', 'things', "don't", "can't"}
        technical_indicators = {'algorithm', 'implementation', 'optimization',
                              'parameter', 'configuration', 'module', 'function'}

        text_words = set(text.lower().split())

        formality_score = len(text_words & formal_indicators)
        informality_score = len(text_words & informal_indicators)
        technicality_score = len(text_words & technical_indicators)

        if expected_tone == 'formal':
            return min(1.0, 0.5 + formality_score * 0.1 - informality_score * 0.1)
        elif expected_tone == 'informal':
            return min(1.0, 0.5 + informality_score * 0.1 - formality_score * 0.1)
        elif expected_tone == 'technical':
            return min(1.0, 0.5 + technicality_score * 0.1)

        return 0.7

    def _analyze_audience_fit(self, text: str,
                              context: Optional[Dict[str, Any]]) -> float:
        """Analyze fit for target audience."""
        if not context or 'audience' not in context:
            return 0.7

        audience = context['audience']

        # Simple complexity analysis
        words = text.split()
        if not words:
            return 0.5

        avg_word_length = np.mean([len(w) for w in words])
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        words_per_sentence = len(words) / max(sentence_count, 1)

        if audience == 'expert':
            # Expect longer words and complex sentences
            complexity_score = min(1.0, avg_word_length / 7 + words_per_sentence / 25)
            return complexity_score
        elif audience == 'general':
            # Prefer moderate complexity
            if 4 <= avg_word_length <= 6 and 10 <= words_per_sentence <= 20:
                return 0.9
            return 0.6
        elif audience == 'beginner':
            # Prefer simpler language
            simplicity_score = max(0, 1.0 - (avg_word_length - 4) * 0.1 -
                                  (words_per_sentence - 12) * 0.02)
            return simplicity_score

        return 0.7

    def _analyze_situational_relevance(self, text: str,
                                        context: Optional[Dict[str, Any]]) -> float:
        """Analyze situational relevance."""
        if not context or 'situation' not in context:
            return 0.6

        situation = context['situation']
        text_lower = text.lower()

        # Situation-specific keyword checking
        situation_keywords = {
            'question_answering': {'answer', 'because', 'reason', 'explanation'},
            'summarization': {'summary', 'main', 'key', 'essentially', 'briefly'},
            'creative_writing': {'story', 'character', 'scene', 'imagine'},
            'instruction': {'step', 'first', 'then', 'finally', 'next'},
            'analysis': {'analysis', 'examine', 'evaluate', 'assess', 'findings'}
        }

        if situation in situation_keywords:
            keywords = situation_keywords[situation]
            hits = sum(1 for kw in keywords if kw in text_lower)
            return min(1.0, 0.5 + hits * 0.15)

        return 0.6


class NegativeRelevancyAnalyzer(RelevancyDimensionAnalyzer):
    """
    Analyzer for negative relevancy detection.

    Identifies harmful, incorrect, biased, or inappropriate
    content that should reduce overall relevancy scores.
    """

    def __init__(self, weight: float = 1.0):
        """Initialize negative relevancy analyzer."""
        super().__init__("negative_relevancy", weight)
        self.sub_dimensions = [
            'contradiction_detection',
            'hallucination_indicators',
            'bias_detection',
            'toxicity_indicators',
            'misinformation_risk'
        ]

    def analyze(self, text: str, reference: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None) -> RelevancyScore:
        """
        Analyze negative relevancy indicators.

        Args:
            text: Generated text to evaluate
            reference: Reference text for contradiction detection
            context: Context for bias and appropriateness checking

        Returns:
            NegativeRelevancyScore (lower is better - indicates problems)
        """
        sub_scores = {}
        evidence = []

        # Contradiction detection (lower = more contradictions = worse)
        contradiction_score = self._detect_contradictions(text, reference)
        sub_scores['contradiction_detection'] = contradiction_score

        # Hallucination indicators
        hallucination_score = self._detect_hallucination_indicators(text, context)
        sub_scores['hallucination_indicators'] = hallucination_score

        # Bias detection
        bias_score = self._detect_bias(text)
        sub_scores['bias_detection'] = bias_score

        # Toxicity indicators
        toxicity_score = self._detect_toxicity(text)
        sub_scores['toxicity_indicators'] = toxicity_score

        # Misinformation risk
        misinfo_score = self._assess_misinformation_risk(text, context)
        sub_scores['misinformation_risk'] = misinfo_score

        # Aggregate score (higher = better = fewer problems)
        aggregate = np.mean(list(sub_scores.values()))
        confidence = self._compute_confidence(list(sub_scores.values()))

        # Generate evidence for problems found
        if contradiction_score < 0.7:
            evidence.append("Potential contradictions detected")
        if hallucination_score < 0.7:
            evidence.append("Hallucination indicators present")
        if bias_score < 0.7:
            evidence.append("Potential bias detected")
        if toxicity_score < 0.7:
            evidence.append("Toxicity indicators found")
        if misinfo_score < 0.7:
            evidence.append("Misinformation risk detected")

        return RelevancyScore(
            dimension=self.dimension_name,
            score=aggregate,
            confidence=confidence,
            evidence=evidence,
            sub_scores=sub_scores,
            metadata={
                'analyzer': 'negative',
                'issues_found': len(evidence),
                'severity': 'high' if aggregate < 0.5 else 'medium' if aggregate < 0.7 else 'low'
            }
        )

    def _detect_contradictions(self, text: str,
                                reference: Optional[str]) -> float:
        """Detect contradictions within text or with reference."""
        issues = 0

        # Internal contradiction check
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if len(sentences) >= 2:
            negation_words = {'not', 'no', 'never', 'none', "n't", 'neither',
                            'without', 'lack', 'absent'}

            for i, sent1 in enumerate(sentences):
                words1 = set(sent1.lower().split())
                has_neg1 = bool(words1 & negation_words)

                for sent2 in sentences[i+1:]:
                    words2 = set(sent2.lower().split())
                    has_neg2 = bool(words2 & negation_words)

                    # Same topic, opposite negation = contradiction
                    common_content = words1 & words2 - negation_words - \
                                   {'the', 'a', 'an', 'is', 'are', 'was', 'were'}

                    if len(common_content) >= 3 and has_neg1 != has_neg2:
                        issues += 1

        # Reference contradiction check
        if reference:
            ref_lower = reference.lower()
            text_lower = text.lower()

            # Check for explicit contradictions
            if 'true' in text_lower and 'false' in ref_lower:
                issues += 1
            if 'false' in text_lower and 'true' in ref_lower:
                issues += 1

        return max(0.0, 1.0 - issues * 0.15)

    def _detect_hallucination_indicators(self, text: str,
                                          context: Optional[Dict[str, Any]]) -> float:
        """Detect indicators of hallucinated content."""
        indicators = 0

        # Overly specific unsupported claims
        import re

        # Specific numbers without source
        numbers = re.findall(r'\b\d{4,}\b', text)  # Long specific numbers
        if len(numbers) > 3:
            indicators += 1

        # Specific dates without context
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        dates = re.findall(date_pattern, text)
        if dates and (not context or 'dates' not in context):
            indicators += 1

        # Excessive certainty markers
        certainty_markers = {'definitely', 'certainly', 'absolutely',
                           'undoubtedly', 'always', 'never', 'all', 'none'}
        text_words = set(text.lower().split())
        if len(text_words & certainty_markers) > 2:
            indicators += 1

        # Unsupported quotes
        quotes = re.findall(r'"[^"]{20,}"', text)
        if quotes and (not context or 'quotes' not in context):
            indicators += 1

        return max(0.0, 1.0 - indicators * 0.2)

    def _detect_bias(self, text: str) -> float:
        """Detect potential bias indicators."""
        bias_indicators = 0
        text_lower = text.lower()

        # One-sided language
        positive_extremes = {'always', 'best', 'perfect', 'amazing', 'wonderful'}
        negative_extremes = {'worst', 'terrible', 'awful', 'horrible', 'never'}

        text_words = set(text_lower.split())

        positive_count = len(text_words & positive_extremes)
        negative_count = len(text_words & negative_extremes)

        # Strong imbalance suggests bias
        if abs(positive_count - negative_count) > 2:
            bias_indicators += 1

        # Stereotyping language patterns (simplified)
        stereotype_patterns = ['all men', 'all women', 'all people from',
                              'they always', 'they never', 'those people']
        for pattern in stereotype_patterns:
            if pattern in text_lower:
                bias_indicators += 1

        return max(0.0, 1.0 - bias_indicators * 0.2)

    def _detect_toxicity(self, text: str) -> float:
        """Detect toxicity indicators."""
        # Simplified toxicity detection
        text_lower = text.lower()

        # Hostile language indicators (non-exhaustive, safe examples)
        hostile_indicators = {'hate', 'stupid', 'idiot', 'loser', 'disgusting',
                            'pathetic', 'worthless', 'inferior'}

        text_words = set(text_lower.split())
        matches = len(text_words & hostile_indicators)

        # All caps (shouting)
        words = text.split()
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)

        issues = matches + (caps_words // 3)

        return max(0.0, 1.0 - issues * 0.15)

    def _assess_misinformation_risk(self, text: str,
                                     context: Optional[Dict[str, Any]]) -> float:
        """Assess risk of misinformation."""
        risk_indicators = 0
        text_lower = text.lower()

        # Unsupported health/medical claims
        medical_keywords = {'cure', 'treatment', 'vaccine', 'medicine', 'disease'}
        if any(kw in text_lower for kw in medical_keywords):
            if not context or 'medical_context' not in context:
                risk_indicators += 1

        # Financial guarantees
        financial_keywords = {'guaranteed return', 'risk-free', 'get rich',
                            'double your money', 'investment opportunity'}
        if any(kw in text_lower for kw in financial_keywords):
            risk_indicators += 1

        # Conspiracy language
        conspiracy_indicators = {'they don\'t want you to know', 'secret truth',
                                'mainstream media lies', 'cover up', 'hidden agenda'}
        if any(ind in text_lower for ind in conspiracy_indicators):
            risk_indicators += 1

        return max(0.0, 1.0 - risk_indicators * 0.25)


class UncertaintyRelevancyAnalyzer(RelevancyDimensionAnalyzer):
    """
    Analyzer for uncertainty quantification in text.

    Evaluates how well text expresses appropriate uncertainty,
    calibration of confidence, and handling of unknowns.
    """

    def __init__(self, weight: float = 1.0):
        """Initialize uncertainty relevancy analyzer."""
        super().__init__("uncertainty_relevancy", weight)
        self.sub_dimensions = [
            'uncertainty_expression',
            'confidence_calibration',
            'hedge_appropriateness',
            'unknown_acknowledgment',
            'probability_language'
        ]

    def analyze(self, text: str, reference: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None) -> RelevancyScore:
        """
        Analyze uncertainty handling in text.

        Args:
            text: Generated text to evaluate
            reference: Reference for expected uncertainty levels
            context: Context indicating uncertainty requirements

        Returns:
            UncertaintyRelevancyScore with calibration results
        """
        sub_scores = {}
        evidence = []

        # Uncertainty expression analysis
        expression_score = self._analyze_uncertainty_expression(text)
        sub_scores['uncertainty_expression'] = expression_score

        # Confidence calibration
        calibration_score = self._analyze_confidence_calibration(text, context)
        sub_scores['confidence_calibration'] = calibration_score

        # Hedge appropriateness
        hedge_score = self._analyze_hedge_appropriateness(text, context)
        sub_scores['hedge_appropriateness'] = hedge_score

        # Unknown acknowledgment
        unknown_score = self._analyze_unknown_acknowledgment(text, context)
        sub_scores['unknown_acknowledgment'] = unknown_score

        # Probability language
        probability_score = self._analyze_probability_language(text)
        sub_scores['probability_language'] = probability_score

        # Aggregate score
        aggregate = np.mean(list(sub_scores.values()))
        confidence = self._compute_confidence(list(sub_scores.values()))

        # Generate evidence
        uncertainty_indicators = self._extract_uncertainty_indicators(text)
        if uncertainty_indicators:
            evidence.append(f"Found {len(uncertainty_indicators)} uncertainty markers")
        if calibration_score < 0.5:
            evidence.append("Potential overconfidence detected")

        return RelevancyScore(
            dimension=self.dimension_name,
            score=aggregate,
            confidence=confidence,
            evidence=evidence,
            sub_scores=sub_scores,
            metadata={
                'analyzer': 'uncertainty',
                'uncertainty_markers': len(uncertainty_indicators)
            }
        )

    def _analyze_uncertainty_expression(self, text: str) -> float:
        """Analyze how uncertainty is expressed."""
        text_lower = text.lower()

        # Good uncertainty expressions
        good_expressions = {'may', 'might', 'could', 'possibly', 'perhaps',
                          'likely', 'unlikely', 'probably', 'approximately',
                          'estimated', 'uncertain', 'unclear'}

        # Overconfident expressions
        overconfident = {'definitely', 'certainly', 'absolutely', 'always',
                        'never', 'guaranteed', 'proven', 'fact'}

        text_words = set(text_lower.split())

        good_count = len(text_words & good_expressions)
        bad_count = len(text_words & overconfident)

        if good_count + bad_count == 0:
            return 0.6  # Neutral

        return good_count / (good_count + bad_count + 1)

    def _analyze_confidence_calibration(self, text: str,
                                         context: Optional[Dict[str, Any]]) -> float:
        """Analyze if confidence level is appropriate."""
        text_lower = text.lower()

        # Detect stated confidence
        high_confidence = {'certain', 'sure', 'confident', 'know', 'clear'}
        low_confidence = {'unsure', 'uncertain', 'unclear', 'unknown', 'doubt'}

        text_words = set(text_lower.split())

        has_high = bool(text_words & high_confidence)
        has_low = bool(text_words & low_confidence)

        if context and 'expected_confidence' in context:
            expected = context['expected_confidence']
            if expected == 'high' and has_high:
                return 0.9
            if expected == 'low' and has_low:
                return 0.9
            if expected == 'high' and has_low:
                return 0.5
            if expected == 'low' and has_high:
                return 0.5

        # Default: mixed confidence is often appropriate
        if has_high and has_low:
            return 0.8
        return 0.6

    def _analyze_hedge_appropriateness(self, text: str,
                                        context: Optional[Dict[str, Any]]) -> float:
        """Analyze appropriateness of hedging language."""
        hedges = ['might', 'may', 'could', 'possibly', 'perhaps', 'seems',
                 'appears', 'suggests', 'indicates', 'tends to']

        text_lower = text.lower()
        hedge_count = sum(1 for h in hedges if h in text_lower)
        word_count = len(text.split())

        if word_count == 0:
            return 0.5

        hedge_ratio = hedge_count / (word_count / 10)  # Per 10 words

        # Moderate hedging is good
        if 0.5 <= hedge_ratio <= 2.0:
            return 0.9
        elif hedge_ratio < 0.5:
            return 0.6  # Too little hedging
        else:
            return 0.5  # Too much hedging

    def _analyze_unknown_acknowledgment(self, text: str,
                                         context: Optional[Dict[str, Any]]) -> float:
        """Analyze acknowledgment of unknowns and limitations."""
        acknowledgment_phrases = [
            'not known', 'unknown', 'unclear', 'uncertain',
            'more research', 'further study', 'limited information',
            'cannot determine', 'difficult to say', 'depends on',
            'varies', 'not enough information'
        ]

        text_lower = text.lower()
        acknowledgments = sum(1 for phrase in acknowledgment_phrases
                             if phrase in text_lower)

        if context and 'has_unknowns' in context and context['has_unknowns']:
            # Should acknowledge unknowns
            if acknowledgments > 0:
                return min(1.0, 0.5 + acknowledgments * 0.2)
            return 0.3  # Should have acknowledged but didn't

        return min(1.0, 0.6 + acknowledgments * 0.1)

    def _analyze_probability_language(self, text: str) -> float:
        """Analyze use of probability language."""
        probability_terms = {
            'percent', '%', 'probability', 'likelihood', 'chance',
            'odds', 'rate', 'frequency', 'proportion', 'ratio'
        }

        quantified_hedges = {
            'very likely', 'somewhat likely', 'highly probable',
            'low probability', 'high chance', 'small chance'
        }

        text_lower = text.lower()

        # Check for probability terms
        has_prob_terms = any(term in text_lower for term in probability_terms)
        has_quantified = any(hedge in text_lower for hedge in quantified_hedges)

        if has_prob_terms and has_quantified:
            return 0.9  # Good probability language
        elif has_prob_terms or has_quantified:
            return 0.7
        else:
            return 0.5

    def _extract_uncertainty_indicators(self, text: str) -> List[str]:
        """Extract all uncertainty indicators from text."""
        indicators = []
        text_lower = text.lower()

        uncertainty_words = ['may', 'might', 'could', 'possibly', 'perhaps',
                           'likely', 'unlikely', 'probably', 'uncertain',
                           'unclear', 'unknown', 'estimated', 'approximately']

        for word in uncertainty_words:
            if word in text_lower:
                indicators.append(word)

        return indicators


class TemporalRelevancyAnalyzer(RelevancyDimensionAnalyzer):
    """
    Analyzer for temporal relevancy.

    Evaluates time-sensitive aspects of text including
    currency, temporal consistency, and time-aware statements.
    """

    def __init__(self, weight: float = 1.0):
        """Initialize temporal relevancy analyzer."""
        super().__init__("temporal_relevancy", weight)
        self.sub_dimensions = [
            'temporal_currency',
            'time_consistency',
            'temporal_markers',
            'historical_accuracy',
            'future_appropriateness'
        ]

    def analyze(self, text: str, reference: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None) -> RelevancyScore:
        """
        Analyze temporal relevancy of text.

        Args:
            text: Generated text to evaluate
            reference: Reference with temporal information
            context: Context with current date and time requirements

        Returns:
            TemporalRelevancyScore with time-based analysis
        """
        sub_scores = {}
        evidence = []

        # Temporal currency
        currency_score = self._analyze_temporal_currency(text, context)
        sub_scores['temporal_currency'] = currency_score

        # Time consistency
        consistency_score = self._analyze_time_consistency(text)
        sub_scores['time_consistency'] = consistency_score

        # Temporal markers
        markers_score = self._analyze_temporal_markers(text, context)
        sub_scores['temporal_markers'] = markers_score

        # Historical accuracy
        historical_score = self._analyze_historical_accuracy(text, context)
        sub_scores['historical_accuracy'] = historical_score

        # Future appropriateness
        future_score = self._analyze_future_appropriateness(text, context)
        sub_scores['future_appropriateness'] = future_score

        # Aggregate score
        aggregate = np.mean(list(sub_scores.values()))
        confidence = self._compute_confidence(list(sub_scores.values()))

        # Generate evidence
        temporal_refs = self._extract_temporal_references(text)
        if temporal_refs:
            evidence.append(f"Found {len(temporal_refs)} temporal references")
        if currency_score < 0.5:
            evidence.append("Potential outdated information")

        return RelevancyScore(
            dimension=self.dimension_name,
            score=aggregate,
            confidence=confidence,
            evidence=evidence,
            sub_scores=sub_scores,
            metadata={
                'analyzer': 'temporal',
                'temporal_references': len(temporal_refs)
            }
        )

    def _analyze_temporal_currency(self, text: str,
                                    context: Optional[Dict[str, Any]]) -> float:
        """Analyze if information is current."""
        import re

        # Extract years
        years = [int(y) for y in re.findall(r'\b(19|20)\d{2}\b', text)]

        if not years:
            return 0.6  # No temporal references

        current_year = 2024
        if context and 'current_year' in context:
            current_year = context['current_year']

        # Check recency
        max_year = max(years)
        min_year = min(years)

        if max_year > current_year:
            return 0.4  # Future dates (potentially problematic)

        age = current_year - max_year
        if age <= 1:
            return 0.9  # Very recent
        elif age <= 3:
            return 0.7  # Recent
        elif age <= 5:
            return 0.5  # Moderately old
        else:
            return 0.3  # Potentially outdated

    def _analyze_time_consistency(self, text: str) -> float:
        """Analyze internal temporal consistency."""
        # Detect tense mixing
        past_markers = {'was', 'were', 'had', 'did', 'went', 'said', 'made'}
        present_markers = {'is', 'are', 'has', 'does', 'goes', 'says', 'makes'}
        future_markers = {'will', 'shall', 'going to', 'would', 'could'}

        text_words = set(text.lower().split())

        has_past = bool(text_words & past_markers)
        has_present = bool(text_words & present_markers)
        has_future = bool(text_words & future_markers)

        tense_count = sum([has_past, has_present, has_future])

        if tense_count <= 1:
            return 0.9  # Consistent tense
        elif tense_count == 2:
            return 0.7  # Some mixing (may be appropriate)
        else:
            return 0.5  # Heavy mixing

    def _analyze_temporal_markers(self, text: str,
                                   context: Optional[Dict[str, Any]]) -> float:
        """Analyze use of temporal markers."""
        temporal_words = {
            'today', 'yesterday', 'tomorrow', 'now', 'then',
            'recently', 'currently', 'previously', 'soon',
            'before', 'after', 'during', 'since', 'until'
        }

        text_words = set(text.lower().split())
        marker_count = len(text_words & temporal_words)
        word_count = len(text.split())

        if word_count == 0:
            return 0.5

        marker_ratio = marker_count / (word_count / 20)  # Per 20 words

        if context and 'needs_temporal_context' in context:
            if context['needs_temporal_context'] and marker_count == 0:
                return 0.3  # Should have temporal markers
            elif context['needs_temporal_context'] and marker_count > 0:
                return 0.9

        # Moderate use is good
        if 0.5 <= marker_ratio <= 2.0:
            return 0.8
        return 0.6

    def _analyze_historical_accuracy(self, text: str,
                                      context: Optional[Dict[str, Any]]) -> float:
        """Analyze historical accuracy of statements."""
        if not context or 'historical_facts' not in context:
            return 0.6

        historical_facts = context['historical_facts']
        text_lower = text.lower()

        verified = 0
        total_claims = 0

        for fact_key, fact_value in historical_facts.items():
            if fact_key.lower() in text_lower:
                total_claims += 1
                if str(fact_value).lower() in text_lower:
                    verified += 1

        if total_claims == 0:
            return 0.6

        return verified / total_claims

    def _analyze_future_appropriateness(self, text: str,
                                         context: Optional[Dict[str, Any]]) -> float:
        """Analyze appropriateness of future-oriented statements."""
        future_patterns = ['will be', 'will have', 'going to', 'expected to',
                         'predicted to', 'projected', 'forecast']

        text_lower = text.lower()
        future_count = sum(1 for p in future_patterns if p in text_lower)

        if future_count == 0:
            return 0.7  # No future claims

        # Check for hedging with future claims
        hedged_future = ['may be', 'might be', 'could be', 'possibly will',
                        'likely to', 'expected to']
        hedged_count = sum(1 for h in hedged_future if h in text_lower)

        if hedged_count >= future_count * 0.5:
            return 0.9  # Appropriately hedged
        else:
            return 0.5  # Unhedged future claims

    def _extract_temporal_references(self, text: str) -> List[str]:
        """Extract all temporal references from text."""
        import re

        references = []

        # Years
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        references.extend(years)

        # Dates
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
        references.extend(dates)

        # Temporal words
        temporal_words = {'today', 'yesterday', 'tomorrow', 'now', 'then',
                         'recently', 'currently', 'previously'}
        text_words = set(text.lower().split())
        references.extend(list(text_words & temporal_words))

        return references


class ComprehensiveRelevancyScorer:
    """
    Comprehensive relevancy scorer combining all 27 dimensions.

    Provides unified scoring across semantic, factual, contextual,
    negative, uncertainty, and temporal relevancy dimensions.
    """

    # Define all 27 relevancy dimensions
    DIMENSION_CATEGORIES = {
        'semantic': [
            'meaning_preservation', 'concept_alignment', 'entity_consistency',
            'relationship_preservation', 'semantic_completeness'
        ],
        'factual': [
            'claim_accuracy', 'fact_consistency', 'source_alignment',
            'temporal_accuracy', 'numerical_precision'
        ],
        'contextual': [
            'context_adherence', 'topic_consistency', 'tone_appropriateness',
            'audience_fit', 'situational_relevance'
        ],
        'negative': [
            'contradiction_detection', 'hallucination_indicators',
            'bias_detection', 'toxicity_indicators', 'misinformation_risk'
        ],
        'uncertainty': [
            'uncertainty_expression', 'confidence_calibration',
            'hedge_appropriateness', 'unknown_acknowledgment', 'probability_language'
        ],
        'temporal': [
            'temporal_currency', 'time_consistency'
        ]
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize comprehensive relevancy scorer.

        Args:
            weights: Optional custom weights for each dimension category
        """
        self.weights = weights or {
            'semantic': 1.0,
            'factual': 1.2,
            'contextual': 0.9,
            'negative': 1.5,
            'uncertainty': 0.8,
            'temporal': 0.7
        }

        # Initialize analyzers
        self.analyzers = {
            'semantic': SemanticRelevancyAnalyzer(self.weights['semantic']),
            'factual': FactualRelevancyAnalyzer(self.weights['factual']),
            'contextual': ContextualRelevancyAnalyzer(self.weights['contextual']),
            'negative': NegativeRelevancyAnalyzer(self.weights['negative']),
            'uncertainty': UncertaintyRelevancyAnalyzer(self.weights['uncertainty']),
            'temporal': TemporalRelevancyAnalyzer(self.weights['temporal'])
        }

    def score(self, text: str, reference: Optional[str] = None,
              context: Optional[Dict[str, Any]] = None) -> RelevancyProfile:
        """
        Generate comprehensive relevancy profile.

        Args:
            text: Text to evaluate
            reference: Optional reference text
            context: Optional context information

        Returns:
            Complete RelevancyProfile with all dimension scores
        """
        profile = RelevancyProfile(text=text, reference=reference)

        all_scores = []
        weighted_sum = 0.0
        weight_sum = 0.0

        for category, analyzer in self.analyzers.items():
            score = analyzer.analyze(text, reference, context)
            profile.scores[category] = score
            all_scores.append(score.score)

            weighted_sum += score.score * analyzer.weight
            weight_sum += analyzer.weight

            # Collect negative flags
            if category == 'negative' and score.score < 0.7:
                profile.negative_flags.extend(score.evidence)

            # Collect uncertainty indicators
            if category == 'uncertainty':
                if 'uncertainty_markers' in score.metadata:
                    profile.uncertainty_indicators.extend(
                        score.evidence[:score.metadata['uncertainty_markers']])

        # Calculate aggregate scores
        profile.aggregate_score = np.mean(all_scores)
        profile.weighted_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0

        return profile

    def score_batch(self, texts: List[str],
                    references: Optional[List[str]] = None,
                    context: Optional[Dict[str, Any]] = None) -> List[RelevancyProfile]:
        """
        Score multiple texts.

        Args:
            texts: List of texts to evaluate
            references: Optional list of reference texts
            context: Shared context for all evaluations

        Returns:
            List of RelevancyProfiles
        """
        profiles = []

        for i, text in enumerate(texts):
            ref = references[i] if references and i < len(references) else None
            profile = self.score(text, ref, context)
            profiles.append(profile)

        return profiles

    def get_dimension_breakdown(self, profile: RelevancyProfile) -> Dict[str, Dict[str, float]]:
        """
        Get detailed breakdown of all 27 dimensions.

        Args:
            profile: RelevancyProfile to analyze

        Returns:
            Nested dictionary of category -> dimension -> score
        """
        breakdown = {}

        for category, score in profile.scores.items():
            breakdown[category] = score.sub_scores.copy()
            breakdown[category]['_aggregate'] = score.score

        return breakdown

    def compare_profiles(self, profile1: RelevancyProfile,
                         profile2: RelevancyProfile) -> Dict[str, Any]:
        """
        Compare two relevancy profiles.

        Args:
            profile1: First profile
            profile2: Second profile

        Returns:
            Comparison results with differences and statistics
        """
        comparison = {
            'aggregate_diff': profile1.aggregate_score - profile2.aggregate_score,
            'weighted_diff': profile1.weighted_score - profile2.weighted_score,
            'category_diffs': {},
            'winner': None
        }

        for category in self.analyzers.keys():
            if category in profile1.scores and category in profile2.scores:
                diff = profile1.scores[category].score - profile2.scores[category].score
                comparison['category_diffs'][category] = diff

        # Determine winner
        if comparison['weighted_diff'] > 0.05:
            comparison['winner'] = 'profile1'
        elif comparison['weighted_diff'] < -0.05:
            comparison['winner'] = 'profile2'
        else:
            comparison['winner'] = 'tie'

        return comparison

    def generate_report(self, profile: RelevancyProfile) -> str:
        """
        Generate human-readable relevancy report.

        Args:
            profile: RelevancyProfile to report on

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "COMPREHENSIVE RELEVANCY REPORT",
            "=" * 60,
            f"Aggregate Score: {profile.aggregate_score:.3f}",
            f"Weighted Score: {profile.weighted_score:.3f}",
            "",
            "DIMENSION SCORES:",
            "-" * 40
        ]

        for category, score in profile.scores.items():
            lines.append(f"\n{category.upper()} ({score.score:.3f}):")
            for sub_dim, sub_score in score.sub_scores.items():
                lines.append(f"  - {sub_dim}: {sub_score:.3f}")

            if score.evidence:
                lines.append("  Evidence:")
                for ev in score.evidence:
                    lines.append(f"    * {ev}")

        if profile.negative_flags:
            lines.extend([
                "",
                "NEGATIVE FLAGS:",
                "-" * 40
            ])
            for flag in profile.negative_flags:
                lines.append(f"  ! {flag}")

        if profile.uncertainty_indicators:
            lines.extend([
                "",
                "UNCERTAINTY INDICATORS:",
                "-" * 40
            ])
            for indicator in profile.uncertainty_indicators:
                lines.append(f"  ? {indicator}")

        lines.append("=" * 60)

        return "\n".join(lines)


# Convenience functions
def analyze_relevancy(text: str, reference: Optional[str] = None,
                      context: Optional[Dict[str, Any]] = None) -> RelevancyProfile:
    """
    Convenience function for quick relevancy analysis.

    Args:
        text: Text to analyze
        reference: Optional reference text
        context: Optional context

    Returns:
        Complete RelevancyProfile
    """
    scorer = ComprehensiveRelevancyScorer()
    return scorer.score(text, reference, context)


def compare_texts_relevancy(text1: str, text2: str,
                            reference: Optional[str] = None,
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compare relevancy of two texts.

    Args:
        text1: First text
        text2: Second text
        reference: Optional reference for both
        context: Optional context for both

    Returns:
        Comparison results
    """
    scorer = ComprehensiveRelevancyScorer()
    profile1 = scorer.score(text1, reference, context)
    profile2 = scorer.score(text2, reference, context)
    return scorer.compare_profiles(profile1, profile2)


__all__ = [
    # Enums
    'RelevancyType',
    'RelevancyLevel',

    # Data classes
    'RelevancyScore',
    'RelevancyProfile',

    # Analyzers
    'RelevancyDimensionAnalyzer',
    'SemanticRelevancyAnalyzer',
    'FactualRelevancyAnalyzer',
    'ContextualRelevancyAnalyzer',
    'NegativeRelevancyAnalyzer',
    'UncertaintyRelevancyAnalyzer',
    'TemporalRelevancyAnalyzer',

    # Comprehensive Scorer
    'ComprehensiveRelevancyScorer',

    # Convenience functions
    'analyze_relevancy',
    'compare_texts_relevancy',
]
