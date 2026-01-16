"""
Diversity and Creativity Analysis Module for GenAI Evaluation.

This module provides comprehensive frameworks for analyzing text diversity,
creativity, and novelty in AI-generated content. Includes metrics for
lexical diversity, semantic diversity, structural variety, and creative
expression analysis.

Classes:
    LexicalDiversityAnalyzer: Token-level diversity metrics (TTR, MTLD, etc.)
    SemanticDiversityAnalyzer: Meaning-level diversity analysis
    StructuralDiversityAnalyzer: Syntactic and structural variety
    CreativityAnalyzer: Creative expression and novelty metrics
    NoveltyAnalyzer: Novel content and idea detection
    SelfBLEUAnalyzer: Self-BLEU for diversity measurement
    DistinctNAnalyzer: Distinct-N metrics for n-gram diversity
    ComprehensiveDiversityEvaluator: Unified diversity evaluation

Author: AgenticFinder Research Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Counter
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict, Counter as CounterClass
import re
import math


class DiversityType(Enum):
    """Types of diversity measures."""
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    TOPICAL = "topical"
    STYLISTIC = "stylistic"


class CreativityLevel(Enum):
    """Levels of creative expression."""
    HIGHLY_CREATIVE = 5
    CREATIVE = 4
    MODERATELY_CREATIVE = 3
    CONVENTIONAL = 2
    REPETITIVE = 1


class NoveltyType(Enum):
    """Types of novelty in content."""
    LEXICAL_NOVELTY = "lexical"
    CONCEPTUAL_NOVELTY = "conceptual"
    STRUCTURAL_NOVELTY = "structural"
    STYLISTIC_NOVELTY = "stylistic"


@dataclass
class DiversityScore:
    """Container for diversity scoring results."""
    diversity_type: DiversityType
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'diversity_type': self.diversity_type.value,
            'score': self.score,
            'metrics': self.metrics,
            'details': self.details
        }


@dataclass
class CreativityScore:
    """Container for creativity analysis results."""
    level: CreativityLevel
    score: float
    dimensions: Dict[str, float] = field(default_factory=dict)
    creative_elements: List[str] = field(default_factory=list)
    conventional_elements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'level': self.level.name,
            'score': self.score,
            'dimensions': self.dimensions,
            'creative_elements': self.creative_elements,
            'conventional_elements': self.conventional_elements
        }


@dataclass
class NoveltyScore:
    """Container for novelty analysis results."""
    overall_novelty: float
    novelty_types: Dict[NoveltyType, float] = field(default_factory=dict)
    novel_elements: List[str] = field(default_factory=list)
    reference_overlap: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'overall_novelty': self.overall_novelty,
            'novelty_types': {k.value: v for k, v in self.novelty_types.items()},
            'novel_elements': self.novel_elements,
            'reference_overlap': self.reference_overlap
        }


@dataclass
class DiversityProfile:
    """Comprehensive diversity profile."""
    text: str
    lexical_diversity: DiversityScore
    semantic_diversity: DiversityScore
    structural_diversity: DiversityScore
    creativity: CreativityScore
    novelty: NoveltyScore
    overall_score: float = 0.0

    def get_summary(self) -> Dict[str, float]:
        """Get score summary."""
        return {
            'lexical': self.lexical_diversity.score,
            'semantic': self.semantic_diversity.score,
            'structural': self.structural_diversity.score,
            'creativity': self.creativity.score,
            'novelty': self.novelty.overall_novelty,
            'overall': self.overall_score
        }


class LexicalDiversityAnalyzer:
    """
    Analyzer for lexical diversity metrics.

    Calculates various token-level diversity measures including
    TTR, MTLD, HD-D, and vocabulary richness metrics.
    """

    def __init__(self):
        """Initialize lexical diversity analyzer."""
        self.metrics = ['ttr', 'rttr', 'cttr', 'mtld', 'hdd', 'maas', 'vocabulary_richness']

    def analyze(self, text: str,
                context: Optional[Dict[str, Any]] = None) -> DiversityScore:
        """
        Analyze lexical diversity of text.

        Args:
            text: Text to analyze
            context: Optional analysis context

        Returns:
            DiversityScore with lexical metrics
        """
        tokens = self._tokenize(text)

        if not tokens:
            return DiversityScore(
                diversity_type=DiversityType.LEXICAL,
                score=0.0,
                metrics={},
                details={'error': 'No tokens found'}
            )

        metrics = {}

        # Type-Token Ratio (TTR)
        metrics['ttr'] = self._calculate_ttr(tokens)

        # Root TTR
        metrics['rttr'] = self._calculate_rttr(tokens)

        # Corrected TTR
        metrics['cttr'] = self._calculate_cttr(tokens)

        # MTLD (Measure of Textual Lexical Diversity)
        metrics['mtld'] = self._calculate_mtld(tokens)

        # HD-D (Hypergeometric Distribution D)
        metrics['hdd'] = self._calculate_hdd(tokens)

        # Maas index
        metrics['maas'] = self._calculate_maas(tokens)

        # Vocabulary richness
        metrics['vocabulary_richness'] = self._calculate_vocabulary_richness(tokens)

        # Hapax legomena ratio
        metrics['hapax_ratio'] = self._calculate_hapax_ratio(tokens)

        # Calculate overall score
        overall_score = np.mean([
            metrics['ttr'],
            metrics['mtld'] / 100,  # Normalize MTLD
            metrics['vocabulary_richness']
        ])

        return DiversityScore(
            diversity_type=DiversityType.LEXICAL,
            score=min(1.0, overall_score),
            metrics=metrics,
            details={
                'token_count': len(tokens),
                'type_count': len(set(tokens)),
                'avg_word_length': np.mean([len(t) for t in tokens])
            }
        )

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r'\b\w+\b', text.lower())

    def _calculate_ttr(self, tokens: List[str]) -> float:
        """Calculate Type-Token Ratio."""
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def _calculate_rttr(self, tokens: List[str]) -> float:
        """Calculate Root Type-Token Ratio."""
        if not tokens:
            return 0.0
        return len(set(tokens)) / np.sqrt(len(tokens))

    def _calculate_cttr(self, tokens: List[str]) -> float:
        """Calculate Corrected Type-Token Ratio."""
        if not tokens:
            return 0.0
        return len(set(tokens)) / np.sqrt(2 * len(tokens))

    def _calculate_mtld(self, tokens: List[str], threshold: float = 0.72) -> float:
        """
        Calculate MTLD (Measure of Textual Lexical Diversity).

        MTLD calculates the average length of sequential word strings
        that maintain a TTR above a threshold.
        """
        if len(tokens) < 10:
            return len(tokens)

        def mtld_forward(tokens):
            factors = 0
            current_ttr = 1.0
            token_count = 0
            type_set = set()

            for token in tokens:
                type_set.add(token)
                token_count += 1
                current_ttr = len(type_set) / token_count

                if current_ttr <= threshold:
                    factors += 1
                    token_count = 0
                    type_set = set()
                    current_ttr = 1.0

            # Handle remainder
            if token_count > 0:
                factors += (1 - current_ttr) / (1 - threshold)

            return len(tokens) / factors if factors > 0 else len(tokens)

        # Calculate forward and backward
        forward = mtld_forward(tokens)
        backward = mtld_forward(tokens[::-1])

        return (forward + backward) / 2

    def _calculate_hdd(self, tokens: List[str], sample_size: int = 42) -> float:
        """
        Calculate HD-D (Hypergeometric Distribution D).

        Uses hypergeometric distribution to estimate vocabulary diversity.
        """
        if len(tokens) < sample_size:
            return self._calculate_ttr(tokens)

        type_counts = CounterClass(tokens)
        types = len(type_counts)
        n = len(tokens)

        if types == 0 or n == 0:
            return 0.0

        hdd = 0.0
        for count in type_counts.values():
            # Contribution of each type
            contrib = 1 - (
                self._comb(n - count, sample_size) /
                self._comb(n, sample_size)
            ) if self._comb(n, sample_size) > 0 else 0
            hdd += contrib

        return hdd / sample_size

    def _comb(self, n: int, k: int) -> float:
        """Calculate combination (n choose k)."""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result

    def _calculate_maas(self, tokens: List[str]) -> float:
        """Calculate Maas index (inverse measure - lower is more diverse)."""
        n = len(tokens)
        v = len(set(tokens))

        if n <= 1 or v <= 1:
            return 0.0

        log_n = np.log(n)
        log_v = np.log(v)

        if log_n == 0:
            return 0.0

        # Maas index (a^2)
        a2 = (log_n - log_v) / (log_n ** 2)
        # Return inverted for consistency (higher = more diverse)
        return 1 - min(1.0, a2 * 10)

    def _calculate_vocabulary_richness(self, tokens: List[str]) -> float:
        """Calculate vocabulary richness score."""
        if not tokens:
            return 0.0

        type_counts = CounterClass(tokens)

        # Yule's K (modified for higher = richer)
        m1 = len(tokens)
        m2 = sum(count ** 2 for count in type_counts.values())

        if m1 == 0 or m1 == m2:
            return 0.0

        k = 10000 * (m2 - m1) / (m1 ** 2)
        # Invert and normalize
        return 1 - min(1.0, k / 200)

    def _calculate_hapax_ratio(self, tokens: List[str]) -> float:
        """Calculate ratio of words appearing only once (hapax legomena)."""
        if not tokens:
            return 0.0

        type_counts = CounterClass(tokens)
        hapax = sum(1 for count in type_counts.values() if count == 1)

        return hapax / len(set(tokens)) if len(set(tokens)) > 0 else 0.0


class SemanticDiversityAnalyzer:
    """
    Analyzer for semantic diversity.

    Evaluates diversity at the meaning level, including
    topic coverage, concept variety, and semantic spread.
    """

    def __init__(self):
        """Initialize semantic diversity analyzer."""
        self.topic_keywords = {
            'technology': {'computer', 'software', 'digital', 'data', 'algorithm', 'system'},
            'science': {'research', 'study', 'experiment', 'theory', 'analysis', 'scientific'},
            'business': {'market', 'company', 'revenue', 'growth', 'strategy', 'customer'},
            'health': {'medical', 'health', 'treatment', 'patient', 'disease', 'care'},
            'education': {'learning', 'student', 'education', 'teaching', 'school', 'knowledge'}
        }

    def analyze(self, text: str,
                context: Optional[Dict[str, Any]] = None) -> DiversityScore:
        """
        Analyze semantic diversity of text.

        Args:
            text: Text to analyze
            context: Optional analysis context

        Returns:
            DiversityScore with semantic metrics
        """
        tokens = set(text.lower().split())
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        metrics = {}

        # Topic coverage
        metrics['topic_coverage'] = self._calculate_topic_coverage(tokens)

        # Concept variety
        metrics['concept_variety'] = self._calculate_concept_variety(text)

        # Semantic spread
        metrics['semantic_spread'] = self._calculate_semantic_spread(sentences)

        # Entity diversity
        metrics['entity_diversity'] = self._calculate_entity_diversity(text)

        # Predicate diversity
        metrics['predicate_diversity'] = self._calculate_predicate_diversity(text)

        # Calculate overall score
        overall_score = np.mean(list(metrics.values()))

        return DiversityScore(
            diversity_type=DiversityType.SEMANTIC,
            score=overall_score,
            metrics=metrics,
            details={
                'topics_covered': self._get_covered_topics(tokens),
                'sentence_count': len(sentences)
            }
        )

    def _calculate_topic_coverage(self, tokens: Set[str]) -> float:
        """Calculate topic coverage across domains."""
        covered_topics = 0
        for topic, keywords in self.topic_keywords.items():
            if tokens & keywords:
                covered_topics += 1

        return covered_topics / len(self.topic_keywords)

    def _get_covered_topics(self, tokens: Set[str]) -> List[str]:
        """Get list of covered topics."""
        covered = []
        for topic, keywords in self.topic_keywords.items():
            if tokens & keywords:
                covered.append(topic)
        return covered

    def _calculate_concept_variety(self, text: str) -> float:
        """Calculate variety of concepts mentioned."""
        # Extract potential concepts (nouns - simplified using capitalized words and long words)
        words = text.split()
        concepts = set()

        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) > 5 or (clean_word and clean_word[0].isupper()):
                concepts.add(clean_word.lower())

        # Normalize by text length
        word_count = len(words)
        if word_count == 0:
            return 0.0

        concept_ratio = len(concepts) / (word_count / 10)  # Per 10 words
        return min(1.0, concept_ratio)

    def _calculate_semantic_spread(self, sentences: List[str]) -> float:
        """Calculate semantic spread across sentences."""
        if len(sentences) < 2:
            return 0.5

        # Calculate pairwise dissimilarity
        dissimilarities = []
        for i, sent1 in enumerate(sentences):
            words1 = set(sent1.lower().split())
            for sent2 in sentences[i+1:]:
                words2 = set(sent2.lower().split())
                if words1 and words2:
                    overlap = len(words1 & words2) / min(len(words1), len(words2))
                    dissimilarities.append(1 - overlap)

        return np.mean(dissimilarities) if dissimilarities else 0.5

    def _calculate_entity_diversity(self, text: str) -> float:
        """Calculate diversity of named entities."""
        # Extract named entities (simplified: capitalized multi-word phrases)
        words = text.split()
        entities = set()

        current_entity = []
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper():
                current_entity.append(clean_word)
            else:
                if current_entity:
                    entities.add(' '.join(current_entity))
                current_entity = []

        if current_entity:
            entities.add(' '.join(current_entity))

        # Normalize
        word_count = len(words)
        if word_count == 0:
            return 0.0

        return min(1.0, len(entities) / (word_count / 20))

    def _calculate_predicate_diversity(self, text: str) -> float:
        """Calculate diversity of predicates/actions."""
        # Common verbs to identify
        common_verbs = {
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'has', 'have', 'had', 'do', 'does', 'did',
            'can', 'could', 'will', 'would', 'should', 'may', 'might'
        }

        words = text.lower().split()
        action_verbs = set()

        # Simple heuristic: words ending in common verb suffixes
        verb_suffixes = ('ed', 'ing', 'es', 'ize', 'ify', 'ate')

        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word not in common_verbs:
                if clean_word.endswith(verb_suffixes) or clean_word in common_verbs:
                    action_verbs.add(clean_word)

        word_count = len(words)
        if word_count == 0:
            return 0.0

        return min(1.0, len(action_verbs) / (word_count / 15))


class StructuralDiversityAnalyzer:
    """
    Analyzer for structural diversity.

    Evaluates syntactic variety, sentence structure diversity,
    and discourse pattern variation.
    """

    def __init__(self):
        """Initialize structural diversity analyzer."""
        pass

    def analyze(self, text: str,
                context: Optional[Dict[str, Any]] = None) -> DiversityScore:
        """
        Analyze structural diversity of text.

        Args:
            text: Text to analyze
            context: Optional analysis context

        Returns:
            DiversityScore with structural metrics
        """
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if not sentences:
            return DiversityScore(
                diversity_type=DiversityType.STRUCTURAL,
                score=0.0,
                metrics={},
                details={'error': 'No sentences found'}
            )

        metrics = {}

        # Sentence length diversity
        metrics['sentence_length_diversity'] = self._calculate_length_diversity(sentences)

        # Sentence structure diversity
        metrics['structure_diversity'] = self._calculate_structure_diversity(sentences)

        # Punctuation diversity
        metrics['punctuation_diversity'] = self._calculate_punctuation_diversity(text)

        # Opening word diversity
        metrics['opening_diversity'] = self._calculate_opening_diversity(sentences)

        # Clause complexity diversity
        metrics['complexity_diversity'] = self._calculate_complexity_diversity(sentences)

        # Paragraph structure (if applicable)
        metrics['paragraph_diversity'] = self._calculate_paragraph_diversity(text)

        # Calculate overall score
        overall_score = np.mean(list(metrics.values()))

        return DiversityScore(
            diversity_type=DiversityType.STRUCTURAL,
            score=overall_score,
            metrics=metrics,
            details={
                'sentence_count': len(sentences),
                'avg_sentence_length': np.mean([len(s.split()) for s in sentences])
            }
        )

    def _calculate_length_diversity(self, sentences: List[str]) -> float:
        """Calculate diversity in sentence lengths."""
        if len(sentences) < 2:
            return 0.5

        lengths = [len(s.split()) for s in sentences]
        std = np.std(lengths)
        mean = np.mean(lengths)

        if mean == 0:
            return 0.0

        # Coefficient of variation
        cv = std / mean
        return min(1.0, cv)

    def _calculate_structure_diversity(self, sentences: List[str]) -> float:
        """Calculate diversity in sentence structures."""
        if not sentences:
            return 0.0

        structures = []
        for sent in sentences:
            # Simplified structure: (first word type, length category)
            words = sent.split()
            if not words:
                continue

            first_word = words[0].lower()
            length_cat = 'short' if len(words) < 8 else 'medium' if len(words) < 15 else 'long'

            # Classify sentence type
            if first_word in {'what', 'who', 'where', 'when', 'why', 'how', 'which'}:
                sent_type = 'question'
            elif first_word in {'the', 'a', 'an', 'this', 'that', 'these', 'those'}:
                sent_type = 'declarative_det'
            elif first_word in {'i', 'we', 'you', 'he', 'she', 'it', 'they'}:
                sent_type = 'declarative_pron'
            else:
                sent_type = 'other'

            structures.append((sent_type, length_cat))

        unique_structures = len(set(structures))
        return unique_structures / len(sentences) if sentences else 0.0

    def _calculate_punctuation_diversity(self, text: str) -> float:
        """Calculate diversity in punctuation usage."""
        punctuation_marks = {'.': 0, ',': 0, ';': 0, ':': 0, '!': 0,
                           '?': 0, '-': 0, '(': 0, '"': 0}

        for char in text:
            if char in punctuation_marks:
                punctuation_marks[char] += 1

        used_marks = sum(1 for v in punctuation_marks.values() if v > 0)
        return used_marks / len(punctuation_marks)

    def _calculate_opening_diversity(self, sentences: List[str]) -> float:
        """Calculate diversity in sentence openings."""
        openings = []
        for sent in sentences:
            words = sent.split()[:3]  # First 3 words
            if words:
                opening = ' '.join(words).lower()
                openings.append(opening)

        if not openings:
            return 0.0

        unique_openings = len(set(openings))
        return unique_openings / len(openings)

    def _calculate_complexity_diversity(self, sentences: List[str]) -> float:
        """Calculate diversity in clause complexity."""
        complexities = []

        clause_markers = {',', ';', ':', 'and', 'but', 'or', 'while', 'because',
                         'although', 'when', 'if', 'since', 'that', 'which'}

        for sent in sentences:
            words = sent.lower().split()
            marker_count = sum(1 for w in words if w.strip('.,;:') in clause_markers)
            # Complexity level: simple (0), compound (1-2), complex (3+)
            if marker_count == 0:
                complexities.append('simple')
            elif marker_count <= 2:
                complexities.append('compound')
            else:
                complexities.append('complex')

        if not complexities:
            return 0.0

        unique_types = len(set(complexities))
        return unique_types / 3  # Max 3 types

    def _calculate_paragraph_diversity(self, text: str) -> float:
        """Calculate diversity in paragraph structure."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if len(paragraphs) <= 1:
            return 0.5

        # Paragraph lengths
        lengths = [len(p.split()) for p in paragraphs]
        std = np.std(lengths)
        mean = np.mean(lengths)

        if mean == 0:
            return 0.0

        return min(1.0, std / mean)


class CreativityAnalyzer:
    """
    Analyzer for creativity and creative expression.

    Evaluates novelty of expression, use of figurative language,
    unconventional structures, and originality.
    """

    def __init__(self):
        """Initialize creativity analyzer."""
        # Common/cliche phrases to detect
        self.cliche_phrases = {
            'at the end of the day', 'when all is said and done',
            'it goes without saying', 'needless to say',
            'in today\'s world', 'in this day and age',
            'at this point in time', 'first and foremost'
        }

        # Figurative language indicators
        self.figurative_markers = {
            'like': 'simile', 'as if': 'simile', 'as though': 'simile',
            'metaphor': 'metaphor', 'imagine': 'metaphor',
            'literally': 'hyperbole', 'never': 'hyperbole', 'always': 'hyperbole'
        }

    def analyze(self, text: str,
                context: Optional[Dict[str, Any]] = None) -> CreativityScore:
        """
        Analyze creativity of text.

        Args:
            text: Text to analyze
            context: Optional analysis context

        Returns:
            CreativityScore with creativity metrics
        """
        dimensions = {}

        # Originality (inverse of cliche usage)
        dimensions['originality'] = self._calculate_originality(text)

        # Figurative language usage
        dimensions['figurative_language'] = self._calculate_figurative_usage(text)

        # Vocabulary sophistication
        dimensions['vocabulary_sophistication'] = self._calculate_vocabulary_sophistication(text)

        # Structural creativity
        dimensions['structural_creativity'] = self._calculate_structural_creativity(text)

        # Expressiveness
        dimensions['expressiveness'] = self._calculate_expressiveness(text)

        # Unexpectedness
        dimensions['unexpectedness'] = self._calculate_unexpectedness(text)

        # Calculate overall score
        overall_score = np.mean(list(dimensions.values()))
        level = self._score_to_level(overall_score)

        # Identify creative and conventional elements
        creative_elements = self._identify_creative_elements(text)
        conventional_elements = self._identify_conventional_elements(text)

        return CreativityScore(
            level=level,
            score=overall_score,
            dimensions=dimensions,
            creative_elements=creative_elements,
            conventional_elements=conventional_elements
        )

    def _calculate_originality(self, text: str) -> float:
        """Calculate originality score (inverse of cliche usage)."""
        text_lower = text.lower()

        cliche_count = sum(1 for phrase in self.cliche_phrases
                          if phrase in text_lower)

        word_count = len(text.split())
        if word_count == 0:
            return 0.5

        # Penalize for cliches
        cliche_penalty = min(1.0, cliche_count * 0.15)
        return 1.0 - cliche_penalty

    def _calculate_figurative_usage(self, text: str) -> float:
        """Calculate use of figurative language."""
        text_lower = text.lower()

        # Count figurative markers
        figurative_count = 0
        for marker in self.figurative_markers:
            if marker in text_lower:
                figurative_count += 1

        # Also check for less common metaphorical patterns
        metaphor_patterns = [
            r'\w+ is a \w+',  # X is a Y (metaphor)
            r'like a \w+',  # simile
            r'as \w+ as',  # simile
        ]

        for pattern in metaphor_patterns:
            matches = re.findall(pattern, text_lower)
            figurative_count += len(matches)

        word_count = len(text.split())
        if word_count == 0:
            return 0.0

        return min(1.0, figurative_count / (word_count / 50))

    def _calculate_vocabulary_sophistication(self, text: str) -> float:
        """Calculate vocabulary sophistication."""
        words = re.findall(r'\b\w+\b', text.lower())

        if not words:
            return 0.0

        # Measure: average word length and uncommon words
        avg_length = np.mean([len(w) for w in words])

        # Common words (simplified list)
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will',
                       'would', 'could', 'should', 'may', 'might', 'must',
                       'and', 'or', 'but', 'if', 'when', 'where', 'what',
                       'who', 'which', 'that', 'this', 'these', 'those',
                       'it', 'its', 'i', 'you', 'he', 'she', 'we', 'they',
                       'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by'}

        uncommon_count = sum(1 for w in words if w not in common_words and len(w) > 4)
        uncommon_ratio = uncommon_count / len(words)

        # Combine metrics
        length_score = min(1.0, (avg_length - 3) / 5)
        return (length_score + uncommon_ratio) / 2

    def _calculate_structural_creativity(self, text: str) -> float:
        """Calculate creativity in text structure."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if len(sentences) < 2:
            return 0.5

        # Check for varied openings
        openings = [s.split()[0].lower() if s.split() else '' for s in sentences]
        unique_openings = len(set(openings)) / len(openings)

        # Check for varied lengths
        lengths = [len(s.split()) for s in sentences]
        length_std = np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0

        # Check for non-standard structures
        non_standard = 0
        for sent in sentences:
            # Starts with conjunction
            if sent.split() and sent.split()[0].lower() in {'and', 'but', 'or', 'so'}:
                non_standard += 1
            # Very short for effect
            if len(sent.split()) <= 3:
                non_standard += 1
            # Ends with ellipsis pattern
            if '...' in sent:
                non_standard += 1

        non_standard_ratio = non_standard / len(sentences)

        return (unique_openings + min(1.0, length_std) + min(1.0, non_standard_ratio)) / 3

    def _calculate_expressiveness(self, text: str) -> float:
        """Calculate expressiveness of writing."""
        # Emotional/descriptive words
        expressive_words = {
            'amazing', 'beautiful', 'wonderful', 'terrible', 'horrible',
            'fantastic', 'incredible', 'extraordinary', 'remarkable',
            'stunning', 'breathtaking', 'magnificent', 'delightful',
            'passionate', 'vibrant', 'vivid', 'brilliant', 'exquisite'
        }

        text_lower = text.lower()
        words = set(text_lower.split())

        expressive_count = len(words & expressive_words)

        # Also count intensifiers
        intensifiers = {'very', 'extremely', 'incredibly', 'absolutely',
                       'totally', 'completely', 'utterly', 'quite'}
        intensifier_count = len(words & intensifiers)

        # Exclamation marks
        exclamation_count = text.count('!')

        word_count = len(text.split())
        if word_count == 0:
            return 0.0

        expressive_score = (expressive_count + intensifier_count * 0.5 +
                           exclamation_count * 0.3) / (word_count / 20)
        return min(1.0, expressive_score)

    def _calculate_unexpectedness(self, text: str) -> float:
        """Calculate unexpectedness/surprise in text."""
        words = text.lower().split()

        if not words:
            return 0.0

        # Check for unexpected word combinations (simplified)
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]

        # Score based on unique bigrams ratio
        unique_bigrams = len(set(bigrams))
        bigram_uniqueness = unique_bigrams / len(bigrams) if bigrams else 0

        # Check for unusual patterns
        unusual_patterns = [
            r'\?!', r'\.\.\.', r'—', r'"[^"]*"',  # Punctuation patterns
        ]

        unusual_count = 0
        for pattern in unusual_patterns:
            unusual_count += len(re.findall(pattern, text))

        unusual_ratio = unusual_count / (len(words) / 10) if words else 0

        return (bigram_uniqueness + min(1.0, unusual_ratio)) / 2

    def _score_to_level(self, score: float) -> CreativityLevel:
        """Convert score to creativity level."""
        if score >= 0.8:
            return CreativityLevel.HIGHLY_CREATIVE
        elif score >= 0.6:
            return CreativityLevel.CREATIVE
        elif score >= 0.4:
            return CreativityLevel.MODERATELY_CREATIVE
        elif score >= 0.2:
            return CreativityLevel.CONVENTIONAL
        else:
            return CreativityLevel.REPETITIVE

    def _identify_creative_elements(self, text: str) -> List[str]:
        """Identify creative elements in text."""
        elements = []
        text_lower = text.lower()

        # Figurative language
        for marker in ['like a', 'as if', 'as though']:
            if marker in text_lower:
                elements.append(f"Figurative language: '{marker}'")

        # Unusual punctuation
        if '...' in text:
            elements.append("Ellipsis for effect")
        if '!' in text:
            elements.append("Exclamatory expression")

        return elements[:5]  # Limit to 5

    def _identify_conventional_elements(self, text: str) -> List[str]:
        """Identify conventional/cliche elements."""
        elements = []
        text_lower = text.lower()

        for phrase in self.cliche_phrases:
            if phrase in text_lower:
                elements.append(f"Cliche: '{phrase}'")

        return elements[:5]


class NoveltyAnalyzer:
    """
    Analyzer for novelty and originality.

    Compares generated content against reference corpus
    to measure novelty of ideas and expressions.
    """

    def __init__(self, reference_corpus: Optional[List[str]] = None):
        """
        Initialize novelty analyzer.

        Args:
            reference_corpus: Optional reference texts for comparison
        """
        self.reference_corpus = reference_corpus or []
        self._reference_ngrams = self._build_reference_ngrams()

    def _build_reference_ngrams(self) -> Dict[int, Set[str]]:
        """Build n-gram sets from reference corpus."""
        ngrams = {1: set(), 2: set(), 3: set()}

        for text in self.reference_corpus:
            words = text.lower().split()
            for n in [1, 2, 3]:
                for i in range(len(words) - n + 1):
                    ngram = ' '.join(words[i:i+n])
                    ngrams[n].add(ngram)

        return ngrams

    def analyze(self, text: str,
                reference: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None) -> NoveltyScore:
        """
        Analyze novelty of text.

        Args:
            text: Text to analyze
            reference: Optional specific reference text
            context: Optional analysis context

        Returns:
            NoveltyScore with novelty metrics
        """
        novelty_types = {}
        novel_elements = []

        # Lexical novelty
        lexical = self._calculate_lexical_novelty(text, reference)
        novelty_types[NoveltyType.LEXICAL_NOVELTY] = lexical['score']
        novel_elements.extend(lexical['elements'])

        # Conceptual novelty
        conceptual = self._calculate_conceptual_novelty(text, reference)
        novelty_types[NoveltyType.CONCEPTUAL_NOVELTY] = conceptual['score']
        novel_elements.extend(conceptual['elements'])

        # Structural novelty
        structural = self._calculate_structural_novelty(text, reference)
        novelty_types[NoveltyType.STRUCTURAL_NOVELTY] = structural['score']

        # Stylistic novelty
        stylistic = self._calculate_stylistic_novelty(text, reference)
        novelty_types[NoveltyType.STYLISTIC_NOVELTY] = stylistic['score']

        # Calculate overall novelty
        overall_novelty = np.mean(list(novelty_types.values()))

        # Calculate reference overlap
        reference_overlap = self._calculate_reference_overlap(text, reference)

        return NoveltyScore(
            overall_novelty=overall_novelty,
            novelty_types=novelty_types,
            novel_elements=novel_elements[:10],  # Limit elements
            reference_overlap=reference_overlap
        )

    def _calculate_lexical_novelty(self, text: str,
                                    reference: Optional[str]) -> Dict[str, Any]:
        """Calculate lexical novelty."""
        words = set(text.lower().split())

        if reference:
            ref_words = set(reference.lower().split())
            novel_words = words - ref_words
            novelty = len(novel_words) / len(words) if words else 0
        elif self._reference_ngrams[1]:
            novel_words = words - self._reference_ngrams[1]
            novelty = len(novel_words) / len(words) if words else 0
        else:
            novelty = 0.5
            novel_words = set()

        elements = [f"Novel word: '{w}'" for w in list(novel_words)[:5]]
        return {'score': novelty, 'elements': elements}

    def _calculate_conceptual_novelty(self, text: str,
                                       reference: Optional[str]) -> Dict[str, Any]:
        """Calculate conceptual novelty."""
        # Extract concepts (long words, capitalized)
        words = text.split()
        concepts = set()
        for w in words:
            clean = re.sub(r'[^\w]', '', w)
            if len(clean) > 6 or (clean and clean[0].isupper()):
                concepts.add(clean.lower())

        if reference:
            ref_words = reference.split()
            ref_concepts = set()
            for w in ref_words:
                clean = re.sub(r'[^\w]', '', w)
                if len(clean) > 6 or (clean and clean[0].isupper()):
                    ref_concepts.add(clean.lower())

            novel_concepts = concepts - ref_concepts
            novelty = len(novel_concepts) / len(concepts) if concepts else 0
        else:
            novelty = 0.5
            novel_concepts = set()

        elements = [f"Novel concept: '{c}'" for c in list(novel_concepts)[:5]]
        return {'score': novelty, 'elements': elements}

    def _calculate_structural_novelty(self, text: str,
                                       reference: Optional[str]) -> Dict[str, Any]:
        """Calculate structural novelty."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        # Analyze sentence patterns
        patterns = []
        for sent in sentences:
            words = sent.split()
            if not words:
                continue
            # Pattern: (length_category, first_word_type)
            length = 'short' if len(words) < 8 else 'medium' if len(words) < 15 else 'long'
            first = 'pronoun' if words[0].lower() in {'i', 'we', 'you', 'he', 'she', 'it', 'they'} else 'other'
            patterns.append((length, first))

        # Check pattern variety
        if patterns:
            unique_patterns = len(set(patterns)) / len(patterns)
        else:
            unique_patterns = 0.5

        return {'score': unique_patterns, 'elements': []}

    def _calculate_stylistic_novelty(self, text: str,
                                      reference: Optional[str]) -> Dict[str, Any]:
        """Calculate stylistic novelty."""
        # Punctuation patterns
        punct_patterns = {
            '...': text.count('...'),
            '!': text.count('!'),
            '?': text.count('?'),
            ';': text.count(';'),
            ':': text.count(':'),
            '"': text.count('"')
        }

        # Variety of punctuation usage
        used_punct = sum(1 for v in punct_patterns.values() if v > 0)
        punct_diversity = used_punct / len(punct_patterns)

        # Word length variety
        words = text.split()
        if words:
            lengths = [len(w) for w in words]
            length_std = np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0
        else:
            length_std = 0

        novelty = (punct_diversity + min(1.0, length_std)) / 2
        return {'score': novelty, 'elements': []}

    def _calculate_reference_overlap(self, text: str,
                                      reference: Optional[str]) -> float:
        """Calculate overlap with reference."""
        if not reference:
            return 0.0

        text_words = set(text.lower().split())
        ref_words = set(reference.lower().split())

        if not ref_words:
            return 0.0

        overlap = len(text_words & ref_words) / len(ref_words)
        return overlap


class SelfBLEUAnalyzer:
    """
    Self-BLEU analyzer for diversity measurement.

    Calculates Self-BLEU to measure diversity among
    a set of generated texts (lower = more diverse).
    """

    def __init__(self, n_gram_range: Tuple[int, int] = (1, 4)):
        """
        Initialize Self-BLEU analyzer.

        Args:
            n_gram_range: Range of n-grams to consider (min, max)
        """
        self.n_gram_range = n_gram_range

    def calculate(self, texts: List[str]) -> Dict[str, float]:
        """
        Calculate Self-BLEU scores.

        Args:
            texts: List of generated texts

        Returns:
            Dictionary with Self-BLEU scores for each n-gram
        """
        if len(texts) < 2:
            return {f'self_bleu_{n}': 0.0 for n in range(self.n_gram_range[0], self.n_gram_range[1] + 1)}

        results = {}

        for n in range(self.n_gram_range[0], self.n_gram_range[1] + 1):
            bleu_scores = []

            for i, text in enumerate(texts):
                # Use other texts as references
                references = texts[:i] + texts[i+1:]
                bleu = self._calculate_bleu_n(text, references, n)
                bleu_scores.append(bleu)

            results[f'self_bleu_{n}'] = np.mean(bleu_scores)

        # Average Self-BLEU
        results['self_bleu_avg'] = np.mean(list(results.values()))

        # Diversity score (inverse of Self-BLEU)
        results['diversity'] = 1.0 - results['self_bleu_avg']

        return results

    def _calculate_bleu_n(self, hypothesis: str,
                           references: List[str], n: int) -> float:
        """Calculate BLEU-n score."""
        hyp_tokens = hypothesis.lower().split()
        hyp_ngrams = self._get_ngrams(hyp_tokens, n)

        if not hyp_ngrams:
            return 0.0

        # Get reference n-grams
        ref_ngrams = CounterClass()
        for ref in references:
            ref_tokens = ref.lower().split()
            for ngram in self._get_ngrams(ref_tokens, n):
                ref_ngrams[ngram] += 1

        # Count matches
        matches = 0
        for ngram in hyp_ngrams:
            if ngram in ref_ngrams and ref_ngrams[ngram] > 0:
                matches += 1
                ref_ngrams[ngram] -= 1

        return matches / len(hyp_ngrams)

    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Get n-grams from tokens."""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


class DistinctNAnalyzer:
    """
    Distinct-N analyzer for n-gram diversity.

    Calculates Distinct-1, Distinct-2, etc. to measure
    the proportion of unique n-grams in text.
    """

    def __init__(self, max_n: int = 4):
        """
        Initialize Distinct-N analyzer.

        Args:
            max_n: Maximum n for n-gram analysis
        """
        self.max_n = max_n

    def calculate(self, text: str) -> Dict[str, float]:
        """
        Calculate Distinct-N scores for text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with Distinct-N scores
        """
        tokens = text.lower().split()

        if not tokens:
            return {f'distinct_{n}': 0.0 for n in range(1, self.max_n + 1)}

        results = {}

        for n in range(1, self.max_n + 1):
            ngrams = self._get_ngrams(tokens, n)

            if ngrams:
                unique = len(set(ngrams))
                total = len(ngrams)
                results[f'distinct_{n}'] = unique / total
            else:
                results[f'distinct_{n}'] = 0.0

        # Average Distinct
        results['distinct_avg'] = np.mean(list(results.values()))

        return results

    def calculate_batch(self, texts: List[str]) -> Dict[str, float]:
        """
        Calculate Distinct-N across multiple texts.

        Args:
            texts: List of texts

        Returns:
            Aggregate Distinct-N scores
        """
        all_tokens = []
        for text in texts:
            all_tokens.extend(text.lower().split())

        if not all_tokens:
            return {f'distinct_{n}': 0.0 for n in range(1, self.max_n + 1)}

        results = {}

        for n in range(1, self.max_n + 1):
            ngrams = self._get_ngrams(all_tokens, n)

            if ngrams:
                unique = len(set(ngrams))
                total = len(ngrams)
                results[f'distinct_{n}'] = unique / total
            else:
                results[f'distinct_{n}'] = 0.0

        results['distinct_avg'] = np.mean([results[f'distinct_{n}']
                                           for n in range(1, self.max_n + 1)])

        return results

    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Get n-grams from tokens."""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


class ComprehensiveDiversityEvaluator:
    """
    Comprehensive diversity evaluator.

    Combines all diversity metrics for unified analysis
    of text diversity, creativity, and novelty.
    """

    def __init__(self,
                 reference_corpus: Optional[List[str]] = None,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize comprehensive evaluator.

        Args:
            reference_corpus: Optional reference texts
            weights: Optional weights for each dimension
        """
        self.weights = weights or {
            'lexical': 0.25,
            'semantic': 0.25,
            'structural': 0.2,
            'creativity': 0.15,
            'novelty': 0.15
        }

        self.lexical_analyzer = LexicalDiversityAnalyzer()
        self.semantic_analyzer = SemanticDiversityAnalyzer()
        self.structural_analyzer = StructuralDiversityAnalyzer()
        self.creativity_analyzer = CreativityAnalyzer()
        self.novelty_analyzer = NoveltyAnalyzer(reference_corpus)
        self.self_bleu_analyzer = SelfBLEUAnalyzer()
        self.distinct_n_analyzer = DistinctNAnalyzer()

    def evaluate(self, text: str,
                 reference: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None) -> DiversityProfile:
        """
        Perform comprehensive diversity evaluation.

        Args:
            text: Text to evaluate
            reference: Optional reference text
            context: Optional evaluation context

        Returns:
            Comprehensive DiversityProfile
        """
        # Analyze each dimension
        lexical = self.lexical_analyzer.analyze(text, context)
        semantic = self.semantic_analyzer.analyze(text, context)
        structural = self.structural_analyzer.analyze(text, context)
        creativity = self.creativity_analyzer.analyze(text, context)
        novelty = self.novelty_analyzer.analyze(text, reference, context)

        # Calculate overall score
        overall_score = (
            lexical.score * self.weights['lexical'] +
            semantic.score * self.weights['semantic'] +
            structural.score * self.weights['structural'] +
            creativity.score * self.weights['creativity'] +
            novelty.overall_novelty * self.weights['novelty']
        )

        return DiversityProfile(
            text=text,
            lexical_diversity=lexical,
            semantic_diversity=semantic,
            structural_diversity=structural,
            creativity=creativity,
            novelty=novelty,
            overall_score=overall_score
        )

    def evaluate_batch(self, texts: List[str],
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate diversity across multiple texts.

        Args:
            texts: List of texts to evaluate
            context: Optional context

        Returns:
            Batch evaluation results
        """
        # Individual profiles
        profiles = [self.evaluate(text, context=context) for text in texts]

        # Self-BLEU for inter-text diversity
        self_bleu = self.self_bleu_analyzer.calculate(texts)

        # Distinct-N across all texts
        distinct_n = self.distinct_n_analyzer.calculate_batch(texts)

        # Aggregate scores
        avg_scores = {
            'lexical': np.mean([p.lexical_diversity.score for p in profiles]),
            'semantic': np.mean([p.semantic_diversity.score for p in profiles]),
            'structural': np.mean([p.structural_diversity.score for p in profiles]),
            'creativity': np.mean([p.creativity.score for p in profiles]),
            'novelty': np.mean([p.novelty.overall_novelty for p in profiles]),
            'overall': np.mean([p.overall_score for p in profiles])
        }

        return {
            'individual_profiles': profiles,
            'self_bleu': self_bleu,
            'distinct_n': distinct_n,
            'average_scores': avg_scores,
            'inter_text_diversity': self_bleu.get('diversity', 0),
            'text_count': len(texts)
        }

    def compare_texts(self, text1: str, text2: str,
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compare diversity of two texts.

        Args:
            text1: First text
            text2: Second text
            context: Optional context

        Returns:
            Comparison results
        """
        profile1 = self.evaluate(text1, context=context)
        profile2 = self.evaluate(text2, context=context)

        comparison = {
            'text1_overall': profile1.overall_score,
            'text2_overall': profile2.overall_score,
            'difference': profile1.overall_score - profile2.overall_score,
            'winner': 'text1' if profile1.overall_score > profile2.overall_score else 'text2',
            'dimension_comparison': {}
        }

        for dim in ['lexical', 'semantic', 'structural', 'creativity', 'novelty']:
            if dim == 'creativity':
                score1 = profile1.creativity.score
                score2 = profile2.creativity.score
            elif dim == 'novelty':
                score1 = profile1.novelty.overall_novelty
                score2 = profile2.novelty.overall_novelty
            else:
                score1 = getattr(profile1, f'{dim}_diversity').score
                score2 = getattr(profile2, f'{dim}_diversity').score

            comparison['dimension_comparison'][dim] = {
                'text1': score1,
                'text2': score2,
                'difference': score1 - score2
            }

        return comparison

    def generate_report(self, profile: DiversityProfile) -> str:
        """
        Generate diversity report.

        Args:
            profile: DiversityProfile to report on

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "DIVERSITY & CREATIVITY ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Overall Diversity Score: {profile.overall_score:.3f}",
            "",
            "DIMENSION SCORES",
            "-" * 40,
            f"  Lexical Diversity:    {profile.lexical_diversity.score:.3f}",
            f"  Semantic Diversity:   {profile.semantic_diversity.score:.3f}",
            f"  Structural Diversity: {profile.structural_diversity.score:.3f}",
            f"  Creativity:           {profile.creativity.score:.3f} ({profile.creativity.level.name})",
            f"  Novelty:              {profile.novelty.overall_novelty:.3f}",
            "",
            "LEXICAL METRICS",
            "-" * 40
        ]

        for metric, value in profile.lexical_diversity.metrics.items():
            lines.append(f"  {metric}: {value:.3f}")

        lines.extend([
            "",
            "CREATIVITY DIMENSIONS",
            "-" * 40
        ])

        for dim, score in profile.creativity.dimensions.items():
            lines.append(f"  {dim}: {score:.3f}")

        if profile.creativity.creative_elements:
            lines.extend([
                "",
                "CREATIVE ELEMENTS FOUND",
                "-" * 40
            ])
            for elem in profile.creativity.creative_elements:
                lines.append(f"  + {elem}")

        if profile.creativity.conventional_elements:
            lines.extend([
                "",
                "CONVENTIONAL ELEMENTS",
                "-" * 40
            ])
            for elem in profile.creativity.conventional_elements:
                lines.append(f"  - {elem}")

        lines.append("=" * 60)

        return "\n".join(lines)


# Convenience functions
def analyze_diversity(text: str) -> DiversityProfile:
    """Quick diversity analysis."""
    evaluator = ComprehensiveDiversityEvaluator()
    return evaluator.evaluate(text)


def analyze_creativity(text: str) -> CreativityScore:
    """Quick creativity analysis."""
    analyzer = CreativityAnalyzer()
    return analyzer.analyze(text)


def calculate_distinct_n(text: str, max_n: int = 4) -> Dict[str, float]:
    """Calculate Distinct-N metrics."""
    analyzer = DistinctNAnalyzer(max_n)
    return analyzer.calculate(text)


def calculate_self_bleu(texts: List[str]) -> Dict[str, float]:
    """Calculate Self-BLEU across texts."""
    analyzer = SelfBLEUAnalyzer()
    return analyzer.calculate(texts)


__all__ = [
    # Enums
    'DiversityType',
    'CreativityLevel',
    'NoveltyType',

    # Data classes
    'DiversityScore',
    'CreativityScore',
    'NoveltyScore',
    'DiversityProfile',

    # Analyzers
    'LexicalDiversityAnalyzer',
    'SemanticDiversityAnalyzer',
    'StructuralDiversityAnalyzer',
    'CreativityAnalyzer',
    'NoveltyAnalyzer',
    'SelfBLEUAnalyzer',
    'DistinctNAnalyzer',
    'ComprehensiveDiversityEvaluator',

    # Convenience functions
    'analyze_diversity',
    'analyze_creativity',
    'calculate_distinct_n',
    'calculate_self_bleu',
]
