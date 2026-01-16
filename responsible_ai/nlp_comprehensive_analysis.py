"""
NLP Comprehensive Analysis Module
=================================

Comprehensive NLP analysis framework covering text quality, hallucination detection,
bias/toxicity analysis, prompt sensitivity, and language model evaluation.

Categories:
1. Text Data Quality Analysis - Tokenization, vocabulary, text statistics
2. Hallucination Analysis - Factuality, groundedness, attribution
3. Bias and Toxicity Analysis - Harmful content, demographic bias
4. Prompt Sensitivity Analysis - Prompt robustness, injection detection
5. Language Model Metrics - Perplexity, generation quality
6. Semantic Analysis - Embeddings, similarity, coherence
7. Summarization Metrics - ROUGE, factual consistency
8. Translation Metrics - BLEU, chrF, semantic similarity
9. Question Answering Metrics - EM, F1, answer quality
10. Dialogue Analysis - Coherence, engagement, safety
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import math
import re


# =============================================================================
# ENUMS
# =============================================================================

class TextQualityDimension(Enum):
    """Dimensions of text quality."""
    FLUENCY = auto()
    COHERENCE = auto()
    RELEVANCE = auto()
    INFORMATIVENESS = auto()
    CONCISENESS = auto()
    GRAMMAR = auto()
    STYLE = auto()
    READABILITY = auto()


class HallucinationType(Enum):
    """Types of hallucinations in text generation."""
    FACTUAL = auto()  # Factually incorrect
    INTRINSIC = auto()  # Contradicts source
    EXTRINSIC = auto()  # Not verifiable from source
    ENTITY = auto()  # Wrong entity mentioned
    RELATION = auto()  # Wrong relationship
    TEMPORAL = auto()  # Wrong temporal information
    NUMERICAL = auto()  # Wrong numbers/statistics
    FABRICATION = auto()  # Completely made up


class BiasType(Enum):
    """Types of bias in text."""
    GENDER = auto()
    RACIAL = auto()
    RELIGIOUS = auto()
    POLITICAL = auto()
    AGE = auto()
    DISABILITY = auto()
    NATIONALITY = auto()
    SOCIOECONOMIC = auto()
    STEREOTYPING = auto()


class ToxicityType(Enum):
    """Types of toxic content."""
    HATE_SPEECH = auto()
    HARASSMENT = auto()
    PROFANITY = auto()
    THREAT = auto()
    SEXUALLY_EXPLICIT = auto()
    SELF_HARM = auto()
    VIOLENCE = auto()
    DISCRIMINATION = auto()


class PromptAttackType(Enum):
    """Types of prompt-based attacks."""
    INJECTION = auto()
    JAILBREAK = auto()
    EXTRACTION = auto()
    MANIPULATION = auto()
    ROLE_PLAY = auto()
    ENCODING_BYPASS = auto()


class NLPTaskType(Enum):
    """NLP task types."""
    TEXT_CLASSIFICATION = auto()
    NAMED_ENTITY_RECOGNITION = auto()
    SENTIMENT_ANALYSIS = auto()
    SUMMARIZATION = auto()
    TRANSLATION = auto()
    QUESTION_ANSWERING = auto()
    TEXT_GENERATION = auto()
    DIALOGUE = auto()
    INFORMATION_EXTRACTION = auto()
    SEMANTIC_SIMILARITY = auto()


class SeverityLevel(Enum):
    """Severity levels for issues."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TextStatistics:
    """Statistics about a text."""
    char_count: int
    word_count: int
    sentence_count: int
    avg_word_length: float
    avg_sentence_length: float
    vocabulary_size: int
    type_token_ratio: float
    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0


@dataclass
class TextQualityMetrics:
    """Text quality assessment metrics."""
    dimension: TextQualityDimension
    score: float
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class HallucinationInstance:
    """Instance of detected hallucination."""
    hallucination_type: HallucinationType
    text_span: str
    start_idx: int
    end_idx: int
    confidence: float
    evidence: str = ""
    correction: str = ""


@dataclass
class HallucinationAnalysisResult:
    """Result from hallucination analysis."""
    total_claims: int
    verified_claims: int
    hallucinated_claims: int
    hallucination_rate: float
    instances: List[HallucinationInstance] = field(default_factory=list)
    factuality_score: float = 0.0
    groundedness_score: float = 0.0


@dataclass
class BiasInstance:
    """Instance of detected bias."""
    bias_type: BiasType
    text_span: str
    confidence: float
    explanation: str = ""
    affected_groups: List[str] = field(default_factory=list)


@dataclass
class BiasAnalysisResult:
    """Result from bias analysis."""
    bias_score: float
    bias_types_detected: List[BiasType]
    instances: List[BiasInstance] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ToxicityInstance:
    """Instance of detected toxicity."""
    toxicity_type: ToxicityType
    text_span: str
    severity: SeverityLevel
    confidence: float


@dataclass
class ToxicityAnalysisResult:
    """Result from toxicity analysis."""
    is_toxic: bool
    overall_score: float
    instances: List[ToxicityInstance] = field(default_factory=list)
    type_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class PromptSensitivityResult:
    """Result from prompt sensitivity analysis."""
    original_prompt: str
    variations_tested: int
    output_stability: float
    semantic_consistency: float
    vulnerability_score: float
    detected_attacks: List[PromptAttackType] = field(default_factory=list)


@dataclass
class LanguageModelMetrics:
    """Metrics for language model evaluation."""
    perplexity: float
    bits_per_character: float
    vocabulary_coverage: float
    oov_rate: float
    repetition_rate: float
    diversity_score: float


@dataclass
class SemanticSimilarityResult:
    """Result from semantic similarity analysis."""
    text_a: str
    text_b: str
    cosine_similarity: float
    jaccard_similarity: float
    semantic_score: float
    method: str = "embedding"


@dataclass
class SummarizationMetrics:
    """Metrics for summarization evaluation."""
    rouge_1: float
    rouge_2: float
    rouge_l: float
    factual_consistency: float
    compression_ratio: float
    key_info_coverage: float


@dataclass
class TranslationMetrics:
    """Metrics for translation evaluation."""
    bleu_score: float
    chrf_score: float
    ter_score: float
    semantic_similarity: float
    adequacy_score: float
    fluency_score: float


@dataclass
class QAMetrics:
    """Metrics for question answering evaluation."""
    exact_match: float
    f1_score: float
    answer_relevance: float
    answer_completeness: float
    has_answer_accuracy: float


@dataclass
class DialogueMetrics:
    """Metrics for dialogue evaluation."""
    coherence_score: float
    engagement_score: float
    informativeness_score: float
    safety_score: float
    context_relevance: float
    response_diversity: float


@dataclass
class NLPAssessment:
    """Comprehensive NLP assessment."""
    assessment_id: str
    timestamp: datetime
    task_type: NLPTaskType
    text_quality_score: float
    hallucination_risk_score: float
    bias_risk_score: float
    toxicity_risk_score: float
    prompt_robustness_score: float
    task_performance_score: float
    overall_score: float
    critical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# ANALYZERS - TEXT QUALITY
# =============================================================================

class TextQualityAnalyzer:
    """Analyzer for text data quality."""

    def compute_statistics(self, text: str) -> TextStatistics:
        """Compute basic text statistics."""
        if not text:
            return TextStatistics(
                char_count=0, word_count=0, sentence_count=0,
                avg_word_length=0, avg_sentence_length=0,
                vocabulary_size=0, type_token_ratio=0
            )

        # Character count
        char_count = len(text)

        # Word tokenization (simple)
        words = text.split()
        word_count = len(words)

        # Sentence count (simple)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)

        # Average word length
        avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0

        # Average sentence length
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        # Vocabulary
        unique_words = set(w.lower() for w in words)
        vocabulary_size = len(unique_words)
        type_token_ratio = vocabulary_size / word_count if word_count > 0 else 0

        # Flesch Reading Ease (simplified)
        flesch = 206.835 - 1.015 * avg_sentence_length - 84.6 * (
            sum(self._count_syllables(w) for w in words) / max(word_count, 1)
        )

        return TextStatistics(
            char_count=char_count,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=avg_word_length,
            avg_sentence_length=avg_sentence_length,
            vocabulary_size=vocabulary_size,
            type_token_ratio=type_token_ratio,
            flesch_reading_ease=max(0, min(100, flesch)),
        )

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)."""
        word = word.lower()
        vowels = 'aeiouy'
        count = 0
        prev_is_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel

        # Handle silent e
        if word.endswith('e'):
            count -= 1

        return max(count, 1)

    def analyze_fluency(self, text: str) -> TextQualityMetrics:
        """Analyze text fluency."""
        issues = []
        suggestions = []

        # Check for repeated words
        words = text.split()
        for i in range(len(words) - 1):
            if words[i].lower() == words[i+1].lower():
                issues.append(f"Repeated word: '{words[i]}'")

        # Check sentence length variation
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            lengths = [len(s.split()) for s in sentences]
            if len(lengths) > 1:
                length_variance = sum((l - sum(lengths)/len(lengths))**2 for l in lengths) / len(lengths)
                if length_variance < 5:
                    issues.append("Low sentence length variation")
                    suggestions.append("Vary sentence lengths for better flow")

        # Score
        score = 1.0 - min(len(issues) * 0.1, 0.5)

        return TextQualityMetrics(
            dimension=TextQualityDimension.FLUENCY,
            score=score,
            issues=issues,
            suggestions=suggestions,
        )

    def analyze_coherence(self, sentences: List[str]) -> TextQualityMetrics:
        """Analyze text coherence."""
        issues = []
        suggestions = []

        if len(sentences) < 2:
            return TextQualityMetrics(
                dimension=TextQualityDimension.COHERENCE,
                score=1.0,
            )

        # Check for transition words
        transition_words = ['however', 'therefore', 'furthermore', 'moreover',
                          'additionally', 'consequently', 'nevertheless', 'thus']

        has_transitions = any(
            any(tw in sent.lower() for tw in transition_words)
            for sent in sentences
        )

        if not has_transitions and len(sentences) > 3:
            suggestions.append("Consider adding transition words for better coherence")

        # Simple coherence score
        score = 0.8 if has_transitions else 0.6

        return TextQualityMetrics(
            dimension=TextQualityDimension.COHERENCE,
            score=score,
            issues=issues,
            suggestions=suggestions,
        )


class TextPreprocessingAnalyzer:
    """Analyzer for text preprocessing quality."""

    def analyze_tokenization(
        self,
        text: str,
        tokens: List[str],
        expected_vocab: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """Analyze tokenization quality."""
        # OOV analysis
        oov_tokens = []
        if expected_vocab:
            oov_tokens = [t for t in tokens if t.lower() not in expected_vocab]

        return {
            "num_tokens": len(tokens),
            "avg_token_length": sum(len(t) for t in tokens) / len(tokens) if tokens else 0,
            "unique_tokens": len(set(tokens)),
            "oov_count": len(oov_tokens),
            "oov_rate": len(oov_tokens) / len(tokens) if tokens else 0,
        }


# =============================================================================
# ANALYZERS - HALLUCINATION
# =============================================================================

class HallucinationAnalyzer:
    """Analyzer for hallucination detection."""

    def analyze_hallucinations(
        self,
        generated_text: str,
        source_documents: List[str],
        claims: Optional[List[str]] = None
    ) -> HallucinationAnalysisResult:
        """Analyze hallucinations in generated text."""
        if claims is None:
            # Extract claims (simplified - just use sentences)
            claims = [s.strip() for s in re.split(r'[.!?]+', generated_text) if s.strip()]

        if not claims:
            return HallucinationAnalysisResult(
                total_claims=0,
                verified_claims=0,
                hallucinated_claims=0,
                hallucination_rate=0.0,
                factuality_score=1.0,
                groundedness_score=1.0,
            )

        # Build source content set (for simple checking)
        source_content = " ".join(source_documents).lower()
        source_words = set(source_content.split())

        verified = 0
        instances = []

        for i, claim in enumerate(claims):
            claim_words = set(claim.lower().split())

            # Simple overlap-based verification
            overlap = len(claim_words & source_words) / len(claim_words) if claim_words else 0

            if overlap > 0.5:
                verified += 1
            else:
                # Detect hallucination type
                if any(char.isdigit() for char in claim):
                    hall_type = HallucinationType.NUMERICAL
                elif any(word[0].isupper() for word in claim.split() if word):
                    hall_type = HallucinationType.ENTITY
                else:
                    hall_type = HallucinationType.EXTRINSIC

                instances.append(HallucinationInstance(
                    hallucination_type=hall_type,
                    text_span=claim,
                    start_idx=i,
                    end_idx=i + 1,
                    confidence=1 - overlap,
                ))

        hallucinated = len(claims) - verified
        hallucination_rate = hallucinated / len(claims) if claims else 0
        factuality_score = 1 - hallucination_rate
        groundedness_score = verified / len(claims) if claims else 1.0

        return HallucinationAnalysisResult(
            total_claims=len(claims),
            verified_claims=verified,
            hallucinated_claims=hallucinated,
            hallucination_rate=hallucination_rate,
            instances=instances,
            factuality_score=factuality_score,
            groundedness_score=groundedness_score,
        )

    def check_factual_consistency(
        self,
        source: str,
        generated: str
    ) -> float:
        """Check factual consistency between source and generated text."""
        # Simple word overlap method
        source_words = set(source.lower().split())
        generated_words = set(generated.lower().split())

        overlap = len(source_words & generated_words)
        consistency = overlap / len(generated_words) if generated_words else 0

        return min(consistency * 1.5, 1.0)  # Scale up since overlap is conservative


# =============================================================================
# ANALYZERS - BIAS AND TOXICITY
# =============================================================================

class BiasAnalyzer:
    """Analyzer for bias in text."""

    def __init__(self):
        # Simplified bias indicators
        self.bias_indicators = {
            BiasType.GENDER: ['he always', 'she always', 'women are', 'men are',
                             'typical woman', 'typical man', 'girls cant', 'boys cant'],
            BiasType.STEREOTYPING: ['all of them', 'they always', 'those people',
                                   'you know how they', 'typical'],
        }

    def analyze_bias(self, text: str) -> BiasAnalysisResult:
        """Analyze bias in text."""
        text_lower = text.lower()
        instances = []
        types_detected = set()

        for bias_type, indicators in self.bias_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    types_detected.add(bias_type)
                    # Find the context
                    idx = text_lower.find(indicator)
                    start = max(0, idx - 20)
                    end = min(len(text), idx + len(indicator) + 20)
                    span = text[start:end]

                    instances.append(BiasInstance(
                        bias_type=bias_type,
                        text_span=span,
                        confidence=0.7,
                        explanation=f"Contains potentially biased language: '{indicator}'",
                    ))

        bias_score = min(len(instances) * 0.2, 1.0)

        recommendations = []
        if instances:
            recommendations.append("Review highlighted text for potential bias")
            recommendations.append("Consider using more inclusive language")

        return BiasAnalysisResult(
            bias_score=bias_score,
            bias_types_detected=list(types_detected),
            instances=instances,
            recommendations=recommendations,
        )


class ToxicityAnalyzer:
    """Analyzer for toxic content."""

    def __init__(self):
        # Simplified toxicity patterns (in practice, use ML models)
        self.toxicity_patterns = {
            ToxicityType.PROFANITY: [],  # Would contain patterns
            ToxicityType.HATE_SPEECH: ['hate', 'kill all', 'death to'],
            ToxicityType.THREAT: ['will hurt', 'going to kill', 'watch out'],
        }

    def analyze_toxicity(self, text: str) -> ToxicityAnalysisResult:
        """Analyze toxicity in text."""
        text_lower = text.lower()
        instances = []
        type_scores = {}

        for tox_type, patterns in self.toxicity_patterns.items():
            found = False
            for pattern in patterns:
                if pattern in text_lower:
                    found = True
                    idx = text_lower.find(pattern)
                    instances.append(ToxicityInstance(
                        toxicity_type=tox_type,
                        text_span=text[max(0, idx-10):min(len(text), idx+len(pattern)+10)],
                        severity=SeverityLevel.HIGH if tox_type == ToxicityType.THREAT else SeverityLevel.MEDIUM,
                        confidence=0.8,
                    ))
            type_scores[tox_type.name] = 1.0 if found else 0.0

        is_toxic = len(instances) > 0
        overall_score = min(len(instances) * 0.3, 1.0)

        return ToxicityAnalysisResult(
            is_toxic=is_toxic,
            overall_score=overall_score,
            instances=instances,
            type_scores=type_scores,
        )


# =============================================================================
# ANALYZERS - PROMPT SENSITIVITY
# =============================================================================

class PromptSensitivityAnalyzer:
    """Analyzer for prompt sensitivity and robustness."""

    def __init__(self):
        self.injection_patterns = [
            'ignore previous',
            'ignore all instructions',
            'disregard',
            'new instructions',
            'system prompt',
            'you are now',
            'pretend to be',
            'act as',
        ]

    def analyze_prompt_robustness(
        self,
        original_prompt: str,
        prompt_variations: List[str],
        outputs: List[str]
    ) -> PromptSensitivityResult:
        """Analyze prompt robustness across variations."""
        if not outputs or len(outputs) < 2:
            return PromptSensitivityResult(
                original_prompt=original_prompt,
                variations_tested=len(prompt_variations),
                output_stability=1.0,
                semantic_consistency=1.0,
                vulnerability_score=0.0,
            )

        # Output stability (how similar outputs are)
        # Simple: use word overlap between consecutive outputs
        stabilities = []
        for i in range(len(outputs) - 1):
            words1 = set(outputs[i].lower().split())
            words2 = set(outputs[i+1].lower().split())
            overlap = len(words1 & words2) / max(len(words1 | words2), 1)
            stabilities.append(overlap)

        output_stability = sum(stabilities) / len(stabilities) if stabilities else 1.0

        # Check for injection vulnerabilities
        detected_attacks = []
        for pattern in self.injection_patterns:
            if pattern in original_prompt.lower():
                detected_attacks.append(PromptAttackType.INJECTION)
                break

        vulnerability_score = 0.5 if detected_attacks else 0.1

        return PromptSensitivityResult(
            original_prompt=original_prompt,
            variations_tested=len(prompt_variations),
            output_stability=output_stability,
            semantic_consistency=output_stability,  # Simplified
            vulnerability_score=vulnerability_score,
            detected_attacks=list(set(detected_attacks)),
        )

    def detect_injection_attempts(self, prompt: str) -> List[PromptAttackType]:
        """Detect potential prompt injection attempts."""
        prompt_lower = prompt.lower()
        attacks = []

        for pattern in self.injection_patterns:
            if pattern in prompt_lower:
                attacks.append(PromptAttackType.INJECTION)
                break

        if 'jailbreak' in prompt_lower or 'bypass' in prompt_lower:
            attacks.append(PromptAttackType.JAILBREAK)

        if 'extract' in prompt_lower and ('system' in prompt_lower or 'prompt' in prompt_lower):
            attacks.append(PromptAttackType.EXTRACTION)

        return list(set(attacks))


# =============================================================================
# ANALYZERS - LANGUAGE MODEL METRICS
# =============================================================================

class LanguageModelAnalyzer:
    """Analyzer for language model evaluation."""

    def calculate_perplexity(
        self,
        log_probabilities: List[float]
    ) -> float:
        """Calculate perplexity from log probabilities."""
        if not log_probabilities:
            return float('inf')

        avg_log_prob = sum(log_probabilities) / len(log_probabilities)
        return math.exp(-avg_log_prob)

    def analyze_generation_quality(
        self,
        generated_text: str,
        log_probs: Optional[List[float]] = None
    ) -> LanguageModelMetrics:
        """Analyze generation quality."""
        words = generated_text.split()

        # Repetition rate
        if len(words) > 1:
            repetitions = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
            repetition_rate = repetitions / (len(words) - 1)
        else:
            repetition_rate = 0

        # Diversity (unique n-grams)
        if len(words) >= 2:
            bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
            unique_bigrams = len(set(bigrams))
            diversity_score = unique_bigrams / len(bigrams) if bigrams else 1.0
        else:
            diversity_score = 1.0

        # Perplexity
        perplexity = self.calculate_perplexity(log_probs) if log_probs else 0

        return LanguageModelMetrics(
            perplexity=perplexity,
            bits_per_character=math.log2(perplexity) if perplexity > 0 else 0,
            vocabulary_coverage=len(set(words)) / len(words) if words else 0,
            oov_rate=0.0,  # Would need vocabulary
            repetition_rate=repetition_rate,
            diversity_score=diversity_score,
        )


# =============================================================================
# ANALYZERS - SEMANTIC ANALYSIS
# =============================================================================

class SemanticAnalyzer:
    """Analyzer for semantic analysis."""

    def calculate_similarity(
        self,
        text_a: str,
        text_b: str,
        embeddings_a: Optional[List[float]] = None,
        embeddings_b: Optional[List[float]] = None
    ) -> SemanticSimilarityResult:
        """Calculate semantic similarity between texts."""
        # Jaccard similarity (word-level)
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        jaccard = intersection / union if union > 0 else 0

        # Cosine similarity (if embeddings provided)
        cosine = 0.0
        if embeddings_a and embeddings_b and len(embeddings_a) == len(embeddings_b):
            dot_product = sum(a * b for a, b in zip(embeddings_a, embeddings_b))
            norm_a = sum(a * a for a in embeddings_a) ** 0.5
            norm_b = sum(b * b for b in embeddings_b) ** 0.5
            cosine = dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0

        # Semantic score (weighted average)
        semantic_score = cosine if embeddings_a else jaccard

        return SemanticSimilarityResult(
            text_a=text_a[:50] + "..." if len(text_a) > 50 else text_a,
            text_b=text_b[:50] + "..." if len(text_b) > 50 else text_b,
            cosine_similarity=cosine,
            jaccard_similarity=jaccard,
            semantic_score=semantic_score,
            method="embedding" if embeddings_a else "word_overlap",
        )


# =============================================================================
# ANALYZERS - TASK-SPECIFIC METRICS
# =============================================================================

class SummarizationAnalyzer:
    """Analyzer for summarization quality."""

    def calculate_rouge(
        self,
        summary: str,
        reference: str
    ) -> SummarizationMetrics:
        """Calculate ROUGE scores."""
        # Tokenize
        summary_words = summary.lower().split()
        reference_words = reference.lower().split()

        # ROUGE-1 (unigram overlap)
        overlap_1 = len(set(summary_words) & set(reference_words))
        rouge_1_recall = overlap_1 / len(reference_words) if reference_words else 0
        rouge_1_precision = overlap_1 / len(summary_words) if summary_words else 0
        rouge_1 = 2 * rouge_1_recall * rouge_1_precision / (rouge_1_recall + rouge_1_precision) if (rouge_1_recall + rouge_1_precision) > 0 else 0

        # ROUGE-2 (bigram overlap)
        def get_bigrams(words):
            return set((words[i], words[i+1]) for i in range(len(words)-1)) if len(words) > 1 else set()

        summary_bigrams = get_bigrams(summary_words)
        reference_bigrams = get_bigrams(reference_words)
        overlap_2 = len(summary_bigrams & reference_bigrams)
        rouge_2 = overlap_2 / len(reference_bigrams) if reference_bigrams else 0

        # ROUGE-L (LCS-based)
        rouge_l = self._lcs_ratio(summary_words, reference_words)

        # Compression ratio
        compression_ratio = len(summary_words) / len(reference_words) if reference_words else 0

        return SummarizationMetrics(
            rouge_1=rouge_1,
            rouge_2=rouge_2,
            rouge_l=rouge_l,
            factual_consistency=rouge_1,  # Simplified
            compression_ratio=compression_ratio,
            key_info_coverage=rouge_1,
        )

    def _lcs_ratio(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate LCS ratio."""
        m, n = len(seq1), len(seq2)
        if m == 0 or n == 0:
            return 0

        # Dynamic programming for LCS length
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs_length = dp[m][n]
        return lcs_length / max(m, n)


class TranslationAnalyzer:
    """Analyzer for translation quality."""

    def calculate_bleu(
        self,
        translation: str,
        reference: str,
        max_n: int = 4
    ) -> float:
        """Calculate BLEU score."""
        trans_words = translation.lower().split()
        ref_words = reference.lower().split()

        if not trans_words:
            return 0.0

        # Calculate n-gram precisions
        precisions = []

        for n in range(1, min(max_n + 1, len(trans_words) + 1)):
            trans_ngrams = self._get_ngrams(trans_words, n)
            ref_ngrams = self._get_ngrams(ref_words, n)

            matches = sum(min(trans_ngrams.get(ng, 0), ref_ngrams.get(ng, 0))
                         for ng in trans_ngrams)
            total = sum(trans_ngrams.values())

            precision = matches / total if total > 0 else 0
            precisions.append(precision)

        if not precisions or all(p == 0 for p in precisions):
            return 0.0

        # Geometric mean
        log_precision = sum(math.log(p + 1e-10) for p in precisions) / len(precisions)
        geo_mean = math.exp(log_precision)

        # Brevity penalty
        bp = min(1, math.exp(1 - len(ref_words) / len(trans_words))) if len(trans_words) > 0 else 0

        return bp * geo_mean

    def _get_ngrams(self, words: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        """Get n-gram counts."""
        ngrams = {}
        for i in range(len(words) - n + 1):
            ng = tuple(words[i:i+n])
            ngrams[ng] = ngrams.get(ng, 0) + 1
        return ngrams

    def analyze_translation(
        self,
        translation: str,
        reference: str
    ) -> TranslationMetrics:
        """Analyze translation quality."""
        bleu = self.calculate_bleu(translation, reference)

        # chrF (character n-gram F-score) - simplified
        chrf = self._calculate_chrf(translation, reference)

        return TranslationMetrics(
            bleu_score=bleu,
            chrf_score=chrf,
            ter_score=0.0,  # Would need edit distance calculation
            semantic_similarity=bleu,  # Simplified
            adequacy_score=bleu,
            fluency_score=0.8,  # Would need fluency model
        )

    def _calculate_chrf(self, translation: str, reference: str) -> float:
        """Calculate chrF score (simplified)."""
        trans_chars = list(translation.lower())
        ref_chars = list(reference.lower())

        if not trans_chars or not ref_chars:
            return 0.0

        # Character 3-gram overlap
        def char_ngrams(chars, n):
            return set(tuple(chars[i:i+n]) for i in range(len(chars)-n+1))

        trans_3grams = char_ngrams(trans_chars, 3)
        ref_3grams = char_ngrams(ref_chars, 3)

        overlap = len(trans_3grams & ref_3grams)
        precision = overlap / len(trans_3grams) if trans_3grams else 0
        recall = overlap / len(ref_3grams) if ref_3grams else 0

        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


class QAAnalyzer:
    """Analyzer for question answering quality."""

    def calculate_metrics(
        self,
        predicted_answer: str,
        ground_truth: str
    ) -> QAMetrics:
        """Calculate QA metrics."""
        # Normalize
        pred_normalized = predicted_answer.lower().strip()
        gt_normalized = ground_truth.lower().strip()

        # Exact match
        exact_match = 1.0 if pred_normalized == gt_normalized else 0.0

        # F1 score (token-level)
        pred_tokens = set(pred_normalized.split())
        gt_tokens = set(gt_normalized.split())

        common = len(pred_tokens & gt_tokens)
        precision = common / len(pred_tokens) if pred_tokens else 0
        recall = common / len(gt_tokens) if gt_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return QAMetrics(
            exact_match=exact_match,
            f1_score=f1,
            answer_relevance=f1,
            answer_completeness=recall,
            has_answer_accuracy=1.0 if predicted_answer else 0.0,
        )


class DialogueAnalyzer:
    """Analyzer for dialogue quality."""

    def analyze_dialogue(
        self,
        context: List[str],
        response: str
    ) -> DialogueMetrics:
        """Analyze dialogue response quality."""
        # Coherence (word overlap with context)
        context_words = set(word.lower() for turn in context for word in turn.split())
        response_words = set(response.lower().split())

        overlap = len(context_words & response_words)
        coherence = overlap / len(response_words) if response_words else 0

        # Informativeness (unique words in response)
        unique_response_words = response_words - context_words
        informativeness = len(unique_response_words) / len(response_words) if response_words else 0

        # Response diversity
        words = response.split()
        diversity = len(set(words)) / len(words) if words else 0

        # Safety (simple check)
        safety = 1.0  # Would use toxicity analyzer

        return DialogueMetrics(
            coherence_score=min(coherence * 2, 1.0),
            engagement_score=informativeness,
            informativeness_score=informativeness,
            safety_score=safety,
            context_relevance=coherence,
            response_diversity=diversity,
        )


# =============================================================================
# COMPREHENSIVE ANALYZER
# =============================================================================

class NLPComprehensiveAnalyzer:
    """Comprehensive NLP analyzer."""

    def __init__(self):
        self.text_quality_analyzer = TextQualityAnalyzer()
        self.hallucination_analyzer = HallucinationAnalyzer()
        self.bias_analyzer = BiasAnalyzer()
        self.toxicity_analyzer = ToxicityAnalyzer()
        self.prompt_analyzer = PromptSensitivityAnalyzer()
        self.lm_analyzer = LanguageModelAnalyzer()
        self.semantic_analyzer = SemanticAnalyzer()
        self.summarization_analyzer = SummarizationAnalyzer()
        self.translation_analyzer = TranslationAnalyzer()
        self.qa_analyzer = QAAnalyzer()
        self.dialogue_analyzer = DialogueAnalyzer()

    def comprehensive_assessment(
        self,
        task_type: NLPTaskType,
        text: str,
        source_documents: Optional[List[str]] = None,
        reference: Optional[str] = None,
        task_metrics: Optional[Dict[str, float]] = None
    ) -> NLPAssessment:
        """Perform comprehensive NLP assessment."""
        assessment_id = f"NLP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        critical_issues = []
        recommendations = []

        # Text quality
        stats = self.text_quality_analyzer.compute_statistics(text)
        fluency = self.text_quality_analyzer.analyze_fluency(text)
        text_quality_score = fluency.score

        # Hallucination risk
        if source_documents:
            hall_result = self.hallucination_analyzer.analyze_hallucinations(text, source_documents)
            hallucination_risk = hall_result.hallucination_rate
            if hall_result.hallucination_rate > 0.2:
                critical_issues.append(f"High hallucination rate: {hall_result.hallucination_rate:.1%}")
        else:
            hallucination_risk = 0.1  # Default low risk

        # Bias risk
        bias_result = self.bias_analyzer.analyze_bias(text)
        bias_risk = bias_result.bias_score
        if bias_result.instances:
            recommendations.extend(bias_result.recommendations)

        # Toxicity risk
        toxicity_result = self.toxicity_analyzer.analyze_toxicity(text)
        toxicity_risk = toxicity_result.overall_score
        if toxicity_result.is_toxic:
            critical_issues.append("Toxic content detected")

        # Prompt robustness (placeholder)
        prompt_robustness = 0.8

        # Task performance
        task_performance = 0.0
        if task_metrics:
            task_performance = sum(task_metrics.values()) / len(task_metrics)
        elif reference:
            if task_type == NLPTaskType.SUMMARIZATION:
                rouge = self.summarization_analyzer.calculate_rouge(text, reference)
                task_performance = (rouge.rouge_1 + rouge.rouge_2 + rouge.rouge_l) / 3
            elif task_type == NLPTaskType.TRANSLATION:
                trans = self.translation_analyzer.analyze_translation(text, reference)
                task_performance = trans.bleu_score
            elif task_type == NLPTaskType.QUESTION_ANSWERING:
                qa = self.qa_analyzer.calculate_metrics(text, reference)
                task_performance = qa.f1_score
        else:
            task_performance = 0.7  # Default

        # Overall score
        overall = (
            text_quality_score * 0.15 +
            (1 - hallucination_risk) * 0.2 +
            (1 - bias_risk) * 0.15 +
            (1 - toxicity_risk) * 0.15 +
            prompt_robustness * 0.1 +
            task_performance * 0.25
        )

        return NLPAssessment(
            assessment_id=assessment_id,
            timestamp=datetime.now(),
            task_type=task_type,
            text_quality_score=text_quality_score,
            hallucination_risk_score=hallucination_risk,
            bias_risk_score=bias_risk,
            toxicity_risk_score=toxicity_risk,
            prompt_robustness_score=prompt_robustness,
            task_performance_score=task_performance,
            overall_score=overall,
            critical_issues=critical_issues,
            recommendations=recommendations,
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def analyze_text_quality(text: str) -> TextStatistics:
    """Analyze text quality statistics."""
    analyzer = TextQualityAnalyzer()
    return analyzer.compute_statistics(text)


def detect_hallucinations(
    generated_text: str,
    source_documents: List[str]
) -> HallucinationAnalysisResult:
    """Detect hallucinations in generated text."""
    analyzer = HallucinationAnalyzer()
    return analyzer.analyze_hallucinations(generated_text, source_documents)


def analyze_bias(text: str) -> BiasAnalysisResult:
    """Analyze bias in text."""
    analyzer = BiasAnalyzer()
    return analyzer.analyze_bias(text)


def analyze_toxicity(text: str) -> ToxicityAnalysisResult:
    """Analyze toxicity in text."""
    analyzer = ToxicityAnalyzer()
    return analyzer.analyze_toxicity(text)


def calculate_rouge_scores(summary: str, reference: str) -> SummarizationMetrics:
    """Calculate ROUGE scores for summarization."""
    analyzer = SummarizationAnalyzer()
    return analyzer.calculate_rouge(summary, reference)


def calculate_bleu_score(translation: str, reference: str) -> float:
    """Calculate BLEU score for translation."""
    analyzer = TranslationAnalyzer()
    return analyzer.calculate_bleu(translation, reference)


def calculate_qa_metrics(predicted: str, ground_truth: str) -> QAMetrics:
    """Calculate QA metrics."""
    analyzer = QAAnalyzer()
    return analyzer.calculate_metrics(predicted, ground_truth)


def calculate_semantic_similarity(text_a: str, text_b: str) -> SemanticSimilarityResult:
    """Calculate semantic similarity between texts."""
    analyzer = SemanticAnalyzer()
    return analyzer.calculate_similarity(text_a, text_b)


def comprehensive_nlp_assessment(
    task_type: NLPTaskType,
    text: str,
    source_documents: Optional[List[str]] = None
) -> NLPAssessment:
    """Perform comprehensive NLP assessment."""
    analyzer = NLPComprehensiveAnalyzer()
    return analyzer.comprehensive_assessment(task_type, text, source_documents)
