"""
Factual Consistency Analysis Module for GenAI Evaluation.

This module provides implementations for factual consistency checking,
hallucination detection, and question-based evaluation frameworks
including QuestEval, FactCC, and related methodologies.

Classes:
    QuestEvalAnalyzer: Question-based evaluation for factual consistency
    FactCCAnalyzer: Fact checking with entailment-based scoring
    HallucinationDetector: Multi-method hallucination detection
    BERTScoreAnalyzer: BERT-based semantic similarity scoring
    BARTScoreAnalyzer: BART-based generation evaluation
    METEORAnalyzer: METEOR metric for translation/generation quality
    FactualConsistencyEvaluator: Comprehensive factual evaluation

Author: AgenticFinder Research Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
import re


class ConsistencyLevel(Enum):
    """Levels of factual consistency."""
    FULLY_CONSISTENT = 5
    MOSTLY_CONSISTENT = 4
    PARTIALLY_CONSISTENT = 3
    INCONSISTENT = 2
    CONTRADICTORY = 1


class HallucinationType(Enum):
    """Types of hallucination."""
    INTRINSIC = "intrinsic"  # Contradicts source
    EXTRINSIC = "extrinsic"  # Unverifiable claims
    ENTITY = "entity"  # Wrong entity
    RELATION = "relation"  # Wrong relationship
    NUMERICAL = "numerical"  # Wrong numbers
    TEMPORAL = "temporal"  # Wrong time/date


class EntailmentLabel(Enum):
    """Entailment classification labels."""
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


@dataclass
class ConsistencyScore:
    """Container for consistency scoring results."""
    score: float
    level: ConsistencyLevel
    method: str
    details: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'score': self.score,
            'level': self.level.name,
            'method': self.method,
            'details': self.details,
            'evidence': self.evidence,
            'issues': self.issues
        }


@dataclass
class HallucinationResult:
    """Container for hallucination detection results."""
    has_hallucination: bool
    confidence: float
    hallucination_types: List[HallucinationType] = field(default_factory=list)
    hallucinated_spans: List[Dict[str, Any]] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    severity: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'has_hallucination': self.has_hallucination,
            'confidence': self.confidence,
            'hallucination_types': [ht.value for ht in self.hallucination_types],
            'hallucinated_spans': self.hallucinated_spans,
            'supporting_evidence': self.supporting_evidence,
            'severity': self.severity
        }


@dataclass
class QuestionAnswerPair:
    """Container for QA-based evaluation."""
    question: str
    source_answer: Optional[str] = None
    generated_answer: Optional[str] = None
    match_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactualEvaluationResult:
    """Comprehensive factual evaluation result."""
    overall_score: float
    consistency_score: ConsistencyScore
    hallucination_result: HallucinationResult
    method_scores: Dict[str, float] = field(default_factory=dict)
    qa_pairs: List[QuestionAnswerPair] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation."""
        return {
            'overall_score': self.overall_score,
            'consistency_level': self.consistency_score.level.name,
            'has_hallucination': self.hallucination_result.has_hallucination,
            'hallucination_severity': self.hallucination_result.severity,
            'method_scores': self.method_scores,
            'recommendation_count': len(self.recommendations)
        }


class QuestEvalAnalyzer:
    """
    QuestEval-style question-based evaluation for factual consistency.

    Generates questions from source text and evaluates if generated
    text provides consistent answers to those questions.
    """

    def __init__(self,
                 question_types: Optional[List[str]] = None,
                 answer_matching_threshold: float = 0.7):
        """
        Initialize QuestEval analyzer.

        Args:
            question_types: Types of questions to generate
            answer_matching_threshold: Threshold for answer matching
        """
        self.question_types = question_types or ['who', 'what', 'when', 'where', 'why', 'how']
        self.answer_matching_threshold = answer_matching_threshold

    def evaluate(self, source: str, generated: str,
                 context: Optional[Dict[str, Any]] = None) -> ConsistencyScore:
        """
        Evaluate factual consistency using question-based approach.

        Args:
            source: Source/reference text
            generated: Generated text to evaluate
            context: Optional evaluation context

        Returns:
            ConsistencyScore with QA-based evaluation
        """
        # Generate questions from source
        questions = self._generate_questions(source)

        # Extract answers from both source and generated
        qa_pairs = []
        for question in questions:
            source_answer = self._extract_answer(source, question)
            generated_answer = self._extract_answer(generated, question)

            match_score = self._calculate_answer_match(source_answer, generated_answer)

            qa_pairs.append(QuestionAnswerPair(
                question=question,
                source_answer=source_answer,
                generated_answer=generated_answer,
                match_score=match_score
            ))

        # Calculate overall score
        if qa_pairs:
            avg_score = np.mean([qa.match_score for qa in qa_pairs])
        else:
            avg_score = 0.5  # Neutral if no questions generated

        # Determine consistency level
        level = self._score_to_level(avg_score)

        # Identify issues
        issues = []
        for qa in qa_pairs:
            if qa.match_score < self.answer_matching_threshold:
                issues.append(f"Inconsistent answer to: {qa.question[:50]}...")

        return ConsistencyScore(
            score=avg_score,
            level=level,
            method='QuestEval',
            details={
                'question_count': len(questions),
                'qa_pairs': [{'q': qa.question, 'score': qa.match_score} for qa in qa_pairs],
                'threshold': self.answer_matching_threshold
            },
            evidence=[f"Evaluated {len(questions)} questions"],
            issues=issues
        )

    def _generate_questions(self, text: str) -> List[str]:
        """Generate questions from text."""
        questions = []
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        for sent in sentences[:10]:  # Limit to first 10 sentences
            # Extract potential question subjects
            words = sent.split()
            if len(words) < 3:
                continue

            # Simple question generation based on sentence structure
            # Who/What questions
            if any(w.istitle() for w in words[1:]):
                questions.append(f"Who or what is mentioned in: {sent[:50]}?")

            # When questions (if temporal markers present)
            temporal_markers = {'in', 'on', 'at', 'during', 'after', 'before', 'since'}
            if any(w.lower() in temporal_markers for w in words):
                questions.append(f"When did the event occur in: {sent[:50]}?")

            # What happened question
            verbs = ['is', 'are', 'was', 'were', 'has', 'have', 'had', 'does', 'do', 'did']
            if any(w.lower() in verbs for w in words):
                questions.append(f"What is stated about: {words[0]}?")

        return questions[:15]  # Limit total questions

    def _extract_answer(self, text: str, question: str) -> Optional[str]:
        """Extract answer from text for given question."""
        text_lower = text.lower()
        question_lower = question.lower()

        # Extract key terms from question
        key_terms = [w for w in question_lower.split()
                    if len(w) > 3 and w not in {'what', 'when', 'where', 'who', 'how', 'why', 'which', 'does', 'about', 'mentioned'}]

        # Find sentences containing key terms
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        for sent in sentences:
            sent_lower = sent.lower()
            if any(term in sent_lower for term in key_terms):
                return sent

        return None

    def _calculate_answer_match(self, source_answer: Optional[str],
                                 generated_answer: Optional[str]) -> float:
        """Calculate match score between answers."""
        if source_answer is None or generated_answer is None:
            return 0.3  # Partial score if one answer missing

        if source_answer is None and generated_answer is None:
            return 0.5  # Both missing is neutral

        source_words = set(source_answer.lower().split())
        gen_words = set(generated_answer.lower().split())

        if not source_words:
            return 0.5

        # Jaccard similarity
        intersection = len(source_words & gen_words)
        union = len(source_words | gen_words)

        return intersection / union if union > 0 else 0.0

    def _score_to_level(self, score: float) -> ConsistencyLevel:
        """Convert score to consistency level."""
        if score >= 0.9:
            return ConsistencyLevel.FULLY_CONSISTENT
        elif score >= 0.7:
            return ConsistencyLevel.MOSTLY_CONSISTENT
        elif score >= 0.5:
            return ConsistencyLevel.PARTIALLY_CONSISTENT
        elif score >= 0.3:
            return ConsistencyLevel.INCONSISTENT
        else:
            return ConsistencyLevel.CONTRADICTORY


class FactCCAnalyzer:
    """
    FactCC-style factual consistency checking.

    Uses entailment-based approach to verify claims in generated
    text against source document.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize FactCC analyzer.

        Args:
            strict_mode: Use strict entailment checking
        """
        self.strict_mode = strict_mode

    def evaluate(self, source: str, generated: str,
                 context: Optional[Dict[str, Any]] = None) -> ConsistencyScore:
        """
        Evaluate factual consistency using entailment approach.

        Args:
            source: Source/reference text
            generated: Generated text (claim) to verify
            context: Optional evaluation context

        Returns:
            ConsistencyScore with entailment-based evaluation
        """
        # Extract claims from generated text
        claims = self._extract_claims(generated)

        # Verify each claim against source
        claim_results = []
        for claim in claims:
            entailment = self._check_entailment(source, claim)
            claim_results.append({
                'claim': claim,
                'entailment': entailment,
                'score': self._entailment_to_score(entailment)
            })

        # Calculate overall score
        if claim_results:
            scores = [r['score'] for r in claim_results]
            if self.strict_mode:
                # All claims must be entailed
                avg_score = min(scores)
            else:
                avg_score = np.mean(scores)
        else:
            avg_score = 0.5

        # Determine consistency level
        level = self._score_to_level(avg_score)

        # Identify issues
        issues = []
        contradictions = [r for r in claim_results
                         if r['entailment'] == EntailmentLabel.CONTRADICTION]
        if contradictions:
            for c in contradictions:
                issues.append(f"Contradictory claim: {c['claim'][:50]}...")

        return ConsistencyScore(
            score=avg_score,
            level=level,
            method='FactCC',
            details={
                'claim_count': len(claims),
                'claim_results': claim_results,
                'strict_mode': self.strict_mode,
                'contradiction_count': len(contradictions)
            },
            evidence=[f"Verified {len(claims)} claims against source"],
            issues=issues
        )

    def _extract_claims(self, text: str) -> List[str]:
        """Extract verifiable claims from text."""
        claims = []
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        # Claim indicators
        claim_verbs = {'is', 'are', 'was', 'were', 'has', 'have', 'had',
                      'states', 'shows', 'indicates', 'reveals', 'confirms'}

        for sent in sentences:
            words = sent.lower().split()
            # Sentence with claim verbs likely contains claims
            if any(v in words for v in claim_verbs):
                claims.append(sent)

        return claims

    def _check_entailment(self, premise: str, hypothesis: str) -> EntailmentLabel:
        """Check entailment relationship between premise and hypothesis."""
        premise_lower = premise.lower()
        hypothesis_lower = hypothesis.lower()

        # Simple lexical entailment check
        premise_words = set(premise_lower.split())
        hypothesis_words = set(hypothesis_lower.split())

        # Check for contradiction indicators
        negation_words = {'not', 'no', 'never', 'none', "n't", 'without', 'lack'}

        hyp_has_negation = bool(hypothesis_words & negation_words)
        prem_has_negation = bool(premise_words & negation_words)

        # Content overlap
        content_words = hypothesis_words - {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at'}
        overlap = content_words & premise_words

        overlap_ratio = len(overlap) / len(content_words) if content_words else 0

        # Determine entailment
        if overlap_ratio > 0.7 and hyp_has_negation == prem_has_negation:
            return EntailmentLabel.ENTAILMENT
        elif overlap_ratio > 0.5 and hyp_has_negation != prem_has_negation:
            return EntailmentLabel.CONTRADICTION
        elif overlap_ratio > 0.3:
            return EntailmentLabel.NEUTRAL
        else:
            return EntailmentLabel.NEUTRAL

    def _entailment_to_score(self, entailment: EntailmentLabel) -> float:
        """Convert entailment label to score."""
        if entailment == EntailmentLabel.ENTAILMENT:
            return 1.0
        elif entailment == EntailmentLabel.NEUTRAL:
            return 0.5
        else:
            return 0.0

    def _score_to_level(self, score: float) -> ConsistencyLevel:
        """Convert score to consistency level."""
        if score >= 0.9:
            return ConsistencyLevel.FULLY_CONSISTENT
        elif score >= 0.7:
            return ConsistencyLevel.MOSTLY_CONSISTENT
        elif score >= 0.5:
            return ConsistencyLevel.PARTIALLY_CONSISTENT
        elif score >= 0.3:
            return ConsistencyLevel.INCONSISTENT
        else:
            return ConsistencyLevel.CONTRADICTORY


class HallucinationDetector:
    """
    Multi-method hallucination detection.

    Combines various approaches to detect different types of
    hallucinations in generated text.
    """

    def __init__(self,
                 detection_threshold: float = 0.5,
                 sensitivity: str = "medium"):
        """
        Initialize hallucination detector.

        Args:
            detection_threshold: Threshold for hallucination detection
            sensitivity: Detection sensitivity (low, medium, high)
        """
        self.detection_threshold = detection_threshold
        self.sensitivity = sensitivity
        self.sensitivity_multiplier = {'low': 0.8, 'medium': 1.0, 'high': 1.2}.get(sensitivity, 1.0)

    def detect(self, source: str, generated: str,
               context: Optional[Dict[str, Any]] = None) -> HallucinationResult:
        """
        Detect hallucinations in generated text.

        Args:
            source: Source/reference text
            generated: Generated text to check
            context: Optional detection context

        Returns:
            HallucinationResult with detection details
        """
        hallucinated_spans = []
        hallucination_types = []
        supporting_evidence = []

        # Check for intrinsic hallucinations (contradictions)
        intrinsic_result = self._detect_intrinsic(source, generated)
        if intrinsic_result['detected']:
            hallucination_types.append(HallucinationType.INTRINSIC)
            hallucinated_spans.extend(intrinsic_result['spans'])
            supporting_evidence.extend(intrinsic_result['evidence'])

        # Check for entity hallucinations
        entity_result = self._detect_entity_hallucination(source, generated)
        if entity_result['detected']:
            hallucination_types.append(HallucinationType.ENTITY)
            hallucinated_spans.extend(entity_result['spans'])
            supporting_evidence.extend(entity_result['evidence'])

        # Check for numerical hallucinations
        numerical_result = self._detect_numerical_hallucination(source, generated)
        if numerical_result['detected']:
            hallucination_types.append(HallucinationType.NUMERICAL)
            hallucinated_spans.extend(numerical_result['spans'])
            supporting_evidence.extend(numerical_result['evidence'])

        # Check for extrinsic hallucinations (unverifiable claims)
        extrinsic_result = self._detect_extrinsic(source, generated, context)
        if extrinsic_result['detected']:
            hallucination_types.append(HallucinationType.EXTRINSIC)
            hallucinated_spans.extend(extrinsic_result['spans'])
            supporting_evidence.extend(extrinsic_result['evidence'])

        # Calculate confidence and severity
        has_hallucination = len(hallucination_types) > 0
        confidence = self._calculate_confidence(hallucinated_spans, generated)
        severity = self._calculate_severity(hallucination_types, hallucinated_spans)

        return HallucinationResult(
            has_hallucination=has_hallucination,
            confidence=confidence,
            hallucination_types=hallucination_types,
            hallucinated_spans=hallucinated_spans,
            supporting_evidence=supporting_evidence,
            severity=severity
        )

    def _detect_intrinsic(self, source: str, generated: str) -> Dict[str, Any]:
        """Detect intrinsic hallucinations (contradictions with source)."""
        result = {'detected': False, 'spans': [], 'evidence': []}

        source_lower = source.lower()
        gen_sentences = [s.strip() for s in generated.split('.') if s.strip()]

        negation_words = {'not', 'no', 'never', 'none', "n't", 'without'}

        for sent in gen_sentences:
            sent_lower = sent.lower()
            sent_words = set(sent_lower.split())

            # Check for contradictory negation patterns
            gen_has_neg = bool(sent_words & negation_words)

            # Find related content in source
            content_words = sent_words - {'the', 'a', 'an', 'is', 'are', 'was', 'were'} - negation_words

            for source_sent in source.split('.'):
                source_sent_lower = source_sent.lower()
                source_words = set(source_sent_lower.split())

                # Check if same topic
                overlap = content_words & source_words
                if len(overlap) >= 3:
                    source_has_neg = bool(source_words & negation_words)

                    # Contradiction: same topic, opposite negation
                    if gen_has_neg != source_has_neg:
                        result['detected'] = True
                        result['spans'].append({
                            'text': sent,
                            'type': 'contradiction',
                            'source_reference': source_sent.strip()
                        })
                        result['evidence'].append(
                            f"Contradicts source: '{sent[:40]}...' vs '{source_sent.strip()[:40]}...'"
                        )
                        break

        return result

    def _detect_entity_hallucination(self, source: str, generated: str) -> Dict[str, Any]:
        """Detect entity hallucinations (wrong entities)."""
        result = {'detected': False, 'spans': [], 'evidence': []}

        # Extract entities (simplified: capitalized words)
        source_entities = {w for w in source.split() if w and w[0].isupper() and len(w) > 1}
        gen_entities = {w for w in generated.split() if w and w[0].isupper() and len(w) > 1}

        # Find entities in generated not in source
        novel_entities = gen_entities - source_entities

        # Filter out common words that might be capitalized (start of sentence)
        common_words = {'The', 'A', 'An', 'This', 'That', 'These', 'Those', 'It', 'He', 'She', 'They', 'We', 'I'}
        suspicious_entities = novel_entities - common_words

        if suspicious_entities:
            # Check if these entities appear in factual claims
            for entity in suspicious_entities:
                for sent in generated.split('.'):
                    if entity in sent:
                        claim_indicators = ['is', 'was', 'are', 'were', 'has', 'have']
                        if any(ci in sent.lower() for ci in claim_indicators):
                            result['detected'] = True
                            result['spans'].append({
                                'text': entity,
                                'type': 'novel_entity',
                                'context': sent.strip()
                            })
                            result['evidence'].append(
                                f"Entity '{entity}' not found in source"
                            )

        return result

    def _detect_numerical_hallucination(self, source: str, generated: str) -> Dict[str, Any]:
        """Detect numerical hallucinations (wrong numbers)."""
        result = {'detected': False, 'spans': [], 'evidence': []}

        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?(?:\s*(?:percent|%|million|billion|thousand))?\b'

        source_numbers = set(re.findall(number_pattern, source.lower()))
        gen_numbers = set(re.findall(number_pattern, generated.lower()))

        # Find numbers in generated not in source
        novel_numbers = gen_numbers - source_numbers

        if novel_numbers:
            for num in novel_numbers:
                # Find context
                for sent in generated.split('.'):
                    if num in sent.lower():
                        result['detected'] = True
                        result['spans'].append({
                            'text': num,
                            'type': 'novel_number',
                            'context': sent.strip()
                        })
                        result['evidence'].append(
                            f"Number '{num}' not found in source"
                        )

        return result

    def _detect_extrinsic(self, source: str, generated: str,
                          context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect extrinsic hallucinations (unverifiable claims)."""
        result = {'detected': False, 'spans': [], 'evidence': []}

        source_lower = source.lower()
        gen_sentences = [s.strip() for s in generated.split('.') if s.strip()]

        # Patterns indicating specific claims
        specific_claim_patterns = [
            r'exactly \d+',
            r'on \d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'at \d{1,2}:\d{2}',
            r'according to [A-Z][a-z]+',
            r'"[^"]{10,}"',  # Quoted text
        ]

        for sent in gen_sentences:
            sent_lower = sent.lower()

            for pattern in specific_claim_patterns:
                matches = re.findall(pattern, sent_lower)
                for match in matches:
                    if match not in source_lower:
                        result['detected'] = True
                        result['spans'].append({
                            'text': match,
                            'type': 'unverifiable_claim',
                            'context': sent
                        })
                        result['evidence'].append(
                            f"Specific claim '{match}' cannot be verified from source"
                        )

        return result

    def _calculate_confidence(self, spans: List[Dict], generated: str) -> float:
        """Calculate detection confidence."""
        if not spans:
            return 0.0

        # Confidence based on number and type of hallucinations
        base_confidence = min(1.0, len(spans) * 0.2)

        # Adjust based on span coverage
        total_span_length = sum(len(s.get('text', '')) for s in spans)
        coverage = total_span_length / len(generated) if generated else 0

        confidence = (base_confidence + coverage) / 2
        return min(1.0, confidence * self.sensitivity_multiplier)

    def _calculate_severity(self, types: List[HallucinationType],
                            spans: List[Dict]) -> str:
        """Calculate hallucination severity."""
        if not types:
            return "none"

        # Severity based on type
        severe_types = {HallucinationType.INTRINSIC, HallucinationType.NUMERICAL}
        has_severe = bool(set(types) & severe_types)

        if has_severe and len(spans) > 2:
            return "high"
        elif has_severe or len(spans) > 3:
            return "medium"
        elif len(spans) > 0:
            return "low"
        else:
            return "none"


class BERTScoreAnalyzer:
    """
    BERT-based semantic similarity scoring.

    Uses BERT embeddings to calculate semantic similarity between
    generated and reference text.
    """

    def __init__(self, model_type: str = "bert-base"):
        """
        Initialize BERTScore analyzer.

        Args:
            model_type: BERT model type to simulate
        """
        self.model_type = model_type

    def calculate(self, reference: str, generated: str,
                  context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calculate BERTScore-style metrics.

        Args:
            reference: Reference text
            generated: Generated text
            context: Optional context

        Returns:
            Dictionary with precision, recall, F1 scores
        """
        # Tokenize (simplified)
        ref_tokens = self._tokenize(reference)
        gen_tokens = self._tokenize(generated)

        if not ref_tokens or not gen_tokens:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        # Calculate token-level similarities (simplified using word overlap)
        # In real BERTScore, this uses contextual embeddings
        precision = self._calculate_precision(gen_tokens, ref_tokens)
        recall = self._calculate_recall(gen_tokens, ref_tokens)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model': self.model_type
        }

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text (simplified)."""
        # Simple word tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        return words

    def _calculate_precision(self, gen_tokens: List[str],
                              ref_tokens: List[str]) -> float:
        """Calculate precision (how much of generated is in reference)."""
        if not gen_tokens:
            return 0.0

        ref_set = set(ref_tokens)
        matches = sum(1 for t in gen_tokens if t in ref_set)
        return matches / len(gen_tokens)

    def _calculate_recall(self, gen_tokens: List[str],
                          ref_tokens: List[str]) -> float:
        """Calculate recall (how much of reference is covered)."""
        if not ref_tokens:
            return 0.0

        gen_set = set(gen_tokens)
        matches = sum(1 for t in ref_tokens if t in gen_set)
        return matches / len(ref_tokens)


class BARTScoreAnalyzer:
    """
    BART-based generation evaluation scoring.

    Uses BART model likelihood to evaluate text generation quality
    and factual consistency.
    """

    def __init__(self, direction: str = "both"):
        """
        Initialize BARTScore analyzer.

        Args:
            direction: Scoring direction (src2tgt, tgt2src, both)
        """
        self.direction = direction

    def calculate(self, source: str, generated: str,
                  context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calculate BARTScore-style metrics.

        Args:
            source: Source text
            generated: Generated text
            context: Optional context

        Returns:
            Dictionary with BARTScore metrics
        """
        # Simplified BARTScore using text overlap and length ratios
        # Real BARTScore uses BART model likelihood

        src_tokens = source.lower().split()
        gen_tokens = generated.lower().split()

        if not src_tokens or not gen_tokens:
            return {'score': 0.0, 'src2tgt': 0.0, 'tgt2src': 0.0}

        # Source to target score (generation adequacy)
        src2tgt = self._calculate_direction_score(src_tokens, gen_tokens)

        # Target to source score (generation precision)
        tgt2src = self._calculate_direction_score(gen_tokens, src_tokens)

        if self.direction == "src2tgt":
            score = src2tgt
        elif self.direction == "tgt2src":
            score = tgt2src
        else:
            score = (src2tgt + tgt2src) / 2

        return {
            'score': score,
            'src2tgt': src2tgt,
            'tgt2src': tgt2src,
            'direction': self.direction
        }

    def _calculate_direction_score(self, from_tokens: List[str],
                                    to_tokens: List[str]) -> float:
        """Calculate directional score."""
        if not from_tokens:
            return 0.0

        to_set = set(to_tokens)
        matches = sum(1 for t in from_tokens if t in to_set)

        # Base overlap score
        overlap = matches / len(from_tokens)

        # Length penalty
        len_ratio = len(to_tokens) / len(from_tokens)
        if len_ratio < 0.5 or len_ratio > 2.0:
            length_penalty = 0.8
        else:
            length_penalty = 1.0

        return overlap * length_penalty


class METEORAnalyzer:
    """
    METEOR metric for text evaluation.

    Evaluates translation/generation quality using word alignment,
    stemming, synonymy, and word order.
    """

    def __init__(self,
                 alpha: float = 0.9,
                 beta: float = 3.0,
                 gamma: float = 0.5):
        """
        Initialize METEOR analyzer.

        Args:
            alpha: Weight for precision vs recall
            beta: Penalty shape parameter
            gamma: Relative weight of fragmentation penalty
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def calculate(self, reference: str, generated: str,
                  context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calculate METEOR score.

        Args:
            reference: Reference text
            generated: Generated text (hypothesis)
            context: Optional context

        Returns:
            Dictionary with METEOR metrics
        """
        # Tokenize
        ref_tokens = self._tokenize(reference)
        gen_tokens = self._tokenize(generated)

        if not ref_tokens or not gen_tokens:
            return {'meteor': 0.0, 'precision': 0.0, 'recall': 0.0, 'penalty': 0.0}

        # Calculate alignment (simplified: exact matches)
        alignment = self._align(gen_tokens, ref_tokens)

        # Calculate precision and recall
        precision = len(alignment) / len(gen_tokens) if gen_tokens else 0
        recall = len(alignment) / len(ref_tokens) if ref_tokens else 0

        # Calculate F-mean
        if precision + recall > 0:
            f_mean = (precision * recall) / (self.alpha * precision + (1 - self.alpha) * recall)
        else:
            f_mean = 0.0

        # Calculate fragmentation penalty
        chunks = self._count_chunks(alignment, gen_tokens, ref_tokens)
        if alignment:
            frag = chunks / len(alignment)
        else:
            frag = 0.0

        penalty = self.gamma * (frag ** self.beta)

        # Final METEOR score
        meteor = f_mean * (1 - penalty)

        return {
            'meteor': meteor,
            'precision': precision,
            'recall': recall,
            'f_mean': f_mean,
            'penalty': penalty,
            'chunks': chunks,
            'alignments': len(alignment)
        }

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        return re.findall(r'\b\w+\b', text.lower())

    def _align(self, gen_tokens: List[str],
               ref_tokens: List[str]) -> List[Tuple[int, int]]:
        """Align tokens between generated and reference."""
        alignment = []
        used_ref = set()

        for i, gen_tok in enumerate(gen_tokens):
            for j, ref_tok in enumerate(ref_tokens):
                if j not in used_ref and gen_tok == ref_tok:
                    alignment.append((i, j))
                    used_ref.add(j)
                    break

        return alignment

    def _count_chunks(self, alignment: List[Tuple[int, int]],
                      gen_tokens: List[str],
                      ref_tokens: List[str]) -> int:
        """Count contiguous chunks in alignment."""
        if not alignment:
            return 0

        # Sort alignment by hypothesis position
        sorted_align = sorted(alignment, key=lambda x: x[0])

        chunks = 1
        for i in range(1, len(sorted_align)):
            # Check if consecutive in both
            prev_gen, prev_ref = sorted_align[i-1]
            curr_gen, curr_ref = sorted_align[i]

            if curr_gen != prev_gen + 1 or curr_ref != prev_ref + 1:
                chunks += 1

        return chunks


class FactualConsistencyEvaluator:
    """
    Comprehensive factual consistency evaluator.

    Combines multiple evaluation methods for thorough
    factual consistency assessment.
    """

    def __init__(self,
                 methods: Optional[List[str]] = None,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize comprehensive evaluator.

        Args:
            methods: Evaluation methods to use
            weights: Weights for each method
        """
        self.methods = methods or ['questeval', 'factcc', 'hallucination', 'bertscore', 'meteor']
        self.weights = weights or {
            'questeval': 0.25,
            'factcc': 0.25,
            'hallucination': 0.20,
            'bertscore': 0.15,
            'meteor': 0.15
        }

        # Initialize analyzers
        self.questeval = QuestEvalAnalyzer()
        self.factcc = FactCCAnalyzer()
        self.hallucination = HallucinationDetector()
        self.bertscore = BERTScoreAnalyzer()
        self.bartscore = BARTScoreAnalyzer()
        self.meteor = METEORAnalyzer()

    def evaluate(self, source: str, generated: str,
                 context: Optional[Dict[str, Any]] = None) -> FactualEvaluationResult:
        """
        Perform comprehensive factual evaluation.

        Args:
            source: Source/reference text
            generated: Generated text to evaluate
            context: Optional evaluation context

        Returns:
            Comprehensive FactualEvaluationResult
        """
        method_scores = {}

        # QuestEval evaluation
        if 'questeval' in self.methods:
            questeval_result = self.questeval.evaluate(source, generated, context)
            method_scores['questeval'] = questeval_result.score
            consistency_score = questeval_result
        else:
            consistency_score = None

        # FactCC evaluation
        if 'factcc' in self.methods:
            factcc_result = self.factcc.evaluate(source, generated, context)
            method_scores['factcc'] = factcc_result.score
            if consistency_score is None:
                consistency_score = factcc_result

        # Hallucination detection
        if 'hallucination' in self.methods:
            hall_result = self.hallucination.detect(source, generated, context)
            # Invert: no hallucination = 1.0
            method_scores['hallucination'] = 1.0 - hall_result.confidence if hall_result.has_hallucination else 1.0
        else:
            hall_result = HallucinationResult(has_hallucination=False, confidence=0.0)

        # BERTScore evaluation
        if 'bertscore' in self.methods:
            bert_result = self.bertscore.calculate(source, generated, context)
            method_scores['bertscore'] = bert_result['f1']

        # BARTScore evaluation
        if 'bartscore' in self.methods:
            bart_result = self.bartscore.calculate(source, generated, context)
            method_scores['bartscore'] = bart_result['score']

        # METEOR evaluation
        if 'meteor' in self.methods:
            meteor_result = self.meteor.calculate(source, generated, context)
            method_scores['meteor'] = meteor_result['meteor']

        # Calculate overall score
        overall_score = self._calculate_overall_score(method_scores)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            method_scores, consistency_score, hall_result
        )

        # Create default consistency score if not computed
        if consistency_score is None:
            consistency_score = ConsistencyScore(
                score=overall_score,
                level=self._score_to_level(overall_score),
                method='combined'
            )

        return FactualEvaluationResult(
            overall_score=overall_score,
            consistency_score=consistency_score,
            hallucination_result=hall_result,
            method_scores=method_scores,
            recommendations=recommendations
        )

    def _calculate_overall_score(self, method_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score."""
        if not method_scores:
            return 0.0

        weighted_sum = 0.0
        weight_sum = 0.0

        for method, score in method_scores.items():
            weight = self.weights.get(method, 1.0)
            weighted_sum += score * weight
            weight_sum += weight

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0

    def _score_to_level(self, score: float) -> ConsistencyLevel:
        """Convert score to consistency level."""
        if score >= 0.9:
            return ConsistencyLevel.FULLY_CONSISTENT
        elif score >= 0.7:
            return ConsistencyLevel.MOSTLY_CONSISTENT
        elif score >= 0.5:
            return ConsistencyLevel.PARTIALLY_CONSISTENT
        elif score >= 0.3:
            return ConsistencyLevel.INCONSISTENT
        else:
            return ConsistencyLevel.CONTRADICTORY

    def _generate_recommendations(self,
                                   method_scores: Dict[str, float],
                                   consistency: Optional[ConsistencyScore],
                                   hallucination: HallucinationResult) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        # Hallucination recommendations
        if hallucination.has_hallucination:
            if hallucination.severity == 'high':
                recommendations.append("Critical: Review and correct hallucinated content")
            else:
                recommendations.append("Verify unsubstantiated claims against source material")

        # Low scoring method recommendations
        for method, score in method_scores.items():
            if score < 0.5:
                if method == 'questeval':
                    recommendations.append("Improve answer consistency for key questions")
                elif method == 'factcc':
                    recommendations.append("Address factual inconsistencies with source")
                elif method == 'bertscore':
                    recommendations.append("Improve semantic alignment with reference")
                elif method == 'meteor':
                    recommendations.append("Improve lexical and structural alignment")

        # Consistency recommendations
        if consistency and consistency.level in [ConsistencyLevel.INCONSISTENT, ConsistencyLevel.CONTRADICTORY]:
            recommendations.append("Major revision needed for factual consistency")

        return recommendations

    def compare_generations(self,
                            source: str,
                            generation1: str,
                            generation2: str,
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compare two generations for factual consistency.

        Args:
            source: Source text
            generation1: First generated text
            generation2: Second generated text
            context: Optional context

        Returns:
            Comparison results
        """
        result1 = self.evaluate(source, generation1, context)
        result2 = self.evaluate(source, generation2, context)

        comparison = {
            'generation1_score': result1.overall_score,
            'generation2_score': result2.overall_score,
            'difference': result1.overall_score - result2.overall_score,
            'winner': 'generation1' if result1.overall_score > result2.overall_score else 'generation2',
            'generation1_has_hallucination': result1.hallucination_result.has_hallucination,
            'generation2_has_hallucination': result2.hallucination_result.has_hallucination,
            'method_comparison': {}
        }

        for method in self.methods:
            if method in result1.method_scores and method in result2.method_scores:
                comparison['method_comparison'][method] = {
                    'generation1': result1.method_scores[method],
                    'generation2': result2.method_scores[method],
                    'difference': result1.method_scores[method] - result2.method_scores[method]
                }

        return comparison

    def generate_report(self, result: FactualEvaluationResult) -> str:
        """
        Generate evaluation report.

        Args:
            result: Evaluation result

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "FACTUAL CONSISTENCY EVALUATION REPORT",
            "=" * 60,
            "",
            f"Overall Score: {result.overall_score:.3f}",
            f"Consistency Level: {result.consistency_score.level.name}",
            f"Has Hallucination: {result.hallucination_result.has_hallucination}",
            "",
            "METHOD SCORES",
            "-" * 40
        ]

        for method, score in result.method_scores.items():
            lines.append(f"  {method}: {score:.3f}")

        if result.hallucination_result.has_hallucination:
            lines.extend([
                "",
                "HALLUCINATION DETAILS",
                "-" * 40,
                f"  Confidence: {result.hallucination_result.confidence:.3f}",
                f"  Severity: {result.hallucination_result.severity}",
                f"  Types: {[t.value for t in result.hallucination_result.hallucination_types]}"
            ])

        if result.recommendations:
            lines.extend([
                "",
                "RECOMMENDATIONS",
                "-" * 40
            ])
            for rec in result.recommendations:
                lines.append(f"  • {rec}")

        lines.append("=" * 60)

        return "\n".join(lines)


# Convenience functions
def check_factual_consistency(source: str, generated: str) -> float:
    """Quick factual consistency check."""
    evaluator = FactualConsistencyEvaluator(methods=['factcc', 'questeval'])
    result = evaluator.evaluate(source, generated)
    return result.overall_score


def detect_hallucinations(source: str, generated: str) -> HallucinationResult:
    """Quick hallucination detection."""
    detector = HallucinationDetector()
    return detector.detect(source, generated)


def evaluate_factual_quality(source: str, generated: str) -> FactualEvaluationResult:
    """Comprehensive factual quality evaluation."""
    evaluator = FactualConsistencyEvaluator()
    return evaluator.evaluate(source, generated)


__all__ = [
    # Enums
    'ConsistencyLevel',
    'HallucinationType',
    'EntailmentLabel',

    # Data classes
    'ConsistencyScore',
    'HallucinationResult',
    'QuestionAnswerPair',
    'FactualEvaluationResult',

    # Analyzers
    'QuestEvalAnalyzer',
    'FactCCAnalyzer',
    'HallucinationDetector',
    'BERTScoreAnalyzer',
    'BARTScoreAnalyzer',
    'METEORAnalyzer',
    'FactualConsistencyEvaluator',

    # Convenience functions
    'check_factual_consistency',
    'detect_hallucinations',
    'evaluate_factual_quality',
]
