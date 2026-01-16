"""
Fidelity Analysis Module for AI/GenAI Evaluation

This module provides comprehensive fidelity analysis for evaluating generative AI
outputs, including Inception Score (IS), Frechet Inception Distance (FID),
F1-Score analysis, and various fidelity evaluation frameworks.

Fidelity in AI refers to how accurately and faithfully a model reproduces or
generates content that matches expected distributions, quality standards,
and semantic meaning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from abc import ABC, abstractmethod
import math
from datetime import datetime


class FidelityType(Enum):
    """Types of fidelity evaluation."""
    PERCEPTUAL = "perceptual"
    STATISTICAL = "statistical"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    TEMPORAL = "temporal"
    DISTRIBUTIONAL = "distributional"
    RECONSTRUCTION = "reconstruction"
    GENERATIVE = "generative"


class QualityMetricType(Enum):
    """Types of quality metrics."""
    INCEPTION_SCORE = "inception_score"
    FID = "frechet_inception_distance"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    CLIP_SCORE = "clip_score"
    LPIPS = "learned_perceptual_image_patch_similarity"
    SSIM = "structural_similarity_index"
    PSNR = "peak_signal_to_noise_ratio"
    BLEU = "bilingual_evaluation_understudy"
    ROUGE = "recall_oriented_understudy_gisting_evaluation"
    METEOR = "metric_for_evaluation_of_translation"
    BERTSCORE = "bert_score"


class FidelityDimension(Enum):
    """Dimensions for fidelity evaluation."""
    QUALITY = "quality"
    DIVERSITY = "diversity"
    CONSISTENCY = "consistency"
    AUTHENTICITY = "authenticity"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    NOVELTY = "novelty"
    FAITHFULNESS = "faithfulness"


@dataclass
class InceptionScoreResult:
    """Results from Inception Score calculation."""
    score: float
    score_std: float
    quality_component: float
    diversity_component: float
    num_samples: int
    split_scores: List[float]
    confidence_interval: Tuple[float, float]
    interpretation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "score_std": self.score_std,
            "quality_component": self.quality_component,
            "diversity_component": self.diversity_component,
            "num_samples": self.num_samples,
            "split_scores": self.split_scores,
            "confidence_interval": self.confidence_interval,
            "interpretation": self.interpretation,
            "timestamp": self.timestamp
        }


@dataclass
class FIDResult:
    """Results from Frechet Inception Distance calculation."""
    distance: float
    real_mean: List[float]
    real_covariance_trace: float
    generated_mean: List[float]
    generated_covariance_trace: float
    mean_difference_norm: float
    covariance_term: float
    num_real_samples: int
    num_generated_samples: int
    feature_dimension: int
    interpretation: str
    quality_assessment: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "distance": self.distance,
            "real_mean_summary": {"length": len(self.real_mean)},
            "real_covariance_trace": self.real_covariance_trace,
            "generated_mean_summary": {"length": len(self.generated_mean)},
            "generated_covariance_trace": self.generated_covariance_trace,
            "mean_difference_norm": self.mean_difference_norm,
            "covariance_term": self.covariance_term,
            "num_real_samples": self.num_real_samples,
            "num_generated_samples": self.num_generated_samples,
            "feature_dimension": self.feature_dimension,
            "interpretation": self.interpretation,
            "quality_assessment": self.quality_assessment,
            "timestamp": self.timestamp
        }


@dataclass
class F1ScoreResult:
    """Results from F1 Score calculation."""
    f1_score: float
    precision: float
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    support: int
    balanced_accuracy: float
    matthews_correlation: float
    interpretation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "support": self.support,
            "balanced_accuracy": self.balanced_accuracy,
            "matthews_correlation": self.matthews_correlation,
            "interpretation": self.interpretation,
            "timestamp": self.timestamp
        }


@dataclass
class FidelityAssessment:
    """Comprehensive fidelity assessment result."""
    fidelity_type: FidelityType
    overall_score: float
    dimension_scores: Dict[str, float]
    quality_metrics: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    confidence_level: float
    methodology: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fidelity_type": self.fidelity_type.value,
            "overall_score": self.overall_score,
            "dimension_scores": self.dimension_scores,
            "quality_metrics": self.quality_metrics,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
            "confidence_level": self.confidence_level,
            "methodology": self.methodology,
            "timestamp": self.timestamp
        }


class BaseFidelityAnalyzer(ABC):
    """Abstract base class for fidelity analyzers."""

    @abstractmethod
    def analyze(self, data: Any) -> Any:
        """Perform fidelity analysis."""
        pass

    @abstractmethod
    def get_analyzer_type(self) -> str:
        """Return the type of analyzer."""
        pass


class InceptionScoreAnalyzer(BaseFidelityAnalyzer):
    """
    Analyzer for calculating and interpreting Inception Score (IS).

    Inception Score measures:
    1. Quality: How confident the classifier is about each image (low entropy of p(y|x))
    2. Diversity: How varied the generated images are (high entropy of p(y))

    IS = exp(E[KL(p(y|x) || p(y))])

    Higher scores indicate better quality AND diversity.
    """

    def __init__(self, num_splits: int = 10):
        self.num_splits = num_splits
        self.score_ranges = {
            "excellent": (300, float('inf')),
            "very_good": (200, 300),
            "good": (100, 200),
            "moderate": (50, 100),
            "fair": (20, 50),
            "poor": (0, 20)
        }

    def get_analyzer_type(self) -> str:
        return "inception_score"

    def analyze(self, predictions: List[List[float]]) -> InceptionScoreResult:
        """
        Calculate Inception Score from class predictions.

        Args:
            predictions: List of softmax probability vectors for each sample

        Returns:
            InceptionScoreResult with score and interpretation
        """
        if not predictions:
            return self._empty_result()

        n = len(predictions)
        split_scores = []

        # Calculate score for each split
        split_size = n // self.num_splits
        for i in range(self.num_splits):
            start = i * split_size
            end = start + split_size if i < self.num_splits - 1 else n
            split_preds = predictions[start:end]

            if split_preds:
                score = self._calculate_is_for_split(split_preds)
                split_scores.append(score)

        # Calculate overall statistics
        mean_score = sum(split_scores) / len(split_scores) if split_scores else 0
        variance = sum((s - mean_score) ** 2 for s in split_scores) / len(split_scores) if split_scores else 0
        std_score = math.sqrt(variance)

        # Calculate confidence interval (95%)
        confidence_margin = 1.96 * std_score / math.sqrt(len(split_scores)) if split_scores else 0
        confidence_interval = (mean_score - confidence_margin, mean_score + confidence_margin)

        # Estimate quality and diversity components
        quality_component = self._estimate_quality_component(predictions)
        diversity_component = self._estimate_diversity_component(predictions)

        interpretation = self._interpret_score(mean_score)

        return InceptionScoreResult(
            score=mean_score,
            score_std=std_score,
            quality_component=quality_component,
            diversity_component=diversity_component,
            num_samples=n,
            split_scores=split_scores,
            confidence_interval=confidence_interval,
            interpretation=interpretation
        )

    def _calculate_is_for_split(self, predictions: List[List[float]]) -> float:
        """Calculate IS for a split of predictions."""
        n = len(predictions)
        if n == 0:
            return 0.0

        # Calculate marginal p(y)
        num_classes = len(predictions[0])
        marginal = [0.0] * num_classes
        for pred in predictions:
            for j, p in enumerate(pred):
                marginal[j] += p / n

        # Calculate KL divergence for each sample
        kl_sum = 0.0
        for pred in predictions:
            for j, p in enumerate(pred):
                if p > 1e-10 and marginal[j] > 1e-10:
                    kl_sum += p * math.log(p / marginal[j])

        kl_mean = kl_sum / n
        return math.exp(kl_mean)

    def _estimate_quality_component(self, predictions: List[List[float]]) -> float:
        """Estimate quality component from prediction confidence."""
        if not predictions:
            return 0.0

        # Higher confidence = lower entropy = better quality
        total_confidence = 0.0
        for pred in predictions:
            max_prob = max(pred)
            total_confidence += max_prob

        return total_confidence / len(predictions)

    def _estimate_diversity_component(self, predictions: List[List[float]]) -> float:
        """Estimate diversity component from class distribution."""
        if not predictions:
            return 0.0

        num_classes = len(predictions[0])
        class_counts = [0.0] * num_classes

        for pred in predictions:
            max_idx = pred.index(max(pred))
            class_counts[max_idx] += 1

        # Calculate entropy of class distribution
        n = len(predictions)
        entropy = 0.0
        for count in class_counts:
            if count > 0:
                p = count / n
                entropy -= p * math.log(p)

        # Normalize by max entropy
        max_entropy = math.log(num_classes) if num_classes > 0 else 1
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _interpret_score(self, score: float) -> str:
        """Provide interpretation of the Inception Score."""
        for level, (low, high) in self.score_ranges.items():
            if low <= score < high:
                interpretations = {
                    "excellent": f"Excellent IS ({score:.2f}): Generated samples show exceptional quality and diversity, comparable to or exceeding real data distributions.",
                    "very_good": f"Very Good IS ({score:.2f}): High quality and diverse outputs indicating strong generative performance.",
                    "good": f"Good IS ({score:.2f}): Generated samples demonstrate good quality and reasonable diversity.",
                    "moderate": f"Moderate IS ({score:.2f}): Acceptable quality but may lack diversity or have some quality issues.",
                    "fair": f"Fair IS ({score:.2f}): Limited quality or diversity; model may need improvement.",
                    "poor": f"Poor IS ({score:.2f}): Low quality and/or diversity; significant model improvements needed."
                }
                return interpretations.get(level, f"IS: {score:.2f}")
        return f"IS: {score:.2f}"

    def _empty_result(self) -> InceptionScoreResult:
        """Return empty result for invalid input."""
        return InceptionScoreResult(
            score=0.0,
            score_std=0.0,
            quality_component=0.0,
            diversity_component=0.0,
            num_samples=0,
            split_scores=[],
            confidence_interval=(0.0, 0.0),
            interpretation="No data provided for analysis"
        )


class FrechetInceptionDistanceAnalyzer(BaseFidelityAnalyzer):
    """
    Analyzer for calculating and interpreting Frechet Inception Distance (FID).

    FID compares the distribution of generated images to real images using
    features extracted from an inception network. It measures both:
    - Quality: How realistic individual samples are
    - Diversity: How well the generated distribution covers the real distribution

    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*(Sigma_r * Sigma_g)^(1/2))

    Lower FID indicates better similarity to real data distribution.
    """

    def __init__(self):
        self.quality_thresholds = {
            "excellent": (0, 10),
            "very_good": (10, 25),
            "good": (25, 50),
            "moderate": (50, 100),
            "fair": (100, 200),
            "poor": (200, float('inf'))
        }

    def get_analyzer_type(self) -> str:
        return "frechet_inception_distance"

    def analyze(
        self,
        real_features: List[List[float]],
        generated_features: List[List[float]]
    ) -> FIDResult:
        """
        Calculate FID between real and generated feature distributions.

        Args:
            real_features: Features extracted from real samples
            generated_features: Features extracted from generated samples

        Returns:
            FIDResult with distance and interpretation
        """
        if not real_features or not generated_features:
            return self._empty_result()

        # Calculate statistics for real features
        real_mean = self._calculate_mean(real_features)
        real_cov = self._calculate_covariance(real_features, real_mean)
        real_cov_trace = sum(real_cov[i][i] for i in range(len(real_cov)))

        # Calculate statistics for generated features
        gen_mean = self._calculate_mean(generated_features)
        gen_cov = self._calculate_covariance(generated_features, gen_mean)
        gen_cov_trace = sum(gen_cov[i][i] for i in range(len(gen_cov)))

        # Calculate FID components
        mean_diff_norm = self._calculate_squared_norm_difference(real_mean, gen_mean)
        cov_term = self._calculate_covariance_term(real_cov, gen_cov)

        fid = mean_diff_norm + cov_term

        interpretation = self._interpret_fid(fid)
        quality_assessment = self._assess_quality(fid, mean_diff_norm, cov_term)

        return FIDResult(
            distance=fid,
            real_mean=real_mean,
            real_covariance_trace=real_cov_trace,
            generated_mean=gen_mean,
            generated_covariance_trace=gen_cov_trace,
            mean_difference_norm=mean_diff_norm,
            covariance_term=cov_term,
            num_real_samples=len(real_features),
            num_generated_samples=len(generated_features),
            feature_dimension=len(real_features[0]) if real_features else 0,
            interpretation=interpretation,
            quality_assessment=quality_assessment
        )

    def _calculate_mean(self, features: List[List[float]]) -> List[float]:
        """Calculate mean of feature vectors."""
        n = len(features)
        dim = len(features[0])
        mean = [0.0] * dim

        for feature in features:
            for i, val in enumerate(feature):
                mean[i] += val / n

        return mean

    def _calculate_covariance(
        self,
        features: List[List[float]],
        mean: List[float]
    ) -> List[List[float]]:
        """Calculate covariance matrix of features."""
        n = len(features)
        dim = len(features[0])
        cov = [[0.0] * dim for _ in range(dim)]

        for feature in features:
            centered = [feature[i] - mean[i] for i in range(dim)]
            for i in range(dim):
                for j in range(dim):
                    cov[i][j] += centered[i] * centered[j] / (n - 1)

        return cov

    def _calculate_squared_norm_difference(
        self,
        mean1: List[float],
        mean2: List[float]
    ) -> float:
        """Calculate squared L2 norm of mean difference."""
        return sum((m1 - m2) ** 2 for m1, m2 in zip(mean1, mean2))

    def _calculate_covariance_term(
        self,
        cov1: List[List[float]],
        cov2: List[List[float]]
    ) -> float:
        """Calculate the covariance term of FID (simplified approximation)."""
        dim = len(cov1)

        trace1 = sum(cov1[i][i] for i in range(dim))
        trace2 = sum(cov2[i][i] for i in range(dim))

        # Simplified: approximate sqrt(cov1 * cov2) trace
        # In practice, this requires proper matrix operations
        sqrt_trace = math.sqrt(trace1 * trace2) if trace1 > 0 and trace2 > 0 else 0

        return trace1 + trace2 - 2 * sqrt_trace

    def _interpret_fid(self, fid: float) -> str:
        """Provide interpretation of the FID score."""
        for level, (low, high) in self.quality_thresholds.items():
            if low <= fid < high:
                interpretations = {
                    "excellent": f"Excellent FID ({fid:.2f}): Generated distribution is nearly indistinguishable from real data.",
                    "very_good": f"Very Good FID ({fid:.2f}): High fidelity generation with minor distribution differences.",
                    "good": f"Good FID ({fid:.2f}): Generated samples closely match real data characteristics.",
                    "moderate": f"Moderate FID ({fid:.2f}): Noticeable differences between generated and real distributions.",
                    "fair": f"Fair FID ({fid:.2f}): Significant distribution gap; model may need improvement.",
                    "poor": f"Poor FID ({fid:.2f}): Large distribution mismatch; substantial improvements needed."
                }
                return interpretations.get(level, f"FID: {fid:.2f}")
        return f"FID: {fid:.2f}"

    def _assess_quality(
        self,
        fid: float,
        mean_diff: float,
        cov_term: float
    ) -> str:
        """Provide detailed quality assessment."""
        total = mean_diff + cov_term if cov_term != 0 else 1
        mean_contribution = mean_diff / total * 100 if total > 0 else 0
        cov_contribution = cov_term / total * 100 if total > 0 else 0

        assessment_parts = [
            f"FID decomposition: Mean difference contributes {mean_contribution:.1f}%, "
            f"covariance term contributes {cov_contribution:.1f}%."
        ]

        if mean_contribution > 70:
            assessment_parts.append(
                "High mean contribution suggests generated samples cluster differently than real data."
            )
        elif cov_contribution > 70:
            assessment_parts.append(
                "High covariance contribution suggests different variance/spread in generated samples."
            )
        else:
            assessment_parts.append(
                "Balanced contribution indicates both location and spread differences."
            )

        return " ".join(assessment_parts)

    def _empty_result(self) -> FIDResult:
        """Return empty result for invalid input."""
        return FIDResult(
            distance=float('inf'),
            real_mean=[],
            real_covariance_trace=0.0,
            generated_mean=[],
            generated_covariance_trace=0.0,
            mean_difference_norm=0.0,
            covariance_term=0.0,
            num_real_samples=0,
            num_generated_samples=0,
            feature_dimension=0,
            interpretation="Insufficient data for FID calculation",
            quality_assessment="N/A"
        )


class F1ScoreAnalyzer(BaseFidelityAnalyzer):
    """
    Analyzer for calculating and interpreting F1 Score and related metrics.

    F1 Score is the harmonic mean of precision and recall:
    F1 = 2 * (precision * recall) / (precision + recall)

    Where:
    - Precision = TP / (TP + FP) - accuracy of positive predictions
    - Recall = TP / (TP + FN) - coverage of actual positives
    """

    def __init__(self):
        self.score_thresholds = {
            "excellent": (0.9, 1.0),
            "very_good": (0.8, 0.9),
            "good": (0.7, 0.8),
            "moderate": (0.6, 0.7),
            "fair": (0.5, 0.6),
            "poor": (0.0, 0.5)
        }

    def get_analyzer_type(self) -> str:
        return "f1_score"

    def analyze(
        self,
        true_labels: List[int],
        predicted_labels: List[int],
        positive_class: int = 1
    ) -> F1ScoreResult:
        """
        Calculate F1 score from true and predicted labels.

        Args:
            true_labels: Ground truth labels
            predicted_labels: Model predictions
            positive_class: Label value for positive class

        Returns:
            F1ScoreResult with detailed metrics
        """
        if not true_labels or not predicted_labels:
            return self._empty_result()

        if len(true_labels) != len(predicted_labels):
            return self._empty_result()

        # Calculate confusion matrix components
        tp, fp, fn, tn = 0, 0, 0, 0
        for true, pred in zip(true_labels, predicted_labels):
            if true == positive_class and pred == positive_class:
                tp += 1
            elif true != positive_class and pred == positive_class:
                fp += 1
            elif true == positive_class and pred != positive_class:
                fn += 1
            else:
                tn += 1

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Additional metrics
        accuracy = (tp + tn) / len(true_labels)
        balanced_accuracy = (recall + tn / (tn + fp) if (tn + fp) > 0 else 0) / 2

        # Matthews Correlation Coefficient
        mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0

        interpretation = self._interpret_f1(f1, precision, recall)

        return F1ScoreResult(
            f1_score=f1,
            precision=precision,
            recall=recall,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            support=tp + fn,
            balanced_accuracy=balanced_accuracy,
            matthews_correlation=mcc,
            interpretation=interpretation
        )

    def _interpret_f1(self, f1: float, precision: float, recall: float) -> str:
        """Provide interpretation of F1 score."""
        for level, (low, high) in self.score_thresholds.items():
            if low <= f1 < high:
                base_interp = {
                    "excellent": f"Excellent F1 ({f1:.3f}): Outstanding balance of precision and recall.",
                    "very_good": f"Very Good F1 ({f1:.3f}): Strong predictive performance.",
                    "good": f"Good F1 ({f1:.3f}): Solid performance with room for improvement.",
                    "moderate": f"Moderate F1 ({f1:.3f}): Acceptable but could be improved.",
                    "fair": f"Fair F1 ({f1:.3f}): Performance at threshold level.",
                    "poor": f"Poor F1 ({f1:.3f}): Significant improvements needed."
                }

                interp = base_interp.get(level, f"F1: {f1:.3f}")

                # Add precision/recall balance analysis
                if precision > 0 and recall > 0:
                    ratio = precision / recall
                    if ratio > 1.5:
                        interp += " Precision-biased (may be missing positive cases)."
                    elif ratio < 0.67:
                        interp += " Recall-biased (may have many false positives)."
                    else:
                        interp += " Well-balanced precision and recall."

                return interp
        return f"F1: {f1:.3f}"

    def _empty_result(self) -> F1ScoreResult:
        """Return empty result for invalid input."""
        return F1ScoreResult(
            f1_score=0.0,
            precision=0.0,
            recall=0.0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            true_negatives=0,
            support=0,
            balanced_accuracy=0.0,
            matthews_correlation=0.0,
            interpretation="Insufficient data for F1 calculation"
        )


class PerceptualFidelityAnalyzer(BaseFidelityAnalyzer):
    """
    Analyzer for perceptual fidelity evaluation.

    Evaluates how well generated content matches human perception of quality,
    including structural similarity, perceptual loss, and visual coherence.
    """

    def __init__(self):
        self.metrics = [
            "structural_similarity",
            "perceptual_loss",
            "visual_coherence",
            "artifact_detection",
            "naturalness",
            "sharpness"
        ]

    def get_analyzer_type(self) -> str:
        return "perceptual_fidelity"

    def analyze(
        self,
        quality_scores: Dict[str, float],
        reference_available: bool = True
    ) -> FidelityAssessment:
        """
        Perform perceptual fidelity analysis.

        Args:
            quality_scores: Dictionary of quality metric scores
            reference_available: Whether reference data is available

        Returns:
            FidelityAssessment with comprehensive evaluation
        """
        # Normalize and process scores
        dimension_scores = {}
        for metric in self.metrics:
            if metric in quality_scores:
                dimension_scores[metric] = min(1.0, max(0.0, quality_scores[metric]))
            else:
                dimension_scores[metric] = 0.5  # Default neutral score

        # Calculate overall score
        weights = {
            "structural_similarity": 0.2,
            "perceptual_loss": 0.2,
            "visual_coherence": 0.15,
            "artifact_detection": 0.15,
            "naturalness": 0.15,
            "sharpness": 0.15
        }

        overall_score = sum(
            dimension_scores.get(m, 0.5) * weights.get(m, 0.1)
            for m in self.metrics
        )

        # Identify strengths and weaknesses
        strengths = [m for m, s in dimension_scores.items() if s >= 0.7]
        weaknesses = [m for m, s in dimension_scores.items() if s < 0.5]

        # Generate recommendations
        recommendations = self._generate_recommendations(dimension_scores, weaknesses)

        return FidelityAssessment(
            fidelity_type=FidelityType.PERCEPTUAL,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            quality_metrics=quality_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            confidence_level=0.8 if reference_available else 0.6,
            methodology="Perceptual quality metrics with weighted aggregation"
        )

    def _generate_recommendations(
        self,
        scores: Dict[str, float],
        weaknesses: List[str]
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        recommendation_map = {
            "structural_similarity": "Improve structural preservation in generation process",
            "perceptual_loss": "Reduce perceptual loss through better feature matching",
            "visual_coherence": "Enhance visual coherence with consistency mechanisms",
            "artifact_detection": "Implement artifact reduction techniques",
            "naturalness": "Improve naturalness through better training data or augmentation",
            "sharpness": "Address blurriness issues with sharper generation methods"
        }

        for weakness in weaknesses:
            if weakness in recommendation_map:
                recommendations.append(recommendation_map[weakness])

        if not recommendations:
            recommendations.append("Maintain current quality levels and monitor for regression")

        return recommendations


class SemanticFidelityAnalyzer(BaseFidelityAnalyzer):
    """
    Analyzer for semantic fidelity evaluation.

    Evaluates how well generated content preserves semantic meaning,
    relevance, and coherence with respect to inputs or references.
    """

    def __init__(self):
        self.semantic_dimensions = [
            "meaning_preservation",
            "factual_accuracy",
            "contextual_relevance",
            "logical_coherence",
            "semantic_consistency",
            "intent_alignment"
        ]

    def get_analyzer_type(self) -> str:
        return "semantic_fidelity"

    def analyze(
        self,
        semantic_scores: Dict[str, float],
        context: Optional[str] = None
    ) -> FidelityAssessment:
        """
        Perform semantic fidelity analysis.

        Args:
            semantic_scores: Dictionary of semantic evaluation scores
            context: Optional context for evaluation

        Returns:
            FidelityAssessment with semantic evaluation
        """
        dimension_scores = {}
        for dim in self.semantic_dimensions:
            if dim in semantic_scores:
                dimension_scores[dim] = min(1.0, max(0.0, semantic_scores[dim]))
            else:
                dimension_scores[dim] = 0.5

        # Calculate weighted overall score
        weights = {
            "meaning_preservation": 0.25,
            "factual_accuracy": 0.2,
            "contextual_relevance": 0.15,
            "logical_coherence": 0.15,
            "semantic_consistency": 0.15,
            "intent_alignment": 0.1
        }

        overall_score = sum(
            dimension_scores.get(d, 0.5) * weights.get(d, 0.1)
            for d in self.semantic_dimensions
        )

        strengths = [d for d, s in dimension_scores.items() if s >= 0.7]
        weaknesses = [d for d, s in dimension_scores.items() if s < 0.5]

        recommendations = self._generate_recommendations(weaknesses)

        return FidelityAssessment(
            fidelity_type=FidelityType.SEMANTIC,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            quality_metrics=semantic_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            confidence_level=0.75,
            methodology="Semantic similarity and coherence analysis"
        )

    def _generate_recommendations(self, weaknesses: List[str]) -> List[str]:
        """Generate semantic improvement recommendations."""
        recommendations = []

        recommendation_map = {
            "meaning_preservation": "Enhance semantic encoding to better preserve meaning",
            "factual_accuracy": "Implement fact-checking mechanisms or knowledge grounding",
            "contextual_relevance": "Improve context understanding with better attention mechanisms",
            "logical_coherence": "Add logical consistency checks to generation pipeline",
            "semantic_consistency": "Ensure consistent semantic representation throughout",
            "intent_alignment": "Better align generation with user/input intent"
        }

        for weakness in weaknesses:
            if weakness in recommendation_map:
                recommendations.append(recommendation_map[weakness])

        if not recommendations:
            recommendations.append("Semantic fidelity is satisfactory; continue monitoring")

        return recommendations


class GenerativeFidelityAnalyzer(BaseFidelityAnalyzer):
    """
    Comprehensive analyzer for generative model fidelity.

    Combines multiple metrics (IS, FID, perceptual, semantic) for
    holistic evaluation of generative model quality.
    """

    def __init__(self):
        self.is_analyzer = InceptionScoreAnalyzer()
        self.fid_analyzer = FrechetInceptionDistanceAnalyzer()
        self.perceptual_analyzer = PerceptualFidelityAnalyzer()
        self.semantic_analyzer = SemanticFidelityAnalyzer()

    def get_analyzer_type(self) -> str:
        return "generative_fidelity"

    def analyze(
        self,
        predictions: Optional[List[List[float]]] = None,
        real_features: Optional[List[List[float]]] = None,
        generated_features: Optional[List[List[float]]] = None,
        perceptual_scores: Optional[Dict[str, float]] = None,
        semantic_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive generative fidelity analysis.

        Args:
            predictions: Class predictions for IS calculation
            real_features: Real sample features for FID
            generated_features: Generated sample features for FID
            perceptual_scores: Perceptual quality scores
            semantic_scores: Semantic quality scores

        Returns:
            Dictionary with all analysis results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": "generative_fidelity",
            "components": {}
        }

        # Inception Score analysis
        if predictions:
            is_result = self.is_analyzer.analyze(predictions)
            results["components"]["inception_score"] = is_result.to_dict()

        # FID analysis
        if real_features and generated_features:
            fid_result = self.fid_analyzer.analyze(real_features, generated_features)
            results["components"]["fid"] = fid_result.to_dict()

        # Perceptual analysis
        if perceptual_scores:
            perceptual_result = self.perceptual_analyzer.analyze(perceptual_scores)
            results["components"]["perceptual"] = perceptual_result.to_dict()

        # Semantic analysis
        if semantic_scores:
            semantic_result = self.semantic_analyzer.analyze(semantic_scores)
            results["components"]["semantic"] = semantic_result.to_dict()

        # Calculate composite score
        results["composite_score"] = self._calculate_composite_score(results["components"])
        results["overall_assessment"] = self._generate_overall_assessment(results)

        return results

    def _calculate_composite_score(self, components: Dict[str, Any]) -> float:
        """Calculate composite fidelity score."""
        scores = []
        weights = []

        if "inception_score" in components:
            # Normalize IS to 0-1 range (assuming max IS of 300)
            is_score = min(1.0, components["inception_score"]["score"] / 300)
            scores.append(is_score)
            weights.append(0.25)

        if "fid" in components:
            # Invert and normalize FID (lower is better)
            fid = components["fid"]["distance"]
            fid_score = max(0, 1 - fid / 200) if fid < 200 else 0
            scores.append(fid_score)
            weights.append(0.25)

        if "perceptual" in components:
            scores.append(components["perceptual"]["overall_score"])
            weights.append(0.25)

        if "semantic" in components:
            scores.append(components["semantic"]["overall_score"])
            weights.append(0.25)

        if not scores:
            return 0.0

        total_weight = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / total_weight if total_weight > 0 else 0.0

    def _generate_overall_assessment(self, results: Dict[str, Any]) -> str:
        """Generate overall assessment text."""
        score = results.get("composite_score", 0)

        if score >= 0.8:
            return "Excellent generative fidelity across all dimensions."
        elif score >= 0.6:
            return "Good generative fidelity with minor areas for improvement."
        elif score >= 0.4:
            return "Moderate fidelity; several areas need attention."
        else:
            return "Low fidelity scores; significant improvements recommended."


class FidelityBenchmark:
    """
    Benchmark utility for comparing fidelity across models or configurations.
    """

    def __init__(self):
        self.results_history: List[Dict[str, Any]] = []

    def add_result(
        self,
        model_name: str,
        result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a benchmark result."""
        entry = {
            "model_name": model_name,
            "result": result,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        self.results_history.append(entry)

    def compare_models(self) -> Dict[str, Any]:
        """Compare all benchmarked models."""
        if not self.results_history:
            return {"error": "No results to compare"}

        comparison = {
            "num_models": len(self.results_history),
            "models": [],
            "best_model": None,
            "ranking": []
        }

        scores = []
        for entry in self.results_history:
            model_info = {
                "name": entry["model_name"],
                "composite_score": entry["result"].get("composite_score", 0),
                "timestamp": entry["timestamp"]
            }
            comparison["models"].append(model_info)
            scores.append((entry["model_name"], model_info["composite_score"]))

        # Rank models
        scores.sort(key=lambda x: x[1], reverse=True)
        comparison["ranking"] = [name for name, _ in scores]
        comparison["best_model"] = scores[0][0] if scores else None

        return comparison

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all benchmarks."""
        if not self.results_history:
            return {"error": "No results available"}

        scores = [r["result"].get("composite_score", 0) for r in self.results_history]

        return {
            "num_benchmarks": len(self.results_history),
            "mean_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "score_range": max(scores) - min(scores)
        }


# Utility functions
def calculate_inception_score(predictions: List[List[float]], num_splits: int = 10) -> InceptionScoreResult:
    """Convenience function to calculate Inception Score."""
    analyzer = InceptionScoreAnalyzer(num_splits=num_splits)
    return analyzer.analyze(predictions)


def calculate_fid(
    real_features: List[List[float]],
    generated_features: List[List[float]]
) -> FIDResult:
    """Convenience function to calculate FID."""
    analyzer = FrechetInceptionDistanceAnalyzer()
    return analyzer.analyze(real_features, generated_features)


def calculate_f1_score(
    true_labels: List[int],
    predicted_labels: List[int],
    positive_class: int = 1
) -> F1ScoreResult:
    """Convenience function to calculate F1 score."""
    analyzer = F1ScoreAnalyzer()
    return analyzer.analyze(true_labels, predicted_labels, positive_class)


def evaluate_generative_fidelity(
    predictions: Optional[List[List[float]]] = None,
    real_features: Optional[List[List[float]]] = None,
    generated_features: Optional[List[List[float]]] = None,
    perceptual_scores: Optional[Dict[str, float]] = None,
    semantic_scores: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Convenience function for comprehensive fidelity evaluation."""
    analyzer = GenerativeFidelityAnalyzer()
    return analyzer.analyze(
        predictions=predictions,
        real_features=real_features,
        generated_features=generated_features,
        perceptual_scores=perceptual_scores,
        semantic_scores=semantic_scores
    )
