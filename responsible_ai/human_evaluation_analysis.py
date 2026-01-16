"""
Human Evaluation Analysis Module for AI/GenAI Research

This module provides comprehensive human evaluation analysis capabilities
for assessing AI system quality through human judgment metrics, including
Mean Opinion Score (MOS), preference ratings, comparative evaluations,
and inter-rater reliability analysis.

Human evaluation remains the gold standard for many AI quality assessments,
especially for subjective qualities like naturalness, coherence, and helpfulness.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from abc import ABC, abstractmethod
import math
from datetime import datetime


class EvaluationScale(Enum):
    """Types of rating scales for human evaluation."""
    MOS_5 = "mos_5"  # 1-5 scale (ITU-T standard)
    MOS_7 = "mos_7"  # 1-7 scale (extended)
    LIKERT_5 = "likert_5"  # 5-point Likert
    LIKERT_7 = "likert_7"  # 7-point Likert
    BINARY = "binary"  # Yes/No
    RANKING = "ranking"  # Comparative ranking
    CONTINUOUS = "continuous"  # 0-100 slider
    PAIRWISE = "pairwise"  # A vs B comparison


class EvaluationDimension(Enum):
    """Dimensions for human evaluation."""
    QUALITY = "quality"
    NATURALNESS = "naturalness"
    FLUENCY = "fluency"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    HELPFULNESS = "helpfulness"
    HARMLESSNESS = "harmlessness"
    HONESTY = "honesty"
    CREATIVITY = "creativity"
    ENGAGEMENT = "engagement"
    OVERALL = "overall"


class AgreementMetric(Enum):
    """Metrics for inter-rater agreement."""
    COHENS_KAPPA = "cohens_kappa"
    FLEISS_KAPPA = "fleiss_kappa"
    KRIPPENDORFF_ALPHA = "krippendorff_alpha"
    ICC = "intraclass_correlation"
    PERCENT_AGREEMENT = "percent_agreement"
    SPEARMAN_CORRELATION = "spearman_correlation"
    KENDALL_TAU = "kendall_tau"


@dataclass
class MOSResult:
    """Result of Mean Opinion Score analysis."""
    mean_score: float
    std_deviation: float
    confidence_interval: Tuple[float, float]
    median_score: float
    mode_score: float
    num_ratings: int
    num_raters: int
    rating_distribution: Dict[int, int]
    quality_category: str
    interpretation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_score": self.mean_score,
            "std_deviation": self.std_deviation,
            "confidence_interval": self.confidence_interval,
            "median_score": self.median_score,
            "mode_score": self.mode_score,
            "num_ratings": self.num_ratings,
            "num_raters": self.num_raters,
            "rating_distribution": self.rating_distribution,
            "quality_category": self.quality_category,
            "interpretation": self.interpretation,
            "timestamp": self.timestamp
        }


@dataclass
class InterRaterAgreement:
    """Result of inter-rater agreement analysis."""
    agreement_metric: AgreementMetric
    agreement_value: float
    agreement_level: str
    confidence_interval: Optional[Tuple[float, float]]
    num_raters: int
    num_items: int
    pairwise_agreements: Optional[Dict[str, float]]
    interpretation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agreement_metric": self.agreement_metric.value,
            "agreement_value": self.agreement_value,
            "agreement_level": self.agreement_level,
            "confidence_interval": self.confidence_interval,
            "num_raters": self.num_raters,
            "num_items": self.num_items,
            "pairwise_agreements": self.pairwise_agreements,
            "interpretation": self.interpretation,
            "timestamp": self.timestamp
        }


@dataclass
class PairwiseComparisonResult:
    """Result of pairwise comparison analysis."""
    win_rate: Dict[str, float]
    total_comparisons: int
    preference_matrix: Dict[str, Dict[str, int]]
    bradley_terry_scores: Optional[Dict[str, float]]
    elo_ratings: Optional[Dict[str, float]]
    statistical_significance: Dict[str, bool]
    interpretation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "win_rate": self.win_rate,
            "total_comparisons": self.total_comparisons,
            "preference_matrix": self.preference_matrix,
            "bradley_terry_scores": self.bradley_terry_scores,
            "elo_ratings": self.elo_ratings,
            "statistical_significance": self.statistical_significance,
            "interpretation": self.interpretation,
            "timestamp": self.timestamp
        }


@dataclass
class HumanEvaluationReport:
    """Comprehensive human evaluation report."""
    evaluation_id: str
    overall_mos: MOSResult
    dimension_scores: Dict[str, MOSResult]
    inter_rater_agreement: InterRaterAgreement
    rater_statistics: Dict[str, Any]
    quality_breakdown: Dict[str, float]
    recommendations: List[str]
    methodology: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "overall_mos": self.overall_mos.to_dict(),
            "dimension_scores": {k: v.to_dict() for k, v in self.dimension_scores.items()},
            "inter_rater_agreement": self.inter_rater_agreement.to_dict(),
            "rater_statistics": self.rater_statistics,
            "quality_breakdown": self.quality_breakdown,
            "recommendations": self.recommendations,
            "methodology": self.methodology,
            "timestamp": self.timestamp
        }


class BaseHumanEvaluationAnalyzer(ABC):
    """Abstract base class for human evaluation analyzers."""

    @abstractmethod
    def analyze(self, data: Any) -> Any:
        """Perform human evaluation analysis."""
        pass

    @abstractmethod
    def get_analyzer_type(self) -> str:
        """Return the type of analyzer."""
        pass


class MOSAnalyzer(BaseHumanEvaluationAnalyzer):
    """
    Analyzer for Mean Opinion Score (MOS) computation.

    MOS is the arithmetic mean of all individual ratings for a stimulus.
    It is the standard metric for subjective quality assessment in
    telecommunications (ITU-T P.800) and is widely used for AI evaluation.

    Quality Categories (5-point scale):
    - 5: Excellent
    - 4: Good
    - 3: Fair
    - 2: Poor
    - 1: Bad

    The 95% confidence interval is computed using:
    CI = mean ± 1.96 * (std / sqrt(n))
    """

    def __init__(self, scale: EvaluationScale = EvaluationScale.MOS_5):
        self.scale = scale
        self.scale_range = self._get_scale_range()

    def get_analyzer_type(self) -> str:
        return "mean_opinion_score"

    def _get_scale_range(self) -> Tuple[int, int]:
        """Get the min and max values for the scale."""
        scale_ranges = {
            EvaluationScale.MOS_5: (1, 5),
            EvaluationScale.MOS_7: (1, 7),
            EvaluationScale.LIKERT_5: (1, 5),
            EvaluationScale.LIKERT_7: (1, 7),
            EvaluationScale.BINARY: (0, 1),
            EvaluationScale.CONTINUOUS: (0, 100),
        }
        return scale_ranges.get(self.scale, (1, 5))

    def analyze(
        self,
        ratings: List[List[float]],
        rater_ids: Optional[List[str]] = None
    ) -> MOSResult:
        """
        Compute MOS from ratings matrix.

        Args:
            ratings: Matrix of ratings [items x raters] or flat list
            rater_ids: Optional identifiers for raters

        Returns:
            MOSResult with MOS statistics
        """
        # Flatten ratings if nested
        all_ratings = []
        num_raters = 0

        if ratings and isinstance(ratings[0], list):
            for item_ratings in ratings:
                all_ratings.extend(item_ratings)
            num_raters = len(ratings[0]) if ratings else 0
        else:
            all_ratings = ratings
            num_raters = 1

        if not all_ratings:
            return self._empty_result()

        n = len(all_ratings)

        # Basic statistics
        mean_score = sum(all_ratings) / n
        variance = sum((r - mean_score) ** 2 for r in all_ratings) / (n - 1) if n > 1 else 0
        std_dev = math.sqrt(variance)

        # Confidence interval (95%)
        margin = 1.96 * std_dev / math.sqrt(n) if n > 0 else 0
        ci = (mean_score - margin, mean_score + margin)

        # Median
        sorted_ratings = sorted(all_ratings)
        median = sorted_ratings[n // 2] if n % 2 == 1 else (sorted_ratings[n // 2 - 1] + sorted_ratings[n // 2]) / 2

        # Mode
        rating_counts = {}
        for r in all_ratings:
            rating_counts[r] = rating_counts.get(r, 0) + 1
        mode = max(rating_counts.items(), key=lambda x: x[1])[0]

        # Rating distribution
        min_val, max_val = self.scale_range
        distribution = {i: 0 for i in range(int(min_val), int(max_val) + 1)}
        for r in all_ratings:
            r_int = int(round(r))
            if r_int in distribution:
                distribution[r_int] += 1

        # Quality category
        quality_category = self._categorize_mos(mean_score)

        interpretation = self._interpret_mos(mean_score, std_dev, n)

        return MOSResult(
            mean_score=mean_score,
            std_deviation=std_dev,
            confidence_interval=ci,
            median_score=median,
            mode_score=mode,
            num_ratings=n,
            num_raters=num_raters,
            rating_distribution=distribution,
            quality_category=quality_category,
            interpretation=interpretation
        )

    def _categorize_mos(self, mos: float) -> str:
        """Categorize MOS into quality levels."""
        min_val, max_val = self.scale_range
        normalized = (mos - min_val) / (max_val - min_val)

        if normalized >= 0.8:
            return "Excellent"
        elif normalized >= 0.6:
            return "Good"
        elif normalized >= 0.4:
            return "Fair"
        elif normalized >= 0.2:
            return "Poor"
        else:
            return "Bad"

    def _interpret_mos(self, mos: float, std: float, n: int) -> str:
        """Generate interpretation of MOS result."""
        category = self._categorize_mos(mos)
        min_val, max_val = self.scale_range

        consistency = "high" if std < 0.5 else "moderate" if std < 1.0 else "low"

        return (
            f"MOS = {mos:.2f} ± {std:.2f} ({category} quality) on {min_val}-{max_val} scale. "
            f"Based on {n} ratings with {consistency} inter-rater consistency."
        )

    def _empty_result(self) -> MOSResult:
        """Return empty result for invalid input."""
        return MOSResult(
            mean_score=0.0,
            std_deviation=0.0,
            confidence_interval=(0.0, 0.0),
            median_score=0.0,
            mode_score=0.0,
            num_ratings=0,
            num_raters=0,
            rating_distribution={},
            quality_category="Unknown",
            interpretation="Insufficient data for MOS computation"
        )


class InterRaterReliabilityAnalyzer(BaseHumanEvaluationAnalyzer):
    """
    Analyzer for inter-rater reliability/agreement.

    Measures how consistently different raters evaluate the same items.
    High agreement indicates reliable human evaluation.

    Supported metrics:
    - Cohen's Kappa (2 raters)
    - Fleiss' Kappa (multiple raters)
    - Krippendorff's Alpha
    - Intraclass Correlation (ICC)
    - Percent Agreement
    """

    def __init__(self, metric: AgreementMetric = AgreementMetric.FLEISS_KAPPA):
        self.metric = metric

    def get_analyzer_type(self) -> str:
        return "inter_rater_reliability"

    def analyze(
        self,
        ratings_matrix: List[List[float]]
    ) -> InterRaterAgreement:
        """
        Analyze inter-rater agreement.

        Args:
            ratings_matrix: Matrix of ratings [items x raters]

        Returns:
            InterRaterAgreement with agreement statistics
        """
        if not ratings_matrix or not ratings_matrix[0]:
            return self._empty_result()

        num_items = len(ratings_matrix)
        num_raters = len(ratings_matrix[0])

        # Compute agreement based on selected metric
        if self.metric == AgreementMetric.FLEISS_KAPPA:
            agreement_value = self._compute_fleiss_kappa(ratings_matrix)
        elif self.metric == AgreementMetric.PERCENT_AGREEMENT:
            agreement_value = self._compute_percent_agreement(ratings_matrix)
        elif self.metric == AgreementMetric.KRIPPENDORFF_ALPHA:
            agreement_value = self._compute_krippendorff_alpha(ratings_matrix)
        elif self.metric == AgreementMetric.ICC:
            agreement_value = self._compute_icc(ratings_matrix)
        else:
            agreement_value = self._compute_fleiss_kappa(ratings_matrix)

        agreement_level = self._interpret_agreement_level(agreement_value)
        interpretation = self._interpret_agreement(agreement_value, num_raters, num_items)

        # Compute pairwise agreements if more than 2 raters
        pairwise = None
        if num_raters > 2:
            pairwise = self._compute_pairwise_agreements(ratings_matrix)

        return InterRaterAgreement(
            agreement_metric=self.metric,
            agreement_value=agreement_value,
            agreement_level=agreement_level,
            confidence_interval=None,  # Would require bootstrap
            num_raters=num_raters,
            num_items=num_items,
            pairwise_agreements=pairwise,
            interpretation=interpretation
        )

    def _compute_fleiss_kappa(self, ratings_matrix: List[List[float]]) -> float:
        """Compute Fleiss' Kappa for multiple raters."""
        n_items = len(ratings_matrix)
        n_raters = len(ratings_matrix[0])

        # Get unique categories
        all_ratings = [r for item in ratings_matrix for r in item]
        categories = sorted(set(all_ratings))
        n_categories = len(categories)

        if n_categories < 2:
            return 1.0  # Perfect agreement if only one category

        # Count ratings per category per item
        category_counts = []
        for item_ratings in ratings_matrix:
            counts = {cat: 0 for cat in categories}
            for r in item_ratings:
                counts[r] += 1
            category_counts.append(counts)

        # Compute P_i for each item
        p_i = []
        for counts in category_counts:
            sum_sq = sum(c * c for c in counts.values())
            p_item = (sum_sq - n_raters) / (n_raters * (n_raters - 1)) if n_raters > 1 else 1
            p_i.append(p_item)

        # Observed agreement
        p_bar = sum(p_i) / n_items if n_items > 0 else 0

        # Expected agreement by chance
        p_j = {}
        for cat in categories:
            total_cat = sum(counts[cat] for counts in category_counts)
            p_j[cat] = total_cat / (n_items * n_raters)

        p_e = sum(p ** 2 for p in p_j.values())

        # Fleiss' Kappa
        if p_e == 1:
            return 1.0
        kappa = (p_bar - p_e) / (1 - p_e)

        return max(-1, min(1, kappa))

    def _compute_percent_agreement(self, ratings_matrix: List[List[float]]) -> float:
        """Compute simple percent agreement."""
        n_items = len(ratings_matrix)
        agreements = 0

        for item_ratings in ratings_matrix:
            # Check if all ratings are the same
            if len(set(item_ratings)) == 1:
                agreements += 1

        return agreements / n_items if n_items > 0 else 0

    def _compute_krippendorff_alpha(self, ratings_matrix: List[List[float]]) -> float:
        """Compute Krippendorff's Alpha (simplified interval scale)."""
        n_items = len(ratings_matrix)
        n_raters = len(ratings_matrix[0])

        # All valid pairs
        all_pairs = []
        for item_ratings in ratings_matrix:
            for i in range(n_raters):
                for j in range(i + 1, n_raters):
                    all_pairs.append((item_ratings[i], item_ratings[j]))

        if not all_pairs:
            return 0.0

        # Observed disagreement
        d_o = sum((p[0] - p[1]) ** 2 for p in all_pairs) / len(all_pairs)

        # Expected disagreement
        all_ratings = [r for item in ratings_matrix for r in item]
        n_total = len(all_ratings)
        mean_all = sum(all_ratings) / n_total if n_total > 0 else 0
        var_all = sum((r - mean_all) ** 2 for r in all_ratings) / n_total if n_total > 0 else 0
        d_e = 2 * var_all

        # Alpha
        if d_e == 0:
            return 1.0
        alpha = 1 - (d_o / d_e)

        return max(-1, min(1, alpha))

    def _compute_icc(self, ratings_matrix: List[List[float]]) -> float:
        """Compute Intraclass Correlation Coefficient (ICC 2,k)."""
        n_items = len(ratings_matrix)
        n_raters = len(ratings_matrix[0])

        # Grand mean
        all_ratings = [r for item in ratings_matrix for r in item]
        grand_mean = sum(all_ratings) / len(all_ratings) if all_ratings else 0

        # Item means
        item_means = [sum(item) / len(item) for item in ratings_matrix]

        # Rater means
        rater_means = [sum(ratings_matrix[i][j] for i in range(n_items)) / n_items for j in range(n_raters)]

        # Sum of squares
        ss_between = n_raters * sum((im - grand_mean) ** 2 for im in item_means)
        ss_within = sum(
            sum((ratings_matrix[i][j] - item_means[i]) ** 2 for j in range(n_raters))
            for i in range(n_items)
        )
        ss_raters = n_items * sum((rm - grand_mean) ** 2 for rm in rater_means)
        ss_error = ss_within - ss_raters

        # Mean squares
        ms_between = ss_between / (n_items - 1) if n_items > 1 else 0
        ms_error = ss_error / ((n_items - 1) * (n_raters - 1)) if (n_items > 1 and n_raters > 1) else 1

        # ICC
        if ms_between + (n_raters - 1) * ms_error == 0:
            return 0.0
        icc = (ms_between - ms_error) / (ms_between + (n_raters - 1) * ms_error)

        return max(-1, min(1, icc))

    def _compute_pairwise_agreements(self, ratings_matrix: List[List[float]]) -> Dict[str, float]:
        """Compute pairwise agreement between all rater pairs."""
        n_raters = len(ratings_matrix[0])
        pairwise = {}

        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                # Extract ratings for this pair
                pair_ratings = [[item[i], item[j]] for item in ratings_matrix]
                kappa = self._compute_fleiss_kappa(pair_ratings)
                pairwise[f"rater_{i}_vs_rater_{j}"] = kappa

        return pairwise

    def _interpret_agreement_level(self, value: float) -> str:
        """Interpret agreement level based on value."""
        if value >= 0.8:
            return "Almost Perfect"
        elif value >= 0.6:
            return "Substantial"
        elif value >= 0.4:
            return "Moderate"
        elif value >= 0.2:
            return "Fair"
        elif value >= 0:
            return "Slight"
        else:
            return "Poor"

    def _interpret_agreement(self, value: float, n_raters: int, n_items: int) -> str:
        """Generate interpretation of agreement result."""
        level = self._interpret_agreement_level(value)

        if value >= 0.6:
            reliability = "reliable"
            recommendation = "Human evaluation results can be used with confidence."
        elif value >= 0.4:
            reliability = "moderately reliable"
            recommendation = "Consider additional training or clearer guidelines for raters."
        else:
            reliability = "unreliable"
            recommendation = "Significant disagreement detected; review evaluation criteria."

        return (
            f"Inter-rater agreement = {value:.3f} ({level}). "
            f"Based on {n_raters} raters evaluating {n_items} items. "
            f"Evaluation is {reliability}. {recommendation}"
        )

    def _empty_result(self) -> InterRaterAgreement:
        """Return empty result for invalid input."""
        return InterRaterAgreement(
            agreement_metric=self.metric,
            agreement_value=0.0,
            agreement_level="Unknown",
            confidence_interval=None,
            num_raters=0,
            num_items=0,
            pairwise_agreements=None,
            interpretation="Insufficient data for agreement analysis"
        )


class PairwiseComparisonAnalyzer(BaseHumanEvaluationAnalyzer):
    """
    Analyzer for pairwise comparison evaluations.

    Pairwise comparison asks raters to choose between two options (A vs B).
    This method is often more reliable than absolute ratings because
    it's easier for humans to make relative judgments.

    Supports:
    - Win rate calculation
    - Bradley-Terry model
    - Elo rating system
    - Statistical significance testing
    """

    def __init__(self, elo_k_factor: float = 32.0):
        self.elo_k_factor = elo_k_factor

    def get_analyzer_type(self) -> str:
        return "pairwise_comparison"

    def analyze(
        self,
        comparisons: List[Tuple[str, str, str]]
    ) -> PairwiseComparisonResult:
        """
        Analyze pairwise comparison results.

        Args:
            comparisons: List of (item_a, item_b, winner) tuples
                        winner is either item_a, item_b, or "tie"

        Returns:
            PairwiseComparisonResult with analysis
        """
        if not comparisons:
            return self._empty_result()

        # Build preference matrix
        items = set()
        for a, b, _ in comparisons:
            items.add(a)
            items.add(b)
        items = list(items)

        # Initialize preference matrix
        preference_matrix = {a: {b: 0 for b in items} for a in items}

        # Count wins
        win_counts = {item: 0 for item in items}
        loss_counts = {item: 0 for item in items}

        for item_a, item_b, winner in comparisons:
            if winner == item_a:
                preference_matrix[item_a][item_b] += 1
                win_counts[item_a] += 1
                loss_counts[item_b] += 1
            elif winner == item_b:
                preference_matrix[item_b][item_a] += 1
                win_counts[item_b] += 1
                loss_counts[item_a] += 1
            # Ties don't affect counts

        # Calculate win rates
        win_rate = {}
        for item in items:
            total = win_counts[item] + loss_counts[item]
            win_rate[item] = win_counts[item] / total if total > 0 else 0.5

        # Bradley-Terry scores
        bt_scores = self._compute_bradley_terry(preference_matrix, items)

        # Elo ratings
        elo_ratings = self._compute_elo_ratings(comparisons, items)

        # Statistical significance
        significance = self._compute_significance(win_counts, loss_counts, items)

        interpretation = self._interpret_results(win_rate, bt_scores)

        return PairwiseComparisonResult(
            win_rate=win_rate,
            total_comparisons=len(comparisons),
            preference_matrix=preference_matrix,
            bradley_terry_scores=bt_scores,
            elo_ratings=elo_ratings,
            statistical_significance=significance,
            interpretation=interpretation
        )

    def _compute_bradley_terry(
        self,
        preference_matrix: Dict[str, Dict[str, int]],
        items: List[str]
    ) -> Dict[str, float]:
        """Compute Bradley-Terry model scores (simplified)."""
        # Initialize scores
        scores = {item: 1.0 for item in items}

        # Iterative estimation
        for _ in range(100):  # Max iterations
            new_scores = {}
            for i in items:
                numerator = sum(preference_matrix[i].values())
                denominator = 0
                for j in items:
                    if i != j:
                        n_ij = preference_matrix[i][j] + preference_matrix[j][i]
                        if n_ij > 0:
                            denominator += n_ij / (scores[i] + scores[j])

                new_scores[i] = numerator / denominator if denominator > 0 else 1.0

            # Normalize
            total = sum(new_scores.values())
            new_scores = {k: v / total * len(items) for k, v in new_scores.items()}

            # Check convergence
            max_diff = max(abs(new_scores[i] - scores[i]) for i in items)
            scores = new_scores
            if max_diff < 1e-6:
                break

        return scores

    def _compute_elo_ratings(
        self,
        comparisons: List[Tuple[str, str, str]],
        items: List[str]
    ) -> Dict[str, float]:
        """Compute Elo ratings from comparisons."""
        # Initialize ratings
        ratings = {item: 1500.0 for item in items}

        for item_a, item_b, winner in comparisons:
            # Expected scores
            r_a = ratings[item_a]
            r_b = ratings[item_b]
            e_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
            e_b = 1 / (1 + 10 ** ((r_a - r_b) / 400))

            # Actual scores
            if winner == item_a:
                s_a, s_b = 1, 0
            elif winner == item_b:
                s_a, s_b = 0, 1
            else:
                s_a, s_b = 0.5, 0.5

            # Update ratings
            ratings[item_a] += self.elo_k_factor * (s_a - e_a)
            ratings[item_b] += self.elo_k_factor * (s_b - e_b)

        return ratings

    def _compute_significance(
        self,
        win_counts: Dict[str, int],
        loss_counts: Dict[str, int],
        items: List[str]
    ) -> Dict[str, bool]:
        """Compute statistical significance using binomial test."""
        significance = {}

        for item in items:
            wins = win_counts[item]
            total = wins + loss_counts[item]

            if total < 5:
                significance[item] = False
                continue

            # Two-sided binomial test (p=0.5 under null)
            # Using normal approximation for simplicity
            expected = total * 0.5
            std = math.sqrt(total * 0.5 * 0.5)
            z = abs(wins - expected) / std if std > 0 else 0

            # Significant at alpha=0.05 (z > 1.96)
            significance[item] = z > 1.96

        return significance

    def _interpret_results(
        self,
        win_rate: Dict[str, float],
        bt_scores: Dict[str, float]
    ) -> str:
        """Generate interpretation of comparison results."""
        sorted_items = sorted(win_rate.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_items) < 2:
            return "Insufficient items for comparison"

        best = sorted_items[0]
        second = sorted_items[1]

        gap = best[1] - second[1]
        if gap > 0.2:
            confidence = "clear"
        elif gap > 0.1:
            confidence = "moderate"
        else:
            confidence = "slight"

        return (
            f"'{best[0]}' has {confidence} preference over other items "
            f"(win rate: {best[1]:.1%}). "
            f"Second place: '{second[0]}' (win rate: {second[1]:.1%})."
        )

    def _empty_result(self) -> PairwiseComparisonResult:
        """Return empty result for invalid input."""
        return PairwiseComparisonResult(
            win_rate={},
            total_comparisons=0,
            preference_matrix={},
            bradley_terry_scores=None,
            elo_ratings=None,
            statistical_significance={},
            interpretation="No comparison data provided"
        )


class MultiDimensionalEvaluator(BaseHumanEvaluationAnalyzer):
    """
    Analyzer for multi-dimensional human evaluation.

    Evaluates items across multiple quality dimensions and provides
    comprehensive analysis including dimension correlations and
    overall quality assessment.
    """

    def __init__(self, dimensions: Optional[List[EvaluationDimension]] = None):
        self.dimensions = dimensions or [
            EvaluationDimension.QUALITY,
            EvaluationDimension.FLUENCY,
            EvaluationDimension.COHERENCE,
            EvaluationDimension.RELEVANCE
        ]
        self.mos_analyzer = MOSAnalyzer()

    def get_analyzer_type(self) -> str:
        return "multi_dimensional_evaluation"

    def analyze(
        self,
        dimension_ratings: Dict[str, List[List[float]]],
        weights: Optional[Dict[str, float]] = None
    ) -> HumanEvaluationReport:
        """
        Perform multi-dimensional human evaluation analysis.

        Args:
            dimension_ratings: Dictionary mapping dimension names to ratings matrices
            weights: Optional weights for each dimension

        Returns:
            HumanEvaluationReport with comprehensive analysis
        """
        if not dimension_ratings:
            return self._empty_report()

        # Default equal weights
        if weights is None:
            weights = {dim: 1.0 / len(dimension_ratings) for dim in dimension_ratings}

        # Analyze each dimension
        dimension_scores = {}
        for dim_name, ratings in dimension_ratings.items():
            dimension_scores[dim_name] = self.mos_analyzer.analyze(ratings)

        # Compute overall MOS (weighted average)
        overall_mos = self._compute_weighted_overall(dimension_scores, weights)

        # Compute inter-rater agreement (using first available dimension)
        first_dim = list(dimension_ratings.keys())[0]
        irr_analyzer = InterRaterReliabilityAnalyzer()
        agreement = irr_analyzer.analyze(dimension_ratings[first_dim])

        # Rater statistics
        rater_stats = self._compute_rater_statistics(dimension_ratings)

        # Quality breakdown
        quality_breakdown = {
            dim: scores.mean_score for dim, scores in dimension_scores.items()
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(dimension_scores, agreement)

        return HumanEvaluationReport(
            evaluation_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            overall_mos=overall_mos,
            dimension_scores=dimension_scores,
            inter_rater_agreement=agreement,
            rater_statistics=rater_stats,
            quality_breakdown=quality_breakdown,
            recommendations=recommendations,
            methodology="Multi-dimensional human evaluation with MOS aggregation"
        )

    def _compute_weighted_overall(
        self,
        dimension_scores: Dict[str, MOSResult],
        weights: Dict[str, float]
    ) -> MOSResult:
        """Compute weighted overall MOS."""
        total_weight = sum(weights.get(d, 0) for d in dimension_scores)
        if total_weight == 0:
            total_weight = 1

        weighted_mean = sum(
            scores.mean_score * weights.get(dim, 0)
            for dim, scores in dimension_scores.items()
        ) / total_weight

        # Pooled standard deviation
        n_total = sum(s.num_ratings for s in dimension_scores.values())
        weighted_var = sum(
            scores.std_deviation ** 2 * weights.get(dim, 0) ** 2
            for dim, scores in dimension_scores.items()
        ) / (total_weight ** 2)
        weighted_std = math.sqrt(weighted_var)

        # Confidence interval
        margin = 1.96 * weighted_std / math.sqrt(n_total) if n_total > 0 else 0
        ci = (weighted_mean - margin, weighted_mean + margin)

        return MOSResult(
            mean_score=weighted_mean,
            std_deviation=weighted_std,
            confidence_interval=ci,
            median_score=weighted_mean,  # Approximation
            mode_score=weighted_mean,
            num_ratings=n_total,
            num_raters=list(dimension_scores.values())[0].num_raters if dimension_scores else 0,
            rating_distribution={},
            quality_category=self._categorize_weighted(weighted_mean),
            interpretation=f"Weighted overall MOS = {weighted_mean:.2f}"
        )

    def _categorize_weighted(self, score: float) -> str:
        """Categorize weighted score."""
        if score >= 4.0:
            return "Excellent"
        elif score >= 3.5:
            return "Good"
        elif score >= 2.5:
            return "Fair"
        elif score >= 1.5:
            return "Poor"
        else:
            return "Bad"

    def _compute_rater_statistics(
        self,
        dimension_ratings: Dict[str, List[List[float]]]
    ) -> Dict[str, Any]:
        """Compute statistics about raters."""
        first_dim = list(dimension_ratings.values())[0]
        n_items = len(first_dim)
        n_raters = len(first_dim[0]) if first_dim else 0

        return {
            "num_items": n_items,
            "num_raters": n_raters,
            "num_dimensions": len(dimension_ratings),
            "total_ratings": n_items * n_raters * len(dimension_ratings)
        }

    def _generate_recommendations(
        self,
        dimension_scores: Dict[str, MOSResult],
        agreement: InterRaterAgreement
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Check for low-scoring dimensions
        for dim, scores in dimension_scores.items():
            if scores.mean_score < 3.0:
                recommendations.append(
                    f"Improve {dim}: current score ({scores.mean_score:.2f}) is below acceptable threshold."
                )

        # Check agreement
        if agreement.agreement_value < 0.4:
            recommendations.append(
                "Low inter-rater agreement detected. Consider clarifying evaluation guidelines."
            )

        # Check for high variance
        high_var_dims = [d for d, s in dimension_scores.items() if s.std_deviation > 1.0]
        if high_var_dims:
            recommendations.append(
                f"High rating variance in {', '.join(high_var_dims)}. Consider additional calibration."
            )

        if not recommendations:
            recommendations.append("Evaluation results are satisfactory across all dimensions.")

        return recommendations

    def _empty_report(self) -> HumanEvaluationReport:
        """Return empty report for invalid input."""
        empty_mos = MOSResult(
            mean_score=0, std_deviation=0, confidence_interval=(0, 0),
            median_score=0, mode_score=0, num_ratings=0, num_raters=0,
            rating_distribution={}, quality_category="Unknown",
            interpretation="No data"
        )
        empty_agreement = InterRaterAgreement(
            agreement_metric=AgreementMetric.FLEISS_KAPPA,
            agreement_value=0, agreement_level="Unknown",
            confidence_interval=None, num_raters=0, num_items=0,
            pairwise_agreements=None, interpretation="No data"
        )
        return HumanEvaluationReport(
            evaluation_id="empty",
            overall_mos=empty_mos,
            dimension_scores={},
            inter_rater_agreement=empty_agreement,
            rater_statistics={},
            quality_breakdown={},
            recommendations=["No data provided for analysis"],
            methodology="N/A"
        )


class RaterQualityAnalyzer(BaseHumanEvaluationAnalyzer):
    """
    Analyzer for assessing individual rater quality.

    Identifies raters who may need additional training or whose
    ratings should be weighted differently.
    """

    def __init__(self):
        pass

    def get_analyzer_type(self) -> str:
        return "rater_quality"

    def analyze(
        self,
        ratings_matrix: List[List[float]],
        rater_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze quality of individual raters.

        Args:
            ratings_matrix: Matrix of ratings [items x raters]
            rater_ids: Optional identifiers for raters

        Returns:
            Dictionary with rater quality metrics
        """
        if not ratings_matrix or not ratings_matrix[0]:
            return {"error": "Insufficient data"}

        n_items = len(ratings_matrix)
        n_raters = len(ratings_matrix[0])

        if rater_ids is None:
            rater_ids = [f"rater_{i}" for i in range(n_raters)]

        # Compute consensus (mean rating per item)
        consensus = [sum(item) / n_raters for item in ratings_matrix]

        rater_analysis = {}
        for j in range(n_raters):
            rater_ratings = [ratings_matrix[i][j] for i in range(n_items)]

            # Mean rating
            mean_rating = sum(rater_ratings) / n_items

            # Deviation from consensus
            deviations = [abs(r - c) for r, c in zip(rater_ratings, consensus)]
            mean_deviation = sum(deviations) / n_items

            # Correlation with consensus
            correlation = self._compute_correlation(rater_ratings, consensus)

            # Rating variance
            variance = sum((r - mean_rating) ** 2 for r in rater_ratings) / n_items

            # Quality score
            quality_score = self._compute_quality_score(correlation, mean_deviation, variance)

            rater_analysis[rater_ids[j]] = {
                "mean_rating": mean_rating,
                "mean_deviation_from_consensus": mean_deviation,
                "correlation_with_consensus": correlation,
                "rating_variance": variance,
                "quality_score": quality_score,
                "needs_review": quality_score < 0.5
            }

        # Overall statistics
        quality_scores = [r["quality_score"] for r in rater_analysis.values()]

        return {
            "rater_analysis": rater_analysis,
            "summary": {
                "mean_quality_score": sum(quality_scores) / len(quality_scores),
                "raters_needing_review": sum(1 for r in rater_analysis.values() if r["needs_review"]),
                "total_raters": n_raters
            }
        }

    def _compute_correlation(self, x: List[float], y: List[float]) -> float:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

        if denom_x * denom_y == 0:
            return 0.0
        return numerator / (denom_x * denom_y)

    def _compute_quality_score(
        self,
        correlation: float,
        deviation: float,
        variance: float
    ) -> float:
        """Compute overall rater quality score."""
        # Higher correlation = better
        corr_score = (correlation + 1) / 2  # Normalize to [0, 1]

        # Lower deviation = better
        dev_score = max(0, 1 - deviation / 2)

        # Moderate variance is expected; too high or too low is suspicious
        var_score = 1 - abs(variance - 1) / 2
        var_score = max(0, min(1, var_score))

        # Weighted combination
        return 0.5 * corr_score + 0.3 * dev_score + 0.2 * var_score


# Utility functions
def compute_mos(ratings: List[List[float]]) -> MOSResult:
    """Convenience function for MOS computation."""
    analyzer = MOSAnalyzer()
    return analyzer.analyze(ratings)


def compute_inter_rater_agreement(
    ratings_matrix: List[List[float]],
    metric: AgreementMetric = AgreementMetric.FLEISS_KAPPA
) -> InterRaterAgreement:
    """Convenience function for inter-rater agreement."""
    analyzer = InterRaterReliabilityAnalyzer(metric=metric)
    return analyzer.analyze(ratings_matrix)


def analyze_pairwise_comparisons(
    comparisons: List[Tuple[str, str, str]]
) -> PairwiseComparisonResult:
    """Convenience function for pairwise comparison analysis."""
    analyzer = PairwiseComparisonAnalyzer()
    return analyzer.analyze(comparisons)


def evaluate_multiple_dimensions(
    dimension_ratings: Dict[str, List[List[float]]],
    weights: Optional[Dict[str, float]] = None
) -> HumanEvaluationReport:
    """Convenience function for multi-dimensional evaluation."""
    analyzer = MultiDimensionalEvaluator()
    return analyzer.analyze(dimension_ratings, weights)


def assess_rater_quality(
    ratings_matrix: List[List[float]],
    rater_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Convenience function for rater quality assessment."""
    analyzer = RaterQualityAnalyzer()
    return analyzer.analyze(ratings_matrix, rater_ids)
