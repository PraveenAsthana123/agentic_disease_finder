"""
Divergence Analysis Module for AI/GenAI Research Evaluation

This module provides comprehensive divergence and distance metrics for
comparing probability distributions, essential for evaluating generative
models, measuring distribution shift, and assessing model quality.

Key Metrics:
- KL Divergence: Asymmetric measure of distribution difference
- Jensen-Shannon Divergence: Symmetric, bounded version of KL
- Wasserstein Distance: Earth mover's distance between distributions
- Maximum Mean Discrepancy (MMD): Kernel-based distribution comparison
- Total Variation Distance: Maximum probability difference
- Hellinger Distance: Geometric mean-based distance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum
from abc import ABC, abstractmethod
import math
from datetime import datetime


class DivergenceType(Enum):
    """Types of divergence/distance metrics."""
    KL_DIVERGENCE = "kl_divergence"
    REVERSE_KL = "reverse_kl"
    JENSEN_SHANNON = "jensen_shannon"
    WASSERSTEIN = "wasserstein"
    MMD = "maximum_mean_discrepancy"
    TOTAL_VARIATION = "total_variation"
    HELLINGER = "hellinger"
    BHATTACHARYYA = "bhattacharyya"
    CHI_SQUARED = "chi_squared"
    RENYI = "renyi"
    F_DIVERGENCE = "f_divergence"
    ALPHA_DIVERGENCE = "alpha_divergence"


class KernelType(Enum):
    """Types of kernels for MMD computation."""
    RBF = "rbf"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    LAPLACIAN = "laplacian"
    IMQ = "inverse_multiquadric"


@dataclass
class DivergenceResult:
    """Result of a divergence computation."""
    divergence_type: DivergenceType
    value: float
    is_symmetric: bool
    is_bounded: bool
    bound_range: Optional[Tuple[float, float]]
    p_distribution_info: Dict[str, Any]
    q_distribution_info: Dict[str, Any]
    interpretation: str
    quality_assessment: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "divergence_type": self.divergence_type.value,
            "value": self.value,
            "is_symmetric": self.is_symmetric,
            "is_bounded": self.is_bounded,
            "bound_range": self.bound_range,
            "p_distribution_info": self.p_distribution_info,
            "q_distribution_info": self.q_distribution_info,
            "interpretation": self.interpretation,
            "quality_assessment": self.quality_assessment,
            "timestamp": self.timestamp
        }


@dataclass
class MMDResult:
    """Result of Maximum Mean Discrepancy computation."""
    mmd_value: float
    mmd_squared: float
    kernel_type: KernelType
    kernel_params: Dict[str, float]
    p_sample_size: int
    q_sample_size: int
    significance_threshold: float
    is_significant: bool
    confidence_interval: Optional[Tuple[float, float]]
    interpretation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mmd_value": self.mmd_value,
            "mmd_squared": self.mmd_squared,
            "kernel_type": self.kernel_type.value,
            "kernel_params": self.kernel_params,
            "p_sample_size": self.p_sample_size,
            "q_sample_size": self.q_sample_size,
            "significance_threshold": self.significance_threshold,
            "is_significant": self.is_significant,
            "confidence_interval": self.confidence_interval,
            "interpretation": self.interpretation,
            "timestamp": self.timestamp
        }


@dataclass
class WassersteinResult:
    """Result of Wasserstein distance computation."""
    distance: float
    order: int
    transport_cost: float
    p_mean: float
    q_mean: float
    p_std: float
    q_std: float
    optimal_transport_plan: Optional[List[Tuple[int, int, float]]]
    interpretation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "distance": self.distance,
            "order": self.order,
            "transport_cost": self.transport_cost,
            "p_mean": self.p_mean,
            "q_mean": self.q_mean,
            "p_std": self.p_std,
            "q_std": self.q_std,
            "has_transport_plan": self.optimal_transport_plan is not None,
            "interpretation": self.interpretation,
            "timestamp": self.timestamp
        }


@dataclass
class DivergenceComparison:
    """Comparison of multiple divergence metrics."""
    metrics: Dict[str, float]
    rankings: Dict[str, int]
    best_metric: str
    worst_metric: str
    agreement_score: float
    summary: str
    recommendations: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics,
            "rankings": self.rankings,
            "best_metric": self.best_metric,
            "worst_metric": self.worst_metric,
            "agreement_score": self.agreement_score,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp
        }


class BaseDivergenceAnalyzer(ABC):
    """Abstract base class for divergence analyzers."""

    @abstractmethod
    def compute(self, p: Any, q: Any) -> Any:
        """Compute divergence between distributions."""
        pass

    @abstractmethod
    def get_divergence_type(self) -> DivergenceType:
        """Return the type of divergence."""
        pass


class KLDivergenceAnalyzer(BaseDivergenceAnalyzer):
    """
    Analyzer for Kullback-Leibler (KL) Divergence.

    KL Divergence measures how one probability distribution P diverges
    from a reference distribution Q:

    D_KL(P || Q) = Sum_x P(x) * log(P(x) / Q(x))

    Properties:
    - Non-negative: D_KL(P || Q) >= 0
    - Zero iff P = Q
    - Asymmetric: D_KL(P || Q) != D_KL(Q || P)
    - Not a true metric (doesn't satisfy triangle inequality)

    Applications:
    - Measuring information loss when Q approximates P
    - VAE loss function (ELBO)
    - Model comparison
    - Distribution shift detection
    """

    def __init__(self, epsilon: float = 1e-10):
        self.epsilon = epsilon  # For numerical stability

    def get_divergence_type(self) -> DivergenceType:
        return DivergenceType.KL_DIVERGENCE

    def compute(
        self,
        p: List[float],
        q: List[float]
    ) -> DivergenceResult:
        """
        Compute KL divergence D_KL(P || Q).

        Args:
            p: Probability distribution P (true/target distribution)
            q: Probability distribution Q (approximating distribution)

        Returns:
            DivergenceResult with KL divergence value and interpretation
        """
        if len(p) != len(q):
            raise ValueError("Distributions must have same length")

        # Normalize if needed
        p_sum = sum(p)
        q_sum = sum(q)
        p_norm = [x / p_sum for x in p] if p_sum > 0 else p
        q_norm = [x / q_sum for x in q] if q_sum > 0 else q

        # Compute KL divergence
        kl = 0.0
        for p_i, q_i in zip(p_norm, q_norm):
            if p_i > self.epsilon:
                q_safe = max(q_i, self.epsilon)
                kl += p_i * math.log(p_i / q_safe)

        interpretation = self._interpret_kl(kl)
        quality = self._assess_quality(kl)

        return DivergenceResult(
            divergence_type=DivergenceType.KL_DIVERGENCE,
            value=kl,
            is_symmetric=False,
            is_bounded=False,
            bound_range=(0, float('inf')),
            p_distribution_info={"length": len(p), "sum": p_sum, "entropy": self._entropy(p_norm)},
            q_distribution_info={"length": len(q), "sum": q_sum, "entropy": self._entropy(q_norm)},
            interpretation=interpretation,
            quality_assessment=quality
        )

    def compute_reverse(
        self,
        p: List[float],
        q: List[float]
    ) -> DivergenceResult:
        """Compute reverse KL divergence D_KL(Q || P)."""
        result = self.compute(q, p)
        result.divergence_type = DivergenceType.REVERSE_KL
        result.interpretation = result.interpretation.replace("D_KL(P||Q)", "D_KL(Q||P)")
        return result

    def compute_both(
        self,
        p: List[float],
        q: List[float]
    ) -> Dict[str, DivergenceResult]:
        """Compute both forward and reverse KL divergence."""
        return {
            "forward": self.compute(p, q),
            "reverse": self.compute_reverse(p, q)
        }

    def _entropy(self, p: List[float]) -> float:
        """Compute entropy of distribution."""
        h = 0.0
        for p_i in p:
            if p_i > self.epsilon:
                h -= p_i * math.log(p_i)
        return h

    def _interpret_kl(self, kl: float) -> str:
        """Interpret KL divergence value."""
        if kl < 0.01:
            level = "negligible"
            desc = "Distributions are nearly identical."
        elif kl < 0.1:
            level = "small"
            desc = "Minor differences between distributions."
        elif kl < 0.5:
            level = "moderate"
            desc = "Noticeable distribution divergence."
        elif kl < 1.0:
            level = "substantial"
            desc = "Significant distribution differences."
        elif kl < 2.0:
            level = "large"
            desc = "Major distribution mismatch."
        else:
            level = "very large"
            desc = "Extreme distribution divergence."

        return f"D_KL(P||Q) = {kl:.6f} ({level}). {desc}"

    def _assess_quality(self, kl: float) -> str:
        """Assess approximation quality based on KL."""
        if kl < 0.1:
            return "Excellent approximation quality"
        elif kl < 0.5:
            return "Good approximation quality"
        elif kl < 1.0:
            return "Acceptable approximation quality"
        elif kl < 2.0:
            return "Poor approximation quality"
        else:
            return "Very poor approximation quality"


class JensenShannonDivergenceAnalyzer(BaseDivergenceAnalyzer):
    """
    Analyzer for Jensen-Shannon Divergence (JSD).

    JSD is a symmetrized and smoothed version of KL divergence:

    JSD(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)

    Where M = 0.5 * (P + Q) is the mixture distribution.

    Properties:
    - Symmetric: JSD(P || Q) = JSD(Q || P)
    - Bounded: 0 <= JSD <= ln(2) ≈ 0.693 (using natural log)
    - Square root is a true metric
    - More numerically stable than KL
    """

    def __init__(self, base: str = "e"):
        self.base = base
        self.max_value = math.log(2) if base == "e" else 1.0  # log2(2) = 1

    def get_divergence_type(self) -> DivergenceType:
        return DivergenceType.JENSEN_SHANNON

    def compute(
        self,
        p: List[float],
        q: List[float]
    ) -> DivergenceResult:
        """
        Compute Jensen-Shannon Divergence.

        Args:
            p: First probability distribution
            q: Second probability distribution

        Returns:
            DivergenceResult with JSD value and interpretation
        """
        if len(p) != len(q):
            raise ValueError("Distributions must have same length")

        # Normalize
        p_sum = sum(p)
        q_sum = sum(q)
        p_norm = [x / p_sum for x in p] if p_sum > 0 else p
        q_norm = [x / q_sum for x in q] if q_sum > 0 else q

        # Compute mixture M = (P + Q) / 2
        m = [(p_i + q_i) / 2 for p_i, q_i in zip(p_norm, q_norm)]

        # Compute JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        kl_pm = self._kl_divergence(p_norm, m)
        kl_qm = self._kl_divergence(q_norm, m)
        jsd = 0.5 * kl_pm + 0.5 * kl_qm

        # Also compute JS distance (square root)
        js_distance = math.sqrt(jsd)

        interpretation = self._interpret_jsd(jsd)
        quality = self._assess_quality(jsd)

        return DivergenceResult(
            divergence_type=DivergenceType.JENSEN_SHANNON,
            value=jsd,
            is_symmetric=True,
            is_bounded=True,
            bound_range=(0, self.max_value),
            p_distribution_info={"length": len(p), "entropy": self._entropy(p_norm)},
            q_distribution_info={"length": len(q), "entropy": self._entropy(q_norm), "js_distance": js_distance},
            interpretation=interpretation,
            quality_assessment=quality
        )

    def _kl_divergence(self, p: List[float], q: List[float]) -> float:
        """Compute KL divergence."""
        eps = 1e-10
        kl = 0.0
        for p_i, q_i in zip(p, q):
            if p_i > eps:
                q_safe = max(q_i, eps)
                kl += p_i * math.log(p_i / q_safe)
        return kl

    def _entropy(self, p: List[float]) -> float:
        """Compute entropy."""
        eps = 1e-10
        h = 0.0
        for p_i in p:
            if p_i > eps:
                h -= p_i * math.log(p_i)
        return h

    def _interpret_jsd(self, jsd: float) -> str:
        """Interpret JSD value."""
        normalized = jsd / self.max_value  # Normalize to [0, 1]

        if normalized < 0.1:
            level = "very similar"
            desc = "Distributions are nearly identical."
        elif normalized < 0.3:
            level = "similar"
            desc = "Distributions have minor differences."
        elif normalized < 0.5:
            level = "moderately different"
            desc = "Notable differences between distributions."
        elif normalized < 0.7:
            level = "different"
            desc = "Significant distribution differences."
        else:
            level = "very different"
            desc = "Distributions are highly divergent."

        return f"JSD = {jsd:.6f} ({normalized*100:.1f}% of max). Distributions are {level}. {desc}"

    def _assess_quality(self, jsd: float) -> str:
        """Assess similarity quality based on JSD."""
        normalized = jsd / self.max_value
        if normalized < 0.1:
            return "Excellent distribution match"
        elif normalized < 0.3:
            return "Good distribution similarity"
        elif normalized < 0.5:
            return "Moderate distribution similarity"
        else:
            return "Poor distribution similarity"


class WassersteinDistanceAnalyzer(BaseDivergenceAnalyzer):
    """
    Analyzer for Wasserstein Distance (Earth Mover's Distance).

    The Wasserstein distance measures the minimum "cost" of transforming
    one distribution into another, where cost is based on the amount of
    "earth" (probability mass) moved times the distance moved.

    W_p(P, Q) = (inf_γ E_(x,y)~γ [d(x,y)^p])^(1/p)

    Properties:
    - True metric (symmetric, triangle inequality)
    - Meaningful even when distributions don't overlap
    - Accounts for geometry of the space
    - Used in WGAN training
    """

    def __init__(self, order: int = 1):
        self.order = order  # p in W_p distance

    def get_divergence_type(self) -> DivergenceType:
        return DivergenceType.WASSERSTEIN

    def compute(
        self,
        p_samples: List[float],
        q_samples: List[float]
    ) -> WassersteinResult:
        """
        Compute Wasserstein distance between empirical distributions.

        For 1D case, this uses the sorted samples approach.

        Args:
            p_samples: Samples from distribution P
            q_samples: Samples from distribution Q

        Returns:
            WassersteinResult with distance and interpretation
        """
        if not p_samples or not q_samples:
            return self._empty_result()

        # Sort samples for 1D Wasserstein
        p_sorted = sorted(p_samples)
        q_sorted = sorted(q_samples)

        # Statistics
        p_mean = sum(p_sorted) / len(p_sorted)
        q_mean = sum(q_sorted) / len(q_sorted)
        p_std = math.sqrt(sum((x - p_mean) ** 2 for x in p_sorted) / len(p_sorted))
        q_std = math.sqrt(sum((x - q_mean) ** 2 for x in q_sorted) / len(q_sorted))

        # For equal sample sizes: W_1 = average of |sorted_p - sorted_q|
        if len(p_sorted) == len(q_sorted):
            distance = self._compute_equal_samples(p_sorted, q_sorted)
        else:
            # Interpolate to same grid
            distance = self._compute_unequal_samples(p_sorted, q_sorted)

        # Compute transport cost (same as distance for order=1)
        transport_cost = distance ** self.order

        interpretation = self._interpret_wasserstein(distance, p_mean, q_mean, p_std, q_std)

        return WassersteinResult(
            distance=distance,
            order=self.order,
            transport_cost=transport_cost,
            p_mean=p_mean,
            q_mean=q_mean,
            p_std=p_std,
            q_std=q_std,
            optimal_transport_plan=None,  # Simplified version
            interpretation=interpretation
        )

    def _compute_equal_samples(
        self,
        p_sorted: List[float],
        q_sorted: List[float]
    ) -> float:
        """Compute Wasserstein for equal sample sizes."""
        n = len(p_sorted)
        if self.order == 1:
            return sum(abs(p - q) for p, q in zip(p_sorted, q_sorted)) / n
        else:
            return (sum(abs(p - q) ** self.order for p, q in zip(p_sorted, q_sorted)) / n) ** (1 / self.order)

    def _compute_unequal_samples(
        self,
        p_sorted: List[float],
        q_sorted: List[float]
    ) -> float:
        """Compute Wasserstein for unequal sample sizes using linear interpolation."""
        # Create common grid based on quantiles
        n_points = max(len(p_sorted), len(q_sorted))

        def interpolate_quantile(sorted_samples: List[float], q: float) -> float:
            """Interpolate value at quantile q."""
            n = len(sorted_samples)
            idx = q * (n - 1)
            lower = int(idx)
            upper = min(lower + 1, n - 1)
            frac = idx - lower
            return sorted_samples[lower] * (1 - frac) + sorted_samples[upper] * frac

        total = 0.0
        for i in range(n_points):
            q = i / (n_points - 1) if n_points > 1 else 0
            p_val = interpolate_quantile(p_sorted, q)
            q_val = interpolate_quantile(q_sorted, q)
            total += abs(p_val - q_val) ** self.order

        return (total / n_points) ** (1 / self.order)

    def _interpret_wasserstein(
        self,
        distance: float,
        p_mean: float,
        q_mean: float,
        p_std: float,
        q_std: float
    ) -> str:
        """Interpret Wasserstein distance."""
        mean_diff = abs(p_mean - q_mean)
        std_diff = abs(p_std - q_std)

        # Relative distance (normalized by combined std)
        combined_std = (p_std + q_std) / 2 if (p_std + q_std) > 0 else 1
        normalized_dist = distance / combined_std

        if normalized_dist < 0.1:
            level = "very small"
        elif normalized_dist < 0.5:
            level = "small"
        elif normalized_dist < 1.0:
            level = "moderate"
        elif normalized_dist < 2.0:
            level = "large"
        else:
            level = "very large"

        return (
            f"Wasserstein-{self.order} distance = {distance:.6f} ({level}). "
            f"Mean difference: {mean_diff:.4f}, Std difference: {std_diff:.4f}."
        )

    def _empty_result(self) -> WassersteinResult:
        """Return empty result for invalid input."""
        return WassersteinResult(
            distance=float('inf'),
            order=self.order,
            transport_cost=float('inf'),
            p_mean=0.0,
            q_mean=0.0,
            p_std=0.0,
            q_std=0.0,
            optimal_transport_plan=None,
            interpretation="Insufficient data for Wasserstein computation"
        )


class MMDAnalyzer(BaseDivergenceAnalyzer):
    """
    Analyzer for Maximum Mean Discrepancy (MMD).

    MMD is a kernel-based statistical test for comparing distributions:

    MMD^2(P, Q) = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]

    Where k is a kernel function, x,x' ~ P, y,y' ~ Q.

    Properties:
    - Non-parametric (no distributional assumptions)
    - Can detect any difference if using characteristic kernel
    - Used in generative model evaluation
    - Basis for two-sample tests
    """

    def __init__(
        self,
        kernel_type: KernelType = KernelType.RBF,
        kernel_bandwidth: float = 1.0
    ):
        self.kernel_type = kernel_type
        self.kernel_bandwidth = kernel_bandwidth

    def get_divergence_type(self) -> DivergenceType:
        return DivergenceType.MMD

    def compute(
        self,
        p_samples: List[List[float]],
        q_samples: List[List[float]]
    ) -> MMDResult:
        """
        Compute MMD between two sets of samples.

        Args:
            p_samples: Samples from distribution P (each sample is a feature vector)
            q_samples: Samples from distribution Q

        Returns:
            MMDResult with MMD value and interpretation
        """
        if not p_samples or not q_samples:
            return self._empty_result()

        n_p = len(p_samples)
        n_q = len(q_samples)

        # Compute kernel matrices
        k_pp = self._kernel_matrix(p_samples, p_samples)
        k_qq = self._kernel_matrix(q_samples, q_samples)
        k_pq = self._kernel_matrix(p_samples, q_samples)

        # Compute MMD^2 (unbiased estimator)
        sum_kpp = sum(k_pp[i][j] for i in range(n_p) for j in range(n_p) if i != j)
        sum_kqq = sum(k_qq[i][j] for i in range(n_q) for j in range(n_q) if i != j)
        sum_kpq = sum(k_pq[i][j] for i in range(n_p) for j in range(n_q))

        mmd_squared = (
            sum_kpp / (n_p * (n_p - 1)) if n_p > 1 else 0
        ) - 2 * (
            sum_kpq / (n_p * n_q)
        ) + (
            sum_kqq / (n_q * (n_q - 1)) if n_q > 1 else 0
        )

        # Handle numerical issues
        mmd_squared = max(0, mmd_squared)
        mmd_value = math.sqrt(mmd_squared)

        # Significance threshold (approximate)
        significance_threshold = self._compute_threshold(n_p, n_q)
        is_significant = mmd_value > significance_threshold

        interpretation = self._interpret_mmd(mmd_value, is_significant)

        return MMDResult(
            mmd_value=mmd_value,
            mmd_squared=mmd_squared,
            kernel_type=self.kernel_type,
            kernel_params={"bandwidth": self.kernel_bandwidth},
            p_sample_size=n_p,
            q_sample_size=n_q,
            significance_threshold=significance_threshold,
            is_significant=is_significant,
            confidence_interval=None,
            interpretation=interpretation
        )

    def _kernel(self, x: List[float], y: List[float]) -> float:
        """Compute kernel value k(x, y)."""
        if self.kernel_type == KernelType.RBF:
            # RBF (Gaussian) kernel: exp(-||x-y||^2 / (2*sigma^2))
            sq_dist = sum((xi - yi) ** 2 for xi, yi in zip(x, y))
            return math.exp(-sq_dist / (2 * self.kernel_bandwidth ** 2))

        elif self.kernel_type == KernelType.LINEAR:
            return sum(xi * yi for xi, yi in zip(x, y))

        elif self.kernel_type == KernelType.POLYNOMIAL:
            # Polynomial kernel: (x·y + 1)^d, using d=2
            dot = sum(xi * yi for xi, yi in zip(x, y))
            return (dot + 1) ** 2

        elif self.kernel_type == KernelType.LAPLACIAN:
            # Laplacian kernel: exp(-||x-y|| / sigma)
            dist = math.sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x, y)))
            return math.exp(-dist / self.kernel_bandwidth)

        elif self.kernel_type == KernelType.IMQ:
            # Inverse multiquadric: 1 / sqrt(c^2 + ||x-y||^2)
            sq_dist = sum((xi - yi) ** 2 for xi, yi in zip(x, y))
            return 1 / math.sqrt(self.kernel_bandwidth ** 2 + sq_dist)

        else:
            # Default to RBF
            sq_dist = sum((xi - yi) ** 2 for xi, yi in zip(x, y))
            return math.exp(-sq_dist / (2 * self.kernel_bandwidth ** 2))

    def _kernel_matrix(
        self,
        samples1: List[List[float]],
        samples2: List[List[float]]
    ) -> List[List[float]]:
        """Compute kernel matrix between two sets of samples."""
        n1 = len(samples1)
        n2 = len(samples2)
        return [[self._kernel(samples1[i], samples2[j]) for j in range(n2)] for i in range(n1)]

    def _compute_threshold(self, n_p: int, n_q: int) -> float:
        """Compute approximate significance threshold."""
        # Simplified threshold based on sample sizes
        # In practice, this would use permutation testing or bootstrap
        n = min(n_p, n_q)
        return 2.0 / math.sqrt(n) if n > 0 else float('inf')

    def _interpret_mmd(self, mmd: float, is_significant: bool) -> str:
        """Interpret MMD result."""
        if mmd < 0.01:
            level = "negligible"
        elif mmd < 0.05:
            level = "very small"
        elif mmd < 0.1:
            level = "small"
        elif mmd < 0.2:
            level = "moderate"
        else:
            level = "large"

        significance = "significantly different" if is_significant else "not significantly different"

        return f"MMD = {mmd:.6f} ({level}). Distributions are {significance}."

    def _empty_result(self) -> MMDResult:
        """Return empty result for invalid input."""
        return MMDResult(
            mmd_value=float('inf'),
            mmd_squared=float('inf'),
            kernel_type=self.kernel_type,
            kernel_params={"bandwidth": self.kernel_bandwidth},
            p_sample_size=0,
            q_sample_size=0,
            significance_threshold=float('inf'),
            is_significant=False,
            confidence_interval=None,
            interpretation="Insufficient data for MMD computation"
        )


class TotalVariationAnalyzer(BaseDivergenceAnalyzer):
    """
    Analyzer for Total Variation Distance.

    Total Variation (TV) distance measures the maximum difference in
    probability between two distributions:

    TV(P, Q) = 0.5 * Sum_x |P(x) - Q(x)| = sup_A |P(A) - Q(A)|

    Properties:
    - Symmetric
    - Bounded: 0 <= TV <= 1
    - True metric
    - Equals 0 iff P = Q
    """

    def get_divergence_type(self) -> DivergenceType:
        return DivergenceType.TOTAL_VARIATION

    def compute(
        self,
        p: List[float],
        q: List[float]
    ) -> DivergenceResult:
        """
        Compute Total Variation distance.

        Args:
            p: First probability distribution
            q: Second probability distribution

        Returns:
            DivergenceResult with TV distance
        """
        if len(p) != len(q):
            raise ValueError("Distributions must have same length")

        # Normalize
        p_sum = sum(p)
        q_sum = sum(q)
        p_norm = [x / p_sum for x in p] if p_sum > 0 else p
        q_norm = [x / q_sum for x in q] if q_sum > 0 else q

        # TV = 0.5 * sum |p - q|
        tv = 0.5 * sum(abs(pi - qi) for pi, qi in zip(p_norm, q_norm))

        interpretation = self._interpret_tv(tv)

        return DivergenceResult(
            divergence_type=DivergenceType.TOTAL_VARIATION,
            value=tv,
            is_symmetric=True,
            is_bounded=True,
            bound_range=(0, 1),
            p_distribution_info={"length": len(p)},
            q_distribution_info={"length": len(q)},
            interpretation=interpretation,
            quality_assessment=self._assess_quality(tv)
        )

    def _interpret_tv(self, tv: float) -> str:
        """Interpret Total Variation distance."""
        if tv < 0.05:
            level = "nearly identical"
        elif tv < 0.15:
            level = "very similar"
        elif tv < 0.3:
            level = "similar"
        elif tv < 0.5:
            level = "moderately different"
        else:
            level = "very different"

        return f"TV distance = {tv:.6f}. Distributions are {level}."

    def _assess_quality(self, tv: float) -> str:
        """Assess similarity quality."""
        if tv < 0.1:
            return "Excellent distribution match"
        elif tv < 0.25:
            return "Good distribution match"
        elif tv < 0.4:
            return "Moderate distribution match"
        else:
            return "Poor distribution match"


class HellingerDistanceAnalyzer(BaseDivergenceAnalyzer):
    """
    Analyzer for Hellinger Distance.

    Hellinger distance is related to Bhattacharyya coefficient:

    H(P, Q) = sqrt(1 - BC(P, Q))

    Where BC(P, Q) = Sum_x sqrt(P(x) * Q(x)) is Bhattacharyya coefficient.

    Properties:
    - Symmetric
    - Bounded: 0 <= H <= 1
    - Related to TV: H^2 <= TV <= sqrt(2) * H
    - True metric
    """

    def get_divergence_type(self) -> DivergenceType:
        return DivergenceType.HELLINGER

    def compute(
        self,
        p: List[float],
        q: List[float]
    ) -> DivergenceResult:
        """
        Compute Hellinger distance.

        Args:
            p: First probability distribution
            q: Second probability distribution

        Returns:
            DivergenceResult with Hellinger distance
        """
        if len(p) != len(q):
            raise ValueError("Distributions must have same length")

        # Normalize
        p_sum = sum(p)
        q_sum = sum(q)
        p_norm = [x / p_sum for x in p] if p_sum > 0 else p
        q_norm = [x / q_sum for x in q] if q_sum > 0 else q

        # Bhattacharyya coefficient
        bc = sum(math.sqrt(pi * qi) for pi, qi in zip(p_norm, q_norm))

        # Hellinger distance
        hellinger = math.sqrt(1 - min(1, bc))  # min to handle numerical errors

        interpretation = self._interpret_hellinger(hellinger)

        return DivergenceResult(
            divergence_type=DivergenceType.HELLINGER,
            value=hellinger,
            is_symmetric=True,
            is_bounded=True,
            bound_range=(0, 1),
            p_distribution_info={"length": len(p), "bhattacharyya_coef": bc},
            q_distribution_info={"length": len(q)},
            interpretation=interpretation,
            quality_assessment=self._assess_quality(hellinger)
        )

    def _interpret_hellinger(self, h: float) -> str:
        """Interpret Hellinger distance."""
        if h < 0.1:
            level = "nearly identical"
        elif h < 0.25:
            level = "very similar"
        elif h < 0.4:
            level = "similar"
        elif h < 0.6:
            level = "moderately different"
        else:
            level = "very different"

        return f"Hellinger distance = {h:.6f}. Distributions are {level}."

    def _assess_quality(self, h: float) -> str:
        """Assess similarity quality."""
        if h < 0.15:
            return "Excellent distribution match"
        elif h < 0.3:
            return "Good distribution match"
        elif h < 0.5:
            return "Moderate distribution match"
        else:
            return "Poor distribution match"


class RenyiDivergenceAnalyzer(BaseDivergenceAnalyzer):
    """
    Analyzer for Rényi Divergence.

    Rényi divergence generalizes KL divergence with parameter alpha:

    D_α(P || Q) = 1/(α-1) * log(Sum_x P(x)^α * Q(x)^(1-α))

    Special cases:
    - α → 1: KL divergence
    - α = 0.5: Related to Hellinger
    - α = 2: Chi-squared divergence
    - α → ∞: Max-divergence
    """

    def __init__(self, alpha: float = 0.5):
        if alpha == 1:
            raise ValueError("Alpha cannot be exactly 1 (use KL divergence instead)")
        self.alpha = alpha

    def get_divergence_type(self) -> DivergenceType:
        return DivergenceType.RENYI

    def compute(
        self,
        p: List[float],
        q: List[float]
    ) -> DivergenceResult:
        """
        Compute Rényi divergence.

        Args:
            p: First probability distribution
            q: Second probability distribution

        Returns:
            DivergenceResult with Rényi divergence
        """
        if len(p) != len(q):
            raise ValueError("Distributions must have same length")

        eps = 1e-10

        # Normalize
        p_sum = sum(p)
        q_sum = sum(q)
        p_norm = [x / p_sum for x in p] if p_sum > 0 else p
        q_norm = [x / q_sum for x in q] if q_sum > 0 else q

        # Compute Rényi divergence
        sum_term = sum(
            (pi ** self.alpha) * (max(qi, eps) ** (1 - self.alpha))
            for pi, qi in zip(p_norm, q_norm)
            if pi > eps
        )

        if sum_term > 0:
            renyi = math.log(sum_term) / (self.alpha - 1)
        else:
            renyi = float('inf')

        interpretation = f"Rényi divergence (α={self.alpha}) = {renyi:.6f}."

        return DivergenceResult(
            divergence_type=DivergenceType.RENYI,
            value=renyi,
            is_symmetric=False,
            is_bounded=False,
            bound_range=(0, float('inf')),
            p_distribution_info={"length": len(p), "alpha": self.alpha},
            q_distribution_info={"length": len(q)},
            interpretation=interpretation,
            quality_assessment=self._assess_quality(renyi)
        )

    def _assess_quality(self, renyi: float) -> str:
        """Assess based on Rényi divergence."""
        if renyi < 0.1:
            return "Excellent distribution match"
        elif renyi < 0.5:
            return "Good distribution match"
        elif renyi < 1.0:
            return "Moderate distribution match"
        else:
            return "Poor distribution match"


class ComprehensiveDivergenceAnalyzer:
    """
    Comprehensive analyzer that computes multiple divergence metrics.

    Useful for comparing distributions using various perspectives
    and for research requiring multiple metrics.
    """

    def __init__(self):
        self.kl_analyzer = KLDivergenceAnalyzer()
        self.js_analyzer = JensenShannonDivergenceAnalyzer()
        self.tv_analyzer = TotalVariationAnalyzer()
        self.hellinger_analyzer = HellingerDistanceAnalyzer()

    def analyze(
        self,
        p: List[float],
        q: List[float],
        include_asymmetric: bool = True
    ) -> DivergenceComparison:
        """
        Compute multiple divergence metrics.

        Args:
            p: First probability distribution
            q: Second probability distribution
            include_asymmetric: Whether to include asymmetric metrics

        Returns:
            DivergenceComparison with all metrics
        """
        metrics = {}

        # Symmetric metrics
        js_result = self.js_analyzer.compute(p, q)
        metrics["jensen_shannon"] = js_result.value

        tv_result = self.tv_analyzer.compute(p, q)
        metrics["total_variation"] = tv_result.value

        hellinger_result = self.hellinger_analyzer.compute(p, q)
        metrics["hellinger"] = hellinger_result.value

        # Asymmetric metrics
        if include_asymmetric:
            kl_forward = self.kl_analyzer.compute(p, q)
            kl_reverse = self.kl_analyzer.compute(q, p)
            metrics["kl_forward"] = kl_forward.value
            metrics["kl_reverse"] = kl_reverse.value

        # Rank metrics (lower is better for all)
        rankings = {k: i + 1 for i, (k, v) in enumerate(sorted(metrics.items(), key=lambda x: x[1]))}

        best_metric = min(metrics.items(), key=lambda x: x[1])[0]
        worst_metric = max(metrics.items(), key=lambda x: x[1])[0]

        # Agreement score (how consistent are the metrics)
        normalized_metrics = [v / max(metrics.values()) if max(metrics.values()) > 0 else 0 for v in metrics.values()]
        agreement_score = 1 - (max(normalized_metrics) - min(normalized_metrics)) if normalized_metrics else 0

        summary = self._generate_summary(metrics, agreement_score)
        recommendations = self._generate_recommendations(metrics)

        return DivergenceComparison(
            metrics=metrics,
            rankings=rankings,
            best_metric=best_metric,
            worst_metric=worst_metric,
            agreement_score=agreement_score,
            summary=summary,
            recommendations=recommendations
        )

    def _generate_summary(self, metrics: Dict[str, float], agreement: float) -> str:
        """Generate summary of divergence analysis."""
        avg_divergence = sum(metrics.values()) / len(metrics) if metrics else 0

        if avg_divergence < 0.1:
            similarity = "very similar"
        elif avg_divergence < 0.3:
            similarity = "similar"
        elif avg_divergence < 0.5:
            similarity = "moderately different"
        else:
            similarity = "very different"

        agreement_level = "high" if agreement > 0.7 else "moderate" if agreement > 0.4 else "low"

        return (
            f"Distributions are {similarity} across metrics. "
            f"Metric agreement is {agreement_level} ({agreement:.2f})."
        )

    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        js = metrics.get("jensen_shannon", 0)
        if js > 0.3:
            recommendations.append("Consider distribution alignment techniques")

        kl_forward = metrics.get("kl_forward", 0)
        kl_reverse = metrics.get("kl_reverse", 0)
        if kl_forward > 0 and kl_reverse > 0:
            asymmetry = abs(kl_forward - kl_reverse) / max(kl_forward, kl_reverse)
            if asymmetry > 0.5:
                recommendations.append("Significant KL asymmetry detected; consider direction-specific analysis")

        if not recommendations:
            recommendations.append("Distribution similarity is acceptable")

        return recommendations


# Utility functions
def compute_kl_divergence(p: List[float], q: List[float]) -> DivergenceResult:
    """Convenience function for KL divergence."""
    analyzer = KLDivergenceAnalyzer()
    return analyzer.compute(p, q)


def compute_js_divergence(p: List[float], q: List[float]) -> DivergenceResult:
    """Convenience function for Jensen-Shannon divergence."""
    analyzer = JensenShannonDivergenceAnalyzer()
    return analyzer.compute(p, q)


def compute_wasserstein_distance(
    p_samples: List[float],
    q_samples: List[float],
    order: int = 1
) -> WassersteinResult:
    """Convenience function for Wasserstein distance."""
    analyzer = WassersteinDistanceAnalyzer(order=order)
    return analyzer.compute(p_samples, q_samples)


def compute_mmd(
    p_samples: List[List[float]],
    q_samples: List[List[float]],
    kernel: KernelType = KernelType.RBF,
    bandwidth: float = 1.0
) -> MMDResult:
    """Convenience function for MMD computation."""
    analyzer = MMDAnalyzer(kernel_type=kernel, kernel_bandwidth=bandwidth)
    return analyzer.compute(p_samples, q_samples)


def compute_all_divergences(
    p: List[float],
    q: List[float]
) -> DivergenceComparison:
    """Convenience function for comprehensive divergence analysis."""
    analyzer = ComprehensiveDivergenceAnalyzer()
    return analyzer.analyze(p, q)
