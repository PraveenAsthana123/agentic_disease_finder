"""
Probability Analysis Module for AI/GenAI Research Evaluation

This module provides comprehensive probability analysis capabilities for
evaluating AI systems, including marginal, joint, conditional, posterior,
and prior probability computations essential for research quality assessment.

Key Concepts:
- Marginal Probability P(X): Probability of X regardless of other variables
- Joint Probability P(X,Y): Probability of X and Y occurring together
- Conditional Probability P(X|Y): Probability of X given Y
- Posterior Probability P(H|E): Updated probability after observing evidence
- Prior Probability P(H): Initial probability before observing evidence
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum
from abc import ABC, abstractmethod
import math
from datetime import datetime


class ProbabilityType(Enum):
    """Types of probability distributions and computations."""
    MARGINAL = "marginal"
    JOINT = "joint"
    CONDITIONAL = "conditional"
    POSTERIOR = "posterior"
    PRIOR = "prior"
    LIKELIHOOD = "likelihood"
    PREDICTIVE = "predictive"
    EMPIRICAL = "empirical"


class DistributionType(Enum):
    """Types of probability distributions."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    GAUSSIAN = "gaussian"
    BERNOULLI = "bernoulli"
    MULTINOMIAL = "multinomial"
    POISSON = "poisson"
    UNIFORM = "uniform"
    BETA = "beta"
    DIRICHLET = "dirichlet"


class InferenceMethod(Enum):
    """Bayesian inference methods."""
    EXACT = "exact"
    MCMC = "mcmc"
    VARIATIONAL = "variational"
    LAPLACE = "laplace"
    MAP = "maximum_a_posteriori"
    MLE = "maximum_likelihood"


@dataclass
class ProbabilityResult:
    """Result of a probability computation."""
    probability_type: ProbabilityType
    value: float
    confidence_interval: Optional[Tuple[float, float]]
    standard_error: Optional[float]
    sample_size: int
    distribution_type: DistributionType
    parameters: Dict[str, Any]
    interpretation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probability_type": self.probability_type.value,
            "value": self.value,
            "confidence_interval": self.confidence_interval,
            "standard_error": self.standard_error,
            "sample_size": self.sample_size,
            "distribution_type": self.distribution_type.value,
            "parameters": self.parameters,
            "interpretation": self.interpretation,
            "timestamp": self.timestamp
        }


@dataclass
class JointProbabilityResult:
    """Result of joint probability analysis."""
    joint_probability: float
    marginal_x: float
    marginal_y: float
    conditional_x_given_y: float
    conditional_y_given_x: float
    mutual_information: float
    correlation: float
    independence_test: bool
    interpretation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joint_probability": self.joint_probability,
            "marginal_x": self.marginal_x,
            "marginal_y": self.marginal_y,
            "conditional_x_given_y": self.conditional_x_given_y,
            "conditional_y_given_x": self.conditional_y_given_x,
            "mutual_information": self.mutual_information,
            "correlation": self.correlation,
            "independence_test": self.independence_test,
            "interpretation": self.interpretation,
            "timestamp": self.timestamp
        }


@dataclass
class BayesianInferenceResult:
    """Result of Bayesian inference computation."""
    prior: Dict[str, float]
    likelihood: Dict[str, float]
    posterior: Dict[str, float]
    evidence: float
    map_estimate: str
    posterior_mean: float
    posterior_variance: float
    credible_interval: Tuple[float, float]
    bayes_factor: Optional[float]
    inference_method: InferenceMethod
    interpretation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prior": self.prior,
            "likelihood": self.likelihood,
            "posterior": self.posterior,
            "evidence": self.evidence,
            "map_estimate": self.map_estimate,
            "posterior_mean": self.posterior_mean,
            "posterior_variance": self.posterior_variance,
            "credible_interval": self.credible_interval,
            "bayes_factor": self.bayes_factor,
            "inference_method": self.inference_method.value,
            "interpretation": self.interpretation,
            "timestamp": self.timestamp
        }


@dataclass
class ProbabilityDistribution:
    """Representation of a probability distribution."""
    distribution_type: DistributionType
    parameters: Dict[str, float]
    support: Tuple[float, float]
    mean: float
    variance: float
    entropy: float
    samples: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "distribution_type": self.distribution_type.value,
            "parameters": self.parameters,
            "support": self.support,
            "mean": self.mean,
            "variance": self.variance,
            "entropy": self.entropy,
            "num_samples": len(self.samples) if self.samples else 0
        }


class BaseProbabilityAnalyzer(ABC):
    """Abstract base class for probability analyzers."""

    @abstractmethod
    def analyze(self, data: Any) -> Any:
        """Perform probability analysis."""
        pass

    @abstractmethod
    def get_analyzer_type(self) -> str:
        """Return the type of analyzer."""
        pass


class MarginalProbabilityAnalyzer(BaseProbabilityAnalyzer):
    """
    Analyzer for computing marginal probabilities.

    Marginal probability P(X) is the probability of event X occurring
    regardless of the values of other variables. It is computed by
    summing (discrete) or integrating (continuous) over all other variables.

    P(X) = Sum_Y P(X, Y) for discrete variables
    """

    def __init__(self, smoothing: float = 0.0):
        self.smoothing = smoothing  # Laplace smoothing parameter

    def get_analyzer_type(self) -> str:
        return "marginal_probability"

    def analyze(
        self,
        observations: List[Any],
        categories: Optional[List[Any]] = None
    ) -> Dict[str, ProbabilityResult]:
        """
        Compute marginal probabilities from observations.

        Args:
            observations: List of observed values
            categories: Optional list of all possible categories

        Returns:
            Dictionary mapping categories to their marginal probabilities
        """
        if not observations:
            return {}

        # Count occurrences
        counts: Dict[Any, int] = {}
        for obs in observations:
            counts[obs] = counts.get(obs, 0) + 1

        # Include zero-count categories if provided
        if categories:
            for cat in categories:
                if cat not in counts:
                    counts[cat] = 0

        # Calculate probabilities with optional smoothing
        n = len(observations)
        k = len(counts)  # Number of categories
        results = {}

        for category, count in counts.items():
            # Laplace smoothing: (count + alpha) / (n + alpha * k)
            smoothed_count = count + self.smoothing
            smoothed_total = n + self.smoothing * k

            prob = smoothed_count / smoothed_total if smoothed_total > 0 else 0.0

            # Calculate confidence interval (Wilson score interval)
            ci = self._wilson_confidence_interval(count, n)

            # Standard error
            se = math.sqrt(prob * (1 - prob) / n) if n > 0 else 0.0

            results[str(category)] = ProbabilityResult(
                probability_type=ProbabilityType.MARGINAL,
                value=prob,
                confidence_interval=ci,
                standard_error=se,
                sample_size=n,
                distribution_type=DistributionType.CATEGORICAL,
                parameters={"count": count, "smoothing": self.smoothing},
                interpretation=self._interpret_marginal(category, prob, n)
            )

        return results

    def _wilson_confidence_interval(
        self,
        successes: int,
        n: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate Wilson score confidence interval."""
        if n == 0:
            return (0.0, 1.0)

        z = 1.96  # 95% confidence
        p_hat = successes / n

        denominator = 1 + z * z / n
        center = (p_hat + z * z / (2 * n)) / denominator
        margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denominator

        return (max(0, center - margin), min(1, center + margin))

    def _interpret_marginal(self, category: Any, prob: float, n: int) -> str:
        """Generate interpretation of marginal probability."""
        if prob >= 0.5:
            level = "dominant"
        elif prob >= 0.25:
            level = "significant"
        elif prob >= 0.1:
            level = "moderate"
        elif prob >= 0.01:
            level = "minor"
        else:
            level = "rare"

        return (
            f"Category '{category}' has {level} marginal probability of {prob:.4f} "
            f"based on {n} observations."
        )


class JointProbabilityAnalyzer(BaseProbabilityAnalyzer):
    """
    Analyzer for computing joint probabilities.

    Joint probability P(X, Y) is the probability of both X and Y
    occurring together. Key relationships:
    - P(X, Y) = P(X|Y) * P(Y) = P(Y|X) * P(X)
    - P(X) = Sum_Y P(X, Y) (marginalization)
    - If independent: P(X, Y) = P(X) * P(Y)
    """

    def __init__(self, independence_threshold: float = 0.05):
        self.independence_threshold = independence_threshold

    def get_analyzer_type(self) -> str:
        return "joint_probability"

    def analyze(
        self,
        observations_x: List[Any],
        observations_y: List[Any]
    ) -> JointProbabilityResult:
        """
        Compute joint probability analysis for two variables.

        Args:
            observations_x: Observations of variable X
            observations_y: Observations of variable Y

        Returns:
            JointProbabilityResult with comprehensive analysis
        """
        if len(observations_x) != len(observations_y):
            raise ValueError("Observation lists must have equal length")

        n = len(observations_x)
        if n == 0:
            return self._empty_result()

        # Count joint occurrences
        joint_counts: Dict[Tuple[Any, Any], int] = {}
        for x, y in zip(observations_x, observations_y):
            key = (x, y)
            joint_counts[key] = joint_counts.get(key, 0) + 1

        # Count marginal occurrences
        x_counts: Dict[Any, int] = {}
        y_counts: Dict[Any, int] = {}
        for x in observations_x:
            x_counts[x] = x_counts.get(x, 0) + 1
        for y in observations_y:
            y_counts[y] = y_counts.get(y, 0) + 1

        # Calculate probabilities for most common pair
        most_common_pair = max(joint_counts.items(), key=lambda x: x[1])
        (x_val, y_val), joint_count = most_common_pair

        joint_prob = joint_count / n
        marginal_x = x_counts[x_val] / n
        marginal_y = y_counts[y_val] / n

        # Conditional probabilities
        cond_x_given_y = joint_count / y_counts[y_val] if y_counts[y_val] > 0 else 0
        cond_y_given_x = joint_count / x_counts[x_val] if x_counts[x_val] > 0 else 0

        # Mutual information
        mutual_info = self._calculate_mutual_information(
            joint_counts, x_counts, y_counts, n
        )

        # Correlation (Cramér's V for categorical)
        correlation = self._calculate_cramers_v(joint_counts, x_counts, y_counts, n)

        # Independence test
        expected_under_independence = marginal_x * marginal_y
        independence = abs(joint_prob - expected_under_independence) < self.independence_threshold

        interpretation = self._interpret_joint(
            joint_prob, marginal_x, marginal_y, mutual_info, independence
        )

        return JointProbabilityResult(
            joint_probability=joint_prob,
            marginal_x=marginal_x,
            marginal_y=marginal_y,
            conditional_x_given_y=cond_x_given_y,
            conditional_y_given_x=cond_y_given_x,
            mutual_information=mutual_info,
            correlation=correlation,
            independence_test=independence,
            interpretation=interpretation
        )

    def _calculate_mutual_information(
        self,
        joint_counts: Dict[Tuple[Any, Any], int],
        x_counts: Dict[Any, int],
        y_counts: Dict[Any, int],
        n: int
    ) -> float:
        """Calculate mutual information I(X; Y)."""
        mi = 0.0
        for (x, y), joint_count in joint_counts.items():
            p_xy = joint_count / n
            p_x = x_counts[x] / n
            p_y = y_counts[y] / n
            if p_xy > 0 and p_x > 0 and p_y > 0:
                mi += p_xy * math.log(p_xy / (p_x * p_y))
        return mi

    def _calculate_cramers_v(
        self,
        joint_counts: Dict[Tuple[Any, Any], int],
        x_counts: Dict[Any, int],
        y_counts: Dict[Any, int],
        n: int
    ) -> float:
        """Calculate Cramér's V correlation coefficient."""
        # Chi-squared statistic
        chi2 = 0.0
        for (x, y), observed in joint_counts.items():
            expected = (x_counts[x] * y_counts[y]) / n
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected

        # Cramér's V
        k_x = len(x_counts)
        k_y = len(y_counts)
        min_dim = min(k_x, k_y) - 1
        if min_dim > 0 and n > 0:
            return math.sqrt(chi2 / (n * min_dim))
        return 0.0

    def _interpret_joint(
        self,
        joint: float,
        marg_x: float,
        marg_y: float,
        mi: float,
        independent: bool
    ) -> str:
        """Generate interpretation of joint probability analysis."""
        parts = [
            f"Joint probability P(X,Y) = {joint:.4f}, "
            f"marginals P(X) = {marg_x:.4f}, P(Y) = {marg_y:.4f}."
        ]

        if independent:
            parts.append("Variables appear to be independent.")
        else:
            parts.append("Variables show dependence.")

        if mi > 0.5:
            parts.append(f"Strong mutual information ({mi:.4f}) indicates high dependency.")
        elif mi > 0.1:
            parts.append(f"Moderate mutual information ({mi:.4f}).")
        else:
            parts.append(f"Low mutual information ({mi:.4f}).")

        return " ".join(parts)

    def _empty_result(self) -> JointProbabilityResult:
        """Return empty result for invalid input."""
        return JointProbabilityResult(
            joint_probability=0.0,
            marginal_x=0.0,
            marginal_y=0.0,
            conditional_x_given_y=0.0,
            conditional_y_given_x=0.0,
            mutual_information=0.0,
            correlation=0.0,
            independence_test=True,
            interpretation="Insufficient data for analysis"
        )


class ConditionalProbabilityAnalyzer(BaseProbabilityAnalyzer):
    """
    Analyzer for computing conditional probabilities.

    Conditional probability P(A|B) is the probability of A given that
    B has occurred. Computed as:
    P(A|B) = P(A, B) / P(B) = P(A ∩ B) / P(B)

    Key properties:
    - P(A|B) ≠ P(B|A) in general
    - Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)
    """

    def __init__(self):
        pass

    def get_analyzer_type(self) -> str:
        return "conditional_probability"

    def analyze(
        self,
        observations_a: List[Any],
        observations_b: List[Any],
        condition_value: Any
    ) -> Dict[str, ProbabilityResult]:
        """
        Compute conditional probabilities P(A|B=condition_value).

        Args:
            observations_a: Observations of variable A
            observations_b: Observations of variable B (conditioning variable)
            condition_value: Value of B to condition on

        Returns:
            Dictionary of conditional probabilities for each value of A
        """
        if len(observations_a) != len(observations_b):
            raise ValueError("Observation lists must have equal length")

        # Filter observations where B = condition_value
        filtered_a = [
            a for a, b in zip(observations_a, observations_b)
            if b == condition_value
        ]

        n_condition = len(filtered_a)
        if n_condition == 0:
            return {}

        # Count occurrences of each A value given the condition
        a_counts: Dict[Any, int] = {}
        for a in filtered_a:
            a_counts[a] = a_counts.get(a, 0) + 1

        # Calculate conditional probabilities
        results = {}
        for a_val, count in a_counts.items():
            prob = count / n_condition

            # Confidence interval
            ci = self._wilson_ci(count, n_condition)

            results[str(a_val)] = ProbabilityResult(
                probability_type=ProbabilityType.CONDITIONAL,
                value=prob,
                confidence_interval=ci,
                standard_error=math.sqrt(prob * (1 - prob) / n_condition) if n_condition > 0 else 0,
                sample_size=n_condition,
                distribution_type=DistributionType.CATEGORICAL,
                parameters={"condition": str(condition_value), "count": count},
                interpretation=(
                    f"P({a_val}|{condition_value}) = {prob:.4f} based on "
                    f"{n_condition} observations where condition holds."
                )
            )

        return results

    def analyze_all_conditions(
        self,
        observations_a: List[Any],
        observations_b: List[Any]
    ) -> Dict[Any, Dict[str, ProbabilityResult]]:
        """
        Compute conditional probabilities for all values of B.

        Args:
            observations_a: Observations of variable A
            observations_b: Observations of variable B

        Returns:
            Nested dictionary: {b_value: {a_value: probability}}
        """
        unique_b = set(observations_b)
        results = {}

        for b_val in unique_b:
            results[b_val] = self.analyze(observations_a, observations_b, b_val)

        return results

    def _wilson_ci(self, successes: int, n: int) -> Tuple[float, float]:
        """Wilson score confidence interval."""
        if n == 0:
            return (0.0, 1.0)
        z = 1.96
        p_hat = successes / n
        denom = 1 + z * z / n
        center = (p_hat + z * z / (2 * n)) / denom
        margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denom
        return (max(0, center - margin), min(1, center + margin))


class BayesianInferenceAnalyzer(BaseProbabilityAnalyzer):
    """
    Analyzer for Bayesian inference computations.

    Implements Bayes' Theorem: P(H|E) = P(E|H) * P(H) / P(E)

    Where:
    - P(H): Prior probability of hypothesis
    - P(E|H): Likelihood of evidence given hypothesis
    - P(E): Evidence (marginal likelihood)
    - P(H|E): Posterior probability of hypothesis given evidence
    """

    def __init__(self, inference_method: InferenceMethod = InferenceMethod.EXACT):
        self.inference_method = inference_method

    def get_analyzer_type(self) -> str:
        return "bayesian_inference"

    def analyze(
        self,
        prior: Dict[str, float],
        likelihood: Dict[str, float],
        evidence_key: Optional[str] = None
    ) -> BayesianInferenceResult:
        """
        Perform Bayesian inference to compute posterior probabilities.

        Args:
            prior: Dictionary of prior probabilities P(H) for each hypothesis
            likelihood: Dictionary of likelihoods P(E|H) for each hypothesis
            evidence_key: Optional key for specific evidence

        Returns:
            BayesianInferenceResult with posterior analysis
        """
        if not prior or not likelihood:
            return self._empty_result()

        # Ensure priors sum to 1
        prior_sum = sum(prior.values())
        normalized_prior = {k: v / prior_sum for k, v in prior.items()}

        # Compute evidence P(E) = Sum_H P(E|H) * P(H)
        evidence = sum(
            likelihood.get(h, 0) * normalized_prior[h]
            for h in normalized_prior
        )

        if evidence == 0:
            return self._empty_result()

        # Compute posterior P(H|E) = P(E|H) * P(H) / P(E)
        posterior = {}
        for hypothesis in normalized_prior:
            p_e_given_h = likelihood.get(hypothesis, 0)
            p_h = normalized_prior[hypothesis]
            posterior[hypothesis] = (p_e_given_h * p_h) / evidence

        # Find MAP estimate
        map_estimate = max(posterior.items(), key=lambda x: x[1])[0]

        # Calculate posterior statistics
        # Assuming hypotheses are numeric or can be indexed
        try:
            numeric_hypotheses = [(float(h), p) for h, p in posterior.items()]
            posterior_mean = sum(h * p for h, p in numeric_hypotheses)
            posterior_variance = sum(
                p * (h - posterior_mean) ** 2 for h, p in numeric_hypotheses
            )
        except (ValueError, TypeError):
            # Non-numeric hypotheses
            posterior_mean = 0.0
            posterior_variance = 0.0

        # Credible interval (for posterior distribution)
        credible_interval = self._compute_credible_interval(posterior)

        # Bayes factor for MAP vs second best
        sorted_posterior = sorted(posterior.values(), reverse=True)
        bayes_factor = None
        if len(sorted_posterior) >= 2 and sorted_posterior[1] > 0:
            bayes_factor = sorted_posterior[0] / sorted_posterior[1]

        interpretation = self._interpret_inference(
            map_estimate, posterior, bayes_factor
        )

        return BayesianInferenceResult(
            prior=normalized_prior,
            likelihood=likelihood,
            posterior=posterior,
            evidence=evidence,
            map_estimate=map_estimate,
            posterior_mean=posterior_mean,
            posterior_variance=posterior_variance,
            credible_interval=credible_interval,
            bayes_factor=bayes_factor,
            inference_method=self.inference_method,
            interpretation=interpretation
        )

    def update_posterior(
        self,
        current_posterior: Dict[str, float],
        new_likelihood: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Update posterior with new evidence (sequential Bayesian updating).

        Args:
            current_posterior: Current posterior (becomes new prior)
            new_likelihood: Likelihood of new evidence

        Returns:
            Updated posterior distribution
        """
        result = self.analyze(current_posterior, new_likelihood)
        return result.posterior

    def _compute_credible_interval(
        self,
        posterior: Dict[str, float],
        credibility: float = 0.95
    ) -> Tuple[float, float]:
        """Compute credible interval from posterior."""
        sorted_items = sorted(posterior.items(), key=lambda x: x[1], reverse=True)

        cumulative = 0.0
        included = []
        for h, p in sorted_items:
            cumulative += p
            try:
                included.append(float(h))
            except (ValueError, TypeError):
                pass
            if cumulative >= credibility:
                break

        if included:
            return (min(included), max(included))
        return (0.0, 1.0)

    def _interpret_inference(
        self,
        map_estimate: str,
        posterior: Dict[str, float],
        bayes_factor: Optional[float]
    ) -> str:
        """Generate interpretation of Bayesian inference."""
        map_prob = posterior[map_estimate]
        parts = [
            f"MAP estimate is '{map_estimate}' with posterior probability {map_prob:.4f}."
        ]

        if map_prob > 0.9:
            parts.append("Very strong evidence for this hypothesis.")
        elif map_prob > 0.7:
            parts.append("Strong evidence supporting this hypothesis.")
        elif map_prob > 0.5:
            parts.append("Moderate evidence; hypothesis is most likely but not certain.")
        else:
            parts.append("Weak evidence; considerable uncertainty remains.")

        if bayes_factor:
            if bayes_factor > 100:
                parts.append(f"Bayes factor ({bayes_factor:.1f}) indicates decisive evidence.")
            elif bayes_factor > 10:
                parts.append(f"Bayes factor ({bayes_factor:.1f}) indicates strong evidence.")
            elif bayes_factor > 3:
                parts.append(f"Bayes factor ({bayes_factor:.1f}) indicates moderate evidence.")
            else:
                parts.append(f"Bayes factor ({bayes_factor:.1f}) indicates weak evidence.")

        return " ".join(parts)

    def _empty_result(self) -> BayesianInferenceResult:
        """Return empty result for invalid input."""
        return BayesianInferenceResult(
            prior={},
            likelihood={},
            posterior={},
            evidence=0.0,
            map_estimate="",
            posterior_mean=0.0,
            posterior_variance=0.0,
            credible_interval=(0.0, 1.0),
            bayes_factor=None,
            inference_method=self.inference_method,
            interpretation="Insufficient data for Bayesian inference"
        )


class LikelihoodAnalyzer(BaseProbabilityAnalyzer):
    """
    Analyzer for likelihood computations in AI model evaluation.

    Likelihood P(D|θ) measures how probable the observed data is
    given model parameters. Used for:
    - Model comparison
    - Parameter estimation (MLE)
    - Model selection
    """

    def __init__(self):
        pass

    def get_analyzer_type(self) -> str:
        return "likelihood"

    def analyze(
        self,
        predictions: List[float],
        targets: List[float],
        distribution: DistributionType = DistributionType.GAUSSIAN
    ) -> Dict[str, Any]:
        """
        Compute likelihood of observations under predicted distribution.

        Args:
            predictions: Model predictions (means for Gaussian)
            targets: Actual target values
            distribution: Assumed distribution type

        Returns:
            Dictionary with likelihood analysis
        """
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have equal length")

        n = len(predictions)
        if n == 0:
            return {"error": "No data provided"}

        if distribution == DistributionType.GAUSSIAN:
            return self._gaussian_likelihood(predictions, targets)
        elif distribution == DistributionType.BERNOULLI:
            return self._bernoulli_likelihood(predictions, targets)
        else:
            return self._gaussian_likelihood(predictions, targets)

    def _gaussian_likelihood(
        self,
        predictions: List[float],
        targets: List[float]
    ) -> Dict[str, Any]:
        """Compute Gaussian likelihood."""
        n = len(predictions)

        # Estimate variance from residuals
        residuals = [t - p for t, p in zip(targets, predictions)]
        mean_residual = sum(residuals) / n
        variance = sum((r - mean_residual) ** 2 for r in residuals) / (n - 1) if n > 1 else 1.0
        std = math.sqrt(variance)

        # Log-likelihood
        log_likelihood = 0.0
        for pred, target in zip(predictions, targets):
            if std > 0:
                log_prob = -0.5 * math.log(2 * math.pi * variance)
                log_prob -= (target - pred) ** 2 / (2 * variance)
                log_likelihood += log_prob

        # Normalized log-likelihood per sample
        nll_per_sample = -log_likelihood / n if n > 0 else float('inf')

        # AIC and BIC for model comparison
        k = 2  # Number of parameters (mean, variance)
        aic = 2 * k - 2 * log_likelihood
        bic = k * math.log(n) - 2 * log_likelihood if n > 0 else float('inf')

        return {
            "log_likelihood": log_likelihood,
            "nll_per_sample": nll_per_sample,
            "estimated_std": std,
            "estimated_variance": variance,
            "aic": aic,
            "bic": bic,
            "n_samples": n,
            "distribution": "gaussian",
            "interpretation": self._interpret_gaussian_likelihood(nll_per_sample, std)
        }

    def _bernoulli_likelihood(
        self,
        predictions: List[float],
        targets: List[float]
    ) -> Dict[str, Any]:
        """Compute Bernoulli likelihood (for binary classification)."""
        n = len(predictions)

        # Binary cross-entropy
        log_likelihood = 0.0
        eps = 1e-15  # Numerical stability

        for pred, target in zip(predictions, targets):
            pred_clipped = max(eps, min(1 - eps, pred))
            if target == 1:
                log_likelihood += math.log(pred_clipped)
            else:
                log_likelihood += math.log(1 - pred_clipped)

        nll_per_sample = -log_likelihood / n if n > 0 else float('inf')

        return {
            "log_likelihood": log_likelihood,
            "nll_per_sample": nll_per_sample,
            "binary_cross_entropy": nll_per_sample,
            "n_samples": n,
            "distribution": "bernoulli",
            "interpretation": self._interpret_bernoulli_likelihood(nll_per_sample)
        }

    def _interpret_gaussian_likelihood(self, nll: float, std: float) -> str:
        """Interpret Gaussian likelihood results."""
        if nll < 1.0:
            quality = "excellent"
        elif nll < 2.0:
            quality = "good"
        elif nll < 3.0:
            quality = "moderate"
        else:
            quality = "poor"

        return (
            f"Model fit quality is {quality} with NLL/sample = {nll:.4f}. "
            f"Estimated standard deviation = {std:.4f}."
        )

    def _interpret_bernoulli_likelihood(self, bce: float) -> str:
        """Interpret Bernoulli likelihood results."""
        if bce < 0.3:
            quality = "excellent"
        elif bce < 0.5:
            quality = "good"
        elif bce < 0.7:
            quality = "moderate"
        else:
            quality = "poor"

        return (
            f"Classification quality is {quality} with binary cross-entropy = {bce:.4f}."
        )


class PredictiveProbabilityAnalyzer(BaseProbabilityAnalyzer):
    """
    Analyzer for predictive probability computations.

    Predictive probability integrates over model uncertainty:
    P(y*|x*, D) = ∫ P(y*|x*, θ) P(θ|D) dθ

    Used for uncertainty quantification in predictions.
    """

    def __init__(self):
        pass

    def get_analyzer_type(self) -> str:
        return "predictive_probability"

    def analyze(
        self,
        ensemble_predictions: List[List[float]],
        target_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze predictive probabilities from ensemble predictions.

        Args:
            ensemble_predictions: List of prediction vectors from ensemble members
            target_index: Optional index for specific target class

        Returns:
            Dictionary with predictive probability analysis
        """
        if not ensemble_predictions:
            return {"error": "No predictions provided"}

        n_models = len(ensemble_predictions)
        n_classes = len(ensemble_predictions[0])

        # Calculate mean prediction (predictive probability)
        mean_pred = [0.0] * n_classes
        for pred in ensemble_predictions:
            for i, p in enumerate(pred):
                mean_pred[i] += p / n_models

        # Calculate predictive uncertainty (entropy of mean prediction)
        predictive_entropy = 0.0
        for p in mean_pred:
            if p > 0:
                predictive_entropy -= p * math.log(p)

        # Calculate expected entropy (aleatoric uncertainty)
        expected_entropy = 0.0
        for pred in ensemble_predictions:
            for p in pred:
                if p > 0:
                    expected_entropy -= p * math.log(p) / n_models

        # Mutual information (epistemic uncertainty)
        mutual_info = predictive_entropy - expected_entropy

        # Variance of predictions
        variance_pred = [0.0] * n_classes
        for pred in ensemble_predictions:
            for i, p in enumerate(pred):
                variance_pred[i] += (p - mean_pred[i]) ** 2 / n_models

        result = {
            "predictive_mean": mean_pred,
            "predictive_variance": variance_pred,
            "predictive_entropy": predictive_entropy,
            "expected_entropy": expected_entropy,
            "mutual_information": mutual_info,
            "aleatoric_uncertainty": expected_entropy,
            "epistemic_uncertainty": mutual_info,
            "total_uncertainty": predictive_entropy,
            "n_ensemble_members": n_models,
            "n_classes": n_classes,
            "interpretation": self._interpret_uncertainty(
                predictive_entropy, mutual_info, expected_entropy
            )
        }

        if target_index is not None and 0 <= target_index < n_classes:
            result["target_probability"] = mean_pred[target_index]
            result["target_variance"] = variance_pred[target_index]

        return result

    def _interpret_uncertainty(
        self,
        total: float,
        epistemic: float,
        aleatoric: float
    ) -> str:
        """Interpret uncertainty decomposition."""
        parts = []

        if total < 0.5:
            parts.append("Low total uncertainty; model is confident.")
        elif total < 1.0:
            parts.append("Moderate total uncertainty.")
        else:
            parts.append("High total uncertainty; predictions are unreliable.")

        if epistemic > aleatoric:
            parts.append(
                "Epistemic uncertainty dominates; more training data may help."
            )
        else:
            parts.append(
                "Aleatoric uncertainty dominates; inherent data noise present."
            )

        return " ".join(parts)


class ProbabilityCalibrationAnalyzer(BaseProbabilityAnalyzer):
    """
    Analyzer for probability calibration assessment.

    Evaluates whether predicted probabilities match empirical frequencies:
    A well-calibrated model has P(Y=1|P(Y=1)=p) ≈ p

    Metrics include:
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Reliability diagrams
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def get_analyzer_type(self) -> str:
        return "probability_calibration"

    def analyze(
        self,
        predicted_probs: List[float],
        true_labels: List[int]
    ) -> Dict[str, Any]:
        """
        Analyze calibration of predicted probabilities.

        Args:
            predicted_probs: Predicted probabilities for positive class
            true_labels: True binary labels (0 or 1)

        Returns:
            Dictionary with calibration analysis
        """
        if len(predicted_probs) != len(true_labels):
            raise ValueError("Predictions and labels must have equal length")

        n = len(predicted_probs)
        if n == 0:
            return {"error": "No data provided"}

        # Create bins
        bin_boundaries = [i / self.n_bins for i in range(self.n_bins + 1)]
        bin_data = {i: {"predictions": [], "labels": []} for i in range(self.n_bins)}

        # Assign to bins
        for prob, label in zip(predicted_probs, true_labels):
            bin_idx = min(int(prob * self.n_bins), self.n_bins - 1)
            bin_data[bin_idx]["predictions"].append(prob)
            bin_data[bin_idx]["labels"].append(label)

        # Calculate bin statistics
        bin_stats = []
        ece = 0.0
        mce = 0.0

        for i in range(self.n_bins):
            preds = bin_data[i]["predictions"]
            labels = bin_data[i]["labels"]

            if preds:
                avg_confidence = sum(preds) / len(preds)
                avg_accuracy = sum(labels) / len(labels)
                bin_size = len(preds)

                calibration_error = abs(avg_accuracy - avg_confidence)
                ece += (bin_size / n) * calibration_error
                mce = max(mce, calibration_error)

                bin_stats.append({
                    "bin_index": i,
                    "bin_range": (bin_boundaries[i], bin_boundaries[i + 1]),
                    "avg_confidence": avg_confidence,
                    "avg_accuracy": avg_accuracy,
                    "calibration_error": calibration_error,
                    "bin_size": bin_size
                })

        # Brier score
        brier_score = sum(
            (p - l) ** 2 for p, l in zip(predicted_probs, true_labels)
        ) / n

        # Overall accuracy
        binary_preds = [1 if p >= 0.5 else 0 for p in predicted_probs]
        accuracy = sum(
            1 for p, l in zip(binary_preds, true_labels) if p == l
        ) / n

        return {
            "ece": ece,
            "mce": mce,
            "brier_score": brier_score,
            "accuracy": accuracy,
            "n_bins": self.n_bins,
            "n_samples": n,
            "bin_statistics": bin_stats,
            "is_well_calibrated": ece < 0.1,
            "interpretation": self._interpret_calibration(ece, mce, brier_score)
        }

    def _interpret_calibration(
        self,
        ece: float,
        mce: float,
        brier: float
    ) -> str:
        """Interpret calibration results."""
        parts = []

        if ece < 0.05:
            parts.append(f"Excellent calibration (ECE = {ece:.4f}).")
        elif ece < 0.1:
            parts.append(f"Good calibration (ECE = {ece:.4f}).")
        elif ece < 0.2:
            parts.append(f"Moderate calibration (ECE = {ece:.4f}); consider calibration methods.")
        else:
            parts.append(f"Poor calibration (ECE = {ece:.4f}); calibration strongly recommended.")

        if mce > 0.3:
            parts.append(f"High maximum calibration error ({mce:.4f}) in some confidence regions.")

        return " ".join(parts)


# Utility functions
def compute_marginal_probability(
    observations: List[Any],
    smoothing: float = 0.0
) -> Dict[str, ProbabilityResult]:
    """Convenience function for marginal probability computation."""
    analyzer = MarginalProbabilityAnalyzer(smoothing=smoothing)
    return analyzer.analyze(observations)


def compute_joint_probability(
    observations_x: List[Any],
    observations_y: List[Any]
) -> JointProbabilityResult:
    """Convenience function for joint probability computation."""
    analyzer = JointProbabilityAnalyzer()
    return analyzer.analyze(observations_x, observations_y)


def compute_conditional_probability(
    observations_a: List[Any],
    observations_b: List[Any],
    condition_value: Any
) -> Dict[str, ProbabilityResult]:
    """Convenience function for conditional probability computation."""
    analyzer = ConditionalProbabilityAnalyzer()
    return analyzer.analyze(observations_a, observations_b, condition_value)


def perform_bayesian_inference(
    prior: Dict[str, float],
    likelihood: Dict[str, float]
) -> BayesianInferenceResult:
    """Convenience function for Bayesian inference."""
    analyzer = BayesianInferenceAnalyzer()
    return analyzer.analyze(prior, likelihood)


def analyze_calibration(
    predicted_probs: List[float],
    true_labels: List[int],
    n_bins: int = 10
) -> Dict[str, Any]:
    """Convenience function for calibration analysis."""
    analyzer = ProbabilityCalibrationAnalyzer(n_bins=n_bins)
    return analyzer.analyze(predicted_probs, true_labels)


def compute_predictive_uncertainty(
    ensemble_predictions: List[List[float]]
) -> Dict[str, Any]:
    """Convenience function for predictive uncertainty analysis."""
    analyzer = PredictiveProbabilityAnalyzer()
    return analyzer.analyze(ensemble_predictions)
