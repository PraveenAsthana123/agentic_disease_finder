"""
Validation Techniques Analysis Module
=====================================

Comprehensive framework for model validation and verification including:
1. Statistical Validation - Cross-validation, bootstrapping, hypothesis testing
2. Performance Validation - Metrics analysis, threshold validation
3. Fairness Validation - Bias detection, demographic parity testing
4. Robustness Validation - Adversarial testing, sensitivity analysis
5. Calibration Validation - Reliability diagrams, expected calibration error

This module provides analyzers for validating AI/ML models across
multiple dimensions to ensure quality and responsible deployment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from abc import ABC, abstractmethod
import math
import statistics
from collections import defaultdict


# =============================================================================
# Enums
# =============================================================================

class ValidationMethod(Enum):
    """Validation methodology types."""
    HOLDOUT = "holdout"
    K_FOLD = "k_fold"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    LEAVE_ONE_OUT = "leave_one_out"
    BOOTSTRAP = "bootstrap"
    TIME_SERIES_SPLIT = "time_series_split"
    GROUP_K_FOLD = "group_k_fold"
    NESTED_CV = "nested_cv"


class ValidationStatus(Enum):
    """Validation outcome status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NOT_RUN = "not_run"
    INCONCLUSIVE = "inconclusive"


class MetricType(Enum):
    """Types of validation metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    AUC_PR = "auc_pr"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R_SQUARED = "r_squared"
    LOG_LOSS = "log_loss"
    BRIER_SCORE = "brier_score"


class FairnessMetric(Enum):
    """Fairness validation metrics."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    PREDICTIVE_PARITY = "predictive_parity"
    CALIBRATION_PARITY = "calibration_parity"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    COUNTERFACTUAL_FAIRNESS = "counterfactual_fairness"


class RobustnessTest(Enum):
    """Robustness testing types."""
    NOISE_INJECTION = "noise_injection"
    ADVERSARIAL_PERTURBATION = "adversarial_perturbation"
    FEATURE_ABLATION = "feature_ablation"
    DISTRIBUTION_SHIFT = "distribution_shift"
    EDGE_CASE = "edge_case"
    BOUNDARY_TESTING = "boundary_testing"
    STRESS_TESTING = "stress_testing"


class CalibrationMethod(Enum):
    """Calibration assessment methods."""
    RELIABILITY_DIAGRAM = "reliability_diagram"
    ECE = "expected_calibration_error"
    MCE = "maximum_calibration_error"
    BRIER_DECOMPOSITION = "brier_decomposition"
    HOSMER_LEMESHOW = "hosmer_lemeshow"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ValidationResult:
    """General validation result."""
    validation_type: str
    status: ValidationStatus
    score: float
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class CrossValidationResult:
    """Cross-validation results."""
    method: ValidationMethod
    n_folds: int
    scores: List[float]
    mean_score: float
    std_score: float
    confidence_interval: Tuple[float, float]
    fold_details: List[Dict[str, Any]] = field(default_factory=list)
    overfitting_detected: bool = False
    variance_analysis: Dict[str, float] = field(default_factory=dict)


@dataclass
class MetricValidationResult:
    """Performance metric validation result."""
    metric: MetricType
    value: float
    threshold: float
    status: ValidationStatus
    baseline_comparison: Optional[float] = None
    trend: Optional[str] = None
    statistical_significance: Optional[float] = None


@dataclass
class FairnessValidationResult:
    """Fairness validation result."""
    metric: FairnessMetric
    protected_attribute: str
    group_metrics: Dict[str, float]
    disparity_ratio: float
    threshold: float
    status: ValidationStatus
    affected_groups: List[str] = field(default_factory=list)
    remediation_suggestions: List[str] = field(default_factory=list)


@dataclass
class RobustnessValidationResult:
    """Robustness validation result."""
    test_type: RobustnessTest
    original_performance: float
    perturbed_performance: float
    degradation: float
    tolerance_threshold: float
    status: ValidationStatus
    vulnerable_features: List[str] = field(default_factory=list)
    attack_success_rate: Optional[float] = None


@dataclass
class CalibrationResult:
    """Model calibration assessment result."""
    method: CalibrationMethod
    calibration_error: float
    threshold: float
    status: ValidationStatus
    bin_statistics: List[Dict[str, float]] = field(default_factory=list)
    reliability_data: Dict[str, List[float]] = field(default_factory=dict)
    recalibration_needed: bool = False


@dataclass
class ComprehensiveValidationReport:
    """Comprehensive validation report."""
    statistical_validation: CrossValidationResult
    metric_validations: List[MetricValidationResult]
    fairness_validations: List[FairnessValidationResult]
    robustness_validations: List[RobustnessValidationResult]
    calibration_result: CalibrationResult
    overall_status: ValidationStatus
    readiness_score: float
    critical_findings: List[str]
    deployment_recommendations: List[str]


# =============================================================================
# Statistical Validation Analyzer
# =============================================================================

class StatisticalValidationAnalyzer:
    """Analyzer for statistical validation techniques."""

    def __init__(self):
        self.confidence_level = 0.95

    def perform_cross_validation(
        self,
        scores: List[float],
        method: ValidationMethod = ValidationMethod.K_FOLD,
        n_folds: int = 5,
        train_scores: Optional[List[float]] = None
    ) -> CrossValidationResult:
        """Analyze cross-validation results."""
        if not scores:
            return CrossValidationResult(
                method=method,
                n_folds=n_folds,
                scores=[],
                mean_score=0.0,
                std_score=0.0,
                confidence_interval=(0.0, 0.0),
                overfitting_detected=False
            )

        mean_score = statistics.mean(scores)
        std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0

        # Calculate confidence interval
        ci = self._calculate_confidence_interval(scores)

        # Detect overfitting
        overfitting_detected = False
        if train_scores:
            train_mean = statistics.mean(train_scores)
            overfitting_detected = (train_mean - mean_score) > 0.1

        # Variance analysis
        variance_analysis = self._analyze_variance(scores)

        # Build fold details
        fold_details = [
            {'fold': i + 1, 'score': score}
            for i, score in enumerate(scores)
        ]

        return CrossValidationResult(
            method=method,
            n_folds=n_folds,
            scores=scores,
            mean_score=mean_score,
            std_score=std_score,
            confidence_interval=ci,
            fold_details=fold_details,
            overfitting_detected=overfitting_detected,
            variance_analysis=variance_analysis
        )

    def perform_bootstrap_validation(
        self,
        scores: List[float],
        n_iterations: int = 1000
    ) -> Dict[str, Any]:
        """Perform bootstrap validation analysis."""
        if not scores:
            return {'error': 'No scores provided'}

        # Simulate bootstrap iterations
        bootstrap_means = []
        n = len(scores)

        for _ in range(n_iterations):
            # Resample with replacement
            sample = [scores[int(len(scores) * (i / n_iterations)) % len(scores)]
                      for i in range(n)]
            bootstrap_means.append(statistics.mean(sample))

        return {
            'original_mean': statistics.mean(scores),
            'bootstrap_mean': statistics.mean(bootstrap_means),
            'bootstrap_std': statistics.stdev(bootstrap_means) if len(bootstrap_means) > 1 else 0,
            'percentile_2.5': sorted(bootstrap_means)[int(0.025 * len(bootstrap_means))],
            'percentile_97.5': sorted(bootstrap_means)[int(0.975 * len(bootstrap_means))],
            'bias': statistics.mean(bootstrap_means) - statistics.mean(scores),
            'n_iterations': n_iterations
        }

    def perform_hypothesis_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        alternative: str = 'two-sided'
    ) -> Dict[str, Any]:
        """Perform hypothesis test between two score sets."""
        if not scores_a or not scores_b:
            return {'error': 'Insufficient data'}

        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)

        std_a = statistics.stdev(scores_a) if len(scores_a) > 1 else 0.01
        std_b = statistics.stdev(scores_b) if len(scores_b) > 1 else 0.01

        # Calculate pooled standard error
        n_a = len(scores_a)
        n_b = len(scores_b)
        se = math.sqrt(std_a**2 / n_a + std_b**2 / n_b)

        # T-statistic
        t_stat = (mean_a - mean_b) / max(se, 0.0001)

        # Simplified p-value estimation
        # In production, use scipy.stats.t.sf
        p_value = self._estimate_p_value(abs(t_stat), n_a + n_b - 2)

        significant = p_value < 0.05

        return {
            'mean_a': mean_a,
            'mean_b': mean_b,
            'difference': mean_a - mean_b,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': significant,
            'alternative': alternative,
            'effect_size': (mean_a - mean_b) / max(std_a, std_b, 0.0001)
        }

    def _calculate_confidence_interval(
        self,
        scores: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval."""
        n = len(scores)
        if n < 2:
            mean = scores[0] if scores else 0
            return (mean, mean)

        mean = statistics.mean(scores)
        std = statistics.stdev(scores)

        # Z-value for 95% CI
        z = 1.96

        margin = z * std / math.sqrt(n)

        return (mean - margin, mean + margin)

    def _analyze_variance(self, scores: List[float]) -> Dict[str, float]:
        """Analyze score variance."""
        if len(scores) < 2:
            return {'variance': 0, 'cv': 0, 'range': 0}

        mean = statistics.mean(scores)
        variance = statistics.variance(scores)
        std = statistics.stdev(scores)

        return {
            'variance': variance,
            'std': std,
            'cv': std / mean if mean != 0 else 0,
            'range': max(scores) - min(scores),
            'iqr': self._calculate_iqr(scores)
        }

    def _calculate_iqr(self, scores: List[float]) -> float:
        """Calculate interquartile range."""
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        return sorted_scores[q3_idx] - sorted_scores[q1_idx]

    def _estimate_p_value(self, t_stat: float, df: int) -> float:
        """Estimate p-value from t-statistic (simplified)."""
        # Simplified approximation
        # In production, use scipy.stats.t.sf
        z = abs(t_stat)
        if z > 3.5:
            return 0.001
        elif z > 2.5:
            return 0.01
        elif z > 2.0:
            return 0.05
        elif z > 1.5:
            return 0.1
        return 0.3


# =============================================================================
# Performance Validation Analyzer
# =============================================================================

class PerformanceValidationAnalyzer:
    """Analyzer for performance metric validation."""

    def __init__(self):
        self.default_thresholds = {
            MetricType.ACCURACY: 0.8,
            MetricType.PRECISION: 0.75,
            MetricType.RECALL: 0.75,
            MetricType.F1_SCORE: 0.75,
            MetricType.AUC_ROC: 0.8,
            MetricType.AUC_PR: 0.7,
            MetricType.MAE: 0.1,
            MetricType.MSE: 0.05,
            MetricType.RMSE: 0.15,
            MetricType.R_SQUARED: 0.7,
            MetricType.LOG_LOSS: 0.5,
            MetricType.BRIER_SCORE: 0.25,
        }

    def validate_metric(
        self,
        metric: MetricType,
        value: float,
        threshold: Optional[float] = None,
        baseline: Optional[float] = None
    ) -> MetricValidationResult:
        """Validate a single performance metric."""
        threshold = threshold or self.default_thresholds.get(metric, 0.7)

        # Determine if metric is "higher is better" or "lower is better"
        lower_is_better = metric in {
            MetricType.MAE, MetricType.MSE, MetricType.RMSE,
            MetricType.LOG_LOSS, MetricType.BRIER_SCORE
        }

        if lower_is_better:
            status = ValidationStatus.PASSED if value <= threshold else ValidationStatus.FAILED
        else:
            status = ValidationStatus.PASSED if value >= threshold else ValidationStatus.FAILED

        # Baseline comparison
        baseline_comparison = None
        trend = None
        if baseline is not None:
            baseline_comparison = value - baseline
            if lower_is_better:
                trend = 'improved' if value < baseline else 'degraded'
            else:
                trend = 'improved' if value > baseline else 'degraded'

        return MetricValidationResult(
            metric=metric,
            value=value,
            threshold=threshold,
            status=status,
            baseline_comparison=baseline_comparison,
            trend=trend
        )

    def validate_all_metrics(
        self,
        metrics: Dict[MetricType, float],
        thresholds: Optional[Dict[MetricType, float]] = None,
        baselines: Optional[Dict[MetricType, float]] = None
    ) -> List[MetricValidationResult]:
        """Validate all provided metrics."""
        thresholds = thresholds or {}
        baselines = baselines or {}

        results = []
        for metric, value in metrics.items():
            result = self.validate_metric(
                metric,
                value,
                thresholds.get(metric),
                baselines.get(metric)
            )
            results.append(result)

        return results

    def calculate_composite_score(
        self,
        results: List[MetricValidationResult],
        weights: Optional[Dict[MetricType, float]] = None
    ) -> float:
        """Calculate composite validation score."""
        if not results:
            return 0.0

        weights = weights or {}

        total_weight = 0
        weighted_score = 0

        for result in results:
            weight = weights.get(result.metric, 1.0)
            total_weight += weight

            # Normalize score
            lower_is_better = result.metric in {
                MetricType.MAE, MetricType.MSE, MetricType.RMSE,
                MetricType.LOG_LOSS, MetricType.BRIER_SCORE
            }

            if lower_is_better:
                normalized = max(0, 1 - result.value / max(result.threshold, 0.0001))
            else:
                normalized = min(1, result.value / max(result.threshold, 0.0001))

            weighted_score += weight * normalized

        return weighted_score / max(total_weight, 1)


# =============================================================================
# Fairness Validation Analyzer
# =============================================================================

class FairnessValidationAnalyzer:
    """Analyzer for fairness validation."""

    def __init__(self):
        self.disparity_threshold = 0.8  # 80% rule

    def validate_demographic_parity(
        self,
        group_positive_rates: Dict[str, float],
        protected_attribute: str,
        threshold: Optional[float] = None
    ) -> FairnessValidationResult:
        """Validate demographic parity."""
        threshold = threshold or self.disparity_threshold

        if not group_positive_rates or len(group_positive_rates) < 2:
            return FairnessValidationResult(
                metric=FairnessMetric.DEMOGRAPHIC_PARITY,
                protected_attribute=protected_attribute,
                group_metrics=group_positive_rates,
                disparity_ratio=1.0,
                threshold=threshold,
                status=ValidationStatus.INCONCLUSIVE
            )

        rates = list(group_positive_rates.values())
        max_rate = max(rates)
        min_rate = min(rates)

        disparity_ratio = min_rate / max_rate if max_rate > 0 else 0

        status = ValidationStatus.PASSED if disparity_ratio >= threshold else ValidationStatus.FAILED

        affected_groups = [
            group for group, rate in group_positive_rates.items()
            if max_rate > 0 and rate / max_rate < threshold
        ]

        remediation = []
        if status == ValidationStatus.FAILED:
            remediation.append(f'Consider resampling or reweighting for {", ".join(affected_groups)}')
            remediation.append('Review feature selection for discriminatory proxies')

        return FairnessValidationResult(
            metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            protected_attribute=protected_attribute,
            group_metrics=group_positive_rates,
            disparity_ratio=disparity_ratio,
            threshold=threshold,
            status=status,
            affected_groups=affected_groups,
            remediation_suggestions=remediation
        )

    def validate_equalized_odds(
        self,
        group_tpr: Dict[str, float],
        group_fpr: Dict[str, float],
        protected_attribute: str,
        threshold: Optional[float] = None
    ) -> FairnessValidationResult:
        """Validate equalized odds."""
        threshold = threshold or self.disparity_threshold

        # Check both TPR and FPR parity
        tpr_values = list(group_tpr.values())
        fpr_values = list(group_fpr.values())

        tpr_disparity = min(tpr_values) / max(tpr_values) if max(tpr_values) > 0 else 0
        fpr_disparity = min(fpr_values) / max(fpr_values) if max(fpr_values) > 0 else 0

        # Both need to pass
        combined_disparity = min(tpr_disparity, fpr_disparity)
        status = ValidationStatus.PASSED if combined_disparity >= threshold else ValidationStatus.FAILED

        group_metrics = {
            f'{g}_tpr': v for g, v in group_tpr.items()
        }
        group_metrics.update({
            f'{g}_fpr': v for g, v in group_fpr.items()
        })

        affected_groups = []
        for group in group_tpr.keys():
            if (group_tpr[group] / max(tpr_values) < threshold if max(tpr_values) > 0 else False):
                affected_groups.append(f'{group} (TPR)')
            if (group_fpr[group] / max(fpr_values) < threshold if max(fpr_values) > 0 else False):
                affected_groups.append(f'{group} (FPR)')

        return FairnessValidationResult(
            metric=FairnessMetric.EQUALIZED_ODDS,
            protected_attribute=protected_attribute,
            group_metrics=group_metrics,
            disparity_ratio=combined_disparity,
            threshold=threshold,
            status=status,
            affected_groups=affected_groups,
            remediation_suggestions=['Consider post-processing calibration'] if status == ValidationStatus.FAILED else []
        )

    def validate_equal_opportunity(
        self,
        group_tpr: Dict[str, float],
        protected_attribute: str,
        threshold: Optional[float] = None
    ) -> FairnessValidationResult:
        """Validate equal opportunity (TPR parity)."""
        threshold = threshold or self.disparity_threshold

        tpr_values = list(group_tpr.values())
        disparity_ratio = min(tpr_values) / max(tpr_values) if max(tpr_values) > 0 else 0

        status = ValidationStatus.PASSED if disparity_ratio >= threshold else ValidationStatus.FAILED

        affected_groups = [
            group for group, tpr in group_tpr.items()
            if max(tpr_values) > 0 and tpr / max(tpr_values) < threshold
        ]

        return FairnessValidationResult(
            metric=FairnessMetric.EQUAL_OPPORTUNITY,
            protected_attribute=protected_attribute,
            group_metrics=group_tpr,
            disparity_ratio=disparity_ratio,
            threshold=threshold,
            status=status,
            affected_groups=affected_groups,
            remediation_suggestions=['Adjust decision threshold per group'] if status == ValidationStatus.FAILED else []
        )

    def validate_predictive_parity(
        self,
        group_ppv: Dict[str, float],
        protected_attribute: str,
        threshold: Optional[float] = None
    ) -> FairnessValidationResult:
        """Validate predictive parity (PPV equality)."""
        threshold = threshold or self.disparity_threshold

        ppv_values = list(group_ppv.values())
        disparity_ratio = min(ppv_values) / max(ppv_values) if max(ppv_values) > 0 else 0

        status = ValidationStatus.PASSED if disparity_ratio >= threshold else ValidationStatus.FAILED

        affected_groups = [
            group for group, ppv in group_ppv.items()
            if max(ppv_values) > 0 and ppv / max(ppv_values) < threshold
        ]

        return FairnessValidationResult(
            metric=FairnessMetric.PREDICTIVE_PARITY,
            protected_attribute=protected_attribute,
            group_metrics=group_ppv,
            disparity_ratio=disparity_ratio,
            threshold=threshold,
            status=status,
            affected_groups=affected_groups
        )

    def comprehensive_fairness_audit(
        self,
        predictions: List[int],
        labels: List[int],
        protected_attributes: Dict[str, List[Any]],
        thresholds: Optional[Dict[FairnessMetric, float]] = None
    ) -> List[FairnessValidationResult]:
        """Perform comprehensive fairness audit."""
        results = []
        thresholds = thresholds or {}

        for attr_name, attr_values in protected_attributes.items():
            # Group predictions and labels by attribute
            groups = defaultdict(lambda: {'pred': [], 'label': []})

            for pred, label, attr in zip(predictions, labels, attr_values):
                groups[attr]['pred'].append(pred)
                groups[attr]['label'].append(label)

            # Calculate group metrics
            group_positive_rates = {}
            group_tpr = {}
            group_fpr = {}
            group_ppv = {}

            for group, data in groups.items():
                preds = data['pred']
                labels_g = data['label']

                # Positive rate (demographic parity)
                group_positive_rates[str(group)] = sum(preds) / len(preds) if preds else 0

                # TPR, FPR, PPV
                tp = sum(1 for p, l in zip(preds, labels_g) if p == 1 and l == 1)
                fp = sum(1 for p, l in zip(preds, labels_g) if p == 1 and l == 0)
                tn = sum(1 for p, l in zip(preds, labels_g) if p == 0 and l == 0)
                fn = sum(1 for p, l in zip(preds, labels_g) if p == 0 and l == 1)

                group_tpr[str(group)] = tp / (tp + fn) if (tp + fn) > 0 else 0
                group_fpr[str(group)] = fp / (fp + tn) if (fp + tn) > 0 else 0
                group_ppv[str(group)] = tp / (tp + fp) if (tp + fp) > 0 else 0

            # Run fairness validations
            results.append(self.validate_demographic_parity(
                group_positive_rates, attr_name,
                thresholds.get(FairnessMetric.DEMOGRAPHIC_PARITY)
            ))

            results.append(self.validate_equalized_odds(
                group_tpr, group_fpr, attr_name,
                thresholds.get(FairnessMetric.EQUALIZED_ODDS)
            ))

            results.append(self.validate_equal_opportunity(
                group_tpr, attr_name,
                thresholds.get(FairnessMetric.EQUAL_OPPORTUNITY)
            ))

            results.append(self.validate_predictive_parity(
                group_ppv, attr_name,
                thresholds.get(FairnessMetric.PREDICTIVE_PARITY)
            ))

        return results


# =============================================================================
# Robustness Validation Analyzer
# =============================================================================

class RobustnessValidationAnalyzer:
    """Analyzer for robustness validation."""

    def __init__(self):
        self.degradation_threshold = 0.1  # 10% max degradation

    def validate_noise_robustness(
        self,
        original_performance: float,
        noisy_performances: Dict[str, float],
        tolerance: Optional[float] = None
    ) -> RobustnessValidationResult:
        """Validate robustness to noise injection."""
        tolerance = tolerance or self.degradation_threshold

        avg_noisy = statistics.mean(noisy_performances.values()) if noisy_performances else original_performance
        degradation = (original_performance - avg_noisy) / max(original_performance, 0.0001)

        status = ValidationStatus.PASSED if degradation <= tolerance else ValidationStatus.FAILED

        vulnerable_features = [
            noise_type for noise_type, perf in noisy_performances.items()
            if (original_performance - perf) / max(original_performance, 0.0001) > tolerance
        ]

        return RobustnessValidationResult(
            test_type=RobustnessTest.NOISE_INJECTION,
            original_performance=original_performance,
            perturbed_performance=avg_noisy,
            degradation=degradation,
            tolerance_threshold=tolerance,
            status=status,
            vulnerable_features=vulnerable_features
        )

    def validate_adversarial_robustness(
        self,
        original_performance: float,
        adversarial_performance: float,
        attack_success_rate: float,
        tolerance: Optional[float] = None
    ) -> RobustnessValidationResult:
        """Validate robustness to adversarial attacks."""
        tolerance = tolerance or self.degradation_threshold

        degradation = (original_performance - adversarial_performance) / max(original_performance, 0.0001)

        # Stricter for adversarial: both degradation and attack success matter
        effective_tolerance = tolerance * (1 - attack_success_rate)
        status = ValidationStatus.PASSED if degradation <= effective_tolerance else ValidationStatus.FAILED

        return RobustnessValidationResult(
            test_type=RobustnessTest.ADVERSARIAL_PERTURBATION,
            original_performance=original_performance,
            perturbed_performance=adversarial_performance,
            degradation=degradation,
            tolerance_threshold=tolerance,
            status=status,
            attack_success_rate=attack_success_rate
        )

    def validate_feature_ablation(
        self,
        original_performance: float,
        ablation_results: Dict[str, float],
        tolerance: Optional[float] = None
    ) -> RobustnessValidationResult:
        """Validate robustness to feature ablation."""
        tolerance = tolerance or self.degradation_threshold * 2  # More lenient for ablation

        avg_ablated = statistics.mean(ablation_results.values()) if ablation_results else original_performance
        degradation = (original_performance - avg_ablated) / max(original_performance, 0.0001)

        status = ValidationStatus.PASSED if degradation <= tolerance else ValidationStatus.FAILED

        # Features that cause significant degradation when removed
        critical_features = [
            feature for feature, perf in ablation_results.items()
            if (original_performance - perf) / max(original_performance, 0.0001) > tolerance
        ]

        return RobustnessValidationResult(
            test_type=RobustnessTest.FEATURE_ABLATION,
            original_performance=original_performance,
            perturbed_performance=avg_ablated,
            degradation=degradation,
            tolerance_threshold=tolerance,
            status=status,
            vulnerable_features=critical_features
        )

    def validate_distribution_shift(
        self,
        original_performance: float,
        shifted_performance: float,
        shift_magnitude: float,
        tolerance: Optional[float] = None
    ) -> RobustnessValidationResult:
        """Validate robustness to distribution shift."""
        tolerance = tolerance or self.degradation_threshold

        degradation = (original_performance - shifted_performance) / max(original_performance, 0.0001)

        # Adjust tolerance based on shift magnitude
        adjusted_tolerance = tolerance * (1 + shift_magnitude)
        status = ValidationStatus.PASSED if degradation <= adjusted_tolerance else ValidationStatus.FAILED

        return RobustnessValidationResult(
            test_type=RobustnessTest.DISTRIBUTION_SHIFT,
            original_performance=original_performance,
            perturbed_performance=shifted_performance,
            degradation=degradation,
            tolerance_threshold=adjusted_tolerance,
            status=status
        )

    def comprehensive_robustness_test(
        self,
        original_performance: float,
        test_results: Dict[RobustnessTest, Dict[str, Any]]
    ) -> List[RobustnessValidationResult]:
        """Perform comprehensive robustness testing."""
        results = []

        for test_type, params in test_results.items():
            if test_type == RobustnessTest.NOISE_INJECTION:
                results.append(self.validate_noise_robustness(
                    original_performance,
                    params.get('performances', {}),
                    params.get('tolerance')
                ))

            elif test_type == RobustnessTest.ADVERSARIAL_PERTURBATION:
                results.append(self.validate_adversarial_robustness(
                    original_performance,
                    params.get('adversarial_performance', original_performance),
                    params.get('attack_success_rate', 0),
                    params.get('tolerance')
                ))

            elif test_type == RobustnessTest.FEATURE_ABLATION:
                results.append(self.validate_feature_ablation(
                    original_performance,
                    params.get('ablation_results', {}),
                    params.get('tolerance')
                ))

            elif test_type == RobustnessTest.DISTRIBUTION_SHIFT:
                results.append(self.validate_distribution_shift(
                    original_performance,
                    params.get('shifted_performance', original_performance),
                    params.get('shift_magnitude', 0.1),
                    params.get('tolerance')
                ))

        return results


# =============================================================================
# Calibration Validation Analyzer
# =============================================================================

class CalibrationValidationAnalyzer:
    """Analyzer for model calibration validation."""

    def __init__(self):
        self.ece_threshold = 0.1  # Expected Calibration Error threshold
        self.n_bins = 10

    def calculate_ece(
        self,
        predicted_probabilities: List[float],
        true_labels: List[int],
        n_bins: Optional[int] = None
    ) -> CalibrationResult:
        """Calculate Expected Calibration Error."""
        n_bins = n_bins or self.n_bins

        if not predicted_probabilities or not true_labels:
            return CalibrationResult(
                method=CalibrationMethod.ECE,
                calibration_error=0.0,
                threshold=self.ece_threshold,
                status=ValidationStatus.INCONCLUSIVE,
                recalibration_needed=False
            )

        # Create bins
        bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
        bin_statistics = []

        total_samples = len(predicted_probabilities)
        ece = 0.0

        reliability_predicted = []
        reliability_actual = []

        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]

            # Get samples in this bin
            bin_indices = [
                j for j, p in enumerate(predicted_probabilities)
                if lower <= p < upper or (i == n_bins - 1 and p == upper)
            ]

            if bin_indices:
                bin_probs = [predicted_probabilities[j] for j in bin_indices]
                bin_labels = [true_labels[j] for j in bin_indices]

                avg_confidence = statistics.mean(bin_probs)
                avg_accuracy = statistics.mean(bin_labels)

                bin_size = len(bin_indices)
                bin_weight = bin_size / total_samples

                ece += bin_weight * abs(avg_confidence - avg_accuracy)

                bin_statistics.append({
                    'bin': i + 1,
                    'lower': lower,
                    'upper': upper,
                    'count': bin_size,
                    'avg_confidence': avg_confidence,
                    'avg_accuracy': avg_accuracy,
                    'gap': abs(avg_confidence - avg_accuracy)
                })

                reliability_predicted.append(avg_confidence)
                reliability_actual.append(avg_accuracy)

        status = ValidationStatus.PASSED if ece <= self.ece_threshold else ValidationStatus.FAILED
        recalibration_needed = ece > self.ece_threshold

        return CalibrationResult(
            method=CalibrationMethod.ECE,
            calibration_error=ece,
            threshold=self.ece_threshold,
            status=status,
            bin_statistics=bin_statistics,
            reliability_data={
                'predicted': reliability_predicted,
                'actual': reliability_actual
            },
            recalibration_needed=recalibration_needed
        )

    def calculate_mce(
        self,
        predicted_probabilities: List[float],
        true_labels: List[int],
        n_bins: Optional[int] = None
    ) -> CalibrationResult:
        """Calculate Maximum Calibration Error."""
        n_bins = n_bins or self.n_bins

        if not predicted_probabilities or not true_labels:
            return CalibrationResult(
                method=CalibrationMethod.MCE,
                calibration_error=0.0,
                threshold=self.ece_threshold * 2,
                status=ValidationStatus.INCONCLUSIVE
            )

        # Create bins
        bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
        max_gap = 0.0
        bin_statistics = []

        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]

            bin_indices = [
                j for j, p in enumerate(predicted_probabilities)
                if lower <= p < upper or (i == n_bins - 1 and p == upper)
            ]

            if bin_indices:
                bin_probs = [predicted_probabilities[j] for j in bin_indices]
                bin_labels = [true_labels[j] for j in bin_indices]

                avg_confidence = statistics.mean(bin_probs)
                avg_accuracy = statistics.mean(bin_labels)
                gap = abs(avg_confidence - avg_accuracy)

                max_gap = max(max_gap, gap)

                bin_statistics.append({
                    'bin': i + 1,
                    'gap': gap
                })

        threshold = self.ece_threshold * 2  # MCE threshold typically higher
        status = ValidationStatus.PASSED if max_gap <= threshold else ValidationStatus.FAILED

        return CalibrationResult(
            method=CalibrationMethod.MCE,
            calibration_error=max_gap,
            threshold=threshold,
            status=status,
            bin_statistics=bin_statistics,
            recalibration_needed=max_gap > threshold
        )

    def calculate_brier_decomposition(
        self,
        predicted_probabilities: List[float],
        true_labels: List[int]
    ) -> Dict[str, float]:
        """Calculate Brier score decomposition."""
        if not predicted_probabilities or not true_labels:
            return {'reliability': 0, 'resolution': 0, 'uncertainty': 0, 'brier': 0}

        n = len(predicted_probabilities)
        base_rate = statistics.mean(true_labels)

        # Brier score
        brier = sum(
            (p - l) ** 2
            for p, l in zip(predicted_probabilities, true_labels)
        ) / n

        # Uncertainty (entropy of base rate)
        uncertainty = base_rate * (1 - base_rate)

        # For decomposition, use binned estimates
        # Simplified: use ECE as proxy for reliability component
        ece_result = self.calculate_ece(predicted_probabilities, true_labels)
        reliability = ece_result.calibration_error ** 2

        # Resolution
        resolution = uncertainty - brier + reliability

        return {
            'reliability': reliability,
            'resolution': max(0, resolution),
            'uncertainty': uncertainty,
            'brier': brier
        }


# =============================================================================
# Comprehensive Validation Analyzer
# =============================================================================

class ComprehensiveValidationAnalyzer:
    """Comprehensive model validation analyzer."""

    def __init__(self):
        self.statistical_analyzer = StatisticalValidationAnalyzer()
        self.performance_analyzer = PerformanceValidationAnalyzer()
        self.fairness_analyzer = FairnessValidationAnalyzer()
        self.robustness_analyzer = RobustnessValidationAnalyzer()
        self.calibration_analyzer = CalibrationValidationAnalyzer()

    def perform_comprehensive_validation(
        self,
        cv_scores: List[float],
        metrics: Dict[MetricType, float],
        predictions: List[int],
        labels: List[int],
        probabilities: List[float],
        protected_attributes: Dict[str, List[Any]],
        robustness_tests: Dict[RobustnessTest, Dict[str, Any]],
        config: Optional[Dict] = None
    ) -> ComprehensiveValidationReport:
        """Perform comprehensive model validation."""
        config = config or {}

        # Statistical validation
        statistical_result = self.statistical_analyzer.perform_cross_validation(
            cv_scores,
            config.get('cv_method', ValidationMethod.K_FOLD),
            config.get('n_folds', 5)
        )

        # Metric validation
        metric_results = self.performance_analyzer.validate_all_metrics(
            metrics,
            config.get('metric_thresholds'),
            config.get('metric_baselines')
        )

        # Fairness validation
        fairness_results = self.fairness_analyzer.comprehensive_fairness_audit(
            predictions, labels, protected_attributes,
            config.get('fairness_thresholds')
        )

        # Robustness validation
        original_perf = metrics.get(MetricType.ACCURACY, 0.8)
        robustness_results = self.robustness_analyzer.comprehensive_robustness_test(
            original_perf, robustness_tests
        )

        # Calibration validation
        calibration_result = self.calibration_analyzer.calculate_ece(
            probabilities, labels
        )

        # Determine overall status
        all_statuses = (
            [r.status for r in metric_results] +
            [r.status for r in fairness_results] +
            [r.status for r in robustness_results] +
            [calibration_result.status]
        )

        failed_count = sum(1 for s in all_statuses if s == ValidationStatus.FAILED)

        if failed_count == 0:
            overall_status = ValidationStatus.PASSED
        elif failed_count <= 2:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.FAILED

        # Calculate readiness score
        passed_count = sum(1 for s in all_statuses if s == ValidationStatus.PASSED)
        readiness_score = passed_count / max(len(all_statuses), 1)

        # Collect critical findings
        critical_findings = []

        for r in metric_results:
            if r.status == ValidationStatus.FAILED:
                critical_findings.append(f'{r.metric.value} below threshold: {r.value:.3f} < {r.threshold:.3f}')

        for r in fairness_results:
            if r.status == ValidationStatus.FAILED:
                critical_findings.append(f'{r.metric.value} violation for {r.protected_attribute}')

        for r in robustness_results:
            if r.status == ValidationStatus.FAILED:
                critical_findings.append(f'{r.test_type.value} failed: {r.degradation:.1%} degradation')

        if calibration_result.status == ValidationStatus.FAILED:
            critical_findings.append(f'Poor calibration: ECE = {calibration_result.calibration_error:.3f}')

        # Deployment recommendations
        recommendations = self._generate_deployment_recommendations(
            overall_status, critical_findings, readiness_score
        )

        return ComprehensiveValidationReport(
            statistical_validation=statistical_result,
            metric_validations=metric_results,
            fairness_validations=fairness_results,
            robustness_validations=robustness_results,
            calibration_result=calibration_result,
            overall_status=overall_status,
            readiness_score=readiness_score,
            critical_findings=critical_findings,
            deployment_recommendations=recommendations
        )

    def _generate_deployment_recommendations(
        self,
        status: ValidationStatus,
        findings: List[str],
        readiness: float
    ) -> List[str]:
        """Generate deployment recommendations."""
        recommendations = []

        if status == ValidationStatus.PASSED:
            recommendations.append('Model is ready for deployment')
            recommendations.append('Continue monitoring in production')
        elif status == ValidationStatus.WARNING:
            recommendations.append('Address warnings before full deployment')
            recommendations.append('Consider staged rollout')
            recommendations.append('Implement enhanced monitoring')
        else:
            recommendations.append('Do not deploy until critical issues are resolved')
            recommendations.append('Review and address all failed validations')
            recommendations.append('Consider model retraining or architectural changes')

        if readiness < 0.8:
            recommendations.append(f'Readiness score ({readiness:.1%}) below recommended threshold (80%)')

        return recommendations


# =============================================================================
# Convenience Functions
# =============================================================================

def perform_cross_validation(
    scores: List[float],
    method: ValidationMethod = ValidationMethod.K_FOLD,
    n_folds: int = 5
) -> CrossValidationResult:
    """Convenience function for cross-validation."""
    analyzer = StatisticalValidationAnalyzer()
    return analyzer.perform_cross_validation(scores, method, n_folds)


def validate_metrics(
    metrics: Dict[MetricType, float],
    thresholds: Optional[Dict[MetricType, float]] = None
) -> List[MetricValidationResult]:
    """Convenience function for metric validation."""
    analyzer = PerformanceValidationAnalyzer()
    return analyzer.validate_all_metrics(metrics, thresholds)


def validate_fairness(
    predictions: List[int],
    labels: List[int],
    protected_attributes: Dict[str, List[Any]]
) -> List[FairnessValidationResult]:
    """Convenience function for fairness validation."""
    analyzer = FairnessValidationAnalyzer()
    return analyzer.comprehensive_fairness_audit(predictions, labels, protected_attributes)


def validate_robustness(
    original_performance: float,
    test_results: Dict[RobustnessTest, Dict[str, Any]]
) -> List[RobustnessValidationResult]:
    """Convenience function for robustness validation."""
    analyzer = RobustnessValidationAnalyzer()
    return analyzer.comprehensive_robustness_test(original_performance, test_results)


def validate_calibration(
    probabilities: List[float],
    labels: List[int]
) -> CalibrationResult:
    """Convenience function for calibration validation."""
    analyzer = CalibrationValidationAnalyzer()
    return analyzer.calculate_ece(probabilities, labels)


def comprehensive_validation(
    cv_scores: List[float],
    metrics: Dict[MetricType, float],
    predictions: List[int],
    labels: List[int],
    probabilities: List[float],
    protected_attributes: Dict[str, List[Any]],
    robustness_tests: Dict[RobustnessTest, Dict[str, Any]]
) -> ComprehensiveValidationReport:
    """Convenience function for comprehensive validation."""
    analyzer = ComprehensiveValidationAnalyzer()
    return analyzer.perform_comprehensive_validation(
        cv_scores, metrics, predictions, labels,
        probabilities, protected_attributes, robustness_tests
    )
