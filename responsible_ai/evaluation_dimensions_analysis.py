"""
Evaluation Dimensions Analysis Module for AI/GenAI Research

This module provides comprehensive multi-dimensional evaluation frameworks
for assessing AI systems across Business, Technology, Sustainability,
Compliance, Performance, Statistical, and Human dimensions.

Includes radar matrix analysis for comparative visualization and
holistic scoring methodologies for research and production evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from abc import ABC, abstractmethod
import math
from datetime import datetime


class EvaluationDimensionType(Enum):
    """Primary evaluation dimension types."""
    BUSINESS = "business"
    TECHNOLOGY = "technology"
    SUSTAINABILITY = "sustainability"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    STATISTICAL = "statistical"
    HUMAN = "human"
    SECURITY = "security"
    ETHICS = "ethics"
    OPERATIONAL = "operational"


class BusinessMetric(Enum):
    """Business dimension metrics."""
    ROI = "return_on_investment"
    COST_EFFICIENCY = "cost_efficiency"
    TIME_TO_VALUE = "time_to_value"
    USER_ADOPTION = "user_adoption"
    REVENUE_IMPACT = "revenue_impact"
    MARKET_FIT = "market_fit"
    SCALABILITY = "scalability"
    COMPETITIVE_ADVANTAGE = "competitive_advantage"
    CUSTOMER_SATISFACTION = "customer_satisfaction"
    OPERATIONAL_SAVINGS = "operational_savings"


class TechnologyMetric(Enum):
    """Technology dimension metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    AVAILABILITY = "availability"
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    INTEROPERABILITY = "interoperability"
    SECURITY_POSTURE = "security_posture"
    API_QUALITY = "api_quality"
    CODE_QUALITY = "code_quality"
    INFRASTRUCTURE_COST = "infrastructure_cost"


class SustainabilityMetric(Enum):
    """Sustainability dimension metrics."""
    CARBON_FOOTPRINT = "carbon_footprint"
    ENERGY_CONSUMPTION = "energy_consumption"
    COMPUTE_EFFICIENCY = "compute_efficiency"
    RESOURCE_UTILIZATION = "resource_utilization"
    GREEN_HOSTING = "green_hosting"
    LIFECYCLE_IMPACT = "lifecycle_impact"
    E_WASTE_REDUCTION = "e_waste_reduction"
    RENEWABLE_ENERGY = "renewable_energy"
    WATER_USAGE = "water_usage"
    SUSTAINABILITY_SCORE = "sustainability_score"


class ComplianceMetric(Enum):
    """Compliance dimension metrics."""
    GDPR_COMPLIANCE = "gdpr_compliance"
    HIPAA_COMPLIANCE = "hipaa_compliance"
    SOC2_COMPLIANCE = "soc2_compliance"
    ISO27001_COMPLIANCE = "iso27001_compliance"
    DATA_PRIVACY = "data_privacy"
    AUDIT_READINESS = "audit_readiness"
    POLICY_ADHERENCE = "policy_adherence"
    REGULATORY_RISK = "regulatory_risk"
    DOCUMENTATION_QUALITY = "documentation_quality"
    CONSENT_MANAGEMENT = "consent_management"


class PerformanceMetric(Enum):
    """Performance dimension metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    LATENCY_P50 = "latency_p50"
    LATENCY_P99 = "latency_p99"
    ERROR_RATE = "error_rate"
    UPTIME = "uptime"
    RESPONSE_TIME = "response_time"
    CONCURRENCY = "concurrency"


class StatisticalMetric(Enum):
    """Statistical dimension metrics."""
    DISTRIBUTION_MATCH = "distribution_match"
    CALIBRATION = "calibration"
    CONFIDENCE = "confidence"
    VARIANCE = "variance"
    BIAS = "bias"
    CONSISTENCY = "consistency"
    REPRODUCIBILITY = "reproducibility"
    STATISTICAL_POWER = "statistical_power"
    EFFECT_SIZE = "effect_size"
    SIGNIFICANCE = "significance"


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""
    dimension: EvaluationDimensionType
    score: float  # 0-100
    metrics: Dict[str, float]
    weight: float
    confidence: float
    trend: str  # "improving", "stable", "declining"
    strengths: List[str]
    weaknesses: List[str]
    interpretation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "metrics": self.metrics,
            "weight": self.weight,
            "confidence": self.confidence,
            "trend": self.trend,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "interpretation": self.interpretation,
            "timestamp": self.timestamp
        }


@dataclass
class RadarMatrixResult:
    """Result of radar matrix analysis."""
    dimensions: List[str]
    scores: Dict[str, float]
    normalized_scores: Dict[str, float]
    area_score: float
    balance_score: float
    min_dimension: str
    max_dimension: str
    improvement_priorities: List[str]
    comparison_data: Optional[Dict[str, Dict[str, float]]]
    interpretation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimensions": self.dimensions,
            "scores": self.scores,
            "normalized_scores": self.normalized_scores,
            "area_score": self.area_score,
            "balance_score": self.balance_score,
            "min_dimension": self.min_dimension,
            "max_dimension": self.max_dimension,
            "improvement_priorities": self.improvement_priorities,
            "comparison_data": self.comparison_data,
            "interpretation": self.interpretation,
            "timestamp": self.timestamp
        }


@dataclass
class ComprehensiveEvaluationResult:
    """Comprehensive multi-dimensional evaluation result."""
    evaluation_id: str
    overall_score: float
    dimension_scores: Dict[str, DimensionScore]
    radar_analysis: RadarMatrixResult
    weighted_aggregate: float
    risk_assessment: Dict[str, str]
    recommendations: List[str]
    executive_summary: str
    detailed_analysis: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "overall_score": self.overall_score,
            "dimension_scores": {k: v.to_dict() for k, v in self.dimension_scores.items()},
            "radar_analysis": self.radar_analysis.to_dict(),
            "weighted_aggregate": self.weighted_aggregate,
            "risk_assessment": self.risk_assessment,
            "recommendations": self.recommendations,
            "executive_summary": self.executive_summary,
            "detailed_analysis": self.detailed_analysis,
            "timestamp": self.timestamp
        }


class BaseDimensionAnalyzer(ABC):
    """Abstract base class for dimension analyzers."""

    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> DimensionScore:
        """Analyze dimension-specific metrics."""
        pass

    @abstractmethod
    def get_dimension_type(self) -> EvaluationDimensionType:
        """Return the dimension type."""
        pass


class BusinessDimensionAnalyzer(BaseDimensionAnalyzer):
    """
    Analyzer for Business dimension evaluation.

    Assesses business value, ROI, market fit, and organizational impact
    of AI systems.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "roi": 0.2,
            "cost_efficiency": 0.15,
            "user_adoption": 0.15,
            "revenue_impact": 0.15,
            "scalability": 0.1,
            "competitive_advantage": 0.1,
            "customer_satisfaction": 0.15
        }

    def get_dimension_type(self) -> EvaluationDimensionType:
        return EvaluationDimensionType.BUSINESS

    def analyze(self, data: Dict[str, Any]) -> DimensionScore:
        """
        Analyze business dimension metrics.

        Args:
            data: Dictionary with business metrics (0-100 scale)

        Returns:
            DimensionScore with business evaluation
        """
        metrics = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, weight in self.weights.items():
            if metric in data:
                value = min(100, max(0, data[metric]))
                metrics[metric] = value
                weighted_sum += value * weight
                total_weight += weight

        score = weighted_sum / total_weight if total_weight > 0 else 0

        # Identify strengths and weaknesses
        strengths = [m for m, v in metrics.items() if v >= 70]
        weaknesses = [m for m, v in metrics.items() if v < 50]

        # Determine trend
        trend = data.get("trend", "stable")

        interpretation = self._interpret_score(score, strengths, weaknesses)

        return DimensionScore(
            dimension=EvaluationDimensionType.BUSINESS,
            score=score,
            metrics=metrics,
            weight=1.0,
            confidence=self._calculate_confidence(metrics),
            trend=trend,
            strengths=strengths,
            weaknesses=weaknesses,
            interpretation=interpretation
        )

    def _interpret_score(
        self,
        score: float,
        strengths: List[str],
        weaknesses: List[str]
    ) -> str:
        """Generate business dimension interpretation."""
        if score >= 80:
            level = "excellent"
            outlook = "Strong business value demonstrated."
        elif score >= 60:
            level = "good"
            outlook = "Positive business impact with room for optimization."
        elif score >= 40:
            level = "moderate"
            outlook = "Business value present but improvements needed."
        else:
            level = "concerning"
            outlook = "Significant business value concerns to address."

        return (
            f"Business dimension score: {score:.1f}/100 ({level}). {outlook} "
            f"Key strengths: {', '.join(strengths) if strengths else 'None identified'}. "
            f"Areas needing attention: {', '.join(weaknesses) if weaknesses else 'None critical'}."
        )

    def _calculate_confidence(self, metrics: Dict[str, float]) -> float:
        """Calculate confidence based on data completeness."""
        expected_metrics = len(self.weights)
        actual_metrics = len(metrics)
        return actual_metrics / expected_metrics if expected_metrics > 0 else 0


class TechnologyDimensionAnalyzer(BaseDimensionAnalyzer):
    """
    Analyzer for Technology dimension evaluation.

    Assesses technical quality, performance, reliability, and
    architectural soundness of AI systems.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "latency": 0.15,
            "throughput": 0.1,
            "availability": 0.15,
            "reliability": 0.15,
            "maintainability": 0.1,
            "security_posture": 0.15,
            "code_quality": 0.1,
            "interoperability": 0.1
        }

    def get_dimension_type(self) -> EvaluationDimensionType:
        return EvaluationDimensionType.TECHNOLOGY

    def analyze(self, data: Dict[str, Any]) -> DimensionScore:
        """
        Analyze technology dimension metrics.

        Args:
            data: Dictionary with technology metrics

        Returns:
            DimensionScore with technology evaluation
        """
        metrics = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, weight in self.weights.items():
            if metric in data:
                value = min(100, max(0, data[metric]))
                metrics[metric] = value
                weighted_sum += value * weight
                total_weight += weight

        score = weighted_sum / total_weight if total_weight > 0 else 0

        strengths = [m for m, v in metrics.items() if v >= 70]
        weaknesses = [m for m, v in metrics.items() if v < 50]
        trend = data.get("trend", "stable")

        interpretation = self._interpret_score(score, strengths, weaknesses)

        return DimensionScore(
            dimension=EvaluationDimensionType.TECHNOLOGY,
            score=score,
            metrics=metrics,
            weight=1.0,
            confidence=len(metrics) / len(self.weights) if self.weights else 0,
            trend=trend,
            strengths=strengths,
            weaknesses=weaknesses,
            interpretation=interpretation
        )

    def _interpret_score(
        self,
        score: float,
        strengths: List[str],
        weaknesses: List[str]
    ) -> str:
        """Generate technology dimension interpretation."""
        if score >= 80:
            level = "excellent"
            outlook = "Technical implementation is robust and well-architected."
        elif score >= 60:
            level = "good"
            outlook = "Sound technical foundation with some areas for improvement."
        elif score >= 40:
            level = "moderate"
            outlook = "Technical debt present; optimization recommended."
        else:
            level = "concerning"
            outlook = "Significant technical issues require attention."

        return (
            f"Technology dimension score: {score:.1f}/100 ({level}). {outlook}"
        )


class SustainabilityDimensionAnalyzer(BaseDimensionAnalyzer):
    """
    Analyzer for Sustainability dimension evaluation.

    Assesses environmental impact, energy efficiency, and
    resource utilization of AI systems.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "carbon_footprint": 0.2,
            "energy_consumption": 0.2,
            "compute_efficiency": 0.15,
            "resource_utilization": 0.15,
            "green_hosting": 0.1,
            "renewable_energy": 0.1,
            "lifecycle_impact": 0.1
        }

    def get_dimension_type(self) -> EvaluationDimensionType:
        return EvaluationDimensionType.SUSTAINABILITY

    def analyze(self, data: Dict[str, Any]) -> DimensionScore:
        """
        Analyze sustainability dimension metrics.

        Args:
            data: Dictionary with sustainability metrics

        Returns:
            DimensionScore with sustainability evaluation
        """
        metrics = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, weight in self.weights.items():
            if metric in data:
                value = min(100, max(0, data[metric]))
                metrics[metric] = value
                weighted_sum += value * weight
                total_weight += weight

        score = weighted_sum / total_weight if total_weight > 0 else 0

        strengths = [m for m, v in metrics.items() if v >= 70]
        weaknesses = [m for m, v in metrics.items() if v < 50]
        trend = data.get("trend", "stable")

        interpretation = self._interpret_score(score, metrics)

        return DimensionScore(
            dimension=EvaluationDimensionType.SUSTAINABILITY,
            score=score,
            metrics=metrics,
            weight=1.0,
            confidence=len(metrics) / len(self.weights) if self.weights else 0,
            trend=trend,
            strengths=strengths,
            weaknesses=weaknesses,
            interpretation=interpretation
        )

    def _interpret_score(self, score: float, metrics: Dict[str, float]) -> str:
        """Generate sustainability interpretation."""
        if score >= 80:
            level = "sustainable"
            outlook = "Excellent environmental responsibility demonstrated."
        elif score >= 60:
            level = "moderately sustainable"
            outlook = "Good sustainability practices with improvement opportunities."
        elif score >= 40:
            level = "needs improvement"
            outlook = "Sustainability gaps identified; action recommended."
        else:
            level = "concerning"
            outlook = "Significant environmental impact; urgent improvements needed."

        carbon = metrics.get("carbon_footprint", 0)
        energy = metrics.get("energy_consumption", 0)

        return (
            f"Sustainability score: {score:.1f}/100 ({level}). {outlook} "
            f"Carbon efficiency: {carbon:.0f}/100, Energy efficiency: {energy:.0f}/100."
        )


class ComplianceDimensionAnalyzer(BaseDimensionAnalyzer):
    """
    Analyzer for Compliance dimension evaluation.

    Assesses regulatory compliance, data privacy, and
    governance adherence of AI systems.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "gdpr_compliance": 0.2,
            "data_privacy": 0.2,
            "audit_readiness": 0.15,
            "policy_adherence": 0.15,
            "regulatory_risk": 0.15,
            "documentation_quality": 0.15
        }

    def get_dimension_type(self) -> EvaluationDimensionType:
        return EvaluationDimensionType.COMPLIANCE

    def analyze(self, data: Dict[str, Any]) -> DimensionScore:
        """
        Analyze compliance dimension metrics.

        Args:
            data: Dictionary with compliance metrics

        Returns:
            DimensionScore with compliance evaluation
        """
        metrics = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, weight in self.weights.items():
            if metric in data:
                value = min(100, max(0, data[metric]))
                metrics[metric] = value
                weighted_sum += value * weight
                total_weight += weight

        score = weighted_sum / total_weight if total_weight > 0 else 0

        strengths = [m for m, v in metrics.items() if v >= 80]
        weaknesses = [m for m, v in metrics.items() if v < 60]
        trend = data.get("trend", "stable")

        # Compliance requires higher thresholds
        critical_gaps = [m for m, v in metrics.items() if v < 50]

        interpretation = self._interpret_score(score, critical_gaps)

        return DimensionScore(
            dimension=EvaluationDimensionType.COMPLIANCE,
            score=score,
            metrics=metrics,
            weight=1.0,
            confidence=len(metrics) / len(self.weights) if self.weights else 0,
            trend=trend,
            strengths=strengths,
            weaknesses=weaknesses,
            interpretation=interpretation
        )

    def _interpret_score(self, score: float, critical_gaps: List[str]) -> str:
        """Generate compliance interpretation."""
        if score >= 90:
            level = "fully compliant"
            risk = "low"
        elif score >= 70:
            level = "substantially compliant"
            risk = "moderate"
        elif score >= 50:
            level = "partially compliant"
            risk = "elevated"
        else:
            level = "non-compliant"
            risk = "high"

        if critical_gaps:
            gaps_str = f" Critical gaps in: {', '.join(critical_gaps)}."
        else:
            gaps_str = ""

        return (
            f"Compliance score: {score:.1f}/100 ({level}). "
            f"Regulatory risk level: {risk}.{gaps_str}"
        )


class PerformanceDimensionAnalyzer(BaseDimensionAnalyzer):
    """
    Analyzer for Performance dimension evaluation.

    Assesses model accuracy, latency, throughput, and
    operational performance of AI systems.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "accuracy": 0.2,
            "f1_score": 0.15,
            "latency_p50": 0.15,
            "latency_p99": 0.1,
            "error_rate": 0.15,
            "uptime": 0.15,
            "throughput": 0.1
        }

    def get_dimension_type(self) -> EvaluationDimensionType:
        return EvaluationDimensionType.PERFORMANCE

    def analyze(self, data: Dict[str, Any]) -> DimensionScore:
        """
        Analyze performance dimension metrics.

        Args:
            data: Dictionary with performance metrics

        Returns:
            DimensionScore with performance evaluation
        """
        metrics = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, weight in self.weights.items():
            if metric in data:
                value = min(100, max(0, data[metric]))
                metrics[metric] = value
                weighted_sum += value * weight
                total_weight += weight

        score = weighted_sum / total_weight if total_weight > 0 else 0

        strengths = [m for m, v in metrics.items() if v >= 75]
        weaknesses = [m for m, v in metrics.items() if v < 50]
        trend = data.get("trend", "stable")

        interpretation = self._interpret_score(score, metrics)

        return DimensionScore(
            dimension=EvaluationDimensionType.PERFORMANCE,
            score=score,
            metrics=metrics,
            weight=1.0,
            confidence=len(metrics) / len(self.weights) if self.weights else 0,
            trend=trend,
            strengths=strengths,
            weaknesses=weaknesses,
            interpretation=interpretation
        )

    def _interpret_score(self, score: float, metrics: Dict[str, float]) -> str:
        """Generate performance interpretation."""
        if score >= 85:
            level = "excellent"
        elif score >= 70:
            level = "good"
        elif score >= 50:
            level = "acceptable"
        else:
            level = "needs improvement"

        accuracy = metrics.get("accuracy", 0)
        uptime = metrics.get("uptime", 0)

        return (
            f"Performance score: {score:.1f}/100 ({level}). "
            f"Accuracy: {accuracy:.0f}/100, Uptime: {uptime:.0f}/100."
        )


class StatisticalDimensionAnalyzer(BaseDimensionAnalyzer):
    """
    Analyzer for Statistical dimension evaluation.

    Assesses statistical validity, calibration, reproducibility,
    and methodological rigor of AI systems.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "distribution_match": 0.15,
            "calibration": 0.15,
            "confidence": 0.15,
            "consistency": 0.15,
            "reproducibility": 0.15,
            "statistical_power": 0.15,
            "bias": 0.1
        }

    def get_dimension_type(self) -> EvaluationDimensionType:
        return EvaluationDimensionType.STATISTICAL

    def analyze(self, data: Dict[str, Any]) -> DimensionScore:
        """
        Analyze statistical dimension metrics.

        Args:
            data: Dictionary with statistical metrics

        Returns:
            DimensionScore with statistical evaluation
        """
        metrics = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, weight in self.weights.items():
            if metric in data:
                value = min(100, max(0, data[metric]))
                metrics[metric] = value
                weighted_sum += value * weight
                total_weight += weight

        score = weighted_sum / total_weight if total_weight > 0 else 0

        strengths = [m for m, v in metrics.items() if v >= 70]
        weaknesses = [m for m, v in metrics.items() if v < 50]
        trend = data.get("trend", "stable")

        interpretation = self._interpret_score(score, metrics)

        return DimensionScore(
            dimension=EvaluationDimensionType.STATISTICAL,
            score=score,
            metrics=metrics,
            weight=1.0,
            confidence=len(metrics) / len(self.weights) if self.weights else 0,
            trend=trend,
            strengths=strengths,
            weaknesses=weaknesses,
            interpretation=interpretation
        )

    def _interpret_score(self, score: float, metrics: Dict[str, float]) -> str:
        """Generate statistical interpretation."""
        if score >= 80:
            rigor = "high"
            validity = "strong"
        elif score >= 60:
            rigor = "moderate"
            validity = "acceptable"
        elif score >= 40:
            rigor = "low"
            validity = "questionable"
        else:
            rigor = "insufficient"
            validity = "weak"

        return (
            f"Statistical rigor: {rigor}, Validity: {validity}. "
            f"Overall score: {score:.1f}/100."
        )


class RadarMatrixAnalyzer:
    """
    Analyzer for radar/spider chart matrix evaluation.

    Creates multi-dimensional visualizations and computes
    aggregate scores across evaluation dimensions.
    """

    def __init__(self, dimensions: Optional[List[str]] = None):
        self.dimensions = dimensions or [
            "Business", "Technology", "Sustainability",
            "Compliance", "Performance", "Statistical", "Human"
        ]

    def analyze(
        self,
        scores: Dict[str, float],
        comparison_scores: Optional[Dict[str, Dict[str, float]]] = None
    ) -> RadarMatrixResult:
        """
        Perform radar matrix analysis.

        Args:
            scores: Dictionary mapping dimensions to scores (0-100)
            comparison_scores: Optional comparison data for benchmarking

        Returns:
            RadarMatrixResult with comprehensive analysis
        """
        # Validate and normalize scores
        validated_scores = {}
        for dim in self.dimensions:
            if dim in scores:
                validated_scores[dim] = min(100, max(0, scores[dim]))
            else:
                validated_scores[dim] = 50  # Default neutral score

        # Normalize to 0-1 for area calculation
        normalized = {k: v / 100 for k, v in validated_scores.items()}

        # Calculate area score (polygon area)
        area_score = self._calculate_polygon_area(normalized)

        # Calculate balance score (uniformity)
        balance_score = self._calculate_balance_score(validated_scores)

        # Find min and max dimensions
        min_dim = min(validated_scores.items(), key=lambda x: x[1])[0]
        max_dim = max(validated_scores.items(), key=lambda x: x[1])[0]

        # Determine improvement priorities
        priorities = self._determine_priorities(validated_scores)

        interpretation = self._generate_interpretation(
            area_score, balance_score, min_dim, max_dim
        )

        return RadarMatrixResult(
            dimensions=self.dimensions,
            scores=validated_scores,
            normalized_scores=normalized,
            area_score=area_score * 100,  # Convert back to percentage
            balance_score=balance_score * 100,
            min_dimension=min_dim,
            max_dimension=max_dim,
            improvement_priorities=priorities,
            comparison_data=comparison_scores,
            interpretation=interpretation
        )

    def _calculate_polygon_area(self, normalized_scores: Dict[str, float]) -> float:
        """Calculate the area of the radar chart polygon."""
        n = len(self.dimensions)
        if n < 3:
            return 0.0

        # Get scores in order
        scores = [normalized_scores.get(dim, 0.5) for dim in self.dimensions]

        # Calculate area using shoelace formula for regular polygon
        angle = 2 * math.pi / n
        area = 0.0

        for i in range(n):
            j = (i + 1) % n
            # Each triangle from center
            area += 0.5 * scores[i] * scores[j] * math.sin(angle)

        # Normalize by maximum possible area (unit radius)
        max_area = 0.5 * n * math.sin(angle)
        return area / max_area if max_area > 0 else 0

    def _calculate_balance_score(self, scores: Dict[str, float]) -> float:
        """Calculate how balanced the scores are across dimensions."""
        values = list(scores.values())
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance)

        # Higher balance when lower std (more uniform)
        # Normalize: std of 50 (max with 0-100 range) would give 0 balance
        balance = 1 - (std / 50)
        return max(0, min(1, balance))

    def _determine_priorities(self, scores: Dict[str, float]) -> List[str]:
        """Determine improvement priorities based on scores."""
        # Sort by score (lowest first)
        sorted_dims = sorted(scores.items(), key=lambda x: x[1])

        # Prioritize dimensions below 60
        priorities = []
        for dim, score in sorted_dims:
            if score < 60:
                priorities.append(f"{dim} ({score:.0f})")
            if len(priorities) >= 3:
                break

        if not priorities:
            priorities.append("All dimensions at acceptable levels")

        return priorities

    def _generate_interpretation(
        self,
        area: float,
        balance: float,
        min_dim: str,
        max_dim: str
    ) -> str:
        """Generate radar matrix interpretation."""
        # Overall health assessment
        if area >= 0.7 and balance >= 0.7:
            health = "excellent"
            advice = "Maintain current performance across dimensions."
        elif area >= 0.5 and balance >= 0.5:
            health = "good"
            advice = f"Focus on improving {min_dim} for better balance."
        elif area >= 0.3:
            health = "moderate"
            advice = "Multiple dimensions need attention."
        else:
            health = "needs improvement"
            advice = "Comprehensive improvement program recommended."

        return (
            f"Overall evaluation health: {health}. "
            f"Coverage: {area*100:.1f}%, Balance: {balance*100:.1f}%. "
            f"Strongest: {max_dim}, Weakest: {min_dim}. {advice}"
        )

    def compare(
        self,
        baseline: Dict[str, float],
        comparison: Dict[str, float],
        labels: Tuple[str, str] = ("Baseline", "Comparison")
    ) -> Dict[str, Any]:
        """
        Compare two radar matrices.

        Args:
            baseline: Baseline scores
            comparison: Comparison scores
            labels: Names for the two datasets

        Returns:
            Comparison analysis
        """
        baseline_result = self.analyze(baseline)
        comparison_result = self.analyze(comparison)

        differences = {}
        for dim in self.dimensions:
            b_score = baseline.get(dim, 0)
            c_score = comparison.get(dim, 0)
            differences[dim] = {
                "baseline": b_score,
                "comparison": c_score,
                "delta": c_score - b_score,
                "improved": c_score > b_score
            }

        improved_dims = [d for d, v in differences.items() if v["improved"]]
        declined_dims = [d for d, v in differences.items() if not v["improved"] and v["delta"] != 0]

        return {
            "baseline": {
                "label": labels[0],
                "area_score": baseline_result.area_score,
                "balance_score": baseline_result.balance_score
            },
            "comparison": {
                "label": labels[1],
                "area_score": comparison_result.area_score,
                "balance_score": comparison_result.balance_score
            },
            "differences": differences,
            "improved_dimensions": improved_dims,
            "declined_dimensions": declined_dims,
            "overall_delta": comparison_result.area_score - baseline_result.area_score
        }


class ComprehensiveEvaluator:
    """
    Comprehensive evaluator combining all dimension analyzers.

    Provides holistic AI system evaluation across Business, Technology,
    Sustainability, Compliance, Performance, and Statistical dimensions.
    """

    def __init__(
        self,
        dimension_weights: Optional[Dict[str, float]] = None
    ):
        self.analyzers = {
            "business": BusinessDimensionAnalyzer(),
            "technology": TechnologyDimensionAnalyzer(),
            "sustainability": SustainabilityDimensionAnalyzer(),
            "compliance": ComplianceDimensionAnalyzer(),
            "performance": PerformanceDimensionAnalyzer(),
            "statistical": StatisticalDimensionAnalyzer()
        }

        self.dimension_weights = dimension_weights or {
            "business": 0.2,
            "technology": 0.2,
            "sustainability": 0.1,
            "compliance": 0.15,
            "performance": 0.2,
            "statistical": 0.15
        }

        self.radar_analyzer = RadarMatrixAnalyzer()

    def evaluate(
        self,
        dimension_data: Dict[str, Dict[str, Any]]
    ) -> ComprehensiveEvaluationResult:
        """
        Perform comprehensive multi-dimensional evaluation.

        Args:
            dimension_data: Dictionary mapping dimension names to their metrics

        Returns:
            ComprehensiveEvaluationResult with full analysis
        """
        # Analyze each dimension
        dimension_scores = {}
        radar_scores = {}

        for dim_name, analyzer in self.analyzers.items():
            if dim_name in dimension_data:
                score = analyzer.analyze(dimension_data[dim_name])
                dimension_scores[dim_name] = score
                radar_scores[dim_name.capitalize()] = score.score

        # Calculate weighted aggregate
        weighted_sum = 0.0
        total_weight = 0.0
        for dim_name, score in dimension_scores.items():
            weight = self.dimension_weights.get(dim_name, 0.1)
            weighted_sum += score.score * weight
            total_weight += weight

        weighted_aggregate = weighted_sum / total_weight if total_weight > 0 else 0

        # Radar analysis
        radar_result = self.radar_analyzer.analyze(radar_scores)

        # Risk assessment
        risk_assessment = self._assess_risks(dimension_scores)

        # Generate recommendations
        recommendations = self._generate_recommendations(dimension_scores, radar_result)

        # Executive summary
        executive_summary = self._generate_executive_summary(
            weighted_aggregate, dimension_scores, risk_assessment
        )

        # Detailed analysis
        detailed_analysis = self._compile_detailed_analysis(dimension_scores)

        return ComprehensiveEvaluationResult(
            evaluation_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            overall_score=weighted_aggregate,
            dimension_scores=dimension_scores,
            radar_analysis=radar_result,
            weighted_aggregate=weighted_aggregate,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            executive_summary=executive_summary,
            detailed_analysis=detailed_analysis
        )

    def _assess_risks(self, dimension_scores: Dict[str, DimensionScore]) -> Dict[str, str]:
        """Assess risks based on dimension scores."""
        risks = {}

        for dim_name, score in dimension_scores.items():
            if score.score < 40:
                risks[dim_name] = "high"
            elif score.score < 60:
                risks[dim_name] = "medium"
            elif score.score < 80:
                risks[dim_name] = "low"
            else:
                risks[dim_name] = "minimal"

        return risks

    def _generate_recommendations(
        self,
        dimension_scores: Dict[str, DimensionScore],
        radar_result: RadarMatrixResult
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Add priority improvements from radar
        for priority in radar_result.improvement_priorities[:3]:
            if "acceptable" not in priority.lower():
                recommendations.append(f"Priority: Improve {priority}")

        # Add dimension-specific recommendations
        for dim_name, score in dimension_scores.items():
            if score.weaknesses:
                top_weakness = score.weaknesses[0]
                recommendations.append(
                    f"{dim_name.capitalize()}: Address {top_weakness} (current score: {score.metrics.get(top_weakness, 0):.0f})"
                )

        if not recommendations:
            recommendations.append("Performance is satisfactory across all dimensions. Focus on maintaining current levels.")

        return recommendations[:5]  # Top 5 recommendations

    def _generate_executive_summary(
        self,
        overall_score: float,
        dimension_scores: Dict[str, DimensionScore],
        risks: Dict[str, str]
    ) -> str:
        """Generate executive summary."""
        # Overall assessment
        if overall_score >= 80:
            assessment = "Excellent"
            outlook = "AI system demonstrates strong performance across all dimensions."
        elif overall_score >= 60:
            assessment = "Good"
            outlook = "AI system performs well with identified areas for improvement."
        elif overall_score >= 40:
            assessment = "Moderate"
            outlook = "AI system has significant gaps requiring attention."
        else:
            assessment = "Needs Improvement"
            outlook = "AI system requires comprehensive improvement program."

        # High risk count
        high_risks = [d for d, r in risks.items() if r == "high"]
        risk_note = f" {len(high_risks)} high-risk areas identified." if high_risks else ""

        return (
            f"Overall Assessment: {assessment} ({overall_score:.1f}/100). {outlook}{risk_note}"
        )

    def _compile_detailed_analysis(
        self,
        dimension_scores: Dict[str, DimensionScore]
    ) -> Dict[str, Any]:
        """Compile detailed analysis data."""
        return {
            "dimension_breakdown": {
                dim: {
                    "score": score.score,
                    "confidence": score.confidence,
                    "trend": score.trend,
                    "top_metric": max(score.metrics.items(), key=lambda x: x[1])[0] if score.metrics else None,
                    "bottom_metric": min(score.metrics.items(), key=lambda x: x[1])[0] if score.metrics else None
                }
                for dim, score in dimension_scores.items()
            },
            "cross_dimension_correlation": "Analysis available",
            "trend_analysis": "Historical comparison available"
        }


# Utility functions
def analyze_business_dimension(data: Dict[str, Any]) -> DimensionScore:
    """Convenience function for business dimension analysis."""
    analyzer = BusinessDimensionAnalyzer()
    return analyzer.analyze(data)


def analyze_technology_dimension(data: Dict[str, Any]) -> DimensionScore:
    """Convenience function for technology dimension analysis."""
    analyzer = TechnologyDimensionAnalyzer()
    return analyzer.analyze(data)


def analyze_sustainability_dimension(data: Dict[str, Any]) -> DimensionScore:
    """Convenience function for sustainability dimension analysis."""
    analyzer = SustainabilityDimensionAnalyzer()
    return analyzer.analyze(data)


def analyze_compliance_dimension(data: Dict[str, Any]) -> DimensionScore:
    """Convenience function for compliance dimension analysis."""
    analyzer = ComplianceDimensionAnalyzer()
    return analyzer.analyze(data)


def create_radar_matrix(scores: Dict[str, float]) -> RadarMatrixResult:
    """Convenience function for radar matrix analysis."""
    analyzer = RadarMatrixAnalyzer()
    return analyzer.analyze(scores)


def perform_comprehensive_evaluation(
    dimension_data: Dict[str, Dict[str, Any]]
) -> ComprehensiveEvaluationResult:
    """Convenience function for comprehensive evaluation."""
    evaluator = ComprehensiveEvaluator()
    return evaluator.evaluate(dimension_data)
