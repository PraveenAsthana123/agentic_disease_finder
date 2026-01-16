"""
Performance and Governance Matrix Analysis Module for AI Evaluation.

This module provides comprehensive frameworks for analyzing AI systems
through Performance Matrix (execution, efficiency, optimization) and
Governance Matrix (permission, risk, compliance) perspectives.

Classes:
    PerformanceMatrixAnalyzer: Execution, efficiency, and optimization metrics
    GovernanceMatrixAnalyzer: Permission, risk, and compliance assessment
    ExecutionPerformanceAnalyzer: Task execution quality analysis
    EfficiencyAnalyzer: Resource and computational efficiency
    PermissionAnalyzer: Access control and authorization analysis
    RiskAssessmentAnalyzer: Risk identification and mitigation analysis
    ComplianceScoreAnalyzer: Regulatory and policy compliance
    IntegratedMatrixAnalyzer: Combined performance-governance analysis

Author: AgenticFinder Research Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict


class PerformanceCategory(Enum):
    """Categories of performance metrics."""
    EXECUTION = "execution"
    EFFICIENCY = "efficiency"
    OPTIMIZATION = "optimization"
    SCALABILITY = "scalability"
    RELIABILITY = "reliability"


class GovernanceCategory(Enum):
    """Categories of governance metrics."""
    PERMISSION = "permission"
    RISK = "risk"
    COMPLIANCE = "compliance"
    ACCOUNTABILITY = "accountability"
    TRANSPARENCY = "transparency"


class RiskLevel(Enum):
    """Risk level classifications."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1


class ComplianceStatus(Enum):
    """Compliance status classifications."""
    FULLY_COMPLIANT = "fully_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class PerformanceMetric:
    """Container for performance metric results."""
    name: str
    value: float
    unit: str
    threshold: Optional[float] = None
    status: str = "normal"
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def meets_threshold(self) -> bool:
        """Check if metric meets threshold."""
        if self.threshold is None:
            return True
        return self.value >= self.threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'threshold': self.threshold,
            'status': self.status,
            'meets_threshold': self.meets_threshold,
            'details': self.details
        }


@dataclass
class GovernanceMetric:
    """Container for governance metric results."""
    name: str
    score: float
    category: GovernanceCategory
    risk_level: RiskLevel = RiskLevel.MINIMAL
    compliance_status: ComplianceStatus = ComplianceStatus.FULLY_COMPLIANT
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'score': self.score,
            'category': self.category.value,
            'risk_level': self.risk_level.name,
            'compliance_status': self.compliance_status.value,
            'issues': self.issues,
            'recommendations': self.recommendations,
            'metadata': self.metadata
        }


@dataclass
class MatrixResult:
    """Combined matrix analysis result."""
    performance_scores: Dict[str, PerformanceMetric] = field(default_factory=dict)
    governance_scores: Dict[str, GovernanceMetric] = field(default_factory=dict)
    overall_performance: float = 0.0
    overall_governance: float = 0.0
    combined_score: float = 0.0
    risk_summary: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def get_radar_data(self) -> Dict[str, float]:
        """Get data for radar chart visualization."""
        data = {}
        for name, metric in self.performance_scores.items():
            data[f"perf_{name}"] = metric.value
        for name, metric in self.governance_scores.items():
            data[f"gov_{name}"] = metric.score
        return data


class PerformanceMatrixAnalyzer:
    """
    Analyzer for Performance Matrix metrics.

    Evaluates AI system performance across execution quality,
    efficiency, optimization, and reliability dimensions.
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize performance matrix analyzer.

        Args:
            thresholds: Optional custom thresholds for metrics
        """
        self.thresholds = thresholds or {
            'accuracy': 0.9,
            'latency': 1.0,
            'throughput': 100,
            'reliability': 0.99,
            'efficiency': 0.8
        }

        self.analyzers = {
            'execution': ExecutionPerformanceAnalyzer(),
            'efficiency': EfficiencyAnalyzer(),
            'optimization': OptimizationAnalyzer(),
            'reliability': ReliabilityAnalyzer()
        }

    def analyze(self, system_data: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None) -> Dict[str, PerformanceMetric]:
        """
        Analyze system performance.

        Args:
            system_data: System performance data
            context: Optional context information

        Returns:
            Dictionary of performance metrics
        """
        results = {}

        for category, analyzer in self.analyzers.items():
            category_metrics = analyzer.analyze(system_data, context)
            for name, metric in category_metrics.items():
                results[f"{category}_{name}"] = metric

        return results

    def calculate_aggregate(self, metrics: Dict[str, PerformanceMetric],
                            weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate aggregate performance score.

        Args:
            metrics: Dictionary of performance metrics
            weights: Optional weights for each metric

        Returns:
            Aggregate performance score
        """
        if not metrics:
            return 0.0

        weights = weights or {name: 1.0 for name in metrics}

        weighted_sum = 0.0
        weight_sum = 0.0

        for name, metric in metrics.items():
            w = weights.get(name, 1.0)
            # Normalize value if threshold exists
            if metric.threshold and metric.threshold > 0:
                normalized = min(1.0, metric.value / metric.threshold)
            else:
                normalized = metric.value
            weighted_sum += normalized * w
            weight_sum += w

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0


class ExecutionPerformanceAnalyzer:
    """Analyzer for execution performance metrics."""

    def __init__(self):
        """Initialize execution performance analyzer."""
        self.metrics = ['accuracy', 'completeness', 'correctness', 'response_quality']

    def analyze(self, data: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None) -> Dict[str, PerformanceMetric]:
        """
        Analyze execution performance.

        Args:
            data: Execution data
            context: Optional context

        Returns:
            Execution performance metrics
        """
        results = {}

        # Accuracy analysis
        accuracy = self._calculate_accuracy(data)
        results['accuracy'] = PerformanceMetric(
            name='accuracy',
            value=accuracy,
            unit='ratio',
            threshold=0.9,
            status='good' if accuracy >= 0.9 else 'warning' if accuracy >= 0.7 else 'critical'
        )

        # Completeness analysis
        completeness = self._calculate_completeness(data)
        results['completeness'] = PerformanceMetric(
            name='completeness',
            value=completeness,
            unit='ratio',
            threshold=0.95,
            status='good' if completeness >= 0.95 else 'warning'
        )

        # Correctness analysis
        correctness = self._calculate_correctness(data)
        results['correctness'] = PerformanceMetric(
            name='correctness',
            value=correctness,
            unit='ratio',
            threshold=0.98,
            status='good' if correctness >= 0.98 else 'warning'
        )

        # Response quality
        quality = self._calculate_response_quality(data, context)
        results['response_quality'] = PerformanceMetric(
            name='response_quality',
            value=quality,
            unit='score',
            threshold=0.8,
            status='good' if quality >= 0.8 else 'warning'
        )

        return results

    def _calculate_accuracy(self, data: Dict[str, Any]) -> float:
        """Calculate accuracy metric."""
        if 'predictions' in data and 'labels' in data:
            predictions = np.array(data['predictions'])
            labels = np.array(data['labels'])
            return float(np.mean(predictions == labels))

        if 'accuracy' in data:
            return float(data['accuracy'])

        if 'correct' in data and 'total' in data:
            return data['correct'] / data['total'] if data['total'] > 0 else 0.0

        return 0.8  # Default

    def _calculate_completeness(self, data: Dict[str, Any]) -> float:
        """Calculate completeness metric."""
        if 'completed_tasks' in data and 'total_tasks' in data:
            return data['completed_tasks'] / data['total_tasks'] if data['total_tasks'] > 0 else 0.0

        if 'completeness' in data:
            return float(data['completeness'])

        if 'missing_fields' in data and 'total_fields' in data:
            return 1.0 - (data['missing_fields'] / data['total_fields'])

        return 0.85  # Default

    def _calculate_correctness(self, data: Dict[str, Any]) -> float:
        """Calculate correctness metric."""
        if 'errors' in data and 'total' in data:
            return 1.0 - (data['errors'] / data['total']) if data['total'] > 0 else 0.0

        if 'correctness' in data:
            return float(data['correctness'])

        return 0.9  # Default

    def _calculate_response_quality(self, data: Dict[str, Any],
                                     context: Optional[Dict[str, Any]]) -> float:
        """Calculate response quality metric."""
        quality_factors = []

        if 'relevance_score' in data:
            quality_factors.append(data['relevance_score'])

        if 'coherence_score' in data:
            quality_factors.append(data['coherence_score'])

        if 'fluency_score' in data:
            quality_factors.append(data['fluency_score'])

        if quality_factors:
            return float(np.mean(quality_factors))

        return 0.75  # Default


class EfficiencyAnalyzer:
    """Analyzer for efficiency metrics."""

    def __init__(self):
        """Initialize efficiency analyzer."""
        self.metrics = ['latency', 'throughput', 'resource_utilization', 'cost_efficiency']

    def analyze(self, data: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None) -> Dict[str, PerformanceMetric]:
        """
        Analyze efficiency metrics.

        Args:
            data: Efficiency data
            context: Optional context

        Returns:
            Efficiency metrics
        """
        results = {}

        # Latency analysis
        latency = self._calculate_latency(data)
        results['latency'] = PerformanceMetric(
            name='latency',
            value=latency,
            unit='ms',
            threshold=1000,  # 1 second threshold
            status='good' if latency <= 1000 else 'warning' if latency <= 5000 else 'critical',
            details={'p50': latency * 0.8, 'p95': latency * 1.5, 'p99': latency * 2.0}
        )

        # Throughput analysis
        throughput = self._calculate_throughput(data)
        results['throughput'] = PerformanceMetric(
            name='throughput',
            value=throughput,
            unit='requests/sec',
            threshold=100,
            status='good' if throughput >= 100 else 'warning'
        )

        # Resource utilization
        utilization = self._calculate_resource_utilization(data)
        results['resource_utilization'] = PerformanceMetric(
            name='resource_utilization',
            value=utilization,
            unit='ratio',
            threshold=0.7,
            status='good' if 0.3 <= utilization <= 0.8 else 'warning'
        )

        # Cost efficiency
        cost_eff = self._calculate_cost_efficiency(data, context)
        results['cost_efficiency'] = PerformanceMetric(
            name='cost_efficiency',
            value=cost_eff,
            unit='score',
            threshold=0.7,
            status='good' if cost_eff >= 0.7 else 'warning'
        )

        return results

    def _calculate_latency(self, data: Dict[str, Any]) -> float:
        """Calculate latency metric."""
        if 'latency_ms' in data:
            return float(data['latency_ms'])

        if 'response_time' in data:
            return float(data['response_time'])

        if 'latencies' in data:
            return float(np.mean(data['latencies']))

        return 500.0  # Default 500ms

    def _calculate_throughput(self, data: Dict[str, Any]) -> float:
        """Calculate throughput metric."""
        if 'throughput' in data:
            return float(data['throughput'])

        if 'requests_per_second' in data:
            return float(data['requests_per_second'])

        if 'total_requests' in data and 'duration_seconds' in data:
            return data['total_requests'] / data['duration_seconds']

        return 50.0  # Default

    def _calculate_resource_utilization(self, data: Dict[str, Any]) -> float:
        """Calculate resource utilization."""
        utilizations = []

        if 'cpu_utilization' in data:
            utilizations.append(data['cpu_utilization'])

        if 'memory_utilization' in data:
            utilizations.append(data['memory_utilization'])

        if 'gpu_utilization' in data:
            utilizations.append(data['gpu_utilization'])

        if utilizations:
            return float(np.mean(utilizations))

        return 0.5  # Default

    def _calculate_cost_efficiency(self, data: Dict[str, Any],
                                    context: Optional[Dict[str, Any]]) -> float:
        """Calculate cost efficiency."""
        if 'cost_per_request' in data and 'value_per_request' in data:
            if data['cost_per_request'] > 0:
                return min(1.0, data['value_per_request'] / data['cost_per_request'])

        if 'cost_efficiency' in data:
            return float(data['cost_efficiency'])

        return 0.6  # Default


class OptimizationAnalyzer:
    """Analyzer for optimization metrics."""

    def __init__(self):
        """Initialize optimization analyzer."""
        self.metrics = ['convergence', 'improvement_rate', 'stability']

    def analyze(self, data: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None) -> Dict[str, PerformanceMetric]:
        """
        Analyze optimization metrics.

        Args:
            data: Optimization data
            context: Optional context

        Returns:
            Optimization metrics
        """
        results = {}

        # Convergence analysis
        convergence = self._calculate_convergence(data)
        results['convergence'] = PerformanceMetric(
            name='convergence',
            value=convergence,
            unit='ratio',
            threshold=0.95,
            status='good' if convergence >= 0.95 else 'warning'
        )

        # Improvement rate
        improvement = self._calculate_improvement_rate(data)
        results['improvement_rate'] = PerformanceMetric(
            name='improvement_rate',
            value=improvement,
            unit='percent',
            threshold=5.0,
            status='good' if improvement >= 5.0 else 'warning'
        )

        # Stability
        stability = self._calculate_stability(data)
        results['stability'] = PerformanceMetric(
            name='stability',
            value=stability,
            unit='ratio',
            threshold=0.9,
            status='good' if stability >= 0.9 else 'warning'
        )

        return results

    def _calculate_convergence(self, data: Dict[str, Any]) -> float:
        """Calculate convergence metric."""
        if 'convergence' in data:
            return float(data['convergence'])

        if 'loss_history' in data:
            losses = data['loss_history']
            if len(losses) >= 2:
                # Check if converged (small change in recent losses)
                recent_change = abs(losses[-1] - losses[-2]) / (losses[-2] + 1e-10)
                return 1.0 - min(1.0, recent_change * 10)

        return 0.8  # Default

    def _calculate_improvement_rate(self, data: Dict[str, Any]) -> float:
        """Calculate improvement rate."""
        if 'improvement_rate' in data:
            return float(data['improvement_rate'])

        if 'baseline_score' in data and 'current_score' in data:
            improvement = (data['current_score'] - data['baseline_score']) / data['baseline_score'] * 100
            return max(0.0, improvement)

        return 10.0  # Default 10% improvement

    def _calculate_stability(self, data: Dict[str, Any]) -> float:
        """Calculate stability metric."""
        if 'stability' in data:
            return float(data['stability'])

        if 'score_history' in data:
            scores = data['score_history']
            if len(scores) >= 2:
                std = np.std(scores)
                mean = np.mean(scores)
                cv = std / (mean + 1e-10)
                return 1.0 - min(1.0, cv)

        return 0.85  # Default


class ReliabilityAnalyzer:
    """Analyzer for reliability metrics."""

    def __init__(self):
        """Initialize reliability analyzer."""
        self.metrics = ['uptime', 'error_rate', 'recovery_time', 'consistency']

    def analyze(self, data: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None) -> Dict[str, PerformanceMetric]:
        """
        Analyze reliability metrics.

        Args:
            data: Reliability data
            context: Optional context

        Returns:
            Reliability metrics
        """
        results = {}

        # Uptime analysis
        uptime = self._calculate_uptime(data)
        results['uptime'] = PerformanceMetric(
            name='uptime',
            value=uptime,
            unit='percent',
            threshold=99.9,
            status='good' if uptime >= 99.9 else 'warning' if uptime >= 99.0 else 'critical'
        )

        # Error rate
        error_rate = self._calculate_error_rate(data)
        results['error_rate'] = PerformanceMetric(
            name='error_rate',
            value=error_rate,
            unit='percent',
            threshold=1.0,  # Max 1% errors
            status='good' if error_rate <= 1.0 else 'warning' if error_rate <= 5.0 else 'critical'
        )

        # Recovery time
        recovery = self._calculate_recovery_time(data)
        results['recovery_time'] = PerformanceMetric(
            name='recovery_time',
            value=recovery,
            unit='seconds',
            threshold=60,  # 1 minute recovery
            status='good' if recovery <= 60 else 'warning'
        )

        # Consistency
        consistency = self._calculate_consistency(data)
        results['consistency'] = PerformanceMetric(
            name='consistency',
            value=consistency,
            unit='ratio',
            threshold=0.95,
            status='good' if consistency >= 0.95 else 'warning'
        )

        return results

    def _calculate_uptime(self, data: Dict[str, Any]) -> float:
        """Calculate uptime metric."""
        if 'uptime' in data:
            return float(data['uptime'])

        if 'downtime_minutes' in data and 'total_minutes' in data:
            return (1 - data['downtime_minutes'] / data['total_minutes']) * 100

        return 99.5  # Default

    def _calculate_error_rate(self, data: Dict[str, Any]) -> float:
        """Calculate error rate."""
        if 'error_rate' in data:
            return float(data['error_rate'])

        if 'errors' in data and 'total_requests' in data:
            return (data['errors'] / data['total_requests']) * 100 if data['total_requests'] > 0 else 0.0

        return 0.5  # Default 0.5%

    def _calculate_recovery_time(self, data: Dict[str, Any]) -> float:
        """Calculate recovery time."""
        if 'mean_time_to_recovery' in data:
            return float(data['mean_time_to_recovery'])

        if 'recovery_times' in data:
            return float(np.mean(data['recovery_times']))

        return 30.0  # Default 30 seconds

    def _calculate_consistency(self, data: Dict[str, Any]) -> float:
        """Calculate consistency metric."""
        if 'consistency' in data:
            return float(data['consistency'])

        if 'outputs' in data:
            outputs = data['outputs']
            if len(outputs) >= 2:
                # Check output variance
                unique_ratio = len(set(str(o) for o in outputs)) / len(outputs)
                return 1.0 - min(1.0, unique_ratio - 0.1)

        return 0.9  # Default


class GovernanceMatrixAnalyzer:
    """
    Analyzer for Governance Matrix metrics.

    Evaluates AI system governance across permission, risk,
    compliance, and transparency dimensions.
    """

    def __init__(self, policies: Optional[Dict[str, Any]] = None):
        """
        Initialize governance matrix analyzer.

        Args:
            policies: Optional policy configurations
        """
        self.policies = policies or {}
        self.analyzers = {
            'permission': PermissionAnalyzer(),
            'risk': RiskAssessmentAnalyzer(),
            'compliance': ComplianceScoreAnalyzer(),
            'accountability': AccountabilityAnalyzer(),
            'transparency': TransparencyAnalyzer()
        }

    def analyze(self, system_data: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None) -> Dict[str, GovernanceMetric]:
        """
        Analyze system governance.

        Args:
            system_data: System governance data
            context: Optional context information

        Returns:
            Dictionary of governance metrics
        """
        results = {}

        for category, analyzer in self.analyzers.items():
            category_metrics = analyzer.analyze(system_data, context, self.policies)
            for name, metric in category_metrics.items():
                results[f"{category}_{name}"] = metric

        return results

    def calculate_aggregate(self, metrics: Dict[str, GovernanceMetric],
                            weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate aggregate governance score.

        Args:
            metrics: Dictionary of governance metrics
            weights: Optional weights for each metric

        Returns:
            Aggregate governance score
        """
        if not metrics:
            return 0.0

        weights = weights or {name: 1.0 for name in metrics}

        weighted_sum = 0.0
        weight_sum = 0.0

        for name, metric in metrics.items():
            w = weights.get(name, 1.0)
            weighted_sum += metric.score * w
            weight_sum += w

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0

    def get_risk_summary(self, metrics: Dict[str, GovernanceMetric]) -> Dict[str, int]:
        """
        Get risk level summary.

        Args:
            metrics: Dictionary of governance metrics

        Returns:
            Count of metrics at each risk level
        """
        summary = {level.name: 0 for level in RiskLevel}

        for metric in metrics.values():
            summary[metric.risk_level.name] += 1

        return summary


class PermissionAnalyzer:
    """Analyzer for permission and access control metrics."""

    def analyze(self, data: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None,
                policies: Optional[Dict[str, Any]] = None) -> Dict[str, GovernanceMetric]:
        """Analyze permission metrics."""
        results = {}

        # Access control score
        access_score = self._calculate_access_control_score(data, policies)
        results['access_control'] = GovernanceMetric(
            name='access_control',
            score=access_score,
            category=GovernanceCategory.PERMISSION,
            risk_level=self._score_to_risk_level(access_score),
            compliance_status=self._score_to_compliance_status(access_score),
            issues=self._identify_access_issues(data, policies),
            recommendations=self._generate_access_recommendations(access_score)
        )

        # Authorization coverage
        auth_coverage = self._calculate_authorization_coverage(data)
        results['authorization_coverage'] = GovernanceMetric(
            name='authorization_coverage',
            score=auth_coverage,
            category=GovernanceCategory.PERMISSION,
            risk_level=self._score_to_risk_level(auth_coverage)
        )

        # Privilege management
        privilege_score = self._calculate_privilege_management(data)
        results['privilege_management'] = GovernanceMetric(
            name='privilege_management',
            score=privilege_score,
            category=GovernanceCategory.PERMISSION,
            risk_level=self._score_to_risk_level(privilege_score)
        )

        return results

    def _calculate_access_control_score(self, data: Dict[str, Any],
                                         policies: Optional[Dict[str, Any]]) -> float:
        """Calculate access control score."""
        if 'access_control_score' in data:
            return float(data['access_control_score'])

        score = 0.8  # Base score

        if 'unauthorized_access_attempts' in data:
            if data['unauthorized_access_attempts'] > 0:
                score -= min(0.3, data['unauthorized_access_attempts'] * 0.05)

        if 'authentication_enabled' in data and data['authentication_enabled']:
            score += 0.1

        return min(1.0, max(0.0, score))

    def _calculate_authorization_coverage(self, data: Dict[str, Any]) -> float:
        """Calculate authorization coverage."""
        if 'covered_endpoints' in data and 'total_endpoints' in data:
            return data['covered_endpoints'] / data['total_endpoints'] if data['total_endpoints'] > 0 else 0.0
        return 0.85

    def _calculate_privilege_management(self, data: Dict[str, Any]) -> float:
        """Calculate privilege management score."""
        if 'least_privilege_compliance' in data:
            return float(data['least_privilege_compliance'])
        return 0.8

    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """Convert score to risk level."""
        if score >= 0.95:
            return RiskLevel.MINIMAL
        elif score >= 0.85:
            return RiskLevel.LOW
        elif score >= 0.7:
            return RiskLevel.MEDIUM
        elif score >= 0.5:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _score_to_compliance_status(self, score: float) -> ComplianceStatus:
        """Convert score to compliance status."""
        if score >= 0.95:
            return ComplianceStatus.FULLY_COMPLIANT
        elif score >= 0.7:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            return ComplianceStatus.NON_COMPLIANT

    def _identify_access_issues(self, data: Dict[str, Any],
                                 policies: Optional[Dict[str, Any]]) -> List[str]:
        """Identify access control issues."""
        issues = []

        if data.get('unauthorized_access_attempts', 0) > 0:
            issues.append("Unauthorized access attempts detected")

        if not data.get('authentication_enabled', True):
            issues.append("Authentication not enabled")

        if not data.get('mfa_enabled', False):
            issues.append("Multi-factor authentication not enabled")

        return issues

    def _generate_access_recommendations(self, score: float) -> List[str]:
        """Generate access control recommendations."""
        recommendations = []

        if score < 0.9:
            recommendations.append("Review and strengthen access control policies")

        if score < 0.8:
            recommendations.append("Implement additional authentication mechanisms")

        if score < 0.7:
            recommendations.append("Conduct access control audit")

        return recommendations


class RiskAssessmentAnalyzer:
    """Analyzer for risk assessment metrics."""

    def analyze(self, data: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None,
                policies: Optional[Dict[str, Any]] = None) -> Dict[str, GovernanceMetric]:
        """Analyze risk metrics."""
        results = {}

        # Overall risk score
        risk_score = self._calculate_risk_score(data)
        results['overall_risk'] = GovernanceMetric(
            name='overall_risk',
            score=1.0 - risk_score,  # Invert: higher score = lower risk
            category=GovernanceCategory.RISK,
            risk_level=self._risk_score_to_level(risk_score),
            issues=self._identify_risks(data),
            recommendations=self._generate_risk_recommendations(risk_score)
        )

        # Vulnerability score
        vuln_score = self._calculate_vulnerability_score(data)
        results['vulnerability'] = GovernanceMetric(
            name='vulnerability',
            score=1.0 - vuln_score,
            category=GovernanceCategory.RISK,
            risk_level=self._risk_score_to_level(vuln_score)
        )

        # Mitigation effectiveness
        mitigation = self._calculate_mitigation_effectiveness(data)
        results['mitigation_effectiveness'] = GovernanceMetric(
            name='mitigation_effectiveness',
            score=mitigation,
            category=GovernanceCategory.RISK,
            risk_level=self._score_to_risk_level(mitigation)
        )

        return results

    def _calculate_risk_score(self, data: Dict[str, Any]) -> float:
        """Calculate overall risk score (0 = no risk, 1 = max risk)."""
        if 'risk_score' in data:
            return float(data['risk_score'])

        risk_factors = []

        if 'security_vulnerabilities' in data:
            risk_factors.append(min(1.0, data['security_vulnerabilities'] * 0.1))

        if 'data_exposure_risk' in data:
            risk_factors.append(data['data_exposure_risk'])

        if 'operational_risk' in data:
            risk_factors.append(data['operational_risk'])

        if risk_factors:
            return float(np.mean(risk_factors))

        return 0.3  # Default moderate risk

    def _calculate_vulnerability_score(self, data: Dict[str, Any]) -> float:
        """Calculate vulnerability score."""
        if 'vulnerability_score' in data:
            return float(data['vulnerability_score'])

        if 'vulnerabilities' in data:
            vulns = data['vulnerabilities']
            critical = vulns.get('critical', 0) * 0.4
            high = vulns.get('high', 0) * 0.3
            medium = vulns.get('medium', 0) * 0.2
            low = vulns.get('low', 0) * 0.1
            return min(1.0, critical + high + medium + low)

        return 0.2

    def _calculate_mitigation_effectiveness(self, data: Dict[str, Any]) -> float:
        """Calculate mitigation effectiveness."""
        if 'mitigation_effectiveness' in data:
            return float(data['mitigation_effectiveness'])

        if 'mitigated_risks' in data and 'total_risks' in data:
            return data['mitigated_risks'] / data['total_risks'] if data['total_risks'] > 0 else 1.0

        return 0.75

    def _risk_score_to_level(self, risk_score: float) -> RiskLevel:
        """Convert risk score to risk level."""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.4:
            return RiskLevel.MEDIUM
        elif risk_score >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL

    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """Convert effectiveness score to risk level."""
        if score >= 0.9:
            return RiskLevel.MINIMAL
        elif score >= 0.7:
            return RiskLevel.LOW
        elif score >= 0.5:
            return RiskLevel.MEDIUM
        elif score >= 0.3:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _identify_risks(self, data: Dict[str, Any]) -> List[str]:
        """Identify current risks."""
        risks = []

        if data.get('security_vulnerabilities', 0) > 0:
            risks.append(f"Security vulnerabilities detected: {data['security_vulnerabilities']}")

        if data.get('data_exposure_risk', 0) > 0.5:
            risks.append("High data exposure risk")

        if not data.get('backup_enabled', False):
            risks.append("Backup not enabled")

        if not data.get('monitoring_active', False):
            risks.append("Active monitoring not in place")

        return risks

    def _generate_risk_recommendations(self, risk_score: float) -> List[str]:
        """Generate risk mitigation recommendations."""
        recommendations = []

        if risk_score > 0.6:
            recommendations.append("Conduct comprehensive security assessment")
            recommendations.append("Implement additional security controls")

        if risk_score > 0.4:
            recommendations.append("Review and update risk mitigation strategies")

        if risk_score > 0.2:
            recommendations.append("Monitor risk indicators regularly")

        return recommendations


class ComplianceScoreAnalyzer:
    """Analyzer for compliance metrics."""

    def analyze(self, data: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None,
                policies: Optional[Dict[str, Any]] = None) -> Dict[str, GovernanceMetric]:
        """Analyze compliance metrics."""
        results = {}

        # Regulatory compliance
        reg_compliance = self._calculate_regulatory_compliance(data, policies)
        results['regulatory_compliance'] = GovernanceMetric(
            name='regulatory_compliance',
            score=reg_compliance,
            category=GovernanceCategory.COMPLIANCE,
            compliance_status=self._score_to_compliance_status(reg_compliance),
            issues=self._identify_compliance_issues(data, policies),
            recommendations=self._generate_compliance_recommendations(reg_compliance)
        )

        # Policy compliance
        policy_compliance = self._calculate_policy_compliance(data, policies)
        results['policy_compliance'] = GovernanceMetric(
            name='policy_compliance',
            score=policy_compliance,
            category=GovernanceCategory.COMPLIANCE,
            compliance_status=self._score_to_compliance_status(policy_compliance)
        )

        # Audit readiness
        audit_readiness = self._calculate_audit_readiness(data)
        results['audit_readiness'] = GovernanceMetric(
            name='audit_readiness',
            score=audit_readiness,
            category=GovernanceCategory.COMPLIANCE,
            compliance_status=self._score_to_compliance_status(audit_readiness)
        )

        return results

    def _calculate_regulatory_compliance(self, data: Dict[str, Any],
                                          policies: Optional[Dict[str, Any]]) -> float:
        """Calculate regulatory compliance score."""
        if 'regulatory_compliance' in data:
            return float(data['regulatory_compliance'])

        compliance_scores = []

        # GDPR compliance
        if 'gdpr_compliant' in data:
            compliance_scores.append(1.0 if data['gdpr_compliant'] else 0.0)

        # Industry standards
        if 'iso_certified' in data:
            compliance_scores.append(1.0 if data['iso_certified'] else 0.5)

        # Data protection
        if 'data_protection_score' in data:
            compliance_scores.append(data['data_protection_score'])

        if compliance_scores:
            return float(np.mean(compliance_scores))

        return 0.8

    def _calculate_policy_compliance(self, data: Dict[str, Any],
                                      policies: Optional[Dict[str, Any]]) -> float:
        """Calculate policy compliance score."""
        if 'policy_compliance' in data:
            return float(data['policy_compliance'])

        if 'policy_violations' in data and 'total_policies' in data:
            return 1.0 - (data['policy_violations'] / data['total_policies']) if data['total_policies'] > 0 else 1.0

        return 0.85

    def _calculate_audit_readiness(self, data: Dict[str, Any]) -> float:
        """Calculate audit readiness score."""
        readiness_factors = []

        if 'documentation_complete' in data:
            readiness_factors.append(1.0 if data['documentation_complete'] else 0.5)

        if 'logging_enabled' in data:
            readiness_factors.append(1.0 if data['logging_enabled'] else 0.0)

        if 'audit_trail_complete' in data:
            readiness_factors.append(1.0 if data['audit_trail_complete'] else 0.3)

        if readiness_factors:
            return float(np.mean(readiness_factors))

        return 0.7

    def _score_to_compliance_status(self, score: float) -> ComplianceStatus:
        """Convert score to compliance status."""
        if score >= 0.95:
            return ComplianceStatus.FULLY_COMPLIANT
        elif score >= 0.7:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            return ComplianceStatus.NON_COMPLIANT

    def _identify_compliance_issues(self, data: Dict[str, Any],
                                     policies: Optional[Dict[str, Any]]) -> List[str]:
        """Identify compliance issues."""
        issues = []

        if not data.get('gdpr_compliant', True):
            issues.append("GDPR compliance requirements not met")

        if data.get('policy_violations', 0) > 0:
            issues.append(f"Policy violations detected: {data['policy_violations']}")

        if not data.get('documentation_complete', True):
            issues.append("Documentation incomplete")

        return issues

    def _generate_compliance_recommendations(self, score: float) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []

        if score < 0.9:
            recommendations.append("Review compliance requirements")

        if score < 0.7:
            recommendations.append("Implement compliance remediation plan")
            recommendations.append("Conduct compliance training")

        return recommendations


class AccountabilityAnalyzer:
    """Analyzer for accountability metrics."""

    def analyze(self, data: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None,
                policies: Optional[Dict[str, Any]] = None) -> Dict[str, GovernanceMetric]:
        """Analyze accountability metrics."""
        results = {}

        # Audit trail completeness
        audit_trail = self._calculate_audit_trail_score(data)
        results['audit_trail'] = GovernanceMetric(
            name='audit_trail',
            score=audit_trail,
            category=GovernanceCategory.ACCOUNTABILITY,
            risk_level=self._score_to_risk_level(audit_trail)
        )

        # Decision traceability
        traceability = self._calculate_traceability(data)
        results['decision_traceability'] = GovernanceMetric(
            name='decision_traceability',
            score=traceability,
            category=GovernanceCategory.ACCOUNTABILITY,
            risk_level=self._score_to_risk_level(traceability)
        )

        # Responsibility assignment
        responsibility = self._calculate_responsibility_score(data)
        results['responsibility_assignment'] = GovernanceMetric(
            name='responsibility_assignment',
            score=responsibility,
            category=GovernanceCategory.ACCOUNTABILITY,
            risk_level=self._score_to_risk_level(responsibility)
        )

        return results

    def _calculate_audit_trail_score(self, data: Dict[str, Any]) -> float:
        """Calculate audit trail completeness."""
        if 'audit_trail_score' in data:
            return float(data['audit_trail_score'])

        if 'logged_actions' in data and 'total_actions' in data:
            return data['logged_actions'] / data['total_actions'] if data['total_actions'] > 0 else 0.0

        return 0.85

    def _calculate_traceability(self, data: Dict[str, Any]) -> float:
        """Calculate decision traceability."""
        if 'traceability_score' in data:
            return float(data['traceability_score'])

        if 'traceable_decisions' in data and 'total_decisions' in data:
            return data['traceable_decisions'] / data['total_decisions'] if data['total_decisions'] > 0 else 0.0

        return 0.8

    def _calculate_responsibility_score(self, data: Dict[str, Any]) -> float:
        """Calculate responsibility assignment score."""
        if 'responsibility_score' in data:
            return float(data['responsibility_score'])

        if 'assigned_owners' in data and 'total_components' in data:
            return data['assigned_owners'] / data['total_components'] if data['total_components'] > 0 else 0.0

        return 0.9

    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """Convert score to risk level."""
        if score >= 0.9:
            return RiskLevel.MINIMAL
        elif score >= 0.7:
            return RiskLevel.LOW
        elif score >= 0.5:
            return RiskLevel.MEDIUM
        elif score >= 0.3:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL


class TransparencyAnalyzer:
    """Analyzer for transparency metrics."""

    def analyze(self, data: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None,
                policies: Optional[Dict[str, Any]] = None) -> Dict[str, GovernanceMetric]:
        """Analyze transparency metrics."""
        results = {}

        # Explainability score
        explainability = self._calculate_explainability(data)
        results['explainability'] = GovernanceMetric(
            name='explainability',
            score=explainability,
            category=GovernanceCategory.TRANSPARENCY,
            risk_level=self._score_to_risk_level(explainability)
        )

        # Documentation coverage
        documentation = self._calculate_documentation_coverage(data)
        results['documentation_coverage'] = GovernanceMetric(
            name='documentation_coverage',
            score=documentation,
            category=GovernanceCategory.TRANSPARENCY,
            risk_level=self._score_to_risk_level(documentation)
        )

        # Disclosure completeness
        disclosure = self._calculate_disclosure_score(data)
        results['disclosure_completeness'] = GovernanceMetric(
            name='disclosure_completeness',
            score=disclosure,
            category=GovernanceCategory.TRANSPARENCY,
            risk_level=self._score_to_risk_level(disclosure)
        )

        return results

    def _calculate_explainability(self, data: Dict[str, Any]) -> float:
        """Calculate explainability score."""
        if 'explainability_score' in data:
            return float(data['explainability_score'])

        factors = []

        if 'model_interpretable' in data:
            factors.append(1.0 if data['model_interpretable'] else 0.5)

        if 'explanations_available' in data:
            factors.append(1.0 if data['explanations_available'] else 0.0)

        if factors:
            return float(np.mean(factors))

        return 0.7

    def _calculate_documentation_coverage(self, data: Dict[str, Any]) -> float:
        """Calculate documentation coverage."""
        if 'documentation_coverage' in data:
            return float(data['documentation_coverage'])

        if 'documented_components' in data and 'total_components' in data:
            return data['documented_components'] / data['total_components'] if data['total_components'] > 0 else 0.0

        return 0.8

    def _calculate_disclosure_score(self, data: Dict[str, Any]) -> float:
        """Calculate disclosure completeness."""
        if 'disclosure_score' in data:
            return float(data['disclosure_score'])

        disclosures = ['data_sources_disclosed', 'model_limitations_disclosed',
                      'usage_policies_published']

        score = sum(1 for d in disclosures if data.get(d, False)) / len(disclosures)
        return score

    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """Convert score to risk level."""
        if score >= 0.9:
            return RiskLevel.MINIMAL
        elif score >= 0.7:
            return RiskLevel.LOW
        elif score >= 0.5:
            return RiskLevel.MEDIUM
        elif score >= 0.3:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL


class IntegratedMatrixAnalyzer:
    """
    Integrated analyzer combining Performance and Governance matrices.

    Provides unified analysis across both matrices with weighted
    aggregation and comprehensive reporting.
    """

    def __init__(self,
                 performance_weight: float = 0.5,
                 governance_weight: float = 0.5,
                 thresholds: Optional[Dict[str, float]] = None,
                 policies: Optional[Dict[str, Any]] = None):
        """
        Initialize integrated matrix analyzer.

        Args:
            performance_weight: Weight for performance matrix
            governance_weight: Weight for governance matrix
            thresholds: Performance thresholds
            policies: Governance policies
        """
        self.performance_weight = performance_weight
        self.governance_weight = governance_weight

        self.performance_analyzer = PerformanceMatrixAnalyzer(thresholds)
        self.governance_analyzer = GovernanceMatrixAnalyzer(policies)

    def analyze(self, system_data: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None) -> MatrixResult:
        """
        Perform integrated matrix analysis.

        Args:
            system_data: Combined system data
            context: Optional context information

        Returns:
            MatrixResult with performance and governance scores
        """
        result = MatrixResult()

        # Performance analysis
        perf_metrics = self.performance_analyzer.analyze(system_data, context)
        result.performance_scores = perf_metrics
        result.overall_performance = self.performance_analyzer.calculate_aggregate(perf_metrics)

        # Governance analysis
        gov_metrics = self.governance_analyzer.analyze(system_data, context)
        result.governance_scores = gov_metrics
        result.overall_governance = self.governance_analyzer.calculate_aggregate(gov_metrics)

        # Combined score
        result.combined_score = (
            result.overall_performance * self.performance_weight +
            result.overall_governance * self.governance_weight
        )

        # Risk summary
        result.risk_summary = self.governance_analyzer.get_risk_summary(gov_metrics)

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        return result

    def _generate_recommendations(self, result: MatrixResult) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []

        # Performance recommendations
        if result.overall_performance < 0.7:
            recommendations.append("Prioritize performance optimization")

            for name, metric in result.performance_scores.items():
                if not metric.meets_threshold:
                    recommendations.append(f"Improve {name}: current {metric.value:.2f}, target {metric.threshold:.2f}")

        # Governance recommendations
        if result.overall_governance < 0.8:
            recommendations.append("Strengthen governance controls")

        # Risk-based recommendations
        if result.risk_summary.get('CRITICAL', 0) > 0:
            recommendations.append("Address critical risk items immediately")

        if result.risk_summary.get('HIGH', 0) > 0:
            recommendations.append("Develop mitigation plans for high-risk items")

        # Balance recommendations
        perf_gov_gap = abs(result.overall_performance - result.overall_governance)
        if perf_gov_gap > 0.2:
            if result.overall_performance > result.overall_governance:
                recommendations.append("Governance lagging behind performance - focus on compliance and risk")
            else:
                recommendations.append("Performance lagging behind governance - focus on optimization")

        return recommendations

    def generate_report(self, result: MatrixResult) -> str:
        """
        Generate comprehensive matrix report.

        Args:
            result: MatrixResult from analysis

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 70,
            "INTEGRATED PERFORMANCE & GOVERNANCE MATRIX REPORT",
            "=" * 70,
            "",
            "SUMMARY SCORES",
            "-" * 40,
            f"Overall Performance:  {result.overall_performance:.3f}",
            f"Overall Governance:   {result.overall_governance:.3f}",
            f"Combined Score:       {result.combined_score:.3f}",
            "",
            "PERFORMANCE METRICS",
            "-" * 40
        ]

        for name, metric in result.performance_scores.items():
            status_icon = "✓" if metric.meets_threshold else "✗"
            lines.append(f"  {status_icon} {name}: {metric.value:.3f} {metric.unit}")

        lines.extend([
            "",
            "GOVERNANCE METRICS",
            "-" * 40
        ])

        for name, metric in result.governance_scores.items():
            risk_icon = {"CRITICAL": "⚠", "HIGH": "!", "MEDIUM": "~", "LOW": "-", "MINIMAL": "✓"}
            icon = risk_icon.get(metric.risk_level.name, "?")
            lines.append(f"  {icon} {name}: {metric.score:.3f} (Risk: {metric.risk_level.name})")

        lines.extend([
            "",
            "RISK SUMMARY",
            "-" * 40
        ])

        for level, count in result.risk_summary.items():
            if count > 0:
                lines.append(f"  {level}: {count}")

        if result.recommendations:
            lines.extend([
                "",
                "RECOMMENDATIONS",
                "-" * 40
            ])
            for rec in result.recommendations:
                lines.append(f"  • {rec}")

        lines.append("=" * 70)

        return "\n".join(lines)

    def compare_systems(self, system1_data: Dict[str, Any],
                        system2_data: Dict[str, Any],
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compare two systems using integrated matrices.

        Args:
            system1_data: First system data
            system2_data: Second system data
            context: Shared context

        Returns:
            Comparison results
        """
        result1 = self.analyze(system1_data, context)
        result2 = self.analyze(system2_data, context)

        comparison = {
            'performance_comparison': {
                'system1': result1.overall_performance,
                'system2': result2.overall_performance,
                'difference': result1.overall_performance - result2.overall_performance,
                'winner': 'system1' if result1.overall_performance > result2.overall_performance else 'system2'
            },
            'governance_comparison': {
                'system1': result1.overall_governance,
                'system2': result2.overall_governance,
                'difference': result1.overall_governance - result2.overall_governance,
                'winner': 'system1' if result1.overall_governance > result2.overall_governance else 'system2'
            },
            'overall_comparison': {
                'system1': result1.combined_score,
                'system2': result2.combined_score,
                'difference': result1.combined_score - result2.combined_score,
                'winner': 'system1' if result1.combined_score > result2.combined_score else 'system2'
            },
            'risk_comparison': {
                'system1_critical': result1.risk_summary.get('CRITICAL', 0),
                'system2_critical': result2.risk_summary.get('CRITICAL', 0),
                'system1_high': result1.risk_summary.get('HIGH', 0),
                'system2_high': result2.risk_summary.get('HIGH', 0)
            }
        }

        return comparison


# Convenience functions
def analyze_performance(data: Dict[str, Any],
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, PerformanceMetric]:
    """Quick performance analysis."""
    analyzer = PerformanceMatrixAnalyzer()
    return analyzer.analyze(data, context)


def analyze_governance(data: Dict[str, Any],
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, GovernanceMetric]:
    """Quick governance analysis."""
    analyzer = GovernanceMatrixAnalyzer()
    return analyzer.analyze(data, context)


def analyze_integrated(data: Dict[str, Any],
                       context: Optional[Dict[str, Any]] = None) -> MatrixResult:
    """Quick integrated analysis."""
    analyzer = IntegratedMatrixAnalyzer()
    return analyzer.analyze(data, context)


__all__ = [
    # Enums
    'PerformanceCategory',
    'GovernanceCategory',
    'RiskLevel',
    'ComplianceStatus',

    # Data classes
    'PerformanceMetric',
    'GovernanceMetric',
    'MatrixResult',

    # Performance Analyzers
    'PerformanceMatrixAnalyzer',
    'ExecutionPerformanceAnalyzer',
    'EfficiencyAnalyzer',
    'OptimizationAnalyzer',
    'ReliabilityAnalyzer',

    # Governance Analyzers
    'GovernanceMatrixAnalyzer',
    'PermissionAnalyzer',
    'RiskAssessmentAnalyzer',
    'ComplianceScoreAnalyzer',
    'AccountabilityAnalyzer',
    'TransparencyAnalyzer',

    # Integrated Analyzer
    'IntegratedMatrixAnalyzer',

    # Convenience functions
    'analyze_performance',
    'analyze_governance',
    'analyze_integrated',
]
