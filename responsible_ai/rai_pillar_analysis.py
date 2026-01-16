"""
RAI Pillar Analysis Module
==========================

Unified framework for analyzing the Five Pillars of Responsible AI:
1. Privacy - Data protection, PII handling, de-identification
2. Transparency - Explainability, interpretability, user notification
3. Robustness - Model stability, adversarial resilience, reliability
4. Safety - Content safety, harm prevention, risk management
5. Accountability - Audit trails, compliance, governance

This module provides comprehensive analysis across all five pillars
with integrated scoring and cross-pillar dependency analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import re
from collections import defaultdict


# =============================================================================
# Enums
# =============================================================================

class PillarType(Enum):
    """Five pillars of Responsible AI."""
    PRIVACY = "privacy"
    TRANSPARENCY = "transparency"
    ROBUSTNESS = "robustness"
    SAFETY = "safety"
    ACCOUNTABILITY = "accountability"


class ComplianceLevel(Enum):
    """Compliance level for pillar assessment."""
    FULL = "full"
    PARTIAL = "partial"
    MINIMAL = "minimal"
    NON_COMPLIANT = "non_compliant"


class MaturityLevel(Enum):
    """Maturity level for pillar implementation."""
    INITIAL = "initial"
    DEVELOPING = "developing"
    DEFINED = "defined"
    MANAGED = "managed"
    OPTIMIZED = "optimized"


class RiskCategory(Enum):
    """Risk categories for RAI assessment."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DataSensitivity(Enum):
    """Data sensitivity levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class ControlType(Enum):
    """Types of controls."""
    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"
    COMPENSATING = "compensating"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PillarScore:
    """Score for a single RAI pillar."""
    pillar: PillarType
    score: float  # 0-1
    compliance_level: ComplianceLevel
    maturity_level: MaturityLevel
    sub_scores: Dict[str, float] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risk_level: RiskCategory = RiskCategory.LOW


@dataclass
class PrivacyMetrics:
    """Metrics for Privacy pillar analysis."""
    pii_detection_rate: float = 0.0
    de_identification_coverage: float = 0.0
    consent_compliance: float = 0.0
    data_minimization_score: float = 0.0
    retention_compliance: float = 0.0
    access_control_score: float = 0.0
    encryption_coverage: float = 0.0
    anonymization_quality: float = 0.0
    cross_border_compliance: float = 0.0
    breach_risk_score: float = 0.0


@dataclass
class TransparencyMetrics:
    """Metrics for Transparency pillar analysis."""
    explainability_score: float = 0.0
    interpretability_score: float = 0.0
    documentation_completeness: float = 0.0
    user_notification_coverage: float = 0.0
    decision_traceability: float = 0.0
    model_card_completeness: float = 0.0
    feature_importance_clarity: float = 0.0
    audit_trail_coverage: float = 0.0
    disclosure_compliance: float = 0.0
    communication_clarity: float = 0.0


@dataclass
class RobustnessMetrics:
    """Metrics for Robustness pillar analysis."""
    model_stability_score: float = 0.0
    adversarial_resilience: float = 0.0
    distribution_shift_tolerance: float = 0.0
    edge_case_handling: float = 0.0
    error_recovery_rate: float = 0.0
    input_validation_coverage: float = 0.0
    output_bounds_compliance: float = 0.0
    performance_consistency: float = 0.0
    degradation_graceful: float = 0.0
    uncertainty_quantification: float = 0.0


@dataclass
class SafetyMetrics:
    """Metrics for Safety pillar analysis."""
    harm_prevention_score: float = 0.0
    content_safety_score: float = 0.0
    bias_mitigation_coverage: float = 0.0
    toxicity_detection_rate: float = 0.0
    guardrail_effectiveness: float = 0.0
    fail_safe_coverage: float = 0.0
    human_oversight_score: float = 0.0
    risk_assessment_quality: float = 0.0
    incident_response_readiness: float = 0.0
    safety_testing_coverage: float = 0.0


@dataclass
class AccountabilityMetrics:
    """Metrics for Accountability pillar analysis."""
    audit_trail_completeness: float = 0.0
    governance_structure_score: float = 0.0
    role_clarity_score: float = 0.0
    decision_documentation: float = 0.0
    compliance_monitoring: float = 0.0
    remediation_process_score: float = 0.0
    stakeholder_engagement: float = 0.0
    policy_enforcement: float = 0.0
    incident_tracking: float = 0.0
    continuous_improvement: float = 0.0


@dataclass
class CrossPillarDependency:
    """Dependency between pillars."""
    source_pillar: PillarType
    target_pillar: PillarType
    dependency_type: str
    strength: float  # 0-1
    description: str


@dataclass
class RAIPillarAssessment:
    """Complete RAI Pillar Assessment."""
    pillar_scores: Dict[PillarType, PillarScore]
    overall_score: float
    overall_compliance: ComplianceLevel
    overall_maturity: MaturityLevel
    cross_pillar_dependencies: List[CrossPillarDependency]
    priority_actions: List[str]
    risk_summary: Dict[RiskCategory, int]
    timestamp: str = ""


# =============================================================================
# Privacy Pillar Analyzer
# =============================================================================

class PrivacyPillarAnalyzer:
    """Analyzer for Privacy pillar of Responsible AI."""

    def __init__(self):
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'date_of_birth': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'name_pattern': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
        }

        self.sensitive_terms = {
            'health': ['diagnosis', 'treatment', 'medical', 'patient', 'prescription'],
            'financial': ['salary', 'income', 'credit', 'debt', 'account'],
            'personal': ['address', 'location', 'home', 'residence'],
        }

    def analyze(self, data: Dict[str, Any], config: Optional[Dict] = None) -> PillarScore:
        """Analyze privacy compliance."""
        config = config or {}

        metrics = self._calculate_metrics(data)
        score = self._compute_score(metrics)
        compliance = self._determine_compliance(score)
        maturity = self._assess_maturity(data, metrics)
        findings = self._generate_findings(metrics, data)
        recommendations = self._generate_recommendations(metrics)
        risk_level = self._assess_risk(metrics)

        return PillarScore(
            pillar=PillarType.PRIVACY,
            score=score,
            compliance_level=compliance,
            maturity_level=maturity,
            sub_scores={
                'pii_protection': metrics.pii_detection_rate,
                'de_identification': metrics.de_identification_coverage,
                'consent': metrics.consent_compliance,
                'data_minimization': metrics.data_minimization_score,
                'retention': metrics.retention_compliance,
                'access_control': metrics.access_control_score,
                'encryption': metrics.encryption_coverage,
                'anonymization': metrics.anonymization_quality,
            },
            findings=findings,
            recommendations=recommendations,
            risk_level=risk_level
        )

    def _calculate_metrics(self, data: Dict[str, Any]) -> PrivacyMetrics:
        """Calculate privacy metrics."""
        text_content = data.get('text_content', '')
        policies = data.get('privacy_policies', {})
        controls = data.get('privacy_controls', {})

        # PII detection
        pii_count = self._detect_pii(text_content)
        pii_detection_rate = 1.0 if pii_count == 0 else max(0, 1.0 - pii_count * 0.1)

        # De-identification coverage
        de_id_methods = controls.get('de_identification_methods', [])
        de_identification_coverage = min(1.0, len(de_id_methods) * 0.2)

        # Consent compliance
        consent_mechanisms = policies.get('consent_mechanisms', [])
        consent_compliance = min(1.0, len(consent_mechanisms) * 0.25)

        # Data minimization
        data_collected = data.get('data_fields_collected', [])
        data_necessary = data.get('data_fields_necessary', [])
        if data_collected:
            data_minimization = len(data_necessary) / len(data_collected) if data_collected else 1.0
        else:
            data_minimization = 1.0

        # Retention compliance
        retention_policy = policies.get('retention_policy', {})
        retention_compliance = 1.0 if retention_policy.get('defined', False) else 0.5

        # Access control
        access_controls = controls.get('access_controls', [])
        access_control_score = min(1.0, len(access_controls) * 0.2)

        # Encryption
        encryption_methods = controls.get('encryption', {})
        encryption_coverage = 0.5 if encryption_methods.get('at_rest') else 0.0
        encryption_coverage += 0.5 if encryption_methods.get('in_transit') else 0.0

        return PrivacyMetrics(
            pii_detection_rate=pii_detection_rate,
            de_identification_coverage=de_identification_coverage,
            consent_compliance=consent_compliance,
            data_minimization_score=data_minimization,
            retention_compliance=retention_compliance,
            access_control_score=access_control_score,
            encryption_coverage=encryption_coverage,
            anonymization_quality=de_identification_coverage * 0.8,
        )

    def _detect_pii(self, text: str) -> int:
        """Detect PII in text."""
        pii_count = 0
        for pattern_name, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            pii_count += len(matches)
        return pii_count

    def _compute_score(self, metrics: PrivacyMetrics) -> float:
        """Compute overall privacy score."""
        weights = {
            'pii_detection_rate': 0.15,
            'de_identification_coverage': 0.15,
            'consent_compliance': 0.15,
            'data_minimization_score': 0.10,
            'retention_compliance': 0.10,
            'access_control_score': 0.15,
            'encryption_coverage': 0.15,
            'anonymization_quality': 0.05,
        }

        score = sum(
            getattr(metrics, key) * weight
            for key, weight in weights.items()
        )
        return min(1.0, max(0.0, score))

    def _determine_compliance(self, score: float) -> ComplianceLevel:
        """Determine compliance level from score."""
        if score >= 0.9:
            return ComplianceLevel.FULL
        elif score >= 0.7:
            return ComplianceLevel.PARTIAL
        elif score >= 0.5:
            return ComplianceLevel.MINIMAL
        return ComplianceLevel.NON_COMPLIANT

    def _assess_maturity(self, data: Dict, metrics: PrivacyMetrics) -> MaturityLevel:
        """Assess privacy maturity level."""
        maturity_indicators = 0

        if metrics.pii_detection_rate > 0.8:
            maturity_indicators += 1
        if metrics.encryption_coverage > 0.8:
            maturity_indicators += 1
        if data.get('privacy_by_design', False):
            maturity_indicators += 1
        if data.get('privacy_impact_assessment', False):
            maturity_indicators += 1
        if data.get('continuous_monitoring', False):
            maturity_indicators += 1

        if maturity_indicators >= 5:
            return MaturityLevel.OPTIMIZED
        elif maturity_indicators >= 4:
            return MaturityLevel.MANAGED
        elif maturity_indicators >= 3:
            return MaturityLevel.DEFINED
        elif maturity_indicators >= 2:
            return MaturityLevel.DEVELOPING
        return MaturityLevel.INITIAL

    def _generate_findings(self, metrics: PrivacyMetrics, data: Dict) -> List[str]:
        """Generate privacy findings."""
        findings = []

        if metrics.pii_detection_rate < 0.8:
            findings.append("PII protection requires improvement")
        if metrics.encryption_coverage < 0.8:
            findings.append("Encryption coverage is incomplete")
        if metrics.consent_compliance < 0.7:
            findings.append("Consent mechanisms need strengthening")
        if metrics.de_identification_coverage < 0.6:
            findings.append("De-identification methods are limited")

        return findings

    def _generate_recommendations(self, metrics: PrivacyMetrics) -> List[str]:
        """Generate privacy recommendations."""
        recommendations = []

        if metrics.pii_detection_rate < 0.9:
            recommendations.append("Implement automated PII scanning")
        if metrics.encryption_coverage < 1.0:
            recommendations.append("Enable encryption for data at rest and in transit")
        if metrics.de_identification_coverage < 0.8:
            recommendations.append("Expand de-identification methods")
        if metrics.access_control_score < 0.8:
            recommendations.append("Strengthen access control mechanisms")

        return recommendations

    def _assess_risk(self, metrics: PrivacyMetrics) -> RiskCategory:
        """Assess privacy risk level."""
        avg_score = (
            metrics.pii_detection_rate +
            metrics.encryption_coverage +
            metrics.access_control_score
        ) / 3

        if avg_score >= 0.8:
            return RiskCategory.LOW
        elif avg_score >= 0.6:
            return RiskCategory.MEDIUM
        elif avg_score >= 0.4:
            return RiskCategory.HIGH
        return RiskCategory.CRITICAL


# =============================================================================
# Transparency Pillar Analyzer
# =============================================================================

class TransparencyPillarAnalyzer:
    """Analyzer for Transparency pillar of Responsible AI."""

    def __init__(self):
        self.explanation_types = [
            'feature_importance', 'decision_path', 'counterfactual',
            'example_based', 'rule_based', 'attention_based'
        ]

    def analyze(self, data: Dict[str, Any], config: Optional[Dict] = None) -> PillarScore:
        """Analyze transparency compliance."""
        config = config or {}

        metrics = self._calculate_metrics(data)
        score = self._compute_score(metrics)
        compliance = self._determine_compliance(score)
        maturity = self._assess_maturity(data, metrics)
        findings = self._generate_findings(metrics)
        recommendations = self._generate_recommendations(metrics)
        risk_level = self._assess_risk(metrics)

        return PillarScore(
            pillar=PillarType.TRANSPARENCY,
            score=score,
            compliance_level=compliance,
            maturity_level=maturity,
            sub_scores={
                'explainability': metrics.explainability_score,
                'interpretability': metrics.interpretability_score,
                'documentation': metrics.documentation_completeness,
                'user_notification': metrics.user_notification_coverage,
                'traceability': metrics.decision_traceability,
                'model_card': metrics.model_card_completeness,
                'audit_trail': metrics.audit_trail_coverage,
            },
            findings=findings,
            recommendations=recommendations,
            risk_level=risk_level
        )

    def _calculate_metrics(self, data: Dict[str, Any]) -> TransparencyMetrics:
        """Calculate transparency metrics."""
        explanations = data.get('explanations', {})
        documentation = data.get('documentation', {})
        notifications = data.get('user_notifications', {})
        audit = data.get('audit_config', {})

        # Explainability
        explanation_methods = explanations.get('methods', [])
        explainability_score = min(1.0, len(explanation_methods) / len(self.explanation_types))

        # Interpretability
        model_type = data.get('model_type', 'black_box')
        interpretability_map = {
            'linear': 1.0, 'tree': 0.9, 'rule_based': 0.85,
            'ensemble': 0.6, 'neural': 0.4, 'black_box': 0.2
        }
        interpretability_score = interpretability_map.get(model_type, 0.3)

        # Documentation
        doc_sections = documentation.get('sections', [])
        required_sections = ['purpose', 'methodology', 'limitations', 'data', 'performance']
        doc_coverage = sum(1 for s in required_sections if s in doc_sections)
        documentation_completeness = doc_coverage / len(required_sections)

        # User notifications
        notification_types = notifications.get('types', [])
        required_notifications = ['ai_usage', 'data_collection', 'decision_basis', 'opt_out']
        notification_coverage = sum(1 for n in required_notifications if n in notification_types)
        user_notification_coverage = notification_coverage / len(required_notifications)

        # Decision traceability
        traceability_enabled = audit.get('decision_logging', False)
        traceability_completeness = audit.get('completeness', 0.5)
        decision_traceability = 1.0 if traceability_enabled else 0.0
        decision_traceability *= traceability_completeness

        # Model card
        model_card = data.get('model_card', {})
        model_card_fields = ['intended_use', 'limitations', 'ethical_considerations', 'metrics']
        card_coverage = sum(1 for f in model_card_fields if f in model_card)
        model_card_completeness = card_coverage / len(model_card_fields)

        # Audit trail
        audit_trail_coverage = 0.0
        if audit.get('enabled', False):
            audit_trail_coverage = 0.5
            if audit.get('immutable', False):
                audit_trail_coverage += 0.3
            if audit.get('searchable', False):
                audit_trail_coverage += 0.2

        return TransparencyMetrics(
            explainability_score=explainability_score,
            interpretability_score=interpretability_score,
            documentation_completeness=documentation_completeness,
            user_notification_coverage=user_notification_coverage,
            decision_traceability=decision_traceability,
            model_card_completeness=model_card_completeness,
            audit_trail_coverage=audit_trail_coverage,
            disclosure_compliance=user_notification_coverage * 0.9,
            communication_clarity=documentation_completeness * 0.8,
        )

    def _compute_score(self, metrics: TransparencyMetrics) -> float:
        """Compute overall transparency score."""
        weights = {
            'explainability_score': 0.20,
            'interpretability_score': 0.15,
            'documentation_completeness': 0.15,
            'user_notification_coverage': 0.15,
            'decision_traceability': 0.15,
            'model_card_completeness': 0.10,
            'audit_trail_coverage': 0.10,
        }

        score = sum(
            getattr(metrics, key) * weight
            for key, weight in weights.items()
        )
        return min(1.0, max(0.0, score))

    def _determine_compliance(self, score: float) -> ComplianceLevel:
        """Determine compliance level."""
        if score >= 0.9:
            return ComplianceLevel.FULL
        elif score >= 0.7:
            return ComplianceLevel.PARTIAL
        elif score >= 0.5:
            return ComplianceLevel.MINIMAL
        return ComplianceLevel.NON_COMPLIANT

    def _assess_maturity(self, data: Dict, metrics: TransparencyMetrics) -> MaturityLevel:
        """Assess transparency maturity."""
        indicators = 0

        if metrics.explainability_score > 0.7:
            indicators += 1
        if metrics.documentation_completeness > 0.8:
            indicators += 1
        if metrics.audit_trail_coverage > 0.7:
            indicators += 1
        if data.get('transparency_by_design', False):
            indicators += 1
        if data.get('regular_audits', False):
            indicators += 1

        if indicators >= 5:
            return MaturityLevel.OPTIMIZED
        elif indicators >= 4:
            return MaturityLevel.MANAGED
        elif indicators >= 3:
            return MaturityLevel.DEFINED
        elif indicators >= 2:
            return MaturityLevel.DEVELOPING
        return MaturityLevel.INITIAL

    def _generate_findings(self, metrics: TransparencyMetrics) -> List[str]:
        """Generate transparency findings."""
        findings = []

        if metrics.explainability_score < 0.6:
            findings.append("Model explanations are limited")
        if metrics.documentation_completeness < 0.7:
            findings.append("Documentation is incomplete")
        if metrics.user_notification_coverage < 0.8:
            findings.append("User notifications need improvement")
        if metrics.audit_trail_coverage < 0.6:
            findings.append("Audit trail coverage is insufficient")

        return findings

    def _generate_recommendations(self, metrics: TransparencyMetrics) -> List[str]:
        """Generate transparency recommendations."""
        recommendations = []

        if metrics.explainability_score < 0.8:
            recommendations.append("Implement additional explanation methods (SHAP, LIME)")
        if metrics.documentation_completeness < 0.9:
            recommendations.append("Complete model documentation with all required sections")
        if metrics.model_card_completeness < 0.8:
            recommendations.append("Create comprehensive model cards")
        if metrics.audit_trail_coverage < 0.8:
            recommendations.append("Enable immutable audit trails")

        return recommendations

    def _assess_risk(self, metrics: TransparencyMetrics) -> RiskCategory:
        """Assess transparency risk."""
        avg = (metrics.explainability_score + metrics.audit_trail_coverage) / 2

        if avg >= 0.8:
            return RiskCategory.LOW
        elif avg >= 0.6:
            return RiskCategory.MEDIUM
        elif avg >= 0.4:
            return RiskCategory.HIGH
        return RiskCategory.CRITICAL


# =============================================================================
# Robustness Pillar Analyzer
# =============================================================================

class RobustnessPillarAnalyzer:
    """Analyzer for Robustness pillar of Responsible AI."""

    def __init__(self):
        self.adversarial_tests = [
            'noise_injection', 'input_perturbation', 'feature_occlusion',
            'gradient_attack', 'boundary_testing', 'distribution_shift'
        ]

    def analyze(self, data: Dict[str, Any], config: Optional[Dict] = None) -> PillarScore:
        """Analyze robustness compliance."""
        config = config or {}

        metrics = self._calculate_metrics(data)
        score = self._compute_score(metrics)
        compliance = self._determine_compliance(score)
        maturity = self._assess_maturity(data, metrics)
        findings = self._generate_findings(metrics)
        recommendations = self._generate_recommendations(metrics)
        risk_level = self._assess_risk(metrics)

        return PillarScore(
            pillar=PillarType.ROBUSTNESS,
            score=score,
            compliance_level=compliance,
            maturity_level=maturity,
            sub_scores={
                'stability': metrics.model_stability_score,
                'adversarial_resilience': metrics.adversarial_resilience,
                'distribution_shift': metrics.distribution_shift_tolerance,
                'edge_cases': metrics.edge_case_handling,
                'error_recovery': metrics.error_recovery_rate,
                'input_validation': metrics.input_validation_coverage,
                'uncertainty': metrics.uncertainty_quantification,
            },
            findings=findings,
            recommendations=recommendations,
            risk_level=risk_level
        )

    def _calculate_metrics(self, data: Dict[str, Any]) -> RobustnessMetrics:
        """Calculate robustness metrics."""
        testing = data.get('robustness_testing', {})
        validation = data.get('validation', {})
        performance = data.get('performance_metrics', {})

        # Model stability
        variance = performance.get('prediction_variance', 0.1)
        model_stability_score = max(0, 1.0 - variance * 2)

        # Adversarial resilience
        adversarial_tests_run = testing.get('adversarial_tests', [])
        adversarial_pass_rate = testing.get('adversarial_pass_rate', 0.0)
        test_coverage = len(adversarial_tests_run) / len(self.adversarial_tests)
        adversarial_resilience = test_coverage * 0.5 + adversarial_pass_rate * 0.5

        # Distribution shift tolerance
        ood_detection = testing.get('ood_detection_enabled', False)
        drift_monitoring = testing.get('drift_monitoring', False)
        distribution_shift_tolerance = 0.0
        if ood_detection:
            distribution_shift_tolerance += 0.5
        if drift_monitoring:
            distribution_shift_tolerance += 0.5

        # Edge case handling
        edge_cases_tested = testing.get('edge_cases_tested', 0)
        edge_cases_total = testing.get('edge_cases_identified', 1)
        edge_case_handling = min(1.0, edge_cases_tested / max(1, edge_cases_total))

        # Error recovery
        error_handling = validation.get('error_handling', {})
        recovery_mechanisms = error_handling.get('recovery_mechanisms', [])
        error_recovery_rate = min(1.0, len(recovery_mechanisms) * 0.25)

        # Input validation
        input_checks = validation.get('input_validation', {})
        validation_types = ['type_check', 'range_check', 'format_check', 'schema_validation']
        enabled_validations = sum(1 for v in validation_types if input_checks.get(v, False))
        input_validation_coverage = enabled_validations / len(validation_types)

        # Output bounds
        output_bounds = validation.get('output_bounds', {})
        output_bounds_compliance = 1.0 if output_bounds.get('enforced', False) else 0.5

        # Uncertainty quantification
        uncertainty_config = data.get('uncertainty', {})
        uncertainty_methods = ['confidence_intervals', 'calibration', 'ensemble_variance']
        enabled_uncertainty = sum(1 for m in uncertainty_methods if uncertainty_config.get(m, False))
        uncertainty_quantification = enabled_uncertainty / len(uncertainty_methods)

        return RobustnessMetrics(
            model_stability_score=model_stability_score,
            adversarial_resilience=adversarial_resilience,
            distribution_shift_tolerance=distribution_shift_tolerance,
            edge_case_handling=edge_case_handling,
            error_recovery_rate=error_recovery_rate,
            input_validation_coverage=input_validation_coverage,
            output_bounds_compliance=output_bounds_compliance,
            performance_consistency=model_stability_score * 0.9,
            degradation_graceful=error_recovery_rate * 0.8,
            uncertainty_quantification=uncertainty_quantification
        )

    def _compute_score(self, metrics: RobustnessMetrics) -> float:
        """Compute overall robustness score."""
        weights = {
            'model_stability_score': 0.15,
            'adversarial_resilience': 0.20,
            'distribution_shift_tolerance': 0.15,
            'edge_case_handling': 0.10,
            'error_recovery_rate': 0.10,
            'input_validation_coverage': 0.15,
            'uncertainty_quantification': 0.15,
        }

        score = sum(
            getattr(metrics, key) * weight
            for key, weight in weights.items()
        )
        return min(1.0, max(0.0, score))

    def _determine_compliance(self, score: float) -> ComplianceLevel:
        """Determine compliance level."""
        if score >= 0.9:
            return ComplianceLevel.FULL
        elif score >= 0.7:
            return ComplianceLevel.PARTIAL
        elif score >= 0.5:
            return ComplianceLevel.MINIMAL
        return ComplianceLevel.NON_COMPLIANT

    def _assess_maturity(self, data: Dict, metrics: RobustnessMetrics) -> MaturityLevel:
        """Assess robustness maturity."""
        indicators = 0

        if metrics.adversarial_resilience > 0.7:
            indicators += 1
        if metrics.input_validation_coverage > 0.8:
            indicators += 1
        if metrics.uncertainty_quantification > 0.6:
            indicators += 1
        if data.get('continuous_testing', False):
            indicators += 1
        if data.get('automated_monitoring', False):
            indicators += 1

        if indicators >= 5:
            return MaturityLevel.OPTIMIZED
        elif indicators >= 4:
            return MaturityLevel.MANAGED
        elif indicators >= 3:
            return MaturityLevel.DEFINED
        elif indicators >= 2:
            return MaturityLevel.DEVELOPING
        return MaturityLevel.INITIAL

    def _generate_findings(self, metrics: RobustnessMetrics) -> List[str]:
        """Generate robustness findings."""
        findings = []

        if metrics.adversarial_resilience < 0.6:
            findings.append("Adversarial testing coverage is limited")
        if metrics.input_validation_coverage < 0.7:
            findings.append("Input validation is incomplete")
        if metrics.uncertainty_quantification < 0.5:
            findings.append("Uncertainty quantification is missing")
        if metrics.error_recovery_rate < 0.6:
            findings.append("Error recovery mechanisms need improvement")

        return findings

    def _generate_recommendations(self, metrics: RobustnessMetrics) -> List[str]:
        """Generate robustness recommendations."""
        recommendations = []

        if metrics.adversarial_resilience < 0.8:
            recommendations.append("Expand adversarial testing suite")
        if metrics.input_validation_coverage < 0.9:
            recommendations.append("Implement comprehensive input validation")
        if metrics.uncertainty_quantification < 0.7:
            recommendations.append("Add uncertainty quantification methods")
        if metrics.distribution_shift_tolerance < 0.7:
            recommendations.append("Enable distribution shift detection")

        return recommendations

    def _assess_risk(self, metrics: RobustnessMetrics) -> RiskCategory:
        """Assess robustness risk."""
        avg = (metrics.adversarial_resilience + metrics.input_validation_coverage) / 2

        if avg >= 0.8:
            return RiskCategory.LOW
        elif avg >= 0.6:
            return RiskCategory.MEDIUM
        elif avg >= 0.4:
            return RiskCategory.HIGH
        return RiskCategory.CRITICAL


# =============================================================================
# Safety Pillar Analyzer
# =============================================================================

class SafetyPillarAnalyzer:
    """Analyzer for Safety pillar of Responsible AI."""

    def __init__(self):
        self.harm_categories = [
            'physical', 'psychological', 'financial', 'reputational',
            'societal', 'environmental', 'legal'
        ]
        self.safety_controls = [
            'content_filtering', 'guardrails', 'human_review',
            'rate_limiting', 'access_control', 'monitoring'
        ]

    def analyze(self, data: Dict[str, Any], config: Optional[Dict] = None) -> PillarScore:
        """Analyze safety compliance."""
        config = config or {}

        metrics = self._calculate_metrics(data)
        score = self._compute_score(metrics)
        compliance = self._determine_compliance(score)
        maturity = self._assess_maturity(data, metrics)
        findings = self._generate_findings(metrics)
        recommendations = self._generate_recommendations(metrics)
        risk_level = self._assess_risk(metrics)

        return PillarScore(
            pillar=PillarType.SAFETY,
            score=score,
            compliance_level=compliance,
            maturity_level=maturity,
            sub_scores={
                'harm_prevention': metrics.harm_prevention_score,
                'content_safety': metrics.content_safety_score,
                'bias_mitigation': metrics.bias_mitigation_coverage,
                'toxicity_detection': metrics.toxicity_detection_rate,
                'guardrails': metrics.guardrail_effectiveness,
                'fail_safe': metrics.fail_safe_coverage,
                'human_oversight': metrics.human_oversight_score,
                'incident_response': metrics.incident_response_readiness,
            },
            findings=findings,
            recommendations=recommendations,
            risk_level=risk_level
        )

    def _calculate_metrics(self, data: Dict[str, Any]) -> SafetyMetrics:
        """Calculate safety metrics."""
        controls = data.get('safety_controls', {})
        testing = data.get('safety_testing', {})
        policies = data.get('safety_policies', {})
        incidents = data.get('incident_management', {})

        # Harm prevention
        harm_assessments = controls.get('harm_assessments', [])
        assessed_categories = sum(1 for h in self.harm_categories if h in harm_assessments)
        harm_prevention_score = assessed_categories / len(self.harm_categories)

        # Content safety
        content_filters = controls.get('content_filters', {})
        filter_types = ['profanity', 'hate_speech', 'violence', 'pii', 'misinformation']
        enabled_filters = sum(1 for f in filter_types if content_filters.get(f, False))
        content_safety_score = enabled_filters / len(filter_types)

        # Bias mitigation
        bias_controls = controls.get('bias_mitigation', {})
        bias_methods = ['pre_processing', 'in_processing', 'post_processing', 'monitoring']
        enabled_bias = sum(1 for m in bias_methods if bias_controls.get(m, False))
        bias_mitigation_coverage = enabled_bias / len(bias_methods)

        # Toxicity detection
        toxicity_config = controls.get('toxicity_detection', {})
        toxicity_enabled = toxicity_config.get('enabled', False)
        toxicity_threshold = toxicity_config.get('threshold', 0.5)
        toxicity_detection_rate = 1.0 if toxicity_enabled else 0.0
        toxicity_detection_rate *= (1.0 - toxicity_threshold + 0.5)

        # Guardrails
        guardrails = controls.get('guardrails', [])
        guardrail_effectiveness = min(1.0, len(guardrails) / len(self.safety_controls))

        # Fail-safe
        fail_safe = controls.get('fail_safe', {})
        fail_safe_coverage = 0.0
        if fail_safe.get('enabled', False):
            fail_safe_coverage = 0.5
            if fail_safe.get('tested', False):
                fail_safe_coverage += 0.3
            if fail_safe.get('documented', False):
                fail_safe_coverage += 0.2

        # Human oversight
        human_review = policies.get('human_review', {})
        human_oversight_score = 0.0
        if human_review.get('high_risk_decisions', False):
            human_oversight_score += 0.4
        if human_review.get('periodic_audits', False):
            human_oversight_score += 0.3
        if human_review.get('escalation_path', False):
            human_oversight_score += 0.3

        # Risk assessment
        risk_framework = policies.get('risk_framework', {})
        risk_assessment_quality = 1.0 if risk_framework.get('defined', False) else 0.5

        # Incident response
        incident_plan = incidents.get('response_plan', {})
        incident_response_readiness = 0.0
        if incident_plan.get('defined', False):
            incident_response_readiness = 0.4
            if incident_plan.get('tested', False):
                incident_response_readiness += 0.3
            if incident_plan.get('communication_plan', False):
                incident_response_readiness += 0.3

        # Safety testing
        safety_tests = testing.get('tests_performed', [])
        safety_testing_coverage = min(1.0, len(safety_tests) * 0.2)

        return SafetyMetrics(
            harm_prevention_score=harm_prevention_score,
            content_safety_score=content_safety_score,
            bias_mitigation_coverage=bias_mitigation_coverage,
            toxicity_detection_rate=min(1.0, toxicity_detection_rate),
            guardrail_effectiveness=guardrail_effectiveness,
            fail_safe_coverage=fail_safe_coverage,
            human_oversight_score=human_oversight_score,
            risk_assessment_quality=risk_assessment_quality,
            incident_response_readiness=incident_response_readiness,
            safety_testing_coverage=safety_testing_coverage
        )

    def _compute_score(self, metrics: SafetyMetrics) -> float:
        """Compute overall safety score."""
        weights = {
            'harm_prevention_score': 0.15,
            'content_safety_score': 0.15,
            'bias_mitigation_coverage': 0.10,
            'toxicity_detection_rate': 0.10,
            'guardrail_effectiveness': 0.15,
            'fail_safe_coverage': 0.10,
            'human_oversight_score': 0.15,
            'incident_response_readiness': 0.10,
        }

        score = sum(
            getattr(metrics, key) * weight
            for key, weight in weights.items()
        )
        return min(1.0, max(0.0, score))

    def _determine_compliance(self, score: float) -> ComplianceLevel:
        """Determine compliance level."""
        if score >= 0.9:
            return ComplianceLevel.FULL
        elif score >= 0.7:
            return ComplianceLevel.PARTIAL
        elif score >= 0.5:
            return ComplianceLevel.MINIMAL
        return ComplianceLevel.NON_COMPLIANT

    def _assess_maturity(self, data: Dict, metrics: SafetyMetrics) -> MaturityLevel:
        """Assess safety maturity."""
        indicators = 0

        if metrics.harm_prevention_score > 0.7:
            indicators += 1
        if metrics.guardrail_effectiveness > 0.8:
            indicators += 1
        if metrics.human_oversight_score > 0.7:
            indicators += 1
        if data.get('safety_by_design', False):
            indicators += 1
        if data.get('continuous_safety_monitoring', False):
            indicators += 1

        if indicators >= 5:
            return MaturityLevel.OPTIMIZED
        elif indicators >= 4:
            return MaturityLevel.MANAGED
        elif indicators >= 3:
            return MaturityLevel.DEFINED
        elif indicators >= 2:
            return MaturityLevel.DEVELOPING
        return MaturityLevel.INITIAL

    def _generate_findings(self, metrics: SafetyMetrics) -> List[str]:
        """Generate safety findings."""
        findings = []

        if metrics.harm_prevention_score < 0.7:
            findings.append("Harm assessment coverage is incomplete")
        if metrics.content_safety_score < 0.8:
            findings.append("Content filtering needs enhancement")
        if metrics.human_oversight_score < 0.6:
            findings.append("Human oversight mechanisms are limited")
        if metrics.incident_response_readiness < 0.7:
            findings.append("Incident response plan needs improvement")

        return findings

    def _generate_recommendations(self, metrics: SafetyMetrics) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []

        if metrics.harm_prevention_score < 0.8:
            recommendations.append("Conduct comprehensive harm assessments")
        if metrics.guardrail_effectiveness < 0.9:
            recommendations.append("Strengthen guardrail mechanisms")
        if metrics.human_oversight_score < 0.8:
            recommendations.append("Implement human-in-the-loop for high-risk decisions")
        if metrics.fail_safe_coverage < 0.8:
            recommendations.append("Test and document fail-safe mechanisms")

        return recommendations

    def _assess_risk(self, metrics: SafetyMetrics) -> RiskCategory:
        """Assess safety risk."""
        avg = (metrics.harm_prevention_score + metrics.guardrail_effectiveness) / 2

        if avg >= 0.8:
            return RiskCategory.LOW
        elif avg >= 0.6:
            return RiskCategory.MEDIUM
        elif avg >= 0.4:
            return RiskCategory.HIGH
        return RiskCategory.CRITICAL


# =============================================================================
# Accountability Pillar Analyzer
# =============================================================================

class AccountabilityPillarAnalyzer:
    """Analyzer for Accountability pillar of Responsible AI."""

    def __init__(self):
        self.raci_roles = ['responsible', 'accountable', 'consulted', 'informed']
        self.governance_elements = [
            'policies', 'procedures', 'standards', 'guidelines',
            'roles', 'responsibilities', 'oversight', 'reporting'
        ]

    def analyze(self, data: Dict[str, Any], config: Optional[Dict] = None) -> PillarScore:
        """Analyze accountability compliance."""
        config = config or {}

        metrics = self._calculate_metrics(data)
        score = self._compute_score(metrics)
        compliance = self._determine_compliance(score)
        maturity = self._assess_maturity(data, metrics)
        findings = self._generate_findings(metrics)
        recommendations = self._generate_recommendations(metrics)
        risk_level = self._assess_risk(metrics)

        return PillarScore(
            pillar=PillarType.ACCOUNTABILITY,
            score=score,
            compliance_level=compliance,
            maturity_level=maturity,
            sub_scores={
                'audit_trail': metrics.audit_trail_completeness,
                'governance': metrics.governance_structure_score,
                'role_clarity': metrics.role_clarity_score,
                'documentation': metrics.decision_documentation,
                'compliance_monitoring': metrics.compliance_monitoring,
                'remediation': metrics.remediation_process_score,
                'stakeholder_engagement': metrics.stakeholder_engagement,
                'continuous_improvement': metrics.continuous_improvement,
            },
            findings=findings,
            recommendations=recommendations,
            risk_level=risk_level
        )

    def _calculate_metrics(self, data: Dict[str, Any]) -> AccountabilityMetrics:
        """Calculate accountability metrics."""
        governance = data.get('governance', {})
        audit = data.get('audit', {})
        roles = data.get('roles', {})
        processes = data.get('processes', {})

        # Audit trail
        audit_config = audit.get('configuration', {})
        audit_trail_completeness = 0.0
        if audit_config.get('enabled', False):
            audit_trail_completeness = 0.4
            if audit_config.get('immutable', False):
                audit_trail_completeness += 0.2
            if audit_config.get('timestamped', False):
                audit_trail_completeness += 0.2
            if audit_config.get('signed', False):
                audit_trail_completeness += 0.2

        # Governance structure
        governance_elements_defined = governance.get('elements', [])
        defined_elements = sum(1 for e in self.governance_elements if e in governance_elements_defined)
        governance_structure_score = defined_elements / len(self.governance_elements)

        # Role clarity (RACI)
        raci_matrix = roles.get('raci_matrix', {})
        roles_defined = sum(1 for r in self.raci_roles if r in raci_matrix)
        role_clarity_score = roles_defined / len(self.raci_roles)

        # Decision documentation
        documentation = processes.get('decision_documentation', {})
        decision_documentation = 0.0
        if documentation.get('required', False):
            decision_documentation = 0.5
            if documentation.get('template', False):
                decision_documentation += 0.25
            if documentation.get('review_process', False):
                decision_documentation += 0.25

        # Compliance monitoring
        monitoring = processes.get('compliance_monitoring', {})
        compliance_monitoring = 0.0
        if monitoring.get('enabled', False):
            compliance_monitoring = 0.4
            if monitoring.get('automated', False):
                compliance_monitoring += 0.3
            if monitoring.get('regular_reporting', False):
                compliance_monitoring += 0.3

        # Remediation process
        remediation = processes.get('remediation', {})
        remediation_process_score = 0.0
        if remediation.get('defined', False):
            remediation_process_score = 0.4
            if remediation.get('escalation_path', False):
                remediation_process_score += 0.3
            if remediation.get('sla_defined', False):
                remediation_process_score += 0.3

        # Stakeholder engagement
        stakeholders = governance.get('stakeholder_engagement', {})
        stakeholder_engagement = 0.0
        if stakeholders.get('identified', False):
            stakeholder_engagement = 0.4
            if stakeholders.get('communication_plan', False):
                stakeholder_engagement += 0.3
            if stakeholders.get('feedback_mechanism', False):
                stakeholder_engagement += 0.3

        # Policy enforcement
        enforcement = governance.get('policy_enforcement', {})
        policy_enforcement = 1.0 if enforcement.get('automated', False) else 0.5

        # Incident tracking
        incident_tracking = processes.get('incident_tracking', {})
        incident_tracking_score = 0.0
        if incident_tracking.get('enabled', False):
            incident_tracking_score = 0.5
            if incident_tracking.get('root_cause_analysis', False):
                incident_tracking_score += 0.25
            if incident_tracking.get('lessons_learned', False):
                incident_tracking_score += 0.25

        # Continuous improvement
        improvement = processes.get('continuous_improvement', {})
        continuous_improvement = 0.0
        if improvement.get('process', False):
            continuous_improvement = 0.4
            if improvement.get('metrics_tracked', False):
                continuous_improvement += 0.3
            if improvement.get('regular_reviews', False):
                continuous_improvement += 0.3

        return AccountabilityMetrics(
            audit_trail_completeness=audit_trail_completeness,
            governance_structure_score=governance_structure_score,
            role_clarity_score=role_clarity_score,
            decision_documentation=decision_documentation,
            compliance_monitoring=compliance_monitoring,
            remediation_process_score=remediation_process_score,
            stakeholder_engagement=stakeholder_engagement,
            policy_enforcement=policy_enforcement,
            incident_tracking=incident_tracking_score,
            continuous_improvement=continuous_improvement
        )

    def _compute_score(self, metrics: AccountabilityMetrics) -> float:
        """Compute overall accountability score."""
        weights = {
            'audit_trail_completeness': 0.15,
            'governance_structure_score': 0.15,
            'role_clarity_score': 0.10,
            'decision_documentation': 0.10,
            'compliance_monitoring': 0.15,
            'remediation_process_score': 0.10,
            'stakeholder_engagement': 0.10,
            'continuous_improvement': 0.15,
        }

        score = sum(
            getattr(metrics, key) * weight
            for key, weight in weights.items()
        )
        return min(1.0, max(0.0, score))

    def _determine_compliance(self, score: float) -> ComplianceLevel:
        """Determine compliance level."""
        if score >= 0.9:
            return ComplianceLevel.FULL
        elif score >= 0.7:
            return ComplianceLevel.PARTIAL
        elif score >= 0.5:
            return ComplianceLevel.MINIMAL
        return ComplianceLevel.NON_COMPLIANT

    def _assess_maturity(self, data: Dict, metrics: AccountabilityMetrics) -> MaturityLevel:
        """Assess accountability maturity."""
        indicators = 0

        if metrics.audit_trail_completeness > 0.8:
            indicators += 1
        if metrics.governance_structure_score > 0.7:
            indicators += 1
        if metrics.compliance_monitoring > 0.7:
            indicators += 1
        if data.get('accountability_framework', False):
            indicators += 1
        if data.get('regular_governance_reviews', False):
            indicators += 1

        if indicators >= 5:
            return MaturityLevel.OPTIMIZED
        elif indicators >= 4:
            return MaturityLevel.MANAGED
        elif indicators >= 3:
            return MaturityLevel.DEFINED
        elif indicators >= 2:
            return MaturityLevel.DEVELOPING
        return MaturityLevel.INITIAL

    def _generate_findings(self, metrics: AccountabilityMetrics) -> List[str]:
        """Generate accountability findings."""
        findings = []

        if metrics.audit_trail_completeness < 0.8:
            findings.append("Audit trail coverage is incomplete")
        if metrics.role_clarity_score < 0.7:
            findings.append("RACI matrix is not fully defined")
        if metrics.compliance_monitoring < 0.7:
            findings.append("Compliance monitoring needs enhancement")
        if metrics.continuous_improvement < 0.6:
            findings.append("Continuous improvement process is limited")

        return findings

    def _generate_recommendations(self, metrics: AccountabilityMetrics) -> List[str]:
        """Generate accountability recommendations."""
        recommendations = []

        if metrics.audit_trail_completeness < 0.9:
            recommendations.append("Implement immutable, signed audit trails")
        if metrics.governance_structure_score < 0.8:
            recommendations.append("Define comprehensive governance structure")
        if metrics.role_clarity_score < 0.8:
            recommendations.append("Complete RACI matrix for all AI decisions")
        if metrics.compliance_monitoring < 0.8:
            recommendations.append("Enable automated compliance monitoring")

        return recommendations

    def _assess_risk(self, metrics: AccountabilityMetrics) -> RiskCategory:
        """Assess accountability risk."""
        avg = (metrics.audit_trail_completeness + metrics.compliance_monitoring) / 2

        if avg >= 0.8:
            return RiskCategory.LOW
        elif avg >= 0.6:
            return RiskCategory.MEDIUM
        elif avg >= 0.4:
            return RiskCategory.HIGH
        return RiskCategory.CRITICAL


# =============================================================================
# Integrated Five Pillar Analyzer
# =============================================================================

class FivePillarAnalyzer:
    """Integrated analyzer for all five RAI pillars."""

    def __init__(self):
        self.privacy_analyzer = PrivacyPillarAnalyzer()
        self.transparency_analyzer = TransparencyPillarAnalyzer()
        self.robustness_analyzer = RobustnessPillarAnalyzer()
        self.safety_analyzer = SafetyPillarAnalyzer()
        self.accountability_analyzer = AccountabilityPillarAnalyzer()

        self.pillar_weights = {
            PillarType.PRIVACY: 0.20,
            PillarType.TRANSPARENCY: 0.20,
            PillarType.ROBUSTNESS: 0.20,
            PillarType.SAFETY: 0.20,
            PillarType.ACCOUNTABILITY: 0.20,
        }

    def analyze(self, data: Dict[str, Any], config: Optional[Dict] = None) -> RAIPillarAssessment:
        """Perform comprehensive five-pillar analysis."""
        config = config or {}

        # Analyze each pillar
        pillar_scores = {
            PillarType.PRIVACY: self.privacy_analyzer.analyze(data.get('privacy', {})),
            PillarType.TRANSPARENCY: self.transparency_analyzer.analyze(data.get('transparency', {})),
            PillarType.ROBUSTNESS: self.robustness_analyzer.analyze(data.get('robustness', {})),
            PillarType.SAFETY: self.safety_analyzer.analyze(data.get('safety', {})),
            PillarType.ACCOUNTABILITY: self.accountability_analyzer.analyze(data.get('accountability', {})),
        }

        # Calculate overall score
        overall_score = sum(
            pillar_scores[pillar].score * weight
            for pillar, weight in self.pillar_weights.items()
        )

        # Determine overall compliance
        overall_compliance = self._determine_overall_compliance(pillar_scores)

        # Determine overall maturity
        overall_maturity = self._determine_overall_maturity(pillar_scores)

        # Analyze cross-pillar dependencies
        cross_pillar_dependencies = self._analyze_dependencies(pillar_scores)

        # Generate priority actions
        priority_actions = self._generate_priority_actions(pillar_scores)

        # Risk summary
        risk_summary = self._summarize_risks(pillar_scores)

        return RAIPillarAssessment(
            pillar_scores=pillar_scores,
            overall_score=overall_score,
            overall_compliance=overall_compliance,
            overall_maturity=overall_maturity,
            cross_pillar_dependencies=cross_pillar_dependencies,
            priority_actions=priority_actions,
            risk_summary=risk_summary
        )

    def _determine_overall_compliance(self, scores: Dict[PillarType, PillarScore]) -> ComplianceLevel:
        """Determine overall compliance level."""
        compliance_values = {
            ComplianceLevel.FULL: 4,
            ComplianceLevel.PARTIAL: 3,
            ComplianceLevel.MINIMAL: 2,
            ComplianceLevel.NON_COMPLIANT: 1,
        }

        avg_compliance = sum(
            compliance_values[s.compliance_level] for s in scores.values()
        ) / len(scores)

        if avg_compliance >= 3.5:
            return ComplianceLevel.FULL
        elif avg_compliance >= 2.5:
            return ComplianceLevel.PARTIAL
        elif avg_compliance >= 1.5:
            return ComplianceLevel.MINIMAL
        return ComplianceLevel.NON_COMPLIANT

    def _determine_overall_maturity(self, scores: Dict[PillarType, PillarScore]) -> MaturityLevel:
        """Determine overall maturity level."""
        maturity_values = {
            MaturityLevel.OPTIMIZED: 5,
            MaturityLevel.MANAGED: 4,
            MaturityLevel.DEFINED: 3,
            MaturityLevel.DEVELOPING: 2,
            MaturityLevel.INITIAL: 1,
        }

        avg_maturity = sum(
            maturity_values[s.maturity_level] for s in scores.values()
        ) / len(scores)

        if avg_maturity >= 4.5:
            return MaturityLevel.OPTIMIZED
        elif avg_maturity >= 3.5:
            return MaturityLevel.MANAGED
        elif avg_maturity >= 2.5:
            return MaturityLevel.DEFINED
        elif avg_maturity >= 1.5:
            return MaturityLevel.DEVELOPING
        return MaturityLevel.INITIAL

    def _analyze_dependencies(self, scores: Dict[PillarType, PillarScore]) -> List[CrossPillarDependency]:
        """Analyze cross-pillar dependencies."""
        dependencies = [
            CrossPillarDependency(
                source_pillar=PillarType.TRANSPARENCY,
                target_pillar=PillarType.ACCOUNTABILITY,
                dependency_type="enabler",
                strength=0.8,
                description="Transparency enables accountability through audit trails"
            ),
            CrossPillarDependency(
                source_pillar=PillarType.PRIVACY,
                target_pillar=PillarType.TRANSPARENCY,
                dependency_type="constraint",
                strength=0.6,
                description="Privacy requirements may limit transparency"
            ),
            CrossPillarDependency(
                source_pillar=PillarType.SAFETY,
                target_pillar=PillarType.ROBUSTNESS,
                dependency_type="enabler",
                strength=0.9,
                description="Safety depends on system robustness"
            ),
            CrossPillarDependency(
                source_pillar=PillarType.ACCOUNTABILITY,
                target_pillar=PillarType.SAFETY,
                dependency_type="enabler",
                strength=0.7,
                description="Accountability frameworks support safety governance"
            ),
            CrossPillarDependency(
                source_pillar=PillarType.ROBUSTNESS,
                target_pillar=PillarType.PRIVACY,
                dependency_type="enabler",
                strength=0.5,
                description="Robust systems better protect privacy"
            ),
        ]
        return dependencies

    def _generate_priority_actions(self, scores: Dict[PillarType, PillarScore]) -> List[str]:
        """Generate prioritized action items."""
        actions = []

        # Sort pillars by score (lowest first)
        sorted_pillars = sorted(
            scores.items(),
            key=lambda x: x[1].score
        )

        for pillar, score in sorted_pillars:
            if score.score < 0.7:
                actions.extend([
                    f"[{pillar.value.upper()}] {rec}"
                    for rec in score.recommendations[:2]
                ])

        return actions[:10]  # Top 10 priority actions

    def _summarize_risks(self, scores: Dict[PillarType, PillarScore]) -> Dict[RiskCategory, int]:
        """Summarize risk levels across pillars."""
        risk_summary = defaultdict(int)

        for score in scores.values():
            risk_summary[score.risk_level] += 1

        return dict(risk_summary)

    def get_pillar_radar_data(self, assessment: RAIPillarAssessment) -> Dict[str, float]:
        """Get data for radar chart visualization."""
        return {
            pillar.value: score.score
            for pillar, score in assessment.pillar_scores.items()
        }

    def get_gap_analysis(self, assessment: RAIPillarAssessment, target_score: float = 0.8) -> Dict[str, float]:
        """Calculate gap to target for each pillar."""
        return {
            pillar.value: max(0, target_score - score.score)
            for pillar, score in assessment.pillar_scores.items()
        }


# =============================================================================
# Pillar Benchmark Analyzer
# =============================================================================

class PillarBenchmarkAnalyzer:
    """Analyzer for benchmarking pillar scores against standards."""

    def __init__(self):
        self.industry_benchmarks = {
            'healthcare': {
                PillarType.PRIVACY: 0.9,
                PillarType.SAFETY: 0.95,
                PillarType.ACCOUNTABILITY: 0.85,
                PillarType.TRANSPARENCY: 0.8,
                PillarType.ROBUSTNESS: 0.85,
            },
            'finance': {
                PillarType.PRIVACY: 0.85,
                PillarType.SAFETY: 0.85,
                PillarType.ACCOUNTABILITY: 0.9,
                PillarType.TRANSPARENCY: 0.85,
                PillarType.ROBUSTNESS: 0.9,
            },
            'general': {
                PillarType.PRIVACY: 0.75,
                PillarType.SAFETY: 0.8,
                PillarType.ACCOUNTABILITY: 0.75,
                PillarType.TRANSPARENCY: 0.75,
                PillarType.ROBUSTNESS: 0.75,
            },
        }

    def benchmark(
        self,
        assessment: RAIPillarAssessment,
        industry: str = 'general'
    ) -> Dict[str, Any]:
        """Benchmark assessment against industry standards."""
        benchmarks = self.industry_benchmarks.get(industry, self.industry_benchmarks['general'])

        comparison = {}
        for pillar, target in benchmarks.items():
            actual = assessment.pillar_scores[pillar].score
            comparison[pillar.value] = {
                'actual': actual,
                'target': target,
                'gap': target - actual,
                'status': 'meets' if actual >= target else 'below',
                'percentage_of_target': (actual / target) * 100 if target > 0 else 100
            }

        overall_benchmark = sum(benchmarks.values()) / len(benchmarks)

        return {
            'industry': industry,
            'pillar_comparison': comparison,
            'overall_actual': assessment.overall_score,
            'overall_target': overall_benchmark,
            'overall_gap': overall_benchmark - assessment.overall_score,
            'pillars_meeting_benchmark': sum(
                1 for p in comparison.values() if p['status'] == 'meets'
            )
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_privacy_pillar(data: Dict[str, Any]) -> PillarScore:
    """Convenience function for privacy pillar analysis."""
    analyzer = PrivacyPillarAnalyzer()
    return analyzer.analyze(data)


def analyze_transparency_pillar(data: Dict[str, Any]) -> PillarScore:
    """Convenience function for transparency pillar analysis."""
    analyzer = TransparencyPillarAnalyzer()
    return analyzer.analyze(data)


def analyze_robustness_pillar(data: Dict[str, Any]) -> PillarScore:
    """Convenience function for robustness pillar analysis."""
    analyzer = RobustnessPillarAnalyzer()
    return analyzer.analyze(data)


def analyze_safety_pillar(data: Dict[str, Any]) -> PillarScore:
    """Convenience function for safety pillar analysis."""
    analyzer = SafetyPillarAnalyzer()
    return analyzer.analyze(data)


def analyze_accountability_pillar(data: Dict[str, Any]) -> PillarScore:
    """Convenience function for accountability pillar analysis."""
    analyzer = AccountabilityPillarAnalyzer()
    return analyzer.analyze(data)


def analyze_all_pillars(data: Dict[str, Any]) -> RAIPillarAssessment:
    """Convenience function for complete five-pillar analysis."""
    analyzer = FivePillarAnalyzer()
    return analyzer.analyze(data)


def benchmark_pillars(
    assessment: RAIPillarAssessment,
    industry: str = 'general'
) -> Dict[str, Any]:
    """Convenience function for pillar benchmarking."""
    analyzer = PillarBenchmarkAnalyzer()
    return analyzer.benchmark(assessment, industry)
