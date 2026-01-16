"""
Control Framework Analysis Module
=================================

Comprehensive framework for analyzing AI system controls including:
1. Hard Controls - Technical enforcement mechanisms (DLP, access control, encryption)
2. Soft Controls - Policy-based controls (guidelines, training, notifications)
3. User Notification - AI usage disclosure and transparency notifications
4. Control Effectiveness - Measuring control implementation and efficacy
5. Control Gap Analysis - Identifying missing or weak controls

This module provides analyzers for assessing control frameworks
in AI/ML systems for responsible AI governance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict
import math


# =============================================================================
# Enums
# =============================================================================

class ControlType(Enum):
    """Types of controls."""
    HARD = "hard"  # Technical enforcement
    SOFT = "soft"  # Policy-based


class ControlCategory(Enum):
    """Control categories."""
    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"
    COMPENSATING = "compensating"
    DIRECTIVE = "directive"


class ControlDomain(Enum):
    """Control domains."""
    DATA_PROTECTION = "data_protection"
    ACCESS_CONTROL = "access_control"
    MODEL_GOVERNANCE = "model_governance"
    OUTPUT_CONTROL = "output_control"
    AUDIT_LOGGING = "audit_logging"
    USER_NOTIFICATION = "user_notification"
    INCIDENT_RESPONSE = "incident_response"
    PRIVACY = "privacy"
    SAFETY = "safety"
    FAIRNESS = "fairness"


class ControlStatus(Enum):
    """Control implementation status."""
    IMPLEMENTED = "implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    PLANNED = "planned"
    NOT_IMPLEMENTED = "not_implemented"
    NOT_APPLICABLE = "not_applicable"


class EffectivenessRating(Enum):
    """Control effectiveness rating."""
    HIGHLY_EFFECTIVE = "highly_effective"
    EFFECTIVE = "effective"
    PARTIALLY_EFFECTIVE = "partially_effective"
    INEFFECTIVE = "ineffective"
    NOT_TESTED = "not_tested"


class NotificationType(Enum):
    """User notification types."""
    AI_USAGE = "ai_usage"
    DATA_COLLECTION = "data_collection"
    DECISION_MAKING = "decision_making"
    PROFILING = "profiling"
    CONSENT_REQUEST = "consent_request"
    OPT_OUT = "opt_out"
    EXPLANATION = "explanation"
    INCIDENT = "incident"


class NotificationChannel(Enum):
    """Notification delivery channels."""
    IN_APP = "in_app"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    API_RESPONSE = "api_response"
    UI_BANNER = "ui_banner"
    TERMS_OF_SERVICE = "terms_of_service"


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Control:
    """Representation of a control."""
    control_id: str
    name: str
    description: str
    control_type: ControlType
    category: ControlCategory
    domain: ControlDomain
    status: ControlStatus
    effectiveness: EffectivenessRating = EffectivenessRating.NOT_TESTED
    owner: Optional[str] = None
    implementation_details: Optional[str] = None
    test_results: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)
    risk_mitigation: List[str] = field(default_factory=list)


@dataclass
class ControlAssessment:
    """Assessment result for a control."""
    control: Control
    score: float  # 0-1
    findings: List[str]
    recommendations: List[str]
    evidence: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)


@dataclass
class HardControlMetrics:
    """Metrics for hard controls."""
    dlp_coverage: float = 0.0
    encryption_coverage: float = 0.0
    access_control_score: float = 0.0
    authentication_strength: float = 0.0
    input_validation_coverage: float = 0.0
    output_filtering_coverage: float = 0.0
    audit_logging_coverage: float = 0.0
    automated_enforcement_rate: float = 0.0


@dataclass
class SoftControlMetrics:
    """Metrics for soft controls."""
    policy_coverage: float = 0.0
    training_completion: float = 0.0
    awareness_score: float = 0.0
    guideline_adherence: float = 0.0
    review_process_maturity: float = 0.0
    escalation_path_defined: bool = False
    documentation_completeness: float = 0.0


@dataclass
class NotificationMetrics:
    """Metrics for user notifications."""
    notification_coverage: float = 0.0
    disclosure_completeness: float = 0.0
    consent_capture_rate: float = 0.0
    opt_out_availability: float = 0.0
    explanation_quality: float = 0.0
    channel_accessibility: float = 0.0
    timing_appropriateness: float = 0.0


@dataclass
class ControlFrameworkAssessment:
    """Complete control framework assessment."""
    hard_control_score: float
    soft_control_score: float
    notification_score: float
    overall_score: float
    effectiveness_rating: EffectivenessRating
    control_assessments: List[ControlAssessment]
    gap_analysis: Dict[ControlDomain, List[str]]
    risk_exposure: Dict[RiskLevel, int]
    priority_actions: List[str]
    maturity_level: str


# =============================================================================
# Hard Control Analyzer
# =============================================================================

class HardControlAnalyzer:
    """Analyzer for technical (hard) controls."""

    def __init__(self):
        self.required_hard_controls = {
            ControlDomain.DATA_PROTECTION: [
                'dlp_scanning', 'encryption_at_rest', 'encryption_in_transit',
                'data_masking', 'tokenization', 'secure_deletion'
            ],
            ControlDomain.ACCESS_CONTROL: [
                'authentication', 'authorization', 'rbac', 'mfa',
                'session_management', 'api_key_rotation'
            ],
            ControlDomain.MODEL_GOVERNANCE: [
                'model_versioning', 'model_registry', 'deployment_gates',
                'rollback_capability', 'feature_flags', 'a_b_testing'
            ],
            ControlDomain.OUTPUT_CONTROL: [
                'content_filtering', 'guardrails', 'rate_limiting',
                'output_validation', 'response_sanitization'
            ],
            ControlDomain.AUDIT_LOGGING: [
                'access_logging', 'decision_logging', 'change_tracking',
                'log_integrity', 'log_retention', 'log_monitoring'
            ],
        }

    def analyze(
        self,
        implemented_controls: Dict[str, Any],
        config: Optional[Dict] = None
    ) -> Tuple[HardControlMetrics, List[ControlAssessment]]:
        """Analyze hard controls."""
        config = config or {}

        metrics = self._calculate_metrics(implemented_controls)
        assessments = self._assess_controls(implemented_controls)

        return metrics, assessments

    def _calculate_metrics(self, controls: Dict[str, Any]) -> HardControlMetrics:
        """Calculate hard control metrics."""
        # DLP coverage
        dlp_config = controls.get('dlp', {})
        dlp_types = ['pii', 'phi', 'pci', 'secrets', 'custom']
        dlp_enabled = sum(1 for t in dlp_types if dlp_config.get(t, False))
        dlp_coverage = dlp_enabled / len(dlp_types)

        # Encryption coverage
        encryption = controls.get('encryption', {})
        encryption_coverage = 0.0
        if encryption.get('at_rest', False):
            encryption_coverage += 0.5
        if encryption.get('in_transit', False):
            encryption_coverage += 0.5

        # Access control score
        access = controls.get('access_control', {})
        access_features = ['rbac', 'mfa', 'sso', 'api_keys', 'least_privilege']
        access_enabled = sum(1 for f in access_features if access.get(f, False))
        access_control_score = access_enabled / len(access_features)

        # Authentication strength
        auth = controls.get('authentication', {})
        auth_score = 0.0
        if auth.get('password_policy', False):
            auth_score += 0.2
        if auth.get('mfa', False):
            auth_score += 0.3
        if auth.get('biometric', False):
            auth_score += 0.2
        if auth.get('certificate', False):
            auth_score += 0.3

        # Input validation
        validation = controls.get('input_validation', {})
        validation_types = ['type_check', 'format_check', 'range_check', 'injection_prevention']
        validation_enabled = sum(1 for v in validation_types if validation.get(v, False))
        input_validation_coverage = validation_enabled / len(validation_types)

        # Output filtering
        output = controls.get('output_control', {})
        output_filters = ['content_filter', 'pii_redaction', 'toxicity_filter', 'guardrails']
        output_enabled = sum(1 for f in output_filters if output.get(f, False))
        output_filtering_coverage = output_enabled / len(output_filters)

        # Audit logging
        logging = controls.get('audit_logging', {})
        log_types = ['access', 'decisions', 'changes', 'errors', 'security']
        log_enabled = sum(1 for l in log_types if logging.get(l, False))
        audit_logging_coverage = log_enabled / len(log_types)

        # Automated enforcement
        enforcement = controls.get('automated_enforcement', {})
        automated_enforcement_rate = enforcement.get('coverage', 0.0)

        return HardControlMetrics(
            dlp_coverage=dlp_coverage,
            encryption_coverage=encryption_coverage,
            access_control_score=access_control_score,
            authentication_strength=auth_score,
            input_validation_coverage=input_validation_coverage,
            output_filtering_coverage=output_filtering_coverage,
            audit_logging_coverage=audit_logging_coverage,
            automated_enforcement_rate=automated_enforcement_rate
        )

    def _assess_controls(
        self,
        implemented_controls: Dict[str, Any]
    ) -> List[ControlAssessment]:
        """Assess individual hard controls."""
        assessments = []

        for domain, required_controls in self.required_hard_controls.items():
            domain_controls = implemented_controls.get(domain.value, {})

            for control_name in required_controls:
                control_config = domain_controls.get(control_name, {})

                # Determine status
                if isinstance(control_config, dict):
                    status = ControlStatus.IMPLEMENTED if control_config.get('enabled', False) else ControlStatus.NOT_IMPLEMENTED
                elif isinstance(control_config, bool):
                    status = ControlStatus.IMPLEMENTED if control_config else ControlStatus.NOT_IMPLEMENTED
                else:
                    status = ControlStatus.NOT_IMPLEMENTED

                # Create control object
                control = Control(
                    control_id=f"{domain.value}_{control_name}",
                    name=control_name.replace('_', ' ').title(),
                    description=f"Hard control for {control_name}",
                    control_type=ControlType.HARD,
                    category=ControlCategory.PREVENTIVE,
                    domain=domain,
                    status=status
                )

                # Calculate score
                score = 1.0 if status == ControlStatus.IMPLEMENTED else 0.0

                # Generate findings and recommendations
                findings = []
                recommendations = []
                gaps = []

                if status != ControlStatus.IMPLEMENTED:
                    findings.append(f"{control_name} is not implemented")
                    recommendations.append(f"Implement {control_name} control")
                    gaps.append(f"Missing {domain.value} control: {control_name}")

                assessment = ControlAssessment(
                    control=control,
                    score=score,
                    findings=findings,
                    recommendations=recommendations,
                    gaps=gaps
                )
                assessments.append(assessment)

        return assessments


# =============================================================================
# Soft Control Analyzer
# =============================================================================

class SoftControlAnalyzer:
    """Analyzer for policy-based (soft) controls."""

    def __init__(self):
        self.required_soft_controls = {
            'policies': [
                'acceptable_use', 'data_handling', 'model_governance',
                'incident_response', 'privacy', 'ethics'
            ],
            'processes': [
                'review_process', 'approval_workflow', 'change_management',
                'escalation_procedure', 'audit_process'
            ],
            'training': [
                'rai_awareness', 'data_privacy', 'security_basics',
                'bias_recognition', 'incident_reporting'
            ],
            'documentation': [
                'system_documentation', 'api_documentation', 'runbooks',
                'model_cards', 'datasheets'
            ],
        }

    def analyze(
        self,
        soft_controls: Dict[str, Any],
        config: Optional[Dict] = None
    ) -> Tuple[SoftControlMetrics, List[ControlAssessment]]:
        """Analyze soft controls."""
        config = config or {}

        metrics = self._calculate_metrics(soft_controls)
        assessments = self._assess_controls(soft_controls)

        return metrics, assessments

    def _calculate_metrics(self, controls: Dict[str, Any]) -> SoftControlMetrics:
        """Calculate soft control metrics."""
        # Policy coverage
        policies = controls.get('policies', {})
        policy_count = sum(1 for p in self.required_soft_controls['policies'] if policies.get(p, {}).get('defined', False))
        policy_coverage = policy_count / len(self.required_soft_controls['policies'])

        # Training completion
        training = controls.get('training', {})
        training_completion = training.get('completion_rate', 0.0)

        # Awareness score
        awareness = controls.get('awareness', {})
        awareness_score = awareness.get('score', 0.5)

        # Guideline adherence
        adherence = controls.get('adherence', {})
        guideline_adherence = adherence.get('rate', 0.0)

        # Review process maturity
        review = controls.get('review_process', {})
        review_maturity = 0.0
        if review.get('defined', False):
            review_maturity = 0.4
        if review.get('automated', False):
            review_maturity += 0.3
        if review.get('measured', False):
            review_maturity += 0.3

        # Escalation path
        escalation = controls.get('escalation', {})
        escalation_defined = escalation.get('defined', False)

        # Documentation completeness
        docs = controls.get('documentation', {})
        doc_count = sum(1 for d in self.required_soft_controls['documentation'] if docs.get(d, {}).get('exists', False))
        documentation_completeness = doc_count / len(self.required_soft_controls['documentation'])

        return SoftControlMetrics(
            policy_coverage=policy_coverage,
            training_completion=training_completion,
            awareness_score=awareness_score,
            guideline_adherence=guideline_adherence,
            review_process_maturity=review_maturity,
            escalation_path_defined=escalation_defined,
            documentation_completeness=documentation_completeness
        )

    def _assess_controls(
        self,
        soft_controls: Dict[str, Any]
    ) -> List[ControlAssessment]:
        """Assess individual soft controls."""
        assessments = []

        for category, controls in self.required_soft_controls.items():
            category_config = soft_controls.get(category, {})

            for control_name in controls:
                control_config = category_config.get(control_name, {})

                # Determine status
                if isinstance(control_config, dict):
                    if control_config.get('defined', False) or control_config.get('exists', False):
                        status = ControlStatus.IMPLEMENTED
                    elif control_config.get('planned', False):
                        status = ControlStatus.PLANNED
                    else:
                        status = ControlStatus.NOT_IMPLEMENTED
                elif isinstance(control_config, bool):
                    status = ControlStatus.IMPLEMENTED if control_config else ControlStatus.NOT_IMPLEMENTED
                else:
                    status = ControlStatus.NOT_IMPLEMENTED

                control = Control(
                    control_id=f"soft_{category}_{control_name}",
                    name=control_name.replace('_', ' ').title(),
                    description=f"Soft control for {control_name}",
                    control_type=ControlType.SOFT,
                    category=ControlCategory.DIRECTIVE,
                    domain=ControlDomain.MODEL_GOVERNANCE,
                    status=status
                )

                score = 1.0 if status == ControlStatus.IMPLEMENTED else 0.5 if status == ControlStatus.PLANNED else 0.0

                findings = []
                recommendations = []
                gaps = []

                if status == ControlStatus.NOT_IMPLEMENTED:
                    findings.append(f"{control_name} policy/process not defined")
                    recommendations.append(f"Define and implement {control_name}")
                    gaps.append(f"Missing {category}: {control_name}")
                elif status == ControlStatus.PLANNED:
                    findings.append(f"{control_name} is planned but not implemented")
                    recommendations.append(f"Prioritize implementation of {control_name}")

                assessment = ControlAssessment(
                    control=control,
                    score=score,
                    findings=findings,
                    recommendations=recommendations,
                    gaps=gaps
                )
                assessments.append(assessment)

        return assessments


# =============================================================================
# User Notification Analyzer
# =============================================================================

class UserNotificationAnalyzer:
    """Analyzer for user notification controls."""

    def __init__(self):
        self.required_notifications = {
            NotificationType.AI_USAGE: {
                'description': 'Disclosure that AI is being used',
                'mandatory': True,
                'timing': 'before_interaction'
            },
            NotificationType.DATA_COLLECTION: {
                'description': 'Notice about data being collected',
                'mandatory': True,
                'timing': 'before_collection'
            },
            NotificationType.DECISION_MAKING: {
                'description': 'Explanation of AI-assisted decisions',
                'mandatory': True,
                'timing': 'with_decision'
            },
            NotificationType.CONSENT_REQUEST: {
                'description': 'Request for user consent',
                'mandatory': True,
                'timing': 'before_processing'
            },
            NotificationType.OPT_OUT: {
                'description': 'Option to opt out of AI processing',
                'mandatory': True,
                'timing': 'always_available'
            },
            NotificationType.EXPLANATION: {
                'description': 'Explanation of how AI works',
                'mandatory': False,
                'timing': 'on_request'
            },
        }

    def analyze(
        self,
        notification_config: Dict[str, Any],
        config: Optional[Dict] = None
    ) -> Tuple[NotificationMetrics, List[ControlAssessment]]:
        """Analyze notification controls."""
        config = config or {}

        metrics = self._calculate_metrics(notification_config)
        assessments = self._assess_notifications(notification_config)

        return metrics, assessments

    def _calculate_metrics(self, notifications: Dict[str, Any]) -> NotificationMetrics:
        """Calculate notification metrics."""
        # Notification coverage
        implemented = 0
        mandatory_count = sum(1 for n in self.required_notifications.values() if n['mandatory'])

        for notif_type, details in self.required_notifications.items():
            notif_config = notifications.get(notif_type.value, {})
            if notif_config.get('enabled', False):
                implemented += 1

        notification_coverage = implemented / len(self.required_notifications)

        # Disclosure completeness
        disclosure_config = notifications.get('disclosure', {})
        disclosure_elements = ['ai_involvement', 'data_usage', 'decision_factors', 'limitations']
        disclosure_present = sum(1 for e in disclosure_elements if disclosure_config.get(e, False))
        disclosure_completeness = disclosure_present / len(disclosure_elements)

        # Consent capture rate
        consent_config = notifications.get('consent', {})
        consent_capture_rate = consent_config.get('capture_rate', 0.0)

        # Opt-out availability
        opt_out_config = notifications.get('opt_out', {})
        opt_out_availability = 1.0 if opt_out_config.get('available', False) else 0.0
        if opt_out_config.get('easy_to_find', False):
            opt_out_availability = min(1.0, opt_out_availability + 0.2)

        # Explanation quality
        explanation_config = notifications.get('explanation', {})
        explanation_factors = ['comprehensible', 'accurate', 'complete', 'accessible']
        explanation_quality = sum(
            1 for f in explanation_factors if explanation_config.get(f, False)
        ) / len(explanation_factors)

        # Channel accessibility
        channels = notifications.get('channels', [])
        all_channels = [c.value for c in NotificationChannel]
        channel_accessibility = len(set(channels) & set(all_channels)) / len(all_channels)

        # Timing appropriateness
        timing_config = notifications.get('timing', {})
        timing_appropriate = timing_config.get('appropriate', True)
        timing_appropriateness = 1.0 if timing_appropriate else 0.5

        return NotificationMetrics(
            notification_coverage=notification_coverage,
            disclosure_completeness=disclosure_completeness,
            consent_capture_rate=consent_capture_rate,
            opt_out_availability=opt_out_availability,
            explanation_quality=explanation_quality,
            channel_accessibility=channel_accessibility,
            timing_appropriateness=timing_appropriateness
        )

    def _assess_notifications(
        self,
        notification_config: Dict[str, Any]
    ) -> List[ControlAssessment]:
        """Assess notification implementations."""
        assessments = []

        for notif_type, requirements in self.required_notifications.items():
            notif_config = notification_config.get(notif_type.value, {})

            # Determine status
            if notif_config.get('enabled', False):
                status = ControlStatus.IMPLEMENTED
            elif notif_config.get('planned', False):
                status = ControlStatus.PLANNED
            else:
                status = ControlStatus.NOT_IMPLEMENTED

            control = Control(
                control_id=f"notification_{notif_type.value}",
                name=f"{notif_type.value.replace('_', ' ').title()} Notification",
                description=requirements['description'],
                control_type=ControlType.SOFT,
                category=ControlCategory.DIRECTIVE,
                domain=ControlDomain.USER_NOTIFICATION,
                status=status
            )

            score = 1.0 if status == ControlStatus.IMPLEMENTED else 0.0
            findings = []
            recommendations = []
            gaps = []

            if status != ControlStatus.IMPLEMENTED:
                if requirements['mandatory']:
                    findings.append(f"Mandatory {notif_type.value} notification missing")
                    recommendations.append(f"Implement {notif_type.value} notification immediately")
                    gaps.append(f"Missing mandatory notification: {notif_type.value}")
                else:
                    findings.append(f"Optional {notif_type.value} notification not implemented")
                    recommendations.append(f"Consider implementing {notif_type.value} notification")

            assessment = ControlAssessment(
                control=control,
                score=score,
                findings=findings,
                recommendations=recommendations,
                gaps=gaps
            )
            assessments.append(assessment)

        return assessments


# =============================================================================
# Control Effectiveness Analyzer
# =============================================================================

class ControlEffectivenessAnalyzer:
    """Analyzer for measuring control effectiveness."""

    def __init__(self):
        self.effectiveness_criteria = {
            'design_effectiveness': {
                'description': 'Control is properly designed',
                'weight': 0.3
            },
            'operational_effectiveness': {
                'description': 'Control operates as intended',
                'weight': 0.4
            },
            'monitoring_effectiveness': {
                'description': 'Control is properly monitored',
                'weight': 0.3
            }
        }

    def assess_effectiveness(
        self,
        control: Control,
        test_results: Dict[str, Any]
    ) -> Tuple[EffectivenessRating, Dict[str, float]]:
        """Assess control effectiveness."""
        scores = {}

        # Design effectiveness
        design_score = test_results.get('design_review', {}).get('score', 0.0)
        scores['design_effectiveness'] = design_score

        # Operational effectiveness
        operational_tests = test_results.get('operational_tests', {})
        if operational_tests:
            op_scores = list(operational_tests.values())
            scores['operational_effectiveness'] = sum(op_scores) / len(op_scores) if op_scores else 0.0
        else:
            scores['operational_effectiveness'] = 0.0

        # Monitoring effectiveness
        monitoring = test_results.get('monitoring', {})
        monitoring_score = 0.0
        if monitoring.get('enabled', False):
            monitoring_score = 0.5
        if monitoring.get('alerts_configured', False):
            monitoring_score += 0.3
        if monitoring.get('regular_review', False):
            monitoring_score += 0.2
        scores['monitoring_effectiveness'] = monitoring_score

        # Calculate weighted overall score
        overall_score = sum(
            scores[criterion] * details['weight']
            for criterion, details in self.effectiveness_criteria.items()
        )

        # Determine rating
        if overall_score >= 0.9:
            rating = EffectivenessRating.HIGHLY_EFFECTIVE
        elif overall_score >= 0.7:
            rating = EffectivenessRating.EFFECTIVE
        elif overall_score >= 0.5:
            rating = EffectivenessRating.PARTIALLY_EFFECTIVE
        elif overall_score > 0:
            rating = EffectivenessRating.INEFFECTIVE
        else:
            rating = EffectivenessRating.NOT_TESTED

        return rating, scores

    def calculate_control_coverage(
        self,
        implemented_controls: List[Control],
        required_controls: List[str]
    ) -> Dict[str, Any]:
        """Calculate control coverage metrics."""
        implemented_ids = {c.control_id for c in implemented_controls if c.status == ControlStatus.IMPLEMENTED}
        required_set = set(required_controls)

        covered = implemented_ids & required_set
        missing = required_set - implemented_ids

        coverage = len(covered) / len(required_set) if required_set else 1.0

        return {
            'coverage_ratio': coverage,
            'covered_controls': list(covered),
            'missing_controls': list(missing),
            'total_required': len(required_set),
            'total_implemented': len(covered)
        }


# =============================================================================
# Control Gap Analyzer
# =============================================================================

class ControlGapAnalyzer:
    """Analyzer for identifying control gaps."""

    def __init__(self):
        self.risk_mappings = {
            ControlDomain.DATA_PROTECTION: RiskLevel.HIGH,
            ControlDomain.ACCESS_CONTROL: RiskLevel.HIGH,
            ControlDomain.MODEL_GOVERNANCE: RiskLevel.MEDIUM,
            ControlDomain.OUTPUT_CONTROL: RiskLevel.HIGH,
            ControlDomain.AUDIT_LOGGING: RiskLevel.MEDIUM,
            ControlDomain.USER_NOTIFICATION: RiskLevel.MEDIUM,
            ControlDomain.INCIDENT_RESPONSE: RiskLevel.HIGH,
            ControlDomain.PRIVACY: RiskLevel.CRITICAL,
            ControlDomain.SAFETY: RiskLevel.CRITICAL,
            ControlDomain.FAIRNESS: RiskLevel.HIGH,
        }

    def analyze_gaps(
        self,
        assessments: List[ControlAssessment]
    ) -> Tuple[Dict[ControlDomain, List[str]], Dict[RiskLevel, int]]:
        """Analyze control gaps by domain."""
        gaps_by_domain = defaultdict(list)
        risk_exposure = defaultdict(int)

        for assessment in assessments:
            if assessment.gaps:
                domain = assessment.control.domain
                gaps_by_domain[domain].extend(assessment.gaps)

                # Calculate risk exposure
                risk_level = self.risk_mappings.get(domain, RiskLevel.MEDIUM)
                risk_exposure[risk_level] += len(assessment.gaps)

        return dict(gaps_by_domain), dict(risk_exposure)

    def prioritize_gaps(
        self,
        gaps_by_domain: Dict[ControlDomain, List[str]],
        risk_exposure: Dict[RiskLevel, int]
    ) -> List[Tuple[str, RiskLevel, ControlDomain]]:
        """Prioritize gaps by risk level."""
        prioritized = []

        # Sort domains by risk level
        risk_priority = {
            RiskLevel.CRITICAL: 0,
            RiskLevel.HIGH: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 3
        }

        sorted_domains = sorted(
            gaps_by_domain.items(),
            key=lambda x: risk_priority.get(self.risk_mappings.get(x[0], RiskLevel.MEDIUM), 2)
        )

        for domain, gaps in sorted_domains:
            risk_level = self.risk_mappings.get(domain, RiskLevel.MEDIUM)
            for gap in gaps:
                prioritized.append((gap, risk_level, domain))

        return prioritized


# =============================================================================
# Comprehensive Control Framework Analyzer
# =============================================================================

class ControlFrameworkAnalyzer:
    """Comprehensive analyzer for control frameworks."""

    def __init__(self):
        self.hard_analyzer = HardControlAnalyzer()
        self.soft_analyzer = SoftControlAnalyzer()
        self.notification_analyzer = UserNotificationAnalyzer()
        self.effectiveness_analyzer = ControlEffectivenessAnalyzer()
        self.gap_analyzer = ControlGapAnalyzer()

    def analyze(
        self,
        hard_controls: Dict[str, Any],
        soft_controls: Dict[str, Any],
        notification_controls: Dict[str, Any],
        config: Optional[Dict] = None
    ) -> ControlFrameworkAssessment:
        """Perform comprehensive control framework analysis."""
        config = config or {}

        # Analyze hard controls
        hard_metrics, hard_assessments = self.hard_analyzer.analyze(hard_controls)
        hard_score = self._calculate_control_score(hard_metrics)

        # Analyze soft controls
        soft_metrics, soft_assessments = self.soft_analyzer.analyze(soft_controls)
        soft_score = self._calculate_soft_score(soft_metrics)

        # Analyze notifications
        notif_metrics, notif_assessments = self.notification_analyzer.analyze(notification_controls)
        notification_score = self._calculate_notification_score(notif_metrics)

        # Combine assessments
        all_assessments = hard_assessments + soft_assessments + notif_assessments

        # Analyze gaps
        gaps_by_domain, risk_exposure = self.gap_analyzer.analyze_gaps(all_assessments)

        # Calculate overall score
        weights = config.get('weights', {
            'hard': 0.4,
            'soft': 0.3,
            'notification': 0.3
        })

        overall_score = (
            hard_score * weights.get('hard', 0.4) +
            soft_score * weights.get('soft', 0.3) +
            notification_score * weights.get('notification', 0.3)
        )

        # Determine effectiveness rating
        if overall_score >= 0.9:
            effectiveness = EffectivenessRating.HIGHLY_EFFECTIVE
        elif overall_score >= 0.7:
            effectiveness = EffectivenessRating.EFFECTIVE
        elif overall_score >= 0.5:
            effectiveness = EffectivenessRating.PARTIALLY_EFFECTIVE
        else:
            effectiveness = EffectivenessRating.INEFFECTIVE

        # Generate priority actions
        priority_actions = self._generate_priority_actions(all_assessments, risk_exposure)

        # Determine maturity level
        maturity_level = self._determine_maturity(overall_score, all_assessments)

        return ControlFrameworkAssessment(
            hard_control_score=hard_score,
            soft_control_score=soft_score,
            notification_score=notification_score,
            overall_score=overall_score,
            effectiveness_rating=effectiveness,
            control_assessments=all_assessments,
            gap_analysis=gaps_by_domain,
            risk_exposure=risk_exposure,
            priority_actions=priority_actions,
            maturity_level=maturity_level
        )

    def _calculate_control_score(self, metrics: HardControlMetrics) -> float:
        """Calculate hard control score."""
        values = [
            metrics.dlp_coverage,
            metrics.encryption_coverage,
            metrics.access_control_score,
            metrics.authentication_strength,
            metrics.input_validation_coverage,
            metrics.output_filtering_coverage,
            metrics.audit_logging_coverage,
            metrics.automated_enforcement_rate
        ]
        return sum(values) / len(values)

    def _calculate_soft_score(self, metrics: SoftControlMetrics) -> float:
        """Calculate soft control score."""
        values = [
            metrics.policy_coverage,
            metrics.training_completion,
            metrics.awareness_score,
            metrics.guideline_adherence,
            metrics.review_process_maturity,
            1.0 if metrics.escalation_path_defined else 0.0,
            metrics.documentation_completeness
        ]
        return sum(values) / len(values)

    def _calculate_notification_score(self, metrics: NotificationMetrics) -> float:
        """Calculate notification score."""
        values = [
            metrics.notification_coverage,
            metrics.disclosure_completeness,
            metrics.consent_capture_rate,
            metrics.opt_out_availability,
            metrics.explanation_quality,
            metrics.channel_accessibility,
            metrics.timing_appropriateness
        ]
        return sum(values) / len(values)

    def _generate_priority_actions(
        self,
        assessments: List[ControlAssessment],
        risk_exposure: Dict[RiskLevel, int]
    ) -> List[str]:
        """Generate prioritized action items."""
        actions = []

        # Sort by gap severity
        for assessment in sorted(
            assessments,
            key=lambda a: (
                0 if a.control.domain in [ControlDomain.PRIVACY, ControlDomain.SAFETY] else 1,
                -a.score
            )
        ):
            for rec in assessment.recommendations:
                if rec not in actions:
                    actions.append(rec)
                    if len(actions) >= 10:
                        return actions

        return actions

    def _determine_maturity(
        self,
        score: float,
        assessments: List[ControlAssessment]
    ) -> str:
        """Determine control framework maturity level."""
        implemented_count = sum(
            1 for a in assessments
            if a.control.status == ControlStatus.IMPLEMENTED
        )
        total_count = len(assessments)
        implementation_rate = implemented_count / total_count if total_count > 0 else 0

        if score >= 0.9 and implementation_rate >= 0.9:
            return 'Optimized'
        elif score >= 0.75 and implementation_rate >= 0.75:
            return 'Managed'
        elif score >= 0.6 and implementation_rate >= 0.6:
            return 'Defined'
        elif score >= 0.4 and implementation_rate >= 0.4:
            return 'Developing'
        return 'Initial'


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_hard_controls(controls: Dict[str, Any]) -> Tuple[HardControlMetrics, List[ControlAssessment]]:
    """Convenience function for hard control analysis."""
    analyzer = HardControlAnalyzer()
    return analyzer.analyze(controls)


def analyze_soft_controls(controls: Dict[str, Any]) -> Tuple[SoftControlMetrics, List[ControlAssessment]]:
    """Convenience function for soft control analysis."""
    analyzer = SoftControlAnalyzer()
    return analyzer.analyze(controls)


def analyze_notifications(config: Dict[str, Any]) -> Tuple[NotificationMetrics, List[ControlAssessment]]:
    """Convenience function for notification analysis."""
    analyzer = UserNotificationAnalyzer()
    return analyzer.analyze(config)


def analyze_control_framework(
    hard_controls: Dict[str, Any],
    soft_controls: Dict[str, Any],
    notification_controls: Dict[str, Any]
) -> ControlFrameworkAssessment:
    """Convenience function for comprehensive control framework analysis."""
    analyzer = ControlFrameworkAnalyzer()
    return analyzer.analyze(hard_controls, soft_controls, notification_controls)


def identify_control_gaps(
    assessments: List[ControlAssessment]
) -> Tuple[Dict[ControlDomain, List[str]], Dict[RiskLevel, int]]:
    """Convenience function for gap analysis."""
    analyzer = ControlGapAnalyzer()
    return analyzer.analyze_gaps(assessments)
