"""
Trust Calibration Analysis Module - Pillar 1: Trust AI (Expanded)
=================================================================

Comprehensive analysis framework for AI trust calibration, confidence signaling,
trust zones, and trust failure handling following the 12-Pillar Trustworthy AI Framework.

Analysis Categories:
- Confidence Signaling Analysis: Confidence expression, signal calibration, uncertainty communication
- Trust Calibration Analysis: Calibration metrics, reliability diagrams, calibration gaps
- Trust Zone Analysis: Trust boundaries, zone transitions, zone-specific policies
- Trust Failure Handling: Failure detection, recovery mechanisms, user notification
- User Trust Dynamics: Trust evolution, repair strategies, over/under trust detection

Key Components:
- ConfidenceSignalAnalyzer: Confidence expression and calibration
- TrustZoneAnalyzer: Trust boundary and zone management
- TrustFailureAnalyzer: Trust failure detection and recovery
- TrustCalibrationMetricsAnalyzer: Calibration quality metrics
- UserTrustDynamicsAnalyzer: User trust modeling and repair
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import math


# ============================================================================
# ENUMS - Trust Classifications
# ============================================================================

class ConfidenceLevel(Enum):
    """AI confidence levels for output"""
    VERY_HIGH = "very_high"  # >95% confident
    HIGH = "high"  # 80-95% confident
    MODERATE = "moderate"  # 60-80% confident
    LOW = "low"  # 40-60% confident
    VERY_LOW = "very_low"  # <40% confident
    UNKNOWN = "unknown"  # Cannot determine confidence


class TrustZone(Enum):
    """Trust zones defining operational boundaries"""
    FULL_AUTONOMY = "full_autonomy"  # AI operates independently
    GUIDED_AUTONOMY = "guided_autonomy"  # AI with soft constraints
    SUPERVISED = "supervised"  # Human oversight required
    RESTRICTED = "restricted"  # Limited AI capabilities
    HUMAN_ONLY = "human_only"  # AI assistance disabled


class CalibrationQuality(Enum):
    """Trust calibration quality assessment"""
    EXCELLENT = "excellent"  # ECE < 0.02
    GOOD = "good"  # ECE 0.02-0.05
    MODERATE = "moderate"  # ECE 0.05-0.10
    POOR = "poor"  # ECE 0.10-0.20
    MISCALIBRATED = "miscalibrated"  # ECE > 0.20


class TrustFailureType(Enum):
    """Types of trust failures"""
    OVERCONFIDENT_ERROR = "overconfident_error"  # High confidence, wrong answer
    UNDERCONFIDENT_CORRECT = "underconfident_correct"  # Low confidence, right answer
    HALLUCINATION = "hallucination"  # Fabricated information
    INCONSISTENCY = "inconsistency"  # Contradictory outputs
    BOUNDARY_VIOLATION = "boundary_violation"  # Exceeded trust zone
    SAFETY_BREACH = "safety_breach"  # Safety constraint violation
    RELIABILITY_FAILURE = "reliability_failure"  # System reliability issue


class TrustSignalType(Enum):
    """Types of trust signals communicated to users"""
    CONFIDENCE_SCORE = "confidence_score"  # Numeric confidence
    CONFIDENCE_LABEL = "confidence_label"  # Categorical confidence
    UNCERTAINTY_BOUNDS = "uncertainty_bounds"  # Range/interval
    SOURCE_CITATION = "source_citation"  # Evidence reference
    LIMITATION_DISCLOSURE = "limitation_disclosure"  # Capability limits
    VERIFICATION_PROMPT = "verification_prompt"  # Encourage verification
    ALTERNATIVE_SUGGESTION = "alternative_suggestion"  # Other options


class TrustRepairStrategy(Enum):
    """Strategies for repairing trust after failure"""
    ACKNOWLEDGE_ERROR = "acknowledge_error"
    EXPLAIN_CAUSE = "explain_cause"
    PROVIDE_EVIDENCE = "provide_evidence"
    OFFER_ALTERNATIVES = "offer_alternatives"
    REDUCE_AUTONOMY = "reduce_autonomy"
    INCREASE_TRANSPARENCY = "increase_transparency"
    HUMAN_ESCALATION = "human_escalation"


class UserTrustState(Enum):
    """User trust state classifications"""
    OVERTRUST = "overtrust"  # User trusts AI too much
    APPROPRIATE_TRUST = "appropriate_trust"  # Calibrated trust
    UNDERTRUST = "undertrust"  # User trusts AI too little
    DISTRUST = "distrust"  # User actively distrusts AI
    UNKNOWN = "unknown"  # Trust state undetermined


# ============================================================================
# DATA CLASSES - Trust Metrics and Results
# ============================================================================

@dataclass
class ConfidenceSignal:
    """A confidence signal communicated to user"""
    signal_type: TrustSignalType
    confidence_value: float  # 0-1 numeric confidence
    confidence_level: ConfidenceLevel
    uncertainty_lower: Optional[float] = None
    uncertainty_upper: Optional[float] = None
    supporting_evidence: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    verification_suggestions: List[str] = field(default_factory=list)


@dataclass
class ConfidenceSignalingMetrics:
    """Metrics for confidence signaling analysis"""
    signals_generated: int
    signal_types_used: List[TrustSignalType]
    average_confidence: float
    confidence_distribution: Dict[ConfidenceLevel, int]
    uncertainty_expressed: bool
    limitations_disclosed: bool
    verification_prompted: bool
    signal_consistency: float  # 0-1, consistency across outputs
    user_comprehension_score: float  # 0-1, estimated comprehensibility


@dataclass
class CalibrationBin:
    """Single bin for calibration analysis"""
    bin_start: float
    bin_end: float
    bin_midpoint: float
    num_samples: int
    avg_confidence: float
    actual_accuracy: float
    calibration_gap: float  # |confidence - accuracy|


@dataclass
class CalibrationMetrics:
    """Comprehensive calibration metrics"""
    expected_calibration_error: float  # ECE
    maximum_calibration_error: float  # MCE
    average_calibration_error: float  # ACE
    calibration_quality: CalibrationQuality
    overconfidence_score: float  # Tendency to be overconfident
    underconfidence_score: float  # Tendency to be underconfident
    calibration_bins: List[CalibrationBin]
    reliability_diagram_data: List[Tuple[float, float]]
    brier_score: float  # Brier score for probabilistic predictions
    log_loss: float  # Log loss score


@dataclass
class TrustZonePolicy:
    """Policy for a specific trust zone"""
    zone: TrustZone
    allowed_actions: List[str]
    prohibited_actions: List[str]
    required_approvals: List[str]
    confidence_threshold: float  # Minimum confidence to operate
    escalation_triggers: List[str]
    audit_level: str  # none, basic, detailed, comprehensive
    human_oversight_required: bool


@dataclass
class TrustZoneMetrics:
    """Metrics for trust zone analysis"""
    current_zone: TrustZone
    zone_policies: Dict[TrustZone, TrustZonePolicy]
    zone_transitions: List[Dict[str, Any]]
    boundary_violations: int
    zone_compliance_rate: float
    escalation_count: int
    autonomous_decision_count: int
    supervised_decision_count: int


@dataclass
class TrustFailureEvent:
    """Record of a trust failure event"""
    failure_id: str
    timestamp: datetime
    failure_type: TrustFailureType
    severity: str  # low, medium, high, critical
    context: Dict[str, Any]
    confidence_at_failure: float
    expected_outcome: Any
    actual_outcome: Any
    user_impact: str
    recovery_action: str
    repair_strategy: TrustRepairStrategy
    resolved: bool


@dataclass
class TrustFailureMetrics:
    """Metrics for trust failure analysis"""
    total_failures: int
    failures_by_type: Dict[TrustFailureType, int]
    failures_by_severity: Dict[str, int]
    failure_rate: float  # Failures per total interactions
    mean_time_to_detection: float  # Average detection time
    mean_time_to_recovery: float  # Average recovery time
    recovery_success_rate: float
    repeat_failure_rate: float
    failure_events: List[TrustFailureEvent]


@dataclass
class UserTrustProfile:
    """Profile of user trust behavior"""
    user_id: str
    trust_state: UserTrustState
    trust_score: float  # 0-1 overall trust level
    overtrust_indicators: List[str]
    undertrust_indicators: List[str]
    trust_history: List[Dict[str, Any]]
    ai_reliance_rate: float  # How often user accepts AI suggestion
    verification_rate: float  # How often user verifies AI output
    override_rate: float  # How often user overrides AI
    trust_calibration_gap: float  # Difference from appropriate trust


@dataclass
class TrustDynamicsMetrics:
    """Metrics for user trust dynamics"""
    user_profiles: List[UserTrustProfile]
    average_trust_score: float
    trust_state_distribution: Dict[UserTrustState, int]
    overtrust_percentage: float
    undertrust_percentage: float
    trust_trend: str  # increasing, stable, decreasing
    repair_success_rate: float
    trust_recovery_time: float  # Average time to recover trust


@dataclass
class TrustAssessment:
    """Comprehensive trust assessment"""
    assessment_id: str
    assessment_date: datetime
    confidence_metrics: ConfidenceSignalingMetrics
    calibration_metrics: CalibrationMetrics
    zone_metrics: TrustZoneMetrics
    failure_metrics: TrustFailureMetrics
    dynamics_metrics: TrustDynamicsMetrics
    overall_trust_health: str  # healthy, warning, critical
    recommendations: List[str]
    risk_factors: List[str]


# ============================================================================
# ANALYZERS - Confidence Signaling
# ============================================================================

class ConfidenceSignalAnalyzer:
    """Analyzes confidence signaling in AI outputs"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.signal_history: List[ConfidenceSignal] = []

    def analyze_confidence_signaling(
        self,
        outputs: List[Dict[str, Any]]
    ) -> ConfidenceSignalingMetrics:
        """Analyze confidence signaling across outputs"""
        signals = [self._extract_signal(output) for output in outputs]
        self.signal_history.extend(signals)

        signal_types = list(set(s.signal_type for s in signals))
        confidence_dist = self._calculate_confidence_distribution(signals)

        return ConfidenceSignalingMetrics(
            signals_generated=len(signals),
            signal_types_used=signal_types,
            average_confidence=sum(s.confidence_value for s in signals) / len(signals) if signals else 0,
            confidence_distribution=confidence_dist,
            uncertainty_expressed=any(s.uncertainty_lower is not None for s in signals),
            limitations_disclosed=any(len(s.limitations) > 0 for s in signals),
            verification_prompted=any(len(s.verification_suggestions) > 0 for s in signals),
            signal_consistency=self._calculate_signal_consistency(signals),
            user_comprehension_score=self._estimate_comprehensibility(signals)
        )

    def _extract_signal(self, output: Dict[str, Any]) -> ConfidenceSignal:
        """Extract confidence signal from output"""
        confidence = output.get('confidence', 0.5)

        return ConfidenceSignal(
            signal_type=self._determine_signal_type(output),
            confidence_value=confidence,
            confidence_level=self._map_to_confidence_level(confidence),
            uncertainty_lower=output.get('uncertainty_lower'),
            uncertainty_upper=output.get('uncertainty_upper'),
            supporting_evidence=output.get('evidence', []),
            limitations=output.get('limitations', []),
            verification_suggestions=output.get('verify', [])
        )

    def _determine_signal_type(self, output: Dict[str, Any]) -> TrustSignalType:
        """Determine the type of trust signal"""
        if 'confidence_score' in output:
            return TrustSignalType.CONFIDENCE_SCORE
        elif 'confidence_label' in output:
            return TrustSignalType.CONFIDENCE_LABEL
        elif 'uncertainty_bounds' in output:
            return TrustSignalType.UNCERTAINTY_BOUNDS
        elif 'sources' in output:
            return TrustSignalType.SOURCE_CITATION
        else:
            return TrustSignalType.CONFIDENCE_SCORE

    def _map_to_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Map numeric confidence to confidence level"""
        if confidence >= 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.80:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.60:
            return ConfidenceLevel.MODERATE
        elif confidence >= 0.40:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _calculate_confidence_distribution(
        self,
        signals: List[ConfidenceSignal]
    ) -> Dict[ConfidenceLevel, int]:
        """Calculate distribution of confidence levels"""
        distribution = {level: 0 for level in ConfidenceLevel}
        for signal in signals:
            distribution[signal.confidence_level] += 1
        return distribution

    def _calculate_signal_consistency(
        self,
        signals: List[ConfidenceSignal]
    ) -> float:
        """Calculate consistency of confidence signals"""
        if len(signals) < 2:
            return 1.0

        confidences = [s.confidence_value for s in signals]
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)

        # Lower variance = higher consistency
        return max(0, 1 - math.sqrt(variance))

    def _estimate_comprehensibility(
        self,
        signals: List[ConfidenceSignal]
    ) -> float:
        """Estimate how comprehensible signals are to users"""
        score = 0.5  # Base score

        # Categorical labels are more comprehensible
        categorical_ratio = sum(
            1 for s in signals if s.signal_type == TrustSignalType.CONFIDENCE_LABEL
        ) / len(signals) if signals else 0
        score += categorical_ratio * 0.2

        # Limitations disclosed improve comprehension
        limitation_ratio = sum(
            1 for s in signals if s.limitations
        ) / len(signals) if signals else 0
        score += limitation_ratio * 0.15

        # Verification suggestions help
        verify_ratio = sum(
            1 for s in signals if s.verification_suggestions
        ) / len(signals) if signals else 0
        score += verify_ratio * 0.15

        return min(1.0, score)

    def generate_confidence_signal(
        self,
        prediction: Any,
        confidence: float,
        evidence: List[str] = None,
        limitations: List[str] = None
    ) -> ConfidenceSignal:
        """Generate a confidence signal for a prediction"""
        signal = ConfidenceSignal(
            signal_type=TrustSignalType.CONFIDENCE_SCORE,
            confidence_value=confidence,
            confidence_level=self._map_to_confidence_level(confidence),
            uncertainty_lower=max(0, confidence - 0.1),
            uncertainty_upper=min(1, confidence + 0.1),
            supporting_evidence=evidence or [],
            limitations=limitations or [],
            verification_suggestions=self._generate_verification_suggestions(confidence)
        )

        self.signal_history.append(signal)
        return signal

    def _generate_verification_suggestions(
        self,
        confidence: float
    ) -> List[str]:
        """Generate verification suggestions based on confidence"""
        suggestions = []

        if confidence < 0.8:
            suggestions.append("Consider verifying this information with additional sources")

        if confidence < 0.6:
            suggestions.append("This response has moderate uncertainty - cross-check recommended")

        if confidence < 0.4:
            suggestions.append("High uncertainty - human expert review recommended")

        return suggestions


class UncertaintyCommunicationAnalyzer:
    """Analyzes uncertainty communication effectiveness"""

    def __init__(self):
        self.communication_patterns: List[Dict[str, Any]] = []

    def analyze_uncertainty_communication(
        self,
        outputs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze how well uncertainty is communicated"""
        analysis = {
            'quantitative_uncertainty': 0,
            'qualitative_uncertainty': 0,
            'no_uncertainty': 0,
            'communication_quality': 0.0,
            'recommendations': []
        }

        for output in outputs:
            if 'uncertainty_bounds' in output or 'confidence_interval' in output:
                analysis['quantitative_uncertainty'] += 1
            elif 'uncertainty_label' in output or 'confidence_level' in output:
                analysis['qualitative_uncertainty'] += 1
            else:
                analysis['no_uncertainty'] += 1

        total = len(outputs) or 1
        communicated = analysis['quantitative_uncertainty'] + analysis['qualitative_uncertainty']

        analysis['communication_rate'] = communicated / total
        analysis['communication_quality'] = self._assess_quality(outputs)

        if analysis['no_uncertainty'] > total * 0.3:
            analysis['recommendations'].append(
                "Increase uncertainty communication in outputs"
            )

        return analysis

    def _assess_quality(self, outputs: List[Dict[str, Any]]) -> float:
        """Assess quality of uncertainty communication"""
        if not outputs:
            return 0.0

        scores = []
        for output in outputs:
            score = 0.0
            if 'uncertainty_bounds' in output:
                score += 0.4
            if 'confidence_level' in output:
                score += 0.2
            if 'limitations' in output:
                score += 0.2
            if 'verification_suggestions' in output:
                score += 0.2
            scores.append(score)

        return sum(scores) / len(scores)


# ============================================================================
# ANALYZERS - Trust Calibration
# ============================================================================

class TrustCalibrationMetricsAnalyzer:
    """Analyzes trust calibration quality"""

    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins
        self.calibration_history: List[CalibrationMetrics] = []

    def analyze_calibration(
        self,
        predictions: List[Dict[str, Any]]
    ) -> CalibrationMetrics:
        """Analyze calibration of predictions"""
        bins = self._create_calibration_bins(predictions)

        ece = self._calculate_ece(bins)
        mce = self._calculate_mce(bins)
        ace = self._calculate_ace(bins)

        overconf, underconf = self._calculate_confidence_bias(bins)

        metrics = CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            average_calibration_error=ace,
            calibration_quality=self._assess_calibration_quality(ece),
            overconfidence_score=overconf,
            underconfidence_score=underconf,
            calibration_bins=bins,
            reliability_diagram_data=[(b.avg_confidence, b.actual_accuracy) for b in bins],
            brier_score=self._calculate_brier_score(predictions),
            log_loss=self._calculate_log_loss(predictions)
        )

        self.calibration_history.append(metrics)
        return metrics

    def _create_calibration_bins(
        self,
        predictions: List[Dict[str, Any]]
    ) -> List[CalibrationBin]:
        """Create calibration bins from predictions"""
        bins = []
        bin_width = 1.0 / self.num_bins

        for i in range(self.num_bins):
            bin_start = i * bin_width
            bin_end = (i + 1) * bin_width

            bin_preds = [
                p for p in predictions
                if bin_start <= p.get('confidence', 0) < bin_end
            ]

            if bin_preds:
                avg_conf = sum(p.get('confidence', 0) for p in bin_preds) / len(bin_preds)
                accuracy = sum(1 for p in bin_preds if p.get('correct', False)) / len(bin_preds)
            else:
                avg_conf = (bin_start + bin_end) / 2
                accuracy = 0

            bins.append(CalibrationBin(
                bin_start=bin_start,
                bin_end=bin_end,
                bin_midpoint=(bin_start + bin_end) / 2,
                num_samples=len(bin_preds),
                avg_confidence=avg_conf,
                actual_accuracy=accuracy,
                calibration_gap=abs(avg_conf - accuracy)
            ))

        return bins

    def _calculate_ece(self, bins: List[CalibrationBin]) -> float:
        """Calculate Expected Calibration Error"""
        total_samples = sum(b.num_samples for b in bins)
        if total_samples == 0:
            return 0.0

        ece = sum(
            (b.num_samples / total_samples) * b.calibration_gap
            for b in bins
        )
        return ece

    def _calculate_mce(self, bins: List[CalibrationBin]) -> float:
        """Calculate Maximum Calibration Error"""
        if not bins:
            return 0.0
        return max(b.calibration_gap for b in bins)

    def _calculate_ace(self, bins: List[CalibrationBin]) -> float:
        """Calculate Average Calibration Error"""
        non_empty_bins = [b for b in bins if b.num_samples > 0]
        if not non_empty_bins:
            return 0.0
        return sum(b.calibration_gap for b in non_empty_bins) / len(non_empty_bins)

    def _calculate_confidence_bias(
        self,
        bins: List[CalibrationBin]
    ) -> Tuple[float, float]:
        """Calculate overconfidence and underconfidence scores"""
        overconf_sum = 0.0
        underconf_sum = 0.0
        total_samples = sum(b.num_samples for b in bins) or 1

        for b in bins:
            if b.avg_confidence > b.actual_accuracy:
                overconf_sum += b.num_samples * (b.avg_confidence - b.actual_accuracy)
            else:
                underconf_sum += b.num_samples * (b.actual_accuracy - b.avg_confidence)

        return overconf_sum / total_samples, underconf_sum / total_samples

    def _assess_calibration_quality(self, ece: float) -> CalibrationQuality:
        """Assess calibration quality from ECE"""
        if ece < 0.02:
            return CalibrationQuality.EXCELLENT
        elif ece < 0.05:
            return CalibrationQuality.GOOD
        elif ece < 0.10:
            return CalibrationQuality.MODERATE
        elif ece < 0.20:
            return CalibrationQuality.POOR
        else:
            return CalibrationQuality.MISCALIBRATED

    def _calculate_brier_score(
        self,
        predictions: List[Dict[str, Any]]
    ) -> float:
        """Calculate Brier score"""
        if not predictions:
            return 0.0

        score = sum(
            (p.get('confidence', 0) - (1 if p.get('correct', False) else 0)) ** 2
            for p in predictions
        ) / len(predictions)

        return score

    def _calculate_log_loss(
        self,
        predictions: List[Dict[str, Any]]
    ) -> float:
        """Calculate log loss"""
        if not predictions:
            return 0.0

        eps = 1e-15  # Prevent log(0)
        loss = 0.0

        for p in predictions:
            conf = max(eps, min(1 - eps, p.get('confidence', 0.5)))
            correct = 1 if p.get('correct', False) else 0
            loss -= correct * math.log(conf) + (1 - correct) * math.log(1 - conf)

        return loss / len(predictions)

    def get_calibration_recommendations(
        self,
        metrics: CalibrationMetrics
    ) -> List[str]:
        """Generate recommendations for improving calibration"""
        recommendations = []

        if metrics.calibration_quality in [CalibrationQuality.POOR, CalibrationQuality.MISCALIBRATED]:
            recommendations.append(
                "Model is significantly miscalibrated - implement temperature scaling or Platt scaling"
            )

        if metrics.overconfidence_score > 0.1:
            recommendations.append(
                f"Model shows overconfidence (score: {metrics.overconfidence_score:.3f}) - "
                "consider confidence dampening or ensemble methods"
            )

        if metrics.underconfidence_score > 0.1:
            recommendations.append(
                f"Model shows underconfidence (score: {metrics.underconfidence_score:.3f}) - "
                "review confidence estimation methodology"
            )

        return recommendations


class ReliabilityDiagramAnalyzer:
    """Analyzes reliability diagrams for calibration visualization"""

    def __init__(self):
        self.diagram_data: List[Dict[str, Any]] = []

    def generate_reliability_diagram_data(
        self,
        calibration_metrics: CalibrationMetrics
    ) -> Dict[str, Any]:
        """Generate data for reliability diagram visualization"""
        diagram = {
            'bins': [],
            'perfect_calibration_line': [(0, 0), (1, 1)],
            'calibration_gap_areas': [],
            'ece': calibration_metrics.expected_calibration_error,
            'quality': calibration_metrics.calibration_quality.value
        }

        for bin_data in calibration_metrics.calibration_bins:
            diagram['bins'].append({
                'confidence': bin_data.avg_confidence,
                'accuracy': bin_data.actual_accuracy,
                'samples': bin_data.num_samples,
                'gap': bin_data.calibration_gap
            })

            # Calculate gap area for visualization
            if bin_data.num_samples > 0:
                diagram['calibration_gap_areas'].append({
                    'x': bin_data.bin_midpoint,
                    'y_pred': bin_data.avg_confidence,
                    'y_actual': bin_data.actual_accuracy,
                    'gap': bin_data.calibration_gap
                })

        self.diagram_data.append(diagram)
        return diagram


# ============================================================================
# ANALYZERS - Trust Zones
# ============================================================================

class TrustZoneAnalyzer:
    """Analyzes trust zones and boundary management"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.zone_policies: Dict[TrustZone, TrustZonePolicy] = self._initialize_policies()
        self.transition_history: List[Dict[str, Any]] = []

    def _initialize_policies(self) -> Dict[TrustZone, TrustZonePolicy]:
        """Initialize default trust zone policies"""
        return {
            TrustZone.FULL_AUTONOMY: TrustZonePolicy(
                zone=TrustZone.FULL_AUTONOMY,
                allowed_actions=['all'],
                prohibited_actions=[],
                required_approvals=[],
                confidence_threshold=0.95,
                escalation_triggers=['safety_concern', 'ethical_issue'],
                audit_level='basic',
                human_oversight_required=False
            ),
            TrustZone.GUIDED_AUTONOMY: TrustZonePolicy(
                zone=TrustZone.GUIDED_AUTONOMY,
                allowed_actions=['standard_operations', 'recommendations'],
                prohibited_actions=['high_risk_actions'],
                required_approvals=['high_impact_decisions'],
                confidence_threshold=0.80,
                escalation_triggers=['uncertainty', 'policy_violation'],
                audit_level='detailed',
                human_oversight_required=False
            ),
            TrustZone.SUPERVISED: TrustZonePolicy(
                zone=TrustZone.SUPERVISED,
                allowed_actions=['suggestions', 'analysis'],
                prohibited_actions=['autonomous_actions', 'modifications'],
                required_approvals=['all_actions'],
                confidence_threshold=0.60,
                escalation_triggers=['any_decision'],
                audit_level='comprehensive',
                human_oversight_required=True
            ),
            TrustZone.RESTRICTED: TrustZonePolicy(
                zone=TrustZone.RESTRICTED,
                allowed_actions=['read_only', 'basic_queries'],
                prohibited_actions=['modifications', 'recommendations'],
                required_approvals=['all'],
                confidence_threshold=0.0,
                escalation_triggers=['any_action'],
                audit_level='comprehensive',
                human_oversight_required=True
            ),
            TrustZone.HUMAN_ONLY: TrustZonePolicy(
                zone=TrustZone.HUMAN_ONLY,
                allowed_actions=[],
                prohibited_actions=['all'],
                required_approvals=[],
                confidence_threshold=1.0,
                escalation_triggers=[],
                audit_level='none',
                human_oversight_required=True
            )
        }

    def analyze_trust_zones(
        self,
        operations: List[Dict[str, Any]],
        current_zone: TrustZone
    ) -> TrustZoneMetrics:
        """Analyze trust zone compliance and transitions"""
        violations = 0
        escalations = 0
        autonomous = 0
        supervised = 0
        transitions = []

        policy = self.zone_policies[current_zone]

        for op in operations:
            action = op.get('action', '')

            # Check for violations
            if action in policy.prohibited_actions:
                violations += 1

            # Check for escalations
            if any(trigger in str(op) for trigger in policy.escalation_triggers):
                escalations += 1

            # Count operation types
            if op.get('autonomous', False):
                autonomous += 1
            else:
                supervised += 1

            # Track zone transitions
            if 'zone_change' in op:
                transition = {
                    'from_zone': current_zone.value,
                    'to_zone': op['zone_change'],
                    'reason': op.get('reason', 'unspecified'),
                    'timestamp': op.get('timestamp', datetime.now().isoformat())
                }
                transitions.append(transition)
                self.transition_history.append(transition)

        total_ops = len(operations) or 1

        return TrustZoneMetrics(
            current_zone=current_zone,
            zone_policies=self.zone_policies,
            zone_transitions=transitions,
            boundary_violations=violations,
            zone_compliance_rate=(total_ops - violations) / total_ops,
            escalation_count=escalations,
            autonomous_decision_count=autonomous,
            supervised_decision_count=supervised
        )

    def determine_appropriate_zone(
        self,
        confidence: float,
        risk_level: str,
        context: Dict[str, Any]
    ) -> TrustZone:
        """Determine appropriate trust zone based on factors"""
        # High risk always requires supervision
        if risk_level == 'critical':
            return TrustZone.HUMAN_ONLY
        elif risk_level == 'high':
            return TrustZone.SUPERVISED

        # Confidence-based zone selection
        if confidence >= 0.95 and risk_level == 'low':
            return TrustZone.FULL_AUTONOMY
        elif confidence >= 0.80:
            return TrustZone.GUIDED_AUTONOMY
        elif confidence >= 0.60:
            return TrustZone.SUPERVISED
        else:
            return TrustZone.RESTRICTED

    def validate_zone_transition(
        self,
        from_zone: TrustZone,
        to_zone: TrustZone,
        justification: str
    ) -> Dict[str, Any]:
        """Validate a zone transition request"""
        # Define valid transitions
        valid_transitions = {
            TrustZone.HUMAN_ONLY: [TrustZone.RESTRICTED],
            TrustZone.RESTRICTED: [TrustZone.SUPERVISED, TrustZone.HUMAN_ONLY],
            TrustZone.SUPERVISED: [TrustZone.GUIDED_AUTONOMY, TrustZone.RESTRICTED],
            TrustZone.GUIDED_AUTONOMY: [TrustZone.FULL_AUTONOMY, TrustZone.SUPERVISED],
            TrustZone.FULL_AUTONOMY: [TrustZone.GUIDED_AUTONOMY]
        }

        is_valid = to_zone in valid_transitions.get(from_zone, [])

        return {
            'valid': is_valid,
            'from_zone': from_zone.value,
            'to_zone': to_zone.value,
            'justification': justification,
            'reason': 'Valid transition' if is_valid else 'Invalid zone transition - must progress incrementally'
        }


class TrustBoundaryAnalyzer:
    """Analyzes trust boundaries and enforcement"""

    def __init__(self):
        self.boundary_violations: List[Dict[str, Any]] = []

    def analyze_boundary_enforcement(
        self,
        zone_metrics: TrustZoneMetrics
    ) -> Dict[str, Any]:
        """Analyze how well trust boundaries are enforced"""
        analysis = {
            'enforcement_rate': zone_metrics.zone_compliance_rate,
            'violations': zone_metrics.boundary_violations,
            'escalation_effectiveness': self._assess_escalation_effectiveness(zone_metrics),
            'boundary_clarity': self._assess_boundary_clarity(zone_metrics),
            'recommendations': []
        }

        if zone_metrics.zone_compliance_rate < 0.95:
            analysis['recommendations'].append(
                "Improve boundary enforcement mechanisms"
            )

        if zone_metrics.boundary_violations > 0:
            analysis['recommendations'].append(
                f"Address {zone_metrics.boundary_violations} boundary violations"
            )

        return analysis

    def _assess_escalation_effectiveness(
        self,
        metrics: TrustZoneMetrics
    ) -> float:
        """Assess how effective escalation triggers are"""
        # Higher escalation rate in supervised zones is expected
        if metrics.current_zone in [TrustZone.SUPERVISED, TrustZone.RESTRICTED]:
            return min(1.0, metrics.escalation_count / max(1, metrics.supervised_decision_count))
        return 0.8  # Default for autonomous zones

    def _assess_boundary_clarity(
        self,
        metrics: TrustZoneMetrics
    ) -> float:
        """Assess clarity of trust boundaries"""
        policy = metrics.zone_policies.get(metrics.current_zone)
        if not policy:
            return 0.5

        clarity_score = 0.0
        if policy.allowed_actions:
            clarity_score += 0.25
        if policy.prohibited_actions:
            clarity_score += 0.25
        if policy.escalation_triggers:
            clarity_score += 0.25
        if policy.confidence_threshold > 0:
            clarity_score += 0.25

        return clarity_score


# ============================================================================
# ANALYZERS - Trust Failure
# ============================================================================

class TrustFailureAnalyzer:
    """Analyzes trust failures and recovery"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.failure_registry: List[TrustFailureEvent] = []

    def analyze_trust_failures(
        self,
        failure_events: List[Dict[str, Any]]
    ) -> TrustFailureMetrics:
        """Analyze trust failure patterns"""
        events = [self._parse_failure_event(e) for e in failure_events]
        self.failure_registry.extend(events)

        failures_by_type = {}
        failures_by_severity = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}

        for event in events:
            failures_by_type[event.failure_type] = failures_by_type.get(event.failure_type, 0) + 1
            failures_by_severity[event.severity] = failures_by_severity.get(event.severity, 0) + 1

        recovery_success = sum(1 for e in events if e.resolved)

        return TrustFailureMetrics(
            total_failures=len(events),
            failures_by_type=failures_by_type,
            failures_by_severity=failures_by_severity,
            failure_rate=len(events) / max(1, self.config.get('total_interactions', 1000)),
            mean_time_to_detection=self._calculate_mean_detection_time(events),
            mean_time_to_recovery=self._calculate_mean_recovery_time(events),
            recovery_success_rate=recovery_success / len(events) if events else 1.0,
            repeat_failure_rate=self._calculate_repeat_rate(events),
            failure_events=events
        )

    def _parse_failure_event(self, event_data: Dict[str, Any]) -> TrustFailureEvent:
        """Parse failure event from raw data"""
        return TrustFailureEvent(
            failure_id=event_data.get('id', f"fail_{datetime.now().timestamp()}"),
            timestamp=datetime.fromisoformat(event_data.get('timestamp', datetime.now().isoformat())),
            failure_type=TrustFailureType(event_data.get('type', 'reliability_failure')),
            severity=event_data.get('severity', 'medium'),
            context=event_data.get('context', {}),
            confidence_at_failure=event_data.get('confidence', 0.5),
            expected_outcome=event_data.get('expected'),
            actual_outcome=event_data.get('actual'),
            user_impact=event_data.get('impact', 'unknown'),
            recovery_action=event_data.get('recovery_action', ''),
            repair_strategy=TrustRepairStrategy(event_data.get('repair_strategy', 'acknowledge_error')),
            resolved=event_data.get('resolved', False)
        )

    def _calculate_mean_detection_time(
        self,
        events: List[TrustFailureEvent]
    ) -> float:
        """Calculate mean time to detection"""
        # Simplified - would need actual detection timestamps
        return 5.0  # Default 5 seconds

    def _calculate_mean_recovery_time(
        self,
        events: List[TrustFailureEvent]
    ) -> float:
        """Calculate mean time to recovery"""
        # Simplified - would need actual recovery timestamps
        return 30.0  # Default 30 seconds

    def _calculate_repeat_rate(
        self,
        events: List[TrustFailureEvent]
    ) -> float:
        """Calculate repeat failure rate"""
        if len(events) < 2:
            return 0.0

        type_counts = {}
        for event in events:
            type_counts[event.failure_type] = type_counts.get(event.failure_type, 0) + 1

        repeats = sum(1 for count in type_counts.values() if count > 1)
        return repeats / len(type_counts) if type_counts else 0.0

    def determine_repair_strategy(
        self,
        failure: TrustFailureEvent
    ) -> TrustRepairStrategy:
        """Determine appropriate repair strategy for a failure"""
        if failure.severity == 'critical':
            return TrustRepairStrategy.HUMAN_ESCALATION

        if failure.failure_type == TrustFailureType.OVERCONFIDENT_ERROR:
            return TrustRepairStrategy.ACKNOWLEDGE_ERROR

        if failure.failure_type == TrustFailureType.HALLUCINATION:
            return TrustRepairStrategy.PROVIDE_EVIDENCE

        if failure.failure_type == TrustFailureType.INCONSISTENCY:
            return TrustRepairStrategy.EXPLAIN_CAUSE

        if failure.failure_type == TrustFailureType.BOUNDARY_VIOLATION:
            return TrustRepairStrategy.REDUCE_AUTONOMY

        return TrustRepairStrategy.INCREASE_TRANSPARENCY

    def generate_recovery_plan(
        self,
        failure: TrustFailureEvent
    ) -> Dict[str, Any]:
        """Generate a recovery plan for a trust failure"""
        strategy = self.determine_repair_strategy(failure)

        plan = {
            'failure_id': failure.failure_id,
            'strategy': strategy.value,
            'immediate_actions': [],
            'communication_approach': '',
            'follow_up_actions': [],
            'prevention_measures': []
        }

        if strategy == TrustRepairStrategy.ACKNOWLEDGE_ERROR:
            plan['immediate_actions'] = ['Issue correction', 'Apologize for error']
            plan['communication_approach'] = 'Direct acknowledgment with corrected information'

        elif strategy == TrustRepairStrategy.EXPLAIN_CAUSE:
            plan['immediate_actions'] = ['Analyze root cause', 'Document findings']
            plan['communication_approach'] = 'Transparent explanation of what went wrong'

        elif strategy == TrustRepairStrategy.PROVIDE_EVIDENCE:
            plan['immediate_actions'] = ['Gather supporting evidence', 'Cite sources']
            plan['communication_approach'] = 'Evidence-based correction with citations'

        elif strategy == TrustRepairStrategy.HUMAN_ESCALATION:
            plan['immediate_actions'] = ['Notify human operator', 'Pause autonomous operations']
            plan['communication_approach'] = 'Escalate to human review'

        plan['follow_up_actions'] = ['Monitor for recurrence', 'Update trust model']
        plan['prevention_measures'] = ['Review confidence calibration', 'Add guardrails']

        return plan


# ============================================================================
# ANALYZERS - User Trust Dynamics
# ============================================================================

class UserTrustDynamicsAnalyzer:
    """Analyzes user trust dynamics and patterns"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.user_profiles: Dict[str, UserTrustProfile] = {}

    def analyze_user_trust(
        self,
        user_interactions: List[Dict[str, Any]]
    ) -> TrustDynamicsMetrics:
        """Analyze user trust dynamics from interactions"""
        # Group interactions by user
        by_user: Dict[str, List[Dict[str, Any]]] = {}
        for interaction in user_interactions:
            user_id = interaction.get('user_id', 'anonymous')
            if user_id not in by_user:
                by_user[user_id] = []
            by_user[user_id].append(interaction)

        # Analyze each user
        profiles = []
        for user_id, interactions in by_user.items():
            profile = self._analyze_user_profile(user_id, interactions)
            profiles.append(profile)
            self.user_profiles[user_id] = profile

        # Calculate aggregate metrics
        trust_states = {state: 0 for state in UserTrustState}
        for profile in profiles:
            trust_states[profile.trust_state] += 1

        overtrust_count = trust_states[UserTrustState.OVERTRUST]
        undertrust_count = trust_states[UserTrustState.UNDERTRUST]
        total = len(profiles) or 1

        return TrustDynamicsMetrics(
            user_profiles=profiles,
            average_trust_score=sum(p.trust_score for p in profiles) / total,
            trust_state_distribution=trust_states,
            overtrust_percentage=overtrust_count / total,
            undertrust_percentage=undertrust_count / total,
            trust_trend=self._determine_trust_trend(profiles),
            repair_success_rate=self._calculate_repair_success(profiles),
            trust_recovery_time=self._calculate_recovery_time(profiles)
        )

    def _analyze_user_profile(
        self,
        user_id: str,
        interactions: List[Dict[str, Any]]
    ) -> UserTrustProfile:
        """Analyze individual user trust profile"""
        accepts = sum(1 for i in interactions if i.get('accepted', False))
        verifies = sum(1 for i in interactions if i.get('verified', False))
        overrides = sum(1 for i in interactions if i.get('overridden', False))
        total = len(interactions) or 1

        reliance_rate = accepts / total
        verification_rate = verifies / total
        override_rate = overrides / total

        trust_state = self._determine_trust_state(reliance_rate, verification_rate)
        trust_score = self._calculate_trust_score(reliance_rate, verification_rate, override_rate)

        return UserTrustProfile(
            user_id=user_id,
            trust_state=trust_state,
            trust_score=trust_score,
            overtrust_indicators=self._identify_overtrust_indicators(interactions),
            undertrust_indicators=self._identify_undertrust_indicators(interactions),
            trust_history=[{'timestamp': i.get('timestamp'), 'action': i.get('action')} for i in interactions[-10:]],
            ai_reliance_rate=reliance_rate,
            verification_rate=verification_rate,
            override_rate=override_rate,
            trust_calibration_gap=abs(trust_score - 0.7)  # 0.7 as ideal trust level
        )

    def _determine_trust_state(
        self,
        reliance_rate: float,
        verification_rate: float
    ) -> UserTrustState:
        """Determine user trust state"""
        if reliance_rate > 0.9 and verification_rate < 0.1:
            return UserTrustState.OVERTRUST
        elif reliance_rate < 0.3:
            return UserTrustState.DISTRUST
        elif reliance_rate < 0.5 or verification_rate > 0.8:
            return UserTrustState.UNDERTRUST
        else:
            return UserTrustState.APPROPRIATE_TRUST

    def _calculate_trust_score(
        self,
        reliance: float,
        verification: float,
        override: float
    ) -> float:
        """Calculate overall trust score"""
        # Balanced trust = moderate reliance with reasonable verification
        ideal_reliance = 0.7
        ideal_verification = 0.3

        reliance_factor = 1 - abs(reliance - ideal_reliance)
        verification_factor = 1 - abs(verification - ideal_verification)
        override_factor = 1 - override  # Lower override is better

        return (reliance_factor + verification_factor + override_factor) / 3

    def _identify_overtrust_indicators(
        self,
        interactions: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify indicators of overtrust"""
        indicators = []

        accepts_without_check = sum(
            1 for i in interactions
            if i.get('accepted', False) and not i.get('verified', False)
        )

        if accepts_without_check > len(interactions) * 0.8:
            indicators.append("Accepts AI output without verification")

        low_conf_accepts = sum(
            1 for i in interactions
            if i.get('accepted', False) and i.get('ai_confidence', 1.0) < 0.6
        )

        if low_conf_accepts > len(interactions) * 0.3:
            indicators.append("Accepts low-confidence outputs")

        return indicators

    def _identify_undertrust_indicators(
        self,
        interactions: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify indicators of undertrust"""
        indicators = []

        rejects_high_conf = sum(
            1 for i in interactions
            if not i.get('accepted', True) and i.get('ai_confidence', 0) > 0.9
        )

        if rejects_high_conf > len(interactions) * 0.3:
            indicators.append("Frequently rejects high-confidence outputs")

        excessive_verification = sum(
            1 for i in interactions if i.get('verified', False)
        )

        if excessive_verification > len(interactions) * 0.8:
            indicators.append("Excessively verifies all outputs")

        return indicators

    def _determine_trust_trend(
        self,
        profiles: List[UserTrustProfile]
    ) -> str:
        """Determine overall trust trend"""
        if not profiles:
            return 'stable'

        # Simplified trend analysis
        avg_score = sum(p.trust_score for p in profiles) / len(profiles)

        if avg_score > 0.7:
            return 'healthy'
        elif avg_score > 0.5:
            return 'stable'
        else:
            return 'declining'

    def _calculate_repair_success(
        self,
        profiles: List[UserTrustProfile]
    ) -> float:
        """Calculate trust repair success rate"""
        return 0.75  # Placeholder

    def _calculate_recovery_time(
        self,
        profiles: List[UserTrustProfile]
    ) -> float:
        """Calculate average trust recovery time"""
        return 24.0  # Placeholder - hours


class OvertrustPreventionAnalyzer:
    """Analyzes and prevents overtrust patterns"""

    def __init__(self):
        self.interventions: List[Dict[str, Any]] = []

    def analyze_overtrust_risk(
        self,
        user_profile: UserTrustProfile
    ) -> Dict[str, Any]:
        """Analyze risk of overtrust for a user"""
        risk_analysis = {
            'user_id': user_profile.user_id,
            'overtrust_risk': 'low',
            'risk_factors': [],
            'recommended_interventions': []
        }

        if user_profile.trust_state == UserTrustState.OVERTRUST:
            risk_analysis['overtrust_risk'] = 'high'
            risk_analysis['risk_factors'] = user_profile.overtrust_indicators
            risk_analysis['recommended_interventions'] = [
                'Display confidence scores more prominently',
                'Add verification prompts for important decisions',
                'Provide alternative options for comparison'
            ]
        elif user_profile.ai_reliance_rate > 0.85:
            risk_analysis['overtrust_risk'] = 'moderate'
            risk_analysis['risk_factors'].append('High AI reliance rate')
            risk_analysis['recommended_interventions'].append(
                'Encourage occasional manual verification'
            )

        return risk_analysis

    def generate_intervention(
        self,
        risk_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate intervention for overtrust"""
        intervention = {
            'type': 'overtrust_prevention',
            'user_id': risk_analysis['user_id'],
            'actions': risk_analysis['recommended_interventions'],
            'ui_changes': [],
            'messaging': []
        }

        if risk_analysis['overtrust_risk'] == 'high':
            intervention['ui_changes'].append('Enable mandatory verification step')
            intervention['messaging'].append(
                'Remember to verify AI suggestions for important decisions'
            )

        self.interventions.append(intervention)
        return intervention


# ============================================================================
# COMPREHENSIVE ANALYZER
# ============================================================================

class TrustCalibrationAnalyzer:
    """Comprehensive trust calibration analyzer"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.signal_analyzer = ConfidenceSignalAnalyzer(config)
        self.calibration_analyzer = TrustCalibrationMetricsAnalyzer()
        self.zone_analyzer = TrustZoneAnalyzer(config)
        self.failure_analyzer = TrustFailureAnalyzer(config)
        self.dynamics_analyzer = UserTrustDynamicsAnalyzer(config)
        self.assessments: List[TrustAssessment] = []

    def analyze_trust(
        self,
        outputs: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        operations: List[Dict[str, Any]],
        failure_events: List[Dict[str, Any]],
        user_interactions: List[Dict[str, Any]],
        current_zone: TrustZone
    ) -> TrustAssessment:
        """Perform comprehensive trust analysis"""
        # Run all analyzers
        confidence_metrics = self.signal_analyzer.analyze_confidence_signaling(outputs)
        calibration_metrics = self.calibration_analyzer.analyze_calibration(predictions)
        zone_metrics = self.zone_analyzer.analyze_trust_zones(operations, current_zone)
        failure_metrics = self.failure_analyzer.analyze_trust_failures(failure_events)
        dynamics_metrics = self.dynamics_analyzer.analyze_user_trust(user_interactions)

        # Determine overall health
        overall_health = self._assess_overall_health(
            confidence_metrics, calibration_metrics, zone_metrics,
            failure_metrics, dynamics_metrics
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            calibration_metrics, zone_metrics, failure_metrics, dynamics_metrics
        )

        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            calibration_metrics, failure_metrics, dynamics_metrics
        )

        assessment = TrustAssessment(
            assessment_id=f"trust_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            assessment_date=datetime.now(),
            confidence_metrics=confidence_metrics,
            calibration_metrics=calibration_metrics,
            zone_metrics=zone_metrics,
            failure_metrics=failure_metrics,
            dynamics_metrics=dynamics_metrics,
            overall_trust_health=overall_health,
            recommendations=recommendations,
            risk_factors=risk_factors
        )

        self.assessments.append(assessment)
        return assessment

    def _assess_overall_health(
        self,
        confidence: ConfidenceSignalingMetrics,
        calibration: CalibrationMetrics,
        zones: TrustZoneMetrics,
        failures: TrustFailureMetrics,
        dynamics: TrustDynamicsMetrics
    ) -> str:
        """Assess overall trust health"""
        health_score = 0.0

        # Calibration quality
        if calibration.calibration_quality in [CalibrationQuality.EXCELLENT, CalibrationQuality.GOOD]:
            health_score += 0.25
        elif calibration.calibration_quality == CalibrationQuality.MODERATE:
            health_score += 0.15

        # Zone compliance
        health_score += zones.zone_compliance_rate * 0.25

        # Failure rate (inverse)
        health_score += (1 - failures.failure_rate) * 0.25

        # User trust dynamics
        if dynamics.trust_trend == 'healthy':
            health_score += 0.25
        elif dynamics.trust_trend == 'stable':
            health_score += 0.15

        if health_score >= 0.8:
            return 'healthy'
        elif health_score >= 0.5:
            return 'warning'
        else:
            return 'critical'

    def _generate_recommendations(
        self,
        calibration: CalibrationMetrics,
        zones: TrustZoneMetrics,
        failures: TrustFailureMetrics,
        dynamics: TrustDynamicsMetrics
    ) -> List[str]:
        """Generate comprehensive recommendations"""
        recommendations = []

        # Calibration recommendations
        recommendations.extend(
            self.calibration_analyzer.get_calibration_recommendations(calibration)
        )

        # Zone recommendations
        if zones.boundary_violations > 0:
            recommendations.append(
                f"Address {zones.boundary_violations} trust boundary violations"
            )

        # Failure recommendations
        if failures.failure_rate > 0.05:
            recommendations.append(
                "High failure rate detected - implement additional guardrails"
            )

        # Dynamics recommendations
        if dynamics.overtrust_percentage > 0.2:
            recommendations.append(
                "Significant overtrust detected - enhance uncertainty communication"
            )

        if dynamics.undertrust_percentage > 0.3:
            recommendations.append(
                "User undertrust detected - improve confidence signaling"
            )

        return recommendations

    def _identify_risk_factors(
        self,
        calibration: CalibrationMetrics,
        failures: TrustFailureMetrics,
        dynamics: TrustDynamicsMetrics
    ) -> List[str]:
        """Identify trust risk factors"""
        risks = []

        if calibration.calibration_quality == CalibrationQuality.MISCALIBRATED:
            risks.append("Model is significantly miscalibrated")

        if calibration.overconfidence_score > 0.15:
            risks.append(f"High overconfidence: {calibration.overconfidence_score:.2f}")

        if failures.failures_by_severity.get('critical', 0) > 0:
            risks.append(f"Critical trust failures: {failures.failures_by_severity['critical']}")

        if dynamics.overtrust_percentage > 0.3:
            risks.append(f"High overtrust rate: {dynamics.overtrust_percentage:.1%}")

        return risks

    def generate_report(
        self,
        assessment: TrustAssessment
    ) -> Dict[str, Any]:
        """Generate comprehensive trust report"""
        return {
            'assessment_id': assessment.assessment_id,
            'assessment_date': assessment.assessment_date.isoformat(),
            'overall_health': assessment.overall_trust_health,
            'summary': {
                'calibration_quality': assessment.calibration_metrics.calibration_quality.value,
                'ece': assessment.calibration_metrics.expected_calibration_error,
                'trust_zone': assessment.zone_metrics.current_zone.value,
                'zone_compliance': assessment.zone_metrics.zone_compliance_rate,
                'failure_rate': assessment.failure_metrics.failure_rate,
                'user_trust_trend': assessment.dynamics_metrics.trust_trend
            },
            'recommendations': assessment.recommendations,
            'risk_factors': assessment.risk_factors,
            'detailed_metrics': {
                'confidence': {
                    'signals_generated': assessment.confidence_metrics.signals_generated,
                    'average_confidence': assessment.confidence_metrics.average_confidence,
                    'uncertainty_expressed': assessment.confidence_metrics.uncertainty_expressed
                },
                'calibration': {
                    'ece': assessment.calibration_metrics.expected_calibration_error,
                    'mce': assessment.calibration_metrics.maximum_calibration_error,
                    'brier_score': assessment.calibration_metrics.brier_score,
                    'overconfidence': assessment.calibration_metrics.overconfidence_score,
                    'underconfidence': assessment.calibration_metrics.underconfidence_score
                },
                'trust_zones': {
                    'current_zone': assessment.zone_metrics.current_zone.value,
                    'violations': assessment.zone_metrics.boundary_violations,
                    'escalations': assessment.zone_metrics.escalation_count
                },
                'failures': {
                    'total': assessment.failure_metrics.total_failures,
                    'recovery_rate': assessment.failure_metrics.recovery_success_rate
                },
                'user_dynamics': {
                    'average_trust': assessment.dynamics_metrics.average_trust_score,
                    'overtrust_pct': assessment.dynamics_metrics.overtrust_percentage,
                    'undertrust_pct': assessment.dynamics_metrics.undertrust_percentage
                }
            }
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    'ConfidenceLevel',
    'TrustZone',
    'CalibrationQuality',
    'TrustFailureType',
    'TrustSignalType',
    'TrustRepairStrategy',
    'UserTrustState',
    # Data Classes
    'ConfidenceSignal',
    'ConfidenceSignalingMetrics',
    'CalibrationBin',
    'CalibrationMetrics',
    'TrustZonePolicy',
    'TrustZoneMetrics',
    'TrustFailureEvent',
    'TrustFailureMetrics',
    'UserTrustProfile',
    'TrustDynamicsMetrics',
    'TrustAssessment',
    # Confidence Signaling Analyzers
    'ConfidenceSignalAnalyzer',
    'UncertaintyCommunicationAnalyzer',
    # Calibration Analyzers
    'TrustCalibrationMetricsAnalyzer',
    'ReliabilityDiagramAnalyzer',
    # Trust Zone Analyzers
    'TrustZoneAnalyzer',
    'TrustBoundaryAnalyzer',
    # Trust Failure Analyzers
    'TrustFailureAnalyzer',
    # User Trust Dynamics Analyzers
    'UserTrustDynamicsAnalyzer',
    'OvertrustPreventionAnalyzer',
    # Comprehensive Analyzer
    'TrustCalibrationAnalyzer',
]
