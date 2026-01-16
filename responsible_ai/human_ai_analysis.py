"""
Human-AI Analysis Module - Human-Centered AI, Human-in-the-Loop AI
===================================================================

Comprehensive analysis for human-AI interaction, collaboration, and oversight.
Implements 36 analysis types across two related frameworks.

Frameworks:
- Human-Centered AI (18 types): User Experience, Accessibility, Cognitive Load, Trust
- Human-in-the-Loop AI (18 types): Feedback Integration, Oversight, Control, Correction
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from collections import defaultdict
import json


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class HumanCenteredMetrics:
    """Metrics for human-centered AI analysis."""
    user_satisfaction: float = 0.0
    usability_score: float = 0.0
    cognitive_load: float = 0.0
    accessibility_compliance: float = 0.0
    trust_calibration: float = 0.0
    error_recovery: float = 0.0


@dataclass
class HITLMetrics:
    """Metrics for human-in-the-loop analysis."""
    feedback_integration_rate: float = 0.0
    oversight_effectiveness: float = 0.0
    human_control_score: float = 0.0
    correction_accuracy: float = 0.0
    intervention_rate: float = 0.0
    collaboration_efficiency: float = 0.0


@dataclass
class UserInteraction:
    """Represents a user interaction with the AI system."""
    interaction_id: str
    user_id: str
    timestamp: datetime
    interaction_type: str
    duration_seconds: float
    success: bool
    user_satisfaction: Optional[float] = None
    errors_encountered: int = 0
    corrections_made: int = 0


@dataclass
class HumanFeedback:
    """Represents human feedback on AI output."""
    feedback_id: str
    output_id: str
    feedback_type: str  # 'correction', 'approval', 'rejection', 'suggestion'
    original_output: Any
    corrected_output: Optional[Any] = None
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    incorporated: bool = False


# ============================================================================
# Human-Centered AI Analyzers
# ============================================================================

class UserExperienceAnalyzer:
    """Analyzes user experience with AI systems."""

    def analyze_user_experience(self,
                               interactions: List[UserInteraction]) -> Dict[str, Any]:
        """Analyze user experience metrics."""
        if not interactions:
            return {'ux_score': 0.0, 'interactions_analyzed': 0}

        satisfaction_scores = [i.user_satisfaction for i in interactions if i.user_satisfaction is not None]
        success_rate = sum(1 for i in interactions if i.success) / len(interactions)
        avg_duration = np.mean([i.duration_seconds for i in interactions])
        error_rate = sum(i.errors_encountered for i in interactions) / len(interactions)

        # UX score combines satisfaction, success, and low error rate
        ux_score = (
            (np.mean(satisfaction_scores) if satisfaction_scores else 0.5) * 0.4 +
            success_rate * 0.4 +
            (1 - min(1, error_rate)) * 0.2
        )

        return {
            'ux_score': float(ux_score),
            'mean_satisfaction': float(np.mean(satisfaction_scores)) if satisfaction_scores else None,
            'success_rate': float(success_rate),
            'error_rate': float(error_rate),
            'average_duration': float(avg_duration),
            'interactions_analyzed': len(interactions),
            'satisfaction_distribution': self._get_distribution(satisfaction_scores)
        }

    def _get_distribution(self, values: List[float]) -> Dict[str, int]:
        if not values:
            return {}
        bins = {'low': 0, 'medium': 0, 'high': 0}
        for v in values:
            if v < 0.4:
                bins['low'] += 1
            elif v < 0.7:
                bins['medium'] += 1
            else:
                bins['high'] += 1
        return bins


class CognitiveLoadAnalyzer:
    """Analyzes cognitive load on users."""

    def analyze_cognitive_load(self,
                               task_complexity: List[Dict[str, Any]],
                               user_performance: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cognitive load indicators."""
        if not task_complexity:
            return {'cognitive_load_score': 0.5, 'overload_risk': 'unknown'}

        load_indicators = []

        for i, task in enumerate(task_complexity):
            complexity = task.get('complexity', 0.5)
            information_density = task.get('information_density', 0.5)
            decision_points = task.get('decision_points', 1)

            # Calculate task load
            task_load = (complexity * 0.4 + information_density * 0.3 + min(1, decision_points / 10) * 0.3)
            load_indicators.append(task_load)

        avg_load = np.mean(load_indicators)

        # Analyze performance correlation
        if user_performance:
            performance_scores = [p.get('accuracy', 0.5) for p in user_performance]
            avg_performance = np.mean(performance_scores)

            # High load with low performance indicates overload
            if avg_load > 0.7 and avg_performance < 0.5:
                overload_risk = 'high'
            elif avg_load > 0.5 and avg_performance < 0.6:
                overload_risk = 'medium'
            else:
                overload_risk = 'low'
        else:
            avg_performance = None
            overload_risk = 'high' if avg_load > 0.7 else ('medium' if avg_load > 0.5 else 'low')

        return {
            'cognitive_load_score': float(avg_load),
            'average_performance': float(avg_performance) if avg_performance else None,
            'overload_risk': overload_risk,
            'tasks_analyzed': len(task_complexity),
            'high_load_tasks': sum(1 for l in load_indicators if l > 0.7),
            'recommendations': self._generate_load_recommendations(avg_load)
        }

    def _generate_load_recommendations(self, load: float) -> List[str]:
        recommendations = []
        if load > 0.7:
            recommendations.append("Simplify interface and reduce information density")
            recommendations.append("Provide progressive disclosure of complex features")
        if load > 0.5:
            recommendations.append("Add visual hierarchy to prioritize information")
        return recommendations


class TrustCalibrationAnalyzer:
    """Analyzes human trust calibration with AI."""

    def analyze_trust_calibration(self,
                                  user_trust_reports: List[Dict[str, Any]],
                                  system_accuracy: float) -> Dict[str, Any]:
        """Analyze trust calibration between humans and AI."""
        if not user_trust_reports:
            return {'calibration_score': 0.0, 'trust_gap': 0.0}

        trust_levels = [r.get('trust_level', 0.5) for r in user_trust_reports]
        avg_trust = np.mean(trust_levels)

        # Trust gap: difference between trust and actual accuracy
        trust_gap = avg_trust - system_accuracy

        # Calibration is best when trust matches accuracy
        calibration_score = 1 - abs(trust_gap)

        # Identify over/under trust
        if trust_gap > 0.1:
            trust_status = 'overtrust'
        elif trust_gap < -0.1:
            trust_status = 'undertrust'
        else:
            trust_status = 'calibrated'

        return {
            'calibration_score': float(calibration_score),
            'trust_gap': float(trust_gap),
            'average_user_trust': float(avg_trust),
            'system_accuracy': system_accuracy,
            'trust_status': trust_status,
            'trust_variance': float(np.var(trust_levels)),
            'users_analyzed': len(user_trust_reports)
        }


class AccessibilityAnalyzer:
    """Analyzes accessibility compliance."""

    def analyze_accessibility(self,
                             accessibility_audit: List[Dict[str, Any]],
                             wcag_level: str = 'AA') -> Dict[str, Any]:
        """Analyze accessibility compliance."""
        if not accessibility_audit:
            return {'accessibility_score': 0.0, 'violations': 0}

        passed = sum(1 for a in accessibility_audit if a.get('passed', False))
        total = len(accessibility_audit)

        violations_by_category = defaultdict(int)
        for audit in accessibility_audit:
            if not audit.get('passed', False):
                category = audit.get('category', 'unknown')
                violations_by_category[category] += 1

        accessibility_score = passed / total if total > 0 else 0

        return {
            'accessibility_score': float(accessibility_score),
            'wcag_level': wcag_level,
            'tests_passed': passed,
            'tests_failed': total - passed,
            'total_tests': total,
            'violations_by_category': dict(violations_by_category),
            'compliance_status': 'compliant' if accessibility_score >= 0.95 else 'non_compliant'
        }


# ============================================================================
# Human-in-the-Loop AI Analyzers
# ============================================================================

class FeedbackIntegrationAnalyzer:
    """Analyzes human feedback integration."""

    def analyze_feedback_integration(self,
                                    feedback_records: List[HumanFeedback]) -> Dict[str, Any]:
        """Analyze how human feedback is integrated."""
        if not feedback_records:
            return {'integration_rate': 0.0, 'total_feedback': 0}

        incorporated = [f for f in feedback_records if f.incorporated]
        integration_rate = len(incorporated) / len(feedback_records)

        feedback_by_type = defaultdict(lambda: {'total': 0, 'incorporated': 0})
        for feedback in feedback_records:
            feedback_by_type[feedback.feedback_type]['total'] += 1
            if feedback.incorporated:
                feedback_by_type[feedback.feedback_type]['incorporated'] += 1

        type_analysis = {
            ftype: {
                'total': data['total'],
                'incorporated': data['incorporated'],
                'rate': data['incorporated'] / data['total'] if data['total'] > 0 else 0
            }
            for ftype, data in feedback_by_type.items()
        }

        return {
            'integration_rate': float(integration_rate),
            'total_feedback': len(feedback_records),
            'incorporated_feedback': len(incorporated),
            'pending_feedback': len(feedback_records) - len(incorporated),
            'type_analysis': type_analysis,
            'most_common_type': max(feedback_by_type, key=lambda x: feedback_by_type[x]['total']) if feedback_by_type else None
        }


class OversightEffectivenessAnalyzer:
    """Analyzes human oversight effectiveness."""

    def analyze_oversight(self,
                         oversight_actions: List[Dict[str, Any]],
                         model_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze human oversight effectiveness."""
        if not oversight_actions:
            return {'oversight_effectiveness': 0.0, 'catch_rate': 0.0}

        errors_caught = sum(1 for o in oversight_actions if o.get('error_caught', False))
        errors_missed = sum(1 for e in model_errors if not e.get('caught_by_oversight', False))
        total_errors = len(model_errors)

        catch_rate = errors_caught / total_errors if total_errors > 0 else 1.0

        # False positive rate (unnecessary interventions)
        unnecessary_interventions = sum(1 for o in oversight_actions if not o.get('was_necessary', True))
        precision = (len(oversight_actions) - unnecessary_interventions) / len(oversight_actions) if oversight_actions else 1

        effectiveness = (catch_rate * 0.6 + precision * 0.4)

        return {
            'oversight_effectiveness': float(effectiveness),
            'catch_rate': float(catch_rate),
            'precision': float(precision),
            'errors_caught': errors_caught,
            'errors_missed': errors_missed,
            'total_errors': total_errors,
            'total_interventions': len(oversight_actions),
            'unnecessary_interventions': unnecessary_interventions
        }


class HumanControlAnalyzer:
    """Analyzes human control over AI systems."""

    def analyze_human_control(self,
                             control_actions: List[Dict[str, Any]],
                             system_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze human control effectiveness."""
        if not control_actions:
            return {'control_score': 0.0, 'control_actions': 0}

        successful_controls = sum(1 for c in control_actions if c.get('successful', False))
        control_success_rate = successful_controls / len(control_actions)

        # Analyze control types
        control_types = defaultdict(lambda: {'total': 0, 'successful': 0})
        for action in control_actions:
            ctype = action.get('type', 'unknown')
            control_types[ctype]['total'] += 1
            if action.get('successful', False):
                control_types[ctype]['successful'] += 1

        # Response time analysis
        response_times = [s.get('response_time', 0) for s in system_responses if 'response_time' in s]
        avg_response_time = np.mean(response_times) if response_times else 0

        return {
            'control_score': float(control_success_rate),
            'successful_controls': successful_controls,
            'failed_controls': len(control_actions) - successful_controls,
            'total_actions': len(control_actions),
            'control_type_analysis': dict(control_types),
            'average_response_time': float(avg_response_time),
            'control_types_available': list(control_types.keys())
        }


class CorrectionAnalyzer:
    """Analyzes human corrections to AI outputs."""

    def analyze_corrections(self,
                           corrections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze human corrections."""
        if not corrections:
            return {'correction_rate': 0.0, 'total_corrections': 0}

        # Analyze correction types
        correction_types = defaultdict(int)
        correction_magnitudes = []

        for correction in corrections:
            ctype = correction.get('type', 'unknown')
            correction_types[ctype] += 1

            magnitude = correction.get('magnitude', 0.5)
            correction_magnitudes.append(magnitude)

        avg_magnitude = np.mean(correction_magnitudes)

        # Analyze correction patterns
        pattern_analysis = {
            'minor_corrections': sum(1 for m in correction_magnitudes if m < 0.3),
            'moderate_corrections': sum(1 for m in correction_magnitudes if 0.3 <= m < 0.7),
            'major_corrections': sum(1 for m in correction_magnitudes if m >= 0.7)
        }

        return {
            'total_corrections': len(corrections),
            'average_magnitude': float(avg_magnitude),
            'correction_types': dict(correction_types),
            'pattern_analysis': pattern_analysis,
            'most_common_correction': max(correction_types, key=correction_types.get) if correction_types else None,
            'major_correction_rate': pattern_analysis['major_corrections'] / len(corrections) if corrections else 0
        }


class CollaborationAnalyzer:
    """Analyzes human-AI collaboration efficiency."""

    def analyze_collaboration(self,
                             collaborative_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze human-AI collaboration."""
        if not collaborative_tasks:
            return {'collaboration_score': 0.0, 'tasks_analyzed': 0}

        completion_rates = []
        efficiency_scores = []

        for task in collaborative_tasks:
            completed = task.get('completed', False)
            human_time = task.get('human_time', 0)
            ai_time = task.get('ai_time', 0)
            total_time = task.get('total_time', human_time + ai_time)
            quality = task.get('quality_score', 0.5)

            if completed:
                completion_rates.append(1)
                # Efficiency: quality relative to time spent
                if total_time > 0:
                    efficiency = quality / (total_time / 3600)  # quality per hour
                    efficiency_scores.append(min(1, efficiency))
            else:
                completion_rates.append(0)

        avg_completion = np.mean(completion_rates)
        avg_efficiency = np.mean(efficiency_scores) if efficiency_scores else 0

        collaboration_score = (avg_completion * 0.5 + avg_efficiency * 0.5)

        return {
            'collaboration_score': float(collaboration_score),
            'completion_rate': float(avg_completion),
            'efficiency_score': float(avg_efficiency),
            'tasks_analyzed': len(collaborative_tasks),
            'completed_tasks': sum(completion_rates),
            'avg_human_time': float(np.mean([t.get('human_time', 0) for t in collaborative_tasks])),
            'avg_ai_time': float(np.mean([t.get('ai_time', 0) for t in collaborative_tasks]))
        }


# ============================================================================
# Report Generator
# ============================================================================

class HumanAIReportGenerator:
    """Generates comprehensive human-AI analysis reports."""

    def __init__(self):
        self.ux_analyzer = UserExperienceAnalyzer()
        self.cognitive_analyzer = CognitiveLoadAnalyzer()
        self.trust_analyzer = TrustCalibrationAnalyzer()
        self.accessibility_analyzer = AccessibilityAnalyzer()
        self.feedback_analyzer = FeedbackIntegrationAnalyzer()
        self.oversight_analyzer = OversightEffectivenessAnalyzer()
        self.control_analyzer = HumanControlAnalyzer()
        self.correction_analyzer = CorrectionAnalyzer()
        self.collaboration_analyzer = CollaborationAnalyzer()

    def generate_full_report(self,
                            interactions: List[UserInteraction] = None,
                            feedback_records: List[HumanFeedback] = None,
                            oversight_actions: List[Dict[str, Any]] = None,
                            collaborative_tasks: List[Dict[str, Any]] = None,
                            system_accuracy: float = 0.85) -> Dict[str, Any]:
        """Generate comprehensive human-AI report."""
        report = {
            'report_type': 'comprehensive_human_ai_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        if interactions:
            report['user_experience'] = self.ux_analyzer.analyze_user_experience(interactions)

        if feedback_records:
            report['feedback_integration'] = self.feedback_analyzer.analyze_feedback_integration(feedback_records)

        if oversight_actions:
            report['oversight'] = self.oversight_analyzer.analyze_oversight(oversight_actions, [])

        if collaborative_tasks:
            report['collaboration'] = self.collaboration_analyzer.analyze_collaboration(collaborative_tasks)

        # Calculate overall score
        scores = []
        if 'user_experience' in report:
            scores.append(report['user_experience'].get('ux_score', 0))
        if 'feedback_integration' in report:
            scores.append(report['feedback_integration'].get('integration_rate', 0))
        if 'oversight' in report:
            scores.append(report['oversight'].get('oversight_effectiveness', 0))
        if 'collaboration' in report:
            scores.append(report['collaboration'].get('collaboration_score', 0))

        report['overall_score'] = float(np.mean(scores)) if scores else 0.0

        return report

    def export_report(self, report: Dict[str, Any], filepath: str, format: str = 'json') -> None:
        """Export report to file."""
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
