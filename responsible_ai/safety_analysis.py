"""
Safety Analysis Module - Safe AI, Long-Term Risk Management
============================================================

Comprehensive analysis for AI system safety, harm prevention, and long-term risk management.
Implements 38 analysis types across two related frameworks.

Frameworks:
- Safe AI (18 types): Harm Prevention, Safety Constraints, Fail-Safe, Content Safety
- Long-Term Risk Management (20 types): Existential Risk, Value Alignment, Control, Capability
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import numpy as np
from collections import defaultdict
import json
import re


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SafetyMetrics:
    """Metrics for AI safety analysis."""
    harm_prevention_score: float = 0.0
    safety_constraint_compliance: float = 0.0
    fail_safe_coverage: float = 0.0
    content_safety_score: float = 0.0
    adversarial_robustness: float = 0.0
    boundary_adherence: float = 0.0
    safety_margin: float = 0.0
    incident_rate: float = 0.0


@dataclass
class RiskMetrics:
    """Metrics for long-term risk analysis."""
    existential_risk_score: float = 0.0
    value_alignment_score: float = 0.0
    control_robustness: float = 0.0
    capability_risk: float = 0.0
    goal_stability: float = 0.0
    containment_effectiveness: float = 0.0
    reversibility_score: float = 0.0
    uncertainty_quantification: float = 0.0


@dataclass
class SafetyIncident:
    """Represents a safety incident."""
    incident_id: str
    timestamp: datetime
    severity: str  # 'critical', 'high', 'medium', 'low'
    incident_type: str
    description: str
    harm_caused: str
    affected_users: int
    root_cause: str
    mitigation_taken: str
    prevented: bool = False
    resolved: bool = True


@dataclass
class SafetyConstraint:
    """Represents a safety constraint."""
    constraint_id: str
    name: str
    description: str
    constraint_type: str  # 'hard', 'soft'
    enforcement_level: str  # 'strict', 'moderate', 'lenient'
    applies_to: List[str]
    violation_penalty: float = 1.0


@dataclass
class RiskScenario:
    """Represents a risk scenario for analysis."""
    scenario_id: str
    name: str
    description: str
    risk_category: str
    probability: float
    impact_severity: float
    time_horizon: str  # 'immediate', 'short-term', 'medium-term', 'long-term'
    mitigation_strategies: List[str]
    residual_risk: float = 0.0


# ============================================================================
# Safe AI Analyzers (18 Analysis Types)
# ============================================================================

class HarmPreventionAnalyzer:
    """Analyzes harm prevention mechanisms and effectiveness."""

    def __init__(self, harm_categories: List[str] = None):
        self.harm_categories = harm_categories or [
            'physical', 'psychological', 'financial', 'reputational',
            'privacy', 'discrimination', 'misinformation', 'manipulation'
        ]

    def analyze_harm_prevention(self,
                                outputs: List[Dict[str, Any]],
                                harm_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze harm prevention effectiveness."""
        if not outputs:
            return {'harm_prevention_score': 1.0, 'harmful_outputs': 0}

        harmful_outputs = []
        harm_by_category = defaultdict(int)
        prevented_harms = []

        for i, output in enumerate(outputs):
            assessment = harm_assessments[i] if i < len(harm_assessments) else {}

            is_harmful = assessment.get('is_harmful', False)
            harm_category = assessment.get('harm_category')
            was_prevented = assessment.get('prevented', False)

            if is_harmful:
                harmful_outputs.append({
                    'output_index': i,
                    'category': harm_category,
                    'severity': assessment.get('severity', 'unknown')
                })
                if harm_category:
                    harm_by_category[harm_category] += 1
            elif was_prevented:
                prevented_harms.append({
                    'output_index': i,
                    'category': harm_category,
                    'prevented_by': assessment.get('prevented_by')
                })

        total_outputs = len(outputs)
        harmful_count = len(harmful_outputs)
        prevented_count = len(prevented_harms)

        harm_prevention_score = 1 - (harmful_count / total_outputs) if total_outputs > 0 else 1.0
        prevention_effectiveness = prevented_count / (harmful_count + prevented_count) if (harmful_count + prevented_count) > 0 else 1.0

        return {
            'harm_prevention_score': harm_prevention_score,
            'prevention_effectiveness': prevention_effectiveness,
            'total_outputs': total_outputs,
            'harmful_outputs': harmful_count,
            'prevented_harms': prevented_count,
            'harm_by_category': dict(harm_by_category),
            'most_common_harm': max(harm_by_category, key=harm_by_category.get) if harm_by_category else None,
            'harmful_output_details': harmful_outputs[:20]
        }


class SafetyConstraintAnalyzer:
    """Analyzes safety constraint compliance and effectiveness."""

    def analyze_constraints(self,
                           actions: List[Dict[str, Any]],
                           constraints: List[SafetyConstraint]) -> Dict[str, Any]:
        """Analyze safety constraint compliance."""
        if not actions or not constraints:
            return {'constraint_compliance': 1.0, 'violations': 0}

        violations = []
        compliance_by_constraint = {}

        for constraint in constraints:
            constraint_violations = []

            for action in actions:
                if self._check_violation(action, constraint):
                    constraint_violations.append({
                        'action': action.get('id', 'unknown'),
                        'constraint': constraint.constraint_id,
                        'constraint_type': constraint.constraint_type
                    })

            total_applicable = len([a for a in actions if self._is_applicable(a, constraint)])
            violation_count = len(constraint_violations)

            compliance_rate = 1 - (violation_count / total_applicable) if total_applicable > 0 else 1.0

            compliance_by_constraint[constraint.constraint_id] = {
                'name': constraint.name,
                'compliance_rate': compliance_rate,
                'violations': violation_count,
                'total_applicable': total_applicable,
                'constraint_type': constraint.constraint_type
            }

            violations.extend(constraint_violations)

        # Calculate overall compliance
        hard_constraints = [c for c in constraints if c.constraint_type == 'hard']
        soft_constraints = [c for c in constraints if c.constraint_type == 'soft']

        hard_compliance = np.mean([
            compliance_by_constraint[c.constraint_id]['compliance_rate']
            for c in hard_constraints
        ]) if hard_constraints else 1.0

        soft_compliance = np.mean([
            compliance_by_constraint[c.constraint_id]['compliance_rate']
            for c in soft_constraints
        ]) if soft_constraints else 1.0

        # Hard constraints weighted more heavily
        overall_compliance = 0.7 * hard_compliance + 0.3 * soft_compliance

        return {
            'constraint_compliance': overall_compliance,
            'hard_constraint_compliance': float(hard_compliance),
            'soft_constraint_compliance': float(soft_compliance),
            'total_violations': len(violations),
            'violations': violations[:50],
            'compliance_by_constraint': compliance_by_constraint
        }

    def _check_violation(self, action: Dict[str, Any], constraint: SafetyConstraint) -> bool:
        """Check if action violates constraint."""
        if not self._is_applicable(action, constraint):
            return False

        # Check if action has explicit violation flag
        if action.get('violates_constraints'):
            return constraint.constraint_id in action.get('violates_constraints', [])

        return False

    def _is_applicable(self, action: Dict[str, Any], constraint: SafetyConstraint) -> bool:
        """Check if constraint applies to action."""
        action_type = action.get('type', 'unknown')
        return action_type in constraint.applies_to or 'all' in constraint.applies_to


class FailSafeAnalyzer:
    """Analyzes fail-safe mechanisms and coverage."""

    def analyze_fail_safes(self,
                          system_components: List[Dict[str, Any]],
                          failure_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze fail-safe coverage and effectiveness."""
        if not system_components:
            return {'fail_safe_coverage': 0.0, 'uncovered_components': []}

        covered_components = []
        uncovered_components = []
        fail_safe_types = defaultdict(int)

        for component in system_components:
            component_name = component.get('name', 'unknown')
            has_fail_safe = component.get('has_fail_safe', False)
            fail_safe_type = component.get('fail_safe_type')

            if has_fail_safe:
                covered_components.append({
                    'component': component_name,
                    'fail_safe_type': fail_safe_type,
                    'fail_safe_tested': component.get('fail_safe_tested', False)
                })
                if fail_safe_type:
                    fail_safe_types[fail_safe_type] += 1
            else:
                if component.get('critical', False):
                    uncovered_components.append({
                        'component': component_name,
                        'critical': True,
                        'risk_level': 'high'
                    })
                else:
                    uncovered_components.append({
                        'component': component_name,
                        'critical': False,
                        'risk_level': 'medium'
                    })

        # Analyze failure scenario coverage
        scenario_coverage = []
        for scenario in failure_scenarios:
            handled = scenario.get('handled_by_fail_safe', False)
            scenario_coverage.append({
                'scenario': scenario.get('name'),
                'covered': handled,
                'recovery_action': scenario.get('recovery_action')
            })

        covered_scenarios = sum(1 for s in scenario_coverage if s['covered'])
        total_scenarios = len(failure_scenarios)

        coverage = len(covered_components) / len(system_components) if system_components else 0
        scenario_coverage_rate = covered_scenarios / total_scenarios if total_scenarios > 0 else 1.0

        # Critical component coverage
        critical_components = [c for c in system_components if c.get('critical', False)]
        critical_covered = len([c for c in covered_components if c['component'] in [cc.get('name') for cc in critical_components]])
        critical_coverage = critical_covered / len(critical_components) if critical_components else 1.0

        return {
            'fail_safe_coverage': coverage,
            'critical_coverage': critical_coverage,
            'scenario_coverage': scenario_coverage_rate,
            'covered_components': len(covered_components),
            'uncovered_components': uncovered_components,
            'total_components': len(system_components),
            'fail_safe_types': dict(fail_safe_types),
            'scenario_analysis': scenario_coverage,
            'risk_assessment': 'high' if critical_coverage < 0.8 else ('medium' if coverage < 0.9 else 'low')
        }


class ContentSafetyAnalyzer:
    """Analyzes content safety and harmful content detection."""

    def __init__(self, unsafe_patterns: List[str] = None):
        self.unsafe_patterns = unsafe_patterns or [
            r'\b(hate|violence|harm|kill|attack)\b',
            r'\b(illegal|drugs|weapons)\b',
            r'\b(explicit|adult|nsfw)\b'
        ]

    def analyze_content_safety(self,
                               contents: List[str],
                               content_labels: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze content for safety issues."""
        if not contents:
            return {'content_safety_score': 1.0, 'unsafe_content': 0}

        unsafe_contents = []
        category_counts = defaultdict(int)
        pattern_matches = defaultdict(int)

        for i, content in enumerate(contents):
            label = content_labels[i] if content_labels and i < len(content_labels) else {}

            # Check for unsafe patterns
            detected_patterns = []
            for pattern in self.unsafe_patterns:
                if re.search(pattern, content.lower()):
                    detected_patterns.append(pattern)
                    pattern_matches[pattern] += 1

            # Check provided labels
            is_unsafe = label.get('is_unsafe', False) or len(detected_patterns) > 0
            category = label.get('category', 'pattern_match' if detected_patterns else None)

            if is_unsafe:
                unsafe_contents.append({
                    'index': i,
                    'category': category,
                    'severity': label.get('severity', 'unknown'),
                    'patterns_matched': detected_patterns
                })
                if category:
                    category_counts[category] += 1

        total_contents = len(contents)
        unsafe_count = len(unsafe_contents)

        content_safety_score = 1 - (unsafe_count / total_contents) if total_contents > 0 else 1.0

        return {
            'content_safety_score': content_safety_score,
            'total_contents': total_contents,
            'unsafe_content': unsafe_count,
            'safe_content': total_contents - unsafe_count,
            'unsafe_rate': unsafe_count / total_contents if total_contents > 0 else 0,
            'category_distribution': dict(category_counts),
            'pattern_matches': dict(pattern_matches),
            'unsafe_details': unsafe_contents[:30]
        }


class BoundaryAnalyzer:
    """Analyzes operational boundary adherence."""

    def analyze_boundaries(self,
                          operations: List[Dict[str, Any]],
                          boundaries: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Analyze boundary adherence for operations."""
        if not operations or not boundaries:
            return {'boundary_adherence': 1.0, 'violations': 0}

        violations = []
        adherence_by_boundary = {}

        for boundary_name, (lower, upper) in boundaries.items():
            boundary_violations = []

            for op in operations:
                value = op.get(boundary_name)
                if value is not None:
                    if value < lower or value > upper:
                        boundary_violations.append({
                            'operation_id': op.get('id'),
                            'boundary': boundary_name,
                            'value': value,
                            'lower_bound': lower,
                            'upper_bound': upper,
                            'violation_type': 'below' if value < lower else 'above'
                        })

            total_ops_with_metric = len([o for o in operations if o.get(boundary_name) is not None])
            violation_count = len(boundary_violations)

            adherence = 1 - (violation_count / total_ops_with_metric) if total_ops_with_metric > 0 else 1.0

            adherence_by_boundary[boundary_name] = {
                'adherence': adherence,
                'violations': violation_count,
                'total_checked': total_ops_with_metric,
                'bounds': (lower, upper)
            }

            violations.extend(boundary_violations)

        overall_adherence = np.mean([v['adherence'] for v in adherence_by_boundary.values()]) if adherence_by_boundary else 1.0

        return {
            'boundary_adherence': float(overall_adherence),
            'total_violations': len(violations),
            'adherence_by_boundary': adherence_by_boundary,
            'violations': violations[:50],
            'most_violated_boundary': min(adherence_by_boundary, key=lambda x: adherence_by_boundary[x]['adherence']) if adherence_by_boundary else None
        }


class SafetyMarginAnalyzer:
    """Analyzes safety margins and buffer zones."""

    def analyze_safety_margins(self,
                               metrics: Dict[str, float],
                               thresholds: Dict[str, float],
                               safety_factors: Dict[str, float] = None) -> Dict[str, Any]:
        """Analyze safety margins relative to thresholds."""
        if not metrics or not thresholds:
            return {'average_safety_margin': 0.0, 'critical_margins': []}

        safety_factors = safety_factors or {k: 1.5 for k in thresholds}

        margin_analysis = {}
        critical_margins = []

        for metric_name, value in metrics.items():
            if metric_name not in thresholds:
                continue

            threshold = thresholds[metric_name]
            safety_factor = safety_factors.get(metric_name, 1.5)

            # Calculate margin as percentage of threshold
            margin = (threshold - value) / threshold if threshold != 0 else 0
            safe_threshold = threshold / safety_factor  # More conservative threshold

            is_within_margin = value <= safe_threshold
            criticality = 'safe' if margin > 0.3 else ('warning' if margin > 0.1 else 'critical')

            margin_analysis[metric_name] = {
                'current_value': value,
                'threshold': threshold,
                'safe_threshold': safe_threshold,
                'margin': float(margin),
                'margin_percentage': float(margin * 100),
                'within_safety_margin': is_within_margin,
                'criticality': criticality
            }

            if criticality == 'critical':
                critical_margins.append(metric_name)

        avg_margin = np.mean([v['margin'] for v in margin_analysis.values()]) if margin_analysis else 0

        return {
            'average_safety_margin': float(avg_margin),
            'average_margin_percentage': float(avg_margin * 100),
            'metrics_analyzed': len(margin_analysis),
            'critical_margins': critical_margins,
            'margin_analysis': margin_analysis,
            'overall_status': 'critical' if critical_margins else ('warning' if avg_margin < 0.2 else 'safe')
        }


class SafetyIncidentAnalyzer:
    """Analyzes safety incidents and patterns."""

    def analyze_incidents(self,
                         incidents: List[SafetyIncident]) -> Dict[str, Any]:
        """Analyze safety incidents for patterns and trends."""
        if not incidents:
            return {
                'incident_rate': 0.0,
                'total_incidents': 0,
                'severity_distribution': {}
            }

        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)
        root_cause_counts = defaultdict(int)
        total_affected_users = 0

        prevented_incidents = []
        unresolved_incidents = []

        for incident in incidents:
            severity_counts[incident.severity] += 1
            type_counts[incident.incident_type] += 1
            root_cause_counts[incident.root_cause] += 1
            total_affected_users += incident.affected_users

            if incident.prevented:
                prevented_incidents.append(incident)
            if not incident.resolved:
                unresolved_incidents.append(incident)

        # Calculate time-based metrics
        sorted_incidents = sorted(incidents, key=lambda x: x.timestamp)
        if len(sorted_incidents) >= 2:
            time_span = (sorted_incidents[-1].timestamp - sorted_incidents[0].timestamp).days
            incident_rate = len(incidents) / max(time_span, 1)
        else:
            incident_rate = len(incidents)

        # Severity weighting
        severity_weights = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        weighted_severity = sum(
            severity_weights.get(s, 1) * count
            for s, count in severity_counts.items()
        )

        return {
            'incident_rate': incident_rate,
            'total_incidents': len(incidents),
            'prevented_incidents': len(prevented_incidents),
            'unresolved_incidents': len(unresolved_incidents),
            'severity_distribution': dict(severity_counts),
            'type_distribution': dict(type_counts),
            'root_cause_distribution': dict(root_cause_counts),
            'total_affected_users': total_affected_users,
            'weighted_severity_score': weighted_severity,
            'most_common_type': max(type_counts, key=type_counts.get) if type_counts else None,
            'most_common_root_cause': max(root_cause_counts, key=root_cause_counts.get) if root_cause_counts else None,
            'prevention_rate': len(prevented_incidents) / len(incidents) if incidents else 0,
            'resolution_rate': (len(incidents) - len(unresolved_incidents)) / len(incidents) if incidents else 1.0
        }


class AdversarialSafetyAnalyzer:
    """Analyzes safety against adversarial attacks."""

    def analyze_adversarial_safety(self,
                                   normal_outputs: List[Dict[str, Any]],
                                   adversarial_outputs: List[Dict[str, Any]],
                                   attack_types: List[str] = None) -> Dict[str, Any]:
        """Analyze robustness against adversarial attacks."""
        attack_types = attack_types or ['prompt_injection', 'jailbreak', 'data_poisoning', 'evasion']

        if not adversarial_outputs:
            return {'adversarial_robustness': 1.0, 'attack_success_rate': 0.0}

        successful_attacks = []
        blocked_attacks = []
        attack_results_by_type = defaultdict(lambda: {'success': 0, 'blocked': 0})

        for output in adversarial_outputs:
            attack_type = output.get('attack_type', 'unknown')
            attack_successful = output.get('attack_successful', False)
            was_blocked = output.get('blocked', False)

            if attack_successful:
                successful_attacks.append({
                    'attack_type': attack_type,
                    'severity': output.get('severity', 'unknown')
                })
                attack_results_by_type[attack_type]['success'] += 1
            elif was_blocked:
                blocked_attacks.append({
                    'attack_type': attack_type,
                    'blocked_by': output.get('blocked_by')
                })
                attack_results_by_type[attack_type]['blocked'] += 1

        total_attacks = len(adversarial_outputs)
        success_count = len(successful_attacks)
        blocked_count = len(blocked_attacks)

        attack_success_rate = success_count / total_attacks if total_attacks > 0 else 0
        adversarial_robustness = 1 - attack_success_rate

        # Calculate robustness by attack type
        robustness_by_type = {}
        for attack_type, results in attack_results_by_type.items():
            total = results['success'] + results['blocked']
            robustness_by_type[attack_type] = {
                'robustness': 1 - (results['success'] / total) if total > 0 else 1.0,
                'successful_attacks': results['success'],
                'blocked_attacks': results['blocked']
            }

        return {
            'adversarial_robustness': adversarial_robustness,
            'attack_success_rate': attack_success_rate,
            'total_attacks': total_attacks,
            'successful_attacks': success_count,
            'blocked_attacks': blocked_count,
            'robustness_by_type': robustness_by_type,
            'most_effective_attack': min(robustness_by_type, key=lambda x: robustness_by_type[x]['robustness']) if robustness_by_type else None,
            'successful_attack_details': successful_attacks[:20]
        }


# ============================================================================
# Long-Term Risk Management Analyzers (20 Analysis Types)
# ============================================================================

class ExistentialRiskAnalyzer:
    """Analyzes existential and catastrophic risk factors."""

    def analyze_existential_risk(self,
                                 risk_scenarios: List[RiskScenario],
                                 capability_levels: Dict[str, float]) -> Dict[str, Any]:
        """Analyze existential risk factors."""
        if not risk_scenarios:
            return {'existential_risk_score': 0.0, 'high_risk_scenarios': []}

        # Filter scenarios by risk category
        existential_scenarios = [
            s for s in risk_scenarios
            if s.risk_category in ['existential', 'catastrophic', 'global']
        ]

        high_risk = []
        medium_risk = []
        low_risk = []

        for scenario in existential_scenarios:
            risk_score = scenario.probability * scenario.impact_severity

            scenario_analysis = {
                'scenario_id': scenario.scenario_id,
                'name': scenario.name,
                'risk_score': risk_score,
                'probability': scenario.probability,
                'impact': scenario.impact_severity,
                'time_horizon': scenario.time_horizon
            }

            if risk_score > 0.7:
                high_risk.append(scenario_analysis)
            elif risk_score > 0.3:
                medium_risk.append(scenario_analysis)
            else:
                low_risk.append(scenario_analysis)

        # Analyze capability risks
        capability_risk_analysis = {}
        high_capability_risks = []

        for capability, level in capability_levels.items():
            # High capability in certain areas increases risk
            risk_multiplier = 1.0
            if capability in ['autonomous_action', 'self_improvement', 'resource_acquisition']:
                risk_multiplier = 1.5

            adjusted_risk = level * risk_multiplier
            capability_risk_analysis[capability] = {
                'capability_level': level,
                'risk_adjusted': adjusted_risk,
                'risk_multiplier': risk_multiplier
            }

            if adjusted_risk > 0.7:
                high_capability_risks.append(capability)

        # Calculate overall existential risk score
        if existential_scenarios:
            avg_risk = np.mean([s.probability * s.impact_severity for s in existential_scenarios])
        else:
            avg_risk = 0

        return {
            'existential_risk_score': float(avg_risk),
            'high_risk_scenarios': high_risk,
            'medium_risk_scenarios': medium_risk,
            'low_risk_scenarios': low_risk,
            'total_existential_scenarios': len(existential_scenarios),
            'capability_risk_analysis': capability_risk_analysis,
            'high_capability_risks': high_capability_risks,
            'risk_level': 'critical' if avg_risk > 0.7 else ('high' if avg_risk > 0.4 else ('medium' if avg_risk > 0.2 else 'low'))
        }


class ValueAlignmentAnalyzer:
    """Analyzes value alignment and goal stability."""

    def analyze_value_alignment(self,
                                system_values: Dict[str, float],
                                target_values: Dict[str, float],
                                behavioral_samples: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze value alignment between system and target values."""
        if not system_values or not target_values:
            return {'value_alignment_score': 0.0, 'misaligned_values': []}

        alignment_analysis = {}
        misaligned_values = []

        for value_name, target_level in target_values.items():
            system_level = system_values.get(value_name, 0)
            alignment = 1 - abs(target_level - system_level)

            alignment_analysis[value_name] = {
                'target_level': target_level,
                'system_level': system_level,
                'alignment': alignment,
                'gap': abs(target_level - system_level)
            }

            if alignment < 0.7:
                misaligned_values.append({
                    'value': value_name,
                    'alignment': alignment,
                    'gap': abs(target_level - system_level),
                    'direction': 'over' if system_level > target_level else 'under'
                })

        overall_alignment = np.mean([v['alignment'] for v in alignment_analysis.values()])

        # Analyze behavioral alignment if samples provided
        behavioral_alignment = 1.0
        if behavioral_samples:
            aligned_behaviors = sum(1 for b in behavioral_samples if b.get('aligned', False))
            behavioral_alignment = aligned_behaviors / len(behavioral_samples)

        return {
            'value_alignment_score': float(overall_alignment),
            'behavioral_alignment': behavioral_alignment,
            'combined_alignment': (overall_alignment + behavioral_alignment) / 2,
            'alignment_by_value': alignment_analysis,
            'misaligned_values': misaligned_values,
            'critical_misalignments': [m for m in misaligned_values if m['gap'] > 0.5],
            'alignment_status': 'aligned' if overall_alignment > 0.8 else ('partial' if overall_alignment > 0.5 else 'misaligned')
        }


class ControlRobustnessAnalyzer:
    """Analyzes control mechanism robustness."""

    def analyze_control_robustness(self,
                                   control_mechanisms: List[Dict[str, Any]],
                                   override_attempts: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze robustness of control mechanisms."""
        if not control_mechanisms:
            return {'control_robustness': 0.0, 'vulnerable_controls': []}

        robust_controls = []
        vulnerable_controls = []
        control_analysis = {}

        for control in control_mechanisms:
            control_id = control.get('id', 'unknown')
            robustness_score = control.get('robustness_score', 0.5)
            is_tested = control.get('tested', False)
            redundancy = control.get('redundancy_level', 0)

            # Adjust score based on testing and redundancy
            adjusted_score = robustness_score
            if is_tested:
                adjusted_score += 0.1
            adjusted_score += redundancy * 0.1
            adjusted_score = min(1.0, adjusted_score)

            control_analysis[control_id] = {
                'base_robustness': robustness_score,
                'adjusted_robustness': adjusted_score,
                'tested': is_tested,
                'redundancy_level': redundancy,
                'control_type': control.get('type')
            }

            if adjusted_score >= 0.7:
                robust_controls.append(control_id)
            else:
                vulnerable_controls.append({
                    'control_id': control_id,
                    'robustness': adjusted_score,
                    'issues': control.get('known_issues', [])
                })

        # Analyze override attempts
        override_analysis = {'total_attempts': 0, 'successful_overrides': 0}
        if override_attempts:
            override_analysis['total_attempts'] = len(override_attempts)
            override_analysis['successful_overrides'] = sum(
                1 for a in override_attempts if a.get('successful', False)
            )
            override_analysis['override_rate'] = (
                override_analysis['successful_overrides'] / override_analysis['total_attempts']
                if override_analysis['total_attempts'] > 0 else 0
            )

        overall_robustness = np.mean([v['adjusted_robustness'] for v in control_analysis.values()])

        return {
            'control_robustness': float(overall_robustness),
            'robust_controls': len(robust_controls),
            'vulnerable_controls': vulnerable_controls,
            'total_controls': len(control_mechanisms),
            'control_analysis': control_analysis,
            'override_analysis': override_analysis,
            'robustness_level': 'high' if overall_robustness > 0.8 else ('medium' if overall_robustness > 0.5 else 'low')
        }


class CapabilityRiskAnalyzer:
    """Analyzes risks associated with AI capabilities."""

    def analyze_capability_risk(self,
                               capabilities: Dict[str, Dict[str, Any]],
                               capability_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """Analyze capability-associated risks."""
        if not capabilities:
            return {'capability_risk': 0.0, 'high_risk_capabilities': []}

        capability_thresholds = capability_thresholds or {
            'autonomous_decision': 0.7,
            'resource_access': 0.6,
            'self_modification': 0.5,
            'external_communication': 0.7,
            'learning_rate': 0.8
        }

        risk_analysis = {}
        high_risk_capabilities = []
        moderate_risk_capabilities = []

        for cap_name, cap_data in capabilities.items():
            level = cap_data.get('level', 0)
            is_bounded = cap_data.get('bounded', True)
            has_oversight = cap_data.get('oversight', True)

            threshold = capability_thresholds.get(cap_name, 0.8)

            # Calculate risk factors
            level_risk = level / threshold if threshold > 0 else level
            unbounded_risk = 0.3 if not is_bounded else 0
            oversight_risk = 0.2 if not has_oversight else 0

            total_risk = min(1.0, level_risk + unbounded_risk + oversight_risk)

            risk_analysis[cap_name] = {
                'capability_level': level,
                'threshold': threshold,
                'is_bounded': is_bounded,
                'has_oversight': has_oversight,
                'risk_score': total_risk,
                'risk_factors': {
                    'level_risk': level_risk,
                    'unbounded_risk': unbounded_risk,
                    'oversight_risk': oversight_risk
                }
            }

            if total_risk > 0.7:
                high_risk_capabilities.append(cap_name)
            elif total_risk > 0.4:
                moderate_risk_capabilities.append(cap_name)

        overall_risk = np.mean([v['risk_score'] for v in risk_analysis.values()])

        return {
            'capability_risk': float(overall_risk),
            'high_risk_capabilities': high_risk_capabilities,
            'moderate_risk_capabilities': moderate_risk_capabilities,
            'risk_analysis': risk_analysis,
            'capabilities_over_threshold': len(high_risk_capabilities),
            'risk_level': 'critical' if len(high_risk_capabilities) > 2 else (
                'high' if high_risk_capabilities else ('moderate' if moderate_risk_capabilities else 'low')
            )
        }


class GoalStabilityAnalyzer:
    """Analyzes goal stability and drift."""

    def analyze_goal_stability(self,
                               goal_history: List[Dict[str, Any]],
                               original_goals: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze goal stability over time."""
        if not goal_history:
            return {'goal_stability': 1.0, 'goal_drift': 0.0}

        # Sort by timestamp
        sorted_history = sorted(goal_history, key=lambda x: x.get('timestamp', datetime.min))

        drift_measurements = []
        goal_changes = []

        for i, snapshot in enumerate(sorted_history):
            current_goals = snapshot.get('goals', {})

            # Compare with original goals
            drift = self._calculate_goal_drift(original_goals, current_goals)
            drift_measurements.append(drift)

            if i > 0:
                prev_goals = sorted_history[i-1].get('goals', {})
                incremental_drift = self._calculate_goal_drift(prev_goals, current_goals)
                if incremental_drift > 0.05:
                    goal_changes.append({
                        'timestamp': snapshot.get('timestamp'),
                        'drift': incremental_drift,
                        'changed_goals': self._identify_changed_goals(prev_goals, current_goals)
                    })

        avg_drift = np.mean(drift_measurements) if drift_measurements else 0
        max_drift = max(drift_measurements) if drift_measurements else 0
        goal_stability = 1 - avg_drift

        # Analyze drift trend
        if len(drift_measurements) >= 3:
            x = np.arange(len(drift_measurements))
            slope = np.polyfit(x, drift_measurements, 1)[0]
            trend = 'increasing' if slope > 0.01 else ('decreasing' if slope < -0.01 else 'stable')
        else:
            slope = 0
            trend = 'insufficient_data'

        return {
            'goal_stability': float(goal_stability),
            'goal_drift': float(avg_drift),
            'max_drift': float(max_drift),
            'drift_trend': trend,
            'drift_slope': float(slope),
            'significant_goal_changes': len(goal_changes),
            'goal_change_details': goal_changes,
            'stability_status': 'stable' if avg_drift < 0.1 else ('drifting' if avg_drift < 0.3 else 'unstable')
        }

    def _calculate_goal_drift(self, original: Dict[str, Any], current: Dict[str, Any]) -> float:
        """Calculate drift between goal sets."""
        if not original:
            return 0

        total_drift = 0
        comparison_count = 0

        for goal_name, original_value in original.items():
            if goal_name in current:
                current_value = current[goal_name]
                if isinstance(original_value, (int, float)) and isinstance(current_value, (int, float)):
                    if original_value != 0:
                        drift = abs(current_value - original_value) / abs(original_value)
                    else:
                        drift = abs(current_value)
                    total_drift += min(1, drift)
                elif original_value != current_value:
                    total_drift += 1
                comparison_count += 1
            else:
                total_drift += 1
                comparison_count += 1

        # Check for new goals not in original
        for goal_name in current:
            if goal_name not in original:
                total_drift += 0.5
                comparison_count += 1

        return total_drift / comparison_count if comparison_count > 0 else 0

    def _identify_changed_goals(self, prev: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
        """Identify which goals changed."""
        changed = []
        for goal in set(prev.keys()) | set(current.keys()):
            if prev.get(goal) != current.get(goal):
                changed.append(goal)
        return changed


class ContainmentAnalyzer:
    """Analyzes containment and isolation effectiveness."""

    def analyze_containment(self,
                           containment_config: Dict[str, Any],
                           breach_attempts: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze containment effectiveness."""
        containment_layers = containment_config.get('layers', [])
        isolation_level = containment_config.get('isolation_level', 'none')
        resource_limits = containment_config.get('resource_limits', {})

        layer_analysis = []
        for layer in containment_layers:
            layer_analysis.append({
                'name': layer.get('name'),
                'type': layer.get('type'),
                'effectiveness': layer.get('effectiveness', 0.5),
                'last_tested': layer.get('last_tested'),
                'known_bypasses': layer.get('known_bypasses', 0)
            })

        # Calculate overall containment effectiveness
        if layer_analysis:
            # Layers compound - more layers = better containment
            individual_effectiveness = [l['effectiveness'] for l in layer_analysis]
            # Combined effectiveness: 1 - product of failure rates
            failure_rates = [1 - e for e in individual_effectiveness]
            combined_failure = np.prod(failure_rates)
            containment_effectiveness = 1 - combined_failure
        else:
            containment_effectiveness = 0

        # Analyze breach attempts
        breach_analysis = {'total_attempts': 0, 'successful_breaches': 0, 'blocked': 0}
        if breach_attempts:
            breach_analysis['total_attempts'] = len(breach_attempts)
            breach_analysis['successful_breaches'] = sum(
                1 for b in breach_attempts if b.get('successful', False)
            )
            breach_analysis['blocked'] = (
                breach_analysis['total_attempts'] - breach_analysis['successful_breaches']
            )
            breach_analysis['breach_rate'] = (
                breach_analysis['successful_breaches'] / breach_analysis['total_attempts']
                if breach_analysis['total_attempts'] > 0 else 0
            )

        # Isolation level scoring
        isolation_scores = {
            'none': 0,
            'process': 0.3,
            'container': 0.5,
            'vm': 0.7,
            'air_gapped': 0.9,
            'physical': 1.0
        }
        isolation_score = isolation_scores.get(isolation_level, 0)

        return {
            'containment_effectiveness': containment_effectiveness,
            'isolation_level': isolation_level,
            'isolation_score': isolation_score,
            'containment_layers': len(containment_layers),
            'layer_analysis': layer_analysis,
            'resource_limits': resource_limits,
            'breach_analysis': breach_analysis,
            'overall_score': (containment_effectiveness + isolation_score) / 2,
            'containment_status': 'strong' if containment_effectiveness > 0.8 else (
                'moderate' if containment_effectiveness > 0.5 else 'weak'
            )
        }


class ReversibilityAnalyzer:
    """Analyzes action reversibility and undo capabilities."""

    def analyze_reversibility(self,
                              actions: List[Dict[str, Any]],
                              undo_capabilities: Dict[str, bool]) -> Dict[str, Any]:
        """Analyze reversibility of actions."""
        if not actions:
            return {'reversibility_score': 1.0, 'irreversible_actions': 0}

        reversible_actions = []
        irreversible_actions = []
        partially_reversible = []

        for action in actions:
            action_type = action.get('type', 'unknown')
            is_reversible = undo_capabilities.get(action_type, False)
            partial_undo = action.get('partial_undo_available', False)

            action_analysis = {
                'action_id': action.get('id'),
                'type': action_type,
                'impact_level': action.get('impact_level', 'unknown')
            }

            if is_reversible:
                reversible_actions.append(action_analysis)
            elif partial_undo:
                partially_reversible.append(action_analysis)
            else:
                irreversible_actions.append(action_analysis)

        total_actions = len(actions)
        reversibility_score = (
            (len(reversible_actions) + 0.5 * len(partially_reversible)) / total_actions
            if total_actions > 0 else 1.0
        )

        # High-impact irreversible actions are more concerning
        high_impact_irreversible = [
            a for a in irreversible_actions
            if a['impact_level'] in ['high', 'critical']
        ]

        return {
            'reversibility_score': reversibility_score,
            'reversible_actions': len(reversible_actions),
            'partially_reversible_actions': len(partially_reversible),
            'irreversible_actions': len(irreversible_actions),
            'high_impact_irreversible': len(high_impact_irreversible),
            'total_actions': total_actions,
            'irreversible_action_details': irreversible_actions,
            'high_impact_details': high_impact_irreversible,
            'risk_level': 'high' if high_impact_irreversible else (
                'medium' if irreversible_actions else 'low'
            )
        }


class UncertaintyRiskAnalyzer:
    """Analyzes uncertainty and unknown risks."""

    def analyze_uncertainty(self,
                           known_risks: List[Dict[str, Any]],
                           uncertainty_sources: List[str],
                           confidence_intervals: Dict[str, Tuple[float, float]] = None) -> Dict[str, Any]:
        """Analyze uncertainty and unknown risk factors."""
        if not known_risks and not uncertainty_sources:
            return {'uncertainty_score': 0.0, 'known_unknowns': 0}

        # Analyze known unknowns
        known_unknowns = len(uncertainty_sources)

        # Analyze confidence intervals
        confidence_analysis = {}
        if confidence_intervals:
            for metric, (lower, upper) in confidence_intervals.items():
                width = upper - lower
                confidence_analysis[metric] = {
                    'lower': lower,
                    'upper': upper,
                    'width': width,
                    'uncertainty_level': 'low' if width < 0.1 else (
                        'medium' if width < 0.3 else 'high'
                    )
                }

        # Calculate average uncertainty from confidence intervals
        if confidence_analysis:
            widths = [v['width'] for v in confidence_analysis.values()]
            avg_uncertainty = np.mean(widths)
        else:
            avg_uncertainty = 0.5  # Default moderate uncertainty

        # Analyze risk coverage
        risk_categories = set(r.get('category', 'unknown') for r in known_risks)
        expected_categories = {
            'technical', 'operational', 'ethical', 'legal',
            'reputational', 'financial', 'safety', 'security'
        }
        coverage = len(risk_categories & expected_categories) / len(expected_categories)

        # Unknown unknowns estimation (based on coverage gaps)
        estimated_unknown_unknowns = (1 - coverage) * 10  # Rough estimate

        return {
            'uncertainty_score': float(avg_uncertainty),
            'known_unknowns': known_unknowns,
            'estimated_unknown_unknowns': estimated_unknown_unknowns,
            'uncertainty_sources': uncertainty_sources,
            'risk_coverage': coverage,
            'confidence_analysis': confidence_analysis,
            'uncovered_risk_categories': list(expected_categories - risk_categories),
            'uncertainty_level': 'high' if avg_uncertainty > 0.3 or known_unknowns > 5 else (
                'medium' if avg_uncertainty > 0.15 or known_unknowns > 2 else 'low'
            )
        }


# ============================================================================
# Report Generator
# ============================================================================

class SafetyReportGenerator:
    """Generates comprehensive safety and long-term risk reports."""

    def __init__(self):
        self.harm_prevention_analyzer = HarmPreventionAnalyzer()
        self.safety_constraint_analyzer = SafetyConstraintAnalyzer()
        self.fail_safe_analyzer = FailSafeAnalyzer()
        self.content_safety_analyzer = ContentSafetyAnalyzer()
        self.boundary_analyzer = BoundaryAnalyzer()
        self.safety_margin_analyzer = SafetyMarginAnalyzer()
        self.safety_incident_analyzer = SafetyIncidentAnalyzer()
        self.adversarial_safety_analyzer = AdversarialSafetyAnalyzer()

        self.existential_risk_analyzer = ExistentialRiskAnalyzer()
        self.value_alignment_analyzer = ValueAlignmentAnalyzer()
        self.control_robustness_analyzer = ControlRobustnessAnalyzer()
        self.capability_risk_analyzer = CapabilityRiskAnalyzer()
        self.goal_stability_analyzer = GoalStabilityAnalyzer()
        self.containment_analyzer = ContainmentAnalyzer()
        self.reversibility_analyzer = ReversibilityAnalyzer()
        self.uncertainty_analyzer = UncertaintyRiskAnalyzer()

    def generate_safety_report(self,
                               outputs: List[Dict[str, Any]] = None,
                               harm_assessments: List[Dict[str, Any]] = None,
                               safety_constraints: List[SafetyConstraint] = None,
                               actions: List[Dict[str, Any]] = None,
                               incidents: List[SafetyIncident] = None) -> Dict[str, Any]:
        """Generate safety analysis report."""
        report = {
            'report_type': 'safety_analysis',
            'timestamp': datetime.now().isoformat()
        }

        if outputs is not None and harm_assessments is not None:
            report['harm_prevention'] = self.harm_prevention_analyzer.analyze_harm_prevention(
                outputs, harm_assessments
            )

        if safety_constraints and actions:
            report['constraint_compliance'] = self.safety_constraint_analyzer.analyze_constraints(
                actions, safety_constraints
            )

        if incidents:
            report['incident_analysis'] = self.safety_incident_analyzer.analyze_incidents(incidents)

        return report

    def generate_risk_report(self,
                            risk_scenarios: List[RiskScenario] = None,
                            capabilities: Dict[str, Dict[str, Any]] = None,
                            goal_history: List[Dict[str, Any]] = None,
                            original_goals: Dict[str, Any] = None,
                            containment_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate long-term risk analysis report."""
        report = {
            'report_type': 'long_term_risk_analysis',
            'timestamp': datetime.now().isoformat()
        }

        if risk_scenarios:
            report['existential_risk'] = self.existential_risk_analyzer.analyze_existential_risk(
                risk_scenarios, capabilities or {}
            )

        if capabilities:
            report['capability_risk'] = self.capability_risk_analyzer.analyze_capability_risk(
                capabilities
            )

        if goal_history and original_goals:
            report['goal_stability'] = self.goal_stability_analyzer.analyze_goal_stability(
                goal_history, original_goals
            )

        if containment_config:
            report['containment'] = self.containment_analyzer.analyze_containment(
                containment_config
            )

        return report

    def generate_full_report(self,
                            outputs: List[Dict[str, Any]] = None,
                            harm_assessments: List[Dict[str, Any]] = None,
                            safety_constraints: List[SafetyConstraint] = None,
                            actions: List[Dict[str, Any]] = None,
                            incidents: List[SafetyIncident] = None,
                            risk_scenarios: List[RiskScenario] = None,
                            capabilities: Dict[str, Dict[str, Any]] = None,
                            goal_history: List[Dict[str, Any]] = None,
                            original_goals: Dict[str, Any] = None,
                            containment_config: Dict[str, Any] = None,
                            system_values: Dict[str, float] = None,
                            target_values: Dict[str, float] = None) -> Dict[str, Any]:
        """Generate comprehensive safety and risk report."""
        report = {
            'report_type': 'comprehensive_safety_risk_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        # Safety analysis
        safety_report = self.generate_safety_report(
            outputs, harm_assessments, safety_constraints, actions, incidents
        )
        for key, value in safety_report.items():
            if key not in ['report_type', 'timestamp']:
                report[f'safety_{key}'] = value

        # Risk analysis
        risk_report = self.generate_risk_report(
            risk_scenarios, capabilities, goal_history, original_goals, containment_config
        )
        for key, value in risk_report.items():
            if key not in ['report_type', 'timestamp']:
                report[f'risk_{key}'] = value

        # Value alignment
        if system_values and target_values:
            report['value_alignment'] = self.value_alignment_analyzer.analyze_value_alignment(
                system_values, target_values
            )

        # Calculate overall safety score
        scores = []
        if 'safety_harm_prevention' in report:
            scores.append(report['safety_harm_prevention'].get('harm_prevention_score', 0))
        if 'safety_constraint_compliance' in report:
            scores.append(report['safety_constraint_compliance'].get('constraint_compliance', 0))
        if 'risk_capability_risk' in report:
            scores.append(1 - report['risk_capability_risk'].get('capability_risk', 0))
        if 'value_alignment' in report:
            scores.append(report['value_alignment'].get('value_alignment_score', 0))

        report['overall_safety_score'] = float(np.mean(scores)) if scores else 0.0

        return report

    def export_report(self, report: Dict[str, Any], filepath: str, format: str = 'json') -> None:
        """Export report to file."""
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format == 'markdown':
            md_content = self._report_to_markdown(report)
            with open(filepath, 'w') as f:
                f.write(md_content)

    def _report_to_markdown(self, report: Dict[str, Any]) -> str:
        """Convert report to markdown format."""
        lines = [
            f"# {report.get('report_type', 'Safety Analysis Report')}",
            f"\n**Generated:** {report.get('timestamp', 'N/A')}",
            f"\n**Overall Safety Score:** {report.get('overall_safety_score', 0):.2%}",
            "\n---\n"
        ]

        if 'safety_harm_prevention' in report:
            lines.append("## Harm Prevention\n")
            hp = report['safety_harm_prevention']
            lines.append(f"- **Harm Prevention Score:** {hp.get('harm_prevention_score', 0):.2%}")
            lines.append(f"- **Harmful Outputs:** {hp.get('harmful_outputs', 0)}")
            lines.append("")

        if 'risk_existential_risk' in report:
            lines.append("## Existential Risk\n")
            er = report['risk_existential_risk']
            lines.append(f"- **Risk Score:** {er.get('existential_risk_score', 0):.2%}")
            lines.append(f"- **Risk Level:** {er.get('risk_level', 'unknown')}")
            lines.append("")

        if 'value_alignment' in report:
            lines.append("## Value Alignment\n")
            va = report['value_alignment']
            lines.append(f"- **Alignment Score:** {va.get('value_alignment_score', 0):.2%}")
            lines.append(f"- **Status:** {va.get('alignment_status', 'unknown')}")
            lines.append("")

        return "\n".join(lines)
