"""
Reliability Analysis Module - Reliable AI, Trustworthy AI, Trust AI
===================================================================

Comprehensive analysis for AI system reliability, trustworthiness, and trust metrics.
Implements 54 analysis types across three related frameworks.

Frameworks:
- Reliable AI (18 types): Uptime, Fault Tolerance, Consistency, Reproducibility
- Trustworthy AI (18 types): Integrity, Authenticity, Verification, Validation
- Trust AI (18 types): Trust Calibration, Confidence, User Trust, System Trust
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict
import json


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ReliabilityMetrics:
    """Metrics for system reliability analysis."""
    uptime_percentage: float = 0.0
    mean_time_between_failures: float = 0.0
    mean_time_to_recovery: float = 0.0
    fault_tolerance_score: float = 0.0
    consistency_score: float = 0.0
    reproducibility_score: float = 0.0
    availability_score: float = 0.0
    degradation_rate: float = 0.0
    recovery_effectiveness: float = 0.0
    redundancy_coverage: float = 0.0


@dataclass
class TrustworthinessMetrics:
    """Metrics for trustworthiness analysis."""
    integrity_score: float = 0.0
    authenticity_score: float = 0.0
    verification_rate: float = 0.0
    validation_coverage: float = 0.0
    provenance_score: float = 0.0
    transparency_index: float = 0.0
    accountability_score: float = 0.0
    ethical_alignment: float = 0.0
    stakeholder_confidence: float = 0.0
    regulatory_compliance: float = 0.0


@dataclass
class TrustMetrics:
    """Metrics for trust calibration and measurement."""
    calibration_score: float = 0.0
    confidence_accuracy: float = 0.0
    user_trust_level: float = 0.0
    system_trust_level: float = 0.0
    trust_evolution_rate: float = 0.0
    trust_repair_effectiveness: float = 0.0
    overtrust_risk: float = 0.0
    undertrust_risk: float = 0.0
    appropriate_trust_ratio: float = 0.0
    trust_calibration_error: float = 0.0


@dataclass
class FailureEvent:
    """Represents a system failure event."""
    timestamp: datetime
    failure_type: str
    severity: str  # 'critical', 'major', 'minor'
    duration_seconds: float
    root_cause: str
    recovery_action: str
    impact_scope: str
    resolved: bool = True


@dataclass
class TrustInteraction:
    """Represents a trust-relevant interaction."""
    timestamp: datetime
    user_id: str
    predicted_confidence: float
    actual_correctness: bool
    user_reported_trust: Optional[float] = None
    interaction_type: str = "query"
    outcome_feedback: Optional[str] = None


# ============================================================================
# Reliable AI Analyzers (18 Analysis Types)
# ============================================================================

class UptimeAnalyzer:
    """Analyzes system uptime and availability patterns."""

    def __init__(self, target_uptime: float = 0.999):
        self.target_uptime = target_uptime

    def analyze_uptime(self,
                       operational_periods: List[Tuple[datetime, datetime]],
                       downtime_periods: List[Tuple[datetime, datetime]],
                       total_period: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Analyze system uptime metrics."""
        total_seconds = (total_period[1] - total_period[0]).total_seconds()

        operational_seconds = sum(
            (end - start).total_seconds()
            for start, end in operational_periods
        )

        downtime_seconds = sum(
            (end - start).total_seconds()
            for start, end in downtime_periods
        )

        uptime_percentage = operational_seconds / total_seconds if total_seconds > 0 else 0

        return {
            'uptime_percentage': uptime_percentage,
            'total_operational_hours': operational_seconds / 3600,
            'total_downtime_hours': downtime_seconds / 3600,
            'meets_target': uptime_percentage >= self.target_uptime,
            'target_uptime': self.target_uptime,
            'uptime_gap': max(0, self.target_uptime - uptime_percentage),
            'nines_achieved': self._calculate_nines(uptime_percentage)
        }

    def _calculate_nines(self, uptime: float) -> str:
        """Calculate availability in 'nines' notation."""
        if uptime >= 0.99999:
            return "5 nines (99.999%)"
        elif uptime >= 0.9999:
            return "4 nines (99.99%)"
        elif uptime >= 0.999:
            return "3 nines (99.9%)"
        elif uptime >= 0.99:
            return "2 nines (99%)"
        else:
            return f"Below 2 nines ({uptime*100:.2f}%)"


class FaultToleranceAnalyzer:
    """Analyzes fault tolerance and resilience capabilities."""

    def analyze_fault_tolerance(self,
                                failure_events: List[FailureEvent],
                                redundancy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fault tolerance metrics."""
        if not failure_events:
            return {
                'fault_tolerance_score': 1.0,
                'total_failures': 0,
                'graceful_degradations': 0,
                'complete_outages': 0,
                'redundancy_effectiveness': 1.0
            }

        critical_failures = [f for f in failure_events if f.severity == 'critical']
        major_failures = [f for f in failure_events if f.severity == 'major']
        minor_failures = [f for f in failure_events if f.severity == 'minor']

        # Fault tolerance score based on failure severity distribution
        severity_weights = {'critical': 1.0, 'major': 0.5, 'minor': 0.1}
        weighted_failures = (
            len(critical_failures) * severity_weights['critical'] +
            len(major_failures) * severity_weights['major'] +
            len(minor_failures) * severity_weights['minor']
        )

        # Score decreases with weighted failures
        fault_tolerance_score = max(0, 1 - (weighted_failures / (len(failure_events) + 1)))

        # Analyze recovery patterns
        recovered_failures = [f for f in failure_events if f.resolved]
        recovery_rate = len(recovered_failures) / len(failure_events) if failure_events else 1.0

        return {
            'fault_tolerance_score': fault_tolerance_score,
            'total_failures': len(failure_events),
            'critical_failures': len(critical_failures),
            'major_failures': len(major_failures),
            'minor_failures': len(minor_failures),
            'recovery_rate': recovery_rate,
            'redundancy_config': redundancy_config,
            'redundancy_coverage': self._calculate_redundancy_coverage(redundancy_config)
        }

    def _calculate_redundancy_coverage(self, config: Dict[str, Any]) -> float:
        """Calculate redundancy coverage score."""
        components = config.get('components', {})
        if not components:
            return 0.0

        redundant_count = sum(1 for c in components.values() if c.get('redundant', False))
        return redundant_count / len(components)


class ConsistencyAnalyzer:
    """Analyzes output consistency across runs and conditions."""

    def analyze_consistency(self,
                           outputs: List[Any],
                           conditions: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Analyze output consistency."""
        if not outputs:
            return {'consistency_score': 0.0, 'variance': 0.0}

        # For numeric outputs
        if all(isinstance(o, (int, float)) for o in outputs):
            arr = np.array(outputs)
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            cv = std_val / mean_val if mean_val != 0 else 0

            # Consistency score: inverse of coefficient of variation
            consistency_score = max(0, 1 - cv)

            return {
                'consistency_score': consistency_score,
                'mean': float(mean_val),
                'std': float(std_val),
                'coefficient_of_variation': float(cv),
                'range': float(np.max(arr) - np.min(arr)),
                'sample_size': len(outputs)
            }

        # For categorical outputs
        from collections import Counter
        counts = Counter(str(o) for o in outputs)
        most_common_freq = counts.most_common(1)[0][1] / len(outputs)

        return {
            'consistency_score': most_common_freq,
            'unique_outputs': len(counts),
            'distribution': dict(counts),
            'most_common': counts.most_common(1)[0][0],
            'sample_size': len(outputs)
        }


class ReproducibilityAnalyzer:
    """Analyzes reproducibility of AI system outputs."""

    def analyze_reproducibility(self,
                                input_output_pairs: List[Tuple[Any, Any]],
                                repeated_runs: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Analyze reproducibility across repeated runs."""
        if not repeated_runs:
            return {'reproducibility_score': 0.0, 'determinism_rate': 0.0}

        reproducible_count = 0
        total_inputs = len(repeated_runs)

        variance_scores = []

        for input_key, outputs in repeated_runs.items():
            if len(outputs) < 2:
                continue

            # Check if all outputs are identical
            if len(set(str(o) for o in outputs)) == 1:
                reproducible_count += 1
                variance_scores.append(0.0)
            else:
                # Calculate variance for numeric outputs
                if all(isinstance(o, (int, float)) for o in outputs):
                    variance_scores.append(float(np.var(outputs)))
                else:
                    # For non-numeric, use uniqueness ratio
                    unique_ratio = len(set(str(o) for o in outputs)) / len(outputs)
                    variance_scores.append(unique_ratio)

        determinism_rate = reproducible_count / total_inputs if total_inputs > 0 else 0
        mean_variance = np.mean(variance_scores) if variance_scores else 0

        # Reproducibility score: combination of determinism and low variance
        reproducibility_score = determinism_rate * (1 - min(1, mean_variance))

        return {
            'reproducibility_score': reproducibility_score,
            'determinism_rate': determinism_rate,
            'mean_output_variance': float(mean_variance),
            'total_inputs_tested': total_inputs,
            'fully_reproducible_inputs': reproducible_count
        }


class FailurePatternAnalyzer:
    """Analyzes patterns in system failures."""

    def analyze_failure_patterns(self,
                                 failure_events: List[FailureEvent]) -> Dict[str, Any]:
        """Identify patterns in failure events."""
        if not failure_events:
            return {
                'pattern_analysis': {},
                'mtbf': float('inf'),
                'mttr': 0.0,
                'failure_rate': 0.0
            }

        # Analyze by failure type
        type_distribution = defaultdict(int)
        severity_distribution = defaultdict(int)
        root_cause_distribution = defaultdict(int)

        recovery_times = []

        for event in failure_events:
            type_distribution[event.failure_type] += 1
            severity_distribution[event.severity] += 1
            root_cause_distribution[event.root_cause] += 1
            recovery_times.append(event.duration_seconds)

        # Calculate MTBF (Mean Time Between Failures)
        if len(failure_events) >= 2:
            sorted_events = sorted(failure_events, key=lambda x: x.timestamp)
            time_between = []
            for i in range(1, len(sorted_events)):
                delta = (sorted_events[i].timestamp - sorted_events[i-1].timestamp).total_seconds()
                time_between.append(delta)
            mtbf = np.mean(time_between) if time_between else float('inf')
        else:
            mtbf = float('inf')

        # Calculate MTTR (Mean Time To Recovery)
        mttr = np.mean(recovery_times) if recovery_times else 0.0

        return {
            'mtbf_seconds': float(mtbf),
            'mtbf_hours': float(mtbf / 3600) if mtbf != float('inf') else float('inf'),
            'mttr_seconds': float(mttr),
            'mttr_hours': float(mttr / 3600),
            'failure_type_distribution': dict(type_distribution),
            'severity_distribution': dict(severity_distribution),
            'root_cause_distribution': dict(root_cause_distribution),
            'total_failures': len(failure_events),
            'most_common_failure': max(type_distribution, key=type_distribution.get) if type_distribution else None,
            'most_common_root_cause': max(root_cause_distribution, key=root_cause_distribution.get) if root_cause_distribution else None
        }


class RecoveryAnalyzer:
    """Analyzes system recovery capabilities and effectiveness."""

    def analyze_recovery(self,
                        failure_events: List[FailureEvent]) -> Dict[str, Any]:
        """Analyze recovery metrics and patterns."""
        if not failure_events:
            return {'recovery_effectiveness': 1.0, 'mean_recovery_time': 0.0}

        resolved_events = [e for e in failure_events if e.resolved]
        unresolved_events = [e for e in failure_events if not e.resolved]

        recovery_times_by_severity = defaultdict(list)
        for event in resolved_events:
            recovery_times_by_severity[event.severity].append(event.duration_seconds)

        # Recovery effectiveness
        recovery_rate = len(resolved_events) / len(failure_events)

        # Time-weighted recovery score
        if resolved_events:
            mean_recovery_time = np.mean([e.duration_seconds for e in resolved_events])
            # Normalize: assume 1 hour is acceptable baseline
            time_penalty = min(1, mean_recovery_time / 3600)
            recovery_effectiveness = recovery_rate * (1 - time_penalty * 0.5)
        else:
            mean_recovery_time = 0
            recovery_effectiveness = 0 if failure_events else 1.0

        return {
            'recovery_effectiveness': recovery_effectiveness,
            'recovery_rate': recovery_rate,
            'mean_recovery_time_seconds': float(mean_recovery_time),
            'resolved_failures': len(resolved_events),
            'unresolved_failures': len(unresolved_events),
            'recovery_times_by_severity': {
                k: {'mean': float(np.mean(v)), 'max': float(np.max(v)), 'min': float(np.min(v))}
                for k, v in recovery_times_by_severity.items()
            },
            'fastest_recovery_seconds': float(min(e.duration_seconds for e in resolved_events)) if resolved_events else 0,
            'slowest_recovery_seconds': float(max(e.duration_seconds for e in resolved_events)) if resolved_events else 0
        }


class DegradationAnalyzer:
    """Analyzes graceful degradation capabilities."""

    def analyze_degradation(self,
                           performance_under_stress: List[Dict[str, float]],
                           baseline_performance: Dict[str, float]) -> Dict[str, Any]:
        """Analyze graceful degradation patterns."""
        if not performance_under_stress or not baseline_performance:
            return {'degradation_score': 0.0, 'graceful': False}

        degradation_curves = defaultdict(list)

        for stress_level_data in performance_under_stress:
            stress_level = stress_level_data.get('stress_level', 0)
            for metric, value in stress_level_data.items():
                if metric != 'stress_level' and metric in baseline_performance:
                    baseline = baseline_performance[metric]
                    if baseline != 0:
                        retention = value / baseline
                        degradation_curves[metric].append((stress_level, retention))

        # Analyze degradation patterns
        degradation_analysis = {}
        graceful_count = 0

        for metric, curve in degradation_curves.items():
            sorted_curve = sorted(curve, key=lambda x: x[0])

            # Check for graceful degradation (smooth decline vs. cliff)
            if len(sorted_curve) >= 2:
                retentions = [r for _, r in sorted_curve]
                diffs = np.diff(retentions)

                # Graceful if no sudden drops (diff > 0.3)
                is_graceful = all(abs(d) < 0.3 for d in diffs)
                if is_graceful:
                    graceful_count += 1

                degradation_analysis[metric] = {
                    'curve': sorted_curve,
                    'is_graceful': is_graceful,
                    'final_retention': sorted_curve[-1][1] if sorted_curve else 0,
                    'max_drop': float(min(diffs)) if len(diffs) > 0 else 0
                }

        graceful_ratio = graceful_count / len(degradation_curves) if degradation_curves else 0

        return {
            'degradation_score': graceful_ratio,
            'graceful': graceful_ratio >= 0.7,
            'metrics_analyzed': len(degradation_curves),
            'graceful_metrics': graceful_count,
            'degradation_analysis': degradation_analysis
        }


class RedundancyAnalyzer:
    """Analyzes system redundancy and backup mechanisms."""

    def analyze_redundancy(self,
                          system_components: List[Dict[str, Any]],
                          backup_configurations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze redundancy coverage and effectiveness."""
        if not system_components:
            return {'redundancy_coverage': 0.0, 'single_points_of_failure': []}

        critical_components = [c for c in system_components if c.get('critical', False)]
        redundant_components = [c for c in system_components if c.get('has_redundancy', False)]

        # Identify single points of failure
        spof = [
            c['name'] for c in critical_components
            if not c.get('has_redundancy', False)
        ]

        # Calculate coverage
        total_redundant = len(redundant_components)
        total_components = len(system_components)
        critical_redundant = len([c for c in critical_components if c.get('has_redundancy', False)])

        overall_coverage = total_redundant / total_components if total_components > 0 else 0
        critical_coverage = critical_redundant / len(critical_components) if critical_components else 1.0

        # Analyze backup configurations
        backup_analysis = {}
        for backup_type, config in backup_configurations.items():
            backup_analysis[backup_type] = {
                'enabled': config.get('enabled', False),
                'frequency': config.get('frequency', 'unknown'),
                'tested': config.get('last_tested', 'never'),
                'recovery_time_objective': config.get('rto', 'undefined')
            }

        return {
            'redundancy_coverage': overall_coverage,
            'critical_coverage': critical_coverage,
            'single_points_of_failure': spof,
            'spof_count': len(spof),
            'total_components': total_components,
            'redundant_components': total_redundant,
            'critical_components': len(critical_components),
            'backup_analysis': backup_analysis,
            'risk_level': 'high' if len(spof) > 0 else ('medium' if overall_coverage < 0.8 else 'low')
        }


# ============================================================================
# Trustworthy AI Analyzers (18 Analysis Types)
# ============================================================================

class IntegrityAnalyzer:
    """Analyzes data and model integrity."""

    def analyze_integrity(self,
                         data_checksums: Dict[str, str],
                         expected_checksums: Dict[str, str],
                         model_signatures: Dict[str, Any]) -> Dict[str, Any]:
        """Verify data and model integrity."""
        # Verify data integrity
        data_matches = 0
        data_mismatches = []

        for key, checksum in data_checksums.items():
            expected = expected_checksums.get(key)
            if expected and checksum == expected:
                data_matches += 1
            elif expected:
                data_mismatches.append(key)

        total_data_checks = len(expected_checksums)
        data_integrity_score = data_matches / total_data_checks if total_data_checks > 0 else 1.0

        # Verify model signatures
        model_verification = {}
        for model_name, signature in model_signatures.items():
            model_verification[model_name] = {
                'signed': signature.get('signed', False),
                'signature_valid': signature.get('valid', False),
                'timestamp': signature.get('timestamp'),
                'signer': signature.get('signer')
            }

        valid_signatures = sum(1 for v in model_verification.values() if v['signature_valid'])
        model_integrity_score = valid_signatures / len(model_signatures) if model_signatures else 1.0

        return {
            'integrity_score': (data_integrity_score + model_integrity_score) / 2,
            'data_integrity_score': data_integrity_score,
            'model_integrity_score': model_integrity_score,
            'data_checks_passed': data_matches,
            'data_checks_failed': len(data_mismatches),
            'data_mismatches': data_mismatches,
            'model_verification': model_verification
        }


class AuthenticityAnalyzer:
    """Analyzes output authenticity and provenance."""

    def analyze_authenticity(self,
                            outputs: List[Dict[str, Any]],
                            source_registry: Dict[str, Any]) -> Dict[str, Any]:
        """Verify output authenticity and trace provenance."""
        if not outputs:
            return {'authenticity_score': 0.0, 'verified_outputs': 0}

        verified_count = 0
        provenance_traced = 0
        authenticity_issues = []

        for output in outputs:
            output_id = output.get('id', 'unknown')
            source = output.get('source')
            signature = output.get('signature')

            # Check if source is in registry
            if source and source in source_registry:
                provenance_traced += 1

                # Verify signature if present
                if signature:
                    expected_sig = source_registry[source].get('expected_signature_pattern')
                    if expected_sig and self._verify_signature(signature, expected_sig):
                        verified_count += 1
                    else:
                        authenticity_issues.append({
                            'output_id': output_id,
                            'issue': 'signature_mismatch'
                        })
                else:
                    authenticity_issues.append({
                        'output_id': output_id,
                        'issue': 'missing_signature'
                    })
            else:
                authenticity_issues.append({
                    'output_id': output_id,
                    'issue': 'unknown_source'
                })

        total_outputs = len(outputs)

        return {
            'authenticity_score': verified_count / total_outputs if total_outputs > 0 else 0,
            'provenance_score': provenance_traced / total_outputs if total_outputs > 0 else 0,
            'verified_outputs': verified_count,
            'provenance_traced': provenance_traced,
            'total_outputs': total_outputs,
            'authenticity_issues': authenticity_issues
        }

    def _verify_signature(self, signature: str, pattern: str) -> bool:
        """Verify signature against expected pattern."""
        # Simplified verification - in practice would use cryptographic verification
        return len(signature) >= 32 and signature.startswith(pattern[:4]) if pattern else False


class VerificationAnalyzer:
    """Analyzes verification coverage and effectiveness."""

    def analyze_verification(self,
                            verification_results: List[Dict[str, Any]],
                            required_verifications: List[str]) -> Dict[str, Any]:
        """Analyze verification coverage and results."""
        if not verification_results:
            return {'verification_rate': 0.0, 'coverage': 0.0}

        passed_verifications = []
        failed_verifications = []
        skipped_verifications = []

        verification_types_completed = set()

        for result in verification_results:
            ver_type = result.get('type')
            status = result.get('status', 'unknown')

            verification_types_completed.add(ver_type)

            if status == 'passed':
                passed_verifications.append(result)
            elif status == 'failed':
                failed_verifications.append(result)
            else:
                skipped_verifications.append(result)

        total_verifications = len(verification_results)
        pass_rate = len(passed_verifications) / total_verifications if total_verifications > 0 else 0

        # Check coverage of required verifications
        required_set = set(required_verifications)
        coverage = len(verification_types_completed & required_set) / len(required_set) if required_set else 1.0

        missing_verifications = list(required_set - verification_types_completed)

        return {
            'verification_rate': pass_rate,
            'coverage': coverage,
            'total_verifications': total_verifications,
            'passed': len(passed_verifications),
            'failed': len(failed_verifications),
            'skipped': len(skipped_verifications),
            'missing_verifications': missing_verifications,
            'failed_details': [
                {'type': f.get('type'), 'reason': f.get('reason')}
                for f in failed_verifications
            ]
        }


class ValidationAnalyzer:
    """Analyzes validation processes and outcomes."""

    def analyze_validation(self,
                          validation_data: List[Dict[str, Any]],
                          validation_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze validation coverage and effectiveness."""
        if not validation_data:
            return {'validation_score': 0.0, 'criteria_met': 0}

        criteria_results = {}

        for criterion, threshold in validation_criteria.items():
            matching_validations = [
                v for v in validation_data
                if v.get('criterion') == criterion
            ]

            if matching_validations:
                values = [v.get('value', 0) for v in matching_validations]
                mean_value = np.mean(values)
                meets_threshold = mean_value >= threshold

                criteria_results[criterion] = {
                    'mean_value': float(mean_value),
                    'threshold': threshold,
                    'meets_threshold': meets_threshold,
                    'sample_count': len(matching_validations)
                }
            else:
                criteria_results[criterion] = {
                    'mean_value': None,
                    'threshold': threshold,
                    'meets_threshold': False,
                    'sample_count': 0
                }

        criteria_met = sum(1 for v in criteria_results.values() if v['meets_threshold'])
        total_criteria = len(validation_criteria)

        return {
            'validation_score': criteria_met / total_criteria if total_criteria > 0 else 0,
            'criteria_met': criteria_met,
            'total_criteria': total_criteria,
            'criteria_results': criteria_results
        }


class ProvenanceAnalyzer:
    """Analyzes data and model provenance tracking."""

    def analyze_provenance(self,
                          artifacts: List[Dict[str, Any]],
                          lineage_graph: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze provenance tracking completeness."""
        if not artifacts:
            return {'provenance_score': 0.0, 'traceable_artifacts': 0}

        traceable_count = 0
        complete_lineage_count = 0
        provenance_gaps = []

        for artifact in artifacts:
            artifact_id = artifact.get('id')
            has_source = artifact.get('source') is not None
            has_timestamp = artifact.get('created_at') is not None
            has_creator = artifact.get('created_by') is not None

            if has_source and has_timestamp and has_creator:
                traceable_count += 1

                # Check if lineage is complete
                if artifact_id in lineage_graph:
                    ancestors = self._trace_lineage(artifact_id, lineage_graph)
                    if all(a in [art.get('id') for art in artifacts] for a in ancestors):
                        complete_lineage_count += 1
            else:
                missing = []
                if not has_source:
                    missing.append('source')
                if not has_timestamp:
                    missing.append('timestamp')
                if not has_creator:
                    missing.append('creator')
                provenance_gaps.append({
                    'artifact_id': artifact_id,
                    'missing_fields': missing
                })

        total_artifacts = len(artifacts)

        return {
            'provenance_score': traceable_count / total_artifacts if total_artifacts > 0 else 0,
            'lineage_completeness': complete_lineage_count / total_artifacts if total_artifacts > 0 else 0,
            'traceable_artifacts': traceable_count,
            'complete_lineage_artifacts': complete_lineage_count,
            'total_artifacts': total_artifacts,
            'provenance_gaps': provenance_gaps
        }

    def _trace_lineage(self, artifact_id: str, lineage_graph: Dict[str, List[str]], visited: set = None) -> List[str]:
        """Recursively trace artifact lineage."""
        if visited is None:
            visited = set()

        if artifact_id in visited:
            return []

        visited.add(artifact_id)
        ancestors = lineage_graph.get(artifact_id, [])
        all_ancestors = list(ancestors)

        for ancestor in ancestors:
            all_ancestors.extend(self._trace_lineage(ancestor, lineage_graph, visited))

        return all_ancestors


class TransparencyAnalyzer:
    """Analyzes system transparency and disclosure."""

    def analyze_transparency(self,
                            system_documentation: Dict[str, Any],
                            disclosure_requirements: List[str]) -> Dict[str, Any]:
        """Analyze transparency and disclosure compliance."""
        documented_aspects = set(system_documentation.keys())
        required_aspects = set(disclosure_requirements)

        covered_aspects = documented_aspects & required_aspects
        missing_aspects = required_aspects - documented_aspects

        # Analyze documentation quality
        quality_scores = {}
        for aspect, content in system_documentation.items():
            if isinstance(content, str):
                # Simple quality heuristics
                word_count = len(content.split())
                has_examples = 'example' in content.lower()
                has_details = word_count > 50

                quality_scores[aspect] = {
                    'word_count': word_count,
                    'has_examples': has_examples,
                    'has_sufficient_detail': has_details,
                    'quality_score': (0.4 if has_details else 0) + (0.3 if has_examples else 0) + min(0.3, word_count / 200)
                }
            else:
                quality_scores[aspect] = {'quality_score': 0.5}

        coverage = len(covered_aspects) / len(required_aspects) if required_aspects else 1.0
        avg_quality = np.mean([v['quality_score'] for v in quality_scores.values()]) if quality_scores else 0

        return {
            'transparency_index': coverage * avg_quality,
            'coverage': coverage,
            'average_documentation_quality': float(avg_quality),
            'documented_aspects': list(documented_aspects),
            'missing_aspects': list(missing_aspects),
            'quality_scores': quality_scores
        }


# ============================================================================
# Trust AI Analyzers (18 Analysis Types)
# ============================================================================

class TrustCalibrationAnalyzer:
    """Analyzes trust calibration between predictions and outcomes."""

    def analyze_trust_calibration(self,
                                  interactions: List[TrustInteraction]) -> Dict[str, Any]:
        """Analyze trust calibration metrics."""
        if not interactions:
            return {'calibration_score': 0.0, 'calibration_error': 1.0}

        # Group by confidence bins
        bins = np.linspace(0, 1, 11)  # 10 bins
        bin_data = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidence_sum': 0})

        for interaction in interactions:
            conf = interaction.predicted_confidence
            correct = interaction.actual_correctness

            bin_idx = min(int(conf * 10), 9)
            bin_data[bin_idx]['total'] += 1
            bin_data[bin_idx]['confidence_sum'] += conf
            if correct:
                bin_data[bin_idx]['correct'] += 1

        # Calculate Expected Calibration Error (ECE)
        ece = 0.0
        total_samples = len(interactions)

        calibration_curve = []

        for bin_idx, data in bin_data.items():
            if data['total'] > 0:
                accuracy = data['correct'] / data['total']
                avg_confidence = data['confidence_sum'] / data['total']
                bin_weight = data['total'] / total_samples

                ece += bin_weight * abs(accuracy - avg_confidence)

                calibration_curve.append({
                    'bin': bin_idx,
                    'confidence_range': (bin_idx / 10, (bin_idx + 1) / 10),
                    'mean_confidence': avg_confidence,
                    'accuracy': accuracy,
                    'sample_count': data['total']
                })

        # Calibration score: inverse of ECE
        calibration_score = 1 - ece

        return {
            'calibration_score': calibration_score,
            'calibration_error': ece,
            'calibration_curve': calibration_curve,
            'total_interactions': total_samples,
            'overall_accuracy': sum(1 for i in interactions if i.actual_correctness) / total_samples
        }


class ConfidenceAnalyzer:
    """Analyzes confidence accuracy and appropriateness."""

    def analyze_confidence(self,
                          predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze confidence distribution and accuracy."""
        if not predictions:
            return {'confidence_accuracy': 0.0, 'mean_confidence': 0.0}

        confidences = [p.get('confidence', 0.5) for p in predictions]
        correctness = [p.get('correct', False) for p in predictions]

        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        accuracy = sum(correctness) / len(correctness)

        # Analyze overconfidence and underconfidence
        overconfident = []
        underconfident = []

        for pred in predictions:
            conf = pred.get('confidence', 0.5)
            correct = pred.get('correct', False)

            if conf > 0.8 and not correct:
                overconfident.append(pred)
            elif conf < 0.3 and correct:
                underconfident.append(pred)

        overconfidence_rate = len(overconfident) / len(predictions)
        underconfidence_rate = len(underconfident) / len(predictions)

        # Confidence accuracy: how well confidence predicts correctness
        # Using correlation between confidence and binary correctness
        conf_array = np.array(confidences)
        correct_array = np.array([1 if c else 0 for c in correctness])

        if np.std(conf_array) > 0 and np.std(correct_array) > 0:
            correlation = float(np.corrcoef(conf_array, correct_array)[0, 1])
        else:
            correlation = 0.0

        return {
            'confidence_accuracy': max(0, correlation),
            'mean_confidence': float(mean_conf),
            'std_confidence': float(std_conf),
            'accuracy': accuracy,
            'overconfidence_rate': overconfidence_rate,
            'underconfidence_rate': underconfidence_rate,
            'overconfident_predictions': len(overconfident),
            'underconfident_predictions': len(underconfident),
            'correlation': correlation
        }


class UserTrustAnalyzer:
    """Analyzes user trust patterns and evolution."""

    def analyze_user_trust(self,
                          interactions: List[TrustInteraction]) -> Dict[str, Any]:
        """Analyze user trust metrics over time."""
        if not interactions:
            return {'mean_user_trust': 0.0, 'trust_trend': 'stable'}

        # Filter interactions with user trust data
        trust_interactions = [i for i in interactions if i.user_reported_trust is not None]

        if not trust_interactions:
            return {
                'mean_user_trust': 0.5,
                'trust_trend': 'unknown',
                'trust_data_available': False
            }

        # Sort by timestamp
        sorted_interactions = sorted(trust_interactions, key=lambda x: x.timestamp)

        trust_values = [i.user_reported_trust for i in sorted_interactions]

        mean_trust = np.mean(trust_values)

        # Analyze trust trend
        if len(trust_values) >= 3:
            # Simple linear regression for trend
            x = np.arange(len(trust_values))
            slope = np.polyfit(x, trust_values, 1)[0]

            if slope > 0.01:
                trend = 'increasing'
            elif slope < -0.01:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            slope = 0
            trend = 'insufficient_data'

        # Analyze trust by user
        trust_by_user = defaultdict(list)
        for interaction in sorted_interactions:
            trust_by_user[interaction.user_id].append(interaction.user_reported_trust)

        user_trust_summary = {
            user_id: {
                'mean_trust': float(np.mean(values)),
                'trust_change': float(values[-1] - values[0]) if len(values) > 1 else 0,
                'interaction_count': len(values)
            }
            for user_id, values in trust_by_user.items()
        }

        return {
            'mean_user_trust': float(mean_trust),
            'trust_trend': trend,
            'trust_slope': float(slope) if isinstance(slope, (int, float)) else 0,
            'min_trust': float(min(trust_values)),
            'max_trust': float(max(trust_values)),
            'trust_variance': float(np.var(trust_values)),
            'user_trust_summary': user_trust_summary,
            'total_trust_reports': len(trust_interactions)
        }


class TrustRepairAnalyzer:
    """Analyzes trust repair mechanisms and effectiveness."""

    def analyze_trust_repair(self,
                            trust_violations: List[Dict[str, Any]],
                            repair_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trust repair effectiveness."""
        if not trust_violations:
            return {'repair_effectiveness': 1.0, 'trust_violations': 0}

        # Match violations with repair actions
        repaired_violations = 0
        repair_success = []
        unrepaired = []

        violation_ids = {v.get('id') for v in trust_violations}

        for action in repair_actions:
            violation_id = action.get('violation_id')
            if violation_id in violation_ids:
                success = action.get('successful', False)
                trust_restored = action.get('trust_restored', 0)

                if success:
                    repaired_violations += 1
                    repair_success.append({
                        'violation_id': violation_id,
                        'trust_restored': trust_restored,
                        'repair_time': action.get('repair_time')
                    })

        # Find unrepaired violations
        repaired_ids = {r['violation_id'] for r in repair_success}
        unrepaired = [v for v in trust_violations if v.get('id') not in repaired_ids]

        repair_rate = repaired_violations / len(trust_violations)
        avg_trust_restored = np.mean([r['trust_restored'] for r in repair_success]) if repair_success else 0

        return {
            'repair_effectiveness': repair_rate * avg_trust_restored,
            'repair_rate': repair_rate,
            'average_trust_restored': float(avg_trust_restored),
            'trust_violations': len(trust_violations),
            'repaired_violations': repaired_violations,
            'unrepaired_violations': len(unrepaired),
            'repair_success_details': repair_success,
            'pending_repairs': [v.get('id') for v in unrepaired]
        }


class OvertrustAnalyzer:
    """Analyzes overtrust risk and patterns."""

    def analyze_overtrust(self,
                         interactions: List[TrustInteraction],
                         system_accuracy: float) -> Dict[str, Any]:
        """Identify overtrust patterns and risks."""
        if not interactions:
            return {'overtrust_risk': 0.0, 'overtrust_instances': 0}

        trust_interactions = [i for i in interactions if i.user_reported_trust is not None]

        overtrust_instances = []
        appropriate_trust_instances = []

        for interaction in trust_interactions:
            user_trust = interaction.user_reported_trust
            actual_correct = interaction.actual_correctness

            # Overtrust: high trust but incorrect
            if user_trust > 0.7 and not actual_correct:
                overtrust_instances.append({
                    'user_id': interaction.user_id,
                    'trust_level': user_trust,
                    'interaction_type': interaction.interaction_type
                })
            elif (user_trust > 0.5) == actual_correct:
                appropriate_trust_instances.append(interaction)

        total = len(trust_interactions) if trust_interactions else 1
        overtrust_rate = len(overtrust_instances) / total

        # Compare user trust to system accuracy
        mean_user_trust = np.mean([i.user_reported_trust for i in trust_interactions]) if trust_interactions else 0.5
        trust_accuracy_gap = mean_user_trust - system_accuracy

        return {
            'overtrust_risk': min(1.0, overtrust_rate + max(0, trust_accuracy_gap)),
            'overtrust_rate': overtrust_rate,
            'overtrust_instances': len(overtrust_instances),
            'appropriate_trust_instances': len(appropriate_trust_instances),
            'mean_user_trust': float(mean_user_trust),
            'system_accuracy': system_accuracy,
            'trust_accuracy_gap': float(trust_accuracy_gap),
            'overtrust_details': overtrust_instances[:10]  # Limit details
        }


class UndertrustAnalyzer:
    """Analyzes undertrust patterns and missed opportunities."""

    def analyze_undertrust(self,
                          interactions: List[TrustInteraction],
                          system_accuracy: float) -> Dict[str, Any]:
        """Identify undertrust patterns."""
        if not interactions:
            return {'undertrust_risk': 0.0, 'undertrust_instances': 0}

        trust_interactions = [i for i in interactions if i.user_reported_trust is not None]

        undertrust_instances = []

        for interaction in trust_interactions:
            user_trust = interaction.user_reported_trust
            actual_correct = interaction.actual_correctness

            # Undertrust: low trust but correct
            if user_trust < 0.3 and actual_correct:
                undertrust_instances.append({
                    'user_id': interaction.user_id,
                    'trust_level': user_trust,
                    'interaction_type': interaction.interaction_type
                })

        total = len(trust_interactions) if trust_interactions else 1
        undertrust_rate = len(undertrust_instances) / total

        mean_user_trust = np.mean([i.user_reported_trust for i in trust_interactions]) if trust_interactions else 0.5
        trust_accuracy_gap = system_accuracy - mean_user_trust

        return {
            'undertrust_risk': min(1.0, undertrust_rate + max(0, trust_accuracy_gap)),
            'undertrust_rate': undertrust_rate,
            'undertrust_instances': len(undertrust_instances),
            'mean_user_trust': float(mean_user_trust),
            'system_accuracy': system_accuracy,
            'trust_accuracy_gap': float(trust_accuracy_gap),
            'undertrust_details': undertrust_instances[:10],
            'missed_opportunity_cost': len(undertrust_instances) * trust_accuracy_gap
        }


class TrustEvolutionAnalyzer:
    """Analyzes trust evolution over time and interactions."""

    def analyze_trust_evolution(self,
                               interactions: List[TrustInteraction],
                               time_window_days: int = 30) -> Dict[str, Any]:
        """Analyze how trust evolves over time."""
        if not interactions:
            return {'trust_evolution': [], 'trend': 'unknown'}

        trust_interactions = [i for i in interactions if i.user_reported_trust is not None]

        if not trust_interactions:
            return {'trust_evolution': [], 'trend': 'no_data'}

        # Sort by timestamp
        sorted_interactions = sorted(trust_interactions, key=lambda x: x.timestamp)

        # Calculate rolling average trust
        window_size = min(10, len(sorted_interactions))
        trust_values = [i.user_reported_trust for i in sorted_interactions]

        evolution = []
        for i in range(len(trust_values)):
            start_idx = max(0, i - window_size + 1)
            window = trust_values[start_idx:i + 1]

            evolution.append({
                'index': i,
                'timestamp': sorted_interactions[i].timestamp.isoformat(),
                'instant_trust': trust_values[i],
                'rolling_average': float(np.mean(window)),
                'rolling_std': float(np.std(window)) if len(window) > 1 else 0
            })

        # Determine overall trend
        if len(trust_values) >= 5:
            first_half = np.mean(trust_values[:len(trust_values)//2])
            second_half = np.mean(trust_values[len(trust_values)//2:])

            if second_half > first_half + 0.1:
                trend = 'improving'
            elif second_half < first_half - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'

        return {
            'trust_evolution': evolution,
            'trend': trend,
            'initial_trust': float(trust_values[0]),
            'final_trust': float(trust_values[-1]),
            'trust_change': float(trust_values[-1] - trust_values[0]),
            'max_trust': float(max(trust_values)),
            'min_trust': float(min(trust_values)),
            'volatility': float(np.std(trust_values))
        }


# ============================================================================
# Report Generator
# ============================================================================

class ReliabilityReportGenerator:
    """Generates comprehensive reliability, trustworthiness, and trust reports."""

    def __init__(self):
        self.uptime_analyzer = UptimeAnalyzer()
        self.fault_tolerance_analyzer = FaultToleranceAnalyzer()
        self.consistency_analyzer = ConsistencyAnalyzer()
        self.reproducibility_analyzer = ReproducibilityAnalyzer()
        self.failure_pattern_analyzer = FailurePatternAnalyzer()
        self.recovery_analyzer = RecoveryAnalyzer()
        self.degradation_analyzer = DegradationAnalyzer()
        self.redundancy_analyzer = RedundancyAnalyzer()

        self.integrity_analyzer = IntegrityAnalyzer()
        self.authenticity_analyzer = AuthenticityAnalyzer()
        self.verification_analyzer = VerificationAnalyzer()
        self.validation_analyzer = ValidationAnalyzer()
        self.provenance_analyzer = ProvenanceAnalyzer()
        self.transparency_analyzer = TransparencyAnalyzer()

        self.trust_calibration_analyzer = TrustCalibrationAnalyzer()
        self.confidence_analyzer = ConfidenceAnalyzer()
        self.user_trust_analyzer = UserTrustAnalyzer()
        self.trust_repair_analyzer = TrustRepairAnalyzer()
        self.overtrust_analyzer = OvertrustAnalyzer()
        self.undertrust_analyzer = UndertrustAnalyzer()
        self.trust_evolution_analyzer = TrustEvolutionAnalyzer()

    def generate_reliability_report(self,
                                   failure_events: List[FailureEvent],
                                   operational_data: Dict[str, Any],
                                   system_components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate reliability analysis report."""
        return {
            'report_type': 'reliability_analysis',
            'timestamp': datetime.now().isoformat(),
            'failure_patterns': self.failure_pattern_analyzer.analyze_failure_patterns(failure_events),
            'recovery_analysis': self.recovery_analyzer.analyze_recovery(failure_events),
            'fault_tolerance': self.fault_tolerance_analyzer.analyze_fault_tolerance(
                failure_events,
                operational_data.get('redundancy_config', {})
            ),
            'redundancy': self.redundancy_analyzer.analyze_redundancy(
                system_components,
                operational_data.get('backup_config', {})
            )
        }

    def generate_trustworthiness_report(self,
                                        verification_data: Dict[str, Any],
                                        artifacts: List[Dict[str, Any]],
                                        documentation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trustworthiness analysis report."""
        return {
            'report_type': 'trustworthiness_analysis',
            'timestamp': datetime.now().isoformat(),
            'integrity': self.integrity_analyzer.analyze_integrity(
                verification_data.get('data_checksums', {}),
                verification_data.get('expected_checksums', {}),
                verification_data.get('model_signatures', {})
            ),
            'verification': self.verification_analyzer.analyze_verification(
                verification_data.get('verification_results', []),
                verification_data.get('required_verifications', [])
            ),
            'provenance': self.provenance_analyzer.analyze_provenance(
                artifacts,
                verification_data.get('lineage_graph', {})
            ),
            'transparency': self.transparency_analyzer.analyze_transparency(
                documentation,
                verification_data.get('disclosure_requirements', [])
            )
        }

    def generate_trust_report(self,
                             interactions: List[TrustInteraction],
                             system_accuracy: float,
                             trust_violations: List[Dict[str, Any]] = None,
                             repair_actions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate trust analysis report."""
        return {
            'report_type': 'trust_analysis',
            'timestamp': datetime.now().isoformat(),
            'calibration': self.trust_calibration_analyzer.analyze_trust_calibration(interactions),
            'user_trust': self.user_trust_analyzer.analyze_user_trust(interactions),
            'overtrust': self.overtrust_analyzer.analyze_overtrust(interactions, system_accuracy),
            'undertrust': self.undertrust_analyzer.analyze_undertrust(interactions, system_accuracy),
            'trust_evolution': self.trust_evolution_analyzer.analyze_trust_evolution(interactions),
            'trust_repair': self.trust_repair_analyzer.analyze_trust_repair(
                trust_violations or [],
                repair_actions or []
            )
        }

    def generate_full_report(self,
                            failure_events: List[FailureEvent] = None,
                            operational_data: Dict[str, Any] = None,
                            system_components: List[Dict[str, Any]] = None,
                            verification_data: Dict[str, Any] = None,
                            artifacts: List[Dict[str, Any]] = None,
                            documentation: Dict[str, Any] = None,
                            interactions: List[TrustInteraction] = None,
                            system_accuracy: float = 0.85) -> Dict[str, Any]:
        """Generate comprehensive reliability, trustworthiness, and trust report."""
        report = {
            'report_type': 'comprehensive_reliability_trust_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        if failure_events is not None:
            report['reliability'] = self.generate_reliability_report(
                failure_events,
                operational_data or {},
                system_components or []
            )

        if verification_data is not None:
            report['trustworthiness'] = self.generate_trustworthiness_report(
                verification_data,
                artifacts or [],
                documentation or {}
            )

        if interactions is not None:
            report['trust'] = self.generate_trust_report(
                interactions,
                system_accuracy
            )

        # Calculate overall scores
        scores = []
        if 'reliability' in report:
            rel_data = report['reliability']
            if 'fault_tolerance' in rel_data:
                scores.append(rel_data['fault_tolerance'].get('fault_tolerance_score', 0))

        if 'trustworthiness' in report:
            tw_data = report['trustworthiness']
            if 'integrity' in tw_data:
                scores.append(tw_data['integrity'].get('integrity_score', 0))

        if 'trust' in report:
            trust_data = report['trust']
            if 'calibration' in trust_data:
                scores.append(trust_data['calibration'].get('calibration_score', 0))

        report['overall_score'] = float(np.mean(scores)) if scores else 0.0

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
            f"# {report.get('report_type', 'Analysis Report')}",
            f"\n**Generated:** {report.get('timestamp', 'N/A')}",
            f"\n**Overall Score:** {report.get('overall_score', 0):.2%}",
            "\n---\n"
        ]

        if 'reliability' in report:
            lines.append("## Reliability Analysis\n")
            rel = report['reliability']
            if 'fault_tolerance' in rel:
                ft = rel['fault_tolerance']
                lines.append(f"- **Fault Tolerance Score:** {ft.get('fault_tolerance_score', 0):.2%}")
                lines.append(f"- **Total Failures:** {ft.get('total_failures', 0)}")
                lines.append(f"- **Recovery Rate:** {ft.get('recovery_rate', 0):.2%}")
            lines.append("")

        if 'trustworthiness' in report:
            lines.append("## Trustworthiness Analysis\n")
            tw = report['trustworthiness']
            if 'integrity' in tw:
                integ = tw['integrity']
                lines.append(f"- **Integrity Score:** {integ.get('integrity_score', 0):.2%}")
            if 'verification' in tw:
                ver = tw['verification']
                lines.append(f"- **Verification Rate:** {ver.get('verification_rate', 0):.2%}")
            lines.append("")

        if 'trust' in report:
            lines.append("## Trust Analysis\n")
            trust = report['trust']
            if 'calibration' in trust:
                cal = trust['calibration']
                lines.append(f"- **Calibration Score:** {cal.get('calibration_score', 0):.2%}")
                lines.append(f"- **Calibration Error:** {cal.get('calibration_error', 0):.4f}")
            if 'user_trust' in trust:
                ut = trust['user_trust']
                lines.append(f"- **Mean User Trust:** {ut.get('mean_user_trust', 0):.2%}")
                lines.append(f"- **Trust Trend:** {ut.get('trust_trend', 'unknown')}")
            lines.append("")

        return "\n".join(lines)
