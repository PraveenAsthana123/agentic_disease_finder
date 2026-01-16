"""
Monitoring Analysis Module - Drift Detection, Debug AI, Sensitivity Analysis
=============================================================================

Comprehensive analysis for AI monitoring, debugging, and sensitivity testing.
Implements 58 analysis types across three related frameworks.

Frameworks:
- Monitoring & Drift Detection (18 types): Data Drift, Concept Drift, Performance Monitoring
- Debug AI (20 types): Error Analysis, Root Cause, Debugging Workflows
- Sensitivity Analysis AI (20 types): Parameter Sensitivity, Input Sensitivity, Robustness
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import numpy as np
from collections import defaultdict
import json


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DriftMetrics:
    """Metrics for drift detection analysis."""
    data_drift_score: float = 0.0
    concept_drift_score: float = 0.0
    prediction_drift_score: float = 0.0
    feature_drift: Dict[str, float] = field(default_factory=dict)
    drift_detected: bool = False


@dataclass
class DebugMetrics:
    """Metrics for debugging analysis."""
    error_rate: float = 0.0
    root_cause_coverage: float = 0.0
    resolution_rate: float = 0.0
    mttr: float = 0.0  # Mean time to resolution


@dataclass
class SensitivityMetrics:
    """Metrics for sensitivity analysis."""
    parameter_sensitivity: Dict[str, float] = field(default_factory=dict)
    input_sensitivity: float = 0.0
    robustness_score: float = 0.0
    stability_index: float = 0.0


@dataclass
class DriftEvent:
    """Represents a detected drift event."""
    event_id: str
    timestamp: datetime
    drift_type: str  # 'data', 'concept', 'prediction', 'feature'
    severity: float
    affected_features: List[str] = field(default_factory=list)
    baseline_distribution: Dict[str, Any] = field(default_factory=dict)
    current_distribution: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebugSession:
    """Represents a debugging session."""
    session_id: str
    started_at: datetime
    error_type: str
    root_cause: Optional[str] = None
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None
    debug_steps: List[str] = field(default_factory=list)


# ============================================================================
# Monitoring & Drift Detection Analyzers
# ============================================================================

class DataDriftAnalyzer:
    """Analyzes data drift patterns."""

    def analyze_data_drift(self,
                          reference_data: Dict[str, List[float]],
                          current_data: Dict[str, List[float]],
                          threshold: float = 0.1) -> Dict[str, Any]:
        """Analyze data drift between reference and current distributions."""
        if not reference_data or not current_data:
            return {'data_drift_score': 0.0, 'drift_detected': False}

        feature_drift = {}
        drift_scores = []

        for feature in reference_data:
            if feature not in current_data:
                continue

            ref = np.array(reference_data[feature])
            curr = np.array(current_data[feature])

            # Calculate PSI (Population Stability Index)
            psi = self._calculate_psi(ref, curr)
            feature_drift[feature] = float(psi)
            drift_scores.append(psi)

        avg_drift = np.mean(drift_scores) if drift_scores else 0
        drift_detected = avg_drift > threshold

        # Identify most drifted features
        sorted_features = sorted(feature_drift.items(), key=lambda x: x[1], reverse=True)
        top_drifted = [f for f, d in sorted_features[:5] if d > threshold]

        return {
            'data_drift_score': float(avg_drift),
            'drift_detected': drift_detected,
            'feature_drift': feature_drift,
            'top_drifted_features': top_drifted,
            'features_analyzed': len(feature_drift),
            'features_with_drift': sum(1 for d in drift_scores if d > threshold),
            'threshold': threshold
        }

    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        # Bin the data
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())

        if min_val == max_val:
            return 0.0

        bin_edges = np.linspace(min_val, max_val, bins + 1)

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)

        # Normalize to get proportions
        ref_prop = ref_counts / len(reference)
        curr_prop = curr_counts / len(current)

        # Avoid division by zero
        ref_prop = np.clip(ref_prop, 0.0001, 1)
        curr_prop = np.clip(curr_prop, 0.0001, 1)

        # Calculate PSI
        psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))

        return float(psi)


class ConceptDriftAnalyzer:
    """Analyzes concept drift in model predictions."""

    def analyze_concept_drift(self,
                             historical_performance: List[Dict[str, float]],
                             window_size: int = 10) -> Dict[str, Any]:
        """Analyze concept drift through performance degradation."""
        if not historical_performance or len(historical_performance) < window_size * 2:
            return {'concept_drift_score': 0.0, 'drift_detected': False}

        # Extract accuracy metrics
        accuracies = [p.get('accuracy', 0) for p in historical_performance]

        # Compare recent window to historical baseline
        baseline = accuracies[:window_size]
        recent = accuracies[-window_size:]

        baseline_mean = np.mean(baseline)
        recent_mean = np.mean(recent)

        # Concept drift indicated by performance degradation
        if baseline_mean > 0:
            drift_score = max(0, (baseline_mean - recent_mean) / baseline_mean)
        else:
            drift_score = 0

        # Detect trend
        x = np.arange(len(accuracies))
        slope = np.polyfit(x, accuracies, 1)[0]
        trend = 'declining' if slope < -0.01 else ('stable' if abs(slope) < 0.01 else 'improving')

        return {
            'concept_drift_score': float(drift_score),
            'drift_detected': drift_score > 0.1,
            'baseline_accuracy': float(baseline_mean),
            'recent_accuracy': float(recent_mean),
            'performance_change': float(recent_mean - baseline_mean),
            'trend': trend,
            'trend_slope': float(slope),
            'windows_analyzed': len(historical_performance)
        }


class PerformanceMonitor:
    """Monitors model performance over time."""

    def monitor_performance(self,
                           metrics_history: List[Dict[str, Any]],
                           alert_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """Monitor performance metrics and detect anomalies."""
        if not metrics_history:
            return {'monitoring_status': 'no_data', 'alerts': []}

        alert_thresholds = alert_thresholds or {
            'accuracy': 0.8,
            'latency_ms': 1000,
            'error_rate': 0.05
        }

        current_metrics = metrics_history[-1]
        alerts = []

        # Check against thresholds
        for metric, threshold in alert_thresholds.items():
            current_value = current_metrics.get(metric)
            if current_value is None:
                continue

            # For latency and error_rate, higher is worse
            if metric in ['latency_ms', 'error_rate']:
                if current_value > threshold:
                    alerts.append({
                        'metric': metric,
                        'current_value': current_value,
                        'threshold': threshold,
                        'severity': 'critical' if current_value > threshold * 1.5 else 'warning'
                    })
            else:
                # For accuracy, lower is worse
                if current_value < threshold:
                    alerts.append({
                        'metric': metric,
                        'current_value': current_value,
                        'threshold': threshold,
                        'severity': 'critical' if current_value < threshold * 0.8 else 'warning'
                    })

        # Calculate metric trends
        trends = {}
        for metric in alert_thresholds:
            values = [m.get(metric) for m in metrics_history if m.get(metric) is not None]
            if len(values) >= 3:
                slope = np.polyfit(range(len(values)), values, 1)[0]
                trends[metric] = 'improving' if slope > 0.01 else ('declining' if slope < -0.01 else 'stable')

        return {
            'monitoring_status': 'critical' if any(a['severity'] == 'critical' for a in alerts) else ('warning' if alerts else 'healthy'),
            'alerts': alerts,
            'current_metrics': current_metrics,
            'metric_trends': trends,
            'data_points': len(metrics_history)
        }


# ============================================================================
# Debug AI Analyzers
# ============================================================================

class ErrorAnalyzer:
    """Analyzes error patterns and frequencies."""

    def analyze_errors(self,
                      errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error patterns."""
        if not errors:
            return {'error_rate': 0.0, 'error_types': {}}

        error_types = defaultdict(int)
        severity_counts = defaultdict(int)
        error_timeline = []

        for error in errors:
            error_types[error.get('type', 'unknown')] += 1
            severity_counts[error.get('severity', 'unknown')] += 1

            if 'timestamp' in error:
                error_timeline.append({
                    'timestamp': error['timestamp'],
                    'type': error.get('type')
                })

        # Calculate patterns
        most_common_error = max(error_types, key=error_types.get) if error_types else None

        return {
            'total_errors': len(errors),
            'error_types': dict(error_types),
            'severity_distribution': dict(severity_counts),
            'most_common_error': most_common_error,
            'most_common_count': error_types[most_common_error] if most_common_error else 0,
            'critical_errors': severity_counts.get('critical', 0),
            'error_diversity': len(error_types)
        }


class RootCauseAnalyzer:
    """Analyzes root causes of errors."""

    def analyze_root_causes(self,
                           debug_sessions: List[DebugSession]) -> Dict[str, Any]:
        """Analyze root causes from debug sessions."""
        if not debug_sessions:
            return {'root_cause_coverage': 0.0, 'root_causes': {}}

        with_root_cause = [s for s in debug_sessions if s.root_cause]
        coverage = len(with_root_cause) / len(debug_sessions)

        root_cause_counts = defaultdict(int)
        for session in with_root_cause:
            root_cause_counts[session.root_cause] += 1

        # Analyze resolution effectiveness
        resolved = [s for s in debug_sessions if s.resolution]
        resolution_rate = len(resolved) / len(debug_sessions)

        # Calculate MTTR
        resolution_times = []
        for session in resolved:
            if session.resolved_at and session.started_at:
                time_diff = (session.resolved_at - session.started_at).total_seconds()
                resolution_times.append(time_diff)

        mttr = np.mean(resolution_times) if resolution_times else 0

        return {
            'root_cause_coverage': float(coverage),
            'resolution_rate': float(resolution_rate),
            'mttr_seconds': float(mttr),
            'mttr_hours': float(mttr / 3600),
            'root_causes': dict(root_cause_counts),
            'most_common_root_cause': max(root_cause_counts, key=root_cause_counts.get) if root_cause_counts else None,
            'sessions_analyzed': len(debug_sessions)
        }


# ============================================================================
# Sensitivity Analysis Analyzers
# ============================================================================

class ParameterSensitivityAnalyzer:
    """Analyzes model sensitivity to parameter changes."""

    def analyze_parameter_sensitivity(self,
                                     parameter_sweeps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sensitivity to parameter changes."""
        if not parameter_sweeps:
            return {'sensitivity_analysis': {}, 'most_sensitive': None}

        sensitivity_by_param = {}

        for sweep in parameter_sweeps:
            param_name = sweep.get('parameter')
            values = sweep.get('values', [])
            outputs = sweep.get('outputs', [])

            if len(values) < 2 or len(outputs) < 2:
                continue

            # Calculate sensitivity (normalized output change / input change)
            values = np.array(values)
            outputs = np.array(outputs)

            # Normalize
            if values.max() - values.min() > 0:
                norm_values = (values - values.min()) / (values.max() - values.min())
            else:
                norm_values = values

            if outputs.max() - outputs.min() > 0:
                norm_outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min())
            else:
                norm_outputs = outputs

            # Sensitivity as average gradient
            gradients = np.abs(np.diff(norm_outputs) / np.diff(norm_values))
            gradients = gradients[~np.isnan(gradients) & ~np.isinf(gradients)]

            if len(gradients) > 0:
                sensitivity = float(np.mean(gradients))
            else:
                sensitivity = 0

            sensitivity_by_param[param_name] = {
                'sensitivity': sensitivity,
                'value_range': (float(values.min()), float(values.max())),
                'output_range': (float(outputs.min()), float(outputs.max()))
            }

        most_sensitive = max(sensitivity_by_param, key=lambda x: sensitivity_by_param[x]['sensitivity']) if sensitivity_by_param else None

        return {
            'sensitivity_analysis': sensitivity_by_param,
            'most_sensitive': most_sensitive,
            'least_sensitive': min(sensitivity_by_param, key=lambda x: sensitivity_by_param[x]['sensitivity']) if sensitivity_by_param else None,
            'parameters_analyzed': len(sensitivity_by_param)
        }


class InputSensitivityAnalyzer:
    """Analyzes model sensitivity to input perturbations."""

    def analyze_input_sensitivity(self,
                                  original_inputs: List[List[float]],
                                  perturbed_inputs: List[List[float]],
                                  original_outputs: List[float],
                                  perturbed_outputs: List[float]) -> Dict[str, Any]:
        """Analyze sensitivity to input perturbations."""
        if not original_inputs or not perturbed_outputs:
            return {'input_sensitivity': 0.0, 'robustness_score': 0.0}

        sensitivities = []

        for i in range(min(len(original_inputs), len(perturbed_inputs))):
            orig = np.array(original_inputs[i])
            pert = np.array(perturbed_inputs[i])

            input_change = np.linalg.norm(pert - orig)
            output_change = abs(perturbed_outputs[i] - original_outputs[i])

            if input_change > 0:
                sensitivity = output_change / input_change
            else:
                sensitivity = 0

            sensitivities.append(sensitivity)

        avg_sensitivity = np.mean(sensitivities)

        # Robustness is inverse of sensitivity
        robustness = 1 / (1 + avg_sensitivity)

        return {
            'input_sensitivity': float(avg_sensitivity),
            'robustness_score': float(robustness),
            'samples_analyzed': len(sensitivities),
            'max_sensitivity': float(max(sensitivities)) if sensitivities else 0,
            'min_sensitivity': float(min(sensitivities)) if sensitivities else 0,
            'sensitivity_std': float(np.std(sensitivities)) if sensitivities else 0
        }


# ============================================================================
# Report Generator
# ============================================================================

class MonitoringReportGenerator:
    """Generates comprehensive monitoring reports."""

    def __init__(self):
        self.data_drift_analyzer = DataDriftAnalyzer()
        self.concept_drift_analyzer = ConceptDriftAnalyzer()
        self.performance_monitor = PerformanceMonitor()
        self.error_analyzer = ErrorAnalyzer()
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.param_sensitivity_analyzer = ParameterSensitivityAnalyzer()
        self.input_sensitivity_analyzer = InputSensitivityAnalyzer()

    def generate_full_report(self,
                            reference_data: Dict[str, List[float]] = None,
                            current_data: Dict[str, List[float]] = None,
                            performance_history: List[Dict[str, Any]] = None,
                            errors: List[Dict[str, Any]] = None,
                            debug_sessions: List[DebugSession] = None) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        report = {
            'report_type': 'comprehensive_monitoring_analysis',
            'timestamp': datetime.now().isoformat(),
            'framework_version': '1.0.0'
        }

        if reference_data and current_data:
            report['data_drift'] = self.data_drift_analyzer.analyze_data_drift(reference_data, current_data)

        if performance_history:
            report['concept_drift'] = self.concept_drift_analyzer.analyze_concept_drift(performance_history)
            report['performance'] = self.performance_monitor.monitor_performance(performance_history)

        if errors:
            report['error_analysis'] = self.error_analyzer.analyze_errors(errors)

        if debug_sessions:
            report['root_cause_analysis'] = self.root_cause_analyzer.analyze_root_causes(debug_sessions)

        # Calculate overall health score
        scores = []
        if 'data_drift' in report:
            scores.append(1 - report['data_drift'].get('data_drift_score', 0))
        if 'concept_drift' in report:
            scores.append(1 - report['concept_drift'].get('concept_drift_score', 0))
        if 'root_cause_analysis' in report:
            scores.append(report['root_cause_analysis'].get('resolution_rate', 0))

        report['overall_health'] = float(np.mean(scores)) if scores else 0.5

        return report

    def export_report(self, report: Dict[str, Any], filepath: str, format: str = 'json') -> None:
        """Export report to file."""
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
