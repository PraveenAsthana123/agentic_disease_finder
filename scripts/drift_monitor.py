#!/usr/bin/env python3
"""
AgenticFinder Drift Monitoring System
=====================================

Automated drift detection for EEG classification models.
Monitors input features, predictions, and performance metrics.

Usage:
    python drift_monitor.py --baseline results/baseline_stats.json --current data/current_batch.npz
    python drift_monitor.py --init --data data/training_data.npz  # Initialize baseline
"""

import numpy as np
import json
import argparse
from scipy import stats
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DriftAlert:
    """Drift alert data structure"""
    drift_type: str
    feature_name: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    score: float
    threshold: float
    timestamp: str
    action_required: str
    details: Dict


class DriftMonitor:
    """
    Automated drift detection for AgenticFinder models

    Monitors:
    - Feature distribution drift (PSI, KS-test)
    - Prediction distribution drift (JS divergence)
    - Performance drift (accuracy, sensitivity, specificity)
    - Calibration drift (ECE)
    """

    # Severity thresholds
    THRESHOLDS = {
        'psi': {'low': 0.10, 'medium': 0.20, 'high': 0.25},
        'ks': {'low': 0.05, 'medium': 0.10, 'high': 0.15},
        'js': {'low': 0.05, 'medium': 0.10, 'high': 0.15},
        'accuracy_drop': {'low': 0.02, 'medium': 0.05, 'high': 0.10},
        'ece': {'low': 0.05, 'medium': 0.10, 'high': 0.15}
    }

    # Response times by severity
    RESPONSE_TIMES = {
        'LOW': '7 days',
        'MEDIUM': '3 days',
        'HIGH': '24 hours',
        'CRITICAL': '2 hours'
    }

    def __init__(self, baseline_path: Optional[str] = None):
        """
        Initialize drift monitor

        Args:
            baseline_path: Path to baseline statistics JSON file
        """
        self.baseline = None
        self.alerts: List[DriftAlert] = []

        if baseline_path and Path(baseline_path).exists():
            with open(baseline_path, 'r') as f:
                self.baseline = json.load(f)
            print(f"Loaded baseline from {baseline_path}")

    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray,
                      bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)

        PSI Interpretation:
        - < 0.10: No significant shift
        - 0.10-0.25: Moderate shift (investigate)
        - > 0.25: Significant shift (action required)

        Args:
            expected: Baseline distribution
            actual: Current distribution
            bins: Number of bins for discretization

        Returns:
            PSI score
        """
        # Handle edge cases
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Create bins based on baseline percentiles
        try:
            breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
            breakpoints[0] = -np.inf
            breakpoints[-1] = np.inf

            # Ensure unique breakpoints
            breakpoints = np.unique(breakpoints)
            if len(breakpoints) < 3:
                return 0.0

            expected_counts = np.histogram(expected, breakpoints)[0]
            actual_counts = np.histogram(actual, breakpoints)[0]

            # Add small value to avoid division by zero
            eps = 1e-10
            expected_percents = (expected_counts + eps) / (len(expected) + eps * len(expected_counts))
            actual_percents = (actual_counts + eps) / (len(actual) + eps * len(actual_counts))

            # Calculate PSI
            psi = np.sum((actual_percents - expected_percents) *
                        np.log(actual_percents / expected_percents))

            return float(max(0, psi))  # Ensure non-negative

        except Exception as e:
            print(f"PSI calculation error: {e}")
            return 0.0

    @staticmethod
    def ks_test(baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test for distribution shift

        Args:
            baseline: Baseline distribution
            current: Current distribution

        Returns:
            (KS statistic, p-value)
        """
        if len(baseline) < 2 or len(current) < 2:
            return 0.0, 1.0

        try:
            statistic, p_value = stats.ks_2samp(baseline, current)
            return float(statistic), float(p_value)
        except Exception:
            return 0.0, 1.0

    @staticmethod
    def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Jensen-Shannon divergence between two distributions

        Args:
            p: First distribution
            q: Second distribution

        Returns:
            JS divergence (0 to 1)
        """
        # Ensure valid probability distributions
        p = np.array(p, dtype=float)
        q = np.array(q, dtype=float)

        p = p / (p.sum() + 1e-10)
        q = q / (q.sum() + 1e-10)

        m = 0.5 * (p + q)

        # Avoid log(0)
        eps = 1e-10
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        m = np.clip(m, eps, 1)

        js = 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))

        return float(max(0, min(1, js)))  # Clamp to [0, 1]

    def _get_severity(self, score: float, metric_type: str) -> str:
        """Determine severity level based on score and metric type"""
        thresholds = self.THRESHOLDS.get(metric_type, self.THRESHOLDS['psi'])

        if score > thresholds['high']:
            return 'CRITICAL'
        elif score > thresholds['medium']:
            return 'HIGH'
        elif score > thresholds['low']:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _get_action(self, severity: str, drift_type: str) -> str:
        """Determine recommended action based on severity"""
        actions = {
            'LOW': 'Continue monitoring. No immediate action required.',
            'MEDIUM': f'Investigate {drift_type}. Review within {self.RESPONSE_TIMES["MEDIUM"]}.',
            'HIGH': f'Schedule model review. Address within {self.RESPONSE_TIMES["HIGH"]}.',
            'CRITICAL': f'IMMEDIATE ACTION REQUIRED. {drift_type} detected. Response within {self.RESPONSE_TIMES["CRITICAL"]}.'
        }
        return actions.get(severity, 'Unknown severity')

    def detect_feature_drift(self, feature_name: str,
                            baseline_values: np.ndarray,
                            current_values: np.ndarray) -> DriftAlert:
        """
        Detect drift for a single feature

        Args:
            feature_name: Name of the feature
            baseline_values: Baseline feature values
            current_values: Current feature values

        Returns:
            DriftAlert object
        """
        # Calculate PSI
        psi = self.calculate_psi(baseline_values, current_values)

        # Calculate KS test
        ks_stat, ks_pvalue = self.ks_test(baseline_values, current_values)

        # Determine severity (use PSI as primary metric)
        severity = self._get_severity(psi, 'psi')

        # Get recommended action
        action = self._get_action(severity, f'feature drift ({feature_name})')

        return DriftAlert(
            drift_type='FEATURE_DRIFT',
            feature_name=feature_name,
            severity=severity,
            score=psi,
            threshold=self.THRESHOLDS['psi']['medium'],
            timestamp=datetime.now().isoformat(),
            action_required=action,
            details={
                'psi': psi,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'baseline_mean': float(np.mean(baseline_values)),
                'baseline_std': float(np.std(baseline_values)),
                'current_mean': float(np.mean(current_values)),
                'current_std': float(np.std(current_values)),
                'sample_size_baseline': len(baseline_values),
                'sample_size_current': len(current_values)
            }
        )

    def detect_prediction_drift(self, baseline_predictions: np.ndarray,
                                current_predictions: np.ndarray) -> DriftAlert:
        """
        Detect drift in prediction distribution

        Args:
            baseline_predictions: Baseline class predictions
            current_predictions: Current class predictions

        Returns:
            DriftAlert object
        """
        # Calculate class distributions
        n_classes = max(np.max(baseline_predictions), np.max(current_predictions)) + 1

        baseline_dist = np.bincount(baseline_predictions.astype(int),
                                   minlength=n_classes) / len(baseline_predictions)
        current_dist = np.bincount(current_predictions.astype(int),
                                  minlength=n_classes) / len(current_predictions)

        # Calculate JS divergence
        js_div = self.js_divergence(baseline_dist, current_dist)

        # Determine severity
        severity = self._get_severity(js_div, 'js')

        # Get action
        action = self._get_action(severity, 'prediction distribution drift')

        return DriftAlert(
            drift_type='PREDICTION_DRIFT',
            feature_name='class_distribution',
            severity=severity,
            score=js_div,
            threshold=self.THRESHOLDS['js']['medium'],
            timestamp=datetime.now().isoformat(),
            action_required=action,
            details={
                'js_divergence': js_div,
                'baseline_distribution': baseline_dist.tolist(),
                'current_distribution': current_dist.tolist(),
                'baseline_positive_rate': float(baseline_dist[1]) if len(baseline_dist) > 1 else 0,
                'current_positive_rate': float(current_dist[1]) if len(current_dist) > 1 else 0
            }
        )

    def detect_performance_drift(self, baseline_accuracy: float,
                                 current_accuracy: float,
                                 current_sensitivity: Optional[float] = None,
                                 current_specificity: Optional[float] = None) -> DriftAlert:
        """
        Detect drop in model performance

        Args:
            baseline_accuracy: Baseline accuracy
            current_accuracy: Current accuracy
            current_sensitivity: Current sensitivity (optional)
            current_specificity: Current specificity (optional)

        Returns:
            DriftAlert object
        """
        accuracy_drop = baseline_accuracy - current_accuracy

        # Determine severity
        severity = self._get_severity(accuracy_drop, 'accuracy_drop')

        # Get action
        if severity == 'CRITICAL':
            action = 'IMMEDIATE RETRAINING REQUIRED. Accuracy dropped below acceptable threshold.'
        elif severity == 'HIGH':
            action = 'Schedule model retraining. Investigate data quality and distribution.'
        elif severity == 'MEDIUM':
            action = 'Monitor closely. Prepare retraining pipeline if trend continues.'
        else:
            action = 'Performance within acceptable range. Continue monitoring.'

        return DriftAlert(
            drift_type='PERFORMANCE_DRIFT',
            feature_name='accuracy',
            severity=severity,
            score=accuracy_drop,
            threshold=self.THRESHOLDS['accuracy_drop']['medium'],
            timestamp=datetime.now().isoformat(),
            action_required=action,
            details={
                'baseline_accuracy': baseline_accuracy,
                'current_accuracy': current_accuracy,
                'accuracy_drop': accuracy_drop,
                'current_sensitivity': current_sensitivity,
                'current_specificity': current_specificity,
                'accuracy_target': 0.90
            }
        )

    def detect_calibration_drift(self, predictions: np.ndarray,
                                 probabilities: np.ndarray,
                                 labels: np.ndarray,
                                 n_bins: int = 10) -> DriftAlert:
        """
        Detect calibration drift using Expected Calibration Error (ECE)

        Args:
            predictions: Predicted classes
            probabilities: Prediction probabilities
            labels: True labels
            n_bins: Number of calibration bins

        Returns:
            DriftAlert object
        """
        # Calculate ECE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (probabilities > bin_boundaries[i]) & (probabilities <= bin_boundaries[i+1])
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                avg_confidence = np.mean(probabilities[in_bin])
                avg_accuracy = np.mean(labels[in_bin] == predictions[in_bin])
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

        # Determine severity
        severity = self._get_severity(ece, 'ece')

        # Get action
        action = self._get_action(severity, 'calibration drift')

        return DriftAlert(
            drift_type='CALIBRATION_DRIFT',
            feature_name='expected_calibration_error',
            severity=severity,
            score=ece,
            threshold=self.THRESHOLDS['ece']['medium'],
            timestamp=datetime.now().isoformat(),
            action_required=action,
            details={
                'ece': ece,
                'n_bins': n_bins,
                'mean_confidence': float(np.mean(probabilities)),
                'mean_accuracy': float(np.mean(labels == predictions))
            }
        )

    def run_monitoring(self, current_data: Dict) -> Dict:
        """
        Run complete monitoring pipeline

        Args:
            current_data: Dictionary containing:
                - features: Dict of feature arrays
                - predictions: Array of predictions
                - probabilities: Array of prediction probabilities (optional)
                - labels: Array of true labels (optional)
                - accuracy: Current accuracy (optional)

        Returns:
            Monitoring report dictionary
        """
        if self.baseline is None:
            return {
                'status': 'ERROR',
                'message': 'No baseline loaded. Initialize baseline first.',
                'timestamp': datetime.now().isoformat()
            }

        self.alerts = []

        # 1. Feature drift detection
        if 'features' in current_data and 'features' in self.baseline:
            print("\nChecking feature drift...")
            for feature_name in current_data['features'].keys():
                if feature_name in self.baseline['features']:
                    baseline_values = np.array(self.baseline['features'][feature_name])
                    current_values = np.array(current_data['features'][feature_name])

                    alert = self.detect_feature_drift(
                        feature_name, baseline_values, current_values
                    )

                    if alert.severity != 'LOW':
                        self.alerts.append(alert)
                        print(f"  [{alert.severity}] {feature_name}: PSI={alert.score:.4f}")

        # 2. Prediction drift detection
        if 'predictions' in current_data and 'predictions' in self.baseline:
            print("\nChecking prediction drift...")
            baseline_preds = np.array(self.baseline['predictions'])
            current_preds = np.array(current_data['predictions'])

            alert = self.detect_prediction_drift(baseline_preds, current_preds)

            if alert.severity != 'LOW':
                self.alerts.append(alert)
                print(f"  [{alert.severity}] Prediction distribution: JS={alert.score:.4f}")

        # 3. Performance drift detection (if labels available)
        if 'accuracy' in current_data and 'accuracy' in self.baseline:
            print("\nChecking performance drift...")
            alert = self.detect_performance_drift(
                self.baseline['accuracy'],
                current_data['accuracy'],
                current_data.get('sensitivity'),
                current_data.get('specificity')
            )

            if alert.severity != 'LOW':
                self.alerts.append(alert)
                print(f"  [{alert.severity}] Accuracy drop: {alert.score:.4f}")

        # 4. Calibration drift detection (if probabilities and labels available)
        if all(k in current_data for k in ['predictions', 'probabilities', 'labels']):
            print("\nChecking calibration drift...")
            alert = self.detect_calibration_drift(
                np.array(current_data['predictions']),
                np.array(current_data['probabilities']),
                np.array(current_data['labels'])
            )

            if alert.severity != 'LOW':
                self.alerts.append(alert)
                print(f"  [{alert.severity}] ECE: {alert.score:.4f}")

        # Generate report
        critical_count = sum(1 for a in self.alerts if a.severity == 'CRITICAL')
        high_count = sum(1 for a in self.alerts if a.severity == 'HIGH')

        if critical_count > 0:
            status = 'CRITICAL'
        elif high_count > 0:
            status = 'WARNING'
        elif len(self.alerts) > 0:
            status = 'ATTENTION'
        else:
            status = 'OK'

        report = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_alerts': len(self.alerts),
                'critical': critical_count,
                'high': high_count,
                'medium': sum(1 for a in self.alerts if a.severity == 'MEDIUM')
            },
            'alerts': [asdict(a) for a in self.alerts],
            'response_required': status in ['CRITICAL', 'WARNING']
        }

        return report

    @staticmethod
    def create_baseline(features: Dict[str, np.ndarray],
                       predictions: np.ndarray,
                       accuracy: float,
                       output_path: str,
                       metadata: Optional[Dict] = None) -> None:
        """
        Create baseline statistics file

        Args:
            features: Dictionary of feature arrays
            predictions: Array of predictions
            accuracy: Baseline accuracy
            output_path: Path to save baseline JSON
            metadata: Optional metadata dictionary
        """
        baseline = {
            'created': datetime.now().isoformat(),
            'version': '1.0',
            'features': {k: v.tolist() for k, v in features.items()},
            'predictions': predictions.tolist(),
            'accuracy': accuracy,
            'metadata': metadata or {}
        }

        # Add feature statistics
        baseline['feature_stats'] = {}
        for name, values in features.items():
            baseline['feature_stats'][name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'n_samples': len(values)
            }

        with open(output_path, 'w') as f:
            json.dump(baseline, f, indent=2)

        print(f"Baseline saved to {output_path}")


def print_report(report: Dict) -> None:
    """Pretty print monitoring report"""
    print("\n" + "="*70)
    print("AGENTICFINDER DRIFT MONITORING REPORT")
    print("="*70)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Status: {report['status']}")
    print("-"*70)
    print(f"Total Alerts: {report['summary']['total_alerts']}")
    print(f"  CRITICAL: {report['summary']['critical']}")
    print(f"  HIGH:     {report['summary']['high']}")
    print(f"  MEDIUM:   {report['summary']['medium']}")
    print("-"*70)

    if report['alerts']:
        print("\nALERT DETAILS:")
        for i, alert in enumerate(report['alerts'], 1):
            print(f"\n[{i}] {alert['drift_type']} - {alert['feature_name']}")
            print(f"    Severity:  {alert['severity']}")
            print(f"    Score:     {alert['score']:.4f} (threshold: {alert['threshold']:.4f})")
            print(f"    Action:    {alert['action_required']}")
    else:
        print("\nNo alerts. All metrics within acceptable range.")

    print("\n" + "="*70)
    if report['response_required']:
        print("*** RESPONSE REQUIRED - See alerts above ***")
    else:
        print("System healthy. Continue normal monitoring.")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='AgenticFinder Drift Monitor')
    parser.add_argument('--baseline', type=str, help='Path to baseline JSON file')
    parser.add_argument('--current', type=str, help='Path to current data (NPZ or JSON)')
    parser.add_argument('--init', action='store_true', help='Initialize baseline from data')
    parser.add_argument('--data', type=str, help='Path to training data for baseline init')
    parser.add_argument('--output', type=str, default='results/drift_report.json',
                       help='Output path for drift report')

    args = parser.parse_args()

    if args.init:
        if not args.data:
            print("Error: --data required for baseline initialization")
            return

        print(f"Initializing baseline from {args.data}...")

        # Load training data
        data = np.load(args.data, allow_pickle=True)

        # Create baseline
        features = {}
        if 'X' in data:
            X = data['X']
            for i in range(X.shape[1]):
                features[f'feature_{i}'] = X[:, i]

        predictions = data.get('y_pred', data.get('y', np.zeros(len(X))))
        accuracy = data.get('accuracy', 0.95)

        baseline_path = args.baseline or 'results/baseline_stats.json'
        DriftMonitor.create_baseline(
            features=features,
            predictions=np.array(predictions),
            accuracy=float(accuracy),
            output_path=baseline_path,
            metadata={'source': args.data}
        )

    else:
        if not args.baseline or not args.current:
            print("Error: --baseline and --current required for monitoring")
            print("Usage: python drift_monitor.py --baseline baseline.json --current current.npz")
            return

        print(f"Running drift monitoring...")
        print(f"Baseline: {args.baseline}")
        print(f"Current:  {args.current}")

        # Initialize monitor
        monitor = DriftMonitor(args.baseline)

        # Load current data
        if args.current.endswith('.json'):
            with open(args.current, 'r') as f:
                current_data = json.load(f)
        else:
            data = np.load(args.current, allow_pickle=True)
            current_data = {
                'features': {f'feature_{i}': data['X'][:, i]
                            for i in range(data['X'].shape[1])},
                'predictions': data.get('y_pred', data.get('y', [])),
            }
            if 'accuracy' in data:
                current_data['accuracy'] = float(data['accuracy'])

        # Run monitoring
        report = monitor.run_monitoring(current_data)

        # Print report
        print_report(report)

        # Save report
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")


if __name__ == '__main__':
    main()
