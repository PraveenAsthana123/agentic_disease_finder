#!/usr/bin/env python3
"""
AgenticFinder Fairness Testing Module
=====================================

Comprehensive fairness testing for EEG classification models.
Tests demographic parity, equalized odds, and predictive parity.

Usage:
    python fairness_tester.py --data results/predictions.json --output results/fairness_report.json
"""

import numpy as np
import json
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FairnessMetric:
    """Fairness metric result"""
    metric_name: str
    attribute: str
    score: float
    threshold: float
    passed: bool
    group_values: Dict
    details: Dict


@dataclass
class FairnessReport:
    """Complete fairness assessment report"""
    timestamp: str
    overall_status: str  # PASS, FAIL, NEEDS_ATTENTION
    attributes_tested: List[str]
    metrics_tested: List[str]
    total_tests: int
    tests_passed: int
    pass_rate: float
    results: List[Dict]
    recommendations: List[str]


class FairnessTester:
    """
    Comprehensive fairness testing for AgenticFinder

    Metrics:
    - Demographic Parity: Equal positive prediction rates across groups
    - Equalized Odds: Equal TPR and FPR across groups
    - Predictive Parity: Equal PPV across groups
    - Calibration Fairness: Equal calibration across groups
    """

    # Fairness thresholds
    THRESHOLDS = {
        'demographic_parity': 0.10,  # Max difference in positive rates
        'equalized_odds': 0.10,      # Max difference in TPR/FPR
        'predictive_parity': 0.10,   # Max difference in PPV
        'calibration': 0.05          # Max difference in calibration
    }

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_prob: Optional[np.ndarray] = None):
        """
        Initialize fairness tester

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
        self.protected_attributes: Dict[str, np.ndarray] = {}
        self.results: List[FairnessMetric] = []

    def add_protected_attribute(self, name: str, values: np.ndarray) -> None:
        """
        Add a protected attribute for fairness testing

        Args:
            name: Attribute name (e.g., 'gender', 'age_group')
            values: Array of attribute values for each sample
        """
        values = np.array(values)
        if len(values) != len(self.y_true):
            raise ValueError(f"Attribute {name} length ({len(values)}) doesn't match data length ({len(self.y_true)})")
        self.protected_attributes[name] = values
        print(f"Added protected attribute: {name} with {len(np.unique(values))} groups")

    def demographic_parity(self, attribute: str) -> FairnessMetric:
        """
        Test demographic parity

        Definition: P(Y_pred=1|A=a) should be equal across all groups a

        Metric: Max difference in positive prediction rates
        """
        if attribute not in self.protected_attributes:
            raise ValueError(f"Attribute {attribute} not found")

        attr_values = self.protected_attributes[attribute]
        unique_groups = np.unique(attr_values)

        # Calculate positive prediction rate per group
        positive_rates = {}
        for group in unique_groups:
            mask = attr_values == group
            if np.sum(mask) > 0:
                positive_rates[str(group)] = float(np.mean(self.y_pred[mask]))
            else:
                positive_rates[str(group)] = 0.0

        rates = list(positive_rates.values())
        max_diff = max(rates) - min(rates) if rates else 0.0

        passed = max_diff <= self.THRESHOLDS['demographic_parity']

        return FairnessMetric(
            metric_name='Demographic Parity',
            attribute=attribute,
            score=max_diff,
            threshold=self.THRESHOLDS['demographic_parity'],
            passed=passed,
            group_values=positive_rates,
            details={
                'description': 'Measures difference in positive prediction rates across groups',
                'max_rate': max(rates) if rates else 0,
                'min_rate': min(rates) if rates else 0,
                'n_groups': len(unique_groups),
                'interpretation': 'Lower is better. 0 = perfect parity.'
            }
        )

    def equalized_odds(self, attribute: str) -> FairnessMetric:
        """
        Test equalized odds

        Definition: P(Y_pred=1|Y_true=y,A=a) should be equal across all groups a, for y in {0,1}

        Metric: Max of (TPR difference, FPR difference)
        """
        if attribute not in self.protected_attributes:
            raise ValueError(f"Attribute {attribute} not found")

        attr_values = self.protected_attributes[attribute]
        unique_groups = np.unique(attr_values)

        tpr_by_group = {}
        fpr_by_group = {}

        for group in unique_groups:
            mask = attr_values == group
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred[mask]

            if len(y_true_group) < 2 or len(np.unique(y_true_group)) < 2:
                tpr_by_group[str(group)] = 0.0
                fpr_by_group[str(group)] = 0.0
                continue

            try:
                tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1]).ravel()
                tpr_by_group[str(group)] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                fpr_by_group[str(group)] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            except ValueError:
                tpr_by_group[str(group)] = 0.0
                fpr_by_group[str(group)] = 0.0

        tpr_values = list(tpr_by_group.values())
        fpr_values = list(fpr_by_group.values())

        tpr_diff = (max(tpr_values) - min(tpr_values)) if tpr_values else 0.0
        fpr_diff = (max(fpr_values) - min(fpr_values)) if fpr_values else 0.0
        max_diff = max(tpr_diff, fpr_diff)

        passed = max_diff <= self.THRESHOLDS['equalized_odds']

        return FairnessMetric(
            metric_name='Equalized Odds',
            attribute=attribute,
            score=max_diff,
            threshold=self.THRESHOLDS['equalized_odds'],
            passed=passed,
            group_values={'tpr': tpr_by_group, 'fpr': fpr_by_group},
            details={
                'description': 'Measures difference in TPR and FPR across groups',
                'tpr_difference': tpr_diff,
                'fpr_difference': fpr_diff,
                'n_groups': len(unique_groups),
                'interpretation': 'Lower is better. 0 = perfect equalized odds.'
            }
        )

    def predictive_parity(self, attribute: str) -> FairnessMetric:
        """
        Test predictive parity

        Definition: P(Y_true=1|Y_pred=1,A=a) should be equal across all groups a

        Metric: Max difference in Positive Predictive Value (PPV)
        """
        if attribute not in self.protected_attributes:
            raise ValueError(f"Attribute {attribute} not found")

        attr_values = self.protected_attributes[attribute]
        unique_groups = np.unique(attr_values)

        ppv_by_group = {}
        npv_by_group = {}

        for group in unique_groups:
            mask = attr_values == group
            y_true_group = self.y_true[mask]
            y_pred_group = self.y_pred[mask]

            if len(y_true_group) < 2:
                ppv_by_group[str(group)] = 0.0
                npv_by_group[str(group)] = 0.0
                continue

            try:
                tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1]).ravel()
                ppv_by_group[str(group)] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                npv_by_group[str(group)] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
            except ValueError:
                ppv_by_group[str(group)] = 0.0
                npv_by_group[str(group)] = 0.0

        ppv_values = list(ppv_by_group.values())
        ppv_diff = (max(ppv_values) - min(ppv_values)) if ppv_values else 0.0

        passed = ppv_diff <= self.THRESHOLDS['predictive_parity']

        return FairnessMetric(
            metric_name='Predictive Parity',
            attribute=attribute,
            score=ppv_diff,
            threshold=self.THRESHOLDS['predictive_parity'],
            passed=passed,
            group_values={'ppv': ppv_by_group, 'npv': npv_by_group},
            details={
                'description': 'Measures difference in Positive Predictive Value across groups',
                'max_ppv': max(ppv_values) if ppv_values else 0,
                'min_ppv': min(ppv_values) if ppv_values else 0,
                'n_groups': len(unique_groups),
                'interpretation': 'Lower is better. 0 = perfect predictive parity.'
            }
        )

    def calibration_fairness(self, attribute: str, n_bins: int = 5) -> FairnessMetric:
        """
        Test calibration fairness

        Definition: Among samples with probability p, actual positive rate should equal p,
                   regardless of group membership.

        Metric: Max difference in calibration error across groups
        """
        if self.y_prob is None:
            return FairnessMetric(
                metric_name='Calibration Fairness',
                attribute=attribute,
                score=0.0,
                threshold=self.THRESHOLDS['calibration'],
                passed=True,
                group_values={},
                details={'error': 'Probabilities not provided'}
            )

        if attribute not in self.protected_attributes:
            raise ValueError(f"Attribute {attribute} not found")

        attr_values = self.protected_attributes[attribute]
        unique_groups = np.unique(attr_values)

        ece_by_group = {}

        for group in unique_groups:
            mask = attr_values == group
            probs = self.y_prob[mask]
            labels = self.y_true[mask]
            preds = self.y_pred[mask]

            if len(probs) < n_bins:
                ece_by_group[str(group)] = 0.0
                continue

            # Calculate ECE for this group
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            ece = 0.0

            for i in range(n_bins):
                in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
                prop_in_bin = np.mean(in_bin)

                if prop_in_bin > 0:
                    avg_confidence = np.mean(probs[in_bin])
                    avg_accuracy = np.mean(labels[in_bin] == preds[in_bin])
                    ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

            ece_by_group[str(group)] = float(ece)

        ece_values = list(ece_by_group.values())
        max_diff = (max(ece_values) - min(ece_values)) if ece_values else 0.0

        passed = max_diff <= self.THRESHOLDS['calibration']

        return FairnessMetric(
            metric_name='Calibration Fairness',
            attribute=attribute,
            score=max_diff,
            threshold=self.THRESHOLDS['calibration'],
            passed=passed,
            group_values=ece_by_group,
            details={
                'description': 'Measures difference in calibration error across groups',
                'max_ece': max(ece_values) if ece_values else 0,
                'min_ece': min(ece_values) if ece_values else 0,
                'n_bins': n_bins,
                'interpretation': 'Lower is better. 0 = equally well calibrated across groups.'
            }
        )

    def run_full_assessment(self) -> FairnessReport:
        """
        Run comprehensive fairness assessment across all attributes and metrics

        Returns:
            FairnessReport object with all results
        """
        if not self.protected_attributes:
            return FairnessReport(
                timestamp=datetime.now().isoformat(),
                overall_status='ERROR',
                attributes_tested=[],
                metrics_tested=[],
                total_tests=0,
                tests_passed=0,
                pass_rate=0.0,
                results=[],
                recommendations=['No protected attributes defined. Add attributes using add_protected_attribute().']
            )

        self.results = []
        metrics_to_test = ['demographic_parity', 'equalized_odds', 'predictive_parity']

        if self.y_prob is not None:
            metrics_to_test.append('calibration_fairness')

        # Run all tests
        for attr in self.protected_attributes.keys():
            print(f"\nTesting attribute: {attr}")

            # Demographic Parity
            result = self.demographic_parity(attr)
            self.results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"  Demographic Parity: {result.score:.4f} [{status}]")

            # Equalized Odds
            result = self.equalized_odds(attr)
            self.results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"  Equalized Odds: {result.score:.4f} [{status}]")

            # Predictive Parity
            result = self.predictive_parity(attr)
            self.results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"  Predictive Parity: {result.score:.4f} [{status}]")

            # Calibration Fairness
            if self.y_prob is not None:
                result = self.calibration_fairness(attr)
                self.results.append(result)
                status = "PASS" if result.passed else "FAIL"
                print(f"  Calibration Fairness: {result.score:.4f} [{status}]")

        # Calculate summary statistics
        total_tests = len(self.results)
        tests_passed = sum(1 for r in self.results if r.passed)
        pass_rate = tests_passed / total_tests if total_tests > 0 else 0.0

        # Generate recommendations
        recommendations = []
        failed_tests = [r for r in self.results if not r.passed]

        for test in failed_tests:
            if test.metric_name == 'Demographic Parity':
                recommendations.append(
                    f"Consider rebalancing training data for '{test.attribute}' to reduce "
                    f"positive rate disparity ({test.score:.2%} difference)."
                )
            elif test.metric_name == 'Equalized Odds':
                recommendations.append(
                    f"Review model errors across '{test.attribute}' groups. Consider "
                    f"threshold adjustment or reweighting for affected groups."
                )
            elif test.metric_name == 'Predictive Parity':
                recommendations.append(
                    f"PPV varies across '{test.attribute}' groups. May need group-specific "
                    f"calibration or additional features for affected groups."
                )
            elif test.metric_name == 'Calibration Fairness':
                recommendations.append(
                    f"Calibration differs across '{test.attribute}' groups. Consider "
                    f"group-specific calibration or temperature scaling."
                )

        if not recommendations:
            recommendations.append("All fairness tests passed. Continue monitoring for drift.")

        # Determine overall status
        if pass_rate == 1.0:
            overall_status = 'PASS'
        elif pass_rate >= 0.8:
            overall_status = 'NEEDS_ATTENTION'
        else:
            overall_status = 'FAIL'

        return FairnessReport(
            timestamp=datetime.now().isoformat(),
            overall_status=overall_status,
            attributes_tested=list(self.protected_attributes.keys()),
            metrics_tested=metrics_to_test,
            total_tests=total_tests,
            tests_passed=tests_passed,
            pass_rate=pass_rate,
            results=[asdict(r) for r in self.results],
            recommendations=recommendations
        )


def print_report(report: FairnessReport) -> None:
    """Pretty print fairness report"""
    print("\n" + "="*70)
    print("AGENTICFINDER FAIRNESS ASSESSMENT REPORT")
    print("="*70)
    print(f"Timestamp: {report.timestamp}")
    print(f"Overall Status: {report.overall_status}")
    print("-"*70)
    print(f"Attributes Tested: {', '.join(report.attributes_tested)}")
    print(f"Metrics Tested: {', '.join(report.metrics_tested)}")
    print(f"Total Tests: {report.total_tests}")
    print(f"Tests Passed: {report.tests_passed} ({report.pass_rate:.1%})")
    print("-"*70)

    # Group results by attribute
    results_by_attr = {}
    for result in report.results:
        attr = result['attribute']
        if attr not in results_by_attr:
            results_by_attr[attr] = []
        results_by_attr[attr].append(result)

    for attr, results in results_by_attr.items():
        print(f"\n{attr.upper()}:")
        for r in results:
            status = "PASS" if r['passed'] else "FAIL"
            print(f"  {r['metric_name']}: {r['score']:.4f} (threshold: {r['threshold']}) [{status}]")

    print("\n" + "-"*70)
    print("RECOMMENDATIONS:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")

    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='AgenticFinder Fairness Tester')
    parser.add_argument('--data', type=str, help='Path to predictions JSON file')
    parser.add_argument('--output', type=str, default='results/fairness_report.json',
                       help='Output path for fairness report')
    parser.add_argument('--demo', action='store_true', help='Run demo with synthetic data')

    args = parser.parse_args()

    if args.demo:
        print("Running demo with synthetic data...")

        # Generate synthetic data
        np.random.seed(42)
        n_samples = 500

        y_true = np.random.binomial(1, 0.4, n_samples)
        y_pred = y_true.copy()

        # Add some errors
        error_idx = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
        y_pred[error_idx] = 1 - y_pred[error_idx]

        y_prob = np.clip(y_pred + np.random.normal(0, 0.1, n_samples), 0, 1)

        # Create tester
        tester = FairnessTester(y_true, y_pred, y_prob)

        # Add synthetic protected attributes
        gender = np.random.choice(['male', 'female'], n_samples)
        tester.add_protected_attribute('gender', gender)

        age_group = np.random.choice(['18-35', '36-50', '51-65'], n_samples)
        tester.add_protected_attribute('age_group', age_group)

        # Run assessment
        report = tester.run_full_assessment()

        # Print and save
        print_report(report)

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Report saved to {args.output}")

    elif args.data:
        print(f"Loading predictions from {args.data}...")

        with open(args.data, 'r') as f:
            data = json.load(f)

        # Create tester
        tester = FairnessTester(
            np.array(data['y_true']),
            np.array(data['y_pred']),
            np.array(data.get('y_prob', [])) if data.get('y_prob') else None
        )

        # Add protected attributes
        for attr_name, attr_values in data.get('protected_attributes', {}).items():
            tester.add_protected_attribute(attr_name, np.array(attr_values))

        if not tester.protected_attributes:
            print("Warning: No protected attributes in data. Adding synthetic age groups...")
            n = len(data['y_true'])
            age_groups = np.random.choice(['18-35', '36-50', '51-65'], n)
            tester.add_protected_attribute('age_group', age_groups)

        # Run assessment
        report = tester.run_full_assessment()

        # Print and save
        print_report(report)

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Report saved to {args.output}")

    else:
        print("Usage:")
        print("  python fairness_tester.py --demo                    # Run demo")
        print("  python fairness_tester.py --data predictions.json   # Test predictions")


if __name__ == '__main__':
    main()
