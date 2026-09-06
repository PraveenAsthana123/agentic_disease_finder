#!/usr/bin/env python3
"""
Evaluation Script for EEG-Based Neurological Disease Classification
====================================================================

This script provides comprehensive evaluation metrics for trained models
including confusion matrices, ROC curves, per-subject analysis, and
statistical significance testing.

Author: AgenticFinder Research Team
License: MIT
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import joblib
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics."""

    def __init__(self, model_path: str):
        """
        Initialize evaluator with trained model.

        Args:
            model_path: Path to saved model file (.joblib)
        """
        self.model_path = Path(model_path)
        self.model = None
        self.feature_extractor = None
        self.results = {}

        self._load_model()

    def _load_model(self):
        """Load trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info(f"Loading model from {self.model_path}")
        checkpoint = joblib.load(self.model_path)

        self.model = checkpoint.get('model')
        self.feature_extractor = checkpoint.get('feature_extractor')
        self.training_info = checkpoint.get('training_info', {})

        logger.info(f"Model loaded successfully")

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)

        Returns:
            Dictionary of metric names and values
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
            roc_auc_score, average_precision_score
        )

        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

        # Per-class metrics (weighted average for multi-class)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Additional metrics
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        # Probability-based metrics
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                    metrics['avg_precision'] = average_precision_score(y_true, y_prob[:, 1])
                else:
                    # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except Exception as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")

        return metrics

    def confusion_matrix_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate detailed confusion matrix analysis.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_names: Optional list of class names

        Returns:
            Dictionary with confusion matrix and per-class metrics
        """
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(y_true, y_pred)

        # Per-class metrics
        report = classification_report(y_true, y_pred, target_names=class_names,
                                       output_dict=True, zero_division=0)

        # Calculate per-class sensitivity and specificity
        n_classes = len(np.unique(y_true))
        per_class_metrics = []

        for i in range(n_classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - tp - fn - fp

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            class_name = class_names[i] if class_names else f"Class {i}"
            per_class_metrics.append({
                'class': class_name,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv,
                'support': int(np.sum(cm[i, :]))
            })

        return {
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'per_class_metrics': per_class_metrics
        }

    def cross_validation_analysis(self, X: np.ndarray, y: np.ndarray,
                                  cv_folds: int = 10) -> Dict[str, Any]:
        """
        Perform cross-validation analysis with confidence intervals.

        Args:
            X: Feature matrix
            y: Labels
            cv_folds: Number of CV folds

        Returns:
            Dictionary with CV results and statistics
        """
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # Multiple scoring metrics
        scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        cv_results = {}

        for metric in scoring_metrics:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=metric)

            # Calculate statistics
            mean = np.mean(scores)
            std = np.std(scores)
            ci_95 = stats.t.interval(0.95, len(scores)-1, loc=mean, scale=stats.sem(scores))

            cv_results[metric] = {
                'scores': scores.tolist(),
                'mean': mean,
                'std': std,
                'ci_95_lower': ci_95[0],
                'ci_95_upper': ci_95[1],
                'min': np.min(scores),
                'max': np.max(scores)
            }

        return cv_results

    def leave_one_subject_out(self, X: np.ndarray, y: np.ndarray,
                              subject_ids: np.ndarray) -> Dict[str, Any]:
        """
        Perform Leave-One-Subject-Out Cross-Validation.

        Args:
            X: Feature matrix
            y: Labels
            subject_ids: Subject identifiers for each sample

        Returns:
            Dictionary with LOSO-CV results
        """
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.base import clone

        logo = LeaveOneGroupOut()
        unique_subjects = np.unique(subject_ids)

        subject_results = []
        all_y_true = []
        all_y_pred = []

        for train_idx, test_idx in logo.split(X, y, subject_ids):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            subject_id = subject_ids[test_idx[0]]

            # Clone and train model
            model_clone = clone(self.model)
            model_clone.fit(X_train, y_train)

            # Predict
            y_pred = model_clone.predict(X_test)

            # Calculate metrics for this subject
            accuracy = np.mean(y_pred == y_test)

            subject_results.append({
                'subject_id': int(subject_id),
                'n_samples': len(y_test),
                'accuracy': accuracy,
                'n_correct': int(np.sum(y_pred == y_test))
            })

            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())

        # Overall metrics
        overall_accuracy = np.mean(np.array(all_y_true) == np.array(all_y_pred))
        subject_accuracies = [r['accuracy'] for r in subject_results]

        return {
            'n_subjects': len(unique_subjects),
            'subject_results': subject_results,
            'overall_accuracy': overall_accuracy,
            'mean_subject_accuracy': np.mean(subject_accuracies),
            'std_subject_accuracy': np.std(subject_accuracies),
            'min_subject_accuracy': np.min(subject_accuracies),
            'max_subject_accuracy': np.max(subject_accuracies)
        }

    def statistical_significance_test(self, scores1: np.ndarray, scores2: np.ndarray,
                                      test_name: str = 'wilcoxon') -> Dict[str, Any]:
        """
        Perform statistical significance testing between two models.

        Args:
            scores1: Scores from first model
            scores2: Scores from second model
            test_name: Statistical test to use ('wilcoxon', 'ttest')

        Returns:
            Dictionary with test results
        """
        if test_name == 'wilcoxon':
            statistic, p_value = wilcoxon(scores1, scores2)
        else:
            statistic, p_value = stats.ttest_rel(scores1, scores2)

        return {
            'test': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01,
            'mean_diff': np.mean(scores1) - np.mean(scores2)
        }

    def feature_importance_analysis(self) -> Dict[str, Any]:
        """
        Analyze feature importance if model supports it.

        Returns:
            Dictionary with feature importance rankings
        """
        importance_results = {}

        # Try to get feature importances from different model types
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            importance_results['method'] = 'built-in'
            importance_results['importances'] = importances.tolist()

        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_).mean(axis=0) if self.model.coef_.ndim > 1 else np.abs(self.model.coef_)
            importance_results['method'] = 'coefficients'
            importance_results['importances'] = importances.tolist()

        elif hasattr(self.model, 'base_estimators_'):
            # Ensemble model - aggregate from base estimators
            all_importances = []
            for estimator in self.model.base_estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    all_importances.append(estimator.feature_importances_)

            if all_importances:
                importances = np.mean(all_importances, axis=0)
                importance_results['method'] = 'ensemble_average'
                importance_results['importances'] = importances.tolist()

        # Rank features
        if 'importances' in importance_results:
            importances = np.array(importance_results['importances'])
            ranked_idx = np.argsort(importances)[::-1]
            importance_results['ranked_indices'] = ranked_idx.tolist()
            importance_results['top_10_indices'] = ranked_idx[:10].tolist()

        return importance_results

    def generate_report(self, X: np.ndarray, y: np.ndarray,
                       subject_ids: Optional[np.ndarray] = None,
                       class_names: Optional[List[str]] = None,
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.

        Args:
            X: Feature matrix
            y: Labels
            subject_ids: Optional subject identifiers
            class_names: Optional class names
            output_dir: Optional directory to save results

        Returns:
            Complete evaluation report
        """
        logger.info("Starting comprehensive evaluation...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'n_samples': len(y),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y))
        }

        # Make predictions
        logger.info("Making predictions...")
        y_pred = self.model.predict(X)
        y_prob = None
        if hasattr(self.model, 'predict_proba'):
            try:
                y_prob = self.model.predict_proba(X)
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")

        # Basic metrics
        logger.info("Calculating metrics...")
        report['metrics'] = self.calculate_metrics(y, y_pred, y_prob)

        # Confusion matrix analysis
        logger.info("Generating confusion matrix analysis...")
        report['confusion_matrix_analysis'] = self.confusion_matrix_analysis(y, y_pred, class_names)

        # Cross-validation
        logger.info("Performing cross-validation...")
        report['cross_validation'] = self.cross_validation_analysis(X, y)

        # LOSO-CV if subject IDs provided
        if subject_ids is not None:
            logger.info("Performing Leave-One-Subject-Out CV...")
            report['loso_cv'] = self.leave_one_subject_out(X, y, subject_ids)

        # Feature importance
        logger.info("Analyzing feature importance...")
        report['feature_importance'] = self.feature_importance_analysis()

        # Save report if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            report_file = output_path / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {report_file}")

        return report


def print_report(report: Dict[str, Any]):
    """Pretty print evaluation report."""
    print("\n" + "="*70)
    print("EVALUATION REPORT")
    print("="*70)

    print(f"\nTimestamp: {report['timestamp']}")
    print(f"Model: {report['model_path']}")
    print(f"Samples: {report['n_samples']}")
    print(f"Features: {report['n_features']}")
    print(f"Classes: {report['n_classes']}")

    print("\n" + "-"*70)
    print("PERFORMANCE METRICS")
    print("-"*70)

    metrics = report['metrics']
    print(f"\nAccuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1 Score:          {metrics['f1_score']:.4f}")
    print(f"MCC:               {metrics['mcc']:.4f}")
    print(f"Cohen's Kappa:     {metrics['cohen_kappa']:.4f}")

    if 'roc_auc' in metrics:
        print(f"ROC AUC:           {metrics['roc_auc']:.4f}")

    print("\n" + "-"*70)
    print("CROSS-VALIDATION RESULTS (10-fold)")
    print("-"*70)

    cv = report['cross_validation']
    for metric_name, results in cv.items():
        print(f"\n{metric_name}:")
        print(f"  Mean: {results['mean']:.4f} (+/- {results['std']:.4f})")
        print(f"  95% CI: [{results['ci_95_lower']:.4f}, {results['ci_95_upper']:.4f}]")

    if 'loso_cv' in report:
        print("\n" + "-"*70)
        print("LEAVE-ONE-SUBJECT-OUT RESULTS")
        print("-"*70)

        loso = report['loso_cv']
        print(f"\nNumber of subjects: {loso['n_subjects']}")
        print(f"Overall accuracy: {loso['overall_accuracy']:.4f}")
        print(f"Mean subject accuracy: {loso['mean_subject_accuracy']:.4f} (+/- {loso['std_subject_accuracy']:.4f})")
        print(f"Range: [{loso['min_subject_accuracy']:.4f}, {loso['max_subject_accuracy']:.4f}]")

    print("\n" + "-"*70)
    print("CONFUSION MATRIX")
    print("-"*70)

    cm = np.array(report['confusion_matrix_analysis']['confusion_matrix'])
    print("\n" + str(cm))

    print("\n" + "-"*70)
    print("PER-CLASS METRICS")
    print("-"*70)

    for class_metrics in report['confusion_matrix_analysis']['per_class_metrics']:
        print(f"\n{class_metrics['class']}:")
        print(f"  Sensitivity: {class_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {class_metrics['specificity']:.4f}")
        print(f"  PPV: {class_metrics['ppv']:.4f}")
        print(f"  NPV: {class_metrics['npv']:.4f}")

    print("\n" + "="*70)


def generate_synthetic_test_data(n_samples: int = 500, n_subjects: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data for testing evaluation."""
    np.random.seed(42)

    # Generate features (47 features as per our model)
    n_features = 47
    X = np.random.randn(n_samples, n_features)

    # Generate binary labels
    y = np.random.randint(0, 2, n_samples)

    # Generate subject IDs
    subject_ids = np.random.randint(0, n_subjects, n_samples)

    return X, y, subject_ids


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained EEG classification model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model file (.joblib)'
    )

    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Path to test data file (NPZ format)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results/evaluation',
        help='Output directory for results'
    )

    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data for testing'
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ModelEvaluator(args.model)

    # Load or generate data
    if args.synthetic:
        logger.info("Generating synthetic test data...")
        X, y, subject_ids = generate_synthetic_test_data()
        class_names = ['Control', 'Disease']
    elif args.data:
        logger.info(f"Loading test data from {args.data}")
        data = np.load(args.data)
        X = data['X']
        y = data['y']
        subject_ids = data.get('subject_ids', None)
        class_names = data.get('class_names', None)
    else:
        logger.error("Please provide either --data or --synthetic flag")
        sys.exit(1)

    # Generate report
    report = evaluator.generate_report(
        X, y,
        subject_ids=subject_ids,
        class_names=class_names,
        output_dir=args.output
    )

    # Print report
    print_report(report)

    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
