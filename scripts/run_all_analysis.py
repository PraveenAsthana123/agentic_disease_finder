#!/usr/bin/env python3
"""
Run Complete Analysis for All Diseases
========================================

Trains and evaluates models for all 7 diseases,
generating comprehensive analysis reports.

Author: AgenticFinder Research Team
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'scripts'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
import joblib

DISEASES = ['parkinson', 'epilepsy', 'autism', 'schizophrenia', 'stress', 'alzheimer', 'depression']


def load_sample_data(disease: str, data_dir: Path):
    """Load sample data for a disease."""
    npz_path = data_dir / disease / 'sample' / f'{disease}_sample_100.npz'
    data = np.load(npz_path, allow_pickle=True)
    return data['X'], data['y'], data.get('subject_ids', None)


def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate comprehensive metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['roc_auc'] = 0.5

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0

    return metrics


def cross_validate(X, y, n_folds=5):
    """Perform cross-validation."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_prob)
        metrics['fold'] = fold + 1
        fold_results.append(metrics)

    return fold_results


def run_disease_analysis(disease: str, data_dir: Path, output_dir: Path):
    """Run complete analysis for a disease."""
    print(f"\n{'='*60}")
    print(f" Analyzing: {disease.upper()}")
    print(f"{'='*60}")

    # Load data
    X, y, subject_ids = load_sample_data(disease, data_dir)
    print(f"  Samples: {len(y)} (Control: {sum(y==0)}, Disease: {sum(y==1)})")

    # Cross-validation
    print("  Running 5-fold cross-validation...")
    cv_results = cross_validate(X, y)

    # Calculate summary statistics
    cv_df = pd.DataFrame(cv_results)
    summary = {
        'disease': disease,
        'n_samples': len(y),
        'n_control': int(sum(y == 0)),
        'n_disease': int(sum(y == 1)),
        'accuracy_mean': cv_df['accuracy'].mean(),
        'accuracy_std': cv_df['accuracy'].std(),
        'precision_mean': cv_df['precision'].mean(),
        'recall_mean': cv_df['recall'].mean(),
        'f1_mean': cv_df['f1_score'].mean(),
        'auc_mean': cv_df['roc_auc'].mean(),
        'sensitivity_mean': cv_df['sensitivity'].mean(),
        'specificity_mean': cv_df['specificity'].mean()
    }

    # Print results
    print(f"\n  Results:")
    print(f"    Accuracy:    {summary['accuracy_mean']:.4f} (+/- {summary['accuracy_std']:.4f})")
    print(f"    Precision:   {summary['precision_mean']:.4f}")
    print(f"    Recall:      {summary['recall_mean']:.4f}")
    print(f"    F1-Score:    {summary['f1_mean']:.4f}")
    print(f"    ROC-AUC:     {summary['auc_mean']:.4f}")
    print(f"    Sensitivity: {summary['sensitivity_mean']:.4f}")
    print(f"    Specificity: {summary['specificity_mean']:.4f}")

    # Train final model
    print("  Training final model...")
    final_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    final_clf.fit(X, y)

    # Save model
    model_dir = output_dir / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f'{disease}_model.joblib'
    joblib.dump({
        'model': final_clf,
        'disease': disease,
        'n_features': X.shape[1],
        'class_names': ['Control', disease.capitalize()],
        'training_date': datetime.now().isoformat(),
        'metrics': summary
    }, model_path)
    print(f"  Model saved: {model_path}")

    # Save results
    results_dir = output_dir / 'results' / disease
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save CV results
    cv_df.to_csv(results_dir / 'cv_results.csv', index=False)

    # Save summary
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Results saved: {results_dir}")

    return summary


def main():
    print("=" * 60)
    print(" AgenticFinder - Complete Analysis Pipeline")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    data_dir = Path('/media/praveen/Asthana3/rajveer/agenticfinder/data')
    output_dir = Path('/media/praveen/Asthana3/rajveer/agenticfinder')

    all_results = []

    for disease in DISEASES:
        summary = run_disease_analysis(disease, data_dir, output_dir)
        all_results.append(summary)

    # Create combined report
    print("\n" + "=" * 60)
    print(" COMBINED RESULTS")
    print("=" * 60)

    results_df = pd.DataFrame(all_results)

    print("\n Performance Summary:")
    print("-" * 80)
    print(f"{'Disease':<15} {'Accuracy':<12} {'F1-Score':<12} {'AUC':<12} {'Sens':<10} {'Spec':<10}")
    print("-" * 80)

    for _, row in results_df.iterrows():
        print(f"{row['disease']:<15} {row['accuracy_mean']:.4f}       {row['f1_mean']:.4f}       "
              f"{row['auc_mean']:.4f}       {row['sensitivity_mean']:.4f}     {row['specificity_mean']:.4f}")

    print("-" * 80)
    print(f"{'AVERAGE':<15} {results_df['accuracy_mean'].mean():.4f}       "
          f"{results_df['f1_mean'].mean():.4f}       {results_df['auc_mean'].mean():.4f}       "
          f"{results_df['sensitivity_mean'].mean():.4f}     {results_df['specificity_mean'].mean():.4f}")

    # Save combined results
    combined_dir = output_dir / 'results'
    combined_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(combined_dir / 'all_diseases_summary.csv', index=False)

    with open(combined_dir / 'all_diseases_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nCombined results saved: {combined_dir}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
