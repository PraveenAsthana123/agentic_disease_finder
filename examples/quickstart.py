#!/usr/bin/env python3
"""
AgenticFinder Quick Start Example
===================================

This script demonstrates the complete workflow for EEG-based
neurological disease classification.

Usage:
    python examples/quickstart.py
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

import numpy as np
from datetime import datetime


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def main():
    print_header("AgenticFinder Quick Start Demo")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Generate synthetic EEG data
    print_header("Step 1: Generate Synthetic EEG Data")

    from generate_sample_data import EEGDataGenerator, extract_simple_features

    generator = EEGDataGenerator(sampling_rate=256, seed=42)

    # Generate Parkinson's dataset
    print("Generating Parkinson's disease dataset...")
    data = generator.generate_dataset(
        disease='parkinson',
        n_subjects=15,
        samples_per_subject=8,
        duration=2.0,
        include_controls=True
    )

    print(f"  - Generated {len(data['signals'])} samples")
    print(f"  - Disease samples: {np.sum(data['labels'] == 1)}")
    print(f"  - Control samples: {np.sum(data['labels'] == 0)}")
    print(f"  - Unique subjects: {len(np.unique(data['subject_ids']))}")

    # Step 2: Extract features
    print_header("Step 2: Extract Features")

    print("Extracting 47 EEG features...")
    X = extract_simple_features(data['signals'], data['sampling_rate'])
    y = data['labels']

    print(f"  - Feature matrix shape: {X.shape}")
    print(f"  - Features per sample: {X.shape[1]}")

    # Step 3: Train model
    print_header("Step 3: Train Ultra Stacking Ensemble")

    from train import UltraStackingEnsemble
    from sklearn.model_selection import train_test_split

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")

    # Train ensemble
    print("\nTraining ensemble classifier...")
    ensemble = UltraStackingEnsemble(random_state=42)
    ensemble.fit(X_train, y_train)

    print("  - Training complete!")

    # Step 4: Evaluate
    print_header("Step 4: Evaluate Model")

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )

    # Predictions
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\nTest Set Performance:")
    print(f"  - Accuracy:  {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1-Score:  {f1:.4f}")
    print(f"  - ROC-AUC:   {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"  [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
    print(f"   [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")

    # Step 5: Make predictions
    print_header("Step 5: Make Predictions on New Data")

    # Generate new samples
    new_signal = generator.generate_eeg_signal(duration=2.0, disease='parkinson')
    new_features = extract_simple_features(new_signal.reshape(1, -1), 256)

    prediction = ensemble.predict(new_features)[0]
    probability = ensemble.predict_proba(new_features)[0]

    print(f"  - Prediction: {'Disease' if prediction == 1 else 'Control'}")
    print(f"  - Confidence: {max(probability):.2%}")
    print(f"  - P(Control): {probability[0]:.4f}")
    print(f"  - P(Disease): {probability[1]:.4f}")

    # Step 6: Manual Cross-validation
    print_header("Step 6: Cross-Validation Results")

    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    print("Running 5-fold cross-validation...")
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_cv_train, X_cv_test = X[train_idx], X[test_idx]
        y_cv_train, y_cv_test = y[train_idx], y[test_idx]

        cv_ensemble = UltraStackingEnsemble(random_state=42)
        cv_ensemble.fit(X_cv_train, y_cv_train)
        y_cv_pred = cv_ensemble.predict(X_cv_test)
        fold_acc = accuracy_score(y_cv_test, y_cv_pred)
        cv_scores.append(fold_acc)
        print(f"    Fold {fold+1}: {fold_acc:.4f}")

    cv_scores = np.array(cv_scores)
    print(f"\n  - 5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Summary
    print_header("Summary")
    print("""
    This demo showed the complete workflow:

    1. Generated synthetic EEG data with disease characteristics
    2. Extracted 47 features (statistical, spectral, temporal, nonlinear)
    3. Trained Ultra Stacking Ensemble (15 classifiers + meta-learner)
    4. Evaluated with comprehensive metrics
    5. Made predictions on new samples
    6. Performed cross-validation

    For real applications:
    - Use actual EEG datasets (PPMI, CHB-MIT, ABIDE, etc.)
    - Apply proper preprocessing (filtering, artifact removal)
    - Use Leave-One-Subject-Out CV for unbiased evaluation
    - Validate with independent test sets
    """)

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
