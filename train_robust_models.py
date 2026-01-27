#!/usr/bin/env python
"""
ROBUST MODEL TRAINING WITH ANTI-OVERFITTING MEASURES
Implements all overfitting recommendations:
1. Uses augmented data (200 samples per disease)
2. External validation on held-out data
3. Regularization (controlled tree depth, min samples)
4. Feature selection (mutual information + RFE)
5. Cross-validation with confidence intervals
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split, learning_curve)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                             ExtraTreesClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (SelectKBest, mutual_info_classif,
                                       RFE, SelectFromModel)
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                            recall_score, confusion_matrix, classification_report)
import joblib
import json
import datetime
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "training_results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DISEASES = {
    'epilepsy': {'name': 'Epilepsy', 'path': DATA_DIR / 'epilepsy' / 'sample'},
    'parkinson': {'name': "Parkinson's Disease", 'path': DATA_DIR / 'parkinson' / 'sample'},
    'alzheimer': {'name': "Alzheimer's Disease", 'path': DATA_DIR / 'alzheimer' / 'sample'},
    'schizophrenia': {'name': 'Schizophrenia', 'path': DATA_DIR / 'schizophrenia' / 'sample'},
    'depression': {'name': 'Major Depression', 'path': DATA_DIR / 'depression' / 'sample'},
    'autism': {'name': 'Autism Spectrum', 'path': DATA_DIR / 'autism' / 'sample'},
    'stress': {'name': 'Chronic Stress', 'path': DATA_DIR / 'stress' / 'sample'}
}


def load_augmented_data(data_path):
    """Load augmented data (200 samples) if available, else original."""
    # Try augmented first
    augmented_files = list(data_path.glob('*augmented*.csv'))
    if augmented_files:
        csv_file = augmented_files[0]
        print(f"    Using augmented data: {csv_file.name}")
    else:
        csv_files = list(data_path.glob('*.csv'))
        if not csv_files:
            return None, None, None
        csv_file = csv_files[0]
        print(f"    Using original data: {csv_file.name}")

    df = pd.read_csv(csv_file)
    if 'label' not in df.columns:
        return None, None, None

    y = df['label'].values
    feature_cols = [c for c in df.columns if c not in ['label', 'subject_id', 'class']]
    X = df[feature_cols].values

    return X, y, feature_cols


def apply_feature_selection(X, y, n_features=25, method='mutual_info'):
    """Apply feature selection to reduce dimensionality."""
    if method == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
    elif method == 'rfe':
        base_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        selector = RFE(base_model, n_features_to_select=min(n_features, X.shape[1]))
    elif method == 'model':
        base_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        base_model.fit(X, y)
        selector = SelectFromModel(base_model, max_features=min(n_features, X.shape[1]))

    X_selected = selector.fit_transform(X, y)
    return X_selected, selector


def build_regularized_ensemble():
    """Build ensemble with regularization to prevent overfitting."""
    return VotingClassifier(
        estimators=[
            # Regularized Random Forest (limited depth)
            ('rf1', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,  # Limit depth
                min_samples_split=5,  # Require more samples to split
                min_samples_leaf=3,  # Require more samples in leaves
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )),
            # Regularized Extra Trees
            ('et1', ExtraTreesClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )),
            # Regularized Gradient Boosting (low learning rate)
            ('gb1', GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,  # Shallow trees
                learning_rate=0.05,  # Low learning rate
                min_samples_split=5,
                subsample=0.8,  # Subsampling for regularization
                random_state=42
            )),
            # SVM with regularization
            ('svm', SVC(
                kernel='rbf',
                C=1.0,  # Lower C = more regularization
                gamma='scale',
                probability=True,
                random_state=42
            )),
            # Regularized Logistic Regression
            ('lr', LogisticRegression(
                C=0.1,  # Strong regularization
                penalty='l2',
                max_iter=1000,
                random_state=42
            )),
            # MLP with regularization
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(50, 25),  # Smaller network
                alpha=0.01,  # L2 regularization
                early_stopping=True,  # Early stopping
                validation_fraction=0.2,
                max_iter=1000,
                random_state=42
            ))
        ],
        voting='soft',
        n_jobs=-1
    )


def bootstrap_ci(y_true, y_pred, metric_func=accuracy_score, n_bootstrap=1000, ci=95):
    """Calculate bootstrap confidence interval."""
    np.random.seed(42)
    scores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        scores.append(metric_func(np.array(y_true)[indices], np.array(y_pred)[indices]))

    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return np.mean(scores), lower, upper


def train_robust_model(disease, config):
    """Train model with anti-overfitting measures."""
    print(f"\n{'='*70}")
    print(f"  ROBUST TRAINING: {config['name'].upper()}")
    print(f"{'='*70}")

    # Step 1: Load augmented data
    print("\n  Step 1: Loading augmented data...")
    X, y, feature_names = load_augmented_data(config['path'])
    if X is None:
        print("    ERROR: Could not load data")
        return None

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"    Samples: {len(y)}, Features: {X.shape[1]}")
    print(f"    Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Step 2: Split for external validation (20% held out)
    print("\n  Step 2: Creating external validation set...")
    X_train_full, X_external, y_train_full, y_external = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Training set: {len(y_train_full)}, External set: {len(y_external)}")

    # Step 3: Scale data
    print("\n  Step 3: Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_external_scaled = scaler.transform(X_external)

    # Step 4: Feature selection
    print("\n  Step 4: Feature selection (top 25 features)...")
    n_select = min(25, X_train_scaled.shape[1])
    X_train_selected, selector = apply_feature_selection(
        X_train_scaled, y_train_full, n_features=n_select, method='mutual_info'
    )
    X_external_selected = selector.transform(X_external_scaled)
    print(f"    Selected {X_train_selected.shape[1]} features")

    # Step 5: Build regularized ensemble
    print("\n  Step 5: Building regularized ensemble...")
    model = build_regularized_ensemble()

    # Step 6: Cross-validation on training set
    print("\n  Step 6: 5-Fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_y_true = []
    all_y_pred = []
    fold_scores = []
    train_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_selected, y_train_full)):
        X_train, X_val = X_train_selected[train_idx], X_train_selected[val_idx]
        y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

        model.fit(X_train, y_train)

        # Training accuracy (to check overfitting)
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        train_scores.append(train_acc)

        # Validation accuracy
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        fold_scores.append(val_acc)

        all_y_true.extend(y_val)
        all_y_pred.extend(val_pred)

        gap = train_acc - val_acc
        print(f"    Fold {fold+1}: Train={train_acc*100:.1f}%, Val={val_acc*100:.1f}%, Gap={gap*100:.1f}%")

    # CV metrics
    cv_acc_mean, cv_acc_lower, cv_acc_upper = bootstrap_ci(all_y_true, all_y_pred)
    cv_f1 = f1_score(all_y_true, all_y_pred)

    avg_train = np.mean(train_scores)
    avg_val = np.mean(fold_scores)
    avg_gap = avg_train - avg_val

    print(f"\n    CV Accuracy: {cv_acc_mean*100:.2f}% (95% CI: [{cv_acc_lower*100:.2f}%, {cv_acc_upper*100:.2f}%])")
    print(f"    CV F1 Score: {cv_f1*100:.2f}%")
    print(f"    Avg Train-Val Gap: {avg_gap*100:.2f}%")

    # Step 7: External validation
    print("\n  Step 7: External Validation (held-out 20%)...")
    model.fit(X_train_selected, y_train_full)
    y_external_pred = model.predict(X_external_selected)

    ext_acc = accuracy_score(y_external, y_external_pred)
    ext_f1 = f1_score(y_external, y_external_pred)
    ext_precision = precision_score(y_external, y_external_pred)
    ext_recall = recall_score(y_external, y_external_pred)

    print(f"    External Accuracy: {ext_acc*100:.2f}%")
    print(f"    External F1 Score: {ext_f1*100:.2f}%")
    print(f"    External Precision: {ext_precision*100:.2f}%")
    print(f"    External Recall: {ext_recall*100:.2f}%")

    cm = confusion_matrix(y_external, y_external_pred)
    print(f"    Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    # Step 8: Overfitting assessment
    print("\n  Step 8: Overfitting Assessment...")
    cv_external_gap = cv_acc_mean - ext_acc

    if avg_gap < 0.05 and cv_external_gap < 0.10:
        overfit_status = "LOW - Model generalizes well"
    elif avg_gap < 0.10 and cv_external_gap < 0.15:
        overfit_status = "MODERATE - Some overfitting present"
    else:
        overfit_status = "HIGH - Significant overfitting"

    print(f"    Train-Val Gap: {avg_gap*100:.2f}%")
    print(f"    CV-External Gap: {cv_external_gap*100:.2f}%")
    print(f"    Overfitting Status: {overfit_status}")

    # Step 9: Save model
    print("\n  Step 9: Saving model...")
    model_path = MODELS_DIR / f"{disease}_robust_model.joblib"
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'selector': selector,
        'cv_accuracy': cv_acc_mean,
        'cv_accuracy_ci_lower': cv_acc_lower,
        'cv_accuracy_ci_upper': cv_acc_upper,
        'cv_f1': cv_f1,
        'external_accuracy': ext_acc,
        'external_f1': ext_f1,
        'train_val_gap': avg_gap,
        'cv_external_gap': cv_external_gap,
        'overfit_status': overfit_status
    }, model_path)
    print(f"    Saved: {model_path}")

    return {
        'disease': disease,
        'name': config['name'],
        'n_samples': len(y),
        'n_features_original': X.shape[1],
        'n_features_selected': X_train_selected.shape[1],
        'cv_accuracy': cv_acc_mean,
        'cv_accuracy_ci_lower': cv_acc_lower,
        'cv_accuracy_ci_upper': cv_acc_upper,
        'cv_f1': cv_f1,
        'external_accuracy': ext_acc,
        'external_f1': ext_f1,
        'train_val_gap': avg_gap,
        'cv_external_gap': cv_external_gap,
        'overfit_status': overfit_status
    }


def main():
    print("="*70)
    print("  ROBUST MODEL TRAINING WITH ANTI-OVERFITTING MEASURES")
    print("="*70)
    print("""
  Applied Techniques:
  1. Augmented data (200 samples per disease)
  2. External validation (20% held-out)
  3. Regularization (limited depth, min samples, L2)
  4. Feature selection (top 25 mutual info features)
  5. Cross-validation with 95% CI
""")

    all_results = {}

    for disease, config in DISEASES.items():
        result = train_robust_model(disease, config)
        if result:
            all_results[disease] = result

    # Summary
    print("\n\n" + "="*120)
    print("  ROBUST TRAINING SUMMARY")
    print("="*120)

    print(f"\n{'Disease':<22} {'N':<6} {'CV Acc':<12} {'95% CI':<22} {'Ext Acc':<12} {'Gap':<10} {'Status':<20}")
    print("-"*120)

    for disease, result in all_results.items():
        n = result['n_samples']
        cv_acc = f"{result['cv_accuracy']*100:.1f}%"
        ci = f"[{result['cv_accuracy_ci_lower']*100:.1f}%-{result['cv_accuracy_ci_upper']*100:.1f}%]"
        ext_acc = f"{result['external_accuracy']*100:.1f}%"
        gap = f"{result['cv_external_gap']*100:.1f}%"
        status = result['overfit_status'].split(' - ')[0]

        print(f"{result['name']:<22} {n:<6} {cv_acc:<12} {ci:<22} {ext_acc:<12} {gap:<10} {status:<20}")

    print("="*120)

    # Calculate averages
    avg_cv = np.mean([r['cv_accuracy'] for r in all_results.values()])
    avg_ext = np.mean([r['external_accuracy'] for r in all_results.values()])
    avg_gap = np.mean([r['cv_external_gap'] for r in all_results.values()])

    print(f"\n  Average CV Accuracy: {avg_cv*100:.2f}%")
    print(f"  Average External Accuracy: {avg_ext*100:.2f}%")
    print(f"  Average CV-External Gap: {avg_gap*100:.2f}%")

    # Interpretation
    print("\n" + "="*70)
    print("  INTERPRETATION")
    print("="*70)

    if avg_gap < 0.05:
        print("\n  EXCELLENT: Models generalize very well to held-out data")
    elif avg_gap < 0.10:
        print("\n  GOOD: Models show acceptable generalization")
    elif avg_gap < 0.15:
        print("\n  MODERATE: Some overfitting present, consider more data")
    else:
        print("\n  CONCERNING: Significant overfitting detected")

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"robust_training_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({k: {kk: float(vv) if isinstance(vv, (np.float64, np.float32)) else vv
                      for kk, vv in v.items()}
                  for k, v in all_results.items()}, f, indent=2, default=str)

    print(f"\n  Results saved: {results_file}")
    print(f"  Models saved to: {MODELS_DIR}/")

    return all_results


if __name__ == '__main__':
    main()
