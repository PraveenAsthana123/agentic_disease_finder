#!/usr/bin/env python
"""
OVERFITTING ANALYSIS REPORT
Comprehensive analysis to detect overfitting in disease classification models:
1. Training vs Test accuracy comparison
2. Cross-validation variance analysis
3. Learning curve analysis
4. Original vs Augmented data comparison
5. Model complexity analysis
6. Recommendations
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, learning_curve, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"

DISEASES = {
    'epilepsy': {'name': 'Epilepsy', 'path': DATA_DIR / 'epilepsy' / 'sample'},
    'parkinson': {'name': "Parkinson's Disease", 'path': DATA_DIR / 'parkinson' / 'sample'},
    'alzheimer': {'name': "Alzheimer's Disease", 'path': DATA_DIR / 'alzheimer' / 'sample'},
    'schizophrenia': {'name': 'Schizophrenia', 'path': DATA_DIR / 'schizophrenia' / 'sample'},
    'depression': {'name': 'Major Depression', 'path': DATA_DIR / 'depression' / 'sample'},
    'autism': {'name': 'Autism Spectrum', 'path': DATA_DIR / 'autism' / 'sample'},
    'stress': {'name': 'Chronic Stress', 'path': DATA_DIR / 'stress' / 'sample'}
}


def load_disease_data(data_path):
    """Load disease data from CSV."""
    for csv_file in data_path.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file)
            if 'label' in df.columns:
                y = df['label'].values
                feature_cols = [c for c in df.columns if c not in ['label', 'subject_id', 'class']]
                X = df[feature_cols].values
                return X, y, feature_cols
        except:
            continue
    return None, None, None


def analyze_overfitting(disease, config):
    """Comprehensive overfitting analysis for a disease."""
    print(f"\n{'='*70}")
    print(f"  OVERFITTING ANALYSIS: {config['name'].upper()}")
    print(f"{'='*70}")

    # Load data
    X, y, feature_names = load_disease_data(config['path'])
    if X is None:
        print("  ERROR: Could not load data")
        return None

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_samples = len(y)
    n_features = X.shape[1]

    print(f"\n  Dataset: {n_samples} samples, {n_features} features")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    results = {
        'disease': disease,
        'name': config['name'],
        'n_samples': n_samples,
        'n_features': n_features,
        'overfitting_indicators': []
    }

    # =========================================
    # 1. TRAINING VS TEST ACCURACY
    # =========================================
    print(f"\n  1. TRAINING VS TEST ACCURACY")
    print(f"  {'-'*50}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Simple model
    rf_simple = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    train_accs_simple = []
    test_accs_simple = []

    for train_idx, test_idx in cv.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf_simple.fit(X_train, y_train)
        train_accs_simple.append(accuracy_score(y_train, rf_simple.predict(X_train)))
        test_accs_simple.append(accuracy_score(y_test, rf_simple.predict(X_test)))

    # Complex model (no depth limit)
    rf_complex = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42)
    train_accs_complex = []
    test_accs_complex = []

    for train_idx, test_idx in cv.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf_complex.fit(X_train, y_train)
        train_accs_complex.append(accuracy_score(y_train, rf_complex.predict(X_train)))
        test_accs_complex.append(accuracy_score(y_test, rf_complex.predict(X_test)))

    train_mean_simple = np.mean(train_accs_simple)
    test_mean_simple = np.mean(test_accs_simple)
    gap_simple = train_mean_simple - test_mean_simple

    train_mean_complex = np.mean(train_accs_complex)
    test_mean_complex = np.mean(test_accs_complex)
    gap_complex = train_mean_complex - test_mean_complex

    print(f"\n  Simple Model (RF, max_depth=5):")
    print(f"    Training Accuracy: {train_mean_simple*100:.2f}%")
    print(f"    Test Accuracy:     {test_mean_simple*100:.2f}%")
    print(f"    Gap:               {gap_simple*100:.2f}%")

    print(f"\n  Complex Model (RF, max_depth=None):")
    print(f"    Training Accuracy: {train_mean_complex*100:.2f}%")
    print(f"    Test Accuracy:     {test_mean_complex*100:.2f}%")
    print(f"    Gap:               {gap_complex*100:.2f}%")

    results['train_acc_simple'] = train_mean_simple
    results['test_acc_simple'] = test_mean_simple
    results['gap_simple'] = gap_simple
    results['train_acc_complex'] = train_mean_complex
    results['test_acc_complex'] = test_mean_complex
    results['gap_complex'] = gap_complex

    if gap_complex > 0.10:
        results['overfitting_indicators'].append("HIGH train-test gap (>10%)")
    elif gap_complex > 0.05:
        results['overfitting_indicators'].append("MODERATE train-test gap (5-10%)")

    # =========================================
    # 2. CROSS-VALIDATION VARIANCE
    # =========================================
    print(f"\n  2. CROSS-VALIDATION VARIANCE")
    print(f"  {'-'*50}")

    cv_scores = cross_val_score(rf_complex, X_scaled, y, cv=cv, scoring='accuracy')
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    cv_range = np.max(cv_scores) - np.min(cv_scores)

    print(f"\n  5-Fold CV Scores: {[f'{s*100:.1f}%' for s in cv_scores]}")
    print(f"  Mean: {cv_mean*100:.2f}%")
    print(f"  Std:  {cv_std*100:.2f}%")
    print(f"  Range: {cv_range*100:.2f}%")

    results['cv_mean'] = cv_mean
    results['cv_std'] = cv_std
    results['cv_range'] = cv_range
    results['cv_scores'] = cv_scores.tolist()

    if cv_std > 0.10:
        results['overfitting_indicators'].append("HIGH CV variance (std >10%)")
    elif cv_std > 0.05:
        results['overfitting_indicators'].append("MODERATE CV variance (std 5-10%)")

    # =========================================
    # 3. LEARNING CURVE ANALYSIS
    # =========================================
    print(f"\n  3. LEARNING CURVE ANALYSIS")
    print(f"  {'-'*50}")

    train_sizes = np.linspace(0.2, 1.0, 5)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        rf_complex, X_scaled, y,
        train_sizes=train_sizes,
        cv=3,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )

    print(f"\n  {'Train Size':<15} {'Train Acc':<15} {'Test Acc':<15} {'Gap':<15}")
    print(f"  {'-'*60}")

    for i, size in enumerate(train_sizes_abs):
        train_mean = np.mean(train_scores[i])
        test_mean = np.mean(test_scores[i])
        gap = train_mean - test_mean
        print(f"  {size:<15} {train_mean*100:<15.2f} {test_mean*100:<15.2f} {gap*100:<15.2f}")

    # Check if gap increases with training size (sign of overfitting)
    gaps = [np.mean(train_scores[i]) - np.mean(test_scores[i]) for i in range(len(train_sizes_abs))]
    gap_trend = gaps[-1] - gaps[0]

    results['learning_curve_gap_trend'] = gap_trend

    if gap_trend > 0.05:
        results['overfitting_indicators'].append("Gap INCREASES with more data (overfitting sign)")
    elif gaps[-1] > 0.15:
        results['overfitting_indicators'].append("Large final gap in learning curve")

    # =========================================
    # 4. SAMPLE SIZE VS FEATURES RATIO
    # =========================================
    print(f"\n  4. SAMPLE SIZE VS FEATURES RATIO")
    print(f"  {'-'*50}")

    ratio = n_samples / n_features
    print(f"\n  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Ratio (samples/features): {ratio:.2f}")

    if ratio < 2:
        status = "CRITICAL - Very high overfitting risk"
        results['overfitting_indicators'].append("Sample/feature ratio < 2 (very high risk)")
    elif ratio < 5:
        status = "WARNING - Moderate overfitting risk"
        results['overfitting_indicators'].append("Sample/feature ratio < 5 (moderate risk)")
    elif ratio < 10:
        status = "CAUTION - Some overfitting risk"
    else:
        status = "OK - Good sample/feature ratio"

    print(f"  Status: {status}")
    results['sample_feature_ratio'] = ratio

    # =========================================
    # 5. MODEL COMPLEXITY COMPARISON
    # =========================================
    print(f"\n  5. MODEL COMPLEXITY COMPARISON")
    print(f"  {'-'*50}")

    models = {
        'RF (depth=3)': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
        'RF (depth=5)': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'RF (depth=10)': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'RF (depth=None)': RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
        'GB (depth=3)': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        'GB (depth=5)': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    }

    print(f"\n  {'Model':<20} {'Train Acc':<15} {'Test Acc':<15} {'Gap':<10}")
    print(f"  {'-'*60}")

    complexity_results = {}
    for name, model in models.items():
        train_accs = []
        test_accs = []

        for train_idx, test_idx in cv.split(X_scaled, y):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            train_accs.append(accuracy_score(y_train, model.predict(X_train)))
            test_accs.append(accuracy_score(y_test, model.predict(X_test)))

        train_mean = np.mean(train_accs)
        test_mean = np.mean(test_accs)
        gap = train_mean - test_mean

        complexity_results[name] = {'train': train_mean, 'test': test_mean, 'gap': gap}
        print(f"  {name:<20} {train_mean*100:<15.2f} {test_mean*100:<15.2f} {gap*100:<10.2f}")

    results['complexity_analysis'] = complexity_results

    # =========================================
    # 6. PERFECT ACCURACY WARNING
    # =========================================
    print(f"\n  6. PERFECT ACCURACY ANALYSIS")
    print(f"  {'-'*50}")

    if test_mean_complex >= 0.99:
        print(f"\n  WARNING: Near-perfect test accuracy ({test_mean_complex*100:.2f}%)")
        print(f"  This could indicate:")
        print(f"    - Data leakage between train/test")
        print(f"    - Overly simple classification task")
        print(f"    - Synthetic/simulated data with clear patterns")
        print(f"    - Small sample size with lucky splits")
        results['overfitting_indicators'].append("Near-perfect accuracy (potential data issues)")
        results['perfect_accuracy_warning'] = True
    else:
        print(f"\n  Test accuracy: {test_mean_complex*100:.2f}% (not suspicious)")
        results['perfect_accuracy_warning'] = False

    # =========================================
    # 7. OVERFITTING SCORE
    # =========================================
    print(f"\n  7. OVERFITTING RISK SCORE")
    print(f"  {'-'*50}")

    # Calculate overfitting score (0-100)
    score = 0

    # Train-test gap contribution (0-30)
    score += min(30, gap_complex * 300)

    # CV variance contribution (0-20)
    score += min(20, cv_std * 200)

    # Sample/feature ratio contribution (0-30)
    if ratio < 2:
        score += 30
    elif ratio < 5:
        score += 20
    elif ratio < 10:
        score += 10

    # Learning curve trend contribution (0-20)
    score += min(20, max(0, gap_trend * 200))

    results['overfitting_score'] = score

    if score < 20:
        risk_level = "LOW"
        risk_color = "GREEN"
    elif score < 40:
        risk_level = "MODERATE"
        risk_color = "YELLOW"
    elif score < 60:
        risk_level = "HIGH"
        risk_color = "ORANGE"
    else:
        risk_level = "CRITICAL"
        risk_color = "RED"

    results['risk_level'] = risk_level

    print(f"\n  Overfitting Risk Score: {score:.1f}/100")
    print(f"  Risk Level: {risk_level} ({risk_color})")

    # =========================================
    # 8. RECOMMENDATIONS
    # =========================================
    print(f"\n  8. RECOMMENDATIONS")
    print(f"  {'-'*50}")

    recommendations = []

    if ratio < 5:
        recommendations.append("Collect more training data (current ratio is low)")
    if gap_complex > 0.05:
        recommendations.append("Use regularization (reduce max_depth, increase min_samples_leaf)")
    if cv_std > 0.05:
        recommendations.append("Use stratified sampling and more CV folds")
    if test_mean_complex >= 0.99:
        recommendations.append("Verify no data leakage; test on truly independent data")
    if n_features > n_samples:
        recommendations.append("Apply feature selection to reduce dimensionality")

    if not recommendations:
        recommendations.append("Model appears well-calibrated based on available data")

    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    results['recommendations'] = recommendations

    return results


def main():
    print("="*70)
    print("  COMPREHENSIVE OVERFITTING ANALYSIS REPORT")
    print("  Analyzing all disease classification models")
    print("="*70)

    all_results = {}

    for disease, config in DISEASES.items():
        result = analyze_overfitting(disease, config)
        if result:
            all_results[disease] = result

    # =========================================
    # SUMMARY TABLE
    # =========================================
    print("\n\n" + "="*100)
    print("  OVERFITTING SUMMARY - ALL DISEASES")
    print("="*100)

    print(f"\n{'Disease':<25} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<10} {'CV Std':<10} {'Risk Score':<12} {'Risk Level':<12}")
    print("-"*100)

    for disease, result in all_results.items():
        train = f"{result['train_acc_complex']*100:.1f}%"
        test = f"{result['test_acc_complex']*100:.1f}%"
        gap = f"{result['gap_complex']*100:.1f}%"
        cv_std = f"{result['cv_std']*100:.1f}%"
        score = f"{result['overfitting_score']:.1f}"
        risk = result['risk_level']

        print(f"{result['name']:<25} {train:<12} {test:<12} {gap:<10} {cv_std:<10} {score:<12} {risk:<12}")

    print("="*100)

    # =========================================
    # OVERALL ASSESSMENT
    # =========================================
    print("\n" + "="*70)
    print("  OVERALL ASSESSMENT")
    print("="*70)

    avg_score = np.mean([r['overfitting_score'] for r in all_results.values()])
    max_score = max([r['overfitting_score'] for r in all_results.values()])
    high_risk_diseases = [r['name'] for r in all_results.values() if r['overfitting_score'] >= 40]

    print(f"\n  Average Overfitting Score: {avg_score:.1f}/100")
    print(f"  Maximum Overfitting Score: {max_score:.1f}/100")

    if high_risk_diseases:
        print(f"\n  HIGH RISK DISEASES: {', '.join(high_risk_diseases)}")
    else:
        print(f"\n  No diseases with high overfitting risk detected.")

    # Key findings
    print("\n  KEY FINDINGS:")
    print("  " + "-"*50)

    findings = []

    # Check for perfect accuracy
    perfect_acc = [r['name'] for r in all_results.values() if r['test_acc_complex'] >= 0.99]
    if perfect_acc:
        findings.append(f"Near-perfect accuracy in: {', '.join(perfect_acc)}")

    # Check for high train-test gap
    high_gap = [r['name'] for r in all_results.values() if r['gap_complex'] >= 0.05]
    if high_gap:
        findings.append(f"Significant train-test gap in: {', '.join(high_gap)}")

    # Check for small sample sizes
    small_samples = [r['name'] for r in all_results.values() if r['n_samples'] < 100]
    if small_samples:
        findings.append(f"Small sample size (<100) in: {', '.join(small_samples)}")

    # Check sample/feature ratio
    low_ratio = [r['name'] for r in all_results.values() if r['sample_feature_ratio'] < 3]
    if low_ratio:
        findings.append(f"Low sample/feature ratio in: {', '.join(low_ratio)}")

    if not findings:
        findings.append("No major overfitting concerns detected")

    for i, finding in enumerate(findings, 1):
        print(f"  {i}. {finding}")

    # =========================================
    # INTERPRETATION GUIDE
    # =========================================
    print("\n" + "="*70)
    print("  INTERPRETATION GUIDE")
    print("="*70)

    print("""
  TRAIN-TEST GAP:
    < 2%  : Excellent - model generalizes well
    2-5%  : Good - minimal overfitting
    5-10% : Moderate - some overfitting present
    > 10% : High - significant overfitting

  CV STANDARD DEVIATION:
    < 3%  : Excellent - stable across folds
    3-5%  : Good - acceptable variance
    5-10% : Moderate - some instability
    > 10% : High - unstable model

  SAMPLE/FEATURE RATIO:
    > 10  : Excellent - sufficient data
    5-10  : Good - adequate data
    2-5   : Moderate - limited data
    < 2   : Poor - high overfitting risk

  OVERFITTING RISK SCORE:
    0-20  : LOW (Green) - Well-calibrated model
    20-40 : MODERATE (Yellow) - Minor concerns
    40-60 : HIGH (Orange) - Significant overfitting
    60+   : CRITICAL (Red) - Severe overfitting

  NOTES ON PERFECT ACCURACY:
    - 100% accuracy on small datasets is often suspicious
    - Could indicate data leakage or synthetic data patterns
    - Always validate on truly independent external data
    - Consider if the classification task is inherently easy
""")

    return all_results


if __name__ == '__main__':
    main()
