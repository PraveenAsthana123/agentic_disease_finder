#!/usr/bin/env python
"""
Deep Training for All Diseases
Applies the same advanced techniques that improved autism to all diseases:
1. Data augmentation (noise injection)
2. Advanced feature engineering (interaction features)
3. Hyperparameter optimization
4. Advanced oversampling (SMOTE variants)
5. Ultimate ensemble (25 models)
6. Bootstrap confidence intervals
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import (StratifiedKFold, RandomizedSearchCV, GridSearchCV,
                                     cross_val_score)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                             ExtraTreesClassifier, AdaBoostClassifier,
                             VotingClassifier, BaggingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import json
import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.combine import SMOTETomek
    HAS_IMBLEARN = True
except:
    HAS_IMBLEARN = False

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


def analyze_features(X, y, feature_names):
    """Analyze feature separability."""
    class_0 = X[y == 0]
    class_1 = X[y == 1]

    separability = []
    for i, name in enumerate(feature_names):
        diff = abs(np.mean(class_0[:, i]) - np.mean(class_1[:, i]))
        pooled_std = np.sqrt((np.std(class_0[:, i])**2 + np.std(class_1[:, i])**2) / 2)
        effect_size = diff / (pooled_std + 1e-10)
        separability.append((name, effect_size, i))

    separability.sort(key=lambda x: x[1], reverse=True)
    return separability


def create_interaction_features(X, feature_names, separability, top_k=8):
    """Create interaction features between top discriminative features."""
    top_indices = [sep[2] for sep in separability[:top_k]]

    interactions = []
    interaction_names = []

    for i in range(len(top_indices)):
        for j in range(i + 1, len(top_indices)):
            idx_i, idx_j = top_indices[i], top_indices[j]

            # Product interaction
            interactions.append(X[:, idx_i] * X[:, idx_j])
            interaction_names.append(f"{feature_names[idx_i]}_x_{feature_names[idx_j]}")

            # Ratio interaction
            ratio = X[:, idx_i] / (X[:, idx_j] + 1e-10)
            interactions.append(ratio)
            interaction_names.append(f"{feature_names[idx_i]}_div_{feature_names[idx_j]}")

    if interactions:
        X_interactions = np.column_stack(interactions)
        X_augmented = np.hstack([X, X_interactions])
        all_names = list(feature_names) + interaction_names
        return X_augmented, all_names

    return X, feature_names


def augment_with_noise(X, y, n_augmented=100, noise_level=0.03):
    """Augment data by adding Gaussian noise."""
    np.random.seed(42)
    X_aug_list = [X]
    y_aug_list = [y]

    for _ in range(n_augmented):
        idx = np.random.randint(0, len(X))
        sample = X[idx].copy()
        label = y[idx]

        noise = np.random.normal(0, noise_level * np.std(X, axis=0), sample.shape)
        augmented_sample = sample + noise

        X_aug_list.append(augmented_sample.reshape(1, -1))
        y_aug_list.append([label])

    X_augmented = np.vstack(X_aug_list)
    y_augmented = np.concatenate(y_aug_list)

    return X_augmented, y_augmented


def apply_oversampling(X, y):
    """Apply best oversampling strategy."""
    if not HAS_IMBLEARN:
        return X, y

    strategies = {}

    for k in [3, 5]:
        try:
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_res, y_res = smote.fit_resample(X, y)
            strategies[f'SMOTE_k{k}'] = (X_res, y_res)
        except:
            pass

    try:
        bl_smote = BorderlineSMOTE(random_state=42, k_neighbors=3)
        X_res, y_res = bl_smote.fit_resample(X, y)
        strategies['BorderlineSMOTE'] = (X_res, y_res)
    except:
        pass

    try:
        smote_tomek = SMOTETomek(random_state=42)
        X_res, y_res = smote_tomek.fit_resample(X, y)
        strategies['SMOTETomek'] = (X_res, y_res)
    except:
        pass

    # Quick evaluation to find best
    best_strategy = None
    best_score = 0

    for name, (X_res, y_res) in strategies.items():
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        try:
            scores = cross_val_score(rf, X_res, y_res, cv=cv, scoring='accuracy')
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_strategy = name
        except:
            pass

    if best_strategy:
        return strategies[best_strategy]

    return X, y


def optimize_hyperparameters(X, y):
    """Quick hyperparameter optimization."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    optimized = {}

    # RF
    rf_params = {
        'n_estimators': [200, 300, 500],
        'max_depth': [None, 10, 15],
        'min_samples_split': [2, 3],
        'max_features': ['sqrt', 'log2']
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_search = RandomizedSearchCV(rf, rf_params, n_iter=15, cv=cv, scoring='accuracy',
                                   random_state=42, n_jobs=-1)
    rf_search.fit(X, y)
    optimized['rf'] = rf_search.best_estimator_

    # SVM
    svm_params = {'C': [1, 10, 100], 'gamma': ['scale', 0.01, 0.001], 'kernel': ['rbf']}
    svm = SVC(probability=True, random_state=42)
    svm_search = GridSearchCV(svm, svm_params, cv=cv, scoring='accuracy', n_jobs=-1)
    svm_search.fit(X, y)
    optimized['svm'] = svm_search.best_estimator_

    # MLP
    mlp_params = {
        'hidden_layer_sizes': [(100,), (200, 100), (100, 50)],
        'alpha': [0.0001, 0.001],
        'activation': ['relu', 'tanh']
    }
    mlp = MLPClassifier(max_iter=2000, early_stopping=True, random_state=42)
    mlp_search = GridSearchCV(mlp, mlp_params, cv=cv, scoring='accuracy', n_jobs=-1)
    mlp_search.fit(X, y)
    optimized['mlp'] = mlp_search.best_estimator_

    return optimized


def build_ultimate_ensemble(optimized_models):
    """Build the ultimate 25-model ensemble."""
    rf_opt = optimized_models.get('rf', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    svm_opt = optimized_models.get('svm', SVC(kernel='rbf', C=10, probability=True, random_state=42))
    mlp_opt = optimized_models.get('mlp', MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=2000, random_state=42))

    ensemble = VotingClassifier(
        estimators=[
            # Optimized models
            ('rf_opt', rf_opt),
            ('svm_opt', svm_opt),
            ('mlp_opt', mlp_opt),
            # RF variants
            ('rf1', RandomForestClassifier(n_estimators=500, max_depth=None, random_state=42, n_jobs=-1)),
            ('rf2', RandomForestClassifier(n_estimators=500, max_depth=15, random_state=43, n_jobs=-1)),
            ('rf3', RandomForestClassifier(n_estimators=500, max_depth=20, random_state=44, n_jobs=-1)),
            # ExtraTrees
            ('et1', ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1)),
            ('et2', ExtraTreesClassifier(n_estimators=500, max_depth=15, random_state=43, n_jobs=-1)),
            # GradientBoosting
            ('gb1', GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=42)),
            ('gb2', GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=43)),
            ('gb3', GradientBoostingClassifier(n_estimators=200, max_depth=7, random_state=44)),
            # AdaBoost
            ('ada1', AdaBoostClassifier(n_estimators=300, random_state=42)),
            ('ada2', AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=43)),
            # Bagging
            ('bag1', BaggingClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
            ('bag2', BaggingClassifier(n_estimators=200, max_samples=0.8, random_state=43, n_jobs=-1)),
            # SVM variants
            ('svm1', SVC(kernel='rbf', C=10, probability=True, random_state=42)),
            ('svm2', SVC(kernel='rbf', C=100, gamma=0.01, probability=True, random_state=42)),
            ('svm3', SVC(kernel='poly', degree=3, C=10, probability=True, random_state=42)),
            # KNN
            ('knn1', KNeighborsClassifier(n_neighbors=3)),
            ('knn2', KNeighborsClassifier(n_neighbors=5, weights='distance')),
            ('knn3', KNeighborsClassifier(n_neighbors=7, weights='distance')),
            # MLP
            ('mlp1', MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=2000, early_stopping=True, random_state=42)),
            ('mlp2', MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=2000, early_stopping=True, random_state=43)),
            # Logistic Regression
            ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ],
        voting='soft',
        n_jobs=-1
    )
    return ensemble


def bootstrap_ci(y_true, y_pred, n_bootstrap=1000):
    """Calculate bootstrap 95% CI."""
    np.random.seed(42)
    accs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_boot = np.array(y_true)[indices]
        y_pred_boot = np.array(y_pred)[indices]
        accs.append(accuracy_score(y_true_boot, y_pred_boot))

    return np.percentile(accs, 2.5), np.percentile(accs, 97.5)


def deep_train_disease(disease, config):
    """Deep training for a single disease."""
    print(f"\n{'='*70}")
    print(f"  DEEP TRAINING: {config['name'].upper()}")
    print(f"{'='*70}")

    # Load data
    X, y, feature_names = load_disease_data(config['path'])
    if X is None:
        print("  ERROR: Could not load data")
        return None

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n  Original data: {len(y)} samples, {X.shape[1]} features")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Step 1: Analyze features
    print("\n  Step 1: Analyzing features...")
    separability = analyze_features(X, y, feature_names)
    print(f"  Top features: {', '.join([s[0] for s in separability[:5]])}")

    # Step 2: Create interaction features
    print("\n  Step 2: Creating interaction features...")
    X_interact, enhanced_names = create_interaction_features(X, feature_names, separability, top_k=8)
    print(f"  Features: {X.shape[1]} -> {X_interact.shape[1]}")

    # Step 3: Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_interact)

    # Step 4: Data augmentation
    print("\n  Step 3: Data augmentation...")
    X_aug, y_aug = augment_with_noise(X_scaled, y, n_augmented=100, noise_level=0.03)
    print(f"  Samples: {len(y)} -> {len(y_aug)}")

    # Step 5: Oversampling
    print("\n  Step 4: Oversampling...")
    X_over, y_over = apply_oversampling(X_aug, y_aug)
    print(f"  After oversampling: {len(y_over)} samples")

    # Step 6: Feature selection
    print("\n  Step 5: Feature selection...")
    n_features = min(40, X_over.shape[1])
    selector = SelectKBest(mutual_info_classif, k=n_features)
    X_selected = selector.fit_transform(X_over, y_over)
    print(f"  Selected {n_features} best features")

    # Step 7: Hyperparameter optimization
    print("\n  Step 6: Hyperparameter optimization...")
    optimized_models = optimize_hyperparameters(X_selected, y_over)

    # Step 8: Build and train ensemble
    print("\n  Step 7: Building ultimate ensemble (25 models)...")
    ensemble = build_ultimate_ensemble(optimized_models)

    # Step 9: Cross-validation
    print("\n  Step 8: 5-Fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_y_true = []
    all_y_pred = []
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_selected, y_over)):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y_over[train_idx], y_over[test_idx]

        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        acc = accuracy_score(y_test, y_pred)
        fold_scores.append(acc)
        print(f"    Fold {fold+1}: {acc*100:.2f}%")

    # Calculate metrics
    final_acc = accuracy_score(all_y_true, all_y_pred)
    final_f1 = f1_score(all_y_true, all_y_pred)
    ci_lower, ci_upper = bootstrap_ci(all_y_true, all_y_pred)

    print(f"\n  FINAL ACCURACY: {final_acc*100:.2f}% (95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%])")
    print(f"  F1 Score: {final_f1*100:.2f}%")
    print(f"  Fold std: {np.std(fold_scores)*100:.2f}%")

    cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"\n  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    # Save model
    print("\n  Saving model...")
    ensemble.fit(X_selected, y_over)

    model_path = MODELS_DIR / f"{disease}_deep_model.joblib"
    joblib.dump({
        'model': ensemble,
        'scaler': scaler,
        'selector': selector,
        'accuracy': final_acc,
        'accuracy_ci_lower': ci_lower,
        'accuracy_ci_upper': ci_upper,
        'f1': final_f1,
        'fold_scores': fold_scores
    }, model_path)
    print(f"  Model saved: {model_path}")

    return {
        'disease': disease,
        'name': config['name'],
        'accuracy': final_acc,
        'accuracy_ci_lower': ci_lower,
        'accuracy_ci_upper': ci_upper,
        'f1': final_f1,
        'fold_std': np.std(fold_scores)
    }


def main():
    print("="*70)
    print("  DEEP TRAINING FOR ALL DISEASES")
    print("  Using Advanced Techniques:")
    print("  - Feature Engineering (interaction features)")
    print("  - Data Augmentation (noise injection)")
    print("  - Advanced Oversampling (SMOTE variants)")
    print("  - Hyperparameter Optimization")
    print("  - Ultimate Ensemble (25 models)")
    print("="*70)

    all_results = {}

    for disease, config in DISEASES.items():
        result = deep_train_disease(disease, config)
        if result:
            all_results[disease] = result

    # Summary
    print("\n\n" + "="*100)
    print("  DEEP TRAINING SUMMARY - ALL DISEASES")
    print("="*100)

    print(f"\n{'Disease':<25} {'Accuracy':<15} {'95% CI':<25} {'F1 Score':<15}")
    print("-"*80)

    for disease, result in all_results.items():
        acc = f"{result['accuracy']*100:.2f}%"
        ci = f"[{result['accuracy_ci_lower']*100:.2f}%, {result['accuracy_ci_upper']*100:.2f}%]"
        f1 = f"{result['f1']*100:.2f}%"
        print(f"{result['name']:<25} {acc:<15} {ci:<25} {f1:<15}")

    print("="*100)

    # Calculate average
    avg_acc = np.mean([r['accuracy'] for r in all_results.values()])
    print(f"\n  Average Accuracy: {avg_acc*100:.2f}%")

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"deep_training_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({k: {kk: float(vv) if isinstance(vv, (np.float64, np.float32)) else vv
                      for kk, vv in v.items()}
                  for k, v in all_results.items()}, f, indent=2, default=str)

    print(f"\n  Results saved: {results_file}")
    print("\n  All models saved to: saved_models/")

    return all_results


if __name__ == '__main__':
    main()
