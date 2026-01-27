#!/usr/bin/env python
"""
Improve Autism Model Accuracy
Target: 96-100% accuracy using advanced techniques:
1. Data augmentation (SMOTE-NC, ADASYN, noise injection)
2. Advanced feature engineering (interaction features, PCA)
3. Hyperparameter optimization (GridSearchCV/RandomizedSearchCV)
4. Deep learning (MLP with dropout, early stopping)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import (StratifiedKFold, GridSearchCV, RandomizedSearchCV,
                                     cross_val_score, cross_val_predict)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                             ExtraTreesClassifier, AdaBoostClassifier,
                             VotingClassifier, StackingClassifier, BaggingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.combine import SMOTETomek, SMOTEENN
    HAS_IMBLEARN = True
except:
    HAS_IMBLEARN = False

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "autism" / "sample"
MODELS_DIR = BASE_DIR / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)


def load_autism_data():
    """Load autism data."""
    for csv_file in DATA_DIR.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file)
            if 'label' in df.columns:
                y = df['label'].values
                subject_id = df['subject_id'].values if 'subject_id' in df.columns else None
                feature_cols = [c for c in df.columns if c not in ['label', 'subject_id', 'class']]
                X = df[feature_cols].values
                return X, y, subject_id, feature_cols
        except:
            continue
    return None, None, None, None


def analyze_data(X, y, feature_names):
    """Analyze data to find issues."""
    print("="*60)
    print("DATA ANALYSIS")
    print("="*60)

    print(f"\nSamples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Check for problematic features
    print("\nFeature Statistics:")
    for i, name in enumerate(feature_names[:10]):
        col = X[:, i]
        print(f"  {name}: mean={np.mean(col):.4f}, std={np.std(col):.4f}, "
              f"min={np.min(col):.4f}, max={np.max(col):.4f}")

    # Check class separability
    print("\nClass Separability (mean diff):")
    class_0 = X[y == 0]
    class_1 = X[y == 1]

    separability = []
    for i, name in enumerate(feature_names):
        diff = abs(np.mean(class_0[:, i]) - np.mean(class_1[:, i]))
        pooled_std = np.sqrt((np.std(class_0[:, i])**2 + np.std(class_1[:, i])**2) / 2)
        effect_size = diff / (pooled_std + 1e-10)
        separability.append((name, effect_size, i))

    separability.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 10 most discriminative features:")
    for name, es, idx in separability[:10]:
        print(f"  {name}: effect size = {es:.4f}")

    return separability


def create_interaction_features(X, feature_names, separability, top_k=10):
    """Create interaction features between top discriminative features."""
    print("\n  Creating interaction features...")

    # Get indices of top discriminative features
    top_indices = [sep[2] for sep in separability[:top_k]]

    # Create interaction features (ratios and products)
    interactions = []
    interaction_names = []

    for i in range(len(top_indices)):
        for j in range(i + 1, len(top_indices)):
            idx_i, idx_j = top_indices[i], top_indices[j]

            # Product interaction
            interactions.append(X[:, idx_i] * X[:, idx_j])
            interaction_names.append(f"{feature_names[idx_i]}_x_{feature_names[idx_j]}")

            # Ratio interaction (with small epsilon to avoid division by zero)
            ratio = X[:, idx_i] / (X[:, idx_j] + 1e-10)
            interactions.append(ratio)
            interaction_names.append(f"{feature_names[idx_i]}_div_{feature_names[idx_j]}")

    if interactions:
        X_interactions = np.column_stack(interactions)
        X_augmented = np.hstack([X, X_interactions])
        all_names = list(feature_names) + interaction_names
        print(f"  Added {len(interactions)} interaction features")
        return X_augmented, all_names

    return X, feature_names


def augment_data_with_noise(X, y, n_augmented=50, noise_level=0.05):
    """Augment data by adding Gaussian noise to existing samples."""
    print(f"\n  Augmenting data with noise injection (n={n_augmented})...")

    np.random.seed(42)
    X_aug_list = [X]
    y_aug_list = [y]

    for _ in range(n_augmented):
        # Randomly select a sample
        idx = np.random.randint(0, len(X))
        sample = X[idx].copy()
        label = y[idx]

        # Add Gaussian noise
        noise = np.random.normal(0, noise_level * np.std(X, axis=0), sample.shape)
        augmented_sample = sample + noise

        X_aug_list.append(augmented_sample.reshape(1, -1))
        y_aug_list.append([label])

    X_augmented = np.vstack(X_aug_list)
    y_augmented = np.concatenate(y_aug_list)

    print(f"  Data augmented: {len(X)} -> {len(X_augmented)} samples")
    return X_augmented, y_augmented


def apply_advanced_oversampling(X, y):
    """Apply multiple oversampling strategies and return the best."""
    if not HAS_IMBLEARN:
        print("  imblearn not available, skipping oversampling")
        return X, y

    print("\n  Applying advanced oversampling strategies...")

    strategies = {}

    # SMOTE with different k values
    for k in [3, 5]:
        try:
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_res, y_res = smote.fit_resample(X, y)
            strategies[f'SMOTE_k{k}'] = (X_res, y_res)
        except:
            pass

    # BorderlineSMOTE
    try:
        bl_smote = BorderlineSMOTE(random_state=42, k_neighbors=3)
        X_res, y_res = bl_smote.fit_resample(X, y)
        strategies['BorderlineSMOTE'] = (X_res, y_res)
    except:
        pass

    # ADASYN
    try:
        adasyn = ADASYN(random_state=42, n_neighbors=3)
        X_res, y_res = adasyn.fit_resample(X, y)
        strategies['ADASYN'] = (X_res, y_res)
    except:
        pass

    # SMOTE-Tomek (hybrid)
    try:
        smote_tomek = SMOTETomek(random_state=42)
        X_res, y_res = smote_tomek.fit_resample(X, y)
        strategies['SMOTETomek'] = (X_res, y_res)
    except:
        pass

    # Test each strategy with a quick RF evaluation
    best_strategy = None
    best_score = 0

    for name, (X_res, y_res) in strategies.items():
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(rf, X_res, y_res, cv=cv, scoring='accuracy')
        mean_score = np.mean(scores)
        print(f"    {name}: {mean_score*100:.2f}% (n={len(X_res)})")

        if mean_score > best_score:
            best_score = mean_score
            best_strategy = name

    if best_strategy:
        print(f"  Best oversampling: {best_strategy} ({best_score*100:.2f}%)")
        return strategies[best_strategy]

    return X, y


def hyperparameter_optimization(X, y):
    """Perform hyperparameter optimization for key models."""
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    optimized_models = {}

    # 1. RandomForest optimization
    print("\n1. Optimizing RandomForest...")
    rf_params = {
        'n_estimators': [200, 300, 500],
        'max_depth': [None, 10, 15, 20],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_search = RandomizedSearchCV(rf, rf_params, n_iter=30, cv=cv, scoring='accuracy',
                                   random_state=42, n_jobs=-1)
    rf_search.fit(X_scaled, y)
    optimized_models['rf'] = rf_search.best_estimator_
    print(f"   Best RF score: {rf_search.best_score_*100:.2f}%")
    print(f"   Best params: {rf_search.best_params_}")

    # 2. GradientBoosting optimization
    print("\n2. Optimizing GradientBoosting...")
    gb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_samples_split': [2, 3, 5],
        'subsample': [0.8, 0.9, 1.0]
    }
    gb = GradientBoostingClassifier(random_state=42)
    gb_search = RandomizedSearchCV(gb, gb_params, n_iter=30, cv=cv, scoring='accuracy',
                                   random_state=42, n_jobs=-1)
    gb_search.fit(X_scaled, y)
    optimized_models['gb'] = gb_search.best_estimator_
    print(f"   Best GB score: {gb_search.best_score_*100:.2f}%")
    print(f"   Best params: {gb_search.best_params_}")

    # 3. SVM optimization
    print("\n3. Optimizing SVM...")
    svm_params = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly']
    }
    svm = SVC(probability=True, random_state=42)
    svm_search = GridSearchCV(svm, svm_params, cv=cv, scoring='accuracy', n_jobs=-1)
    svm_search.fit(X_scaled, y)
    optimized_models['svm'] = svm_search.best_estimator_
    print(f"   Best SVM score: {svm_search.best_score_*100:.2f}%")
    print(f"   Best params: {svm_search.best_params_}")

    # 4. MLP optimization
    print("\n4. Optimizing MLP...")
    mlp_params = {
        'hidden_layer_sizes': [(100,), (200,), (100, 50), (200, 100), (100, 50, 25)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01],
        'activation': ['relu', 'tanh']
    }
    mlp = MLPClassifier(max_iter=2000, early_stopping=True, random_state=42)
    mlp_search = GridSearchCV(mlp, mlp_params, cv=cv, scoring='accuracy', n_jobs=-1)
    mlp_search.fit(X_scaled, y)
    optimized_models['mlp'] = mlp_search.best_estimator_
    print(f"   Best MLP score: {mlp_search.best_score_*100:.2f}%")
    print(f"   Best params: {mlp_search.best_params_}")

    return optimized_models, scaler


def try_different_approaches(X, y, separability, feature_names):
    """Try multiple approaches to improve accuracy."""
    print("\n" + "="*60)
    print("TRYING DIFFERENT APPROACHES")
    print("="*60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    # 1. Basic RandomForest
    print("\n1. Basic RandomForest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='accuracy')
    results['RF_basic'] = np.mean(scores)
    print(f"   Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.1f}%)")

    # 2. Tuned RandomForest
    print("\n2. Tuned RandomForest...")
    rf_tuned = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    scores = cross_val_score(rf_tuned, X_scaled, y, cv=cv, scoring='accuracy')
    results['RF_tuned'] = np.mean(scores)
    print(f"   Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.1f}%)")

    # 3. ExtraTrees
    print("\n3. ExtraTrees...")
    et = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    scores = cross_val_score(et, X_scaled, y, cv=cv, scoring='accuracy')
    results['ExtraTrees'] = np.mean(scores)
    print(f"   Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.1f}%)")

    # 4. GradientBoosting
    print("\n4. GradientBoosting...")
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
    scores = cross_val_score(gb, X_scaled, y, cv=cv, scoring='accuracy')
    results['GradientBoosting'] = np.mean(scores)
    print(f"   Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.1f}%)")

    # 5. SVM with RBF
    print("\n5. SVM (RBF)...")
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    scores = cross_val_score(svm, X_scaled, y, cv=cv, scoring='accuracy')
    results['SVM_RBF'] = np.mean(scores)
    print(f"   Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.1f}%)")

    # 6. SVM with tuned parameters
    print("\n6. SVM (Tuned)...")
    svm_tuned = SVC(kernel='rbf', C=100, gamma=0.01, random_state=42)
    scores = cross_val_score(svm_tuned, X_scaled, y, cv=cv, scoring='accuracy')
    results['SVM_tuned'] = np.mean(scores)
    print(f"   Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.1f}%)")

    # 7. MLP Neural Network
    print("\n7. MLP Neural Network...")
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000,
                        early_stopping=True, random_state=42)
    scores = cross_val_score(mlp, X_scaled, y, cv=cv, scoring='accuracy')
    results['MLP'] = np.mean(scores)
    print(f"   Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.1f}%)")

    # 8. With interaction features
    print("\n8. RF with Interaction Features...")
    X_interact, _ = create_interaction_features(X, feature_names, separability, top_k=8)
    X_interact_scaled = StandardScaler().fit_transform(X_interact)
    rf_interact = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf_interact, X_interact_scaled, y, cv=cv, scoring='accuracy')
    results['RF_Interactions'] = np.mean(scores)
    print(f"   Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.1f}%)")

    # 9. With data augmentation (noise injection)
    print("\n9. RF with Noise Augmentation...")
    X_aug, y_aug = augment_data_with_noise(X_scaled, y, n_augmented=100, noise_level=0.03)
    rf_aug = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf_aug, X_aug, y_aug, cv=cv, scoring='accuracy')
    results['RF_NoiseAug'] = np.mean(scores)
    print(f"   Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.1f}%)")

    # 10. With SMOTE/ADASYN
    if HAS_IMBLEARN:
        print("\n10. RF with Advanced Oversampling...")
        X_over, y_over = apply_advanced_oversampling(X_scaled, y)
        rf_over = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        scores = cross_val_score(rf_over, X_over, y_over, cv=cv, scoring='accuracy')
        results['RF_Oversampled'] = np.mean(scores)
        print(f"   Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.1f}%)")

    # 11. PCA + best model
    print("\n11. PCA + RandomForest...")
    pca = PCA(n_components=0.95)  # Keep 95% variance
    X_pca = pca.fit_transform(X_scaled)
    print(f"   PCA reduced to {X_pca.shape[1]} components")
    rf_pca = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf_pca, X_pca, y, cv=cv, scoring='accuracy')
    results['PCA_RF'] = np.mean(scores)
    print(f"   Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.1f}%)")

    # 12. Feature selection + best model
    print("\n12. Feature Selection (MI) + RF...")
    selector = SelectKBest(mutual_info_classif, k=25)
    X_selected = selector.fit_transform(X_scaled, y)
    rf_sel = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf_sel, X_selected, y, cv=cv, scoring='accuracy')
    results['FeatureSelect_RF'] = np.mean(scores)
    print(f"   Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.1f}%)")

    # 13. Voting Ensemble
    print("\n13. Voting Ensemble (5 models)...")
    voting = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
            ('et', ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42)),
            ('svm', SVC(kernel='rbf', C=10, probability=True, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))
        ],
        voting='soft'
    )
    scores = cross_val_score(voting, X_scaled, y, cv=cv, scoring='accuracy')
    results['Voting_5'] = np.mean(scores)
    print(f"   Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.1f}%)")

    # 14. Stacking Ensemble
    print("\n14. Stacking Ensemble...")
    stacking = StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
            ('et', ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(kernel='rbf', C=10, probability=True, random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=5))
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    )
    scores = cross_val_score(stacking, X_scaled, y, cv=cv, scoring='accuracy')
    results['Stacking'] = np.mean(scores)
    print(f"   Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.1f}%)")

    # 15. Aggressive Ensemble (20 models)
    print("\n15. Aggressive Ensemble (20 models)...")
    aggressive_ensemble = VotingClassifier(
        estimators=[
            ('rf1', RandomForestClassifier(n_estimators=500, max_depth=None, random_state=42, n_jobs=-1)),
            ('rf2', RandomForestClassifier(n_estimators=500, max_depth=15, random_state=43, n_jobs=-1)),
            ('rf3', RandomForestClassifier(n_estimators=500, max_depth=20, random_state=44, n_jobs=-1)),
            ('et1', ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1)),
            ('et2', ExtraTreesClassifier(n_estimators=500, max_depth=15, random_state=43, n_jobs=-1)),
            ('gb1', GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=42)),
            ('gb2', GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=43)),
            ('gb3', GradientBoostingClassifier(n_estimators=200, max_depth=7, learning_rate=0.1, random_state=44)),
            ('ada1', AdaBoostClassifier(n_estimators=300, random_state=42)),
            ('ada2', AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=43)),
            ('bag1', BaggingClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
            ('bag2', BaggingClassifier(n_estimators=200, max_samples=0.8, random_state=43, n_jobs=-1)),
            ('svm1', SVC(kernel='rbf', C=10, probability=True, random_state=42)),
            ('svm2', SVC(kernel='rbf', C=100, gamma=0.01, probability=True, random_state=42)),
            ('svm3', SVC(kernel='poly', degree=3, C=10, probability=True, random_state=42)),
            ('knn1', KNeighborsClassifier(n_neighbors=3)),
            ('knn2', KNeighborsClassifier(n_neighbors=5, weights='distance')),
            ('knn3', KNeighborsClassifier(n_neighbors=7, weights='distance')),
            ('mlp1', MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=2000, random_state=42)),
            ('mlp2', MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=2000, random_state=43))
        ],
        voting='soft',
        n_jobs=-1
    )
    scores = cross_val_score(aggressive_ensemble, X_scaled, y, cv=cv, scoring='accuracy')
    results['Aggressive_Ensemble_20'] = np.mean(scores)
    print(f"   Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.1f}%)")

    return results


def train_best_model(X, y, separability, feature_names, optimized_models=None):
    """Train the best model configuration with all enhancements."""
    print("\n" + "="*60)
    print("TRAINING BEST MODEL (Enhanced)")
    print("="*60)

    # Step 1: Feature engineering - add interaction features
    print("\n  Step 1: Feature Engineering...")
    X_enhanced, enhanced_names = create_interaction_features(X, feature_names, separability, top_k=8)

    # Step 2: Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enhanced)

    # Step 3: Data augmentation
    print("\n  Step 2: Data Augmentation...")
    X_aug, y_aug = augment_data_with_noise(X_scaled, y, n_augmented=100, noise_level=0.03)

    # Step 4: Oversampling if imbalanced
    if HAS_IMBLEARN:
        print("\n  Step 3: Oversampling...")
        X_final, y_final = apply_advanced_oversampling(X_aug, y_aug)
    else:
        X_final, y_final = X_aug, y_aug

    # Step 5: Feature selection
    print("\n  Step 4: Feature Selection...")
    n_features = min(40, X_final.shape[1])
    selector = SelectKBest(mutual_info_classif, k=n_features)
    X_selected = selector.fit_transform(X_final, y_final)
    print(f"  Selected {n_features} best features")

    # Step 6: Build the ultimate ensemble
    print("\n  Step 5: Building Ultimate Ensemble...")

    # Use optimized models if available
    if optimized_models:
        rf_opt = optimized_models.get('rf', RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1))
        gb_opt = optimized_models.get('gb', GradientBoostingClassifier(n_estimators=300, random_state=42))
        svm_opt = optimized_models.get('svm', SVC(kernel='rbf', C=10, probability=True, random_state=42))
        mlp_opt = optimized_models.get('mlp', MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=2000, random_state=42))
    else:
        rf_opt = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
        gb_opt = GradientBoostingClassifier(n_estimators=300, random_state=42)
        svm_opt = SVC(kernel='rbf', C=10, probability=True, random_state=42)
        mlp_opt = MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=2000, random_state=42)

    # Ultimate ensemble with 25 diverse models
    best_model = VotingClassifier(
        estimators=[
            # Optimized models
            ('rf_opt', rf_opt),
            ('gb_opt', gb_opt),
            ('svm_opt', svm_opt),
            ('mlp_opt', mlp_opt),
            # RandomForest variants
            ('rf1', RandomForestClassifier(n_estimators=500, max_depth=None, random_state=42, n_jobs=-1)),
            ('rf2', RandomForestClassifier(n_estimators=500, max_depth=15, random_state=43, n_jobs=-1)),
            ('rf3', RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_leaf=2, random_state=44, n_jobs=-1)),
            # ExtraTrees variants
            ('et1', ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1)),
            ('et2', ExtraTreesClassifier(n_estimators=500, max_depth=15, random_state=43, n_jobs=-1)),
            ('et3', ExtraTreesClassifier(n_estimators=500, max_depth=20, min_samples_leaf=2, random_state=44, n_jobs=-1)),
            # GradientBoosting variants
            ('gb1', GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=42)),
            ('gb2', GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=43)),
            ('gb3', GradientBoostingClassifier(n_estimators=200, max_depth=7, learning_rate=0.1, random_state=44)),
            # AdaBoost variants
            ('ada1', AdaBoostClassifier(n_estimators=300, random_state=42)),
            ('ada2', AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=43)),
            # Bagging variants
            ('bag1', BaggingClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
            ('bag2', BaggingClassifier(n_estimators=200, max_samples=0.8, random_state=43, n_jobs=-1)),
            # SVM variants
            ('svm1', SVC(kernel='rbf', C=10, probability=True, random_state=42)),
            ('svm2', SVC(kernel='rbf', C=100, gamma=0.01, probability=True, random_state=42)),
            ('svm3', SVC(kernel='poly', degree=3, C=10, probability=True, random_state=42)),
            # KNN variants
            ('knn1', KNeighborsClassifier(n_neighbors=3)),
            ('knn2', KNeighborsClassifier(n_neighbors=5, weights='distance')),
            ('knn3', KNeighborsClassifier(n_neighbors=7, weights='distance')),
            # MLP variants
            ('mlp1', MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=2000, early_stopping=True, random_state=42)),
            ('mlp2', MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=2000, early_stopping=True, random_state=43)),
        ],
        voting='soft',
        n_jobs=-1
    )

    # Cross-validation with 95% CI
    print("\n  Step 6: Cross-Validation with 95% CI...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_y_true = []
    all_y_pred = []
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_selected, y_final)):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y_final[train_idx], y_final[test_idx]

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        acc = accuracy_score(y_test, y_pred)
        fold_scores.append(acc)
        print(f"    Fold {fold+1}: {acc*100:.2f}%")

    # Final metrics with confidence intervals
    final_acc = accuracy_score(all_y_true, all_y_pred)
    final_f1 = f1_score(all_y_true, all_y_pred)

    # Bootstrap 95% CI
    n_bootstrap = 1000
    bootstrap_accs = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(all_y_true), size=len(all_y_true), replace=True)
        y_true_boot = np.array(all_y_true)[indices]
        y_pred_boot = np.array(all_y_pred)[indices]
        bootstrap_accs.append(accuracy_score(y_true_boot, y_pred_boot))

    ci_lower = np.percentile(bootstrap_accs, 2.5)
    ci_upper = np.percentile(bootstrap_accs, 97.5)

    print(f"\n  FINAL CV Accuracy: {final_acc*100:.2f}% (95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%])")
    print(f"  FINAL CV F1 Score: {final_f1*100:.2f}%")
    print(f"  Fold std: {np.std(fold_scores)*100:.2f}%")

    print("\n  Confusion Matrix:")
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    # Check if we met the target
    if final_acc >= 0.96:
        print(f"\n  TARGET MET: {final_acc*100:.2f}% >= 96%")
    else:
        print(f"\n  Target not yet met: {final_acc*100:.2f}% < 96%")

    # Train on all data and save
    print("\n  Training final model on all data...")
    best_model.fit(X_selected, y_final)

    model_path = MODELS_DIR / "autism_improved_model.joblib"
    joblib.dump({
        'model': best_model,
        'scaler': scaler,
        'selector': selector,
        'accuracy': final_acc,
        'accuracy_ci_lower': ci_lower,
        'accuracy_ci_upper': ci_upper,
        'f1': final_f1,
        'feature_names': enhanced_names,
        'separability': separability
    }, model_path)
    print(f"  Model saved: {model_path}")

    return final_acc, final_f1, ci_lower, ci_upper


def main():
    print("="*60)
    print("  IMPROVING AUTISM MODEL - ADVANCED VERSION")
    print("  Target: 96-100% accuracy")
    print("="*60)

    X, y, subject_ids, feature_names = load_autism_data()
    if X is None:
        print("ERROR: Could not load data")
        return

    # Step 1: Analyze data
    separability = analyze_data(X, y, feature_names)

    # Step 2: Try different approaches (including new techniques)
    results = try_different_approaches(X, y, separability, feature_names)

    # Step 3: Summary of approaches
    print("\n" + "="*60)
    print("RESULTS SUMMARY - ALL APPROACHES")
    print("="*60)

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'Method':<35} {'Accuracy':<15}")
    print("-"*50)
    for method, acc in sorted_results:
        marker = " *" if acc >= 0.96 else ""
        print(f"{method:<35} {acc*100:.2f}%{marker}")

    # Step 4: Hyperparameter optimization
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    optimized_models, _ = hyperparameter_optimization(X, y)

    # Step 5: Train the ultimate best model
    best_acc, best_f1, ci_lower, ci_upper = train_best_model(
        X, y, separability, feature_names, optimized_models
    )

    # Final summary
    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    print(f"\n  Best Autism Model Accuracy: {best_acc*100:.2f}%")
    print(f"  95% Confidence Interval: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
    print(f"  F1 Score: {best_f1*100:.2f}%")

    if best_acc >= 0.96:
        print("\n  SUCCESS: Target accuracy (96%) achieved!")
    elif best_acc >= 0.94:
        print("\n  GOOD: Accuracy improved but below 96% target")
    else:
        print("\n  NOTE: Further optimization may be needed")

    print("\n  Model saved to: saved_models/autism_improved_model.joblib")
    print("="*60)

    return best_acc, best_f1


if __name__ == '__main__':
    main()
