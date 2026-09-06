#!/usr/bin/env python3
"""
NeuroMCP-Agent Training Script
Trains Ultra Stacking Ensemble for EEG-based neurological disease detection.

Usage:
    python scripts/train.py --disease epilepsy --data_path data/epilepsy/
    python scripts/train.py --disease all --data_path data/

Author: Praveen Asthana
"""

import os
import sys
import argparse
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import (
    ExtraTreesClassifier, RandomForestClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib

# Optional imports
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Using fallback.")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Using fallback.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Supported diseases
DISEASES = [
    'epilepsy', 'parkinson', 'alzheimer', 'schizophrenia',
    'autism', 'stress', 'depression'
]

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class EEGFeatureExtractor:
    """Extract 47 features from EEG signals."""

    def __init__(self, sampling_rate=256):
        self.sampling_rate = sampling_rate
        self.feature_names = self._get_feature_names()

    def _get_feature_names(self):
        """Get list of 47 feature names."""
        statistical = [
            'mean', 'std', 'var', 'min', 'max', 'range', 'median',
            'skewness', 'kurtosis', 'rms', 'iqr', 'mad', 'cv',
            'peak_to_peak', 'crest_factor'
        ]
        spectral = [
            'delta_power', 'theta_power', 'alpha_power', 'beta_power',
            'gamma_power', 'total_power', 'delta_ratio', 'theta_ratio',
            'alpha_ratio', 'beta_ratio', 'gamma_ratio', 'theta_beta_ratio',
            'alpha_theta_ratio', 'spectral_entropy', 'spectral_edge_freq',
            'peak_freq', 'mean_freq', 'bandwidth'
        ]
        temporal = [
            'zero_crossings', 'line_length', 'hjorth_activity',
            'hjorth_mobility', 'hjorth_complexity', 'autocorr_lag1',
            'autocorr_lag2', 'diff_entropy', 'log_energy'
        ]
        nonlinear = [
            'sample_entropy', 'approx_entropy', 'hurst_exp',
            'lyapunov_exp', 'fractal_dim'
        ]
        return statistical + spectral + temporal + nonlinear

    def extract_features(self, signal):
        """Extract 47 features from a single EEG signal."""
        features = {}

        # Statistical features (15)
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['var'] = np.var(signal)
        features['min'] = np.min(signal)
        features['max'] = np.max(signal)
        features['range'] = np.ptp(signal)
        features['median'] = np.median(signal)
        features['skewness'] = self._skewness(signal)
        features['kurtosis'] = self._kurtosis(signal)
        features['rms'] = np.sqrt(np.mean(signal**2))
        features['iqr'] = np.percentile(signal, 75) - np.percentile(signal, 25)
        features['mad'] = np.median(np.abs(signal - np.median(signal)))
        features['cv'] = np.std(signal) / (np.mean(signal) + 1e-10)
        features['peak_to_peak'] = np.max(signal) - np.min(signal)
        features['crest_factor'] = np.max(np.abs(signal)) / (features['rms'] + 1e-10)

        # Spectral features (18)
        freqs, psd = self._compute_psd(signal)
        features['delta_power'] = self._bandpower(freqs, psd, 0.5, 4)
        features['theta_power'] = self._bandpower(freqs, psd, 4, 8)
        features['alpha_power'] = self._bandpower(freqs, psd, 8, 13)
        features['beta_power'] = self._bandpower(freqs, psd, 13, 30)
        features['gamma_power'] = self._bandpower(freqs, psd, 30, 45)
        features['total_power'] = np.sum(psd)
        total = features['total_power'] + 1e-10
        features['delta_ratio'] = features['delta_power'] / total
        features['theta_ratio'] = features['theta_power'] / total
        features['alpha_ratio'] = features['alpha_power'] / total
        features['beta_ratio'] = features['beta_power'] / total
        features['gamma_ratio'] = features['gamma_power'] / total
        features['theta_beta_ratio'] = features['theta_power'] / (features['beta_power'] + 1e-10)
        features['alpha_theta_ratio'] = features['alpha_power'] / (features['theta_power'] + 1e-10)
        features['spectral_entropy'] = self._spectral_entropy(psd)
        features['spectral_edge_freq'] = self._spectral_edge_freq(freqs, psd)
        features['peak_freq'] = freqs[np.argmax(psd)] if len(psd) > 0 else 0
        features['mean_freq'] = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
        features['bandwidth'] = np.sqrt(np.sum(((freqs - features['mean_freq'])**2) * psd) / (np.sum(psd) + 1e-10))

        # Temporal features (9)
        features['zero_crossings'] = np.sum(np.diff(np.sign(signal)) != 0)
        features['line_length'] = np.sum(np.abs(np.diff(signal)))
        features['hjorth_activity'] = np.var(signal)
        diff1 = np.diff(signal)
        features['hjorth_mobility'] = np.sqrt(np.var(diff1) / (np.var(signal) + 1e-10))
        diff2 = np.diff(diff1)
        mob1 = np.sqrt(np.var(diff1) / (np.var(signal) + 1e-10))
        mob2 = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10))
        features['hjorth_complexity'] = mob2 / (mob1 + 1e-10)
        features['autocorr_lag1'] = np.corrcoef(signal[:-1], signal[1:])[0, 1] if len(signal) > 1 else 0
        features['autocorr_lag2'] = np.corrcoef(signal[:-2], signal[2:])[0, 1] if len(signal) > 2 else 0
        features['diff_entropy'] = 0.5 * np.log2(2 * np.pi * np.e * (np.var(signal) + 1e-10))
        features['log_energy'] = np.sum(np.log(signal**2 + 1e-10))

        # Nonlinear features (5) - simplified versions
        features['sample_entropy'] = self._sample_entropy(signal)
        features['approx_entropy'] = self._approx_entropy(signal)
        features['hurst_exp'] = self._hurst_exponent(signal)
        features['lyapunov_exp'] = self._lyapunov_exponent(signal)
        features['fractal_dim'] = self._fractal_dimension(signal)

        return np.array([features[name] for name in self.feature_names])

    def _skewness(self, x):
        n = len(x)
        mean = np.mean(x)
        std = np.std(x) + 1e-10
        return np.sum(((x - mean) / std) ** 3) / n

    def _kurtosis(self, x):
        n = len(x)
        mean = np.mean(x)
        std = np.std(x) + 1e-10
        return np.sum(((x - mean) / std) ** 4) / n - 3

    def _compute_psd(self, signal):
        """Compute power spectral density using Welch's method."""
        from scipy.signal import welch
        freqs, psd = welch(signal, fs=self.sampling_rate, nperseg=min(256, len(signal)))
        return freqs, psd

    def _bandpower(self, freqs, psd, fmin, fmax):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return np.trapz(psd[idx], freqs[idx]) if np.any(idx) else 0

    def _spectral_entropy(self, psd):
        psd_norm = psd / (np.sum(psd) + 1e-10)
        return -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

    def _spectral_edge_freq(self, freqs, psd, threshold=0.95):
        cumsum = np.cumsum(psd)
        total = cumsum[-1]
        idx = np.where(cumsum >= threshold * total)[0]
        return freqs[idx[0]] if len(idx) > 0 else freqs[-1]

    def _sample_entropy(self, x, m=2, r=0.2):
        """Simplified sample entropy calculation."""
        n = len(x)
        if n < m + 1:
            return 0
        r *= np.std(x)

        def count_matches(template_len):
            count = 0
            for i in range(n - template_len):
                for j in range(i + 1, n - template_len):
                    if np.max(np.abs(x[i:i+template_len] - x[j:j+template_len])) < r:
                        count += 1
            return count

        A = count_matches(m + 1)
        B = count_matches(m)
        return -np.log(A / (B + 1e-10) + 1e-10) if B > 0 else 0

    def _approx_entropy(self, x, m=2, r=0.2):
        """Simplified approximate entropy."""
        return self._sample_entropy(x, m, r) * 0.9  # Approximation

    def _hurst_exponent(self, x):
        """Simplified Hurst exponent using R/S analysis."""
        n = len(x)
        if n < 20:
            return 0.5

        max_k = min(n // 2, 100)
        rs = []
        ns = []

        for k in range(10, max_k, 10):
            rs_k = []
            for i in range(0, n - k, k):
                segment = x[i:i+k]
                mean = np.mean(segment)
                cumdev = np.cumsum(segment - mean)
                r = np.max(cumdev) - np.min(cumdev)
                s = np.std(segment) + 1e-10
                rs_k.append(r / s)
            if rs_k:
                rs.append(np.mean(rs_k))
                ns.append(k)

        if len(rs) > 1:
            log_rs = np.log(rs)
            log_ns = np.log(ns)
            slope, _ = np.polyfit(log_ns, log_rs, 1)
            return slope
        return 0.5

    def _lyapunov_exponent(self, x):
        """Simplified Lyapunov exponent estimation."""
        n = len(x)
        if n < 50:
            return 0

        diffs = []
        for i in range(n - 10):
            # Find nearest neighbor
            min_dist = float('inf')
            min_j = -1
            for j in range(n - 10):
                if abs(i - j) > 5:
                    dist = abs(x[i] - x[j])
                    if dist < min_dist and dist > 1e-10:
                        min_dist = dist
                        min_j = j

            if min_j >= 0:
                new_dist = abs(x[i + 10] - x[min_j + 10])
                if new_dist > 1e-10:
                    diffs.append(np.log(new_dist / min_dist))

        return np.mean(diffs) if diffs else 0

    def _fractal_dimension(self, x):
        """Higuchi fractal dimension."""
        n = len(x)
        kmax = min(10, n // 4)

        L = []
        for k in range(1, kmax + 1):
            Lk = []
            for m in range(1, k + 1):
                idx = np.arange(m - 1, n, k)
                Lmk = np.sum(np.abs(np.diff(x[idx]))) * (n - 1) / (k * len(idx) * k)
                Lk.append(Lmk)
            L.append(np.mean(Lk))

        if len(L) > 1:
            log_L = np.log(np.array(L) + 1e-10)
            log_k = np.log(np.arange(1, kmax + 1))
            slope, _ = np.polyfit(log_k, log_L, 1)
            return -slope
        return 1.5


class UltraStackingEnsemble:
    """Ultra Stacking Ensemble with 15 base classifiers and MLP meta-learner."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_classifiers = self._create_base_classifiers()
        self.meta_learner = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            random_state=random_state
        )
        self.feature_selector = SelectKBest(mutual_info_classif, k=40)
        self.scaler = RobustScaler()
        self.is_fitted = False

    def _create_base_classifiers(self):
        """Create 15 diverse base classifiers."""
        classifiers = {
            'et1': ExtraTreesClassifier(n_estimators=500, max_depth=None,
                                        min_samples_split=2, random_state=self.random_state),
            'et2': ExtraTreesClassifier(n_estimators=300, max_depth=20,
                                        min_samples_split=5, random_state=self.random_state+1),
            'rf1': RandomForestClassifier(n_estimators=500, max_depth=None,
                                          random_state=self.random_state+2),
            'rf2': RandomForestClassifier(n_estimators=300, max_depth=20,
                                          random_state=self.random_state+3),
            'gb1': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                              max_depth=5, random_state=self.random_state+4),
            'gb2': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                              max_depth=3, random_state=self.random_state+5),
            'ada1': AdaBoostClassifier(n_estimators=100, learning_rate=1.0,
                                       random_state=self.random_state+10),
            'ada2': AdaBoostClassifier(n_estimators=50, learning_rate=0.5,
                                       random_state=self.random_state+11),
            'mlp1': MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500,
                                  random_state=self.random_state+12),
            'mlp2': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                  random_state=self.random_state+13),
            'svm': SVC(C=1.0, kernel='rbf', probability=True,
                       random_state=self.random_state+14),
        }

        # Add XGBoost if available
        if HAS_XGBOOST:
            classifiers['xgb1'] = XGBClassifier(n_estimators=200, learning_rate=0.1,
                                                max_depth=6, random_state=self.random_state+6,
                                                use_label_encoder=False, eval_metric='logloss')
            classifiers['xgb2'] = XGBClassifier(n_estimators=100, learning_rate=0.05,
                                                max_depth=4, random_state=self.random_state+7,
                                                use_label_encoder=False, eval_metric='logloss')
        else:
            classifiers['xgb1'] = GradientBoostingClassifier(n_estimators=200, random_state=self.random_state+6)
            classifiers['xgb2'] = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state+7)

        # Add LightGBM if available
        if HAS_LIGHTGBM:
            classifiers['lgb1'] = LGBMClassifier(n_estimators=200, learning_rate=0.1,
                                                 num_leaves=31, random_state=self.random_state+8,
                                                 verbose=-1)
            classifiers['lgb2'] = LGBMClassifier(n_estimators=100, learning_rate=0.05,
                                                 num_leaves=15, random_state=self.random_state+9,
                                                 verbose=-1)
        else:
            classifiers['lgb1'] = GradientBoostingClassifier(n_estimators=200, random_state=self.random_state+8)
            classifiers['lgb2'] = GradientBoostingClassifier(n_estimators=100, random_state=self.random_state+9)

        return classifiers

    def fit(self, X, y):
        """Fit the stacking ensemble."""
        logger.info("Fitting Ultra Stacking Ensemble...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Select features
        if X_scaled.shape[1] > 40:
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
        else:
            X_selected = X_scaled

        # Fit base classifiers
        meta_features = []
        for name, clf in self.base_classifiers.items():
            logger.info(f"  Training {name}...")
            clf.fit(X_selected, y)
            proba = clf.predict_proba(X_selected)
            meta_features.append(proba)

        # Stack predictions for meta-learner
        meta_X = np.hstack(meta_features)

        # Fit meta-learner
        logger.info("  Training meta-learner...")
        self.meta_learner.fit(meta_X, y)

        self.is_fitted = True
        logger.info("Ensemble training complete.")
        return self

    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        if hasattr(self.feature_selector, 'get_support'):
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled

        # Get base predictions
        meta_features = []
        for name, clf in self.base_classifiers.items():
            proba = clf.predict_proba(X_selected)
            meta_features.append(proba)

        meta_X = np.hstack(meta_features)
        return self.meta_learner.predict_proba(meta_X)

    def save(self, path):
        """Save the model to disk."""
        joblib.dump(self, path)
        logger.info(f"Model saved to {path}")

    @staticmethod
    def load(path):
        """Load model from disk."""
        return joblib.load(path)


def generate_synthetic_data(disease, n_samples=1000, n_channels=19, signal_length=512):
    """Generate synthetic EEG data for testing."""
    logger.info(f"Generating synthetic data for {disease}...")

    np.random.seed(RANDOM_SEED)
    extractor = EEGFeatureExtractor(sampling_rate=256)

    X = []
    y = []

    # Disease-specific signal characteristics
    disease_params = {
        'epilepsy': {'spike_prob': 0.3, 'noise_level': 0.5},
        'parkinson': {'tremor_freq': 5, 'noise_level': 0.3},
        'alzheimer': {'slow_wave': True, 'noise_level': 0.4},
        'schizophrenia': {'gamma_reduction': 0.5, 'noise_level': 0.4},
        'autism': {'connectivity_diff': True, 'noise_level': 0.3},
        'stress': {'beta_increase': 1.5, 'noise_level': 0.5},
        'depression': {'alpha_asymmetry': True, 'noise_level': 0.4},
    }

    params = disease_params.get(disease, {'noise_level': 0.3})

    for i in range(n_samples):
        # Generate healthy or disease signal
        label = i % 2  # Alternate between healthy (0) and disease (1)

        # Base signal
        t = np.linspace(0, 2, signal_length)
        signal = np.zeros(signal_length)

        # Add frequency components
        signal += 0.5 * np.sin(2 * np.pi * 2 * t)  # Delta
        signal += 0.3 * np.sin(2 * np.pi * 6 * t)  # Theta
        signal += 0.4 * np.sin(2 * np.pi * 10 * t)  # Alpha
        signal += 0.2 * np.sin(2 * np.pi * 20 * t)  # Beta
        signal += 0.1 * np.sin(2 * np.pi * 35 * t)  # Gamma

        # Add disease-specific patterns
        if label == 1:  # Disease
            if disease == 'epilepsy' and np.random.random() < params['spike_prob']:
                spike_loc = np.random.randint(100, signal_length - 100)
                signal[spike_loc:spike_loc+20] += 3 * np.exp(-np.linspace(0, 3, 20))
            elif disease == 'parkinson':
                signal += 0.5 * np.sin(2 * np.pi * params['tremor_freq'] * t)
            elif disease == 'alzheimer':
                signal += 0.3 * np.sin(2 * np.pi * 1 * t)  # More slow waves
            elif disease == 'schizophrenia':
                signal -= 0.05 * np.sin(2 * np.pi * 35 * t)  # Reduced gamma
            elif disease == 'stress':
                signal += 0.3 * np.sin(2 * np.pi * 20 * t)  # More beta
            elif disease == 'depression':
                signal += 0.2 * np.sin(2 * np.pi * 10 * t + np.pi/4)  # Alpha asymmetry

        # Add noise
        signal += params['noise_level'] * np.random.randn(signal_length)

        # Extract features from average of channels (simulated)
        features = extractor.extract_features(signal)

        # Add some variation across channels
        channel_features = []
        for ch in range(n_channels):
            ch_signal = signal + 0.1 * np.random.randn(signal_length)
            ch_features = extractor.extract_features(ch_signal)
            channel_features.append(ch_features)

        # Average features across channels
        avg_features = np.mean(channel_features, axis=0)

        X.append(avg_features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    logger.info(f"Generated {n_samples} samples with {X.shape[1]} features")
    return X, y


def train_disease_model(disease, data_path=None, output_dir='saved_models', n_folds=5):
    """Train model for a specific disease."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training model for: {disease.upper()}")
    logger.info(f"{'='*60}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load or generate data
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        # Load real data here
        data = np.load(data_path)
        X, y = data['X'], data['y']
    else:
        logger.info("Using synthetic data (no real data provided)")
        X, y = generate_synthetic_data(disease, n_samples=2000)

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    results = {
        'disease': disease,
        'n_samples': len(y),
        'n_features': X.shape[1],
        'n_folds': n_folds,
        'fold_results': [],
        'timestamp': datetime.now().isoformat()
    }

    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\nFold {fold + 1}/{n_folds}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model
        model = UltraStackingEnsemble(random_state=RANDOM_SEED + fold)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        fold_metrics = {
            'fold': fold + 1,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
        }

        results['fold_results'].append(fold_metrics)
        logger.info(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
        logger.info(f"  AUC: {fold_metrics['auc']:.4f}")

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

    # Calculate overall metrics
    results['overall'] = {
        'accuracy': accuracy_score(all_y_true, all_y_pred),
        'accuracy_std': np.std([f['accuracy'] for f in results['fold_results']]),
        'precision': precision_score(all_y_true, all_y_pred, zero_division=0),
        'recall': recall_score(all_y_true, all_y_pred, zero_division=0),
        'f1': f1_score(all_y_true, all_y_pred, zero_division=0),
        'auc': roc_auc_score(all_y_true, all_y_proba),
        'confusion_matrix': confusion_matrix(all_y_true, all_y_pred).tolist(),
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"OVERALL RESULTS for {disease.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {results['overall']['accuracy']:.4f} ± {results['overall']['accuracy_std']:.4f}")
    logger.info(f"Precision: {results['overall']['precision']:.4f}")
    logger.info(f"Recall: {results['overall']['recall']:.4f}")
    logger.info(f"F1-Score: {results['overall']['f1']:.4f}")
    logger.info(f"AUC-ROC: {results['overall']['auc']:.4f}")

    # Train final model on all data
    logger.info("\nTraining final model on all data...")
    final_model = UltraStackingEnsemble(random_state=RANDOM_SEED)
    final_model.fit(X, y)

    # Save model and results
    model_path = os.path.join(output_dir, f'{disease}_model.joblib')
    final_model.save(model_path)

    results_path = os.path.join(output_dir, f'{disease}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train NeuroMCP-Agent models')
    parser.add_argument('--disease', type=str, default='all',
                        choices=DISEASES + ['all'],
                        help='Disease to train model for')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='saved_models',
                        help='Output directory for models')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds')

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("NeuroMCP-Agent Training Script")
    logger.info("="*60)
    logger.info(f"Disease: {args.disease}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"CV folds: {args.n_folds}")
    logger.info("="*60)

    if args.disease == 'all':
        all_results = {}
        for disease in DISEASES:
            results = train_disease_model(
                disease,
                data_path=args.data_path,
                output_dir=args.output_dir,
                n_folds=args.n_folds
            )
            all_results[disease] = results

        # Save summary
        summary_path = os.path.join(args.output_dir, 'all_results_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nAll results saved to {summary_path}")

        # Print summary table
        logger.info("\n" + "="*80)
        logger.info("SUMMARY OF ALL DISEASES")
        logger.info("="*80)
        logger.info(f"{'Disease':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
        logger.info("-"*80)
        for disease, results in all_results.items():
            o = results['overall']
            logger.info(f"{disease:<15} {o['accuracy']:.4f}       {o['precision']:.4f}       "
                       f"{o['recall']:.4f}       {o['f1']:.4f}       {o['auc']:.4f}")
    else:
        train_disease_model(
            args.disease,
            data_path=args.data_path,
            output_dir=args.output_dir,
            n_folds=args.n_folds
        )

    logger.info("\nTraining complete!")


if __name__ == '__main__':
    main()
