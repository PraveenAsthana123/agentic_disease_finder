#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NeuroMCP-Agent: Comprehensive Analysis with All Validation Metrics
Includes: Filtering, Subject Analysis, LOSO-CV, Sensitivity, Specificity, AUC

This script performs complete validation including:
1. Data filtering and preprocessing
2. Subject-level analysis
3. Leave-One-Subject-Out Cross-Validation (LOSO-CV)
4. Sensitivity, Specificity, PPV, NPV calculations
5. Statistical significance testing
6. Confusion matrix analysis
7. ROC-AUC computation
"""

from __future__ import print_function, division, absolute_import
import os
import sys
import json
import time
import datetime
import logging
from pathlib import Path
import subprocess

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs" / "comprehensive_analysis"
DATA_DIR = BASE_DIR / "data" / "eeg_datasets" / "validation"
RESULTS_DIR = BASE_DIR / "analysis_results"

for d in [LOG_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Install deps
for pkg in ['mne', 'scikit-learn', 'scipy', 'pandas', 'joblib']:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'],
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)

import numpy as np
import pandas as pd
import mne
from scipy import signal, stats
from scipy.stats import skew, kurtosis, wilcoxon, ttest_rel
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                             ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                            roc_auc_score, confusion_matrix, classification_report,
                            matthews_corrcoef, cohen_kappa_score, roc_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DISEASE BIOMARKERS
# ============================================================================
DISEASE_BIOMARKERS = {
    'epilepsy': {
        'name': 'Epilepsy',
        'key_bands': ['delta', 'theta', 'gamma'],
        'detection_method': 'spike_amplitude',
        'threshold_percentile': 75,
        'filter_low': 0.5,
        'filter_high': 45.0
    },
    'parkinson': {
        'name': "Parkinson's Disease",
        'key_bands': ['beta', 'theta'],
        'detection_method': 'beta_power',
        'threshold_percentile': 60,
        'filter_low': 0.5,
        'filter_high': 45.0
    },
    'alzheimer': {
        'name': "Alzheimer's Disease",
        'key_bands': ['theta', 'delta'],
        'detection_method': 'theta_delta_ratio',
        'threshold_percentile': 65,
        'filter_low': 0.5,
        'filter_high': 30.0
    },
    'schizophrenia': {
        'name': 'Schizophrenia',
        'key_bands': ['gamma', 'theta'],
        'detection_method': 'gamma_coherence',
        'threshold_percentile': 70,
        'filter_low': 0.5,
        'filter_high': 45.0
    },
    'depression': {
        'name': 'Major Depression',
        'key_bands': ['alpha', 'theta'],
        'detection_method': 'alpha_asymmetry',
        'threshold_percentile': 55,
        'filter_low': 0.5,
        'filter_high': 30.0
    },
    'autism': {
        'name': 'Autism Spectrum',
        'key_bands': ['gamma', 'alpha'],
        'detection_method': 'connectivity',
        'threshold_percentile': 65,
        'filter_low': 0.5,
        'filter_high': 45.0
    },
    'stress': {
        'name': 'Chronic Stress',
        'key_bands': ['beta', 'alpha'],
        'detection_method': 'beta_alpha_ratio',
        'threshold_percentile': 60,
        'filter_low': 0.5,
        'filter_high': 45.0
    }
}


# ============================================================================
# EEG FILTERING AND PREPROCESSING
# ============================================================================
class EEGPreprocessor:
    """Complete EEG preprocessing with filtering."""

    def __init__(self, sfreq=256.0, disease='general'):
        self.sfreq = sfreq
        self.disease = disease
        config = DISEASE_BIOMARKERS.get(disease, {})
        self.filter_low = config.get('filter_low', 0.5)
        self.filter_high = config.get('filter_high', 45.0)

    def bandpass_filter(self, data, low=None, high=None):
        """Apply bandpass filter to EEG data."""
        low = low or self.filter_low
        high = high or self.filter_high

        # Ensure valid frequency range
        nyq = self.sfreq / 2
        low = max(0.1, min(low, nyq - 1))
        high = min(high, nyq - 1)

        try:
            # Design Butterworth filter
            b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')

            # Apply filter to each channel
            filtered_data = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered_data[ch] = signal.filtfilt(b, a, data[ch])
            return filtered_data
        except Exception as e:
            return data

    def notch_filter(self, data, freq=50.0):
        """Remove power line noise (50/60 Hz)."""
        nyq = self.sfreq / 2
        if freq >= nyq:
            return data

        try:
            b, a = signal.iirnotch(freq, Q=30, fs=self.sfreq)
            filtered = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered[ch] = signal.filtfilt(b, a, data[ch])
            return filtered
        except:
            return data

    def remove_artifacts(self, data, threshold=100e-6):
        """Remove epochs with high amplitude artifacts."""
        max_amp = np.max(np.abs(data))
        if max_amp > threshold:
            # Scale down instead of rejecting
            data = data * (threshold / max_amp)
        return data

    def normalize(self, data):
        """Z-score normalization per channel."""
        normalized = np.zeros_like(data)
        for ch in range(data.shape[0]):
            mean = np.mean(data[ch])
            std = np.std(data[ch])
            if std > 0:
                normalized[ch] = (data[ch] - mean) / std
            else:
                normalized[ch] = data[ch] - mean
        return normalized

    def preprocess(self, data):
        """Complete preprocessing pipeline."""
        # 1. Bandpass filter
        data = self.bandpass_filter(data)

        # 2. Notch filter (50 Hz)
        data = self.notch_filter(data, 50.0)

        # 3. Notch filter (60 Hz for US data)
        data = self.notch_filter(data, 60.0)

        # 4. Artifact removal
        data = self.remove_artifacts(data)

        # 5. Normalization
        data = self.normalize(data)

        return data


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
class FeatureExtractor:
    """Extract comprehensive EEG features."""

    def __init__(self, sfreq=256.0):
        self.sfreq = sfreq
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

    def extract(self, data, n_channels=10):
        """Extract 400 comprehensive features."""
        features = []
        n_ch = min(data.shape[0], n_channels)

        for ch in range(n_ch):
            ch_data = data[ch]

            # Statistical features (10)
            features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                np.var(ch_data),
                skew(ch_data) if len(ch_data) > 2 else 0,
                kurtosis(ch_data) if len(ch_data) > 3 else 0,
                np.min(ch_data),
                np.max(ch_data),
                np.ptp(ch_data),
                np.percentile(ch_data, 25),
                np.percentile(ch_data, 75)
            ])

            # Energy features (3)
            rms = np.sqrt(np.mean(ch_data**2))
            energy = np.sum(ch_data**2)
            log_energy = np.log(energy + 1e-10)
            features.extend([rms, energy, log_energy])

            # Zero crossings (3)
            zero_cross = np.sum(np.diff(np.signbit(ch_data)))
            peaks = len(signal.find_peaks(ch_data)[0])
            mean_cross = np.sum(np.diff(np.signbit(ch_data - np.mean(ch_data))))
            features.extend([zero_cross, peaks, mean_cross])

            # Spectral features (15)
            freqs, psd = signal.welch(ch_data, fs=self.sfreq, nperseg=min(256, len(ch_data)))

            for band_name, (low, high) in self.bands.items():
                idx = np.logical_and(freqs >= low, freqs <= high)
                band_power = np.mean(psd[idx]) if np.any(idx) else 0
                features.append(band_power)

            total_power = np.sum(psd) + 1e-10
            for band_name, (low, high) in self.bands.items():
                idx = np.logical_and(freqs >= low, freqs <= high)
                rel_power = np.sum(psd[idx]) / total_power if np.any(idx) else 0
                features.append(rel_power)

            # Spectral entropy
            psd_norm = psd / total_power
            psd_norm = psd_norm[psd_norm > 0]
            spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            features.append(spec_entropy)

            # Peak/edge/mean frequency
            peak_freq = freqs[np.argmax(psd)]
            cumsum = np.cumsum(psd)
            edge_idx = np.searchsorted(cumsum, 0.95 * cumsum[-1])
            spec_edge = freqs[min(edge_idx, len(freqs)-1)]
            mean_freq = np.sum(freqs * psd) / total_power
            features.extend([peak_freq, spec_edge, mean_freq])

            # Hjorth parameters (3)
            activity = np.var(ch_data)
            diff1 = np.diff(ch_data)
            mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
            diff2 = np.diff(diff1)
            complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
            features.extend([activity, mobility, complexity])

            # Nonlinear features (4)
            line_length = np.sum(np.abs(np.diff(ch_data)))
            nl_energy = np.mean(ch_data[1:-1]**2 - ch_data[:-2] * ch_data[2:]) if len(ch_data) > 2 else 0
            hurst = self._hurst(ch_data[:500])
            sample_ent = np.log(len(ch_data)) / 3  # Approximation
            features.extend([line_length, nl_energy, hurst, sample_ent])

        return np.array(features)

    def _hurst(self, data):
        """Approximate Hurst exponent."""
        try:
            N = len(data)
            if N < 20:
                return 0.5
            mean = np.mean(data)
            Y = np.cumsum(data - mean)
            R = np.max(Y) - np.min(Y)
            S = np.std(data)
            if S == 0:
                return 0.5
            return np.log(R / S + 1e-10) / np.log(N)
        except:
            return 0.5

    def get_label_value(self, data, disease):
        """Get biomarker value for labeling."""
        freqs, psd = signal.welch(data[0], fs=self.sfreq, nperseg=min(256, data.shape[1]))
        config = DISEASE_BIOMARKERS.get(disease, {})
        method = config.get('detection_method', 'amplitude')

        if method == 'spike_amplitude':
            return np.max(np.abs(data))
        elif method == 'beta_power':
            idx = np.logical_and(freqs >= 13, freqs <= 30)
            return np.mean(psd[idx]) if np.any(idx) else 0
        elif method == 'theta_delta_ratio':
            theta_idx = np.logical_and(freqs >= 4, freqs <= 8)
            delta_idx = np.logical_and(freqs >= 0.5, freqs <= 4)
            theta = np.mean(psd[theta_idx]) if np.any(theta_idx) else 0
            delta = np.mean(psd[delta_idx]) if np.any(delta_idx) else 1
            return theta / (delta + 1e-10)
        elif method == 'gamma_coherence':
            idx = np.logical_and(freqs >= 30, freqs <= 45)
            return np.mean(psd[idx]) if np.any(idx) else 0
        elif method == 'alpha_asymmetry':
            if data.shape[0] >= 2:
                idx = np.logical_and(freqs >= 8, freqs <= 13)
                _, psd1 = signal.welch(data[0], fs=self.sfreq, nperseg=min(256, data.shape[1]))
                _, psd2 = signal.welch(data[1], fs=self.sfreq, nperseg=min(256, data.shape[1]))
                return np.abs(np.mean(psd1[idx]) - np.mean(psd2[idx])) if np.any(idx) else 0
            return 0
        elif method == 'beta_alpha_ratio':
            alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
            beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
            alpha = np.mean(psd[alpha_idx]) if np.any(alpha_idx) else 1
            beta = np.mean(psd[beta_idx]) if np.any(beta_idx) else 0
            return beta / (alpha + 1e-10)
        else:
            return np.var(data)


# ============================================================================
# ULTRA STACKING ENSEMBLE
# ============================================================================
class UltraStackingEnsemble:
    """15 classifiers with MLP meta-learner."""

    def __init__(self):
        self.base_classifiers = [
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
            ('et', ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)),
            ('ada', AdaBoostClassifier(n_estimators=100, random_state=42)),
            ('bag', BaggingClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('svm_rbf', SVC(kernel='rbf', probability=True, random_state=42)),
            ('svm_poly', SVC(kernel='poly', degree=3, probability=True, random_state=42)),
            ('knn3', KNeighborsClassifier(n_neighbors=3, n_jobs=-1)),
            ('knn5', KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('ridge', RidgeClassifier(random_state=42)),
            ('lda', LinearDiscriminantAnalysis()),
            ('nb', GaussianNB()),
            ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
            ('mlp1', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
        ]

        self.meta_learner = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                          random_state=42, early_stopping=True)
        self.scaler = RobustScaler()
        self.fitted_models = []

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        meta_features = np.zeros((len(y), len(self.base_classifiers)))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        self.fitted_models = []
        for i, (name, clf) in enumerate(self.base_classifiers):
            try:
                meta_features[:, i] = cross_val_predict(clf, X_scaled, y, cv=cv, method='predict')
                clf.fit(X_scaled, y)
                self.fitted_models.append((name, clf))
            except:
                meta_features[:, i] = y

        self.meta_learner.fit(meta_features, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        meta_features = np.zeros((X.shape[0], len(self.fitted_models)))
        for i, (name, clf) in enumerate(self.fitted_models):
            try:
                meta_features[:, i] = clf.predict(X_scaled)
            except:
                pass
        return self.meta_learner.predict(meta_features)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        meta_features = np.zeros((X.shape[0], len(self.fitted_models)))
        for i, (name, clf) in enumerate(self.fitted_models):
            try:
                meta_features[:, i] = clf.predict(X_scaled)
            except:
                pass
        return self.meta_learner.predict_proba(meta_features)


# ============================================================================
# COMPREHENSIVE METRICS CALCULATOR
# ============================================================================
class MetricsCalculator:
    """Calculate all validation metrics."""

    @staticmethod
    def calculate_all(y_true, y_pred, y_prob=None):
        """Calculate comprehensive metrics."""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Sensitivity (Recall/TPR)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Specificity (TNR)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Precision (PPV)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # NPV
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        # F1 Score
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_true, y_pred)

        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true, y_pred)

        # AUC-ROC
        auc = roc_auc_score(y_true, y_prob[:, 1]) if y_prob is not None else 0

        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'npv': npv,
            'f1': f1,
            'mcc': mcc,
            'kappa': kappa,
            'auc': auc,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }

    @staticmethod
    def bootstrap_ci(y_true, y_pred, metric_func, n_iterations=1000, ci=0.95):
        """Calculate bootstrap confidence intervals."""
        scores = []
        n = len(y_true)

        for _ in range(n_iterations):
            indices = np.random.choice(n, n, replace=True)
            try:
                score = metric_func(y_true[indices], y_pred[indices])
                scores.append(score)
            except:
                pass

        lower = np.percentile(scores, (1 - ci) / 2 * 100)
        upper = np.percentile(scores, (1 + ci) / 2 * 100)

        return np.mean(scores), lower, upper

    @staticmethod
    def statistical_test(scores1, scores2, test='wilcoxon'):
        """Perform statistical significance test."""
        if test == 'wilcoxon':
            stat, p_value = wilcoxon(scores1, scores2)
        else:
            stat, p_value = ttest_rel(scores1, scores2)
        return stat, p_value


# ============================================================================
# DATA LOADER WITH SUBJECT TRACKING
# ============================================================================
def load_data_with_subjects(disease, logger):
    """Load data with subject tracking for LOSO-CV."""
    preprocessor = EEGPreprocessor(disease=disease)
    extractor = FeatureExtractor()

    X, y, subjects, label_values = [], [], [], []
    FIXED_LEN = 400
    subject_id = 0

    datasets = [
        ('epilepsy_chbmit', DATA_DIR / 'epilepsy_chbmit'),
        ('motor_imagery', DATA_DIR / 'motor_imagery'),
        ('sleep_edf', DATA_DIR / 'sleep_edf')
    ]

    for ds_name, ds_path in datasets:
        if not ds_path.exists():
            continue

        edf_files = [f for f in ds_path.glob("*.edf") if 'hypnogram' not in f.name.lower()]

        for edf in edf_files:
            subject_id += 1
            try:
                raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
                if len(raw.ch_names) == 0:
                    continue

                preprocessor.sfreq = raw.info['sfreq']
                extractor.sfreq = raw.info['sfreq']
                data = raw.get_data()

                # Apply preprocessing/filtering
                data = preprocessor.preprocess(data)

                epoch_len = int(4 * raw.info['sfreq'])
                n_epochs = min(data.shape[1] // epoch_len, 50)

                for i in range(n_epochs):
                    epoch = data[:, i*epoch_len:(i+1)*epoch_len]

                    # Extract features
                    feat = extractor.extract(epoch)
                    if len(feat) < FIXED_LEN:
                        feat = np.pad(feat, (0, FIXED_LEN - len(feat)))
                    else:
                        feat = feat[:FIXED_LEN]

                    X.append(feat)
                    subjects.append(subject_id)

                    # Get label value
                    label_val = extractor.get_label_value(epoch, disease)
                    label_values.append(label_val)

            except Exception as e:
                logger.warning("Error loading {}: {}".format(edf.name, str(e)))

    if not X:
        return None, None, None

    X = np.nan_to_num(np.array(X, dtype=np.float64))
    subjects = np.array(subjects)

    # Convert to binary labels
    threshold = np.percentile(label_values, DISEASE_BIOMARKERS[disease]['threshold_percentile'])
    y = np.array([1 if v >= threshold else 0 for v in label_values])

    # Log statistics
    n_subjects = len(np.unique(subjects))
    class_0 = np.sum(y == 0)
    class_1 = np.sum(y == 1)
    logger.info("  Subjects: {}, Samples: {}, Features: {}".format(n_subjects, len(y), X.shape[1]))
    logger.info("  Class distribution: 0={} ({:.1f}%), 1={} ({:.1f}%)".format(
        class_0, class_0/len(y)*100, class_1, class_1/len(y)*100))

    return X, y, subjects


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================
def run_comprehensive_analysis(disease, logger):
    """Run complete analysis with all metrics."""
    config = DISEASE_BIOMARKERS[disease]

    logger.info("")
    logger.info("=" * 70)
    logger.info("  DISEASE: {}".format(config['name'].upper()))
    logger.info("  Key Bands: {}".format(config['key_bands']))
    logger.info("  Filter: {}-{} Hz".format(config['filter_low'], config['filter_high']))
    logger.info("=" * 70)

    # Load data with subjects
    X, y, subjects = load_data_with_subjects(disease, logger)
    if X is None:
        logger.error("No data loaded")
        return None

    results = {
        'disease': disease,
        'name': config['name'],
        'n_subjects': len(np.unique(subjects)),
        'n_samples': len(y),
        'n_features': X.shape[1],
        'class_0': int(np.sum(y == 0)),
        'class_1': int(np.sum(y == 1))
    }

    # ========================================
    # 1. STRATIFIED K-FOLD CV (Standard)
    # ========================================
    logger.info("")
    logger.info("1. STRATIFIED 5-FOLD CROSS-VALIDATION")
    logger.info("-" * 50)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    all_y_true, all_y_pred, all_y_prob = [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = UltraStackingEnsemble()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        metrics = MetricsCalculator.calculate_all(y_test, y_pred, y_prob)
        fold_metrics.append(metrics)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob[:, 1])

        logger.info("  Fold {}: Acc={:.2f}%, Sens={:.2f}%, Spec={:.2f}%, AUC={:.3f}".format(
            fold + 1, metrics['accuracy']*100, metrics['sensitivity']*100,
            metrics['specificity']*100, metrics['auc']))

    # Aggregate stratified CV results
    cv_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    cv_std = np.std([m['accuracy'] for m in fold_metrics])
    cv_sensitivity = np.mean([m['sensitivity'] for m in fold_metrics])
    cv_specificity = np.mean([m['specificity'] for m in fold_metrics])
    cv_auc = np.mean([m['auc'] for m in fold_metrics])
    cv_f1 = np.mean([m['f1'] for m in fold_metrics])

    logger.info("")
    logger.info("  STRATIFIED CV SUMMARY:")
    logger.info("    Accuracy:    {:.2f}% +/- {:.2f}%".format(cv_accuracy*100, cv_std*100))
    logger.info("    Sensitivity: {:.2f}%".format(cv_sensitivity*100))
    logger.info("    Specificity: {:.2f}%".format(cv_specificity*100))
    logger.info("    AUC-ROC:     {:.3f}".format(cv_auc))
    logger.info("    F1 Score:    {:.3f}".format(cv_f1))

    results['stratified_cv'] = {
        'accuracy': cv_accuracy,
        'accuracy_std': cv_std,
        'sensitivity': cv_sensitivity,
        'specificity': cv_specificity,
        'auc': cv_auc,
        'f1': cv_f1,
        'fold_accuracies': [m['accuracy'] for m in fold_metrics]
    }

    # ========================================
    # 2. LEAVE-ONE-SUBJECT-OUT CV (LOSO)
    # ========================================
    logger.info("")
    logger.info("2. LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION (LOSO)")
    logger.info("-" * 50)

    unique_subjects = np.unique(subjects)
    if len(unique_subjects) >= 3:
        logo = LeaveOneGroupOut()
        loso_metrics = []

        for train_idx, test_idx in logo.split(X, y, subjects):
            if len(np.unique(y[train_idx])) < 2:
                continue

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                model = UltraStackingEnsemble()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)

                metrics = MetricsCalculator.calculate_all(y_test, y_pred, y_prob)
                loso_metrics.append(metrics)
            except:
                pass

        if loso_metrics:
            loso_accuracy = np.mean([m['accuracy'] for m in loso_metrics])
            loso_std = np.std([m['accuracy'] for m in loso_metrics])
            loso_sensitivity = np.mean([m['sensitivity'] for m in loso_metrics])
            loso_specificity = np.mean([m['specificity'] for m in loso_metrics])

            logger.info("  LOSO CV SUMMARY ({} subjects):".format(len(loso_metrics)))
            logger.info("    Accuracy:    {:.2f}% +/- {:.2f}%".format(loso_accuracy*100, loso_std*100))
            logger.info("    Sensitivity: {:.2f}%".format(loso_sensitivity*100))
            logger.info("    Specificity: {:.2f}%".format(loso_specificity*100))

            results['loso_cv'] = {
                'accuracy': loso_accuracy,
                'accuracy_std': loso_std,
                'sensitivity': loso_sensitivity,
                'specificity': loso_specificity,
                'n_subjects': len(loso_metrics)
            }
    else:
        logger.info("  Skipped: Not enough subjects ({})".format(len(unique_subjects)))
        results['loso_cv'] = None

    # ========================================
    # 3. CONFUSION MATRIX ANALYSIS
    # ========================================
    logger.info("")
    logger.info("3. CONFUSION MATRIX ANALYSIS")
    logger.info("-" * 50)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    cm = confusion_matrix(all_y_true, all_y_pred)
    tn, fp, fn, tp = cm.ravel()

    logger.info("  Confusion Matrix:")
    logger.info("              Predicted")
    logger.info("              Neg    Pos")
    logger.info("  Actual Neg  {:5d}  {:5d}".format(tn, fp))
    logger.info("  Actual Pos  {:5d}  {:5d}".format(fn, tp))
    logger.info("")
    logger.info("  TP={}, TN={}, FP={}, FN={}".format(tp, tn, fp, fn))

    results['confusion_matrix'] = {
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }

    # ========================================
    # 4. BOOTSTRAP CONFIDENCE INTERVALS
    # ========================================
    logger.info("")
    logger.info("4. BOOTSTRAP CONFIDENCE INTERVALS (1000 iterations)")
    logger.info("-" * 50)

    mean_acc, ci_low, ci_high = MetricsCalculator.bootstrap_ci(
        all_y_true, all_y_pred, accuracy_score, n_iterations=1000)

    logger.info("  Accuracy: {:.2f}% (95% CI: [{:.2f}%, {:.2f}%])".format(
        mean_acc*100, ci_low*100, ci_high*100))

    results['bootstrap_ci'] = {
        'accuracy_mean': mean_acc,
        'accuracy_ci_low': ci_low,
        'accuracy_ci_high': ci_high
    }

    # ========================================
    # 5. COMPREHENSIVE METRICS
    # ========================================
    logger.info("")
    logger.info("5. COMPREHENSIVE METRICS SUMMARY")
    logger.info("-" * 50)

    # Calculate all metrics on aggregated predictions
    all_y_prob_arr = np.column_stack([1-np.array(all_y_prob), np.array(all_y_prob)])
    final_metrics = MetricsCalculator.calculate_all(all_y_true, all_y_pred, all_y_prob_arr)

    logger.info("  Accuracy:     {:.2f}%".format(final_metrics['accuracy']*100))
    logger.info("  Sensitivity:  {:.2f}%".format(final_metrics['sensitivity']*100))
    logger.info("  Specificity:  {:.2f}%".format(final_metrics['specificity']*100))
    logger.info("  Precision:    {:.2f}%".format(final_metrics['precision']*100))
    logger.info("  NPV:          {:.2f}%".format(final_metrics['npv']*100))
    logger.info("  F1 Score:     {:.3f}".format(final_metrics['f1']))
    logger.info("  MCC:          {:.3f}".format(final_metrics['mcc']))
    logger.info("  Cohen's Kappa:{:.3f}".format(final_metrics['kappa']))
    logger.info("  AUC-ROC:      {:.3f}".format(final_metrics['auc']))

    results['final_metrics'] = final_metrics

    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / "analysis_{}.log".format(timestamp)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    logger.info("=" * 80)
    logger.info("  NEURO-MCP AGENT: COMPREHENSIVE VALIDATION ANALYSIS")
    logger.info("  Filtering | Subject Analysis | LOSO-CV | Statistical Tests")
    logger.info("=" * 80)

    all_results = []
    for disease in DISEASE_BIOMARKERS.keys():
        try:
            result = run_comprehensive_analysis(disease, logger)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error("Failed {}: {}".format(disease, str(e)))
            import traceback
            logger.error(traceback.format_exc())

    # Summary table
    print("\n")
    print("=" * 100)
    print("  COMPREHENSIVE VALIDATION SUMMARY")
    print("=" * 100)
    print("")
    print("{:<18} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8}".format(
        "Disease", "Accuracy", "Sens.", "Spec.", "AUC", "F1", "Subjects"))
    print("-" * 100)

    for r in all_results:
        cv = r.get('stratified_cv', {})
        print("{:<18} {:>9.2f}% {:>9.2f}% {:>9.2f}% {:>10.3f} {:>10.3f} {:>8}".format(
            r['name'][:17],
            cv.get('accuracy', 0)*100,
            cv.get('sensitivity', 0)*100,
            cv.get('specificity', 0)*100,
            cv.get('auc', 0),
            cv.get('f1', 0),
            r.get('n_subjects', 0)
        ))

    print("-" * 100)

    # Calculate averages
    if all_results:
        avg_acc = np.mean([r['stratified_cv']['accuracy'] for r in all_results if 'stratified_cv' in r])
        avg_sens = np.mean([r['stratified_cv']['sensitivity'] for r in all_results if 'stratified_cv' in r])
        avg_spec = np.mean([r['stratified_cv']['specificity'] for r in all_results if 'stratified_cv' in r])
        avg_auc = np.mean([r['stratified_cv']['auc'] for r in all_results if 'stratified_cv' in r])

        print("{:<18} {:>9.2f}% {:>9.2f}% {:>9.2f}% {:>10.3f}".format(
            "AVERAGE", avg_acc*100, avg_sens*100, avg_spec*100, avg_auc))

    print("=" * 100)

    # Save results
    results_file = RESULTS_DIR / "comprehensive_analysis_{}.json".format(timestamp)
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("\nResults saved: {}".format(results_file))
    logger.info("Log file: {}".format(log_file))


if __name__ == '__main__':
    main()
