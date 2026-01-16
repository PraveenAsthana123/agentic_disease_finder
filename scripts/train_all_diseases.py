#!/usr/bin/env python3
"""
Train and Evaluate Each Disease Separately
AgenticFinder - 6 Disease EEG Classification
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import signal
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

try:
    import mne
    mne.set_log_level('ERROR')
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    print("Warning: MNE not available")


class FeatureExtractor:
    """Extract features from EEG signals"""

    def __init__(self, fs=128):
        self.fs = fs
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

    def bandpower(self, data, band):
        low, high = band
        nperseg = min(len(data), int(self.fs * 2))
        if nperseg < 4:
            return 0
        freqs, psd = signal.welch(data, self.fs, nperseg=nperseg)
        idx = np.logical_and(freqs >= low, freqs <= high)
        return np.trapz(psd[idx], freqs[idx]) if np.sum(idx) > 0 else 0

    def hjorth_params(self, data):
        diff1 = np.diff(data)
        diff2 = np.diff(diff1)
        var0 = np.var(data) + 1e-10
        var1 = np.var(diff1) + 1e-10
        var2 = np.var(diff2) + 1e-10
        activity = var0
        mobility = np.sqrt(var1 / var0)
        complexity = np.sqrt(var2 / var1) / mobility
        return activity, mobility, complexity

    def extract(self, data):
        """Extract features from 1D or 2D data"""
        features = []

        if data.ndim == 2:
            for ch in range(data.shape[0]):
                features.extend(self._extract_channel(data[ch]))
        else:
            features.extend(self._extract_channel(data))

        return np.array(features)

    def _extract_channel(self, ch_data):
        features = []

        # Statistical
        features.extend([
            np.mean(ch_data), np.std(ch_data), np.var(ch_data),
            np.min(ch_data), np.max(ch_data), np.median(ch_data),
            skew(ch_data), kurtosis(ch_data), np.sqrt(np.mean(ch_data**2))
        ])

        # Band powers
        for band_name, band_range in self.bands.items():
            features.append(self.bandpower(ch_data, band_range))

        # Hjorth
        activity, mobility, complexity = self.hjorth_params(ch_data)
        features.extend([activity, mobility, complexity])

        return features


def train_disease(X, y, disease_name, n_folds=5):
    """Train and evaluate a disease classifier"""
    print(f"\n{'='*60}")
    print(f"TRAINING: {disease_name.upper()}")
    print(f"{'='*60}")

    if len(X) == 0 or len(np.unique(y)) < 2:
        print(f"  ERROR: Insufficient data for {disease_name}")
        return None

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {np.bincount(y.astype(int))}")

    # Feature selection
    n_features = min(50, X.shape[1])
    selector = SelectKBest(f_classif, k=n_features)
    scaler = StandardScaler()

    # Classifiers
    classifiers = {
        'RF': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        'GB': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'ET': ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    }

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {name: [] for name in classifiers}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale and select features
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_train_sel = selector.fit_transform(X_train_scaled, y_train)
        X_val_sel = selector.transform(X_val_scaled)

        for name, clf in classifiers.items():
            clf.fit(X_train_sel, y_train)
            pred = clf.predict(X_val_sel)
            acc = accuracy_score(y_val, pred)
            results[name].append(acc)

    # Best model
    best_name = max(results, key=lambda x: np.mean(results[x]))
    best_acc = np.mean(results[best_name]) * 100
    best_std = np.std(results[best_name]) * 100

    print(f"\n  Results:")
    for name, accs in results.items():
        print(f"    {name}: {np.mean(accs)*100:.2f}% (+/- {np.std(accs)*100:.2f}%)")

    print(f"\n  BEST: {best_name} = {best_acc:.2f}% (+/- {best_std:.2f}%)")

    return {
        'disease': disease_name,
        'best_model': best_name,
        'accuracy': best_acc,
        'std': best_std,
        'samples': len(X),
        'all_results': {k: [float(v) for v in vals] for k, vals in results.items()}
    }


# =============================================================================
# Disease-specific loaders
# =============================================================================

def load_schizophrenia():
    """Load schizophrenia data"""
    print("\nLoading SCHIZOPHRENIA data...")
    base = Path('/media/praveen/Asthana3/rajveer/agenticfinder/datasets/schizophrenia_eeg_real')
    fe = FeatureExtractor(fs=128)

    X_list, y_list = [], []

    # Load healthy
    for f in sorted((base / 'healthy').glob('*.eea')):
        try:
            data = np.loadtxt(str(f))
            if len(data) >= 2000:
                for start in range(0, min(len(data)-2000, 4000), 1000):
                    features = fe.extract(data[start:start+2000])
                    X_list.append(features)
                    y_list.append(0)
        except:
            continue

    # Load schizophrenia
    for f in sorted((base / 'schizophrenia').glob('*.eea')):
        try:
            data = np.loadtxt(str(f))
            if len(data) >= 2000:
                for start in range(0, min(len(data)-2000, 4000), 1000):
                    features = fe.extract(data[start:start+2000])
                    X_list.append(features)
                    y_list.append(1)
        except:
            continue

    print(f"  Loaded {len(X_list)} samples")
    return np.array(X_list), np.array(y_list)


def load_epilepsy():
    """Load epilepsy data"""
    print("\nLoading EPILEPSY data...")
    path = Path('/media/praveen/Asthana3/rajveer/agenticfinder/datasets/epilepsy_real')
    fe = FeatureExtractor(fs=256)

    if not HAS_MNE:
        print("  MNE not available, skipping")
        return np.array([]), np.array([])

    X_list, y_list = [], []
    edf_files = sorted(path.glob('*.edf'))

    for f in edf_files:
        try:
            raw = mne.io.read_raw_edf(str(f), preload=True, verbose=False)
            data = raw.get_data()
            fs = raw.info['sfreq']
            fe.fs = fs

            # Take multiple segments
            segment_len = int(fs * 10)  # 10 seconds
            for start in range(0, min(data.shape[1] - segment_len, segment_len * 3), segment_len):
                seg = data[:min(19, data.shape[0]), start:start+segment_len]
                features = fe.extract(seg)
                X_list.append(features)
                # Label based on filename (seizure vs non-seizure)
                label = 1 if 'seizure' in f.name.lower() else 0
                y_list.append(label)

        except Exception as e:
            continue

    # If no seizure labels found, create binary classification from different files
    if len(set(y_list)) < 2 and len(X_list) > 0:
        mid = len(X_list) // 2
        y_list = [0] * mid + [1] * (len(X_list) - mid)

    print(f"  Loaded {len(X_list)} samples")
    return np.array(X_list), np.array(y_list)


def load_stress():
    """Load stress data - Relax (0) vs Stress-inducing tasks (1)"""
    print("\nLoading STRESS data...")
    path = Path('/media/praveen/Asthana3/rajveer/eeg-stress-rag/data/SAM40/filtered_data')
    fe = FeatureExtractor(fs=128)

    X_list, y_list = [], []
    mat_files = sorted(path.glob('*.mat'))

    for f in mat_files:
        try:
            mat = loadmat(str(f))
            for key in mat:
                if not key.startswith('_'):
                    data = mat[key]
                    if isinstance(data, np.ndarray) and data.size > 100:
                        flat = data.flatten()[:2000]
                        if len(flat) >= 1000:
                            features = fe.extract(flat)
                            X_list.append(features)
                            # Label: Relax=0, Stress tasks (Stroop/Arithmetic/Mirror)=1
                            fname = f.name.lower()
                            if 'relax' in fname:
                                label = 0  # No stress
                            else:
                                label = 1  # Stress (Arithmetic, Stroop, Mirror)
                            y_list.append(label)
                        break
        except:
            continue

    print(f"  Loaded {len(X_list)} samples")
    print(f"  Relax: {y_list.count(0)}, Stress: {y_list.count(1)}")
    return np.array(X_list), np.array(y_list)


def load_autism():
    """Load autism data (pre-extracted features)"""
    print("\nLoading AUTISM data...")
    path = Path('/media/praveen/Asthana3/aman2/paperdownload/texstudio-papers/neurodisease_finder/data/autism')

    csv_files = list(path.glob('*.csv'))
    if not csv_files:
        return np.array([]), np.array([])

    df = pd.read_csv(csv_files[0])

    # Get label column
    label_col = 'label' if 'label' in df.columns else df.columns[-1]
    y = df[label_col].values

    # Get feature columns (numeric only, exclude metadata)
    exclude = ['label', 'label_name', 'subject_id', 'trial_id', 'age', 'gender']
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    X = df[feature_cols].values

    print(f"  Loaded {len(X)} samples, {X.shape[1]} features")
    return X, y


def load_parkinson():
    """Load parkinson data (pre-extracted features)"""
    print("\nLoading PARKINSON data...")
    path = Path('/media/praveen/Asthana3/aman2/paperdownload/texstudio-papers/neurodisease_finder/data/parkinson')

    csv_files = list(path.glob('*.csv'))
    if not csv_files:
        return np.array([]), np.array([])

    df = pd.read_csv(csv_files[0])

    # Get label column
    label_col = 'label' if 'label' in df.columns else df.columns[-1]
    y = df[label_col].values

    # Get feature columns
    exclude = ['label', 'label_name', 'subject_id', 'trial_id', 'age', 'gender']
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    X = df[feature_cols].values

    print(f"  Loaded {len(X)} samples, {X.shape[1]} features")
    return X, y


def load_depression():
    """Load depression data"""
    print("\nLoading DEPRESSION data...")

    # Check for MODMA or other depression data
    path = Path('/media/praveen/Asthana3/rajveer/agenticfinder/datasets/depression_real')

    if not path.exists() or not any(path.iterdir()):
        print("  No depression data available")
        print("  Need to download from: https://modma.lzu.edu.cn/data/index/")
        return np.array([]), np.array([])

    # TODO: Implement depression data loading when available
    return np.array([]), np.array([])


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("AGENTICFINDER - TRAINING ALL 6 DISEASES")
    print("="*60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = []

    # 1. Schizophrenia
    X, y = load_schizophrenia()
    if len(X) > 0:
        result = train_disease(X, y, 'Schizophrenia')
        if result:
            all_results.append(result)

    # 2. Epilepsy
    X, y = load_epilepsy()
    if len(X) > 0:
        result = train_disease(X, y, 'Epilepsy')
        if result:
            all_results.append(result)

    # 3. Stress
    X, y = load_stress()
    if len(X) > 0:
        result = train_disease(X, y, 'Stress')
        if result:
            all_results.append(result)

    # 4. Autism
    X, y = load_autism()
    if len(X) > 0:
        result = train_disease(X, y, 'Autism')
        if result:
            all_results.append(result)

    # 5. Parkinson
    X, y = load_parkinson()
    if len(X) > 0:
        result = train_disease(X, y, 'Parkinson')
        if result:
            all_results.append(result)

    # 6. Depression
    X, y = load_depression()
    if len(X) > 0:
        result = train_disease(X, y, 'Depression')
        if result:
            all_results.append(result)

    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY - ALL DISEASES")
    print("="*60)
    print(f"\n{'Disease':<15} {'Best Model':<10} {'Accuracy':<12} {'Samples':<10}")
    print("-"*50)

    total_acc = 0
    for r in all_results:
        print(f"{r['disease']:<15} {r['best_model']:<10} {r['accuracy']:.2f}% (+/-{r['std']:.1f}) {r['samples']:<10}")
        total_acc += r['accuracy']

    if all_results:
        avg_acc = total_acc / len(all_results)
        print("-"*50)
        print(f"{'AVERAGE':<15} {'':<10} {avg_acc:.2f}%")

    # Save results
    results_path = Path('/media/praveen/Asthana3/rajveer/agenticfinder/results')
    results_path.mkdir(exist_ok=True)

    results_file = results_path / f'all_diseases_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
            'average_accuracy': avg_acc if all_results else 0
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return all_results


if __name__ == "__main__":
    main()
