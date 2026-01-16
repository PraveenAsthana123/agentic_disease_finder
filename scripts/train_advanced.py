#!/usr/bin/env python3
"""
Advanced EEG Classification with Feature Engineering
AgenticFinder - Target: 90% Accuracy
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier,
                              StackingClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif

try:
    import mne
    mne.set_log_level('ERROR')
    HAS_MNE = True
except ImportError:
    HAS_MNE = False


class AdvancedFeatureExtractor:
    """Extract comprehensive features from EEG signals"""

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
        """Calculate band power using Welch's method"""
        low, high = band
        nperseg = min(len(data), int(self.fs * 2))
        if nperseg < 4:
            return 0
        freqs, psd = signal.welch(data, self.fs, nperseg=nperseg)
        idx = np.logical_and(freqs >= low, freqs <= high)
        if np.sum(idx) == 0:
            return 0
        return np.trapz(psd[idx], freqs[idx])

    def spectral_entropy(self, data):
        """Calculate spectral entropy"""
        nperseg = min(len(data), int(self.fs * 2))
        if nperseg < 4:
            return 0
        freqs, psd = signal.welch(data, self.fs, nperseg=nperseg)
        psd_norm = psd / (psd.sum() + 1e-10)
        psd_norm = psd_norm[psd_norm > 0]
        return -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

    def hjorth_params(self, data):
        """Calculate Hjorth parameters"""
        diff1 = np.diff(data)
        diff2 = np.diff(diff1)
        var0 = np.var(data)
        var1 = np.var(diff1)
        var2 = np.var(diff2)

        activity = var0
        mobility = np.sqrt(var1 / (var0 + 1e-10))
        complexity = np.sqrt(var2 / (var1 + 1e-10)) / (mobility + 1e-10)

        return activity, mobility, complexity

    def zero_crossings(self, data):
        """Count zero crossings"""
        return np.sum(np.diff(np.sign(data)) != 0)

    def line_length(self, data):
        """Calculate line length"""
        return np.sum(np.abs(np.diff(data)))

    def sample_entropy(self, data, m=2, r=0.2):
        """Simplified sample entropy"""
        N = len(data)
        if N < m + 1:
            return 0

        r_val = r * np.std(data)
        if r_val == 0:
            return 0

        # Simplified approximation
        diff = np.abs(np.diff(data))
        return np.log(np.sum(diff < r_val) / (len(diff) + 1) + 1e-10)

    def extract_features(self, data):
        """Extract all features from signal"""
        features = []

        # Handle 2D data (channels x samples)
        if data.ndim == 2:
            for ch in range(data.shape[0]):
                features.extend(self._extract_channel_features(data[ch]))
        else:
            features.extend(self._extract_channel_features(data))

        return np.array(features)

    def _extract_channel_features(self, ch_data):
        """Extract features from single channel"""
        features = []

        # Statistical features
        features.append(np.mean(ch_data))
        features.append(np.std(ch_data))
        features.append(np.var(ch_data))
        features.append(np.min(ch_data))
        features.append(np.max(ch_data))
        features.append(np.median(ch_data))
        features.append(np.percentile(ch_data, 25))
        features.append(np.percentile(ch_data, 75))
        features.append(skew(ch_data))
        features.append(kurtosis(ch_data))
        features.append(np.sqrt(np.mean(ch_data**2)))  # RMS

        # Band powers
        total_power = 0
        band_powers = []
        for band_name, band_range in self.bands.items():
            bp = self.bandpower(ch_data, band_range)
            band_powers.append(bp)
            total_power += bp
        features.extend(band_powers)

        # Relative band powers
        for bp in band_powers:
            features.append(bp / (total_power + 1e-10))

        # Spectral features
        features.append(self.spectral_entropy(ch_data))

        # Hjorth parameters
        activity, mobility, complexity = self.hjorth_params(ch_data)
        features.extend([activity, mobility, complexity])

        # Time-domain features
        features.append(self.zero_crossings(ch_data))
        features.append(self.line_length(ch_data))
        features.append(self.sample_entropy(ch_data))

        # Band ratios (clinically relevant)
        if band_powers[1] > 0:  # theta
            features.append(band_powers[2] / (band_powers[1] + 1e-10))  # alpha/theta
        else:
            features.append(0)

        if band_powers[3] > 0:  # beta
            features.append(band_powers[2] / (band_powers[3] + 1e-10))  # alpha/beta
        else:
            features.append(0)

        return features


class DataLoader:
    """Load EEG data from multiple sources"""

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.feature_extractor = AdvancedFeatureExtractor(fs=128)

    def load_eea_files(self, healthy_path, disease_path, augment=True):
        """Load EEA files with optional augmentation"""
        print("Loading MHRC dataset...")
        X_list, y_list, subj_list = [], [], []

        # Load healthy
        for i, f in enumerate(sorted(Path(healthy_path).glob('*.eea'))):
            try:
                data = np.loadtxt(str(f))
                if len(data) >= 2000:
                    segments = self._segment_data(data[:8000], augment)
                    for seg in segments:
                        features = self.feature_extractor.extract_features(seg)
                        X_list.append(features)
                        y_list.append(0)
                        subj_list.append(f'H_{i}')
            except:
                continue

        # Load schizophrenia
        for i, f in enumerate(sorted(Path(disease_path).glob('*.eea'))):
            try:
                data = np.loadtxt(str(f))
                if len(data) >= 2000:
                    segments = self._segment_data(data[:8000], augment)
                    for seg in segments:
                        features = self.feature_extractor.extract_features(seg)
                        X_list.append(features)
                        y_list.append(1)
                        subj_list.append(f'S_{i}')
            except:
                continue

        print(f"  Loaded {len(X_list)} samples")
        return np.array(X_list), np.array(y_list), np.array(subj_list)

    def load_edf_files(self, path, label, prefix, max_files=100, augment=True):
        """Load EDF files"""
        if not HAS_MNE:
            return np.array([]), np.array([]), np.array([])

        print(f"Loading EDF files from {path}...")
        X_list, y_list, subj_list = [], [], []

        edf_files = sorted(Path(path).glob('**/*.edf'))[:max_files]
        for i, f in enumerate(edf_files):
            try:
                raw = mne.io.read_raw_edf(str(f), preload=True, verbose=False)
                data = raw.get_data()
                fs = raw.info['sfreq']

                # Resample if needed
                if fs != 128:
                    target_len = int(data.shape[1] * 128 / fs)
                    data = signal.resample(data, target_len, axis=1)

                # Use all available channels
                self.feature_extractor.fs = 128

                segments = self._segment_2d_data(data[:19, :], augment)
                for seg in segments:
                    features = self.feature_extractor.extract_features(seg)
                    X_list.append(features)
                    y_list.append(label)
                    subj_list.append(f'{prefix}_{i}')

            except Exception as e:
                continue

        print(f"  Loaded {len(X_list)} samples")
        return np.array(X_list), np.array(y_list), np.array(subj_list)

    def _segment_data(self, data, augment=True):
        """Segment 1D data into overlapping windows"""
        segment_len = 2000
        segments = []

        if len(data) >= segment_len:
            segments.append(data[:segment_len])

            if augment and len(data) >= segment_len * 2:
                # Overlapping segments
                for start in range(500, len(data) - segment_len, 500):
                    segments.append(data[start:start + segment_len])
                    if len(segments) >= 4:
                        break

                # Add noise augmentation
                for seg in segments[:2]:
                    noisy = seg + np.random.normal(0, 0.01 * np.std(seg), len(seg))
                    segments.append(noisy)

        return segments

    def _segment_2d_data(self, data, augment=True):
        """Segment 2D data (channels x samples)"""
        segment_len = 1280  # 10 seconds at 128 Hz
        segments = []

        if data.shape[1] >= segment_len:
            segments.append(data[:, :segment_len])

            if augment and data.shape[1] >= segment_len * 2:
                for start in range(640, data.shape[1] - segment_len, 640):
                    segments.append(data[:, start:start + segment_len])
                    if len(segments) >= 3:
                        break

        return segments


class EnsembleClassifier:
    """Advanced ensemble classifier"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.selector = None

        # Base classifiers
        self.base_classifiers = {
            'rf': RandomForestClassifier(n_estimators=200, max_depth=15,
                                         min_samples_split=5, random_state=42, n_jobs=-1),
            'et': ExtraTreesClassifier(n_estimators=200, max_depth=15,
                                       random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                             learning_rate=0.1, random_state=42),
            'ada': AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
            'svm': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=1000,
                                 early_stopping=True, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        }

    def train_cv(self, X, y, subjects, n_folds=5):
        """Train with cross-validation"""
        print(f"\n{'='*60}")
        print(f"TRAINING ENSEMBLE WITH {n_folds}-FOLD CV")
        print(f"{'='*60}")

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Feature selection
        n_features = min(100, X.shape[1])
        self.selector = SelectKBest(f_classif, k=n_features)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        all_results = defaultdict(list)
        all_preds = []
        all_labels = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n--- Fold {fold + 1}/{n_folds} ---")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Scale
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Feature selection
            X_train_selected = self.selector.fit_transform(X_train_scaled, y_train)
            X_val_selected = self.selector.transform(X_val_scaled)

            fold_preds = {}

            # Train each classifier
            for name, clf in self.base_classifiers.items():
                try:
                    clf.fit(X_train_selected, y_train)
                    pred = clf.predict(X_val_selected)
                    acc = accuracy_score(y_val, pred)
                    all_results[name].append(acc)
                    fold_preds[name] = pred
                    print(f"  {name.upper():6} Accuracy: {acc*100:.2f}%")
                except Exception as e:
                    print(f"  {name.upper():6} Error: {e}")

            # Voting ensemble
            if fold_preds:
                # Majority voting
                pred_matrix = np.array(list(fold_preds.values()))
                ensemble_pred = np.apply_along_axis(
                    lambda x: np.bincount(x.astype(int)).argmax(),
                    axis=0,
                    arr=pred_matrix
                )
                ensemble_acc = accuracy_score(y_val, ensemble_pred)
                all_results['ensemble'].append(ensemble_acc)
                print(f"  ENSEMBLE Accuracy: {ensemble_acc*100:.2f}%")

                all_preds.extend(ensemble_pred)
                all_labels.extend(y_val)

        # Summary
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")

        best_model = None
        best_acc = 0

        for name, accs in all_results.items():
            mean_acc = np.mean(accs) * 100
            std_acc = np.std(accs) * 100
            print(f"{name.upper():10} : {mean_acc:.2f}% (+/- {std_acc:.2f}%)")

            if mean_acc > best_acc:
                best_acc = mean_acc
                best_model = name

        print(f"\nBest Model: {best_model.upper()} with {best_acc:.2f}%")

        print(f"\nClassification Report (Ensemble):")
        print(classification_report(all_labels, all_preds,
                                    target_names=['Healthy', 'Schizophrenia']))

        return best_acc, all_results


def main():
    print("="*60)
    print("AGENTICFINDER - ADVANCED EEG CLASSIFICATION")
    print("TARGET: 90% ACCURACY")
    print("="*60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    base_path = Path('/media/praveen/Asthana3/rajveer/agenticfinder/datasets/schizophrenia_eeg_real')
    loader = DataLoader(base_path)

    # Load data with augmentation
    X_mhrc, y_mhrc, subj_mhrc = loader.load_eea_files(
        base_path / 'healthy',
        base_path / 'schizophrenia',
        augment=True
    )

    # Load RepOD data
    repod_path = base_path / 'repod_dataset'
    X_repod_h, y_repod_h, subj_repod_h = np.array([]), np.array([]), np.array([])
    X_repod_s, y_repod_s, subj_repod_s = np.array([]), np.array([]), np.array([])

    if repod_path.exists():
        for subdir in repod_path.iterdir():
            if subdir.is_dir():
                if 'healthy' in subdir.name.lower():
                    X_repod_h, y_repod_h, subj_repod_h = loader.load_edf_files(
                        subdir, label=0, prefix='RepOD_H', augment=True
                    )
                elif 'schiz' in subdir.name.lower():
                    X_repod_s, y_repod_s, subj_repod_s = loader.load_edf_files(
                        subdir, label=1, prefix='RepOD_S', augment=True
                    )

    # Combine all data
    X_list = [X_mhrc]
    y_list = [y_mhrc]
    subj_list = [subj_mhrc]

    for X, y, s in [(X_repod_h, y_repod_h, subj_repod_h),
                    (X_repod_s, y_repod_s, subj_repod_s)]:
        if len(X) > 0:
            X_list.append(X)
            y_list.append(y)
            subj_list.append(s)

    # Handle different feature dimensions
    if len(X_list) > 1:
        max_features = max(x.shape[1] for x in X_list if len(x) > 0)
        X_padded = []
        for X in X_list:
            if len(X) > 0:
                if X.shape[1] < max_features:
                    X = np.pad(X, ((0, 0), (0, max_features - X.shape[1])))
                X_padded.append(X)
        X = np.vstack(X_padded)
        y = np.concatenate([y for y in y_list if len(y) > 0])
        subjects = np.concatenate([s for s in subj_list if len(s) > 0])
    else:
        X = X_mhrc
        y = y_mhrc
        subjects = subj_mhrc

    print(f"\n{'='*60}")
    print("DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Healthy: {np.sum(y == 0)}, Schizophrenia: {np.sum(y == 1)}")

    # Train
    classifier = EnsembleClassifier()
    best_acc, results = classifier.train_cv(X, y, subjects, n_folds=5)

    # Save results
    results_path = Path('/media/praveen/Asthana3/rajveer/agenticfinder/results')
    results_path.mkdir(exist_ok=True)

    results_data = {
        'timestamp': datetime.now().isoformat(),
        'best_accuracy': best_acc,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'results': {k: [float(v) for v in vals] for k, vals in results.items()}
    }

    results_file = results_path / f'advanced_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {results_file}")
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if best_acc >= 90:
        print("\n TARGET 90% ACHIEVED!")
    else:
        print(f"\n Gap to target: {90 - best_acc:.2f}%")

    return best_acc


if __name__ == "__main__":
    main()
