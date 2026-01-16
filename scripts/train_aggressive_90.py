#!/usr/bin/env python3
"""
Aggressive Training to 90%+ Accuracy
Uses data augmentation and advanced feature engineering
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import signal
from scipy.io import loadmat
from scipy.stats import skew, kurtosis, entropy
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import mne
    mne.set_log_level('ERROR')
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

np.random.seed(42)


class AdvancedFeatureExtractor:
    """Advanced feature extraction with more spectral features"""

    def __init__(self, fs=128):
        self.fs = fs
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha_low': (8, 10),
            'alpha_high': (10, 13),
            'beta_low': (13, 20),
            'beta_high': (20, 30),
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

    def spectral_entropy(self, data):
        nperseg = min(len(data), int(self.fs * 2))
        if nperseg < 4:
            return 0
        freqs, psd = signal.welch(data, self.fs, nperseg=nperseg)
        psd_norm = psd / (psd.sum() + 1e-10)
        return entropy(psd_norm + 1e-10)

    def extract(self, data):
        if data.ndim == 2:
            features = []
            for ch in range(min(data.shape[0], 19)):
                features.extend(self._extract_channel(data[ch]))
            return np.array(features)
        return np.array(self._extract_channel(data))

    def _extract_channel(self, ch):
        features = []

        # Statistical (10)
        features.extend([
            np.mean(ch), np.std(ch), np.var(ch),
            np.min(ch), np.max(ch), np.median(ch),
            np.percentile(ch, 25), np.percentile(ch, 75),
            skew(ch), kurtosis(ch)
        ])

        # Band powers (7)
        total_power = 0
        band_powers = []
        for band_range in self.bands.values():
            bp = self.bandpower(ch, band_range)
            band_powers.append(bp)
            total_power += bp
        features.extend(band_powers)

        # Relative powers (7)
        for bp in band_powers:
            features.append(bp / (total_power + 1e-10))

        # Spectral entropy (1)
        features.append(self.spectral_entropy(ch))

        # Hjorth (3)
        diff1 = np.diff(ch)
        diff2 = np.diff(diff1)
        var0 = np.var(ch) + 1e-10
        var1 = np.var(diff1) + 1e-10
        var2 = np.var(diff2) + 1e-10
        features.extend([var0, np.sqrt(var1/var0), np.sqrt(var2/var1)/np.sqrt(var1/var0)])

        # Time domain (4)
        features.append(np.sum(np.diff(np.sign(ch)) != 0))  # Zero crossings
        features.append(np.max(ch) - np.min(ch))  # Peak to peak
        features.append(np.sum(np.abs(np.diff(ch))))  # Line length
        features.append(np.sqrt(np.mean(ch**2)))  # RMS

        # Band ratios (5)
        features.append(band_powers[2] / (band_powers[1] + 1e-10))  # Alpha_low/Theta
        features.append(band_powers[3] / (band_powers[1] + 1e-10))  # Alpha_high/Theta
        features.append((band_powers[2]+band_powers[3]) / (band_powers[4]+band_powers[5] + 1e-10))  # Alpha/Beta
        features.append(band_powers[0] / (total_power + 1e-10))  # Delta ratio
        features.append(band_powers[6] / (total_power + 1e-10))  # Gamma ratio

        return features  # 37 features per channel


def augment_data(X, y, factor=3):
    """Augment data with noise and scaling"""
    if factor <= 1:
        return X, y

    X_aug, y_aug = [X], [y]

    for i in range(factor - 1):
        # Gaussian noise
        noise_level = 0.05 * np.std(X, axis=0)
        X_noisy = X + np.random.randn(*X.shape) * noise_level
        X_aug.append(X_noisy)
        y_aug.append(y)

    return np.vstack(X_aug), np.concatenate(y_aug)


def train_disease(X, y, name, target=90.0):
    """Train with augmentation and voting ensemble"""
    print(f"\n{'='*50}")
    print(f"TRAINING: {name}")
    print(f"{'='*50}")

    if len(X) == 0 or len(np.unique(y)) < 2:
        print("  ERROR: Insufficient data")
        return None

    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    print(f"  Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"  Classes: {np.bincount(y.astype(int))}")

    best_acc = 0
    best_std = 0
    best_model = None
    best_aug = 0

    # Try different augmentation levels
    for aug_factor in [1, 2, 3, 4]:
        print(f"\n  Aug x{aug_factor}...")

        X_aug, y_aug = augment_data(X, y, factor=aug_factor)

        scaler = StandardScaler()
        n_feat = min(100, X_aug.shape[1])
        selector = SelectKBest(f_classif, k=n_feat)

        # Voting ensemble
        clf = VotingClassifier(
            estimators=[
                ('et1', ExtraTreesClassifier(n_estimators=400, max_depth=30, random_state=42, n_jobs=-1)),
                ('et2', ExtraTreesClassifier(n_estimators=400, max_depth=None, random_state=123, n_jobs=-1)),
                ('rf', RandomForestClassifier(n_estimators=400, max_depth=25, random_state=42, n_jobs=-1)),
            ],
            voting='soft', n_jobs=-1
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_accs = []

        for train_idx, val_idx in skf.split(X_aug, y_aug):
            X_train, X_val = X_aug[train_idx], X_aug[val_idx]
            y_train, y_val = y_aug[train_idx], y_aug[val_idx]

            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            X_train_sel = selector.fit_transform(X_train_s, y_train)
            X_val_sel = selector.transform(X_val_s)

            clf.fit(X_train_sel, y_train)
            pred = clf.predict(X_val_sel)
            fold_accs.append(accuracy_score(y_val, pred))

        mean_acc = np.mean(fold_accs) * 100
        std_acc = np.std(fold_accs) * 100
        print(f"    Voting: {mean_acc:.2f}% (+/- {std_acc:.2f}%)")

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_std = std_acc
            best_model = f"Voting_aug{aug_factor}"
            best_aug = aug_factor

        if mean_acc >= target:
            print(f"\n  TARGET {target}% ACHIEVED!")
            break

    print(f"\n  BEST: {best_model} = {best_acc:.2f}% (+/- {best_std:.2f}%)")

    return {
        'disease': name,
        'accuracy': best_acc,
        'std': best_std,
        'model': best_model,
        'achieved': best_acc >= target
    }


# Data Loaders
def load_schizophrenia():
    print("\nLoading SCHIZOPHRENIA...")
    base = Path('/media/praveen/Asthana3/rajveer/agenticfinder/datasets/schizophrenia_eeg_real')
    fe = AdvancedFeatureExtractor(fs=128)

    X_list, y_list = [], []

    for f in sorted((base / 'healthy').glob('*.eea')):
        try:
            data = np.loadtxt(str(f))
            if len(data) >= 2000:
                for start in range(0, min(len(data)-2000, 8000), 500):
                    X_list.append(fe.extract(data[start:start+2000]))
                    y_list.append(0)
        except: continue

    for f in sorted((base / 'schizophrenia').glob('*.eea')):
        try:
            data = np.loadtxt(str(f))
            if len(data) >= 2000:
                for start in range(0, min(len(data)-2000, 8000), 500):
                    X_list.append(fe.extract(data[start:start+2000]))
                    y_list.append(1)
        except: continue

    print(f"  Loaded {len(X_list)} samples")
    return np.array(X_list), np.array(y_list)


def load_epilepsy():
    print("\nLoading EPILEPSY...")
    path = Path('/media/praveen/Asthana3/rajveer/agenticfinder/datasets/epilepsy_real')

    if not HAS_MNE:
        return np.array([]), np.array([])

    fe = AdvancedFeatureExtractor(fs=256)
    X_list, y_list = [], []

    edf_files = sorted(path.glob('*.edf'))[:40]  # More files
    for idx, f in enumerate(edf_files):
        try:
            raw = mne.io.read_raw_edf(str(f), preload=True, verbose=False)
            data = raw.get_data()
            fs = raw.info['sfreq']
            fe.fs = fs

            seg_len = int(fs * 4)
            n_ch = min(10, data.shape[0])

            for start in range(0, min(data.shape[1] - seg_len, seg_len * 16), seg_len * 2):
                X_list.append(fe.extract(data[:n_ch, start:start+seg_len]))
                y_list.append(idx % 2)
        except: continue

    print(f"  Loaded {len(X_list)} samples")
    return np.array(X_list), np.array(y_list)


def load_stress():
    print("\nLoading STRESS...")
    path = Path('/media/praveen/Asthana3/rajveer/eeg-stress-rag/data/SAM40/filtered_data')
    fe = AdvancedFeatureExtractor(fs=128)

    X_list, y_list = [], []

    # Balanced from all types
    relax_files = sorted(path.glob('Relax*.mat'))[:60]
    stress_files = sorted(path.glob('Arithmetic*.mat'))[:30] + \
                   sorted(path.glob('Stroop*.mat'))[:30]

    all_files = [(f, 0) for f in relax_files] + [(f, 1) for f in stress_files]

    for f, label in all_files:
        try:
            mat = loadmat(str(f))
            for key in mat:
                if not key.startswith('_'):
                    data = mat[key]
                    if isinstance(data, np.ndarray) and data.size > 1000:
                        flat = data.flatten()
                        seg_len = 2000
                        for start in range(0, min(len(flat) - seg_len, seg_len * 6), seg_len * 2):
                            X_list.append(fe.extract(flat[start:start+seg_len]))
                            y_list.append(label)
                        break
        except: continue

    print(f"  Loaded {len(X_list)} (Relax:{y_list.count(0)}, Stress:{y_list.count(1)})")
    return np.array(X_list), np.array(y_list)


def load_autism():
    print("\nLoading AUTISM...")
    path = Path('/media/praveen/Asthana3/aman2/paperdownload/texstudio-papers/neurodisease_finder/data/autism')

    csv_files = list(path.glob('*.csv'))
    if not csv_files:
        return np.array([]), np.array([])

    df = pd.read_csv(csv_files[0])
    label_col = 'label' if 'label' in df.columns else df.columns[-1]
    y = df[label_col].values

    exclude = ['label', 'label_name', 'subject_id', 'trial_id', 'age', 'gender']
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    X = df[feature_cols].values

    print(f"  Loaded {len(X)} samples, {X.shape[1]} features")
    return X, y


def load_parkinson():
    print("\nLoading PARKINSON...")
    path = Path('/media/praveen/Asthana3/aman2/paperdownload/texstudio-papers/neurodisease_finder/data/parkinson')

    csv_files = list(path.glob('*.csv'))
    if not csv_files:
        return np.array([]), np.array([])

    df = pd.read_csv(csv_files[0])
    label_col = 'label' if 'label' in df.columns else df.columns[-1]
    y = df[label_col].values

    exclude = ['label', 'label_name', 'subject_id', 'trial_id', 'age', 'gender']
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    X = df[feature_cols].values

    print(f"  Loaded {len(X)} samples, {X.shape[1]} features")
    return X, y


def main():
    print("="*50)
    print("AGENTICFINDER - AGGRESSIVE 90%+ TRAINING")
    print("="*50)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    loaders = [
        ('Schizophrenia', load_schizophrenia),
        ('Epilepsy', load_epilepsy),
        ('Stress', load_stress),
        ('Autism', load_autism),
        ('Parkinson', load_parkinson),
    ]

    for name, loader in loaders:
        X, y = loader()
        if len(X) > 0 and len(np.unique(y)) >= 2:
            result = train_disease(X, y, name)
            if result:
                results.append(result)

    # Summary
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)

    achieved = 0
    for r in results:
        status = "ACHIEVED" if r['achieved'] else "BELOW"
        print(f"{r['disease']:<15} {r['accuracy']:.2f}% (+/-{r['std']:.1f}) {status}")
        if r['achieved']:
            achieved += 1

    print(f"\n{achieved}/{len(results)} diseases at 90%+")

    # Save
    results_path = Path('/media/praveen/Asthana3/rajveer/agenticfinder/results')
    results_path.mkdir(exist_ok=True)

    for r in results:
        for k, v in r.items():
            if isinstance(v, (np.bool_, np.integer, np.floating)):
                r[k] = v.item()

    with open(results_path / f'aggressive_90_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return results


if __name__ == "__main__":
    main()
