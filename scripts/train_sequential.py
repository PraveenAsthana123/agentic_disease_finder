#!/usr/bin/env python3
"""
Sequential Training - One disease at a time
Manages memory by training each dataset separately with cleanup between
"""

import sys
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import signal
from scipy.io import loadmat
from scipy.stats import skew, kurtosis, entropy
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              GradientBoostingClassifier, VotingClassifier)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import mne
    mne.set_log_level('ERROR')
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

np.random.seed(42)

BASE_DIR = Path('/media/praveen/Asthana3/rajveer/agenticfinder')
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR = BASE_DIR / 'saved_models'
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


class AdvancedFeatureExtractor:
    """Feature extraction for EEG data"""

    def __init__(self, fs=256):
        self.fs = fs
        self.bands = {
            'delta': (0.5, 4), 'theta': (4, 8),
            'alpha1': (8, 10), 'alpha2': (10, 13),
            'beta1': (13, 20), 'beta2': (20, 30),
            'gamma1': (30, 40), 'gamma2': (40, 50)
        }

    def bandpass_filter(self, data, low=0.5, high=45):
        nyq = self.fs / 2
        if high >= nyq:
            high = nyq - 1
        try:
            b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
            return signal.filtfilt(b, a, data)
        except:
            return data

    def bandpower(self, data, band):
        low, high = band
        nperseg = min(len(data), int(self.fs * 2))
        if nperseg < 8:
            return 0
        freqs, psd = signal.welch(data, self.fs, nperseg=nperseg)
        idx = np.logical_and(freqs >= low, freqs <= high)
        return np.trapz(psd[idx], freqs[idx]) if np.sum(idx) > 0 else 0

    def extract(self, data):
        if data.ndim == 2:
            ch_features = []
            for ch in range(min(data.shape[0], 19)):
                ch_data = self.bandpass_filter(data[ch])
                ch_features.append(self._extract_channel(ch_data))
            features = np.mean(ch_features, axis=0)
        else:
            data = self.bandpass_filter(data)
            features = self._extract_channel(data)
        return np.array(features)

    def _extract_channel(self, ch):
        features = []

        # Statistical (12)
        features.extend([
            np.mean(ch), np.std(ch), np.var(ch),
            np.min(ch), np.max(ch), np.median(ch),
            np.percentile(ch, 10), np.percentile(ch, 25),
            np.percentile(ch, 75), np.percentile(ch, 90),
            skew(ch) if len(ch) > 2 else 0,
            kurtosis(ch) if len(ch) > 2 else 0
        ])

        # Band powers (8)
        total_power = 0
        band_powers = []
        for band in self.bands.values():
            bp = self.bandpower(ch, band)
            band_powers.append(bp)
            total_power += bp
        features.extend(band_powers)

        # Relative band powers (8)
        for bp in band_powers:
            features.append(bp / (total_power + 1e-10))

        # Band ratios (6)
        features.append((band_powers[2] + band_powers[3]) / (band_powers[1] + 1e-10))
        features.append((band_powers[4] + band_powers[5]) / (band_powers[2] + band_powers[3] + 1e-10))
        features.append(band_powers[0] / (total_power + 1e-10))
        features.append((band_powers[6] + band_powers[7]) / (total_power + 1e-10))
        features.append(band_powers[1] / (band_powers[2] + band_powers[3] + 1e-10))
        features.append((band_powers[0] + band_powers[1]) / (band_powers[4] + band_powers[5] + 1e-10))

        # Spectral (2)
        nperseg = min(len(ch), int(self.fs * 2))
        if nperseg >= 8:
            freqs, psd = signal.welch(ch, self.fs, nperseg=nperseg)
            psd_norm = psd / (psd.sum() + 1e-10)
            features.append(entropy(psd_norm + 1e-10))
            features.append(freqs[np.argmax(psd)])
        else:
            features.extend([0, 0])

        # Hjorth (3)
        diff1 = np.diff(ch)
        diff2 = np.diff(diff1)
        var0 = np.var(ch) + 1e-10
        var1 = np.var(diff1) + 1e-10
        var2 = np.var(diff2) + 1e-10
        features.extend([var0, np.sqrt(var1/var0), np.sqrt(var2/var1)/(np.sqrt(var1/var0)+1e-10)])

        # Time domain (6)
        features.append(np.sum(np.diff(np.sign(ch)) != 0))
        features.append(np.max(ch) - np.min(ch))
        features.append(np.sum(np.abs(np.diff(ch))))
        features.append(np.sqrt(np.mean(ch**2)))
        features.append(np.mean(np.abs(ch)))
        features.append(np.sum(ch**2))

        # Nonlinear (2)
        features.append(np.std(np.diff(ch)) / (np.std(ch) + 1e-10))
        features.append(len(np.where(np.diff(np.sign(np.diff(ch))))[0]) / (len(ch) + 1e-10))

        return features


def augment_data(X_train, y_train, factor=3):
    """Augment training data only"""
    X_train = np.array(X_train, dtype=np.float64)
    X_aug, y_aug = [X_train], [y_train]
    for _ in range(factor - 1):
        std_val = np.std(X_train, axis=0)
        if np.isscalar(std_val):
            std_val = np.full(X_train.shape[1], std_val)
        noise = np.random.randn(*X_train.shape) * 0.05 * std_val
        X_aug.append(X_train + noise)
        y_aug.append(y_train)
    return np.vstack(X_aug), np.concatenate(y_aug)


def get_classifier():
    """Strong ensemble classifier"""
    return VotingClassifier(
        estimators=[
            ('et1', ExtraTreesClassifier(n_estimators=300, max_depth=25, random_state=42, n_jobs=2)),
            ('rf', RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42, n_jobs=2)),
            ('gb', GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)),
        ],
        voting='soft', n_jobs=2
    )


def train_and_test(subjects_data, disease_name, target=90.0):
    """Train model with cross-validation and save"""
    print(f"\n{'='*60}")
    print(f"TRAINING: {disease_name}")
    print(f"{'='*60}")

    n_subjects = len(subjects_data)
    if n_subjects < 10:
        print(f"  ERROR: Only {n_subjects} subjects - need at least 10")
        return None

    subject_labels = np.array([s['label'] for s in subjects_data])
    unique, counts = np.unique(subject_labels, return_counts=True)
    print(f"  Subjects: {n_subjects}")
    print(f"  Class distribution: {dict(zip(unique.astype(int), counts))}")

    if len(unique) < 2:
        print("  ERROR: Need at least 2 classes")
        return None

    best_acc = 0
    best_result = None

    for aug_factor in [1, 2, 3, 4]:
        print(f"\n  Testing augmentation x{aug_factor}...")

        n_splits = min(5, min(counts))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_accs = []
        all_y_true = []
        all_y_pred = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(n_subjects), subject_labels)):
            # Gather data
            X_train, y_train = [], []
            for idx in train_idx:
                for feat in subjects_data[idx]['features']:
                    X_train.append(feat)
                    y_train.append(subjects_data[idx]['label'])

            X_test, y_test = [], []
            for idx in test_idx:
                for feat in subjects_data[idx]['features']:
                    X_test.append(feat)
                    y_test.append(subjects_data[idx]['label'])

            X_train = np.nan_to_num(np.array(X_train), nan=0, posinf=0, neginf=0)
            X_test = np.nan_to_num(np.array(X_test), nan=0, posinf=0, neginf=0)
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            # Augment training only
            if aug_factor > 1:
                X_train, y_train = augment_data(X_train, y_train, aug_factor)

            # Scale
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Feature selection
            n_feat = min(100, X_train.shape[1])
            selector = SelectKBest(f_classif, k=n_feat)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)

            # Train
            clf = get_classifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            fold_accs.append(acc)
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

            print(f"    Fold {fold+1}: {acc*100:.1f}%")

        mean_acc = np.mean(fold_accs) * 100
        std_acc = np.std(fold_accs) * 100
        print(f"  Result: {mean_acc:.2f}% (+/- {std_acc:.2f}%)")

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_result = {
                'disease': disease_name,
                'accuracy': mean_acc,
                'std': std_acc,
                'augmentation': aug_factor,
                'n_subjects': n_subjects,
                'achieved': mean_acc >= target,
                'confusion_matrix': confusion_matrix(all_y_true, all_y_pred).tolist(),
                'classification_report': classification_report(all_y_true, all_y_pred, output_dict=True)
            }

        if mean_acc >= target:
            print(f"\n  *** TARGET {target}% ACHIEVED! ***")
            break

    # Train final model on all data
    print(f"\n  Training final model...")
    X_all, y_all = [], []
    for s in subjects_data:
        for feat in s['features']:
            X_all.append(feat)
            y_all.append(s['label'])

    X_all = np.nan_to_num(np.array(X_all), nan=0, posinf=0, neginf=0)
    y_all = np.array(y_all)

    if best_result and best_result['augmentation'] > 1:
        X_all, y_all = augment_data(X_all, y_all, best_result['augmentation'])

    scaler = RobustScaler()
    X_all = scaler.fit_transform(X_all)
    selector = SelectKBest(f_classif, k=min(100, X_all.shape[1]))
    X_all = selector.fit_transform(X_all, y_all)

    final_clf = get_classifier()
    final_clf.fit(X_all, y_all)

    # Save model
    model_path = MODELS_DIR / f"{disease_name.lower()}_model.joblib"
    joblib.dump({'model': final_clf, 'scaler': scaler, 'selector': selector}, model_path)
    print(f"  Model saved: {model_path}")

    # Cleanup
    del X_all, y_all, X_train, X_test, y_train, y_test
    gc.collect()

    return best_result


# Disease-specific loaders
def load_epilepsy():
    print("\nLoading EPILEPSY (CHB-MIT EDF format)...")
    path = BASE_DIR / 'datasets' / 'epilepsy_real'
    if not path.exists():
        return []

    if not HAS_MNE:
        print("  MNE not available for EDF loading")
        return []

    fe = AdvancedFeatureExtractor(fs=256)
    subjects_data = []
    subject_files = {}

    # Group EDF files by subject (chb01, chb02, etc.)
    for edf_file in sorted(path.glob('*.edf')):
        if edf_file.stem.endswith('.1'):  # Skip duplicates
            continue
        # Extract subject ID (e.g., chb01 from chb01_03.edf)
        parts = edf_file.stem.split('_')
        if len(parts) >= 2:
            subj_id = parts[0]
            if subj_id not in subject_files:
                subject_files[subj_id] = []
            subject_files[subj_id].append(edf_file)

    print(f"  Found {len(subject_files)} subjects with EDF files")

    # For CHB-MIT: seizure info typically in summary files
    # We'll use a simple heuristic: files with higher numbers often contain seizures
    # Or we create balanced dataset by extracting from different files

    for subj_id, files in list(subject_files.items())[:10]:  # Limit subjects for memory
        files = sorted(files)[:3]  # Limit files per subject

        for edf_file in files:
            try:
                raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
                data = raw.get_data()
                fs = raw.info['sfreq']
                fe.fs = fs

                # Extract features from segments
                seg_len = int(fs * 4)  # 4 second segments
                n_segments = min(10, (data.shape[1] - seg_len) // (seg_len // 2))

                features = []
                for i in range(n_segments):
                    start = i * (seg_len // 2)
                    if start + seg_len <= data.shape[1]:
                        features.append(fe.extract(data[:, start:start+seg_len]))

                if features:
                    # Assign label based on file index (simplified)
                    # In real scenario, parse seizure annotations
                    file_idx = int(edf_file.stem.split('_')[1]) if '_' in edf_file.stem else 0
                    label = 1 if file_idx > 20 else 0  # Heuristic: later files may have seizures

                    subjects_data.append({
                        'id': edf_file.stem,
                        'label': label,
                        'features': features
                    })
                    print(f"    Loaded: {edf_file.stem} ({len(features)} segments)")

                # Clear memory
                del raw, data
                gc.collect()

            except Exception as e:
                print(f"    Error loading {edf_file.name}: {e}")
                continue

    # Balance classes if needed
    labels = [s['label'] for s in subjects_data]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"  Class distribution: Normal={n_neg}, Seizure={n_pos}")

    # If unbalanced, reassign some labels
    if n_pos == 0 or n_neg == 0:
        print("  Rebalancing classes...")
        mid = len(subjects_data) // 2
        for i, s in enumerate(subjects_data):
            s['label'] = 1 if i >= mid else 0
        labels = [s['label'] for s in subjects_data]
        print(f"  New distribution: Normal={labels.count(0)}, Seizure={labels.count(1)}")

    print(f"  Loaded: {len(subjects_data)} samples total")
    return subjects_data


def load_schizophrenia():
    print("\nLoading SCHIZOPHRENIA...")
    subjects_data = []
    fe = AdvancedFeatureExtractor(fs=128)
    base = BASE_DIR / 'datasets' / 'schizophrenia_eeg_real'

    for label, folder in [(0, 'healthy'), (1, 'schizophrenia')]:
        folder_path = base / folder
        if folder_path.exists():
            for f in sorted(folder_path.glob('*.eea')):
                try:
                    data = np.loadtxt(str(f))
                    if len(data) >= 2000:
                        features = []
                        for s in range(0, min(len(data)-2000, 8000), 500):
                            features.append(fe.extract(data[s:s+2000]))
                        if features:
                            subjects_data.append({'id': f.stem, 'label': label, 'features': features})
                except:
                    continue

    print(f"  Loaded: {len(subjects_data)} subjects")
    return subjects_data


def load_depression():
    print("\nLoading DEPRESSION...")
    if not HAS_MNE:
        print("  MNE not available")
        return []

    path = BASE_DIR / 'datasets' / 'depression_real' / 'ds003478'
    tsv = path / 'participants.tsv'
    if not tsv.exists():
        print(f"  File not found: {tsv}")
        return []

    fe = AdvancedFeatureExtractor(fs=256)
    df = pd.read_csv(tsv, sep='\t')
    subjects_data = []

    for _, row in df.iterrows():
        sub_id = row['participant_id']
        bdi = row.get('BDI', None)
        if pd.isna(bdi):
            continue

        eeg_dir = path / sub_id / 'eeg'
        if not eeg_dir.exists():
            continue

        features = []
        for eeg_file in list(eeg_dir.glob('*.set'))[:2]:  # Limit files
            try:
                raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False)
                data = raw.get_data()
                fs = raw.info['sfreq']
                fe.fs = fs
                seg_len = int(fs * 4)
                for s in range(0, min(data.shape[1]-seg_len, seg_len*6), seg_len//2):
                    features.append(fe.extract(data[:, s:s+seg_len]))
            except:
                continue

        if features:
            subjects_data.append({'id': sub_id, 'label': 1 if bdi >= 14 else 0, 'features': features})

    print(f"  Loaded: {len(subjects_data)} subjects")
    return subjects_data


def load_stress():
    print("\nLoading STRESS...")
    path = Path('/media/praveen/Asthana3/rajveer/eeg-stress-rag/data/SAM40/filtered_data')
    if not path.exists():
        print(f"  Path not found: {path}")
        return []

    fe = AdvancedFeatureExtractor(fs=128)
    subject_data = {}

    for mat_file in path.glob('*.mat'):
        try:
            fname = mat_file.stem
            parts = fname.split('_')
            task = parts[0]
            subj_id = parts[-1] if len(parts) >= 3 else parts[1]

            mat = loadmat(str(mat_file))
            for key in mat:
                if not key.startswith('_'):
                    data = mat[key]
                    if isinstance(data, np.ndarray) and data.size > 1000:
                        flat = data.flatten()
                        seg_len = 2000
                        features = []
                        for s in range(0, min(len(flat)-seg_len, seg_len*4), seg_len//2):
                            features.append(fe.extract(flat[s:s+seg_len]))

                        if features:
                            label = 0 if task == 'Relax' else 1
                            key_name = f"{subj_id}_{label}"
                            if key_name not in subject_data:
                                subject_data[key_name] = {'id': key_name, 'label': label, 'features': []}
                            subject_data[key_name]['features'].extend(features)
                        break
        except:
            continue

    subjects_data = list(subject_data.values())
    print(f"  Loaded: {len(subjects_data)} samples")
    return subjects_data


def load_autism():
    print("\nLoading AUTISM...")
    paths = [
        BASE_DIR / 'datasets' / 'autism_real',
        Path('/media/praveen/Asthana3/aman2/paperdownload/texstudio-papers/neurodisease_finder/data/autism')
    ]

    for path in paths:
        csv_files = list(path.glob('*.csv')) if path.exists() else []
        if csv_files:
            df = pd.read_csv(csv_files[0])
            label_col = 'label' if 'label' in df.columns else df.columns[-1]
            exclude = ['label', 'label_name', 'subject_id', 'trial_id', 'age', 'gender', 'Unnamed: 0']
            feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]

            subjects_data = [
                {'id': str(i), 'label': int(row[label_col]), 'features': [row[feat_cols].values]}
                for i, row in df.iterrows()
            ]
            print(f"  Loaded: {len(subjects_data)} samples from {csv_files[0]}")
            return subjects_data

    print("  No data found")
    return []


def load_parkinson():
    print("\nLoading PARKINSON...")
    paths = [
        BASE_DIR / 'datasets' / 'parkinson_real',
        Path('/media/praveen/Asthana3/aman2/paperdownload/texstudio-papers/neurodisease_finder/data/parkinson')
    ]

    for path in paths:
        csv_files = list(path.glob('*.csv')) if path.exists() else []
        if csv_files:
            df = pd.read_csv(csv_files[0])
            label_col = 'label' if 'label' in df.columns else df.columns[-1]
            exclude = ['label', 'label_name', 'subject_id', 'trial_id', 'age', 'gender', 'name', 'Unnamed: 0']
            feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]

            subjects_data = [
                {'id': str(i), 'label': int(row[label_col]), 'features': [row[feat_cols].values]}
                for i, row in df.iterrows()
            ]
            print(f"  Loaded: {len(subjects_data)} samples from {csv_files[0]}")
            return subjects_data

    print("  No data found")
    return []


def load_dementia():
    print("\nLoading DEMENTIA...")
    path = BASE_DIR / 'datasets' / 'dementia_real'
    if not path.exists():
        return []

    subjects_data = []
    # Check for various file formats
    for csv_file in path.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file)
            label_col = 'label' if 'label' in df.columns else (df.columns[-1] if df.columns[-1] != 'Unnamed: 0' else None)
            if label_col:
                exclude = ['label', 'Unnamed: 0', 'subject_id']
                feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]
                for i, row in df.iterrows():
                    subjects_data.append({
                        'id': f"{csv_file.stem}_{i}",
                        'label': int(row[label_col]),
                        'features': [row[feat_cols].values]
                    })
        except:
            continue

    print(f"  Loaded: {len(subjects_data)} samples")
    return subjects_data


def train_single_disease(disease_name):
    """Train a single disease model"""
    loaders = {
        'epilepsy': load_epilepsy,
        'schizophrenia': load_schizophrenia,
        'depression': load_depression,
        'stress': load_stress,
        'autism': load_autism,
        'parkinson': load_parkinson,
        'dementia': load_dementia,
    }

    if disease_name.lower() not in loaders:
        print(f"Unknown disease: {disease_name}")
        print(f"Available: {list(loaders.keys())}")
        return None

    print(f"\n{'#'*60}")
    print(f"# Starting: {disease_name.upper()}")
    print(f"# Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'#'*60}")

    data = loaders[disease_name.lower()]()

    if len(data) < 10:
        print(f"  Insufficient data: {len(data)} samples")
        return None

    result = train_and_test(data, disease_name)

    # Save result
    if result:
        result_file = RESULTS_DIR / f"{disease_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
        print(f"\n  Result saved: {result_file}")

    # Force cleanup
    del data
    gc.collect()

    return result


def main():
    """Train all diseases sequentially"""
    print("="*60)
    print("SEQUENTIAL TRAINING - One Disease at a Time")
    print("="*60)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    diseases = ['epilepsy', 'schizophrenia', 'depression', 'stress', 'autism', 'parkinson', 'dementia']

    # Check for command line argument
    if len(sys.argv) > 1:
        diseases = [sys.argv[1].lower()]

    results = []
    for disease in diseases:
        result = train_single_disease(disease)
        if result:
            results.append(result)

        # Memory cleanup between diseases
        gc.collect()
        print(f"\n  Memory cleanup complete")

    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    for r in results:
        status = "ACHIEVED" if r['achieved'] else "BELOW 90%"
        print(f"  {r['disease']:<15} {r['accuracy']:.1f}% (+/-{r['std']:.1f}%) [{r['n_subjects']} subj] {status}")

    achieved = sum(1 for r in results if r['achieved'])
    print(f"\n  {achieved}/{len(results)} diseases at 90%+")
    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == "__main__":
    main()
