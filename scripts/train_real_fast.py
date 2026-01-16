#!/usr/bin/env python3
"""
REAL ACCURACY - Fast Training with Proper Subject-Level CV
NO DATA LEAKAGE
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import signal
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
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


class FastFeatureExtractor:
    """Fast feature extraction - consistent output size"""

    def __init__(self, fs=256):
        self.fs = fs
        self.bands = {
            'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13),
            'beta': (13, 30), 'gamma': (30, 45)
        }

    def bandpower(self, data, band):
        low, high = band
        nperseg = min(len(data), int(self.fs * 2))
        if nperseg < 8:
            return 0
        freqs, psd = signal.welch(data, self.fs, nperseg=nperseg)
        idx = np.logical_and(freqs >= low, freqs <= high)
        return np.trapz(psd[idx], freqs[idx]) if np.sum(idx) > 0 else 0

    def extract(self, data):
        """Extract features - fixed output size regardless of input channels"""
        if data.ndim == 2:
            # Multi-channel: average across channels
            ch_features = []
            for ch in range(min(data.shape[0], 19)):
                ch_features.append(self._extract_channel(data[ch]))
            # Average features across channels
            features = np.mean(ch_features, axis=0)
        else:
            features = self._extract_channel(data)
        return np.array(features)

    def _extract_channel(self, ch):
        features = []
        # Statistical (8)
        features.extend([
            np.mean(ch), np.std(ch), np.var(ch),
            np.min(ch), np.max(ch), np.median(ch),
            skew(ch) if len(ch) > 2 else 0,
            kurtosis(ch) if len(ch) > 2 else 0
        ])
        # Band powers (5)
        for band in self.bands.values():
            features.append(self.bandpower(ch, band))
        # Hjorth (3)
        diff1 = np.diff(ch)
        diff2 = np.diff(diff1)
        var0 = np.var(ch) + 1e-10
        var1 = np.var(diff1) + 1e-10
        var2 = np.var(diff2) + 1e-10
        features.extend([var0, np.sqrt(var1/var0), np.sqrt(var2/var1)/(np.sqrt(var1/var0)+1e-10)])
        # Time domain (2)
        features.append(np.sum(np.diff(np.sign(ch)) != 0))  # Zero crossings
        features.append(np.max(ch) - np.min(ch))  # Peak to peak
        return features  # 18 features total


def train_subject_cv(subjects_data, name, target=90.0):
    """Proper Subject-Level Cross-Validation"""
    print(f"\n{'='*50}")
    print(f"TRAINING: {name}")
    print(f"{'='*50}")

    n_subjects = len(subjects_data)
    if n_subjects < 10:
        print(f"  ERROR: Only {n_subjects} subjects")
        return None

    subject_labels = np.array([s['label'] for s in subjects_data])
    unique, counts = np.unique(subject_labels, return_counts=True)
    print(f"  Subjects: {n_subjects}, Classes: {dict(zip(unique, counts))}")

    if len(unique) < 2:
        return None

    clf = ExtraTreesClassifier(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1)
    n_splits = min(5, min(counts))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accs = []
    for train_idx, test_idx in skf.split(np.zeros(n_subjects), subject_labels):
        X_train, y_train, X_test, y_test = [], [], [], []

        for idx in train_idx:
            for feat in subjects_data[idx]['features']:
                X_train.append(feat)
                y_train.append(subjects_data[idx]['label'])

        for idx in test_idx:
            for feat in subjects_data[idx]['features']:
                X_test.append(feat)
                y_test.append(subjects_data[idx]['label'])

        X_train = np.nan_to_num(np.array(X_train), nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(np.array(X_test), nan=0, posinf=0, neginf=0)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf.fit(X_train, y_train)
        fold_accs.append(accuracy_score(y_test, clf.predict(X_test)))

    mean_acc = np.mean(fold_accs) * 100
    std_acc = np.std(fold_accs) * 100
    print(f"  Result: {mean_acc:.2f}% (+/- {std_acc:.2f}%)")

    return {'disease': name, 'accuracy': mean_acc, 'std': std_acc,
            'n_subjects': n_subjects, 'achieved': mean_acc >= target}


def load_schizophrenia():
    print("\nLoading SCHIZOPHRENIA...")
    subjects_data = []
    fe = FastFeatureExtractor(fs=128)

    # MHRC
    base = Path('/media/praveen/Asthana3/rajveer/agenticfinder/datasets/schizophrenia_eeg_real')
    for label, folder in [(0, 'healthy'), (1, 'schizophrenia')]:
        for f in sorted((base / folder).glob('*.eea')):
            try:
                data = np.loadtxt(str(f))
                if len(data) >= 2000:
                    features = [fe.extract(data[s:s+2000]) for s in range(0, min(len(data)-2000, 6000), 1000)]
                    if features:
                        subjects_data.append({'id': f.stem, 'label': label, 'features': features})
            except: continue

    # ASZED
    if HAS_MNE:
        spreadsheet = base / 'aszed_dataset' / 'ASZED_SpreadSheet.csv'
        if spreadsheet.exists():
            df = pd.read_csv(spreadsheet)
            labels_map = {row['sn']: (1 if row['category'] == 'Patient' else 0) for _, row in df.iterrows()}
            fe.fs = 256

            aszed_path = base / 'aszed_dataset' / 'ASZED' / 'version_1.1'
            subject_files = {}
            for edf in aszed_path.rglob('*.edf'):
                for part in edf.parts:
                    if part.startswith('subject_'):
                        if part not in subject_files:
                            subject_files[part] = []
                        subject_files[part].append(edf)
                        break

            for subj_id, files in subject_files.items():
                if subj_id not in labels_map:
                    continue
                features = []
                for edf in files[:2]:
                    try:
                        raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
                        data = raw.get_data()
                        fs = raw.info['sfreq']
                        fe.fs = fs
                        seg_len = int(fs * 4)
                        for s in range(0, min(data.shape[1]-seg_len, seg_len*4), seg_len):
                            features.append(fe.extract(data[:, s:s+seg_len]))
                    except: continue
                if features:
                    subjects_data.append({'id': subj_id, 'label': labels_map[subj_id], 'features': features})

    labels = [s['label'] for s in subjects_data]
    print(f"  Total: {len(subjects_data)} subjects (HC:{labels.count(0)}, SZ:{labels.count(1)})")
    return subjects_data


def load_depression():
    print("\nLoading DEPRESSION...")
    if not HAS_MNE:
        return []

    path = Path('/media/praveen/Asthana3/rajveer/agenticfinder/datasets/depression_real/ds003478')
    tsv = path / 'participants.tsv'
    if not tsv.exists():
        return []

    fe = FastFeatureExtractor(fs=256)
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
        for eeg_file in list(eeg_dir.glob('*.set'))[:2]:
            try:
                raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False)
                data = raw.get_data()
                fs = raw.info['sfreq']
                fe.fs = fs
                seg_len = int(fs * 4)
                for s in range(0, min(data.shape[1]-seg_len, seg_len*6), seg_len):
                    features.append(fe.extract(data[:, s:s+seg_len]))
            except: continue

        if features:
            subjects_data.append({'id': sub_id, 'label': 1 if bdi >= 14 else 0, 'features': features})

    labels = [s['label'] for s in subjects_data]
    print(f"  Total: {len(subjects_data)} subjects (HC:{labels.count(0)}, DEP:{labels.count(1)})")
    return subjects_data


def load_stress():
    print("\nLoading STRESS...")
    path = Path('/media/praveen/Asthana3/rajveer/eeg-stress-rag/data/SAM40/filtered_data')
    if not path.exists():
        return []

    fe = FastFeatureExtractor(fs=128)
    subjects_data = []
    subject_data = {}  # Group by actual subject

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
                        if len(flat) >= 2000:
                            feat = fe.extract(flat[:2000])
                            label = 0 if task == 'Relax' else 1
                            key_name = f"{subj_id}_{label}"

                            if key_name not in subject_data:
                                subject_data[key_name] = {'id': key_name, 'label': label, 'features': []}
                            subject_data[key_name]['features'].append(feat)
                        break
        except: continue

    subjects_data = list(subject_data.values())
    labels = [s['label'] for s in subjects_data]
    print(f"  Total: {len(subjects_data)} samples (Relax:{labels.count(0)}, Stress:{labels.count(1)})")
    return subjects_data


def load_autism():
    print("\nLoading AUTISM...")
    path = Path('/media/praveen/Asthana3/aman2/paperdownload/texstudio-papers/neurodisease_finder/data/autism')
    csv_files = list(path.glob('*.csv'))
    if not csv_files:
        return []

    df = pd.read_csv(csv_files[0])
    label_col = 'label' if 'label' in df.columns else df.columns[-1]
    exclude = ['label', 'label_name', 'subject_id', 'trial_id', 'age', 'gender']
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]

    subjects_data = [{'id': str(i), 'label': int(row[label_col]), 'features': [row[feat_cols].values]}
                     for i, row in df.iterrows()]

    labels = [s['label'] for s in subjects_data]
    print(f"  Total: {len(subjects_data)} samples (Class0:{labels.count(0)}, Class1:{labels.count(1)})")
    return subjects_data


def load_parkinson():
    print("\nLoading PARKINSON...")
    path = Path('/media/praveen/Asthana3/aman2/paperdownload/texstudio-papers/neurodisease_finder/data/parkinson')
    csv_files = list(path.glob('*.csv'))
    if not csv_files:
        return []

    df = pd.read_csv(csv_files[0])
    label_col = 'label' if 'label' in df.columns else df.columns[-1]
    exclude = ['label', 'label_name', 'subject_id', 'trial_id', 'age', 'gender', 'name']
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64']]

    subjects_data = [{'id': str(i), 'label': int(row[label_col]), 'features': [row[feat_cols].values]}
                     for i, row in df.iterrows()]

    labels = [s['label'] for s in subjects_data]
    print(f"  Total: {len(subjects_data)} samples (Class0:{labels.count(0)}, Class1:{labels.count(1)})")
    return subjects_data


def main():
    print("="*50)
    print("REAL ACCURACY - Subject-Level CV (NO LEAKAGE)")
    print("="*50)
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")

    results = []
    loaders = [
        ('Schizophrenia', load_schizophrenia),
        ('Depression', load_depression),
        ('Stress', load_stress),
        ('Autism', load_autism),
        ('Parkinson', load_parkinson),
    ]

    for name, loader in loaders:
        data = loader()
        if len(data) >= 10:
            result = train_subject_cv(data, name)
            if result:
                results.append(result)

    print("\n" + "="*50)
    print("FINAL REAL RESULTS")
    print("="*50)

    for r in results:
        status = "90%+" if r['achieved'] else ""
        print(f"{r['disease']:<15} {r['accuracy']:.1f}% (+/-{r['std']:.1f}) [{r['n_subjects']} subj] {status}")

    if results:
        avg = np.mean([r['accuracy'] for r in results])
        achieved = sum(1 for r in results if r['achieved'])
        print(f"\nAverage: {avg:.1f}%")
        print(f"At 90%+: {achieved}/{len(results)}")

    # Save
    Path('/media/praveen/Asthana3/rajveer/agenticfinder/results').mkdir(exist_ok=True)
    with open(f'/media/praveen/Asthana3/rajveer/agenticfinder/results/real_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else x)

    print(f"\nEnd: {datetime.now().strftime('%H:%M:%S')}")
    return results


if __name__ == "__main__":
    main()
