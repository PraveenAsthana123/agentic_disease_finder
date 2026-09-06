#!/usr/bin/env python3
"""
Multi-Disease EEG Classification Training Script
AgenticFinder - 6 Disease Detection System

Diseases: Schizophrenia, Autism, Parkinson, Stress, Epilepsy, Depression
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from scipy import signal
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, '/media/praveen/Asthana3/rajveer/agenticfinder')
from config.dataset_loader import DatasetConfig

# Try importing optional libraries
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    print("Warning: MNE not available, using basic EDF loading")


class EEGFeatureExtractor:
    """Extract features from EEG signals"""

    def __init__(self, fs=256):
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
        nperseg = min(len(data), self.fs * 2)
        freqs, psd = signal.welch(data, self.fs, nperseg=nperseg)
        idx = np.logical_and(freqs >= low, freqs <= high)
        return np.trapz(psd[idx], freqs[idx])

    def extract_features(self, eeg_data):
        """Extract features from EEG epoch"""
        features = []

        # Handle multi-channel data
        if eeg_data.ndim == 1:
            channels = [eeg_data]
        else:
            channels = eeg_data

        for ch_data in channels:
            # Statistical features
            features.extend([
                np.mean(ch_data),
                np.std(ch_data),
                np.var(ch_data),
                np.min(ch_data),
                np.max(ch_data),
                np.median(ch_data),
            ])

            # Band powers
            for band_name, band_range in self.bands.items():
                try:
                    bp = self.bandpower(ch_data, band_range)
                    features.append(bp)
                except:
                    features.append(0)

            # Hjorth parameters
            try:
                diff1 = np.diff(ch_data)
                diff2 = np.diff(diff1)
                activity = np.var(ch_data)
                mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
                complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
                features.extend([activity, mobility, complexity])
            except:
                features.extend([0, 0, 0])

        return np.array(features)


class MultiDiseaseDataLoader:
    """Load and preprocess data from all 6 diseases"""

    def __init__(self):
        self.config = DatasetConfig()
        self.feature_extractor = EEGFeatureExtractor()
        self.diseases = ['schizophrenia', 'autism', 'parkinson', 'stress', 'epilepsy', 'depression']

    def load_csv_data(self, path, file_name=None):
        """Load CSV dataset"""
        if file_name:
            csv_path = Path(path) / file_name
        else:
            csv_files = list(Path(path).glob('*.csv'))
            if not csv_files:
                return None, None
            csv_path = csv_files[0]

        if not csv_path.exists():
            return None, None

        try:
            df = pd.read_csv(csv_path)

            # Try to identify label column
            label_cols = [c for c in df.columns if c.lower() in ['label', 'class', 'target', 'diagnosis', 'y']]
            if label_cols:
                y = df[label_cols[0]].values
                X = df.drop(columns=label_cols)
            else:
                # Assume last column is label
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1].values

            # Keep only numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X = X[numeric_cols].values

            # Convert y to numeric if possible
            if y.dtype == object:
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Handle NaN
            X = np.nan_to_num(X, nan=0)

            return X, y
        except Exception as e:
            print(f"    Error loading CSV: {e}")
            return None, None

    def load_mat_data(self, path):
        """Load MAT files or NPY/NPZ files"""
        path = Path(path)

        # First check for pre-processed NPY/NPZ files
        npy_files = list(path.glob('**/*.npy')) + list(path.glob('**/*.npz'))
        if npy_files:
            try:
                # Look for X and y files
                x_files = [f for f in npy_files if 'X' in f.name or 'x' in f.name]
                y_files = [f for f in npy_files if 'y' in f.name or 'Y' in f.name]

                if x_files and y_files:
                    X = np.load(str(x_files[0]))
                    y = np.load(str(y_files[0]))
                    if isinstance(X, np.lib.npyio.NpzFile):
                        X = X[list(X.keys())[0]]
                    if isinstance(y, np.lib.npyio.NpzFile):
                        y = y[list(y.keys())[0]]

                    # Handle 3D data (samples, channels, time)
                    if X.ndim == 3:
                        # Extract features from each sample
                        X_features = []
                        for sample in X:
                            features = self.feature_extractor.extract_features(sample)
                            X_features.append(features)
                        X = np.array(X_features)

                    return X, y
            except Exception as e:
                print(f"    Error loading NPY: {e}")

        # Load MAT files
        mat_files = list(path.glob('**/*.mat'))
        if not mat_files:
            return None, None

        X_list, y_list = [], []
        for mat_file in mat_files[:100]:  # Limit for speed
            try:
                mat_data = loadmat(str(mat_file))
                # Find data array
                for key in mat_data:
                    if not key.startswith('_'):
                        data = mat_data[key]
                        if isinstance(data, np.ndarray) and data.size > 10:
                            features = self.feature_extractor.extract_features(data.flatten()[:1000])
                            X_list.append(features)
                            # Label based on filename
                            label = 1 if 'stress' in mat_file.name.lower() or 'stroop' in mat_file.name.lower() else 0
                            y_list.append(label)
                            break
            except Exception as e:
                continue

        if X_list:
            return np.array(X_list), np.array(y_list)
        return None, None

    def load_edf_data(self, path, max_files=20):
        """Load EDF files"""
        edf_files = list(Path(path).glob('**/*.edf'))
        if not edf_files:
            return None, None

        X_list, y_list = [], []

        for edf_file in edf_files[:max_files]:
            try:
                if HAS_MNE:
                    raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
                    data = raw.get_data()
                    fs = raw.info['sfreq']
                else:
                    # Basic EDF loading without MNE
                    continue

                # Extract features from each channel
                self.feature_extractor.fs = fs

                # Use first 10 seconds
                samples = int(min(10 * fs, data.shape[1]))
                epoch_data = data[:, :samples]

                features = self.feature_extractor.extract_features(epoch_data)
                X_list.append(features)

                # Label based on path or filename
                label = 1  # Disease present
                y_list.append(label)

            except Exception as e:
                continue

        if X_list:
            # Pad features to same length
            max_len = max(len(f) for f in X_list)
            X_padded = [np.pad(f, (0, max_len - len(f))) for f in X_list]
            return np.array(X_padded), np.array(y_list)
        return None, None

    def load_eea_data(self, healthy_path, disease_path, max_files=50):
        """Load EEA files (MHRC format)"""
        X_list, y_list = [], []

        # Load healthy controls
        healthy_files = list(Path(healthy_path).glob('*.eea'))
        for eea_file in healthy_files[:max_files]:
            try:
                data = np.loadtxt(str(eea_file))
                features = self.feature_extractor.extract_features(data[:2000])  # First 2000 samples
                X_list.append(features)
                y_list.append(0)  # Healthy
            except Exception as e:
                continue

        # Load schizophrenia patients
        disease_files = list(Path(disease_path).glob('*.eea'))
        for eea_file in disease_files[:max_files]:
            try:
                data = np.loadtxt(str(eea_file))
                features = self.feature_extractor.extract_features(data[:2000])
                X_list.append(features)
                y_list.append(1)  # Disease
            except Exception as e:
                continue

        if X_list:
            return np.array(X_list), np.array(y_list)
        return None, None

    def load_disease_data(self, disease):
        """Load data for a specific disease"""
        datasets = self.config.get_disease_datasets(disease)

        X_all, y_all = [], []

        # Special handling for schizophrenia MHRC data
        if disease == 'schizophrenia':
            base_path = Path('/media/praveen/Asthana3/rajveer/agenticfinder/datasets/schizophrenia_eeg_real')
            healthy_path = base_path / 'healthy'
            sz_path = base_path / 'schizophrenia'
            if healthy_path.exists() and sz_path.exists():
                print(f"  Loading MHRC (EEA)...")
                X, y = self.load_eea_data(healthy_path, sz_path)
                if X is not None:
                    X_all.append(X)
                    y_all.append(y)
                    print(f"    Loaded {len(X)} samples")

        # Special handling for stress data
        if disease == 'stress':
            stress_path = Path('/media/praveen/Asthana3/rajveer/eeg-stress-rag/data/SAM40/sample_100')
            if stress_path.exists():
                print(f"  Loading SAM40 preprocessed (NPY)...")
                X, y = self.load_mat_data(stress_path)
                if X is not None:
                    X_all.append(X)
                    y_all.append(y)
                    print(f"    Loaded {len(X)} samples")

        for ds in datasets:
            path = ds['path']
            file_name = ds.get('file_name')
            data_format = ds.get('format', 'CSV')

            print(f"  Loading {ds['name']} ({data_format})...")

            X, y = None, None

            if 'CSV' in data_format.upper():
                X, y = self.load_csv_data(path, file_name)
            elif 'MAT' in data_format.upper():
                X, y = self.load_mat_data(path)
            elif 'EDF' in data_format.upper():
                X, y = self.load_edf_data(path)

            if X is not None and len(X) > 0:
                X_all.append(X)
                y_all.append(y)
                print(f"    Loaded {len(X)} samples")

        if X_all:
            # Combine all datasets for this disease
            # Need to handle different feature dimensions
            max_features = max(x.shape[1] if x.ndim > 1 else len(x[0]) for x in X_all)

            X_combined = []
            y_combined = []
            for X, y in zip(X_all, y_all):
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                # Pad to max features
                if X.shape[1] < max_features:
                    X = np.pad(X, ((0, 0), (0, max_features - X.shape[1])))
                X_combined.append(X)
                y_combined.append(y)

            return np.vstack(X_combined), np.concatenate(y_combined)

        return None, None

    def load_all_data(self):
        """Load data from all 6 diseases"""
        all_X, all_y, all_diseases = [], [], []

        for i, disease in enumerate(self.diseases):
            print(f"\nLoading {disease.upper()} data...")
            X, y = self.load_disease_data(disease)

            if X is not None and len(X) > 0:
                all_X.append(X)
                # Create disease label
                disease_labels = np.full(len(X), i)
                all_diseases.append(disease_labels)
                all_y.append(y)
                print(f"  Total: {len(X)} samples for {disease}")

        if all_X:
            # Pad all to same feature dimension
            max_features = max(x.shape[1] for x in all_X)
            X_padded = []
            for X in all_X:
                if X.shape[1] < max_features:
                    X = np.pad(X, ((0, 0), (0, max_features - X.shape[1])))
                X_padded.append(X)

            return (np.vstack(X_padded),
                    np.concatenate(all_diseases),
                    self.diseases)

        return None, None, None


class MultiDiseaseClassifier:
    """Ensemble classifier for multi-disease detection"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Ensemble of classifiers
        self.classifiers = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        }

        # Voting ensemble
        self.ensemble = VotingClassifier(
            estimators=[(name, clf) for name, clf in self.classifiers.items()],
            voting='soft'
        )

    def train(self, X, y, diseases):
        """Train the ensemble classifier"""
        print("\n" + "="*60)
        print("TRAINING MULTI-DISEASE CLASSIFIER")
        print("="*60)

        # Handle NaN and Inf
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\nTraining samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Classes: {len(np.unique(y_encoded))}")

        results = {}

        # Train individual classifiers
        for name, clf in self.classifiers.items():
            print(f"\nTraining {name.upper()}...")
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results[name] = {'accuracy': acc, 'f1': f1}
            print(f"  Accuracy: {acc*100:.2f}%")
            print(f"  F1 Score: {f1:.4f}")

        # Train ensemble
        print(f"\nTraining ENSEMBLE...")
        self.ensemble.fit(X_train_scaled, y_train)
        y_pred_ensemble = self.ensemble.predict(X_test_scaled)
        acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
        f1_ensemble = f1_score(y_test, y_pred_ensemble, average='weighted')
        results['ensemble'] = {'accuracy': acc_ensemble, 'f1': f1_ensemble}

        print(f"  Accuracy: {acc_ensemble*100:.2f}%")
        print(f"  F1 Score: {f1_ensemble:.4f}")

        # Per-disease accuracy
        print("\n" + "="*60)
        print("PER-DISEASE ACCURACY")
        print("="*60)

        disease_results = {}
        for i, disease in enumerate(diseases):
            mask = y_test == i
            if mask.sum() > 0:
                disease_acc = accuracy_score(y_test[mask], y_pred_ensemble[mask])
                disease_results[disease] = disease_acc
                print(f"{disease.upper():15} : {disease_acc*100:.2f}%")

        # Classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_test, y_pred_ensemble,
                                    target_names=diseases[:len(np.unique(y_encoded))]))

        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred_ensemble)
        print(cm)

        return {
            'overall_accuracy': acc_ensemble,
            'overall_f1': f1_ensemble,
            'per_disease': disease_results,
            'individual_models': results,
            'confusion_matrix': cm.tolist()
        }


def main():
    """Main training function"""
    print("="*60)
    print("AGENTICFINDER - MULTI-DISEASE EEG CLASSIFICATION")
    print("="*60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    loader = MultiDiseaseDataLoader()
    X, y, diseases = loader.load_all_data()

    if X is None:
        print("Error: No data loaded!")
        return

    print(f"\n{'='*60}")
    print("DATA SUMMARY")
    print("="*60)
    print(f"Total samples: {len(X)}")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Diseases: {diseases}")

    # Distribution
    print("\nClass distribution:")
    for i, disease in enumerate(diseases):
        count = np.sum(y == i)
        print(f"  {disease}: {count} samples")

    # Train classifier
    classifier = MultiDiseaseClassifier()
    results = classifier.train(X, y, diseases)

    # Save results
    results_path = Path('/media/praveen/Asthana3/rajveer/agenticfinder/results')
    results_path.mkdir(exist_ok=True)

    results_file = results_path / f'training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {results['overall_accuracy']*100:.2f}%")
    print(f"Overall F1 Score: {results['overall_f1']:.4f}")
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == "__main__":
    main()
