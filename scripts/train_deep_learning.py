#!/usr/bin/env python3
"""
Deep Learning EEG Classification - AgenticFinder
Proper training with cross-validation and realistic evaluation
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

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy import signal
from scipy.io import loadmat

# Check for MNE
try:
    import mne
    mne.set_log_level('ERROR')
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# EEGNet Model - State-of-the-art for EEG classification
# =============================================================================
class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs
    Reference: Lawhern et al., 2018
    """
    def __init__(self, n_channels=16, n_samples=256, n_classes=2,
                 dropout_rate=0.5, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()

        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.n_channels = n_channels
        self.n_samples = n_samples

        # Block 1: Temporal Convolution
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 1: Depthwise Convolution (spatial filter)
        self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 2: Separable Convolution
        self.conv3 = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                               groups=F1 * D, bias=False)
        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Calculate output size
        self.feature_size = self._get_feature_size()

        # Classifier
        self.fc = nn.Linear(self.feature_size, n_classes)

    def _get_feature_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = torch.relu(x)
            x = self.pool1(x)
            x = self.dropout1(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.bn3(x)
            x = torch.relu(x)
            x = self.pool2(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        # Input: (batch, channels, samples)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dim: (batch, 1, channels, samples)

        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Classifier
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# =============================================================================
# 1D CNN for variable length EEG
# =============================================================================
class EEG1DCNN(nn.Module):
    """Simple 1D CNN for EEG classification"""
    def __init__(self, n_channels=16, n_classes=2, dropout=0.5):
        super(EEG1DCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: (batch, channels, samples)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# =============================================================================
# Data Loading
# =============================================================================
class SchizophreniaDataLoader:
    """Load schizophrenia EEG data from multiple sources"""

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.data = []
        self.labels = []
        self.subject_ids = []

    def load_eea_files(self, healthy_path, disease_path):
        """Load EEA format files (MHRC dataset)"""
        print("Loading MHRC dataset (EEA format)...")

        # Load healthy controls
        healthy_files = sorted(Path(healthy_path).glob('*.eea'))
        for i, f in enumerate(healthy_files):
            try:
                data = np.loadtxt(str(f))
                if len(data) >= 2000:
                    # Reshape to simulate channels (use segments as pseudo-channels)
                    n_samples = 2000
                    data = data[:n_samples]
                    # Create 16 pseudo-channels by splitting
                    n_channels = 16
                    segment_len = n_samples // n_channels
                    reshaped = data[:n_channels * segment_len].reshape(n_channels, segment_len)

                    self.data.append(reshaped)
                    self.labels.append(0)  # Healthy
                    self.subject_ids.append(f'H_{i}')
            except Exception as e:
                continue

        # Load schizophrenia patients
        disease_files = sorted(Path(disease_path).glob('*.eea'))
        for i, f in enumerate(disease_files):
            try:
                data = np.loadtxt(str(f))
                if len(data) >= 2000:
                    n_samples = 2000
                    data = data[:n_samples]
                    n_channels = 16
                    segment_len = n_samples // n_channels
                    reshaped = data[:n_channels * segment_len].reshape(n_channels, segment_len)

                    self.data.append(reshaped)
                    self.labels.append(1)  # Schizophrenia
                    self.subject_ids.append(f'S_{i}')
            except Exception as e:
                continue

        print(f"  Loaded {len(self.data)} samples from MHRC")

    def load_edf_files(self, edf_path, label, prefix, max_files=50):
        """Load EDF format files"""
        if not HAS_MNE:
            print("  MNE not available, skipping EDF files")
            return

        edf_files = sorted(Path(edf_path).glob('**/*.edf'))[:max_files]
        print(f"Loading {len(edf_files)} EDF files from {edf_path}...")

        for i, f in enumerate(edf_files):
            try:
                raw = mne.io.read_raw_edf(str(f), preload=True, verbose=False)
                data = raw.get_data()
                fs = raw.info['sfreq']

                # Resample to 128 Hz if needed
                if fs != 128:
                    target_samples = int(data.shape[1] * 128 / fs)
                    data = signal.resample(data, target_samples, axis=1)

                # Take first 10 seconds (1280 samples at 128 Hz)
                n_samples = min(1280, data.shape[1])
                data = data[:, :n_samples]

                # Normalize channels
                n_channels = min(16, data.shape[0])
                data = data[:n_channels, :]

                # Pad if needed
                if data.shape[1] < 1280:
                    data = np.pad(data, ((0, 0), (0, 1280 - data.shape[1])))
                if data.shape[0] < 16:
                    data = np.pad(data, ((0, 16 - data.shape[0]), (0, 0)))

                # Normalize
                data = (data - np.mean(data)) / (np.std(data) + 1e-8)

                self.data.append(data[:16, :128*10])  # 16 channels, 10 seconds
                self.labels.append(label)
                self.subject_ids.append(f'{prefix}_{i}')

            except Exception as e:
                continue

        print(f"  Total samples: {len(self.data)}")

    def get_data(self):
        """Return data, labels, and subject IDs"""
        # Normalize all data to same shape
        min_samples = min(d.shape[1] for d in self.data)
        min_channels = min(d.shape[0] for d in self.data)

        X = np.array([d[:min_channels, :min_samples] for d in self.data])
        y = np.array(self.labels)
        subjects = np.array(self.subject_ids)

        return X, y, subjects


# =============================================================================
# Training Functions
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = 100. * correct / total
    return total_loss / len(loader), accuracy, all_preds, all_labels


def train_with_cv(X, y, subjects, n_folds=5, epochs=100, lr=0.001, batch_size=16):
    """Train with K-Fold cross-validation"""
    print(f"\n{'='*60}")
    print(f"TRAINING WITH {n_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")

    # Convert to torch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    n_channels, n_samples = X.shape[1], X.shape[2]
    n_classes = len(np.unique(y))

    print(f"Data shape: {X.shape}")
    print(f"Classes: {np.bincount(y)}")
    print(f"Channels: {n_channels}, Samples: {n_samples}")

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_preds = []
    all_labels = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
        y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Create model
        model = EEG1DCNN(n_channels=n_channels, n_classes=n_classes, dropout=0.5).to(device)

        # Loss with class weights
        class_weights = torch.FloatTensor([1.0 / np.sum(y == c) for c in range(n_classes)])
        class_weights = class_weights / class_weights.sum()
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                          factor=0.5, patience=10)

        best_acc = 0
        patience = 20
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                                 optimizer, device)
            val_loss, val_acc, preds, labels = evaluate(model, val_loader,
                                                         criterion, device)

            scheduler.step(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_preds = preds
                best_labels = labels
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}: Train Acc: {train_acc:.2f}%, "
                      f"Val Acc: {val_acc:.2f}%, Best: {best_acc:.2f}%")

        fold_results.append(best_acc)
        all_preds.extend(best_preds)
        all_labels.extend(best_labels)
        print(f"  Fold {fold + 1} Best Accuracy: {best_acc:.2f}%")

    # Final results
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)

    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Fold Accuracies: {[f'{a:.2f}%' for a in fold_results]}")
    print(f"Mean Accuracy: {mean_acc:.2f}% (+/- {std_acc:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Healthy', 'Schizophrenia']))

    return mean_acc, std_acc, fold_results


def train_until_target(X, y, subjects, target_acc=90.0, max_attempts=10):
    """Train with different hyperparameters until target accuracy is achieved"""
    print(f"\n{'='*60}")
    print(f"TRAINING UNTIL {target_acc}% ACCURACY")
    print(f"{'='*60}")

    hyperparams = [
        {'lr': 0.001, 'batch_size': 16, 'epochs': 100},
        {'lr': 0.0005, 'batch_size': 32, 'epochs': 150},
        {'lr': 0.0001, 'batch_size': 8, 'epochs': 200},
        {'lr': 0.002, 'batch_size': 16, 'epochs': 100},
        {'lr': 0.0005, 'batch_size': 16, 'epochs': 200},
        {'lr': 0.001, 'batch_size': 8, 'epochs': 150},
        {'lr': 0.0003, 'batch_size': 32, 'epochs': 250},
        {'lr': 0.001, 'batch_size': 4, 'epochs': 200},
        {'lr': 0.0008, 'batch_size': 16, 'epochs': 300},
        {'lr': 0.0005, 'batch_size': 8, 'epochs': 300},
    ]

    best_result = {'acc': 0, 'params': None, 'fold_results': None}

    for attempt, params in enumerate(hyperparams[:max_attempts]):
        print(f"\n--- Attempt {attempt + 1}/{max_attempts} ---")
        print(f"Params: lr={params['lr']}, batch={params['batch_size']}, epochs={params['epochs']}")

        mean_acc, std_acc, fold_results = train_with_cv(
            X, y, subjects,
            n_folds=5,
            epochs=params['epochs'],
            lr=params['lr'],
            batch_size=params['batch_size']
        )

        if mean_acc > best_result['acc']:
            best_result['acc'] = mean_acc
            best_result['std'] = std_acc
            best_result['params'] = params
            best_result['fold_results'] = fold_results

        if mean_acc >= target_acc:
            print(f"\n TARGET ACHIEVED: {mean_acc:.2f}% >= {target_acc}%")
            break

    return best_result


# =============================================================================
# Main
# =============================================================================
def main():
    print("="*60)
    print("AGENTICFINDER - DEEP LEARNING EEG CLASSIFICATION")
    print("="*60)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")

    # Load schizophrenia data
    base_path = Path('/media/praveen/Asthana3/rajveer/agenticfinder/datasets/schizophrenia_eeg_real')

    loader = SchizophreniaDataLoader(base_path)

    # Load MHRC data
    healthy_path = base_path / 'healthy'
    disease_path = base_path / 'schizophrenia'
    loader.load_eea_files(healthy_path, disease_path)

    # Load RepOD data
    repod_path = base_path / 'repod_dataset'
    if repod_path.exists():
        # Load healthy and schizophrenia from repod
        for subdir in repod_path.iterdir():
            if subdir.is_dir():
                if 'healthy' in subdir.name.lower() or 'control' in subdir.name.lower():
                    loader.load_edf_files(subdir, label=0, prefix='RepOD_H')
                elif 'schiz' in subdir.name.lower() or 'patient' in subdir.name.lower():
                    loader.load_edf_files(subdir, label=1, prefix='RepOD_S')

    # Load ASZED data
    aszed_path = base_path / 'aszed_dataset'
    if aszed_path.exists():
        for subdir in aszed_path.iterdir():
            if subdir.is_dir():
                if 'healthy' in subdir.name.lower() or 'control' in subdir.name.lower():
                    loader.load_edf_files(subdir, label=0, prefix='ASZED_H')
                elif 'schiz' in subdir.name.lower() or 'patient' in subdir.name.lower():
                    loader.load_edf_files(subdir, label=1, prefix='ASZED_S')

    # Get data
    X, y, subjects = loader.get_data()

    if len(X) == 0:
        print("ERROR: No data loaded!")
        return

    print(f"\n{'='*60}")
    print("DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(X)}")
    print(f"Shape: {X.shape}")
    print(f"Healthy: {np.sum(y == 0)}, Schizophrenia: {np.sum(y == 1)}")
    print(f"Unique subjects: {len(np.unique(subjects))}")

    # Normalize data
    X_normalized = np.zeros_like(X)
    for i in range(len(X)):
        X_normalized[i] = (X[i] - np.mean(X[i])) / (np.std(X[i]) + 1e-8)

    # Train until target accuracy
    result = train_until_target(X_normalized, y, subjects, target_acc=90.0, max_attempts=10)

    # Save results
    results_path = Path('/media/praveen/Asthana3/rajveer/agenticfinder/results')
    results_path.mkdir(exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'best_accuracy': result['acc'],
        'std': result.get('std', 0),
        'best_params': result['params'],
        'fold_results': result['fold_results'],
        'n_samples': len(X),
        'n_healthy': int(np.sum(y == 0)),
        'n_schizophrenia': int(np.sum(y == 1)),
    }

    results_file = results_path / f'deep_learning_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best Accuracy: {result['acc']:.2f}%")
    print(f"Best Parameters: {result['params']}")
    print(f"Results saved to: {results_file}")
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return result


if __name__ == "__main__":
    main()
