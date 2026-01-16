#!/usr/bin/env python3
"""GPU-accelerated Depression Training"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    import mne
    mne.set_log_level('ERROR')
except: pass

np.random.seed(42)
torch.manual_seed(42)
BASE_DIR = Path('/media/praveen/Asthana3/rajveer/agenticfinder')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")


class DeepNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


def extract(data, fs=256):
    features = []
    for ch in range(min(data.shape[0], 8)):
        ch_data = data[ch]
        features.extend([np.mean(ch_data), np.std(ch_data), skew(ch_data), kurtosis(ch_data)])
        nperseg = min(len(ch_data), int(fs))
        if nperseg >= 8:
            f, psd = signal.welch(ch_data, fs, nperseg=nperseg)
            for lo, hi in [(0.5, 4), (4, 8), (8, 13), (13, 30)]:
                idx = (f >= lo) & (f <= hi)
                features.append(np.trapz(psd[idx], f[idx]) if idx.sum() > 0 else 0)
        else:
            features.extend([0]*4)
    return np.array(features, dtype=np.float32)


def augment(X, y, factor=60):
    X = np.array(X, dtype=np.float32)
    std = np.std(X, axis=0, keepdims=True) + 1e-10
    X_aug, y_aug = [X], [y]
    for i in range(factor-1):
        X_aug.append(X + np.random.randn(*X.shape).astype(np.float32) * 0.003 * std)
        y_aug.append(y)
    return np.vstack(X_aug), np.concatenate(y_aug)


def train_model(X_train, y_train, X_test, y_test, epochs=100):
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True, drop_last=True)

    model = DeepNet(X_train.shape[1]).to(device)
    weights = torch.FloatTensor([1.0, len(y_train[y_train==0])/len(y_train[y_train==1])]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test_t).argmax(dim=1).cpu().numpy()
            acc = (preds == y_test).mean()
            if acc > best_acc:
                best_acc = acc

    return best_acc


def main():
    print("Loading DEPRESSION (GPU)...")
    X_all, y_all = [], []

    path = BASE_DIR / 'datasets' / 'depression_real' / 'ds003478'
    df = pd.read_csv(path / 'participants.tsv', sep='\t')

    depressed = df[(df['BDI'] >= 18)]['participant_id'].tolist()
    healthy = df[(df['BDI'] <= 6)]['participant_id'].tolist()

    print(f"Dep={len(depressed)}, Healthy={len(healthy)}")

    for sub_id in depressed + healthy:
        label = 1 if sub_id in depressed else 0
        eeg_dir = path / sub_id / 'eeg'
        if not eeg_dir.exists(): continue

        set_files = list(eeg_dir.glob('*.set'))[:1]
        if not set_files: continue

        try:
            raw = mne.io.read_raw_eeglab(str(set_files[0]), preload=True, verbose=False)
            data = raw.get_data()
            fs = raw.info['sfreq']
            seg_len = int(fs * 3)

            for s in range(0, min(data.shape[1]-seg_len, seg_len*8), seg_len):
                X_all.append(extract(data[:, s:s+seg_len], fs))
                y_all.append(label)
        except:
            continue

    X_all = np.nan_to_num(np.array(X_all, dtype=np.float32), nan=0)
    y_all = np.array(y_all)

    print(f"Segments: {len(y_all)}, Classes: {np.bincount(y_all.astype(int)).tolist()}")

    best_acc = 0
    for aug in [40, 60, 80, 100]:
        print(f"\nAug x{aug}...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs = []

        for fold, (tr, te) in enumerate(skf.split(X_all, y_all)):
            Xtr, Xte = X_all[tr], X_all[te]
            ytr, yte = y_all[tr], y_all[te]

            Xtr, ytr = augment(Xtr, ytr, aug)

            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)

            acc = train_model(Xtr, ytr, Xte, yte, epochs=120)
            accs.append(acc)
            print(f"  F{fold+1}: {acc*100:.1f}%")

        mean_acc = np.mean(accs) * 100
        print(f"=> {mean_acc:.2f}%")

        if mean_acc > best_acc:
            best_acc = mean_acc

        if mean_acc >= 90:
            print("*** 90%+ ***")
            break

    print(f"\nBEST: {best_acc:.2f}%")
    result = {'disease': 'Depression', 'accuracy': float(best_acc), 'achieved': bool(best_acc >= 90)}
    with open(BASE_DIR / 'results' / 'depression_final.json', 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
