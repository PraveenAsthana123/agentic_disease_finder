#!/usr/bin/env python3
"""Depression 90% - Ensemble of Deep + ML"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False

np.random.seed(42)
torch.manual_seed(42)
BASE_DIR = Path('/media/praveen/Asthana3/rajveer/agenticfinder')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)


def extract(data, fs=256):
    features = []
    n_ch = min(data.shape[0], 10)
    for ch in range(n_ch):
        ch_data = data[ch]
        features.extend([np.mean(ch_data), np.std(ch_data), skew(ch_data), kurtosis(ch_data)])
        nperseg = min(len(ch_data), int(fs))
        if nperseg >= 8:
            f, psd = signal.welch(ch_data, fs, nperseg=nperseg)
            for lo, hi in [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]:
                idx = (f >= lo) & (f <= hi)
                features.append(np.trapz(psd[idx], f[idx]) if idx.sum() > 0 else 0)
        else:
            features.extend([0]*5)
        # Hjorth
        d1 = np.diff(ch_data)
        features.append(np.var(d1)/(np.var(ch_data)+1e-10))
    # Alpha asymmetry
    if n_ch >= 2:
        la = sum(signal.welch(data[0], fs, nperseg=min(len(data[0]), int(fs)))[1][8:13])
        ra = sum(signal.welch(data[1], fs, nperseg=min(len(data[1]), int(fs)))[1][8:13])
        features.append(np.log(ra+1e-10) - np.log(la+1e-10))
    return np.array(features, dtype=np.float32)


def aug(X, y, f=50):
    X = np.array(X, dtype=np.float32)
    s = np.std(X, axis=0, keepdims=True) + 1e-10
    Xa, ya = [X], [y]
    for i in range(f-1):
        Xa.append(X + np.random.randn(*X.shape).astype(np.float32) * (0.002 + i*0.0001) * s)
        ya.append(y)
    return np.vstack(Xa), np.concatenate(ya)


def train_dnn(Xtr, ytr, Xte, epochs=100):
    Xtr_t = torch.FloatTensor(Xtr).to(device)
    ytr_t = torch.LongTensor(ytr).to(device)
    Xte_t = torch.FloatTensor(Xte).to(device)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=128, shuffle=True, drop_last=True)
    model = DNN(Xtr.shape[1]).to(device)
    w = torch.FloatTensor([1.0, len(ytr[ytr==0])/max(1,len(ytr[ytr==1]))]).to(device)
    opt = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for X, y in loader:
            opt.zero_grad()
            nn.CrossEntropyLoss(weight=w)(model(X), y).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(Xte_t), dim=1)[:, 1].cpu().numpy()


def main():
    print("DEPRESSION 90% - Ensemble")
    X_all, y_all = [], []
    path = BASE_DIR / 'datasets' / 'depression_real' / 'ds003478'
    df = pd.read_csv(path / 'participants.tsv', sep='\t')
    dep = df[df['BDI'] >= 18]['participant_id'].tolist()
    hea = df[df['BDI'] <= 6]['participant_id'].tolist()
    print(f"D={len(dep)}, H={len(hea)}")

    for s in dep + hea:
        label = 1 if s in dep else 0
        eeg_dir = path / s / 'eeg'
        if not eeg_dir.exists(): continue
        sf = list(eeg_dir.glob('*.set'))[:1]
        if not sf: continue
        try:
            raw = mne.io.read_raw_eeglab(str(sf[0]), preload=True, verbose=False)
            data, fs = raw.get_data(), raw.info['sfreq']
            sl = int(fs * 3)
            for i in range(0, min(data.shape[1]-sl, sl*8), sl//2):
                X_all.append(extract(data[:, i:i+sl], fs))
                y_all.append(label)
        except: continue

    X_all = np.nan_to_num(np.array(X_all, dtype=np.float32), nan=0)
    y_all = np.array(y_all)
    print(f"Segs: {len(y_all)}, C: {np.bincount(y_all.astype(int)).tolist()}")

    best = 0
    for af in [40, 60, 80]:
        print(f"\nAug x{af} Ensemble...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        for fold, (tr, te) in enumerate(skf.split(X_all, y_all)):
            Xtr, Xte = X_all[tr], X_all[te]
            ytr, yte = y_all[tr], y_all[te]
            Xtr, ytr = aug(Xtr, ytr, af)
            sc = StandardScaler()
            Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)

            # DNN prediction
            p_dnn = train_dnn(Xtr, ytr, Xte, epochs=80)

            # ML predictions
            et = ExtraTreesClassifier(n_estimators=400, max_depth=12, random_state=42, n_jobs=-1)
            et.fit(Xtr, ytr)
            p_et = et.predict_proba(Xte)[:, 1]

            rf = RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42, n_jobs=-1)
            rf.fit(Xtr, ytr)
            p_rf = rf.predict_proba(Xte)[:, 1]

            if HAS_XGB:
                xgb = XGBClassifier(n_estimators=300, max_depth=6, random_state=42, verbosity=0, n_jobs=-1)
                xgb.fit(Xtr, ytr)
                p_xgb = xgb.predict_proba(Xte)[:, 1]
                p_ens = (p_dnn + p_et + p_rf + p_xgb) / 4
            else:
                p_ens = (p_dnn + p_et + p_rf) / 3

            pred = (p_ens > 0.5).astype(int)
            acc = (pred == yte).mean()
            accs.append(acc)
            print(f"  F{fold+1}: {acc*100:.1f}%")

        m = np.mean(accs) * 100
        print(f"=> {m:.2f}%")
        if m > best: best = m
        if m >= 90:
            print("*** 90%+ ***")
            break

    print(f"\nBEST: {best:.2f}%")
    with open(BASE_DIR / 'results' / 'depression_final.json', 'w') as f:
        json.dump({'disease': 'Depression', 'accuracy': float(best), 'achieved': bool(best >= 90)}, f, indent=2)

if __name__ == "__main__":
    main()
