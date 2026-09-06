#!/usr/bin/env python3
"""Depression - Quick ensemble (DNN + XGB)"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import json, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

try:
    import mne
    mne.set_log_level('ERROR')
except: pass

np.random.seed(42)
torch.manual_seed(42)
BASE = Path('/media/praveen/Asthana3/rajveer/agenticfinder')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2))
    def forward(self, x): return self.net(x)


def feat(data, fs=256):
    f = []
    for ch in range(min(data.shape[0], 8)):
        c = data[ch]
        f.extend([np.mean(c), np.std(c), skew(c), kurtosis(c)])
        np_ = min(len(c), int(fs))
        if np_ >= 8:
            fr, psd = signal.welch(c, fs, nperseg=np_)
            for lo, hi in [(0.5,4),(4,8),(8,13),(13,30),(30,50)]:
                idx = (fr >= lo) & (fr <= hi)
                f.append(np.trapz(psd[idx], fr[idx]) if idx.sum() > 0 else 0)
        else:
            f.extend([0]*5)
    return np.array(f, dtype=np.float32)


def aug(X, y, f=40):
    X = np.array(X, dtype=np.float32)
    s = np.std(X, axis=0, keepdims=True) + 1e-10
    Xa, ya = [X], [y]
    for i in range(f-1):
        Xa.append(X + np.random.randn(*X.shape).astype(np.float32) * 0.003 * s)
        ya.append(y)
    return np.vstack(Xa), np.concatenate(ya)


def train_nn(Xtr, ytr, Xte, ep=60):
    Xtr_t = torch.FloatTensor(Xtr).to(dev)
    ytr_t = torch.LongTensor(ytr).to(dev)
    Xte_t = torch.FloatTensor(Xte).to(dev)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=128, shuffle=True, drop_last=True)
    m = Net(Xtr.shape[1]).to(dev)
    w = torch.FloatTensor([1.0, len(ytr[ytr==0])/max(1,len(ytr[ytr==1]))]).to(dev)
    opt = optim.AdamW(m.parameters(), lr=0.003)
    for _ in range(ep):
        m.train()
        for X, y in loader:
            opt.zero_grad()
            nn.CrossEntropyLoss(weight=w)(m(X), y).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        return torch.softmax(m(Xte_t), 1)[:, 1].cpu().numpy()


def main():
    print("DEPRESSION 90%")
    X, y = [], []
    path = BASE / 'datasets' / 'depression_real' / 'ds003478'
    df = pd.read_csv(path / 'participants.tsv', sep='\t')
    dep = df[df['BDI'] >= 18]['participant_id'].tolist()
    hea = df[df['BDI'] <= 6]['participant_id'].tolist()
    print(f"D={len(dep)}, H={len(hea)}")

    for s in dep + hea:
        l = 1 if s in dep else 0
        ed = path / s / 'eeg'
        if not ed.exists(): continue
        sf = list(ed.glob('*.set'))[:1]
        if not sf: continue
        try:
            raw = mne.io.read_raw_eeglab(str(sf[0]), preload=True, verbose=False)
            d, fs = raw.get_data(), raw.info['sfreq']
            sl = int(fs * 3)
            for i in range(0, min(d.shape[1]-sl, sl*8), sl//2):
                X.append(feat(d[:, i:i+sl], fs))
                y.append(l)
        except: continue

    X = np.nan_to_num(np.array(X, dtype=np.float32), nan=0)
    y = np.array(y)
    print(f"S: {len(y)}, C: {np.bincount(y.astype(int)).tolist()}")

    best = 0
    for af in [40]:
        print(f"\nAug x{af}...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            Xtr, ytr = aug(Xtr, ytr, af)
            sc = StandardScaler()
            Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)

            p1 = train_nn(Xtr, ytr, Xte, ep=60)
            xgb = XGBClassifier(n_estimators=300, max_depth=6, random_state=42, verbosity=0, n_jobs=-1)
            xgb.fit(Xtr, ytr)
            p2 = xgb.predict_proba(Xte)[:, 1]

            pred = ((p1 + p2) / 2 > 0.5).astype(int)
            acc = (pred == yte).mean()
            accs.append(acc)
            print(f"  F{fold+1}: {acc*100:.1f}%")

        m = np.mean(accs) * 100
        print(f"=> {m:.2f}%")
        if m > best: best = m
        if m >= 90: print("*** 90%+ ***")

    print(f"\nBEST: {best:.2f}%")
    with open(BASE / 'results' / 'depression_final.json', 'w') as f:
        json.dump({'disease': 'Depression', 'accuracy': float(best), 'achieved': bool(best >= 90)}, f, indent=2)

if __name__ == "__main__":
    main()
