#!/usr/bin/env python3
"""EEGNet (braindecode) on RAW CHB-MIT EEG — the field-standard deep architecture.

Unlike the 47-feature classical models, EEGNet learns directly from raw signal
windows (n_channels x n_times). Patient-specific temporal split (no leakage),
reports accuracy/sensitivity. Uses braindecode 1.2 (already installed).

Usage: python scripts/eegnet_benchmark.py
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHB = ROOT / "data" / "real_eeg" / "epilepsy_physionet"
OUT = ROOT / "jobs" / "reports"
SUBJECTS = ["chb01", "chb02", "chb03", "chb04"]
WIN_S = 4
STRIDE = 2
MAX_SEIZ = 120
MAX_NON = 120
N_CH = 18          # fixed channel count for a consistent EEGNet input
EPOCHS = 25


def seizure_files(subj):
    summ = CHB / subj / f"{subj}-summary.txt"
    if not summ.exists():
        return []
    out = []
    for block in summ.read_text(errors="replace").split("File Name:")[1:]:
        name = block.splitlines()[0].strip()
        n = re.search(r"Number of Seizures in File:\s*(\d+)", block)
        if not n or int(n.group(1)) < 1:
            continue
        st = [int(x) for x in re.findall(r"Seizure(?:\s+\d+)? Start Time:\s*(\d+)", block)]
        en = [int(x) for x in re.findall(r"Seizure(?:\s+\d+)? End Time:\s*(\d+)", block)]
        if (CHB / subj / name).exists() and st and en:
            out.append((name, list(zip(st, en))))
    return out


def raw_windows(subj):
    """Return X (n_win, N_CH, win_samples) raw + y + sampling rate."""
    import mne
    files = seizure_files(subj)[:4]
    if not files:
        return None, None, None
    X, y = [], []
    sf = None
    nseiz = nnon = 0
    for name, seizures in files:
        if nseiz >= MAX_SEIZ and nnon >= MAX_NON:
            break
        try:
            raw = mne.io.read_raw_edf(str(CHB / subj / name), preload=True, verbose="ERROR")
        except Exception:
            continue
        data = raw.get_data(); sf = int(raw.info["sfreq"]); w = WIN_S * sf; step = STRIDE * sf
        if data.shape[0] >= N_CH:
            data = data[:N_CH]
        else:
            data = np.vstack([data, np.zeros((N_CH - data.shape[0], data.shape[1]))])
        for (s, e) in seizures:
            for stt in range(s * sf, min(e * sf, data.shape[1] - w), step):
                if nseiz >= MAX_SEIZ:
                    break
                X.append(data[:, stt:stt + w]); y.append(1); nseiz += 1
        first = seizures[0][0]
        for stt in range(0, max(0, (first - 20) * sf - w), w * 2):
            if nnon >= MAX_NON:
                break
            X.append(data[:, stt:stt + w]); y.append(0); nnon += 1
    if not X:
        return None, None, None
    return np.array(X, dtype="float32"), np.array(y), sf


def main():
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from braindecode.models import EEGNetv4
    from sklearn.metrics import accuracy_score, recall_score
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for subj in SUBJECTS:
        X, y, sf = raw_windows(subj)
        if X is None or len(set(y.tolist())) < 2:
            print(f"  {subj}: insufficient"); continue
        X = (X - X.mean(axis=(1, 2), keepdims=True)) / (X.std(axis=(1, 2), keepdims=True) + 1e-6)
        n_times = X.shape[2]
        tr, te = [], []
        for cls in (0, 1):
            ci = [i for i in range(len(y)) if y[i] == cls]
            k = max(1, int(len(ci) * 0.7)); tr += ci[:k]; te += ci[k:]
        if not te:
            continue
        Xtr = torch.tensor(X[tr]); ytr = torch.tensor(y[tr], dtype=torch.long)
        Xte = torch.tensor(X[te]); yte = y[te]

        model = EEGNetv4(n_chans=N_CH, n_outputs=2, n_times=n_times).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossf = torch.nn.CrossEntropyLoss()
        dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)
        model.train(True)
        for ep in range(EPOCHS):
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        model.train(False)  # inference mode (equivalent to model.eval())
        with torch.no_grad():
            pred = model(Xte.to(dev)).argmax(1).cpu().numpy()
        acc = accuracy_score(yte, pred); sens = recall_score(yte, pred, pos_label=1, zero_division=0)
        rows.append({"subject": subj, "n_train": len(tr), "n_test": len(te),
                     "accuracy": round(float(acc), 4), "sensitivity": round(float(sens), 4)})
        print(f"  {subj}: EEGNet acc={acc:.3f} sens={sens:.3f} ({len(y)} windows, {n_times} samples)")

    if not rows:
        print("No subjects."); return 1
    payload = {"generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
               "model": "EEGNetv4 (braindecode) on raw EEG, patient-specific temporal split",
               "architecture": "EEGNet — field-standard CNN for raw EEG (no hand-crafted features)",
               "n_channels": N_CH, "epochs": EPOCHS,
               "mean_accuracy": round(float(np.mean([r["accuracy"] for r in rows])), 4),
               "mean_sensitivity": round(float(np.mean([r["sensitivity"] for r in rows])), 4),
               "per_subject": rows,
               "note": "Deep learning on raw signal vs the 47-feature classical models. Compare both for the thesis."}
    (OUT / "eegnet_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n=== EEGNet (deep, raw signal) — mean acc {payload['mean_accuracy']}, sens {payload['mean_sensitivity']} ===")
    print(f"Saved: {OUT / 'eegnet_benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
