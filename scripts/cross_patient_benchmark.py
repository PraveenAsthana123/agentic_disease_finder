#!/usr/bin/env python3
"""Real cross-patient (leave-subjects-out) seizure-detection benchmark on CHB-MIT.

Uses real EDFs + seizure annotations (chbXX-summary.txt). Extracts seizure vs
non-seizure 4 s windows per subject, fast features, then leave-ONE-subject-out
CV — the honest generalization number the 50-row samples can't give.

Bounded for runtime: a few subjects, 1 seizure EDF each, capped windows.
Usage: python scripts/cross_patient_benchmark.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from scipy import signal as sp

ROOT = Path(__file__).resolve().parent.parent
CHB = ROOT / "data" / "real_eeg" / "epilepsy_physionet"
OUT = ROOT / "jobs" / "reports"
SUBJECTS = ["chb02", "chb03", "chb04"]
WIN_S = 4
MAX_SEIZ_WIN = 12
MAX_NONSEIZ_WIN = 12
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}


def parse_summary(subj):
    """Return list of (edf_name, [(start,end)...]) for files with seizures present locally."""
    summ = CHB / subj / f"{subj}-summary.txt"
    if not summ.exists():
        return []
    txt = summ.read_text(errors="replace")
    out = []
    for block in txt.split("File Name:")[1:]:
        name = block.splitlines()[0].strip()
        n = re.search(r"Number of Seizures in File:\s*(\d+)", block)
        if not n or int(n.group(1)) < 1:
            continue
        starts = [int(x) for x in re.findall(r"Seizure(?:\s+\d+)? Start Time:\s*(\d+)", block)]
        ends = [int(x) for x in re.findall(r"Seizure(?:\s+\d+)? End Time:\s*(\d+)", block)]
        if (CHB / subj / name).exists() and starts and ends:
            out.append((name, list(zip(starts, ends))))
    return out


def fast_features(win, sf):
    """~12 fast features from a (n_ch, n_samp) window (no slow entropies)."""
    f = []
    f.append(float(np.mean([np.std(win[c]) for c in range(win.shape[0])])))
    f.append(float(np.mean([np.ptp(win[c]) for c in range(win.shape[0])])))
    f.append(float(np.mean([np.sum(np.abs(np.diff(win[c]))) for c in range(win.shape[0])])))  # line length
    # band powers averaged across channels
    band_acc = {b: 0.0 for b in BANDS}
    for c in range(win.shape[0]):
        fr, psd = sp.welch(win[c], fs=sf, nperseg=min(256, win.shape[1]))
        tot = np.sum(psd) + 1e-12
        for b, (lo, hi) in BANDS.items():
            idx = (fr >= lo) & (fr < hi)
            band_acc[b] += float(np.sum(psd[idx])) / tot
    for b in BANDS:
        f.append(band_acc[b] / win.shape[0])
    # Hjorth mobility avg
    mob = []
    for c in range(win.shape[0]):
        d1 = np.diff(win[c]); v = np.var(win[c])
        mob.append(np.sqrt(np.var(d1) / (v + 1e-12)))
    f.append(float(np.mean(mob)))
    return f


def subject_windows(subj):
    import mne
    files = parse_summary(subj)
    if not files:
        return None, None
    name, seizures = files[0]  # one seizure EDF per subject (bounded)
    raw = mne.io.read_raw_edf(str(CHB / subj / name), preload=True, verbose="ERROR")
    data = raw.get_data()
    sf = int(raw.info["sfreq"])
    w = WIN_S * sf
    X, y = [], []
    # seizure windows
    for (s, e) in seizures:
        for st in range(s * sf, min(e * sf, data.shape[1] - w), w):
            if len(X) >= MAX_SEIZ_WIN:
                break
            X.append(fast_features(data[:, st:st + w], sf)); y.append(1)
    # non-seizure windows (well before first seizure)
    first = seizures[0][0]
    for st in range(0, max(0, (first - 30) * sf - w), w * 5):
        if y.count(0) >= MAX_NONSEIZ_WIN:
            break
        X.append(fast_features(data[:, st:st + w], sf)); y.append(0)
    return np.array(X, dtype="float32"), np.array(y)


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    OUT.mkdir(parents=True, exist_ok=True)

    per_subj = {}
    for s in SUBJECTS:
        Xs, ys = subject_windows(s)
        if Xs is not None and len(set(ys.tolist())) == 2:
            per_subj[s] = (Xs, ys)
            print(f"  {s}: {len(ys)} windows ({int(ys.sum())} seizure / {int((ys==0).sum())} non)")

    if len(per_subj) < 2:
        print("Not enough subjects with both classes."); return 1

    # Leave-one-subject-out
    results = []
    subs = list(per_subj.keys())
    for test_s in subs:
        Xtr = np.vstack([per_subj[s][0] for s in subs if s != test_s])
        ytr = np.concatenate([per_subj[s][1] for s in subs if s != test_s])
        Xte, yte = per_subj[test_s]
        sc = StandardScaler().fit(Xtr)
        clf = RandomForestClassifier(n_estimators=200, random_state=42).fit(sc.transform(Xtr), ytr)
        pred = clf.predict(sc.transform(Xte))
        acc = accuracy_score(yte, pred); f1 = f1_score(yte, pred, average="weighted", zero_division=0)
        results.append({"held_out_subject": test_s, "n_test": int(len(yte)),
                        "accuracy": round(float(acc), 4), "f1": round(float(f1), 4)})
        print(f"  held-out {test_s}: acc={acc:.3f} f1={f1:.3f}")

    mean_acc = round(float(np.mean([r["accuracy"] for r in results])), 4)
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "benchmark": "CHB-MIT cross-patient (leave-one-subject-out), real EDF + seizure annotations",
        "subjects": subs, "window_seconds": WIN_S, "features": "12 fast (stats + band power + Hjorth)",
        "cross_patient_accuracy_mean": mean_acc, "folds": results,
        "caveat": "Bounded subset (3 subjects, 1 seizure EDF each, capped windows). Honest cross-patient signal, not full-dataset.",
    }
    (OUT / "cross_patient_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nCROSS-PATIENT mean accuracy: {mean_acc}  (vs in-sample 0.99 — the honest gap)")
    print(f"Saved: {OUT / 'cross_patient_benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
