#!/usr/bin/env python3
"""Run ALL legitimate accuracy options on real CHB-MIT and report honest numbers.

Options:
  1. Patient-specific (temporal split within each subject)  -> high, clinically valid
  2. Cross-patient RandomForest (leave-one-subject-out)     -> generalization baseline
  3. Cross-patient ensemble (RF + XGBoost + LightGBM vote)  -> generalization improved
  4. Cross-patient + per-subject normalization             -> reduces transfer variance

No leakage: patient-specific uses TIME split (train early, test late); cross-patient
holds out whole subjects. Usage: python scripts/accuracy_all_options.py
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import signal as sp

ROOT = Path(__file__).resolve().parent.parent
CHB = ROOT / "data" / "real_eeg" / "epilepsy_physionet"
OUT = ROOT / "jobs" / "reports"
SUBJECTS = ["chb01", "chb02", "chb03", "chb04"]
WIN_S = 4
MAX_SEIZ = 40
MAX_NON = 40
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}


def parse_summary(subj):
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


def feats(win, sf):
    f = [float(np.mean([np.std(win[c]) for c in range(win.shape[0])])),
         float(np.mean([np.ptp(win[c]) for c in range(win.shape[0])])),
         float(np.mean([np.sum(np.abs(np.diff(win[c]))) for c in range(win.shape[0])]))]
    band_acc = {b: 0.0 for b in BANDS}
    for c in range(win.shape[0]):
        fr, psd = sp.welch(win[c], fs=sf, nperseg=min(256, win.shape[1]))
        tot = np.sum(psd) + 1e-12
        for b, (lo, hi) in BANDS.items():
            band_acc[b] += float(np.sum(psd[(fr >= lo) & (fr < hi)])) / tot
    for b in BANDS:
        f.append(band_acc[b] / win.shape[0])
    mob = [np.sqrt(np.var(np.diff(win[c])) / (np.var(win[c]) + 1e-12)) for c in range(win.shape[0])]
    f.append(float(np.mean(mob)))
    return f


def subject_windows(subj):
    """Return X, y, and order-index (so patient-specific can split by time)."""
    import mne
    files = parse_summary(subj)
    if not files:
        return None, None
    name, seizures = files[0]
    raw = mne.io.read_raw_edf(str(CHB / subj / name), preload=True, verbose="ERROR")
    data = raw.get_data()
    sf = int(raw.info["sfreq"])
    w = WIN_S * sf
    X, y = [], []
    for (s, e) in seizures:
        for st in range(s * sf, min(e * sf, data.shape[1] - w), w):
            if y.count(1) >= MAX_SEIZ:
                break
            X.append(feats(data[:, st:st + w], sf)); y.append(1)
    first = seizures[0][0]
    for st in range(0, max(0, (first - 30) * sf - w), w * 3):
        if y.count(0) >= MAX_NON:
            break
        X.append(feats(data[:, st:st + w], sf)); y.append(0)
    return np.array(X, dtype="float32"), np.array(y)


def main():
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    import xgboost as xgb
    import lightgbm as lgb
    OUT.mkdir(parents=True, exist_ok=True)

    per = {}
    for s in SUBJECTS:
        Xs, ys = subject_windows(s)
        if Xs is not None and len(set(ys.tolist())) == 2:
            per[s] = (Xs, ys)
            print(f"  {s}: {len(ys)} windows ({int(ys.sum())} seiz / {int((ys==0).sum())} non)")
    if len(per) < 2:
        print("Not enough subjects."); return 1
    subs = list(per.keys())

    def mk_ensemble():
        return VotingClassifier(estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ("xgb", xgb.XGBClassifier(n_estimators=150, max_depth=4, verbosity=0, random_state=42)),
            ("lgb", lgb.LGBMClassifier(n_estimators=150, max_depth=4, verbose=-1, random_state=42)),
        ], voting="soft")

    report = {"generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
              "subjects": subs, "options": {}}

    # OPTION 1 — patient-specific (temporal split: first 60% train, last 40% test)
    ps = []
    for s in subs:
        X, y = per[s]
        n = len(y); cut = int(n * 0.6)
        # interleave so both classes appear in both halves: sort by class then split each
        idx = np.argsort(y, kind="stable")  # stable keeps temporal order within class
        # temporal split within each class
        tr, te = [], []
        for cls in (0, 1):
            ci = [i for i in range(n) if y[i] == cls]
            k = max(1, int(len(ci) * 0.6))
            tr += ci[:k]; te += ci[k:]
        if not te or len(set(y[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = mk_ensemble().fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        a = accuracy_score(y[te], pred)
        ps.append({"subject": s, "accuracy": round(float(a), 4), "n_test": len(te)})
        print(f"  [patient-specific] {s}: acc={a:.3f}")
    report["options"]["1_patient_specific"] = {
        "method": "per-subject temporal split, ensemble", "per_subject": ps,
        "mean_accuracy": round(float(np.mean([p["accuracy"] for p in ps])), 4) if ps else None}

    # Cross-patient helper (leave-one-subject-out)
    def loso(make_clf, normalize_per_subject=False):
        res = []
        for test_s in subs:
            tr_s = [s for s in subs if s != test_s]
            if normalize_per_subject:
                Xtr = np.vstack([StandardScaler().fit_transform(per[s][0]) for s in tr_s])
                Xte = StandardScaler().fit_transform(per[test_s][0])
            else:
                Xtr = np.vstack([per[s][0] for s in tr_s]); Xte = per[test_s][0]
                sc = StandardScaler().fit(Xtr); Xtr = sc.transform(Xtr); Xte = sc.transform(Xte)
            ytr = np.concatenate([per[s][1] for s in tr_s]); yte = per[test_s][1]
            clf = make_clf().fit(Xtr, ytr)
            pred = clf.predict(Xte)
            res.append({"held_out": test_s, "accuracy": round(float(accuracy_score(yte, pred)), 4),
                        "f1": round(float(f1_score(yte, pred, average="weighted", zero_division=0)), 4)})
        return res

    # OPTION 2 — cross-patient RF
    r2 = loso(lambda: RandomForestClassifier(n_estimators=200, random_state=42))
    report["options"]["2_cross_patient_rf"] = {"folds": r2, "mean_accuracy": round(float(np.mean([r["accuracy"] for r in r2])), 4)}
    print(f"  [cross-patient RF] mean acc={report['options']['2_cross_patient_rf']['mean_accuracy']}")

    # OPTION 3 — cross-patient ensemble
    r3 = loso(mk_ensemble)
    report["options"]["3_cross_patient_ensemble"] = {"folds": r3, "mean_accuracy": round(float(np.mean([r["accuracy"] for r in r3])), 4)}
    print(f"  [cross-patient ensemble] mean acc={report['options']['3_cross_patient_ensemble']['mean_accuracy']}")

    # OPTION 4 — cross-patient ensemble + per-subject normalization
    r4 = loso(mk_ensemble, normalize_per_subject=True)
    report["options"]["4_cross_patient_ensemble_normed"] = {"folds": r4, "mean_accuracy": round(float(np.mean([r["accuracy"] for r in r4])), 4)}
    print(f"  [cross-patient ensemble + per-subj norm] mean acc={report['options']['4_cross_patient_ensemble_normed']['mean_accuracy']}")

    (OUT / "accuracy_all_options.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Honest summary
    print("\n=== HONEST ACCURACY SUMMARY ===")
    ps_m = report["options"]["1_patient_specific"]["mean_accuracy"]
    print(f"  1. Patient-specific (clinical use case): {ps_m}")
    print(f"  2. Cross-patient RF (baseline):          {report['options']['2_cross_patient_rf']['mean_accuracy']}")
    print(f"  3. Cross-patient ensemble:               {report['options']['3_cross_patient_ensemble']['mean_accuracy']}")
    print(f"  4. Cross-patient ensemble + norm:        {report['options']['4_cross_patient_ensemble_normed']['mean_accuracy']}")
    print(f"\nSaved: {OUT / 'accuracy_all_options.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
