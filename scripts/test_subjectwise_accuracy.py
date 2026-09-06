#!/usr/bin/env python3
"""Subject-wise (leakage-free) accuracy job — the defensible number.

The in-sample job (test_multi_disease_accuracy.py) reports ~99-100% because
the shipped models are scored on data that overlaps their training set. This
job removes that leakage:

  * GroupKFold grouped by subject_id  -> no subject in both train and test
  * a FRESH model is trained inside each fold (shipped models never reused)
  * StandardScaler is fit on the training fold only (no scaler leakage)

It reports subject-wise CV accuracy (mean ± std) per disease and the
leakage gap vs the in-sample score. This is the number to cite in the DBA.

Usage: python scripts/test_subjectwise_accuracy.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "jobs" / "reports"
DISEASES = ["alzheimer", "parkinson", "schizophrenia", "epilepsy", "autism", "stress", "depression"]


def _now():
    return datetime.now(timezone.utc).astimezone()


def sample_path(dz: str) -> Path | None:
    for n in (f"{dz}_50rows.npz", f"{dz}_sample_100.npz"):
        p = DATA / dz / "sample" / n
        if p.exists():
            return p
    return None


def evaluate(dz: str) -> dict:
    sp = sample_path(dz)
    if sp is None:
        return {"disease": dz, "status": "skip", "reason": "no sample"}
    d = np.load(sp)
    X, y = d["X"], d["y"]
    if "subject_ids" not in d:
        return {"disease": dz, "status": "skip", "reason": "no subject_ids"}
    groups = d["subject_ids"]

    n_subjects = len(np.unique(groups))
    n_splits = min(5, n_subjects)
    if n_splits < 2:
        return {"disease": dz, "status": "skip", "reason": "too few subjects"}

    gkf = GroupKFold(n_splits=n_splits)
    fold_acc, fold_f1, insample_acc = [], [], []

    for train_idx, test_idx in gkf.split(X, y, groups):
        # Guard: a fold must contain both classes in train to be scoreable.
        if len(np.unique(y[train_idx])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            RandomForestClassifier(n_estimators=200, random_state=42))
        clf.fit(X[train_idx], y[train_idx])
        y_hat = clf.predict(X[test_idx])
        fold_acc.append(accuracy_score(y[test_idx], y_hat))
        fold_f1.append(f1_score(y[test_idx], y_hat, average="weighted", zero_division=0))
        # In-sample reference: same fresh model scored on its own train fold.
        insample_acc.append(accuracy_score(y[train_idx], clf.predict(X[train_idx])))

    if not fold_acc:
        return {"disease": dz, "status": "skip", "reason": "no scoreable folds"}

    cv_acc = float(np.mean(fold_acc))
    return {
        "disease": dz, "status": "ok",
        "n_subjects": int(n_subjects), "n_splits": len(fold_acc),
        "subjectwise_acc_mean": round(cv_acc, 4),
        "subjectwise_acc_std": round(float(np.std(fold_acc)), 4),
        "subjectwise_f1_mean": round(float(np.mean(fold_f1)), 4),
        "insample_acc_mean": round(float(np.mean(insample_acc)), 4),
        "leakage_gap": round(float(np.mean(insample_acc)) - cv_acc, 4),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = [evaluate(dz) for dz in DISEASES]
    ok = [r for r in results if r["status"] == "ok"]
    mean_cv = round(sum(r["subjectwise_acc_mean"] for r in ok) / len(ok), 4) if ok else None
    mean_gap = round(sum(r["leakage_gap"] for r in ok) / len(ok), 4) if ok else None

    print(f"\n=== Subject-Wise (leakage-free) Accuracy — {_now().isoformat(timespec='seconds')} ===")
    print(f"{'disease':<15}{'cv_acc':>9}{'±std':>8}{'cv_f1':>8}{'insample':>10}{'gap':>8}")
    print("-" * 58)
    for r in results:
        if r["status"] == "ok":
            print(f"{r['disease']:<15}{r['subjectwise_acc_mean']:>9}{r['subjectwise_acc_std']:>8}"
                  f"{r['subjectwise_f1_mean']:>8}{r['insample_acc_mean']:>10}{r['leakage_gap']:>8}")
        else:
            print(f"{r['disease']:<15}{'SKIP':>9}  ({r['reason']})")
    print("-" * 58)
    print(f"{'MEAN':<15}{mean_cv if mean_cv is not None else '-':>9}{'':>8}{'':>8}{'':>10}{mean_gap if mean_gap is not None else '-':>8}")
    print(f"\n✅ Subject-wise CV is the DEFENSIBLE number (~{mean_cv}). The 'gap' column")
    print("   quantifies how much in-sample scoring inflated accuracy (leakage).")

    payload = {
        "generated_at": _now().isoformat(timespec="seconds"),
        "evaluation_type": "subject-wise GroupKFold (leakage-free)",
        "mean_subjectwise_accuracy": mean_cv,
        "mean_leakage_gap": mean_gap,
        "results": results,
        "note": "Fresh model per fold; scaler fit on train only; grouped by subject_id.",
    }
    (OUT / "subjectwise_accuracy.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Subject-Wise (Leakage-Free) Accuracy",
        "", f"_Generated {payload['generated_at']}_", "",
        f"**Mean subject-wise CV accuracy: {mean_cv}**  ·  mean leakage gap: **{mean_gap}**", "",
        "| Disease | CV acc | ±std | CV F1 | In-sample | Leakage gap | Subjects |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r["status"] == "ok":
            lines.append(f"| {r['disease']} | {r['subjectwise_acc_mean']} | {r['subjectwise_acc_std']} | "
                         f"{r['subjectwise_f1_mean']} | {r['insample_acc_mean']} | {r['leakage_gap']} | {r['n_subjects']} |")
        else:
            lines.append(f"| {r['disease']} | SKIP | — | — | — | — | {r['reason']} |")
    lines += [
        "",
        "> ✅ **This is the defensible number for the DBA.** No subject appears in both",
        "> train and test; a fresh model is trained per fold; the scaler is fit on the",
        "> training fold only. The leakage gap = in-sample minus subject-wise CV.",
        "> Note: samples are small (10 subjects/disease) — re-run on the full feature",
        "> dataset for tighter confidence intervals.",
        "",
    ]
    (OUT / "subjectwise_accuracy.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved: {OUT / 'subjectwise_accuracy.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
