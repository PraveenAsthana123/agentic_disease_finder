#!/usr/bin/env python3
"""Multi-disease accuracy test job.

For each disease: load the trained model bundle (models/<disease>_model.joblib)
and evaluate it on the on-disk feature sample (data/<disease>/sample/*.npz).
Reports accuracy / precision / recall / F1 / confusion matrix per disease, plus
the model's stored training metrics, and writes a JSON + Markdown report.

⚠️ HONESTY: evaluating on the sample that (likely) overlaps training data is
IN-SAMPLE and optimistic. Treat near-perfect scores as a data-leakage flag, not
a generalization claim. For a real number, use a subject-wise hold-out split.

Usage: python scripts/test_multi_disease_accuracy.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data"
OUT = ROOT / "jobs" / "reports"
DISEASES = ["alzheimer", "parkinson", "schizophrenia", "epilepsy", "autism", "stress", "depression"]


def _now():
    return datetime.now(timezone.utc).astimezone()


def sample_path(disease: str) -> Path | None:
    for name in (f"{disease}_50rows.npz", f"{disease}_sample_100.npz"):
        p = DATA / disease / "sample" / name
        if p.exists():
            return p
    return None


def evaluate(disease: str) -> dict:
    mp = MODELS / f"{disease}_model.joblib"
    sp = sample_path(disease)
    if not mp.exists():
        return {"disease": disease, "status": "skip", "reason": "no model"}
    if sp is None:
        return {"disease": disease, "status": "skip", "reason": "no sample"}

    bundle = joblib.load(mp)
    model = bundle["model"]
    d = np.load(sp)
    X, y = d["X"], d["y"]

    n_expected = bundle.get("n_features", getattr(model, "n_features_in_", X.shape[1]))
    if X.shape[1] != n_expected:
        return {"disease": disease, "status": "skip", "reason": f"feature mismatch {X.shape[1]}!={n_expected}"}

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y, y_pred).tolist()

    return {
        "disease": disease,
        "status": "ok",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "accuracy": round(float(acc), 4),
        "precision": round(float(pr), 4),
        "recall": round(float(rc), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm,
        "class_names": [str(c) for c in d["class_names"]] if "class_names" in d else None,
        "training_metrics": bundle.get("metrics", {}),
        "evaluation": "in-sample (optimistic; not a generalization estimate)",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = [evaluate(dz) for dz in DISEASES]
    ok = [r for r in results if r["status"] == "ok"]
    mean_acc = round(sum(r["accuracy"] for r in ok) / len(ok), 4) if ok else None

    # Console table.
    print(f"\n=== Multi-Disease Accuracy Test — {_now().isoformat(timespec='seconds')} ===")
    print(f"{'disease':<15}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}{'n':>6}")
    print("-" * 53)
    for r in results:
        if r["status"] == "ok":
            print(f"{r['disease']:<15}{r['accuracy']:>8}{r['precision']:>8}{r['recall']:>8}{r['f1']:>8}{r['n_samples']:>6}")
        else:
            print(f"{r['disease']:<15}{'SKIP':>8}  ({r['reason']})")
    print("-" * 53)
    print(f"{'MEAN (in-sample)':<15}{mean_acc if mean_acc is not None else '-':>8}")
    print("\n⚠️  In-sample evaluation — near-perfect scores indicate leakage, not generalization.")

    payload = {
        "generated_at": _now().isoformat(timespec="seconds"),
        "evaluation_type": "in-sample (optimistic)",
        "mean_accuracy": mean_acc,
        "results": results,
        "caveat": "Use subject-wise hold-out for a real generalization number.",
    }
    (OUT / "multi_disease_accuracy.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Multi-Disease Accuracy Test",
        "", f"_Generated {payload['generated_at']} — **in-sample (optimistic)**_", "",
        f"**Mean in-sample accuracy: {mean_acc}**", "",
        "| Disease | Accuracy | Precision | Recall | F1 | N |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        if r["status"] == "ok":
            lines.append(f"| {r['disease']} | {r['accuracy']} | {r['precision']} | {r['recall']} | {r['f1']} | {r['n_samples']} |")
        else:
            lines.append(f"| {r['disease']} | SKIP | — | — | — | {r['reason']} |")
    lines += [
        "",
        "> ⚠️ **In-sample evaluation.** Models are scored on the sample that likely",
        "> overlaps their training data, so these are optimistic. Near-perfect scores",
        "> are a data-leakage red flag, not generalization. For thesis claims, re-run",
        "> with a subject-wise hold-out split (no subject in both train and test).",
        "",
    ]
    (OUT / "multi_disease_accuracy.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved: {OUT / 'multi_disease_accuracy.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
