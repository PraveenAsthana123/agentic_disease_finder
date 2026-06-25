#!/usr/bin/env python3
"""Clinical Data Manager — Label Validation.

Two real label-QC views (no synthetic):

1. analysis_labels(): consistency of `analyses.predicted_label` against the
   patient's recorded disease, single-class flags, and low-confidence labels.
2. dataset_labels(): per reference .npz — class distribution, imbalance ratio,
   class_names integrity (cardinality matches distinct y), and balance flags.

Reads live tables + real on-disk datasets. Report only — no mutation.
"""

import glob
import os
import sqlite3
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "clinical.db")


def analysis_labels():
    """Validate analyses.predicted_label vs patient disease + confidence."""
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute(
        """SELECT a.id, a.patient_id, a.predicted_label, a.disease AS analysis_disease,
                  a.confidence, p.disease AS patient_disease
           FROM analyses a LEFT JOIN patients p ON a.patient_id = p.patient_id""")]
    c.close()

    dist = Counter(r["predicted_label"] for r in rows)
    mismatches = [{"analysis_id": r["id"], "patient_id": r["patient_id"],
                   "analysis_disease": r["analysis_disease"], "patient_disease": r["patient_disease"]}
                  for r in rows if r["analysis_disease"] and r["patient_disease"]
                  and r["analysis_disease"].lower() != r["patient_disease"].lower()]
    low_conf = [{"analysis_id": r["id"], "predicted_label": r["predicted_label"], "confidence": r["confidence"]}
                for r in rows if (r["confidence"] or 0) < 0.6]
    flags = []
    if len(dist) <= 1 and rows:
        flags.append(f"single predicted class ('{next(iter(dist))}') — no negative/control labels to validate against")
    if mismatches:
        flags.append(f"{len(mismatches)} analysis/patient disease mismatch(es)")
    return {
        "available": True, "n_analyses": len(rows),
        "predicted_label_distribution": dict(dist),
        "disease_mismatches": mismatches,
        "low_confidence_labels": low_conf,
        "flags": flags,
        "note": "Label consistency from live analyses-patients join; low-confidence (<0.6) surfaced for review.",
    }


def dataset_labels():
    """Validate class balance + class_names integrity across reference .npz datasets."""
    try:
        import numpy as np
    except ImportError:
        return {"available": False, "error": "numpy unavailable"}

    datasets = []
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "*", "sample", "*sample_100.npz"))):
        rel = os.path.relpath(p, ROOT)
        try:
            with np.load(p) as z:  # safe loader (numeric arrays only)
                if "y" not in z.files:
                    datasets.append({"dataset": rel, "available": False, "issue": "no y array"})
                    continue
                y = z["y"]
                dist = {int(k): int(v) for k, v in Counter(int(v) for v in y).items()}
                names = [str(x) for x in z["class_names"]] if "class_names" in z.files else None
        except Exception as e:  # noqa: BLE001 — report-only, never crash the QC pass
            datasets.append({"dataset": rel, "available": False, "issue": f"{type(e).__name__}: {e}"})
            continue
        counts = sorted(dist.values())
        imbalance = round(counts[-1] / counts[0], 2) if counts and counts[0] else None
        flags = []
        if len(dist) < 2:
            flags.append("single-class dataset")
        if imbalance and imbalance > 1.5:
            flags.append(f"class imbalance {imbalance}:1")
        if names is not None and len(names) != len(dist):
            flags.append(f"class_names count ({len(names)}) != distinct labels ({len(dist)})")
        datasets.append({"dataset": rel, "available": True, "n": int(len(y)),
                         "class_distribution": dist, "class_names": names,
                         "imbalance_ratio": imbalance, "flags": flags})
    ok = [d for d in datasets if d.get("available")]
    return {
        "available": True, "n_datasets": len(datasets),
        "datasets": datasets,
        "summary": {"datasets_ok": len(ok),
                    "datasets_with_flags": sum(1 for d in ok if d.get("flags")),
                    "balanced_datasets": sum(1 for d in ok if not d.get("flags"))},
        "note": "Class balance + class_names integrity over real reference .npz (safe loader).",
    }


def full_report():
    return {
        "role": "Clinical Data Manager — Label Validation",
        "analysis_labels": analysis_labels(),
        "dataset_labels": dataset_labels(),
    }


if __name__ == "__main__":
    r = full_report()
    al, dl = r["analysis_labels"], r["dataset_labels"]
    print("Label validation:")
    print("  analyses:", al["predicted_label_distribution"], "flags:", al["flags"])
    print("  datasets:", dl["summary"])
