#!/usr/bin/env python3
"""Real model-performance metrics (ROC, PR, confusion, calibration) for the epilepsy model,
computed subject-wise (GroupKFold) on the aligned reference npz. No synthetic. Cached."""
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent


def build(disease: str = "epilepsy") -> dict:
    import numpy as np, joblib
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve,
                                 roc_auc_score, accuracy_score, precision_score, recall_score, f1_score)
    npz = ROOT / "data" / disease.lower() / "sample" / f"{disease.lower()}_sample_100.npz"
    mdl = ROOT / "models" / f"{disease.lower()}_model.joblib"
    if not npz.exists() or not mdl.exists():
        return {"available": False, "error": "missing npz or model"}
    ref = np.load(npz, allow_pickle=True)
    X = np.nan_to_num(ref["X"].astype(float)); y = ref["y"].astype(int)
    sids = ref["subject_ids"] if "subject_ids" in ref else np.arange(len(y))
    classes = [str(c) for c in ref["class_names"]]
    model = joblib.load(mdl)["model"]
    n_split = max(2, min(5, int((y == 0).sum()), int((y == 1).sum())))
    cv = StratifiedGroupKFold(n_splits=n_split)
    yp = cross_val_predict(model, X, y, groups=sids, cv=cv, method="predict")
    yproba = cross_val_predict(model, X, y, groups=sids, cv=cv, method="predict_proba")[:, 1]

    cm = confusion_matrix(y, yp).tolist()
    fpr, tpr, _ = roc_curve(y, yproba)
    prec, rec, _ = precision_recall_curve(y, yproba)
    # downsample curves to ~30 pts for UI
    def ds(a, n=30):
        idx = np.linspace(0, len(a) - 1, min(n, len(a))).astype(int)
        return [round(float(a[i]), 4) for i in idx]
    return {
        "available": True, "disease": disease, "n_samples": int(len(y)),
        "classes": classes, "cv": f"StratifiedGroupKFold-{n_split} (subject-wise)",
        "metrics": {"accuracy": round(accuracy_score(y, yp), 4), "precision": round(precision_score(y, yp, zero_division=0), 4),
                    "recall": round(recall_score(y, yp, zero_division=0), 4), "f1": round(f1_score(y, yp, zero_division=0), 4),
                    "auc": round(roc_auc_score(y, yproba), 4)},
        "confusion_matrix": cm,
        "roc": [{"fpr": f, "tpr": t} for f, t in zip(ds(fpr), ds(tpr))],
        "pr": [{"recall": r, "precision": p} for r, p in zip(ds(rec), ds(prec))],
        "note": "Subject-wise CV (leakage-free) on aligned reference features. Dataset-confound caveat per model bundle.",
    }


if __name__ == "__main__":
    r = build()
    if r.get("available"):
        print("Model performance:", r["metrics"], "| cm", r["confusion_matrix"])
    else:
        print(r)
