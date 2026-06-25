#!/usr/bin/env python3
"""CatBoost model — gradient-boosted alternative on the real epilepsy features.

Trains a CatBoost classifier on the aligned reference features (epilepsy_sample_
100.npz) with leakage-free subject-wise CV (StratifiedGroupKFold), reports
metrics + top feature importances, and compares to the deployed model's AUC.

Wires the CatBoost EEG-stack lib (installed in the canonical venv per §61.11).
100% real (reference features) — no synthetic.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build(disease: str = "epilepsy") -> dict:
    import numpy as np
    from catboost import CatBoostClassifier
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    npz = ROOT / "data" / disease.lower() / "sample" / f"{disease.lower()}_sample_100.npz"
    if not npz.exists():
        return {"available": False, "error": "reference npz not found"}
    z = np.load(npz)
    X = np.nan_to_num(z["X"].astype(float)); y = z["y"].astype(int)
    sids = z["subject_ids"] if "subject_ids" in z.files else np.arange(len(y))
    fnames = [str(f) for f in z["feature_names"]] if "feature_names" in z.files else [f"f{i}" for i in range(X.shape[1])]

    n = max(2, min(5, int((y == 0).sum()), int((y == 1).sum())))
    cv = StratifiedGroupKFold(n_splits=n)
    model = CatBoostClassifier(iterations=150, depth=4, learning_rate=0.05, verbose=0, random_seed=42)
    yp = cross_val_predict(model, X, y, groups=sids, cv=cv, method="predict")
    yproba = cross_val_predict(model, X, y, groups=sids, cv=cv, method="predict_proba")[:, 1]

    # fit on all to extract importances (illustrative)
    model.fit(X, y)
    imp = model.get_feature_importance()
    top = sorted(zip(fnames, imp), key=lambda t: t[1], reverse=True)[:12]

    # deployed model AUC (from bundle metrics if present)
    deployed_auc = None
    bundle = ROOT / "models" / f"{disease.lower()}_model.joblib"
    if bundle.exists():
        try:
            import joblib
            m = joblib.load(bundle).get("metrics", {})
            deployed_auc = m.get("auc") or m.get("roc_auc")
        except Exception:
            deployed_auc = None

    cat_auc = round(float(roc_auc_score(y, yproba)), 4)
    return {
        "available": True, "model": "CatBoost", "disease": disease,
        "n_samples": int(len(y)), "n_features": int(X.shape[1]),
        "cv": f"StratifiedGroupKFold-{n} (subject-wise, leakage-free)",
        "metrics": {
            "accuracy": round(float(accuracy_score(y, yp)), 4),
            "precision": round(float(precision_score(y, yp, zero_division=0)), 4),
            "recall": round(float(recall_score(y, yp, zero_division=0)), 4),
            "f1": round(float(f1_score(y, yp, zero_division=0)), 4),
            "auc": cat_auc,
        },
        "top_features": [{"feature": f, "importance": round(float(v), 3)} for f, v in top],
        "comparison": {"catboost_auc": cat_auc, "deployed_model_auc": deployed_auc,
                       "delta": (round(cat_auc - deployed_auc, 4) if isinstance(deployed_auc, (int, float)) else None)},
        "note": "CatBoost alternative model, subject-wise CV (leakage-free). Illustrative comparison, not a redeploy.",
        "source": "Real aligned reference features via CatBoost + scikit-learn.",
    }


if __name__ == "__main__":
    r = build()
    print("CatBoost:", r.get("metrics"))
    print("  comparison:", r.get("comparison"))
    print("  top feature:", r.get("top_features", [{}])[0])
