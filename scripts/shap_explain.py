#!/usr/bin/env python3
"""REAL local SHAP explanation for an EEG prediction — 'why did the model predict X?'.
Uses the SAME model + raw 47-feature vector the pipeline classifies with, with the
training reference samples (epilepsy_sample_100.npz) as SHAP background. No synthetic.
Loads first-party trained model bundles via joblib (same pattern as eeg_analysis_pipeline.py)."""
from __future__ import annotations

import json
import sqlite3
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent


def _target_features(analysis_id=None, patient_id=None):
    c = sqlite3.connect(str(ROOT / "data" / "clinical.db"))
    c.row_factory = sqlite3.Row
    if analysis_id:
        r = c.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
    elif patient_id:
        r = c.execute("SELECT * FROM analyses WHERE patient_id=? ORDER BY id DESC LIMIT 1", (patient_id,)).fetchone()
    else:
        r = c.execute("SELECT * FROM analyses WHERE result_json LIKE '%features%' ORDER BY id DESC LIMIT 1").fetchone()
    if not r:
        return None, None, None
    d = dict(r)
    res = json.loads(d.get("result_json") or "{}")
    return d, res.get("features", {}), res.get("disease", d.get("disease", "epilepsy"))


def explain(analysis_id=None, patient_id=None, top_n: int = 8) -> dict:
    import joblib
    import numpy as np
    import shap

    row, feats, disease = _target_features(analysis_id, patient_id)
    if not feats:
        return {"available": False, "error": "No analysis with feature vector found."}

    npz_path = ROOT / "data" / disease.lower() / "sample" / f"{disease.lower()}_sample_100.npz"
    model_path = ROOT / "saved_models" / f"{disease.lower()}_model.joblib"
    if not npz_path.exists() or not model_path.exists():
        return {"available": False, "error": f"Missing reference samples or model for '{disease}'."}

    ref = np.load(npz_path, allow_pickle=True)
    feat_names = [str(f) for f in ref["feature_names"]]
    class_names = [str(c) for c in ref["class_names"]]
    X_bg = ref["X"]

    instance = np.array([float(feats.get(fn, 0.0) or 0.0) for fn in feat_names], dtype=float)
    X_bg = np.nan_to_num(X_bg, nan=0.0, posinf=0.0, neginf=0.0)
    col_mean = np.nanmean(X_bg, axis=0)
    instance = np.where(np.isfinite(instance), instance, col_mean)  # NaN-safe: degenerate signals -> background mean
    bundle = joblib.load(model_path)            # first-party model artifact (same as pipeline)
    model = bundle["model"]
    if instance.shape[0] != getattr(model, "n_features_in_", instance.shape[0]):
        return {"available": False, "error": f"Feature mismatch: {instance.shape[0]} vs model {model.n_features_in_}"}

    proba = model.predict_proba(instance.reshape(1, -1))[0]
    pred_idx = int(np.argmax(proba))

    bg = shap.kmeans(X_bg, 15)
    expl = shap.KernelExplainer(model.predict_proba, bg, silent=True)
    sv = expl.shap_values(instance.reshape(1, -1), nsamples=120, silent=True)
    if isinstance(sv, list):
        vals = np.asarray(sv[pred_idx]).reshape(-1)
        base = float(np.asarray(expl.expected_value).reshape(-1)[pred_idx])
    else:
        arr = np.asarray(sv)
        vals = arr[0, :, pred_idx] if arr.ndim == 3 else arr.reshape(-1)
        ev = np.asarray(expl.expected_value).reshape(-1)
        base = float(ev[pred_idx]) if ev.size > pred_idx else float(ev[0])

    order = np.argsort(np.abs(vals))[::-1][:top_n]
    pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
    top = [{
        "feature": feat_names[i],
        "shap": round(float(vals[i]), 4),
        "value": round(float(instance[i]), 4),
        "direction": f"↑ {pred_label}" if vals[i] > 0 else f"↓ {pred_label}",
    } for i in order]

    return {
        "available": True,
        "analysis_id": row["id"], "patient_id": row["patient_id"], "disease": disease,
        "predicted_label": pred_label,
        "confidence": round(float(proba[pred_idx]), 4),
        "base_value": round(base, 4),
        "top_features": top,
        "method": "SHAP KernelExplainer (model-agnostic) on the VotingClassifier",
        "background": f"{X_bg.shape[0]} real reference samples (kmeans-15 summary)",
        "interpretation": f"For this recording, the features above most drove the '{pred_label}' prediction "
                          f"(↑ = increased {pred_label} probability, ↓ = decreased it).",
        "note": "Local explanation for THIS prediction. Feature-level (model is feature-based, not per-channel).",
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis_id", type=int)
    ap.add_argument("--patient_id")
    a = ap.parse_args()
    out = explain(analysis_id=a.analysis_id, patient_id=a.patient_id)
    if out.get("available"):
        print(f"{out['predicted_label']} (conf {out['confidence']}) top SHAP features:")
        for t in out["top_features"]:
            print(f"  {t['direction']:14s} {t['feature']:20s} shap={t['shap']:+.4f} (value {t['value']})")
    else:
        print(out)
