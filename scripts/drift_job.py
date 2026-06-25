#!/usr/bin/env python3
"""Scheduled DRIFT job — measures real distribution drift between the model's TRAINING
reference (epilepsy_sample_100.npz) and the LIVE feature extractor's recent outputs
(stored analyses). Uses PSI + KS-test (the standard drift metrics, same as Evidently).
Directly quantifies train/serve skew. Writes jobs/reports/drift_latest.json. No synthetic."""
from __future__ import annotations
import json, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))


def _now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def run(disease: str = "epilepsy") -> dict:
    import numpy as np
    from drift_monitor import DriftMonitor  # reuse proven PSI/KS code

    npz = ROOT / "data" / disease.lower() / "sample" / f"{disease.lower()}_sample_100.npz"
    if not npz.exists():
        return {"available": False, "error": f"No reference samples for {disease}"}
    ref = np.load(npz, allow_pickle=True)
    feat_names = [str(f) for f in ref["feature_names"]]
    X_ref = np.nan_to_num(ref["X"].astype(float))

    # live feature vectors from stored analyses (the live extractor's output)
    c = sqlite3.connect(str(ROOT / "data" / "clinical.db"))
    rows = c.execute("SELECT result_json FROM analyses WHERE result_json LIKE '%features%'").fetchall()
    live = []
    for (rj,) in rows:
        try:
            f = json.loads(rj).get("features", {})
            if len(f) == 47:
                live.append([float(f.get(fn, 0) or 0) for fn in feat_names])
        except (ValueError, TypeError):
            pass
    if len(live) < 3:
        return {"available": False, "error": f"Only {len(live)} live samples (need >=3)"}
    X_live = np.nan_to_num(np.array(live, dtype=float))

    dm = DriftMonitor()
    per_feature = []
    drifted = 0
    for i, fn in enumerate(feat_names):
        psi = dm.calculate_psi(X_ref[:, i], X_live[:, i])
        ks_stat, ks_p = dm.ks_test(X_ref[:, i], X_live[:, i])
        sev = "high" if psi >= 0.25 else "moderate" if psi >= 0.1 else "low"
        if psi >= 0.25:
            drifted += 1
        per_feature.append({"feature": fn, "psi": round(psi, 4), "ks_stat": round(ks_stat, 4),
                            "ks_p": round(ks_p, 4), "severity": sev})
    per_feature.sort(key=lambda x: x["psi"], reverse=True)
    frac = drifted / len(feat_names)
    verdict = "SEVERE drift" if frac > 0.5 else "MODERATE drift" if frac > 0.2 else "stable"

    report = {
        "available": True, "run_at_local": _now(), "disease": disease,
        "n_reference": int(X_ref.shape[0]), "n_live": int(X_live.shape[0]),
        "n_features": len(feat_names), "n_high_drift": drifted,
        "frac_drifted": round(frac, 3), "verdict": verdict,
        "top_drift": per_feature[:12],
        "method": "PSI (>=0.25 high) + KS-test, training-reference vs live-extractor features",
        "interpretation": ("Live EEG features differ substantially from the model's training "
                           "distribution (train/serve skew) — model confidence is not trustworthy; "
                           "human oversight required." if frac > 0.2 else
                           "Live features track the training distribution."),
        "thresholds": {"psi_high": 0.25, "psi_moderate": 0.1},
    }
    out = ROOT / "jobs" / "reports" / "drift_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    try:
        import clinical_db as cdb
        cdb.log_transaction("_system", component="drift", action="monitor",
                            detail=f"{verdict} frac={frac} high={drifted}/{len(feat_names)}")
    except Exception:
        pass
    return report


if __name__ == "__main__":
    r = run()
    if r.get("available"):
        print(f"DRIFT: {r['verdict']} — {r['n_high_drift']}/{r['n_features']} features high-PSI "
              f"(ref={r['n_reference']} vs live={r['n_live']})")
        for f in r["top_drift"][:6]:
            print(f"  {f['severity']:9s} {f['feature']:20s} PSI={f['psi']}")
    else:
        print(r)
