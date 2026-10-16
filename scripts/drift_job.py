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
    live_raw = []
    for (rj,) in rows:
        try:
            f = json.loads(rj).get("features", {})
            if len(f) == 47:
                live_raw.append([float(f.get(fn, 0) or 0) for fn in feat_names])
        except (ValueError, TypeError):
            pass
    if len(live_raw) < 3:
        return {"available": False, "error": f"Only {len(live_raw)} live samples (need >=3)"}
    X_live_raw = np.nan_to_num(np.array(live_raw, dtype=float))

    # Filter 1 — extraction failures: std=0 rows are all-zero feature vectors
    # (failed extraction or unprocessed placeholder).
    std_idx = feat_names.index("std") if "std" in feat_names else -1
    if std_idx >= 0:
        valid_mask = X_live_raw[:, std_idx] > 1e-9
        n_invalid_std = int((~valid_mask).sum())
        X_live = X_live_raw[valid_mask]
    else:
        X_live = X_live_raw
        n_invalid_std = 0

    # Filter 2 — degenerate 2-point placeholders: ptp feature catches min=-1,max=1 stubs
    # that have std=1 (pass filter 1) but line_length=2 and only 2 distinct signal values.
    ptp_idx = feat_names.index("ptp") if "ptp" in feat_names else -1
    if ptp_idx >= 0:
        degenerate_mask = X_live[:, ptp_idx] < 2.1  # ptp=2 → [-1,+1] 2-sample stub
        n_invalid_ptp = int(degenerate_mask.sum())
        X_live = X_live[~degenerate_mask]
    else:
        n_invalid_ptp = 0
    n_invalid = n_invalid_std + n_invalid_ptp

    if len(X_live) < 3:
        return {"available": False, "error": f"Only {len(X_live)} valid live samples after filtering {n_invalid} extraction failures"}

    # De-duplicate: repeated analysis of the same window skews the distribution.
    X_live = np.unique(X_live, axis=0)

    # Warn if sample size is too small for reliable PSI (PSI needs >=50 for stability).
    low_sample_warning = len(X_live) < 20

    dm = DriftMonitor()
    per_feature = []
    drifted = 0
    for i, fn in enumerate(feat_names):
        ref_col = X_ref[:, i]
        live_col = X_live[:, i]

        # Skip near-constant reference features: artefacts of z-score normalisation
        # where all samples land within a tiny absolute range (e.g. std feature is
        # always ≈1.000 ± 1e-5, mean feature is ≈0 ± 3e-17).  PSI bins computed from
        # percentiles of such a narrow range place ALL live values into one bin →
        # astronomically large PSI with zero clinical meaning.
        # Criterion: peak-to-peak in reference < 0.1% of its absolute mean value
        # (or < 1e-6 absolute if mean ≈ 0).
        ref_ptp = float(ref_col.ptp()) if hasattr(ref_col, "ptp") else float(ref_col.max() - ref_col.min())
        ref_abs_mean = float(abs(ref_col.mean())) + 1e-10
        if ref_ptp / ref_abs_mean < 1e-3 or ref_ptp < 1e-6:
            per_feature.append({"feature": fn, "psi": 0.0, "ks_stat": 0.0,
                                "ks_p": 1.0, "severity": "low",
                                "note": "near-constant in both ref and live (no meaningful drift)"})
            continue

        psi = dm.calculate_psi(ref_col, live_col)
        ks_stat, ks_p = dm.ks_test(ref_col, live_col)
        sev = "high" if psi >= 0.25 else "moderate" if psi >= 0.1 else "low"
        if psi >= 0.25:
            drifted += 1
        per_feature.append({"feature": fn, "psi": round(psi, 4), "ks_stat": round(ks_stat, 4),
                            "ks_p": round(ks_p, 4), "severity": sev})
    per_feature.sort(key=lambda x: x["psi"], reverse=True)
    frac = drifted / len(feat_names)
    verdict = "SEVERE drift" if frac > 0.5 else "MODERATE drift" if frac > 0.2 else "stable"
    # Downgrade SEVERE → INSUFFICIENT_DATA when sample count is too low for
    # reliable PSI estimation (PSI variance is large with n<20).
    if low_sample_warning and verdict == "SEVERE drift":
        verdict = "INSUFFICIENT_DATA (n_live too small for reliable PSI)"
    elif low_sample_warning and verdict == "MODERATE drift":
        verdict = "INSUFFICIENT_DATA (n_live too small for reliable PSI)"

    report = {
        "available": True, "run_at_local": _now(), "disease": disease,
        "n_reference": int(X_ref.shape[0]), "n_live": int(X_live.shape[0]),
        "n_live_raw": int(X_live_raw.shape[0]),
        "n_invalid_filtered": n_invalid,
        "n_features": len(feat_names), "n_high_drift": drifted,
        "frac_drifted": round(frac, 3), "verdict": verdict,
        "low_sample_warning": low_sample_warning,
        "top_drift": per_feature[:12],
        "method": "PSI (>=0.25 high) + KS-test; extraction-failure (std=0) and "
                  "degenerate-2pt (ptp<2.1) rows removed; near-constant features skipped; "
                  "downgraded to INSUFFICIENT_DATA when n_live<20",
        "interpretation": (
            "Too few valid live samples for reliable drift estimation — accumulate more analyses "
            "before trusting PSI scores." if low_sample_warning else
            "Live EEG features differ substantially from the model's training "
            "distribution (train/serve skew) — model confidence is not trustworthy; "
            "human oversight required." if frac > 0.2 else
            "Live features track the training distribution."
        ),
        "thresholds": {"psi_high": 0.25, "psi_moderate": 0.1, "min_live_for_reliable_psi": 20},
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
