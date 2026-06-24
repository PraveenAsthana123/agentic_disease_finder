#!/usr/bin/env python3
"""Subject-level bootstrap confidence intervals + published-baseline comparison.

(1) Bootstrap CIs by RESAMPLING SUBJECTS (not windows) — the statistically correct
    method when windows from the same patient are not independent.
(2) Baseline table: our numbers vs published CHB-MIT seizure-detection methods.

Reads the real benchmark outputs produced this session. Usage:
  python scripts/bootstrap_ci_baselines.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "jobs" / "reports"
RNG = np.random.RandomState(42)


def subject_bootstrap(per_subject_acc, n_boot=2000):
    """CI by resampling SUBJECTS with replacement (correct for grouped data)."""
    accs = np.array(per_subject_acc, dtype=float)
    n = len(accs)
    if n < 2:
        return {"mean": round(float(accs.mean()), 4), "ci95": None, "note": "n<2, no CI"}
    means = [RNG.choice(accs, size=n, replace=True).mean() for _ in range(n_boot)]
    return {"mean": round(float(accs.mean()), 4),
            "ci95_low": round(float(np.percentile(means, 2.5)), 4),
            "ci95_high": round(float(np.percentile(means, 97.5)), 4),
            "n_subjects": n, "n_boot": n_boot}


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
              "method": "subject-level bootstrap (resample subjects, 2000 iters) — correct for non-independent windows"}

    # Patient-specific CIs
    ps = load("accuracy_patient_specific.json")
    if ps:
        accs = [r["accuracy"] for r in ps["per_subject"]]
        sens = [r["sensitivity"] for r in ps["per_subject"]]
        report["patient_specific_accuracy_ci"] = subject_bootstrap(accs)
        report["patient_specific_sensitivity_ci"] = subject_bootstrap(sens)

    # Cross-patient CIs
    allo = load("accuracy_all_options.json")
    if allo:
        rf = [f["accuracy"] for f in allo["options"]["2_cross_patient_rf"]["folds"]]
        report["cross_patient_rf_accuracy_ci"] = subject_bootstrap(rf)

    # Published-baseline comparison table (CHB-MIT seizure detection)
    report["baseline_comparison"] = {
        "note": "Comparison is indicative; methods differ in windows/channels/metrics. Our patient-specific uses temporal split + ensemble; cross-patient is leave-one-subject-out.",
        "methods": [
            {"method": "Shoeb (2010) patient-specific SVM", "setting": "patient-specific", "reported": "~0.96 sensitivity", "source": "MIT thesis / CHB-MIT origin"},
            {"method": "Truong et al. (2018) CNN", "setting": "patient-specific", "reported": "~0.97 AUC", "source": "Neural Networks"},
            {"method": "Ours (ensemble, temporal split)", "setting": "patient-specific", "reported": f"{ps['mean_accuracy'] if ps else 'n/a'} acc / {ps['mean_sensitivity'] if ps else 'n/a'} sens", "source": "this project (4 subjects)"},
            {"method": "Typical cross-patient (literature)", "setting": "cross-patient", "reported": "0.65-0.85", "source": "various"},
            {"method": "Ours RF (leave-one-subject-out)", "setting": "cross-patient", "reported": f"{allo['options']['2_cross_patient_rf']['mean_accuracy'] if allo else 'n/a'} acc", "source": "this project (4 subjects)"},
        ]}

    (OUT / "bootstrap_ci_baselines.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== SUBJECT-LEVEL BOOTSTRAP CIs ===")
    for k in ("patient_specific_accuracy_ci", "patient_specific_sensitivity_ci", "cross_patient_rf_accuracy_ci"):
        if k in report:
            c = report[k]
            ci = f"[{c.get('ci95_low')}, {c.get('ci95_high')}]" if c.get("ci95_low") else "n/a"
            print(f"  {k}: mean={c['mean']} 95%CI={ci}")
    print(f"Saved: {OUT / 'bootstrap_ci_baselines.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
