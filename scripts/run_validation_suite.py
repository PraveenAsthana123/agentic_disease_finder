#!/usr/bin/env python3
"""Validation suite runner (cron target). Runs every benchmark + writes a
consolidated VALIDATION_SUMMARY.md so the project's evidence is always current.

Usage: python scripts/run_validation_suite.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "jobs" / "reports"
PY = sys.executable

SCRIPTS = [
    "accuracy_patient_specific.py",
    "accuracy_all_options.py",
    "bonn_external_validation.py",
    "ica_noise_cleaning.py",
    "concordance_analysis.py",
    "bootstrap_ci_baselines.py",  # last — consumes the others' outputs
]


def run(name):
    try:
        r = subprocess.run([PY, str(ROOT / "scripts" / name)], capture_output=True, text=True, timeout=900)
        return r.returncode == 0
    except Exception:
        return False


def jload(name):
    p = OUT / name
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"[validation-suite] {ts}")
    status = {n: run(n) for n in SCRIPTS}
    for n, ok in status.items():
        print(f"  {'OK ' if ok else 'FAIL'} {n}")

    ps = jload("accuracy_patient_specific.json") or {}
    allo = jload("accuracy_all_options.json") or {}
    bonn = jload("bonn_external_validation.json") or {}
    ica = jload("ica_noise_cleaning.json") or {}
    ci = jload("bootstrap_ci_baselines.json") or {}

    def ci_str(c):
        return f"{c.get('mean')} [{c.get('ci95_low')}, {c.get('ci95_high')}]" if c and c.get("ci95_low") else (c.get("mean") if c else "n/a")

    md = [f"# Validation Summary", f"_generated {ts}_", "",
          "| Metric | Value | 95% CI (subject bootstrap) |", "|---|---|---|",
          f"| Patient-specific accuracy | {ps.get('mean_accuracy','n/a')} | {ci_str(ci.get('patient_specific_accuracy_ci'))} |",
          f"| Patient-specific sensitivity | {ps.get('mean_sensitivity','n/a')} | {ci_str(ci.get('patient_specific_sensitivity_ci'))} |",
          f"| Cross-patient RF accuracy | {allo.get('options',{}).get('2_cross_patient_rf',{}).get('mean_accuracy','n/a')} | {ci_str(ci.get('cross_patient_rf_accuracy_ci'))} |",
          f"| Bonn external (RF) accuracy | {bonn.get('results',{}).get('rf',{}).get('accuracy_mean','n/a')} | 5-fold |",
          f"| ICA variance removed | {ica.get('mean_variance_removed_pct','n/a')}% | mean |",
          "",
          "## Honest notes",
          "- Patient-specific (calibrated detector) is the clinical use case; high + tight CI.",
          "- Cross-patient CI is WIDE (few subjects, chb04 hard) → generalization needs oversight.",
          "- Bonn healthy-vs-seizure is near-perfectly separable; confirms generalization on an EASY task.",
          "- Ensemble/normalization did NOT improve cross-patient (honest negative).",
          ""]
    (OUT / "VALIDATION_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {OUT / 'VALIDATION_SUMMARY.md'}")
    return 0 if all(status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
