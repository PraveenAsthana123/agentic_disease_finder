# Epilepsy Paper / Thesis — Roadmap & Evidence Map

_Honest status. ✅ have & verified · 🟡 partial · 🔴 needs work/data._

## Central contribution (the spine)
Leakage-aware evaluation of epilepsy EEG AI + a Responsible-AI governance framework
for deployment under human oversight. Headline: **patient-specific 0.98 vs cross-patient 0.73.**

## Evidence map (claim → file)
| Claim | Evidence | Status |
|---|---|---|
| Patient-specific 0.98 [0.973–0.987] | `jobs/reports/accuracy_patient_specific.json` | ✅ |
| Cross-patient 0.73 [0.40–0.93] | `jobs/reports/accuracy_all_options.json` + `bootstrap_ci_baselines.json` | ✅ |
| Ensemble/norm did NOT help (honest negative) | `accuracy_all_options.json` | ✅ |
| Subject-level bootstrap CIs | `bootstrap_ci_baselines.json` | ✅ |
| Baseline comparison (Shoeb/Truong) | `bootstrap_ci_baselines.json` | ✅ |
| Bonn 2nd dataset (easy) | `bonn_external_validation.json` | ✅ |
| ICA artifact removal (42%) | `ica_noise_cleaning.json` | ✅ |
| SHAP + surrogate explainability | `eeg_explainability.py` | ✅ |
| Fairness (DI/EO) | `rai_checks.py` | 🟡 (synthetic strata) |
| **AI–expert concordance** (unique) | `concordance_analysis.json` | 🟡 (needs more reviews) |
| Governance: audit + HITL + council | `clinical_db`, `council_orchestrator` | ✅ |

## Pending for Q1 (priority)
1. 🔴 **More subjects** — only 4 CHB-MIT on disk; download 10–20 (PhysioNet). Benchmark auto-discovers — drop EDFs in `data/real_eeg/epilepsy_physionet/chbXX/` and it scales.
2. 🔴 **Hard 2nd dataset** — TUSZ or Siena cross-dataset (Bonn too easy).
3. 🟡 **Real fairness strata** — replace synthetic protected attrs with age/sex/site.
4. 🟡 **Concordance data** — accumulate expert reviews → the unique result table.
5. 🟠 **Prospective reader study** — clinicians read with vs without AI (IRB; elevates to top-Q1).

## Buildable improvements (no new data)
- Event-level metrics (sensitivity / FAR-per-hour per seizure event)
- Probability calibration (Platt/isotonic)
- Spectrogram CNN arm (STFT already computed)
- Concordance analysis by confidence band (built; grows with reviews)

## Paper figures/tables (data ready)
T1 per-subject acc/sens + CI · F1 the 0.98→0.73 gap (`papers/q1_fig_bar_300dpi`) ·
F2 SHAP global · T2 baselines · F3 ICA before/after · T3 fairness + concordance.

## Crons (auto-refresh evidence)
VALIDATION-SUITE (Sun 06:00) re-runs all benchmarks + concordance → `VALIDATION_SUMMARY.md`.

## Realistic verdict
Now: solid **Q2 / conference**. +subjects +hard-dataset → **Q1**. +reader-study → top-Q1.
