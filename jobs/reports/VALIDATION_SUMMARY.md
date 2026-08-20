# Validation Summary
_generated 2026-08-20T03:00:01-06:00_

| Metric | Value | 95% CI (subject bootstrap) |
|---|---|---|
| Patient-specific accuracy | 0.9805 | 0.9805 [0.973, 0.9866] |
| Patient-specific sensitivity | 0.9403 | 0.9403 [0.8662, 0.9908] |
| Cross-patient RF accuracy | 0.7277 | 0.7277 [0.4035, 0.9269] |
| Bonn external (RF) accuracy | 1.0 | 5-fold |
| ICA variance removed | 42.08% | mean |

## Honest notes
- Patient-specific (calibrated detector) is the clinical use case; high + tight CI.
- Cross-patient CI is WIDE (few subjects, chb04 hard) → generalization needs oversight.
- Bonn healthy-vs-seizure is near-perfectly separable; confirms generalization on an EASY task.
- Ensemble/normalization did NOT improve cross-patient (honest negative).
