# Subject-Wise (Leakage-Free) Accuracy

_Generated 2026-06-23T17:17:48-06:00_

**Mean subject-wise CV accuracy: 0.9886**  ·  mean leakage gap: **0.0114**

| Disease | CV acc | ±std | CV F1 | In-sample | Leakage gap | Subjects |
|---|---|---|---|---|---|---|
| alzheimer | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | 10 |
| parkinson | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | 10 |
| schizophrenia | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | 10 |
| epilepsy | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | 10 |
| autism | 0.94 | 0.08 | 0.9398 | 1.0 | 0.06 | 10 |
| stress | 0.98 | 0.04 | 0.9798 | 1.0 | 0.02 | 10 |
| depression | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | 10 |

> ✅ **This is the defensible number for the DBA.** No subject appears in both
> train and test; a fresh model is trained per fold; the scaler is fit on the
> training fold only. The leakage gap = in-sample minus subject-wise CV.
> Note: samples are small (10 subjects/disease) — re-run on the full feature
> dataset for tighter confidence intervals.
