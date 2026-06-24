# NeuroLab AI — Plan, Status & Roadmap

_Honest status. built = runs end-to-end & verified · partial = pieces exist · planned = needs lib/infra · n/a = out of scope for a research DBA._

## 1. What's REAL (built + verified this project)
| Capability | Evidence |
|---|---|
| EEG AI seizure read (47 features, RF/ensemble) | `eeg_analysis_pipeline.py`, `/api/analyze-upload` |
| **Patient-specific accuracy 0.98** (temporal split, no leakage) | `scripts/accuracy_patient_specific.py` → `accuracy_patient_specific.json` |
| **Cross-patient 0.73** (leave-one-subject-out) | `scripts/accuracy_all_options.py` |
| **Subject-level bootstrap CIs** | `scripts/bootstrap_ci_baselines.py` |
| **Bonn 2nd-dataset validation** | `scripts/bonn_external_validation.py` |
| **ICA noise cleaning** (28-60% variance removed) | `scripts/ica_noise_cleaning.py` |
| SHAP + surrogate-tree explainability | `eeg_explainability.py` |
| Fairness (DI/EO) + PII + guardrails | `rai_checks.py` |
| Council-of-Agents governed flow | `council_orchestrator.py` |
| Decision audit trail (UTC+local) | `clinical_db.py` transaction_log |
| Vector RAG conversation AI (ChromaDB+Ollama) | `clinical_db.patient_chat`, VECTOR-INGEST cron |
| 8 coverage registries (pipelines/dashboards/issues/roles/sim/tests/neurolab) | `config/*.json` + UI |

## 2. The honest accuracy story (the thesis spine)
- **Patient-specific: 0.98** [CI ~0.97–0.99] — clinically deployable for a *known* patient (CHB-MIT's design purpose).
- **Cross-patient: 0.73** [CI ~0.40–0.93] — generalization to *new* patients is unstable → **human oversight required.**
- Bonn (healthy-vs-seizure): ~1.00 — confirms generalization on an *easy* task; CHB-MIT cross-patient is the hard one.
- Ensemble & per-subject normalization did **not** help cross-patient (honest negatives).

## 3. Buildable depth fixes — DONE
- [x] ICA noise cleaning
- [x] Subject-level bootstrap CIs (correct stats for non-independent windows)
- [x] Baseline comparison table (vs Shoeb 2010, Truong 2018)
- [x] Bonn second-dataset validation
- [x] Validation suite + weekly cron (`VALIDATION-SUITE`, Sun 06:00)

## 4. Deployment gaps (planned / integrate, don't rebuild)
| Item | Status | Strategy |
|---|---|---|
| Multi-user auth + RBAC | planned | add before any real deploy |
| EMR/HIS (HL7/FHIR) | planned | integrate existing |
| Real device streaming | planned | partner w/ amplifier vendor |
| Video-EEG sync player | planned | clinical phase |
| DICOM imaging | planned | integrate viewer |
| Scheduling / billing / portal | planned | buy/integrate |
| Clinical validation (prospective) | planned | IRB study — the real credibility step |
| Regulatory (FDA/CE) | n/a (research) | only if diagnostic claim |

## 5. Crons (scheduled jobs)
| Cron | Schedule | Purpose |
|---|---|---|
| VECTOR-INGEST | 07:00 + 19:00 | push clinical data → vector DB for RAG |
| CLINICAL-DB-AUDIT | 08:00 + 20:00 | DB backup + integrity |
| THESIS-ASSET-REFRESH | 09:00 + 21:00 | gather figures + tables |
| VALIDATION-SUITE | Sun 06:00 | re-run all benchmarks → VALIDATION_SUMMARY.md |

## 6. Q1 paper / thesis readiness
- ✅ leakage finding · explainability · fairness · governance · 2nd dataset · bootstrap CIs · baseline table
- ❌ remaining for Q1: **prospective clinical validation** (the one thing code can't produce — needs an IRB study with real clinicians)

## 7. The one-line value proposition
> Built the part nobody else has — **governed, explainable AI under multi-role human oversight** — with honest leakage-aware evaluation. Integrate the commodity clinical-IT (EMR, devices, billing). The governance layer is the IP and the precondition for the ROI.
