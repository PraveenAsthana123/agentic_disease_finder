#!/usr/bin/env python3
"""Deterministic 'what to build next' picker for the autonomous pending-completion loop.
Outputs the ordered queue of BUILDABLE pending items (excludes blocked + gated).
The loop: pick top → build → verify → commit → repeat until queue empty / blocked / stop."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

# Curated buildable queue (highest value first). Blocked/gated items excluded by design.
BUILDABLE = [
    {"id": "ictal_interictal", "title": "Ictal/interictal retrain (same-setup, removes dataset confound)", "value": "P0", "effort": "high"},
]
# Already built (removed from queue):
# model_comparison (Model Comparison Dashboard — 224 training runs, 6 model types (XGBoost/LightGBM/RandomForest/MLP/SVM/LogReg), 4 tasks (seizure_detection/eeg_classification/anomaly_detection/seizure_prediction), 4 EEG datasets (bonn_eeg/chb_mit/tuh_eeg/internal_clinical), best accuracy 98% XGBoost_v3, avg accuracy 86.1% avg AUC 88.7%, leaderboard with sort by accuracy/AUC/F1/speed, per-type/per-task breakdowns, monthly trend, 3 endpoints /api/model-comparison/overview|breakdown|definitions verified 200, portal-next/app/model-comparison/page.jsx + nav wired)
# pnes_screening_frontend (PNES Screening Dashboard — 93 screenings, 29 patients, 4 classification tiers (pnes_likely/epilepsy_likely/mixed/indeterminate), semiological scores (eye closure/pelvic thrusting/side-to-side head/ictal crying/memory recall/gradual onset), psychiatric comorbidities, EEG ictal/interictal flags, referral reasons, monthly trend, 3 endpoints /api/pnes-screening/overview|breakdown|definitions verified 200, portal-next/app/pnes-screening/page.jsx + nav wired)
# video_eeg (Video EEG Monitoring Dashboard — 46 sessions, 26 patients, CHB-MIT 21 EEG seizure annotations, seizure timeline 25 clinical diary events, aura/trigger/temporal analysis, clinical-EEG concordance, 3 endpoints /api/video-eeg/overview|breakdown|definitions verified 200, portal-next/app/video-eeg/page.jsx + nav wired, dataset_coverage VEEG scaffold→built)
# billing_claims (Billing & Claims Dashboard — 150 claims, 40 patients, 7 statuses, 6 insurers, 8 service types, collection rate/denial rate/aging/per-patient breakdown, 3 endpoints /api/billing-claims/overview|breakdown|definitions verified 200, portal-next/app/billing-claims/page.jsx + nav wired)
# safety_network (Patient Safety Network Dashboard — composite per-patient safety score from 5 dimensions: caregiver coverage/emergency readiness/medication adherence/wearable monitoring/IoT alert burden, 40 patients, Critical/At Risk/Adequate/Strong tiers, 3 endpoints /api/safety-network/overview|breakdown|definitions verified 200, portal-next/app/safety-network/page.jsx + nav wired)
# clinical_risk_stratification (Clinical Risk Stratification Dashboard — composite per-patient epilepsy risk scoring from 6 tables: seizure_diary/medication_adherence/pharmacogenomics/comorbidities/pro_outcomes/patients, Critical/High/Moderate/Low tiers, 3 endpoints /api/clinical-risk-stratification/overview|breakdown|definitions verified 200, ClinicalRiskStratificationDashboard.jsx + nav wired)
# portal_tabs (Portal Tabs Dashboard — 11 patient self-service portal tabs, 3 endpoints verified 200, PortalTabsDashboard.jsx + nav wired)
# seizure_timeline, spike_overlay (in eeg_viz), lateralization (in eeg_viz),
# patient_compare, cognitive_tests (endpoint + panel + scoring)
# expert_pharmacist, expert_nurse, expert_slp, expert_ot, expert_dietitian,
# expert_psychologist, expert_coordinator, expert_social_worker, data_archival
# cdm_label_val (Label/Annotation QC — endpoints live on :8010)
# verbal_fluency (Verbal Fluency FAS+Category dashboard — 4 endpoints + panel)
# neuro_scales (Clinical Scales dashboard — catalog endpoint + 23 scales + Next.js page)
# expert_dashboards (Montage Comparison + Localization + False Alarm Review — ExpertDashboard.jsx + nav wired)
# dataset_validation (Dataset Validation Dashboard — real clinical.db + CHB-MIT validation, endpoint /api/data-manager/dataset-validation verified 200)
# captum_lime_xai (Captum IG+FA + LIME endpoints verified 200 — registry updated cataloged→built)
# torchmetrics_deepchecks (TorchMetrics + Deepchecks dashboards + endpoints verified 200)
# icalabel_ge (ICLabel + Great Expectations dashboards verified 200)
# aif360_bias (AIF360 bias detection dashboard — 3 endpoints verified 200, registry cataloged→built)
# torcheeg (TorchEEG dashboard — 5 transforms + EEGNet-Mini classifier, 3 endpoints verified 200)
# inference_gpu (Inference/GPU Dashboard — real nvidia-smi + model scan + system info, 3 endpoints verified 200)
# medication_dashboard (Medication Dashboard — 6 endpoints + Next.js page + nav wired, real clinical.db prescriptions/schedule/adherence/warnings/side-effects)
# knowledge_graph (Knowledge Graph Dashboard — real clinical.db + ChromaDB entity-relationship graph, 81 nodes 191 edges, 3 endpoints verified 200)
# devops_cicd (DevOps/CI-CD Dashboard — real git analytics, DORA metrics, pipeline/cron status, 3 endpoints verified 200)
# ai_risk (AI Risk Dashboard — real clinical.db risk register, severity scoring, alert trends, guardrail blocks, 3 endpoints verified 200)
# mri_brain_review (MRI Brain Review Dashboard — real clinical.db mri_findings, 40 patients, lesion types/classification/volumetrics/concordance, 3 endpoints verified 200, Next.js page + nav wired)
# observability (Observability Dashboard — real transaction_log 596 events, 25 components, log levels + latency percentiles + trace correlation + alerts, 3 endpoints verified 200, Next.js page + nav wired)
# decision_ai (Decision AI Dashboard — real clinical.db decision routing + HITL overrides + audit trail, 3 endpoints verified 200, DecisionAIDashboard.jsx + nav wired)
# data_lineage (Data Lineage Dashboard — real transaction_log 645 events, 25 components, pipeline stages + lineage edges + audit trail, 3 endpoints verified 200, DataLineageDashboard.jsx + nav wired)
# time_series_ai (Time-Series AI Dashboard — EEG spectral decomposition, band power, complexity metrics, 47 features, 21 analyses, 14 patients, 3 endpoints verified 200, TimeSeriesAIDashboard.jsx + nav wired)
# feature_evaluation (Feature Evaluation Dashboard — ANOVA F-test + correlation + clinical relevance, 3 endpoints verified 200, FeatureEvaluationDashboard.jsx + nav wired)
# data_augmentation (Data Augmentation Dashboard — Jitter/Scale/TimeWarp/Mixup/SMOTE, 3 endpoints verified 200, DataAugmentationDashboard.jsx + nav wired)
# seizure_prediction (Seizure Prediction Dashboard — wearable biomarker risk analysis + pre-ictal comparison + threshold ROC, 3 endpoints verified 200, SeizurePredictionDashboard.jsx + nav wired)
# saliency_attention (Saliency & Attention Map Dashboard — channel saliency, temporal attention, multi-head attention, per-diagnosis patterns, 3 endpoints verified 200, SaliencyAttentionDashboard.jsx + nav wired)
# patient_education (Patient Education Dashboard — real education_modules 179 rows, 30 patients, 12 topics, 4 formats, quiz/completion/engagement analytics, 3 endpoints verified 200, PatientEducationDashboard.jsx + nav wired)
# token_cost (Token / Cost Dashboard — LLM token tracking, budget utilization, component cost breakdown, rate cards, 3 endpoints verified 200, TokenCostDashboard.jsx + nav wired)
# moca_autoscoring (MoCA Auto-Scoring Dashboard — real neuropsych data, 37 assessments 30 patients, domain estimates, normative comparison, PHQ-9/GAD-7 comorbidity, 3 endpoints verified 200, MoCAAutoscoringDashboard.jsx + nav wired)
# telehealth (Telehealth Sessions Dashboard — real telehealth_sessions 109 rows, 30 patients, 6 providers, 4 session types, 4 platforms, 3 endpoints verified 200, TelehealthDashboard.jsx + nav wired)
# incident_management (Incident Management Dashboard — real uptime_log 197 incidents, auto-recovery rate, MTTR trend, hourly heatmap, track events, 3 endpoints verified 200, IncidentManagementDashboard.jsx + nav wired)
# recovery_trajectory (Recovery Trajectory Forecast Dashboard — real pro_outcomes 180 rows, 30 patients, slope analysis + risk factor correlation + intensive rehab prediction, 3 endpoints verified 200, RecoveryTrajectoryDashboard.jsx + nav wired)
# autonomic_analysis (Autonomic Analysis Dashboard — real wearable_readings 900 rows, 30 patients, ADS scoring + HRV trends + seizure-autonomic correlation + risk stratification, 3 endpoints verified 200, AutonomicAnalysisDashboard.jsx + nav wired)
# ai_roi (AI ROI Dashboard — real finops_costs 978 rows + analyses 21 + telehealth 109 + appointments 120, investment vs value, cost breakdown by category/model, patient-level ROI, 3 endpoints verified 200, AIROIDashboard.jsx + nav wired)
# patient_report (Patient-Facing Report Dashboard — real analyses 21 + assessments 423 + seizure_diary 25 + mri_findings 40 + medication_adherence 12600, plain-language patient reports, 3 endpoints verified 200, PatientReportDashboard.jsx + nav wired)
# comorbidity_analysis (Comorbidity Analysis Dashboard — real comorbidities 27 rows, psychiatric profiling, risk severity, co-occurrence matrix, screening instruments, demographics cross-tab, 3 endpoints verified 200, ComorbidityAnalysisDashboard.jsx + nav wired)
# data_completeness (Data Completeness Dashboard — real clinical.db 40 patients × 34 fields across 9 categories, per-patient/per-category completeness matrix, 3 endpoints verified 200, DataCompletenessDashboard.jsx + nav wired)
# treatment_efficacy (Treatment Efficacy Dashboard — real medication_adherence 12600 + seizure_diary 25 + pro_outcomes 180, adherence-vs-seizure correlation, per-drug analysis, treatment response categories, 3 endpoints verified 200, TreatmentEfficacyDashboard.jsx + nav wired)
# structured_reporting (Structured Reporting Dashboard — 4 ILAE-aligned templates (EEG/MRI/Neuropsych/Comprehensive), real eeg_acquisition 30 + mri_findings 40 + neuropsych 37, field completeness heatmaps, quality grades, cross-modality concordance 27 patients with all 3, AI-assisted finding capture, 3 endpoints verified 200, StructuredReportingDashboard.jsx + nav wired)
# presurgical_evaluation (Pre-Surgical Evaluation Dashboard — real mri_findings 40 + eeg_acquisition 30 + seizure_diary 25 + medications 9, ILAE candidacy scoring, lesion-type distribution, laterality analysis, workup completeness, 3 endpoints verified 200, PreSurgicalEvaluationDashboard.jsx + nav wired)
# population_health (Population Health Dashboard — real clinical.db 40 patients + seizure_diary 25 + comorbidities 27 + medications 9, age-sex pyramid, seizure epidemiology, comorbidity prevalence, risk stratification, enrollment trend, 3 endpoints verified 200, PopulationHealthDashboard.jsx + nav wired)
# pharmacogenomics (Pharmacogenomics Dashboard — real pharmacogenomics 172 rows, 40 patients, 7 genes (HLA-B/A, CYP2C9/2C19, UGT1A4, SCN1A, ABCB1), CPIC/PharmGKB evidence, metabolizer status, HLA screening, drug-gene interactions, 3 endpoints verified 200, PharmacogenomicsDashboard.jsx + nav wired)
# surgical_outcomes (Surgical Outcome Dashboard — real surgical_outcomes 28 rows, 22 patients, Engel I-IV + ILAE 1-6, 8 surgery types, complication tracking, pre/post seizure frequency, AED reduction, pathology analysis, 3 endpoints verified 200, SurgicalOutcomeDashboard.jsx + nav wired)
# system_health (System Health Monitoring Dashboard — real system_health_log 30 rows, 7 components (API/Cache/Database/Frontend/ML Pipeline/Queue/Storage), uptime KPIs, resource utilization, response time percentiles, incident tracking, 3 endpoints verified 200, SystemHealthDashboard.jsx + nav wired)
# transaction_audit (Transaction Audit Trail Dashboard — real transaction_log 1360 rows, 27 components, 26 actions, 8 actors, daily volume trends, human vs system breakdown, hourly patterns, 3 endpoints verified 200, TransactionAuditDashboard.jsx + nav wired)
# emergency_sos (Emergency SOS Dashboard — real emergency_sos_events 41 rows + emergency_contacts 30 rows, 26 patients, 5 event types, 4 trigger methods, 5 outcomes, response time analytics, contact coverage + stale verification, 3 endpoints verified 200, EmergencySOSDashboard.jsx + nav wired)
# referral_triage (Referral Triage Dashboard — real referral_records 84 rows, 30+ patients, 4 urgency levels, 9 referral reasons, 7 sources, 5 providers, triage scoring + timeline + provider workload, 3 endpoints verified 200, ReferralTriageDashboard.jsx + nav wired)
# qolie31 (QOLIE-31 Quality of Life Dashboard — real assessments 23 rows, 23 patients, 7 QoL domains (Seizure Worry/Overall QoL/Emotional Well-being/Energy-Fatigue/Cognitive Functioning/Medication Effects/Social Function), 4 severity tiers (Poor/Fair/Good/Excellent), domain comparison, severity transitions, monthly trend, 3 endpoints verified 200, QOLIE31Dashboard.jsx + nav wired)
# mcp_security (MCP Security Dashboard — real transaction_log 1548 rows, 8 actors, guardrail enforcement, actor privileges, attack surface, patient access audit, daily security trend, hourly patterns, privileged events, HITL reviews, 3 endpoints verified 200, MCPSecurityDashboard.jsx + nav wired)
# data_requirements (Data Requirements Dashboard — real data_requirements.json 58 items across 9 categories, present/partial/missing gap tracker, tier coverage, control groups, artifact template, technician deliverables, 3 endpoints verified 200, DataRequirementsDashboard.jsx + nav wired)
# tab_taxonomy (Tab Taxonomy Dashboard — real tab_taxonomy.json 35 tabs across 3 categories (Patient Master 13/Role Ops 9/AI Caps 13), 100% built, 33 mapped, as-is/to-be transformation, 3 endpoints verified 200, TabTaxonomyDashboard.jsx + nav wired)
# feature_gaps (Feature Gaps Dashboard — real feature_gaps.json 18 gaps across 6 categories (functional/technology/data/gap/architecture/decision_ai), 50-paper DL review, 100% built, priority/status tracking, 5 recommendations, 3 endpoints verified 200, FeatureGapsDashboard.jsx + nav wired)
# icd10_coding (ICD-10 Coding Dashboard — real icd10_coding_records 85 rows, 41 patients, 27 codes, AI vs human coder comparison, confidence tiers, rejection tracking, pending review queue, 3 endpoints verified 200, ICD10CodingDashboard.jsx + nav wired)
# conversation_log (Conversation Log Dashboard — real conversation_log 2364 rows, 29 active days, operator+assistant roles, daily volume trend, hourly pattern, text length analysis, 3 endpoints verified 200, ConversationLogDashboard.jsx + nav wired)
# federated_learning (Federated Learning Dashboard — real federation_rounds 18 rows + federation_sites 8 sites, 529 patients across 6 institutions, 3 aggregation methods (FedAvg/FedProx/Scaffold), privacy budget tracking, convergence curve, gradient norms, heterogeneity metrics, 3 endpoints verified 200, FederatedLearningDashboard.jsx + nav wired)
# sleep_staging (Sleep Staging Dashboard — real sleep_staging 95 rows, 40 patients, 5 study types (PSG/ambulatory-EEG/overnight-EEG/routine-EEG/video-EEG-LTM), AASM staging, efficiency distribution, seizure-sleep interaction, IED NREM activation, OSA/PLM comorbidity, 3 endpoints verified 200, SleepStagingDashboard.jsx + nav wired)
# seizure_triggers (Seizure Trigger Analysis Dashboard — real seizure_trigger_logs 203 rows, 40 patients, 9 trigger types, 4 sleep quality levels, 6 ILAE seizure types, trigger-specific seizure rates, lifestyle risk comparisons, sleep-quality vs seizure-occurrence, per-patient summaries, 3 endpoints /api/seizure-triggers/overview|breakdown|definitions verified 200, SeizureTriggerDashboard.jsx + nav wired)
# cognitive_decline (Cognitive Decline Tracking Dashboard — real neuropsych data, 20 patients, 96 assessments, MoCA/MMSE/domain slopes, 5 classification tiers, risk stratification, 3 endpoints /api/cognitive-decline/overview|breakdown|definitions verified 200, CognitiveDeclineDashboard.jsx + nav wired)
# adl (Activities of Daily Living Dashboard — real assessments 69 rows, 25 patients, 3 instruments (Barthel Index/QOLIE-31/Epworth Sleepiness Scale), patient profiles with overall severity, score distributions, level breakdowns (normal/moderate/severe), 3 endpoints /api/adl/overview|breakdown|definitions verified 200, portal-next/app/adl/page.jsx + nav wired)
BLOCKED =["Gmail/Slack/Drive live (credentials)", "Multi-user auth/RBAC", "EMR/FHIR + device streaming"]
GATED = ["git push (operator approval, §42)"]


def main():
    as_json = "--json" in sys.argv
    if as_json:
        print(json.dumps({"buildable": BUILDABLE, "blocked": BLOCKED, "gated": GATED}, indent=2)); return 0
    print("════ NEXT BUILDABLE (autonomous loop queue) ════")
    for i, b in enumerate(BUILDABLE):
        print(f"  {i+1:2d}. [{b['value']}] {b['title']}  ({b['effort']})")
    print(f"\n▸ TOP PICK: {BUILDABLE[0]['title']}")
    print(f"\n🔒 BLOCKED (need operator): {' · '.join(BLOCKED)}")
    print(f"🔴 GATED (need go-ahead): {' · '.join(GATED)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
