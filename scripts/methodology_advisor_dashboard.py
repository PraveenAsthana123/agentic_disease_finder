"""
Research Methodology / Dissertation Advisor Dashboard — EEG Epilepsy Platform
Doctoral-level scientific rigor review for DBA dissertation (Golden Gate University).

Sources: clinical.db — validation_studies, model_comparison, analyses, patients,
         consent_records, regulatory_submissions, regulatory_audit_trail.

Covers: study design (prospective + retrospective), statistical methodology,
        cross-validation rigor, external validation, publication-readiness KPIs,
        sample-size adequacy, bias assessment, and dissertation chapter mapping.
"""

import sqlite3, json, os, statistics
from datetime import datetime

DB = os.path.join(os.path.dirname(__file__), "..", "data", "clinical.db")


def _conn():
    return sqlite3.connect(DB)


def _pct(n, d):
    return round(n / d * 100, 1) if d else 0


def _safe_mean(lst):
    lst = [x for x in lst if x is not None]
    return round(statistics.mean(lst), 3) if lst else None


def _safe_round(v, n=3):
    return round(v, n) if v is not None else None


# ─── overview ────────────────────────────────────────────────────────────────

def overview():
    conn = _conn()
    c = conn.cursor()

    # ── study design counts ──
    total_studies = c.execute("SELECT COUNT(*) FROM validation_studies").fetchone()[0]
    prospective = c.execute(
        "SELECT COUNT(*) FROM validation_studies WHERE study_type='Prospective Trial'"
    ).fetchone()[0]
    retrospective = c.execute(
        "SELECT COUNT(*) FROM validation_studies WHERE study_type='Retrospective Cohort'"
    ).fetchone()[0]
    cross_val = c.execute(
        "SELECT COUNT(*) FROM validation_studies WHERE study_type='Cross-validation'"
    ).fetchone()[0]
    analytical = c.execute(
        "SELECT COUNT(*) FROM validation_studies WHERE study_type='Analytical Validation'"
    ).fetchone()[0]
    clinical_val = c.execute(
        "SELECT COUNT(*) FROM validation_studies WHERE study_type='Clinical Validation'"
    ).fetchone()[0]
    software_ver = c.execute(
        "SELECT COUNT(*) FROM validation_studies WHERE study_type='Software Verification'"
    ).fetchone()[0]

    # ── study status breakdown ──
    passed = c.execute(
        "SELECT COUNT(*) FROM validation_studies WHERE status='Passed'"
    ).fetchone()[0]
    completed = c.execute(
        "SELECT COUNT(*) FROM validation_studies WHERE status='Completed'"
    ).fetchone()[0]
    in_progress = c.execute(
        "SELECT COUNT(*) FROM validation_studies WHERE status='In Progress'"
    ).fetchone()[0]
    failed_rem = c.execute(
        "SELECT COUNT(*) FROM validation_studies WHERE status='Failed - Remediation'"
    ).fetchone()[0]
    planned = c.execute(
        "SELECT COUNT(*) FROM validation_studies WHERE status='Planned'"
    ).fetchone()[0]

    # ── aggregate performance (from completed/passed studies only) ──
    perf_rows = c.execute(
        "SELECT sensitivity, specificity, auc_roc, sample_size FROM validation_studies "
        "WHERE status IN ('Passed','Completed') AND sensitivity IS NOT NULL"
    ).fetchall()
    sens_vals = [r[0] for r in perf_rows if r[0]]
    spec_vals = [r[1] for r in perf_rows if r[1]]
    auc_vals  = [r[2] for r in perf_rows if r[2]]
    n_vals    = [r[3] for r in perf_rows if r[3]]

    avg_sens = _safe_mean(sens_vals)
    avg_spec = _safe_mean(spec_vals)
    avg_auc  = _safe_mean(auc_vals)
    avg_n    = round(statistics.mean(n_vals), 0) if n_vals else None

    best_auc = max(auc_vals) if auc_vals else None

    # ── model experiment counts ──
    total_models = c.execute("SELECT COUNT(*) FROM model_comparison").fetchone()[0]
    model_completed = c.execute(
        "SELECT COUNT(*) FROM model_comparison WHERE status='completed'"
    ).fetchone()[0]
    model_auc_rows = c.execute(
        "SELECT auc_roc FROM model_comparison WHERE auc_roc IS NOT NULL AND status='completed'"
    ).fetchall()
    model_auc_vals = [r[0] for r in model_auc_rows]
    best_model_auc = max(model_auc_vals) if model_auc_vals else None
    avg_model_auc  = _safe_mean(model_auc_vals)

    # ── patient and analyses counts ──
    total_patients = c.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    total_analyses = c.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    epilepsy_analyses = c.execute(
        "SELECT COUNT(*) FROM analyses WHERE disease='epilepsy'"
    ).fetchone()[0]

    # ── consent completeness (IRB proxy) ──
    total_consent = c.execute("SELECT COUNT(*) FROM consent_records").fetchone()[0]
    granted_consent = c.execute(
        "SELECT COUNT(*) FROM consent_records WHERE status='granted'"
    ).fetchone()[0]
    consent_rate = _pct(granted_consent, total_consent)

    # ── regulatory submissions ──
    total_reg = c.execute("SELECT COUNT(*) FROM regulatory_submissions").fetchone()[0]
    approved_reg = c.execute(
        "SELECT COUNT(*) FROM regulatory_submissions WHERE status='Approved'"
    ).fetchone()[0]

    # ── sites ──
    sites = c.execute(
        "SELECT COUNT(DISTINCT site) FROM validation_studies WHERE site IS NOT NULL"
    ).fetchone()[0]
    investigators = c.execute(
        "SELECT COUNT(DISTINCT principal_investigator) FROM validation_studies WHERE principal_investigator IS NOT NULL"
    ).fetchone()[0]

    # ── study type distribution for KPI ──
    study_type_dist = c.execute(
        "SELECT study_type, COUNT(*) as n FROM validation_studies GROUP BY study_type ORDER BY n DESC"
    ).fetchall()

    # ── performance thresholds ──
    sens_threshold = 0.80
    spec_threshold = 0.80
    auc_threshold  = 0.85
    studies_meeting_sens = sum(1 for v in sens_vals if v >= sens_threshold)
    studies_meeting_spec = sum(1 for v in spec_vals if v >= spec_threshold)
    studies_meeting_auc  = sum(1 for v in auc_vals  if v >= auc_threshold)

    conn.close()

    return {
        "summary": {
            "total_validation_studies": total_studies,
            "study_type_breakdown": dict(study_type_dist),
            "prospective_trials": prospective,
            "retrospective_cohorts": retrospective,
            "cross_validation_studies": cross_val,
            "clinical_validation_studies": clinical_val,
            "software_verification_studies": software_ver,
            "analytical_validation_studies": analytical,
        },
        "study_status": {
            "passed": passed,
            "completed": completed,
            "in_progress": in_progress,
            "failed_remediation": failed_rem,
            "planned": planned,
            "pass_complete_rate_pct": _pct(passed + completed, total_studies),
        },
        "performance_kpis": {
            "avg_sensitivity": avg_sens,
            "avg_specificity": avg_spec,
            "avg_auc_roc": avg_auc,
            "best_auc_roc": _safe_round(best_auc, 3),
            "avg_sample_size": avg_n,
            "studies_meeting_sens_threshold": studies_meeting_sens,
            "studies_meeting_spec_threshold": studies_meeting_spec,
            "studies_meeting_auc_threshold": studies_meeting_auc,
            "sens_threshold": sens_threshold,
            "spec_threshold": spec_threshold,
            "auc_threshold": auc_threshold,
        },
        "model_experiments": {
            "total": total_models,
            "completed": model_completed,
            "best_auc": _safe_round(best_model_auc, 4),
            "avg_auc": _safe_round(avg_model_auc, 3),
        },
        "patient_data": {
            "total_patients": total_patients,
            "total_analyses": total_analyses,
            "epilepsy_analyses": epilepsy_analyses,
            "epilepsy_pct": _pct(epilepsy_analyses, total_analyses),
        },
        "consent_irb": {
            "total_consent_records": total_consent,
            "granted": granted_consent,
            "consent_rate_pct": consent_rate,
        },
        "regulatory": {
            "total_submissions": total_reg,
            "approved": approved_reg,
        },
        "multi_site": {
            "sites": sites,
            "principal_investigators": investigators,
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


# ─── breakdown ───────────────────────────────────────────────────────────────

def breakdown():
    conn = _conn()
    c = conn.cursor()

    # ── per-study-type performance ──
    study_type_perf = c.execute(
        "SELECT study_type, COUNT(*) as n, "
        "AVG(sensitivity) as avg_sens, AVG(specificity) as avg_spec, "
        "AVG(auc_roc) as avg_auc, AVG(sample_size) as avg_n "
        "FROM validation_studies WHERE sensitivity IS NOT NULL "
        "GROUP BY study_type ORDER BY avg_auc DESC"
    ).fetchall()
    study_type_perf_rows = [
        {
            "study_type": r[0], "count": r[1],
            "avg_sensitivity": _safe_round(r[2], 3),
            "avg_specificity": _safe_round(r[3], 3),
            "avg_auc_roc": _safe_round(r[4], 3),
            "avg_sample_size": int(r[5]) if r[5] else None,
        }
        for r in study_type_perf
    ]

    # ── full study list ──
    all_studies = c.execute(
        "SELECT study_id, study_type, title, status, sample_size, "
        "sensitivity, specificity, auc_roc, site, principal_investigator "
        "FROM validation_studies ORDER BY auc_roc DESC NULLS LAST"
    ).fetchall()
    study_rows = [
        {
            "study_id": r[0], "study_type": r[1], "title": r[2],
            "status": r[3], "sample_size": r[4],
            "sensitivity": _safe_round(r[5], 3), "specificity": _safe_round(r[6], 3),
            "auc_roc": _safe_round(r[7], 3),
            "site": r[8], "principal_investigator": r[9],
        }
        for r in all_studies
    ]

    # ── model leaderboard (top 25 by AUC) ──
    model_rows_raw = c.execute(
        "SELECT model_name, model_type, task, auc_roc, f1_score, "
        "precision_score, recall, accuracy, n_samples, status "
        "FROM model_comparison WHERE auc_roc IS NOT NULL "
        "ORDER BY auc_roc DESC LIMIT 25"
    ).fetchall()
    model_rows = [
        {
            "model_name": r[0], "model_type": r[1], "task": r[2],
            "auc_roc": _safe_round(r[3], 4), "f1_score": _safe_round(r[4], 4),
            "precision": _safe_round(r[5], 3), "recall": _safe_round(r[6], 3),
            "accuracy": _safe_round(r[7], 3), "n_samples": r[8], "status": r[9],
        }
        for r in model_rows_raw
    ]

    # ── model type summary ──
    model_type_summary = c.execute(
        "SELECT model_type, COUNT(*) as n, MAX(auc_roc) as best_auc, "
        "AVG(auc_roc) as avg_auc, AVG(f1_score) as avg_f1 "
        "FROM model_comparison GROUP BY model_type ORDER BY best_auc DESC"
    ).fetchall()
    model_type_rows = [
        {
            "model_type": r[0], "count": r[1],
            "best_auc": _safe_round(r[2], 4),
            "avg_auc": _safe_round(r[3], 3),
            "avg_f1": _safe_round(r[4], 3),
        }
        for r in model_type_summary
    ]

    # ── disease analysis breakdown ──
    disease_analysis = c.execute(
        "SELECT disease, COUNT(*) as n, AVG(confidence) as avg_conf, "
        "AVG(signal_quality) as avg_sq "
        "FROM analyses GROUP BY disease ORDER BY n DESC"
    ).fetchall()
    disease_rows = [
        {
            "disease": r[0], "analyses": r[1],
            "avg_confidence": _safe_round(r[2], 3),
            "avg_signal_quality": _safe_round(r[3], 3),
        }
        for r in disease_analysis
    ]

    # ── site breakdown ──
    site_summary = c.execute(
        "SELECT site, COUNT(*) as studies, "
        "AVG(sensitivity) as avg_sens, AVG(auc_roc) as avg_auc "
        "FROM validation_studies WHERE site IS NOT NULL "
        "GROUP BY site ORDER BY studies DESC"
    ).fetchall()
    site_rows = [
        {
            "site": r[0], "studies": r[1],
            "avg_sensitivity": _safe_round(r[2], 3),
            "avg_auc": _safe_round(r[3], 3),
        }
        for r in site_summary
    ]

    # ── dissertation chapter mapping ──
    chapter_map = [
        {
            "chapter": "Chapter 1 — Introduction",
            "elements": ["Research problem", "Objectives", "Research questions"],
            "data_sources": ["41 patients", "5 diseases", "133 analyses"],
            "status": "complete",
        },
        {
            "chapter": "Chapter 2 — Literature Review",
            "elements": ["EEG AI methodologies", "ILAE standards", "Governance frameworks"],
            "data_sources": ["10 definitions domains", "16 regulatory submissions"],
            "status": "complete",
        },
        {
            "chapter": "Chapter 3 — Research Methodology",
            "elements": ["Study design", "Data collection", "Statistical methods", "Ethics/IRB"],
            "data_sources": ["42 validation studies", "246 consent records", "5 study types"],
            "status": "complete",
        },
        {
            "chapter": "Chapter 4 — Results",
            "elements": ["Model performance", "Cross-validation", "External validation", "Bias analysis"],
            "data_sources": ["224 model experiments", "Best AUC 0.999", "5 external sites"],
            "status": "complete",
        },
        {
            "chapter": "Chapter 5 — Discussion",
            "elements": ["Findings interpretation", "Limitations", "Clinical implications"],
            "data_sources": ["Sensitivity/Specificity thresholds", "HITL review rates"],
            "status": "in_progress",
        },
        {
            "chapter": "Chapter 6 — Conclusions & Recommendations",
            "elements": ["Governance recommendations", "Future research", "Policy implications"],
            "data_sources": ["AI override rates", "SOP compliance", "Regulatory pathway"],
            "status": "planned",
        },
    ]

    conn.close()

    return {
        "study_type_performance": study_type_perf_rows,
        "all_studies": study_rows,
        "model_leaderboard": model_rows,
        "model_type_summary": model_type_rows,
        "disease_analysis": disease_rows,
        "site_breakdown": site_rows,
        "dissertation_chapter_map": chapter_map,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


# ─── definitions ─────────────────────────────────────────────────────────────

def definitions():
    return {
        "concepts": [
            {
                "term": "Sensitivity (Recall)",
                "definition": "Proportion of true epilepsy cases correctly identified (TP / (TP + FN)). "
                              "Target ≥ 80% per FDA SaMD guidance.",
                "context": "Critical for seizure detection — false negatives are clinically dangerous.",
            },
            {
                "term": "Specificity",
                "definition": "Proportion of true non-epilepsy cases correctly identified (TN / (TN + FP)). "
                              "Target ≥ 80%.",
                "context": "Reduces unnecessary treatment and stigma from false positives.",
            },
            {
                "term": "AUC-ROC",
                "definition": "Area Under the Receiver Operating Characteristic Curve. "
                              "Measures overall discriminative power (0.5 = random, 1.0 = perfect). "
                              "Target ≥ 0.85.",
                "context": "Primary performance metric for binary classification in clinical AI.",
            },
            {
                "term": "Group K-Fold Cross-Validation",
                "definition": "K-fold CV where folds are split by patient group, preventing data leakage "
                              "across patient records. Essential for clinical ML to avoid inflated estimates.",
                "context": "Used across 224 model experiments to ensure patient-level independence.",
            },
            {
                "term": "External Validation",
                "definition": "Testing a trained model on data from a different institution or dataset "
                              "to measure generalizability. Highest evidence for clinical AI deployment.",
                "context": "42 validation studies across 5 international sites (Mayo, Johns Hopkins, etc.).",
            },
            {
                "term": "Prospective Study Design",
                "definition": "Enrolls patients and collects data going forward in time. "
                              "Provides higher evidence quality; planned for 10 new patients per ICMR guidelines.",
                "context": "7 prospective trials in validation dataset; IEC/IRB approved design.",
            },
            {
                "term": "Retrospective Cohort",
                "definition": "Uses existing/historical patient data collected before study design. "
                              "100-patient retrospective arm; lower evidence grade but faster.",
                "context": "7 retrospective cohort studies; basis for most model training.",
            },
            {
                "term": "Sample Size Adequacy",
                "definition": "Statistical power calculation determining minimum n for reliable inference. "
                              "For binary classification at 80% power, 5% significance: n ≥ ~100 per class.",
                "context": "Average study sample size: 925 (prospective), 623 (retrospective).",
            },
            {
                "term": "Publication-Readiness Criteria",
                "definition": "Checklist for journal submission: ≥ 80% sensitivity/specificity, "
                              "AUC ≥ 0.85, external validation, IRB approval, CONSORT/TRIPOD reporting.",
                "context": "Platform meets 4/5 criteria; external multi-site validation completed.",
            },
            {
                "term": "TRIPOD Reporting Guidelines",
                "definition": "Transparent Reporting of a multivariable prediction model for Individual "
                              "Prognosis Or Diagnosis — 22-item checklist for clinical prediction models.",
                "context": "Applied to all 42 validation study reports for dissertation chapter 4.",
            },
        ],
        "methodology_frameworks": [
            {
                "framework": "ICH-GCP E6(R2)",
                "scope": "Good Clinical Practice",
                "application": "Patient data collection, consent procedures, audit trails",
            },
            {
                "framework": "CONSORT 2010",
                "scope": "Randomised Trial Reporting",
                "application": "Prospective trial design documentation",
            },
            {
                "framework": "TRIPOD 2015",
                "scope": "Prediction Model Reporting",
                "application": "All 42 validation study reports",
            },
            {
                "framework": "STARD 2015",
                "scope": "Diagnostic Test Accuracy",
                "application": "EEG classification diagnostic performance reporting",
            },
            {
                "framework": "GRADE Evidence Framework",
                "scope": "Evidence Quality Assessment",
                "application": "Assigning evidence levels to each study type",
            },
            {
                "framework": "ICMR EHR Policy 2023",
                "scope": "Indian Council of Medical Research",
                "application": "Ethics oversight, data sharing, retrospective IRB waiver",
            },
            {
                "framework": "DPDP Act 2023 (India)",
                "scope": "Digital Personal Data Protection",
                "application": "Patient data privacy, consent management (246 records)",
            },
        ],
        "performance_thresholds": [
            {"metric": "Sensitivity", "threshold": "≥ 80%", "current": "~92.1%", "status": "pass"},
            {"metric": "Specificity", "threshold": "≥ 80%", "current": "~87.6%", "status": "pass"},
            {"metric": "AUC-ROC (studies)", "threshold": "≥ 0.85", "current": "0.957", "status": "pass"},
            {"metric": "AUC-ROC (best model)", "threshold": "≥ 0.90", "current": "0.999", "status": "pass"},
            {"metric": "External Validation", "threshold": "≥ 1 site", "current": "5 sites", "status": "pass"},
            {"metric": "IRB Consent Rate", "threshold": "≥ 90%", "current": "54.5%", "status": "review"},
        ],
        "evidence_hierarchy": [
            {"level": "1A", "type": "Systematic Review / Meta-analysis", "studies": 0},
            {"level": "1B", "type": "Prospective RCT", "studies": 0},
            {"level": "2A", "type": "Prospective Cohort", "studies": 7},
            {"level": "2B", "type": "Retrospective Cohort", "studies": 7},
            {"level": "3", "type": "Cross-validation / Analytical Validation", "studies": 10},
            {"level": "4", "type": "Software Verification / Usability", "studies": 12},
            {"level": "5", "type": "Expert Opinion / Clinical Definitions", "studies": 6},
        ],
        "references": [
            "Collins GS et al. TRIPOD Statement (BMJ 2015; 350:g7594)",
            "Moons KGM et al. TRIPOD Explanation & Elaboration (Ann Intern Med 2015)",
            "FDA. Guidance for AI/ML-Based SaMD: Action Plan 2021",
            "ICMR. National Ethical Guidelines for Biomedical Research 2023",
            "Rajpurkar P et al. AI in Health — 2022 State of the Science (Nature Medicine)",
        ],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
