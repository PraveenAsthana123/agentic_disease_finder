"""DBA Research KPI Dashboard — tracks Praveen Asthana's DBA program at
Golden Gate University: patient recruitment, IRB/IEC regulatory submissions,
AI model performance validation, ethics & consent compliance, and research milestones
for the study "Responsible Explainable AI Governance for EEG-Based Epilepsy Diagnosis."

Reads from real clinical.db tables:
  patients, analyses, consent_records, regulatory_submissions,
  validation_studies, model_comparison, assessments
Study design: retrospective (100 patients target) + prospective (10 patients target).
IRB: GGU + IEC (India). Jurisdictions: ICMR, DPDP Act 2023, HIPAA, TCPS2, ICH-GCP, Helsinki.
References: ICMR 2017, ICH E6(R2), WMA Helsinki 2013, GGU IRB SOP, FDA 21 CFR Part 11.
"""

import os
import sqlite3
from collections import Counter, defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# DBA study design constants
RETRO_TARGET = 100   # retrospective patient target
PROSP_TARGET = 10    # prospective patient target
TOTAL_TARGET = RETRO_TARGET + PROSP_TARGET

# IEC/IRB submission targets across study phases
IEC_IRB_DOC_TARGET = 173  # 173-document master list


def _conn():
    return sqlite3.connect(DB)


def overview(patient_id: str = None):
    """Overview KPIs: patient recruitment, model best perf, consent rate, regulatory status."""
    try:
        con = _conn()
        cur = con.cursor()

        # ── Patient recruitment ────────────────────────────────────────────────
        cur.execute("SELECT COUNT(*) FROM patients")
        n_patients = cur.fetchone()[0] or 0

        cur.execute("SELECT COUNT(*) FROM analyses")
        n_analyses = cur.fetchone()[0] or 0

        cur.execute("SELECT AVG(confidence), MAX(confidence) FROM analyses WHERE confidence IS NOT NULL")
        row = cur.fetchone()
        avg_conf = round((row[0] or 0) * 100, 1)
        max_conf = round((row[1] or 0) * 100, 1)

        # ── Consent compliance ─────────────────────────────────────────────────
        cur.execute("SELECT COUNT(*) FROM consent_records WHERE status='granted'")
        consents_granted = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(*) FROM consent_records")
        consents_total = cur.fetchone()[0] or 0
        consent_rate = round(consents_granted / consents_total * 100, 1) if consents_total else 0

        # Research-specific consent (IRB prerequisite)
        cur.execute("SELECT COUNT(*) FROM consent_records WHERE consent_type='research' AND status='granted'")
        research_consent = cur.fetchone()[0] or 0

        # ── Regulatory / IRB ───────────────────────────────────────────────────
        cur.execute("SELECT COUNT(*) FROM regulatory_submissions")
        n_submissions = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(*) FROM regulatory_submissions WHERE status='Approved'")
        n_approved = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(*) FROM regulatory_submissions WHERE status IN ('Submitted','Under Review')")
        n_under_review = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(*) FROM regulatory_submissions WHERE status='Pre-submission'")
        n_presubmission = cur.fetchone()[0] or 0

        # ── Validation studies ─────────────────────────────────────────────────
        cur.execute("SELECT COUNT(*) FROM validation_studies")
        n_val_studies = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(*) FROM validation_studies WHERE status='Passed'")
        n_val_passed = cur.fetchone()[0] or 0
        cur.execute("SELECT AVG(auc_roc), AVG(sensitivity), AVG(specificity) FROM validation_studies WHERE auc_roc IS NOT NULL")
        row = cur.fetchone()
        avg_auc = round((row[0] or 0), 3)
        avg_sens = round((row[1] or 0), 3)
        avg_spec = round((row[2] or 0), 3)

        # ── Model performance ──────────────────────────────────────────────────
        cur.execute("SELECT model_name, model_type, accuracy, auc_roc, task FROM model_comparison ORDER BY accuracy DESC LIMIT 1")
        best = cur.fetchone()
        best_model = {
            "name": best[0] if best else "—",
            "type": best[1] if best else "—",
            "accuracy": round((best[2] or 0) * 100, 1) if best else 0,
            "auc": round((best[3] or 0), 3) if best else 0,
            "task": best[4] if best else "—",
        }

        cur.execute("SELECT COUNT(*) FROM model_comparison")
        n_model_runs = cur.fetchone()[0] or 0

        # ── Assessments (data richness) ────────────────────────────────────────
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT patient_id) FROM assessments")
        row = cur.fetchone()
        n_assessments = row[0] or 0
        n_assessed_patients = row[1] or 0

        # ── Recruitment progress ───────────────────────────────────────────────
        recruitment_pct = round(n_patients / TOTAL_TARGET * 100, 1)

        con.close()

        return {
            "available": True,
            "study": {
                "title": "Responsible Explainable AI Governance for EEG-Based Epilepsy Diagnosis",
                "researcher": "Praveen Asthana",
                "program": "Doctor of Business Administration (DBA)",
                "institution": "Golden Gate University (GGU)",
                "location": "Canada",
                "design": f"Retrospective (n={RETRO_TARGET}) + Prospective (n={PROSP_TARGET})",
                "irb": "GGU IRB + IEC (India/ICMR)",
                "jurisdictions": ["ICMR 2017", "DPDP Act 2023", "HIPAA", "TCPS2", "ICH-GCP E6(R2)", "WMA Helsinki 2013"],
            },
            "recruitment": {
                "enrolled": n_patients,
                "target_total": TOTAL_TARGET,
                "target_retrospective": RETRO_TARGET,
                "target_prospective": PROSP_TARGET,
                "recruitment_pct": recruitment_pct,
                "n_analyses": n_analyses,
                "avg_confidence_pct": avg_conf,
                "max_confidence_pct": max_conf,
            },
            "ethics_consent": {
                "total_consent_records": consents_total,
                "granted": consents_granted,
                "consent_rate_pct": consent_rate,
                "research_consent_granted": research_consent,
                "research_consent_target": n_patients,
                "research_consent_pct": round(research_consent / n_patients * 100, 1) if n_patients else 0,
            },
            "regulatory": {
                "total_submissions": n_submissions,
                "approved": n_approved,
                "under_review": n_under_review,
                "pre_submission": n_presubmission,
                "approval_rate_pct": round(n_approved / n_submissions * 100, 1) if n_submissions else 0,
                "iec_irb_doc_target": IEC_IRB_DOC_TARGET,
                "iec_irb_doc_note": "173-document master list across 9 categories (IEC India + IRB GGU)",
            },
            "validation": {
                "total_studies": n_val_studies,
                "passed": n_val_passed,
                "pass_rate_pct": round(n_val_passed / n_val_studies * 100, 1) if n_val_studies else 0,
                "avg_auc": avg_auc,
                "avg_sensitivity": avg_sens,
                "avg_specificity": avg_spec,
            },
            "ai_performance": {
                "best_model": best_model,
                "total_training_runs": n_model_runs,
            },
            "data_richness": {
                "total_assessments": n_assessments,
                "assessed_patients": n_assessed_patients,
            },
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def breakdown(patient_id: str = None):
    """Detailed breakdown: consent by type, regulatory by pathway, validation by type, model comparison."""
    try:
        con = _conn()
        cur = con.cursor()

        # ── Consent by type ────────────────────────────────────────────────────
        cur.execute("""
            SELECT consent_type,
                   SUM(status='granted') as granted,
                   SUM(status='pending') as pending,
                   SUM(status='declined') as declined,
                   SUM(status='expired') as expired,
                   SUM(status='withdrawn') as withdrawn,
                   COUNT(*) as total
            FROM consent_records
            GROUP BY consent_type
            ORDER BY consent_type
        """)
        consent_by_type = []
        for row in cur.fetchall():
            total = row[6] or 1
            consent_by_type.append({
                "type": row[0],
                "granted": row[1] or 0,
                "pending": row[2] or 0,
                "declined": row[3] or 0,
                "expired": row[4] or 0,
                "withdrawn": row[5] or 0,
                "total": total,
                "granted_pct": round((row[1] or 0) / total * 100, 1),
            })

        # ── Regulatory by pathway ──────────────────────────────────────────────
        cur.execute("""
            SELECT pathway, status, phase, risk_class,
                   validation_score, submitted_date
            FROM regulatory_submissions
            ORDER BY submitted_date DESC
        """)
        reg_rows = cur.fetchall()
        regulatory_list = [
            {
                "pathway": r[0],
                "status": r[1],
                "phase": r[2],
                "risk_class": r[3],
                "validation_score": r[4],
                "submitted_date": r[5],
            }
            for r in reg_rows
        ]

        # Pathway distribution
        pathway_counts = Counter(r[0] for r in reg_rows)
        status_counts = Counter(r[1] for r in reg_rows)

        # ── Validation studies ─────────────────────────────────────────────────
        cur.execute("""
            SELECT study_type, status, site, auc_roc, sensitivity, specificity,
                   principal_investigator, sample_size, start_date, end_date, title
            FROM validation_studies
            ORDER BY auc_roc DESC
        """)
        val_rows = cur.fetchall()
        val_list = [
            {
                "study_type": r[0],
                "status": r[1],
                "site": r[2],
                "auc": round(r[3], 3) if r[3] is not None else None,
                "sensitivity": round(r[4], 3) if r[4] is not None else None,
                "specificity": round(r[5], 3) if r[5] is not None else None,
                "pi": r[6],
                "sample_size": r[7],
                "start_date": r[8],
                "end_date": r[9],
                "title": r[10],
            }
            for r in val_rows
        ]

        # Val by type summary
        val_type_summary = defaultdict(lambda: {"count": 0, "passed": 0, "auc_sum": 0, "auc_n": 0})
        for v in val_list:
            t = v["study_type"]
            val_type_summary[t]["count"] += 1
            if v["status"] in ("Passed", "Completed"):
                val_type_summary[t]["passed"] += 1
            if v["auc"] is not None:
                val_type_summary[t]["auc_sum"] += v["auc"]
                val_type_summary[t]["auc_n"] += 1

        val_by_type = []
        for t, d in sorted(val_type_summary.items()):
            avg_auc = round(d["auc_sum"] / d["auc_n"], 3) if d["auc_n"] else None
            val_by_type.append({"type": t, "count": d["count"], "passed": d["passed"], "avg_auc": avg_auc})

        # ── Model comparison ───────────────────────────────────────────────────
        cur.execute("""
            SELECT model_type, COUNT(*) as runs,
                   ROUND(MAX(accuracy), 4) as best_acc,
                   ROUND(MAX(auc_roc), 4) as best_auc,
                   ROUND(AVG(accuracy), 4) as avg_acc
            FROM model_comparison
            GROUP BY model_type
            ORDER BY best_acc DESC
        """)
        model_summary = [
            {"type": r[0], "runs": r[1], "best_acc": round((r[2] or 0)*100, 1),
             "best_auc": r[3], "avg_acc": round((r[4] or 0)*100, 1)}
            for r in cur.fetchall()
        ]

        # Monthly analysis trend (patient data collection)
        cur.execute("""
            SELECT SUBSTR(created_at, 1, 7) as month, COUNT(*) as n
            FROM analyses
            WHERE created_at IS NOT NULL
            GROUP BY month
            ORDER BY month
        """)
        monthly_analyses = [{"month": r[0], "count": r[1]} for r in cur.fetchall()]

        con.close()

        return {
            "available": True,
            "consent_by_type": consent_by_type,
            "regulatory": {
                "list": regulatory_list,
                "pathway_distribution": dict(pathway_counts),
                "status_distribution": dict(status_counts),
            },
            "validation": {
                "studies": val_list,
                "by_type": val_by_type,
            },
            "model_comparison": model_summary,
            "monthly_analyses": monthly_analyses,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def definitions():
    """Study design, IRB/IEC framework, glossary, and key references."""
    return {
        "available": True,
        "study_design": {
            "title": "Responsible Explainable AI Governance for EEG-Based Epilepsy Diagnosis",
            "type": "Mixed-methods (quantitative AI evaluation + qualitative governance)",
            "retrospective": {
                "target_n": 100,
                "source": "Existing clinical EEG/MRI/assessment records",
                "inclusion": "Diagnosed epilepsy, EEG available, age ≥2y",
                "exclusion": "Missing informed consent, incomplete EEG record",
            },
            "prospective": {
                "target_n": 10,
                "source": "New patient enrollment at study site",
                "duration": "6 months follow-up",
                "consent": "Prospective written informed consent required",
            },
            "primary_outcome": "AI model diagnostic accuracy (AUC-ROC) vs. neurologist standard",
            "secondary_outcomes": [
                "Explainability (SHAP/LIME fidelity)",
                "Fairness across demographic subgroups",
                "Human override rate and clinical appropriateness",
                "Governance compliance (HIPAA, ICMR, DPDP Act 2023)",
            ],
        },
        "irb_framework": {
            "irb_primary": "Golden Gate University (GGU) IRB",
            "irb_secondary": "Institutional Ethics Committee (IEC) — India (ICMR)",
            "jurisdictions": {
                "India": ["ICMR Ethical Guidelines 2017", "DPDP Act 2023", "ICH-GCP E6(R2)"],
                "Canada": ["TCPS2 (2022)", "PIPEDA", "ICH-GCP E6(R2)"],
                "USA": ["GGU IRB / 45 CFR 46", "HIPAA Privacy Rule", "FDA 21 CFR Part 11"],
                "International": ["WMA Helsinki Declaration 2013", "ISO 14155:2020"],
            },
            "document_master_list": 173,
            "categories": [
                "A. Study Protocol & Design",
                "B. IRB/IEC Submissions",
                "C. Informed Consent",
                "D. Data Management",
                "E. AI Governance",
                "F. Regulatory Compliance",
                "G. Safety Monitoring",
                "H. Statistical Analysis",
                "I. Final Report & Dissemination",
            ],
        },
        "glossary": [
            {"term": "DBA", "definition": "Doctor of Business Administration — terminal professional degree at GGU"},
            {"term": "IEC", "definition": "Institutional Ethics Committee — ethics body for Indian research sites (ICMR)"},
            {"term": "IRB", "definition": "Institutional Review Board — GGU's ethics oversight for human-subjects research"},
            {"term": "ICMR", "definition": "Indian Council of Medical Research — national body for biomedical research ethics in India"},
            {"term": "DPDP Act 2023", "definition": "Digital Personal Data Protection Act 2023 (India) — governs patient data handling"},
            {"term": "TCPS2", "definition": "Tri-Council Policy Statement 2 (Canada) — ethical conduct for research involving humans"},
            {"term": "PIPEDA", "definition": "Personal Information Protection and Electronic Documents Act — Canadian privacy law"},
            {"term": "ICH-GCP E6(R2)", "definition": "International Council for Harmonisation Good Clinical Practice — global standard"},
            {"term": "AUC-ROC", "definition": "Area Under the Receiver Operating Characteristic Curve — primary AI performance metric"},
            {"term": "SHAP", "definition": "SHapley Additive exPlanations — model explainability method (Lundberg & Lee, 2017)"},
            {"term": "SaMD", "definition": "Software as a Medical Device — FDA/CE classification for AI diagnostic tools"},
        ],
        "references": [
            "ICMR. Ethical Guidelines for Biomedical and Health Research Involving Human Participants. New Delhi: ICMR; 2017.",
            "ICH. E6(R2) Guideline for Good Clinical Practice. Step 4 version 2016.",
            "WMA. Declaration of Helsinki — Ethical Principles for Medical Research Involving Human Subjects. Fortaleza; 2013.",
            "Government of India. Digital Personal Data Protection Act (DPDP Act). New Delhi; 2023.",
            "Government of Canada. Tri-Council Policy Statement: Ethical Conduct for Research Involving Humans (TCPS2). 2022.",
            "Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. NeurIPS 2017.",
            "FDA. 21 CFR Part 11: Electronic Records; Electronic Signatures. 1997.",
            "IEC 62304: Medical device software — Software life cycle processes. 2006.",
            "ISO 14971: Medical devices — Application of risk management to medical devices. 2019.",
            "Asthana P. Responsible Explainable AI Governance for EEG-Based Epilepsy Diagnosis. DBA Dissertation, GGU; 2026 (In Progress).",
        ],
    }
