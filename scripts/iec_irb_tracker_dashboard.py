"""
IEC/IRB 173-Document Submission Tracker — EEG Epilepsy Platform
Tracks the 173-document master list (categories A–I) for phased IEC (India) and
IRB (GGU) submission. Covers multi-jurisdiction compliance and document completion.

Sources: clinical.db — regulatory_submissions, consent_records, validation_studies,
         patients, analyses.

Study: Responsible Explainable AI Governance for EEG-Based Epilepsy Diagnosis
Researcher: Praveen Asthana, DBA, Golden Gate University
"""

import sqlite3, os, statistics
from datetime import datetime

DB = os.path.join(os.path.dirname(__file__), "..", "data", "clinical.db")


def _conn():
    return sqlite3.connect(DB)


def _pct(n, d):
    return round(n / d * 100, 1) if d else 0


# ─── Document catalogue: 173 docs across categories A–I ──────────────────────
# Ground-truth status derived from real project milestones (§57.7 honest).
# Real = actual document completed; Partial = drafted/incomplete;
# Design = designed but not drafted; Pending = not yet started.

def _doc_catalogue():
    """Return the 173-document master list with category, phase, status, jurisdiction."""
    return [
        # ── Category A: Study Design & Protocol (25 docs) ──
        {"cat": "A", "id": "A01", "name": "Master Protocol v2.1",          "phase_iec": 1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","Canada","International"]},
        {"cat": "A", "id": "A02", "name": "Study Synopsis (2-page)",        "phase_iec": 1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "A", "id": "A03", "name": "Investigator's Brochure",        "phase_iec": 1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","International"]},
        {"cat": "A", "id": "A04", "name": "Statistical Analysis Plan v1.2", "phase_iec": 1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","Canada"]},
        {"cat": "A", "id": "A05", "name": "Sample Size & Power Calculation","phase_iec": 1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "A", "id": "A06", "name": "Inclusion/Exclusion Criteria",   "phase_iec": 1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","International"]},
        {"cat": "A", "id": "A07", "name": "Recruitment & Retention Plan",   "phase_iec": 1, "phase_irb": 1, "status": "partial", "jurisdiction": ["India","USA"]},
        {"cat": "A", "id": "A08", "name": "Data Management Plan",           "phase_iec": 1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","Canada"]},
        {"cat": "A", "id": "A09", "name": "Risk Management Plan (ISO 14971)","phase_iec": 2,"phase_irb": 2, "status": "real",    "jurisdiction": ["India","USA","International"]},
        {"cat": "A", "id": "A10", "name": "Prospective Sub-Protocol",       "phase_iec": 2, "phase_irb": 1, "status": "partial", "jurisdiction": ["India","USA"]},
        {"cat": "A", "id": "A11", "name": "Retrospective EEG Sub-Protocol", "phase_iec": 1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "A", "id": "A12", "name": "Cross-sectional Design Rationale","phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "A", "id": "A13", "name": "TRIPOD-AI Reporting Checklist",  "phase_iec": 2, "phase_irb": 2, "status": "real",    "jurisdiction": ["USA","International"]},
        {"cat": "A", "id": "A14", "name": "CONSORT-AI Checklist",           "phase_iec": 2, "phase_irb": 2, "status": "partial", "jurisdiction": ["USA","International"]},
        {"cat": "A", "id": "A15", "name": "PROBAST-AI Risk-of-Bias",        "phase_iec": 2, "phase_irb": 2, "status": "partial", "jurisdiction": ["International"]},
        {"cat": "A", "id": "A16", "name": "External Validation Plan (7 sites)","phase_iec":2,"phase_irb":2,"status": "real",    "jurisdiction": ["USA","Canada","International"]},
        {"cat": "A", "id": "A17", "name": "Clinical Validation Protocol",   "phase_iec": 2, "phase_irb": 3, "status": "partial", "jurisdiction": ["India","USA"]},
        {"cat": "A", "id": "A18", "name": "Multi-site Coordination Plan",   "phase_iec": 3, "phase_irb": 2, "status": "design",  "jurisdiction": ["India","USA","Canada"]},
        {"cat": "A", "id": "A19", "name": "Endpoint Definition Document",   "phase_iec": 1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "A", "id": "A20", "name": "Blinding & Randomisation Plan",  "phase_iec": 1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "A", "id": "A21", "name": "Protocol Deviation Procedures",  "phase_iec": 2, "phase_irb": 1, "status": "design",  "jurisdiction": ["India","USA","International"]},
        {"cat": "A", "id": "A22", "name": "Study Closure Procedures",       "phase_iec": 3, "phase_irb": 3, "status": "pending", "jurisdiction": ["India","USA"]},
        {"cat": "A", "id": "A23", "name": "Publication & IP Policy",        "phase_iec": 3, "phase_irb": 3, "status": "design",  "jurisdiction": ["India","USA","Canada"]},
        {"cat": "A", "id": "A24", "name": "Data Sharing Agreement Template","phase_iec": 2, "phase_irb": 2, "status": "design",  "jurisdiction": ["India","USA","Canada","International"]},
        {"cat": "A", "id": "A25", "name": "Protocol Amendment Procedures",  "phase_iec": 2, "phase_irb": 2, "status": "design",  "jurisdiction": ["India","USA","International"]},

        # ── Category B: Prospective Interview/Survey Package (18 docs) ──
        {"cat": "B", "id": "B01", "name": "Patient Interview Guide v1.0",   "phase_iec": 2, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "B", "id": "B02", "name": "Epilepsy Burden Survey (EBS-7)", "phase_iec": 2, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","Canada"]},
        {"cat": "B", "id": "B03", "name": "QOLIE-31-P Questionnaire",       "phase_iec": 2, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "B", "id": "B04", "name": "NDDI-E Depression Screen",       "phase_iec": 2, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","Canada"]},
        {"cat": "B", "id": "B05", "name": "Seizure Diary (SZD-Pro)",        "phase_iec": 2, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "B", "id": "B06", "name": "AED Adherence Questionnaire",    "phase_iec": 2, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "B", "id": "B07", "name": "Caregiver Burden Index",         "phase_iec": 2, "phase_irb": 1, "status": "partial", "jurisdiction": ["India","USA"]},
        {"cat": "B", "id": "B08", "name": "SUDEP Awareness Questionnaire",  "phase_iec": 2, "phase_irb": 2, "status": "design",  "jurisdiction": ["India","USA","Canada"]},
        {"cat": "B", "id": "B09", "name": "Trigger Identification Survey",  "phase_iec": 2, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "B", "id": "B10", "name": "Sleep & Lifestyle Assessment",   "phase_iec": 2, "phase_irb": 1, "status": "partial", "jurisdiction": ["India","USA"]},
        {"cat": "B", "id": "B11", "name": "Cognitive Function Screen",      "phase_iec": 2, "phase_irb": 2, "status": "partial", "jurisdiction": ["India","USA","Canada"]},
        {"cat": "B", "id": "B12", "name": "Social Support Assessment",      "phase_iec": 2, "phase_irb": 2, "status": "design",  "jurisdiction": ["India","USA"]},
        {"cat": "B", "id": "B13", "name": "AI Perception & Trust Survey",   "phase_iec": 2, "phase_irb": 2, "status": "design",  "jurisdiction": ["India","USA","Canada"]},
        {"cat": "B", "id": "B14", "name": "Digital Literacy Screen",        "phase_iec": 3, "phase_irb": 2, "status": "pending", "jurisdiction": ["India","USA"]},
        {"cat": "B", "id": "B15", "name": "Wearable Acceptability Survey",  "phase_iec": 3, "phase_irb": 2, "status": "pending", "jurisdiction": ["India","USA"]},
        {"cat": "B", "id": "B16", "name": "Telemedicine Readiness Survey",  "phase_iec": 3, "phase_irb": 2, "status": "pending", "jurisdiction": ["India","USA","Canada"]},
        {"cat": "B", "id": "B17", "name": "Survey Pilot Test Report",       "phase_iec": 2, "phase_irb": 1, "status": "design",  "jurisdiction": ["India","USA"]},
        {"cat": "B", "id": "B18", "name": "Interview Pilot Test Report",    "phase_iec": 2, "phase_irb": 1, "status": "design",  "jurisdiction": ["India","USA"]},

        # ── Category C: Retrospective EEG & Clinical Records Package (20 docs) ──
        {"cat": "C", "id": "C01", "name": "EEG Data Access Request",        "phase_iec": 1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "C", "id": "C02", "name": "De-identification Protocol (Safe Harbor)","phase_iec":1,"phase_irb":1,"status":"real","jurisdiction": ["India","USA","Canada","International"]},
        {"cat": "C", "id": "C03", "name": "EDF/BDF Processing Pipeline",    "phase_iec": 1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "C", "id": "C04", "name": "Feature Extraction Documentation","phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "C", "id": "C05", "name": "Model Development Report",       "phase_iec": 2, "phase_irb": 2, "status": "real",    "jurisdiction": ["India","USA","International"]},
        {"cat": "C", "id": "C06", "name": "XAI / SHAP Analysis Report",     "phase_iec": 2, "phase_irb": 2, "status": "real",    "jurisdiction": ["India","USA","International"]},
        {"cat": "C", "id": "C07", "name": "GroupKFold CV Report (§no-leakage)","phase_iec":2,"phase_irb":2,"status":"real",     "jurisdiction": ["USA","International"]},
        {"cat": "C", "id": "C08", "name": "External Validation Report",     "phase_iec": 2, "phase_irb": 2, "status": "real",    "jurisdiction": ["India","USA","Canada","International"]},
        {"cat": "C", "id": "C09", "name": "Fairness & Bias Assessment",     "phase_iec": 2, "phase_irb": 2, "status": "real",    "jurisdiction": ["India","USA","International"]},
        {"cat": "C", "id": "C10", "name": "Calibration Report (Hosmer-Lemeshow)","phase_iec":2,"phase_irb":2,"status":"real",   "jurisdiction": ["USA","International"]},
        {"cat": "C", "id": "C11", "name": "Decision Curve Analysis Report", "phase_iec": 2, "phase_irb": 2, "status": "partial", "jurisdiction": ["USA","International"]},
        {"cat": "C", "id": "C12", "name": "Drift Monitoring Report",        "phase_iec": 3, "phase_irb": 3, "status": "partial", "jurisdiction": ["India","USA"]},
        {"cat": "C", "id": "C13", "name": "Benchmark Dataset Comparison",   "phase_iec": 2, "phase_irb": 2, "status": "real",    "jurisdiction": ["USA","International"]},
        {"cat": "C", "id": "C14", "name": "HITL Override Log Analysis",     "phase_iec": 2, "phase_irb": 2, "status": "real",    "jurisdiction": ["India","USA","International"]},
        {"cat": "C", "id": "C15", "name": "EEG Artifact Annotation Protocol","phase_iec":1, "phase_irb": 1, "status": "partial", "jurisdiction": ["India","USA"]},
        {"cat": "C", "id": "C16", "name": "Signal Quality Assessment Report","phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "C", "id": "C17", "name": "Class Imbalance Strategy Report","phase_iec": 2, "phase_irb": 2, "status": "real",    "jurisdiction": ["USA","International"]},
        {"cat": "C", "id": "C18", "name": "Ictal/Interictal Classification Report","phase_iec":2,"phase_irb":2,"status":"partial","jurisdiction":["India","USA"]},
        {"cat": "C", "id": "C19", "name": "Deep Learning Architecture Report","phase_iec":2,"phase_irb":2,"status":"real",      "jurisdiction": ["USA","International"]},
        {"cat": "C", "id": "C20", "name": "Post-Market Surveillance Plan",  "phase_iec": 3, "phase_irb": 3, "status": "design",  "jurisdiction": ["India","USA","International"]},

        # ── Category D: Consent Forms (20 docs) ──
        {"cat": "D", "id": "D01", "name": "Main ICF (Prospective) — English","phase_iec":2,"phase_irb":1,"status":"real",       "jurisdiction": ["India","USA","Canada"]},
        {"cat": "D", "id": "D02", "name": "Main ICF — Hindi",                "phase_iec":2, "phase_irb": 1, "status": "real",    "jurisdiction": ["India"]},
        {"cat": "D", "id": "D03", "name": "Retrospective Waiver of Consent", "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "D", "id": "D04", "name": "AI Use Disclosure Form",          "phase_iec":2, "phase_irb": 2, "status": "real",    "jurisdiction": ["India","USA","Canada"]},
        {"cat": "D", "id": "D05", "name": "Data Sharing Consent",            "phase_iec":2, "phase_irb": 2, "status": "real",    "jurisdiction": ["India","USA","Canada","International"]},
        {"cat": "D", "id": "D06", "name": "Genetic Testing Consent",         "phase_iec":2, "phase_irb": 2, "status": "partial", "jurisdiction": ["India","USA"]},
        {"cat": "D", "id": "D07", "name": "Video-EEG Recording Consent",     "phase_iec":2, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "D", "id": "D08", "name": "Wearable Device Consent",         "phase_iec":3, "phase_irb": 2, "status": "design",  "jurisdiction": ["India","USA"]},
        {"cat": "D", "id": "D09", "name": "Assent Form (Minors <12)",        "phase_iec":2, "phase_irb": 1, "status": "design",  "jurisdiction": ["India","USA","Canada"]},
        {"cat": "D", "id": "D10", "name": "LAR Consent (Cognitively Impaired)","phase_iec":2,"phase_irb":1,"status":"design",   "jurisdiction": ["India","USA"]},
        {"cat": "D", "id": "D11", "name": "Re-consent Procedure",            "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["India","USA"]},
        {"cat": "D", "id": "D12", "name": "Caregiver Consent Form",          "phase_iec":2, "phase_irb": 1, "status": "partial", "jurisdiction": ["India","USA"]},
        {"cat": "D", "id": "D13", "name": "Multi-site Master Consent",       "phase_iec":3, "phase_irb": 2, "status": "pending", "jurisdiction": ["India","USA","Canada"]},
        {"cat": "D", "id": "D14", "name": "Remote/Telehealth Consent",       "phase_iec":3, "phase_irb": 2, "status": "pending", "jurisdiction": ["India","USA","Canada"]},
        {"cat": "D", "id": "D15", "name": "Biobanking Consent",              "phase_iec":3, "phase_irb": 2, "status": "pending", "jurisdiction": ["India","USA"]},
        {"cat": "D", "id": "D16", "name": "Consent Monitoring SOP",         "phase_iec":2, "phase_irb": 1, "status": "design",  "jurisdiction": ["India","USA","International"]},
        {"cat": "D", "id": "D17", "name": "Withdrawal Procedure Document",   "phase_iec":2, "phase_irb": 1, "status": "partial", "jurisdiction": ["India","USA","International"]},
        {"cat": "D", "id": "D18", "name": "Consent Audit Checklist",         "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["India","USA"]},
        {"cat": "D", "id": "D19", "name": "DPDP Act 2023 Consent Addendum",  "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["India"]},
        {"cat": "D", "id": "D20", "name": "PIPEDA / TCPS2 Consent Addendum", "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["Canada"]},

        # ── Category E: Ethics Applications (20 docs) ──
        {"cat": "E", "id": "E01", "name": "IEC Application Form (ICMR 20-section)","phase_iec":1,"phase_irb":1,"status":"real","jurisdiction":["India"]},
        {"cat": "E", "id": "E02", "name": "IRB Application Form (GGU)",      "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["USA"]},
        {"cat": "E", "id": "E03", "name": "IEC Covering Letter — Phase 1",   "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India"]},
        {"cat": "E", "id": "E04", "name": "IRB Covering Letter — Phase 1",   "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["USA"]},
        {"cat": "E", "id": "E05", "name": "Principal Investigator CV",        "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","Canada","International"]},
        {"cat": "E", "id": "E06", "name": "GCP Training Certificate",         "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","International"]},
        {"cat": "E", "id": "E07", "name": "Conflict of Interest Declaration", "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","Canada"]},
        {"cat": "E", "id": "E08", "name": "Funding Disclosure Statement",     "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","Canada"]},
        {"cat": "E", "id": "E09", "name": "Site Feasibility Assessment",      "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "E", "id": "E10", "name": "IEC Amendment — Phase 2",         "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["India"]},
        {"cat": "E", "id": "E11", "name": "IRB Amendment — Phase 2",         "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["USA"]},
        {"cat": "E", "id": "E12", "name": "AI Governance Addendum (IEC)",     "phase_iec":2, "phase_irb": 2, "status": "partial", "jurisdiction": ["India"]},
        {"cat": "E", "id": "E13", "name": "AI Governance Addendum (IRB)",     "phase_iec":2, "phase_irb": 2, "status": "partial", "jurisdiction": ["USA"]},
        {"cat": "E", "id": "E14", "name": "HMSC Applicability Determination", "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["India"]},
        {"cat": "E", "id": "E15", "name": "IEC Progress Report — Phase 1",   "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["India"]},
        {"cat": "E", "id": "E16", "name": "IRB Progress Report — Phase 1",   "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["USA"]},
        {"cat": "E", "id": "E17", "name": "Adverse Event Reporting SOP",      "phase_iec":2, "phase_irb": 1, "status": "design",  "jurisdiction": ["India","USA","International"]},
        {"cat": "E", "id": "E18", "name": "Annual Renewal Application",       "phase_iec":3, "phase_irb": 3, "status": "pending", "jurisdiction": ["India","USA"]},
        {"cat": "E", "id": "E19", "name": "Study Closure Report",             "phase_iec":3, "phase_irb": 3, "status": "pending", "jurisdiction": ["India","USA"]},
        {"cat": "E", "id": "E20", "name": "Final Ethics Approval Certificate","phase_iec":3, "phase_irb": 3, "status": "pending", "jurisdiction": ["India","USA"]},

        # ── Category F: Data Management (15 docs) ──
        {"cat": "F", "id": "F01", "name": "DPIA (Data Protection Impact Assessment)","phase_iec":1,"phase_irb":1,"status":"real","jurisdiction":["India","USA","Canada","International"]},
        {"cat": "F", "id": "F02", "name": "Data Governance Framework",        "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","Canada"]},
        {"cat": "F", "id": "F03", "name": "Data Retention & Destruction Policy","phase_iec":1,"phase_irb":1,"status":"real",     "jurisdiction": ["India","USA","Canada","International"]},
        {"cat": "F", "id": "F04", "name": "Database Specification (ERD)",     "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "F", "id": "F05", "name": "Audit Trail SOP",                  "phase_iec":2, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","International"]},
        {"cat": "F", "id": "F06", "name": "Access Control Matrix",            "phase_iec":2, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","Canada"]},
        {"cat": "F", "id": "F07", "name": "Encryption & Security Protocol",   "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","Canada","International"]},
        {"cat": "F", "id": "F08", "name": "Cloud Storage Agreement",          "phase_iec":2, "phase_irb": 1, "status": "partial", "jurisdiction": ["India","USA","Canada"]},
        {"cat": "F", "id": "F09", "name": "FAIR Data Principles Compliance",  "phase_iec":2, "phase_irb": 2, "status": "real",    "jurisdiction": ["USA","International"]},
        {"cat": "F", "id": "F10", "name": "Data Dictionary v1.0",             "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "F", "id": "F11", "name": "CRF (Case Report Form) Templates", "phase_iec":2, "phase_irb": 1, "status": "partial", "jurisdiction": ["India","USA","International"]},
        {"cat": "F", "id": "F12", "name": "Source Data Verification Plan",    "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["India","USA"]},
        {"cat": "F", "id": "F13", "name": "Federated Learning Data Protocol", "phase_iec":3, "phase_irb": 2, "status": "design",  "jurisdiction": ["India","USA","Canada","International"]},
        {"cat": "F", "id": "F14", "name": "Data Transfer Agreement (DTA)",    "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["India","USA","Canada","International"]},
        {"cat": "F", "id": "F15", "name": "Backup & Disaster Recovery Plan",  "phase_iec":2, "phase_irb": 1, "status": "partial", "jurisdiction": ["India","USA"]},

        # ── Category G: Analysis Plans & Quality (15 docs) ──
        {"cat": "G", "id": "G01", "name": "Primary Endpoint Analysis Plan",   "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "G", "id": "G02", "name": "Secondary Endpoint Analysis Plan", "phase_iec":2, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "G", "id": "G03", "name": "Fairness & Equity Analysis Plan",  "phase_iec":2, "phase_irb": 2, "status": "real",    "jurisdiction": ["India","USA","International"]},
        {"cat": "G", "id": "G04", "name": "Sensitivity Analysis Plan",        "phase_iec":2, "phase_irb": 2, "status": "partial", "jurisdiction": ["USA","International"]},
        {"cat": "G", "id": "G05", "name": "Missing Data Handling Plan",       "phase_iec":2, "phase_irb": 2, "status": "partial", "jurisdiction": ["USA"]},
        {"cat": "G", "id": "G06", "name": "Subgroup Analysis Plan",           "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["USA","International"]},
        {"cat": "G", "id": "G07", "name": "Bootstrap CI Protocol (n=1000)",   "phase_iec":2, "phase_irb": 2, "status": "real",    "jurisdiction": ["USA","International"]},
        {"cat": "G", "id": "G08", "name": "DeLong AUC Test Protocol",         "phase_iec":2, "phase_irb": 2, "status": "real",    "jurisdiction": ["USA","International"]},
        {"cat": "G", "id": "G09", "name": "McNemar's Test Protocol",          "phase_iec":2, "phase_irb": 2, "status": "real",    "jurisdiction": ["USA","International"]},
        {"cat": "G", "id": "G10", "name": "KS Drift Detection Protocol",      "phase_iec":3, "phase_irb": 3, "status": "partial", "jurisdiction": ["USA","International"]},
        {"cat": "G", "id": "G11", "name": "G*Power Sample Size Update",       "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","Canada"]},
        {"cat": "G", "id": "G12", "name": "Interim Analysis Plan",            "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["India","USA"]},
        {"cat": "G", "id": "G13", "name": "Quality Assurance Plan (QAP)",     "phase_iec":2, "phase_irb": 1, "status": "partial", "jurisdiction": ["India","USA","International"]},
        {"cat": "G", "id": "G14", "name": "Quality Management System (QMS)",  "phase_iec":3, "phase_irb": 3, "status": "design",  "jurisdiction": ["India","USA","International"]},
        {"cat": "G", "id": "G15", "name": "Reproducibility Checklist (seed=42)","phase_iec":2,"phase_irb":2,"status":"real",     "jurisdiction": ["USA","International"]},

        # ── Category H: Safety & Monitoring (20 docs) ──
        {"cat": "H", "id": "H01", "name": "Safety Monitoring Plan",           "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","International"]},
        {"cat": "H", "id": "H02", "name": "DSMB Charter",                     "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["India","USA"]},
        {"cat": "H", "id": "H03", "name": "Adverse Event Classification SOP", "phase_iec":2, "phase_irb": 1, "status": "design",  "jurisdiction": ["India","USA","International"]},
        {"cat": "H", "id": "H04", "name": "SAE Reporting Form (24-hr)",       "phase_iec":2, "phase_irb": 1, "status": "design",  "jurisdiction": ["India","USA","International"]},
        {"cat": "H", "id": "H05", "name": "AI Model Failure Mode Analysis",   "phase_iec":2, "phase_irb": 2, "status": "real",    "jurisdiction": ["India","USA","International"]},
        {"cat": "H", "id": "H06", "name": "HITL Override Protocol",           "phase_iec":2, "phase_irb": 2, "status": "real",    "jurisdiction": ["India","USA","International"]},
        {"cat": "H", "id": "H07", "name": "AI Incident Response Plan",        "phase_iec":2, "phase_irb": 2, "status": "real",    "jurisdiction": ["India","USA","International"]},
        {"cat": "H", "id": "H08", "name": "Stopping Rules Document",          "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["India","USA"]},
        {"cat": "H", "id": "H09", "name": "Emergency Alert Protocol",         "phase_iec":2, "phase_irb": 2, "status": "real",    "jurisdiction": ["India","USA"]},
        {"cat": "H", "id": "H10", "name": "Patient Safety Monitoring SOP",    "phase_iec":2, "phase_irb": 1, "status": "partial", "jurisdiction": ["India","USA","International"]},
        {"cat": "H", "id": "H11", "name": "Pharmacovigilance Plan",           "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["India","USA","International"]},
        {"cat": "H", "id": "H12", "name": "Risk-Benefit Assessment",          "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","International"]},
        {"cat": "H", "id": "H13", "name": "Data Safety Monitoring Report",    "phase_iec":3, "phase_irb": 3, "status": "pending", "jurisdiction": ["India","USA"]},
        {"cat": "H", "id": "H14", "name": "Model Drift Alert Protocol",       "phase_iec":3, "phase_irb": 3, "status": "partial", "jurisdiction": ["India","USA","International"]},
        {"cat": "H", "id": "H15", "name": "Bias Monitoring Plan",             "phase_iec":2, "phase_irb": 2, "status": "real",    "jurisdiction": ["India","USA","International"]},
        {"cat": "H", "id": "H16", "name": "Closed-Loop Safety Checklist",     "phase_iec":3, "phase_irb": 3, "status": "design",  "jurisdiction": ["India","USA"]},
        {"cat": "H", "id": "H17", "name": "Vulnerable Population Safeguards", "phase_iec":2, "phase_irb": 1, "status": "design",  "jurisdiction": ["India","USA","Canada","International"]},
        {"cat": "H", "id": "H18", "name": "ICH-GCP E6(R3) Compliance Report","phase_iec":2, "phase_irb": 2, "status": "partial", "jurisdiction": ["India","USA","Canada","International"]},
        {"cat": "H", "id": "H19", "name": "Post-Study Safety Follow-up Plan", "phase_iec":3, "phase_irb": 3, "status": "pending", "jurisdiction": ["India","USA"]},
        {"cat": "H", "id": "H20", "name": "Device Safety Assessment (Wearable)","phase_iec":3,"phase_irb":3,"status":"pending",  "jurisdiction": ["India","USA"]},

        # ── Category I: International Regulatory Binder (20 docs) ──
        {"cat": "I", "id": "I01", "name": "Jurisdiction Matrix (India/USA/Canada/Intl)","phase_iec":1,"phase_irb":1,"status":"real","jurisdiction":["India","USA","Canada","International"]},
        {"cat": "I", "id": "I02", "name": "Foreign Researcher Declaration",   "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["India","USA","Canada"]},
        {"cat": "I", "id": "I03", "name": "ICMR AI Ethics Guidelines Compliance","phase_iec":1,"phase_irb":1,"status":"real",    "jurisdiction": ["India"]},
        {"cat": "I", "id": "I04", "name": "DPDP Act 2023 Compliance Plan",    "phase_iec":2, "phase_irb": 2, "status": "real",    "jurisdiction": ["India"]},
        {"cat": "I", "id": "I05", "name": "HIPAA Compliance Assessment",      "phase_iec":1, "phase_irb": 1, "status": "real",    "jurisdiction": ["USA"]},
        {"cat": "I", "id": "I06", "name": "PIPEDA Compliance Assessment",     "phase_iec":2, "phase_irb": 2, "status": "partial", "jurisdiction": ["Canada"]},
        {"cat": "I", "id": "I07", "name": "TCPS 2 (2022) Compliance Report",  "phase_iec":2, "phase_irb": 2, "status": "partial", "jurisdiction": ["Canada"]},
        {"cat": "I", "id": "I08", "name": "Declaration of Helsinki Attestation","phase_iec":1,"phase_irb":1,"status":"real",     "jurisdiction": ["India","USA","Canada","International"]},
        {"cat": "I", "id": "I09", "name": "CIOMS Guidelines Compliance",      "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["International"]},
        {"cat": "I", "id": "I10", "name": "WHO AI Ethics Principles Mapping", "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["International"]},
        {"cat": "I", "id": "I11", "name": "NIST AI RMF Alignment Report",     "phase_iec":2, "phase_irb": 2, "status": "real",    "jurisdiction": ["USA","International"]},
        {"cat": "I", "id": "I12", "name": "EU AI Act Applicability Analysis", "phase_iec":2, "phase_irb": 2, "status": "partial", "jurisdiction": ["International"]},
        {"cat": "I", "id": "I13", "name": "ISO/IEC 23894 Compliance",         "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["International"]},
        {"cat": "I", "id": "I14", "name": "Cross-Border Data Transfer Agreement","phase_iec":2,"phase_irb":2,"status":"design",  "jurisdiction": ["India","USA","Canada","International"]},
        {"cat": "I", "id": "I15", "name": "MOU with External Sites",          "phase_iec":2, "phase_irb": 2, "status": "design",  "jurisdiction": ["India","USA","Canada"]},
        {"cat": "I", "id": "I16", "name": "Common Rule (45 CFR 46) Compliance","phase_iec":2,"phase_irb":1,"status":"real",      "jurisdiction": ["USA"]},
        {"cat": "I", "id": "I17", "name": "Provincial Privacy Law Compliance (Canada)","phase_iec":2,"phase_irb":2,"status":"design","jurisdiction":["Canada"]},
        {"cat": "I", "id": "I18", "name": "OECD AI Principles Assessment",    "phase_iec":3, "phase_irb": 3, "status": "design",  "jurisdiction": ["International"]},
        {"cat": "I", "id": "I19", "name": "International Regulatory Binder Index","phase_iec":2,"phase_irb":2,"status":"partial","jurisdiction":["India","USA","Canada","International"]},
        {"cat": "I", "id": "I20", "name": "Global Study Coordination Plan",   "phase_iec":3, "phase_irb": 3, "status": "design",  "jurisdiction": ["India","USA","Canada","International"]},
    ]


def _status_weight(s):
    return {"real": 1.0, "partial": 0.5, "design": 0.25, "pending": 0.0}[s]


def _iec_phase_status(doc):
    """Map document status to IEC submission phase status."""
    s = doc["status"]
    ph = doc["phase_iec"]
    if "India" not in doc["jurisdiction"]:
        return "not_applicable"
    if ph == 1 and s in ("real", "partial"):
        return "submitted"
    if ph == 1 and s in ("design", "pending"):
        return "pending"
    if ph == 2 and s in ("real", "partial"):
        return "in_progress"
    return "pending"


def _irb_phase_status(doc):
    """Map document status to IRB submission phase status."""
    s = doc["status"]
    if s == "real":
        return "submitted"
    if s == "partial":
        return "in_progress"
    if s == "design":
        return "drafted"
    return "pending"


# ─── overview ────────────────────────────────────────────────────────────────

def overview():
    conn = _conn()
    c = conn.cursor()

    # real DB anchors
    n_patients    = c.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    n_consent     = c.execute("SELECT COUNT(*) FROM consent_records").fetchone()[0]
    n_reg_subs    = c.execute("SELECT COUNT(*) FROM regulatory_submissions").fetchone()[0]
    n_approved    = c.execute("SELECT COUNT(*) FROM regulatory_submissions WHERE status='Approved'").fetchone()[0]
    n_analyses    = c.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    conn.close()

    docs = _doc_catalogue()
    total = len(docs)

    # completion counts
    real    = sum(1 for d in docs if d["status"] == "real")
    partial = sum(1 for d in docs if d["status"] == "partial")
    design  = sum(1 for d in docs if d["status"] == "design")
    pending = sum(1 for d in docs if d["status"] == "pending")

    # weighted completion score
    weighted = sum(_status_weight(d["status"]) for d in docs)
    completion_pct = round(weighted / total * 100, 1)

    # IEC readiness (Phase 1 only)
    iec_ph1 = [d for d in docs if d["phase_iec"] == 1 and "India" in d["jurisdiction"]]
    iec_ph1_done = sum(1 for d in iec_ph1 if d["status"] in ("real", "partial"))
    iec_ph1_pct = _pct(iec_ph1_done, len(iec_ph1))

    # IRB readiness (Phase 1 only)
    irb_ph1 = [d for d in docs if d["phase_irb"] == 1]
    irb_ph1_done = sum(1 for d in irb_ph1 if d["status"] in ("real", "partial"))
    irb_ph1_pct = _pct(irb_ph1_done, len(irb_ph1))

    # category breakdown
    cat_summary = {}
    for d in docs:
        cat = d["cat"]
        if cat not in cat_summary:
            cat_summary[cat] = {"total": 0, "real": 0, "partial": 0, "design": 0, "pending": 0}
        cat_summary[cat]["total"] += 1
        cat_summary[cat][d["status"]] += 1

    categories = []
    cat_labels = {
        "A": "Study Design & Protocol",
        "B": "Prospective Survey Package",
        "C": "Retrospective EEG Package",
        "D": "Consent Forms",
        "E": "Ethics Applications",
        "F": "Data Management",
        "G": "Analysis Plans & Quality",
        "H": "Safety & Monitoring",
        "I": "International Regulatory Binder",
    }
    for cat, st in cat_summary.items():
        pct = _pct(st["real"] + st["partial"] * 0.5, st["total"])
        categories.append({
            "category": cat,
            "label": cat_labels.get(cat, cat),
            "total": st["total"],
            "real": st["real"],
            "partial": st["partial"],
            "design": st["design"],
            "pending": st["pending"],
            "completion_pct": pct,
        })

    # jurisdiction coverage
    jurisdictions = ["India", "USA", "Canada", "International"]
    jur_coverage = {}
    for j in jurisdictions:
        j_docs = [d for d in docs if j in d["jurisdiction"]]
        j_done = sum(1 for d in j_docs if d["status"] in ("real", "partial"))
        jur_coverage[j] = {"total": len(j_docs), "done": j_done, "pct": _pct(j_done, len(j_docs))}

    return {
        "available": True,
        "study": {
            "title": "Responsible Explainable AI Governance for EEG-Based Epilepsy Diagnosis",
            "researcher": "Praveen Asthana",
            "degree": "Doctor of Business Administration (DBA)",
            "institution": "Golden Gate University",
            "study_type": "Prospective Cross-Sectional + Retrospective EEG",
            "updated": datetime.today().strftime("%Y-%m-%d"),
        },
        "kpis": {
            "total_documents": total,
            "real_complete": real,
            "partial": partial,
            "design": design,
            "pending": pending,
            "completion_pct": completion_pct,
            "iec_phase1_readiness_pct": iec_ph1_pct,
            "irb_phase1_readiness_pct": irb_ph1_pct,
            "patients_enrolled": n_patients,
            "consent_records": n_consent,
            "regulatory_submissions": n_reg_subs,
            "regulatory_approved": n_approved,
            "analyses_completed": n_analyses,
        },
        "category_summary": categories,
        "jurisdiction_coverage": [
            {"jurisdiction": j, **jur_coverage[j]} for j in jurisdictions
        ],
        "phases": {
            "iec": {
                "phase1": {"name": "Retrospective EEG Classification", "target": "Submit first (easiest approval)", "docs": len(iec_ph1), "done": iec_ph1_done, "pct": iec_ph1_pct},
                "phase2": {"name": "XAI + Clinical Review", "target": "Amendment after Phase 1 approval"},
                "phase3": {"name": "Governance + Remote/Observation", "target": "Separate approval"},
            },
            "irb": {
                "phase1": {"name": "Core Approval", "target": "Protocol, SAP, consent forms", "docs": len(irb_ph1), "done": irb_ph1_done, "pct": irb_ph1_pct},
                "phase2": {"name": "AI Governance", "target": "XAI, fairness, validation, monitoring"},
                "phase3": {"name": "Clinical Validation", "target": "Implementation, publication, QMS, closure"},
            },
        },
        "standards": [
            {"name": "ICMR 2017 & AI 2023", "jurisdiction": "India", "status": "partial"},
            {"name": "DPDP Act 2023",         "jurisdiction": "India", "status": "real"},
            {"name": "HIPAA (45 CFR 164)",    "jurisdiction": "USA",   "status": "real"},
            {"name": "Common Rule (45 CFR 46)","jurisdiction": "USA",  "status": "real"},
            {"name": "GGU IRB",               "jurisdiction": "USA",   "status": "real"},
            {"name": "PIPEDA",                "jurisdiction": "Canada","status": "partial"},
            {"name": "TCPS 2 (2022)",         "jurisdiction": "Canada","status": "partial"},
            {"name": "ICH-GCP E6(R3)",        "jurisdiction": "Intl",  "status": "partial"},
            {"name": "Declaration of Helsinki","jurisdiction": "Intl",  "status": "real"},
            {"name": "TRIPOD-AI",             "jurisdiction": "Intl",  "status": "real"},
            {"name": "NIST AI RMF",           "jurisdiction": "Intl",  "status": "real"},
            {"name": "ISO/IEC 23894",         "jurisdiction": "Intl",  "status": "design"},
        ],
    }


# ─── breakdown ────────────────────────────────────────────────────────────────

def breakdown():
    conn = _conn()
    c = conn.cursor()

    # consent type breakdown from DB
    consent_types = c.execute(
        "SELECT consent_type, status, COUNT(*) FROM consent_records GROUP BY consent_type, status"
    ).fetchall()

    # regulatory submissions by status
    reg_by_status = c.execute(
        "SELECT status, COUNT(*) FROM regulatory_submissions GROUP BY status"
    ).fetchall()

    conn.close()

    docs = _doc_catalogue()

    # full document list with submission status
    doc_list = []
    for d in docs:
        doc_list.append({
            "id": d["id"],
            "category": d["cat"],
            "name": d["name"],
            "status": d["status"],
            "iec_phase": d["phase_iec"],
            "irb_phase": d["phase_irb"],
            "iec_submission_status": _iec_phase_status(d),
            "irb_submission_status": _irb_phase_status(d),
            "jurisdiction": d["jurisdiction"],
        })

    # phase-wise summary
    def phase_docs(phase_key, phase_num):
        return [d for d in docs if d[phase_key] == phase_num]

    phase_summary = []
    for system, key in [("IEC", "phase_iec"), ("IRB", "phase_irb")]:
        for ph in [1, 2, 3]:
            ph_docs = phase_docs(key, ph)
            real_ct  = sum(1 for d in ph_docs if d["status"] == "real")
            part_ct  = sum(1 for d in ph_docs if d["status"] == "partial")
            design_ct= sum(1 for d in ph_docs if d["status"] == "design")
            pend_ct  = sum(1 for d in ph_docs if d["status"] == "pending")
            pct = _pct(real_ct + part_ct * 0.5, len(ph_docs)) if ph_docs else 0
            phase_summary.append({
                "system": system, "phase": ph, "total": len(ph_docs),
                "real": real_ct, "partial": part_ct, "design": design_ct,
                "pending": pend_ct, "completion_pct": pct,
            })

    # jurisdiction document map
    jurisdictions = ["India", "USA", "Canada", "International"]
    jur_map = []
    for j in jurisdictions:
        j_docs = [d for d in docs if j in d["jurisdiction"]]
        by_status = {"real": 0, "partial": 0, "design": 0, "pending": 0}
        for d in j_docs:
            by_status[d["status"]] += 1
        jur_map.append({
            "jurisdiction": j,
            "total": len(j_docs),
            **by_status,
            "completion_pct": _pct(by_status["real"] + by_status["partial"] * 0.5, len(j_docs)),
        })

    # consent records from real DB
    consent_breakdown = {}
    for ct, st, cnt in consent_types:
        if ct not in consent_breakdown:
            consent_breakdown[ct] = {}
        consent_breakdown[ct][st] = cnt

    # regulatory submission status from real DB
    reg_status_map = {row[0]: row[1] for row in reg_by_status}

    return {
        "available": True,
        "document_list": doc_list,
        "phase_summary": phase_summary,
        "jurisdiction_map": jur_map,
        "consent_breakdown": [
            {"type": k, **v} for k, v in consent_breakdown.items()
        ],
        "regulatory_submissions": {
            "by_status": reg_status_map,
            "total": sum(reg_status_map.values()),
        },
        "priority_documents": [
            d for d in doc_list
            if d["iec_phase"] == 1 or d["irb_phase"] == 1
        ][:30],
    }


# ─── definitions ─────────────────────────────────────────────────────────────

def definitions():
    return {
        "available": True,
        "concepts": [
            {"term": "IEC",            "def": "Institutional Ethics Committee — India-based ethics review body (ICMR-accredited). Reviews human research protocols for scientific merit, ethical acceptability, and regulatory compliance."},
            {"term": "IRB",            "def": "Institutional Review Board — USA-based ethics review body. GGU IRB provides oversight for the DBA dissertation research per 45 CFR 46 (Common Rule)."},
            {"term": "173-Document Master List", "def": "The complete catalogue of IEC/IRB submission documents across 9 categories (A–I), covering protocol, consent, ethics applications, data management, analysis plans, safety monitoring, and international regulatory compliance."},
            {"term": "Categories A–I", "def": "A=Study Design & Protocol; B=Prospective Survey Package; C=Retrospective EEG Package; D=Consent Forms; E=Ethics Applications; F=Data Management; G=Analysis Plans & Quality; H=Safety & Monitoring; I=International Regulatory Binder."},
            {"term": "Phased Submission","def": "IEC: Phase 1 (Retrospective EEG, easiest approval) → Phase 2 (XAI + clinical, amendment) → Phase 3 (Governance + remote). IRB: Phase 1 (Core) → Phase 2 (AI Governance) → Phase 3 (Clinical Validation/Closure)."},
            {"term": "Real",           "def": "Document is fully drafted, reviewed, and ready for submission with real study data. Status: Real (§57.7 honest — no fabrication)."},
            {"term": "Partial",        "def": "Document is drafted with meaningful content but incomplete — missing sections, data, or signatures. Counts 0.5× toward completion score."},
            {"term": "Design",         "def": "Document structure and key content are designed/planned but not yet fully drafted. Counts 0.25× toward completion score."},
            {"term": "DPIA",           "def": "Data Protection Impact Assessment — mandatory under DPDP Act 2023 (India) and GDPR-aligned frameworks. Assesses privacy risks of processing EEG/clinical data."},
            {"term": "ICH-GCP E6(R3)","def": "International Council for Harmonisation — Good Clinical Practice guideline. Defines international standards for clinical trial conduct, data integrity, and patient protection."},
            {"term": "ICMR AI 2023",  "def": "Indian Council of Medical Research AI ethics guidelines (2023). Covers responsible AI development, bias mitigation, transparency, and patient safety in India."},
            {"term": "HMSC",          "def": "Health Ministry Screening Committee — Indian government body that reviews foreign collaborations involving sensitive health data. Applicability must be determined for cross-border EEG data sharing."},
            {"term": "Common Rule",   "def": "45 CFR 46 — US federal regulation governing the protection of human research subjects. Basis for GGU IRB oversight of the DBA research."},
            {"term": "TCPS 2 (2022)", "def": "Tri-Council Policy Statement: Ethical Conduct for Research Involving Humans — Canada's primary research ethics framework."},
            {"term": "Weighted Completion","def": "Completion score = (1.0×Real + 0.5×Partial + 0.25×Design + 0.0×Pending) / Total × 100%. Reflects document readiness, not just count."},
        ],
        "standards": [
            {"name": "ICMR 2017",         "jurisdiction": "India",         "scope": "Biomedical & health research ethics guidelines"},
            {"name": "ICMR AI 2023",      "jurisdiction": "India",         "scope": "Responsible AI in healthcare"},
            {"name": "DPDP Act 2023",     "jurisdiction": "India",         "scope": "Digital personal data protection"},
            {"name": "HIPAA 45 CFR 164",  "jurisdiction": "USA",           "scope": "Protected health information privacy/security"},
            {"name": "Common Rule 45 CFR 46","jurisdiction": "USA",        "scope": "Human subjects research protection"},
            {"name": "PIPEDA",            "jurisdiction": "Canada",         "scope": "Personal information protection in commercial activity"},
            {"name": "TCPS 2 (2022)",     "jurisdiction": "Canada",         "scope": "Research ethics for human participants"},
            {"name": "ICH-GCP E6(R3)",    "jurisdiction": "International",  "scope": "Good Clinical Practice — trial conduct"},
            {"name": "Declaration of Helsinki","jurisdiction": "International","scope": "WMA ethical principles for medical research"},
            {"name": "CIOMS 2016",        "jurisdiction": "International",  "scope": "International ethical guidelines for research"},
            {"name": "TRIPOD-AI",         "jurisdiction": "International",  "scope": "Transparent reporting of AI prediction models"},
            {"name": "NIST AI RMF",       "jurisdiction": "International",  "scope": "AI risk management framework"},
            {"name": "ISO/IEC 23894",     "jurisdiction": "International",  "scope": "AI risk management guidance"},
        ],
        "thresholds": [
            {"metric": "Phase 1 Readiness", "target": "≥ 80%", "rationale": "Minimum to submit IEC/IRB Phase 1 application"},
            {"metric": "Weighted Completion","target": "≥ 60%", "rationale": "Acceptable progress for active study"},
            {"metric": "Consent Coverage",  "target": "100%",  "rationale": "All enrolled patients must have valid consent"},
            {"metric": "Real documents",    "target": "≥ 70%", "rationale": "Majority of documents fully complete at publication"},
            {"metric": "Jurisdiction Coverage","target": "All 4","rationale": "Multi-jurisdiction compliance mandatory per CLAUDE.md policy"},
            {"metric": "Category Coverage", "target": "All A–I","rationale": "173-doc master list must cover all 9 categories"},
        ],
        "references": [
            "ICMR. National Ethical Guidelines for Biomedical and Health Research Involving Human Participants. 2017.",
            "Ministry of Electronics & Information Technology. Digital Personal Data Protection Act. 2023. India.",
            "Collins GS et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning. BMJ. 2024.",
            "ICH. E6(R3) Good Clinical Practice. International Council for Harmonisation. 2023.",
            "Government of Canada. Tri-Council Policy Statement: Ethical Conduct for Research Involving Humans (TCPS 2). 2022.",
        ],
    }
