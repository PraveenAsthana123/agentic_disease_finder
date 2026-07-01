"""Multimodal AI Dashboard — convergence of EEG, MRI, neuropsychology,
seizure diary, and clinical assessment data per patient.

Shows how different clinical data modalities converge (or diverge) for each
patient, enabling holistic epilepsy evaluation beyond single-modality AI.

Aggregates data from:
- data/clinical.db analyses table    (21 rows  — EEG predictions + 47 features)
- data/clinical.db mri_findings table (40 rows  — MRI lesion characterization)
- data/clinical.db neuropsych table  (1  row   — PHQ9, GAD7, MoCA, MMSE)
- data/clinical.db seizure_diary table (25 rows — seizure events)
- data/clinical.db assessments table (423 rows — clinical instrument scores)
"""

import sqlite3
import json
import os
import math
from datetime import datetime, timezone
from collections import defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# ── Modality registry ────────────────────────────────────────────────────────
MODALITIES = [
    {"name": "EEG Analyses",          "table": "analyses"},
    {"name": "MRI Findings",          "table": "mri_findings"},
    {"name": "Neuropsych",            "table": "neuropsych"},
    {"name": "Seizure Diary",         "table": "seizure_diary"},
    {"name": "Clinical Assessments",  "table": "assessments"},
]

# EEG disease labels that suggest left/right/bilateral laterality
_LEFT_HINTS  = {"left",  "temporal-left",  "l-temporal",  "frontal-left"}
_RIGHT_HINTS = {"right", "temporal-right", "r-temporal",  "frontal-right"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _connect():
    """Return a DB connection with Row factory, or None if DB is missing."""
    if not os.path.exists(DB):
        return None
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn, name):
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row[0] > 0


def _safe_json(blob):
    """Parse a JSON string blob; return {} on any error."""
    if not blob:
        return {}
    try:
        return json.loads(blob)
    except Exception:
        return {}


def _safe_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _round2(v):
    return round(v, 2) if v is not None else None


# ── EEG laterality hint from disease string ──────────────────────────────────

def _eeg_laterality(disease, predicted_label):
    """Guess a rough laterality hint from disease / label text."""
    combined = f"{disease or ''} {predicted_label or ''}".lower()
    if any(h in combined for h in _LEFT_HINTS):
        return "Left"
    if any(h in combined for h in _RIGHT_HINTS):
        return "Right"
    return None   # indeterminate


def _mri_laterality(fields):
    """Extract MRI laterality from parsed fields_json dict."""
    return fields.get("laterality") or None


def _concordance(eeg_lat, mri_lat):
    """
    Compare EEG laterality hint with MRI laterality.
    Returns 'concordant', 'discordant', or 'indeterminate'.
    """
    if not eeg_lat or not mri_lat:
        return "indeterminate"
    el = eeg_lat.lower()
    ml = mri_lat.lower()
    if ml == "bilateral":
        return "indeterminate"
    if el == ml:
        return "concordant"
    return "discordant"


# ════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

def multimodal_overview():
    """
    High-level multimodal coverage summary.

    Returns
    -------
    dict with keys:
        total_patients, modalities, modality_coverage_distribution,
        concordance_summary, kpis, modality_timeline
    """
    conn = _connect()
    if conn is None:
        return {
            "total_patients": 0,
            "modalities": [],
            "modality_coverage_distribution": [],
            "concordance_summary": {"concordant": 0, "discordant": 0, "indeterminate": 0},
            "kpis": {},
            "modality_timeline": [],
        }

    # ── Total patients (union across all modality tables) ────────────────────
    all_patient_ids: set = set()
    for m in MODALITIES:
        tbl = m["table"]
        if _table_exists(conn, tbl):
            rows = conn.execute(f"SELECT DISTINCT patient_id FROM {tbl}").fetchall()
            all_patient_ids.update(r[0] for r in rows)

    # Also pull patients table if available
    if _table_exists(conn, "patients"):
        rows = conn.execute("SELECT DISTINCT patient_id FROM patients").fetchall()
        all_patient_ids.update(r[0] for r in rows)

    total_patients = len(all_patient_ids)

    # ── Per-modality coverage ────────────────────────────────────────────────
    modality_patient_sets = {}
    modality_counts = {}
    modalities_out = []

    for m in MODALITIES:
        tbl = m["table"]
        if _table_exists(conn, tbl):
            rows = conn.execute(f"SELECT patient_id FROM {tbl}").fetchall()
            pts  = set(r[0] for r in rows)
            cnt  = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        else:
            pts = set()
            cnt = 0

        modality_patient_sets[m["name"]] = pts
        modality_counts[m["name"]] = cnt
        cov_pct = _round2(100 * len(pts) / total_patients) if total_patients else 0.0
        modalities_out.append({
            "modality":      m["name"],
            "count":         cnt,
            "patient_count": len(pts),
            "coverage_pct":  cov_pct,
        })

    # ── Modality coverage distribution ──────────────────────────────────────
    patient_modality_count: dict = defaultdict(int)
    for pid in all_patient_ids:
        for m in MODALITIES:
            if pid in modality_patient_sets.get(m["name"], set()):
                patient_modality_count[pid] += 1

    dist_counter: dict = defaultdict(int)
    for pid, cnt in patient_modality_count.items():
        dist_counter[cnt] += 1

    modality_coverage_distribution = [
        {"modality_count": k, "patient_count": v}
        for k, v in sorted(dist_counter.items())
    ]

    # ── EEG ↔ MRI concordance ────────────────────────────────────────────────
    concordance_counts = {"concordant": 0, "discordant": 0, "indeterminate": 0}

    eeg_pts = modality_patient_sets.get("EEG Analyses", set())
    mri_pts = modality_patient_sets.get("MRI Findings", set())
    both_pts = eeg_pts & mri_pts

    # Build quick lookup: patient_id → latest EEG row
    eeg_lookup: dict = {}
    if _table_exists(conn, "analyses"):
        rows = conn.execute(
            "SELECT patient_id, disease, predicted_label FROM analyses ORDER BY created_at DESC"
        ).fetchall()
        for r in rows:
            if r["patient_id"] not in eeg_lookup:
                eeg_lookup[r["patient_id"]] = r

    # Build quick lookup: patient_id → latest MRI row
    mri_lookup: dict = {}
    if _table_exists(conn, "mri_findings"):
        rows = conn.execute(
            "SELECT patient_id, fields_json FROM mri_findings ORDER BY created_at DESC"
        ).fetchall()
        for r in rows:
            if r["patient_id"] not in mri_lookup:
                mri_lookup[r["patient_id"]] = _safe_json(r["fields_json"])

    for pid in both_pts:
        eeg_row  = eeg_lookup.get(pid)
        mri_flds = mri_lookup.get(pid, {})
        eeg_lat  = _eeg_laterality(
            eeg_row["disease"] if eeg_row else None,
            eeg_row["predicted_label"] if eeg_row else None,
        )
        mri_lat  = _mri_laterality(mri_flds)
        verdict  = _concordance(eeg_lat, mri_lat)
        concordance_counts[verdict] += 1

    # ── Modality timeline ────────────────────────────────────────────────────
    modality_timeline = []
    for m in MODALITIES:
        tbl = m["table"]
        if not _table_exists(conn, tbl):
            continue
        try:
            rows = conn.execute(
                f"SELECT created_at FROM {tbl} WHERE created_at IS NOT NULL"
            ).fetchall()
            month_counts: dict = defaultdict(int)
            for r in rows:
                raw = r[0] or ""
                try:
                    month = raw[:7]  # "YYYY-MM"
                    if len(month) == 7:
                        month_counts[month] += 1
                except Exception:
                    pass
            for month, cnt in sorted(month_counts.items()):
                modality_timeline.append({"month": month, "modality": m["name"], "count": cnt})
        except Exception:
            pass

    # ── KPIs ─────────────────────────────────────────────────────────────────
    total_records = sum(modality_counts.values())
    avg_modalities = (
        _round2(sum(patient_modality_count.values()) / total_patients)
        if total_patients else 0.0
    )
    patients_full_coverage = sum(
        1 for v in patient_modality_count.values() if v >= 3
    )
    total_concordance_checked = len(both_pts)
    concordance_rate = (
        _round2(100 * concordance_counts["concordant"] / total_concordance_checked)
        if total_concordance_checked else 0.0
    )

    conn.close()
    return {
        "total_patients":                total_patients,
        "modalities":                    modalities_out,
        "modality_coverage_distribution": modality_coverage_distribution,
        "concordance_summary":           concordance_counts,
        "kpis": {
            "total_patients":              total_patients,
            "total_records":               total_records,
            "avg_modalities_per_patient":  avg_modalities,
            "patients_with_full_coverage": patients_full_coverage,
            "eeg_mri_concordance_rate":    concordance_rate,
        },
        "modality_timeline": modality_timeline,
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. BREAKDOWN
# ════════════════════════════════════════════════════════════════════════════

def multimodal_breakdown():
    """
    Per-patient multimodal profiles plus aggregate distributions.

    Returns
    -------
    dict with keys:
        patient_profiles, modality_correlation_matrix,
        mri_lesion_distribution, eeg_disease_distribution,
        confidence_by_modality_count
    """
    conn = _connect()
    if conn is None:
        return {
            "patient_profiles":             [],
            "modality_correlation_matrix":  [],
            "mri_lesion_distribution":      [],
            "eeg_disease_distribution":     [],
            "confidence_by_modality_count": [],
        }

    # ── Load all modality data indexed by patient_id ─────────────────────────

    # EEG — latest analysis per patient
    eeg_by_patient: dict = {}
    if _table_exists(conn, "analyses"):
        rows = conn.execute(
            "SELECT patient_id, disease, predicted_label, confidence, signal_quality "
            "FROM analyses ORDER BY created_at DESC"
        ).fetchall()
        for r in rows:
            if r["patient_id"] not in eeg_by_patient:
                eeg_by_patient[r["patient_id"]] = {
                    "disease":        r["disease"],
                    "predicted_label": r["predicted_label"],
                    "confidence":     _safe_float(r["confidence"]),
                    "signal_quality": r["signal_quality"],
                }

    # MRI — latest per patient
    mri_by_patient: dict = {}
    if _table_exists(conn, "mri_findings"):
        rows = conn.execute(
            "SELECT patient_id, fields_json FROM mri_findings ORDER BY created_at DESC"
        ).fetchall()
        for r in rows:
            if r["patient_id"] not in mri_by_patient:
                f = _safe_json(r["fields_json"])
                mri_by_patient[r["patient_id"]] = {
                    "lesion_type":     f.get("lesion_type"),
                    "lesion_location": f.get("lesion_location"),
                    "laterality":      f.get("laterality"),
                    "classification":  f.get("classification"),
                }

    # Neuropsych — only table-level scores (single record in DB)
    neuro_by_patient: dict = {}
    if _table_exists(conn, "neuropsych"):
        rows = conn.execute(
            "SELECT patient_id, fields_json FROM neuropsych"
        ).fetchall()
        for r in rows:
            f = _safe_json(r["fields_json"])
            neuro_by_patient[r["patient_id"]] = {
                "phq9": _safe_float(f.get("phq9")),
                "gad7": _safe_float(f.get("gad7")),
                "moca": _safe_float(f.get("moca")),
                "mmse": _safe_float(f.get("mmse")),
            }

    # Seizure diary — count per patient
    seizure_count_by_patient: dict = defaultdict(int)
    if _table_exists(conn, "seizure_diary"):
        rows = conn.execute(
            "SELECT patient_id FROM seizure_diary"
        ).fetchall()
        for r in rows:
            seizure_count_by_patient[r["patient_id"]] += 1

    # Assessments — count per patient
    assessment_count_by_patient: dict = defaultdict(int)
    if _table_exists(conn, "assessments"):
        rows = conn.execute(
            "SELECT patient_id FROM assessments"
        ).fetchall()
        for r in rows:
            assessment_count_by_patient[r["patient_id"]] += 1

    # ── Union of all patient IDs ─────────────────────────────────────────────
    all_pids: set = set()
    all_pids.update(eeg_by_patient.keys())
    all_pids.update(mri_by_patient.keys())
    all_pids.update(neuro_by_patient.keys())
    all_pids.update(seizure_count_by_patient.keys())
    all_pids.update(assessment_count_by_patient.keys())
    if _table_exists(conn, "patients"):
        rows = conn.execute("SELECT DISTINCT patient_id FROM patients").fetchall()
        all_pids.update(r[0] for r in rows)

    # ── Per-patient profiles ─────────────────────────────────────────────────
    patient_profiles = []
    for pid in sorted(all_pids):
        mods_available = []
        if pid in eeg_by_patient:
            mods_available.append("EEG Analyses")
        if pid in mri_by_patient:
            mods_available.append("MRI Findings")
        if pid in neuro_by_patient:
            mods_available.append("Neuropsych")
        if seizure_count_by_patient.get(pid, 0) > 0:
            mods_available.append("Seizure Diary")
        if assessment_count_by_patient.get(pid, 0) > 0:
            mods_available.append("Clinical Assessments")

        eeg_sum  = eeg_by_patient.get(pid)
        mri_sum  = mri_by_patient.get(pid)
        neuro_sum = neuro_by_patient.get(pid)

        # Concordance
        eeg_lat = None
        mri_lat = None
        if eeg_sum:
            eeg_lat = _eeg_laterality(eeg_sum.get("disease"), eeg_sum.get("predicted_label"))
        if mri_sum:
            mri_lat = _mri_laterality(mri_sum)

        if eeg_sum and mri_sum:
            conc = _concordance(eeg_lat, mri_lat)
        else:
            conc = "insufficient_data"

        patient_profiles.append({
            "patient_id":          pid,
            "modalities_available": mods_available,
            "modality_count":      len(mods_available),
            "eeg_summary":         eeg_sum,
            "mri_summary":         mri_sum,
            "neuropsych_summary":  neuro_sum,
            "seizure_count":       seizure_count_by_patient.get(pid, 0),
            "assessment_count":    assessment_count_by_patient.get(pid, 0),
            "concordance_status":  conc,
        })

    # ── Modality co-occurrence (correlation) matrix ───────────────────────────
    mod_names = [m["name"] for m in MODALITIES]

    # patient-level presence sets
    presence: dict = {
        "EEG Analyses":         set(eeg_by_patient.keys()),
        "MRI Findings":         set(mri_by_patient.keys()),
        "Neuropsych":           set(neuro_by_patient.keys()),
        "Seizure Diary":        set(p for p, c in seizure_count_by_patient.items() if c > 0),
        "Clinical Assessments": set(p for p, c in assessment_count_by_patient.items() if c > 0),
    }

    modality_correlation_matrix = []
    for i, m1 in enumerate(mod_names):
        for j, m2 in enumerate(mod_names):
            both = len(presence[m1] & presence[m2])
            modality_correlation_matrix.append({
                "modality_a":    m1,
                "modality_b":    m2,
                "patient_count": both,
            })

    # ── MRI lesion distribution ───────────────────────────────────────────────
    lesion_counter: dict = defaultdict(int)
    if _table_exists(conn, "mri_findings"):
        rows = conn.execute("SELECT fields_json FROM mri_findings").fetchall()
        for r in rows:
            f = _safe_json(r["fields_json"])
            lt = f.get("lesion_type") or "Unknown"
            lesion_counter[lt] += 1

    mri_lesion_distribution = [
        {"lesion_type": k, "count": v}
        for k, v in sorted(lesion_counter.items(), key=lambda x: -x[1])
    ]

    # ── EEG disease distribution ──────────────────────────────────────────────
    disease_counter: dict = defaultdict(int)
    if _table_exists(conn, "analyses"):
        rows = conn.execute("SELECT disease FROM analyses").fetchall()
        for r in rows:
            d = r["disease"] or "Unknown"
            disease_counter[d] += 1

    eeg_disease_distribution = [
        {"disease": k, "count": v}
        for k, v in sorted(disease_counter.items(), key=lambda x: -x[1])
    ]

    # ── Average EEG confidence by modality count ──────────────────────────────
    # Build map patient_id → modality_count from profiles
    pid_modcount = {p["patient_id"]: p["modality_count"] for p in patient_profiles}
    conf_buckets: dict = defaultdict(list)
    for pid, eeg_s in eeg_by_patient.items():
        conf = eeg_s.get("confidence")
        mc   = pid_modcount.get(pid, 0)
        if conf is not None:
            conf_buckets[mc].append(conf)

    confidence_by_modality_count = [
        {
            "modality_count":   mc,
            "avg_confidence":   _round2(sum(confs) / len(confs)),
            "patient_count":    len(confs),
        }
        for mc, confs in sorted(conf_buckets.items())
    ]

    conn.close()
    return {
        "patient_profiles":             patient_profiles,
        "modality_correlation_matrix":  modality_correlation_matrix,
        "mri_lesion_distribution":      mri_lesion_distribution,
        "eeg_disease_distribution":     eeg_disease_distribution,
        "confidence_by_modality_count": confidence_by_modality_count,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════

def multimodal_definitions():
    """
    Clinical and technical definitions for the Multimodal AI Dashboard.

    Returns
    -------
    dict with key 'sections': list of definition section dicts.
    """
    return {
        "sections": [
            {
                "title": "Multimodal Integration Concepts",
                "items": [
                    {
                        "term": "Multimodal AI",
                        "description": (
                            "An AI system that ingests and fuses data from more than one "
                            "clinical modality (e.g., EEG + MRI + neuropsychology) to improve "
                            "diagnostic accuracy, reduce single-modality bias, and provide a "
                            "richer representation of a patient's epileptic condition."
                        ),
                    },
                    {
                        "term": "Data Fusion",
                        "description": (
                            "The process of combining features or decisions from multiple "
                            "modalities. Early fusion merges raw features before classification; "
                            "late fusion combines individual modality predictions (e.g., majority "
                            "vote or weighted ensemble); intermediate fusion integrates learned "
                            "representations at a hidden layer."
                        ),
                    },
                    {
                        "term": "Modality Concordance",
                        "description": (
                            "Agreement between two modalities on a clinically relevant finding "
                            "(e.g., EEG left-temporal focus + MRI left hippocampal sclerosis = "
                            "concordant). Concordance increases confidence in surgical "
                            "candidacy; discordance triggers further investigation."
                        ),
                    },
                    {
                        "term": "Complementary Modalities",
                        "description": (
                            "Modalities that capture distinct, non-redundant aspects of disease: "
                            "EEG captures ictal/interictal electrical activity; MRI reveals "
                            "structural lesions; neuropsychology measures cognitive and mood "
                            "consequences; seizure diary tracks frequency and semiology over time."
                        ),
                    },
                    {
                        "term": "Coverage",
                        "description": (
                            "The percentage of patients for whom a specific modality has at "
                            "least one recorded data point. Low coverage reduces the reliability "
                            "of multimodal fusion conclusions for that cohort."
                        ),
                    },
                ],
            },
            {
                "title": "Data Modalities",
                "items": [
                    {
                        "term": "EEG Analyses",
                        "description": (
                            "Electroencephalography recordings processed to extract 47 time-domain, "
                            "spectral, complexity, and Hjorth features. An AI classifier predicts "
                            "disease label (e.g., Epilepsy vs Control) with an associated "
                            "confidence score and signal quality rating."
                        ),
                    },
                    {
                        "term": "MRI Findings",
                        "description": (
                            "Structural brain MRI reviewed by a radiologist using an epilepsy "
                            "protocol (3 T, 3D-FLAIR, coronal T2, volumetric T1). Fields include "
                            "lesion type (HS, FCD, Cavernoma, Tumour, etc.), laterality, "
                            "hippocampal volume asymmetry, T2/FLAIR signal, and classification "
                            "(LESIONAL / NON-LESIONAL / UNKNOWN)."
                        ),
                    },
                    {
                        "term": "Neuropsychology",
                        "description": (
                            "Standardised cognitive and mood assessments: PHQ-9 (depression "
                            "screen, 0–27), GAD-7 (anxiety screen, 0–21), MoCA (Montreal "
                            "Cognitive Assessment, 0–30), and MMSE (Mini-Mental State Exam, "
                            "0–30). Scores flag psychiatric comorbidities that may affect "
                            "surgical candidacy and quality-of-life outcomes."
                        ),
                    },
                    {
                        "term": "Seizure Diary",
                        "description": (
                            "Patient-reported or clinician-logged seizure events including date, "
                            "duration, severity, falls, and post-ictal features. Longitudinal "
                            "seizure frequency and pattern inform treatment response and "
                            "prognostic modelling."
                        ),
                    },
                    {
                        "term": "Clinical Assessments",
                        "description": (
                            "Structured clinical instrument scores stored per patient (e.g., "
                            "additional PHQ-9/GAD-7 administrations, custom scales). These "
                            "supplement the neuropsych table with longitudinal repeated measures "
                            "and multi-instrument coverage across the cohort."
                        ),
                    },
                ],
            },
            {
                "title": "Concordance & Fusion Metrics",
                "items": [
                    {
                        "term": "EEG–MRI Concordance Rate",
                        "description": (
                            "Percentage of patients with both EEG and MRI data where the EEG "
                            "laterality prediction matches the MRI lesion laterality. Formula: "
                            "concordant_patients / (concordant + discordant) × 100. Bilateral "
                            "MRI lesions are scored as indeterminate."
                        ),
                    },
                    {
                        "term": "Modality Coverage (%)",
                        "description": (
                            "For each modality: (patients with ≥1 record in that modality) / "
                            "total_patients × 100. Target ≥80 % for primary modalities (EEG, "
                            "MRI) in a pre-surgical epilepsy evaluation cohort."
                        ),
                    },
                    {
                        "term": "Modality Co-occurrence",
                        "description": (
                            "Number of patients who have data in both modality A and modality B. "
                            "Presented as a symmetric matrix. High co-occurrence enables "
                            "multimodal fusion; low co-occurrence limits joint analysis."
                        ),
                    },
                    {
                        "term": "Average Modalities per Patient",
                        "description": (
                            "Mean number of distinct data modalities recorded for each patient. "
                            "Higher values indicate richer clinical workup. Patients with ≥3 "
                            "modalities are classified as 'full coverage' for this dashboard."
                        ),
                    },
                    {
                        "term": "Fusion Confidence",
                        "description": (
                            "The degree of certainty a multimodal model assigns to its fused "
                            "prediction. Typically higher than single-modality confidence when "
                            "modalities are concordant, and lower (or flagged for review) when "
                            "modalities are discordant."
                        ),
                    },
                ],
            },
            {
                "title": "Clinical Relevance",
                "items": [
                    {
                        "term": "IEC 62304 — Software Lifecycle",
                        "description": (
                            "Requires that a medical device software system maintain traceability "
                            "between clinical requirements and software behaviour. Multimodal AI "
                            "must document how each modality contributes to the output and define "
                            "failure modes when a modality is missing or of low quality."
                        ),
                    },
                    {
                        "term": "FDA AI/ML PCCP — Predetermined Change Control Plan",
                        "description": (
                            "FDA guidance allows AI/ML-based SaMD to update continuously if a "
                            "PCCP is pre-specified. Multimodal systems must describe which "
                            "modality weights may change, acceptable drift bounds per modality, "
                            "and re-validation triggers before deployment."
                        ),
                    },
                    {
                        "term": "ILAE — Multimodal Pre-surgical Evaluation",
                        "description": (
                            "International League Against Epilepsy guidelines recommend that "
                            "epilepsy surgery candidates undergo a comprehensive multimodal "
                            "evaluation: video-EEG, high-resolution MRI, neuropsychological "
                            "testing, and—where available—FDG-PET, SPECT, and intracranial EEG. "
                            "Concordance across modalities is a key decision criterion."
                        ),
                    },
                    {
                        "term": "ISO 14971 — Risk Management",
                        "description": (
                            "Requires risk analysis of all AI outputs used in clinical decisions. "
                            "For multimodal systems, risks include: discordant modalities causing "
                            "confusion, missing modality leading to incorrect classification, "
                            "and bias introduced by unequal modality coverage across demographic "
                            "groups."
                        ),
                    },
                    {
                        "term": "EU AI Act — High-Risk AI Systems",
                        "description": (
                            "Multimodal AI used in epilepsy pre-surgical evaluation falls under "
                            "Annex III high-risk AI. Obligations include: technical documentation "
                            "of each modality's contribution, human oversight mechanisms, "
                            "accuracy metrics per modality, and post-market monitoring of "
                            "concordance rates across real-world populations."
                        ),
                    },
                ],
            },
            {
                "title": "Remediation Strategies",
                "items": [
                    {
                        "term": "Discordant Modalities",
                        "description": (
                            "When EEG and MRI suggest different laterality: (1) review raw EEG "
                            "for focal interictal discharges; (2) obtain higher-resolution MRI "
                            "or 7 T protocol; (3) consider FDG-PET or ictal SPECT; (4) refer "
                            "to multidisciplinary epilepsy conference before surgery decision."
                        ),
                    },
                    {
                        "term": "Low Modality Coverage",
                        "description": (
                            "If <50 % of patients have a given modality: (1) review data "
                            "ingestion pipelines for that modality; (2) audit whether data "
                            "exists but is not linked to patient_id; (3) assess whether "
                            "the cohort clinically requires that modality (e.g., neuropsych "
                            "may not be universal); (4) impute missing data only with explicit "
                            "clinical justification and uncertainty flagging."
                        ),
                    },
                    {
                        "term": "Missing MRI in EEG-Positive Patients",
                        "description": (
                            "Patients with EEG evidence of epilepsy but no MRI record: flag for "
                            "urgent MRI scheduling. Non-lesional EEG + no MRI is an incomplete "
                            "workup per ILAE pre-surgical standards and may mask surgically "
                            "remediable structural lesions."
                        ),
                    },
                    {
                        "term": "Low Average Modalities per Patient",
                        "description": (
                            "avg_modalities_per_patient < 2 indicates a shallow workup. "
                            "Recommended actions: (1) create care pathways that auto-schedule "
                            "complementary modalities on referral; (2) review exclusion criteria "
                            "that may be filtering out data records; (3) set KPI targets and "
                            "dashboard alerts when coverage drops below threshold."
                        ),
                    },
                    {
                        "term": "Poor EEG Signal Quality with No Alternative Modality",
                        "description": (
                            "If EEG signal_quality is Poor/Artifact and the patient lacks MRI "
                            "or neuropsych data: (1) repeat EEG under controlled conditions; "
                            "(2) escalate to long-term video-EEG monitoring; (3) do not use "
                            "the AI prediction as evidence without at least one corroborating "
                            "modality; (4) document the uncertainty in the clinical record."
                        ),
                    },
                ],
            },
        ]
    }


# ── CLI smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint

    print("=== multimodal_overview ===")
    ov = multimodal_overview()
    pprint.pprint({k: v for k, v in ov.items() if k != "modality_timeline"})
    print(f"  timeline entries: {len(ov.get('modality_timeline', []))}")

    print("\n=== multimodal_breakdown (summary) ===")
    bk = multimodal_breakdown()
    print(f"  patient_profiles:             {len(bk['patient_profiles'])} patients")
    print(f"  modality_correlation_matrix:  {len(bk['modality_correlation_matrix'])} pairs")
    print(f"  mri_lesion_distribution:      {bk['mri_lesion_distribution']}")
    print(f"  eeg_disease_distribution:     {bk['eeg_disease_distribution']}")
    print(f"  confidence_by_modality_count: {bk['confidence_by_modality_count']}")

    print("\n=== multimodal_definitions ===")
    df = multimodal_definitions()
    for s in df["sections"]:
        print(f"  [{s['title']}] — {len(s['items'])} terms")
