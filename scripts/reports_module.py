"""Patient Reports module — real EEG/summary reports from clinical.db.

Pulls from: analyses, assessments, seizure_diary, component_findings, expert_reviews,
clinical_decisions, hitl_reviews.  Each patient gets an aggregate report card.
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def _safe(row):
    """Convert sqlite3.Row to dict with JSON-safe values."""
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, bytes):
            d[k] = v.decode("utf-8", errors="replace")
    return d


# ── overview: all patients with report summaries ────────────────────────
def reports_overview(patient_id=None):
    """Return report cards for all patients (or one), pulling real data."""
    c = _conn()
    try:
        # Get all patients
        if patient_id:
            patients = [_safe(r) for r in c.execute(
                "SELECT * FROM patients WHERE patient_id=?", (patient_id,)).fetchall()]
        else:
            patients = [_safe(r) for r in c.execute(
                "SELECT * FROM patients ORDER BY patient_id").fetchall()]

        reports = []
        for p in patients:
            pid = p["patient_id"]

            # Latest analysis
            analysis = c.execute(
                "SELECT * FROM analyses WHERE patient_id=? ORDER BY id DESC LIMIT 1",
                (pid,)).fetchone()
            analysis = _safe(analysis) if analysis else None

            # Assessment count + latest
            acount = c.execute(
                "SELECT COUNT(*) FROM assessments WHERE patient_id=?", (pid,)).fetchone()[0]
            latest_assess = c.execute(
                "SELECT instrument, score, max_score, interpretation, level, created_at "
                "FROM assessments WHERE patient_id=? ORDER BY id DESC LIMIT 1",
                (pid,)).fetchone()
            latest_assess = _safe(latest_assess) if latest_assess else None

            # Seizure diary count
            sz_count = c.execute(
                "SELECT COUNT(*) FROM seizure_diary WHERE patient_id=?", (pid,)).fetchone()[0]

            # Expert reviews
            reviews = [_safe(r) for r in c.execute(
                "SELECT role, expert, finding, agree_with_ai, created_at "
                "FROM expert_reviews WHERE patient_id=? ORDER BY id DESC LIMIT 5",
                (pid,)).fetchall()]

            # Component findings
            comp_count = c.execute(
                "SELECT COUNT(*) FROM component_findings WHERE patient_id=?", (pid,)).fetchone()[0]

            # Clinical decisions
            decisions = [_safe(r) for r in c.execute(
                "SELECT ai_prediction, ai_confidence, final_decision, reviewer, created_at "
                "FROM clinical_decisions WHERE patient_id=? ORDER BY id DESC LIMIT 3",
                (pid,)).fetchall()]

            # HITL reviews
            hitl = c.execute(
                "SELECT COUNT(*) FROM hitl_reviews WHERE patient_id=?", (pid,)).fetchone()[0]

            # Medication count
            med_count = c.execute(
                "SELECT COUNT(*) FROM medications WHERE patient_id=?", (pid,)).fetchone()[0]

            # Report status
            has_analysis = analysis is not None
            has_assessments = acount > 0
            has_seizure_log = sz_count > 0
            completeness = sum([has_analysis, has_assessments, has_seizure_log, comp_count > 0, len(reviews) > 0]) / 5

            status = "complete" if completeness >= 0.6 else "partial" if completeness > 0 else "pending"

            reports.append({
                "patient_id": pid,
                "name": p.get("name", ""),
                "age": p.get("age"),
                "gender": p.get("gender", ""),
                "disease": p.get("disease", ""),
                "department": p.get("department", ""),
                "status": status,
                "completeness": round(completeness * 100),
                "latest_analysis": {
                    "predicted_label": analysis.get("predicted_label", "") if analysis else "",
                    "confidence": analysis.get("confidence") if analysis else None,
                    "signal_quality": analysis.get("signal_quality", "") if analysis else "",
                    "date": analysis.get("created_at", "") if analysis else "",
                } if analysis else None,
                "assessment_count": acount,
                "latest_assessment": latest_assess,
                "seizure_diary_entries": sz_count,
                "expert_reviews": reviews,
                "component_findings_count": comp_count,
                "clinical_decisions": decisions,
                "hitl_review_count": hitl,
                "medication_count": med_count,
            })

        return {"available": True, "total_patients": len(reports), "reports": reports}
    finally:
        c.close()


# ── summary stats ───────────────────────────────────────────────────────
def reports_summary():
    """Aggregate stats across all patient reports."""
    c = _conn()
    try:
        total_patients = c.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        total_analyses = c.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        total_assessments = c.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
        total_seizure_events = c.execute("SELECT COUNT(*) FROM seizure_diary").fetchone()[0]
        total_reviews = c.execute("SELECT COUNT(*) FROM expert_reviews").fetchone()[0]
        total_decisions = c.execute("SELECT COUNT(*) FROM clinical_decisions").fetchone()[0]
        total_hitl = c.execute("SELECT COUNT(*) FROM hitl_reviews").fetchone()[0]

        # Disease breakdown
        disease_rows = c.execute(
            "SELECT disease, COUNT(*) as cnt FROM patients GROUP BY disease ORDER BY cnt DESC"
        ).fetchall()
        disease_breakdown = [{"disease": r[0] or "Unknown", "count": r[1]} for r in disease_rows]

        # Assessment instrument breakdown
        instrument_rows = c.execute(
            "SELECT instrument, COUNT(*) as cnt FROM assessments GROUP BY instrument ORDER BY cnt DESC"
        ).fetchall()
        instrument_breakdown = [{"instrument": r[0], "count": r[1]} for r in instrument_rows]

        # Analysis prediction breakdown
        pred_rows = c.execute(
            "SELECT predicted_label, COUNT(*) as cnt FROM analyses GROUP BY predicted_label ORDER BY cnt DESC"
        ).fetchall()
        prediction_breakdown = [{"label": r[0] or "Unknown", "count": r[1]} for r in pred_rows]

        # Patients with/without analysis
        patients_with_analysis = c.execute(
            "SELECT COUNT(DISTINCT patient_id) FROM analyses"
        ).fetchone()[0]

        return {
            "available": True,
            "total_patients": total_patients,
            "total_analyses": total_analyses,
            "total_assessments": total_assessments,
            "total_seizure_events": total_seizure_events,
            "total_expert_reviews": total_reviews,
            "total_clinical_decisions": total_decisions,
            "total_hitl_reviews": total_hitl,
            "patients_with_analysis": patients_with_analysis,
            "patients_without_analysis": total_patients - patients_with_analysis,
            "disease_breakdown": disease_breakdown,
            "instrument_breakdown": instrument_breakdown,
            "prediction_breakdown": prediction_breakdown,
        }
    finally:
        c.close()


# ── definitions (tooltip text) ──────────────────────────────────────────
def reports_definitions():
    return {
        "available": True,
        "definitions": {
            "EEG Analysis": "AI-predicted seizure classification from uploaded EEG recordings, including confidence score and signal quality assessment.",
            "Assessments": "Validated clinical instruments (PHQ-9, GAD-7, MoCA, NDDI-E, QOLIE-31, etc.) scored and tracked per patient.",
            "Seizure Diary": "Patient-reported seizure events with duration, severity, triggers, aura, motor signs, and post-ictal state.",
            "Expert Reviews": "Specialist findings by neurologists, epileptologists, and other experts — includes agreement with AI predictions.",
            "Component Findings": "Doctor-entered findings for each EEG report component (background, epileptiform, artifacts, etc.).",
            "Clinical Decisions": "Final clinical decisions made after reviewing AI predictions and expert opinions.",
            "HITL Reviews": "Human-in-the-loop reviews where clinicians verify or correct AI predictions.",
            "Report Completeness": "Percentage based on: has EEG analysis, has assessments, has seizure diary, has component findings, has expert reviews.",
            "Status": "complete (≥60% data), partial (some data), pending (no data yet).",
        }
    }
