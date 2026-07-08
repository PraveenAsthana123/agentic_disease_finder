"""
Patient-Facing Report Dashboard
================================
Generates simplified, patient-friendly reports from existing clinical data.
No jargon, no raw scores — plain-language summaries that patients can understand.

Addresses:
  - Pipeline step 22: "patient-facing report planned"
  - Role challenge: Patient — "Delays in getting results"
  - Accessible health literacy (6th-grade reading level target)

Data Sources:
  - analyses          (21 rows)  — EEG analysis results per patient
  - assessments       (423 rows) — PHQ-9, GAD-7, MoCA, QOLIE-31, etc.
  - patients          (40 rows)  — demographics
  - patient_demographics (30 rows) — extended demographics
  - seizure_diary     (25 rows)  — seizure event log
  - pro_outcomes      (180 rows) — patient-reported outcomes
  - medications       (9 rows)   — current prescriptions
  - medication_adherence (12600 rows) — dose tracking
  - mri_findings      (40 rows)  — imaging results

Author: Research Team
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter, defaultdict

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _safe_json(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _fmt(val, decimals=1):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


# ── Plain-language mappings ─────────────────────────────────────────

INSTRUMENT_NAMES = {
    "PHQ9": "Depression Screening (PHQ-9)",
    "GAD7": "Anxiety Screening (GAD-7)",
    "MOCA": "Memory & Thinking Test (MoCA)",
    "QOLIE31": "Quality of Life (QOLIE-31)",
    "NDDIE": "Epilepsy Depression Screen (NDDI-E)",
    "MMSE": "Mental Status Exam (MMSE)",
    "BARTHEL": "Daily Activity Index (Barthel)",
    "EPWORTH": "Sleepiness Scale (Epworth)",
    "BNT": "Naming Test (BNT)",
    "WAB": "Language Assessment (WAB)",
    "VERBAL_FLUENCY": "Word Fluency Test",
    "MASA": "Swallowing Assessment (MASA)",
    "LSSS": "Seizure Severity Scale (LSSS)",
    "CSSRS": "Safety Screening (C-SSRS)",
    "WAIS": "Intelligence Assessment (WAIS)",
    "DIGIT_SPAN": "Attention Test (Digit Span)",
}

LEVEL_PLAIN = {
    "normal": "Within normal range",
    "mild": "Mildly elevated — worth discussing with your doctor",
    "moderate": "Moderately elevated — your doctor may recommend follow-up",
    "severe": "Significantly elevated — please discuss with your care team soon",
    "critical": "Needs prompt attention — your care team has been notified",
    "minimal": "Within normal range",
    "low": "Below expected range — your doctor may want to follow up",
    "high": "Above expected range — worth discussing at your next visit",
}

CONFIDENCE_PLAIN = {
    "high": "The AI system is quite confident in this result",
    "medium": "The AI system has moderate confidence — your doctor will review",
    "low": "The AI system has lower confidence — your doctor will confirm",
}


def _confidence_category(conf):
    if conf is None:
        return "medium"
    if conf >= 0.8:
        return "high"
    if conf >= 0.6:
        return "medium"
    return "low"


def _score_percent(score, max_score):
    if score is None or max_score is None or max_score == 0:
        return None
    return round(score / max_score * 100, 1)


# ── Overview ────────────────────────────────────────────────────────

def overview():
    """Dashboard KPIs: total patients with reports, report coverage,
    assessment summary distribution, recent report activity."""
    conn = _conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Patients with analyses (have EEG results)
    analysis_patients = [dict(r) for r in cur.execute(
        "SELECT DISTINCT patient_id FROM analyses"
    ).fetchall()]
    analysis_pids = {r["patient_id"] for r in analysis_patients}

    # Patients with assessments
    assessment_patients = [dict(r) for r in cur.execute(
        "SELECT DISTINCT patient_id FROM assessments"
    ).fetchall()]
    assess_pids = {r["patient_id"] for r in assessment_patients}

    # All patients
    all_patients = [dict(r) for r in cur.execute(
        "SELECT patient_id, name, age, gender, disease FROM patients"
    ).fetchall()]
    all_pids = {r["patient_id"] for r in all_patients}

    # Patients with at least some reportable data
    reportable_pids = analysis_pids | assess_pids
    coverage_pct = round(len(reportable_pids) / len(all_pids) * 100, 1) if all_pids else 0

    # Assessment level distribution
    levels = cur.execute(
        "SELECT level, COUNT(*) as cnt FROM assessments GROUP BY level ORDER BY cnt DESC"
    ).fetchall()
    level_dist = [{"name": LEVEL_PLAIN.get(r["level"], r["level"]), "level": r["level"],
                   "value": r["cnt"]} for r in levels]

    # Instrument usage
    instruments = cur.execute(
        "SELECT instrument, COUNT(*) as cnt FROM assessments GROUP BY instrument ORDER BY cnt DESC"
    ).fetchall()
    instrument_dist = [{"name": INSTRUMENT_NAMES.get(r["instrument"], r["instrument"]),
                        "code": r["instrument"], "value": r["cnt"]} for r in instruments]

    # Assessment trend by month
    monthly = cur.execute("""
        SELECT strftime('%Y-%m', created_at) as month, COUNT(*) as cnt
        FROM assessments WHERE created_at IS NOT NULL
        GROUP BY month ORDER BY month
    """).fetchall()
    monthly_trend = [{"month": r["month"], "count": r["cnt"]} for r in monthly]

    # Disease distribution among reportable patients
    disease_dist = Counter()
    for p in all_patients:
        if p["patient_id"] in reportable_pids:
            disease_dist[p.get("disease", "unknown")] += 1
    disease_chart = [{"name": k.title() if k else "Unknown", "value": v}
                     for k, v in disease_dist.most_common()]

    # Seizure diary summary
    seizure_count = cur.execute("SELECT COUNT(*) FROM seizure_diary").fetchone()[0]
    seizure_patients = cur.execute("SELECT COUNT(DISTINCT patient_id) FROM seizure_diary").fetchone()[0]

    # MRI findings summary
    mri_count = cur.execute("SELECT COUNT(*) FROM mri_findings").fetchone()[0]
    mri_patients = cur.execute("SELECT COUNT(DISTINCT patient_id) FROM mri_findings").fetchone()[0]

    # Medication adherence average
    adherence_avg = cur.execute(
        "SELECT AVG(CASE WHEN taken = 'yes' THEN 100.0 ELSE 0.0 END) FROM medication_adherence"
    ).fetchone()[0]

    conn.close()

    return {
        "kpis": {
            "total_patients": len(all_pids),
            "reportable_patients": len(reportable_pids),
            "report_coverage_pct": coverage_pct,
            "total_assessments": sum(r["value"] for r in instrument_dist),
            "instruments_used": len(instrument_dist),
            "seizure_events_logged": seizure_count,
            "mri_scans_reviewed": mri_count,
            "avg_medication_adherence_pct": round(adherence_avg, 1) if adherence_avg else 0,
        },
        "level_distribution": level_dist,
        "instrument_distribution": instrument_dist,
        "monthly_trend": monthly_trend,
        "disease_distribution": disease_chart,
        "data_coverage": {
            "eeg_analyses": len(analysis_pids),
            "assessments": len(assess_pids),
            "seizure_diary": seizure_patients,
            "mri_findings": mri_patients,
        }
    }


# ── Breakdown (per-patient report generation) ──────────────────────

def breakdown(patient_id=None):
    """Per-patient simplified reports. If patient_id given, generate
    a full patient-facing report for that patient."""
    conn = _conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Patient list with report readiness
    patients = [dict(r) for r in cur.execute(
        "SELECT patient_id, name, age, gender, disease FROM patients ORDER BY patient_id"
    ).fetchall()]

    patient_data_map = {}
    for p in patients:
        pid = p["patient_id"]
        has_analysis = cur.execute(
            "SELECT COUNT(*) FROM analyses WHERE patient_id=?", (pid,)
        ).fetchone()[0] > 0
        has_assess = cur.execute(
            "SELECT COUNT(*) FROM assessments WHERE patient_id=?", (pid,)
        ).fetchone()[0] > 0
        has_seizure = cur.execute(
            "SELECT COUNT(*) FROM seizure_diary WHERE patient_id=?", (pid,)
        ).fetchone()[0] > 0
        has_mri = cur.execute(
            "SELECT COUNT(*) FROM mri_findings WHERE patient_id=?", (pid,)
        ).fetchone()[0] > 0

        sections = []
        if has_analysis:
            sections.append("EEG Results")
        if has_assess:
            sections.append("Assessments")
        if has_seizure:
            sections.append("Seizure Log")
        if has_mri:
            sections.append("Brain Imaging")

        patient_data_map[pid] = {
            **p,
            "has_data": len(sections) > 0,
            "available_sections": sections,
            "section_count": len(sections),
        }

    patient_list = sorted(patient_data_map.values(),
                          key=lambda x: (-x["section_count"], x["patient_id"]))

    result = {
        "patient_list": patient_list,
        "total_reportable": sum(1 for p in patient_list if p["has_data"]),
    }

    # Generate full report for a specific patient
    if patient_id:
        report = _generate_patient_report(cur, patient_id)
        result["patient_report"] = report

    conn.close()
    return result


def _generate_patient_report(cur, patient_id):
    """Generate a complete patient-facing report in plain language."""
    # Patient info
    patient = cur.execute(
        "SELECT * FROM patients WHERE patient_id=?", (patient_id,)
    ).fetchone()
    if not patient:
        return {"error": f"Patient {patient_id} not found"}
    patient = dict(patient)

    report = {
        "patient_id": patient_id,
        "patient_name": patient.get("name", "Patient"),
        "generated_at": datetime.now().isoformat(),
        "disclaimer": (
            "This report is a simplified summary of your clinical data. "
            "It is NOT a diagnosis. Please discuss all results with your "
            "healthcare provider who can explain what they mean for you personally."
        ),
        "sections": [],
    }

    # Section 1: EEG Analysis Results
    analyses = [dict(r) for r in cur.execute(
        "SELECT * FROM analyses WHERE patient_id=? ORDER BY created_at DESC",
        (patient_id,)
    ).fetchall()]
    if analyses:
        latest = analyses[0]
        conf_cat = _confidence_category(latest.get("confidence"))
        section = {
            "title": "Your EEG Results",
            "icon": "brain",
            "summary": (
                f"Your most recent EEG recording was analyzed by our AI system. "
                f"The signal quality was rated as '{latest.get('signal_quality', 'N/A')}'. "
                f"{CONFIDENCE_PLAIN.get(conf_cat, '')}."
            ),
            "details": [
                {"label": "Recording Date", "value": (latest.get("created_at") or "N/A")[:10]},
                {"label": "Signal Quality", "value": latest.get("signal_quality", "N/A")},
                {"label": "AI Confidence", "value": f"{conf_cat.title()} ({_fmt(latest.get('confidence', 0) * 100, 0)}%)"},
                {"label": "Total EEG Analyses", "value": str(len(analyses))},
            ],
            "plain_note": (
                "An EEG (electroencephalogram) records the electrical activity in your brain. "
                "Our AI reviews the recording to help your doctor identify any unusual patterns."
            ),
        }
        report["sections"].append(section)

    # Section 2: Assessment Results
    assessments = [dict(r) for r in cur.execute(
        "SELECT * FROM assessments WHERE patient_id=? ORDER BY created_at DESC",
        (patient_id,)
    ).fetchall()]
    if assessments:
        assessment_summaries = []
        seen_instruments = set()
        for a in assessments:
            inst = a.get("instrument", "Unknown")
            if inst in seen_instruments:
                continue
            seen_instruments.add(inst)
            pct = _score_percent(a.get("score"), a.get("max_score"))
            level = a.get("level", "unknown")
            assessment_summaries.append({
                "test": INSTRUMENT_NAMES.get(inst, inst),
                "code": inst,
                "result": LEVEL_PLAIN.get(level, level),
                "level": level,
                "score_pct": pct,
                "date": (a.get("created_at") or "N/A")[:10],
            })

        normal_count = sum(1 for a in assessment_summaries if a["level"] in ("normal", "minimal"))
        elevated_count = len(assessment_summaries) - normal_count

        section = {
            "title": "Your Health Assessments",
            "icon": "clipboard",
            "summary": (
                f"You have completed {len(assessment_summaries)} different health assessments. "
                f"{normal_count} came back within normal range"
                + (f" and {elevated_count} showed areas worth discussing with your doctor." if elevated_count > 0
                   else ".")
            ),
            "assessments": assessment_summaries,
            "plain_note": (
                "These assessments measure different aspects of your health — mood, memory, "
                "sleep, daily functioning, and quality of life. Higher scores on some tests "
                "are better, while on others, lower is better. Your doctor can explain what "
                "each result means for you."
            ),
        }
        report["sections"].append(section)

    # Section 3: Seizure Diary
    seizures = [dict(r) for r in cur.execute(
        "SELECT * FROM seizure_diary WHERE patient_id=? ORDER BY event_date DESC",
        (patient_id,)
    ).fetchall()]
    if seizures:
        total = len(seizures)
        section = {
            "title": "Your Seizure Diary Summary",
            "icon": "activity",
            "summary": (
                f"Your seizure diary has {total} recorded event{'s' if total != 1 else ''}. "
                "Keeping a detailed diary helps your doctor understand your seizure patterns "
                "and adjust your treatment plan."
            ),
            "event_count": total,
            "recent_events": [
                {
                    "date": s.get("event_date", "N/A"),
                    "duration_sec": s.get("duration_sec", "N/A"),
                    "severity": s.get("severity", "N/A"),
                }
                for s in seizures[:5]
            ],
            "plain_note": (
                "Each seizure event is different. Your doctor uses this diary to track "
                "whether your seizures are changing in frequency or severity over time."
            ),
        }
        report["sections"].append(section)

    # Section 4: Brain Imaging (MRI)
    mri_rows = [dict(r) for r in cur.execute(
        "SELECT * FROM mri_findings WHERE patient_id=? ORDER BY created_at DESC",
        (patient_id,)
    ).fetchall()]
    if mri_rows:
        latest_mri = mri_rows[0]
        mri_fields = _safe_json(latest_mri.get("fields_json", "{}"))
        scan_date = (latest_mri.get("created_at") or "N/A")[:10]
        scan_quality = mri_fields.get("quality", "N/A")
        classification = mri_fields.get("classification_label", mri_fields.get("classification", "N/A"))
        section = {
            "title": "Your Brain Imaging Results",
            "icon": "image",
            "summary": (
                f"You have {len(mri_rows)} brain scan{'s' if len(mri_rows) != 1 else ''} on file. "
                "Your doctor reviews these images to look at the structure of your brain."
            ),
            "details": [
                {"label": "Most Recent Scan", "value": scan_date},
                {"label": "Image Quality", "value": scan_quality},
                {"label": "Total Scans", "value": str(len(mri_rows))},
            ],
            "plain_note": (
                "An MRI (magnetic resonance imaging) takes detailed pictures of your brain. "
                "It helps your doctor see if there are any structural changes that might be "
                "related to your condition. Only your doctor can interpret these images."
            ),
        }
        report["sections"].append(section)

    # Section 5: Medication Summary
    adherence = cur.execute(
        "SELECT AVG(CASE WHEN taken='yes' THEN 100.0 ELSE 0.0 END) as avg_adh, "
        "COUNT(*) as total_doses, SUM(CASE WHEN taken='yes' THEN 1 ELSE 0 END) as taken_doses "
        "FROM medication_adherence WHERE patient_id=?",
        (patient_id,)
    ).fetchone()
    if adherence and adherence["total_doses"] and adherence["total_doses"] > 0:
        avg_adh = round(adherence["avg_adh"], 1) if adherence["avg_adh"] else 0
        section = {
            "title": "Your Medication Summary",
            "icon": "pill",
            "summary": (
                f"Over your tracked period, your medication adherence rate is {avg_adh}%. "
                + ("Great job keeping up with your medications! " if avg_adh >= 90
                   else "Consistent medication use is important for seizure control. "
                   "If you're having trouble remembering doses, talk to your care team about strategies. "
                   if avg_adh >= 70
                   else "Your adherence is below target. Please discuss barriers with your care team — "
                   "they can help find solutions. ")
            ),
            "details": [
                {"label": "Adherence Rate", "value": f"{avg_adh}%"},
                {"label": "Doses Taken", "value": str(adherence["taken_doses"])},
                {"label": "Total Scheduled Doses", "value": str(adherence["total_doses"])},
            ],
            "plain_note": (
                "Taking your medications as prescribed is one of the most important things "
                "you can do to manage your condition. If side effects are making it hard, "
                "your doctor may be able to adjust your treatment."
            ),
        }
        report["sections"].append(section)

    # Closing
    report["closing"] = {
        "message": (
            "Thank you for being an active partner in your healthcare. "
            "If you have questions about any of these results, please bring this "
            "report to your next appointment to discuss with your doctor."
        ),
        "next_steps": [
            "Review this report and note any questions for your doctor",
            "Continue logging seizure events in your diary",
            "Take medications as prescribed",
            "Attend your scheduled follow-up appointments",
        ],
    }

    return report


# ── Definitions ─────────────────────────────────────────────────────

def definitions():
    """Metric definitions, glossary, health literacy notes."""
    return {
        "title": "Patient Report — Glossary & Definitions",
        "sections": [
            {
                "heading": "What This Report Includes",
                "items": [
                    {"term": "EEG Results", "definition": "A summary of your brain wave recording analysis, reviewed by AI and your doctor."},
                    {"term": "Health Assessments", "definition": "Questionnaires that measure your mood, memory, sleep, and quality of life."},
                    {"term": "Seizure Diary", "definition": "A log of seizure events you or your caregiver have recorded."},
                    {"term": "Brain Imaging", "definition": "Pictures of your brain taken with an MRI scanner."},
                    {"term": "Medication Summary", "definition": "How well you are keeping up with your prescribed medications."},
                ],
            },
            {
                "heading": "Understanding Assessment Levels",
                "items": [
                    {"term": "Within normal range", "definition": "Your score is in the expected range for the general population."},
                    {"term": "Mildly elevated", "definition": "Your score is slightly above the expected range. Worth monitoring."},
                    {"term": "Moderately elevated", "definition": "Your score suggests a concern that may benefit from treatment or follow-up."},
                    {"term": "Significantly elevated", "definition": "Your score indicates a concern that should be addressed with your care team."},
                ],
            },
            {
                "heading": "About AI-Assisted Analysis",
                "items": [
                    {"term": "AI Confidence", "definition": "How certain the computer system is about its analysis. Your doctor always reviews and confirms."},
                    {"term": "Signal Quality", "definition": "How clean and usable your EEG recording was. Better quality means more reliable results."},
                    {"term": "Not a Diagnosis", "definition": "This report summarizes data but does NOT replace a doctor's evaluation or diagnosis."},
                ],
            },
            {
                "heading": "Your Rights",
                "items": [
                    {"term": "Access", "definition": "You have the right to access all your health information."},
                    {"term": "Questions", "definition": "You can ask your care team to explain any result in more detail."},
                    {"term": "Second Opinion", "definition": "You may request a second opinion from another healthcare provider at any time."},
                    {"term": "Privacy", "definition": "Your health data is protected and only shared with your authorized care team."},
                ],
            },
        ],
        "health_literacy_note": (
            "This report is written in plain language to make your health information "
            "easier to understand. If anything is unclear, please ask your care team."
        ),
    }
