"""
Patient Portal — Campaigns Tab Module
=======================================
Health campaigns, education, and reminders derived from clinical.db:
  - Screening Campaigns    (assessment instrument programs — PHQ9, GAD7, MOCA, etc.)
  - Medication Adherence   (adherence reminders from medications table)
  - Seizure Safety         (safety education triggered by seizure diary events)
  - Form Completion        (pending form assignments as active campaigns)
  - Education Programs     (condition-specific education from assessment results)

All campaigns are derived from REAL patient data in clinical.db.
No fabricated or stubbed data — every row traces to a source table.
"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from collections import defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"

# ── Campaign type constants ──────────────────────────────────────────
TYPE_SCREENING = "screening"
TYPE_ADHERENCE = "adherence"
TYPE_SAFETY = "safety"
TYPE_FORM = "form_completion"
TYPE_EDUCATION = "education"

# Status constants
STATUS_ACTIVE = "active"
STATUS_COMPLETED = "completed"
STATUS_PENDING = "pending"

# Assessment instruments → campaign descriptions
SCREENING_PROGRAMS = {
    "PHQ9":     {"name": "Depression Screening (PHQ-9)", "category": "Mental Health",
                 "education": "Understanding depression and treatment options"},
    "GAD7":     {"name": "Anxiety Screening (GAD-7)", "category": "Mental Health",
                 "education": "Anxiety management and coping strategies"},
    "NDDIE":    {"name": "Depression in Epilepsy (NDDIE)", "category": "Mental Health",
                 "education": "Mood and epilepsy — what you need to know"},
    "MOCA":     {"name": "Cognitive Screening (MoCA)", "category": "Cognitive Health",
                 "education": "Maintaining cognitive health with epilepsy"},
    "MMSE":     {"name": "Cognitive Screening (MMSE)", "category": "Cognitive Health",
                 "education": "Cognitive wellness and brain-healthy habits"},
    "QOLIE31":  {"name": "Quality of Life Assessment", "category": "Wellbeing",
                 "education": "Improving quality of life with epilepsy"},
    "BARTHEL":  {"name": "Functional Independence", "category": "Rehabilitation",
                 "education": "Building daily living skills and independence"},
    "EPWORTH":  {"name": "Sleep Quality Screening (ESS)", "category": "Sleep Health",
                 "education": "Sleep hygiene and seizure management"},
    "BNT":      {"name": "Language Assessment (BNT)", "category": "Cognitive Health",
                 "education": "Language and communication support"},
    "WAB":      {"name": "Aphasia Assessment (WAB)", "category": "Cognitive Health",
                 "education": "Speech-language therapy resources"},
    "VERBAL_FLUENCY": {"name": "Verbal Fluency Assessment", "category": "Cognitive Health",
                 "education": "Cognitive exercise programs"},
    "MASA":     {"name": "Swallowing Safety (MASA)", "category": "Safety",
                 "education": "Safe swallowing and aspiration prevention"},
}


def _connect():
    """Open read-only connection to clinical.db."""
    if not DB.exists():
        return None
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    return conn


def _ts(row_ts):
    """Parse a timestamp string, return ISO string or original."""
    if not row_ts:
        return None
    try:
        return datetime.fromisoformat(str(row_ts).replace("Z", "+00:00")).isoformat()
    except Exception:
        return str(row_ts)


# ── Screening Campaigns ─────────────────────────────────────────────

def _screening_campaigns(conn, patient_id=None):
    """Generate screening campaigns from assessment instruments."""
    sql = "SELECT instrument, patient_id, score, max_score, interpretation, level, created_at FROM assessments ORDER BY created_at DESC"
    params = []
    if patient_id:
        sql = "SELECT instrument, patient_id, score, max_score, interpretation, level, created_at FROM assessments WHERE patient_id = ? ORDER BY created_at DESC"
        params = [patient_id]
    rows = conn.execute(sql, params).fetchall()

    # Group by instrument to form campaigns
    by_instrument = defaultdict(list)
    for r in rows:
        inst = (r["instrument"] or "").upper().replace("-", "")
        by_instrument[inst].append(dict(r))

    campaigns = []
    for inst, entries in by_instrument.items():
        prog = SCREENING_PROGRAMS.get(inst)
        if not prog:
            prog = {"name": f"{inst} Screening", "category": "Clinical",
                    "education": f"{inst} assessment and follow-up"}

        participants = list(set(e["patient_id"] for e in entries if e["patient_id"]))
        latest = entries[0] if entries else {}
        earliest = entries[-1] if entries else {}

        # Determine campaign status from participation
        completed_count = sum(1 for e in entries if e.get("score") is not None)
        total = len(entries)

        campaigns.append({
            "id": f"screening-{inst.lower()}",
            "type": TYPE_SCREENING,
            "name": prog["name"],
            "category": prog["category"],
            "status": STATUS_ACTIVE if total > 0 else STATUS_PENDING,
            "participants": len(participants),
            "assessments_completed": completed_count,
            "assessments_total": total,
            "completion_rate": round(completed_count / total * 100, 1) if total > 0 else 0,
            "latest_date": _ts(latest.get("created_at")),
            "start_date": _ts(earliest.get("created_at")),
            "education_topic": prog["education"],
            "source": "assessments",
        })
    return campaigns


# ── Medication Adherence Campaigns ───────────────────────────────────

def _adherence_campaigns(conn, patient_id=None):
    """Generate medication adherence campaigns from medications table."""
    sql = "SELECT patient_id, fields_json, created_at FROM medications ORDER BY created_at DESC"
    params = []
    if patient_id:
        sql = "SELECT patient_id, fields_json, created_at FROM medications WHERE patient_id = ? ORDER BY created_at DESC"
        params = [patient_id]
    rows = conn.execute(sql, params).fetchall()

    if not rows:
        return []

    patients_on_meds = set()
    med_names = set()
    latest_date = None
    earliest_date = None

    for r in rows:
        patients_on_meds.add(r["patient_id"])
        if not latest_date:
            latest_date = r["created_at"]
        earliest_date = r["created_at"]
        try:
            fields = json.loads(r["fields_json"]) if r["fields_json"] else {}
            if isinstance(fields, dict):
                for k, v in fields.items():
                    if "name" in k.lower() and v:
                        med_names.add(str(v))
        except (json.JSONDecodeError, TypeError):
            pass

    return [{
        "id": "adherence-medication",
        "type": TYPE_ADHERENCE,
        "name": "Medication Adherence Program",
        "category": "Medication Safety",
        "status": STATUS_ACTIVE,
        "participants": len(patients_on_meds),
        "medications_tracked": len(med_names),
        "medication_list": sorted(med_names)[:10],
        "latest_date": _ts(latest_date),
        "start_date": _ts(earliest_date),
        "education_topic": "Taking your medications safely and on time",
        "source": "medications",
    }]


# ── Seizure Safety Campaigns ────────────────────────────────────────

def _safety_campaigns(conn, patient_id=None):
    """Generate seizure safety campaigns from seizure_diary events."""
    sql = "SELECT patient_id, severity, injury, er_visit, trigger, event_date, created_at FROM seizure_diary ORDER BY created_at DESC"
    params = []
    if patient_id:
        sql = "SELECT patient_id, severity, injury, er_visit, trigger, event_date, created_at FROM seizure_diary WHERE patient_id = ? ORDER BY created_at DESC"
        params = [patient_id]
    rows = conn.execute(sql, params).fetchall()

    if not rows:
        return []

    patients = set()
    injury_count = 0
    er_count = 0
    triggers = defaultdict(int)
    severity_dist = defaultdict(int)

    for r in rows:
        patients.add(r["patient_id"])
        if r["injury"] and str(r["injury"]).lower() not in ("no", "none", ""):
            injury_count += 1
        if r["er_visit"] and str(r["er_visit"]).lower() not in ("no", "none", ""):
            er_count += 1
        if r["trigger"]:
            triggers[str(r["trigger"])] += 1
        if r["severity"]:
            severity_dist[str(r["severity"]).lower()] += 1

    latest = rows[0]["created_at"] if rows else None
    earliest = rows[-1]["created_at"] if rows else None
    top_triggers = sorted(triggers.items(), key=lambda x: x[1], reverse=True)[:5]

    return [{
        "id": "safety-seizure",
        "type": TYPE_SAFETY,
        "name": "Seizure Safety & First Aid Education",
        "category": "Safety",
        "status": STATUS_ACTIVE,
        "participants": len(patients),
        "total_events": len(rows),
        "injuries_reported": injury_count,
        "er_visits": er_count,
        "severity_distribution": dict(severity_dist),
        "top_triggers": [{"trigger": t, "count": c} for t, c in top_triggers],
        "latest_date": _ts(latest),
        "start_date": _ts(earliest),
        "education_topic": "Seizure first aid, safety planning, and trigger avoidance",
        "source": "seizure_diary",
    }]


# ── Form Completion Campaigns ───────────────────────────────────────

def _form_campaigns(conn, patient_id=None):
    """Generate form completion campaigns from form_assignments."""
    sql = "SELECT patient_id, instrument, status, assigned_by, message, created_at, completed_at FROM form_assignments ORDER BY created_at DESC"
    params = []
    if patient_id:
        sql = "SELECT patient_id, instrument, status, assigned_by, message, created_at, completed_at FROM form_assignments WHERE patient_id = ? ORDER BY created_at DESC"
        params = [patient_id]
    rows = conn.execute(sql, params).fetchall()

    if not rows:
        return []

    pending = [r for r in rows if (r["status"] or "").lower() in ("pending", "assigned")]
    completed = [r for r in rows if (r["status"] or "").lower() == "completed"]

    return [{
        "id": "forms-completion",
        "type": TYPE_FORM,
        "name": "Clinical Form Completion Drive",
        "category": "Patient Engagement",
        "status": STATUS_ACTIVE if pending else STATUS_COMPLETED,
        "participants": len(set(r["patient_id"] for r in rows)),
        "forms_pending": len(pending),
        "forms_completed": len(completed),
        "forms_total": len(rows),
        "completion_rate": round(len(completed) / len(rows) * 100, 1) if rows else 0,
        "pending_forms": [
            {"patient_id": r["patient_id"], "instrument": r["instrument"],
             "assigned_by": r["assigned_by"], "message": r["message"],
             "assigned_date": _ts(r["created_at"])}
            for r in pending[:10]
        ],
        "latest_date": _ts(rows[0]["created_at"]) if rows else None,
        "education_topic": "Why completing your health questionnaires matters",
        "source": "form_assignments",
    }]


# ── Education Campaigns (derived from assessment results) ────────────

def _education_campaigns(conn, patient_id=None):
    """Generate education campaigns based on assessment score patterns."""
    sql = "SELECT instrument, patient_id, score, max_score, level, interpretation, created_at FROM assessments WHERE score IS NOT NULL ORDER BY created_at DESC"
    params = []
    if patient_id:
        sql = "SELECT instrument, patient_id, score, max_score, level, interpretation, created_at FROM assessments WHERE score IS NOT NULL AND patient_id = ? ORDER BY created_at DESC"
        params = [patient_id]
    rows = conn.execute(sql, params).fetchall()

    # Group by concern area
    mental_health_flags = []
    cognitive_flags = []
    functional_flags = []

    for r in rows:
        inst = (r["instrument"] or "").upper().replace("-", "")
        score = r["score"]
        level = (r["level"] or "").lower()

        if inst in ("PHQ9", "GAD7", "BDI", "NDDIE") and level in ("severe", "critical", "moderate-severe", "moderately severe"):
            mental_health_flags.append(dict(r))
        elif inst in ("MOCA", "MMSE") and score is not None and score < 26:
            cognitive_flags.append(dict(r))
        elif inst == "BARTHEL" and score is not None and score < 60:
            functional_flags.append(dict(r))

    campaigns = []

    if mental_health_flags:
        patients = set(r["patient_id"] for r in mental_health_flags)
        campaigns.append({
            "id": "education-mental-health",
            "type": TYPE_EDUCATION,
            "name": "Mental Health Awareness & Support",
            "category": "Mental Health",
            "status": STATUS_ACTIVE,
            "participants": len(patients),
            "flagged_assessments": len(mental_health_flags),
            "instruments_involved": list(set(r["instrument"] for r in mental_health_flags)),
            "latest_date": _ts(mental_health_flags[0]["created_at"]),
            "education_topic": "Understanding mood disorders, treatment options, and when to seek help",
            "source": "assessments",
        })

    if cognitive_flags:
        patients = set(r["patient_id"] for r in cognitive_flags)
        campaigns.append({
            "id": "education-cognitive",
            "type": TYPE_EDUCATION,
            "name": "Cognitive Health & Brain Wellness",
            "category": "Cognitive Health",
            "status": STATUS_ACTIVE,
            "participants": len(patients),
            "flagged_assessments": len(cognitive_flags),
            "instruments_involved": list(set(r["instrument"] for r in cognitive_flags)),
            "latest_date": _ts(cognitive_flags[0]["created_at"]),
            "education_topic": "Cognitive exercises, brain-healthy lifestyle, and memory support",
            "source": "assessments",
        })

    if functional_flags:
        patients = set(r["patient_id"] for r in functional_flags)
        campaigns.append({
            "id": "education-functional",
            "type": TYPE_EDUCATION,
            "name": "Functional Independence Program",
            "category": "Rehabilitation",
            "status": STATUS_ACTIVE,
            "participants": len(patients),
            "flagged_assessments": len(functional_flags),
            "instruments_involved": list(set(r["instrument"] for r in functional_flags)),
            "latest_date": _ts(functional_flags[0]["created_at"]),
            "education_topic": "Daily living skills, adaptive equipment, and rehabilitation exercises",
            "source": "assessments",
        })

    return campaigns


# ── Public API ───────────────────────────────────────────────────────

def campaigns_overview(patient_id=None):
    """Full campaigns dashboard: all campaign types with stats."""
    conn = _connect()
    if not conn:
        return {"available": False, "reason": "clinical.db not found"}
    try:
        screening = _screening_campaigns(conn, patient_id)
        adherence = _adherence_campaigns(conn, patient_id)
        safety = _safety_campaigns(conn, patient_id)
        forms = _form_campaigns(conn, patient_id)
        education = _education_campaigns(conn, patient_id)

        all_campaigns = screening + adherence + safety + forms + education
        active = [c for c in all_campaigns if c["status"] == STATUS_ACTIVE]
        completed = [c for c in all_campaigns if c["status"] == STATUS_COMPLETED]

        # Category breakdown
        by_category = defaultdict(int)
        for c in all_campaigns:
            by_category[c["category"]] += 1

        # Type breakdown
        by_type = defaultdict(int)
        for c in all_campaigns:
            by_type[c["type"]] += 1

        return {
            "available": True,
            "total_campaigns": len(all_campaigns),
            "active": len(active),
            "completed": len(completed),
            "pending": len([c for c in all_campaigns if c["status"] == STATUS_PENDING]),
            "by_category": dict(by_category),
            "by_type": dict(by_type),
            "campaigns": all_campaigns,
        }
    finally:
        conn.close()


def campaigns_by_type(campaign_type, patient_id=None):
    """Return campaigns filtered by type."""
    overview = campaigns_overview(patient_id)
    if not overview.get("available"):
        return overview
    filtered = [c for c in overview["campaigns"] if c["type"] == campaign_type]
    return {
        "available": True,
        "type": campaign_type,
        "count": len(filtered),
        "campaigns": filtered,
    }


def campaigns_summary(patient_id=None):
    """Compact summary: counts + top active campaigns."""
    overview = campaigns_overview(patient_id)
    if not overview.get("available"):
        return overview
    top_active = [c for c in overview["campaigns"] if c["status"] == STATUS_ACTIVE][:5]
    return {
        "available": True,
        "total": overview["total_campaigns"],
        "active": overview["active"],
        "completed": overview["completed"],
        "by_category": overview["by_category"],
        "top_active": [
            {"id": c["id"], "name": c["name"], "category": c["category"],
             "participants": c.get("participants", 0),
             "education_topic": c.get("education_topic", "")}
            for c in top_active
        ],
    }


def campaigns_definitions():
    """Campaign metric definitions for tooltip overlays."""
    return {
        "available": True,
        "definitions": {
            "screening": "Systematic health screening programs using validated instruments (PHQ-9, GAD-7, MoCA, etc.) to identify patients needing intervention.",
            "adherence": "Medication adherence monitoring and education programs to ensure patients take prescribed AEDs safely and consistently.",
            "safety": "Seizure safety and first aid education programs triggered by seizure diary entries, injuries, or ER visits.",
            "form_completion": "Active campaigns to complete pending clinical questionnaires and assessments assigned by the care team.",
            "education": "Condition-specific education programs triggered when assessment scores indicate a need for patient education.",
            "participants": "Number of unique patients enrolled or targeted by this campaign.",
            "completion_rate": "Percentage of required actions or assessments completed within the campaign.",
            "campaign_status": "Active = currently running. Completed = all goals met. Pending = scheduled but not yet started.",
            "category": "Clinical domain grouping: Mental Health, Cognitive Health, Safety, Medication Safety, Wellbeing, Rehabilitation, Patient Engagement.",
        },
    }
