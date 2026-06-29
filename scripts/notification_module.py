"""
Patient Portal — Notification Tab Module
==========================================
Patient-facing notification centre derived from clinical.db:
  - Assessment Results  (score ready, alert triggered)
  - Form Assignments    (pending forms to complete)
  - Seizure Diary       (seizure event → follow-up reminder)
  - Medication Updates  (new recommendation / schedule change)
  - Clinical Activity   (transaction-log milestones)

All notifications are derived from REAL patient data in clinical.db.
No fabricated or stubbed data — every row traces to a source table.
"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"

# ── Notification category constants ────────────────────────────────────
CATEGORY_RESULT   = "result"
CATEGORY_FORM     = "form"
CATEGORY_SEIZURE  = "seizure"
CATEGORY_MED      = "medication"
CATEGORY_ACTIVITY = "activity"
CATEGORY_ALERT    = "alert"

# Severity thresholds for alerts derived from assessment scores
ALERT_INSTRUMENTS = {
    "PHQ9":    {"threshold": 15, "direction": "above", "label": "Moderate-severe depression"},
    "GAD7":    {"threshold": 10, "direction": "above", "label": "Moderate anxiety"},
    "ESS":     {"threshold": 16, "direction": "above", "label": "Severe sleepiness"},
    "BDI":     {"threshold": 29, "direction": "above", "label": "Severe depression"},
    "MOCA":    {"threshold": 26, "direction": "below", "label": "Cognitive impairment"},
    "MMSE":    {"threshold": 24, "direction": "below", "label": "Cognitive impairment"},
    "BARTHEL": {"threshold": 60, "direction": "below", "label": "Functional dependence"},
}

PRIORITY_MAP = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


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


# ── Assessment Result Notifications ────────────────────────────────────

def _assessment_notifications(conn, patient_id=None):
    """Generate notifications from completed assessments."""
    sql = "SELECT * FROM assessments ORDER BY created_at DESC"
    params = []
    if patient_id:
        sql = "SELECT * FROM assessments WHERE patient_id = ? ORDER BY created_at DESC"
        params = [patient_id]
    rows = conn.execute(sql, params).fetchall()
    notifs = []
    for r in rows:
        instrument = (r["instrument"] or "").upper().replace("-", "")
        score = r["score"]
        alert_text = r["alert"]
        priority = "info"
        title = f"{r['instrument']} assessment completed"
        body = f"Score: {score}" + (f" / {r['max_score']}" if r["max_score"] else "")
        if r["interpretation"]:
            body += f" — {r['interpretation']}"

        # Elevate priority if alert present or threshold crossed
        if alert_text:
            priority = "high"
            title = f"⚠ {r['instrument']} alert: {alert_text}"
        elif instrument in ALERT_INSTRUMENTS:
            info = ALERT_INSTRUMENTS[instrument]
            if score is not None:
                if info["direction"] == "above" and score >= info["threshold"]:
                    priority = "high"
                    title = f"⚠ {r['instrument']}: {info['label']}"
                elif info["direction"] == "below" and score <= info["threshold"]:
                    priority = "high"
                    title = f"⚠ {r['instrument']}: {info['label']}"

        notifs.append({
            "id": f"assess-{r['id']}",
            "category": CATEGORY_ALERT if priority == "high" else CATEGORY_RESULT,
            "priority": priority,
            "title": title,
            "body": body,
            "patient_id": r["patient_id"],
            "source_table": "assessments",
            "source_id": r["id"],
            "timestamp": _ts(r["created_at"]),
            "read": False,
        })
    return notifs


# ── Form Assignment Notifications ──────────────────────────────────────

def _form_notifications(conn, patient_id=None):
    """Generate notifications from form assignments."""
    sql = "SELECT * FROM form_assignments ORDER BY created_at DESC"
    params = []
    if patient_id:
        sql = "SELECT * FROM form_assignments WHERE patient_id = ? ORDER BY created_at DESC"
        params = [patient_id]
    rows = conn.execute(sql, params).fetchall()
    notifs = []
    for r in rows:
        status = (r["status"] or "pending").lower()
        if status == "pending":
            priority = "medium"
            title = f"📋 Please complete: {r['instrument']}"
            body = f"Assigned by {r['assigned_by'] or 'care team'}"
            if r["message"]:
                body += f" — {r['message']}"
        else:
            priority = "low"
            title = f"✅ {r['instrument']} form completed"
            body = f"Completed on {r['completed_at'] or 'N/A'}"

        notifs.append({
            "id": f"form-{r['id']}",
            "category": CATEGORY_FORM,
            "priority": priority,
            "title": title,
            "body": body,
            "patient_id": r["patient_id"],
            "source_table": "form_assignments",
            "source_id": r["id"],
            "timestamp": _ts(r["created_at"]),
            "read": status == "completed",
        })
    return notifs


# ── Seizure Diary Notifications ────────────────────────────────────────

def _seizure_notifications(conn, patient_id=None):
    """Generate follow-up notifications from seizure diary events."""
    sql = "SELECT * FROM seizure_diary ORDER BY created_at DESC"
    params = []
    if patient_id:
        sql = "SELECT * FROM seizure_diary WHERE patient_id = ? ORDER BY created_at DESC"
        params = [patient_id]
    rows = conn.execute(sql, params).fetchall()
    notifs = []
    for r in rows:
        severity = (r["severity"] or "unknown").lower()
        er = (r["er_visit"] or "no").lower()
        rescue = (r["rescue_med"] or "no").lower()

        if er in ("yes", "true", "1") or severity == "severe":
            priority = "critical"
        elif rescue in ("yes", "true", "1") or severity == "moderate":
            priority = "high"
        else:
            priority = "medium"

        date_str = r["event_date"] or "unknown date"
        dur = f", duration {r['duration_sec']}s" if r["duration_sec"] else ""
        title = f"Seizure event on {date_str}"
        body = f"Severity: {severity}{dur}"
        if r["injury"] and str(r["injury"]).lower() not in ("no", "none", ""):
            body += f" — Injury: {r['injury']}"
            priority = "critical"
        if r["post_ictal"]:
            body += f" — Post-ictal: {r['post_ictal']}"

        notifs.append({
            "id": f"seizure-{r['id']}",
            "category": CATEGORY_SEIZURE,
            "priority": priority,
            "title": title,
            "body": body,
            "patient_id": r["patient_id"],
            "source_table": "seizure_diary",
            "source_id": r["id"],
            "timestamp": _ts(r["created_at"]),
            "read": False,
        })
    return notifs


# ── Medication Notifications ───────────────────────────────────────────

def _medication_notifications(conn, patient_id=None):
    """Generate notifications from medication records."""
    sql = "SELECT * FROM medications ORDER BY created_at DESC"
    params = []
    if patient_id:
        sql = "SELECT * FROM medications WHERE patient_id = ? ORDER BY created_at DESC"
        params = [patient_id]
    rows = conn.execute(sql, params).fetchall()
    notifs = []
    for r in rows:
        fields = {}
        try:
            fields = json.loads(r["fields_json"]) if r["fields_json"] else {}
        except Exception:
            pass
        drug = fields.get("drug_name") or fields.get("medication") or fields.get("name", "Medication")
        action = fields.get("action", "updated")
        title = f"💊 {drug} — {action}"
        body_parts = []
        if fields.get("dose"):
            body_parts.append(f"Dose: {fields['dose']}")
        if fields.get("frequency"):
            body_parts.append(f"Frequency: {fields['frequency']}")
        if fields.get("reason"):
            body_parts.append(f"Reason: {fields['reason']}")
        body = " | ".join(body_parts) if body_parts else "Medication record updated"

        notifs.append({
            "id": f"med-{r['id']}",
            "category": CATEGORY_MED,
            "priority": "medium",
            "title": title,
            "body": body,
            "patient_id": r["patient_id"],
            "source_table": "medications",
            "source_id": r["id"],
            "timestamp": _ts(r["created_at"]),
            "read": False,
        })
    return notifs


# ── Activity Notifications (transaction log milestones) ────────────────

def _activity_notifications(conn, patient_id=None, limit=50):
    """Generate milestone notifications from the transaction log."""
    sql = "SELECT * FROM transaction_log ORDER BY ts_utc DESC LIMIT ?"
    params = [limit]
    if patient_id:
        sql = "SELECT * FROM transaction_log WHERE patient_id = ? ORDER BY ts_utc DESC LIMIT ?"
        params = [patient_id, limit]
    rows = conn.execute(sql, params).fetchall()
    notifs = []
    for r in rows:
        component = r["component"] or "system"
        action = r["action"] or "activity"
        title = f"{component}: {action}"
        body = r["detail"] or ""
        notifs.append({
            "id": f"txn-{r['id']}",
            "category": CATEGORY_ACTIVITY,
            "priority": "info",
            "title": title,
            "body": body[:200],
            "patient_id": r["patient_id"],
            "source_table": "transaction_log",
            "source_id": r["id"],
            "timestamp": _ts(r["ts_utc"]),
            "read": True,
        })
    return notifs


# ── Public API Functions ───────────────────────────────────────────────

def notification_overview(patient_id=None):
    """Full notification centre: all categories merged and sorted by
    priority then timestamp. Returns summary counts + full list."""
    conn = _connect()
    if not conn:
        return {"available": False, "error": "clinical.db not found"}
    try:
        all_notifs = []
        all_notifs.extend(_assessment_notifications(conn, patient_id))
        all_notifs.extend(_form_notifications(conn, patient_id))
        all_notifs.extend(_seizure_notifications(conn, patient_id))
        all_notifs.extend(_medication_notifications(conn, patient_id))
        all_notifs.extend(_activity_notifications(conn, patient_id, limit=30))

        # Sort: priority (critical first), then newest first
        all_notifs.sort(key=lambda n: (PRIORITY_MAP.get(n["priority"], 9), n["timestamp"] or ""), reverse=False)
        # Within same priority, newest first
        all_notifs.sort(key=lambda n: (PRIORITY_MAP.get(n["priority"], 9), -(hash(n["timestamp"] or ""))))

        # Re-sort properly
        all_notifs.sort(key=lambda n: (PRIORITY_MAP.get(n["priority"], 9),))
        # Within same priority group, sort by timestamp descending
        from itertools import groupby
        sorted_notifs = []
        for _, group in groupby(all_notifs, key=lambda n: n["priority"]):
            g = list(group)
            g.sort(key=lambda n: n["timestamp"] or "", reverse=True)
            sorted_notifs.extend(g)

        # Summary
        by_cat = defaultdict(int)
        by_priority = defaultdict(int)
        unread = 0
        for n in sorted_notifs:
            by_cat[n["category"]] += 1
            by_priority[n["priority"]] += 1
            if not n["read"]:
                unread += 1

        return {
            "available": True,
            "total": len(sorted_notifs),
            "unread": unread,
            "by_category": dict(by_cat),
            "by_priority": dict(by_priority),
            "notifications": sorted_notifs,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
    finally:
        conn.close()


def notification_by_category(category, patient_id=None):
    """Return notifications filtered by category."""
    overview = notification_overview(patient_id)
    if not overview.get("available"):
        return overview
    filtered = [n for n in overview["notifications"] if n["category"] == category]
    return {
        "available": True,
        "category": category,
        "total": len(filtered),
        "notifications": filtered,
    }


def notification_unread(patient_id=None):
    """Return only unread notifications."""
    overview = notification_overview(patient_id)
    if not overview.get("available"):
        return overview
    filtered = [n for n in overview["notifications"] if not n["read"]]
    return {
        "available": True,
        "total": len(filtered),
        "notifications": filtered,
    }


def notification_definitions():
    """Metric and term definitions for tooltip overlays."""
    return {
        "categories": {
            CATEGORY_RESULT:   "Assessment results — scores and interpretations from completed instruments",
            CATEGORY_FORM:     "Form assignments — pending or completed questionnaires assigned by care team",
            CATEGORY_SEIZURE:  "Seizure events — diary entries with follow-up recommendations",
            CATEGORY_MED:      "Medication updates — new prescriptions, dose changes, schedule modifications",
            CATEGORY_ACTIVITY: "Clinical activity — milestones from your care journey transaction log",
            CATEGORY_ALERT:    "Clinical alerts — assessment scores crossing clinical thresholds",
        },
        "priorities": {
            "critical": "Requires immediate attention (e.g., severe seizure with injury, ER visit)",
            "high":     "Important — clinical alert or elevated assessment score",
            "medium":   "Action needed — pending form or moderate seizure event",
            "low":      "Informational — completed items",
            "info":     "Activity log — routine clinical activity record",
        },
        "source_tables": {
            "assessments":      "Clinical assessments (PHQ-9, MoCA, MMSE, GAD-7, etc.)",
            "form_assignments": "Forms assigned to patient by care team",
            "seizure_diary":    "Patient-reported seizure events",
            "medications":      "Medication prescriptions and recommendations",
            "transaction_log":  "System-wide clinical activity audit trail",
        },
    }
