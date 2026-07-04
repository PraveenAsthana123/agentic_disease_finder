"""Seizure Diary Dashboard — event logging, trend analysis, severity tracking.

All data from REAL seizure_diary table in data/clinical.db.
Columns: id, patient_id, event_date, event_time, duration_sec, location,
         witnessed, aura, awareness, motor_signs, injury, post_ictal,
         recovery_min, er_visit, rescue_med, severity, trigger, notes, created_at.
"""
import sqlite3
from pathlib import Path
from collections import Counter, defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def _severity_score(sev):
    return {"Mild": 1, "Moderate": 2, "Severe": 3, "Critical": 4}.get(sev, 0)


def overview():
    """KPIs + severity distribution + monthly trend + ER summary."""
    rows = _rows("SELECT * FROM seizure_diary ORDER BY event_date DESC")
    if not rows:
        return {"total_events": 0, "patients": 0, "message": "No seizure diary data yet"}

    patients = set()
    severity_dist = Counter()
    monthly = defaultdict(int)
    monthly_er = defaultdict(int)
    durations = []
    er_count = 0
    injury_count = 0
    triggers = Counter()

    for r in rows:
        patients.add(r["patient_id"])
        sev = r["severity"] or "Unknown"
        severity_dist[sev] += 1
        month = (r["event_date"] or "")[:7]
        if month:
            monthly[month] += 1
        if r["duration_sec"] is not None:
            durations.append(r["duration_sec"])
        if r["er_visit"] == "Yes":
            er_count += 1
            if month:
                monthly_er[month] += 1
        if r["injury"] and r["injury"] != "No":
            injury_count += 1
        if r["trigger"]:
            triggers[r["trigger"]] += 1

    avg_dur = round(sum(durations) / len(durations), 1) if durations else 0
    monthly_trend = [{"month": m, "events": monthly[m], "er_visits": monthly_er.get(m, 0)}
                     for m in sorted(monthly.keys())]

    return {
        "total_events": len(rows),
        "unique_patients": len(patients),
        "avg_duration_sec": avg_dur,
        "er_visits": er_count,
        "er_rate_pct": round(100 * er_count / len(rows), 1) if rows else 0,
        "injury_count": injury_count,
        "severity_distribution": dict(severity_dist),
        "monthly_trend": monthly_trend,
        "top_triggers": dict(triggers.most_common(10)),
    }


def breakdown():
    """Per-patient history, event log, trigger analysis, severity timeline."""
    rows = _rows("SELECT * FROM seizure_diary ORDER BY event_date DESC")
    if not rows:
        return {"patients": [], "events": [], "message": "No seizure diary data yet"}

    per_patient = defaultdict(list)
    for r in rows:
        per_patient[r["patient_id"]].append(r)

    patient_summary = []
    for pid in sorted(per_patient.keys()):
        evts = per_patient[pid]
        sev_counts = Counter(e["severity"] for e in evts)
        durs = [e["duration_sec"] for e in evts if e["duration_sec"] is not None]
        er = sum(1 for e in evts if e["er_visit"] == "Yes")
        injuries = sum(1 for e in evts if e["injury"] and e["injury"] != "No")
        patient_summary.append({
            "patient_id": pid,
            "total_events": len(evts),
            "avg_duration_sec": round(sum(durs) / len(durs), 1) if durs else 0,
            "er_visits": er,
            "injuries": injuries,
            "severity_counts": dict(sev_counts),
            "last_event": evts[0]["event_date"],
            "worst_severity": max(evts, key=lambda e: _severity_score(e.get("severity", "")))["severity"],
        })

    # Sort by severity (worst first), then by total events
    patient_summary.sort(key=lambda p: (-_severity_score(p["worst_severity"]), -p["total_events"]))

    event_log = []
    for r in rows[:50]:
        event_log.append({
            "id": r["id"],
            "patient_id": r["patient_id"],
            "event_date": r["event_date"],
            "event_time": r["event_time"],
            "duration_sec": r["duration_sec"],
            "severity": r["severity"],
            "injury": r["injury"],
            "er_visit": r["er_visit"],
            "trigger": r["trigger"],
            "recovery_min": r["recovery_min"],
            "rescue_med": r["rescue_med"],
        })

    trigger_by_severity = defaultdict(lambda: Counter())
    for r in rows:
        if r["trigger"]:
            trigger_by_severity[r["trigger"]][r["severity"] or "Unknown"] += 1
    trigger_analysis = [
        {"trigger": t, "total": sum(sev.values()), "severity": dict(sev)}
        for t, sev in sorted(trigger_by_severity.items(), key=lambda x: -sum(x[1].values()))
    ]

    return {
        "patient_summary": patient_summary,
        "event_log": event_log,
        "trigger_analysis": trigger_analysis,
    }


def definitions():
    """Metric definitions for the Seizure Diary dashboard."""
    return {
        "title": "Seizure Diary Dashboard",
        "description": "Patient-reported seizure event log with severity auto-scoring, "
                       "monthly trend aggregation, trigger analysis, and ER tracking.",
        "metrics": [
            {"name": "Total Events", "description": "Total seizure events recorded across all patients."},
            {"name": "Unique Patients", "description": "Number of distinct patients with diary entries."},
            {"name": "Avg Duration (sec)", "description": "Mean seizure duration in seconds across all events."},
            {"name": "ER Visits", "description": "Number of events resulting in an emergency room visit."},
            {"name": "ER Rate %", "description": "Percentage of events with ER visit out of total events."},
            {"name": "Injury Count", "description": "Events where injury was reported (falls, tongue bites, etc.)."},
            {"name": "Severity Distribution", "description": "Count of events by severity level (Mild/Moderate/Severe/Critical)."},
            {"name": "Monthly Trend", "description": "Events and ER visits aggregated per calendar month."},
            {"name": "Top Triggers", "description": "Most frequently reported seizure triggers."},
        ],
        "severity_levels": [
            {"level": "Mild", "description": "Brief seizure (<60s), no injury, no ER, quick recovery."},
            {"level": "Moderate", "description": "Moderate duration (60-180s), possible confusion, no major injury."},
            {"level": "Severe", "description": "Prolonged seizure (>180s), injury, ER visit, or prolonged recovery."},
            {"level": "Critical", "description": "Status epilepticus, ICU admission, or life-threatening event."},
        ],
        "data_source": "seizure_diary table in data/clinical.db — patient-reported events.",
    }
