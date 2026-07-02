"""Continuous Monitoring Dashboard — seizure diary analytics for ongoing patient monitoring.

All data from REAL seizure_diary table in data/clinical.db (25 seizure events, 22 patients).
Tracks seizure frequency, severity, triggers, injuries, ER visits, duration, and recovery.
"""
import sqlite3, json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def overview():
    """Summary KPIs for continuous seizure monitoring."""
    rows = _rows("SELECT * FROM seizure_diary ORDER BY event_date DESC")
    if not rows:
        return {"total_events": 0, "patients": 0, "message": "No seizure diary data yet"}

    patients = set(r["patient_id"] for r in rows)
    severities = Counter(r["severity"] for r in rows if r["severity"])
    triggers = Counter(r["trigger"] for r in rows if r["trigger"])
    injuries = Counter(r["injury"] for r in rows if r["injury"])

    durations = [r["duration_sec"] for r in rows if r["duration_sec"]]
    avg_duration = round(sum(durations) / len(durations), 1) if durations else 0
    max_duration = max(durations) if durations else 0

    recoveries = [r["recovery_min"] for r in rows if r["recovery_min"]]
    avg_recovery = round(sum(recoveries) / len(recoveries), 1) if recoveries else 0

    er_count = sum(1 for r in rows if r["er_visit"] == "Yes")
    er_rate = round(er_count / len(rows) * 100, 1) if rows else 0

    severe_count = severities.get("Severe", 0)
    mild_count = severities.get("Mild", 0)
    severe_rate = round(severe_count / len(rows) * 100, 1) if rows else 0

    aura_count = sum(1 for r in rows if r["aura"])
    witnessed_count = sum(1 for r in rows if r["witnessed"])

    # Events per patient
    events_per_patient = Counter(r["patient_id"] for r in rows)
    avg_events = round(len(rows) / len(patients), 1) if patients else 0
    max_events_patient = max(events_per_patient.items(), key=lambda x: x[1]) if events_per_patient else ("--", 0)

    # Date range
    dates = [r["event_date"] for r in rows if r["event_date"]]
    date_range = {"earliest": min(dates), "latest": max(dates)} if dates else {}

    return {
        "total_events": len(rows),
        "unique_patients": len(patients),
        "avg_events_per_patient": avg_events,
        "most_active_patient": {"id": max_events_patient[0], "events": max_events_patient[1]},
        "avg_duration_sec": avg_duration,
        "max_duration_sec": max_duration,
        "avg_recovery_min": avg_recovery,
        "er_visits": er_count,
        "er_rate_pct": er_rate,
        "severe_count": severe_count,
        "mild_count": mild_count,
        "severe_rate_pct": severe_rate,
        "aura_reported": aura_count,
        "witnessed_events": witnessed_count,
        "severity_distribution": dict(severities),
        "trigger_distribution": dict(triggers),
        "injury_distribution": dict(injuries),
        "date_range": date_range,
    }


def breakdown():
    """Detailed breakdown: per-patient profiles, daily trend, duration buckets, recent events."""
    rows = _rows("SELECT * FROM seizure_diary ORDER BY event_date ASC")
    if not rows:
        return {"message": "No seizure diary data"}

    # Daily event trend
    daily = Counter(r["event_date"] for r in rows if r["event_date"])
    daily_trend = [{"date": d, "events": c} for d, c in sorted(daily.items())]

    # Severity by trigger
    sev_by_trigger = defaultdict(lambda: Counter())
    for r in rows:
        t = r["trigger"] or "Unknown"
        s = r["severity"] or "Unknown"
        sev_by_trigger[t][s] += 1
    trigger_severity = [
        {"trigger": t, "Mild": c.get("Mild", 0), "Severe": c.get("Severe", 0), "total": sum(c.values())}
        for t, c in sorted(sev_by_trigger.items())
    ]

    # Duration buckets
    dur_buckets = {"<1 min": 0, "1-2 min": 0, "2-5 min": 0, "5-10 min": 0, ">10 min": 0}
    for r in rows:
        d = r["duration_sec"]
        if not d:
            continue
        if d < 60:
            dur_buckets["<1 min"] += 1
        elif d < 120:
            dur_buckets["1-2 min"] += 1
        elif d < 300:
            dur_buckets["2-5 min"] += 1
        elif d < 600:
            dur_buckets["5-10 min"] += 1
        else:
            dur_buckets[">10 min"] += 1
    duration_distribution = [{"bucket": k, "count": v} for k, v in dur_buckets.items()]

    # Per-patient profile
    patient_data = defaultdict(list)
    for r in rows:
        patient_data[r["patient_id"]].append(r)

    patient_profiles = []
    for pid, events in sorted(patient_data.items()):
        sevs = [e["severity"] for e in events if e["severity"]]
        durs = [e["duration_sec"] for e in events if e["duration_sec"]]
        ers = sum(1 for e in events if e["er_visit"] == "Yes")
        injuries = sum(1 for e in events if e["injury"] and e["injury"] != "No")
        worst = "Severe" if "Severe" in sevs else ("Mild" if "Mild" in sevs else "Unknown")
        patient_profiles.append({
            "patient_id": pid,
            "total_events": len(events),
            "worst_severity": worst,
            "avg_duration_sec": round(sum(durs) / len(durs), 1) if durs else 0,
            "er_visits": ers,
            "injuries": injuries,
            "date_range": {
                "first": min(e["event_date"] for e in events if e["event_date"]),
                "last": max(e["event_date"] for e in events if e["event_date"]),
            } if any(e["event_date"] for e in events) else {},
        })
    patient_profiles.sort(key=lambda x: x["total_events"], reverse=True)

    # Recent events (last 20)
    recent = _rows("SELECT * FROM seizure_diary ORDER BY event_date DESC LIMIT 20")
    recent_events = [{
        "patient_id": r["patient_id"],
        "date": r["event_date"],
        "time": r["event_time"],
        "duration_sec": r["duration_sec"],
        "severity": r["severity"],
        "trigger": r["trigger"],
        "injury": r["injury"],
        "er_visit": r["er_visit"],
        "aura": r["aura"],
        "location": r["location"],
    } for r in recent]

    return {
        "daily_trend": daily_trend,
        "trigger_severity": trigger_severity,
        "duration_distribution": duration_distribution,
        "patient_profiles": patient_profiles,
        "recent_events": recent_events,
    }


def definitions():
    """Metric definitions for the Continuous Monitoring dashboard."""
    return {
        "title": "Continuous Monitoring — Seizure Diary Analytics — Metric Definitions",
        "sections": [
            {
                "heading": "Seizure Diary Concepts",
                "items": [
                    {"term": "Seizure Event", "definition": "A single seizure occurrence recorded by the patient or caregiver, including date, time, duration, severity, and associated symptoms."},
                    {"term": "Seizure Frequency", "definition": "Total number of seizure events per patient over the monitoring period. Higher frequency indicates poorer seizure control."},
                    {"term": "Duration (seconds)", "definition": "How long the seizure lasted. Prolonged seizures (>5 min) are medical emergencies (status epilepticus)."},
                    {"term": "Severity", "definition": "Categorized as Mild or Severe based on clinical impact, injury risk, and recovery time."},
                    {"term": "Trigger", "definition": "Known precipitant of the seizure (e.g., sleep deprivation, stress, medication non-compliance)."},
                    {"term": "Aura", "definition": "Warning sensation before seizure onset (e.g., deja vu, visual disturbance). Presence suggests focal onset."},
                    {"term": "Injury", "definition": "Physical harm during the seizure (e.g., fall, tongue bite). Key safety metric."},
                    {"term": "ER Visit", "definition": "Whether the seizure required emergency department evaluation. Indicates severity and healthcare resource utilization."},
                    {"term": "Recovery Time", "definition": "Post-ictal recovery duration in minutes. Longer recovery correlates with greater seizure impact on daily functioning."},
                ]
            },
            {
                "heading": "Quality Metrics",
                "items": [
                    {"term": "ER Rate", "definition": "Percentage of seizure events resulting in ER visits. Target: <15% for well-managed epilepsy."},
                    {"term": "Severe Rate", "definition": "Percentage of events classified as Severe. High rates warrant medication adjustment."},
                    {"term": "Avg Events/Patient", "definition": "Mean seizure frequency per patient. Benchmark varies by epilepsy type; goal is reduction over time."},
                ]
            },
            {
                "heading": "Clinical Relevance",
                "items": [
                    {"term": "ILAE Seizure Classification", "definition": "International League Against Epilepsy classification guides seizure type identification and treatment selection."},
                    {"term": "IEC 62304 §7.1", "definition": "Medical device software maintenance requires continuous monitoring of seizure detection system performance."},
                    {"term": "SUDEP Risk", "definition": "Sudden Unexpected Death in Epilepsy. Frequent generalized tonic-clonic seizures and nocturnal seizures increase risk; monitoring helps identify high-risk patients."},
                    {"term": "AAN Practice Guidelines", "definition": "American Academy of Neurology recommends seizure diary maintenance as standard of care for epilepsy management."},
                ]
            },
            {
                "heading": "Remediation Strategies",
                "items": [
                    {"term": "High ER Rate", "definition": "Review medication regimen, educate on rescue medication use, assess compliance, consider epilepsy surgery evaluation."},
                    {"term": "Rising Frequency", "definition": "Check medication adherence, screen for new triggers, obtain EEG to assess for seizure type evolution."},
                    {"term": "Prolonged Duration", "definition": "Prescribe rescue benzodiazepine, create seizure action plan, evaluate for status epilepticus risk factors."},
                    {"term": "Injury Pattern", "definition": "Occupational therapy safety assessment, helmet evaluation, modify environment (padding, shower seats)."},
                ]
            },
        ],
    }
