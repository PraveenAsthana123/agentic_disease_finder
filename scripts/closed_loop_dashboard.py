#!/usr/bin/env python3
"""
Closed-Loop Neurostimulation Dashboard
=======================================

Computes REAL closed-loop responsive-alert metrics by cross-referencing:

  1. **wearable_readings** — seizure_risk_score, seizure_detected, HRV, SpO2
  2. **seizure_diary** — verified seizure events with severity and triggers
  3. **emergency_sos_events** — SOS triggers, response times, outcomes
  4. **iot_alerts** — device alerts including sos_seizure type

The dashboard visualises the full closed loop:
  pre-ictal detection → alert trigger → responder notification → response → outcome audit

Functions:
  overview()     — loop KPIs, trigger breakdown, response latency, outcome distribution
  breakdown()    — per-patient loop performance, event timeline, alert-response audit
  definitions()  — closed-loop methodology, glossary, clinical references
"""

import json
import math
import os
import sqlite3
from collections import Counter, defaultdict
from typing import Any, Dict, List

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
_DB_PATH = os.path.join(_BASE_DIR, "data", "clinical.db")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _conn():
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    if not os.path.exists(_DB_PATH):
        return []
    conn = _conn()
    try:
        return [dict(r) for r in conn.execute(query, params).fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


def _safe_json(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _avg(vals):
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _std(vals):
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return round(math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)), 4)


def _pct(num, denom):
    return round(num / denom, 4) if denom else 0.0


# ---------------------------------------------------------------------------
# data loaders
# ---------------------------------------------------------------------------

def _load_sos_events():
    """Return all emergency SOS events."""
    return _rows("SELECT * FROM emergency_sos_events ORDER BY event_date")


def _load_seizure_diary():
    """Return all seizure diary entries."""
    return _rows("SELECT * FROM seizure_diary ORDER BY event_date")


def _load_iot_alerts():
    """Return parsed IoT alerts."""
    raw = _rows("SELECT id, patient_id, fields_json, created_at FROM iot_alerts")
    out = []
    for r in raw:
        fields = _safe_json(r.get("fields_json", ""))
        fields["id"] = r["id"]
        fields["db_patient_id"] = r["patient_id"]
        fields["created_at"] = r["created_at"]
        out.append(fields)
    return out


def _load_wearable_risk():
    """Return wearable readings with seizure risk scores."""
    raw = _rows("SELECT patient_id, fields_json FROM wearable_readings")
    out = []
    for r in raw:
        fields = _safe_json(r.get("fields_json", ""))
        fields["patient_id"] = r["patient_id"]
        out.append(fields)
    return out


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------

def overview() -> Dict[str, Any]:
    sos = _load_sos_events()
    diary = _load_seizure_diary()
    alerts = _load_iot_alerts()
    wear = _load_wearable_risk()

    # -- KPIs --
    total_sos = len(sos)
    total_seizures = len(diary)
    seizure_alerts = [a for a in alerts if a.get("alert_type") == "sos_seizure"]
    total_seizure_alerts = len(seizure_alerts)

    # Response times
    response_times = [e["response_time_seconds"] for e in sos
                      if e.get("response_time_seconds") is not None]
    avg_response = _avg(response_times)
    median_response = sorted(response_times)[len(response_times) // 2] if response_times else 0
    p95_response = sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0

    # Responder notification rate
    notified = sum(1 for e in sos if e.get("responder_notified"))
    notification_rate = _pct(notified, total_sos)

    # Location sharing rate
    loc_shared = sum(1 for e in sos if e.get("location_shared"))
    loc_rate = _pct(loc_shared, total_sos)

    # False alarm rate
    false_alarms = sum(1 for e in sos if e.get("outcome") == "false-alarm")
    false_alarm_rate = _pct(false_alarms, total_sos)

    # -- Trigger method breakdown --
    trigger_counts = Counter(e.get("trigger_method", "unknown") for e in sos)
    trigger_breakdown = [{"method": k, "count": v, "pct": _pct(v, total_sos)}
                         for k, v in trigger_counts.most_common()]

    # -- Event type breakdown --
    event_type_counts = Counter(e.get("event_type", "unknown") for e in sos)
    event_type_breakdown = [{"type": k, "count": v, "pct": _pct(v, total_sos)}
                            for k, v in event_type_counts.most_common()]

    # -- Outcome distribution --
    outcome_counts = Counter(e.get("outcome", "unknown") for e in sos)
    outcome_breakdown = [{"outcome": k, "count": v, "pct": _pct(v, total_sos)}
                         for k, v in outcome_counts.most_common()]

    # -- Response time distribution (buckets) --
    buckets = {"<60s": 0, "60-120s": 0, "120-300s": 0, "300-600s": 0, ">600s": 0}
    for rt in response_times:
        if rt < 60:
            buckets["<60s"] += 1
        elif rt < 120:
            buckets["60-120s"] += 1
        elif rt < 300:
            buckets["120-300s"] += 1
        elif rt < 600:
            buckets["300-600s"] += 1
        else:
            buckets[">600s"] += 1
    response_dist = [{"bucket": k, "count": v} for k, v in buckets.items()]

    # -- Wearable risk score distribution --
    risk_scores = [w.get("seizure_risk_score", 0) for w in wear
                   if w.get("seizure_risk_score") is not None]
    high_risk = sum(1 for s in risk_scores if s >= 0.7)
    medium_risk = sum(1 for s in risk_scores if 0.4 <= s < 0.7)
    low_risk = sum(1 for s in risk_scores if s < 0.4)
    risk_distribution = [
        {"tier": "High (≥0.7)", "count": high_risk, "pct": _pct(high_risk, len(risk_scores))},
        {"tier": "Medium (0.4–0.7)", "count": medium_risk, "pct": _pct(medium_risk, len(risk_scores))},
        {"tier": "Low (<0.4)", "count": low_risk, "pct": _pct(low_risk, len(risk_scores))},
    ]

    # -- Seizure detection loop coverage --
    detected = sum(1 for w in wear if w.get("seizure_detected"))
    detection_rate = _pct(detected, len(wear)) if wear else 0.0

    # -- Closed-loop effectiveness (seizure events that had SOS response) --
    sos_patients = set(e.get("patient_id") for e in sos)
    diary_patients = set(e.get("patient_id") for e in diary)
    covered_patients = sos_patients & diary_patients
    loop_coverage = _pct(len(covered_patients), len(diary_patients)) if diary_patients else 0.0

    return {
        "kpis": {
            "total_sos_events": total_sos,
            "total_seizures_logged": total_seizures,
            "seizure_alerts_iot": total_seizure_alerts,
            "avg_response_seconds": avg_response,
            "median_response_seconds": median_response,
            "p95_response_seconds": p95_response,
            "notification_rate": notification_rate,
            "location_sharing_rate": loc_rate,
            "false_alarm_rate": false_alarm_rate,
            "detection_rate": detection_rate,
            "loop_patient_coverage": loop_coverage,
        },
        "trigger_breakdown": trigger_breakdown,
        "event_type_breakdown": event_type_breakdown,
        "outcome_distribution": outcome_breakdown,
        "response_time_distribution": response_dist,
        "risk_distribution": risk_distribution,
    }


# ---------------------------------------------------------------------------
# breakdown
# ---------------------------------------------------------------------------

def breakdown() -> Dict[str, Any]:
    sos = _load_sos_events()
    diary = _load_seizure_diary()
    alerts = _load_iot_alerts()

    # -- Per-patient closed-loop audit --
    patient_sos = defaultdict(list)
    for e in sos:
        pid = e.get("patient_id", "unknown")
        patient_sos[pid].append(e)

    patient_diary = defaultdict(list)
    for e in diary:
        pid = e.get("patient_id", "unknown")
        patient_diary[pid].append(e)

    per_patient = []
    all_pids = sorted(set(list(patient_sos.keys()) + list(patient_diary.keys())))
    for pid in all_pids:
        events = patient_sos.get(pid, [])
        seizures = patient_diary.get(pid, [])
        rts = [e["response_time_seconds"] for e in events
               if e.get("response_time_seconds") is not None]
        outcomes = Counter(e.get("outcome", "unknown") for e in events)
        false_ct = outcomes.get("false-alarm", 0)
        per_patient.append({
            "patient_id": pid,
            "sos_events": len(events),
            "seizures_logged": len(seizures),
            "avg_response_s": _avg(rts),
            "min_response_s": min(rts) if rts else None,
            "max_response_s": max(rts) if rts else None,
            "false_alarms": false_ct,
            "false_alarm_rate": _pct(false_ct, len(events)) if events else 0.0,
            "outcomes": dict(outcomes),
            "trigger_methods": dict(Counter(e.get("trigger_method") for e in events)),
        })

    # -- Event timeline (all SOS events chronologically) --
    timeline = []
    for e in sorted(sos, key=lambda x: x.get("event_date", "")):
        timeline.append({
            "patient_id": e.get("patient_id"),
            "date": e.get("event_date"),
            "event_type": e.get("event_type"),
            "trigger_method": e.get("trigger_method"),
            "response_time_s": e.get("response_time_seconds"),
            "responder_notified": bool(e.get("responder_notified")),
            "location_shared": bool(e.get("location_shared")),
            "outcome": e.get("outcome"),
            "notes": e.get("notes"),
        })

    # -- IoT alert audit (seizure-related) --
    seizure_iot = [a for a in alerts if a.get("alert_type") == "sos_seizure"]
    critical_iot = [a for a in alerts if a.get("severity") == "critical"]
    alert_audit = {
        "total_iot_alerts": len(alerts),
        "seizure_specific": len(seizure_iot),
        "critical_alerts": len(critical_iot),
        "by_type": dict(Counter(a.get("alert_type", "unknown") for a in alerts)),
        "by_severity": dict(Counter(a.get("severity", "unknown") for a in alerts)),
        "resolved_count": sum(1 for a in alerts if a.get("resolved")),
        "acknowledged_count": sum(1 for a in alerts if a.get("acknowledged")),
    }

    # -- Severity vs response analysis --
    severity_response = defaultdict(list)
    for e in sos:
        etype = e.get("event_type", "unknown")
        rt = e.get("response_time_seconds")
        if rt is not None:
            severity_response[etype].append(rt)
    severity_analysis = [
        {
            "event_type": k,
            "count": len(v),
            "avg_response_s": _avg(v),
            "std_response_s": _std(v),
        }
        for k, v in sorted(severity_response.items())
    ]

    # -- Trigger method effectiveness --
    method_outcomes = defaultdict(lambda: defaultdict(int))
    for e in sos:
        method = e.get("trigger_method", "unknown")
        outcome = e.get("outcome", "unknown")
        method_outcomes[method][outcome] += 1
    trigger_effectiveness = [
        {"method": m, "outcomes": dict(oc)}
        for m, oc in sorted(method_outcomes.items())
    ]

    return {
        "per_patient": per_patient,
        "event_timeline": timeline,
        "alert_audit": alert_audit,
        "severity_analysis": severity_analysis,
        "trigger_effectiveness": trigger_effectiveness,
    }


# ---------------------------------------------------------------------------
# definitions
# ---------------------------------------------------------------------------

def definitions() -> Dict[str, Any]:
    return {
        "title": "Closed-Loop Neurostimulation — Methodology & Glossary",
        "loop_stages": [
            {
                "stage": 1,
                "name": "Pre-ictal Detection",
                "description": "Wearable biosensors (HRV, EDA, SpO2, accelerometer) continuously compute a seizure_risk_score. When the score exceeds a configurable threshold, the system enters the alert phase.",
                "data_source": "wearable_readings.seizure_risk_score",
            },
            {
                "stage": 2,
                "name": "Alert Trigger",
                "description": "An SOS event is generated via one of four trigger methods: wearable-auto (automatic detection), app-button (manual in-app), caregiver-initiated, or voice-command. The system classifies the event type (seizure-alert, fall-detected, panic-button, manual-sos, medication-emergency).",
                "data_source": "emergency_sos_events.trigger_method + event_type",
            },
            {
                "stage": 3,
                "name": "Responder Notification",
                "description": "Designated caregivers and/or emergency services are notified with patient location (if shared). The system logs notification success and tracks response latency.",
                "data_source": "emergency_sos_events.responder_notified + location_shared",
            },
            {
                "stage": 4,
                "name": "Response & Intervention",
                "description": "The response is timed from alert trigger to first responder action. Outcomes are classified: caregiver-responded, er-visit, ems-dispatched, resolved-home, or false-alarm.",
                "data_source": "emergency_sos_events.response_time_seconds + outcome",
            },
            {
                "stage": 5,
                "name": "Outcome Audit",
                "description": "Each closed-loop cycle is audited: was the alert genuine? Was the response timely? Did the intervention prevent escalation? False-alarm rates and response-time distributions feed back into threshold tuning.",
                "data_source": "emergency_sos_events + seizure_diary cross-reference",
            },
        ],
        "glossary": [
            {"term": "Closed-loop system", "definition": "A system where the output (intervention) feeds back into the input (detection thresholds), creating a self-improving cycle."},
            {"term": "Pre-ictal", "definition": "The period immediately before a seizure, during which biomarker changes (HRV drop, EDA spike) may be detectable."},
            {"term": "Responsive neurostimulation (RNS)", "definition": "FDA-approved implantable device that detects abnormal brain activity and delivers targeted electrical stimulation to prevent seizures."},
            {"term": "False alarm rate (FAR)", "definition": "Proportion of alerts that did not correspond to a genuine seizure or emergency."},
            {"term": "Response latency", "definition": "Time in seconds from alert trigger to first responder action."},
            {"term": "SOS event", "definition": "An emergency event triggered by the patient, caregiver, or automated detection system."},
            {"term": "Trigger method", "definition": "How the alert was initiated: wearable-auto, app-button, caregiver-initiated, or voice-command."},
            {"term": "Loop coverage", "definition": "Proportion of patients with seizure diary entries who also have corresponding SOS event coverage."},
        ],
        "clinical_references": [
            "Morrell MJ. Responsive cortical stimulation for the treatment of medically intractable partial epilepsy. Neurology. 2011;77(13):1295-1304.",
            "Brinkmann BH et al. Crowdsourcing reproducible seizure forecasting in human and canine epilepsy. Brain. 2016;139(6):1713-1722.",
            "Baud MO et al. Multi-day rhythms modulate seizure risk in epilepsy. Nature Communications. 2018;9(1):88.",
            "Karoly PJ et al. Cycles in epilepsy. Nature Reviews Neurology. 2021;17(5):267-284.",
        ],
        "data_tables_used": [
            "wearable_readings (seizure_risk_score, seizure_detected)",
            "seizure_diary (verified seizure events)",
            "emergency_sos_events (SOS triggers, response, outcomes)",
            "iot_alerts (device alerts, sos_seizure type)",
        ],
    }


if __name__ == "__main__":
    import json as _json
    print("=== OVERVIEW ===")
    print(_json.dumps(overview(), indent=2, default=str)[:2000])
    print("\n=== BREAKDOWN ===")
    print(_json.dumps(breakdown(), indent=2, default=str)[:2000])
    print("\n=== DEFINITIONS ===")
    print(_json.dumps(definitions(), indent=2, default=str)[:1000])
