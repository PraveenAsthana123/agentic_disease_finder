#!/usr/bin/env python3
"""
Caregiver App Dashboard
========================

Analyses REAL caregiver / SOS data from clinical.db:

  * caregivers       — 30 rows: role, availability, burnout, stress, training
  * emergency_contacts — 30 rows: relationship, notification preference
  * emergency_sos_events — 41 rows: event type, trigger, response time, outcome

The dashboard covers the companion mobile app used by caregivers to receive
seizure alerts, record events, confirm medications, and coordinate SOS responses.

Functions:
  overview()    — KPIs, burnout distribution, role/availability breakdown, SOS trend
  breakdown()   — per-caregiver table, SOS alert log, emergency contact list
  definitions() — app features, alert types, burnout scale, references
"""

import json
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


def _avg(vals):
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def _pct(num, denom):
    return round(num / denom * 100, 1) if denom else 0.0


def _burnout_tier(score):
    if score is None:
        return "Unknown"
    if score < 30:
        return "Low"
    if score < 55:
        return "Moderate"
    if score < 75:
        return "High"
    return "Critical"


def _stress_tier(score):
    if score is None:
        return "Unknown"
    if score <= 3:
        return "Low"
    if score <= 6:
        return "Moderate"
    return "High"


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------

def overview() -> Dict[str, Any]:
    """KPIs, burnout distribution, role/availability breakdown, SOS event trend."""
    caregivers = _rows("SELECT * FROM caregivers")
    sos = _rows("SELECT * FROM emergency_sos_events")
    contacts = _rows("SELECT * FROM emergency_contacts")

    total_caregivers = len(caregivers)
    total_contacts = len(contacts)
    total_sos = len(sos)

    # Training / certification KPIs
    training_done = sum(1 for c in caregivers if c.get("epilepsy_training_completed"))
    first_aid = sum(1 for c in caregivers if c.get("first_aid_certified"))
    rescue_med = sum(1 for c in caregivers if c.get("rescue_med_trained"))

    # Burnout
    burnout_scores = [c["burnout_score"] for c in caregivers if c.get("burnout_score") is not None]
    avg_burnout = _avg(burnout_scores)
    high_burnout = sum(1 for s in burnout_scores if s >= 75)

    # Stress
    stress_scores = [c["caregiver_stress"] for c in caregivers if c.get("caregiver_stress") is not None]
    avg_stress = _avg(stress_scores)

    # SOS response metrics
    rt_vals = [s["response_time_seconds"] for s in sos if s.get("response_time_seconds") is not None]
    avg_response_sec = _avg(rt_vals)
    fast_response = sum(1 for r in rt_vals if r <= 120)
    notified = sum(1 for s in sos if s.get("responder_notified"))

    # Distributions
    role_dist = dict(Counter(c.get("role", "unknown") for c in caregivers))
    avail_dist = dict(Counter(c.get("availability", "unknown") for c in caregivers))
    burnout_dist = dict(Counter(_burnout_tier(c.get("burnout_score")) for c in caregivers))
    sos_outcome_dist = dict(Counter(s.get("outcome", "unknown") for s in sos if s.get("outcome")))
    sos_trigger_dist = dict(Counter(s.get("trigger_method", "unknown") for s in sos if s.get("trigger_method")))

    # Monthly SOS trend
    monthly: Dict[str, int] = defaultdict(int)
    for s in sos:
        d = (s.get("event_date") or "")[:7]
        if d:
            monthly[d] += 1
    sos_monthly = [{"month": m, "count": c} for m, c in sorted(monthly.items())]

    # Contact relationship breakdown
    rel_dist = dict(Counter(c.get("relationship", "unknown") for c in contacts if c.get("relationship")))
    notify_pct = _pct(sum(1 for c in contacts if c.get("notify_on_seizure")), total_contacts)

    return {
        "total_caregivers": total_caregivers,
        "total_emergency_contacts": total_contacts,
        "total_sos_events": total_sos,
        "training_completion_pct": _pct(training_done, total_caregivers),
        "first_aid_certified_pct": _pct(first_aid, total_caregivers),
        "rescue_med_trained_pct": _pct(rescue_med, total_caregivers),
        "avg_burnout_score": avg_burnout,
        "high_burnout_count": high_burnout,
        "high_burnout_pct": _pct(high_burnout, total_caregivers),
        "avg_stress_level": avg_stress,
        "sos_avg_response_sec": avg_response_sec,
        "sos_fast_response_pct": _pct(fast_response, total_sos),
        "sos_notified_pct": _pct(notified, total_sos),
        "contact_notify_on_seizure_pct": notify_pct,
        "role_distribution": role_dist,
        "availability_distribution": avail_dist,
        "burnout_tier_distribution": burnout_dist,
        "sos_outcome_distribution": sos_outcome_dist,
        "sos_trigger_distribution": sos_trigger_dist,
        "sos_monthly_trend": sos_monthly,
        "contact_relationship_distribution": rel_dist,
    }


# ---------------------------------------------------------------------------
# breakdown
# ---------------------------------------------------------------------------

def breakdown(patient_id: str = None) -> Dict[str, Any]:
    """Per-caregiver table, SOS alert log, emergency contact list."""
    params_cg = ()
    where_cg = "1=1"
    params_sos = ()
    where_sos = "1=1"

    if patient_id:
        where_cg = "patient_id = ?"
        params_cg = (patient_id,)
        where_sos = "patient_id = ?"
        params_sos = (patient_id,)

    caregivers = _rows(f"SELECT * FROM caregivers WHERE {where_cg} ORDER BY burnout_score DESC", params_cg)
    sos_events = _rows(f"SELECT * FROM emergency_sos_events WHERE {where_sos} ORDER BY event_date DESC", params_sos)
    contacts = _rows(f"SELECT * FROM emergency_contacts WHERE patient_id {'= ?' if patient_id else 'IS NOT NULL'} ORDER BY is_primary DESC, patient_id", params_cg)

    # Per-caregiver table
    caregiver_table = []
    for c in caregivers:
        training_topics = []
        try:
            training_topics = json.loads(c.get("training_topics") or "[]")
        except Exception:
            pass
        caregiver_table.append({
            "patient_id": c.get("patient_id"),
            "name": c.get("name"),
            "role": c.get("role"),
            "availability": c.get("availability"),
            "experience_years": c.get("experience_years"),
            "epilepsy_training": bool(c.get("epilepsy_training_completed")),
            "first_aid_certified": bool(c.get("first_aid_certified")),
            "rescue_med_trained": bool(c.get("rescue_med_trained")),
            "confidence": c.get("seizure_first_aid_confidence"),
            "stress": c.get("caregiver_stress"),
            "stress_tier": _stress_tier(c.get("caregiver_stress")),
            "burnout_score": c.get("burnout_score"),
            "burnout_tier": _burnout_tier(c.get("burnout_score")),
            "safety_plan": bool(c.get("safety_plan_exists")),
            "action_plan": bool(c.get("seizure_action_plan_exists")),
            "training_count": len(training_topics),
            "last_respite": c.get("last_respite_date"),
        })

    # SOS alert log
    sos_log = []
    for s in sos_events[:50]:
        sos_log.append({
            "patient_id": s.get("patient_id"),
            "date": (s.get("event_date") or "")[:10],
            "event_type": s.get("event_type"),
            "trigger_method": s.get("trigger_method"),
            "response_time_sec": s.get("response_time_seconds"),
            "notified": bool(s.get("responder_notified")),
            "location_shared": bool(s.get("location_shared")),
            "outcome": s.get("outcome"),
        })

    # Emergency contact list
    contact_list = []
    for c in contacts[:50]:
        contact_list.append({
            "patient_id": c.get("patient_id"),
            "name": c.get("contact_name"),
            "relationship": c.get("relationship"),
            "is_primary": bool(c.get("is_primary")),
            "notify_on_seizure": bool(c.get("notify_on_seizure")),
            "last_verified": c.get("last_verified"),
        })

    # High-burnout caregivers
    high_burnout = [c for c in caregiver_table if c["burnout_score"] is not None and c["burnout_score"] >= 75]

    # Training gap analysis
    training_gaps = {
        "no_epilepsy_training": sum(1 for c in caregiver_table if not c["epilepsy_training"]),
        "no_first_aid": sum(1 for c in caregiver_table if not c["first_aid_certified"]),
        "no_rescue_med": sum(1 for c in caregiver_table if not c["rescue_med_trained"]),
        "no_safety_plan": sum(1 for c in caregiver_table if not c["safety_plan"]),
        "no_action_plan": sum(1 for c in caregiver_table if not c["action_plan"]),
    }

    return {
        "caregiver_table": caregiver_table,
        "sos_alert_log": sos_log,
        "emergency_contacts": contact_list,
        "high_burnout_caregivers": high_burnout,
        "training_gaps": training_gaps,
        "total_caregivers": len(caregiver_table),
        "total_sos": len(sos_events),
        "total_contacts": len(contacts),
    }


# ---------------------------------------------------------------------------
# definitions
# ---------------------------------------------------------------------------

def definitions() -> Dict[str, Any]:
    """App feature descriptions, alert types, burnout scale, clinical references."""
    return {
        "app_name": "AgenticFinder Caregiver App",
        "description": (
            "A companion mobile application enabling caregivers of epilepsy patients "
            "to receive real-time seizure alerts, log witnessed events, confirm medication "
            "adherence, and coordinate emergency SOS responses."
        ),
        "app_features": [
            {
                "feature": "Real-time Seizure Alerts",
                "description": "Push notifications triggered by wearable seizure detection or patient SOS button",
                "status": "active",
            },
            {
                "feature": "SOS Response Coordination",
                "description": "Caregiver acknowledges alert, shares location, escalates to EMS if needed",
                "status": "active",
            },
            {
                "feature": "Seizure Event Recording",
                "description": "Log witnessed seizure: type, duration, severity, rescue medication administered",
                "status": "active",
            },
            {
                "feature": "Medication Confirmation",
                "description": "Confirm patient took prescribed medication; flag missed doses",
                "status": "active",
            },
            {
                "feature": "Video Upload",
                "description": "Upload short seizure video clips for clinician review",
                "status": "planned",
            },
            {
                "feature": "Caregiver Wellbeing Check-in",
                "description": "Weekly burnout/stress self-assessment; flags high-risk caregivers for support",
                "status": "active",
            },
            {
                "feature": "Training Module Access",
                "description": "In-app epilepsy first-aid training, rescue medication guides, action plans",
                "status": "active",
            },
        ],
        "alert_types": [
            {"type": "seizure-alert", "description": "Wearable detected convulsive seizure"},
            {"type": "fall-detected", "description": "Accelerometer detected fall event"},
            {"type": "panic-button", "description": "Patient pressed SOS button manually"},
            {"type": "manual-sos", "description": "SOS initiated via app voice/tap"},
            {"type": "medication-emergency", "description": "Missed rescue medication alert"},
        ],
        "sos_outcomes": [
            {"outcome": "caregiver-responded", "description": "Caregiver handled the event at home"},
            {"outcome": "ems-dispatched", "description": "Emergency medical services called"},
            {"outcome": "resolved-home", "description": "Resolved without EMS; patient stable"},
            {"outcome": "false-alarm", "description": "Alert triggered in error"},
            {"outcome": "er-visit", "description": "Patient transported to emergency room"},
        ],
        "burnout_scale": {
            "name": "Caregiver Burnout Score (0–100)",
            "thresholds": [
                {"tier": "Low", "range": "0–29", "action": "Routine monitoring"},
                {"tier": "Moderate", "range": "30–54", "action": "Wellbeing check-in recommended"},
                {"tier": "High", "range": "55–74", "action": "Social worker referral"},
                {"tier": "Critical", "range": "75–100", "action": "Urgent intervention required"},
            ],
        },
        "stress_scale": {
            "name": "Caregiver Stress Level (1–10)",
            "thresholds": [
                {"tier": "Low", "range": "1–3"},
                {"tier": "Moderate", "range": "4–6"},
                {"tier": "High", "range": "7–10"},
            ],
        },
        "data_sources": [
            {"table": "caregivers", "rows": 30, "description": "Caregiver profiles, burnout, training"},
            {"table": "emergency_contacts", "rows": 30, "description": "Emergency contact registry"},
            {"table": "emergency_sos_events", "rows": 41, "description": "SOS event log with response times"},
        ],
        "references": [
            "ILAE Caregiver Burden Taskforce (2022) — Epilepsy caregiver wellbeing guidelines",
            "Epilepsy Foundation — Seizure First Aid & Response Protocols",
            "NICE CG137 — Epilepsies: diagnosis and management (caregiver support)",
            "WHO QoL-BREF — Caregiver quality of life assessment framework",
        ],
    }
