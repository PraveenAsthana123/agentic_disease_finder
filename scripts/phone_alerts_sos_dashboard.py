"""
Phone Alerts & SOS Escalation Dashboard
========================================
Mobile and phone-initiated emergency alert analytics.

Focuses on patient-initiated alerts (app-button, voice-command) vs. automatic
triggers (wearable-auto, caregiver-initiated), SLA thresholds, escalation chain
outcomes, and per-patient repeat-alert profiling.

Real data: emergency_sos_events (41 rows, 26 patients) +
           emergency_contacts (30 rows) in clinical.db.
"""

import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


# SLA thresholds (seconds)
SLA_CRITICAL  = 60    # <1 min — ideal for seizure/fall events
SLA_STANDARD  = 120   # <2 min — acceptable
SLA_EXTENDED  = 300   # <5 min — tolerated for lower-urgency events

# Patient-initiated triggers (phone/voice)
PATIENT_TRIGGERS = ('app-button', 'voice-command')
# Automatic / caregiver triggers
AUTO_TRIGGERS = ('wearable-auto', 'caregiver-initiated')

# Severe outcome categories (need EMS or ER)
SEVERE_OUTCOMES = ('ems-dispatched', 'er-visit')


def overview():
    """Phone alerts overview — total events, patient-initiated vs. automated
    split, SLA compliance tiers, severe outcome rate, contact chain coverage,
    monthly alert trend."""
    conn = _conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM emergency_sos_events")
    total_events = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM emergency_sos_events")
    total_patients = cur.fetchone()[0]

    # Patient-initiated (phone) vs. automated split
    triggers_in = ",".join(f"'{t}'" for t in PATIENT_TRIGGERS)
    cur.execute(f"""
        SELECT COUNT(*) FROM emergency_sos_events
        WHERE trigger_method IN ({triggers_in})
    """)
    phone_initiated = cur.fetchone()[0]
    auto_initiated = total_events - phone_initiated

    # Event type distribution
    cur.execute("""
        SELECT event_type, COUNT(*) cnt
        FROM emergency_sos_events
        GROUP BY event_type ORDER BY cnt DESC
    """)
    event_type_dist = {r[0]: r[1] for r in cur.fetchall()}

    # Trigger method distribution
    cur.execute("""
        SELECT trigger_method, COUNT(*) cnt
        FROM emergency_sos_events
        GROUP BY trigger_method ORDER BY cnt DESC
    """)
    trigger_dist = {r[0]: r[1] for r in cur.fetchall()}

    # SLA compliance tiers
    cur.execute(f"""
        SELECT
          SUM(CASE WHEN response_time_seconds <= {SLA_CRITICAL}  THEN 1 ELSE 0 END) critical_met,
          SUM(CASE WHEN response_time_seconds <= {SLA_STANDARD} AND response_time_seconds > {SLA_CRITICAL} THEN 1 ELSE 0 END) standard_met,
          SUM(CASE WHEN response_time_seconds <= {SLA_EXTENDED} AND response_time_seconds > {SLA_STANDARD} THEN 1 ELSE 0 END) extended_met,
          SUM(CASE WHEN response_time_seconds  > {SLA_EXTENDED} THEN 1 ELSE 0 END) sla_breach,
          ROUND(AVG(response_time_seconds), 0) avg_rt,
          MIN(response_time_seconds) min_rt,
          MAX(response_time_seconds) max_rt
        FROM emergency_sos_events
    """)
    r = cur.fetchone()
    sla = {
        "critical_met":  r[0],   # <= 60s
        "standard_met":  r[1],   # 61-120s
        "extended_met":  r[2],   # 121-300s
        "breach":        r[3],   # > 300s
        "avg_rt":        r[4],
        "min_rt":        r[5],
        "max_rt":        r[6],
        "pct_under_60s": round(r[0] / max(total_events, 1) * 100, 1),
        "pct_under_120s": round((r[0] + r[1]) / max(total_events, 1) * 100, 1),
    }

    # Severe outcome rate
    severe_in = ",".join(f"'{o}'" for o in SEVERE_OUTCOMES)
    cur.execute(f"""
        SELECT COUNT(*) FROM emergency_sos_events WHERE outcome IN ({severe_in})
    """)
    severe_count = cur.fetchone()[0]

    # Outcome distribution
    cur.execute("""
        SELECT outcome, COUNT(*) cnt
        FROM emergency_sos_events
        GROUP BY outcome ORDER BY cnt DESC
    """)
    outcome_dist = {r[0]: r[1] for r in cur.fetchall()}

    # Responder notified rate
    cur.execute("""
        SELECT ROUND(AVG(CASE WHEN responder_notified=1 THEN 1.0 ELSE 0.0 END)*100,1)
        FROM emergency_sos_events
    """)
    responder_pct = cur.fetchone()[0]

    # Location shared rate
    cur.execute("""
        SELECT ROUND(AVG(CASE WHEN location_shared=1 THEN 1.0 ELSE 0.0 END)*100,1)
        FROM emergency_sos_events
    """)
    location_pct = cur.fetchone()[0]

    # Monthly trend
    cur.execute("""
        SELECT SUBSTR(event_date,1,7) month, COUNT(*) cnt
        FROM emergency_sos_events
        GROUP BY month ORDER BY month
    """)
    monthly_trend = _dict_rows(cur)

    # Contact chain coverage
    cur.execute("SELECT COUNT(*) FROM emergency_contacts")
    total_contacts = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM emergency_contacts")
    patients_with_contacts = cur.fetchone()[0]

    # Patients with events but no contacts
    cur.execute("""
        SELECT COUNT(DISTINCT e.patient_id) FROM emergency_sos_events e
        WHERE e.patient_id NOT IN (SELECT DISTINCT patient_id FROM emergency_contacts)
    """)
    no_contact_patients = cur.fetchone()[0]

    conn.close()
    return {
        "total_events": total_events,
        "total_patients": total_patients,
        "phone_initiated_events": phone_initiated,
        "auto_initiated_events": auto_initiated,
        "phone_pct": round(phone_initiated / max(total_events, 1) * 100, 1),
        "auto_pct": round(auto_initiated / max(total_events, 1) * 100, 1),
        "event_type_distribution": event_type_dist,
        "trigger_method_distribution": trigger_dist,
        "sla": sla,
        "severe_events": severe_count,
        "severe_rate_pct": round(severe_count / max(total_events, 1) * 100, 1),
        "outcome_distribution": outcome_dist,
        "responder_notified_pct": responder_pct,
        "location_shared_pct": location_pct,
        "monthly_trend": monthly_trend,
        "contacts": {
            "total": total_contacts,
            "patients_covered": patients_with_contacts,
            "patients_no_contact": no_contact_patients,
            "coverage_pct": round(patients_with_contacts / max(total_patients, 1) * 100, 1),
        },
        "sla_thresholds": {
            "critical_s": SLA_CRITICAL,
            "standard_s": SLA_STANDARD,
            "extended_s": SLA_EXTENDED,
        },
    }


def breakdown():
    """Phone alerts breakdown — per-patient repeat alert profile, phone vs. auto
    by event type, SLA by trigger method, escalation chain (trigger→outcome),
    no-response events, recent event log."""
    conn = _conn()
    cur = conn.cursor()

    triggers_in = ",".join(f"'{t}'" for t in PATIENT_TRIGGERS)
    severe_in   = ",".join(f"'{o}'" for o in SEVERE_OUTCOMES)

    # Per-patient alert profile
    cur.execute(f"""
        SELECT e.patient_id,
               COUNT(*) total_events,
               SUM(CASE WHEN e.trigger_method IN ({triggers_in}) THEN 1 ELSE 0 END) phone_alerts,
               SUM(CASE WHEN e.trigger_method NOT IN ({triggers_in}) THEN 1 ELSE 0 END) auto_alerts,
               ROUND(AVG(e.response_time_seconds), 0) avg_rt,
               SUM(CASE WHEN e.outcome IN ({severe_in}) THEN 1 ELSE 0 END) severe_events,
               SUM(CASE WHEN e.outcome = 'false-alarm' THEN 1 ELSE 0 END) false_alarms,
               SUM(CASE WHEN e.location_shared = 1 THEN 1 ELSE 0 END) location_shared,
               (SELECT COUNT(*) FROM emergency_contacts c WHERE c.patient_id = e.patient_id) contacts
        FROM emergency_sos_events e
        GROUP BY e.patient_id
        ORDER BY total_events DESC
    """)
    patient_profile = _dict_rows(cur)

    # Phone-initiated vs. auto by event type
    cur.execute(f"""
        SELECT event_type,
               SUM(CASE WHEN trigger_method IN ({triggers_in}) THEN 1 ELSE 0 END) phone,
               SUM(CASE WHEN trigger_method NOT IN ({triggers_in}) THEN 1 ELSE 0 END) auto,
               COUNT(*) total
        FROM emergency_sos_events
        GROUP BY event_type ORDER BY total DESC
    """)
    phone_vs_auto_by_type = _dict_rows(cur)

    # SLA compliance by trigger method
    cur.execute(f"""
        SELECT trigger_method,
               COUNT(*) total,
               SUM(CASE WHEN response_time_seconds <= {SLA_CRITICAL} THEN 1 ELSE 0 END) critical_met,
               SUM(CASE WHEN response_time_seconds <= {SLA_STANDARD} THEN 1 ELSE 0 END) standard_met,
               SUM(CASE WHEN response_time_seconds  > {SLA_EXTENDED} THEN 1 ELSE 0 END) breach,
               ROUND(AVG(response_time_seconds), 0) avg_rt
        FROM emergency_sos_events
        GROUP BY trigger_method ORDER BY avg_rt
    """)
    sla_by_trigger = _dict_rows(cur)

    # Escalation chain: trigger × outcome
    cur.execute("""
        SELECT trigger_method, outcome, COUNT(*) cnt
        FROM emergency_sos_events
        GROUP BY trigger_method, outcome
        ORDER BY trigger_method, cnt DESC
    """)
    escalation_chain = _dict_rows(cur)

    # Events with slowest response (potential SLA breaches)
    cur.execute(f"""
        SELECT id, patient_id, event_date, event_type, trigger_method,
               response_time_seconds, outcome, location_shared, notes
        FROM emergency_sos_events
        WHERE response_time_seconds > {SLA_EXTENDED}
        ORDER BY response_time_seconds DESC
    """)
    sla_breaches = _dict_rows(cur)

    # Recent 20 events
    cur.execute("""
        SELECT id, patient_id, event_date, event_type, trigger_method,
               responder_notified, response_time_seconds, location_shared, outcome, notes
        FROM emergency_sos_events
        ORDER BY event_date DESC
        LIMIT 20
    """)
    recent_events = _dict_rows(cur)

    # Contact chain — stale contacts (>180 days unverified)
    cur.execute("""
        SELECT patient_id, contact_name, relationship,
               phone, last_verified,
               CAST(JULIANDAY('now') - JULIANDAY(last_verified) AS INTEGER) days_stale
        FROM emergency_contacts
        WHERE JULIANDAY('now') - JULIANDAY(last_verified) > 180
        ORDER BY days_stale DESC
    """)
    stale_contacts = _dict_rows(cur)

    conn.close()
    return {
        "patient_profile": patient_profile,
        "phone_vs_auto_by_type": phone_vs_auto_by_type,
        "sla_by_trigger": sla_by_trigger,
        "escalation_chain": escalation_chain,
        "sla_breaches": sla_breaches,
        "recent_events": recent_events,
        "stale_contacts": stale_contacts,
    }


def definitions():
    """Phone alerts definitions — clinical glossary, trigger categories, SLA
    tiers, escalation outcome descriptions, and readiness standards."""
    return {
        "glossary": [
            {"term": "Phone Alert", "definition": "Any SOS event triggered directly by the patient via a mobile device (app button or voice command)"},
            {"term": "Auto Alert", "definition": "SOS event triggered automatically by a wearable sensor or by a caregiver on the patient's behalf"},
            {"term": "Escalation Chain", "definition": "The sequence from alert trigger → responder notification → outcome resolution"},
            {"term": "SLA (Service Level Agreement)", "definition": "Target response time thresholds: Critical ≤60s, Standard ≤120s, Extended ≤300s"},
            {"term": "SLA Breach", "definition": "Any event where response time exceeds the Extended threshold (>300 seconds)"},
            {"term": "Location Sharing", "definition": "GPS coordinates transmitted with the alert to help responders locate the patient"},
            {"term": "Stale Contact", "definition": "An emergency contact whose details have not been verified in the past 180 days"},
            {"term": "False Alarm", "definition": "An SOS event that required no emergency intervention (accidental trigger or device error)"},
            {"term": "Severe Outcome", "definition": "Events resulting in EMS dispatch or ER visit — the highest acuity tier"},
            {"term": "Contact Chain Coverage", "definition": "Percentage of patients with at least one active, verified emergency contact"},
            {"term": "Post-Ictal Period", "definition": "Recovery phase after a seizure; may impair patient's ability to self-report or use phone"},
            {"term": "SUDEP", "definition": "Sudden Unexpected Death in Epilepsy; the primary motivator for rapid SOS response protocols"},
        ],
        "trigger_categories": [
            {
                "category": "Patient-Initiated (Phone)",
                "triggers": ["app-button", "voice-command"],
                "description": "Patient directly activates an alert via mobile app button or voice assistant"
            },
            {
                "category": "Automated / Caregiver",
                "triggers": ["wearable-auto", "caregiver-initiated"],
                "description": "Alert triggered by wearable sensor detection or by a caregiver witnessing an event"
            },
        ],
        "sla_tiers": [
            {"tier": "Critical", "threshold_s": 60,  "label": "≤60 s",  "description": "Seizure or fall — requires immediate response"},
            {"tier": "Standard", "threshold_s": 120, "label": "≤120 s", "description": "Acceptable for most alert types; target for all events"},
            {"tier": "Extended", "threshold_s": 300, "label": "≤300 s", "description": "Tolerated for lower-urgency alerts (medication, manual SOS)"},
            {"tier": "Breach",   "threshold_s": None,"label": ">300 s", "description": "SLA breach — review required regardless of outcome"},
        ],
        "outcome_descriptions": [
            {"outcome": "caregiver-responded", "severity": "moderate", "description": "Emergency contact or caregiver responded and managed the situation at home"},
            {"outcome": "er-visit",            "severity": "high",     "description": "Patient transported to emergency room for evaluation and treatment"},
            {"outcome": "false-alarm",         "severity": "low",      "description": "No actual emergency — accidental trigger or device malfunction"},
            {"outcome": "resolved-home",       "severity": "low",      "description": "Event resolved at home without hospital or EMS involvement"},
            {"outcome": "ems-dispatched",      "severity": "critical", "description": "Emergency Medical Services (ambulance) dispatched to patient location"},
        ],
        "readiness_standards": [
            {"metric": "Contact Coverage",   "target": "100% of patients with ≥1 verified contact",  "rationale": "Every patient must have a reachable emergency contact"},
            {"metric": "Contact Freshness",  "target": "All contacts verified within 180 days",       "rationale": "Stale contact information delays or blocks response"},
            {"metric": "Location Sharing",   "target": ">90% of events include GPS coordinates",      "rationale": "Location data reduces EMS dispatch and response time"},
            {"metric": "SLA ≤120s",          "target": ">80% of events acknowledged within 120s",    "rationale": "Most seizure emergencies require response within 2 minutes"},
            {"metric": "False Alarm Rate",   "target": "<15% of total events",                        "rationale": "High false-alarm rates desensitize caregivers (cry-wolf effect)"},
            {"metric": "Severe Outcome Rate","target": "Monitored; interventions if trend rising",    "rationale": "Rising ems/er-visit rate signals deteriorating clinical control"},
        ],
    }
