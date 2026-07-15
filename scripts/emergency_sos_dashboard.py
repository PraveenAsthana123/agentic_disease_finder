"""
Neuro AI Ecosystem — Emergency SOS Dashboard
=============================================
Emergency alert analytics from emergency_sos_events + emergency_contacts tables.

Event types: manual-sos, panic-button, fall-detected, seizure-alert, medication-emergency
Trigger methods: wearable-auto, caregiver-initiated, app-button, voice-command
Outcomes: caregiver-responded, er-visit, false-alarm, resolved-home, ems-dispatched

Real data: emergency_sos_events (41 rows, 26 patients) + emergency_contacts (30 rows)
in clinical.db.
"""

import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def overview():
    """Emergency SOS overview — total events, event type distribution, trigger method
    breakdown, outcome analysis, response time stats, location sharing rate."""
    conn = _conn()
    cur = conn.cursor()

    # Total events
    cur.execute("SELECT COUNT(*) FROM emergency_sos_events")
    total_events = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM emergency_sos_events")
    patients_with_events = cur.fetchone()[0]

    # Event type distribution
    cur.execute("""
        SELECT event_type, COUNT(*) cnt
        FROM emergency_sos_events
        GROUP BY event_type
        ORDER BY cnt DESC
    """)
    event_type_dist = {r[0]: r[1] for r in cur.fetchall()}

    # Trigger method distribution
    cur.execute("""
        SELECT trigger_method, COUNT(*) cnt
        FROM emergency_sos_events
        GROUP BY trigger_method
        ORDER BY cnt DESC
    """)
    trigger_method_dist = {r[0]: r[1] for r in cur.fetchall()}

    # Outcome distribution
    cur.execute("""
        SELECT outcome, COUNT(*) cnt
        FROM emergency_sos_events
        GROUP BY outcome
        ORDER BY cnt DESC
    """)
    outcome_dist = {r[0]: r[1] for r in cur.fetchall()}

    # Response time statistics
    cur.execute("""
        SELECT ROUND(AVG(response_time_seconds), 0) avg_rt,
               MIN(response_time_seconds) min_rt,
               MAX(response_time_seconds) max_rt,
               ROUND(AVG(CASE WHEN response_time_seconds <= 120 THEN 1.0 ELSE 0.0 END) * 100, 1) pct_under_2min
        FROM emergency_sos_events
    """)
    r = cur.fetchone()
    response_stats = {
        "avg_seconds": r[0],
        "min_seconds": r[1],
        "max_seconds": r[2],
        "pct_under_2min": r[3]
    }

    # Location sharing rate
    cur.execute("""
        SELECT ROUND(AVG(CASE WHEN location_shared = 1 THEN 1.0 ELSE 0.0 END) * 100, 1)
        FROM emergency_sos_events
    """)
    location_sharing_pct = cur.fetchone()[0]

    # Responder notification rate
    cur.execute("""
        SELECT ROUND(AVG(CASE WHEN responder_notified = 1 THEN 1.0 ELSE 0.0 END) * 100, 1)
        FROM emergency_sos_events
    """)
    responder_notified_pct = cur.fetchone()[0]

    # False alarm rate
    false_alarm_count = outcome_dist.get("false-alarm", 0)
    false_alarm_pct = round(false_alarm_count / max(total_events, 1) * 100, 1)

    # Monthly trend
    cur.execute("""
        SELECT SUBSTR(event_date, 1, 7) month, COUNT(*) cnt
        FROM emergency_sos_events
        GROUP BY month
        ORDER BY month
    """)
    monthly_trend = _dict_rows(cur)

    # Emergency contacts overview
    cur.execute("SELECT COUNT(*) FROM emergency_contacts")
    total_contacts = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM emergency_contacts")
    patients_with_contacts = cur.fetchone()[0]

    cur.execute("""
        SELECT relationship, COUNT(*) cnt
        FROM emergency_contacts
        GROUP BY relationship
        ORDER BY cnt DESC
    """)
    relationship_dist = {r[0]: r[1] for r in cur.fetchall()}

    cur.execute("""
        SELECT ROUND(AVG(CASE WHEN notify_on_seizure = 1 THEN 1.0 ELSE 0.0 END) * 100, 1)
        FROM emergency_contacts
    """)
    seizure_notify_pct = cur.fetchone()[0]

    conn.close()
    return {
        "total_events": total_events,
        "patients_with_events": patients_with_events,
        "event_type_distribution": event_type_dist,
        "trigger_method_distribution": trigger_method_dist,
        "outcome_distribution": outcome_dist,
        "response_time_stats": response_stats,
        "location_sharing_pct": location_sharing_pct,
        "responder_notified_pct": responder_notified_pct,
        "false_alarm_rate_pct": false_alarm_pct,
        "monthly_trend": monthly_trend,
        "contacts": {
            "total": total_contacts,
            "patients_covered": patients_with_contacts,
            "relationship_distribution": relationship_dist,
            "seizure_notify_pct": seizure_notify_pct
        }
    }


def breakdown():
    """Emergency SOS breakdown — per-patient event history, response time by
    event type, trigger-outcome cross-tab, recent events, contact coverage."""
    conn = _conn()
    cur = conn.cursor()

    # Per-patient event counts
    cur.execute("""
        SELECT e.patient_id,
               COUNT(*) total_events,
               ROUND(AVG(e.response_time_seconds), 0) avg_response,
               SUM(CASE WHEN e.outcome = 'false-alarm' THEN 1 ELSE 0 END) false_alarms,
               SUM(CASE WHEN e.outcome = 'er-visit' THEN 1 ELSE 0 END) er_visits,
               SUM(CASE WHEN e.outcome = 'ems-dispatched' THEN 1 ELSE 0 END) ems_dispatched,
               SUM(CASE WHEN e.location_shared = 1 THEN 1 ELSE 0 END) location_shared_count,
               (SELECT COUNT(*) FROM emergency_contacts c WHERE c.patient_id = e.patient_id) contact_count
        FROM emergency_sos_events e
        GROUP BY e.patient_id
        ORDER BY total_events DESC
    """)
    patient_events = _dict_rows(cur)

    # Response time by event type
    cur.execute("""
        SELECT event_type,
               COUNT(*) cnt,
               ROUND(AVG(response_time_seconds), 0) avg_rt,
               MIN(response_time_seconds) min_rt,
               MAX(response_time_seconds) max_rt
        FROM emergency_sos_events
        GROUP BY event_type
        ORDER BY avg_rt
    """)
    response_by_type = _dict_rows(cur)

    # Response time by trigger method
    cur.execute("""
        SELECT trigger_method,
               COUNT(*) cnt,
               ROUND(AVG(response_time_seconds), 0) avg_rt
        FROM emergency_sos_events
        GROUP BY trigger_method
        ORDER BY avg_rt
    """)
    response_by_trigger = _dict_rows(cur)

    # Trigger × outcome cross-tab
    cur.execute("""
        SELECT trigger_method, outcome, COUNT(*) cnt
        FROM emergency_sos_events
        GROUP BY trigger_method, outcome
        ORDER BY trigger_method, outcome
    """)
    trigger_outcome_crosstab = _dict_rows(cur)

    # Recent events (most recent 30)
    cur.execute("""
        SELECT id, patient_id, event_date, event_type, trigger_method,
               responder_notified, response_time_seconds, location_shared,
               outcome, notes
        FROM emergency_sos_events
        ORDER BY event_date DESC
        LIMIT 30
    """)
    recent_events = _dict_rows(cur)

    # Contact coverage — patients with vs without contacts
    cur.execute("""
        SELECT e.patient_id,
               (SELECT COUNT(*) FROM emergency_contacts c WHERE c.patient_id = e.patient_id) contact_count,
               (SELECT GROUP_CONCAT(c.relationship, ', ')
                FROM emergency_contacts c WHERE c.patient_id = e.patient_id) relationships
        FROM (SELECT DISTINCT patient_id FROM emergency_sos_events) e
        ORDER BY e.patient_id
    """)
    contact_coverage = _dict_rows(cur)

    # Contacts needing verification (last_verified > 6 months ago)
    cur.execute("""
        SELECT patient_id, contact_name, relationship, last_verified,
               CAST(JULIANDAY('now') - JULIANDAY(last_verified) AS INTEGER) days_since_verified
        FROM emergency_contacts
        WHERE JULIANDAY('now') - JULIANDAY(last_verified) > 180
        ORDER BY days_since_verified DESC
    """)
    stale_contacts = _dict_rows(cur)

    conn.close()
    return {
        "patient_events": patient_events,
        "response_by_event_type": response_by_type,
        "response_by_trigger_method": response_by_trigger,
        "trigger_outcome_crosstab": trigger_outcome_crosstab,
        "recent_events": recent_events,
        "contact_coverage": contact_coverage,
        "stale_contacts": stale_contacts
    }


def definitions():
    """Emergency SOS definitions — clinical glossary, event types, trigger methods,
    outcome categories, and emergency preparedness references."""
    return {
        "glossary": [
            {"term": "SOS Event", "definition": "Any emergency alert triggered by patient, caregiver, or automated detection system"},
            {"term": "Response Time", "definition": "Seconds from alert trigger to first responder acknowledgement"},
            {"term": "False Alarm", "definition": "SOS event where no actual emergency was occurring (accidental trigger, device malfunction)"},
            {"term": "Tonic-Clonic Seizure", "definition": "Generalized seizure with stiffening (tonic) and jerking (clonic) phases; the most common trigger for emergency SOS"},
            {"term": "Post-Ictal Period", "definition": "Recovery phase after a seizure; patient may be confused, fatigued, or unresponsive"},
            {"term": "Status Epilepticus", "definition": "Prolonged seizure lasting >5 minutes or repeated seizures without recovery; medical emergency requiring EMS"},
            {"term": "SUDEP", "definition": "Sudden Unexpected Death in Epilepsy; primary motivation for emergency response readiness"},
            {"term": "Seizure Action Plan", "definition": "Written instructions for caregivers on what to do during and after a seizure"},
            {"term": "Wearable Detection", "definition": "Automated seizure detection via smartwatch or wrist-worn accelerometer/HR sensor"},
            {"term": "EMS Dispatch", "definition": "Emergency Medical Services activation (911/ambulance) for events exceeding caregiver response capability"},
            {"term": "Location Sharing", "definition": "GPS coordinates transmitted with SOS alert to help responders locate the patient"},
            {"term": "Contact Verification", "definition": "Periodic confirmation that emergency contact information is current and reachable"}
        ],
        "event_types": [
            {"type": "manual-sos", "description": "Patient manually activated SOS button on phone or wearable"},
            {"type": "panic-button", "description": "Dedicated hardware panic button pressed by patient or bystander"},
            {"type": "fall-detected", "description": "Accelerometer-based fall detection triggered by sudden impact"},
            {"type": "seizure-alert", "description": "Wearable or AI system detected possible seizure activity"},
            {"type": "medication-emergency", "description": "Alert triggered for missed critical medication or adverse drug reaction"}
        ],
        "trigger_methods": [
            {"method": "wearable-auto", "description": "Automatic detection by smartwatch or wearable sensor (accelerometer, HR)"},
            {"method": "caregiver-initiated", "description": "Caregiver manually triggered alert after witnessing event"},
            {"method": "app-button", "description": "Patient pressed SOS button in mobile application"},
            {"method": "voice-command", "description": "Voice-activated SOS via smart speaker or phone assistant"}
        ],
        "outcomes": [
            {"outcome": "caregiver-responded", "description": "Emergency contact or caregiver responded and managed the situation"},
            {"outcome": "er-visit", "description": "Patient was taken to emergency room for evaluation"},
            {"outcome": "false-alarm", "description": "No actual emergency — accidental trigger or device error"},
            {"outcome": "resolved-home", "description": "Event resolved at home without hospital visit"},
            {"outcome": "ems-dispatched", "description": "Emergency Medical Services (ambulance) was dispatched"}
        ],
        "preparedness_metrics": [
            {"metric": "Contact Coverage", "target": "100% of patients with >= 1 verified emergency contact"},
            {"metric": "Response Time", "target": "< 120 seconds average from alert to acknowledgement"},
            {"metric": "Location Sharing", "target": "> 90% of events include GPS coordinates"},
            {"metric": "Contact Freshness", "target": "All contacts verified within last 6 months"},
            {"metric": "False Alarm Rate", "target": "< 15% of total events"}
        ]
    }
