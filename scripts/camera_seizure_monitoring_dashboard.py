"""
Neuro AI Ecosystem — Camera Seizure Monitoring Dashboard
========================================================
Video camera seizure monitoring analytics from camera_monitoring_sessions table.

Camera locations: bedroom, living_room, emu_room, hallway, kitchen
Session types: continuous, scheduled, event_triggered
Recording quality: excellent, good, fair, poor
Statuses: completed, active, interrupted, failed

Real data: camera_monitoring_sessions (~80 rows, 27 patients)
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
    """Camera seizure monitoring overview — session counts, location distribution,
    seizure detection rate, response time stats, recording quality."""
    conn = _conn()
    cur = conn.cursor()

    # Total sessions
    cur.execute("SELECT COUNT(*) FROM camera_monitoring_sessions")
    total_sessions = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM camera_monitoring_sessions")
    unique_patients = cur.fetchone()[0]

    # Camera location distribution
    cur.execute("""
        SELECT camera_location, COUNT(*) cnt
        FROM camera_monitoring_sessions
        GROUP BY camera_location
        ORDER BY cnt DESC
    """)
    location_dist = {r[0]: r[1] for r in cur.fetchall()}

    # Session type distribution
    cur.execute("""
        SELECT session_type, COUNT(*) cnt
        FROM camera_monitoring_sessions
        GROUP BY session_type
        ORDER BY cnt DESC
    """)
    session_type_dist = {r[0]: r[1] for r in cur.fetchall()}

    # Status distribution
    cur.execute("""
        SELECT status, COUNT(*) cnt
        FROM camera_monitoring_sessions
        GROUP BY status
        ORDER BY cnt DESC
    """)
    status_dist = {r[0]: r[1] for r in cur.fetchall()}

    # Recording quality distribution
    cur.execute("""
        SELECT recording_quality, COUNT(*) cnt
        FROM camera_monitoring_sessions
        GROUP BY recording_quality
        ORDER BY cnt DESC
    """)
    quality_dist = {r[0]: r[1] for r in cur.fetchall()}

    # Seizure detection rate
    cur.execute("""
        SELECT COUNT(*) FROM camera_monitoring_sessions WHERE seizure_events > 0
    """)
    sessions_with_seizures = cur.fetchone()[0]
    seizure_detection_rate = round(sessions_with_seizures / max(total_sessions, 1) * 100, 1)

    cur.execute("""
        SELECT SUM(seizure_events) FROM camera_monitoring_sessions
    """)
    total_seizure_events = cur.fetchone()[0] or 0

    # Average events per session
    cur.execute("""
        SELECT ROUND(AVG(events_detected), 1) avg_events,
               ROUND(AVG(seizure_events), 2) avg_seizure,
               ROUND(AVG(movement_events), 2) avg_movement,
               ROUND(AVG(false_alarms), 2) avg_false_alarms
        FROM camera_monitoring_sessions
    """)
    r = cur.fetchone()
    avg_events = {
        "avg_events_detected": r[0],
        "avg_seizure_events": r[1],
        "avg_movement_events": r[2],
        "avg_false_alarms": r[3]
    }

    # Average response time (only for sessions with alerts)
    cur.execute("""
        SELECT ROUND(AVG(response_time_seconds), 0) avg_rt,
               MIN(response_time_seconds) min_rt,
               MAX(response_time_seconds) max_rt,
               ROUND(AVG(CASE WHEN response_time_seconds <= 120 THEN 1.0 ELSE 0.0 END) * 100, 1) pct_under_2min
        FROM camera_monitoring_sessions
        WHERE alert_sent = 1 AND response_time_seconds IS NOT NULL
    """)
    r = cur.fetchone()
    response_stats = {
        "avg_seconds": r[0],
        "min_seconds": r[1],
        "max_seconds": r[2],
        "pct_under_2min": r[3]
    }

    # Night vision usage rate
    cur.execute("""
        SELECT ROUND(AVG(CASE WHEN night_vision = 1 THEN 1.0 ELSE 0.0 END) * 100, 1)
        FROM camera_monitoring_sessions
    """)
    night_vision_pct = cur.fetchone()[0]

    # Alert rate
    cur.execute("""
        SELECT ROUND(AVG(CASE WHEN alert_sent = 1 THEN 1.0 ELSE 0.0 END) * 100, 1)
        FROM camera_monitoring_sessions
    """)
    alert_rate_pct = cur.fetchone()[0]

    # Average duration
    cur.execute("""
        SELECT ROUND(AVG(duration_hours), 1) FROM camera_monitoring_sessions
    """)
    avg_duration = cur.fetchone()[0]

    # Monthly trend
    cur.execute("""
        SELECT SUBSTR(session_date, 1, 7) month,
               COUNT(*) sessions,
               SUM(seizure_events) seizure_events
        FROM camera_monitoring_sessions
        GROUP BY month
        ORDER BY month
    """)
    monthly_trend = _dict_rows(cur)

    conn.close()
    return {
        "total_sessions": total_sessions,
        "unique_patients": unique_patients,
        "camera_location_distribution": location_dist,
        "session_type_distribution": session_type_dist,
        "status_distribution": status_dist,
        "recording_quality_distribution": quality_dist,
        "seizure_detection_rate_pct": seizure_detection_rate,
        "total_seizure_events": total_seizure_events,
        "avg_events_per_session": avg_events,
        "response_time_stats": response_stats,
        "night_vision_usage_pct": night_vision_pct,
        "alert_rate_pct": alert_rate_pct,
        "avg_duration_hours": avg_duration,
        "monthly_trend": monthly_trend
    }


def breakdown():
    """Camera seizure monitoring breakdown — per-patient profiles, monthly trends,
    location analysis, recent/active sessions."""
    conn = _conn()
    cur = conn.cursor()

    # Per-patient summary
    cur.execute("""
        SELECT patient_id,
               COUNT(*) total_sessions,
               SUM(seizure_events) total_seizure_events,
               ROUND(AVG(duration_hours), 1) avg_duration,
               SUM(events_detected) total_events_detected,
               SUM(false_alarms) total_false_alarms,
               SUM(CASE WHEN alert_sent = 1 THEN 1 ELSE 0 END) alerts_sent
        FROM camera_monitoring_sessions
        GROUP BY patient_id
        ORDER BY total_seizure_events DESC
    """)
    patient_summary = _dict_rows(cur)

    # Add most-used camera location per patient
    for ps in patient_summary:
        cur.execute("""
            SELECT camera_location, COUNT(*) cnt
            FROM camera_monitoring_sessions
            WHERE patient_id = ?
            GROUP BY camera_location
            ORDER BY cnt DESC
            LIMIT 1
        """, (ps["patient_id"],))
        row = cur.fetchone()
        ps["most_used_location"] = row[0] if row else None

    # Monthly trend (sessions + seizure events by month)
    cur.execute("""
        SELECT SUBSTR(session_date, 1, 7) month,
               COUNT(*) sessions,
               SUM(seizure_events) seizure_events,
               SUM(events_detected) total_events,
               ROUND(AVG(duration_hours), 1) avg_duration
        FROM camera_monitoring_sessions
        GROUP BY month
        ORDER BY month
    """)
    monthly_trend = _dict_rows(cur)

    # Location x session_type cross-tab
    cur.execute("""
        SELECT camera_location, session_type, COUNT(*) cnt
        FROM camera_monitoring_sessions
        GROUP BY camera_location, session_type
        ORDER BY camera_location, session_type
    """)
    location_type_crosstab = _dict_rows(cur)

    # Recent sessions (most recent 30)
    cur.execute("""
        SELECT id, patient_id, session_date, camera_location, session_type,
               duration_hours, events_detected, seizure_events, movement_events,
               false_alarms, alert_sent, response_time_seconds, recording_quality,
               night_vision, status, notes
        FROM camera_monitoring_sessions
        ORDER BY session_date DESC
        LIMIT 30
    """)
    recent_sessions = _dict_rows(cur)

    # High-activity patients (most seizure events)
    cur.execute("""
        SELECT patient_id,
               SUM(seizure_events) total_seizure_events,
               COUNT(*) total_sessions,
               ROUND(AVG(duration_hours), 1) avg_duration
        FROM camera_monitoring_sessions
        GROUP BY patient_id
        HAVING total_seizure_events > 0
        ORDER BY total_seizure_events DESC
        LIMIT 10
    """)
    high_activity = _dict_rows(cur)

    # Active sessions
    cur.execute("""
        SELECT id, patient_id, session_date, camera_location, session_type,
               duration_hours, events_detected, seizure_events, recording_quality,
               night_vision, notes
        FROM camera_monitoring_sessions
        WHERE status = 'active'
        ORDER BY session_date DESC
    """)
    active_sessions = _dict_rows(cur)

    # Location analysis — stats per camera location
    cur.execute("""
        SELECT camera_location,
               COUNT(*) total_sessions,
               SUM(seizure_events) seizure_events,
               SUM(events_detected) total_events,
               ROUND(AVG(duration_hours), 1) avg_duration,
               ROUND(AVG(CASE WHEN night_vision = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) night_vision_pct,
               ROUND(AVG(false_alarms), 2) avg_false_alarms
        FROM camera_monitoring_sessions
        GROUP BY camera_location
        ORDER BY total_sessions DESC
    """)
    location_stats = _dict_rows(cur)

    conn.close()
    return {
        "patient_summary": patient_summary,
        "monthly_trend": monthly_trend,
        "location_type_crosstab": location_type_crosstab,
        "recent_sessions": recent_sessions,
        "high_activity_patients": high_activity,
        "active_sessions": active_sessions,
        "location_stats": location_stats
    }


def definitions():
    """Camera seizure monitoring definitions — clinical glossary, camera locations,
    session types, quality tiers, monitoring guidelines."""
    return {
        "glossary": [
            {"term": "EMU", "definition": "Epilepsy Monitoring Unit — specialized hospital unit for continuous video-EEG monitoring to localize seizure focus and classify seizure types"},
            {"term": "Nocturnal Seizure Monitoring", "definition": "Video/sensor monitoring during sleep to detect seizures that occur primarily at night, often unwitnessed and underreported"},
            {"term": "SUDEP", "definition": "Sudden Unexpected Death in Epilepsy — leading cause of death in refractory epilepsy; nocturnal monitoring is critical for prevention via early detection"},
            {"term": "Video-EEG", "definition": "Gold-standard diagnostic combining synchronized video recording with electroencephalography for seizure classification and surgical planning"},
            {"term": "False Alarm Rate", "definition": "Proportion of detected events that were not true seizures (pet movement, normal repositioning, device artifact); target < 15%"},
            {"term": "Sensitivity", "definition": "Proportion of true seizure events correctly detected by the monitoring system; target > 90% for clinical-grade systems"},
            {"term": "Specificity", "definition": "Proportion of non-seizure periods correctly identified as seizure-free; high specificity reduces caregiver alert fatigue"},
            {"term": "Tonic-Clonic Seizure", "definition": "Generalized convulsive seizure with stiffening and rhythmic jerking — the most reliably detected seizure type on video monitoring"},
            {"term": "Myoclonic Jerk", "definition": "Brief, shock-like involuntary muscle contraction; may be epileptic or non-epileptic, harder to distinguish on video alone"},
            {"term": "Post-Ictal Period", "definition": "Recovery phase after a seizure; patient may be confused, fatigued, or unresponsive — monitoring continues through this phase"},
            {"term": "Seizure Semiology", "definition": "Observable clinical features of a seizure (movement patterns, lateralization, automatisms) captured on video for classification"},
            {"term": "Alert Fatigue", "definition": "Desensitization of caregivers/staff to frequent alarms, leading to delayed response; minimized by reducing false alarm rates"}
        ],
        "camera_locations": [
            {"location": "bedroom", "description": "Primary location for nocturnal seizure monitoring; captures sleep-related seizures, night-time tonic-clonic events, and SUDEP risk assessment"},
            {"location": "living_room", "description": "Daytime activity monitoring; captures seizures during waking hours, falls, and post-ictal wandering"},
            {"location": "emu_room", "description": "Epilepsy Monitoring Unit hospital room; continuous video-EEG synchronized recording for seizure localization and presurgical evaluation"},
            {"location": "hallway", "description": "Transition area monitoring; captures falls, post-ictal wandering, and seizure events during ambulation"},
            {"location": "kitchen", "description": "Safety-critical area monitoring; captures seizures near heat sources, sharp objects, and fall hazards"}
        ],
        "session_types": [
            {"type": "continuous", "description": "Uninterrupted 24/7 recording, standard in EMU settings and for high-risk nocturnal monitoring at home"},
            {"type": "scheduled", "description": "Pre-programmed recording windows (e.g., sleep hours only or specific time blocks) for home monitoring"},
            {"type": "event_triggered", "description": "Recording activated by motion/sound detection or wearable seizure alert; captures events while minimizing storage"}
        ],
        "recording_quality_tiers": [
            {"quality": "excellent", "description": "Full HD/4K, clear lighting, no artifacts, suitable for clinical semiology analysis"},
            {"quality": "good", "description": "Adequate resolution and lighting for seizure detection and basic movement classification"},
            {"quality": "fair", "description": "Reduced clarity due to low light, distance, or compression; seizure detection possible but semiology limited"},
            {"quality": "poor", "description": "Significant quality issues (dark, blurred, dropped frames); event detection unreliable, re-positioning recommended"}
        ],
        "status_definitions": [
            {"status": "completed", "description": "Monitoring session finished as planned; recording saved and available for review"},
            {"status": "active", "description": "Session currently in progress; real-time monitoring and alerting active"},
            {"status": "interrupted", "description": "Session ended prematurely due to technical issue (connectivity, power, camera displacement)"},
            {"status": "failed", "description": "Session could not complete or recording was corrupted; no usable data captured"}
        ],
        "clinical_notes": [
            "Video-EEG monitoring is the gold standard for seizure classification and is required for epilepsy surgery evaluation (ILAE guidelines)",
            "Nocturnal monitoring is critical for SUDEP prevention — most SUDEP events occur during sleep and are unwitnessed without camera systems",
            "Home video monitoring complements clinical video-EEG by capturing seizures in the patient's natural environment over extended periods",
            "False alarm reduction is essential to prevent alert fatigue in caregivers; AI-assisted video analysis can improve specificity",
            "Camera monitoring should be paired with wearable sensors (accelerometer, HR) for multi-modal seizure detection",
            "Privacy-preserving approaches (on-device processing, encrypted storage, consent-based sharing) are required for home camera monitoring"
        ],
        "performance_targets": [
            {"metric": "Seizure Detection Sensitivity", "target": "> 90% for tonic-clonic seizures"},
            {"metric": "False Alarm Rate", "target": "< 15% of total detected events"},
            {"metric": "Response Time", "target": "< 120 seconds from seizure detection to caregiver alert"},
            {"metric": "Recording Uptime", "target": "> 95% for continuous monitoring sessions"},
            {"metric": "Night Vision Coverage", "target": "100% for bedroom monitoring sessions"}
        ]
    }
