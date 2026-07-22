"""Camera Monitoring Sessions Dashboard — backend analytics for camera_monitoring_sessions table."""
import sqlite3, os

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')

def _conn():
    return sqlite3.connect(DB)

def overview():
    conn = _conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    total_sessions = c.execute("SELECT COUNT(*) FROM camera_monitoring_sessions").fetchone()[0]
    total_patients = c.execute("SELECT COUNT(DISTINCT patient_id) FROM camera_monitoring_sessions").fetchone()[0]
    total_seizure_events = c.execute("SELECT COALESCE(SUM(seizure_events),0) FROM camera_monitoring_sessions").fetchone()[0]
    total_movement_events = c.execute("SELECT COALESCE(SUM(movement_events),0) FROM camera_monitoring_sessions").fetchone()[0]
    total_false_alarms = c.execute("SELECT COALESCE(SUM(false_alarms),0) FROM camera_monitoring_sessions").fetchone()[0]
    avg_duration = c.execute("SELECT ROUND(AVG(duration_hours),2) FROM camera_monitoring_sessions").fetchone()[0]
    avg_response_time = c.execute("SELECT ROUND(AVG(response_time_seconds),1) FROM camera_monitoring_sessions WHERE response_time_seconds IS NOT NULL").fetchone()[0]

    alert_count = c.execute("SELECT COUNT(*) FROM camera_monitoring_sessions WHERE alert_sent=1").fetchone()[0]
    alert_rate = round(alert_count / total_sessions * 100, 1) if total_sessions else 0

    total_events_detected = c.execute("SELECT COALESCE(SUM(events_detected),0) FROM camera_monitoring_sessions").fetchone()[0]
    seizure_detection_rate = round(total_seizure_events / total_events_detected * 100, 1) if total_events_detected else 0

    night_vision_count = c.execute("SELECT COUNT(*) FROM camera_monitoring_sessions WHERE night_vision=1").fetchone()[0]
    night_vision_rate = round(night_vision_count / total_sessions * 100, 1) if total_sessions else 0

    location_dist = [dict(r) for r in c.execute(
        "SELECT camera_location AS location, COUNT(*) AS count FROM camera_monitoring_sessions GROUP BY camera_location ORDER BY count DESC")]

    quality_dist = [dict(r) for r in c.execute(
        "SELECT recording_quality AS quality, COUNT(*) AS count FROM camera_monitoring_sessions GROUP BY recording_quality ORDER BY count DESC")]

    session_type_dist = [dict(r) for r in c.execute(
        "SELECT session_type AS type, COUNT(*) AS count FROM camera_monitoring_sessions GROUP BY session_type ORDER BY count DESC")]

    status_dist = [dict(r) for r in c.execute(
        "SELECT status, COUNT(*) AS count FROM camera_monitoring_sessions GROUP BY status ORDER BY count DESC")]

    monthly_trend = [dict(r) for r in c.execute("""
        SELECT SUBSTR(session_date,1,7) AS month,
               COUNT(*) AS sessions,
               COALESCE(SUM(seizure_events),0) AS seizure_events,
               COALESCE(SUM(movement_events),0) AS movement_events,
               COALESCE(SUM(false_alarms),0) AS false_alarms
        FROM camera_monitoring_sessions GROUP BY month ORDER BY month
    """)]

    conn.close()
    return {
        "kpis": {
            "total_sessions": total_sessions,
            "total_patients": total_patients,
            "total_seizure_events": total_seizure_events,
            "total_movement_events": total_movement_events,
            "total_false_alarms": total_false_alarms,
            "avg_duration_hours": avg_duration,
            "avg_response_time_seconds": avg_response_time,
            "alert_rate": alert_rate,
            "seizure_detection_rate": seizure_detection_rate,
            "night_vision_rate": night_vision_rate,
        },
        "location_dist": location_dist,
        "quality_dist": quality_dist,
        "session_type_dist": session_type_dist,
        "status_dist": status_dist,
        "monthly_trend": monthly_trend,
    }

def breakdown():
    conn = _conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    sessions = [dict(r) for r in c.execute(
        "SELECT * FROM camera_monitoring_sessions ORDER BY session_date DESC LIMIT 200")]

    by_patient = [dict(r) for r in c.execute("""
        SELECT patient_id,
               COUNT(*) AS sessions,
               ROUND(SUM(duration_hours),2) AS total_duration,
               COALESCE(SUM(seizure_events),0) AS seizure_events,
               COALESCE(SUM(movement_events),0) AS movement_events,
               COALESCE(SUM(false_alarms),0) AS false_alarms,
               ROUND(AVG(response_time_seconds),1) AS avg_response_time
        FROM camera_monitoring_sessions GROUP BY patient_id ORDER BY sessions DESC
    """)]

    by_location = [dict(r) for r in c.execute("""
        SELECT camera_location AS location,
               COUNT(*) AS sessions,
               COALESCE(SUM(seizure_events),0) AS seizure_events,
               COALESCE(SUM(movement_events),0) AS movement_events,
               COALESCE(SUM(false_alarms),0) AS false_alarms,
               ROUND(AVG(duration_hours),2) AS avg_duration,
               SUM(CASE WHEN recording_quality='good' THEN 1 ELSE 0 END) AS quality_good,
               SUM(CASE WHEN recording_quality='fair' THEN 1 ELSE 0 END) AS quality_fair,
               SUM(CASE WHEN recording_quality='poor' THEN 1 ELSE 0 END) AS quality_poor,
               SUM(CASE WHEN recording_quality='excellent' THEN 1 ELSE 0 END) AS quality_excellent
        FROM camera_monitoring_sessions GROUP BY camera_location ORDER BY sessions DESC
    """)]

    # Restructure quality_breakdown into nested dict
    for loc in by_location:
        loc["quality_breakdown"] = {
            "good": loc.pop("quality_good"),
            "fair": loc.pop("quality_fair"),
            "poor": loc.pop("quality_poor"),
            "excellent": loc.pop("quality_excellent"),
        }

    conn.close()
    return {
        "sessions": sessions,
        "by_patient": by_patient,
        "by_location": by_location,
    }

def definitions():
    return {
        "title": "Camera Monitoring Sessions Dashboard — Definitions",
        "concepts": [
            {"name": "Camera Location", "description": "Physical placement of the monitoring camera. Options: emu_room (Epilepsy Monitoring Unit — inpatient video-EEG suite), living_room, bedroom, hallway, kitchen. Location affects event detection accuracy and clinical relevance."},
            {"name": "Session Type", "description": "Recording mode: continuous (24/7 uninterrupted recording), event_triggered (camera activates on detected motion/sound), scheduled (recording during predefined windows, e.g. nocturnal monitoring)."},
            {"name": "Duration Hours", "description": "Total length of the monitoring session in hours. Longer sessions capture more events but generate larger storage and review burdens."},
            {"name": "Events Detected", "description": "Total number of events flagged by the camera's motion/AI detection algorithm during the session, including seizures, movement, and false alarms."},
            {"name": "Seizure Events", "description": "Number of events classified as epileptic seizures by the detection algorithm or confirmed by clinical review. Key safety metric for home and EMU monitoring."},
            {"name": "Movement Events", "description": "Non-seizure movement events detected (e.g. restlessness, falls, repositioning). Helps distinguish seizure-related from normal activity."},
            {"name": "False Alarms", "description": "Events flagged by the detection algorithm that were not clinically significant upon review. High false alarm rates reduce clinician trust and increase alert fatigue."},
            {"name": "Alert Sent", "description": "Whether the system dispatched a real-time alert (SMS, push notification, pager) to a caregiver or clinician for this session (1=yes, 0=no)."},
            {"name": "Alert Rate", "description": "Percentage of sessions where at least one alert was sent. Tracks system sensitivity and potential alert fatigue burden."},
            {"name": "Response Time (seconds)", "description": "Time in seconds between alert dispatch and caregiver/clinician acknowledgement or intervention. Critical for evaluating emergency response effectiveness."},
            {"name": "Recording Quality", "description": "Technical quality of the video recording: excellent, good, fair, or poor. Affected by lighting, camera angle, occlusion, and connectivity. Poor quality reduces seizure detection accuracy."},
            {"name": "Night Vision", "description": "Whether infrared/night-vision mode was active during the session (1=yes, 0=no). Essential for nocturnal seizure detection where room lighting is absent."},
            {"name": "Night Vision Rate", "description": "Percentage of all sessions using night-vision mode. Indicates coverage of nocturnal monitoring across the patient cohort."},
            {"name": "Seizure Detection Rate", "description": "Ratio of seizure events to total events detected, expressed as a percentage. Measures the proportion of detected events that are clinically meaningful seizures versus noise."},
            {"name": "Status", "description": "Session outcome: completed (finished normally), active (currently recording), interrupted (stopped early due to technical/patient factors), failed (session did not record usable data)."},
        ],
        "data_sources": [
            "camera_monitoring_sessions table — 78 sessions, 27 patients",
            "Video-EEG and home camera monitoring systems",
            "Automated seizure detection algorithms with clinical review confirmation",
        ],
    }

if __name__ == "__main__":
    import json
    print(json.dumps(overview(), indent=2))
