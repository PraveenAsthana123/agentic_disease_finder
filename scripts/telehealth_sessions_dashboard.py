"""Telehealth Sessions Dashboard — telehealth visit analytics from clinical.db.

Tracks patient telehealth sessions including video visits, phone consults,
async messaging, and remote monitoring reviews across providers and platforms.

Sources:
- telehealth_sessions table (id, patient_id, session_date, session_type,
  provider_name, duration_minutes, connection_quality, patient_satisfaction,
  technical_issues, platform, notes, created_at)
"""

import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


# ──────────────────────────────────────────────────────────────
#  /api/telehealth-sessions/overview
# ──────────────────────────────────────────────────────────────

def overview():
    """High-level telehealth session metrics."""
    conn = _conn()
    cur = conn.cursor()

    # KPIs
    cur.execute("SELECT COUNT(*) FROM telehealth_sessions")
    total_sessions = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM telehealth_sessions")
    total_patients = cur.fetchone()[0] or 0

    cur.execute("SELECT ROUND(AVG(duration_minutes), 1) FROM telehealth_sessions")
    avg_duration = cur.fetchone()[0] or 0.0

    cur.execute("SELECT ROUND(AVG(patient_satisfaction), 1) FROM telehealth_sessions")
    avg_satisfaction = cur.fetchone()[0] or 0.0

    cur.execute("SELECT SUM(technical_issues) FROM telehealth_sessions")
    total_tech_issues = cur.fetchone()[0] or 0
    tech_issue_rate_pct = round(total_tech_issues / total_sessions * 100, 1) if total_sessions else 0.0

    cur.execute("SELECT COUNT(DISTINCT provider_name) FROM telehealth_sessions")
    total_providers = cur.fetchone()[0] or 0

    kpis = {
        "total_sessions": total_sessions,
        "total_patients": total_patients,
        "avg_duration": avg_duration,
        "avg_satisfaction": avg_satisfaction,
        "tech_issue_rate_pct": tech_issue_rate_pct,
        "total_providers": total_providers,
    }

    # Session type distribution
    cur.execute(
        "SELECT session_type AS name, COUNT(*) AS count FROM telehealth_sessions "
        "GROUP BY session_type ORDER BY count DESC"
    )
    session_type_distribution = _dict_rows(cur)

    # Platform distribution
    cur.execute(
        "SELECT platform AS name, COUNT(*) AS count FROM telehealth_sessions "
        "GROUP BY platform ORDER BY count DESC"
    )
    platform_distribution = _dict_rows(cur)

    # Connection quality distribution
    cur.execute(
        "SELECT connection_quality AS name, COUNT(*) AS count FROM telehealth_sessions "
        "GROUP BY connection_quality ORDER BY count DESC"
    )
    connection_quality_distribution = _dict_rows(cur)

    # Monthly trend
    cur.execute(
        "SELECT SUBSTR(session_date, 1, 7) AS month, "
        "COUNT(*) AS sessions, "
        "ROUND(AVG(duration_minutes), 1) AS avg_duration, "
        "ROUND(AVG(patient_satisfaction), 1) AS avg_satisfaction "
        "FROM telehealth_sessions GROUP BY month ORDER BY month"
    )
    monthly_trend = _dict_rows(cur)

    # Provider workload
    cur.execute(
        "SELECT provider_name AS provider, "
        "COUNT(*) AS sessions, "
        "ROUND(AVG(duration_minutes), 1) AS avg_duration, "
        "ROUND(AVG(patient_satisfaction), 1) AS avg_satisfaction "
        "FROM telehealth_sessions GROUP BY provider_name ORDER BY sessions DESC"
    )
    provider_workload = _dict_rows(cur)

    conn.close()
    return {
        "available": True,
        "kpis": kpis,
        "session_type_distribution": session_type_distribution,
        "platform_distribution": platform_distribution,
        "connection_quality_distribution": connection_quality_distribution,
        "monthly_trend": monthly_trend,
        "provider_workload": provider_workload,
    }


# ──────────────────────────────────────────────────────────────
#  /api/telehealth-sessions/breakdown
# ──────────────────────────────────────────────────────────────

def breakdown():
    """Per-patient detail, quality issues, recent sessions."""
    conn = _conn()
    cur = conn.cursor()

    # Poor connection sessions
    cur.execute(
        "SELECT id, patient_id, session_date, session_type, provider_name, "
        "duration_minutes, connection_quality, patient_satisfaction, "
        "technical_issues, platform, notes "
        "FROM telehealth_sessions "
        "WHERE connection_quality = 'poor' "
        "ORDER BY session_date DESC LIMIT 20"
    )
    poor_connection_sessions = _dict_rows(cur)

    # Low satisfaction sessions (satisfaction <= 2)
    cur.execute(
        "SELECT id, patient_id, session_date, session_type, provider_name, "
        "duration_minutes, connection_quality, patient_satisfaction, "
        "technical_issues, platform, notes "
        "FROM telehealth_sessions "
        "WHERE patient_satisfaction <= 2 "
        "ORDER BY session_date DESC LIMIT 20"
    )
    low_satisfaction_sessions = _dict_rows(cur)

    # Per-patient summary
    cur.execute(
        "SELECT patient_id, "
        "COUNT(*) AS sessions, "
        "ROUND(AVG(duration_minutes), 1) AS avg_duration, "
        "ROUND(AVG(patient_satisfaction), 1) AS avg_satisfaction, "
        "SUM(technical_issues) AS tech_issues, "
        "( SELECT platform FROM telehealth_sessions t2 "
        "  WHERE t2.patient_id = t1.patient_id "
        "  GROUP BY platform ORDER BY COUNT(*) DESC LIMIT 1 "
        ") AS most_used_platform "
        "FROM telehealth_sessions t1 "
        "GROUP BY patient_id ORDER BY patient_id"
    )
    per_patient_summary = _dict_rows(cur)

    # Recent sessions (last 20)
    cur.execute(
        "SELECT id, patient_id, session_date, session_type, provider_name, "
        "duration_minutes, connection_quality, patient_satisfaction, "
        "technical_issues, platform, notes "
        "FROM telehealth_sessions "
        "ORDER BY session_date DESC LIMIT 20"
    )
    recent_sessions = _dict_rows(cur)

    # Provider by type
    cur.execute(
        "SELECT provider_name AS provider, session_type, COUNT(*) AS count "
        "FROM telehealth_sessions "
        "GROUP BY provider_name, session_type "
        "ORDER BY provider_name, count DESC"
    )
    provider_by_type = _dict_rows(cur)

    conn.close()
    return {
        "poor_connection_sessions": poor_connection_sessions,
        "low_satisfaction_sessions": low_satisfaction_sessions,
        "per_patient_summary": per_patient_summary,
        "recent_sessions": recent_sessions,
        "provider_by_type": provider_by_type,
    }


# ──────────────────────────────────────────────────────────────
#  /api/telehealth-sessions/definitions
# ──────────────────────────────────────────────────────────────

def definitions():
    """Session type definitions, field descriptions, clinical notes, glossary."""
    return {
        "session_types": {
            "video-visit": "Synchronous face-to-face video consultation between patient and "
                           "provider. Highest fidelity telehealth modality, enabling visual "
                           "assessment of patient presentation and seizure semiology.",
            "phone-consult": "Voice-only telephone consultation. Suitable for medication reviews, "
                             "symptom check-ins, and follow-up discussions where visual assessment "
                             "is not required.",
            "async-message": "Asynchronous secure messaging between patient and provider. Used for "
                             "non-urgent queries, medication refill requests, lab result reviews, "
                             "and care plan clarifications.",
            "remote-monitoring-review": "Provider review of remotely collected patient data "
                                        "including EEG telemetry, seizure diary entries, "
                                        "wearable device readings, and medication adherence logs.",
        },
        "connection_quality_levels": {
            "excellent": "Stable, high-bandwidth connection with no interruptions. Full HD video "
                         "and clear audio throughout the session.",
            "good": "Generally stable connection with minor, brief quality fluctuations. Audio "
                    "remains clear; video may have occasional pixelation.",
            "fair": "Intermittent connectivity issues affecting session flow. Possible audio "
                    "dropouts or video freezing requiring brief pauses.",
            "poor": "Significant connectivity problems impacting clinical utility. Frequent "
                    "audio/video disruptions, potential session restart or fallback to phone.",
        },
        "platforms": {
            "Zoom Health": "HIPAA-compliant Zoom for Healthcare platform with BAA, end-to-end "
                           "encryption, and waiting room functionality.",
            "Teams": "Microsoft Teams for Healthcare with HIPAA compliance, integrated EHR "
                     "connectors, and secure file sharing capabilities.",
            "Doxy.me": "Browser-based telehealth platform requiring no downloads. HIPAA-compliant "
                       "with virtual waiting room and simple patient access.",
            "In-house Portal": "Custom-built patient portal integrated with the clinic's EHR "
                               "system. Provides seamless access to records during sessions.",
        },
        "field_descriptions": {
            "session_date": "Date of the telehealth session in ISO 8601 format (YYYY-MM-DD).",
            "session_type": "Category of telehealth encounter: video-visit, phone-consult, "
                            "async-message, or remote-monitoring-review.",
            "provider_name": "Name of the healthcare provider conducting or reviewing the session.",
            "duration_minutes": "Total duration of the session in minutes. Async messages reflect "
                                "estimated provider review time.",
            "connection_quality": "Technical quality of the connection: excellent, good, fair, or "
                                  "poor. Assessed by the platform or provider.",
            "patient_satisfaction": "Patient-reported satisfaction score on a 1-5 Likert scale "
                                    "(1 = very dissatisfied, 5 = very satisfied).",
            "technical_issues": "Binary flag (0/1) indicating whether technical issues occurred "
                                "during the session (audio/video failures, disconnections).",
            "platform": "Telehealth platform used for the session.",
            "notes": "Free-text clinical or technical notes from the session.",
        },
        "clinical_notes": [
            "Telehealth is particularly valuable for epilepsy patients who may have driving "
            "restrictions due to seizure recency, reducing transportation barriers to care.",
            "Video visits allow providers to observe for subtle signs of medication side effects "
            "(e.g., tremor, nystagmus, cognitive slowing) that phone calls would miss.",
            "Remote monitoring reviews of EEG telemetry and seizure diaries enable earlier "
            "detection of treatment failure or breakthrough seizure patterns.",
            "Patients with frequent seizures benefit from async messaging for non-urgent "
            "communication, reducing the cognitive burden of scheduled appointments.",
            "Technical issues during telehealth sessions can compromise clinical assessment "
            "quality; fallback protocols (phone callback, rescheduling) should be established.",
        ],
        "glossary": [
            {"term": "Telehealth", "definition": "Delivery of healthcare services via "
             "telecommunications technology, encompassing video, phone, messaging, and "
             "remote monitoring modalities."},
            {"term": "Synchronous Telehealth", "definition": "Real-time communication between "
             "patient and provider, including video visits and phone consultations."},
            {"term": "Asynchronous Telehealth", "definition": "Store-and-forward communication "
             "where messages, images, or data are sent and reviewed at different times."},
            {"term": "Remote Patient Monitoring", "definition": "Collection and transmission of "
             "patient health data (EEG, vitals, seizure logs) from the patient's location to "
             "the provider for review."},
            {"term": "HIPAA Compliance", "definition": "Adherence to the Health Insurance "
             "Portability and Accountability Act standards for protecting patient health "
             "information during telehealth encounters."},
            {"term": "BAA", "definition": "Business Associate Agreement — a contract between a "
             "healthcare provider and a technology vendor ensuring HIPAA-compliant handling of "
             "protected health information."},
            {"term": "Patient Satisfaction Score", "definition": "A standardized 1-5 Likert scale "
             "rating of the patient's experience with the telehealth session, including ease of "
             "use, communication quality, and clinical value."},
            {"term": "Connection Quality", "definition": "Assessment of the technical reliability "
             "of the telehealth connection, factoring in bandwidth, latency, packet loss, and "
             "audio/video fidelity."},
            {"term": "EHR Integration", "definition": "Electronic Health Record integration — the "
             "ability of the telehealth platform to read from and write to the patient's medical "
             "record during sessions."},
            {"term": "Digital Divide", "definition": "Disparities in access to reliable internet "
             "and technology that can limit telehealth adoption, particularly in rural or "
             "underserved epilepsy populations."},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== Overview ===")
    print(json.dumps(overview(), indent=2))
    print("\n=== Breakdown (summary) ===")
    b = breakdown()
    print(f"Patients: {len(b['per_patient_summary'])}, "
          f"Poor connections: {len(b['poor_connection_sessions'])}")
    print(f"Low satisfaction: {len(b['low_satisfaction_sessions'])}, "
          f"Recent: {len(b['recent_sessions'])}")
    print(f"Provider-by-type combos: {len(b['provider_by_type'])}")
    print("\n=== Definitions ===")
    d = definitions()
    print(f"Session types: {len(d['session_types'])}, "
          f"Quality levels: {len(d['connection_quality_levels'])}")
    print(f"Platforms: {len(d['platforms'])}, "
          f"Fields: {len(d['field_descriptions'])}")
    print(f"Clinical notes: {len(d['clinical_notes'])}, "
          f"Glossary: {len(d['glossary'])}")
