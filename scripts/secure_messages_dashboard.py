"""Secure Messages Dashboard — backend analytics for secure_messages table."""
import sqlite3, os

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')

def _conn():
    return sqlite3.connect(DB)

def overview():
    conn = _conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    total_messages = c.execute("SELECT COUNT(*) FROM secure_messages").fetchone()[0]
    total_patients = c.execute("SELECT COUNT(DISTINCT patient_id) FROM secure_messages").fetchone()[0]
    inbound_count = c.execute("SELECT COUNT(*) FROM secure_messages WHERE direction='inbound'").fetchone()[0]
    outbound_count = c.execute("SELECT COUNT(*) FROM secure_messages WHERE direction='outbound'").fetchone()[0]
    unread_count = c.execute("SELECT COUNT(*) FROM secure_messages WHERE read_status='unread'").fetchone()[0]
    unread_rate = round(unread_count / total_messages * 100, 1) if total_messages else 0
    avg_response_time = c.execute("SELECT ROUND(AVG(response_time_hours),2) FROM secure_messages WHERE response_time_hours IS NOT NULL").fetchone()[0]
    urgent_count = c.execute("SELECT COUNT(*) FROM secure_messages WHERE priority='urgent'").fetchone()[0]
    high_priority_count = c.execute("SELECT COUNT(*) FROM secure_messages WHERE priority='high'").fetchone()[0]

    direction_dist = [dict(r) for r in c.execute(
        "SELECT direction, COUNT(*) AS count FROM secure_messages GROUP BY direction ORDER BY count DESC")]

    category_dist = [dict(r) for r in c.execute(
        "SELECT category, COUNT(*) AS count FROM secure_messages GROUP BY category ORDER BY count DESC")]

    priority_dist = [dict(r) for r in c.execute(
        "SELECT priority, COUNT(*) AS count FROM secure_messages GROUP BY priority ORDER BY count DESC")]

    read_status_dist = [dict(r) for r in c.execute(
        "SELECT read_status AS status, COUNT(*) AS count FROM secure_messages GROUP BY read_status ORDER BY count DESC")]

    monthly_trend = [dict(r) for r in c.execute("""
        SELECT SUBSTR(created_at,1,7) AS month,
               COUNT(*) AS messages,
               SUM(CASE WHEN direction='inbound' THEN 1 ELSE 0 END) AS inbound,
               SUM(CASE WHEN direction='outbound' THEN 1 ELSE 0 END) AS outbound
        FROM secure_messages GROUP BY month ORDER BY month
    """)]

    avg_response_by_priority = [dict(r) for r in c.execute("""
        SELECT priority, ROUND(AVG(response_time_hours),2) AS avg_response_time
        FROM secure_messages WHERE response_time_hours IS NOT NULL
        GROUP BY priority ORDER BY avg_response_time
    """)]

    category_by_direction = [dict(r) for r in c.execute("""
        SELECT category,
               SUM(CASE WHEN direction='inbound' THEN 1 ELSE 0 END) AS inbound,
               SUM(CASE WHEN direction='outbound' THEN 1 ELSE 0 END) AS outbound
        FROM secure_messages GROUP BY category ORDER BY category
    """)]

    conn.close()
    return {
        "kpis": {
            "total_messages": total_messages,
            "total_patients": total_patients,
            "inbound_count": inbound_count,
            "outbound_count": outbound_count,
            "unread_count": unread_count,
            "unread_rate": unread_rate,
            "avg_response_time_hours": avg_response_time,
            "urgent_count": urgent_count,
            "high_priority_count": high_priority_count,
        },
        "direction_dist": direction_dist,
        "category_dist": category_dist,
        "priority_dist": priority_dist,
        "read_status_dist": read_status_dist,
        "monthly_trend": monthly_trend,
        "avg_response_by_priority": avg_response_by_priority,
        "category_by_direction": category_by_direction,
    }

def breakdown():
    conn = _conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    messages = [dict(r) for r in c.execute(
        "SELECT * FROM secure_messages ORDER BY created_at DESC LIMIT 200")]

    by_patient = [dict(r) for r in c.execute("""
        SELECT patient_id,
               COUNT(*) AS messages,
               SUM(CASE WHEN direction='inbound' THEN 1 ELSE 0 END) AS inbound,
               SUM(CASE WHEN direction='outbound' THEN 1 ELSE 0 END) AS outbound,
               SUM(CASE WHEN read_status='unread' THEN 1 ELSE 0 END) AS unread,
               ROUND(AVG(response_time_hours),2) AS avg_response_time,
               SUM(CASE WHEN priority='urgent' THEN 1 ELSE 0 END) AS urgent_count
        FROM secure_messages GROUP BY patient_id ORDER BY messages DESC
    """)]

    by_category = [dict(r) for r in c.execute("""
        SELECT category,
               COUNT(*) AS total,
               SUM(CASE WHEN direction='inbound' THEN 1 ELSE 0 END) AS inbound,
               SUM(CASE WHEN direction='outbound' THEN 1 ELSE 0 END) AS outbound,
               ROUND(AVG(response_time_hours),2) AS avg_response_time,
               SUM(CASE WHEN read_status='unread' THEN 1 ELSE 0 END) AS unread_count
        FROM secure_messages GROUP BY category ORDER BY total DESC
    """)]

    conn.close()
    return {
        "messages": messages,
        "by_patient": by_patient,
        "by_category": by_category,
    }

def definitions():
    return {
        "title": "Secure Messages Dashboard — Definitions",
        "concepts": [
            {"name": "Direction", "description": "Whether the message was sent by the patient (inbound) or by the clinical team (outbound). Tracks communication flow and patient engagement patterns."},
            {"name": "Category", "description": "Classification of the message content: side-effect-report, symptom-report, urgent, test-results, medication-question, appointment-request, general-inquiry, prescription-refill. Drives triage routing and response priority."},
            {"name": "Subject", "description": "Free-text subject line of the secure message. 24 distinct values across the dataset, summarising the message topic for quick clinician review."},
            {"name": "Read Status", "description": "Whether the message has been opened by the recipient: read or unread. Unread rate is a key metric for inbox management and patient communication SLAs."},
            {"name": "Response Time (hours)", "description": "Elapsed time in hours between message receipt and clinician response. Nullable when no response has been sent yet. Lower values indicate faster patient engagement."},
            {"name": "Priority", "description": "Urgency level assigned to the message: low, normal, high, urgent. Determines triage order and escalation pathways. Urgent messages require immediate clinical attention."},
            {"name": "Message Preview", "description": "Truncated preview of the message body content. Used for at-a-glance review without opening the full message thread."},
        ],
        "data_sources": [
            "secure_messages table — 170 messages, 30 patients",
            "Patient portal secure messaging system",
        ],
    }

if __name__ == "__main__":
    import json
    print(json.dumps(overview(), indent=2))
