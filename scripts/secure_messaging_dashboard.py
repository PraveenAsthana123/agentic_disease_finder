"""Secure Messaging Dashboard — clinical messaging analytics from clinical.db.

Tracks patient–provider secure message volume, response times, category
distribution, priority breakdown, read status, and per-patient communication
patterns.

Sources:
- secure_messages table (patient_id, direction, category, subject,
  message_preview, read_status, response_time_hours, priority, created_at)
- 170 records, 30 patients, 8 categories, 4 priority levels
"""

import sqlite3
import os

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql, params=(), default=0):
    try:
        cur.execute(sql, params)
        return cur.fetchone()[0]
    except Exception:
        return default


def _safe_rows(cur, sql, params=()):
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────
#  /api/secure-messaging/overview
# ──────────────────────────────────────────────────────────────

def overview():
    """Aggregate messaging health: total messages, direction split,
    category distribution, priority breakdown, response-time KPIs."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    total_messages = _safe(cur, "SELECT COUNT(*) FROM secure_messages")
    total_patients = _safe(cur, "SELECT COUNT(DISTINCT patient_id) FROM secure_messages")
    inbound_count = _safe(cur, "SELECT COUNT(*) FROM secure_messages WHERE direction = 'inbound'")
    outbound_count = _safe(cur, "SELECT COUNT(*) FROM secure_messages WHERE direction = 'outbound'")
    unread_count = _safe(cur, "SELECT COUNT(*) FROM secure_messages WHERE read_status = 'unread'")
    unread_pct = round(unread_count / total_messages * 100, 1) if total_messages else 0
    avg_response_hours = _safe(cur,
        "SELECT ROUND(AVG(response_time_hours), 1) FROM secure_messages "
        "WHERE response_time_hours IS NOT NULL")
    median_response_hours = _safe(cur,
        "SELECT ROUND(response_time_hours, 1) FROM secure_messages "
        "WHERE response_time_hours IS NOT NULL "
        "ORDER BY response_time_hours "
        "LIMIT 1 OFFSET (SELECT COUNT(*) FROM secure_messages "
        "WHERE response_time_hours IS NOT NULL) / 2")

    # Category distribution
    category_distribution = []
    for row in _safe_rows(cur,
            """SELECT category, COUNT(*) as cnt
               FROM secure_messages GROUP BY category ORDER BY cnt DESC"""):
        category_distribution.append({
            "category": row[0], "count": row[1],
            "pct": round(row[1] / total_messages * 100, 1) if total_messages else 0
        })

    # Priority breakdown
    priority_breakdown = []
    for row in _safe_rows(cur,
            """SELECT priority, COUNT(*) as cnt
               FROM secure_messages GROUP BY priority ORDER BY cnt DESC"""):
        priority_breakdown.append({
            "priority": row[0], "count": row[1],
            "pct": round(row[1] / total_messages * 100, 1) if total_messages else 0
        })

    # Direction split (for pie chart)
    direction_split = [
        {"direction": "Inbound", "count": inbound_count,
         "pct": round(inbound_count / total_messages * 100, 1) if total_messages else 0},
        {"direction": "Outbound", "count": outbound_count,
         "pct": round(outbound_count / total_messages * 100, 1) if total_messages else 0}
    ]

    # Monthly trend
    monthly_trend = []
    for row in _safe_rows(cur,
            """SELECT strftime('%Y-%m', created_at) as month, COUNT(*) as messages
               FROM secure_messages GROUP BY month ORDER BY month"""):
        monthly_trend.append({"month": row[0], "messages": row[1]})

    # Response time by priority
    response_by_priority = []
    for row in _safe_rows(cur,
            """SELECT priority,
                      ROUND(AVG(response_time_hours), 1) as avg_hours,
                      COUNT(*) as cnt
               FROM secure_messages
               WHERE response_time_hours IS NOT NULL
               GROUP BY priority ORDER BY avg_hours"""):
        response_by_priority.append({
            "priority": row[0], "avg_hours": row[1], "count": row[2]
        })

    conn.close()

    return {
        "available": True,
        "kpis": {
            "total_messages": total_messages,
            "total_patients": total_patients,
            "inbound_count": inbound_count,
            "outbound_count": outbound_count,
            "unread_count": unread_count,
            "unread_pct": unread_pct,
            "avg_response_hours": avg_response_hours,
            "median_response_hours": median_response_hours
        },
        "category_distribution": category_distribution,
        "priority_breakdown": priority_breakdown,
        "direction_split": direction_split,
        "monthly_trend": monthly_trend,
        "response_by_priority": response_by_priority
    }


# ──────────────────────────────────────────────────────────────
#  /api/secure-messaging/breakdown
# ──────────────────────────────────────────────────────────────

def breakdown():
    """Per-patient messaging summary, recent messages, category detail,
    and unread queue for triage."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # Per-patient summary (top 20 by message count)
    per_patient = []
    for row in _safe_rows(cur,
            """SELECT patient_id, COUNT(*) as messages,
                      SUM(CASE WHEN direction = 'inbound' THEN 1 ELSE 0 END) as inbound,
                      SUM(CASE WHEN direction = 'outbound' THEN 1 ELSE 0 END) as outbound,
                      SUM(CASE WHEN read_status = 'unread' THEN 1 ELSE 0 END) as unread,
                      ROUND(AVG(CASE WHEN response_time_hours IS NOT NULL
                                     THEN response_time_hours END), 1) as avg_response
               FROM secure_messages GROUP BY patient_id
               ORDER BY messages DESC LIMIT 20"""):
        per_patient.append({
            "patient_id": row[0], "messages": row[1],
            "inbound": row[2], "outbound": row[3],
            "unread": row[4], "avg_response_hours": row[5]
        })

    # Recent messages (last 20)
    recent_messages = []
    for row in _safe_rows(cur,
            """SELECT patient_id, direction, category, subject,
                      read_status, response_time_hours, priority, created_at
               FROM secure_messages ORDER BY created_at DESC LIMIT 20"""):
        recent_messages.append({
            "patient_id": row[0], "direction": row[1],
            "category": row[2], "subject": row[3],
            "read_status": row[4], "response_time_hours": row[5],
            "priority": row[6], "created_at": row[7]
        })

    # Category detail
    category_detail = []
    for row in _safe_rows(cur,
            """SELECT category,
                      COUNT(*) as total,
                      SUM(CASE WHEN direction = 'inbound' THEN 1 ELSE 0 END) as inbound,
                      SUM(CASE WHEN read_status = 'unread' THEN 1 ELSE 0 END) as unread,
                      ROUND(AVG(CASE WHEN response_time_hours IS NOT NULL
                                     THEN response_time_hours END), 1) as avg_response,
                      COUNT(DISTINCT patient_id) as unique_patients
               FROM secure_messages GROUP BY category
               ORDER BY total DESC"""):
        category_detail.append({
            "category": row[0], "total": row[1],
            "inbound": row[2], "unread": row[3],
            "avg_response_hours": row[4], "unique_patients": row[5]
        })

    # Unread queue (messages that haven't been read, by priority)
    unread_queue = []
    for row in _safe_rows(cur,
            """SELECT patient_id, direction, category, subject,
                      priority, created_at
               FROM secure_messages
               WHERE read_status = 'unread'
               ORDER BY
                 CASE priority
                   WHEN 'urgent' THEN 1 WHEN 'high' THEN 2
                   WHEN 'normal' THEN 3 WHEN 'low' THEN 4 ELSE 5
                 END, created_at DESC"""):
        unread_queue.append({
            "patient_id": row[0], "direction": row[1],
            "category": row[2], "subject": row[3],
            "priority": row[4], "created_at": row[5]
        })

    conn.close()

    return {
        "available": True,
        "per_patient": per_patient,
        "recent_messages": recent_messages,
        "category_detail": category_detail,
        "unread_queue": unread_queue
    }


# ──────────────────────────────────────────────────────────────
#  /api/secure-messaging/definitions
# ──────────────────────────────────────────────────────────────

def definitions():
    """Metric definitions for tooltip overlays."""
    return {
        "response_time_note": "Average response time is measured in hours from when an inbound "
                              "message is received to when a provider reply is sent. Lower values "
                              "indicate faster clinical communication. Only inbound messages with "
                              "a recorded response time are included.",
        "unread_note": "Unread messages have been delivered but not yet opened by the recipient. "
                       "High unread counts may indicate communication bottlenecks or staffing gaps.",
        "priority_levels": {
            "urgent": "Requires immediate clinical attention (seizure event, ER visit, acute symptom)",
            "high": "Needs same-day response (new symptoms, medication concerns)",
            "normal": "Standard clinical communication (follow-up questions, appointment requests)",
            "low": "Non-urgent informational messages (general inquiries, refill requests)"
        },
        "categories": {
            "side-effect-report": "Patient reports side effects from current medications",
            "symptom-report": "Patient reports new or changed symptoms",
            "urgent": "Time-sensitive clinical communication requiring immediate attention",
            "test-results": "Questions or communication about diagnostic test results",
            "medication-question": "Inquiries about medication dosing, timing, or interactions",
            "appointment-request": "Requests to schedule, reschedule, or cancel appointments",
            "general-inquiry": "General clinical or administrative questions",
            "prescription-refill": "Requests for prescription renewals or refills"
        },
        "direction_note": "Inbound = patient-to-provider; Outbound = provider-to-patient.",
        "gap_analysis_note": "Messages are analyzed for response latency. Messages with "
                             "response_time_hours > 24 may indicate SLA breaches."
    }
