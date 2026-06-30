"""Event / Kafka / Queue Dashboard — real metrics from clinical.db.

Surfaces the platform's event-processing pipeline as a queue dashboard:
throughput, action-type distribution, component-queue depth, actor attribution,
daily event volume trend, processing latency proxies, and queue health.

Sources:
- transaction_log (every system event → queue items)
- conversation_log (async message turns → message queue)
- assessments (queued evaluation jobs)
- uploads (file-ingest queue)
"""

import sqlite3
import os
from datetime import datetime, timezone, timedelta

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql, default=0):
    try:
        cur.execute(sql)
        return cur.fetchone()[0]
    except Exception:
        return default


def _safe_rows(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchall()
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────
#  /api/event-queue/overview
# ──────────────────────────────────────────────────────────────

def event_queue_overview():
    """Aggregate event-queue health: throughput, distribution, trends."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # ── 1. Total event counts ──────────────────────────────────
    total_events = _safe(cur, "SELECT count(*) FROM transaction_log")
    total_conversations = _safe(cur, "SELECT count(*) FROM conversation_log")
    total_assessments = _safe(cur, "SELECT count(*) FROM assessments")
    total_uploads = _safe(cur, "SELECT count(*) FROM uploads")
    total_queued = total_events + total_conversations + total_assessments + total_uploads

    # ── 2. Action-type distribution (event types) ──────────────
    rows = _safe_rows(cur, """
        SELECT action, count(*) as cnt
        FROM transaction_log
        GROUP BY action ORDER BY cnt DESC
    """)
    action_distribution = [{"action": r[0] or "unknown", "count": r[1]} for r in rows]

    # ── 3. Component queues (which components produce events) ──
    rows = _safe_rows(cur, """
        SELECT component, count(*) as cnt
        FROM transaction_log
        GROUP BY component ORDER BY cnt DESC
    """)
    component_queues = [{"queue": r[0] or "unknown", "events": r[1]} for r in rows]

    # ── 4. Actor attribution (who/what produces events) ────────
    rows = _safe_rows(cur, """
        SELECT actor, count(*) as cnt
        FROM transaction_log
        GROUP BY actor ORDER BY cnt DESC
    """)
    actor_distribution = [{"actor": r[0] or "unknown", "events": r[1]} for r in rows]

    # ── 5. Daily event volume (last 14 days) ───────────────────
    rows = _safe_rows(cur, """
        SELECT date(ts_utc) as day, count(*) as cnt
        FROM transaction_log
        WHERE ts_utc >= date('now', '-14 days')
        GROUP BY day ORDER BY day
    """)
    daily_volume = [{"date": r[0], "events": r[1]} for r in rows]

    # ── 6. Hourly distribution (detect burst patterns) ─────────
    rows = _safe_rows(cur, """
        SELECT cast(strftime('%H', ts_utc) as integer) as hour, count(*) as cnt
        FROM transaction_log
        GROUP BY hour ORDER BY hour
    """)
    hourly_distribution = [{"hour": r[0], "events": r[1]} for r in rows]

    # ── 7. Queue health metrics ────────────────────────────────
    blocked_events = _safe(cur, "SELECT count(*) FROM transaction_log WHERE action = 'blocked'")
    error_events = _safe(cur, "SELECT count(*) FROM transaction_log WHERE action IN ('error', 'fail', 'failed')")

    # Recent 24h throughput
    recent_24h = _safe(cur, """
        SELECT count(*) FROM transaction_log
        WHERE ts_utc >= datetime('now', '-1 day')
    """)

    # Recent 1h throughput
    recent_1h = _safe(cur, """
        SELECT count(*) FROM transaction_log
        WHERE ts_utc >= datetime('now', '-1 hour')
    """)

    # Conversation queue — messages by role
    rows = _safe_rows(cur, """
        SELECT role, count(*) as cnt
        FROM conversation_log
        GROUP BY role ORDER BY cnt DESC
    """)
    message_queue = [{"role": r[0] or "unknown", "messages": r[1]} for r in rows]

    # Upload queue status
    rows = _safe_rows(cur, """
        SELECT status, count(*) as cnt
        FROM uploads
        GROUP BY status ORDER BY cnt DESC
    """)
    upload_queue = [{"status": r[0] or "unknown", "count": r[1]} for r in rows]

    # Unique patients with events
    unique_patients = _safe(cur, "SELECT count(DISTINCT patient_id) FROM transaction_log WHERE patient_id != ''")

    # Date range
    first_event = _safe(cur, "SELECT min(ts_utc) FROM transaction_log", "")
    last_event = _safe(cur, "SELECT max(ts_utc) FROM transaction_log", "")

    # Distinct components and actions
    distinct_components = _safe(cur, "SELECT count(DISTINCT component) FROM transaction_log")
    distinct_actions = _safe(cur, "SELECT count(DISTINCT action) FROM transaction_log")

    health_status = "healthy"
    if blocked_events > 5 or error_events > 10:
        health_status = "degraded"
    if blocked_events > 20 or error_events > 50:
        health_status = "critical"

    conn.close()

    return {
        "available": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_events": total_events,
            "total_queued_items": total_queued,
            "total_conversations": total_conversations,
            "total_assessments": total_assessments,
            "total_uploads": total_uploads,
            "blocked_events": blocked_events,
            "error_events": error_events,
            "throughput_24h": recent_24h,
            "throughput_1h": recent_1h,
            "unique_patients": unique_patients,
            "distinct_queues": distinct_components,
            "distinct_event_types": distinct_actions,
            "health_status": health_status,
            "first_event": first_event,
            "last_event": last_event,
        },
        "action_distribution": action_distribution,
        "component_queues": component_queues,
        "actor_distribution": actor_distribution,
        "daily_volume": daily_volume,
        "hourly_distribution": hourly_distribution,
        "message_queue": message_queue,
        "upload_queue": upload_queue,
    }


# ──────────────────────────────────────────────────────────────
#  /api/event-queue/breakdown
# ──────────────────────────────────────────────────────────────

def event_queue_breakdown():
    """Per-queue breakdown: recent events, component × action cross-tab,
    patient-level event counts."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # ── 1. Component × Action cross-tab ────────────────────────
    rows = _safe_rows(cur, """
        SELECT component, action, count(*) as cnt
        FROM transaction_log
        GROUP BY component, action
        ORDER BY cnt DESC
    """)
    cross_tab = [
        {"component": r[0] or "unknown", "action": r[1] or "unknown", "count": r[2]}
        for r in rows
    ]

    # ── 2. Recent events (last 50) ─────────────────────────────
    rows = _safe_rows(cur, """
        SELECT id, patient_id, component, action, actor, detail, ts_utc
        FROM transaction_log
        ORDER BY id DESC LIMIT 50
    """)
    recent_events = [
        {
            "id": r[0], "patient_id": r[1] or "", "component": r[2] or "",
            "action": r[3] or "", "actor": r[4] or "", "detail": r[5] or "",
            "ts_utc": r[6] or "",
        }
        for r in rows
    ]

    # ── 3. Patient event counts ────────────────────────────────
    rows = _safe_rows(cur, """
        SELECT patient_id, count(*) as cnt
        FROM transaction_log
        WHERE patient_id != ''
        GROUP BY patient_id ORDER BY cnt DESC
        LIMIT 20
    """)
    patient_events = [{"patient_id": r[0], "events": r[1]} for r in rows]

    # ── 4. Per-queue (component) stats ─────────────────────────
    rows = _safe_rows(cur, """
        SELECT component,
               count(*) as total,
               min(ts_utc) as first_ts,
               max(ts_utc) as last_ts,
               count(DISTINCT action) as action_types,
               count(DISTINCT patient_id) as patients
        FROM transaction_log
        GROUP BY component ORDER BY total DESC
    """)
    queue_stats = [
        {
            "queue": r[0] or "unknown", "total_events": r[1],
            "first_event": r[2] or "", "last_event": r[3] or "",
            "action_types": r[4], "patients": r[5],
        }
        for r in rows
    ]

    # ── 5. Conversation timeline (daily) ───────────────────────
    rows = _safe_rows(cur, """
        SELECT date(ts_utc) as day, role, count(*) as cnt
        FROM conversation_log
        GROUP BY day, role ORDER BY day
    """)
    conversation_timeline = [
        {"date": r[0], "role": r[1] or "unknown", "messages": r[2]}
        for r in rows
    ]

    conn.close()

    return {
        "available": True,
        "cross_tab": cross_tab,
        "recent_events": recent_events,
        "patient_events": patient_events,
        "queue_stats": queue_stats,
        "conversation_timeline": conversation_timeline,
    }


# ──────────────────────────────────────────────────────────────
#  /api/event-queue/definitions
# ──────────────────────────────────────────────────────────────

def event_queue_definitions():
    """Metric definitions for the Event / Queue dashboard."""
    return {
        "available": True,
        "definitions": [
            {
                "term": "Total Events",
                "definition": "Count of all rows in the transaction_log table. Each row represents one discrete system event (create, process, log, etc.).",
            },
            {
                "term": "Total Queued Items",
                "definition": "Sum of transaction_log events + conversation messages + assessments + uploads. Represents all items that have passed through the platform's processing queues.",
            },
            {
                "term": "Throughput (24h / 1h)",
                "definition": "Number of transaction_log events recorded in the last 24 hours or 1 hour. Indicates current processing velocity.",
            },
            {
                "term": "Component Queue",
                "definition": "Each distinct 'component' in the transaction_log acts as a logical queue (e.g., assessment, cv_pipeline, seizure_diary). Events are grouped by component to show queue depth.",
            },
            {
                "term": "Action Distribution",
                "definition": "Breakdown of event types (create, process, extract, log, monitor, etc.). Shows what kinds of work the system is performing.",
            },
            {
                "term": "Actor Distribution",
                "definition": "Who or what produced each event: system (automated), compliance_agent, neurologist, etc. Shows human-vs-AI event sourcing.",
            },
            {
                "term": "Blocked Events",
                "definition": "Events with action='blocked'. Indicates queue items that could not be processed due to missing data, permissions, or dependencies.",
            },
            {
                "term": "Health Status",
                "definition": "Overall queue health: healthy (few blocked/errors), degraded (>5 blocked or >10 errors), critical (>20 blocked or >50 errors).",
            },
            {
                "term": "Hourly Distribution",
                "definition": "Event counts grouped by hour-of-day (0-23 UTC). Reveals burst patterns and processing peaks.",
            },
            {
                "term": "Message Queue",
                "definition": "Conversation messages (assistant/operator/system) from conversation_log, treated as a message-passing queue.",
            },
            {
                "term": "Upload Queue",
                "definition": "File uploads grouped by processing status. Shows ingest-queue throughput.",
            },
            {
                "term": "Cross-Tab (Component × Action)",
                "definition": "Two-dimensional breakdown showing which components generate which action types, enabling queue-specific analysis.",
            },
        ],
    }
