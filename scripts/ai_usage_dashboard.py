"""AI Usage Dashboard — real transaction volume, component activity,
actor breakdown, action distribution, and daily trends from clinical.db."""

import sqlite3
import os
from datetime import datetime, timezone
from collections import Counter

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def usage_overview():
    """Total operations, daily trend (last 14 days), top components, actor
    breakdown, action distribution — all from transaction_log."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT count(*) FROM transaction_log")
        total = cur.fetchone()[0]
    except Exception:
        conn.close()
        return {"available": False, "note": "transaction_log table missing"}

    if total == 0:
        conn.close()
        return {"available": False, "note": "No transactions recorded yet"}

    # Date range
    cur.execute("SELECT min(ts_utc), max(ts_utc) FROM transaction_log")
    row = cur.fetchone()
    first_ts, last_ts = row[0] or "", row[1] or ""

    # Daily trend (last 14 days)
    cur.execute("""
        SELECT date(ts_utc) as d, count(*) as cnt
        FROM transaction_log
        GROUP BY d ORDER BY d DESC LIMIT 14
    """)
    daily = [{"date": r[0], "operations": r[1]} for r in cur.fetchall()]
    daily.reverse()

    # By component
    cur.execute("""
        SELECT component, count(*) as cnt
        FROM transaction_log GROUP BY component ORDER BY cnt DESC
    """)
    by_component = {r[0]: r[1] for r in cur.fetchall()}

    # By action
    cur.execute("""
        SELECT action, count(*) as cnt
        FROM transaction_log GROUP BY action ORDER BY cnt DESC
    """)
    by_action = {r[0]: r[1] for r in cur.fetchall()}

    # By actor
    cur.execute("""
        SELECT actor, count(*) as cnt
        FROM transaction_log GROUP BY actor ORDER BY cnt DESC
    """)
    by_actor = {r[0]: r[1] for r in cur.fetchall()}

    # Distinct patients (non-system)
    cur.execute("""
        SELECT count(DISTINCT patient_id) FROM transaction_log
        WHERE patient_id != '_system'
    """)
    patient_count = cur.fetchone()[0]

    # Peak hour
    cur.execute("""
        SELECT cast(strftime('%H', ts_utc) as integer) as hr, count(*) as cnt
        FROM transaction_log GROUP BY hr ORDER BY cnt DESC LIMIT 1
    """)
    peak = cur.fetchone()
    peak_hour = peak[0] if peak else 0

    # Today's ops
    cur.execute("""
        SELECT count(*) FROM transaction_log
        WHERE date(ts_utc) = date('now')
    """)
    today_ops = cur.fetchone()[0]

    conn.close()

    return {
        "available": True,
        "summary": {
            "total_operations": total,
            "today_operations": today_ops,
            "unique_patients": patient_count,
            "unique_components": len(by_component),
            "unique_actors": len(by_actor),
            "peak_hour_utc": peak_hour,
        },
        "daily_trend": daily,
        "by_component": by_component,
        "by_action": by_action,
        "by_actor": by_actor,
        "date_range": {"first": first_ts, "last": last_ts},
    }


def usage_breakdown():
    """Per-component breakdown: operations, top actions, top actors, latest timestamp."""
    if not os.path.exists(DB):
        return {"available": False, "components": []}

    conn = _conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT component, action, count(*) as cnt
            FROM transaction_log GROUP BY component, action ORDER BY component, cnt DESC
        """)
        rows = cur.fetchall()
    except Exception:
        conn.close()
        return {"available": False, "components": []}

    comp_data = {}
    for comp, action, cnt in rows:
        if comp not in comp_data:
            comp_data[comp] = {"component": comp, "operations": 0, "actions": {}}
        comp_data[comp]["operations"] += cnt
        comp_data[comp]["actions"][action] = cnt

    # Get latest ts per component
    cur.execute("""
        SELECT component, max(ts_utc) FROM transaction_log GROUP BY component
    """)
    for comp, ts in cur.fetchall():
        if comp in comp_data:
            comp_data[comp]["last_activity"] = ts or ""

    # Get actors per component
    cur.execute("""
        SELECT component, actor, count(*) as cnt
        FROM transaction_log GROUP BY component, actor ORDER BY component, cnt DESC
    """)
    for comp, actor, cnt in cur.fetchall():
        if comp in comp_data:
            if "actors" not in comp_data[comp]:
                comp_data[comp]["actors"] = {}
            comp_data[comp]["actors"][actor] = cnt

    conn.close()

    components = sorted(comp_data.values(), key=lambda c: c["operations"], reverse=True)
    return {
        "available": True,
        "components": components,
        "total_components": len(components),
    }


def usage_definitions():
    """Metric definitions for AI usage dashboard tooltips."""
    return {
        "available": True,
        "definitions": [
            {"term": "Total Operations", "definition": "Count of all recorded transactions in the system transaction log, including patient actions, system processes, and AI inferences."},
            {"term": "Today's Operations", "definition": "Number of transactions recorded today (UTC), indicating current system activity level."},
            {"term": "Unique Patients", "definition": "Count of distinct patient IDs (excluding _system) that have generated at least one transaction."},
            {"term": "Peak Hour (UTC)", "definition": "The hour of day (0-23 UTC) with the highest number of recorded operations historically."},
            {"term": "Component", "definition": "A subsystem that records transactions — e.g., assessment, cv_pipeline, eeg_upload, patient_chat, seizure_diary."},
            {"term": "Action", "definition": "The type of operation performed — e.g., ingest, query, create, analyze, extract, process."},
            {"term": "Actor", "definition": "The entity that performed the operation — can be 'system' for automated processes, or a clinician/role name."},
            {"term": "Daily Trend", "definition": "Operations per day over the last 14 days, used to spot usage spikes, adoption patterns, or downtime."},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(usage_overview(), indent=2, default=str))
    print("\n=== BREAKDOWN ===")
    print(json.dumps(usage_breakdown(), indent=2, default=str))
