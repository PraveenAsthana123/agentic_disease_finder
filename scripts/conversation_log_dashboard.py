"""
Neuro AI Ecosystem — Conversation Log Dashboard
=================================================
AI-human interaction analytics from conversation_log table.

Roles: operator (human), assistant (AI)
Fields: id, role, text, ts_utc, ts_local

Real data: conversation_log (2364 rows, 29 active days) in clinical.db.
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
    """Conversation log overview — KPIs, role distribution, daily volume trend,
    text length stats, hourly activity pattern."""
    conn = _conn()
    cur = conn.cursor()

    # Total messages
    cur.execute("SELECT COUNT(*) FROM conversation_log")
    total = cur.fetchone()[0]

    # Role counts
    cur.execute("SELECT role, COUNT(*) cnt FROM conversation_log GROUP BY role ORDER BY cnt DESC")
    role_dist = _dict_rows(cur)

    # Date range
    cur.execute("SELECT MIN(date(ts_utc)), MAX(date(ts_utc)), COUNT(DISTINCT date(ts_utc)) FROM conversation_log")
    row = cur.fetchone()
    date_range = {"first_date": row[0], "last_date": row[1], "active_days": row[2]}

    # Average text length by role
    cur.execute("""
        SELECT role,
               ROUND(AVG(LENGTH(text)), 0) avg_len,
               MAX(LENGTH(text)) max_len,
               MIN(LENGTH(text)) min_len,
               ROUND(SUM(LENGTH(text)) / 1000.0, 1) total_kb
        FROM conversation_log
        GROUP BY role
    """)
    text_stats = _dict_rows(cur)

    # Total characters
    cur.execute("SELECT SUM(LENGTH(text)) FROM conversation_log")
    total_chars = cur.fetchone()[0] or 0

    # Daily volume trend
    cur.execute("""
        SELECT date(ts_utc) day, COUNT(*) total,
               SUM(CASE WHEN role='operator' THEN 1 ELSE 0 END) operator_msgs,
               SUM(CASE WHEN role='assistant' THEN 1 ELSE 0 END) assistant_msgs
        FROM conversation_log
        GROUP BY day
        ORDER BY day
    """)
    daily_trend = _dict_rows(cur)

    # Hourly activity pattern
    cur.execute("""
        SELECT CAST(strftime('%H', ts_utc) AS INTEGER) hour, COUNT(*) cnt
        FROM conversation_log
        GROUP BY hour
        ORDER BY hour
    """)
    hourly_pattern = _dict_rows(cur)

    # Messages per day average
    avg_per_day = round(total / max(date_range["active_days"], 1), 1)

    # Operator-to-assistant ratio
    op_count = next((r["cnt"] for r in role_dist if r["role"] == "operator"), 0)
    ast_count = next((r["cnt"] for r in role_dist if r["role"] == "assistant"), 0)
    ratio = round(ast_count / max(op_count, 1), 1)

    conn.close()
    return {
        "total_messages": total,
        "role_distribution": role_dist,
        "date_range": date_range,
        "text_stats": text_stats,
        "total_characters": total_chars,
        "avg_messages_per_day": avg_per_day,
        "assistant_to_operator_ratio": ratio,
        "daily_trend": daily_trend,
        "hourly_pattern": hourly_pattern,
    }


def breakdown():
    """Conversation log breakdown — recent messages, daily detail,
    longest messages, text length distribution."""
    conn = _conn()
    cur = conn.cursor()

    # Recent messages (last 50)
    cur.execute("""
        SELECT id, role, SUBSTR(text, 1, 200) text_preview,
               LENGTH(text) text_length, ts_utc, ts_local
        FROM conversation_log
        ORDER BY id DESC
        LIMIT 50
    """)
    recent = _dict_rows(cur)

    # Daily summary with stats
    cur.execute("""
        SELECT date(ts_utc) day,
               COUNT(*) total,
               SUM(CASE WHEN role='operator' THEN 1 ELSE 0 END) operator_msgs,
               SUM(CASE WHEN role='assistant' THEN 1 ELSE 0 END) assistant_msgs,
               ROUND(AVG(LENGTH(text)), 0) avg_text_len,
               SUM(LENGTH(text)) total_chars
        FROM conversation_log
        GROUP BY day
        ORDER BY day DESC
    """)
    daily_detail = _dict_rows(cur)

    # Longest messages
    cur.execute("""
        SELECT id, role, SUBSTR(text, 1, 200) text_preview,
               LENGTH(text) text_length, ts_utc
        FROM conversation_log
        ORDER BY LENGTH(text) DESC
        LIMIT 10
    """)
    longest = _dict_rows(cur)

    # Text length distribution buckets
    cur.execute("""
        SELECT CASE
                 WHEN LENGTH(text) < 100 THEN '< 100'
                 WHEN LENGTH(text) < 300 THEN '100-299'
                 WHEN LENGTH(text) < 500 THEN '300-499'
                 WHEN LENGTH(text) < 1000 THEN '500-999'
                 WHEN LENGTH(text) < 2000 THEN '1000-1999'
                 ELSE '2000+'
               END bucket,
               COUNT(*) cnt
        FROM conversation_log
        GROUP BY bucket
        ORDER BY MIN(LENGTH(text))
    """)
    length_dist = _dict_rows(cur)

    # Operator message topics (keywords in operator messages)
    cur.execute("""
        SELECT SUBSTR(text, 1, 120) preview, ts_utc
        FROM conversation_log
        WHERE role = 'operator'
        ORDER BY id DESC
        LIMIT 20
    """)
    operator_recent = _dict_rows(cur)

    conn.close()
    return {
        "recent_messages": recent,
        "daily_detail": daily_detail,
        "longest_messages": longest,
        "length_distribution": length_dist,
        "operator_recent": operator_recent,
    }


def definitions():
    """Conversation log definitions — field glossary, role descriptions,
    analytics methodology."""
    return {
        "fields": [
            {"field": "id", "type": "INTEGER", "description": "Auto-increment message ID"},
            {"field": "role", "type": "TEXT", "description": "Message sender role — operator (human) or assistant (AI)"},
            {"field": "text", "type": "TEXT", "description": "Full message text content"},
            {"field": "ts_utc", "type": "TEXT", "description": "UTC timestamp of message"},
            {"field": "ts_local", "type": "TEXT", "description": "Local timezone timestamp"},
        ],
        "roles": {
            "operator": "Human user issuing commands, asking questions, or providing feedback",
            "assistant": "AI assistant generating responses, building features, running analyses",
        },
        "metrics": {
            "assistant_to_operator_ratio": "Number of assistant messages per operator message — higher = more AI-generated content per human prompt",
            "avg_messages_per_day": "Total messages / active days — measures interaction intensity",
            "text_length": "Character count of message text — proxy for message complexity",
            "active_days": "Days with at least one message exchanged",
        },
        "data_sources": [
            "conversation_log table in clinical.db",
            "Logged automatically during AI-human interaction sessions",
        ],
    }
