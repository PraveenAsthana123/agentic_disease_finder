"""
Neuro AI Ecosystem — Transaction Audit Trail Dashboard
======================================================
Audit trail analytics from the transaction_log table.

Components tracked: video_frames, cv_pipeline, assessment, referral, seizure_diary,
drift, training, graph_db, consistency, fairness, eeg_upload, patient_chat, etc.
Actions tracked: process, extract, create, analyze, log, monitor, scheduled_train, etc.
Actors: system, psychiatrist, neurologist, compliance_agent, etc.

Real data: transaction_log (1360 rows, 27 components) in clinical.db.
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
    """Transaction audit overview — volume trends, component/action/actor
    distributions, human vs system breakdown."""
    conn = _conn()
    cur = conn.cursor()

    # Total transactions
    cur.execute("SELECT COUNT(*) FROM transaction_log")
    total_transactions = cur.fetchone()[0]

    # Component distribution
    cur.execute("""
        SELECT component, COUNT(*) cnt
        FROM transaction_log
        GROUP BY component
        ORDER BY cnt DESC
    """)
    component_distribution = {r[0]: r[1] for r in cur.fetchall()}

    # Action distribution
    cur.execute("""
        SELECT action, COUNT(*) cnt
        FROM transaction_log
        GROUP BY action
        ORDER BY cnt DESC
    """)
    action_distribution = {r[0]: r[1] for r in cur.fetchall()}

    # Actor distribution
    cur.execute("""
        SELECT actor, COUNT(*) cnt
        FROM transaction_log
        GROUP BY actor
        ORDER BY cnt DESC
    """)
    actor_distribution = {r[0]: r[1] for r in cur.fetchall()}

    # Daily volume
    cur.execute("""
        SELECT DATE(ts_local) as date, COUNT(*) as count
        FROM transaction_log
        WHERE ts_local IS NOT NULL
        GROUP BY DATE(ts_local)
        ORDER BY date
    """)
    daily_volume = _dict_rows(cur)

    # Human vs system
    cur.execute("""
        SELECT
            SUM(CASE WHEN actor = 'system' THEN 1 ELSE 0 END) as system_count,
            SUM(CASE WHEN actor != 'system' THEN 1 ELSE 0 END) as human_count
        FROM transaction_log
    """)
    row = cur.fetchone()
    human_vs_system = {
        "system": row[0] or 0,
        "human": row[1] or 0,
        "system_pct": round((row[0] or 0) / max(total_transactions, 1) * 100, 1),
        "human_pct": round((row[1] or 0) / max(total_transactions, 1) * 100, 1),
    }

    # Top 10 components by volume with percentage
    top_components = []
    for comp, cnt in list(component_distribution.items())[:10]:
        top_components.append({
            "component": comp,
            "count": cnt,
            "percentage": round(cnt / max(total_transactions, 1) * 100, 1),
        })

    conn.close()
    return {
        "total_transactions": total_transactions,
        "component_distribution": component_distribution,
        "action_distribution": action_distribution,
        "actor_distribution": actor_distribution,
        "daily_volume": daily_volume,
        "human_vs_system": human_vs_system,
        "top_components": top_components,
    }


def breakdown():
    """Transaction audit breakdown — per-component detail, recent
    transactions, hourly patterns, patient activity."""
    conn = _conn()
    cur = conn.cursor()

    # Per-component detail
    cur.execute("SELECT DISTINCT component FROM transaction_log ORDER BY component")
    components = [r[0] for r in cur.fetchall()]

    per_component = []
    for comp in components:
        cur.execute("SELECT COUNT(*) FROM transaction_log WHERE component = ?", (comp,))
        total = cur.fetchone()[0]

        cur.execute("""
            SELECT action, COUNT(*) cnt
            FROM transaction_log WHERE component = ?
            GROUP BY action ORDER BY cnt DESC
        """, (comp,))
        actions = {r[0]: r[1] for r in cur.fetchall()}

        cur.execute("""
            SELECT actor, COUNT(*) cnt
            FROM transaction_log WHERE component = ?
            GROUP BY actor ORDER BY cnt DESC
        """, (comp,))
        actors = {r[0]: r[1] for r in cur.fetchall()}

        cur.execute("""
            SELECT MIN(ts_local) as earliest, MAX(ts_local) as latest
            FROM transaction_log WHERE component = ?
        """, (comp,))
        ts = cur.fetchone()

        per_component.append({
            "component": comp,
            "total": total,
            "actions": actions,
            "actors": actors,
            "earliest_ts": ts[0],
            "latest_ts": ts[1],
        })

    # Per-actor detail
    cur.execute("SELECT DISTINCT actor FROM transaction_log ORDER BY actor")
    actors_list = [r[0] for r in cur.fetchall()]

    per_actor = []
    for actor in actors_list:
        cur.execute("SELECT COUNT(*) FROM transaction_log WHERE actor = ?", (actor,))
        total = cur.fetchone()[0]

        cur.execute("""
            SELECT DISTINCT component FROM transaction_log WHERE actor = ?
            ORDER BY component
        """, (actor,))
        comps = [r[0] for r in cur.fetchall()]

        cur.execute("""
            SELECT DISTINCT action FROM transaction_log WHERE actor = ?
            ORDER BY action
        """, (actor,))
        acts = [r[0] for r in cur.fetchall()]

        per_actor.append({
            "actor": actor,
            "total": total,
            "components": comps,
            "actions": acts,
        })

    # Recent 50 transactions
    cur.execute("""
        SELECT id, patient_id, component, action, actor, ref_id, detail, ts_utc, ts_local
        FROM transaction_log
        ORDER BY id DESC
        LIMIT 50
    """)
    recent_transactions = _dict_rows(cur)

    # Hourly pattern
    cur.execute("""
        SELECT CAST(STRFTIME('%H', ts_local) AS INTEGER) as hour, COUNT(*) as count
        FROM transaction_log
        WHERE ts_local IS NOT NULL
        GROUP BY hour
        ORDER BY hour
    """)
    hourly_raw = {r[0]: r[1] for r in cur.fetchall()}
    hourly_pattern = [{"hour": h, "count": hourly_raw.get(h, 0)} for h in range(24)]

    # Patient activity (top 20)
    cur.execute("""
        SELECT patient_id, COUNT(*) as count
        FROM transaction_log
        WHERE patient_id IS NOT NULL
        GROUP BY patient_id
        ORDER BY count DESC
        LIMIT 20
    """)
    patient_activity = _dict_rows(cur)

    conn.close()
    return {
        "per_component": per_component,
        "per_actor": per_actor,
        "recent_transactions": recent_transactions,
        "hourly_pattern": hourly_pattern,
        "patient_activity": patient_activity,
    }


def definitions():
    """Transaction audit definitions — audit trail glossary,
    component descriptions, action type reference."""
    glossary = [
        {"term": "Transaction", "definition": "A single recorded event in the system representing an action taken by an actor on a component."},
        {"term": "Component", "definition": "The system module or subsystem where the transaction originated (e.g., cv_pipeline, assessment, eeg_upload)."},
        {"term": "Action", "definition": "The type of operation performed (e.g., process, extract, create, analyze, monitor)."},
        {"term": "Actor", "definition": "The entity that initiated the transaction — either 'system' for automated actions or a human role (e.g., neurologist, psychiatrist)."},
        {"term": "Audit Trail", "definition": "A chronological record of all transactions providing accountability, traceability, and compliance evidence."},
        {"term": "ref_id", "definition": "A reference identifier linking the transaction to a specific record in another table (e.g., patient record, model version)."},
        {"term": "ts_utc", "definition": "The timestamp of the transaction in Coordinated Universal Time (UTC)."},
        {"term": "ts_local", "definition": "The timestamp of the transaction in the local timezone of the system."},
        {"term": "Human vs System", "definition": "Classification of transactions by actor type — system-automated actions vs human-initiated actions."},
        {"term": "Daily Volume", "definition": "The number of transactions recorded per calendar day, used to track activity trends."},
        {"term": "Hourly Pattern", "definition": "Distribution of transactions across the 24 hours of the day, revealing peak activity periods."},
        {"term": "Patient Activity", "definition": "Per-patient transaction count showing how many system events are associated with each patient."},
    ]

    component_descriptions = {
        "video_frames": "Video frame extraction and processing from EEG video monitoring sessions.",
        "cv_pipeline": "Computer vision pipeline for automated video analysis and feature extraction.",
        "assessment": "Clinical assessment records including evaluations and diagnostic notes.",
        "referral": "Patient referral events between clinical specialties (e.g., to neurology).",
        "seizure_diary": "Patient-reported seizure diary entries and logging.",
        "drift": "Data and model drift monitoring events.",
        "training": "Model training and retraining pipeline events.",
        "graph_db": "Knowledge graph database build and query operations.",
        "consistency": "Data consistency checks and validation events.",
        "fairness": "Algorithmic fairness and bias monitoring checks.",
        "eeg_upload": "EEG data upload and ingestion events.",
        "patient_chat": "Patient-facing chat interactions and messages.",
        "team_chat": "Internal clinical team communication events.",
        "council": "Multi-disciplinary team council review events.",
        "data_quality": "Data quality assessment and scoring events.",
        "model_registry": "Model version registration and lifecycle events.",
        "feature_store": "Feature store update and retrieval events.",
        "alert": "System and clinical alert generation events.",
        "notification": "User notification delivery events.",
        "audit": "Internal audit and compliance check events.",
    }

    action_descriptions = {
        "process": "Data processing or transformation operation.",
        "extract": "Feature or data extraction from raw inputs.",
        "create": "Creation of a new record (assessment, referral, entry).",
        "analyze": "Analytical operation such as statistical or ML analysis.",
        "log": "Logging of a clinical or system event.",
        "monitor": "Continuous monitoring check (drift, fairness, consistency).",
        "scheduled_train": "Scheduled model training or retraining job.",
        "check": "Validation or quality check operation.",
        "build": "Build operation (e.g., knowledge graph construction).",
        "refer_to_neurology": "Referral action directing patient to neurology specialty.",
        "refer_to_psychiatry": "Referral action directing patient to psychiatry specialty.",
        "upload": "Data upload operation.",
        "send": "Message or notification send operation.",
        "review": "Clinical or administrative review event.",
        "update": "Update to an existing record.",
    }

    return {
        "glossary": glossary,
        "component_descriptions": component_descriptions,
        "action_descriptions": action_descriptions,
    }
