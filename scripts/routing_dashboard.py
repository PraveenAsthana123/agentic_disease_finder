"""Routing Dashboard — real decision-routing metrics from clinical.db.

Tracks how tasks and decisions are routed across components, actors, and
action types in the clinical pipeline. Surfaces routing volume, component
fanout, actor workload distribution, routing patterns over time, and
decision-routing outcomes (AI vs human agreement paths).

Sources:
- transaction_log (component + action + actor → routing paths)
- clinical_decisions (AI prediction → neurologist agreement → final decision)
- component_findings (doctor findings → agree/disagree routing)
- conversation_log (routing of conversation turns by role)
"""

import sqlite3
import os
from datetime import datetime, timezone

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
#  /api/routing/overview
# ──────────────────────────────────────────────────────────────

def routing_overview():
    """Aggregate routing health: volume, fanout, actor distribution,
    component routing, decision routing outcomes, daily trends."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # ── 1. Total routing volume ──────────────────────────────────
    total_routed = _safe(cur, "SELECT count(*) FROM transaction_log")
    total_decisions = _safe(cur, "SELECT count(*) FROM clinical_decisions")
    total_findings = _safe(cur, "SELECT count(*) FROM component_findings")
    total_conversations = _safe(cur, "SELECT count(*) FROM conversation_log")
    distinct_components = _safe(
        cur, "SELECT count(DISTINCT component) FROM transaction_log")
    distinct_actors = _safe(
        cur, "SELECT count(DISTINCT actor) FROM transaction_log")
    distinct_actions = _safe(
        cur, "SELECT count(DISTINCT action) FROM transaction_log")
    unique_patients = _safe(
        cur, "SELECT count(DISTINCT patient_id) FROM transaction_log WHERE patient_id IS NOT NULL")

    # Auto-routed (actor=system) vs human-routed
    auto_routed = _safe(
        cur, "SELECT count(*) FROM transaction_log WHERE actor = 'system'")
    human_routed = total_routed - auto_routed
    auto_rate = round(auto_routed / max(total_routed, 1) * 100, 1)

    # ── 2. Component routing distribution ────────────────────────
    rows = _safe_rows(cur, """
        SELECT component, count(*) as cnt
        FROM transaction_log
        GROUP BY component ORDER BY cnt DESC
    """)
    component_distribution = [
        {"component": r[0], "routed": r[1],
         "pct": round(r[1] / max(total_routed, 1) * 100, 1)}
        for r in rows
    ]

    # ── 3. Action routing distribution ───────────────────────────
    rows = _safe_rows(cur, """
        SELECT action, count(*) as cnt
        FROM transaction_log
        GROUP BY action ORDER BY cnt DESC
    """)
    action_distribution = [
        {"action": r[0], "count": r[1]} for r in rows
    ]

    # ── 4. Actor workload distribution ───────────────────────────
    rows = _safe_rows(cur, """
        SELECT actor, count(*) as cnt
        FROM transaction_log
        GROUP BY actor ORDER BY cnt DESC
    """)
    actor_distribution = [
        {"actor": r[0], "routed": r[1],
         "pct": round(r[1] / max(total_routed, 1) * 100, 1)}
        for r in rows
    ]

    # ── 5. Decision routing outcomes ─────────────────────────────
    rows = _safe_rows(cur, """
        SELECT neurologist_agreement, final_decision, count(*) as cnt
        FROM clinical_decisions
        GROUP BY neurologist_agreement, final_decision
        ORDER BY cnt DESC
    """)
    decision_outcomes = [
        {"agreement": r[0] or "pending", "final_decision": r[1] or "pending",
         "count": r[2]}
        for r in rows
    ]

    agreed = _safe(
        cur, "SELECT count(*) FROM clinical_decisions WHERE neurologist_agreement = 'Yes'")
    disagreed = _safe(
        cur, "SELECT count(*) FROM clinical_decisions WHERE neurologist_agreement = 'No'")
    agreement_rate = round(agreed / max(agreed + disagreed, 1) * 100, 1)

    # ── 6. Component-to-action routing matrix (top combos) ───────
    rows = _safe_rows(cur, """
        SELECT component, action, count(*) as cnt
        FROM transaction_log
        GROUP BY component, action
        ORDER BY cnt DESC LIMIT 20
    """)
    routing_matrix = [
        {"component": r[0], "action": r[1], "count": r[2]}
        for r in rows
    ]

    # ── 7. Daily routing volume trend ────────────────────────────
    rows = _safe_rows(cur, """
        SELECT date(ts_utc) as d, count(*) as cnt
        FROM transaction_log
        WHERE ts_utc IS NOT NULL
        GROUP BY d ORDER BY d DESC LIMIT 30
    """)
    daily_volume = [
        {"date": r[0], "routed": r[1]} for r in reversed(rows)
    ]

    # ── 8. Hourly routing pattern ────────────────────────────────
    rows = _safe_rows(cur, """
        SELECT cast(strftime('%H', ts_utc) as integer) as h, count(*) as cnt
        FROM transaction_log
        WHERE ts_utc IS NOT NULL
        GROUP BY h ORDER BY h
    """)
    hourly_pattern = [
        {"hour": r[0], "routed": r[1]} for r in rows
    ]

    # ── 9. Component finding routing ─────────────────────────────
    rows = _safe_rows(cur, """
        SELECT component, agree_with_ai, count(*) as cnt
        FROM component_findings
        GROUP BY component, agree_with_ai
        ORDER BY cnt DESC
    """)
    finding_routing = [
        {"component": r[0], "agreement": r[1] or "pending", "count": r[2]}
        for r in rows
    ]

    # ── 10. Conversation role routing ────────────────────────────
    rows = _safe_rows(cur, """
        SELECT role, count(*) as cnt
        FROM conversation_log
        GROUP BY role ORDER BY cnt DESC
    """)
    conversation_routing = [
        {"role": r[0], "turns": r[1]} for r in rows
    ]

    conn.close()

    return {
        "available": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_routed_events": total_routed,
            "total_clinical_decisions": total_decisions,
            "total_component_findings": total_findings,
            "total_conversations": total_conversations,
            "distinct_components": distinct_components,
            "distinct_actors": distinct_actors,
            "distinct_actions": distinct_actions,
            "unique_patients": unique_patients,
            "auto_routed": auto_routed,
            "human_routed": human_routed,
            "automation_rate_pct": auto_rate,
            "agreement_rate_pct": agreement_rate,
        },
        "component_distribution": component_distribution,
        "action_distribution": action_distribution,
        "actor_distribution": actor_distribution,
        "decision_outcomes": decision_outcomes,
        "routing_matrix": routing_matrix,
        "daily_volume": daily_volume,
        "hourly_pattern": hourly_pattern,
        "finding_routing": finding_routing,
        "conversation_routing": conversation_routing,
    }


# ──────────────────────────────────────────────────────────────
#  /api/routing/breakdown
# ──────────────────────────────────────────────────────────────

def routing_breakdown():
    """Per-route detail: component cross-tab, recent routing events,
    patient routing paths, decision detail."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # ── 1. Component × actor cross-tab ───────────────────────────
    rows = _safe_rows(cur, """
        SELECT component, actor, action, count(*) as cnt
        FROM transaction_log
        GROUP BY component, actor, action
        ORDER BY cnt DESC LIMIT 40
    """)
    cross_tab = [
        {"component": r[0], "actor": r[1], "action": r[2], "count": r[3]}
        for r in rows
    ]

    # ── 2. Recent routing events ─────────────────────────────────
    rows = _safe_rows(cur, """
        SELECT id, patient_id, component, action, actor, detail, ts_utc
        FROM transaction_log
        ORDER BY id DESC LIMIT 25
    """)
    recent_events = [
        {"id": r[0], "patient_id": r[1], "component": r[2],
         "action": r[3], "actor": r[4],
         "detail": r[5][:120] if r[5] else None,
         "ts_utc": r[6]}
        for r in rows
    ]

    # ── 3. Per-patient routing summary ───────────────────────────
    rows = _safe_rows(cur, """
        SELECT patient_id, count(*) as events,
               count(DISTINCT component) as components,
               count(DISTINCT action) as actions,
               count(DISTINCT actor) as actors
        FROM transaction_log
        WHERE patient_id IS NOT NULL
        GROUP BY patient_id
        ORDER BY events DESC LIMIT 20
    """)
    patient_routing = [
        {"patient_id": r[0], "events": r[1], "components": r[2],
         "actions": r[3], "actors": r[4]}
        for r in rows
    ]

    # ── 4. Clinical decision detail ──────────────────────────────
    rows = _safe_rows(cur, """
        SELECT patient_id, ai_prediction, ai_confidence,
               neurologist_agreement, final_decision, reviewer, created_at
        FROM clinical_decisions
        ORDER BY id DESC LIMIT 20
    """)
    decision_detail = [
        {"patient_id": r[0], "ai_prediction": r[1],
         "ai_confidence": r[2], "agreement": r[3],
         "final_decision": r[4], "reviewer": r[5], "created_at": r[6]}
        for r in rows
    ]

    # ── 5. Component routing stats ───────────────────────────────
    rows = _safe_rows(cur, """
        SELECT component,
               count(*) as total,
               count(DISTINCT patient_id) as patients,
               count(DISTINCT actor) as actors,
               count(DISTINCT action) as actions,
               min(ts_utc) as first_event,
               max(ts_utc) as last_event
        FROM transaction_log
        GROUP BY component
        ORDER BY total DESC
    """)
    component_stats = [
        {"component": r[0], "total": r[1], "patients": r[2],
         "actors": r[3], "actions": r[4],
         "first_event": r[5], "last_event": r[6]}
        for r in rows
    ]

    conn.close()

    return {
        "available": True,
        "cross_tab": cross_tab,
        "recent_events": recent_events,
        "patient_routing": patient_routing,
        "decision_detail": decision_detail,
        "component_stats": component_stats,
    }


# ──────────────────────────────────────────────────────────────
#  /api/routing/definitions
# ──────────────────────────────────────────────────────────────

def routing_definitions():
    """Metric definitions for the Routing dashboard."""
    return {
        "available": True,
        "metrics": [
            {"name": "Total Routed Events",
             "definition": "Count of all transaction_log entries representing tasks or actions routed through the clinical pipeline.",
             "source": "transaction_log"},
            {"name": "Automation Rate",
             "definition": "Percentage of events routed by 'system' actor (automated) vs human actors. Higher = more automated pipeline.",
             "source": "transaction_log.actor"},
            {"name": "Component Distribution",
             "definition": "How routed events are distributed across pipeline components (assessment, eeg_upload, patient_chat, etc.).",
             "source": "transaction_log.component"},
            {"name": "Actor Workload",
             "definition": "Distribution of routing events per actor — shows who (system, doctor, agent) handles what volume.",
             "source": "transaction_log.actor"},
            {"name": "Action Distribution",
             "definition": "Breakdown of routing event types (create, process, analyze, query, log, etc.).",
             "source": "transaction_log.action"},
            {"name": "Decision Routing Outcomes",
             "definition": "How clinical decisions flow: AI prediction → neurologist agreement → final decision. Shows confirm/override/pending paths.",
             "source": "clinical_decisions"},
            {"name": "Agreement Rate",
             "definition": "Percentage of clinical decisions where neurologist agreed with AI prediction. Higher = better AI-human alignment.",
             "source": "clinical_decisions.neurologist_agreement"},
            {"name": "Routing Matrix",
             "definition": "Cross-tabulation of component × action showing the most common routing paths in the pipeline.",
             "source": "transaction_log (component, action)"},
            {"name": "Patient Routing",
             "definition": "Per-patient routing summary — how many components, actions, and actors touched each patient's data.",
             "source": "transaction_log GROUP BY patient_id"},
            {"name": "Component Stats",
             "definition": "Per-component aggregate: total events, unique patients, actors, actions, first/last event timestamps.",
             "source": "transaction_log GROUP BY component"},
            {"name": "Daily Volume Trend",
             "definition": "Routing event count per day over the last 30 days. Shows pipeline activity trends.",
             "source": "transaction_log date(ts_utc)"},
            {"name": "Hourly Pattern",
             "definition": "Routing event distribution by hour of day (UTC). Identifies peak routing hours.",
             "source": "transaction_log strftime('%H', ts_utc)"},
        ]
    }
