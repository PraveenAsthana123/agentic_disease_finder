"""MCP Overview Dashboard -- system-wide MCP health, tool/action catalog,
protocol compliance, security audit trail, conversation flow, and activity patterns.

Different from mcp_federation_dashboard.py (cross-component topology/edges).
This dashboard focuses on the overall MCP system: component registry, action usage,
guardrail enforcement, security events, conversation analysis, and uptime patterns.

Sources:
- transaction_log  -- component/action audit trail
- conversation_log -- role-based conversation flow
- analyses         -- analysis pipeline outputs
- expert_reviews   -- human expert oversight
- hitl_reviews     -- human-in-the-loop gate
- clinical_decisions -- clinical decision routing
"""

import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql):
    try:
        cur.execute(sql)
        r = cur.fetchone()
        return r[0] if r else 0
    except Exception:
        return 0


def _safe_rows(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchall()
    except Exception:
        return []


# ── Overview ─────────────────────────────────────────────────────

def mcp_overview():
    """System-wide MCP health: component registry, action catalog, actor summary,
    protocol compliance, daily/hourly activity, guardrail enforcement."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()
    now = datetime.now(timezone.utc)
    result = {"available": True, "generated_at": now.isoformat()}

    # ── Totals ──
    total_transactions = _safe(cur, "SELECT count(*) FROM transaction_log")
    total_conversations = _safe(cur, "SELECT count(*) FROM conversation_log")
    total_analyses = _safe(cur, "SELECT count(*) FROM analyses")

    components = _safe_rows(
        cur, "SELECT DISTINCT component FROM transaction_log ORDER BY component"
    )
    total_components = len(components)

    actors = _safe_rows(
        cur, "SELECT DISTINCT actor FROM transaction_log"
    )
    total_actors = len(actors)

    actions = _safe_rows(
        cur, "SELECT DISTINCT action FROM transaction_log"
    )
    total_actions = len(actions)

    # ── Guardrail / compliance events ──
    guardrail_events = _safe(
        cur,
        "SELECT count(*) FROM transaction_log "
        "WHERE lower(action) LIKE '%guard%' "
        "OR lower(action) LIKE '%security%' "
        "OR lower(action) LIKE '%audit%'"
    )
    compliance_rate = round(
        guardrail_events / max(total_transactions, 1) * 100, 1
    )

    result["summary"] = {
        "total_components": total_components,
        "total_transactions": total_transactions,
        "total_conversations": total_conversations,
        "total_analyses": total_analyses,
        "total_actors": total_actors,
        "total_actions": total_actions,
        "guardrail_events": guardrail_events,
        "compliance_rate": compliance_rate,
    }

    # ── Component health ──
    # Determine cutoff for "active" (7 days from most recent transaction)
    max_ts = _safe(cur, "SELECT max(ts_utc) FROM transaction_log")
    try:
        if max_ts:
            latest = datetime.fromisoformat(str(max_ts).replace("Z", "+00:00"))
        else:
            latest = now
    except Exception:
        latest = now
    stale_cutoff = (latest - timedelta(days=7)).isoformat()

    comp_health_rows = _safe_rows(
        cur,
        "SELECT component, count(*), max(ts_utc) "
        "FROM transaction_log GROUP BY component ORDER BY count(*) DESC"
    )
    component_health = []
    for comp, txn_count, last_active in comp_health_rows:
        status = "active"
        try:
            if last_active and last_active < stale_cutoff:
                status = "stale"
        except Exception:
            status = "unknown"
        component_health.append({
            "component": comp,
            "transactions": txn_count,
            "last_active": last_active,
            "status": status,
        })
    result["component_health"] = component_health

    # ── Action catalog ──
    action_rows = _safe_rows(
        cur,
        "SELECT action, count(*) FROM transaction_log "
        "GROUP BY action ORDER BY count(*) DESC"
    )
    action_catalog = []
    for action, count in action_rows:
        action_comps = _safe_rows(
            cur,
            f"SELECT DISTINCT component FROM transaction_log "
            f"WHERE action = ?",
        ) if False else []
        # Use parameterized query via raw execute
        try:
            cur.execute(
                "SELECT DISTINCT component FROM transaction_log WHERE action = ?",
                (action,)
            )
            action_comps = [r[0] for r in cur.fetchall()]
        except Exception:
            action_comps = []
        action_catalog.append({
            "action": action,
            "count": count,
            "components": action_comps,
        })
    result["action_catalog"] = action_catalog

    # ── Actor summary ──
    actor_rows = _safe_rows(
        cur,
        "SELECT actor, count(*), count(DISTINCT component), count(DISTINCT action) "
        "FROM transaction_log GROUP BY actor ORDER BY count(*) DESC"
    )
    actor_summary = [
        {
            "actor": r[0],
            "transactions": r[1],
            "components_touched": r[2],
            "actions_used": r[3],
        }
        for r in actor_rows
    ]
    result["actor_summary"] = actor_summary

    # ── Daily activity (last 30 days) ──
    # Find the date range based on actual data
    daily_tx = _safe_rows(
        cur,
        "SELECT substr(ts_utc, 1, 10) AS day, count(*) "
        "FROM transaction_log WHERE ts_utc IS NOT NULL "
        "GROUP BY day ORDER BY day DESC LIMIT 30"
    )
    daily_conv = _safe_rows(
        cur,
        "SELECT substr(ts_utc, 1, 10) AS day, count(*) "
        "FROM conversation_log WHERE ts_utc IS NOT NULL "
        "GROUP BY day ORDER BY day DESC LIMIT 30"
    )
    # Merge into single list
    conv_map = {r[0]: r[1] for r in daily_conv}
    daily_activity = []
    for day, tx_count in reversed(daily_tx):
        daily_activity.append({
            "date": day,
            "transactions": tx_count,
            "conversations": conv_map.get(day, 0),
        })
    result["daily_activity"] = daily_activity

    # ── Hourly heatmap ──
    hourly = _safe_rows(
        cur,
        "SELECT CAST(substr(ts_utc, 12, 2) AS INTEGER) AS hour, count(*) "
        "FROM transaction_log WHERE ts_utc IS NOT NULL AND length(ts_utc) >= 13 "
        "GROUP BY hour ORDER BY hour"
    )
    hourly_heatmap = [{"hour": r[0], "transactions": r[1]} for r in hourly]
    result["hourly_heatmap"] = hourly_heatmap

    # ── Protocol compliance ──
    security_events = _safe(
        cur,
        "SELECT count(*) FROM transaction_log WHERE lower(action) LIKE '%security%'"
    )
    audit_events = _safe(
        cur,
        "SELECT count(*) FROM transaction_log WHERE lower(action) LIKE '%audit%'"
    )
    result["protocol_compliance"] = {
        "total_transactions": total_transactions,
        "guardrail_events": guardrail_events,
        "compliance_rate_pct": compliance_rate,
        "security_events": security_events,
        "audit_events": audit_events,
    }

    conn.close()
    return result


# ── Breakdown ────────────────────────────────────────────────────

def mcp_overview_breakdown():
    """Detailed breakdown: component-action matrix, conversation roles/components,
    patient coverage, recent events, security audit log, component interconnections."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # ── Component-action matrix (top 15 components) ──
    top_comps = _safe_rows(
        cur,
        "SELECT component, count(*) FROM transaction_log "
        "GROUP BY component ORDER BY count(*) DESC LIMIT 15"
    )
    component_action_matrix = []
    for comp, _ in top_comps:
        try:
            cur.execute(
                "SELECT action, count(*) FROM transaction_log "
                "WHERE component = ? GROUP BY action ORDER BY count(*) DESC",
                (comp,)
            )
            actions = {r[0]: r[1] for r in cur.fetchall()}
        except Exception:
            actions = {}
        component_action_matrix.append({
            "component": comp,
            "actions": actions,
        })

    # ── Conversation roles ──
    role_rows = _safe_rows(
        cur,
        "SELECT role, count(*) FROM conversation_log "
        "GROUP BY role ORDER BY count(*) DESC"
    )
    conversation_roles = [{"role": r[0], "count": r[1]} for r in role_rows]

    # ── Conversation components ──
    # conversation_log has no component column; derive from transaction_log
    # where component relates to communication/chat
    conv_comp_rows = _safe_rows(
        cur,
        "SELECT component, count(*) FROM transaction_log "
        "WHERE lower(component) LIKE '%chat%' "
        "OR lower(component) LIKE '%conversation%' "
        "OR lower(component) LIKE '%message%' "
        "OR lower(component) LIKE '%bot%' "
        "GROUP BY component ORDER BY count(*) DESC"
    )
    conversation_components = [{"component": r[0], "count": r[1]} for r in conv_comp_rows]
    # If no chat-specific components found, show all components with their
    # transaction counts as a fallback
    if not conversation_components:
        fallback = _safe_rows(
            cur,
            "SELECT component, count(*) FROM transaction_log "
            "GROUP BY component ORDER BY count(*) DESC LIMIT 10"
        )
        conversation_components = [{"component": r[0], "count": r[1]} for r in fallback]

    # ── Patient coverage ──
    total_patients_tx = _safe(
        cur, "SELECT count(DISTINCT patient_id) FROM transaction_log"
    )
    patients_with_conversations = 0
    # conversation_log has no patient_id; try to count via transaction_log chat components
    try:
        cur.execute(
            "SELECT count(DISTINCT patient_id) FROM transaction_log "
            "WHERE lower(component) LIKE '%chat%'"
        )
        r = cur.fetchone()
        patients_with_conversations = r[0] if r else 0
    except Exception:
        pass

    patients_with_analyses = _safe(
        cur, "SELECT count(DISTINCT patient_id) FROM analyses"
    )
    patients_with_reviews = 0
    try:
        # Union of expert_reviews and hitl_reviews patient_ids
        cur.execute(
            "SELECT count(DISTINCT patient_id) FROM ("
            "  SELECT patient_id FROM expert_reviews "
            "  UNION "
            "  SELECT patient_id FROM hitl_reviews"
            ")"
        )
        r = cur.fetchone()
        patients_with_reviews = r[0] if r else 0
    except Exception:
        pass

    patient_coverage = {
        "total_patients": total_patients_tx,
        "patients_with_conversations": patients_with_conversations,
        "patients_with_analyses": patients_with_analyses,
        "patients_with_reviews": patients_with_reviews,
    }

    # ── Recent events (last 20 transactions) ──
    recent = _safe_rows(
        cur,
        "SELECT id, patient_id, component, action, actor, ts_utc "
        "FROM transaction_log ORDER BY id DESC LIMIT 20"
    )
    recent_events = [
        {
            "id": r[0],
            "patient_id": r[1],
            "component": r[2],
            "action": r[3],
            "actor": r[4],
            "created_at": r[5],
        }
        for r in recent
    ]

    # ── Security audit log ──
    security_rows = _safe_rows(
        cur,
        "SELECT id, patient_id, component, action, actor, detail, ts_utc "
        "FROM transaction_log "
        "WHERE lower(action) LIKE '%guard%' "
        "OR lower(action) LIKE '%security%' "
        "OR lower(action) LIKE '%audit%' "
        "OR lower(action) LIKE '%override%' "
        "ORDER BY id DESC LIMIT 50"
    )
    security_audit_log = [
        {
            "id": r[0],
            "patient_id": r[1],
            "component": r[2],
            "action": r[3],
            "actor": r[4],
            "detail": (r[5] or "")[:200],
            "created_at": r[6],
        }
        for r in security_rows
    ]

    # ── Component interconnections ──
    comp_interconn_rows = _safe_rows(
        cur,
        "SELECT component, "
        "count(DISTINCT patient_id), "
        "count(DISTINCT actor), "
        "count(DISTINCT action) "
        "FROM transaction_log GROUP BY component ORDER BY count(*) DESC"
    )
    component_interconnections = [
        {
            "component": r[0],
            "unique_patients": r[1],
            "unique_actors": r[2],
            "unique_actions": r[3],
        }
        for r in comp_interconn_rows
    ]

    conn.close()

    return {
        "available": True,
        "component_action_matrix": component_action_matrix,
        "conversation_roles": conversation_roles,
        "conversation_components": conversation_components,
        "patient_coverage": patient_coverage,
        "recent_events": recent_events,
        "security_audit_log": security_audit_log,
        "component_interconnections": component_interconnections,
    }


# ── Definitions ──────────────────────────────────────────────────

def mcp_overview_definitions():
    """Metric definitions for the MCP Overview dashboard."""
    return {
        "metrics": [
            {
                "name": "Total Components",
                "description": "Number of distinct MCP components (services) registered in the transaction log.",
            },
            {
                "name": "Total Transactions",
                "description": "Total recorded actions across all components -- the primary throughput metric.",
            },
            {
                "name": "Total Conversations",
                "description": "Total messages in the conversation log across all roles.",
            },
            {
                "name": "Total Analyses",
                "description": "Number of ML/clinical analyses completed and stored.",
            },
            {
                "name": "Total Actors",
                "description": "Distinct actors (users, agents, systems) that have performed actions.",
            },
            {
                "name": "Total Actions",
                "description": "Distinct action types (protocol verbs) used across the system.",
            },
            {
                "name": "Guardrail Events",
                "description": "Transactions whose action contains 'guard', 'security', or 'audit' -- indicates compliance enforcement.",
            },
            {
                "name": "Compliance Rate",
                "description": "Guardrail events as a percentage of total transactions -- higher means more guardrails enforced.",
            },
            {
                "name": "Component Health",
                "description": "Per-component status: 'active' if last transaction within 7 days of latest activity, 'stale' otherwise.",
            },
            {
                "name": "Action Catalog",
                "description": "All distinct actions with usage counts and which components invoke them.",
            },
            {
                "name": "Actor Summary",
                "description": "Per-actor breakdown: transaction count, components touched, distinct actions used.",
            },
            {
                "name": "Daily Activity",
                "description": "Transaction and conversation counts per day for the last 30 days of data.",
            },
            {
                "name": "Hourly Heatmap",
                "description": "Transaction volume by hour of day (0-23) to reveal activity patterns.",
            },
            {
                "name": "Protocol Compliance",
                "description": "Breakdown of compliance signals: guardrail, security, and audit event counts with overall rate.",
            },
            {
                "name": "Component-Action Matrix",
                "description": "For each of the top 15 components, the set of actions and their counts.",
            },
            {
                "name": "Conversation Roles",
                "description": "Distribution of conversation messages by role (system, user, assistant, etc.).",
            },
            {
                "name": "Patient Coverage",
                "description": "How many distinct patients appear in transactions, conversations, analyses, and reviews.",
            },
            {
                "name": "Security Audit Log",
                "description": "Recent transactions involving guard, security, audit, or override actions -- the security trail.",
            },
            {
                "name": "Component Interconnections",
                "description": "Per-component summary of unique patients, actors, and actions -- measures integration breadth.",
            },
        ]
    }
