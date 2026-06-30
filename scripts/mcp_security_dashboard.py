"""MCP Security Dashboard -- guardrail enforcement, access control patterns,
security audit trail, compliance agent activity, actor privilege analysis,
and threat surface from clinical.db transaction/conversation logs.

Surfaces:
- Guardrail events (blocked, compliance checks, sign-offs, human_decisions)
- Actor privilege matrix (who touches what components/actions)
- Security agent and compliance agent activity
- Access pattern anomalies (unusual actor-component pairs)
- Patient data access audit (which actors access which patients)
- Temporal security patterns (guardrail events by hour/day)
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


# ── Security event classification ─────────────────────────────

SECURITY_ACTIONS = {'blocked', 'sign-off', 'human_decision'}
SECURITY_ACTORS = {'security_agent', 'compliance_agent'}
PRIVILEGED_ACTIONS = {'delete', 'update', 'human_decision', 'sign-off', 'blocked'}


# ── Overview ──────────────────────────────────────────────────

def mcp_security_overview():
    """Security posture: guardrail events, compliance rate, actor privilege
    summary, security agent activity, blocked events, access control stats."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()
    now = datetime.now(timezone.utc)
    result = {"available": True, "generated_at": now.isoformat()}

    # ── Totals ──
    total_tx = _safe(cur, "SELECT count(*) FROM transaction_log")
    total_conversations = _safe(cur, "SELECT count(*) FROM conversation_log")

    # Security-relevant events
    blocked_count = _safe(
        cur, "SELECT count(*) FROM transaction_log WHERE action='blocked'"
    )
    signoff_count = _safe(
        cur, "SELECT count(*) FROM transaction_log WHERE action='sign-off'"
    )
    human_decision_count = _safe(
        cur, "SELECT count(*) FROM transaction_log WHERE action='human_decision'"
    )
    expert_review_count = _safe(cur, "SELECT count(*) FROM expert_reviews")
    hitl_review_count = _safe(cur, "SELECT count(*) FROM hitl_reviews")
    clinical_decision_count = _safe(cur, "SELECT count(*) FROM clinical_decisions")

    guardrail_total = blocked_count + signoff_count + human_decision_count
    guardrail_rate = round(guardrail_total / max(total_tx, 1) * 100, 1)

    # Human oversight rate
    human_oversight_events = (
        signoff_count + human_decision_count
        + expert_review_count + hitl_review_count
    )
    oversight_rate = round(human_oversight_events / max(total_tx, 1) * 100, 1)

    result["summary"] = {
        "total_transactions": total_tx,
        "total_conversations": total_conversations,
        "guardrail_events": guardrail_total,
        "guardrail_rate_pct": guardrail_rate,
        "blocked_events": blocked_count,
        "signoff_events": signoff_count,
        "human_decisions": human_decision_count,
        "expert_reviews": expert_review_count,
        "hitl_reviews": hitl_review_count,
        "clinical_decisions": clinical_decision_count,
        "human_oversight_events": human_oversight_events,
        "oversight_rate_pct": oversight_rate,
    }

    # ── Actor privilege matrix ──
    actor_rows = _safe_rows(
        cur,
        "SELECT actor, count(*) as txns, "
        "count(DISTINCT component) as comps, "
        "count(DISTINCT action) as acts, "
        "count(DISTINCT patient_id) as patients "
        "FROM transaction_log GROUP BY actor ORDER BY txns DESC"
    )
    actor_privileges = []
    for r in actor_rows:
        actor = r[0]
        is_security = actor in SECURITY_ACTORS
        # Check if actor has privileged actions
        priv_rows = _safe_rows(
            cur,
            "SELECT DISTINCT action FROM transaction_log "
            f"WHERE actor = '{actor}' AND action IN "
            "('delete','update','human_decision','sign-off','blocked')"
        )
        priv_actions = [p[0] for p in priv_rows]
        risk_level = "high" if len(priv_actions) >= 2 else (
            "medium" if priv_actions else "low"
        )
        actor_privileges.append({
            "actor": actor,
            "transactions": r[1],
            "components": r[2],
            "actions": r[3],
            "patients_accessed": r[4],
            "is_security_actor": is_security,
            "privileged_actions": priv_actions,
            "risk_level": risk_level,
        })
    result["actor_privileges"] = actor_privileges

    # ── Security agent activity ──
    sec_events = _safe_rows(
        cur,
        "SELECT actor, action, component, detail, ts_utc "
        "FROM transaction_log "
        "WHERE actor IN ('security_agent','compliance_agent') "
        "ORDER BY ts_utc DESC"
    )
    security_events = [
        {
            "actor": r[0], "action": r[1], "component": r[2],
            "detail": r[3], "timestamp": r[4],
        }
        for r in sec_events
    ]
    result["security_agent_events"] = security_events

    # ── Guardrail event log ──
    guard_events = _safe_rows(
        cur,
        "SELECT actor, action, component, patient_id, detail, ts_utc "
        "FROM transaction_log "
        "WHERE action IN ('blocked','sign-off','human_decision') "
        "ORDER BY ts_utc DESC"
    )
    guardrail_log = [
        {
            "actor": r[0], "action": r[1], "component": r[2],
            "patient_id": r[3], "detail": r[4], "timestamp": r[5],
        }
        for r in guard_events
    ]
    result["guardrail_log"] = guardrail_log

    # ── Component access frequency (potential attack surface) ──
    comp_access = _safe_rows(
        cur,
        "SELECT component, count(DISTINCT actor) as actors, count(*) as txns "
        "FROM transaction_log GROUP BY component ORDER BY actors DESC"
    )
    attack_surface = [
        {
            "component": r[0],
            "distinct_actors": r[1],
            "transactions": r[2],
            "exposure": "high" if r[1] >= 3 else ("medium" if r[1] >= 2 else "low"),
        }
        for r in comp_access
    ]
    result["attack_surface"] = attack_surface

    conn.close()
    return result


# ── Breakdown ─────────────────────────────────────────────────

def mcp_security_breakdown():
    """Drill-down: patient data access audit, actor-component cross-tab,
    temporal security patterns, conversation role analysis, detailed event log."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()
    result = {"available": True}

    # ── Patient data access audit ──
    patient_access = _safe_rows(
        cur,
        "SELECT patient_id, count(DISTINCT actor) as actors, "
        "count(DISTINCT component) as components, count(*) as events "
        "FROM transaction_log WHERE patient_id IS NOT NULL AND patient_id != '' "
        "GROUP BY patient_id ORDER BY actors DESC, events DESC LIMIT 30"
    )
    result["patient_access_audit"] = [
        {
            "patient_id": r[0],
            "distinct_actors": r[1],
            "distinct_components": r[2],
            "total_events": r[3],
            "access_risk": "elevated" if r[1] > 2 else "normal",
        }
        for r in patient_access
    ]

    # ── Actor-Component cross-tab ──
    cross_tab = _safe_rows(
        cur,
        "SELECT actor, component, count(*) "
        "FROM transaction_log GROUP BY actor, component "
        "ORDER BY actor, count(*) DESC"
    )
    actor_comp_map = defaultdict(list)
    for actor, comp, cnt in cross_tab:
        actor_comp_map[actor].append({"component": comp, "count": cnt})
    result["actor_component_matrix"] = [
        {"actor": a, "components": comps}
        for a, comps in actor_comp_map.items()
    ]

    # ── Actor-Action cross-tab ──
    action_tab = _safe_rows(
        cur,
        "SELECT actor, action, count(*) "
        "FROM transaction_log GROUP BY actor, action "
        "ORDER BY actor, count(*) DESC"
    )
    actor_action_map = defaultdict(list)
    for actor, action, cnt in action_tab:
        actor_action_map[actor].append({"action": action, "count": cnt})
    result["actor_action_matrix"] = [
        {"actor": a, "actions": acts}
        for a, acts in actor_action_map.items()
    ]

    # ── Daily security events ──
    daily_guard = _safe_rows(
        cur,
        "SELECT substr(ts_utc, 1, 10) AS day, "
        "sum(CASE WHEN action='blocked' THEN 1 ELSE 0 END) as blocked, "
        "sum(CASE WHEN action='sign-off' THEN 1 ELSE 0 END) as signoffs, "
        "sum(CASE WHEN action='human_decision' THEN 1 ELSE 0 END) as decisions, "
        "count(*) as total "
        "FROM transaction_log WHERE ts_utc IS NOT NULL "
        "GROUP BY day ORDER BY day"
    )
    result["daily_security"] = [
        {
            "date": r[0], "blocked": r[1], "signoffs": r[2],
            "human_decisions": r[3], "total_events": r[4],
        }
        for r in daily_guard
    ]

    # ── Hourly activity pattern ──
    hourly = _safe_rows(
        cur,
        "SELECT substr(ts_utc, 12, 2) AS hour, count(*) "
        "FROM transaction_log WHERE ts_utc IS NOT NULL "
        "GROUP BY hour ORDER BY hour"
    )
    result["hourly_pattern"] = [
        {"hour": int(r[0]) if r[0] else 0, "events": r[1]}
        for r in hourly
    ]

    # ── Conversation role security ──
    conv_roles = _safe_rows(
        cur,
        "SELECT role, count(*) FROM conversation_log "
        "GROUP BY role ORDER BY count(*) DESC"
    )
    result["conversation_roles"] = [
        {"role": r[0], "count": r[1]}
        for r in conv_roles
    ]

    # ── Recent high-privilege events ──
    recent = _safe_rows(
        cur,
        "SELECT actor, action, component, patient_id, detail, ts_utc "
        "FROM transaction_log "
        "WHERE action IN ('delete','update','human_decision','sign-off','blocked',"
        "'ingest','scheduled_train') "
        "ORDER BY ts_utc DESC LIMIT 20"
    )
    result["recent_privileged_events"] = [
        {
            "actor": r[0], "action": r[1], "component": r[2],
            "patient_id": r[3], "detail": r[4], "timestamp": r[5],
        }
        for r in recent
    ]

    # ── Expert review + HITL detail ──
    expert_rows = _safe_rows(
        cur,
        "SELECT patient_id, fields_json, created_at "
        "FROM expert_reviews ORDER BY created_at DESC"
    )
    result["expert_reviews"] = [
        {"patient_id": r[0], "detail": r[1][:200] if r[1] else "", "timestamp": r[2]}
        for r in expert_rows
    ]

    hitl_rows = _safe_rows(
        cur,
        "SELECT patient_id, fields_json, created_at "
        "FROM hitl_reviews ORDER BY created_at DESC"
    )
    result["hitl_reviews"] = [
        {"patient_id": r[0], "detail": r[1][:200] if r[1] else "", "timestamp": r[2]}
        for r in hitl_rows
    ]

    conn.close()
    return result


# ── Definitions ───────────────────────────────────────────────

def mcp_security_definitions():
    """Metric definitions for tooltip overlays on the MCP Security dashboard."""
    return {
        "available": True,
        "definitions": [
            {
                "term": "Guardrail Events",
                "definition": "Total count of blocked, sign-off, and human_decision actions in the transaction log. These represent security gates that prevent or validate system actions.",
            },
            {
                "term": "Guardrail Rate",
                "definition": "Percentage of all transactions that triggered a guardrail event (blocked + sign-off + human_decision) / total transactions.",
            },
            {
                "term": "Blocked Events",
                "definition": "Actions explicitly blocked by the security agent or compliance rules. Indicates enforcement of access control policies.",
            },
            {
                "term": "Human Oversight Rate",
                "definition": "Percentage of transactions that involved human review: sign-offs + human_decisions + expert_reviews + HITL reviews. Higher = more human-in-the-loop governance.",
            },
            {
                "term": "Actor Privilege Matrix",
                "definition": "Per-actor breakdown of components accessed, actions performed, and patients touched. Identifies over-privileged actors or unusual access patterns.",
            },
            {
                "term": "Attack Surface",
                "definition": "Components ranked by number of distinct actors that access them. More actors = broader exposure = higher attack surface.",
            },
            {
                "term": "Patient Access Audit",
                "definition": "Per-patient count of distinct actors and components that accessed their data. Elevated risk = more than 2 distinct actors accessing a single patient record.",
            },
            {
                "term": "Security Agent Events",
                "definition": "Actions taken by the security_agent and compliance_agent actors, including blocked requests and compliance decisions.",
            },
            {
                "term": "Privileged Actions",
                "definition": "High-impact actions: delete, update, human_decision, sign-off, blocked, ingest, scheduled_train. These require audit scrutiny.",
            },
            {
                "term": "Conversation Roles",
                "definition": "Distribution of conversation participants by role (system, user, assistant). Abnormal role ratios may indicate injection or misuse.",
            },
        ],
    }
