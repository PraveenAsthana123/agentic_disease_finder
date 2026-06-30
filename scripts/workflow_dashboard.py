"""Workflow Dashboard — consultant workflow execution analytics from clinical.db.

Tracks consultant workflow progress across roles (neurologist, psychiatrist,
psychologist, etc.), phase completion, step execution, sign-off status,
and activity patterns over time.

Sources:
- config/consultant_workflows.json (role definitions, phases, steps, signoffs)
- transaction_log (component + action + actor → workflow activity)
- expert_reviews (sign-off events)
- hitl_reviews (human-in-the-loop reviews as workflow gates)
- clinical_decisions (decision outcomes as workflow completions)
- conversation_log (consultation turns by role)
"""

import sqlite3
import json
import os
from datetime import datetime, timezone

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')
WF = os.path.join(os.path.dirname(__file__), '..', 'config', 'consultant_workflows.json')


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


def _load_workflows():
    if not os.path.exists(WF):
        return {}
    with open(WF) as f:
        data = json.load(f)
    return data.get("workflows", {})


# ──────────────────────────────────────────────────────────────
#  /api/workflow/overview
# ──────────────────────────────────────────────────────────────

def workflow_overview():
    """Aggregate workflow health: role coverage, phase/step counts,
    sign-off status, activity volume, actor distribution, daily trends."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    workflows = _load_workflows()
    conn = _conn()
    cur = conn.cursor()

    # ── 1. Workflow definitions summary ──────────────────────
    role_summaries = []
    total_phases = 0
    total_steps = 0
    total_signoffs = 0
    for role_key, role in workflows.items():
        phases = role.get("phases", [])
        steps = sum(len(p.get("steps", [])) for p in phases)
        signoffs = len(role.get("signoffs", []))
        total_phases += len(phases)
        total_steps += steps
        total_signoffs += signoffs
        role_summaries.append({
            "role_key": role_key,
            "name": role.get("name", role_key),
            "summary": role.get("summary", ""),
            "phases": len(phases),
            "steps": steps,
            "signoffs": signoffs,
        })

    # ── 2. Activity volume from transaction_log ─────────────
    total_events = _safe(cur, "SELECT count(*) FROM transaction_log")
    distinct_components = _safe(
        cur, "SELECT count(DISTINCT component) FROM transaction_log")
    distinct_actors = _safe(
        cur, "SELECT count(DISTINCT actor) FROM transaction_log")
    distinct_actions = _safe(
        cur, "SELECT count(DISTINCT action) FROM transaction_log")
    unique_patients = _safe(
        cur, "SELECT count(DISTINCT patient_id) FROM transaction_log "
        "WHERE patient_id IS NOT NULL")

    # Workflow-relevant components (consultant/council/expert_review)
    wf_components = _safe_rows(cur, """
        SELECT component, count(*) as cnt
        FROM transaction_log
        WHERE component LIKE 'consultant%'
           OR component IN ('council', 'expert_review', 'clinical_trust')
        GROUP BY component ORDER BY cnt DESC
    """)
    workflow_events = sum(r[1] for r in wf_components)

    # ── 3. Expert reviews (sign-off proxy) ───────────────────
    total_expert_reviews = _safe(cur, "SELECT count(*) FROM expert_reviews")
    total_hitl_reviews = _safe(cur, "SELECT count(*) FROM hitl_reviews")
    total_decisions = _safe(cur, "SELECT count(*) FROM clinical_decisions")

    # Expert review details
    expert_rows = _safe_rows(cur, """
        SELECT reviewer_role, verdict, count(*) as cnt
        FROM expert_reviews
        GROUP BY reviewer_role, verdict ORDER BY cnt DESC
    """)
    expert_by_role = {}
    for role, verdict, cnt in expert_rows:
        if role not in expert_by_role:
            expert_by_role[role] = {"total": 0, "verdicts": {}}
        expert_by_role[role]["total"] += cnt
        expert_by_role[role]["verdicts"][verdict] = cnt

    # ── 4. Actor workload (who is active in workflows) ──────
    actor_rows = _safe_rows(cur, """
        SELECT actor, count(*) as cnt
        FROM transaction_log
        GROUP BY actor ORDER BY cnt DESC
    """)
    actor_distribution = [{"actor": a, "events": c} for a, c in actor_rows]

    # Human vs system
    system_events = _safe(
        cur, "SELECT count(*) FROM transaction_log WHERE actor = 'system'")
    human_events = total_events - system_events
    human_rate = round(human_events / max(total_events, 1) * 100, 1)

    # ── 5. Daily activity trend ──────────────────────────────
    daily_rows = _safe_rows(cur, """
        SELECT date(ts_utc) as day, count(*) as cnt
        FROM transaction_log
        WHERE ts_utc IS NOT NULL
        GROUP BY day ORDER BY day
    """)
    daily_trend = [{"date": d, "events": c} for d, c in daily_rows]

    # ── 6. Action type distribution ──────────────────────────
    action_rows = _safe_rows(cur, """
        SELECT action, count(*) as cnt
        FROM transaction_log
        GROUP BY action ORDER BY cnt DESC
    """)
    action_distribution = [{"action": a, "count": c} for a, c in action_rows]

    # ── 7. Conversation workflow turns ───────────────────────
    conv_total = _safe(cur, "SELECT count(*) FROM conversation_log")
    conv_by_role = _safe_rows(cur, """
        SELECT role, count(*) as cnt
        FROM conversation_log
        GROUP BY role ORDER BY cnt DESC
    """)
    conversation_roles = [{"role": r, "turns": c} for r, c in conv_by_role]

    conn.close()

    return {
        "available": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_roles": len(role_summaries),
            "total_phases": total_phases,
            "total_steps": total_steps,
            "total_signoffs": total_signoffs,
            "workflow_events": workflow_events,
            "total_events": total_events,
            "expert_reviews": total_expert_reviews,
            "hitl_reviews": total_hitl_reviews,
            "clinical_decisions": total_decisions,
            "distinct_actors": distinct_actors,
            "distinct_components": distinct_components,
            "human_rate_pct": human_rate,
            "unique_patients": unique_patients,
            "conversation_turns": conv_total,
        },
        "roles": role_summaries,
        "workflow_components": [{"component": c, "events": n}
                                for c, n in wf_components],
        "actor_distribution": actor_distribution,
        "action_distribution": action_distribution,
        "expert_by_role": expert_by_role,
        "conversation_roles": conversation_roles,
        "daily_trend": daily_trend,
    }


# ──────────────────────────────────────────────────────────────
#  /api/workflow/breakdown
# ──────────────────────────────────────────────────────────────

def workflow_breakdown():
    """Per-role breakdown: phases, steps, signoffs, and matching
    transaction_log activity for each consultant role."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    workflows = _load_workflows()
    conn = _conn()
    cur = conn.cursor()

    role_details = []
    for role_key, role in workflows.items():
        phases = role.get("phases", [])
        signoffs = role.get("signoffs", [])

        # Count matching events for this role
        role_events = _safe(
            cur,
            f"SELECT count(*) FROM transaction_log "
            f"WHERE component LIKE 'consultant:{role_key}%'"
        )
        # Expert reviews by this role
        expert_cnt = _safe(
            cur,
            f"SELECT count(*) FROM expert_reviews "
            f"WHERE reviewer_role = '{role_key}'"
        )
        # Conversation turns by role
        conv_cnt = _safe(
            cur,
            f"SELECT count(*) FROM conversation_log "
            f"WHERE role = '{role_key}'"
        )

        phase_details = []
        for p in phases:
            steps = p.get("steps", [])
            phase_details.append({
                "name": p.get("name", ""),
                "step_count": len(steps),
                "steps": [{"step": s.get("step", ""),
                           "input": s.get("input", ""),
                           "task": s.get("task", ""),
                           "output": s.get("output", "")}
                          for s in steps],
            })

        role_details.append({
            "role_key": role_key,
            "name": role.get("name", role_key),
            "summary": role.get("summary", ""),
            "total_phases": len(phases),
            "total_steps": sum(len(p.get("steps", [])) for p in phases),
            "signoffs": signoffs,
            "signoff_count": len(signoffs),
            "events_logged": role_events,
            "expert_reviews": expert_cnt,
            "conversation_turns": conv_cnt,
            "phases": phase_details,
        })

    # ── Component-action cross-tab ───────────────────────────
    cross_rows = _safe_rows(cur, """
        SELECT component, action, count(*) as cnt
        FROM transaction_log
        WHERE component LIKE 'consultant%'
           OR component IN ('council', 'expert_review', 'clinical_trust')
        GROUP BY component, action
        ORDER BY cnt DESC
    """)
    component_actions = [{"component": c, "action": a, "count": n}
                         for c, a, n in cross_rows]

    # ── Recent workflow-related events ───────────────────────
    recent_rows = _safe_rows(cur, """
        SELECT ts_utc, component, action, actor, patient_id, detail
        FROM transaction_log
        WHERE component LIKE 'consultant%'
           OR component IN ('council', 'expert_review', 'clinical_trust')
        ORDER BY ts_utc DESC LIMIT 20
    """)
    recent_events = [{"ts": ts, "component": comp, "action": act,
                      "actor": actor, "patient_id": pid,
                      "detail": det}
                     for ts, comp, act, actor, pid, det in recent_rows]

    # ── Hourly pattern ───────────────────────────────────────
    hourly = _safe_rows(cur, """
        SELECT cast(strftime('%H', ts_utc) as integer) as hr, count(*)
        FROM transaction_log
        WHERE ts_utc IS NOT NULL
        GROUP BY hr ORDER BY hr
    """)
    hourly_pattern = [{"hour": h, "events": c} for h, c in hourly]

    # ── Patient workflow depth ───────────────────────────────
    patient_depth = _safe_rows(cur, """
        SELECT patient_id, count(DISTINCT component) as comps,
               count(DISTINCT action) as actions, count(*) as events
        FROM transaction_log
        WHERE patient_id IS NOT NULL
        GROUP BY patient_id
        ORDER BY events DESC LIMIT 20
    """)
    patient_workflows = [{"patient_id": pid, "components": comps,
                          "actions": acts, "events": evts}
                         for pid, comps, acts, evts in patient_depth]

    conn.close()

    return {
        "available": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "roles": role_details,
        "component_actions": component_actions,
        "recent_events": recent_events,
        "hourly_pattern": hourly_pattern,
        "patient_workflows": patient_workflows,
    }


# ──────────────────────────────────────────────────────────────
#  /api/workflow/definitions
# ──────────────────────────────────────────────────────────────

def workflow_definitions():
    """Metric definitions for the Workflow Dashboard."""
    return {
        "available": True,
        "definitions": [
            {"metric": "Total Roles",
             "definition": "Number of consultant roles defined in consultant_workflows.json (neurologist, psychiatrist, etc.)"},
            {"metric": "Total Phases",
             "definition": "Sum of all workflow phases across all roles — each phase is a sequential stage of clinical oversight"},
            {"metric": "Total Steps",
             "definition": "Sum of all individual steps across all phases — each step has input, task, and output"},
            {"metric": "Total Sign-offs",
             "definition": "Sum of sign-off gates across all roles — each sign-off is a required approval checkpoint"},
            {"metric": "Workflow Events",
             "definition": "Transaction log entries from workflow-related components (consultant:*, council, expert_review, clinical_trust)"},
            {"metric": "Expert Reviews",
             "definition": "Count of expert_reviews table rows — formal sign-off events by clinical experts"},
            {"metric": "HITL Reviews",
             "definition": "Count of hitl_reviews rows — human-in-the-loop review gates for AI decisions"},
            {"metric": "Clinical Decisions",
             "definition": "Count of clinical_decisions rows — final clinical decision outcomes after workflow completion"},
            {"metric": "Human Rate %",
             "definition": "Percentage of transaction_log events with actor != 'system' — measures human participation in workflow"},
            {"metric": "Actor Distribution",
             "definition": "Breakdown of events by actor identity — shows workload across human and system actors"},
            {"metric": "Daily Trend",
             "definition": "Transaction events aggregated by date — shows workflow activity volume over time"},
            {"metric": "Hourly Pattern",
             "definition": "Events by hour of day — reveals when workflow activity peaks (working hours vs off-hours)"},
            {"metric": "Patient Workflow Depth",
             "definition": "Per-patient: distinct components touched, distinct actions, total events — measures workflow coverage per patient"},
            {"metric": "Conversation Roles",
             "definition": "Conversation log turns grouped by role — shows consultation discussion distribution"},
        ],
    }
