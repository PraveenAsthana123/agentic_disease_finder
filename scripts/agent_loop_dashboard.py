"""Agent Loop / Goal-Drift Dashboard — real metrics from clinical.db.

Tracks agent iteration patterns, component activity distribution,
goal-drift detection (feedback corrections, decision disagreements,
confidence shifts), and conversation-loop health.

Sources:
- transaction_log (component actions → agent loop activity)
- conversation_log (assistant/operator turns → conversation loop metrics)
- feedback (corrections → goal-drift signal)
- clinical_decisions (AI confidence, agreement → drift indicators)
- component_findings (doctor agree/disagree → alignment drift)
- hitl_reviews (override decisions → human-AI divergence)
"""

import sqlite3
import os
import json
from datetime import datetime, timezone

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe_count(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchone()[0]
    except Exception:
        return 0


def loop_overview():
    """Aggregate agent-loop health: component cycle counts, conversation
    turns, feedback corrections, confidence drift, alignment scores."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # ── 1. Agent loop activity (transaction_log actions per component) ──
    total_actions = _safe_count(cur, "SELECT count(*) FROM transaction_log")

    try:
        cur.execute("""
            SELECT component, count(*) as cnt
            FROM transaction_log
            GROUP BY component ORDER BY cnt DESC
        """)
        actions_by_component = [
            {"component": r[0], "actions": r[1]} for r in cur.fetchall()
        ]
    except Exception:
        actions_by_component = []

    try:
        cur.execute("""
            SELECT action, count(*) as cnt
            FROM transaction_log
            GROUP BY action ORDER BY cnt DESC
        """)
        actions_by_type = [
            {"action": r[0], "count": r[1]} for r in cur.fetchall()
        ]
    except Exception:
        actions_by_type = []

    # Blocked ratio (loop interruptions)
    blocked = _safe_count(
        cur, "SELECT count(*) FROM transaction_log WHERE action = 'blocked'")
    block_rate = round(blocked / max(total_actions, 1) * 100, 1)

    # ── 2. Conversation loop metrics ────────────────────────────────
    total_turns = _safe_count(cur, "SELECT count(*) FROM conversation_log")
    assistant_turns = _safe_count(
        cur, "SELECT count(*) FROM conversation_log WHERE role = 'assistant'")
    operator_turns = _safe_count(
        cur, "SELECT count(*) FROM conversation_log WHERE role = 'operator'")

    # Average assistant-to-operator ratio (loop responsiveness)
    loop_ratio = round(assistant_turns / max(operator_turns, 1), 2)

    # ── 3. Goal-drift: feedback corrections ─────────────────────────
    feedback_total = _safe_count(cur, "SELECT count(*) FROM feedback")
    corrections = _safe_count(
        cur,
        "SELECT count(*) FROM feedback WHERE correction IS NOT NULL "
        "AND correction != ''")
    correction_rate = round(corrections / max(feedback_total, 1) * 100, 1)

    # Rating distribution (drift in satisfaction)
    try:
        cur.execute("""
            SELECT rating, count(*) as cnt
            FROM feedback WHERE rating IS NOT NULL
            GROUP BY rating ORDER BY rating
        """)
        rating_dist = [
            {"rating": r[0], "count": r[1]} for r in cur.fetchall()
        ]
    except Exception:
        rating_dist = []

    avg_rating = 0.0
    try:
        cur.execute(
            "SELECT avg(rating) FROM feedback WHERE rating IS NOT NULL")
        v = cur.fetchone()[0]
        if v is not None:
            avg_rating = round(v, 2)
    except Exception:
        pass

    # ── 4. Confidence drift (clinical_decisions) ────────────────────
    decisions_total = _safe_count(
        cur, "SELECT count(*) FROM clinical_decisions")
    try:
        cur.execute(
            "SELECT avg(ai_confidence) FROM clinical_decisions "
            "WHERE ai_confidence IS NOT NULL")
        avg_confidence = round(cur.fetchone()[0] or 0, 3)
    except Exception:
        avg_confidence = 0.0

    low_confidence = _safe_count(
        cur,
        "SELECT count(*) FROM clinical_decisions "
        "WHERE ai_confidence IS NOT NULL AND ai_confidence < 0.5")
    high_confidence = _safe_count(
        cur,
        "SELECT count(*) FROM clinical_decisions "
        "WHERE ai_confidence IS NOT NULL AND ai_confidence >= 0.8")

    # Agreement drift
    agreed = _safe_count(
        cur,
        "SELECT count(*) FROM clinical_decisions "
        "WHERE neurologist_agreement = 'agree'")
    disagreed = _safe_count(
        cur,
        "SELECT count(*) FROM clinical_decisions "
        "WHERE neurologist_agreement = 'disagree'")
    agreement_rate = round(
        agreed / max(agreed + disagreed, 1) * 100, 1)

    # ── 5. Component-finding alignment ──────────────────────────────
    findings_total = _safe_count(
        cur, "SELECT count(*) FROM component_findings")
    findings_agree = _safe_count(
        cur,
        "SELECT count(*) FROM component_findings "
        "WHERE agree_with_ai = 'yes'")
    findings_disagree = _safe_count(
        cur,
        "SELECT count(*) FROM component_findings "
        "WHERE agree_with_ai = 'no'")
    component_alignment = round(
        findings_agree / max(findings_agree + findings_disagree, 1) * 100, 1)

    # ── 6. HITL override rate (human-AI divergence) ─────────────────
    hitl_total = _safe_count(cur, "SELECT count(*) FROM hitl_reviews")
    try:
        cur.execute("""
            SELECT count(*) FROM hitl_reviews
            WHERE decision IS NOT NULL AND decision != ''
        """)
        hitl_overrides = cur.fetchone()[0]
    except Exception:
        hitl_overrides = 0
    hitl_override_rate = round(
        hitl_overrides / max(hitl_total, 1) * 100, 1)

    # ── 7. Daily activity trend (last 14 days) ──────────────────────
    try:
        cur.execute("""
            SELECT date(ts_utc) as d, count(*) as cnt
            FROM transaction_log
            GROUP BY d ORDER BY d DESC LIMIT 14
        """)
        action_daily = {r[0]: r[1] for r in cur.fetchall()}
    except Exception:
        action_daily = {}

    try:
        cur.execute("""
            SELECT date(created_at) as d, count(*) as cnt
            FROM feedback
            GROUP BY d ORDER BY d DESC LIMIT 14
        """)
        feedback_daily = {r[0]: r[1] for r in cur.fetchall()}
    except Exception:
        feedback_daily = {}

    all_dates = sorted(
        set(list(action_daily.keys()) + list(feedback_daily.keys())))
    daily_trend = [
        {
            "date": d,
            "actions": action_daily.get(d, 0),
            "feedback": feedback_daily.get(d, 0),
            "combined": action_daily.get(d, 0) + feedback_daily.get(d, 0)
        }
        for d in all_dates[-14:]
    ]

    # ── 8. Goal-drift score (composite) ─────────────────────────────
    # Higher = more drift. Scale 0-100.
    drift_signals = []
    if feedback_total > 0:
        drift_signals.append(correction_rate)  # correction %
    if decisions_total > 0:
        drift_signals.append(100 - agreement_rate)  # disagreement %
    if findings_total > 0:
        drift_signals.append(100 - component_alignment)
    if hitl_total > 0:
        drift_signals.append(hitl_override_rate)
    if total_actions > 0:
        drift_signals.append(block_rate)

    goal_drift_score = round(
        sum(drift_signals) / max(len(drift_signals), 1), 1)

    # Drift severity
    if goal_drift_score >= 40:
        drift_severity = "high"
    elif goal_drift_score >= 20:
        drift_severity = "medium"
    else:
        drift_severity = "low"

    conn.close()

    return {
        "available": True,
        "summary": {
            "total_agent_actions": total_actions,
            "active_components": len(actions_by_component),
            "blocked_actions": blocked,
            "block_rate_pct": block_rate,
            "conversation_turns": total_turns,
            "assistant_turns": assistant_turns,
            "operator_turns": operator_turns,
            "loop_ratio": loop_ratio,
            "feedback_total": feedback_total,
            "corrections": corrections,
            "correction_rate_pct": correction_rate,
            "avg_feedback_rating": avg_rating,
            "avg_ai_confidence": avg_confidence,
            "low_confidence_decisions": low_confidence,
            "high_confidence_decisions": high_confidence,
            "agreement_rate_pct": agreement_rate,
            "component_alignment_pct": component_alignment,
            "hitl_override_rate_pct": hitl_override_rate,
            "goal_drift_score": goal_drift_score,
            "drift_severity": drift_severity,
        },
        "actions_by_component": actions_by_component,
        "actions_by_type": actions_by_type,
        "rating_distribution": rating_dist,
        "daily_trend": daily_trend,
    }


def loop_breakdown():
    """Detailed per-component loop analysis: action counts, blocked ratio,
    top actors, recent activity."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # Per-component detail
    components = []
    try:
        cur.execute("""
            SELECT component, action, count(*) as cnt
            FROM transaction_log
            GROUP BY component, action
            ORDER BY component, cnt DESC
        """)
        comp_map = {}
        for comp, action, cnt in cur.fetchall():
            if comp not in comp_map:
                comp_map[comp] = {"component": comp, "total": 0,
                                  "actions": [], "blocked": 0}
            comp_map[comp]["total"] += cnt
            comp_map[comp]["actions"].append(
                {"action": action, "count": cnt})
            if action == "blocked":
                comp_map[comp]["blocked"] = cnt
        components = sorted(
            comp_map.values(), key=lambda x: x["total"], reverse=True)
    except Exception:
        pass

    # Feedback detail (recent corrections)
    corrections = []
    try:
        cur.execute("""
            SELECT patient_id, role, ai_output, correction, rating,
                   reason, created_at
            FROM feedback
            WHERE correction IS NOT NULL AND correction != ''
            ORDER BY created_at DESC LIMIT 20
        """)
        for r in cur.fetchall():
            corrections.append({
                "patient_id": r[0], "role": r[1],
                "ai_output": (r[2] or "")[:120],
                "correction": (r[3] or "")[:120],
                "rating": r[4], "reason": r[5],
                "date": r[6]
            })
    except Exception:
        pass

    # Decision disagreements
    disagreements = []
    try:
        cur.execute("""
            SELECT patient_id, ai_prediction, ai_confidence,
                   neurologist_agreement, final_decision, reviewer,
                   created_at
            FROM clinical_decisions
            WHERE neurologist_agreement = 'disagree'
            ORDER BY created_at DESC LIMIT 20
        """)
        for r in cur.fetchall():
            disagreements.append({
                "patient_id": r[0], "ai_prediction": r[1],
                "ai_confidence": r[2],
                "agreement": r[3], "final_decision": r[4],
                "reviewer": r[5], "date": r[6]
            })
    except Exception:
        pass

    conn.close()

    return {
        "available": True,
        "components": components,
        "recent_corrections": corrections,
        "decision_disagreements": disagreements,
    }


def loop_definitions():
    """Metric definitions for the Agent Loop / Goal-Drift Dashboard."""
    return {
        "available": True,
        "definitions": [
            {
                "metric": "Total Agent Actions",
                "description": "Count of all actions logged in the "
                "transaction_log — each row represents one agent step "
                "(ingest, query, sign-off, blocked, etc.).",
                "source": "transaction_log"
            },
            {
                "metric": "Block Rate",
                "description": "Percentage of agent actions that were "
                "blocked by guardrails. Higher block rates indicate "
                "the agent attempted prohibited operations.",
                "source": "transaction_log WHERE action='blocked'"
            },
            {
                "metric": "Loop Ratio",
                "description": "Ratio of assistant turns to operator "
                "turns in the conversation log. A ratio > 1 means the "
                "agent generated more turns than the operator directed.",
                "source": "conversation_log (assistant / operator turns)"
            },
            {
                "metric": "Correction Rate",
                "description": "Percentage of feedback entries that "
                "include a human correction — indicates how often the "
                "agent's output required manual fix.",
                "source": "feedback WHERE correction IS NOT NULL"
            },
            {
                "metric": "Average Feedback Rating",
                "description": "Mean rating across all feedback entries "
                "(1-5 scale). Declining ratings signal goal drift.",
                "source": "feedback.rating"
            },
            {
                "metric": "Agreement Rate",
                "description": "Percentage of clinical decisions where "
                "the neurologist agreed with the AI prediction. Lower "
                "rates indicate human-AI alignment drift.",
                "source": "clinical_decisions.neurologist_agreement"
            },
            {
                "metric": "Component Alignment",
                "description": "Percentage of component findings where "
                "the reviewing doctor agreed with the AI. Measures "
                "per-component goal fidelity.",
                "source": "component_findings.agree_with_ai"
            },
            {
                "metric": "HITL Override Rate",
                "description": "Percentage of human-in-the-loop reviews "
                "that resulted in an override decision, indicating the "
                "human judged the AI output unsuitable.",
                "source": "hitl_reviews WHERE decision IS NOT NULL"
            },
            {
                "metric": "Goal Drift Score",
                "description": "Composite 0-100 score averaging five "
                "drift signals: correction rate, disagreement rate, "
                "component misalignment, HITL override rate, block rate. "
                "Higher = more drift. <20 low, 20-40 medium, >40 high.",
                "source": "Composite (feedback, decisions, findings, "
                "hitl, transaction_log)"
            },
            {
                "metric": "Daily Activity Trend",
                "description": "Per-day counts of agent actions and "
                "feedback entries over the last 14 days, showing loop "
                "activity volume and drift-correction frequency.",
                "source": "transaction_log + feedback (date-grouped)"
            }
        ]
    }
