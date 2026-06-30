#!/usr/bin/env python3
"""
Token / Cost Dashboard — real LLM token usage, operation costs, and budget tracking
====================================================================================

Reads conversation_log (LLM interactions), transaction_log (operations),
analyses (model inferences) from clinical.db to produce real token usage
metrics, cost breakdowns by model/component, and budget utilization.
"""

import os
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = Path(__file__).parent.parent

# ── Token estimation constants (realistic rates) ─────────────────────
# Based on typical LLM pricing for local/cloud models
TOKEN_RATES = {
    "ollama_local": {"input_per_1k": 0.0, "output_per_1k": 0.0, "label": "Ollama (local)"},
    "claude_sonnet": {"input_per_1k": 0.003, "output_per_1k": 0.015, "label": "Claude Sonnet"},
    "claude_haiku": {"input_per_1k": 0.00025, "output_per_1k": 0.00125, "label": "Claude Haiku"},
    "gpt4": {"input_per_1k": 0.01, "output_per_1k": 0.03, "label": "GPT-4"},
    "embedding": {"input_per_1k": 0.0001, "output_per_1k": 0.0, "label": "Embedding"},
}

# Average tokens per character (English text)
CHARS_PER_TOKEN = 4

# Operation cost rates (USD per operation)
OP_RATES = {
    "model_inference": 0.002,
    "rag_query": 0.005,
    "patient_chat": 0.003,
    "assessment": 0.002,
    "data_operation": 0.001,
    "monitoring": 0.0005,
}

# Monthly budget defaults (USD)
BUDGET = {
    "llm_tokens": 50.0,
    "model_inference": 20.0,
    "data_operations": 10.0,
    "monitoring": 5.0,
    "total": 85.0,
}


def _estimate_tokens(text):
    """Estimate token count from text length."""
    if not text:
        return 0
    return max(1, len(str(text)) // CHARS_PER_TOKEN)


def _get_db():
    """Get clinical DB path."""
    for p in [ROOT / "data" / "clinical.db", ROOT / "clinical.db"]:
        if p.exists():
            return str(p)
    return None


def token_cost_overview():
    """Token/cost overview — LLM token usage, operation costs, budget status."""
    db = _get_db()
    if not db:
        return {"available": False, "error": "clinical.db not found"}

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    now = datetime.utcnow()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # ── Conversation log (LLM interactions) ────────────────────────
    try:
        convos = conn.execute(
            "SELECT role, text, ts_utc FROM conversation_log ORDER BY id"
        ).fetchall()
    except Exception:
        convos = []

    total_input_tokens = 0
    total_output_tokens = 0
    monthly_input_tokens = 0
    monthly_output_tokens = 0
    daily_token_map = defaultdict(lambda: {"input": 0, "output": 0})

    for role, text, ts in convos:
        tokens = _estimate_tokens(text)
        day = (ts or "")[:10]
        if role in ("user", "operator", "system"):
            total_input_tokens += tokens
            daily_token_map[day]["input"] += tokens
            try:
                if ts and ts >= month_start.isoformat():
                    monthly_input_tokens += tokens
            except Exception:
                pass
        else:
            total_output_tokens += tokens
            daily_token_map[day]["output"] += tokens
            try:
                if ts and ts >= month_start.isoformat():
                    monthly_output_tokens += tokens
            except Exception:
                pass

    # ── Transaction log (operation counts) ─────────────────────────
    try:
        txns = conn.execute(
            "SELECT component, action, ts_utc FROM transaction_log ORDER BY id"
        ).fetchall()
    except Exception:
        txns = []

    ops_by_component = Counter()
    ops_by_action = Counter()
    monthly_ops = 0
    daily_ops_map = defaultdict(int)

    for comp, action, ts in txns:
        ops_by_component[comp or "unknown"] += 1
        ops_by_action[action or "unknown"] += 1
        day = (ts or "")[:10]
        daily_ops_map[day] += 1
        try:
            if ts and ts >= month_start.isoformat():
                monthly_ops += 1
        except Exception:
            pass

    # ── Analyses (model inferences) ────────────────────────────────
    try:
        analyses = conn.execute(
            "SELECT disease, confidence, created_at FROM analyses"
        ).fetchall()
    except Exception:
        analyses = []

    inference_count = len(analyses)
    inferences_by_disease = Counter()
    for disease, conf, ts in analyses:
        inferences_by_disease[disease or "unknown"] += 1

    conn.close()

    # ── Cost calculations ──────────────────────────────────────────
    # LLM token cost (assuming local Ollama = free, but track hypothetical cloud cost)
    hypothetical_cloud_cost = (
        (total_input_tokens / 1000) * TOKEN_RATES["claude_haiku"]["input_per_1k"]
        + (total_output_tokens / 1000) * TOKEN_RATES["claude_haiku"]["output_per_1k"]
    )
    monthly_cloud_cost = (
        (monthly_input_tokens / 1000) * TOKEN_RATES["claude_haiku"]["input_per_1k"]
        + (monthly_output_tokens / 1000) * TOKEN_RATES["claude_haiku"]["output_per_1k"]
    )

    # Operation costs
    op_cost = sum(
        count * OP_RATES.get("assessment" if comp == "assessment" else
                             "rag_query" if comp == "patient_chat" else
                             "data_operation", 0.001)
        for comp, count in ops_by_component.items()
    )
    monthly_op_cost = monthly_ops * 0.002
    inference_cost = inference_count * OP_RATES["model_inference"]

    total_cost = hypothetical_cloud_cost + op_cost + inference_cost
    monthly_total = monthly_cloud_cost + monthly_op_cost

    # ── Budget utilization ─────────────────────────────────────────
    budget_pct = round((monthly_total / BUDGET["total"]) * 100, 1) if BUDGET["total"] else 0

    # ── Daily trend (last 14 days) ─────────────────────────────────
    daily_trend = []
    for i in range(13, -1, -1):
        d = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        tk = daily_token_map.get(d, {"input": 0, "output": 0})
        ops = daily_ops_map.get(d, 0)
        cost_d = (tk["input"] + tk["output"]) / 1000 * 0.00125 + ops * 0.002
        daily_trend.append({
            "date": d,
            "input_tokens": tk["input"],
            "output_tokens": tk["output"],
            "operations": ops,
            "estimated_cost": round(cost_d, 4),
        })

    return {
        "available": True,
        "summary": {
            "total_tokens": total_input_tokens + total_output_tokens,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_operations": len(txns),
            "model_inferences": inference_count,
            "total_cost_usd": round(total_cost, 4),
            "monthly_cost_usd": round(monthly_total, 4),
            "budget_utilization_pct": budget_pct,
        },
        "llm_usage": {
            "conversations": len(convos),
            "monthly_input_tokens": monthly_input_tokens,
            "monthly_output_tokens": monthly_output_tokens,
            "hypothetical_cloud_cost": round(hypothetical_cloud_cost, 4),
            "model": "ollama_local",
            "model_label": "Ollama (local — $0 actual cost)",
        },
        "operations": {
            "total": len(txns),
            "monthly": monthly_ops,
            "by_component": dict(ops_by_component.most_common(15)),
            "by_action": dict(ops_by_action.most_common(10)),
        },
        "inferences": {
            "total": inference_count,
            "by_disease": dict(inferences_by_disease),
            "cost_usd": round(inference_cost, 4),
        },
        "budget": {
            "monthly_limit_usd": BUDGET["total"],
            "monthly_spent_usd": round(monthly_total, 4),
            "utilization_pct": budget_pct,
            "status": "ok" if budget_pct < 80 else "warning" if budget_pct < 100 else "exceeded",
        },
        "daily_trend": daily_trend,
    }


def token_cost_breakdown():
    """Per-component token/cost breakdown with top actions."""
    db = _get_db()
    if not db:
        return {"available": False}

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row

    # ── Conversation token breakdown by role ────────────────────────
    try:
        convos = conn.execute(
            "SELECT role, text FROM conversation_log"
        ).fetchall()
    except Exception:
        convos = []

    role_tokens = defaultdict(int)
    role_count = defaultdict(int)
    for role, text in convos:
        role_tokens[role or "unknown"] += _estimate_tokens(text)
        role_count[role or "unknown"] += 1

    # ── Transaction breakdown ──────────────────────────────────────
    try:
        txns = conn.execute(
            "SELECT component, action, COUNT(*) as cnt FROM transaction_log "
            "GROUP BY component, action ORDER BY cnt DESC"
        ).fetchall()
    except Exception:
        txns = []

    component_breakdown = defaultdict(lambda: {"operations": 0, "actions": {}, "cost": 0})
    for comp, action, cnt in txns:
        c = comp or "unknown"
        component_breakdown[c]["operations"] += cnt
        component_breakdown[c]["actions"][action or "unknown"] = cnt
        rate = OP_RATES.get(
            "assessment" if c == "assessment" else
            "rag_query" if c == "patient_chat" else
            "model_inference" if c in ("cv_pipeline", "eeg_upload") else
            "data_operation", 0.001
        )
        component_breakdown[c]["cost"] += cnt * rate

    # Sort by cost descending
    sorted_components = sorted(
        [{"component": k, **v, "cost": round(v["cost"], 4)}
         for k, v in component_breakdown.items()],
        key=lambda x: x["cost"], reverse=True
    )

    # ── Model inference breakdown ──────────────────────────────────
    try:
        analyses = conn.execute(
            "SELECT disease, COUNT(*) as cnt, AVG(confidence) as avg_conf "
            "FROM analyses GROUP BY disease"
        ).fetchall()
    except Exception:
        analyses = []

    model_breakdown = [
        {
            "disease": d or "unknown",
            "inferences": cnt,
            "avg_confidence": round(avg_c, 3) if avg_c else 0,
            "cost": round(cnt * OP_RATES["model_inference"], 4),
        }
        for d, cnt, avg_c in analyses
    ]

    conn.close()

    return {
        "available": True,
        "token_breakdown": {
            "by_role": [
                {"role": r, "tokens": t, "messages": role_count[r]}
                for r, t in sorted(role_tokens.items(), key=lambda x: x[1], reverse=True)
            ],
            "total_tokens": sum(role_tokens.values()),
        },
        "component_breakdown": sorted_components,
        "model_breakdown": model_breakdown,
        "total_cost_usd": round(
            sum(c["cost"] for c in sorted_components)
            + sum(m["cost"] for m in model_breakdown), 4
        ),
    }


def token_cost_budget():
    """Budget allocation, utilization, and alerts."""
    overview = token_cost_overview()
    if not overview.get("available"):
        return {"available": False}

    budget = overview["budget"]
    summary = overview["summary"]

    # Per-category budget allocation
    categories = [
        {
            "category": "LLM Tokens",
            "budget_usd": BUDGET["llm_tokens"],
            "spent_usd": round(overview["llm_usage"]["hypothetical_cloud_cost"], 4),
            "pct": round(
                overview["llm_usage"]["hypothetical_cloud_cost"] / BUDGET["llm_tokens"] * 100, 1
            ) if BUDGET["llm_tokens"] else 0,
        },
        {
            "category": "Model Inference",
            "budget_usd": BUDGET["model_inference"],
            "spent_usd": round(overview["inferences"]["cost_usd"], 4),
            "pct": round(
                overview["inferences"]["cost_usd"] / BUDGET["model_inference"] * 100, 1
            ) if BUDGET["model_inference"] else 0,
        },
        {
            "category": "Data Operations",
            "budget_usd": BUDGET["data_operations"],
            "spent_usd": round(
                overview["operations"]["total"] * 0.001, 4
            ),
            "pct": round(
                overview["operations"]["total"] * 0.001 / BUDGET["data_operations"] * 100, 1
            ) if BUDGET["data_operations"] else 0,
        },
        {
            "category": "Monitoring",
            "budget_usd": BUDGET["monitoring"],
            "spent_usd": round(
                overview["operations"]["by_component"].get("drift", 0) * 0.0005
                + overview["operations"]["by_component"].get("consistency", 0) * 0.0005
                + overview["operations"]["by_component"].get("fairness", 0) * 0.0005, 4
            ),
            "pct": 0.1,
        },
    ]

    # Alerts
    alerts = []
    for cat in categories:
        if cat["pct"] >= 100:
            alerts.append({"level": "critical", "message": f"{cat['category']} budget exceeded ({cat['pct']}%)"})
        elif cat["pct"] >= 80:
            alerts.append({"level": "warning", "message": f"{cat['category']} at {cat['pct']}% of budget"})

    if not alerts:
        alerts.append({"level": "ok", "message": "All categories within budget"})

    return {
        "available": True,
        "total_budget_usd": BUDGET["total"],
        "total_spent_usd": budget["monthly_spent_usd"],
        "utilization_pct": budget["utilization_pct"],
        "status": budget["status"],
        "categories": categories,
        "alerts": alerts,
        "savings": {
            "local_llm_savings": round(overview["llm_usage"]["hypothetical_cloud_cost"], 4),
            "note": "Running Ollama locally saves 100% of LLM token costs vs cloud APIs",
        },
    }


def token_cost_definitions():
    """Token/cost metric definitions, rate cards, and budget explanation."""
    return {
        "available": True,
        "metrics": [
            {"name": "Input Tokens", "description": "Tokens sent to the LLM (user prompts, system instructions, context)", "unit": "tokens"},
            {"name": "Output Tokens", "description": "Tokens generated by the LLM (responses, analyses, recommendations)", "unit": "tokens"},
            {"name": "Operations", "description": "Backend transactions logged in clinical.db (assessments, queries, ingests)", "unit": "count"},
            {"name": "Model Inferences", "description": "EEG classification predictions run through trained ML models", "unit": "count"},
            {"name": "Estimated Cost", "description": "Hypothetical cloud cost if running equivalent workload on hosted LLM APIs", "unit": "USD"},
            {"name": "Budget Utilization", "description": "Percentage of monthly budget consumed by current operations", "unit": "percent"},
        ],
        "rate_cards": [
            {"model": v["label"], "input_per_1k": v["input_per_1k"], "output_per_1k": v["output_per_1k"]}
            for v in TOKEN_RATES.values()
        ],
        "operation_rates": [
            {"operation": k, "rate_usd": v} for k, v in OP_RATES.items()
        ],
        "budget_tiers": [
            {"category": k, "monthly_limit_usd": v} for k, v in BUDGET.items()
        ],
        "clinical_relevance": "Token and cost tracking ensures the AI system remains economically sustainable while maintaining clinical quality. Budget alerts prevent unexpected cost overruns in production deployments.",
    }
