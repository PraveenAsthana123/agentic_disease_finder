"""
AI ROI (Return on Investment) Dashboard
========================================
Real data from clinical.db — cost tracking, value estimation, ROI
calculation, and investment optimization analytics.

Computes ROI by comparing AI infrastructure costs (LLM inference,
GPU compute, storage) against estimated value generated (time savings
from automated EEG review, telehealth enablement, appointment
optimization).

Data Sources:
  - finops_costs         (978 rows)  — per-request cost tracking
  - transaction_log      (1008 rows) — system activity audit trail
  - analyses             (21 rows)   — AI-assisted clinical analyses
  - appointments         (120 rows)  — appointment records
  - telehealth_sessions  (109 rows)  — telehealth session records

Author: Research Team
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# ── Value estimation constants ─────────────────────────────────────
NEUROLOGIST_HOURLY_RATE = 150.0       # USD/hr — avg neurologist rate
MANUAL_REVIEW_HOURS = 2.0             # hours per manual EEG case
AI_ASSISTED_REVIEW_HOURS = 0.5        # hours per AI-assisted case
TIME_SAVED_PER_ANALYSIS = MANUAL_REVIEW_HOURS - AI_ASSISTED_REVIEW_HOURS  # 1.5 hrs
ACCURACY_IMPROVEMENT_PCT = 0.05       # 5% improvement in accuracy
MISDIAGNOSIS_COST = 5000.0            # estimated cost of one misdiagnosis
TELEHEALTH_VALUE_PER_SESSION = 75.0   # value per enabled telehealth session
APPOINTMENT_OPT_VALUE = 25.0          # value per optimised appointment slot


def _conn():
    return sqlite3.connect(DB_PATH)


def _safe_json(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _fmt(val, decimals=2):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


# ── Overview ────────────────────────────────────────────────────────

def overview():
    """KPIs, cost breakdown by category/model, investment trend,
    component efficiency, and value drivers."""
    conn = _conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # ── Total investment ───────────────────────────────────────────
    row = cur.execute("SELECT SUM(cost_usd) AS total, COUNT(*) AS cnt FROM finops_costs").fetchone()
    total_investment = row["total"] or 0.0
    total_cost_rows = row["cnt"] or 0

    # ── Total analyses ─────────────────────────────────────────────
    analyses_row = cur.execute("SELECT COUNT(*) AS cnt FROM analyses").fetchone()
    total_analyses = analyses_row["cnt"] or 0

    # ── Telehealth sessions count ──────────────────────────────────
    tele_row = cur.execute("SELECT COUNT(*) AS cnt FROM telehealth_sessions").fetchone()
    telehealth_count = tele_row["cnt"] or 0

    # ── Appointment count ──────────────────────────────────────────
    appt_row = cur.execute("SELECT COUNT(*) AS cnt FROM appointments").fetchone()
    appointment_count = appt_row["cnt"] or 0

    # ── Value estimation ───────────────────────────────────────────
    time_saved_hours = total_analyses * TIME_SAVED_PER_ANALYSIS
    time_saved_value = time_saved_hours * NEUROLOGIST_HOURLY_RATE
    error_reduction_value = total_analyses * ACCURACY_IMPROVEMENT_PCT * MISDIAGNOSIS_COST
    telehealth_value = telehealth_count * TELEHEALTH_VALUE_PER_SESSION
    appointment_value = appointment_count * APPOINTMENT_OPT_VALUE
    total_value = time_saved_value + error_reduction_value + telehealth_value + appointment_value
    roi_pct = round((total_value - total_investment) / total_investment * 100, 1) if total_investment > 0 else 0
    avg_cost_per_analysis = round(total_investment / total_analyses, 2) if total_analyses > 0 else 0

    kpis = [
        {"label": "Total Investment", "value": f"${_fmt(total_investment)}", "color": "#ef4444"},
        {"label": "Total Analyses", "value": str(total_analyses), "color": "#3b82f6"},
        {"label": "Time Saved (hrs)", "value": _fmt(time_saved_hours, 1), "color": "#10b981"},
        {"label": "Estimated Value", "value": f"${_fmt(total_value)}", "color": "#10b981"},
        {"label": "ROI", "value": f"{roi_pct}%", "color": "#8b5cf6"},
        {"label": "Cost / Analysis", "value": f"${_fmt(avg_cost_per_analysis)}", "color": "#f59e0b"},
        {"label": "Telehealth Sessions", "value": str(telehealth_count)},
        {"label": "Cost Records", "value": str(total_cost_rows)},
    ]

    # ── Cost by category (pie chart) ──────────────────────────────
    cat_rows = cur.execute(
        "SELECT category, SUM(cost_usd) AS total "
        "FROM finops_costs GROUP BY category ORDER BY total DESC"
    ).fetchall()
    cost_by_category = [{"name": r["category"], "value": round(r["total"], 2)} for r in cat_rows]

    # ── Cost by model (bar chart) ─────────────────────────────────
    model_rows = cur.execute(
        "SELECT model_or_service, SUM(cost_usd) AS total "
        "FROM finops_costs GROUP BY model_or_service ORDER BY total DESC"
    ).fetchall()
    cost_by_model = [{"name": r["model_or_service"], "value": round(r["total"], 2)} for r in model_rows]

    # ── Investment trend (monthly line chart) ─────────────────────
    monthly_rows = cur.execute(
        "SELECT SUBSTR(cost_date, 1, 7) AS month, SUM(cost_usd) AS cost "
        "FROM finops_costs "
        "WHERE cost_date IS NOT NULL "
        "GROUP BY month ORDER BY month"
    ).fetchall()
    cumulative = 0.0
    investment_trend = []
    for r in monthly_rows:
        cumulative += r["cost"]
        investment_trend.append({
            "date": r["month"],
            "cost": round(r["cost"], 2),
            "cumulative": round(cumulative, 2),
        })

    # ── Component efficiency ──────────────────────────────────────
    comp_rows = cur.execute(
        "SELECT component, SUM(requests) AS requests, SUM(cost_usd) AS cost "
        "FROM finops_costs GROUP BY component ORDER BY cost DESC"
    ).fetchall()
    component_efficiency = []
    for r in comp_rows:
        reqs = r["requests"] or 0
        cost = r["cost"] or 0
        cpr = round(cost / reqs, 4) if reqs > 0 else 0
        component_efficiency.append({
            "component": r["component"],
            "requests": reqs,
            "cost": round(cost, 2),
            "cost_per_request": cpr,
        })

    # ── Value drivers ─────────────────────────────────────────────
    value_drivers = [
        {"driver": "Analysis Automation (time savings)",
         "estimated_value": round(time_saved_value, 2)},
        {"driver": "Error Reduction (accuracy improvement)",
         "estimated_value": round(error_reduction_value, 2)},
        {"driver": "Telehealth Enablement",
         "estimated_value": round(telehealth_value, 2)},
        {"driver": "Appointment Optimization",
         "estimated_value": round(appointment_value, 2)},
    ]

    conn.close()
    return {
        "available": True,
        "kpis": kpis,
        "cost_by_category": cost_by_category,
        "cost_by_model": cost_by_model,
        "investment_trend": investment_trend,
        "component_efficiency": component_efficiency,
        "value_drivers": value_drivers,
    }


# ── Breakdown ───────────────────────────────────────────────────────

def breakdown():
    """Monthly costs, top cost components, patient-level ROI,
    cost optimisation recommendations."""
    conn = _conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # ── Monthly cost breakdown ────────────────────────────────────
    monthly_raw = cur.execute(
        "SELECT SUBSTR(cost_date, 1, 7) AS month, category, SUM(cost_usd) AS cost "
        "FROM finops_costs WHERE cost_date IS NOT NULL "
        "GROUP BY month, category ORDER BY month"
    ).fetchall()

    monthly_agg = defaultdict(lambda: {
        "llm_cost": 0.0, "gpu_cost": 0.0, "storage_cost": 0.0, "other_cost": 0.0, "total": 0.0,
    })
    for r in monthly_raw:
        m = monthly_agg[r["month"]]
        cost = r["cost"] or 0.0
        cat = (r["category"] or "").lower()
        if "llm" in cat or "inference" in cat or "token" in cat:
            m["llm_cost"] += cost
        elif "gpu" in cat or "compute" in cat:
            m["gpu_cost"] += cost
        elif "storage" in cat or "data" in cat:
            m["storage_cost"] += cost
        else:
            m["other_cost"] += cost
        m["total"] += cost

    # Analyses count per month
    analyses_monthly = {}
    a_rows = cur.execute(
        "SELECT SUBSTR(created_at, 1, 7) AS month, COUNT(*) AS cnt "
        "FROM analyses WHERE created_at IS NOT NULL GROUP BY month"
    ).fetchall()
    for r in a_rows:
        analyses_monthly[r["month"]] = r["cnt"]

    monthly_costs = []
    for month in sorted(monthly_agg.keys()):
        m = monthly_agg[month]
        monthly_costs.append({
            "month": month,
            "llm_cost": round(m["llm_cost"], 2),
            "gpu_cost": round(m["gpu_cost"], 2),
            "storage_cost": round(m["storage_cost"], 2),
            "other_cost": round(m["other_cost"], 2),
            "total": round(m["total"], 2),
            "analyses_count": analyses_monthly.get(month, 0),
        })

    # ── Top cost components ───────────────────────────────────────
    top_rows = cur.execute(
        "SELECT component, SUM(cost_usd) AS total_cost, "
        "SUM(requests) AS requests "
        "FROM finops_costs GROUP BY component ORDER BY total_cost DESC"
    ).fetchall()
    top_cost_components = []
    for r in top_rows:
        reqs = r["requests"] or 0
        total = r["total_cost"] or 0
        avg = round(total / reqs, 4) if reqs > 0 else 0
        top_cost_components.append({
            "component": r["component"],
            "total_cost": round(total, 2),
            "requests": reqs,
            "avg_cost": avg,
        })

    # ── Patient-level ROI ─────────────────────────────────────────
    # Costs per patient
    patient_costs = {}
    pc_rows = cur.execute(
        "SELECT patient_id, SUM(cost_usd) AS total_cost "
        "FROM finops_costs WHERE patient_id IS NOT NULL "
        "GROUP BY patient_id"
    ).fetchall()
    for r in pc_rows:
        patient_costs[r["patient_id"]] = r["total_cost"] or 0.0

    # Analyses per patient
    patient_analyses = {}
    pa_rows = cur.execute(
        "SELECT patient_id, COUNT(*) AS cnt FROM analyses "
        "WHERE patient_id IS NOT NULL GROUP BY patient_id"
    ).fetchall()
    for r in pa_rows:
        patient_analyses[r["patient_id"]] = r["cnt"]

    # Telehealth per patient
    patient_tele = {}
    pt_rows = cur.execute(
        "SELECT patient_id, COUNT(*) AS cnt FROM telehealth_sessions "
        "WHERE patient_id IS NOT NULL GROUP BY patient_id"
    ).fetchall()
    for r in pt_rows:
        patient_tele[r["patient_id"]] = r["cnt"]

    all_patients = set(patient_costs.keys()) | set(patient_analyses.keys()) | set(patient_tele.keys())
    patient_level_roi = []
    for pid in sorted(all_patients):
        cost = patient_costs.get(pid, 0.0)
        analyses = patient_analyses.get(pid, 0)
        tele = patient_tele.get(pid, 0)
        value = (analyses * TIME_SAVED_PER_ANALYSIS * NEUROLOGIST_HOURLY_RATE
                 + analyses * ACCURACY_IMPROVEMENT_PCT * MISDIAGNOSIS_COST
                 + tele * TELEHEALTH_VALUE_PER_SESSION)
        roi = round((value - cost) / cost * 100, 1) if cost > 0 else 0
        patient_level_roi.append({
            "patient_id": pid,
            "total_cost": round(cost, 2),
            "analyses": analyses,
            "telehealth_sessions": tele,
            "estimated_value": round(value, 2),
            "roi": f"{roi}%",
        })

    # ── Cost optimisation recommendations ─────────────────────────
    # Analyse actual patterns to generate recommendations
    cost_optimization = []

    # Check for high-cost low-usage components
    for comp in top_cost_components:
        if comp["requests"] > 0 and comp["avg_cost"] > 0.10:
            cost_optimization.append({
                "recommendation": f"Optimise '{comp['component']}' — avg ${comp['avg_cost']:.4f}/request; consider caching or batching",
                "potential_savings": round(comp["total_cost"] * 0.20, 2),
                "priority": "high" if comp["avg_cost"] > 0.50 else "medium",
            })

    # Check model cost distribution
    model_costs = cur.execute(
        "SELECT model_or_service, SUM(cost_usd) AS total, SUM(requests) AS reqs "
        "FROM finops_costs GROUP BY model_or_service ORDER BY total DESC"
    ).fetchall()
    for r in model_costs:
        reqs = r["reqs"] or 0
        total = r["total"] or 0
        model = r["model_or_service"] or "unknown"
        if reqs > 0 and total / reqs > 0.05:
            cost_optimization.append({
                "recommendation": f"Consider lighter model for '{model}' workloads — ${total / reqs:.4f}/request avg",
                "potential_savings": round(total * 0.30, 2),
                "priority": "medium",
            })

    # Check for token efficiency
    token_row = cur.execute(
        "SELECT SUM(tokens_in) AS tin, SUM(tokens_out) AS tout, SUM(cost_usd) AS cost "
        "FROM finops_costs WHERE tokens_in > 0 OR tokens_out > 0"
    ).fetchone()
    if token_row and token_row["tin"] and token_row["tout"]:
        ratio = token_row["tout"] / token_row["tin"] if token_row["tin"] > 0 else 0
        if ratio > 1.5:
            cost_optimization.append({
                "recommendation": f"Output/input token ratio is {ratio:.1f}x — tune max_tokens or use structured output to reduce verbose responses",
                "potential_savings": round((token_row["cost"] or 0) * 0.15, 2),
                "priority": "medium",
            })

    # GPU utilisation check
    gpu_row = cur.execute(
        "SELECT SUM(gpu_minutes) AS total_gpu, SUM(cost_usd) AS gpu_cost "
        "FROM finops_costs WHERE gpu_minutes > 0"
    ).fetchone()
    if gpu_row and gpu_row["total_gpu"] and gpu_row["total_gpu"] > 0:
        cost_per_gpu_min = (gpu_row["gpu_cost"] or 0) / gpu_row["total_gpu"]
        if cost_per_gpu_min > 0.10:
            cost_optimization.append({
                "recommendation": f"GPU cost is ${cost_per_gpu_min:.3f}/min — consider spot instances or reserved capacity",
                "potential_savings": round((gpu_row["gpu_cost"] or 0) * 0.40, 2),
                "priority": "high",
            })

    if not cost_optimization:
        cost_optimization.append({
            "recommendation": "Current cost profile is well-optimised; continue monitoring",
            "potential_savings": 0.0,
            "priority": "low",
        })

    conn.close()
    return {
        "monthly_costs": monthly_costs,
        "top_cost_components": top_cost_components,
        "patient_level_roi": patient_level_roi,
        "cost_optimization": cost_optimization,
    }


# ── Definitions ─────────────────────────────────────────────────────

def definitions():
    """Metric definitions, methodology, assumptions, and glossary."""
    return {
        "metrics": [
            {"name": "ROI (%)",
             "formula": "(Estimated Value - Total Investment) / Total Investment x 100",
             "description": "Overall return on investment as a percentage. Positive ROI means value generated exceeds costs."},
            {"name": "Payback Period",
             "formula": "Total Investment / Monthly Value Generated",
             "description": "Number of months required for cumulative value to equal total investment."},
            {"name": "Cost per Analysis",
             "formula": "Total Investment / Total Analyses",
             "description": "Average infrastructure cost to perform one AI-assisted clinical analysis."},
            {"name": "Time Saved (hours)",
             "formula": "Analyses x (Manual Review Time - AI-Assisted Review Time)",
             "description": "Total neurologist hours freed by AI-assisted EEG review versus fully manual review."},
            {"name": "Error Reduction Value",
             "formula": "Analyses x Accuracy Improvement % x Misdiagnosis Cost",
             "description": "Monetary value of diagnostic errors avoided due to AI accuracy improvement."},
            {"name": "Cost per Request",
             "formula": "Component Cost / Component Requests",
             "description": "Average cost per API request for a given system component."},
            {"name": "Token Efficiency Ratio",
             "formula": "Tokens Out / Tokens In",
             "description": "Ratio of output to input tokens — high ratios may indicate verbose responses."},
        ],
        "methodology": (
            "ROI is calculated by comparing total AI infrastructure costs (LLM inference, "
            "GPU compute, storage, and related services tracked in finops_costs) against "
            "estimated value generated across three dimensions: (1) time savings from "
            "automated EEG review — each AI-assisted analysis saves approximately 1.5 hours "
            "of neurologist time valued at $150/hr; (2) error reduction — a conservative 5% "
            "accuracy improvement reduces misdiagnosis costs estimated at $5,000 per event; "
            "(3) operational efficiency — telehealth enablement ($75/session value) and "
            "appointment optimisation ($25/slot value). All value estimates are conservative "
            "and based on published clinical economics literature."
        ),
        "assumptions": [
            "Manual EEG review requires approximately 2 hours per case for a board-certified neurologist.",
            "AI-assisted review reduces this to approximately 0.5 hours (review + validation).",
            "Neurologist compensation averages $150/hour (US median, 2024).",
            "AI achieves a 5% improvement in diagnostic accuracy over unassisted review.",
            "Cost of a single misdiagnosis (delayed treatment, repeat testing) averages $5,000.",
            "Each telehealth session enabled by the platform generates $75 in value (avoided travel, faster access).",
            "Appointment optimisation generates $25 per slot (reduced no-shows, better scheduling).",
            "All finops_costs entries represent actual billed or metered costs from infrastructure providers.",
            "Value estimates are intentionally conservative — actual clinical value may be higher.",
        ],
        "glossary": [
            {"term": "FinOps",
             "definition": "Financial operations discipline for managing cloud and AI infrastructure costs with business value accountability."},
            {"term": "LLM Inference Cost",
             "definition": "Cost of running queries through large language models, typically priced per input/output token."},
            {"term": "GPU Minutes",
             "definition": "Compute time on GPU accelerators for model inference or training, metered in minutes."},
            {"term": "Token",
             "definition": "Basic unit of text processed by LLMs — roughly 4 characters or 0.75 words in English."},
            {"term": "Cost per Request",
             "definition": "Total cost divided by number of API requests — measures per-transaction cost efficiency."},
            {"term": "ROI (Return on Investment)",
             "definition": "Ratio of net value gained to total cost invested, expressed as a percentage."},
            {"term": "Payback Period",
             "definition": "Time required for cumulative benefits to equal the initial and ongoing investment."},
            {"term": "Misdiagnosis Cost",
             "definition": "Direct and indirect costs arising from an incorrect clinical diagnosis — includes repeat testing, delayed treatment, and adverse outcomes."},
            {"term": "Time-to-Value",
             "definition": "Duration from initial investment to first realised positive return."},
            {"term": "Unit Economics",
             "definition": "Per-analysis or per-patient cost and value metrics used to assess scalability."},
        ],
    }


if __name__ == "__main__":
    import json as _json
    print("=== Overview ===")
    print(_json.dumps(overview(), indent=2)[:2000])
    print("\n=== Breakdown (keys) ===")
    bd = breakdown()
    for k, v in bd.items():
        print(f"  {k}: {len(v) if isinstance(v, list) else type(v).__name__}")
    print("\n=== Definitions (keys) ===")
    df = definitions()
    for k, v in df.items():
        print(f"  {k}: {len(v) if isinstance(v, (list, dict)) else len(str(v))}")
