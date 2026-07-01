"""
AI Control Tower Dashboard
===========================
Centralized oversight of all AI system components — real-time status,
component activity, orchestration health, decision audit trail, and
cost/resource allocation from real clinical.db data.

Reads REAL data from:
  - data/clinical.db  (transaction_log, analyses, finops_costs,
    hitl_reviews, clinical_decisions, feedback, component_findings)
  - jobs/reports/health_latest.json
  - jobs/reports/drift_latest.json
  - jobs/reports/consistency_latest.json
  - jobs/reports/data_quality_latest.json
  - config/ai_type_coverage.json
"""

import json
import sqlite3
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
REPORTS = BASE / "jobs" / "reports"
DB_PATH = BASE / "data" / "clinical.db"
COVERAGE = BASE / "config" / "ai_type_coverage.json"


def _load_json(filename):
    try:
        with open(REPORTS / filename, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _query_db(sql, params=()):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────
# 1. control_tower_overview
# ──────────────────────────────────────────────────────────────────────

def control_tower_overview() -> dict:
    """KPI-level summary: system components, activity, health, decisions."""

    out = {}

    # ── AI Type Coverage (component registry) ────────────────────────
    try:
        with open(COVERAGE, "r") as f:
            cov = json.load(f)
        counts = cov.get("counts", {})
        out["ai_components"] = {
            "total": cov.get("total", 0),
            "built": counts.get("built", 0),
            "scaffold": counts.get("scaffold", 0),
            "planned": counts.get("planned", 0),
            "not_pulled": counts.get("not-pulled", 0),
        }
        # Built component list
        built_list = [r["type"] for r in cov.get("rows", [])
                      if r.get("status") == "built"]
        out["built_components"] = built_list
    except Exception:
        out["ai_components"] = {"total": 0, "built": 0, "scaffold": 0,
                                "planned": 0, "not_pulled": 0}
        out["built_components"] = []

    # ── Transaction volume ───────────────────────────────────────────
    total_tx = _query_db("SELECT COUNT(*) AS cnt FROM transaction_log")
    out["total_transactions"] = total_tx[0]["cnt"] if total_tx else 0

    # Component activity (top 10)
    comp_activity = _query_db("""
        SELECT component, COUNT(*) AS cnt
        FROM transaction_log
        GROUP BY component ORDER BY cnt DESC LIMIT 10
    """)
    out["component_activity"] = comp_activity

    # Actor activity
    actor_activity = _query_db("""
        SELECT actor, COUNT(*) AS cnt
        FROM transaction_log
        GROUP BY actor ORDER BY cnt DESC
    """)
    out["actor_activity"] = actor_activity

    # Action distribution
    action_dist = _query_db("""
        SELECT action, COUNT(*) AS cnt
        FROM transaction_log
        GROUP BY action ORDER BY cnt DESC LIMIT 10
    """)
    out["action_distribution"] = action_dist

    # Daily transaction volume (last 30 days)
    daily_vol = _query_db("""
        SELECT DATE(ts_local) AS day, COUNT(*) AS cnt
        FROM transaction_log
        GROUP BY DATE(ts_local)
        ORDER BY day DESC LIMIT 30
    """)
    out["daily_volume"] = list(reversed(daily_vol))

    # ── Analyses overview ────────────────────────────────────────────
    total_analyses = _query_db("SELECT COUNT(*) AS cnt FROM analyses")
    avg_conf = _query_db("SELECT AVG(confidence) AS v FROM analyses")
    disease_counts = _query_db(
        "SELECT disease, COUNT(*) AS cnt FROM analyses GROUP BY disease"
    )
    out["analyses"] = {
        "total": total_analyses[0]["cnt"] if total_analyses else 0,
        "avg_confidence": round(avg_conf[0]["v"], 4) if avg_conf and avg_conf[0]["v"] else None,
        "per_disease": {r["disease"]: r["cnt"] for r in disease_counts},
    }

    # ── HITL / Clinical decisions ────────────────────────────────────
    hitl_count = _query_db("SELECT COUNT(*) AS cnt FROM hitl_reviews")
    cd_count = _query_db("SELECT COUNT(*) AS cnt FROM clinical_decisions")
    feedback_count = _query_db("SELECT COUNT(*) AS cnt FROM feedback")
    findings_count = _query_db("SELECT COUNT(*) AS cnt FROM component_findings")

    out["oversight"] = {
        "hitl_reviews": hitl_count[0]["cnt"] if hitl_count else 0,
        "clinical_decisions": cd_count[0]["cnt"] if cd_count else 0,
        "feedback_entries": feedback_count[0]["cnt"] if feedback_count else 0,
        "component_findings": findings_count[0]["cnt"] if findings_count else 0,
    }

    # ── Cost summary ─────────────────────────────────────────────────
    cost_total = _query_db("SELECT SUM(cost_usd) AS v FROM finops_costs")
    cost_records = _query_db("SELECT COUNT(*) AS cnt FROM finops_costs")
    cost_by_cat = _query_db("""
        SELECT category, SUM(cost_usd) AS total_cost, COUNT(*) AS records
        FROM finops_costs GROUP BY category ORDER BY total_cost DESC
    """)
    out["cost_summary"] = {
        "total_cost_usd": round(cost_total[0]["v"], 2) if cost_total and cost_total[0]["v"] else 0,
        "total_records": cost_records[0]["cnt"] if cost_records else 0,
        "by_category": cost_by_cat,
    }

    # ── System health (from report) ─────────────────────────────────
    health = _load_json("health_latest.json")
    if health:
        out["system_health"] = {
            "backend_http": health.get("backend_http"),
            "api_errors": health.get("api_errors"),
            "db_status": health.get("db"),
            "frontend_http": health.get("frontend_http"),
            "total_errors": health.get("total_errors"),
            "timestamp": health.get("now"),
        }
    else:
        out["system_health"] = {}

    # ── Drift status ─────────────────────────────────────────────────
    drift = _load_json("drift_latest.json")
    if drift:
        out["drift_status"] = {
            "verdict": drift.get("verdict"),
            "frac_drifted": drift.get("frac_drifted"),
            "n_high_drift": drift.get("n_high_drift"),
        }
    else:
        out["drift_status"] = {}

    # ── Data quality status ──────────────────────────────────────────
    dq = _load_json("data_quality_latest.json")
    if dq:
        out["data_quality"] = {
            "ai_readiness_score": dq.get("ai_readiness_score"),
            "ai_readiness_grade": dq.get("ai_readiness_grade"),
        }
    else:
        out["data_quality"] = {}

    # ── Consistency status ───────────────────────────────────────────
    consistency = _load_json("consistency_latest.json")
    if consistency:
        out["consistency"] = {
            "verdict": consistency.get("verdict"),
            "mismatches": consistency.get("mismatches"),
        }
    else:
        out["consistency"] = {}

    # ── Error/blocked action rate ────────────────────────────────────
    error_actions = _query_db("""
        SELECT action, COUNT(*) AS cnt
        FROM transaction_log
        WHERE action IN ('blocked', 'delete', 'error')
        GROUP BY action
    """)
    out["error_actions"] = error_actions
    out["error_action_total"] = sum(r["cnt"] for r in error_actions)
    out["error_action_rate"] = (
        round(out["error_action_total"] / out["total_transactions"], 4)
        if out["total_transactions"] > 0 else 0
    )

    return out


# ──────────────────────────────────────────────────────────────────────
# 2. control_tower_breakdown
# ──────────────────────────────────────────────────────────────────────

def control_tower_breakdown() -> dict:
    """Detailed drill-down: per-component action map, per-actor component map,
    HITL review details, clinical decisions, cost by service, recent transactions."""

    out = {}

    # ── Per-component top actions ────────────────────────────────────
    comp_actions = _query_db("""
        SELECT component, action, COUNT(*) AS cnt
        FROM transaction_log
        GROUP BY component, action
        ORDER BY component, cnt DESC
    """)
    comp_map = {}
    for r in comp_actions:
        c = r["component"]
        if c not in comp_map:
            comp_map[c] = []
        comp_map[c].append({"action": r["action"], "count": r["cnt"]})
    # Keep top 5 actions per component
    out["component_action_map"] = {
        c: acts[:5] for c, acts in comp_map.items()
    }

    # ── Per-actor component map ──────────────────────────────────────
    actor_comps = _query_db("""
        SELECT actor, component, COUNT(*) AS cnt
        FROM transaction_log
        GROUP BY actor, component
        ORDER BY actor, cnt DESC
    """)
    actor_map = {}
    for r in actor_comps:
        a = r["actor"]
        if a not in actor_map:
            actor_map[a] = []
        actor_map[a].append({"component": r["component"], "count": r["cnt"]})
    out["actor_component_map"] = actor_map

    # ── Recent transactions (last 50) ────────────────────────────────
    recent = _query_db("""
        SELECT id, patient_id, component, action, actor, detail, ts_local
        FROM transaction_log
        ORDER BY id DESC LIMIT 50
    """)
    out["recent_transactions"] = recent

    # ── HITL reviews detail ──────────────────────────────────────────
    hitl = _query_db("""
        SELECT id, patient_id, analysis_id, fields_json, created_at
        FROM hitl_reviews ORDER BY id DESC
    """)
    for r in hitl:
        try:
            r["fields"] = json.loads(r.pop("fields_json", "{}"))
        except Exception:
            r["fields"] = {}
    out["hitl_reviews"] = hitl

    # ── Clinical decisions ───────────────────────────────────────────
    decisions = _query_db("""
        SELECT id, patient_id, ai_prediction, ai_confidence,
               neurologist_agreement, final_decision, reviewer, note, created_at
        FROM clinical_decisions ORDER BY id DESC
    """)
    out["clinical_decisions"] = decisions

    # ── Feedback entries ─────────────────────────────────────────────
    fb = _query_db("""
        SELECT id, patient_id, role, rating, correction, reason, created_at
        FROM feedback ORDER BY id DESC
    """)
    out["feedback"] = fb

    # ── Component findings ───────────────────────────────────────────
    findings = _query_db("""
        SELECT id, patient_id, component, doctor_finding, doctor,
               agree_with_ai, created_at
        FROM component_findings ORDER BY id DESC
    """)
    out["component_findings"] = findings

    # ── Cost by service (detailed) ───────────────────────────────────
    cost_svc = _query_db("""
        SELECT model_or_service, category, SUM(cost_usd) AS total_cost,
               SUM(requests) AS total_requests,
               SUM(tokens_in) AS total_tokens_in,
               SUM(tokens_out) AS total_tokens_out
        FROM finops_costs
        GROUP BY model_or_service, category
        ORDER BY total_cost DESC LIMIT 20
    """)
    out["cost_by_service"] = cost_svc

    # ── Cost over time ───────────────────────────────────────────────
    cost_time = _query_db("""
        SELECT cost_date, SUM(cost_usd) AS daily_cost,
               SUM(tokens_in) AS tokens_in, SUM(tokens_out) AS tokens_out
        FROM finops_costs
        GROUP BY cost_date
        ORDER BY cost_date DESC LIMIT 30
    """)
    out["cost_timeline"] = list(reversed(cost_time))

    # ── Patient transaction profiles ─────────────────────────────────
    patient_tx = _query_db("""
        SELECT patient_id, COUNT(*) AS tx_count,
               GROUP_CONCAT(DISTINCT component) AS components,
               GROUP_CONCAT(DISTINCT action) AS actions
        FROM transaction_log
        WHERE patient_id IS NOT NULL AND patient_id != ''
        GROUP BY patient_id
        ORDER BY tx_count DESC LIMIT 30
    """)
    out["patient_profiles"] = patient_tx

    # ── Hourly activity heatmap ──────────────────────────────────────
    hourly = _query_db("""
        SELECT CAST(SUBSTR(ts_local, 12, 2) AS INTEGER) AS hour,
               COUNT(*) AS cnt
        FROM transaction_log
        WHERE LENGTH(ts_local) >= 13
        GROUP BY hour ORDER BY hour
    """)
    out["hourly_heatmap"] = hourly

    # ── Component status matrix ──────────────────────────────────────
    try:
        with open(COVERAGE, "r") as f:
            cov = json.load(f)
        rows = cov.get("rows", [])
        status_matrix = [
            {"type": r["type"], "status": r["status"], "note": r.get("note", "")}
            for r in rows if r.get("status") != "not-pulled"
        ]
        out["component_status_matrix"] = status_matrix
    except Exception:
        out["component_status_matrix"] = []

    return out


# ──────────────────────────────────────────────────────────────────────
# 3. definitions
# ──────────────────────────────────────────────────────────────────────

def control_tower_definitions() -> dict:
    return {
        "control_tower_concept": [
            {
                "name": "AI Control Tower",
                "description": (
                    "A centralized oversight dashboard that aggregates status, "
                    "health, activity, and decision data from all AI system "
                    "components. Provides a single pane of glass for monitoring "
                    "system orchestration, component readiness, human oversight "
                    "actions, cost allocation, and data quality across the entire "
                    "clinical decision support pipeline."
                ),
            },
            {
                "name": "Component Registry",
                "description": (
                    "Tracks the lifecycle status of each AI component: planned "
                    "(designed but not built), scaffold (partial implementation), "
                    "built (fully functional with verified endpoints), or not-pulled "
                    "(exists in parent project but not applicable here)."
                ),
            },
        ],
        "system_components": [
            {"name": "Transaction Log", "description": "Audit trail of every system action — who did what, to which patient, via which component, and when."},
            {"name": "Analyses Engine", "description": "EEG classification pipeline producing disease predictions with confidence scores and signal quality assessments."},
            {"name": "HITL Reviews", "description": "Human-in-the-loop review records where clinicians verify, override, or annotate AI predictions."},
            {"name": "Clinical Decisions", "description": "Final clinical decision records capturing AI prediction, neurologist agreement, and final disposition."},
            {"name": "FinOps Cost Tracker", "description": "Financial operations tracking: API costs, GPU minutes, token usage per model/service/category."},
            {"name": "Component Findings", "description": "Doctor-level findings per AI component, including agreement/disagreement with AI output."},
            {"name": "Feedback Loop", "description": "Clinician feedback on AI outputs — ratings, corrections, and reasons for disagreement."},
        ],
        "metrics_and_kpis": [
            {"name": "Total Transactions", "description": "Count of all logged system actions. Higher volume indicates active system usage."},
            {"name": "Error Action Rate", "description": "Fraction of transactions that are blocked/error/delete actions. Lower is better; > 5% warrants investigation."},
            {"name": "HITL Coverage", "description": "Ratio of analyses that received human review. Higher coverage improves clinical safety."},
            {"name": "Cost Per Analysis", "description": "Total FinOps cost divided by number of analyses. Tracks resource efficiency."},
            {"name": "Component Readiness", "description": "Fraction of registered AI components in 'built' status. Tracks overall system maturity."},
            {"name": "Actor Diversity", "description": "Number of distinct actors (users, agents, systems) interacting with the system."},
        ],
        "clinical_relevance": [
            {
                "standard": "IEC 62304",
                "description": (
                    "Medical device software lifecycle. The Control Tower provides "
                    "traceability required by clause 5.5 (software verification) and "
                    "clause 6.3 (post-market surveillance) by logging all component "
                    "interactions and human oversight actions."
                ),
            },
            {
                "standard": "FDA AI/ML PCCP",
                "description": (
                    "Predetermined Change Control Plan. Centralized monitoring of "
                    "component status, drift, consistency, and HITL overrides provides "
                    "evidence for the FDA's total product lifecycle approach."
                ),
            },
            {
                "standard": "ILAE Classification",
                "description": (
                    "International League Against Epilepsy. The Control Tower ensures "
                    "AI predictions are tracked against ILAE taxonomy and that clinical "
                    "decisions align with current classification standards."
                ),
            },
            {
                "standard": "ISO 14971 Risk Management",
                "description": (
                    "Risk management for medical devices. Transaction logging and error "
                    "action tracking support hazard identification (clause 4.3) and "
                    "risk estimation (clause 4.4) for the AI system."
                ),
            },
        ],
        "remediation_strategies": [
            {
                "trigger": "Component status remains 'scaffold' or 'planned'",
                "strategy": (
                    "1. Review component requirements and data availability. "
                    "2. Prioritize based on clinical value and data readiness. "
                    "3. Build real backend + frontend + endpoint verification. "
                    "4. Update registry status to 'built' after verification."
                ),
            },
            {
                "trigger": "Error action rate exceeds 5%",
                "strategy": (
                    "1. Review blocked/error transactions for root cause. "
                    "2. Check if specific components or actors are disproportionately affected. "
                    "3. Fix underlying permission, data, or logic issues. "
                    "4. Monitor rate after remediation."
                ),
            },
            {
                "trigger": "HITL coverage below threshold",
                "strategy": (
                    "1. Identify analyses without human review. "
                    "2. Route high-confidence predictions for spot-check review. "
                    "3. Require mandatory review for low-confidence predictions. "
                    "4. Track coverage trend over time."
                ),
            },
            {
                "trigger": "Cost anomaly detected",
                "strategy": (
                    "1. Identify which service/model caused the spike. "
                    "2. Check for runaway inference loops or duplicate requests. "
                    "3. Implement rate limiting or caching if appropriate. "
                    "4. Set cost alerts at category level."
                ),
            },
            {
                "trigger": "System health degradation",
                "strategy": (
                    "1. Check backend HTTP status and API error count. "
                    "2. Verify database connectivity. "
                    "3. Restart affected services. "
                    "4. Confirm health endpoint returns HTTP 200."
                ),
            },
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# CLI self-test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint

    print("=" * 60)
    print("AI CONTROL TOWER — OVERVIEW")
    print("=" * 60)
    ov = control_tower_overview()
    pprint.pprint(ov, width=120)

    print("\n" + "=" * 60)
    print("AI CONTROL TOWER — BREAKDOWN")
    print("=" * 60)
    bd = control_tower_breakdown()
    pprint.pprint(bd, width=120)

    print("\n" + "=" * 60)
    print("AI CONTROL TOWER — DEFINITIONS")
    print("=" * 60)
    df = control_tower_definitions()
    pprint.pprint(df, width=120)
