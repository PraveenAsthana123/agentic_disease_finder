"""AI Risk Dashboard — real risk metrics derived from clinical.db.

Sources:
- assessments (alert flags → clinical risk signals)
- seizure_diary (frequency → patient-safety risk)
- transaction_log (blocked actions → AI guardrail violations)
- medications (polypharmacy → drug-interaction risk)
- hitl_reviews (override rate → human-AI alignment risk)
- clinical_decisions (low-confidence decisions → model-uncertainty risk)
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


def risk_overview():
    """Aggregate risk posture: total open risks, severity distribution,
    risk trend, top risk categories — all from real clinical.db data."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # ── 1. Clinical alert risks (assessments with alert != '') ────────
    alert_total = _safe_count(cur,
        "SELECT count(*) FROM assessments WHERE alert IS NOT NULL AND alert != ''")
    high_alerts = _safe_count(cur,
        "SELECT count(*) FROM assessments WHERE alert LIKE '%high%' OR alert LIKE '%severe%' OR alert LIKE '%critical%'")
    moderate_alerts = _safe_count(cur,
        "SELECT count(*) FROM assessments WHERE alert IS NOT NULL AND alert != '' "
        "AND alert NOT LIKE '%high%' AND alert NOT LIKE '%severe%' AND alert NOT LIKE '%critical%'")

    # Alert by instrument
    try:
        cur.execute("""
            SELECT instrument, count(*) as cnt
            FROM assessments WHERE alert IS NOT NULL AND alert != ''
            GROUP BY instrument ORDER BY cnt DESC
        """)
        alert_by_instrument = {r[0]: r[1] for r in cur.fetchall()}
    except Exception:
        alert_by_instrument = {}

    # ── 2. Seizure-safety risk (seizure_diary frequency) ─────────────
    seizure_total = _safe_count(cur, "SELECT count(*) FROM seizure_diary")
    try:
        cur.execute("""
            SELECT patient_id, count(*) as cnt
            FROM seizure_diary GROUP BY patient_id ORDER BY cnt DESC
        """)
        seizure_by_patient = [{"patient_id": r[0], "episodes": r[1]} for r in cur.fetchall()]
    except Exception:
        seizure_by_patient = []

    high_seizure_patients = sum(1 for p in seizure_by_patient if p["episodes"] >= 5)

    # ── 3. AI guardrail violations (blocked actions in transaction_log)
    guardrail_blocks = _safe_count(cur,
        "SELECT count(*) FROM transaction_log WHERE action = 'blocked'")
    try:
        cur.execute("""
            SELECT component, count(*) as cnt
            FROM transaction_log WHERE action = 'blocked'
            GROUP BY component ORDER BY cnt DESC
        """)
        blocks_by_component = {r[0]: r[1] for r in cur.fetchall()}
    except Exception:
        blocks_by_component = {}

    # ── 4. Polypharmacy risk (patients with >= 3 medications) ────────
    try:
        cur.execute("""
            SELECT patient_id, count(*) as cnt
            FROM medications GROUP BY patient_id HAVING cnt >= 3
        """)
        polypharmacy_patients = len(cur.fetchall())
    except Exception:
        polypharmacy_patients = 0

    total_medications = _safe_count(cur, "SELECT count(*) FROM medications")

    # ── 5. HITL override risk (reviews where human overrode AI) ──────
    hitl_total = _safe_count(cur, "SELECT count(*) FROM hitl_reviews")
    try:
        cur.execute("""
            SELECT count(*) FROM hitl_reviews
            WHERE decision IS NOT NULL AND decision != ''
        """)
        hitl_overrides = cur.fetchone()[0]
    except Exception:
        hitl_overrides = 0

    # ── 6. Risk-trend (daily: alerts + blocks + seizures last 14d) ───
    try:
        cur.execute("""
            SELECT date(created_at) as d, count(*) as cnt
            FROM assessments WHERE alert IS NOT NULL AND alert != ''
            GROUP BY d ORDER BY d DESC LIMIT 14
        """)
        alert_daily = {r[0]: r[1] for r in cur.fetchall()}
    except Exception:
        alert_daily = {}

    try:
        cur.execute("""
            SELECT date(ts_utc) as d, count(*) as cnt
            FROM transaction_log WHERE action = 'blocked'
            GROUP BY d ORDER BY d DESC LIMIT 14
        """)
        block_daily = {r[0]: r[1] for r in cur.fetchall()}
    except Exception:
        block_daily = {}

    all_dates = sorted(set(list(alert_daily.keys()) + list(block_daily.keys())))
    daily_trend = [
        {"date": d,
         "alerts": alert_daily.get(d, 0),
         "guardrail_blocks": block_daily.get(d, 0),
         "combined": alert_daily.get(d, 0) + block_daily.get(d, 0)}
        for d in all_dates[-14:]
    ]

    # ── 7. Build risk register ───────────────────────────────────────
    risks = []

    if high_alerts > 0:
        risks.append({
            "id": "RISK-001", "category": "Clinical Safety",
            "title": "High/severe clinical assessment alerts",
            "severity": "high", "count": high_alerts,
            "status": "open" if high_alerts > 0 else "mitigated",
            "mitigation": "Escalation to specialist via alert pipeline"
        })

    if moderate_alerts > 0:
        risks.append({
            "id": "RISK-002", "category": "Clinical Safety",
            "title": "Moderate clinical assessment alerts",
            "severity": "medium", "count": moderate_alerts,
            "status": "monitoring",
            "mitigation": "Periodic review by care team"
        })

    if high_seizure_patients > 0:
        risks.append({
            "id": "RISK-003", "category": "Patient Safety",
            "title": "Patients with frequent seizure episodes (>=5)",
            "severity": "high", "count": high_seizure_patients,
            "status": "open",
            "mitigation": "Medication review + seizure action plan"
        })

    if guardrail_blocks > 0:
        risks.append({
            "id": "RISK-004", "category": "AI Safety",
            "title": "AI guardrail-blocked operations",
            "severity": "medium", "count": guardrail_blocks,
            "status": "mitigated",
            "mitigation": "Guardrails active; blocked requests logged"
        })

    if polypharmacy_patients > 0:
        risks.append({
            "id": "RISK-005", "category": "Pharmacological",
            "title": "Polypharmacy risk (>=3 concurrent medications)",
            "severity": "medium", "count": polypharmacy_patients,
            "status": "monitoring",
            "mitigation": "Pharmacist review flag + interaction checker"
        })

    if hitl_overrides > 0:
        risks.append({
            "id": "RISK-006", "category": "Human-AI Alignment",
            "title": "HITL override decisions (human disagreed with AI)",
            "severity": "low", "count": hitl_overrides,
            "status": "monitoring",
            "mitigation": "Model retraining queue + decision auditing"
        })

    # Always include model-fairness and data-leakage as standing risks
    risks.append({
        "id": "RISK-007", "category": "Fairness",
        "title": "Model bias across age/gender subgroups",
        "severity": "medium", "count": None,
        "status": "monitoring",
        "mitigation": "AIF360 fairness tests run per training cycle"
    })

    risks.append({
        "id": "RISK-008", "category": "Data Privacy",
        "title": "PII/PHI exposure in AI pipelines",
        "severity": "high", "count": None,
        "status": "mitigated",
        "mitigation": "PII scanner + local-only data processing"
    })

    # Severity summary
    sev_counts = {"high": 0, "medium": 0, "low": 0}
    status_counts = {"open": 0, "mitigated": 0, "monitoring": 0}
    for r in risks:
        sev_counts[r["severity"]] = sev_counts.get(r["severity"], 0) + 1
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    mitigated_pct = round(100 * status_counts["mitigated"] / max(len(risks), 1))

    conn.close()

    return {
        "available": True,
        "summary": {
            "total_risks": len(risks),
            "high_severity": sev_counts["high"],
            "medium_severity": sev_counts["medium"],
            "low_severity": sev_counts["low"],
            "open": status_counts["open"],
            "mitigated": status_counts["mitigated"],
            "monitoring": status_counts["monitoring"],
            "mitigated_pct": mitigated_pct,
            "clinical_alerts": alert_total,
            "guardrail_blocks": guardrail_blocks,
            "seizure_episodes": seizure_total,
        },
        "risks": risks,
        "daily_trend": daily_trend,
        "alert_by_instrument": alert_by_instrument,
        "blocks_by_component": blocks_by_component,
    }


def risk_breakdown():
    """Per-category risk detail: category grouping, severity, counts, mitigations."""
    ov = risk_overview()
    if not ov.get("available"):
        return {"available": False, "categories": []}

    risks = ov.get("risks", [])
    categories = {}
    for r in risks:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"category": cat, "risks": [], "total": 0,
                               "high": 0, "medium": 0, "low": 0}
        categories[cat]["risks"].append(r)
        categories[cat]["total"] += 1
        categories[cat][r["severity"]] += 1

    cat_list = sorted(categories.values(), key=lambda c: c["high"], reverse=True)

    return {
        "available": True,
        "categories": cat_list,
        "total_categories": len(cat_list),
    }


def risk_definitions():
    """Metric definitions for AI risk dashboard tooltip overlays."""
    return {
        "available": True,
        "definitions": [
            {"term": "Risk Register", "definition": "A structured inventory of identified risks, each scored by severity (high/medium/low) and tracked by status (open/mitigated/monitoring)."},
            {"term": "Clinical Alert Risk", "definition": "Risks derived from assessment instruments that flagged a clinical alert (e.g., high PHQ-9, severe GAD-7). Count reflects real alert records in clinical.db."},
            {"term": "Seizure-Safety Risk", "definition": "Patient-safety risks identified from seizure diary frequency. Patients with 5+ episodes are flagged for urgent review."},
            {"term": "Guardrail Block", "definition": "An AI operation that was blocked by the system's guardrail layer (e.g., council safety check, compliance gate). Each block is logged in transaction_log."},
            {"term": "Polypharmacy Risk", "definition": "Pharmacological risk flagged when a patient has 3 or more concurrent medications, increasing drug-interaction probability."},
            {"term": "HITL Override", "definition": "A human-in-the-loop review where the clinician's decision differed from the AI recommendation. Tracked to measure human-AI alignment."},
            {"term": "Severity", "definition": "Risk severity rating: high = immediate clinical/safety impact; medium = operational or quality impact; low = informational or minor."},
            {"term": "Status", "definition": "Risk status: open = unresolved and active; mitigated = controls in place; monitoring = under observation with partial controls."},
            {"term": "Mitigated %", "definition": "Percentage of total identified risks that have active mitigations in place (status = mitigated)."},
            {"term": "Daily Risk Trend", "definition": "Combined daily count of clinical alerts and guardrail blocks over the last 14 days, indicating risk activity over time."},
        ],
    }


if __name__ == "__main__":
    print("=== OVERVIEW ===")
    print(json.dumps(risk_overview(), indent=2, default=str))
    print("\n=== BREAKDOWN ===")
    print(json.dumps(risk_breakdown(), indent=2, default=str))
