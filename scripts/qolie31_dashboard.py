"""QOLIE-31 Dashboard — Quality of Life in Epilepsy-31 analytics.

All data from REAL QOLIE-31 assessments in data/clinical.db (assessments table, instrument='QOLIE31').

The QOLIE-31 is a validated epilepsy-specific 31-item quality of life measure
with 7 subscale domains scored 0-100 each (higher = better QoL).

Subscale domains (answers_json item1-item7):
  1. Seizure Worry (item1) - Fear and worry about seizures
  2. Overall QoL (item2) - General quality of life rating
  3. Emotional Well-being (item3) - Emotional health and mood
  4. Energy/Fatigue (item4) - Energy levels and fatigue impact
  5. Cognitive Functioning (item5) - Memory, concentration, language
  6. Medication Effects (item6) - Side effects of anti-epileptic drugs
  7. Social Function (item7) - Social activities and relationships

Overall composite score: 0-100 (higher = better QoL)
Severity tiers:
  0-25   Poor QoL
  26-50  Fair QoL
  51-75  Good QoL
  76-100 Excellent QoL

Clinical note: MCID (Minimally Clinically Important Difference) is a 5-point
change. QoL impairment is the primary patient-centered outcome; seizure freedom
alone does not guarantee QoL improvement.

Reference:
  Cramer JA, Perrine K, Devinsky O, Bryant-Comstock L, Meador K, Hermann B.
  Development and cross-cultural translations of a 31-item quality of life in
  epilepsy inventory. Epilepsia. 1998;39(1):81-88.

Author: Research Team
"""
import sqlite3
import json
from pathlib import Path
from collections import Counter, defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"

QOLIE31_ITEMS = [
    {"id": "item1", "label": "Seizure Worry", "description": "Fear and worry about seizures"},
    {"id": "item2", "label": "Overall QoL", "description": "General quality of life rating"},
    {"id": "item3", "label": "Emotional Well-being", "description": "Emotional health and mood"},
    {"id": "item4", "label": "Energy/Fatigue", "description": "Energy levels and fatigue impact"},
    {"id": "item5", "label": "Cognitive Functioning", "description": "Memory, concentration, language"},
    {"id": "item6", "label": "Medication Effects", "description": "Side effects of anti-epileptic drugs"},
    {"id": "item7", "label": "Social Function", "description": "Social activities and relationships"},
]

SEVERITY_TIERS = [
    {"range": [0, 25], "label": "Poor", "color": "#ef4444", "action": "Comprehensive QoL intervention; review seizure control, AED side effects, and psychosocial support"},
    {"range": [26, 50], "label": "Fair", "color": "#eab308", "action": "Targeted intervention on lowest-scoring domains; consider AED optimization and counselling"},
    {"range": [51, 75], "label": "Good", "color": "#84cc16", "action": "Maintain current management; monitor for domain-specific decline"},
    {"range": [76, 100], "label": "Excellent", "color": "#22c55e", "action": "Continue current management; annual reassessment"},
]


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def _severity(score):
    for t in SEVERITY_TIERS:
        if t["range"][0] <= score <= t["range"][1]:
            return t["label"].lower()
    return "excellent"


def overview():
    rows = _rows("SELECT * FROM assessments WHERE instrument='QOLIE31' ORDER BY created_at DESC")
    if not rows:
        return {
            "total_assessments": 0, "unique_patients": 0,
            "avg_score": 0, "pct_poor_fair": 0,
            "severity_distribution": {}, "patient_summary": [],
            "active_alerts": [],
        }

    scores = [r["score"] for r in rows]
    pids = list({r["patient_id"] for r in rows})

    # Severity distribution
    sev_counts = Counter()
    for r in rows:
        sev_counts[_severity(r["score"])] += 1

    # % poor or fair (score < 50)
    poor_fair = sum(1 for s in scores if s < 50)
    pct_pf = round(100 * poor_fair / len(scores), 1) if scores else 0

    # Per-patient latest
    latest = {}
    for r in rows:
        pid = r["patient_id"]
        if pid not in latest:
            latest[pid] = r

    patient_summary = []
    for pid, r in sorted(latest.items()):
        patient_summary.append({
            "patient_id": pid,
            "latest_score": r["score"],
            "max_score": r["max_score"],
            "severity": _severity(r["score"]),
            "interpretation": r.get("interpretation", ""),
            "assessed_at": r.get("created_at", ""),
        })

    # Active alerts: score < 50
    alerts = []
    for pid, r in latest.items():
        if r["score"] < 50:
            alerts.append({
                "patient_id": pid,
                "alert": f"Score {int(r['score'])}/100 — {_severity(r['score'])}",
                "score": int(r["score"]),
                "severity": _severity(r["score"]),
            })

    return {
        "total_assessments": len(rows),
        "unique_patients": len(pids),
        "avg_score": round(sum(scores) / len(scores), 1),
        "pct_poor_fair": pct_pf,
        "severity_distribution": dict(sev_counts),
        "patient_summary": patient_summary,
        "active_alerts": alerts,
    }


def breakdown():
    rows = _rows("SELECT * FROM assessments WHERE instrument='QOLIE31' ORDER BY created_at ASC")
    if not rows:
        return {
            "domain_averages": [], "severity_transitions": [],
            "trend": [], "patient_history": {},
            "domain_comparison": [],
        }

    # Per-domain average scores
    domain_totals = {it["id"]: [] for it in QOLIE31_ITEMS}
    parsed = 0
    for r in rows:
        try:
            ans = json.loads(r["answers_json"]) if r["answers_json"] else {}
            parsed += 1
            for it in QOLIE31_ITEMS:
                val = ans.get(it["id"])
                if val is not None:
                    domain_totals[it["id"]].append(val)
        except Exception:
            pass

    domain_averages = []
    for it in QOLIE31_ITEMS:
        vals = domain_totals[it["id"]]
        avg = round(sum(vals) / len(vals), 1) if vals else 0
        domain_averages.append({
            "id": it["id"],
            "label": it["label"],
            "avg_score": avg,
            "min_score": min(vals) if vals else 0,
            "max_score": max(vals) if vals else 0,
            "n": len(vals),
        })

    # Domain comparison (radar-style data)
    domain_comparison = []
    for it in QOLIE31_ITEMS:
        vals = domain_totals[it["id"]]
        domain_comparison.append({
            "domain": it["label"],
            "value": round(sum(vals) / len(vals), 1) if vals else 0,
            "max": 100,
        })

    # Severity transitions (patients with 2+ assessments)
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r["patient_id"]].append(r)

    severity_transitions = []
    for pid, recs in sorted(by_patient.items()):
        if len(recs) >= 2:
            first = recs[0]
            last = recs[-1]
            change = int(last["score"]) - int(first["score"])
            severity_transitions.append({
                "patient_id": pid,
                "first_score": int(first["score"]),
                "first_severity": _severity(first["score"]),
                "latest_score": int(last["score"]),
                "latest_severity": _severity(last["score"]),
                "change": change,
                "assessments": len(recs),
            })

    # Monthly trend
    monthly = defaultdict(list)
    for r in rows:
        month = (r.get("created_at") or "")[:7]
        if month:
            monthly[month].append(r["score"])

    trend = []
    for month in sorted(monthly.keys()):
        vals = monthly[month]
        trend.append({
            "month": month,
            "avg_score": round(sum(vals) / len(vals), 1),
            "count": len(vals),
            "pct_poor_fair": round(100 * sum(1 for v in vals if v < 50) / len(vals), 1),
        })

    # Per-patient history
    patient_history = {}
    for pid, recs in sorted(by_patient.items()):
        patient_history[pid] = [
            {
                "score": int(r["score"]),
                "severity": _severity(r["score"]),
                "date": r.get("created_at", ""),
            }
            for r in recs
        ]

    return {
        "domain_averages": domain_averages,
        "severity_transitions": severity_transitions,
        "trend": trend,
        "patient_history": patient_history,
        "domain_comparison": domain_comparison,
    }


def definitions():
    return {
        "title": "QOLIE-31 — Quality of Life in Epilepsy, 31-Item Inventory",
        "reference": "Cramer JA, Perrine K, Devinsky O, Bryant-Comstock L, Meador K, Hermann B. Development and cross-cultural translations of a 31-item quality of life in epilepsy inventory. Epilepsia. 1998;39(1):81-88.",
        "domains": [
            {**it, "scoring": "0-100 subscale score (higher = better quality of life)"}
            for it in QOLIE31_ITEMS
        ],
        "severity_tiers": SEVERITY_TIERS,
        "clinical_notes": [
            {"term": "MCID", "definition": "A 5-point change in QOLIE-31 composite score is the minimally clinically important difference."},
            {"term": "Epilepsy relevance", "definition": "QoL impairment is the primary patient-centered outcome; seizure freedom alone does not guarantee QoL improvement. QOLIE-31 captures domains most affected by epilepsy."},
            {"term": "AED effects", "definition": "Cognitive Functioning and Energy/Fatigue subscales are most sensitive to medication changes. AED simplification often improves these domains."},
            {"term": "Seizure worry", "definition": "Often the most impaired domain, correlates with seizure frequency. Seizure Worry may remain elevated even after seizure reduction if patients fear recurrence."},
            {"term": "Treatment response", "definition": "A 10-point improvement in QOLIE-31 composite score indicates a clinically meaningful treatment response."},
        ],
    }
