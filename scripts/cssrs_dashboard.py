"""C-SSRS Dashboard — Columbia Suicide Severity Rating Scale analytics.

All data from REAL C-SSRS assessments in data/clinical.db (assessments table, instrument='CSSRS').

The C-SSRS is a validated, semi-structured interview that measures suicidal ideation
and behavior. It is FDA-cleared and used worldwide in clinical trials, emergency
departments, and psychiatric practice.

Structure:
  Screening (6 binary items):
    1. Wish to be dead
    2. Non-specific active suicidal thoughts
    3. Active suicidal ideation with any methods (not plan)
    4. Active suicidal ideation with some intent to act
    5. Active suicidal ideation with specific plan and intent
    6. Suicidal behavior (lifetime or recent)

  Intensity (5 items, each 1-5, only scored if ideation present):
    1. Frequency
    2. Duration
    3. Controllability
    4. Deterrents
    5. Reasons for ideation

  Score range: 0 (no ideation) to 31 (max: 6 screening + 5*5 intensity)
  Risk tiers: None (0), Low (1-7), Moderate (8-16), High (17-31)

Reference:
  Posner K, Brown GK, Stanley B, et al. The Columbia-Suicide Severity
  Rating Scale: initial validity and internal consistency findings from
  three multisite studies with adolescents and adults. Am J Psychiatry.
  2011;168(12):1266-1277.

Author: Research Team
"""
import sqlite3
import json
from pathlib import Path
from collections import Counter, defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"

SCREEN_ITEMS = [
    {"id": "screen1", "label": "Wish to be dead", "description": "Have you wished you were dead or wished you could go to sleep and not wake up?"},
    {"id": "screen2", "label": "Non-specific active suicidal thoughts", "description": "Have you actually had any thoughts of killing yourself?"},
    {"id": "screen3", "label": "Active ideation with method", "description": "Have you been thinking about how you might do this?"},
    {"id": "screen4", "label": "Active ideation with intent", "description": "Have you had these thoughts and had some intention of acting on them?"},
    {"id": "screen5", "label": "Active ideation with plan and intent", "description": "Have you started to work out or worked out the details of how to kill yourself?"},
    {"id": "screen6", "label": "Suicidal behavior", "description": "Have you ever done anything, started to do anything, or prepared to do anything to end your life?"},
]

INTENSITY_ITEMS = [
    {"id": "intensity1", "label": "Frequency", "range": "1-5", "description": "How often have you had these thoughts?"},
    {"id": "intensity2", "label": "Duration", "range": "1-5", "description": "How long do the thoughts last?"},
    {"id": "intensity3", "label": "Controllability", "range": "1-5", "description": "Can you stop thinking about it? (1=easily, 5=unable)"},
    {"id": "intensity4", "label": "Deterrents", "range": "1-5", "description": "What stops you from acting? (1=strong deterrents, 5=none)"},
    {"id": "intensity5", "label": "Reasons for ideation", "range": "1-5", "description": "What is the reason? (1=to get attention, 5=to end pain permanently)"},
]

RISK_TIERS = [
    {"range": [0, 0], "label": "None", "color": "#22c55e", "action": "Standard follow-up"},
    {"range": [1, 7], "label": "Low", "color": "#eab308", "action": "Monitor at next visit, document in chart"},
    {"range": [8, 16], "label": "Moderate", "color": "#f97316", "action": "Safety planning intervention, increase monitoring frequency"},
    {"range": [17, 31], "label": "High", "color": "#ef4444", "action": "Immediate safety intervention, consider psychiatric referral, restrict access to means"},
]


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def _risk_tier(score):
    for t in RISK_TIERS:
        if t["range"][0] <= score <= t["range"][1]:
            return t["label"]
    return "High"


def overview():
    """Summary KPIs + risk distribution + per-patient latest scores + alerts."""
    rows = _rows("SELECT * FROM assessments WHERE instrument='CSSRS' ORDER BY created_at DESC")
    if not rows:
        return {"total_assessments": 0, "patients": 0, "message": "No C-SSRS data yet"}

    patients = set()
    latest_per_patient = {}
    all_scores = []
    risk_dist = Counter()
    alerts = []

    for r in rows:
        patients.add(r["patient_id"])
        all_scores.append(r["score"])
        risk_dist[r["level"]] += 1
        if r["patient_id"] not in latest_per_patient:
            latest_per_patient[r["patient_id"]] = r
        if r["alert"]:
            alerts.append({
                "patient_id": r["patient_id"],
                "alert": r["alert"],
                "score": r["score"],
                "level": r["level"],
                "date": r["created_at"],
            })

    avg_score = round(sum(all_scores) / len(all_scores), 1)
    ideation_rate = round(sum(1 for s in all_scores if s > 0) / len(all_scores) * 100, 1)

    patient_summary = []
    for pid, r in sorted(latest_per_patient.items()):
        patient_summary.append({
            "patient_id": pid,
            "latest_score": r["score"],
            "max_score": r["max_score"],
            "interpretation": r["interpretation"],
            "level": r["level"],
            "alert": r["alert"],
            "assessed_at": r["created_at"],
        })

    # Sort patient summary by risk (high first)
    level_order = {"high": 0, "moderate": 1, "low": 2, "none": 3}
    patient_summary.sort(key=lambda x: level_order.get(x["level"], 4))

    return {
        "total_assessments": len(rows),
        "unique_patients": len(patients),
        "avg_score": avg_score,
        "ideation_rate_pct": ideation_rate,
        "risk_distribution": dict(risk_dist),
        "active_alerts": alerts[:10],
        "patient_summary": patient_summary,
    }


def breakdown():
    """Screening item rates, intensity analysis, trend, per-patient history."""
    rows = _rows("SELECT * FROM assessments WHERE instrument='CSSRS' ORDER BY created_at ASC")
    if not rows:
        return {"message": "No C-SSRS data"}

    # Screening item endorsement rates
    screen_counts = {s["id"]: 0 for s in SCREEN_ITEMS}
    intensity_totals = {it["id"]: [] for it in INTENSITY_ITEMS}
    n = len(rows)

    for r in rows:
        answers = json.loads(r["answers_json"]) if r["answers_json"] else {}
        for s in SCREEN_ITEMS:
            if answers.get(s["id"], 0) > 0:
                screen_counts[s["id"]] += 1
        for it in INTENSITY_ITEMS:
            val = answers.get(it["id"], 0)
            if val > 0:
                intensity_totals[it["id"]].append(val)

    screening_rates = []
    for s in SCREEN_ITEMS:
        screening_rates.append({
            "id": s["id"],
            "label": s["label"],
            "endorsed_count": screen_counts[s["id"]],
            "total": n,
            "rate_pct": round(screen_counts[s["id"]] / n * 100, 1),
        })

    intensity_summary = []
    for it in INTENSITY_ITEMS:
        vals = intensity_totals[it["id"]]
        intensity_summary.append({
            "id": it["id"],
            "label": it["label"],
            "avg": round(sum(vals) / len(vals), 1) if vals else 0,
            "max": 5,
            "n_rated": len(vals),
        })

    # Monthly trend
    date_scores = defaultdict(list)
    for r in rows:
        month = (r["created_at"] or "")[:7]
        if month:
            date_scores[month].append(r["score"])
    trend = [{"month": m, "avg_score": round(sum(v) / len(v), 1), "count": len(v),
              "ideation_pct": round(sum(1 for s in v if s > 0) / len(v) * 100, 1)}
             for m, v in sorted(date_scores.items())]

    # Per-patient history
    patient_history = defaultdict(list)
    for r in rows:
        patient_history[r["patient_id"]].append({
            "score": r["score"],
            "interpretation": r["interpretation"],
            "level": r["level"],
            "alert": r["alert"],
            "date": r["created_at"],
        })

    # Risk transition: for patients with 2+ assessments, show changes
    risk_transitions = []
    for pid, hist in patient_history.items():
        if len(hist) >= 2:
            risk_transitions.append({
                "patient_id": pid,
                "first_level": hist[0]["level"],
                "latest_level": hist[-1]["level"],
                "first_score": hist[0]["score"],
                "latest_score": hist[-1]["score"],
                "assessments": len(hist),
            })

    return {
        "screening_rates": screening_rates,
        "intensity_summary": intensity_summary,
        "trend": trend,
        "patient_history": dict(patient_history),
        "risk_transitions": risk_transitions,
    }


def definitions():
    """Metric definitions for the C-SSRS dashboard."""
    return {
        "title": "Columbia Suicide Severity Rating Scale (C-SSRS) — Metric Definitions",
        "definitions": [
            {"term": "C-SSRS", "definition": "Columbia Suicide Severity Rating Scale — an FDA-cleared, evidence-based tool for assessing suicidal ideation and behavior. Developed by Columbia University."},
            {"term": "Score Range", "definition": "0 (no ideation) to 31 (maximum severity). Screening contributes 0-6, intensity contributes 0-25."},
            {"term": "Screening Items (6)", "definition": "Binary (yes/no) questions assessing: wish to be dead, non-specific thoughts, ideation with method, intent without plan, intent with plan, suicidal behavior."},
            {"term": "Intensity Items (5)", "definition": "Rated 1-5 each (only if ideation present): frequency, duration, controllability, deterrents, reasons for ideation."},
            {"term": "None (Score 0)", "definition": "No suicidal ideation endorsed. Standard follow-up at next scheduled visit."},
            {"term": "Low (Score 1-7)", "definition": "Passive ideation or low-intensity active ideation. Monitor at next visit, document in chart."},
            {"term": "Moderate (Score 8-16)", "definition": "Active ideation with some method/intent. Safety planning intervention recommended, increase monitoring frequency."},
            {"term": "High (Score 17-31)", "definition": "Active ideation with plan/intent or suicidal behavior. Immediate safety intervention required, psychiatric referral, restrict access to means."},
            {"term": "Ideation Rate", "definition": "Percentage of assessments where any suicidal ideation was endorsed (score > 0)."},
            {"term": "Screening Endorsement Rate", "definition": "Percentage of assessments where each specific screening item was endorsed. Higher items (3-6) indicate greater severity."},
            {"term": "Intensity Profile", "definition": "Average intensity ratings across the 5 dimensions. High controllability/low deterrents scores indicate greatest concern."},
            {"term": "Epilepsy Context", "definition": "Epilepsy patients have 3-5x higher suicide risk than the general population (Christensen et al., 2007). C-SSRS screening is especially important in this population."},
        ],
    }
