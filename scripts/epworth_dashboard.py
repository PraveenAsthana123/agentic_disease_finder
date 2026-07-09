"""Epworth Sleepiness Scale (ESS) Dashboard — daytime sleepiness analytics.

All data from REAL ESS assessments in data/clinical.db (assessments table, instrument='EPWORTH').

The ESS is a validated 8-item self-administered questionnaire that measures a
subject's general level of daytime sleepiness. Respondents rate their likelihood
of dozing off or falling asleep in eight common situations on a 0-3 scale.

Structure:
  8 situational items (each scored 0-3):
    1. Sitting and reading
    2. Watching TV
    3. Sitting inactive in a public place (e.g., a theater or meeting)
    4. As a passenger in a car for an hour without a break
    5. Lying down to rest in the afternoon when circumstances permit
    6. Sitting and talking to someone
    7. Sitting quietly after lunch without alcohol
    8. In a car, while stopped for a few minutes in traffic

  Response options:
    0 = Would never doze
    1 = Slight chance of dozing
    2 = Moderate chance of dozing
    3 = High chance of dozing

  Score range: 0 (no sleepiness) to 24 (maximum sleepiness)
  Severity tiers: Normal (0-10), Mild (11-12), Moderate (13-15), Severe (16-24)

Reference:
  Johns MW. A new method for measuring daytime sleepiness: the Epworth
  sleepiness scale. Sleep. 1991;14(6):540-545.

Author: Research Team
"""
import sqlite3
import json
from pathlib import Path
from collections import Counter, defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"

SITUATION_ITEMS = [
    {"id": "item1", "label": "Sitting and reading", "description": "How likely are you to doze off or fall asleep while sitting and reading?"},
    {"id": "item2", "label": "Watching TV", "description": "How likely are you to doze off or fall asleep while watching TV?"},
    {"id": "item3", "label": "Sitting inactive in a public place", "description": "How likely are you to doze off or fall asleep while sitting inactive in a public place (e.g., a theater or meeting)?"},
    {"id": "item4", "label": "Passenger in a car for an hour", "description": "How likely are you to doze off or fall asleep as a passenger in a car for an hour without a break?"},
    {"id": "item5", "label": "Lying down to rest in the afternoon", "description": "How likely are you to doze off or fall asleep while lying down to rest in the afternoon when circumstances permit?"},
    {"id": "item6", "label": "Sitting and talking to someone", "description": "How likely are you to doze off or fall asleep while sitting and talking to someone?"},
    {"id": "item7", "label": "Sitting quietly after lunch (no alcohol)", "description": "How likely are you to doze off or fall asleep while sitting quietly after lunch without alcohol?"},
    {"id": "item8", "label": "In a car, stopped in traffic", "description": "How likely are you to doze off or fall asleep in a car, while stopped for a few minutes in traffic?"},
]

SEVERITY_TIERS = [
    {"range": [0, 10], "label": "Normal", "color": "#22c55e", "action": "No excessive daytime sleepiness; standard follow-up"},
    {"range": [11, 12], "label": "Mild", "color": "#eab308", "action": "Mild excessive daytime sleepiness; evaluate sleep hygiene, consider sleep study if symptomatic"},
    {"range": [13, 15], "label": "Moderate", "color": "#f97316", "action": "Moderate excessive daytime sleepiness; recommend polysomnography, assess medication side effects"},
    {"range": [16, 24], "label": "Severe", "color": "#ef4444", "action": "Severe excessive daytime sleepiness; urgent sleep medicine referral, driving safety counseling, evaluate for sleep apnea/narcolepsy"},
]


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def _severity_tier(score):
    for t in SEVERITY_TIERS:
        if t["range"][0] <= score <= t["range"][1]:
            return t["label"]
    return "Severe"


def overview():
    """Summary KPIs + severity distribution + per-patient latest scores + alerts."""
    rows = _rows("SELECT * FROM assessments WHERE instrument='EPWORTH' ORDER BY created_at DESC")
    if not rows:
        return {"total_assessments": 0, "patients": 0, "message": "No Epworth data yet"}

    patients = set()
    latest_per_patient = {}
    all_scores = []
    severity_dist = Counter()
    alerts = []

    for r in rows:
        patients.add(r["patient_id"])
        all_scores.append(r["score"])
        severity_dist[r["level"]] += 1
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
    excessive_sleepiness_rate = round(sum(1 for s in all_scores if s > 10) / len(all_scores) * 100, 1)

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

    # Sort patient summary by severity (severe first)
    level_order = {"severe": 0, "moderate": 1, "mild": 2, "normal": 3}
    patient_summary.sort(key=lambda x: level_order.get(x["level"], 4))

    return {
        "total_assessments": len(rows),
        "unique_patients": len(patients),
        "avg_score": avg_score,
        "excessive_sleepiness_rate_pct": excessive_sleepiness_rate,
        "severity_distribution": dict(severity_dist),
        "active_alerts": alerts[:10],
        "patient_summary": patient_summary,
    }


def breakdown():
    """Per-item mean scores, monthly trend, per-patient history, severity transitions."""
    rows = _rows("SELECT * FROM assessments WHERE instrument='EPWORTH' ORDER BY created_at ASC")
    if not rows:
        return {"message": "No Epworth data"}

    # Per-item mean scores (which situations cause most sleepiness)
    item_totals = {it["id"]: [] for it in SITUATION_ITEMS}
    n = len(rows)

    for r in rows:
        answers = json.loads(r["answers_json"]) if r["answers_json"] else {}
        for it in SITUATION_ITEMS:
            val = answers.get(it["id"])
            if val is not None:
                item_totals[it["id"]].append(val)

    item_means = []
    for it in SITUATION_ITEMS:
        vals = item_totals[it["id"]]
        item_means.append({
            "id": it["id"],
            "label": it["label"],
            "mean_score": round(sum(vals) / len(vals), 2) if vals else 0,
            "max": 3,
            "n_rated": len(vals),
        })

    # Sort by mean score descending to show which situations cause most sleepiness
    item_means_ranked = sorted(item_means, key=lambda x: x["mean_score"], reverse=True)

    # Monthly trend
    date_scores = defaultdict(list)
    for r in rows:
        month = (r["created_at"] or "")[:7]
        if month:
            date_scores[month].append(r["score"])
    trend = [{"month": m, "avg_score": round(sum(v) / len(v), 1), "count": len(v),
              "excessive_pct": round(sum(1 for s in v if s > 10) / len(v) * 100, 1)}
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

    # Severity transitions: for patients with 2+ assessments, show changes
    severity_transitions = []
    for pid, hist in patient_history.items():
        if len(hist) >= 2:
            severity_transitions.append({
                "patient_id": pid,
                "first_level": hist[0]["level"],
                "latest_level": hist[-1]["level"],
                "first_score": hist[0]["score"],
                "latest_score": hist[-1]["score"],
                "assessments": len(hist),
            })

    return {
        "item_means": item_means,
        "item_means_ranked": item_means_ranked,
        "trend": trend,
        "patient_history": dict(patient_history),
        "severity_transitions": severity_transitions,
    }


def definitions():
    """Metric definitions for the Epworth Sleepiness Scale dashboard."""
    return {
        "title": "Epworth Sleepiness Scale (ESS) — Metric Definitions",
        "definitions": [
            {"term": "Epworth Sleepiness Scale (ESS)", "definition": "A validated, self-administered 8-item questionnaire that measures a person's general level of daytime sleepiness. Developed by Murray Johns in 1991 at the Epworth Hospital in Melbourne, Australia."},
            {"term": "Score Range", "definition": "0 (no daytime sleepiness) to 24 (maximum sleepiness). Each of 8 situations is scored 0-3."},
            {"term": "Situation Items (8)", "definition": "Eight everyday situations in which respondents rate their chance of dozing: sitting and reading, watching TV, sitting inactive in public, as a car passenger for an hour, lying down in the afternoon, sitting and talking, sitting quietly after lunch (no alcohol), and in a car stopped in traffic."},
            {"term": "Response Scale (0-3)", "definition": "0 = would never doze, 1 = slight chance of dozing, 2 = moderate chance of dozing, 3 = high chance of dozing."},
            {"term": "Normal (Score 0-10)", "definition": "No excessive daytime sleepiness. Within the normal range for healthy adults. Standard follow-up."},
            {"term": "Mild (Score 11-12)", "definition": "Mild excessive daytime sleepiness. May warrant evaluation of sleep hygiene and consideration of a sleep study if other symptoms are present."},
            {"term": "Moderate (Score 13-15)", "definition": "Moderate excessive daytime sleepiness. Polysomnography recommended. Assess whether antiepileptic medications may be contributing."},
            {"term": "Severe (Score 16-24)", "definition": "Severe excessive daytime sleepiness. Urgent sleep medicine referral indicated. Driving safety counseling essential. Evaluate for obstructive sleep apnea, narcolepsy, or other sleep disorders."},
            {"term": "Excessive Sleepiness Rate", "definition": "Percentage of assessments with ESS score > 10, indicating above-normal daytime sleepiness."},
            {"term": "Per-Item Mean Score", "definition": "Average score for each situation across all assessments. Identifies which situations contribute most to overall sleepiness. Higher means (closer to 3) indicate greater propensity to doze in that context."},
            {"term": "Severity Transitions", "definition": "Change in severity tier between a patient's first and most recent ESS assessment. Tracks whether sleepiness is improving, stable, or worsening over time."},
            {"term": "Epilepsy Context", "definition": "Epilepsy patients have a significantly higher prevalence of sleep disorders and excessive daytime sleepiness (EDS) than the general population. Contributing factors include: (1) antiepileptic drug side effects (e.g., sedation from valproate, carbamazepine, phenobarbital), (2) comorbid obstructive sleep apnea (present in up to 30% of refractory epilepsy patients), (3) nocturnal seizures disrupting sleep architecture, and (4) sleep deprivation as a seizure trigger creating a vicious cycle. The ESS helps screen for EDS so that modifiable causes can be identified and treated."},
        ],
        "reference": "Johns MW. A new method for measuring daytime sleepiness: the Epworth sleepiness scale. Sleep. 1991;14(6):540-545.",
    }
