"""Seizure Severity Dashboard — Liverpool Seizure Severity Scale (LSSS) analytics.

All data from REAL LSSS assessments in data/clinical.db (assessments table, instrument='LSSS').
"""
import sqlite3, json, os
from pathlib import Path
from collections import Counter, defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"

LSSS_DOMAINS = {
    "perception_control": {"label": "Perception of Control", "items": ["item1","item2","item3"], "max": 12},
    "ictal": {"label": "Ictal (During Seizure)", "items": [f"item{i}" for i in range(4,15)], "max": 44},
    "postictal": {"label": "Post-Ictal (After Seizure)", "items": [f"item{i}" for i in range(15,21)], "max": 24},
}

LSSS_ITEM_LABELS = [
    "Warning before seizure", "Ability to fight off seizure", "Control during seizure",
    "Loss of consciousness", "Severity of body movements", "Incontinence during seizure",
    "Biting tongue or cheek", "Injury during seizure", "Duration of seizure",
    "Difficulty breathing", "Severity of confusion after", "Falling during seizure",
    "Automatisms during seizure", "Vocalization during seizure",
    "Duration of recovery", "Headache after seizure", "Sleepiness after seizure",
    "Muscle soreness after seizure", "Difficulty speaking after", "Weakness after seizure",
]


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def overview():
    """Summary KPIs + severity distribution + per-patient latest scores."""
    rows = _rows("SELECT * FROM assessments WHERE instrument='LSSS' ORDER BY created_at DESC")
    if not rows:
        return {"total_assessments": 0, "patients": 0, "message": "No LSSS data yet"}

    patients = set()
    latest_per_patient = {}
    all_scores = []
    severity_dist = Counter()
    for r in rows:
        patients.add(r["patient_id"])
        all_scores.append(r["score"])
        severity_dist[r["level"]] += 1
        if r["patient_id"] not in latest_per_patient:
            latest_per_patient[r["patient_id"]] = r

    avg_score = round(sum(all_scores) / len(all_scores), 1)
    min_score = min(all_scores)
    max_score_val = max(all_scores)

    patient_summary = []
    for pid, r in sorted(latest_per_patient.items()):
        patient_summary.append({
            "patient_id": pid,
            "latest_score": r["score"],
            "max_score": r["max_score"],
            "interpretation": r["interpretation"],
            "level": r["level"],
            "assessed_at": r["created_at"],
        })

    return {
        "total_assessments": len(rows),
        "unique_patients": len(patients),
        "avg_score": avg_score,
        "min_score": min_score,
        "max_score_observed": max_score_val,
        "severity_distribution": dict(severity_dist),
        "patient_summary": patient_summary,
    }


def breakdown():
    """Domain-level analysis, per-item heatmap, trend over time, per-patient history."""
    rows = _rows("SELECT * FROM assessments WHERE instrument='LSSS' ORDER BY created_at ASC")
    if not rows:
        return {"message": "No LSSS data"}

    # Domain analysis (aggregate)
    domain_totals = defaultdict(list)
    item_totals = defaultdict(list)

    for r in rows:
        answers = json.loads(r["answers_json"]) if r["answers_json"] else {}
        for domain_id, domain_info in LSSS_DOMAINS.items():
            domain_score = sum(answers.get(it, 0) for it in domain_info["items"])
            domain_totals[domain_id].append(domain_score)
        for i in range(20):
            key = f"item{i+1}"
            item_totals[key].append(answers.get(key, 0))

    domain_summary = []
    for domain_id, domain_info in LSSS_DOMAINS.items():
        vals = domain_totals[domain_id]
        domain_summary.append({
            "domain": domain_id,
            "label": domain_info["label"],
            "avg_score": round(sum(vals)/len(vals), 1) if vals else 0,
            "max_possible": domain_info["max"],
            "n": len(vals),
        })

    # Per-item averages (for heatmap)
    item_heatmap = []
    for i in range(20):
        key = f"item{i+1}"
        vals = item_totals[key]
        item_heatmap.append({
            "item": key,
            "label": LSSS_ITEM_LABELS[i],
            "avg": round(sum(vals)/len(vals), 2) if vals else 0,
            "max": 4,
        })

    # Sort by severity (highest avg first)
    item_heatmap.sort(key=lambda x: x["avg"], reverse=True)

    # Trend: average score per assessment date (grouped by month)
    date_scores = defaultdict(list)
    for r in rows:
        month = (r["created_at"] or "")[:7]
        if month:
            date_scores[month].append(r["score"])
    trend = [{"month": m, "avg_score": round(sum(v)/len(v), 1), "count": len(v)}
             for m, v in sorted(date_scores.items())]

    # Per-patient history (all assessments, for individual tracking)
    patient_history = defaultdict(list)
    for r in rows:
        patient_history[r["patient_id"]].append({
            "score": r["score"],
            "interpretation": r["interpretation"],
            "level": r["level"],
            "date": r["created_at"],
        })

    return {
        "domain_summary": domain_summary,
        "item_heatmap": item_heatmap,
        "trend": trend,
        "patient_history": dict(patient_history),
    }


def definitions():
    """Metric definitions for the Seizure Severity dashboard."""
    return {
        "title": "Liverpool Seizure Severity Scale (LSSS) — Metric Definitions",
        "definitions": [
            {"term": "LSSS", "definition": "Liverpool Seizure Severity Scale — a validated 20-item self-report measure of seizure severity for adults with epilepsy."},
            {"term": "Score Range", "definition": "20 (minimum/least severe) to 80 (maximum/most severe). Each of 20 items scored 1-4."},
            {"term": "Perception of Control", "definition": "Domain covering items 1-3: warning, ability to fight off, and control during seizure."},
            {"term": "Ictal", "definition": "Domain covering items 4-14: symptoms during the seizure (LOC, movements, incontinence, injury, etc.)."},
            {"term": "Post-Ictal", "definition": "Domain covering items 15-20: recovery symptoms (headache, sleepiness, soreness, speech difficulty, weakness)."},
            {"term": "Mild (20-35)", "definition": "Seizures are brief, with preserved awareness, minimal motor involvement, and quick recovery."},
            {"term": "Moderate (36-50)", "definition": "Seizures involve some LOC, moderate motor symptoms, and recovery within an hour."},
            {"term": "Severe (51-65)", "definition": "Seizures involve LOC, significant motor/autonomic symptoms, injury risk, and prolonged recovery."},
            {"term": "Critical (66-80)", "definition": "Seizures are maximally severe: prolonged LOC, major motor activity, injuries, and slow recovery."},
            {"term": "Item Heatmap", "definition": "Per-item average scores across all patients, ranked by severity. Identifies which seizure features are most problematic."},
            {"term": "Trend", "definition": "Monthly average LSSS scores across all assessments. Tracks whether seizure severity is improving or worsening over time."},
        ],
    }
