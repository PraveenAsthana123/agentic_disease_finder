#!/usr/bin/env python3
"""
Barthel Index (ADL) Dashboard
==============================

Analyses REAL Barthel assessment data from clinical.db:

  * assessments table filtered to instrument='BARTHEL'
  * 10-item Activities of Daily Living scale, max score 100
  * Severity levels (higher = better / more independent):
      Independent      80–100
      Minimally dep.   60–79
      Partially dep.   40–59
      Very dependent    0–39

Functions:
  overview()    — KPIs, independence distribution, score histogram, monthly trend
  breakdown()   — per-patient detail, score trajectory, item-level analysis
  definitions() — scale description, item definitions, scoring guide, references
"""

import json
import os
import sqlite3
from collections import Counter, defaultdict
from typing import Any, Dict, List

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
_DB_PATH = os.path.join(_BASE_DIR, "data", "clinical.db")

_ITEMS = [
    "Feeding",
    "Bathing",
    "Grooming",
    "Dressing",
    "Bowel control",
    "Bladder control",
    "Toilet use",
    "Transfers",
    "Mobility",
    "Stairs",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _conn():
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    if not os.path.exists(_DB_PATH):
        return []
    conn = _conn()
    try:
        return [dict(r) for r in conn.execute(query, params).fetchall()]
    finally:
        conn.close()


def _independence_label(score) -> str:
    if score is None:
        return "Unknown"
    if score >= 80:
        return "Independent"
    if score >= 60:
        return "Minimally Dependent"
    if score >= 40:
        return "Partially Dependent"
    return "Very Dependent"


def _independence_level(score) -> str:
    if score is None:
        return "unknown"
    if score >= 80:
        return "normal"
    if score >= 60:
        return "mild"
    if score >= 40:
        return "moderate"
    return "severe"


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------

def overview(patient_id: str = None) -> Dict[str, Any]:
    """KPIs, independence distribution, score histogram, monthly trend."""
    params = ("BARTHEL",)
    where = "instrument = ?"
    if patient_id:
        where += " AND patient_id = ?"
        params = ("BARTHEL", patient_id)

    rows = _rows(
        f"SELECT * FROM assessments WHERE {where} ORDER BY created_at DESC",
        params,
    )

    if not rows:
        return {
            "total_assessments": 0,
            "unique_patients": 0,
            "message": "No Barthel ADL data found",
        }

    scores = [r["score"] for r in rows if r["score"] is not None]
    patients = {r["patient_id"] for r in rows}
    indep_dist = Counter(_independence_label(r["score"]) for r in rows)

    # Score histogram (bins of width 10)
    histogram = defaultdict(int)
    for s in scores:
        bucket = int(s // 10) * 10
        label = f"{bucket}–{bucket + 9}"
        histogram[label] += 1
    hist_list = [{"bin": k, "count": histogram[k]}
                 for k in sorted(histogram.keys())]

    # Monthly trend
    monthly = defaultdict(lambda: {"count": 0, "score_sum": 0})
    for r in rows:
        mo = (r["created_at"] or "")[:7]
        if mo:
            monthly[mo]["count"] += 1
            monthly[mo]["score_sum"] += r["score"] or 0
    trend_list = [
        {"month": m, "count": monthly[m]["count"],
         "avg_score": round(monthly[m]["score_sum"] / monthly[m]["count"], 1)}
        for m in sorted(monthly.keys())
    ]

    dependent_count = sum(1 for r in rows if (r["score"] or 0) < 60)

    return {
        "total_assessments": len(rows),
        "unique_patients": len(patients),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "independence_distribution": dict(indep_dist),
        "score_histogram": hist_list,
        "monthly_trend": trend_list,
        "dependent_patient_count": dependent_count,
        "independent_patient_count": sum(1 for r in rows if (r["score"] or 0) >= 80),
        "scale": {"min": 0, "max": 100, "higher_is_better": True},
    }


# ---------------------------------------------------------------------------
# breakdown
# ---------------------------------------------------------------------------

def breakdown(patient_id: str = None) -> Dict[str, Any]:
    """Per-patient detail, score trajectory, item-level analysis."""
    params = ("BARTHEL",)
    where = "instrument = ?"
    if patient_id:
        where += " AND patient_id = ?"
        params = ("BARTHEL", patient_id)

    rows = _rows(
        f"SELECT * FROM assessments WHERE {where} ORDER BY created_at DESC",
        params,
    )

    if not rows:
        return {"patient_summary": [], "assessment_log": [], "item_averages": []}

    # Per-patient summary
    by_patient: Dict[str, List] = defaultdict(list)
    for r in rows:
        by_patient[r["patient_id"]].append(r)

    patient_summary = []
    for pid, prows in by_patient.items():
        pscores = [r["score"] for r in prows if r["score"] is not None]
        sorted_rows = sorted(prows, key=lambda x: x["created_at"] or "")
        first_score = sorted_rows[0]["score"] if sorted_rows else None
        last_score = sorted_rows[-1]["score"] if sorted_rows else None
        if first_score is not None and last_score is not None:
            diff = last_score - first_score
            trend = "improving" if diff > 2 else ("worsening" if diff < -2 else "stable")
        else:
            trend = "unknown"
        latest = prows[0]
        patient_summary.append({
            "patient_id": pid,
            "assessments": len(prows),
            "avg_score": round(sum(pscores) / len(pscores), 1) if pscores else None,
            "latest_score": latest["score"],
            "latest_level": _independence_label(latest["score"]),
            "latest_level_key": _independence_level(latest["score"]),
            "latest_date": (latest["created_at"] or "")[:10],
            "trend": trend,
            "interpretation": latest.get("interpretation") or _independence_label(latest["score"]),
        })

    # Assessment log (last 100)
    log = [
        {
            "id": r["id"],
            "patient_id": r["patient_id"],
            "score": r["score"],
            "max_score": r["max_score"],
            "level": _independence_label(r["score"]),
            "interpretation": r.get("interpretation") or _independence_label(r["score"]),
            "examiner": r.get("examiner") or "OT",
            "date": (r["created_at"] or "")[:10],
        }
        for r in rows[:100]
    ]

    # Item-level averages (parsed from answers_json)
    item_totals: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        try:
            answers = json.loads(r["answers_json"] or "{}")
            for i, item in enumerate(_ITEMS, 1):
                key = f"item{i}"
                alt_key = item.lower().replace(" ", "_")
                val = answers.get(key) or answers.get(alt_key)
                if val is not None:
                    item_totals[key].append(float(val))
        except Exception:
            pass

    item_averages = []
    for i, item in enumerate(_ITEMS, 1):
        key = f"item{i}"
        vals = item_totals.get(key, [])
        item_averages.append({
            "item": key,
            "label": item,
            "avg": round(sum(vals) / len(vals), 2) if vals else None,
            "n": len(vals),
        })

    return {
        "patient_summary": sorted(patient_summary, key=lambda x: -(x["avg_score"] or 0)),
        "assessment_log": log,
        "item_averages": item_averages,
    }


# ---------------------------------------------------------------------------
# definitions
# ---------------------------------------------------------------------------

def definitions() -> Dict[str, Any]:
    """Scale description, item definitions, scoring guide, references."""
    return {
        "instrument": "BARTHEL",
        "full_name": "Barthel Index of Activities of Daily Living",
        "abbreviation": "BI / Barthel ADL",
        "purpose": (
            "Measures functional independence in 10 personal care and mobility activities. "
            "Widely used in neurology, stroke rehabilitation, and long-term care settings."
        ),
        "population": "Adults with neurological or physical disability; epilepsy patients with motor/cognitive impact",
        "role": "Occupational Therapist (OT)",
        "scale": {
            "min": 0,
            "max": 100,
            "higher_is_better": True,
            "scoring": "Sum of item scores (0, 5, 10, or 15 per item)",
        },
        "items": [
            {"number": i + 1, "name": item}
            for i, item in enumerate(_ITEMS)
        ],
        "bands": [
            {"min": 80, "max": 100, "label": "Independent", "level": "normal",
             "description": "Patient manages all ADLs without assistance"},
            {"min": 60, "max": 79, "label": "Minimally Dependent", "level": "mild",
             "description": "Requires occasional assistance for some tasks"},
            {"min": 40, "max": 59, "label": "Partially Dependent", "level": "moderate",
             "description": "Needs help with several tasks; semi-independent"},
            {"min": 0, "max": 39, "label": "Very Dependent", "level": "severe",
             "description": "Requires substantial help for most or all ADLs"},
        ],
        "administration": {
            "time": "5–10 minutes",
            "method": "Structured observation or caregiver/patient report",
            "frequency": "Monthly or at each clinical review",
        },
        "epilepsy_relevance": (
            "Epilepsy can impair ADL independence through motor deficits (post-ictal weakness, Todd's paresis), "
            "cognitive side effects of AEDs, or comorbid intellectual disability. "
            "The Barthel Index tracks functional impact of seizure burden and treatment response."
        ),
        "scoring_guide": [
            {"item": "Feeding", "options": [
                {"score": 0, "label": "Unable"}, {"score": 5, "label": "Needs help"}, {"score": 10, "label": "Independent"}]},
            {"item": "Bathing", "options": [
                {"score": 0, "label": "Dependent"}, {"score": 5, "label": "Independent"}]},
            {"item": "Grooming", "options": [
                {"score": 0, "label": "Needs help"}, {"score": 5, "label": "Independent (face, hair, teeth)"}]},
            {"item": "Dressing", "options": [
                {"score": 0, "label": "Dependent"}, {"score": 5, "label": "Needs help"}, {"score": 10, "label": "Independent"}]},
            {"item": "Bowel control", "options": [
                {"score": 0, "label": "Incontinent"}, {"score": 5, "label": "Occasional accident"}, {"score": 10, "label": "Continent"}]},
            {"item": "Bladder control", "options": [
                {"score": 0, "label": "Incontinent or catheterised"}, {"score": 5, "label": "Occasional accident"}, {"score": 10, "label": "Continent"}]},
            {"item": "Toilet use", "options": [
                {"score": 0, "label": "Dependent"}, {"score": 5, "label": "Needs some help"}, {"score": 10, "label": "Independent"}]},
            {"item": "Transfers (chair↔bed)", "options": [
                {"score": 0, "label": "Unable; no sitting balance"}, {"score": 5, "label": "Major help (2 people)"}, {"score": 10, "label": "Minor help / verbal"}, {"score": 15, "label": "Independent"}]},
            {"item": "Mobility", "options": [
                {"score": 0, "label": "Immobile"}, {"score": 5, "label": "Wheelchair independent"}, {"score": 10, "label": "Walks with help"}, {"score": 15, "label": "Independent ≥50m"}]},
            {"item": "Stairs", "options": [
                {"score": 0, "label": "Unable"}, {"score": 5, "label": "Needs help"}, {"score": 10, "label": "Independent"}]},
        ],
        "references": [
            "Mahoney FI, Barthel DW (1965). Functional evaluation: the Barthel method. Md State Med J. 14:61–5.",
            "Wade DT, Collin C (1988). The Barthel ADL Index: a standard measure of physical disability? Int Disabil Studies. 10(2):64–7.",
            "Sulter G, et al. (1999). Use of the Barthel Index and Modified Rankin Scale in Acute Stroke Trials. Stroke. 30(8):1538–41.",
        ],
    }
