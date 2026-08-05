#!/usr/bin/env python3
"""
LSSS (Liverpool Seizure Severity Scale) Dashboard
===================================================

Analyses REAL LSSS assessment data from clinical.db:

  * assessments table filtered to instrument='LSSS'
  * 20-item scale, max score 80 (higher = more severe)
  * Severity levels: mild (<40), moderate (40-54), severe (55-69), critical (>=70)

Functions:
  overview()    — KPIs, severity distribution, score histogram, monthly trend
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


def _severity_label(score: float) -> str:
    if score is None:
        return "Unknown"
    if score < 40:
        return "Mild"
    if score < 55:
        return "Moderate"
    if score < 70:
        return "Severe"
    return "Critical"


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------

def overview(patient_id: str = None) -> Dict[str, Any]:
    """KPIs, severity distribution, score histogram, monthly trend."""
    params = ("LSSS",)
    where = "instrument = ?"
    if patient_id:
        where += " AND patient_id = ?"
        params = ("LSSS", patient_id)

    rows = _rows(
        f"SELECT * FROM assessments WHERE {where} ORDER BY created_at DESC",
        params,
    )

    if not rows:
        return {
            "total_assessments": 0,
            "unique_patients": 0,
            "message": "No LSSS data found",
        }

    scores = [r["score"] for r in rows if r["score"] is not None]
    patients = {r["patient_id"] for r in rows}
    severity_dist = Counter(_severity_label(r["score"]) for r in rows)

    # Score histogram (bins of width 10)
    histogram = defaultdict(int)
    for s in scores:
        bucket = int(s // 10) * 10
        label = f"{bucket}–{bucket+9}"
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
    monthly_trend = [
        {
            "month": m,
            "assessments": monthly[m]["count"],
            "avg_score": round(monthly[m]["score_sum"] / monthly[m]["count"], 1),
        }
        for m in sorted(monthly.keys())
    ]

    # High-risk patients (score >= 55 = Severe+)
    high_risk = sorted(
        {
            r["patient_id"]
            for r in rows
            if r["score"] is not None and r["score"] >= 55
        }
    )

    return {
        "total_assessments": len(rows),
        "unique_patients": len(patients),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "high_risk_patient_count": len(high_risk),
        "severity_distribution": dict(severity_dist),
        "score_histogram": hist_list,
        "monthly_trend": monthly_trend,
    }


# ---------------------------------------------------------------------------
# breakdown
# ---------------------------------------------------------------------------

def breakdown(patient_id: str = None) -> Dict[str, Any]:
    """Per-patient trajectory, score table, item-level analysis."""
    params = ("LSSS",)
    where = "instrument = ?"
    if patient_id:
        where += " AND patient_id = ?"
        params = ("LSSS", patient_id)

    rows = _rows(
        f"SELECT * FROM assessments WHERE {where} ORDER BY patient_id, created_at",
        params,
    )

    if not rows:
        return {"patients": [], "assessments": [], "message": "No LSSS data found"}

    # Per-patient summary
    per_patient: Dict[str, List] = defaultdict(list)
    for r in rows:
        per_patient[r["patient_id"]].append(r)

    patient_summary = []
    for pid, pts in sorted(per_patient.items()):
        scores = [p["score"] for p in pts if p["score"] is not None]
        if not scores:
            continue
        first_score = pts[0]["score"]
        last_score = pts[-1]["score"]
        trend = "stable"
        if len(scores) > 1:
            delta = last_score - first_score
            if delta >= 5:
                trend = "worsening"
            elif delta <= -5:
                trend = "improving"
        patient_summary.append({
            "patient_id": pid,
            "assessments": len(pts),
            "avg_score": round(sum(scores) / len(scores), 1),
            "min_score": min(scores),
            "max_score": max(scores),
            "latest_score": last_score,
            "latest_level": _severity_label(last_score),
            "trend": trend,
            "first_date": pts[0]["created_at"][:10] if pts[0]["created_at"] else None,
            "latest_date": pts[-1]["created_at"][:10] if pts[-1]["created_at"] else None,
        })

    # Sort by severity (worst first)
    severity_order = {"Critical": 0, "Severe": 1, "Moderate": 2, "Mild": 3, "Unknown": 4}
    patient_summary.sort(key=lambda p: (severity_order.get(p["latest_level"], 4), -p["avg_score"]))

    # Assessment log (recent 60)
    assessment_log = [
        {
            "id": r["id"],
            "patient_id": r["patient_id"],
            "score": r["score"],
            "max_score": r["max_score"],
            "level": r["level"] or _severity_label(r["score"]).lower(),
            "interpretation": r["interpretation"],
            "examiner": r["examiner"],
            "date": (r["created_at"] or "")[:10],
        }
        for r in sorted(rows, key=lambda x: x["created_at"] or "", reverse=True)[:60]
    ]

    # Item-level analysis: average per item across all assessments
    item_totals: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        try:
            answers = json.loads(r["answers_json"] or "{}")
            for k, v in answers.items():
                item_totals[k].append(float(v))
        except (json.JSONDecodeError, TypeError):
            pass
    item_averages = [
        {"item": k, "avg_score": round(sum(v) / len(v), 2), "responses": len(v)}
        for k, v in sorted(item_totals.items())
    ]

    return {
        "patient_summary": patient_summary,
        "assessment_log": assessment_log,
        "item_averages": item_averages,
    }


# ---------------------------------------------------------------------------
# definitions
# ---------------------------------------------------------------------------

def definitions() -> Dict[str, Any]:
    """Scale description, item list, scoring guide, references."""
    return {
        "title": "Liverpool Seizure Severity Scale (LSSS)",
        "description": (
            "The LSSS is a validated 20-item patient-reported outcome measure (PROM) "
            "designed to quantify the severity of epileptic seizures. It captures the "
            "ictal and post-ictal experience across dimensions including awareness, "
            "injury risk, recovery time, and impact on daily activities. "
            "Scoring range: 20–80 (higher = more severe)."
        ),
        "score_range": {"min": 20, "max": 80, "higher_is_worse": True},
        "severity_thresholds": [
            {"level": "Mild", "min": 20, "max": 39, "description": "Seizures with minimal impact on daily life; brief post-ictal recovery."},
            {"level": "Moderate", "min": 40, "max": 54, "description": "Moderate impact; noticeable post-ictal symptoms, some activity limitation."},
            {"level": "Severe", "min": 55, "max": 69, "description": "Significant impairment; prolonged post-ictal state, frequent injuries or falls."},
            {"level": "Critical", "min": 70, "max": 80, "description": "Severe burden; life-limiting seizures, repeated ER visits, high injury risk."},
        ],
        "subscales": [
            {
                "name": "Ictal",
                "items": ["item1", "item2", "item3", "item4", "item5", "item6", "item7", "item8", "item9", "item10"],
                "description": "Captures seizure characteristics during the ictus: duration, awareness, motor signs, injury, tongue biting.",
            },
            {
                "name": "Post-ictal",
                "items": ["item11", "item12", "item13", "item14", "item15", "item16", "item17", "item18", "item19", "item20"],
                "description": "Post-ictal burden: confusion, headache, fatigue, embarrassment, memory loss, recovery time.",
            },
        ],
        "item_labels": {
            "item1": "Seizure duration",
            "item2": "Loss of awareness",
            "item3": "Convulsive movements",
            "item4": "Tongue biting",
            "item5": "Injury during seizure",
            "item6": "Incontinence",
            "item7": "Breathing difficulty",
            "item8": "Fall or collapse",
            "item9": "Facial colour change",
            "item10": "Aura present",
            "item11": "Confusion post-ictal",
            "item12": "Headache post-ictal",
            "item13": "Fatigue/exhaustion",
            "item14": "Memory loss post-ictal",
            "item15": "Embarrassment",
            "item16": "Low mood",
            "item17": "Recovery time",
            "item18": "Impact on activities",
            "item19": "Fear of next seizure",
            "item20": "Social withdrawal",
        },
        "clinical_use": [
            "Baseline seizure severity at diagnosis",
            "Treatment response monitoring",
            "AED titration decision support",
            "Pre/post surgical evaluation",
            "Quality-of-life research endpoint",
        ],
        "references": [
            "Baker GA et al. (1991). Development of a seizure severity scale as an outcome measure in epilepsy. Epilepsy Research, 8(3), 245-251.",
            "Jacoby A et al. (1993). Reliability and validity of a newly developed instrument for measuring seizure severity. Epilepsia, 34(5), 905-912.",
        ],
        "data_source": "assessments table in data/clinical.db — instrument='LSSS'.",
    }


if __name__ == "__main__":
    import json
    print("=== LSSS Overview ===")
    print(json.dumps(overview(), indent=2))
