#!/usr/bin/env python3
"""
NDDI-E (Neurological Disorders Depression Inventory for Epilepsy) Dashboard
===========================================================================

All data from REAL NDDIE assessments in data/clinical.db
(assessments table, instrument='NDDIE').

The NDDI-E is a validated, epilepsy-specific 6-item depression screening tool
designed to detect major depressive disorder in people with epilepsy while
avoiding overlap with ictal/AED-related symptoms.

Scale structure:
  6 items, each rated 1–4:
    1 = Never | 2 = Sometimes | 3 = Often | 4 = Always

  Items:
    item1 — Everything I do is a struggle
    item2 — Nothing good will ever happen to me
    item3 — I feel guilty
    item4 — I would be better off dead       ← suicidality flag
    item5 — I feel frustrated
    item6 — I have difficulty finding pleasure

  Score range: 6 (no symptoms) to 24 (maximum burden)
  Severity tiers:
    ≤12   Normal     — Depression unlikely
    13–14 Borderline — Monitor closely
    ≥15   Screen +   — Likely major depression; refer to neuropsychiatry

Special alerts:
  • NDDI-E ≥15 — depression screen positive
  • Item 4 ≥3   — suicidality item positive (escalate immediately)
  • Both        — urgent psychiatric referral

Reference:
  Gilliam FG, Barry JJ, Hermann BP, Meador KJ, Vahle V, Kanner AM.
  Rapid detection of major depression in epilepsy: a multicentre study.
  Lancet Neurol. 2006;5(5):399-405.

API functions:
  overview()    — KPIs, severity distribution, score histogram, monthly trend,
                  item-level endorsement rates, suicidality flag summary
  breakdown()   — per-patient detail, score trajectory, item averages, log
  definitions() — scale description, items, thresholds, references
"""

import json
import os
import sqlite3
from collections import Counter, defaultdict
from typing import Any, Dict, List

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
_DB_PATH = os.path.join(_BASE_DIR, "data", "clinical.db")

# ---------------------------------------------------------------------------
# NDDI-E meta
# ---------------------------------------------------------------------------

NDDIE_ITEMS = [
    {"id": "item1", "label": "Everything I do is a struggle",         "subscale": "functional"},
    {"id": "item2", "label": "Nothing good will ever happen to me",   "subscale": "hopelessness"},
    {"id": "item3", "label": "I feel guilty",                          "subscale": "cognitive"},
    {"id": "item4", "label": "I would be better off dead",             "subscale": "suicidality"},
    {"id": "item5", "label": "I feel frustrated",                      "subscale": "emotional"},
    {"id": "item6", "label": "I have difficulty finding pleasure",     "subscale": "anhedonia"},
]

RESPONSE_LABELS = {1: "Never", 2: "Sometimes", 3: "Often", 4: "Always"}

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
    if score <= 12:
        return "Normal"
    if score <= 14:
        return "Borderline"
    return "Screen+"


def _severity_color(label: str) -> str:
    return {
        "Normal": "success",
        "Borderline": "warning",
        "Screen+": "danger",
    }.get(label, "secondary")


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------

def overview(patient_id: str = None) -> Dict[str, Any]:
    """KPIs, severity distribution, score histogram, monthly trend,
    item endorsement rates, and suicidality flag summary."""

    params: tuple = ("NDDIE",)
    where = "instrument = ?"
    if patient_id:
        where += " AND patient_id = ?"
        params = ("NDDIE", patient_id)

    rows = _rows(
        f"SELECT * FROM assessments WHERE {where} ORDER BY created_at DESC",
        params,
    )

    if not rows:
        return {
            "total_assessments": 0,
            "unique_patients": 0,
            "message": "No NDDI-E data found",
        }

    scores = [r["score"] for r in rows if r["score"] is not None]
    patients = {r["patient_id"] for r in rows}
    severity_dist = Counter(_severity_label(r["score"]) for r in rows)

    # Screen positive: score ≥15
    screen_positive_count = sum(1 for s in scores if s >= 15)
    screen_positive_pct = round(100 * screen_positive_count / len(scores), 1) if scores else 0

    # Suicidality flags (item4 ≥3)
    suicidality_flags = 0
    for r in rows:
        try:
            ans = json.loads(r["answers_json"] or "{}")
            if ans.get("item4", 0) >= 3:
                suicidality_flags += 1
        except Exception:
            pass

    # Item endorsement rates (avg score per item across all assessments)
    item_totals: Dict[str, float] = defaultdict(float)
    item_counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        try:
            ans = json.loads(r["answers_json"] or "{}")
            for it in NDDIE_ITEMS:
                val = ans.get(it["id"])
                if val is not None:
                    item_totals[it["id"]] += val
                    item_counts[it["id"]] += 1
        except Exception:
            pass
    item_averages = [
        {
            "item": it["id"],
            "label": it["label"],
            "subscale": it["subscale"],
            "avg_score": round(item_totals[it["id"]] / item_counts[it["id"]], 2)
            if item_counts[it["id"]] > 0 else None,
        }
        for it in NDDIE_ITEMS
    ]

    # Score histogram (bins of width 3: 6-8, 9-11, 12-14, 15-17, 18-20, 21-24)
    bins = [(6, 8), (9, 11), (12, 14), (15, 17), (18, 20), (21, 24)]
    histogram = []
    for lo, hi in bins:
        cnt = sum(1 for s in scores if lo <= s <= hi)
        histogram.append({"bin": f"{lo}–{hi}", "count": cnt})

    # Monthly trend
    monthly: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "score_sum": 0})
    for r in rows:
        mo = (r["created_at"] or "")[:7]
        if mo:
            monthly[mo]["count"] += 1
            monthly[mo]["score_sum"] += r["score"] or 0
    monthly_trend = [
        {
            "month": mo,
            "assessments": v["count"],
            "avg_score": round(v["score_sum"] / v["count"], 1),
        }
        for mo, v in sorted(monthly.items())
    ]

    return {
        "total_assessments": len(rows),
        "unique_patients": len(patients),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "screen_positive_count": screen_positive_count,
        "screen_positive_pct": screen_positive_pct,
        "suicidality_flag_count": suicidality_flags,
        "suicidality_flag_pct": round(100 * suicidality_flags / len(rows), 1) if rows else 0,
        "severity_distribution": dict(severity_dist),
        "score_histogram": histogram,
        "monthly_trend": monthly_trend,
        "item_averages": item_averages,
    }


# ---------------------------------------------------------------------------
# breakdown
# ---------------------------------------------------------------------------

def breakdown(patient_id: str = None) -> Dict[str, Any]:
    """Per-patient summary, score trajectory, item-level averages, assessment log."""

    params: tuple = ("NDDIE",)
    where = "instrument = ?"
    if patient_id:
        where += " AND patient_id = ?"
        params = ("NDDIE", patient_id)

    rows = _rows(
        f"SELECT * FROM assessments WHERE {where} ORDER BY patient_id, created_at ASC",
        params,
    )

    # Per-patient summary
    patient_data: Dict[str, Dict] = {}
    for r in rows:
        pid = r["patient_id"]
        if pid not in patient_data:
            patient_data[pid] = {
                "patient_id": pid,
                "assessments": 0,
                "score_sum": 0.0,
                "scores": [],
                "levels": [],
                "suicidality_positive": False,
                "first_date": r["created_at"],
                "latest_date": r["created_at"],
                "latest_score": r["score"],
                "latest_level": _severity_label(r["score"]),
            }
        d = patient_data[pid]
        d["assessments"] += 1
        if r["score"] is not None:
            d["score_sum"] += r["score"]
            d["scores"].append(r["score"])
        d["levels"].append(_severity_label(r["score"]))
        d["latest_date"] = r["created_at"]
        d["latest_score"] = r["score"]
        d["latest_level"] = _severity_label(r["score"])
        # suicidality flag
        try:
            ans = json.loads(r["answers_json"] or "{}")
            if ans.get("item4", 0) >= 3:
                d["suicidality_positive"] = True
        except Exception:
            pass

    patient_summary = []
    for pid, d in sorted(patient_data.items()):
        n = d["assessments"]
        avg = round(d["score_sum"] / n, 1) if n > 0 else None
        scores = d["scores"]
        trend = "stable"
        if len(scores) >= 2:
            if scores[-1] > scores[0]:
                trend = "worsening"
            elif scores[-1] < scores[0]:
                trend = "improving"
        patient_summary.append({
            "patient_id": pid,
            "assessments": n,
            "avg_score": avg,
            "latest_score": d["latest_score"],
            "latest_level": d["latest_level"],
            "suicidality_positive": d["suicidality_positive"],
            "trend": trend,
            "first_date": d["first_date"],
            "latest_date": d["latest_date"],
        })

    # Assessment log (all rows, recent first)
    log_rows = _rows(
        f"SELECT * FROM assessments WHERE {where} ORDER BY created_at DESC LIMIT 200",
        params,
    )
    assessment_log = [
        {
            "id": r["id"],
            "patient_id": r["patient_id"],
            "score": r["score"],
            "max_score": r["max_score"],
            "level": _severity_label(r["score"]),
            "interpretation": r["interpretation"],
            "alert": r["alert"],
            "examiner": r["examiner"],
            "date": r["created_at"],
        }
        for r in log_rows
    ]

    # Item averages (across all)
    item_totals: Dict[str, float] = defaultdict(float)
    item_counts: Dict[str, int] = defaultdict(int)
    item_response_dist: Dict[str, Dict[int, int]] = {it["id"]: defaultdict(int) for it in NDDIE_ITEMS}
    for r in rows:
        try:
            ans = json.loads(r["answers_json"] or "{}")
            for it in NDDIE_ITEMS:
                val = ans.get(it["id"])
                if val is not None:
                    item_totals[it["id"]] += val
                    item_counts[it["id"]] += 1
                    item_response_dist[it["id"]][int(val)] += 1
        except Exception:
            pass

    item_averages = [
        {
            "item": it["id"],
            "label": it["label"],
            "subscale": it["subscale"],
            "avg_score": round(item_totals[it["id"]] / item_counts[it["id"]], 2)
            if item_counts[it["id"]] > 0 else None,
            "response_distribution": {
                RESPONSE_LABELS.get(k, str(k)): v
                for k, v in sorted(item_response_dist[it["id"]].items())
            },
        }
        for it in NDDIE_ITEMS
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
    """Scale description, items, scoring guide, severity thresholds, references."""
    return {
        "title": "NDDI-E — Neurological Disorders Depression Inventory for Epilepsy",
        "description": (
            "A validated 6-item self-report screening tool developed specifically "
            "for detecting major depressive disorder (MDD) in people with epilepsy. "
            "Items were selected to avoid overlap with ictal symptoms and AED side-effects, "
            "making it more accurate in this population than generic depression screens (e.g., PHQ-9)."
        ),
        "developer": "Gilliam FG, Barry JJ, Hermann BP et al., 2006",
        "administration": "Self-report; ~2 minutes",
        "score_range": {"min": 6, "max": 24},
        "interpretation": "Higher score = greater depression burden",
        "cutoff": 15,
        "cutoff_interpretation": "Score ≥15 suggests likely major depression (sensitivity 81%, specificity 90%)",
        "severity_thresholds": [
            {
                "level": "Normal",
                "min": 6,
                "max": 12,
                "description": "Depression unlikely; routine monitoring at follow-up visits.",
            },
            {
                "level": "Borderline",
                "min": 13,
                "max": 14,
                "description": "Subthreshold symptoms; monitor closely, consider structured interview.",
            },
            {
                "level": "Screen+",
                "min": 15,
                "max": 24,
                "description": "Screen positive for likely major depression; refer to neuropsychiatry/psychiatry.",
            },
        ],
        "items": [
            {
                "id": it["id"],
                "number": int(it["id"].replace("item", "")),
                "label": it["label"],
                "subscale": it["subscale"],
                "clinical_note": (
                    "Suicidality flag: score ≥3 on this item triggers immediate escalation protocol."
                    if it["id"] == "item4" else ""
                ),
            }
            for it in NDDIE_ITEMS
        ],
        "response_scale": {
            "1": "Never",
            "2": "Sometimes",
            "3": "Often",
            "4": "Always",
        },
        "suicidality_protocol": (
            "Item 4 ('I would be better off dead') ≥3 (Often/Always) triggers immediate "
            "suicidality assessment protocol regardless of total score."
        ),
        "advantages_over_phq9": [
            "Omits somatic symptoms (sleep, appetite, fatigue) that overlap with AED side-effects",
            "Omits psychomotor symptoms that overlap with post-ictal states",
            "Shorter (6 vs 9 items) — better tolerated during clinical visits",
            "Validated specifically in epilepsy cohorts across multiple centres",
            "High specificity — reduces false positives from epilepsy-related symptoms",
        ],
        "clinical_use": [
            "Annual depression screening in epilepsy clinic patients",
            "Pre/post AED medication change monitoring",
            "Pre-surgical epilepsy evaluation (psychosocial baseline)",
            "Research outcome measure in clinical trials",
            "Telehealth and remote patient-reported outcomes",
        ],
        "regulatory_context": (
            "Recommended by ILAE Commission on Neuropsychiatric Issues (2010). "
            "Endorsed by NICE guideline NG217 (Epilepsies in children, young people and adults, 2022) "
            "as a validated screening tool for comorbid depression in epilepsy."
        ),
        "references": [
            "Gilliam FG, Barry JJ, Hermann BP, Meador KJ, Vahle V, Kanner AM. "
            "Rapid detection of major depression in epilepsy: a multicentre study. "
            "Lancet Neurol. 2006;5(5):399-405.",
            "Kanner AM, Barry JJ, Gilliam F, Hermann B, Meador KJ. "
            "Anxiety disorders, subsyndromic depressive episodes, and major depressive episodes: "
            "do they differ on their impact on the quality of life of patients with epilepsy? "
            "Epilepsia. 2010;51(7):1152-8.",
            "ILAE Commission on Neuropsychiatric Issues. "
            "Neuropsychiatric comorbidities in epilepsy. 2010.",
            "NICE NG217. Epilepsies in children, young people and adults. 2022.",
            "Jones JE, Hermann BP, Barry JJ, Gilliam F, Kanner AM, Meador KJ. "
            "Clinical assessment of Axis I psychiatric morbidity in chronic epilepsy: "
            "a multicenter investigation. J Neuropsychiatry Clin Neurosci. 2005;17(2):172-9.",
        ],
    }
