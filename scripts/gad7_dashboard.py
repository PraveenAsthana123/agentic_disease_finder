"""GAD-7 Dashboard — Generalized Anxiety Disorder-7 analytics.

All data from REAL GAD-7 assessments in data/clinical.db (assessments table, instrument='GAD7').

The GAD-7 is a validated 7-item self-report measure for screening and monitoring
generalized anxiety disorder. Each item scores 0-3
(0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day).

Items:
  1. Feeling nervous, anxious, or on edge
  2. Not being able to stop or control worrying
  3. Worrying too much about different things
  4. Trouble relaxing
  5. Being so restless that it's hard to sit still
  6. Becoming easily annoyed or irritable
  7. Feeling afraid as if something awful might happen

Score range: 0-21
Severity tiers:
  0-4   Minimal
  5-9   Mild
  10-14 Moderate
  15-21 Severe

Clinical note: GAD-7 >= 10 has 89% sensitivity and 82% specificity for
generalized anxiety disorder (Spitzer et al., 2006). Anxiety disorders occur
in 10-25% of people with epilepsy.

Reference:
  Spitzer RL, Kroenke K, Williams JBW, Lowe B. A brief measure for assessing
  generalized anxiety disorder: the GAD-7. Arch Intern Med. 2006;166(10):1092-1097.

Author: Research Team
"""
import sqlite3
import json
from pathlib import Path
from collections import Counter, defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"

GAD7_ITEMS = [
    {"id": "item1", "label": "Nervous / anxious", "description": "Feeling nervous, anxious, or on edge"},
    {"id": "item2", "label": "Uncontrollable worry", "description": "Not being able to stop or control worrying"},
    {"id": "item3", "label": "Excessive worry", "description": "Worrying too much about different things"},
    {"id": "item4", "label": "Trouble relaxing", "description": "Trouble relaxing"},
    {"id": "item5", "label": "Restlessness", "description": "Being so restless that it is hard to sit still"},
    {"id": "item6", "label": "Irritability", "description": "Becoming easily annoyed or irritable"},
    {"id": "item7", "label": "Feeling afraid", "description": "Feeling afraid, as if something awful might happen"},
]

SEVERITY_TIERS = [
    {"range": [0, 4], "label": "Minimal", "color": "#22c55e", "action": "No treatment indicated; monitor if epilepsy comorbidity"},
    {"range": [5, 9], "label": "Mild", "color": "#84cc16", "action": "Watchful waiting; repeat GAD-7 at follow-up"},
    {"range": [10, 14], "label": "Moderate", "color": "#eab308", "action": "Consider counselling, pharmacotherapy, or both"},
    {"range": [15, 21], "label": "Severe", "color": "#ef4444", "action": "Active treatment with pharmacotherapy and/or psychotherapy; consider psychiatry referral"},
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
    return "severe"


def overview():
    rows = _rows("SELECT * FROM assessments WHERE instrument='GAD7' ORDER BY created_at DESC")
    if not rows:
        return {
            "total_assessments": 0, "unique_patients": 0,
            "avg_score": 0, "pct_moderate_plus": 0,
            "severity_distribution": {}, "patient_summary": [],
            "active_alerts": [],
        }

    scores = [r["score"] for r in rows]
    pids = list({r["patient_id"] for r in rows})

    # Severity distribution
    sev_counts = Counter()
    for r in rows:
        sev_counts[_severity(r["score"])] += 1

    # % moderate or above (score >= 10)
    mod_plus = sum(1 for s in scores if s >= 10)
    pct_mod = round(100 * mod_plus / len(scores), 1) if scores else 0

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

    # Active alerts: score >= 10
    alerts = []
    for pid, r in latest.items():
        if r["score"] >= 10:
            alerts.append({
                "patient_id": pid,
                "alert": f"Score {int(r['score'])}/21 — {_severity(r['score'])} anxiety",
                "score": int(r["score"]),
                "severity": _severity(r["score"]),
            })

    return {
        "total_assessments": len(rows),
        "unique_patients": len(pids),
        "avg_score": round(sum(scores) / len(scores), 1),
        "pct_moderate_plus": pct_mod,
        "severity_distribution": dict(sev_counts),
        "patient_summary": patient_summary,
        "active_alerts": alerts,
    }


def breakdown():
    rows = _rows("SELECT * FROM assessments WHERE instrument='GAD7' ORDER BY created_at ASC")
    if not rows:
        return {
            "item_endorsement": [], "severity_transitions": [],
            "trend": [], "patient_history": {},
        }

    # Per-item endorsement rates (score > 0 for that item)
    item_totals = {it["id"]: 0 for it in GAD7_ITEMS}
    item_severe = {it["id"]: 0 for it in GAD7_ITEMS}  # score >= 2
    parsed = 0
    for r in rows:
        try:
            ans = json.loads(r["answers_json"]) if r["answers_json"] else {}
            parsed += 1
            for it in GAD7_ITEMS:
                val = ans.get(it["id"], 0)
                if val > 0:
                    item_totals[it["id"]] += 1
                if val >= 2:
                    item_severe[it["id"]] += 1
        except Exception:
            pass

    item_endorsement = []
    for it in GAD7_ITEMS:
        any_pct = round(100 * item_totals[it["id"]] / parsed, 1) if parsed else 0
        sev_pct = round(100 * item_severe[it["id"]] / parsed, 1) if parsed else 0
        item_endorsement.append({
            "id": it["id"],
            "label": it["label"],
            "any_pct": any_pct,
            "frequent_pct": sev_pct,
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
            "pct_moderate_plus": round(100 * sum(1 for v in vals if v >= 10) / len(vals), 1),
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
        "item_endorsement": item_endorsement,
        "severity_transitions": severity_transitions,
        "trend": trend,
        "patient_history": patient_history,
    }


def definitions():
    return {
        "title": "GAD-7 — Generalized Anxiety Disorder-7",
        "reference": "Spitzer RL, Kroenke K, Williams JBW, Lowe B. A brief measure for assessing generalized anxiety disorder: the GAD-7. Arch Intern Med. 2006;166(10):1092-1097.",
        "items": [
            {**it, "scoring": "0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day"}
            for it in GAD7_ITEMS
        ],
        "severity_tiers": SEVERITY_TIERS,
        "clinical_notes": [
            {"term": "Screening threshold", "definition": "GAD-7 >= 10 has 89% sensitivity and 82% specificity for generalized anxiety disorder."},
            {"term": "Epilepsy relevance", "definition": "Anxiety disorders occur in 10-25% of people with epilepsy, often under-recognized. The ILAE recommends routine anxiety screening in epilepsy clinics."},
            {"term": "Seizure-anxiety link", "definition": "Pre-ictal anxiety is reported by up to 60% of patients. Fear of seizures (ictal phobia) is distinct from generalized anxiety and may not respond to standard anxiolytics."},
            {"term": "AED effects", "definition": "Some AEDs (levetiracetam, topiramate) may worsen anxiety; others (pregabalin, gabapentin) have anxiolytic properties."},
            {"term": "Treatment response", "definition": "A 5-point decrease in GAD-7 score is considered clinically significant. 50% reduction indicates treatment response."},
            {"term": "Remission", "definition": "GAD-7 < 5 is considered anxiety remission."},
        ],
    }
