"""Mini-Mental State Examination (MMSE) Dashboard — cognitive screening analytics.

All data from REAL MMSE assessments in data/clinical.db (assessments table, instrument='MMSE').

The MMSE is a widely-used, 30-point clinician-administered cognitive screening tool
covering orientation, registration, attention, recall, and language.

Structure:
  7 domain items (max points in parentheses):
    item1: Orientation to time     (max 5)
    item2: Orientation to place    (max 5)
    item3: Registration            (max 3)
    item4: Attention/Calculation   (max 5)
    item5: Recall                  (max 3)
    item6: Language                (max 8)
    item7: Copying                 (max 1)
  Total score range: 0–30 (higher = better cognition)

Severity bands:
  24–30 → Normal cognition (no significant impairment)
  18–23 → Mild impairment (further evaluation warranted)
  10–17 → Moderate impairment (significant deficits)
   0– 9 → Severe impairment (dependent care)

Reference:
  Folstein MF, Folstein SE, McHugh PR. "Mini-mental state": a practical method for
  grading the cognitive state of patients for the clinician. J Psychiatr Res.
  1975;12(3):189-198.

Epilepsy context:
  Cognitive impairment is present in 20–50% of people with epilepsy. Factors
  include seizure frequency, AED side effects, underlying aetiology, and seizure
  duration. The MMSE helps track cognitive trajectory under AED changes.

Author: Research Team
"""
import sqlite3
import json
from pathlib import Path
from collections import Counter, defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"

DOMAIN_ITEMS = [
    {"id": "item1", "label": "Orientation to Time",     "description": "Year, season, date, day, month", "max": 5},
    {"id": "item2", "label": "Orientation to Place",    "description": "State/country, county, town/city, hospital, floor", "max": 5},
    {"id": "item3", "label": "Registration",            "description": "Name 3 objects; patient repeats (1 point each)", "max": 3},
    {"id": "item4", "label": "Attention & Calculation", "description": "Serial 7s (5 subtractions) or WORLD spelled backwards", "max": 5},
    {"id": "item5", "label": "Recall",                  "description": "Ask for the 3 objects registered earlier", "max": 3},
    {"id": "item6", "label": "Language",                "description": "Name 2 items, repeat phrase, 3-step command, read & obey, write sentence, copy design", "max": 8},
    {"id": "item7", "label": "Copying",                 "description": "Copy intersecting pentagons", "max": 1},
]

SEVERITY_BANDS = [
    {"min": 24, "max": 30, "label": "Normal",   "color": "#22c55e", "action": "No significant cognitive impairment; routine follow-up."},
    {"min": 18, "max": 23, "label": "Mild",     "color": "#eab308", "action": "Mild cognitive impairment; further neuropsychological evaluation recommended."},
    {"min": 10, "max": 17, "label": "Moderate", "color": "#f97316", "action": "Moderate impairment; evaluate AED regimen, consider neuroimaging and neuropsychology referral."},
    {"min":  0, "max":  9, "label": "Severe",   "color": "#ef4444", "action": "Severe impairment; dependent care assessment needed, urgent multidisciplinary review."},
]


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def _severity(score):
    for b in SEVERITY_BANDS:
        if b["min"] <= score <= b["max"]:
            return b["label"].lower()
    return "severe"


def overview():
    """MMSE summary KPIs, severity distribution, per-patient latest scores, alerts."""
    rows = _rows("SELECT * FROM assessments WHERE instrument='MMSE' ORDER BY created_at DESC")
    if not rows:
        return {"total_assessments": 0, "patients": 0, "message": "No MMSE data yet"}

    patients = set()
    latest_per_patient = {}
    all_scores = []
    severity_dist = Counter()
    alerts = []

    for r in rows:
        patients.add(r["patient_id"])
        all_scores.append(r["score"])
        lv = r["level"] or _severity(r["score"])
        severity_dist[lv] += 1
        if r["patient_id"] not in latest_per_patient:
            latest_per_patient[r["patient_id"]] = r
        if r["alert"]:
            alerts.append({
                "patient_id": r["patient_id"],
                "alert": r["alert"],
                "score": r["score"],
                "level": lv,
                "date": r["created_at"],
            })

    avg_score = round(sum(all_scores) / len(all_scores), 1)
    impaired_rate = round(sum(1 for s in all_scores if s < 24) / len(all_scores) * 100, 1)

    patient_summary = []
    for pid, r in sorted(latest_per_patient.items()):
        lv = r["level"] or _severity(r["score"])
        patient_summary.append({
            "patient_id": pid,
            "latest_score": r["score"],
            "max_score": r["max_score"],
            "interpretation": r["interpretation"],
            "level": lv,
            "alert": r["alert"],
            "assessed_at": r["created_at"],
        })

    level_order = {"severe": 0, "moderate": 1, "mild": 2, "normal": 3}
    patient_summary.sort(key=lambda x: level_order.get(x["level"], 4))

    return {
        "total_assessments": len(rows),
        "unique_patients": len(patients),
        "avg_score": avg_score,
        "impaired_rate_pct": impaired_rate,
        "severity_distribution": dict(severity_dist),
        "active_alerts": alerts[:10],
        "patient_summary": patient_summary,
        "domain_items": DOMAIN_ITEMS,
        "severity_bands": SEVERITY_BANDS,
    }


def breakdown():
    """Per-domain mean scores, monthly trend, per-patient history, severity transitions."""
    rows = _rows("SELECT * FROM assessments WHERE instrument='MMSE' ORDER BY created_at ASC")
    if not rows:
        return {"message": "No MMSE data"}

    # Per-domain mean scores
    domain_totals = {it["id"]: [] for it in DOMAIN_ITEMS}
    for r in rows:
        answers = json.loads(r["answers_json"]) if r["answers_json"] else {}
        for it in DOMAIN_ITEMS:
            val = answers.get(it["id"])
            if val is not None:
                domain_totals[it["id"]].append(val)

    domain_means = []
    for it in DOMAIN_ITEMS:
        vals = domain_totals[it["id"]]
        mean = round(sum(vals) / len(vals), 2) if vals else 0
        pct_max = round(mean / it["max"] * 100, 1) if it["max"] else 0
        domain_means.append({
            "id": it["id"],
            "label": it["label"],
            "max": it["max"],
            "mean_score": mean,
            "pct_of_max": pct_max,
            "n_rated": len(vals),
        })

    # Rank domains by pct_of_max ascending (worst performance first)
    domain_worst = sorted(domain_means, key=lambda x: x["pct_of_max"])

    # Monthly trend
    date_scores = defaultdict(list)
    for r in rows:
        month = (r["created_at"] or "")[:7]
        if month:
            date_scores[month].append(r["score"])
    trend = [
        {
            "month": m,
            "avg_score": round(sum(v) / len(v), 1),
            "count": len(v),
            "impaired_pct": round(sum(1 for s in v if s < 24) / len(v) * 100, 1),
        }
        for m, v in sorted(date_scores.items())
    ]

    # Per-patient history
    patient_history = defaultdict(list)
    for r in rows:
        lv = r["level"] or _severity(r["score"])
        patient_history[r["patient_id"]].append({
            "score": r["score"],
            "interpretation": r["interpretation"],
            "level": lv,
            "date": r["created_at"],
        })

    # Severity transitions
    severity_transitions = []
    for pid, hist in patient_history.items():
        if len(hist) >= 2:
            severity_transitions.append({
                "patient_id": pid,
                "first_score": hist[0]["score"],
                "first_level": hist[0]["level"],
                "latest_score": hist[-1]["score"],
                "latest_level": hist[-1]["level"],
                "change": round(hist[-1]["score"] - hist[0]["score"], 1),
                "assessments": len(hist),
            })

    return {
        "domain_means": domain_means,
        "domain_worst": domain_worst,
        "trend": trend,
        "patient_history": dict(patient_history),
        "severity_transitions": severity_transitions,
    }


def definitions():
    """Metric definitions for the MMSE dashboard."""
    return {
        "title": "Mini-Mental State Examination (MMSE) — Metric Definitions",
        "reference": "Folstein MF, Folstein SE, McHugh PR. 'Mini-mental state': a practical method for grading the cognitive state of patients for the clinician. J Psychiatr Res. 1975;12(3):189-198.",
        "severity_bands": SEVERITY_BANDS,
        "domain_items": DOMAIN_ITEMS,
        "definitions": [
            {
                "term": "Mini-Mental State Examination (MMSE)",
                "definition": "A 30-point clinician-administered screening tool for cognitive impairment. Developed by Folstein et al. in 1975, it assesses orientation, registration, attention/calculation, recall, and language. Widely used in epilepsy clinics to monitor AED-related cognitive effects.",
            },
            {
                "term": "Score Range",
                "definition": "0 (most severe impairment) to 30 (intact cognition). Scored by adding points across 7 domain clusters. Education-adjusted norms may apply: patients with ≤8 years schooling may score lower without true impairment.",
            },
            {
                "term": "Normal (24–30)",
                "definition": "No clinically significant cognitive impairment. Routine monitoring; rescreen if AED is changed, seizure frequency increases, or clinical concern arises.",
            },
            {
                "term": "Mild Impairment (18–23)",
                "definition": "Mild cognitive impairment. Consider full neuropsychological battery, brain MRI review, AED rationalization, and caregiver interview. May represent early dementia, post-ictal state, or medication effect.",
            },
            {
                "term": "Moderate Impairment (10–17)",
                "definition": "Significant cognitive deficits impacting activities of daily living. Multidisciplinary review required. Evaluate AED burden, seizure control, and structural/metabolic causes.",
            },
            {
                "term": "Severe Impairment (0–9)",
                "definition": "Profound cognitive impairment; patient likely requires supervised care. Urgent investigation for acute causes (post-ictal encephalopathy, non-convulsive status, metabolic derangement, medication toxicity).",
            },
            {
                "term": "Orientation to Time (max 5)",
                "definition": "Year, season, date, day of week, month. Temporal disorientation is an early and sensitive marker of cognitive decline. Epilepsy patients may show post-ictal temporal disorientation that resolves.",
            },
            {
                "term": "Orientation to Place (max 5)",
                "definition": "Country/state, county, town/city, hospital, ward/floor. Spatial disorientation at this level suggests moderate–severe impairment beyond typical post-ictal effects.",
            },
            {
                "term": "Registration (max 3)",
                "definition": "Examiner names 3 objects; patient immediately repeats. Scores 1 per object. Tests verbal encoding and immediate recall. Near-ceiling in most patients; low scores suggest inattention or aphasia.",
            },
            {
                "term": "Attention & Calculation (max 5)",
                "definition": "Serial 7 subtractions from 100 (or WORLD spelled backwards). Most sensitive domain for detecting AED-related cognitive dulling, particularly with benzodiazepines, phenobarbital, and topiramate.",
            },
            {
                "term": "Recall (max 3)",
                "definition": "Recall the 3 objects registered earlier. Tests delayed memory consolidation. Low scores indicate hippocampal dysfunction, particularly relevant in temporal lobe epilepsy.",
            },
            {
                "term": "Language (max 8)",
                "definition": "Naming (2 pts), phrase repetition (1 pt), 3-step command (3 pts), read & obey (1 pt), write a sentence (1 pt). Combines expressive and receptive language assessment.",
            },
            {
                "term": "Copying (max 1)",
                "definition": "Copy two intersecting pentagons. Tests visuospatial and constructional praxis. Impaired in parietal lobe epilepsy and posterior cortical dysfunction.",
            },
            {
                "term": "Epilepsy Context",
                "definition": "20–50% of people with epilepsy experience cognitive impairment. The MMSE is used to: (1) establish baseline before AED initiation, (2) monitor AED titration effects, (3) detect post-ictal cognitive effects, (4) screen for comorbid dementia in older patients, and (5) guide rehabilitation planning.",
            },
            {
                "term": "Impaired Rate",
                "definition": "Percentage of assessments with MMSE < 24, indicating at least mild cognitive impairment. Used to track the overall cognitive burden across the epilepsy patient cohort.",
            },
        ],
    }
