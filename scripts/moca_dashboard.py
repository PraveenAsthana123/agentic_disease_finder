"""MoCA Dashboard — Montreal Cognitive Assessment analytics.

All data from REAL MoCA assessments in data/clinical.db (assessments table, instrument='MOCA').

The MoCA is a validated 30-point cognitive screening tool for mild cognitive impairment (MCI).
It covers 7 cognitive domains scored across item1–item7:
  item1: Visuospatial / Executive  (max 5)
  item2: Naming                    (max 3)
  item3: Attention                 (max 6)
  item4: Language                  (max 3)
  item5: Abstraction               (max 2)
  item6: Delayed Recall            (max 5)
  item7: Orientation               (max 6)
Total: 30 points (+1 bonus if ≤12 years education; cap at 30)

Severity bands:
  26–30  Normal cognition
  18–25  Mild cognitive impairment (MCI)
  10–17  Moderate impairment
   0–9   Severe impairment

Cutoff: <26 = possible MCI (sensitivity 90%, specificity 87%; Nasreddine et al., 2005).
Epilepsy relevance: 20–50% of drug-resistant epilepsy patients show MCI; temporal lobe
epilepsy disproportionately affects memory (delayed recall domain).

References:
  Nasreddine ZS, et al. The Montreal Cognitive Assessment, MoCA: a brief screening tool
  for mild cognitive impairment. J Am Geriatr Soc. 2005;53(4):695-699.
  Griffith HR, et al. Cognitive functions in temporal lobe epilepsy.
  Neuropsychology Review. 2007;17(4):429-455.

Author: Research Team
"""
import sqlite3
import json
from pathlib import Path
from collections import Counter, defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"

DOMAINS = [
    {"id": "item1", "label": "Visuospatial / Executive", "max": 5},
    {"id": "item2", "label": "Naming",                   "max": 3},
    {"id": "item3", "label": "Attention",                 "max": 6},
    {"id": "item4", "label": "Language",                  "max": 3},
    {"id": "item5", "label": "Abstraction",               "max": 2},
    {"id": "item6", "label": "Delayed Recall",            "max": 5},
    {"id": "item7", "label": "Orientation",               "max": 6},
]

SEVERITY_TIERS = [
    {"range": [26, 30], "label": "Normal",   "color": "#22c55e",
     "action": "No cognitive impairment detected; routine monitoring recommended"},
    {"range": [18, 25], "label": "Mild MCI", "color": "#eab308",
     "action": "Mild cognitive impairment; neuropsychological follow-up, discuss AED cognitive load"},
    {"range": [10, 17], "label": "Moderate", "color": "#f97316",
     "action": "Moderate impairment; comprehensive neuropsychological evaluation, caregiver assessment"},
    {"range": [0, 9],   "label": "Severe",   "color": "#ef4444",
     "action": "Severe impairment; urgent neurological review, capacity assessment, full care plan"},
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
            return t["label"].lower().replace(" ", "_")
    return "severe"


def _severity_label(score):
    for t in SEVERITY_TIERS:
        if t["range"][0] <= score <= t["range"][1]:
            return t["label"]
    return "Severe"


def overview():
    rows = _rows("SELECT * FROM assessments WHERE instrument='MOCA' ORDER BY created_at DESC")
    if not rows:
        return {
            "total_assessments": 0, "unique_patients": 0,
            "avg_score": 0, "pct_impaired": 0,
            "severity_distribution": {}, "patient_summary": [],
            "active_alerts": [],
        }

    scores = [r["score"] for r in rows]
    pids = list({r["patient_id"] for r in rows})

    sev_counts = Counter()
    for r in rows:
        sev_counts[_severity(r["score"])] += 1

    # % impaired = score < 26
    n_impaired = sum(1 for s in scores if s < 26)
    pct_impaired = round(100 * n_impaired / len(scores), 1) if scores else 0

    # Latest per patient
    latest = {}
    for r in rows:
        pid = r["patient_id"]
        if pid not in latest:
            latest[pid] = r

    patient_summary = []
    for pid, r in sorted(latest.items()):
        patient_summary.append({
            "patient_id": pid,
            "latest_score": int(r["score"]),
            "max_score": int(r.get("max_score") or 30),
            "severity": _severity_label(r["score"]),
            "interpretation": r.get("interpretation", ""),
            "assessed_at": r.get("created_at", ""),
        })

    # Alerts: score < 18 (moderate/severe)
    alerts = []
    for pid, r in latest.items():
        if r["score"] < 18:
            alerts.append({
                "patient_id": pid,
                "alert": f"Score {int(r['score'])}/30 — {_severity_label(r['score'])} impairment",
                "score": int(r["score"]),
                "severity": _severity(r["score"]),
            })

    return {
        "total_assessments": len(rows),
        "unique_patients": len(pids),
        "avg_score": round(sum(scores) / len(scores), 1),
        "pct_impaired": pct_impaired,
        "n_impaired": n_impaired,
        "n_normal": len(pids) - n_impaired,
        "severity_distribution": dict(sev_counts),
        "patient_summary": patient_summary,
        "active_alerts": alerts,
    }


def breakdown():
    rows = _rows("SELECT * FROM assessments WHERE instrument='MOCA' ORDER BY created_at ASC")
    if not rows:
        return {
            "domain_scores": [], "severity_transitions": [],
            "trend": [], "patient_history": {},
        }

    # Per-domain average scores
    domain_totals = {d["id"]: [] for d in DOMAINS}
    for r in rows:
        try:
            ans = json.loads(r["answers_json"]) if r["answers_json"] else {}
            for d in DOMAINS:
                val = ans.get(d["id"])
                if val is not None:
                    domain_totals[d["id"]].append(float(val))
        except Exception:
            pass

    domain_scores = []
    for d in DOMAINS:
        vals = domain_totals[d["id"]]
        avg = round(sum(vals) / len(vals), 2) if vals else 0
        pct = round(100 * avg / d["max"], 1) if d["max"] else 0
        domain_scores.append({
            "id": d["id"],
            "label": d["label"],
            "avg_score": avg,
            "max": d["max"],
            "pct_of_max": pct,
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
                "first_severity": _severity_label(first["score"]),
                "latest_score": int(last["score"]),
                "latest_severity": _severity_label(last["score"]),
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
            "pct_impaired": round(100 * sum(1 for v in vals if v < 26) / len(vals), 1),
        })

    # Per-patient history
    patient_history = {}
    for pid, recs in sorted(by_patient.items()):
        patient_history[pid] = [
            {
                "score": int(r["score"]),
                "severity": _severity_label(r["score"]),
                "date": r.get("created_at", ""),
            }
            for r in recs
        ]

    return {
        "domain_scores": domain_scores,
        "severity_transitions": severity_transitions,
        "trend": trend,
        "patient_history": patient_history,
    }


def definitions():
    return {
        "title": "MoCA — Montreal Cognitive Assessment",
        "reference": (
            "Nasreddine ZS, Phillips NA, Bédirian V, et al. The Montreal Cognitive Assessment, "
            "MoCA: a brief screening tool for mild cognitive impairment. "
            "J Am Geriatr Soc. 2005;53(4):695-699. doi:10.1111/j.1532-5415.2005.53221.x"
        ),
        "domains": DOMAINS,
        "severity_tiers": SEVERITY_TIERS,
        "clinical_notes": [
            {
                "term": "Screening threshold",
                "definition": (
                    "MoCA < 26 indicates possible cognitive impairment (sensitivity 90%, "
                    "specificity 87% for MCI vs. normal; Nasreddine 2005). "
                    "+1 point correction if ≤12 years education (cap at 30)."
                ),
            },
            {
                "term": "Epilepsy & cognition",
                "definition": (
                    "20–50% of drug-resistant epilepsy patients exhibit measurable cognitive "
                    "decline. Temporal lobe epilepsy preferentially impairs delayed recall "
                    "(item6). Frontal lobe involvement affects visuospatial and executive "
                    "function (item1)."
                ),
            },
            {
                "term": "AED cognitive effects",
                "definition": (
                    "Topiramate and phenobarbital have the highest cognitive load "
                    "(processing speed, attention). Levetiracetam and lamotrigine are "
                    "relatively cognitively neutral. MoCA monitoring is recommended at "
                    "AED initiation, dose change, and annual review."
                ),
            },
            {
                "term": "MCI vs. dementia",
                "definition": (
                    "MoCA detects MCI (18–25) earlier than MMSE, which tends to miss "
                    "subtle deficits. MCI in epilepsy does not automatically indicate "
                    "dementia progression — seizure control improvement may partially "
                    "reverse cognitive deficits."
                ),
            },
            {
                "term": "Delayed Recall domain",
                "definition": (
                    "The 5-word delayed recall (item6) is the most sensitive domain for "
                    "hippocampal dysfunction. Scores ≤2/5 warrant MRI hippocampal volumetry "
                    "and neuropsychological referral."
                ),
            },
            {
                "term": "Serial monitoring",
                "definition": (
                    "A ≥2-point decline on serial MoCA is considered clinically significant. "
                    "Repeat every 12 months for stable patients; every 6 months if score <22 "
                    "or if AED changes occurred."
                ),
            },
        ],
    }
