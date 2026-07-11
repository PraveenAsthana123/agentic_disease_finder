"""PHQ-9 Dashboard — Patient Health Questionnaire-9 analytics.

All data from REAL PHQ-9 assessments in data/clinical.db (assessments table, instrument='PHQ9').

The PHQ-9 is a validated 9-item self-report measure for screening, diagnosing,
monitoring, and measuring severity of depression. Each item scores 0-3
(0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day).

Items:
  1. Little interest or pleasure in doing things (anhedonia)
  2. Feeling down, depressed, or hopeless
  3. Trouble falling/staying asleep, or sleeping too much
  4. Feeling tired or having little energy
  5. Poor appetite or overeating
  6. Feeling bad about yourself / failure
  7. Trouble concentrating
  8. Moving/speaking slowly or being fidgety/restless
  9. Thoughts of self-harm or being better off dead

Score range: 0-27
Severity tiers:
  0-4   Minimal
  5-9   Mild
  10-14 Moderate
  15-19 Moderately severe
  20-27 Severe

Clinical note: Item 9 (suicidal ideation) requires immediate follow-up regardless
of total score. PHQ-9 >= 10 has 88% sensitivity and 88% specificity for major
depression (Kroenke et al., 2001).

Reference:
  Kroenke K, Spitzer RL, Williams JBW. The PHQ-9: validity of a brief
  depression severity measure. J Gen Intern Med. 2001;16(9):606-613.

Author: Research Team
"""
import sqlite3
import json
from pathlib import Path
from collections import Counter, defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"

PHQ9_ITEMS = [
    {"id": "item1", "label": "Anhedonia", "description": "Little interest or pleasure in doing things"},
    {"id": "item2", "label": "Depressed mood", "description": "Feeling down, depressed, or hopeless"},
    {"id": "item3", "label": "Sleep disturbance", "description": "Trouble falling/staying asleep, or sleeping too much"},
    {"id": "item4", "label": "Fatigue", "description": "Feeling tired or having little energy"},
    {"id": "item5", "label": "Appetite change", "description": "Poor appetite or overeating"},
    {"id": "item6", "label": "Guilt / worthlessness", "description": "Feeling bad about yourself — or that you are a failure"},
    {"id": "item7", "label": "Concentration", "description": "Trouble concentrating on things, such as reading or watching TV"},
    {"id": "item8", "label": "Psychomotor change", "description": "Moving or speaking so slowly that others noticed, or being fidgety/restless"},
    {"id": "item9", "label": "Suicidal ideation", "description": "Thoughts that you would be better off dead, or of hurting yourself"},
]

SEVERITY_TIERS = [
    {"range": [0, 4], "label": "Minimal", "color": "#22c55e", "action": "No treatment indicated; monitor if epilepsy comorbidity"},
    {"range": [5, 9], "label": "Mild", "color": "#84cc16", "action": "Watchful waiting; repeat PHQ-9 at follow-up"},
    {"range": [10, 14], "label": "Moderate", "color": "#eab308", "action": "Treatment plan (counselling, pharmacotherapy, or both)"},
    {"range": [15, 19], "label": "Moderately severe", "color": "#f97316", "action": "Active treatment with pharmacotherapy and/or psychotherapy"},
    {"range": [20, 27], "label": "Severe", "color": "#ef4444", "action": "Immediate initiation of pharmacotherapy; consider referral to psychiatry"},
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
    rows = _rows("SELECT * FROM assessments WHERE instrument='PHQ9' ORDER BY created_at DESC")
    if not rows:
        return {
            "total_assessments": 0, "unique_patients": 0,
            "avg_score": 0, "item9_flag_rate_pct": 0,
            "severity_distribution": {}, "patient_summary": [],
            "active_alerts": [],
        }

    scores = [r["score"] for r in rows]
    pids = list({r["patient_id"] for r in rows})

    # Severity distribution
    sev_counts = Counter()
    for r in rows:
        sev_counts[_severity(r["score"])] += 1

    # Item 9 flag rate (suicidal ideation > 0)
    item9_flags = 0
    for r in rows:
        try:
            ans = json.loads(r["answers_json"]) if r["answers_json"] else {}
            if ans.get("item9", 0) > 0:
                item9_flags += 1
        except Exception:
            pass
    item9_pct = round(100 * item9_flags / len(rows), 1) if rows else 0

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

    # Active alerts: item9 > 0 OR score >= 15
    alerts = []
    for pid, r in latest.items():
        reasons = []
        if r["score"] >= 15:
            reasons.append(f"Score {int(r['score'])}/27 — moderately severe+")
        try:
            ans = json.loads(r["answers_json"]) if r["answers_json"] else {}
            if ans.get("item9", 0) > 0:
                reasons.append(f"Item 9 (suicidal ideation) = {ans['item9']}")
        except Exception:
            pass
        for reason in reasons:
            alerts.append({
                "patient_id": pid,
                "alert": reason,
                "score": int(r["score"]),
                "severity": _severity(r["score"]),
            })

    return {
        "total_assessments": len(rows),
        "unique_patients": len(pids),
        "avg_score": round(sum(scores) / len(scores), 1),
        "item9_flag_rate_pct": item9_pct,
        "severity_distribution": dict(sev_counts),
        "patient_summary": patient_summary,
        "active_alerts": alerts,
    }


def breakdown():
    rows = _rows("SELECT * FROM assessments WHERE instrument='PHQ9' ORDER BY created_at ASC")
    if not rows:
        return {
            "item_endorsement": [], "severity_transitions": [],
            "trend": [], "patient_history": {},
        }

    # Per-item endorsement rates (score > 0 for that item)
    item_totals = {it["id"]: 0 for it in PHQ9_ITEMS}
    item_severe = {it["id"]: 0 for it in PHQ9_ITEMS}  # score >= 2
    parsed = 0
    for r in rows:
        try:
            ans = json.loads(r["answers_json"]) if r["answers_json"] else {}
            parsed += 1
            for it in PHQ9_ITEMS:
                val = ans.get(it["id"], 0)
                if val > 0:
                    item_totals[it["id"]] += 1
                if val >= 2:
                    item_severe[it["id"]] += 1
        except Exception:
            pass

    item_endorsement = []
    for it in PHQ9_ITEMS:
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
        "title": "PHQ-9 — Patient Health Questionnaire-9",
        "reference": "Kroenke K, Spitzer RL, Williams JBW. The PHQ-9: validity of a brief depression severity measure. J Gen Intern Med. 2001;16(9):606-613.",
        "items": [
            {**it, "scoring": "0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day"}
            for it in PHQ9_ITEMS
        ],
        "severity_tiers": SEVERITY_TIERS,
        "clinical_notes": [
            {"term": "Item 9 Flag", "definition": "Any non-zero response to item 9 (suicidal ideation) requires immediate clinical follow-up regardless of total score."},
            {"term": "Sensitivity", "definition": "PHQ-9 >= 10 has 88% sensitivity and 88% specificity for major depressive disorder."},
            {"term": "Epilepsy relevance", "definition": "Depression is the most common psychiatric comorbidity in epilepsy (prevalence 20-55%). PHQ-9 is recommended by the ILAE Task Force for routine screening in epilepsy clinics."},
            {"term": "Treatment response", "definition": "A 5-point decrease in PHQ-9 score is considered clinically significant. 50% reduction indicates treatment response."},
            {"term": "Remission", "definition": "PHQ-9 < 5 is considered depression remission."},
        ],
    }
