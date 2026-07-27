"""
Neuro AI Ecosystem — Cognitive Decline Tracking Dashboard
=========================================================
Monitors neurocognitive trajectories in epilepsy patients over time,
tracking MoCA, MMSE, and domain-specific cognitive indices (memory,
attention, executive function, processing speed) across longitudinal
timepoints.

Data: 96 rows, 20 patients (EPAT001-EPAT020), 3-6 timepoints each.
All data is read from the existing cognitive_decline_tracking table
in clinical.db — no data seeding or fabrication.
"""

import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

CLASS_COLORS = {
    "stable": "#22c55e",
    "improving": "#3b82f6",
    "mild_decline": "#f59e0b",
    "moderate_decline": "#f97316",
    "significant_decline": "#ef4444",
}
CLASS_LABELS = {
    "stable": "Stable",
    "improving": "Improving",
    "mild_decline": "Mild Decline",
    "moderate_decline": "Moderate Decline",
    "significant_decline": "Significant Decline",
}
RISK_COLORS = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}


def _conn():
    return sqlite3.connect(DB_PATH)


def _round1(val):
    if val is None:
        return None
    return round(val, 1)


def _calc_slope(x_vals, y_vals):
    """Linear regression slope (y per unit x).  x is months, so slope = pts/month.
    We report pts/year (multiply by 12) for clinical readability."""
    n = len(x_vals)
    if n < 2:
        return 0.0
    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
    den = sum((x - x_mean) ** 2 for x in x_vals)
    if den == 0:
        return 0.0
    return num / den * 12  # convert to pts/year


def _classify(moca_slope_per_year):
    """Classify cognitive trajectory based on annualised MoCA slope."""
    s = moca_slope_per_year
    if s <= -3.0:
        return "significant_decline"
    if s <= -1.0:
        return "moderate_decline"
    if s < -0.2:
        return "mild_decline"
    if s > 0.5:
        return "improving"
    return "stable"


def _risk_level(classification, latest_moca):
    if classification in ("significant_decline", "moderate_decline") or (latest_moca is not None and latest_moca < 23):
        return "high"
    if classification == "mild_decline" or (latest_moca is not None and latest_moca <= 25):
        return "medium"
    return "low"


def _patient_name(pid):
    """Look up patient name from the patients table; fall back to pid."""
    try:
        conn = _conn()
        row = conn.execute("SELECT name FROM patients WHERE patient_id = ?", (pid,)).fetchone()
        conn.close()
        if row and row[0]:
            return row[0]
    except Exception:
        pass
    return pid


# ── Public API ────────────────────────────────────────────────────────

def _build_patients():
    """Build per-patient summaries with trajectories, classification, risk."""
    conn = _conn()

    pids = [r[0] for r in conn.execute(
        "SELECT DISTINCT patient_id FROM cognitive_decline_tracking ORDER BY patient_id"
    ).fetchall()]

    # Try to get patient ages
    age_map = {}
    try:
        for r in conn.execute("SELECT patient_id, age FROM patients"):
            age_map[r[0]] = r[1]
    except Exception:
        pass

    patients = []
    for pid in pids:
        rows = conn.execute("""
            SELECT timepoint, months_from_baseline, assessment_date,
                   moca_score, mmse_score,
                   memory_index, attention_index, executive_index, processing_speed_index,
                   notes
            FROM cognitive_decline_tracking
            WHERE patient_id = ?
            ORDER BY timepoint
        """, (pid,)).fetchall()

        if not rows:
            continue

        timepoints = []
        months_list = []
        moca_list = []
        mmse_list = []
        mem_list = []
        att_list = []
        exe_list = []
        spd_list = []

        for r in rows:
            tp = {
                "timepoint": r[0],
                "months": _round1(r[1]),
                "date": r[2],
                "moca": _round1(r[3]),
                "mmse": _round1(r[4]),
                "memory": _round1(r[5]),
                "attention": _round1(r[6]),
                "executive": _round1(r[7]),
                "processing_speed": _round1(r[8]),
                "notes": r[9],
            }
            timepoints.append(tp)
            if r[1] is not None:
                months_list.append(r[1])
            if r[3] is not None:
                moca_list.append(r[3])
            if r[4] is not None:
                mmse_list.append(r[4])
            if r[5] is not None:
                mem_list.append(r[5])
            if r[6] is not None:
                att_list.append(r[6])
            if r[7] is not None:
                exe_list.append(r[7])
            if r[8] is not None:
                spd_list.append(r[8])

        baseline_moca = _round1(moca_list[0]) if moca_list else None
        latest_moca = _round1(moca_list[-1]) if moca_list else None
        moca_change = _round1(latest_moca - baseline_moca) if (latest_moca is not None and baseline_moca is not None) else None

        moca_slope = _round1(_calc_slope(months_list, moca_list)) if len(months_list) >= 2 and len(moca_list) >= 2 else 0.0

        # Domain slopes (pts/year)
        domain_slopes = {}
        for dname, dvals in [("memory", mem_list), ("attention", att_list),
                             ("executive", exe_list), ("processing_speed", spd_list)]:
            if len(months_list) >= 2 and len(dvals) >= 2:
                domain_slopes[dname] = _round1(_calc_slope(months_list, dvals))
            else:
                domain_slopes[dname] = 0.0

        follow_up_months = _round1(months_list[-1]) if months_list else 0

        classification = _classify(moca_slope)
        risk = _risk_level(classification, latest_moca)

        patients.append({
            "patient_id": pid,
            "name": _patient_name(pid),
            "age": age_map.get(pid),
            "n_timepoints": len(timepoints),
            "baseline_moca": baseline_moca,
            "latest_moca": latest_moca,
            "moca_change": moca_change,
            "moca_slope": moca_slope,
            "follow_up_months": follow_up_months,
            "classification": classification,
            "classification_label": CLASS_LABELS.get(classification, classification),
            "classification_color": CLASS_COLORS.get(classification, "#64748b"),
            "risk_level": risk,
            "risk_color": RISK_COLORS.get(risk, "#64748b"),
            "domain_slopes": domain_slopes,
            "timepoints": timepoints,
        })

    conn.close()
    return patients


def overview():
    patients = _build_patients()

    total_patients = len(patients)
    total_assessments = sum(p["n_timepoints"] for p in patients)
    patients_with_decline = sum(1 for p in patients if p["classification"] in
                                ("mild_decline", "moderate_decline", "significant_decline"))
    decline_rate_pct = _round1(patients_with_decline / total_patients * 100) if total_patients else 0
    avg_follow_up = _round1(sum(p["follow_up_months"] for p in patients) / total_patients) if total_patients else 0

    # Classification distribution
    from collections import Counter
    class_counts = Counter(p["classification"] for p in patients)
    classification_distribution = []
    for cls in ["stable", "improving", "mild_decline", "moderate_decline", "significant_decline"]:
        if class_counts.get(cls, 0) > 0:
            classification_distribution.append({
                "classification": cls,
                "label": CLASS_LABELS[cls],
                "count": class_counts[cls],
                "color": CLASS_COLORS[cls],
            })

    # Domain decline rates (% of patients with negative slope in each domain)
    domains = ["memory", "attention", "executive", "processing_speed"]
    domain_decline_rates = []
    for d in domains:
        declining = sum(1 for p in patients if p["domain_slopes"].get(d, 0) < -1.0)
        rate = _round1(declining / total_patients * 100) if total_patients else 0
        domain_decline_rates.append({"domain": d.replace("_", " ").title(), "rate_pct": rate})

    # MoCA trend (avg by timepoint)
    conn = _conn()
    tp_rows = conn.execute("""
        SELECT timepoint, ROUND(AVG(moca_score), 1) AS avg_moca
        FROM cognitive_decline_tracking
        GROUP BY timepoint ORDER BY timepoint
    """).fetchall()
    conn.close()
    moca_trend = [{"timepoint": r[0], "avg_moca": r[1]} for r in tp_rows]

    # Risk stratification
    risk_counts = Counter(p["risk_level"] for p in patients)
    risk_stratification = [
        {"level": "high", "count": risk_counts.get("high", 0), "color": RISK_COLORS["high"]},
        {"level": "medium", "count": risk_counts.get("medium", 0), "color": RISK_COLORS["medium"]},
        {"level": "low", "count": risk_counts.get("low", 0), "color": RISK_COLORS["low"]},
    ]

    return {
        "kpis": {
            "total_patients": total_patients,
            "patients_with_decline": patients_with_decline,
            "decline_rate_pct": decline_rate_pct,
            "avg_follow_up_months": avg_follow_up,
            "total_assessments": total_assessments,
        },
        "classification_distribution": classification_distribution,
        "domain_decline_rates": domain_decline_rates,
        "moca_trend": moca_trend,
        "risk_stratification": risk_stratification,
    }


def breakdown():
    patients = _build_patients()

    # Flagged patients: those with moderate/significant decline or high risk
    flagged = []
    for p in patients:
        if p["classification"] in ("moderate_decline", "significant_decline") or p["risk_level"] == "high":
            declining_domains = [d for d, s in p["domain_slopes"].items() if s < -3.0]
            rec = _recommendation(p["classification"], p["risk_level"], declining_domains)
            flagged.append({
                "patient_id": p["patient_id"],
                "name": p["name"],
                "age": p["age"],
                "classification": p["classification_label"],
                "classification_color": p["classification_color"],
                "risk_level": p["risk_level"],
                "moca_slope": p["moca_slope"],
                "moca_change": p["moca_change"],
                "follow_up_months": p["follow_up_months"],
                "declining_domains": declining_domains,
                "recommendation": rec,
            })

    return {
        "patients": patients,
        "flagged_patients": flagged,
    }


def _recommendation(classification, risk_level, declining_domains):
    if classification == "significant_decline":
        return ("Urgent neuropsychological evaluation recommended. Consider AED review "
                "for cognitive-sparing alternatives. Refer for cognitive rehabilitation.")
    if classification == "moderate_decline":
        if declining_domains:
            return (f"Follow-up assessment within 3 months. Declining domains: "
                    f"{', '.join(declining_domains)}. Consider targeted cognitive intervention.")
        return "Follow-up assessment within 3 months. Consider AED regimen review."
    if risk_level == "high":
        return ("MoCA below 23 — further neuropsychological workup warranted. "
                "Rule out reversible causes (medication side effects, sleep, mood).")
    return "Continue routine monitoring per protocol."


def definitions():
    return {
        "decline_thresholds": [
            {
                "classification": "significant_decline",
                "label": "Significant Decline",
                "color": "#ef4444",
                "moca_slope": "< -3.0 pts/year",
                "description": "Rapid cognitive deterioration requiring urgent evaluation and intervention.",
            },
            {
                "classification": "moderate_decline",
                "label": "Moderate Decline",
                "color": "#f97316",
                "moca_slope": "-3.0 to -1.0 pts/year",
                "description": "Clinically meaningful decline. Warrants closer monitoring and possible treatment modification.",
            },
            {
                "classification": "mild_decline",
                "label": "Mild Decline",
                "color": "#f59e0b",
                "moca_slope": "-1.0 to -0.2 pts/year",
                "description": "Subtle cognitive slowing. May reflect normal aging or early medication effect.",
            },
            {
                "classification": "stable",
                "label": "Stable",
                "color": "#22c55e",
                "moca_slope": "-0.2 to +0.5 pts/year",
                "description": "No clinically significant change in cognitive function.",
            },
            {
                "classification": "improving",
                "label": "Improving",
                "color": "#3b82f6",
                "moca_slope": "> +0.5 pts/year",
                "description": "Cognitive function improving, possibly reflecting treatment optimisation or practice effect.",
            },
        ],
        "assessment_instruments": [
            {
                "name": "Montreal Cognitive Assessment (MoCA)",
                "description": "Screening tool for mild cognitive impairment. Assesses attention, executive function, memory, language, visuospatial, and orientation.",
                "range": "0-30",
                "cutoff": "<26 abnormal",
                "time": "10 min",
            },
            {
                "name": "Mini-Mental State Examination (MMSE)",
                "description": "Brief standardised test covering orientation, registration, attention, recall, and language.",
                "range": "0-30",
                "cutoff": "<24 abnormal",
                "time": "5-10 min",
            },
            {
                "name": "Memory Index",
                "description": "Composite score from verbal and visual memory tasks (immediate and delayed recall).",
                "range": "Mean=100, SD=15",
                "cutoff": "<85 concerning",
            },
            {
                "name": "Attention Index",
                "description": "Composite score measuring sustained, selective, and divided attention.",
                "range": "Mean=100, SD=15",
                "cutoff": "<85 concerning",
            },
            {
                "name": "Executive Function Index",
                "description": "Planning, cognitive flexibility, inhibition, and working memory composite.",
                "range": "Mean=100, SD=15",
                "cutoff": "<85 concerning",
            },
            {
                "name": "Processing Speed Index",
                "description": "Speed of cognitive operations. AEDs (especially topiramate, phenobarbital) can reduce this.",
                "range": "Mean=100, SD=15",
                "cutoff": "<85 concerning",
            },
        ],
        "risk_factors": [
            {"factor": "Seizure Frequency", "description": "Higher seizure burden is associated with accelerated cognitive decline, particularly for memory and attention."},
            {"factor": "AED Polytherapy", "description": "Multiple anti-epileptic drugs increase risk of cognitive side effects, especially processing speed reduction."},
            {"factor": "Age of Onset", "description": "Earlier seizure onset is linked to greater cumulative cognitive impact over the lifespan."},
            {"factor": "Seizure Type", "description": "Generalised tonic-clonic seizures carry higher cognitive risk than focal aware seizures."},
            {"factor": "Duration of Epilepsy", "description": "Longer disease duration correlates with progressive decline in memory and executive function."},
            {"factor": "Status Epilepticus History", "description": "Episodes of status epilepticus can cause acute hippocampal damage and lasting memory impairment."},
            {"factor": "Sleep Disruption", "description": "Poor sleep quality (common in epilepsy) impairs memory consolidation and attention."},
            {"factor": "Comorbid Depression", "description": "Depression (prevalence 30-50% in epilepsy) independently contributes to cognitive complaints and poor test performance."},
        ],
        "clinical_actions": [
            {
                "category": "Significant Decline",
                "color": "#ef4444",
                "actions": [
                    "Urgent comprehensive neuropsychological evaluation",
                    "AED review — switch to cognitive-sparing alternatives (levetiracetam, lamotrigine)",
                    "MRI to rule out structural progression",
                    "Cognitive rehabilitation referral",
                    "Screen for depression and sleep disorders",
                ],
                "follow_up": "4-6 weeks",
            },
            {
                "category": "Moderate Decline",
                "color": "#f97316",
                "actions": [
                    "Repeat cognitive assessment in 3 months",
                    "Review AED regimen for cognitive side effects",
                    "Consider neuropsychology referral",
                    "Assess seizure control and optimise if needed",
                ],
                "follow_up": "3 months",
            },
            {
                "category": "Mild Decline",
                "color": "#f59e0b",
                "actions": [
                    "Continue routine monitoring",
                    "Educate patient on cognitive strategies",
                    "Monitor for progression at next scheduled visit",
                ],
                "follow_up": "6 months",
            },
            {
                "category": "Stable / Improving",
                "color": "#22c55e",
                "actions": [
                    "Maintain current treatment plan",
                    "Reinforce positive outcomes with patient",
                    "Routine follow-up per protocol",
                ],
                "follow_up": "6-12 months",
            },
        ],
        "glossary": [
            {"term": "Timepoint", "definition": "Sequential assessment number (1 = baseline). Each timepoint is a scheduled neurocognitive evaluation."},
            {"term": "MoCA Slope", "definition": "Annualised rate of change in MoCA score (pts/year) via linear regression over all timepoints."},
            {"term": "Cognitive Index", "definition": "Standardised score (mean=100, SD=15). Scores below 85 (1 SD below mean) are clinically concerning."},
            {"term": "MCI", "definition": "Mild Cognitive Impairment — decline beyond normal ageing but not meeting dementia criteria. MoCA 23-25 in epilepsy patients."},
            {"term": "AED", "definition": "Anti-Epileptic Drug. Some (topiramate, phenobarbital) have significant cognitive side effects."},
            {"term": "Baseline Assessment", "definition": "First cognitive evaluation, serving as the reference for all subsequent comparisons."},
            {"term": "Domain Slope", "definition": "Annualised rate of change in a cognitive domain index (memory, attention, executive, processing speed)."},
            {"term": "Decline Classification", "definition": "Category assigned based on annualised MoCA slope: significant (<-3), moderate (-3 to -1), mild (-1 to -0.2), stable, improving (>0.5)."},
        ],
    }
