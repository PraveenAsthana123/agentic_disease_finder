"""MoCA Auto-Scoring Dashboard — Montreal Cognitive Assessment auto-scoring
with normative comparison, domain breakdown, trend analysis, and impairment
classification.  Real data from clinical.db neuropsych table."""

import sqlite3
import json
from pathlib import Path
from collections import Counter, defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    return sqlite3.connect(str(DB))


# ── MoCA domain scoring reference ──────────────────────────────────
# MoCA total = 30 points across 8 cognitive domains.
DOMAIN_MAP = {
    "visuospatial_executive": {"max": 5, "label": "Visuospatial / Executive"},
    "naming": {"max": 3, "label": "Naming"},
    "attention": {"max": 6, "label": "Attention"},
    "language": {"max": 3, "label": "Language"},
    "abstraction": {"max": 2, "label": "Abstraction"},
    "delayed_recall": {"max": 5, "label": "Delayed Recall"},
    "orientation": {"max": 6, "label": "Orientation"},
}

# Normative thresholds (Nasreddine et al., 2005)
CUTOFFS = {
    "normal": 26,       # >= 26: normal
    "mci": 18,          # 18-25: mild cognitive impairment
    "moderate": 10,     # 10-17: moderate impairment
    # < 10: severe impairment
}

# Education adjustment: +1 point if <= 12 years of education
EDUCATION_ADJUSTMENT = 12


def _classify(score):
    """Classify a MoCA total score into impairment category."""
    if score >= CUTOFFS["normal"]:
        return "normal"
    elif score >= CUTOFFS["mci"]:
        return "mci"
    elif score >= CUTOFFS["moderate"]:
        return "moderate"
    else:
        return "severe"


def _load_records():
    """Load all neuropsych records with MoCA data."""
    con = _conn()
    cur = con.cursor()
    cur.execute("SELECT patient_id, fields_json, created_at FROM neuropsych ORDER BY created_at")
    rows = cur.fetchall()
    con.close()

    records = []
    for pid, fj, created in rows:
        data = json.loads(fj)
        moca = data.get("moca")
        if moca is None:
            continue
        records.append({
            "patient_id": pid,
            "moca_total": moca,
            "mmse": data.get("mmse"),
            "phq9": data.get("phq9"),
            "gad7": data.get("gad7"),
            "memory_index": data.get("memory_index"),
            "attention_index": data.get("attention_index"),
            "executive_index": data.get("executive_index"),
            "language_index": data.get("language_index"),
            "processing_speed_index": data.get("processing_speed_index"),
            "verbal_memory_raw": data.get("verbal_memory_raw"),
            "visual_memory_raw": data.get("visual_memory_raw"),
            "digit_span_forward": data.get("digit_span_forward"),
            "digit_span_backward": data.get("digit_span_backward"),
            "trail_a_seconds": data.get("trail_a_seconds"),
            "trail_b_seconds": data.get("trail_b_seconds"),
            "impairment_flag": data.get("impairment_flag", "unknown"),
            "lateralization": data.get("lateralization_hypothesis", "unknown"),
            "battery_type": data.get("battery_type", "unknown"),
            "referral_reason": data.get("referral_reason", "unknown"),
            "assessor": data.get("assessor", "unknown"),
            "assessed_at": created,
        })
    return records


def _estimate_domains(rec):
    """Estimate domain scores from available indices.

    The neuropsych table stores composite indices (0-130 scale) rather than
    raw MoCA sub-domain scores.  We derive proportional domain estimates
    by mapping each index to the MoCA sub-domain it most closely reflects,
    then scaling to the sub-domain max.  This is an *estimate* — for exact
    MoCA sub-scores, the raw per-item data would be required.
    """
    total = rec["moca_total"]
    domains = {}

    # Map composite indices to domains (index/130 * domain_max, clamped)
    mapping = [
        ("visuospatial_executive", rec.get("executive_index"), 5),
        ("attention", rec.get("attention_index"), 6),
        ("language", rec.get("language_index"), 3),
        ("delayed_recall", rec.get("memory_index"), 5),
    ]

    allocated = 0
    for domain, idx, mx in mapping:
        if idx is not None:
            est = round(min(idx / 130.0, 1.0) * mx)
            est = max(0, min(est, mx))
        else:
            est = round(mx * total / 30.0)
        domains[domain] = est
        allocated += est

    # Naming (from language index, 3 pts)
    lang_idx = rec.get("language_index")
    if lang_idx is not None:
        domains["naming"] = min(3, round(lang_idx / 130.0 * 3))
    else:
        domains["naming"] = min(3, round(3 * total / 30.0))
    allocated += domains["naming"]

    # Abstraction (2 pts) — derive from executive
    exec_idx = rec.get("executive_index")
    if exec_idx is not None:
        domains["abstraction"] = min(2, round(exec_idx / 130.0 * 2))
    else:
        domains["abstraction"] = min(2, round(2 * total / 30.0))
    allocated += domains["abstraction"]

    # Orientation (6 pts) — remaining points
    remaining = max(0, total - allocated)
    domains["orientation"] = min(6, remaining)

    return domains


def overview():
    """Overview: KPIs, classification distribution, score histogram,
    domain averages, assessor stats, correlation with MMSE."""
    records = _load_records()
    if not records:
        return {"available": False}

    scores = [r["moca_total"] for r in records]
    n = len(scores)
    avg = sum(scores) / n
    classifications = Counter(_classify(s) for s in scores)

    # Unique patients
    patients = set(r["patient_id"] for r in records)

    # Assessor stats
    assessors = Counter(r["assessor"] for r in records)

    # Score histogram (bins: 10-15, 16-20, 21-25, 26-30)
    bins = [(10, 15), (16, 20), (21, 25), (26, 30)]
    histogram = []
    for lo, hi in bins:
        count = sum(1 for s in scores if lo <= s <= hi)
        histogram.append({"range": f"{lo}-{hi}", "count": count})

    # MoCA vs MMSE correlation
    moca_mmse = []
    for r in records:
        if r["mmse"] is not None:
            moca_mmse.append({"moca": r["moca_total"], "mmse": r["mmse"],
                              "patient_id": r["patient_id"]})

    # Domain averages
    all_domains = defaultdict(list)
    for r in records:
        domains = _estimate_domains(r)
        for d, v in domains.items():
            all_domains[d].append(v)

    domain_avg = []
    for d, info in DOMAIN_MAP.items():
        vals = all_domains.get(d, [])
        if vals:
            domain_avg.append({
                "domain": info["label"],
                "average": round(sum(vals) / len(vals), 1),
                "max": info["max"],
                "pct": round(sum(vals) / len(vals) / info["max"] * 100, 1),
            })

    # Classification by referral reason
    by_referral = defaultdict(lambda: Counter())
    for r in records:
        cat = _classify(r["moca_total"])
        by_referral[r["referral_reason"]][cat] += 1
    referral_breakdown = [
        {"reason": reason, **dict(cats)}
        for reason, cats in sorted(by_referral.items())
    ]

    return {
        "available": True,
        "kpis": {
            "total_assessments": n,
            "unique_patients": len(patients),
            "mean_score": round(avg, 1),
            "median_score": sorted(scores)[n // 2],
            "min_score": min(scores),
            "max_score": max(scores),
            "below_cutoff": sum(1 for s in scores if s < CUTOFFS["normal"]),
            "below_cutoff_pct": round(sum(1 for s in scores if s < CUTOFFS["normal"]) / n, 3),
            "normal_count": classifications.get("normal", 0),
            "mci_count": classifications.get("mci", 0),
            "moderate_count": classifications.get("moderate", 0),
            "severe_count": classifications.get("severe", 0),
        },
        "classification_distribution": [
            {"category": "Normal (≥26)", "count": classifications.get("normal", 0), "color": "#22c55e"},
            {"category": "MCI (18-25)", "count": classifications.get("mci", 0), "color": "#eab308"},
            {"category": "Moderate (10-17)", "count": classifications.get("moderate", 0), "color": "#f97316"},
            {"category": "Severe (<10)", "count": classifications.get("severe", 0), "color": "#ef4444"},
        ],
        "score_histogram": histogram,
        "domain_averages": domain_avg,
        "moca_vs_mmse": moca_mmse,
        "assessor_stats": [{"assessor": a, "count": c} for a, c in assessors.most_common()],
        "referral_breakdown": referral_breakdown,
    }


def breakdown():
    """Per-patient breakdown: individual scores, domain profiles,
    impairment classification, comorbidity indicators."""
    records = _load_records()
    if not records:
        return {"available": False}

    per_patient = []
    for r in records:
        domains = _estimate_domains(r)
        domain_list = [
            {"domain": DOMAIN_MAP[d]["label"], "score": domains[d], "max": DOMAIN_MAP[d]["max"]}
            for d in DOMAIN_MAP
        ]
        classification = _classify(r["moca_total"])

        # Impaired domains (below 50% of max)
        impaired = [
            DOMAIN_MAP[d]["label"]
            for d in DOMAIN_MAP
            if domains[d] < DOMAIN_MAP[d]["max"] * 0.5
        ]

        # Comorbidity flags
        depression = None
        if r["phq9"] is not None:
            if r["phq9"] >= 20:
                depression = "severe"
            elif r["phq9"] >= 15:
                depression = "moderately-severe"
            elif r["phq9"] >= 10:
                depression = "moderate"
            elif r["phq9"] >= 5:
                depression = "mild"
            else:
                depression = "minimal"

        anxiety = None
        if r["gad7"] is not None:
            if r["gad7"] >= 15:
                anxiety = "severe"
            elif r["gad7"] >= 10:
                anxiety = "moderate"
            elif r["gad7"] >= 5:
                anxiety = "mild"
            else:
                anxiety = "minimal"

        per_patient.append({
            "patient_id": r["patient_id"],
            "moca_total": r["moca_total"],
            "mmse": r["mmse"],
            "classification": classification,
            "domains": domain_list,
            "impaired_domains": impaired,
            "impaired_count": len(impaired),
            "phq9": r["phq9"],
            "depression_severity": depression,
            "gad7": r["gad7"],
            "anxiety_severity": anxiety,
            "trail_a": r["trail_a_seconds"],
            "trail_b": r["trail_b_seconds"],
            "battery_type": r["battery_type"],
            "referral_reason": r["referral_reason"],
            "assessor": r["assessor"],
            "assessed_at": r["assessed_at"],
            "lateralization": r["lateralization"],
        })

    # Sort by MoCA score ascending (most impaired first)
    per_patient.sort(key=lambda x: x["moca_total"])

    # By classification group
    by_class = defaultdict(list)
    for p in per_patient:
        by_class[p["classification"]].append(p["patient_id"])

    classification_groups = [
        {"classification": c, "patients": ids, "count": len(ids)}
        for c, ids in sorted(by_class.items(),
                             key=lambda x: ["severe", "moderate", "mci", "normal"].index(x[0])
                             if x[0] in ["severe", "moderate", "mci", "normal"] else 99)
    ]

    # Domain vulnerability — which domains are most commonly impaired
    domain_vuln = Counter()
    for p in per_patient:
        for d in p["impaired_domains"]:
            domain_vuln[d] += 1

    domain_vulnerability = [
        {"domain": d, "impaired_count": c, "pct": round(c / len(per_patient), 3)}
        for d, c in domain_vuln.most_common()
    ]

    # Depression-cognition correlation
    depression_cognition = []
    for p in per_patient:
        if p["phq9"] is not None:
            depression_cognition.append({
                "patient_id": p["patient_id"],
                "moca": p["moca_total"],
                "phq9": p["phq9"],
                "classification": p["classification"],
            })

    return {
        "available": True,
        "per_patient": per_patient,
        "classification_groups": classification_groups,
        "domain_vulnerability": domain_vulnerability,
        "depression_cognition": depression_cognition,
    }


def definitions():
    """Metric definitions, scoring guide, clinical caveats."""
    return {
        "metrics": [
            {"name": "MoCA Total", "description": "Montreal Cognitive Assessment total score (0-30). Higher = better cognition."},
            {"name": "Normal (≥26)", "description": "Score 26-30: no cognitive impairment detected."},
            {"name": "MCI (18-25)", "description": "Mild Cognitive Impairment: subtle deficits in one or more domains."},
            {"name": "Moderate (10-17)", "description": "Moderate impairment: significant deficits affecting daily function."},
            {"name": "Severe (<10)", "description": "Severe impairment: major cognitive dysfunction."},
            {"name": "MMSE", "description": "Mini-Mental State Examination (0-30). Compared for convergent validity."},
            {"name": "PHQ-9", "description": "Patient Health Questionnaire-9 (depression screening). Depression can lower MoCA."},
            {"name": "GAD-7", "description": "Generalized Anxiety Disorder-7 scale. Anxiety may affect attention/executive domains."},
            {"name": "Trail Making A", "description": "Psychomotor speed (seconds). Higher = slower processing."},
            {"name": "Trail Making B", "description": "Executive function/set-shifting (seconds). Higher = more impaired."},
        ],
        "domains": [
            {"domain": info["label"], "max_score": info["max"],
             "description": f"MoCA sub-domain: {info['label']} (max {info['max']} points)"}
            for info in DOMAIN_MAP.values()
        ],
        "scoring_guide": {
            "total_range": "0-30",
            "normal_cutoff": "≥ 26",
            "mci_range": "18-25",
            "moderate_range": "10-17",
            "severe_range": "< 10",
            "education_adjustment": "+1 point if ≤ 12 years education",
            "administration_time": "~10 minutes",
            "reference": "Nasreddine et al., 2005 (JAGS 53:695-699)",
        },
        "data_sources": [
            "clinical.db → neuropsych table (37 records, 30 patients)",
            "Domain scores estimated from composite neuropsych indices",
            "PHQ-9 and GAD-7 for comorbidity context",
        ],
        "clinical_caveat": (
            "Domain scores are ESTIMATED from composite neuropsych indices, not "
            "raw MoCA item-level data.  For clinical decisions, always use the "
            "official MoCA scoring sheet with item-level recording.  This dashboard "
            "is for screening, trend monitoring, and research — not standalone diagnosis."
        ),
    }
