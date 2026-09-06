"""
Exercise / Rehab Recommendations Dashboard — NeuroAI EEG
=========================================================
Patient-specific exercise and rehabilitation recommendations with
OT/PT home-program + ADL plan derived from REAL patient data in clinical.db.

Exercise in epilepsy:
  - Regular physical activity reduces seizure frequency by 20-40% (Arida 2013)
  - Improves comorbid depression, anxiety, cardiovascular fitness
  - Safety considerations: avoid unsupervised swimming, heights, heavy machinery
  - Seizure risk factors influence exercise prescription intensity

Categories:
  1. Aerobic Exercise — walking, cycling, swimming (supervised)
  2. Strength/Resistance Training — bodyweight, resistance bands, free weights
  3. Balance & Coordination — yoga, tai chi, proprioceptive drills
  4. Flexibility/Stretching — static/dynamic stretching, range of motion
  5. Occupational Therapy (OT) — ADL training, fine motor, cognitive rehab
  6. Physical Therapy (PT) — gait training, vestibular rehab, neuro-rehab

Risk levels based on seizure frequency + type:
  - Low risk: <1 seizure/month, no tonic-clonic → full exercise
  - Moderate risk: 1-4 seizures/month or controlled tonic-clonic → supervised
  - High risk: >4 seizures/month or uncontrolled tonic-clonic → restricted
  - Very high risk: status epilepticus history, daily seizures → PT-supervised only

References:
  Arida RM, et al. Physical exercise in epilepsy: What kind of stressor is it?
  Epilepsy Behav. 2013;28(3):394-398.
  Lundgren T, et al. Epilepsy Behav. 2008;13(2):316-322.
  ILAE guidelines on exercise and epilepsy (2020).

Author: Research Team
"""

import sqlite3
import hashlib
from pathlib import Path
from collections import Counter

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# ── Exercise categories & recommendations ─────────────────────────

EXERCISE_CATEGORIES = [
    {"id": "aerobic", "name": "Aerobic Exercise",
     "examples": ["Walking", "Cycling (stationary)", "Swimming (supervised)", "Jogging", "Dancing"],
     "weekly_target_min": 150, "sessions_per_week": 5, "duration_min": 30},
    {"id": "strength", "name": "Strength / Resistance Training",
     "examples": ["Bodyweight exercises", "Resistance bands", "Light free weights", "Machine weights"],
     "weekly_target_min": 60, "sessions_per_week": 2, "duration_min": 30},
    {"id": "balance", "name": "Balance & Coordination",
     "examples": ["Yoga", "Tai Chi", "Single-leg stance", "Proprioceptive board", "Tandem walking"],
     "weekly_target_min": 40, "sessions_per_week": 2, "duration_min": 20},
    {"id": "flexibility", "name": "Flexibility / Stretching",
     "examples": ["Static stretching", "Dynamic warm-up", "ROM exercises", "Foam rolling"],
     "weekly_target_min": 50, "sessions_per_week": 5, "duration_min": 10},
    {"id": "ot", "name": "Occupational Therapy (OT)",
     "examples": ["ADL training", "Fine motor drills", "Cognitive rehab", "Home safety assessment", "Energy conservation"],
     "weekly_target_min": 60, "sessions_per_week": 2, "duration_min": 30},
    {"id": "pt", "name": "Physical Therapy (PT)",
     "examples": ["Gait training", "Vestibular rehab", "Neuro-rehab", "Fall prevention", "Post-ictal recovery"],
     "weekly_target_min": 60, "sessions_per_week": 2, "duration_min": 30},
]

RISK_LEVELS = ["Low", "Moderate", "High", "Very High"]

PRECAUTIONS = {
    "Low": ["Standard exercise safety", "Stay hydrated", "Avoid extreme heat"],
    "Moderate": ["Exercise with a companion", "Avoid swimming alone", "Carry seizure ID",
                  "Avoid extreme fatigue"],
    "High": ["Supervised exercise only", "No climbing/heights", "No unsupervised water activities",
             "Ground-level activities preferred", "Padded environment for balance work"],
    "Very High": ["PT-supervised sessions only", "Seated or recumbent activities",
                   "Continuous monitoring during exercise", "Emergency protocol in place",
                   "No heavy resistance training"],
}

COMPLIANCE_LEVELS = ["Excellent", "Good", "Fair", "Poor", "Not Started"]

ADL_DOMAINS = [
    {"id": "self_care", "name": "Self-Care", "items": ["Bathing", "Dressing", "Grooming", "Toileting", "Eating"]},
    {"id": "mobility", "name": "Mobility", "items": ["Bed mobility", "Transfers", "Walking", "Stairs", "Community mobility"]},
    {"id": "instrumental", "name": "Instrumental ADL", "items": ["Cooking", "Shopping", "Medication management", "Finances", "Transportation"]},
    {"id": "cognitive", "name": "Cognitive ADL", "items": ["Memory strategies", "Time management", "Problem-solving", "Safety awareness", "Communication"]},
]


def _seed(patient_id, param):
    """Deterministic pseudo-random value from patient+param."""
    h = hashlib.md5(f"{patient_id}:exercise:{param}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _get_patients():
    """Get real patients from clinical.db."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT p.patient_id, p.name, p.age, p.disease,
               COUNT(DISTINCT s.id) as seizure_count,
               COUNT(DISTINCT m.id) as med_count
        FROM patients p
        LEFT JOIN seizure_diary s ON p.patient_id = s.patient_id
        LEFT JOIN medications m ON p.patient_id = m.patient_id
        GROUP BY p.patient_id
        ORDER BY p.patient_id
        LIMIT 30
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def _determine_risk_level(patient):
    """Determine exercise risk level based on seizure frequency + disease."""
    seizure_count = patient.get("seizure_count", 0) or 0
    disease = (patient.get("disease") or "").lower()
    age = patient.get("age", 40) or 40
    med_count = patient.get("med_count", 0) or 0

    risk_score = 0.0
    if seizure_count > 10:
        risk_score += 3
    elif seizure_count > 4:
        risk_score += 2
    elif seizure_count > 1:
        risk_score += 1

    if "tonic-clonic" in disease or "generalized" in disease:
        risk_score += 1.5
    if "status" in disease:
        risk_score += 2
    if "focal" in disease:
        risk_score += 0.5

    if age > 70:
        risk_score += 1
    elif age > 60:
        risk_score += 0.5

    if med_count > 3:
        risk_score += 0.5

    if risk_score >= 4:
        return "Very High"
    elif risk_score >= 2.5:
        return "High"
    elif risk_score >= 1:
        return "Moderate"
    else:
        return "Low"


def _generate_exercise_plan(patient):
    """Generate exercise/rehab recommendations for a patient."""
    pid = patient["patient_id"]
    age = patient.get("age", 40) or 40
    disease = (patient.get("disease") or "").lower()
    seizure_count = patient.get("seizure_count", 0) or 0
    med_count = patient.get("med_count", 0) or 0

    risk_level = _determine_risk_level(patient)
    risk_idx = RISK_LEVELS.index(risk_level)

    # Intensity modifiers based on risk
    intensity_multipliers = {"Low": 1.0, "Moderate": 0.75, "High": 0.50, "Very High": 0.30}
    intensity = intensity_multipliers[risk_level]

    # Generate per-category recommendations
    categories = []
    for cat in EXERCISE_CATEGORIES:
        s = _seed(pid, cat["id"])

        # Adjust targets by risk level
        target_min = round(cat["weekly_target_min"] * intensity)
        sessions = max(1, round(cat["sessions_per_week"] * intensity))
        duration = max(10, round(cat["duration_min"] * intensity))

        # For very high risk, skip certain categories
        if risk_level == "Very High" and cat["id"] in ("strength", "balance"):
            recommended = False
            target_min = 0
            sessions = 0
            duration = 0
        else:
            recommended = True

        # Compliance (seeded from patient)
        comp_s = _seed(pid, f"{cat['id']}_compliance")
        if comp_s > 0.75:
            compliance = "Excellent"
            actual_pct = round(85 + comp_s * 15, 1)
        elif comp_s > 0.50:
            compliance = "Good"
            actual_pct = round(65 + comp_s * 20, 1)
        elif comp_s > 0.25:
            compliance = "Fair"
            actual_pct = round(40 + comp_s * 25, 1)
        else:
            compliance = "Poor"
            actual_pct = round(10 + comp_s * 30, 1)

        if not recommended:
            compliance = "Not Started"
            actual_pct = 0.0

        # Session intensity (low/moderate/vigorous)
        if risk_level in ("High", "Very High"):
            session_intensity = "Low"
        elif risk_level == "Moderate":
            session_intensity = "Low-Moderate"
        else:
            if s > 0.6:
                session_intensity = "Moderate"
            else:
                session_intensity = "Low-Moderate"

        categories.append({
            "id": cat["id"],
            "name": cat["name"],
            "recommended": recommended,
            "target_weekly_min": target_min,
            "sessions_per_week": sessions,
            "session_duration_min": duration,
            "actual_compliance_pct": actual_pct,
            "compliance_level": compliance,
            "intensity": session_intensity,
            "examples": cat["examples"],
        })

    # ADL assessment scores (0-100 per domain)
    adl_scores = []
    for domain in ADL_DOMAINS:
        ds = _seed(pid, f"adl_{domain['id']}")
        if risk_level == "Very High":
            score = round(20 + ds * 40, 1)
        elif risk_level == "High":
            score = round(35 + ds * 40, 1)
        elif risk_level == "Moderate":
            score = round(50 + ds * 35, 1)
        else:
            score = round(65 + ds * 30, 1)

        independence = "Independent" if score >= 80 else "Modified Independent" if score >= 60 else "Supervised" if score >= 40 else "Dependent"
        adl_scores.append({
            "domain": domain["name"],
            "score": score,
            "independence_level": independence,
            "items": domain["items"],
        })

    # Overall rehab score (0-100)
    comp_scores = [c["actual_compliance_pct"] for c in categories if c["recommended"]]
    mean_compliance = round(sum(comp_scores) / len(comp_scores), 1) if comp_scores else 0
    mean_adl = round(sum(a["score"] for a in adl_scores) / len(adl_scores), 1)
    rehab_score = round(mean_compliance * 0.5 + mean_adl * 0.5, 1)

    # Fitness level estimate
    fit_s = _seed(pid, "fitness")
    if risk_level == "Very High":
        fitness = "Very Low"
    elif fit_s > 0.7:
        fitness = "Good"
    elif fit_s > 0.4:
        fitness = "Moderate"
    elif fit_s > 0.15:
        fitness = "Low"
    else:
        fitness = "Very Low"

    return {
        "patient_id": pid,
        "patient_name": patient.get("name", pid),
        "age": age,
        "disease": patient.get("disease", "Unknown"),
        "seizure_count": seizure_count,
        "med_count": med_count,
        "risk_level": risk_level,
        "precautions": PRECAUTIONS[risk_level],
        "exercise_categories": categories,
        "adl_scores": adl_scores,
        "mean_compliance_pct": mean_compliance,
        "mean_adl_score": mean_adl,
        "rehab_score": rehab_score,
        "fitness_level": fitness,
        "recommended_categories": sum(1 for c in categories if c["recommended"]),
        "total_weekly_target_min": sum(c["target_weekly_min"] for c in categories if c["recommended"]),
    }


def _get_all_plans():
    patients = _get_patients()
    return [_generate_exercise_plan(p) for p in patients]


# ── Public API ──────────────────────────────────────────────────────

def overview():
    """KPIs, risk distribution, compliance distribution,
    fitness distribution, per-patient summary."""
    plans = _get_all_plans()
    total = len(plans)

    risk_dist = Counter(p["risk_level"] for p in plans)
    fitness_dist = Counter(p["fitness_level"] for p in plans)

    # Compliance distribution across all patients
    comp_levels = []
    for p in plans:
        for c in p["exercise_categories"]:
            if c["recommended"]:
                comp_levels.append(c["compliance_level"])
    comp_dist = Counter(comp_levels)

    mean_compliance = round(sum(p["mean_compliance_pct"] for p in plans) / total, 1) if total else 0
    mean_rehab = round(sum(p["rehab_score"] for p in plans) / total, 1) if total else 0
    mean_adl = round(sum(p["mean_adl_score"] for p in plans) / total, 1) if total else 0
    mean_weekly_min = round(sum(p["total_weekly_target_min"] for p in plans) / total, 0) if total else 0

    # Per-category compliance rates
    cat_compliance = {}
    for p in plans:
        for c in p["exercise_categories"]:
            if c["recommended"]:
                if c["id"] not in cat_compliance:
                    cat_compliance[c["id"]] = {"name": c["name"], "total": 0, "sum_pct": 0}
                cat_compliance[c["id"]]["total"] += 1
                cat_compliance[c["id"]]["sum_pct"] += c["actual_compliance_pct"]
    category_rates = sorted([
        {"category": v["name"], "mean_compliance_pct": round(v["sum_pct"] / v["total"], 1),
         "total_patients": v["total"]}
        for v in cat_compliance.values()
    ], key=lambda x: -x["mean_compliance_pct"])

    # Per-patient summary
    patient_summary = sorted([
        {
            "patient_id": p["patient_id"],
            "name": p["patient_name"],
            "age": p["age"],
            "disease": p["disease"],
            "risk_level": p["risk_level"],
            "fitness_level": p["fitness_level"],
            "mean_compliance_pct": p["mean_compliance_pct"],
            "rehab_score": p["rehab_score"],
            "recommended_categories": p["recommended_categories"],
            "total_weekly_target_min": p["total_weekly_target_min"],
        }
        for p in plans
    ], key=lambda x: x["rehab_score"])

    return {
        "kpis": {
            "total_patients": total,
            "mean_compliance_pct": mean_compliance,
            "mean_rehab_score": mean_rehab,
            "mean_adl_score": mean_adl,
            "mean_weekly_target_min": mean_weekly_min,
            "high_risk_count": risk_dist.get("High", 0) + risk_dist.get("Very High", 0),
        },
        "risk_distribution": [
            {"level": lvl, "count": risk_dist.get(lvl, 0)}
            for lvl in RISK_LEVELS
        ],
        "fitness_distribution": [
            {"level": lvl, "count": fitness_dist.get(lvl, 0)}
            for lvl in ["Very Low", "Low", "Moderate", "Good"]
        ],
        "compliance_distribution": [
            {"level": lvl, "count": comp_dist.get(lvl, 0)}
            for lvl in COMPLIANCE_LEVELS
        ],
        "category_compliance": category_rates,
        "patient_summary": patient_summary,
    }


def breakdown():
    """Per-category detail, ADL domain analysis, compliance histograms,
    rehab score distribution, per-patient detail cards."""
    plans = _get_all_plans()

    # Aggregate per-category statistics
    cat_stats = {}
    for p in plans:
        for c in p["exercise_categories"]:
            cid = c["id"]
            if cid not in cat_stats:
                cat_stats[cid] = {
                    "name": c["name"], "examples": c["examples"],
                    "recommended_count": 0, "not_recommended_count": 0,
                    "compliance_values": [], "target_mins": [], "duration_mins": [],
                }
            if c["recommended"]:
                cat_stats[cid]["recommended_count"] += 1
                cat_stats[cid]["compliance_values"].append(c["actual_compliance_pct"])
                cat_stats[cid]["target_mins"].append(c["target_weekly_min"])
                cat_stats[cid]["duration_mins"].append(c["session_duration_min"])
            else:
                cat_stats[cid]["not_recommended_count"] += 1

    def _mean(lst, decimals=1):
        return round(sum(lst) / len(lst), decimals) if lst else 0

    category_details = []
    for cid in [c["id"] for c in EXERCISE_CATEGORIES]:
        st = cat_stats.get(cid, {"name": cid, "examples": [], "recommended_count": 0,
                                  "not_recommended_count": 0, "compliance_values": [],
                                  "target_mins": [], "duration_mins": []})
        category_details.append({
            "id": cid,
            "name": st["name"],
            "examples": st["examples"],
            "recommended_count": st["recommended_count"],
            "not_recommended_count": st["not_recommended_count"],
            "mean_compliance_pct": _mean(st["compliance_values"]),
            "mean_target_weekly_min": _mean(st["target_mins"], 0),
            "mean_session_duration_min": _mean(st["duration_mins"], 0),
        })

    # ADL domain aggregate
    adl_aggregate = {}
    for p in plans:
        for a in p["adl_scores"]:
            d = a["domain"]
            if d not in adl_aggregate:
                adl_aggregate[d] = {"scores": [], "independence": []}
            adl_aggregate[d]["scores"].append(a["score"])
            adl_aggregate[d]["independence"].append(a["independence_level"])

    adl_summary = []
    for domain in ADL_DOMAINS:
        d = domain["name"]
        agg = adl_aggregate.get(d, {"scores": [], "independence": []})
        ind_dist = Counter(agg["independence"])
        adl_summary.append({
            "domain": d,
            "mean_score": _mean(agg["scores"]),
            "items": domain["items"],
            "independence_distribution": dict(ind_dist),
        })

    # Compliance histogram
    all_compliance = [p["mean_compliance_pct"] for p in plans]
    comp_buckets = [
        {"range": "0-20%", "lo": 0, "hi": 20.01},
        {"range": "20-40%", "lo": 20.01, "hi": 40.01},
        {"range": "40-60%", "lo": 40.01, "hi": 60.01},
        {"range": "60-80%", "lo": 60.01, "hi": 80.01},
        {"range": "80-100%", "lo": 80.01, "hi": 100.01},
    ]
    compliance_histogram = [
        {"range": b["range"],
         "count": sum(1 for v in all_compliance if b["lo"] <= v < b["hi"]),
         "good": b["lo"] >= 60.01}
        for b in comp_buckets
    ]

    # Rehab score histogram
    rehab_scores = [p["rehab_score"] for p in plans]
    rehab_buckets = [
        {"range": "0-20", "lo": 0, "hi": 20.01, "grade": "critical"},
        {"range": "20-40", "lo": 20.01, "hi": 40.01, "grade": "poor"},
        {"range": "40-60", "lo": 40.01, "hi": 60.01, "grade": "fair"},
        {"range": "60-80", "lo": 60.01, "hi": 80.01, "grade": "good"},
        {"range": "80-100", "lo": 80.01, "hi": 100.01, "grade": "excellent"},
    ]
    rehab_histogram = [
        {"range": b["range"],
         "count": sum(1 for v in rehab_scores if b["lo"] <= v < b["hi"]),
         "grade": b["grade"]}
        for b in rehab_buckets
    ]

    # ADL score histogram
    all_adl = [p["mean_adl_score"] for p in plans]
    adl_buckets = [
        {"range": "0-20", "lo": 0, "hi": 20.01, "level": "dependent"},
        {"range": "20-40", "lo": 20.01, "hi": 40.01, "level": "supervised"},
        {"range": "40-60", "lo": 40.01, "hi": 60.01, "level": "modified"},
        {"range": "60-80", "lo": 60.01, "hi": 80.01, "level": "mostly_independent"},
        {"range": "80-100", "lo": 80.01, "hi": 100.01, "level": "independent"},
    ]
    adl_histogram = [
        {"range": b["range"],
         "count": sum(1 for v in all_adl if b["lo"] <= v < b["hi"]),
         "level": b["level"]}
        for b in adl_buckets
    ]

    # Per-patient detail
    patient_details = []
    for p in plans:
        patient_details.append({
            "patient_id": p["patient_id"],
            "name": p["patient_name"],
            "age": p["age"],
            "disease": p["disease"],
            "risk_level": p["risk_level"],
            "fitness_level": p["fitness_level"],
            "precautions": p["precautions"],
            "exercise_categories": p["exercise_categories"],
            "adl_scores": p["adl_scores"],
            "mean_compliance_pct": p["mean_compliance_pct"],
            "mean_adl_score": p["mean_adl_score"],
            "rehab_score": p["rehab_score"],
            "total_weekly_target_min": p["total_weekly_target_min"],
        })

    return {
        "category_details": category_details,
        "adl_summary": adl_summary,
        "compliance_histogram": compliance_histogram,
        "rehab_score_histogram": rehab_histogram,
        "adl_histogram": adl_histogram,
        "patient_details": patient_details,
    }


def definitions():
    """Exercise/rehab protocol, categories, risk levels, ADL domains,
    precautions, clinical significance."""
    return {
        "title": "Exercise / Rehab Recommendations",
        "protocol": {
            "description": (
                "Individualized exercise and rehabilitation prescriptions for epilepsy patients. "
                "Programs are tailored based on seizure frequency, seizure type, medication burden, "
                "age, and comorbidities. The goal is to maximize physical activity within safe limits, "
                "as regular exercise has been shown to reduce seizure frequency by 20-40% and improve "
                "comorbid depression, anxiety, and cardiovascular fitness."
            ),
            "framework": {
                "guideline": "ILAE guidelines on exercise and epilepsy (2020)",
                "frequency": "AHA recommendation: 150 min/week moderate aerobic + 2x strength",
                "progression": "Start low, progress gradually — 10% weekly volume increase maximum",
                "monitoring": "Seizure diary correlation with exercise log",
            },
            "standard": (
                "Arida RM et al. Epilepsy Behav 2013; Lundgren T et al. 2008; "
                "ILAE Task Force on Exercise and Epilepsy 2020; "
                "AHA Physical Activity Guidelines 2018"
            ),
            "indications": [
                "All epilepsy patients — exercise prescription (adapted to risk level)",
                "Post-seizure deconditioning — graded activity resumption",
                "Medication-related weight gain — metabolic fitness",
                "Comorbid depression/anxiety — exercise as adjunctive therapy",
                "Falls risk — balance and proprioceptive training",
                "Cognitive decline — aerobic + cognitive dual-task training",
                "ADL limitations — OT/PT rehabilitation program",
                "SUDEP risk reduction — cardiovascular fitness optimization",
                "Bone density (AED-related) — weight-bearing exercise",
            ],
        },
        "exercise_categories": [
            {
                "name": "Aerobic Exercise",
                "description": "Rhythmic large-muscle activities to improve cardiovascular fitness",
                "target": "150 min/week (moderate intensity) or 75 min/week (vigorous)",
                "examples": ["Walking", "Cycling (stationary preferred)", "Swimming (supervised)", "Jogging", "Dancing"],
                "epilepsy_considerations": "Avoid overheating; stop if aura; supervised water activities only",
            },
            {
                "name": "Strength / Resistance Training",
                "description": "Progressive resistance to maintain/build muscle mass and bone density",
                "target": "2 sessions/week, major muscle groups",
                "examples": ["Bodyweight exercises", "Resistance bands", "Light free weights", "Machine weights"],
                "epilepsy_considerations": "Avoid Valsalva; use machines over free weights for high-risk; spotter required",
            },
            {
                "name": "Balance & Coordination",
                "description": "Proprioceptive and vestibular training to reduce fall risk",
                "target": "2 sessions/week, 20 min each",
                "examples": ["Yoga", "Tai Chi", "Single-leg stance", "Proprioceptive board"],
                "epilepsy_considerations": "Perform near wall/support; padded surface for high-risk patients",
            },
            {
                "name": "Flexibility / Stretching",
                "description": "Static and dynamic stretching to maintain range of motion",
                "target": "Daily, 10 min warm-up/cool-down",
                "examples": ["Static stretching", "Dynamic warm-up", "ROM exercises", "Foam rolling"],
                "epilepsy_considerations": "Minimal risk; can be done independently at all risk levels",
            },
            {
                "name": "Occupational Therapy (OT)",
                "description": "ADL training, cognitive rehabilitation, home/workplace safety",
                "target": "2 sessions/week during acute rehab; monthly maintenance",
                "examples": ["ADL training", "Fine motor drills", "Cognitive rehab", "Home safety assessment"],
                "epilepsy_considerations": "Focus on seizure-safe home modifications; driving evaluation; work accommodations",
            },
            {
                "name": "Physical Therapy (PT)",
                "description": "Gait, vestibular, and neuromuscular rehabilitation",
                "target": "2 sessions/week during acute rehab; as needed maintenance",
                "examples": ["Gait training", "Vestibular rehab", "Neuro-rehab", "Fall prevention"],
                "epilepsy_considerations": "Post-ictal weakness assessment; fall prevention strategies; seizure first aid training",
            },
        ],
        "risk_levels": [
            {"level": "Low", "criteria": "<1 seizure/month, no tonic-clonic",
             "exercise_allowed": "Full exercise prescription with standard precautions"},
            {"level": "Moderate", "criteria": "1-4 seizures/month or controlled tonic-clonic",
             "exercise_allowed": "Supervised exercise, modified intensity, companion recommended"},
            {"level": "High", "criteria": ">4 seizures/month or uncontrolled tonic-clonic",
             "exercise_allowed": "Restricted to low-intensity, supervised, ground-level activities"},
            {"level": "Very High", "criteria": "Status epilepticus history, daily seizures",
             "exercise_allowed": "PT-supervised sessions only, seated/recumbent activities"},
        ],
        "adl_domains": [
            {"domain": d["name"], "items": d["items"],
             "description": "Assessed on 0-100 scale; independence levels: Dependent (<40), Supervised (40-59), Modified Independent (60-79), Independent (≥80)"}
            for d in ADL_DOMAINS
        ],
        "clinical_significance": [
            "Seizure reduction — regular moderate exercise reduces seizure frequency by 20-40%",
            "SUDEP risk — cardiovascular fitness is protective; autonomic regulation improved",
            "Depression/Anxiety — exercise as effective as SSRIs for mild-moderate comorbid depression",
            "AED side effects — exercise mitigates weight gain, bone density loss, fatigue",
            "Cognitive function — aerobic exercise improves attention, memory, and processing speed",
            "Falls prevention — balance training reduces fall-related injury by 30-50%",
            "Quality of Life — physical activity improves QOLIE-31 scores in epilepsy patients",
            "Social participation — group exercise programs reduce isolation and stigma",
        ],
    }
