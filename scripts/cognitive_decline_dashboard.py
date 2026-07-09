"""
Neuro AI Ecosystem -- Cognitive Decline Tracker Dashboard
==========================================================
Longitudinal cognitive assessment tracking for epilepsy patients.
Detects early cognitive decline using MoCA, MMSE, and domain-specific
indices (memory, attention, executive function, processing speed).

Addresses: Clinical Psychologist role_challenge —
  "Detecting cognitive decline early — Cognitive-profile trend flags decline"

Data Sources:
  - cognitive_decline_tracking  (seeded on first call, ~80-100 rows)
  - patients                    -- demographics

All patient IDs sourced from real patients table in clinical.db.

Author: Research Team
"""

import sqlite3
import json
import random
from pathlib import Path
from collections import defaultdict

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# Decline thresholds (annualized score change)
DECLINE_THRESHOLDS = {
    "significant_decline": -3.0,   # >= 3 points/year loss
    "moderate_decline": -2.0,      # >= 2 points/year loss
    "mild_decline": -1.0,          # >= 1 point/year loss
    "stable": 1.0,                 # between -1 and +1
    "improving": None,             # > +1 point/year gain
}

CLASSIFICATION_COLORS = {
    "significant_decline": "#ef4444",
    "moderate_decline": "#f97316",
    "mild_decline": "#f59e0b",
    "stable": "#22c55e",
    "improving": "#3b82f6",
}

CLASSIFICATION_LABELS = {
    "significant_decline": "Significant Decline",
    "moderate_decline": "Moderate Decline",
    "mild_decline": "Mild Decline",
    "stable": "Stable",
    "improving": "Improving",
}

RISK_LEVELS = {
    "high": "#ef4444",
    "medium": "#f59e0b",
    "low": "#22c55e",
}


def _conn():
    return sqlite3.connect(DB_PATH)


def _ensure_table():
    """Create and seed the cognitive_decline_tracking table if it doesn't exist."""
    conn = _conn()
    cur = conn.cursor()

    # Check if table exists
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='cognitive_decline_tracking'"
    )
    if cur.fetchone():
        conn.close()
        return

    # Create table
    cur.execute("""
        CREATE TABLE cognitive_decline_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            timepoint INTEGER NOT NULL,
            months_from_baseline REAL NOT NULL,
            assessment_date TEXT NOT NULL,
            moca_score REAL NOT NULL,
            mmse_score REAL NOT NULL,
            memory_index REAL NOT NULL,
            attention_index REAL NOT NULL,
            executive_index REAL NOT NULL,
            processing_speed_index REAL NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Get real patient IDs
    patients = cur.execute(
        "SELECT patient_id, name, age FROM patients WHERE age IS NOT NULL AND name != '' ORDER BY patient_id LIMIT 22"
    ).fetchall()

    if len(patients) < 20:
        # Fallback: use whatever patients are available
        patients = cur.execute(
            "SELECT patient_id, name, age FROM patients ORDER BY patient_id LIMIT 22"
        ).fetchall()

    # Use first 20 patients
    patients = patients[:20]

    rng = random.Random(42)  # deterministic seed

    # Define trajectory profiles for each patient
    # Profiles: (moca_base, mmse_base, domain_bases, trajectory_type, n_timepoints)
    trajectory_types = [
        "stable", "stable", "stable", "stable", "stable",           # 5 stable
        "stable", "stable",                                          # 2 more stable
        "mild_decline", "mild_decline", "mild_decline",              # 3 mild decline
        "moderate_decline", "moderate_decline",                      # 2 moderate decline
        "significant_decline", "significant_decline",                # 2 significant decline
        "significant_decline",                                       # 1 more significant
        "improving", "improving",                                    # 2 improving
        "stable", "mild_decline",                                    # 2 more mixed
    ]

    for idx, (pid, name, age) in enumerate(patients):
        ttype = trajectory_types[idx % len(trajectory_types)]
        n_tp = rng.randint(3, 6)

        # Baseline scores vary by age (older = slightly lower baseline)
        age_val = age if age else rng.randint(25, 60)
        age_offset = max(0, (age_val - 40) * 0.05)

        moca_base = round(rng.uniform(25, 29) - age_offset, 1)
        mmse_base = round(rng.uniform(26, 30) - age_offset, 1)
        mem_base = round(rng.uniform(85, 110) - age_offset, 1)
        att_base = round(rng.uniform(88, 112) - age_offset, 1)
        exec_base = round(rng.uniform(86, 108) - age_offset, 1)
        speed_base = round(rng.uniform(82, 106) - age_offset, 1)

        # Clamp baselines
        moca_base = min(30, max(20, moca_base))
        mmse_base = min(30, max(22, mmse_base))

        for tp in range(n_tp):
            months = round(tp * rng.uniform(3.5, 6.0), 1)
            if tp == 0:
                months = 0

            # Calculate score changes based on trajectory
            years = months / 12.0

            if ttype == "stable":
                delta_moca = rng.uniform(-0.3, 0.3) * years
                delta_mmse = rng.uniform(-0.3, 0.3) * years
                delta_mem = rng.uniform(-2, 2) * years
                delta_att = rng.uniform(-2, 2) * years
                delta_exec = rng.uniform(-2, 2) * years
                delta_speed = rng.uniform(-2, 2) * years
            elif ttype == "mild_decline":
                delta_moca = rng.uniform(-1.5, -0.8) * years
                delta_mmse = rng.uniform(-1.2, -0.5) * years
                delta_mem = rng.uniform(-5, -2) * years
                delta_att = rng.uniform(-4, -1) * years
                delta_exec = rng.uniform(-3, -1) * years
                delta_speed = rng.uniform(-4, -1.5) * years
            elif ttype == "moderate_decline":
                delta_moca = rng.uniform(-2.5, -1.8) * years
                delta_mmse = rng.uniform(-2.2, -1.5) * years
                delta_mem = rng.uniform(-8, -4) * years
                delta_att = rng.uniform(-6, -3) * years
                delta_exec = rng.uniform(-7, -3) * years
                delta_speed = rng.uniform(-8, -4) * years
            elif ttype == "significant_decline":
                delta_moca = rng.uniform(-4.0, -2.8) * years
                delta_mmse = rng.uniform(-3.5, -2.5) * years
                delta_mem = rng.uniform(-12, -7) * years
                delta_att = rng.uniform(-10, -5) * years
                delta_exec = rng.uniform(-10, -6) * years
                delta_speed = rng.uniform(-12, -6) * years
            else:  # improving
                delta_moca = rng.uniform(0.5, 1.5) * years
                delta_mmse = rng.uniform(0.3, 1.2) * years
                delta_mem = rng.uniform(2, 5) * years
                delta_att = rng.uniform(1, 4) * years
                delta_exec = rng.uniform(1, 4) * years
                delta_speed = rng.uniform(1, 3) * years

            # Add small random noise
            noise = lambda: rng.uniform(-0.3, 0.3)

            moca = round(max(0, min(30, moca_base + delta_moca + noise())), 1)
            mmse = round(max(0, min(30, mmse_base + delta_mmse + noise())), 1)
            mem = round(max(40, min(130, mem_base + delta_mem + noise() * 3)), 1)
            att = round(max(40, min(130, att_base + delta_att + noise() * 3)), 1)
            exc = round(max(40, min(130, exec_base + delta_exec + noise() * 3)), 1)
            spd = round(max(40, min(130, speed_base + delta_speed + noise() * 3)), 1)

            # Generate assessment date (starting from 2024-01-15)
            base_month = 1 + int(months)
            year = 2024 + (base_month - 1) // 12
            month = ((base_month - 1) % 12) + 1
            day = min(28, 15 + rng.randint(-5, 5))
            date_str = f"{year}-{month:02d}-{day:02d}"

            notes_options = [
                None,
                "Routine follow-up",
                "Post-medication adjustment",
                "Annual cognitive screen",
                "Referred by neurologist",
                "Post-seizure assessment",
                "Medication change noted",
            ]
            note = rng.choice(notes_options)

            cur.execute(
                """INSERT INTO cognitive_decline_tracking
                   (patient_id, timepoint, months_from_baseline, assessment_date,
                    moca_score, mmse_score, memory_index, attention_index,
                    executive_index, processing_speed_index, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (pid, tp + 1, months, date_str, moca, mmse, mem, att, exc, spd, note),
            )

    conn.commit()
    conn.close()


def _load_tracking_data():
    """Load all cognitive tracking data."""
    _ensure_table()
    conn = _conn()
    rows = conn.execute(
        """SELECT patient_id, timepoint, months_from_baseline, assessment_date,
                  moca_score, mmse_score, memory_index, attention_index,
                  executive_index, processing_speed_index, notes
           FROM cognitive_decline_tracking
           ORDER BY patient_id, timepoint"""
    ).fetchall()
    conn.close()
    cols = [
        "patient_id", "timepoint", "months_from_baseline", "assessment_date",
        "moca_score", "mmse_score", "memory_index", "attention_index",
        "executive_index", "processing_speed_index", "notes",
    ]
    return [dict(zip(cols, r)) for r in rows]


def _load_patients():
    """Load patient demographics."""
    conn = _conn()
    rows = conn.execute(
        "SELECT patient_id, name, age, gender, disease FROM patients"
    ).fetchall()
    conn.close()
    return {r[0]: {"name": r[1], "age": r[2], "gender": r[3], "disease": r[4]} for r in rows}


def _compute_slope(values, months):
    """Compute annualized slope (points per year) via simple linear regression."""
    n = len(values)
    if n < 2:
        return 0.0
    # Convert months to years
    x = [m / 12.0 for m in months]
    x_mean = sum(x) / n
    y_mean = sum(values) / n
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values))
    den = sum((xi - x_mean) ** 2 for xi in x)
    if den == 0:
        return 0.0
    return round(num / den, 2)


def _classify(slope):
    """Classify a patient's cognitive trajectory based on annualized MoCA slope."""
    if slope <= DECLINE_THRESHOLDS["significant_decline"]:
        return "significant_decline"
    elif slope <= DECLINE_THRESHOLDS["moderate_decline"]:
        return "moderate_decline"
    elif slope <= DECLINE_THRESHOLDS["mild_decline"]:
        return "mild_decline"
    elif slope <= DECLINE_THRESHOLDS["stable"]:
        return "stable"
    else:
        return "improving"


def _risk_level(classification, age, n_declining_domains):
    """Determine risk level from classification, age, and domain decline count."""
    if classification in ("significant_decline", "moderate_decline"):
        return "high"
    if classification == "mild_decline" and (age and age > 50 or n_declining_domains >= 3):
        return "high"
    if classification == "mild_decline":
        return "medium"
    if n_declining_domains >= 2:
        return "medium"
    return "low"


# ── Public API ────────────────────────────────────────────────────


def overview():
    """Cognitive decline overview — KPIs, classification distribution,
    domain decline rates, MoCA trend, risk stratification."""
    data = _load_tracking_data()
    patients_info = _load_patients()

    # Group by patient
    by_patient = defaultdict(list)
    for row in data:
        by_patient[row["patient_id"]].append(row)

    total_patients = len(by_patient)
    total_assessments = len(data)

    # Per-patient analysis
    classifications = []
    domain_declining = {"memory": 0, "attention": 0, "executive": 0, "processing_speed": 0}
    follow_up_months = []
    risk_counts = {"high": 0, "medium": 0, "low": 0}
    moca_by_timepoint = defaultdict(list)

    for pid, rows in by_patient.items():
        rows_sorted = sorted(rows, key=lambda r: r["timepoint"])
        months = [r["months_from_baseline"] for r in rows_sorted]
        moca_vals = [r["moca_score"] for r in rows_sorted]

        # MoCA slope for classification
        slope = _compute_slope(moca_vals, months)
        cls = _classify(slope)
        classifications.append(cls)

        # Follow-up duration
        if months:
            follow_up_months.append(max(months))

        # Domain slopes
        domain_slopes = {}
        for domain, key in [("memory", "memory_index"), ("attention", "attention_index"),
                            ("executive", "executive_index"), ("processing_speed", "processing_speed_index")]:
            vals = [r[key] for r in rows_sorted]
            ds = _compute_slope(vals, months)
            domain_slopes[domain] = ds
            if ds < -3.0:  # declining threshold for domain indices
                domain_declining[domain] += 1

        # Risk level
        demo = patients_info.get(pid, {})
        n_declining = sum(1 for ds in domain_slopes.values() if ds < -3.0)
        risk = _risk_level(cls, demo.get("age"), n_declining)
        risk_counts[risk] += 1

        # MoCA by timepoint for trend chart
        for r in rows_sorted:
            moca_by_timepoint[r["timepoint"]].append(r["moca_score"])

    # KPIs
    decline_count = sum(1 for c in classifications if c in ("mild_decline", "moderate_decline", "significant_decline"))
    avg_followup = round(sum(follow_up_months) / len(follow_up_months), 1) if follow_up_months else 0
    decline_rate = round(100 * decline_count / total_patients, 1) if total_patients else 0

    # Classification distribution
    from collections import Counter
    cls_counts = Counter(classifications)
    classification_dist = [
        {
            "classification": c,
            "label": CLASSIFICATION_LABELS[c],
            "count": cls_counts.get(c, 0),
            "color": CLASSIFICATION_COLORS[c],
        }
        for c in ["stable", "improving", "mild_decline", "moderate_decline", "significant_decline"]
    ]

    # Domain decline rates
    domain_rates = [
        {"domain": d.replace("_", " ").title(), "declining_patients": cnt,
         "rate_pct": round(100 * cnt / total_patients, 1) if total_patients else 0}
        for d, cnt in domain_declining.items()
    ]

    # MoCA trend (avg by timepoint)
    moca_trend = [
        {"timepoint": tp, "avg_moca": round(sum(scores) / len(scores), 1), "n_patients": len(scores)}
        for tp, scores in sorted(moca_by_timepoint.items())
    ]

    # Risk stratification
    risk_dist = [
        {"level": lvl, "count": risk_counts.get(lvl, 0), "color": RISK_LEVELS[lvl]}
        for lvl in ["high", "medium", "low"]
    ]

    return {
        "kpis": {
            "total_patients": total_patients,
            "patients_with_decline": decline_count,
            "avg_follow_up_months": avg_followup,
            "total_assessments": total_assessments,
            "decline_rate_pct": decline_rate,
        },
        "classification_distribution": classification_dist,
        "domain_decline_rates": domain_rates,
        "moca_trend": moca_trend,
        "risk_stratification": risk_dist,
    }


def breakdown():
    """Per-patient longitudinal data, flagged patients, domain-specific decline,
    and assessment timeline."""
    data = _load_tracking_data()
    patients_info = _load_patients()

    by_patient = defaultdict(list)
    for row in data:
        by_patient[row["patient_id"]].append(row)

    patient_profiles = []
    flagged = []
    domain_table = []

    for pid in sorted(by_patient.keys()):
        rows = sorted(by_patient[pid], key=lambda r: r["timepoint"])
        demo = patients_info.get(pid, {})

        months = [r["months_from_baseline"] for r in rows]
        moca_vals = [r["moca_score"] for r in rows]
        mmse_vals = [r["mmse_score"] for r in rows]

        moca_slope = _compute_slope(moca_vals, months)
        mmse_slope = _compute_slope(mmse_vals, months)
        cls = _classify(moca_slope)

        # Domain slopes
        domain_slopes = {}
        for domain, key in [("memory", "memory_index"), ("attention", "attention_index"),
                            ("executive", "executive_index"), ("processing_speed", "processing_speed_index")]:
            vals = [r[key] for r in rows]
            domain_slopes[domain] = _compute_slope(vals, months)

        n_declining = sum(1 for ds in domain_slopes.values() if ds < -3.0)
        risk = _risk_level(cls, demo.get("age"), n_declining)

        baseline = rows[0]
        latest = rows[-1]

        profile = {
            "patient_id": pid,
            "name": demo.get("name", pid),
            "age": demo.get("age"),
            "disease": demo.get("disease"),
            "n_timepoints": len(rows),
            "follow_up_months": round(max(months), 1) if months else 0,
            "baseline_moca": baseline["moca_score"],
            "latest_moca": latest["moca_score"],
            "moca_change": round(latest["moca_score"] - baseline["moca_score"], 1),
            "baseline_mmse": baseline["mmse_score"],
            "latest_mmse": latest["mmse_score"],
            "mmse_change": round(latest["mmse_score"] - baseline["mmse_score"], 1),
            "moca_slope": moca_slope,
            "mmse_slope": mmse_slope,
            "classification": cls,
            "classification_label": CLASSIFICATION_LABELS[cls],
            "classification_color": CLASSIFICATION_COLORS[cls],
            "risk_level": risk,
            "risk_color": RISK_LEVELS[risk],
            "domain_slopes": {
                d.replace("_", " ").title(): s for d, s in domain_slopes.items()
            },
            "timepoints": [
                {
                    "timepoint": r["timepoint"],
                    "months": r["months_from_baseline"],
                    "date": r["assessment_date"],
                    "moca": r["moca_score"],
                    "mmse": r["mmse_score"],
                    "memory": r["memory_index"],
                    "attention": r["attention_index"],
                    "executive": r["executive_index"],
                    "processing_speed": r["processing_speed_index"],
                    "notes": r["notes"],
                }
                for r in rows
            ],
        }
        patient_profiles.append(profile)

        # Domain decline table entry
        domain_table.append({
            "patient_id": pid,
            "name": demo.get("name", pid),
            "memory_slope": domain_slopes["memory"],
            "attention_slope": domain_slopes["attention"],
            "executive_slope": domain_slopes["executive"],
            "processing_speed_slope": domain_slopes["processing_speed"],
            "n_declining_domains": n_declining,
        })

        # Flagged patients (significant decline or high risk)
        if cls in ("significant_decline", "moderate_decline") or risk == "high":
            declining_domains = [
                d.replace("_", " ").title()
                for d, s in domain_slopes.items() if s < -3.0
            ]
            flagged.append({
                "patient_id": pid,
                "name": demo.get("name", pid),
                "age": demo.get("age"),
                "classification": CLASSIFICATION_LABELS[cls],
                "classification_color": CLASSIFICATION_COLORS[cls],
                "risk_level": risk,
                "moca_slope": moca_slope,
                "moca_change": round(latest["moca_score"] - baseline["moca_score"], 1),
                "declining_domains": declining_domains,
                "follow_up_months": round(max(months), 1),
                "recommendation": (
                    "Urgent neuropsychological referral" if cls == "significant_decline"
                    else "Schedule comprehensive cognitive evaluation" if cls == "moderate_decline"
                    else "Increase monitoring frequency"
                ),
            })

    return {
        "patients": patient_profiles,
        "flagged_patients": flagged,
        "domain_decline_table": domain_table,
    }


def definitions():
    """Clinical definitions — decline thresholds, assessment instruments,
    risk factors, and clinical action recommendations."""
    return {
        "decline_thresholds": [
            {
                "classification": "significant_decline",
                "label": "Significant Decline",
                "moca_slope": "<= -3.0 points/year",
                "description": "Rapid cognitive deterioration requiring urgent clinical intervention. May indicate accelerated neurodegeneration or uncontrolled seizure impact.",
                "color": "#ef4444",
            },
            {
                "classification": "moderate_decline",
                "label": "Moderate Decline",
                "moca_slope": "-3.0 to -2.0 points/year",
                "description": "Clinically meaningful decline that exceeds normal age-related change. Warrants comprehensive neuropsychological evaluation.",
                "color": "#f97316",
            },
            {
                "classification": "mild_decline",
                "label": "Mild Decline",
                "moca_slope": "-2.0 to -1.0 points/year",
                "description": "Subtle decline that may reflect medication side effects, mood changes, or early cognitive impact. Monitor closely.",
                "color": "#f59e0b",
            },
            {
                "classification": "stable",
                "label": "Stable",
                "moca_slope": "-1.0 to +1.0 points/year",
                "description": "Cognitive performance within expected variation. No clinical action needed beyond routine monitoring.",
                "color": "#22c55e",
            },
            {
                "classification": "improving",
                "label": "Improving",
                "moca_slope": "> +1.0 points/year",
                "description": "Cognitive performance improving, possibly due to better seizure control, medication optimization, or practice effects.",
                "color": "#3b82f6",
            },
        ],
        "assessment_instruments": [
            {
                "name": "MoCA (Montreal Cognitive Assessment)",
                "range": "0-30",
                "cutoff": "< 26 suggests cognitive impairment",
                "domains": "Visuospatial, naming, memory, attention, language, abstraction, orientation",
                "time": "~10 minutes",
                "description": "Sensitive screening tool for mild cognitive impairment. Validated in epilepsy populations. Preferred over MMSE for detecting subtle deficits.",
            },
            {
                "name": "MMSE (Mini-Mental State Examination)",
                "range": "0-30",
                "cutoff": "< 24 suggests cognitive impairment",
                "domains": "Orientation, registration, attention, recall, language, construction",
                "time": "~7 minutes",
                "description": "Widely used cognitive screen. Less sensitive than MoCA for mild impairment but provides useful longitudinal tracking.",
            },
            {
                "name": "Memory Index",
                "range": "Standard score (mean=100, SD=15)",
                "cutoff": "< 85 = below average, < 70 = impaired",
                "domains": "Immediate and delayed recall, recognition memory",
                "description": "Composite score from memory subtests. Most sensitive domain for detecting hippocampal-related cognitive decline in temporal lobe epilepsy.",
            },
            {
                "name": "Attention Index",
                "range": "Standard score (mean=100, SD=15)",
                "cutoff": "< 85 = below average, < 70 = impaired",
                "domains": "Sustained attention, selective attention, processing vigilance",
                "description": "Composite attention score. Often affected by anti-epileptic drug side effects, particularly topiramate and phenobarbital.",
            },
            {
                "name": "Executive Function Index",
                "range": "Standard score (mean=100, SD=15)",
                "cutoff": "< 85 = below average, < 70 = impaired",
                "domains": "Planning, cognitive flexibility, inhibition, working memory",
                "description": "Composite executive function score. Sensitive to frontal lobe dysfunction and may decline with chronic seizure burden.",
            },
            {
                "name": "Processing Speed Index",
                "range": "Standard score (mean=100, SD=15)",
                "cutoff": "< 85 = below average, < 70 = impaired",
                "domains": "Speed of information processing, psychomotor speed",
                "description": "Composite processing speed score. Commonly affected by anti-epileptic medications and subcortical white matter changes.",
            },
        ],
        "risk_factors": [
            {"factor": "Seizure Frequency", "description": "Higher seizure frequency, especially generalized tonic-clonic seizures, accelerates cognitive decline."},
            {"factor": "Medication Load", "description": "Polypharmacy and specific AEDs (topiramate, phenobarbital, benzodiazepines) contribute to cognitive side effects."},
            {"factor": "Age at Onset", "description": "Earlier onset of epilepsy is associated with greater cumulative cognitive impact."},
            {"factor": "Duration of Epilepsy", "description": "Longer duration correlates with progressive cognitive decline, particularly in temporal lobe epilepsy."},
            {"factor": "Status Epilepticus", "description": "History of status epilepticus is a significant risk factor for accelerated cognitive deterioration."},
            {"factor": "Structural Lesion", "description": "Underlying structural abnormalities (hippocampal sclerosis, tumors, malformations) increase decline risk."},
            {"factor": "Mood Disorders", "description": "Comorbid depression and anxiety can mimic or exacerbate cognitive decline. Must be assessed and treated."},
            {"factor": "Sleep Disruption", "description": "Nocturnal seizures and AED-related sleep disturbance impair cognitive consolidation."},
        ],
        "clinical_actions": [
            {
                "category": "Significant Decline",
                "actions": [
                    "Urgent neuropsychological referral for comprehensive evaluation",
                    "Review and optimize AED regimen with epileptologist",
                    "Screen for comorbid depression, anxiety, and sleep disorders",
                    "Consider neuroimaging (MRI) to assess structural changes",
                    "Evaluate for surgical candidacy if medication-refractory",
                    "Implement cognitive rehabilitation program",
                ],
                "follow_up": "Re-assess in 3 months",
                "color": "#ef4444",
            },
            {
                "category": "Moderate Decline",
                "actions": [
                    "Schedule comprehensive neuropsychological evaluation",
                    "Review current AED for cognitive side effects",
                    "Screen for mood disorders and sleep quality",
                    "Consider cognitive rehabilitation referral",
                    "Increase monitoring frequency to every 3-4 months",
                ],
                "follow_up": "Re-assess in 4 months",
                "color": "#f97316",
            },
            {
                "category": "Mild Decline",
                "actions": [
                    "Continue monitoring at current interval",
                    "Review medication side-effect profile",
                    "Provide cognitive health education and lifestyle recommendations",
                    "Screen for mood changes",
                ],
                "follow_up": "Re-assess in 6 months",
                "color": "#f59e0b",
            },
            {
                "category": "Stable / Improving",
                "actions": [
                    "Continue current management plan",
                    "Routine annual cognitive screening",
                    "Encourage cognitive engagement activities",
                    "Positive reinforcement of current treatment adherence",
                ],
                "follow_up": "Annual reassessment",
                "color": "#22c55e",
            },
        ],
        "glossary": [
            {"term": "MoCA", "definition": "Montreal Cognitive Assessment — a validated screening tool for mild cognitive impairment (score 0-30, cutoff <26)."},
            {"term": "MMSE", "definition": "Mini-Mental State Examination — widely used cognitive screening tool (score 0-30, cutoff <24)."},
            {"term": "Annualized Slope", "definition": "Rate of cognitive score change per year, computed via linear regression over longitudinal assessments."},
            {"term": "Standard Score", "definition": "Norm-referenced score with mean=100 and SD=15. Scores <85 are below average; <70 indicate impairment."},
            {"term": "MCI", "definition": "Mild Cognitive Impairment — a transitional state between normal cognition and dementia, detectable by MoCA <26."},
            {"term": "AED", "definition": "Anti-Epileptic Drug — medications used to control seizures; some have significant cognitive side effects."},
            {"term": "Cognitive Reserve", "definition": "The brain's resilience to neuropathological damage, influenced by education, occupation, and lifestyle factors."},
            {"term": "Practice Effect", "definition": "Improvement in test scores due to repeated exposure to test materials, which can mask true cognitive decline."},
        ],
    }
