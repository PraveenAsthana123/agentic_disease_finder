"""
Neuro AI Ecosystem -- Recovery Trajectory Forecast Dashboard
=============================================================
Predicts which patients need intensive rehabilitation by analysing
functional recovery trends from real clinical.db data.

Trajectory Classification:
  improving  -- daily_function_rating slope > +0.1 per assessment
  stable     -- daily_function_rating slope between -0.1 and +0.1
  declining  -- daily_function_rating slope < -0.1 per assessment

Intensive Rehab Recommendation:
  Patients with a declining trajectory AND at least one of:
    - PHQ-9 score > 10   (moderate-severe depression)
    - MoCA score < 25    (cognitive impairment)
    - Fatigue level > 6  (high fatigue)
  are flagged for intensive rehabilitation.

Data Sources:
  - pro_outcomes   (180 rows, 30 patients) -- monthly PRO assessments
  - patients       -- demographics (patient_id, name, age, gender, disease)
  - assessments    (423 rows) -- instrument-based clinical assessments
  - neuropsych     (37 rows)  -- neuropsychological evaluations

All patient IDs are sourced from the real patients table in clinical.db.
Hashlib-based deterministic seeding used for any supplementary data.

Author: Research Team
"""

import sqlite3
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# -- Constants ----------------------------------------------------------------

TRAJECTORY_CLASSES = ["improving", "stable", "declining"]

RISK_FACTOR_LABELS = {
    "high_phq9":      "PHQ-9 > 10 (Depression)",
    "low_moca":       "MoCA < 25 (Cognitive Impairment)",
    "high_fatigue":   "Fatigue > 6",
    "poor_sleep":     "Sleep < 6 hours",
}


def _conn():
    return sqlite3.connect(DB_PATH)


def _deterministic_seed(patient_id: str, item_id: str) -> float:
    """Return a reproducible float in [0, 1) from patient + item identifiers."""
    h = hashlib.sha256(f"recovery:{patient_id}:{item_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _seed_int(patient_id: str, item_id: str, lo: int, hi: int) -> int:
    """Return a reproducible int in [lo, hi] from patient + item identifiers."""
    return lo + int(_deterministic_seed(patient_id, item_id) * (hi - lo + 1)) % (hi - lo + 1)


def _safe_json(raw):
    """Safely parse a JSON string, returning {} on failure."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _avg(values):
    """Return mean of a list, rounded to 2 decimal places, or 0 if empty."""
    return round(sum(values) / len(values), 2) if values else 0


def _real_patients(conn):
    """Return list of patient dicts from the real patients table."""
    c = conn.cursor()
    c.execute("""
        SELECT patient_id, name, age, gender, disease, department
        FROM patients
        WHERE patient_id IS NOT NULL
        ORDER BY patient_id
    """)
    rows = []
    for row in c.fetchall():
        rows.append({
            "patient_id": row[0],
            "name":       row[1] or f"Patient {row[0]}",
            "age":        row[2],
            "gender":     row[3],
            "disease":    row[4] or "unspecified",
            "department": row[5],
        })
    return rows


def _load_pro_outcomes(conn):
    """Load all pro_outcomes rows, parsing fields_json into dicts."""
    c = conn.cursor()
    c.execute("SELECT id, patient_id, fields_json, created_at FROM pro_outcomes ORDER BY created_at")
    results = []
    for row in c.fetchall():
        fields = _safe_json(row[2])
        fields["_row_id"] = row[0]
        fields["_patient_id"] = row[1]
        fields["_created_at"] = row[3]
        results.append(fields)
    return results


def _compute_slope(values):
    """Compute linear slope of values over their index positions.

    Uses simple least-squares regression: slope = Cov(x,y) / Var(x)
    where x = [0, 1, 2, ...] and y = values.
    Returns 0.0 if fewer than 2 data points.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _classify_trajectory(slope):
    """Classify trajectory based on slope threshold."""
    if slope > 0.1:
        return "improving"
    elif slope < -0.1:
        return "declining"
    else:
        return "stable"


def _build_patient_trajectories(pro_records, patient_info_map):
    """Build per-patient trajectory analysis from PRO outcome records.

    Returns a list of dicts, one per patient with PRO data, containing
    trajectory class, slope, latest values, and rehab recommendation.
    """
    from collections import defaultdict

    # Group records by patient
    patient_records = defaultdict(list)
    for r in pro_records:
        pid = r.get("patient_id", r.get("_patient_id"))
        patient_records[pid].append(r)

    trajectories = []
    for pid in sorted(patient_records.keys()):
        recs = sorted(
            patient_records[pid],
            key=lambda x: x.get("assessment_date", x.get("_created_at", "")),
        )

        # Extract daily_function_rating series
        daily_ratings = [
            r["daily_function_rating"]
            for r in recs
            if r.get("daily_function_rating") is not None
        ]

        slope = _compute_slope(daily_ratings)
        trajectory_class = _classify_trajectory(slope)

        # Latest record values
        latest = recs[-1]
        latest_daily = latest.get("daily_function_rating")
        latest_social = latest.get("social_function_rating")
        latest_moca = latest.get("moca_score")
        latest_phq9 = latest.get("phq9_score")
        latest_fatigue = latest.get("fatigue_level")
        latest_sleep = latest.get("sleep_hours")
        latest_gad7 = latest.get("gad7_score")
        latest_mood = latest.get("mood_rating")

        # Intensive rehab recommendation
        intensive_rehab = False
        if trajectory_class == "declining":
            if (latest_phq9 is not None and latest_phq9 > 10) or \
               (latest_moca is not None and latest_moca < 25) or \
               (latest_fatigue is not None and latest_fatigue > 6):
                intensive_rehab = True

        # Risk factors present
        risk_factors = []
        if latest_phq9 is not None and latest_phq9 > 10:
            risk_factors.append("high_phq9")
        if latest_moca is not None and latest_moca < 25:
            risk_factors.append("low_moca")
        if latest_fatigue is not None and latest_fatigue > 6:
            risk_factors.append("high_fatigue")
        if latest_sleep is not None and latest_sleep < 6:
            risk_factors.append("poor_sleep")

        info = patient_info_map.get(pid, {})

        trajectories.append({
            "patient_id":                pid,
            "name":                      info.get("name", f"Patient {pid}"),
            "age":                       info.get("age"),
            "gender":                    info.get("gender"),
            "disease":                   info.get("disease", "unspecified"),
            "department":                info.get("department"),
            "trajectory_class":          trajectory_class,
            "slope":                     round(slope, 4),
            "num_assessments":           len(recs),
            "latest_daily_function":     latest_daily,
            "latest_social_function":    latest_social,
            "latest_moca":               latest_moca,
            "latest_phq9":               latest_phq9,
            "latest_fatigue":            latest_fatigue,
            "latest_sleep":              latest_sleep,
            "latest_gad7":               latest_gad7,
            "latest_mood":               latest_mood,
            "risk_factors":              risk_factors,
            "intensive_rehab_recommended": intensive_rehab,
        })

    return trajectories


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def overview(patient_id: Optional[str] = None) -> dict:
    """
    Recovery trajectory forecast overview.

    Returns:
        kpis                     -- total patients tracked, avg trend, patients
                                    needing intensive rehab, avg recovery score
        trajectory_distribution  -- pie chart data: {name, count} per class
        monthly_avg_function     -- line chart: {month, avg_daily_function} over time
        risk_factor_bar          -- bar chart: risk factors correlated with declining
        patient_trajectory_summary -- per-patient: id, name, trajectory, slope
    """
    conn = _conn()
    patients = _real_patients(conn)
    patient_info_map = {p["patient_id"]: p for p in patients}
    pro_records = _load_pro_outcomes(conn)
    conn.close()

    if patient_id:
        pro_records = [r for r in pro_records if r.get("_patient_id") == patient_id]

    if not pro_records:
        return {
            "kpis": {
                "total_patients_tracked": 0,
                "avg_daily_function_trend": "no_data",
                "patients_needing_intensive_rehab": 0,
                "avg_recovery_score": 0.0,
            },
            "trajectory_distribution": [],
            "monthly_avg_function": [],
            "risk_factor_bar": [],
            "patient_trajectory_summary": [],
        }

    trajectories = _build_patient_trajectories(pro_records, patient_info_map)

    # -- KPIs -----------------------------------------------------------------
    total_patients = len(trajectories)
    improving_count = sum(1 for t in trajectories if t["trajectory_class"] == "improving")
    declining_count = sum(1 for t in trajectories if t["trajectory_class"] == "declining")
    stable_count = sum(1 for t in trajectories if t["trajectory_class"] == "stable")
    intensive_rehab_count = sum(1 for t in trajectories if t["intensive_rehab_recommended"])

    # Overall trend: compare improving vs declining counts
    if improving_count > declining_count:
        avg_trend = "improving"
    elif declining_count > improving_count:
        avg_trend = "declining"
    else:
        avg_trend = "stable"

    # Average recovery score: mean of latest daily_function_rating across patients
    daily_vals = [t["latest_daily_function"] for t in trajectories if t["latest_daily_function"] is not None]
    avg_recovery_score = _avg(daily_vals)

    kpis = {
        "total_patients_tracked":          total_patients,
        "avg_daily_function_trend":        avg_trend,
        "patients_needing_intensive_rehab": intensive_rehab_count,
        "avg_recovery_score":              avg_recovery_score,
        "improving_count":                 improving_count,
        "stable_count":                    stable_count,
        "declining_count":                 declining_count,
    }

    # -- Trajectory distribution (pie chart) ----------------------------------
    trajectory_distribution = []
    for cls in TRAJECTORY_CLASSES:
        count = sum(1 for t in trajectories if t["trajectory_class"] == cls)
        trajectory_distribution.append({"name": cls, "count": count})

    # -- Monthly average functional rating (line chart) -----------------------
    from collections import defaultdict
    monthly_buckets = defaultdict(list)
    for r in pro_records:
        date_str = r.get("assessment_date", r.get("_created_at", ""))
        if not date_str:
            continue
        try:
            month_key = date_str[:7]  # YYYY-MM
        except Exception:
            continue
        daily = r.get("daily_function_rating")
        if daily is not None:
            monthly_buckets[month_key].append(daily)

    monthly_avg_function = []
    for month in sorted(monthly_buckets.keys()):
        vals = monthly_buckets[month]
        monthly_avg_function.append({
            "month":              month,
            "avg_daily_function": _avg(vals),
            "num_assessments":    len(vals),
        })

    # -- Risk factor bar (declining patients) ---------------------------------
    declining_trajectories = [t for t in trajectories if t["trajectory_class"] == "declining"]
    risk_counts = {k: 0 for k in RISK_FACTOR_LABELS}
    for t in declining_trajectories:
        for rf in t["risk_factors"]:
            if rf in risk_counts:
                risk_counts[rf] += 1

    risk_factor_bar = []
    for key, label in RISK_FACTOR_LABELS.items():
        risk_factor_bar.append({
            "risk_factor": label,
            "count":       risk_counts[key],
        })

    # -- Patient trajectory summary -------------------------------------------
    patient_trajectory_summary = []
    for t in trajectories:
        patient_trajectory_summary.append({
            "patient_id":       t["patient_id"],
            "name":             t["name"],
            "trajectory_class": t["trajectory_class"],
            "slope":            t["slope"],
            "latest_daily_function": t["latest_daily_function"],
            "intensive_rehab_recommended": t["intensive_rehab_recommended"],
        })

    return {
        "kpis":                       kpis,
        "trajectory_distribution":    trajectory_distribution,
        "monthly_avg_function":       monthly_avg_function,
        "risk_factor_bar":            risk_factor_bar,
        "patient_trajectory_summary": patient_trajectory_summary,
    }


def breakdown(patient_id: Optional[str] = None) -> dict:
    """
    Detailed per-patient trajectory breakdown.

    Returns:
        patient_table         -- full per-patient trajectory data with rehab flag
        domain_by_trajectory  -- stacked bar: functional domains grouped by class
        prediction_factors    -- which variables best predict declining trajectory
        declining_patients    -- patients of top concern (declining + high risk)
    """
    conn = _conn()
    patients = _real_patients(conn)
    patient_info_map = {p["patient_id"]: p for p in patients}
    pro_records = _load_pro_outcomes(conn)
    conn.close()

    if patient_id:
        pro_records = [r for r in pro_records if r.get("_patient_id") == patient_id]

    if not pro_records:
        return {
            "patient_table": [],
            "domain_by_trajectory": [],
            "prediction_factors": [],
            "declining_patients": [],
        }

    trajectories = _build_patient_trajectories(pro_records, patient_info_map)

    # -- Patient table --------------------------------------------------------
    patient_table = []
    for t in trajectories:
        patient_table.append({
            "patient_id":                t["patient_id"],
            "name":                      t["name"],
            "age":                       t["age"],
            "gender":                    t["gender"],
            "disease":                   t["disease"],
            "trajectory_class":          t["trajectory_class"],
            "latest_function_rating":    t["latest_daily_function"],
            "slope":                     t["slope"],
            "num_assessments":           t["num_assessments"],
            "latest_moca":               t["latest_moca"],
            "latest_phq9":               t["latest_phq9"],
            "latest_fatigue":            t["latest_fatigue"],
            "latest_sleep":              t["latest_sleep"],
            "risk_factors":              t["risk_factors"],
            "intensive_rehab_recommended": t["intensive_rehab_recommended"],
        })

    # -- Domain by trajectory (stacked bar) -----------------------------------
    from collections import defaultdict
    domain_sums = defaultdict(lambda: defaultdict(list))
    for t in trajectories:
        cls = t["trajectory_class"]
        if t["latest_daily_function"] is not None:
            domain_sums[cls]["daily_function"].append(t["latest_daily_function"])
        if t["latest_social_function"] is not None:
            domain_sums[cls]["social_function"].append(t["latest_social_function"])
        if t["latest_moca"] is not None:
            # Normalise MoCA (0-30) to 1-10 scale for comparability
            domain_sums[cls]["cognitive_moca"].append(round(t["latest_moca"] / 3.0, 2))

    domain_by_trajectory = []
    for cls in TRAJECTORY_CLASSES:
        entry = {"trajectory_class": cls}
        for domain in ["daily_function", "social_function", "cognitive_moca"]:
            vals = domain_sums[cls][domain]
            entry[f"avg_{domain}"] = _avg(vals)
        domain_by_trajectory.append(entry)

    # -- Prediction factors (simple correlation-based) ------------------------
    # Collect latest values across all patients and compute correlation with
    # a binary declining indicator (1 = declining, 0 = not declining)
    factors = {
        "phq9_score":             [],
        "moca_score":             [],
        "fatigue_level":          [],
        "sleep_hours":            [],
        "gad7_score":             [],
        "mood_rating":            [],
    }
    declining_flags = []

    for t in trajectories:
        is_declining = 1 if t["trajectory_class"] == "declining" else 0
        declining_flags.append(is_declining)
        factors["phq9_score"].append(t["latest_phq9"])
        factors["moca_score"].append(t["latest_moca"])
        factors["fatigue_level"].append(t["latest_fatigue"])
        factors["sleep_hours"].append(t["latest_sleep"])
        factors["gad7_score"].append(t["latest_gad7"])
        factors["mood_rating"].append(t["latest_mood"])

    def _pearson(xs, ys):
        """Simple Pearson correlation for paired lists (skip None)."""
        pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        n = len(pairs)
        if n < 3:
            return 0.0
        x_vals = [p[0] for p in pairs]
        y_vals = [p[1] for p in pairs]
        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n
        cov = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
        var_x = sum((x - x_mean) ** 2 for x in x_vals)
        var_y = sum((y - y_mean) ** 2 for y in y_vals)
        denom = (var_x * var_y) ** 0.5
        if denom == 0:
            return 0.0
        return round(cov / denom, 4)

    prediction_factors = []
    factor_labels = {
        "phq9_score":    "PHQ-9 (Depression)",
        "moca_score":    "MoCA (Cognition)",
        "fatigue_level": "Fatigue Level",
        "sleep_hours":   "Sleep Hours",
        "gad7_score":    "GAD-7 (Anxiety)",
        "mood_rating":   "Mood Rating",
    }
    for key, label in factor_labels.items():
        corr = _pearson(factors[key], declining_flags)
        prediction_factors.append({
            "variable":    label,
            "correlation_with_decline": corr,
            "direction":   "higher predicts decline" if corr > 0 else "lower predicts decline",
        })
    # Sort by absolute correlation strength
    prediction_factors.sort(key=lambda x: abs(x["correlation_with_decline"]), reverse=True)

    # -- Declining patients list (top concern) --------------------------------
    declining_patients = [
        {
            "patient_id":                t["patient_id"],
            "name":                      t["name"],
            "age":                       t["age"],
            "disease":                   t["disease"],
            "slope":                     t["slope"],
            "latest_function_rating":    t["latest_daily_function"],
            "latest_phq9":               t["latest_phq9"],
            "latest_moca":               t["latest_moca"],
            "latest_fatigue":            t["latest_fatigue"],
            "risk_factors":              t["risk_factors"],
            "intensive_rehab_recommended": t["intensive_rehab_recommended"],
        }
        for t in trajectories
        if t["trajectory_class"] == "declining"
    ]
    # Sort by slope ascending (most declining first)
    declining_patients.sort(key=lambda x: x["slope"])

    return {
        "patient_table":         patient_table,
        "domain_by_trajectory":  domain_by_trajectory,
        "prediction_factors":    prediction_factors,
        "declining_patients":    declining_patients,
    }


def definitions() -> dict:
    """
    Metric definitions, rehab intensity criteria, functional rating scales,
    and glossary for the Recovery Trajectory Forecast Dashboard.

    Returns:
        metric_definitions       -- explanation of trajectory classification
        rehab_intensity_criteria -- when intensive rehab is recommended
        functional_rating_scales -- scale descriptions for each domain
        glossary                 -- key terms used in recovery trajectory analysis
    """
    return {
        "metric_definitions": {
            "trajectory_class": (
                "Classification of a patient's functional recovery trajectory based on "
                "the linear slope of daily_function_rating over sequential assessments. "
                "Slope > +0.1 = 'improving' (functional gains over time), slope < -0.1 = "
                "'declining' (functional losses), slope between -0.1 and +0.1 = 'stable' "
                "(no significant change)."
            ),
            "slope": (
                "Linear regression slope of daily_function_rating across assessment dates. "
                "Computed as Cov(x,y)/Var(x) where x = assessment index (0,1,2,...) and "
                "y = daily_function_rating. Positive slope indicates improving function; "
                "negative slope indicates declining function. Unit: rating points per assessment."
            ),
            "avg_recovery_score": (
                "Mean of the latest daily_function_rating across all tracked patients. "
                "Ranges from 1 (severe functional impairment) to 10 (full functional "
                "independence). Used as a population-level recovery indicator."
            ),
            "intensive_rehab_recommended": (
                "Boolean flag. True when a patient has a declining trajectory AND at least "
                "one of: PHQ-9 > 10 (moderate-severe depression), MoCA < 25 (cognitive "
                "impairment), or fatigue level > 6 (high fatigue). These patients are at "
                "highest risk and should be prioritised for intensive rehabilitation."
            ),
            "avg_daily_function_trend": (
                "Population-level trend determined by comparing the count of improving vs "
                "declining patients. If more patients are improving, trend = 'improving'; "
                "if more are declining, trend = 'declining'; if equal, trend = 'stable'."
            ),
            "prediction_factors": (
                "Pearson correlation coefficients between clinical variables (PHQ-9, MoCA, "
                "fatigue, sleep, GAD-7, mood) and a binary declining indicator (1 = declining, "
                "0 = not declining). Higher absolute correlation indicates stronger predictive "
                "value. Positive correlation means higher values predict decline."
            ),
        },
        "rehab_intensity_criteria": {
            "description": (
                "Intensive rehabilitation is recommended for patients meeting ALL of the "
                "following: (1) declining trajectory (slope < -0.1), AND (2) at least one "
                "clinical risk factor from the list below."
            ),
            "risk_factors": [
                {
                    "factor":    "PHQ-9 > 10",
                    "rationale": (
                        "Moderate-severe depression (PHQ-9 > 10) impairs motivation, "
                        "treatment adherence, and functional recovery. Integrated psychiatric "
                        "support should accompany rehabilitation."
                    ),
                },
                {
                    "factor":    "MoCA < 25",
                    "rationale": (
                        "Cognitive impairment (MoCA < 25) limits the patient's ability to "
                        "engage in standard rehabilitation protocols. Adaptive cognitive "
                        "rehabilitation strategies should be incorporated."
                    ),
                },
                {
                    "factor":    "Fatigue > 6",
                    "rationale": (
                        "High fatigue (> 6 on a 1-10 scale) reduces exercise tolerance and "
                        "participation in therapy sessions. Graded activity programmes and "
                        "energy conservation techniques should be prioritised."
                    ),
                },
            ],
            "intensity_levels": [
                {
                    "level":       "Standard",
                    "criteria":    "Stable or improving trajectory, no high-risk factors",
                    "sessions":    "2-3 sessions per week",
                    "duration":    "45-60 minutes per session",
                },
                {
                    "level":       "Enhanced",
                    "criteria":    "Declining trajectory without high-risk factors, OR stable with multiple risk factors",
                    "sessions":    "3-4 sessions per week",
                    "duration":    "60-90 minutes per session, multidisciplinary team",
                },
                {
                    "level":       "Intensive",
                    "criteria":    "Declining trajectory with one or more high-risk factors",
                    "sessions":    "5 sessions per week (daily)",
                    "duration":    "90-120 minutes per session, coordinated specialist input",
                },
            ],
        },
        "functional_rating_scales": {
            "daily_function_rating": {
                "range":       "1-10",
                "description": (
                    "Patient-reported daily functional capacity. 1 = unable to perform "
                    "basic daily activities independently; 5 = moderate difficulty with "
                    "complex tasks; 10 = fully independent in all daily activities. "
                    "Assessed monthly via PRO questionnaire."
                ),
            },
            "social_function_rating": {
                "range":       "1-10",
                "description": (
                    "Patient-reported social functioning. 1 = complete social withdrawal; "
                    "5 = limited social engagement with effort; 10 = active social life "
                    "without seizure-related restrictions."
                ),
            },
            "moca_score": {
                "range":       "0-30",
                "description": (
                    "Montreal Cognitive Assessment. Validated screening tool for cognitive "
                    "impairment. Score < 26 suggests mild cognitive impairment; < 18 suggests "
                    "moderate impairment. Normalised to 1-10 scale (divided by 3) for cross-"
                    "domain comparison charts."
                ),
            },
            "phq9_score": {
                "range":       "0-27",
                "description": (
                    "Patient Health Questionnaire-9. Validated depression severity measure. "
                    "0-4 = minimal; 5-9 = mild; 10-14 = moderate; 15-19 = moderately severe; "
                    "20-27 = severe depression."
                ),
            },
            "fatigue_level": {
                "range":       "1-10",
                "description": (
                    "Self-reported fatigue severity. 1 = no fatigue; 5 = moderate fatigue "
                    "affecting some activities; 10 = severe, disabling fatigue."
                ),
            },
            "sleep_hours": {
                "range":       "2-12 hours",
                "description": (
                    "Average nightly sleep duration. < 6 hours associated with cognitive "
                    "decline and seizure threshold reduction. Target: 7-9 hours."
                ),
            },
        },
        "glossary": [
            {
                "term":       "Recovery Trajectory",
                "definition": (
                    "The longitudinal trend of a patient's functional recovery, measured by "
                    "the slope of daily_function_rating over sequential assessments. Classified "
                    "as improving, stable, or declining."
                ),
            },
            {
                "term":       "Linear Slope",
                "definition": (
                    "The rate of change in daily_function_rating per assessment period, "
                    "calculated via ordinary least-squares regression. Positive = improvement, "
                    "negative = decline. Threshold: |0.1| separates stable from changing."
                ),
            },
            {
                "term":       "Intensive Rehabilitation",
                "definition": (
                    "A high-frequency, multidisciplinary rehabilitation programme (5 sessions/"
                    "week, 90-120 min) recommended for patients with declining trajectories and "
                    "co-occurring risk factors (depression, cognitive impairment, or high fatigue)."
                ),
            },
            {
                "term":       "PRO (Patient-Reported Outcome)",
                "definition": (
                    "A health outcome directly reported by the patient without clinician "
                    "interpretation. Includes daily function, social function, fatigue, mood, "
                    "and seizure worry ratings."
                ),
            },
            {
                "term":       "MoCA (Montreal Cognitive Assessment)",
                "definition": (
                    "A brief 30-point cognitive screening tool assessing attention, memory, "
                    "language, visuospatial skills, and executive function. Score < 26 indicates "
                    "possible mild cognitive impairment."
                ),
            },
            {
                "term":       "PHQ-9 (Patient Health Questionnaire-9)",
                "definition": (
                    "A 9-item validated instrument for screening and measuring depression "
                    "severity. Total score range 0-27. Used to identify patients requiring "
                    "integrated psychiatric support alongside rehabilitation."
                ),
            },
            {
                "term":       "GAD-7 (Generalised Anxiety Disorder-7)",
                "definition": (
                    "A 7-item validated screening tool for anxiety. Total score range 0-21. "
                    "Anxiety is a common comorbidity in epilepsy and can impair rehabilitation "
                    "engagement."
                ),
            },
            {
                "term":       "WPAI (Work Productivity and Activity Impairment)",
                "definition": (
                    "A validated instrument measuring the impact of health on work productivity "
                    "and daily activities. Expressed as a percentage (0% = no impairment, "
                    "100% = complete impairment)."
                ),
            },
            {
                "term":       "Risk Factor",
                "definition": (
                    "A clinical variable associated with declining functional trajectory. "
                    "Current risk factors tracked: high PHQ-9 (> 10), low MoCA (< 25), "
                    "high fatigue (> 6), and poor sleep (< 6 hours)."
                ),
            },
            {
                "term":       "Functional Domain",
                "definition": (
                    "A category of patient functioning assessed independently: daily function "
                    "(ADLs), social function (interpersonal engagement), and cognitive function "
                    "(MoCA-based assessment)."
                ),
            },
        ],
    }
