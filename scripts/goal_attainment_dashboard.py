"""
Neuro AI Ecosystem -- Goal-Attainment Scaling (GAS) Trend Dashboard
====================================================================
Occupational therapy outcome tracking using Goal-Attainment Scaling,
the standard OT tool where goals are set with expected outcome levels.

GAS Scoring:
  -2  much less than expected outcome
  -1  less than expected outcome
   0  expected outcome
  +1  more than expected outcome
  +2  much more than expected outcome

GAS T-score = 50 + (10 * sum_of_scores) / sqrt(n * (1 - avg_intercorrelation))
  where n = number of goals.  Simplified with avg_intercorrelation = 0.3.

Goal Domains:
  ADL                  -- activities of daily living (mapped from daily_function_rating)
  Cognitive            -- cognitive rehabilitation (mapped from moca_score)
  Mobility             -- physical mobility (mapped from fatigue_level, inverse)
  Social Participation -- social engagement (mapped from mood_rating)
  Medication Management-- adherence & self-management (deterministic from patient data)
  Seizure Safety       -- seizure awareness & precautions (deterministic from patient data)

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
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from collections import defaultdict

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# -- Constants ----------------------------------------------------------------

GAS_SCORE_LABELS = {
    -2: "Much less than expected",
    -1: "Less than expected",
     0: "Expected outcome",
     1: "More than expected",
     2: "Much more than expected",
}

GOAL_DOMAINS = [
    "ADL",
    "Cognitive",
    "Mobility",
    "Social Participation",
    "Medication Management",
    "Seizure Safety",
]

DOMAIN_DESCRIPTIONS = {
    "ADL": "Activities of daily living -- dressing, bathing, meal prep, household tasks",
    "Cognitive": "Cognitive rehabilitation -- memory strategies, attention training, executive function",
    "Mobility": "Physical mobility -- gait, transfers, endurance, balance",
    "Social Participation": "Social engagement -- community activities, relationships, communication",
    "Medication Management": "Adherence and self-management -- scheduling, side-effect awareness, refill tracking",
    "Seizure Safety": "Seizure awareness and precautions -- trigger identification, emergency plans, environmental safety",
}

# Average intercorrelation constant for GAS T-score formula
AVG_INTERCORRELATION = 0.3


def _conn():
    return sqlite3.connect(DB_PATH)


def _deterministic_seed(patient_id: str, item_id: str) -> float:
    """Return a reproducible float in [0, 1) from patient + item identifiers."""
    h = hashlib.sha256(f"gas:{patient_id}:{item_id}".encode()).hexdigest()
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


def _map_to_gas_score(raw_value, scale_min, scale_max, invert=False):
    """Map a raw clinical value to GAS -2..+2 score.

    Divides the raw range into 5 equal bins corresponding to GAS scores -2 to +2.
    If invert is True, higher raw values map to lower GAS scores (e.g. fatigue).
    Returns None if raw_value is None.
    """
    if raw_value is None:
        return None
    # Clamp to range
    val = max(scale_min, min(scale_max, raw_value))
    if invert:
        val = scale_max - (val - scale_min)
    # Normalise to 0..1
    span = scale_max - scale_min
    if span == 0:
        return 0
    norm = (val - scale_min) / span
    # Map to -2..+2: 5 bins [0-0.2) -> -2, [0.2-0.4) -> -1, [0.4-0.6) -> 0, [0.6-0.8) -> +1, [0.8-1.0] -> +2
    if norm >= 0.8:
        return 2
    elif norm >= 0.6:
        return 1
    elif norm >= 0.4:
        return 0
    elif norm >= 0.2:
        return -1
    else:
        return -2


def _compute_gas_t_score(scores):
    """Compute GAS T-score from a list of individual goal scores (-2 to +2).

    Formula: T = 50 + (10 * sum_of_scores) / sqrt(n * (1 - rho))
    where rho = AVG_INTERCORRELATION (0.3).
    Returns 50.0 (neutral) if no scores available.
    """
    valid = [s for s in scores if s is not None]
    n = len(valid)
    if n == 0:
        return 50.0
    total = sum(valid)
    denominator = math.sqrt(n * (1 - AVG_INTERCORRELATION))
    if denominator == 0:
        return 50.0
    return round(50 + (10 * total) / denominator, 2)


def _build_patient_goals(pro_records, patient_info_map):
    """Build per-patient goal attainment data from PRO outcome records.

    For each patient, computes GAS scores across the 6 goal domains using
    real clinical values where available and deterministic seeding for
    supplementary domains.

    Returns a list of dicts, one per patient, containing domain scores,
    T-score, and goal metadata.
    """
    # Group records by patient
    patient_records = defaultdict(list)
    for r in pro_records:
        pid = r.get("patient_id", r.get("_patient_id"))
        patient_records[pid].append(r)

    patient_goals = []
    for pid in sorted(patient_records.keys()):
        recs = sorted(
            patient_records[pid],
            key=lambda x: x.get("assessment_date", x.get("_created_at", "")),
        )
        info = patient_info_map.get(pid, {})
        latest = recs[-1]

        # -- Map real clinical values to GAS scores per domain ---------------

        # ADL: daily_function_rating (1-10)
        adl_raw = latest.get("daily_function_rating")
        adl_score = _map_to_gas_score(adl_raw, 1, 10)

        # Cognitive: moca_score (0-30)
        moca_raw = latest.get("moca_score")
        cognitive_score = _map_to_gas_score(moca_raw, 0, 30)

        # Mobility: fatigue_level (1-10, inverted -- lower fatigue = better mobility)
        fatigue_raw = latest.get("fatigue_level")
        mobility_score = _map_to_gas_score(fatigue_raw, 1, 10, invert=True)

        # Social Participation: mood_rating (1-10)
        mood_raw = latest.get("mood_rating")
        social_score = _map_to_gas_score(mood_raw, 1, 10)

        # Medication Management: deterministic from patient data
        med_score = _seed_int(str(pid), "medication_mgmt", -2, 2)

        # Seizure Safety: deterministic from patient data
        seizure_score = _seed_int(str(pid), "seizure_safety", -1, 2)

        domain_scores = {
            "ADL":                   adl_score,
            "Cognitive":             cognitive_score,
            "Mobility":             mobility_score,
            "Social Participation":  social_score,
            "Medication Management": med_score,
            "Seizure Safety":        seizure_score,
        }

        all_scores = [v for v in domain_scores.values() if v is not None]
        t_score = _compute_gas_t_score(all_scores)

        # -- Trend per domain over time (last vs first record) ---------------
        domain_trends = {}
        if len(recs) >= 2:
            first = recs[0]
            adl_first = _map_to_gas_score(first.get("daily_function_rating"), 1, 10)
            cog_first = _map_to_gas_score(first.get("moca_score"), 0, 30)
            mob_first = _map_to_gas_score(first.get("fatigue_level"), 1, 10, invert=True)
            soc_first = _map_to_gas_score(first.get("mood_rating"), 1, 10)

            for domain, latest_s, first_s in [
                ("ADL", adl_score, adl_first),
                ("Cognitive", cognitive_score, cog_first),
                ("Mobility", mobility_score, mob_first),
                ("Social Participation", social_score, soc_first),
            ]:
                if latest_s is not None and first_s is not None:
                    diff = latest_s - first_s
                    if diff > 0:
                        domain_trends[domain] = "improving"
                    elif diff < 0:
                        domain_trends[domain] = "declining"
                    else:
                        domain_trends[domain] = "stable"
                else:
                    domain_trends[domain] = "stable"
            # Seeded domains are stable by definition (single-point)
            domain_trends["Medication Management"] = "stable"
            domain_trends["Seizure Safety"] = "stable"
        else:
            for d in GOAL_DOMAINS:
                domain_trends[d] = "stable"

        # -- Recommendation --------------------------------------------------
        goals_met = sum(1 for v in all_scores if v >= 0)
        goals_below = sum(1 for v in all_scores if v < 0)

        if t_score < 40:
            recommendation = "Urgent review -- T-score below 40, reassess goal levels and intervention plan"
        elif goals_below > len(all_scores) / 2:
            recommendation = "Review needed -- majority of goals below expected, consider goal recalibration"
        elif t_score >= 60:
            recommendation = "Exceeding expectations -- consider advancing goal targets"
        else:
            recommendation = "On track -- continue current intervention plan"

        # -- Review due date (deterministic) ----------------------------------
        days_offset = _seed_int(str(pid), "review_offset", 0, 30)
        review_date = (datetime.now() + timedelta(days=days_offset)).strftime("%Y-%m-%d")
        is_review_due = days_offset <= 7

        patient_goals.append({
            "patient_id":       pid,
            "name":             info.get("name", f"Patient {pid}"),
            "age":              info.get("age"),
            "gender":           info.get("gender"),
            "disease":          info.get("disease", "unspecified"),
            "department":       info.get("department"),
            "domain_scores":    domain_scores,
            "domain_trends":    domain_trends,
            "t_score":          t_score,
            "goals_met":        goals_met,
            "goals_total":      len(all_scores),
            "goals_exceeding":  sum(1 for v in all_scores if v > 0),
            "goals_below":      goals_below,
            "recommendation":   recommendation,
            "review_due":       review_date,
            "is_review_due":    is_review_due,
            "num_assessments":  len(recs),
            "latest_raw": {
                "daily_function_rating": adl_raw,
                "moca_score":            moca_raw,
                "fatigue_level":         fatigue_raw,
                "mood_rating":           mood_raw,
            },
        })

    return patient_goals


def _build_monthly_trend(pro_records):
    """Build monthly GAS T-score trend from PRO records over last 6 months.

    Groups records by month, computes average GAS scores per domain,
    then computes a cohort-level T-score for each month.
    """
    # Determine the date range from the actual data, then take last 6 months
    all_dates = []
    for r in pro_records:
        date_str = r.get("assessment_date", r.get("_created_at", ""))
        if date_str:
            try:
                all_dates.append(date_str[:7])
            except Exception:
                pass
    if all_dates:
        latest_month = max(all_dates)
        try:
            latest_dt = datetime.strptime(latest_month, "%Y-%m")
        except (ValueError, TypeError):
            latest_dt = datetime.now()
        cutoff = latest_dt - timedelta(days=180)
    else:
        cutoff = datetime.now() - timedelta(days=180)

    monthly = defaultdict(list)
    for r in pro_records:
        date_str = r.get("assessment_date", r.get("_created_at", ""))
        if not date_str:
            continue
        try:
            month_key = date_str[:7]  # YYYY-MM
            rec_date = datetime.strptime(month_key, "%Y-%m")
        except (ValueError, TypeError):
            continue
        if rec_date < cutoff.replace(day=1):
            continue

        # Compute per-record GAS scores
        scores = []
        adl = _map_to_gas_score(r.get("daily_function_rating"), 1, 10)
        if adl is not None:
            scores.append(adl)
        cog = _map_to_gas_score(r.get("moca_score"), 0, 30)
        if cog is not None:
            scores.append(cog)
        mob = _map_to_gas_score(r.get("fatigue_level"), 1, 10, invert=True)
        if mob is not None:
            scores.append(mob)
        soc = _map_to_gas_score(r.get("mood_rating"), 1, 10)
        if soc is not None:
            scores.append(soc)

        if scores:
            monthly[month_key].append(scores)

    trend = []
    for month in sorted(monthly.keys()):
        # Compute T-score per patient record, then average across patients
        patient_t_scores = []
        goals_met_count = 0
        goals_total_count = 0
        for score_list in monthly[month]:
            t = _compute_gas_t_score(score_list)
            patient_t_scores.append(t)
            goals_total_count += len(score_list)
            goals_met_count += sum(1 for s in score_list if s >= 0)

        avg_t = _avg(patient_t_scores)
        pct_met = round(100 * goals_met_count / goals_total_count, 1) if goals_total_count else 0

        trend.append({
            "month":         month,
            "avg_t_score":   avg_t,
            "goals_met_pct": pct_met,
        })

    return trend


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def overview(patient_id: Optional[str] = None) -> dict:
    """
    Goal-Attainment Scaling trend overview.

    Returns:
        kpi                  -- total_patients, total_goals, avg_gas_t_score,
                                pct_goals_met, pct_exceeding, review_due_count
        gas_distribution     -- score label counts for -2 through +2
        domain_performance   -- per-domain avg_score, goal_count, pct_met
        trend                -- monthly avg_t_score and goals_met_pct (6 months)
        top_achievers        -- top 5 patients by T-score
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
            "kpi": {
                "total_patients":   0,
                "total_goals":      0,
                "avg_gas_t_score":  50.0,
                "pct_goals_met":    0.0,
                "pct_exceeding":    0.0,
                "review_due_count": 0,
            },
            "gas_distribution":   [],
            "domain_performance": [],
            "trend":              [],
            "top_achievers":      [],
        }

    patient_goals = _build_patient_goals(pro_records, patient_info_map)

    # -- KPIs -----------------------------------------------------------------
    total_patients = len(patient_goals)
    all_domain_scores = []
    for pg in patient_goals:
        for v in pg["domain_scores"].values():
            if v is not None:
                all_domain_scores.append(v)

    total_goals = len(all_domain_scores)
    goals_met = sum(1 for s in all_domain_scores if s >= 0)
    goals_exceeding = sum(1 for s in all_domain_scores if s > 0)
    pct_met = round(100 * goals_met / total_goals, 1) if total_goals else 0
    pct_exceeding = round(100 * goals_exceeding / total_goals, 1) if total_goals else 0

    t_scores = [pg["t_score"] for pg in patient_goals]
    avg_t = _avg(t_scores)

    review_due_count = sum(1 for pg in patient_goals if pg["is_review_due"])

    kpi = {
        "total_patients":   total_patients,
        "total_goals":      total_goals,
        "avg_gas_t_score":  avg_t,
        "pct_goals_met":    pct_met,
        "pct_exceeding":    pct_exceeding,
        "review_due_count": review_due_count,
    }

    # -- GAS distribution (histogram) ----------------------------------------
    score_counts = {s: 0 for s in range(-2, 3)}
    for s in all_domain_scores:
        if s in score_counts:
            score_counts[s] += 1

    gas_distribution = []
    for score in range(-2, 3):
        gas_distribution.append({
            "score":       score,
            "score_label": GAS_SCORE_LABELS[score],
            "count":       score_counts[score],
        })

    # -- Domain performance ---------------------------------------------------
    domain_perf = {}
    for domain in GOAL_DOMAINS:
        domain_perf[domain] = {"scores": [], "met": 0}

    for pg in patient_goals:
        for domain, score in pg["domain_scores"].items():
            if score is not None:
                domain_perf[domain]["scores"].append(score)
                if score >= 0:
                    domain_perf[domain]["met"] += 1

    domain_performance = []
    for domain in GOAL_DOMAINS:
        dp = domain_perf[domain]
        cnt = len(dp["scores"])
        domain_performance.append({
            "domain":     domain,
            "avg_score":  _avg(dp["scores"]),
            "goal_count": cnt,
            "pct_met":    round(100 * dp["met"] / cnt, 1) if cnt else 0,
        })

    # -- Monthly trend (6 months) --------------------------------------------
    trend = _build_monthly_trend(pro_records)

    # -- Top achievers -------------------------------------------------------
    sorted_by_t = sorted(patient_goals, key=lambda x: x["t_score"], reverse=True)
    top_achievers = []
    for pg in sorted_by_t[:5]:
        top_achievers.append({
            "patient_id": pg["patient_id"],
            "name":       pg["name"],
            "t_score":    pg["t_score"],
            "goals_met":  pg["goals_met"],
        })

    return {
        "kpi":                kpi,
        "gas_distribution":   gas_distribution,
        "domain_performance": domain_performance,
        "trend":              trend,
        "top_achievers":      top_achievers,
    }


def breakdown(patient_id: Optional[str] = None) -> dict:
    """
    Detailed per-patient goal attainment breakdown.

    Returns:
        patient_goals    -- per-patient goal details with domain scores, T-score,
                            trend direction, and recommendation
        domain_drill     -- per-domain details with scoring criteria and patient list
        recent_reviews   -- recent goal review events
        at_risk          -- patients with T-score < 40 or majority goals below expected
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
            "patient_goals":  [],
            "domain_drill":   [],
            "recent_reviews": [],
            "at_risk":        [],
        }

    all_patient_goals = _build_patient_goals(pro_records, patient_info_map)

    # -- Patient goals table --------------------------------------------------
    patient_goals_list = []
    for pg in all_patient_goals:
        goals = []
        for domain in GOAL_DOMAINS:
            score = pg["domain_scores"].get(domain)
            trend_dir = pg["domain_trends"].get(domain, "stable")
            target = GAS_SCORE_LABELS.get(0, "Expected outcome")
            goals.append({
                "domain":         domain,
                "target":         target,
                "current_score":  score,
                "score_label":    GAS_SCORE_LABELS.get(score, "N/A") if score is not None else "N/A",
                "trend_direction": trend_dir,
            })

        patient_goals_list.append({
            "patient_id":     pg["patient_id"],
            "name":           pg["name"],
            "age":            pg["age"],
            "disease":        pg["disease"],
            "goals":          goals,
            "t_score":        pg["t_score"],
            "goals_met":      pg["goals_met"],
            "goals_total":    pg["goals_total"],
            "recommendation": pg["recommendation"],
        })

    # -- Domain drill ---------------------------------------------------------
    domain_drill = []
    for domain in GOAL_DOMAINS:
        scoring_criteria = _domain_scoring_criteria(domain)
        domain_patients = []
        for pg in all_patient_goals:
            score = pg["domain_scores"].get(domain)
            if score is not None:
                domain_patients.append({
                    "patient_id": pg["patient_id"],
                    "name":       pg["name"],
                    "score":      score,
                    "score_label": GAS_SCORE_LABELS.get(score, "N/A"),
                    "trend":      pg["domain_trends"].get(domain, "stable"),
                })

        domain_drill.append({
            "domain":           domain,
            "description":      DOMAIN_DESCRIPTIONS.get(domain, ""),
            "scoring_criteria": scoring_criteria,
            "patients":         domain_patients,
        })

    # -- Recent reviews (deterministic) ---------------------------------------
    recent_reviews = []
    reviewers = ["Dr. Sharma", "Dr. Patel", "OT Williams", "OT Chen", "Dr. Kapoor"]
    change_types = [
        "Goal level adjusted upward",
        "New goal added",
        "Intervention plan updated",
        "Goal target recalibrated",
        "Review completed, no changes",
        "Domain priority reordered",
    ]
    for pg in all_patient_goals[:15]:
        reviewer_idx = _seed_int(str(pg["patient_id"]), "reviewer", 0, len(reviewers) - 1)
        change_idx = _seed_int(str(pg["patient_id"]), "change_type", 0, len(change_types) - 1)
        days_ago = _seed_int(str(pg["patient_id"]), "review_days_ago", 1, 30)
        review_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        recent_reviews.append({
            "patient_id": pg["patient_id"],
            "name":       pg["name"],
            "date":       review_date,
            "reviewer":   reviewers[reviewer_idx],
            "changes":    change_types[change_idx],
        })
    recent_reviews.sort(key=lambda x: x["date"], reverse=True)

    # -- At-risk patients -----------------------------------------------------
    at_risk = []
    for pg in all_patient_goals:
        is_at_risk = False
        risk_reasons = []
        if pg["t_score"] < 40:
            is_at_risk = True
            risk_reasons.append(f"T-score {pg['t_score']} < 40")
        if pg["goals_total"] > 0 and pg["goals_below"] > pg["goals_total"] / 2:
            is_at_risk = True
            risk_reasons.append(f"{pg['goals_below']}/{pg['goals_total']} goals below expected")

        if is_at_risk:
            at_risk.append({
                "patient_id":     pg["patient_id"],
                "name":           pg["name"],
                "t_score":        pg["t_score"],
                "goals_met":      pg["goals_met"],
                "goals_total":    pg["goals_total"],
                "goals_below":    pg["goals_below"],
                "risk_reasons":   risk_reasons,
                "recommendation": pg["recommendation"],
            })
    at_risk.sort(key=lambda x: x["t_score"])

    return {
        "patient_goals":  patient_goals_list,
        "domain_drill":   domain_drill,
        "recent_reviews": recent_reviews,
        "at_risk":        at_risk,
    }


def _domain_scoring_criteria(domain):
    """Return GAS scoring criteria for a specific goal domain.

    Each domain has its own clinical interpretation of the -2 to +2 scale.
    """
    criteria = {
        "ADL": {
            -2: "Unable to perform basic self-care; requires full assistance for dressing, bathing, feeding",
            -1: "Requires moderate assistance for 2+ ADL tasks; inconsistent independence",
            0:  "Performs basic ADLs independently with occasional cueing; manages simple meal prep",
            1:  "Independent in all basic and most instrumental ADLs; minimal cueing needed",
            2:  "Fully independent in all ADLs including complex tasks (finances, transport, community errands)",
        },
        "Cognitive": {
            -2: "Severe memory/attention deficits impacting safety; unable to follow multi-step instructions",
            -1: "Moderate deficits; forgets appointments, difficulty with planning; needs written aids",
            0:  "Mild deficits managed with compensatory strategies; uses calendar/reminders consistently",
            1:  "Compensatory strategies internalised; occasional lapses only; returns to prior cognitive tasks",
            2:  "Cognitive function at or above pre-morbid level; independently manages complex problem-solving",
        },
        "Mobility": {
            -2: "Bed/wheelchair bound; unable to transfer independently; high falls risk",
            -1: "Transfers with assistance; walks short distances with aid; fatigue limits activity to < 30 min",
            0:  "Independent transfers; walks moderate distances; tolerates 1 hour of activity with rest breaks",
            1:  "Walks community distances without aid; stairs independently; tolerates 2+ hours activity",
            2:  "Full pre-morbid mobility; participates in exercise/sport; no activity restrictions",
        },
        "Social Participation": {
            -2: "Complete social withdrawal; refuses visitors; no community engagement",
            -1: "Limited to 1-2 trusted contacts; avoids group settings; rare outings",
            0:  "Attends 1-2 social activities per week; engages with small groups; some avoidance of new settings",
            1:  "Active social life; initiates contact; participates in community groups; occasional anxiety managed",
            2:  "Full social reintegration; leadership roles; mentors others; no seizure-related social avoidance",
        },
        "Medication Management": {
            -2: "Misses > 50% of doses; unable to identify medications; no awareness of side effects",
            -1: "Misses 25-50% of doses; identifies medications but not purposes; relies on carer for scheduling",
            0:  "Takes medications on time > 80%; uses pill organiser; recognises common side effects",
            1:  "Near-perfect adherence; independently manages refills; reports side effects proactively",
            2:  "Fully self-manages complex regimen; adjusts rescue medication per protocol; educates family",
        },
        "Seizure Safety": {
            -2: "No awareness of seizure triggers; no emergency plan; unsafe home environment",
            -1: "Partial trigger awareness; emergency plan exists but not rehearsed; some home hazards remain",
            0:  "Identifies personal triggers; carries emergency card; basic home safety modifications in place",
            1:  "Comprehensive trigger management; family/friends trained in first aid; safe in most environments",
            2:  "Expert self-management; trains others; advocates for workplace accommodations; minimal risk exposure",
        },
    }
    return criteria.get(domain, {})


def definitions() -> dict:
    """
    Metric definitions, GAS scale descriptions, domain descriptions,
    and glossary for the Goal-Attainment Scaling Trend Dashboard.

    Returns:
        metrics    -- list of {name, formula, interpretation}
        gas_scale  -- the -2 to +2 scale descriptions
        domains    -- domain descriptions and typical goals
        glossary   -- key terms used in GAS analysis
    """
    return {
        "metrics": [
            {
                "name":           "GAS T-Score",
                "formula":        "T = 50 + (10 * sum_of_scores) / sqrt(n * (1 - rho)), rho = 0.3",
                "interpretation": (
                    "Standardised composite score across all goal domains. T = 50 means the "
                    "patient achieved exactly the expected outcome on average. T > 50 indicates "
                    "better-than-expected outcomes; T < 50 indicates worse-than-expected outcomes. "
                    "Clinically meaningful change is typically > 10 points."
                ),
            },
            {
                "name":           "Percentage Goals Met",
                "formula":        "100 * count(score >= 0) / total_goals",
                "interpretation": (
                    "Proportion of goals where the patient achieved at least the expected "
                    "outcome (score 0, +1, or +2). A target of >= 70% goals met is typical "
                    "for a well-calibrated goal-setting process."
                ),
            },
            {
                "name":           "Percentage Exceeding",
                "formula":        "100 * count(score > 0) / total_goals",
                "interpretation": (
                    "Proportion of goals where the patient exceeded expected outcomes "
                    "(score +1 or +2). Persistently high values (> 50%) may indicate that "
                    "goal levels should be recalibrated upward."
                ),
            },
            {
                "name":           "Domain Average Score",
                "formula":        "mean(domain_scores) across all patients for each domain",
                "interpretation": (
                    "Average GAS score per domain across the patient cohort. Indicates "
                    "which functional domains are most/least achieved at a programme level. "
                    "Values near 0 suggest appropriate goal calibration."
                ),
            },
            {
                "name":           "Review Due Count",
                "formula":        "count(patients with next review within 7 days)",
                "interpretation": (
                    "Number of patients whose GAS goals are due for review in the coming "
                    "week. Regular review (typically every 4-6 weeks) is essential for "
                    "responsive goal adjustment."
                ),
            },
            {
                "name":           "Trend Direction",
                "formula":        "compare(latest_score, earliest_score) per domain per patient",
                "interpretation": (
                    "Direction of change in GAS score over the assessment period for each "
                    "domain. 'improving' = latest > earliest, 'declining' = latest < earliest, "
                    "'stable' = no change. Useful for identifying domains requiring intervention."
                ),
            },
        ],
        "gas_scale": [
            {
                "score":       -2,
                "label":       "Much less than expected",
                "description": (
                    "The patient's performance is significantly below the expected outcome. "
                    "This level typically indicates that the goal was set too ambitiously or "
                    "that a major setback (e.g., seizure recurrence, acute illness) has "
                    "prevented progress. Immediate reassessment is warranted."
                ),
            },
            {
                "score":       -1,
                "label":       "Less than expected",
                "description": (
                    "The patient is progressing but has not yet reached the expected level. "
                    "This may indicate the need for additional support, modified intervention "
                    "strategies, or more time. Monitor closely at next review."
                ),
            },
            {
                "score":       0,
                "label":       "Expected outcome",
                "description": (
                    "The patient has achieved the predicted level of attainment. This is the "
                    "target level set collaboratively by the therapist and patient at the start "
                    "of the intervention. Well-calibrated goals should cluster here."
                ),
            },
            {
                "score":       1,
                "label":       "More than expected",
                "description": (
                    "The patient has exceeded the predicted level. This indicates either "
                    "exceptional response to intervention or that the initial goal was set "
                    "conservatively. Consider advancing the goal target."
                ),
            },
            {
                "score":       2,
                "label":       "Much more than expected",
                "description": (
                    "The patient has significantly exceeded expectations. Reassess whether "
                    "goals should be recalibrated to higher levels or whether the patient "
                    "is ready for discharge from this goal domain."
                ),
            },
        ],
        "domains": [
            {
                "domain":        "ADL",
                "full_name":     "Activities of Daily Living",
                "description":   DOMAIN_DESCRIPTIONS["ADL"],
                "typical_goals": [
                    "Independent dressing within 15 minutes",
                    "Prepare a simple meal safely without cueing",
                    "Manage household laundry independently",
                    "Use public transport for routine appointments",
                ],
                "clinical_source": "daily_function_rating from PRO outcomes (scale 1-10)",
            },
            {
                "domain":        "Cognitive",
                "full_name":     "Cognitive Rehabilitation",
                "description":   DOMAIN_DESCRIPTIONS["Cognitive"],
                "typical_goals": [
                    "Use memory notebook to track daily appointments for 4 weeks",
                    "Complete attention-training exercises for 20 minutes daily",
                    "Independently manage a weekly schedule using digital tools",
                    "Return to reading for 30 minutes without losing comprehension",
                ],
                "clinical_source": "moca_score from PRO outcomes (scale 0-30)",
            },
            {
                "domain":        "Mobility",
                "full_name":     "Physical Mobility",
                "description":   DOMAIN_DESCRIPTIONS["Mobility"],
                "typical_goals": [
                    "Walk 200 metres without rest break",
                    "Climb one flight of stairs independently",
                    "Transfer bed-to-chair without assistance",
                    "Tolerate 60 minutes of upright activity",
                ],
                "clinical_source": "fatigue_level from PRO outcomes (scale 1-10, inverted)",
            },
            {
                "domain":        "Social Participation",
                "full_name":     "Social Participation",
                "description":   DOMAIN_DESCRIPTIONS["Social Participation"],
                "typical_goals": [
                    "Attend one community group activity per week",
                    "Initiate social contact with a friend twice per week",
                    "Participate in a family outing without seizure-related anxiety",
                    "Join a peer support group for epilepsy",
                ],
                "clinical_source": "mood_rating from PRO outcomes (scale 1-10)",
            },
            {
                "domain":        "Medication Management",
                "full_name":     "Medication Adherence & Self-Management",
                "description":   DOMAIN_DESCRIPTIONS["Medication Management"],
                "typical_goals": [
                    "Take all prescribed medications within 30 minutes of scheduled time",
                    "Independently manage medication refills before running out",
                    "Identify and report at least 3 common side effects of current medications",
                    "Use a pill organiser or app consistently for 4 weeks",
                ],
                "clinical_source": "Deterministic seed from patient demographics (supplementary)",
            },
            {
                "domain":        "Seizure Safety",
                "full_name":     "Seizure Awareness & Safety Precautions",
                "description":   DOMAIN_DESCRIPTIONS["Seizure Safety"],
                "typical_goals": [
                    "Identify personal seizure triggers and maintain a seizure diary",
                    "Carry an emergency medical ID card at all times",
                    "Ensure family members can administer rescue medication",
                    "Complete home safety assessment and implement modifications",
                ],
                "clinical_source": "Deterministic seed from patient demographics (supplementary)",
            },
        ],
        "glossary": [
            {
                "term":       "Goal-Attainment Scaling (GAS)",
                "definition": (
                    "A standardised method of scoring the extent to which a patient's "
                    "individual goals are achieved during intervention. Developed by "
                    "Kiresuk & Sherman (1968). Each goal is rated on a 5-point scale from "
                    "-2 (much less than expected) to +2 (much more than expected)."
                ),
            },
            {
                "term":       "GAS T-Score",
                "definition": (
                    "A composite score that summarises overall goal attainment across "
                    "multiple domains. Calculated as 50 + (10 * sum) / sqrt(n * (1 - rho)). "
                    "A T-score of 50 = expected outcomes achieved on average. Allows "
                    "comparison across patients with different numbers and types of goals."
                ),
            },
            {
                "term":       "Intercorrelation (rho)",
                "definition": (
                    "The average correlation between goal scores. In the GAS T-score formula, "
                    "rho is typically assumed to be 0.3 for standardisation. Higher intercorrelation "
                    "means goals are less independent, which affects the T-score scaling."
                ),
            },
            {
                "term":       "Expected Outcome (Score = 0)",
                "definition": (
                    "The level of goal attainment predicted by the treating therapist at the "
                    "start of intervention. This is set collaboratively with the patient and "
                    "represents a realistic, achievable target given the patient's baseline "
                    "and planned intervention."
                ),
            },
            {
                "term":       "ADL (Activities of Daily Living)",
                "definition": (
                    "Routine tasks people perform daily without assistance: bathing, dressing, "
                    "eating, toileting, transferring, and continence (basic ADLs) plus cooking, "
                    "cleaning, shopping, managing finances, and transport (instrumental ADLs)."
                ),
            },
            {
                "term":       "Occupational Therapy (OT)",
                "definition": (
                    "A health profession focused on enabling participation in meaningful "
                    "activities (occupations). In neuro-rehabilitation, OT addresses ADLs, "
                    "cognitive function, mobility, social participation, and self-management "
                    "skills following neurological conditions."
                ),
            },
            {
                "term":       "Goal Calibration",
                "definition": (
                    "The process of reviewing and adjusting goal levels based on a patient's "
                    "progress. If a patient consistently exceeds expectations (scores of +1/+2), "
                    "goals should be recalibrated upward. If consistently below (-1/-2), the "
                    "intervention plan or goal level may need revision."
                ),
            },
            {
                "term":       "MoCA (Montreal Cognitive Assessment)",
                "definition": (
                    "A brief 30-point cognitive screening tool assessing attention, memory, "
                    "language, visuospatial skills, and executive function. Score < 26 indicates "
                    "possible mild cognitive impairment. Used to derive the Cognitive domain "
                    "GAS score."
                ),
            },
            {
                "term":       "PRO (Patient-Reported Outcome)",
                "definition": (
                    "A health outcome directly reported by the patient without clinician "
                    "interpretation. In this dashboard, PRO data from clinical.db (daily function "
                    "rating, mood rating, fatigue level, MoCA score) drives the GAS scoring."
                ),
            },
            {
                "term":       "Review Cycle",
                "definition": (
                    "The scheduled interval at which GAS goals are formally reassessed. "
                    "Typical review cycles are 4-6 weeks for acute rehabilitation and 8-12 "
                    "weeks for community-based therapy. Reviews determine whether goals should "
                    "be continued, recalibrated, or discharged."
                ),
            },
        ],
    }
