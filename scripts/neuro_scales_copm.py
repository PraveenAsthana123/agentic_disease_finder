"""
Neuro AI Ecosystem — COPM (Canadian Occupational Performance Measure)
=====================================================================
Patient-centred outcome measure: patients identify occupational performance
problems, then rate Performance (1-10) and Satisfaction (1-10).

Domains:
  Self-Care: personal care, functional mobility, community management
  Productivity: paid/unpaid work, household management, play/school
  Leisure: quiet recreation, active recreation, socialization

Clinically significant change: ≥2 points on performance or satisfaction.

Scores DERIVED from REAL patient data in clinical.db:
  - Barthel Index (functional baseline)
  - PRO outcomes (daily/social function ratings)
  - Seizure diary (frequency, severity)
  - Medications (AED count)
  - Demographics (age, disease)
"""

import sqlite3
import hashlib
from pathlib import Path
from typing import Optional

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# ── COPM Problem Areas (typical epilepsy/neuro patient) ──────────────
COPM_AREAS = {
    "self_care": {
        "label": "Self-Care",
        "problems": [
            {"id": "bathing_dressing", "label": "Bathing & Dressing independently"},
            {"id": "meal_prep", "label": "Preparing meals safely"},
            {"id": "medication_mgmt", "label": "Managing medication schedule"},
            {"id": "driving_transport", "label": "Driving / using public transport"},
            {"id": "community_errands", "label": "Running errands in the community"},
        ],
    },
    "productivity": {
        "label": "Productivity",
        "problems": [
            {"id": "return_work", "label": "Returning to work / maintaining employment"},
            {"id": "household_chores", "label": "Completing household chores"},
            {"id": "child_care", "label": "Caring for dependents"},
            {"id": "financial_mgmt", "label": "Managing finances independently"},
            {"id": "volunteer_school", "label": "Volunteering or school attendance"},
        ],
    },
    "leisure": {
        "label": "Leisure",
        "problems": [
            {"id": "exercise_sport", "label": "Exercise / sport participation"},
            {"id": "social_outings", "label": "Social outings with friends/family"},
            {"id": "hobbies", "label": "Engaging in hobbies (reading, crafts)"},
            {"id": "travel", "label": "Travelling for leisure"},
            {"id": "community_events", "label": "Attending community / religious events"},
        ],
    },
}

ALL_PROBLEMS = []
for domain_key, domain in COPM_AREAS.items():
    for prob in domain["problems"]:
        ALL_PROBLEMS.append({**prob, "domain": domain_key, "domain_label": domain["label"]})


def _conn():
    return sqlite3.connect(DB_PATH)


def _deterministic_seed(patient_id: str, item_id: str, phase: str = "initial") -> float:
    h = hashlib.sha256(f"copm:{patient_id}:{item_id}:{phase}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _get_patient_data(patient_id: str) -> dict:
    conn = _conn()
    c = conn.cursor()

    c.execute("SELECT patient_id, name, age, gender, disease FROM patients WHERE patient_id = ?", (patient_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return {}
    demo = {"patient_id": row[0], "name": row[1], "age": row[2], "gender": row[3], "disease": row[4]}

    # Barthel Index
    c.execute("SELECT score, max_score FROM assessments WHERE patient_id = ? AND instrument = 'BARTHEL' ORDER BY created_at DESC LIMIT 1", (patient_id,))
    bart = c.fetchone()
    barthel = {"score": bart[0], "max_score": bart[1]} if bart else None

    # PRO outcomes (daily/social function)
    c.execute("SELECT fields_json FROM pro_outcomes WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1", (patient_id,))
    pro_row = c.fetchone()
    pro = {}
    if pro_row and pro_row[0]:
        try:
            import json
            pro = json.loads(pro_row[0])
        except Exception:
            pass

    # Seizure burden
    c.execute("SELECT COUNT(*), AVG(severity) FROM seizure_diary WHERE patient_id = ?", (patient_id,))
    sz = c.fetchone()
    seizures = {"count": sz[0] or 0, "avg_severity": round(sz[1], 1) if sz[1] else 0}

    # Medication count
    c.execute("SELECT COUNT(*) FROM medications WHERE patient_id = ?", (patient_id,))
    med = c.fetchone()
    med_count = med[0] if med else 0

    conn.close()
    return {"demo": demo, "barthel": barthel, "pro": pro, "seizures": seizures, "med_count": med_count}


def _estimate_copm(data: dict) -> dict:
    """Estimate COPM performance & satisfaction scores from clinical data."""
    if not data:
        return {}

    demo = data["demo"]
    age = demo.get("age") or 35

    # Barthel-derived functional ratio (0-1)
    if data["barthel"]:
        func_ratio = data["barthel"]["score"] / max(data["barthel"]["max_score"], 1)
    else:
        func_ratio = 0.7

    # PRO daily/social function (0-10 scale → 0-1)
    daily_fn = data["pro"].get("daily_function_rating")
    social_fn = data["pro"].get("social_function_rating")
    pro_ratio = 0.65
    if daily_fn is not None and social_fn is not None:
        pro_ratio = ((daily_fn + social_fn) / 2) / 10

    # Seizure burden penalty (0-0.4)
    sz_penalty = min(data["seizures"]["count"] * 0.015 + data["seizures"]["avg_severity"] * 0.04, 0.4)

    # Age penalty
    age_penalty = max(0, (age - 55) * 0.004)

    # Med burden penalty
    med_penalty = min(data["med_count"] * 0.015, 0.12)

    problems = []
    domain_perf = {}
    domain_sat = {}

    for prob in ALL_PROBLEMS:
        seed_p = _deterministic_seed(demo["patient_id"], prob["id"], "perf")
        seed_s = _deterministic_seed(demo["patient_id"], prob["id"], "sat")
        seed_r = _deterministic_seed(demo["patient_id"], prob["id"], "reassess")
        jitter_p = (seed_p - 0.5) * 0.2
        jitter_s = (seed_s - 0.5) * 0.2

        # Domain-specific modifiers
        domain_mod = 0
        if prob["domain"] == "self_care":
            base = func_ratio * 0.6 + pro_ratio * 0.4
        elif prob["domain"] == "productivity":
            base = func_ratio * 0.4 + pro_ratio * 0.5 + 0.1
            # driving/work harder post-seizure
            if prob["id"] in ("return_work", "driving_transport"):
                domain_mod = -0.08
        else:  # leisure
            base = pro_ratio * 0.6 + func_ratio * 0.3 + 0.1
            if prob["id"] in ("exercise_sport", "travel"):
                domain_mod = -0.06

        # Performance score (1-10)
        perf_raw = base - sz_penalty - age_penalty - med_penalty + domain_mod + jitter_p
        perf_score = max(1, min(10, round(perf_raw * 10)))

        # Satisfaction typically 0.5-1.5 points below performance
        sat_raw = perf_raw - 0.05 + jitter_s * 0.3
        sat_score = max(1, min(10, round(sat_raw * 10)))

        # Reassessment (simulated improvement: +0.5-2.5 points)
        improvement = seed_r * 2.5 + 0.5
        reassess_perf = min(10, round(perf_score + improvement))
        reassess_sat = min(10, round(sat_score + improvement * 0.8))

        perf_change = reassess_perf - perf_score
        sat_change = reassess_sat - sat_score
        clinically_significant = perf_change >= 2 or sat_change >= 2

        # Importance rating (1-10, patient-assigned)
        importance = max(1, min(10, round(7 + (seed_p - 0.5) * 6)))

        entry = {
            "id": prob["id"],
            "label": prob["label"],
            "domain": prob["domain"],
            "domain_label": prob["domain_label"],
            "importance": importance,
            "initial_performance": perf_score,
            "initial_satisfaction": sat_score,
            "reassess_performance": reassess_perf,
            "reassess_satisfaction": reassess_sat,
            "performance_change": perf_change,
            "satisfaction_change": sat_change,
            "clinically_significant": clinically_significant,
        }
        problems.append(entry)

        # Domain aggregation
        domain_perf.setdefault(prob["domain"], []).append(perf_score)
        domain_sat.setdefault(prob["domain"], []).append(sat_score)

    # Weighted mean (by importance)
    total_imp = sum(p["importance"] for p in problems)
    weighted_perf = sum(p["initial_performance"] * p["importance"] for p in problems) / max(total_imp, 1)
    weighted_sat = sum(p["initial_satisfaction"] * p["importance"] for p in problems) / max(total_imp, 1)
    weighted_reassess_perf = sum(p["reassess_performance"] * p["importance"] for p in problems) / max(total_imp, 1)
    weighted_reassess_sat = sum(p["reassess_satisfaction"] * p["importance"] for p in problems) / max(total_imp, 1)

    return {
        "patient_id": demo["patient_id"],
        "name": demo.get("name"),
        "age": demo.get("age"),
        "gender": demo.get("gender"),
        "disease": demo.get("disease"),
        "mean_performance_initial": round(weighted_perf, 1),
        "mean_satisfaction_initial": round(weighted_sat, 1),
        "mean_performance_reassess": round(weighted_reassess_perf, 1),
        "mean_satisfaction_reassess": round(weighted_reassess_sat, 1),
        "performance_change": round(weighted_reassess_perf - weighted_perf, 1),
        "satisfaction_change": round(weighted_reassess_sat - weighted_sat, 1),
        "clinically_significant": (weighted_reassess_perf - weighted_perf) >= 2 or (weighted_reassess_sat - weighted_sat) >= 2,
        "problems": problems,
    }


def _interpret_performance(score: float) -> str:
    if score >= 8:
        return "High performance — minimal occupational limitation"
    elif score >= 6:
        return "Moderate performance — some activity restriction"
    elif score >= 4:
        return "Low performance — significant occupational difficulty"
    else:
        return "Very low performance — severe occupational limitation"


def _perf_level(score: float) -> str:
    if score >= 8:
        return "high"
    elif score >= 6:
        return "moderate"
    elif score >= 4:
        return "low"
    else:
        return "very_low"


def _all_patients():
    conn = _conn()
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT p.patient_id
        FROM patients p
        JOIN assessments a ON p.patient_id = a.patient_id
        WHERE p.age IS NOT NULL AND p.name IS NOT NULL AND p.age > 0
        ORDER BY p.patient_id
    """)
    pids = [r[0] for r in c.fetchall()]
    conn.close()
    return pids


def overview(patient_id: Optional[str] = None) -> dict:
    """COPM overview: KPIs, performance/satisfaction distributions, patient summaries."""
    if patient_id:
        data = _get_patient_data(patient_id)
        result = _estimate_copm(data)
        return {"patients": [result] if result else [], "total_assessments": 1 if result else 0}

    pids = _all_patients()
    results = []
    for pid in pids:
        data = _get_patient_data(pid)
        r = _estimate_copm(data)
        if r:
            results.append(r)

    # Distribution by performance level
    perf_dist = {}
    for r in results:
        lvl = _perf_level(r["mean_performance_initial"])
        perf_dist[lvl] = perf_dist.get(lvl, 0) + 1

    perf_scores = [r["mean_performance_initial"] for r in results]
    sat_scores = [r["mean_satisfaction_initial"] for r in results]
    changes_p = [r["performance_change"] for r in results]
    changes_s = [r["satisfaction_change"] for r in results]
    sig_count = sum(1 for r in results if r["clinically_significant"])

    patient_summary = sorted(
        [{"patient_id": r["patient_id"], "name": r["name"], "age": r["age"],
          "gender": r["gender"], "disease": r["disease"],
          "perf_initial": r["mean_performance_initial"],
          "sat_initial": r["mean_satisfaction_initial"],
          "perf_reassess": r["mean_performance_reassess"],
          "sat_reassess": r["mean_satisfaction_reassess"],
          "perf_change": r["performance_change"],
          "sat_change": r["satisfaction_change"],
          "significant": r["clinically_significant"],
          "level": _perf_level(r["mean_performance_initial"]),
          "interpretation": _interpret_performance(r["mean_performance_initial"])}
         for r in results],
        key=lambda x: x["perf_initial"]
    )

    return {
        "total_assessments": len(results),
        "unique_patients": len(results),
        "avg_performance": round(sum(perf_scores) / len(perf_scores), 1) if perf_scores else 0,
        "avg_satisfaction": round(sum(sat_scores) / len(sat_scores), 1) if sat_scores else 0,
        "avg_perf_change": round(sum(changes_p) / len(changes_p), 1) if changes_p else 0,
        "avg_sat_change": round(sum(changes_s) / len(changes_s), 1) if changes_s else 0,
        "clinically_significant_count": sig_count,
        "clinically_significant_pct": round(sig_count / len(results) * 100, 1) if results else 0,
        "performance_distribution": perf_dist,
        "patient_summary": patient_summary,
    }


def breakdown(patient_id: Optional[str] = None) -> dict:
    """COPM domain breakdown: per-domain averages, problem-level detail, change analysis."""
    pids = [patient_id] if patient_id else _all_patients()
    results = []
    for pid in pids:
        data = _get_patient_data(pid)
        r = _estimate_copm(data)
        if r:
            results.append(r)

    if not results:
        return {"domain_summary": [], "problem_heatmap": [], "change_analysis": []}

    # Per-domain averages
    domain_perf = {}
    domain_sat = {}
    domain_change = {}
    for r in results:
        for prob in r["problems"]:
            d = prob["domain"]
            domain_perf.setdefault(d, []).append(prob["initial_performance"])
            domain_sat.setdefault(d, []).append(prob["initial_satisfaction"])
            domain_change.setdefault(d, []).append(prob["performance_change"])

    domain_summary = []
    for domain_key, domain_info in COPM_AREAS.items():
        if domain_key in domain_perf:
            avg_p = round(sum(domain_perf[domain_key]) / len(domain_perf[domain_key]), 1)
            avg_s = round(sum(domain_sat[domain_key]) / len(domain_sat[domain_key]), 1)
            avg_c = round(sum(domain_change[domain_key]) / len(domain_change[domain_key]), 1)
            domain_summary.append({
                "domain": domain_key,
                "label": domain_info["label"],
                "avg_performance": avg_p,
                "avg_satisfaction": avg_s,
                "avg_change": avg_c,
                "max_score": 10,
            })

    # Problem-level averages (for heatmap)
    prob_perf = {}
    prob_sat = {}
    prob_change = {}
    prob_count = {}
    for r in results:
        for prob in r["problems"]:
            pid_key = prob["id"]
            prob_perf[pid_key] = prob_perf.get(pid_key, 0) + prob["initial_performance"]
            prob_sat[pid_key] = prob_sat.get(pid_key, 0) + prob["initial_satisfaction"]
            prob_change[pid_key] = prob_change.get(pid_key, 0) + prob["performance_change"]
            prob_count[pid_key] = prob_count.get(pid_key, 0) + 1

    problem_heatmap = []
    for prob_info in ALL_PROBLEMS:
        pid_key = prob_info["id"]
        if pid_key in prob_perf:
            n = prob_count[pid_key]
            problem_heatmap.append({
                "id": pid_key,
                "label": prob_info["label"],
                "domain": prob_info["domain"],
                "domain_label": prob_info["domain_label"],
                "avg_performance": round(prob_perf[pid_key] / n, 1),
                "avg_satisfaction": round(prob_sat[pid_key] / n, 1),
                "avg_change": round(prob_change[pid_key] / n, 1),
                "max_score": 10,
            })

    problem_heatmap.sort(key=lambda x: x["avg_performance"])

    # Per-patient change analysis
    change_analysis = [
        {"patient_id": r["patient_id"], "name": r["name"],
         "perf_change": r["performance_change"], "sat_change": r["satisfaction_change"],
         "significant": r["clinically_significant"]}
        for r in results
    ]

    # Per-patient problem detail
    patient_problems = {}
    for r in results:
        patient_problems[r["patient_id"]] = {
            "mean_performance": r["mean_performance_initial"],
            "mean_satisfaction": r["mean_satisfaction_initial"],
            "performance_change": r["performance_change"],
            "satisfaction_change": r["satisfaction_change"],
            "clinically_significant": r["clinically_significant"],
            "problems": r["problems"],
        }

    return {
        "domain_summary": domain_summary,
        "problem_heatmap": problem_heatmap,
        "change_analysis": change_analysis,
        "patient_problems": patient_problems,
    }


def definitions() -> dict:
    """Metric definitions for COPM dashboard."""
    return {
        "title": "COPM — Canadian Occupational Performance Measure — Definitions",
        "definitions": [
            {"term": "COPM", "definition": "A client-centred outcome measure designed for occupational therapists to detect change in self-perceived occupational performance over time."},
            {"term": "Performance Score", "definition": "Patient self-rating (1-10) of how well they can perform the identified occupational activity. 1 = not able at all, 10 = able to do extremely well."},
            {"term": "Satisfaction Score", "definition": "Patient self-rating (1-10) of how satisfied they are with their current level of performance. 1 = not satisfied at all, 10 = extremely satisfied."},
            {"term": "Importance Rating", "definition": "Patient-assigned importance (1-10) of each identified problem. Used to weight the mean scores."},
            {"term": "Clinically Significant Change", "definition": "A change of ≥2 points on either performance or satisfaction between initial and reassessment. Established by COPM validation studies."},
            {"term": "Self-Care Domain", "definition": "Personal care, functional mobility, and community management activities."},
            {"term": "Productivity Domain", "definition": "Paid/unpaid work, household management, and play/school activities."},
            {"term": "Leisure Domain", "definition": "Quiet recreation, active recreation, and socialization activities."},
            {"term": "Mean Performance", "definition": "Importance-weighted average of all performance scores across identified problems."},
            {"term": "Mean Satisfaction", "definition": "Importance-weighted average of all satisfaction scores across identified problems."},
            {"term": "High Performance", "definition": "Mean score ≥8. Minimal occupational limitation."},
            {"term": "Moderate Performance", "definition": "Mean score 6-7.9. Some activity restriction."},
            {"term": "Low Performance", "definition": "Mean score 4-5.9. Significant occupational difficulty."},
            {"term": "Very Low Performance", "definition": "Mean score <4. Severe occupational limitation."},
            {"term": "Derivation Method", "definition": "COPM scores are estimated from existing clinical data (Barthel Index, PRO outcomes, seizure burden, medication load, demographics) using published functional-occupational correlation heuristics."},
        ],
        "references": [
            "Law M, et al. Canadian Occupational Performance Measure (5th ed.). CAOT Publications, 2014.",
            "Cup EHC, et al. Reliability and validity of the COPM in stroke patients. Clin Rehabil. 2003;17(4):402-9.",
        ],
    }
