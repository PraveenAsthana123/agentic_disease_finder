"""
Wisconsin Card Sorting Test (WCST) — NeuroAI Clinical Dashboard Module
=======================================================================
Measures executive function: cognitive flexibility, set-shifting, and
perseverative responding. Patients sort 128 cards (4 stimulus cards) by
color, form, or number; the sorting rule shifts without warning.

Citation
--------
Heaton RK, Chelune GJ, Talley JL, Kay GG, Curtiss G.
Wisconsin Card Sorting Test Manual: Revised and Expanded.
Psychological Assessment Resources; 2003.

Primary Metrics
---------------
- Categories Completed       (0–6; healthy adults mean ≈ 5, SD ≈ 1.2)
- Perseverative Errors (PE)  (count; healthy adults mean ≈ 12, SD ≈ 6)
- Total Errors               (count)
- Conceptual Level Responses (CLR %)
- Failure to Maintain Set    (FMS; count of category breaks)

Severity (PE-based, Heaton 2003 norms)
---------------------------------------
Normal      0–12 PE
Low-normal  13–19 PE
Borderline  20–25 PE
Impaired    > 25 PE

Clinical context — epilepsy
----------------------------
Frontal-lobe seizures and frontal-temporal AEDs (topiramate, phenobarbital,
phenytoin, zonisamide) specifically impair WCST performance via working-memory
and set-shifting circuits. WCST is the gold-standard frontal-lobe executive
function screen in epilepsy neuropsychology.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

WCST_METRICS = [
    {
        "id": 1,
        "metric": "Categories Completed",
        "abbreviation": "CC",
        "description": "Number of 10-consecutive-correct sorting runs (0–6).",
        "measures": "Ability to discover and apply sorting principles; overall executive success.",
        "norms": {"mean": 5.0, "sd": 1.2, "direction": "higher_better"},
    },
    {
        "id": 2,
        "metric": "Perseverative Errors",
        "abbreviation": "PE",
        "description": "Errors where patient continues a previously correct but now incorrect strategy.",
        "measures": "Cognitive inflexibility; inability to shift mental set.",
        "norms": {"mean": 12.0, "sd": 6.0, "direction": "lower_better"},
    },
    {
        "id": 3,
        "metric": "Total Errors",
        "abbreviation": "TE",
        "description": "All incorrect responses across all 128 trials.",
        "measures": "Overall accuracy and learning efficiency.",
        "norms": {"mean": 22.0, "sd": 9.0, "direction": "lower_better"},
    },
    {
        "id": 4,
        "metric": "Conceptual Level Responses",
        "abbreviation": "CLR%",
        "description": "Percentage of responses in runs of 3+ consecutive correct sorts.",
        "measures": "Sustained conceptual understanding; quality of category maintenance.",
        "norms": {"mean": 72.0, "sd": 12.0, "direction": "higher_better"},
    },
    {
        "id": 5,
        "metric": "Failure to Maintain Set",
        "abbreviation": "FMS",
        "description": "Number of times a category run (≥5 correct) was broken before completion.",
        "measures": "Attentional lapses; difficulty maintaining a correct strategy.",
        "norms": {"mean": 0.8, "sd": 1.1, "direction": "lower_better"},
    },
]

# Severity bands keyed on Perseverative Errors (PE) — Heaton 2003 adult norms
SEVERITY_BANDS = [
    {
        "range": [0, 12],
        "label": "Normal",
        "color": "green",
        "description": "PE within expected range for healthy adults; cognitive flexibility intact.",
    },
    {
        "range": [13, 19],
        "label": "Low-normal",
        "color": "olive",
        "description": "Mildly elevated PE; subtle set-shifting difficulty, borderline concern.",
    },
    {
        "range": [20, 25],
        "label": "Borderline",
        "color": "orange",
        "description": "Elevated PE consistent with frontal-executive dysfunction; monitor closely.",
    },
    {
        "range": [26, 999],
        "label": "Impaired",
        "color": "red",
        "description": "Significantly impaired cognitive flexibility; frontal-lobe assessment indicated.",
    },
]

# All recognised AEDs (same 25-agent set as Stroop module)
AEDS_SET = {
    "levetiracetam",
    "carbamazepine",
    "valproate",
    "lamotrigine",
    "topiramate",
    "oxcarbazepine",
    "phenytoin",
    "phenobarbital",
    "lacosamide",
    "zonisamide",
    "gabapentin",
    "pregabalin",
    "clobazam",
    "clonazepam",
    "ethosuximide",
    "rufinamide",
    "perampanel",
    "brivaracetam",
    "eslicarbazepine",
    "vigabatrin",
    "stiripentol",
    "felbamate",
    "tiagabine",
    "cannabidiol",
    "cenobamate",
}

# AEDs with documented high frontal-executive cognitive burden
HIGH_COGNITIVE_BURDEN_AEDS = {
    "topiramate",
    "phenobarbital",
    "phenytoin",
    "zonisamide",
    "clobazam",
    "clonazepam",
    "vigabatrin",
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _conn():
    """Return a SQLite connection to clinical.db."""
    return sqlite3.connect(DB_PATH)


def _get_patient_data(patient_id: str) -> dict:
    """
    Fetch real patient data from clinical.db for WCST estimation.

    Returns
    -------
    dict with keys:
        demographics, barthel, seizure_count_30d,
        aed_count, aed_names, high_burden_aeds
    """
    result = {
        "demographics": {},
        "barthel": 100,
        "seizure_count_30d": 0,
        "aed_count": 0,
        "aed_names": [],
        "high_burden_aeds": [],
    }

    try:
        with _conn() as con:
            cur = con.cursor()

            # Demographics
            cur.execute(
                "SELECT patient_id, name, age, gender, disease_type "
                "FROM patients WHERE patient_id = ?",
                (patient_id,),
            )
            row = cur.fetchone()
            if row:
                result["demographics"] = {
                    "patient_id": row[0],
                    "name": row[1],
                    "age": row[2],
                    "gender": row[3],
                    "disease_type": row[4],
                }

            # Latest Barthel index
            cur.execute(
                "SELECT score FROM assessments "
                "WHERE patient_id = ? AND assessment_type = 'BARTHEL' "
                "ORDER BY assessment_date DESC LIMIT 1",
                (patient_id,),
            )
            row = cur.fetchone()
            if row:
                result["barthel"] = row[0]

            # Seizure count in last 30 days
            cur.execute(
                "SELECT COUNT(*) FROM seizure_diary "
                "WHERE patient_id = ? "
                "AND seizure_date >= date('now', '-30 days')",
                (patient_id,),
            )
            row = cur.fetchone()
            if row:
                result["seizure_count_30d"] = row[0]

            # Medications
            cur.execute(
                "SELECT fields_json FROM medications WHERE patient_id = ?",
                (patient_id,),
            )
            rows = cur.fetchall()
            aed_names: list[str] = []
            for (fields_raw,) in rows:
                try:
                    fields = json.loads(fields_raw)
                    if isinstance(fields, list):
                        for item in fields:
                            if isinstance(item, dict):
                                name = (
                                    item.get("medication_name", "")
                                    or item.get("drug_name", "")
                                    or item.get("name", "")
                                ).lower().strip()
                                if name in AEDS_SET:
                                    aed_names.append(name)
                    elif isinstance(fields, dict):
                        name = (
                            fields.get("medication_name", "")
                            or fields.get("drug_name", "")
                            or fields.get("name", "")
                        ).lower().strip()
                        if name in AEDS_SET:
                            aed_names.append(name)
                except (json.JSONDecodeError, TypeError):
                    pass

            aed_names = list(set(aed_names))
            result["aed_count"] = len(aed_names)
            result["aed_names"] = aed_names
            result["high_burden_aeds"] = [
                a for a in aed_names if a in HIGH_COGNITIVE_BURDEN_AEDS
            ]

    except Exception:
        pass  # Return defaults if DB unavailable

    return result


# ---------------------------------------------------------------------------
# Core estimation
# ---------------------------------------------------------------------------

def _estimate_wcst(data: dict) -> dict:
    """
    Estimate WCST metrics for a single patient using real clinical data.

    Methodology
    -----------
    Baseline norms (Heaton 2003) are adjusted by deterministic clinical
    modifiers. A hash-seeded per-patient noise term ensures reproducibility
    across calls.

    Modifier logic
    --------------
    Categories Completed (base 5.0, lower is worse):
        age_delta   : −0.03/year after 50; additional −0.02/year after 65
        disease_delta: epilepsy −0.6; parkinson −1.0; depression −0.4;
                       alzheimer −1.5; adhd −0.8; default −0.2
        aed_delta   : count ≥3 → −1.2; 2 → −0.6; 1 → −0.2;
                      high-burden count * −0.5
        sz_delta    : sz >10 → −0.6; >5 → −0.3; >0 → −0.1
        func_delta  : barthel <60 → −0.5; <80 → −0.2

    Perseverative Errors (base 12.0, higher is worse — direction reversed):
        age_delta   : +0.3/year after 50; additional +0.2/year after 65
        disease_delta: epilepsy +7; parkinson +10; depression +5;
                       alzheimer +15; adhd +8; default +3
        aed_delta   : count ≥3 → +12; 2 → +6; 1 → +2;
                      high-burden count * +5
        sz_delta    : sz >10 → +6; >5 → +3; >0 → +1
        func_delta  : barthel <60 → +5; <80 → +2
    """
    demo = data.get("demographics", {})
    pid = demo.get("patient_id", "unknown")
    age = demo.get("age", 40)
    disease = (demo.get("disease_type", "") or "").lower()
    barthel = data.get("barthel", 100)
    sz_count = data.get("seizure_count_30d", 0)
    aed_count = data.get("aed_count", 0)
    high_burden_count = len(data.get("high_burden_aeds", []))

    # --- Deterministic per-patient seed ---
    seed = int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)

    # === Categories Completed (base 5.0, subtract for impairment) ===
    cc_base = 5.0

    # Age delta (worsening after 50)
    age_cc_delta = 0.0
    if age > 65:
        age_cc_delta = -(0.03 * (65 - 50)) - (0.05 * (age - 65))
    elif age > 50:
        age_cc_delta = -0.03 * (age - 50)

    # Disease delta
    cc_disease_map = {
        "epilepsy": -0.6,
        "parkinson": -1.0,
        "depression": -0.4,
        "alzheimer": -1.5,
        "adhd": -0.8,
    }
    cc_disease_delta = next(
        (v for k, v in cc_disease_map.items() if k in disease), -0.2
    )

    # AED delta
    if aed_count >= 3:
        cc_aed_delta = -1.2
    elif aed_count == 2:
        cc_aed_delta = -0.6
    elif aed_count == 1:
        cc_aed_delta = -0.2
    else:
        cc_aed_delta = 0.0
    cc_aed_delta -= high_burden_count * 0.5

    # Seizure delta
    if sz_count > 10:
        cc_sz_delta = -0.6
    elif sz_count > 5:
        cc_sz_delta = -0.3
    elif sz_count > 0:
        cc_sz_delta = -0.1
    else:
        cc_sz_delta = 0.0

    # Functional status delta
    if barthel < 60:
        cc_func_delta = -0.5
    elif barthel < 80:
        cc_func_delta = -0.2
    else:
        cc_func_delta = 0.0

    # Noise: ±0.3 categories (deterministic)
    cc_noise = ((seed % 61) - 30) / 100.0

    cc_raw = (
        cc_base
        + age_cc_delta
        + cc_disease_delta
        + cc_aed_delta
        + cc_sz_delta
        + cc_func_delta
        + cc_noise
    )
    categories_completed = max(0, min(6, round(cc_raw)))

    # === Perseverative Errors (base 12, add for impairment) ===
    pe_base = 12.0

    # Age delta
    age_pe_delta = 0.0
    if age > 65:
        age_pe_delta = (0.3 * (65 - 50)) + (0.5 * (age - 65))
    elif age > 50:
        age_pe_delta = 0.3 * (age - 50)

    # Disease delta
    pe_disease_map = {
        "epilepsy": 7.0,
        "parkinson": 10.0,
        "depression": 5.0,
        "alzheimer": 15.0,
        "adhd": 8.0,
    }
    pe_disease_delta = next(
        (v for k, v in pe_disease_map.items() if k in disease), 3.0
    )

    # AED delta
    if aed_count >= 3:
        pe_aed_delta = 12.0
    elif aed_count == 2:
        pe_aed_delta = 6.0
    elif aed_count == 1:
        pe_aed_delta = 2.0
    else:
        pe_aed_delta = 0.0
    pe_aed_delta += high_burden_count * 5.0

    # Seizure delta
    if sz_count > 10:
        pe_sz_delta = 6.0
    elif sz_count > 5:
        pe_sz_delta = 3.0
    elif sz_count > 0:
        pe_sz_delta = 1.0
    else:
        pe_sz_delta = 0.0

    # Functional status delta
    if barthel < 60:
        pe_func_delta = 5.0
    elif barthel < 80:
        pe_func_delta = 2.0
    else:
        pe_func_delta = 0.0

    # Noise: ±3 PE (deterministic, different seed slice)
    pe_noise = ((seed >> 8) % 61) - 30  # –30 to +30
    pe_noise = pe_noise / 10.0  # –3.0 to +3.0

    pe_raw = (
        pe_base
        + age_pe_delta
        + pe_disease_delta
        + pe_aed_delta
        + pe_sz_delta
        + pe_func_delta
        + pe_noise
    )
    perseverative_errors = max(0, round(pe_raw))

    # === Total Errors (base 22, correlated with PE) ===
    te_noise = ((seed >> 16) % 41) - 20
    total_errors = max(perseverative_errors, round(pe_raw * 1.8 + te_noise / 5.0))
    total_errors = min(128, total_errors)

    # === CLR % (base 72, decreases with impairment) ===
    clr_base = 72.0
    clr_penalty = (pe_raw - 12.0) * 1.2  # 1.2% loss per PE above mean
    clr_noise = ((seed >> 24) % 21) - 10
    clr_pct = max(0.0, min(100.0, round(clr_base - clr_penalty + clr_noise / 5.0, 1)))

    # === FMS (base 0.8, increases with impairment) ===
    fms_extra = max(0.0, (pe_raw - 12.0) / 10.0)
    fms_noise = ((seed >> 32) % 11) - 5
    fms_raw = 0.8 + fms_extra + fms_noise / 20.0
    fms_count = max(0, round(fms_raw))

    # === Z-score and percentile (PE-based, Heaton norms: mean 12, SD 6) ===
    pe_sd = 6.0
    pe_mean = 12.0
    pe_z = round((perseverative_errors - pe_mean) / pe_sd, 2)
    # Higher PE = worse; percentile reflects impairment probability
    # Approximated from z-score (one-tailed: how many SD above mean)
    pe_pct_lookup = {
        -3: 99, -2: 97, -1: 84, 0: 50, 1: 16, 2: 5, 3: 1
    }
    # Linear interpolation
    z_floor = max(-3, min(3, int(pe_z)))
    z_ceil = min(3, z_floor + 1)
    frac = pe_z - z_floor
    pct_floor = pe_pct_lookup.get(z_floor, 50)
    pct_ceil = pe_pct_lookup.get(z_ceil, 50)
    pe_percentile_impairment = max(1, min(99, round(pct_floor + frac * (pct_ceil - pct_floor))))

    # === Severity band (PE-based) ===
    severity_info = SEVERITY_BANDS[-1].copy()
    for band in SEVERITY_BANDS:
        if band["range"][0] <= perseverative_errors <= band["range"][1]:
            severity_info = band.copy()
            break

    return {
        "categories_completed": categories_completed,
        "perseverative_errors": perseverative_errors,
        "total_errors": total_errors,
        "clr_pct": clr_pct,
        "fms_count": fms_count,
        "pe_z_score": pe_z,
        "pe_percentile_impairment": pe_percentile_impairment,
        "severity": severity_info["label"],
        "severity_color": severity_info["color"],
        "severity_description": severity_info["description"],
    }


# ---------------------------------------------------------------------------
# Public API — dashboard
# ---------------------------------------------------------------------------

def wcst_dashboard(patient_id: str = None) -> dict:
    """
    Return WCST results for a single patient or all patients.

    Parameters
    ----------
    patient_id : str, optional
        If provided, return results for that patient only.
        If None (default), aggregate across all patients in the DB.

    Returns (single patient)
    -------------------------
    dict with: patient_id, patient_name, age, disease, data_sources,
               categories_completed, perseverative_errors, total_errors,
               clr_pct, fms_count, pe_z_score, pe_percentile_impairment,
               severity, severity_color, severity_description

    Returns (all patients)
    ----------------------
    dict with: scale_name, total_patients, patients (list),
               severity_distribution, mean_categories_completed,
               mean_pe, impairment_rate_pct, norm_reference
    """
    if patient_id:
        data = _get_patient_data(patient_id)
        est = _estimate_wcst(data)
        demo = data.get("demographics", {})
        return {
            "patient_id": patient_id,
            "patient_name": demo.get("name", "Unknown"),
            "age": demo.get("age", None),
            "disease": demo.get("disease_type", "Unknown"),
            "data_sources": {
                "aed_count": data.get("aed_count", 0),
                "aed_names": data.get("aed_names", []),
                "high_burden_aeds": data.get("high_burden_aeds", []),
                "seizure_count_30d": data.get("seizure_count_30d", 0),
                "barthel": data.get("barthel", 100),
            },
            **est,
        }

    # All patients
    try:
        with _conn() as con:
            cur = con.cursor()
            cur.execute("SELECT patient_id FROM patients")
            pids = [r[0] for r in cur.fetchall()]
    except Exception:
        pids = []

    patients = []
    for pid in pids:
        d = _get_patient_data(pid)
        est = _estimate_wcst(d)
        demo = d.get("demographics", {})
        patients.append({
            "patient_id": pid,
            "patient_name": demo.get("name", "Unknown"),
            "age": demo.get("age", None),
            "disease": demo.get("disease_type", "Unknown"),
            **est,
        })

    if not patients:
        return {
            "scale_name": "Wisconsin Card Sorting Test (WCST)",
            "total_patients": 0,
            "patients": [],
            "severity_distribution": {},
            "mean_categories_completed": None,
            "mean_pe": None,
            "impairment_rate_pct": None,
            "norm_reference": "Heaton RK et al. 2003",
        }

    # Aggregates
    severity_dist: dict[str, int] = {}
    for p in patients:
        sev = p.get("severity", "Unknown")
        severity_dist[sev] = severity_dist.get(sev, 0) + 1

    mean_cc = round(
        sum(p["categories_completed"] for p in patients) / len(patients), 2
    )
    mean_pe = round(
        sum(p["perseverative_errors"] for p in patients) / len(patients), 1
    )
    impaired_count = sum(
        1 for p in patients
        if p.get("severity") in ("Borderline", "Impaired")
    )
    impairment_rate = round(impaired_count / len(patients) * 100, 1)

    return {
        "scale_name": "Wisconsin Card Sorting Test (WCST)",
        "total_patients": len(patients),
        "patients": patients,
        "severity_distribution": severity_dist,
        "mean_categories_completed": mean_cc,
        "mean_pe": mean_pe,
        "impairment_rate_pct": impairment_rate,
        "norm_reference": "Heaton RK et al. 2003",
    }


# ---------------------------------------------------------------------------
# Public API — detail
# ---------------------------------------------------------------------------

def wcst_detail(patient_id: str) -> dict:
    """
    Return full WCST detail for one patient, including contributing factors
    and clinical recommendations.

    Extends wcst_dashboard() with:
        contributing_factors : dict — quantified per-domain contributions
        clinical_recommendations : list[str]
    """
    base = wcst_dashboard(patient_id)
    data = _get_patient_data(patient_id)
    demo = data.get("demographics", {})

    age = demo.get("age", 40)
    disease = (demo.get("disease_type", "") or "").lower()
    barthel = data.get("barthel", 100)
    sz_count = data.get("seizure_count_30d", 0)
    aed_count = data.get("aed_count", 0)
    high_burden = data.get("high_burden_aeds", [])
    severity = base.get("severity", "Normal")
    pe = base.get("perseverative_errors", 12)
    cc = base.get("categories_completed", 5)

    # Contributing factors (qualitative impact estimates on PE)
    contributing_factors = {
        "age_effect": (
            "Minimal" if age <= 50
            else "Moderate" if age <= 65
            else "Significant"
        ),
        "disease_effect": (
            "High" if any(d in disease for d in ["epilepsy", "parkinson", "alzheimer"])
            else "Moderate" if any(d in disease for d in ["adhd", "depression"])
            else "Low"
        ),
        "aed_polypharmacy": f"{aed_count} AED(s) — " + (
            "high burden" if aed_count >= 3 else
            "moderate burden" if aed_count == 2 else
            "low burden" if aed_count == 1 else "none"
        ),
        "high_burden_aeds": high_burden if high_burden else ["none identified"],
        "seizure_frequency": (
            f"{sz_count} seizures/30d — " + (
                "high impact" if sz_count > 10
                else "moderate impact" if sz_count > 5
                else "low impact" if sz_count > 0
                else "no recent seizures"
            )
        ),
        "functional_status": (
            f"Barthel {barthel} — " + (
                "significant impairment" if barthel < 60
                else "mild-moderate impairment" if barthel < 80
                else "functionally independent"
            )
        ),
    }

    # Clinical recommendations
    recs: list[str] = []

    if severity == "Impaired":
        recs.append(
            "WCST scores indicate significant frontal-executive dysfunction; "
            "consider comprehensive neuropsychological assessment and neuroimaging."
        )
    elif severity == "Borderline":
        recs.append(
            "Borderline WCST performance; close cognitive monitoring every 6 months "
            "and functional assessment recommended."
        )

    if high_burden:
        burden_list = ", ".join(high_burden)
        recs.append(
            f"High-burden AEDs identified ({burden_list}); review pharmacotherapy — "
            "switching topiramate or phenobarbital to levetiracetam or lamotrigine "
            "may improve set-shifting and reduce perseverative errors."
        )

    if aed_count >= 3:
        recs.append(
            "Polypharmacy (≥3 AEDs) significantly burdens executive function; "
            "rationalisation of AED regimen to ≤2 agents should be evaluated."
        )

    if sz_count > 5:
        recs.append(
            f"{sz_count} seizures in the past 30 days likely contribute to "
            "post-ictal and inter-ictal frontal-lobe dysfunction; "
            "optimise seizure control as the primary intervention."
        )

    if pe > 25:
        recs.append(
            "Perseverative Error count > 25 meets Heaton 2003 impairment threshold; "
            "consider occupational therapy targeting cognitive flexibility and "
            "structured problem-solving programmes."
        )

    if cc < 3:
        recs.append(
            "Fewer than 3 categories completed (impairment threshold); "
            "patient may benefit from cognitive rehabilitation with set-shifting exercises."
        )

    if not recs:
        recs.append(
            "WCST performance within or near normal limits; "
            "continue routine annual executive-function monitoring."
        )

    base["contributing_factors"] = contributing_factors
    base["clinical_recommendations"] = recs
    return base


# ---------------------------------------------------------------------------
# Public API — trend
# ---------------------------------------------------------------------------

def wcst_trend(patient_id: str) -> dict:
    """
    Project WCST trajectory over 12 months for one patient.

    Trajectory logic
    ----------------
    Categories Completed improve (increase) and PE decrease with:
        - AED optimisation  : starts month 2 if high_burden_count > 0
          CC +0.07/month, PE −0.8/month
        - Seizure control   : starts month 1 if sz_count > 5
          CC +0.04/month, PE −0.5/month
        - Age-related decline: CC −0.015/month, PE +0.2/month if age > 50
          (gradual background worsening regardless of intervention)

    Returns
    -------
    dict with: patient_id, patient_name, baseline_categories,
               baseline_pe, baseline_severity, trajectory (list of 13 points),
               assumptions
    """
    data = _get_patient_data(patient_id)
    demo = data.get("demographics", {})
    est = _estimate_wcst(data)

    age = demo.get("age", 40)
    sz_count = data.get("seizure_count_30d", 0)
    high_burden_count = len(data.get("high_burden_aeds", []))

    baseline_cc = est["categories_completed"]
    baseline_pe = est["perseverative_errors"]
    baseline_severity = est["severity"]

    trajectory = []
    current_cc = float(baseline_cc)
    current_pe = float(baseline_pe)

    for month in range(13):  # 0 to 12
        # AED optimisation effect (month 2+)
        if month >= 2 and high_burden_count > 0:
            current_cc += 0.07
            current_pe -= 0.8

        # Seizure control effect (month 1+)
        if month >= 1 and sz_count > 5:
            current_cc += 0.04
            current_pe -= 0.5

        # Background age-related decline
        if month > 0 and age > 50:
            current_cc -= 0.015
            current_pe += 0.2

        # Clamp to valid ranges
        cc_point = max(0, min(6, round(current_cc, 2)))
        pe_point = max(0, round(current_pe, 1))

        # Severity at this time point
        sev_label = SEVERITY_BANDS[-1]["label"]
        for band in SEVERITY_BANDS:
            if band["range"][0] <= pe_point <= band["range"][1]:
                sev_label = band["label"]
                break

        trajectory.append({
            "month": month,
            "projected_categories_completed": cc_point,
            "projected_pe": pe_point,
            "severity": sev_label,
            "label": f"Month {month}" if month > 0 else "Baseline",
        })

    # Build assumptions list
    assumptions = [
        f"Baseline: {baseline_cc} categories completed, {baseline_pe} PE ({baseline_severity})",
        f"Heaton 2003 norms: mean 5 categories (SD 1.2), mean 12 PE (SD 6)",
        "AED optimisation benefit applies from month 2 when high-burden AEDs are present",
        "Seizure control benefit applies from month 1 when >5 seizures/30 days",
        f"Age-related decline (age {age}): " + (
            "−0.015 CC/month, +0.2 PE/month applied (age >50)" if age > 50
            else "not applied (age ≤50)"
        ),
        "Projections are model estimates only; clinical outcomes may differ",
    ]
    if high_burden_count > 0:
        assumptions.append(
            f"{high_burden_count} high-burden AED(s) identified; "
            "CC +0.07/month, PE −0.8/month improvement modelled from month 2"
        )
    if sz_count > 5:
        assumptions.append(
            f"Frequent seizures ({sz_count}/30d); CC +0.04/month, PE −0.5/month "
            "improvement modelled from month 1 with seizure control"
        )

    return {
        "patient_id": patient_id,
        "patient_name": demo.get("name", "Unknown"),
        "baseline_categories_completed": baseline_cc,
        "baseline_pe": baseline_pe,
        "baseline_severity": baseline_severity,
        "trajectory": trajectory,
        "assumptions": assumptions,
    }


# ---------------------------------------------------------------------------
# Public API — scale definitions
# ---------------------------------------------------------------------------

def scale_definitions() -> dict:
    """
    Return metadata and reference information for the WCST.
    """
    return {
        "scale_name": "Wisconsin Card Sorting Test",
        "abbreviation": "WCST",
        "author": "Berg EA (original); Heaton RK et al. (revised and expanded norms)",
        "reference": (
            "Heaton RK, Chelune GJ, Talley JL, Kay GG, Curtiss G. "
            "Wisconsin Card Sorting Test Manual: Revised and Expanded. "
            "Psychological Assessment Resources; 2003."
        ),
        "purpose": (
            "Assess executive functions including concept formation, cognitive flexibility, "
            "set-shifting, and ability to use feedback to guide behaviour. Gold-standard "
            "frontal-lobe measure in neuropsychological evaluation."
        ),
        "administration": {
            "stimuli": "128 response cards sorted to 4 stimulus cards",
            "rules": "Color, form, or number — shifts without warning",
            "feedback": "Examiner says 'right' or 'wrong' after each sort",
            "duration_minutes": "20–30",
            "version": "Manual (standard) or computerised (WCST-64, PAR)",
        },
        "metrics": WCST_METRICS,
        "primary_metric": "Perseverative Errors (PE) — key indicator of frontal inflexibility",
        "severity_bands": SEVERITY_BANDS,
        "normative_data": {
            "source": "Heaton RK et al. 2003 — age- and education-stratified norms",
            "healthy_adult_categories_mean": 5.0,
            "healthy_adult_categories_sd": 1.2,
            "healthy_adult_pe_mean": 12.0,
            "healthy_adult_pe_sd": 6.0,
            "impairment_threshold_categories": "< 3 categories completed",
            "impairment_threshold_pe": "> 25 perseverative errors",
            "note": (
                "Norms are stratified by age (20–89) and education (8–16+ years); "
                "clinical interpretation requires age/education-corrected T-scores."
            ),
        },
        "psychometrics": {
            "test_retest_reliability": "r = 0.41–0.68 (PE most stable)",
            "validity": (
                "Sensitive to frontal lobe lesions, prefrontal dysfunction, "
                "and conditions affecting the dorsolateral prefrontal cortex"
            ),
            "construct": "Executive function — cognitive flexibility, set-shifting, working memory",
            "sensitivity_frontal": "Moderate-high (frontal lesions produce elevated PE)",
            "ecological_validity": (
                "PE correlates with real-world problems in planning, "
                "decision-making, and adapting to change"
            ),
        },
        "epilepsy_context": {
            "applications": [
                "Pre-surgical neuropsychological evaluation (esp. frontal / temporal lobe epilepsy)",
                "AED cognitive monitoring — detecting set-shifting impairment",
                "Baseline and follow-up in clinical trials assessing cognitive AED side-effects",
                "Frontal lobe epilepsy localisation support",
                "Post-ictal cognitive recovery assessment",
            ],
            "aed_effects": {
                "high_burden_aeds": list(HIGH_COGNITIVE_BURDEN_AEDS),
                "effect_summary": (
                    "Topiramate and phenobarbital most strongly impair set-shifting "
                    "and working memory, producing elevated PE and reduced categories. "
                    "Lamotrigine and levetiracetam show minimal frontal-executive burden. "
                    "Zonisamide and clobazam produce intermediate effects."
                ),
                "frontal_lobe_note": (
                    "Frontal-lobe seizures and ictal/inter-ictal frontal dysfunction "
                    "independently elevate PE irrespective of AED load; "
                    "disentangling AED vs seizure contributions requires repeat testing "
                    "during seizure-free periods."
                ),
            },
        },
        "data_derivation": (
            "Scores are clinically modelled from real patient data (demographics, "
            "AED regimen, seizure frequency, Barthel Index) using Heaton 2003 "
            "normative baselines with deterministic hash-seeded per-patient noise. "
            "Intended for research, educational, and prototype purposes only. "
            "Replace with administered WCST data for clinical decision-making."
        ),
    }


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    d = wcst_dashboard()
    print(f"WCST Dashboard: {d['total_patients']} patients")
    print(f"Mean categories completed: {d['mean_categories_completed']}")
    print(f"Mean perseverative errors: {d['mean_pe']}")
    print(f"Impairment rate: {d['impairment_rate_pct']}%")
    print(f"Distribution: {d['severity_distribution']}")
