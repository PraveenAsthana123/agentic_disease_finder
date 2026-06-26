"""
Rey Auditory Verbal Learning Test (RAVLT) — NeuroAI Clinical Dashboard Module
===============================================================================
Measures verbal learning and memory across five learning trials, an interference
trial, immediate and delayed recall, and recognition. The RAVLT is the most
widely used word-list learning test in clinical neuropsychology.

Citation
--------
Lezak MD, Howieson DB, Bigler ED, Tranel D.
Neuropsychological Assessment. 5th ed. Oxford University Press; 2012.

Mitrushina M, Boone KB, Razani J, D'Elia LF.
Handbook of Normative Data for Neuropsychological Assessment. 2nd ed.
Oxford University Press; 2005.

Primary Metrics
---------------
- Trial 1 (T1)             attention span / immediate registration (mean ~6, SD ~2)
- Trial 5 (T5)             total learning ceiling (mean ~13, SD ~2)
- Total Learning (T1–T5)   sum of words recalled across 5 trials (mean ~52, SD ~8)
- List B                   interference susceptibility (mean ~6, SD ~2)
- Trial 6 (post-interf.)   retroactive interference (mean ~11, SD ~3)
- Delayed Recall (DR)      20-min long-term retention (mean ~11, SD ~3)
- Recognition Hits         out of 15 — discrimination ability (mean ~14, SD ~1.5)
- Learning Slope           (T5 − T1) / 4 — rate of acquisition
- Forgetting Rate          (T5 − DR) / T5 × 100 — retention loss percentage

Severity (DR-based z-scores, Lezak 2012 / Mitrushina 2005 norms)
------------------------------------------------------------------
Normal       z ≥ −1.0        (green)
Low-normal   −1.5 ≤ z < −1.0 (olive)
Borderline   −2.0 ≤ z < −1.5 (orange)
Impaired     z < −2.0        (red)

Clinical context — epilepsy
----------------------------
Temporal-lobe epilepsy (TLE) specifically impairs RAVLT via hippocampal
dysfunction. Left TLE produces dominant-hemisphere verbal memory deficits
(lower delayed recall), while right TLE has less RAVLT impact. AEDs such
as topiramate, phenobarbital, phenytoin, and zonisamide impair verbal
learning and consolidation.

Helmstaedter C, Kurthen M. Memory and temporal lobe epilepsy: a critical
review. Epilepsy & Behavior. 2001;2(3):126–150.
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

RAVLT_METRICS = [
    {
        "id": 1,
        "metric": "Trial 1 (T1)",
        "abbreviation": "T1",
        "description": "Words recalled on the first presentation of List A (0–15).",
        "measures": "Immediate attention span and initial registration capacity.",
        "norms": {"mean": 6.0, "sd": 2.0, "direction": "higher_better"},
    },
    {
        "id": 2,
        "metric": "Trial 5 (T5)",
        "abbreviation": "T5",
        "description": "Words recalled on the fifth presentation of List A (0–15).",
        "measures": "Maximal learning capacity after repeated exposure.",
        "norms": {"mean": 13.0, "sd": 2.0, "direction": "higher_better"},
    },
    {
        "id": 3,
        "metric": "Total Learning (T1–T5)",
        "abbreviation": "TL",
        "description": "Sum of words recalled across all five learning trials (0–75).",
        "measures": "Overall verbal learning capacity and acquisition efficiency.",
        "norms": {"mean": 52.0, "sd": 8.0, "direction": "higher_better"},
    },
    {
        "id": 4,
        "metric": "List B",
        "abbreviation": "LB",
        "description": "Words recalled from a single-trial interference list (0–15).",
        "measures": "Proactive interference susceptibility and incidental learning.",
        "norms": {"mean": 6.0, "sd": 2.0, "direction": "higher_better"},
    },
    {
        "id": 5,
        "metric": "Trial 6 (Post-Interference Recall)",
        "abbreviation": "T6",
        "description": "List A recall immediately after List B interference (0–15).",
        "measures": "Resistance to retroactive interference; short-term retention.",
        "norms": {"mean": 11.0, "sd": 3.0, "direction": "higher_better"},
    },
    {
        "id": 6,
        "metric": "Delayed Recall (20 min)",
        "abbreviation": "DR",
        "description": "List A recall after a 20-minute filled delay (0–15).",
        "measures": "Long-term verbal memory consolidation; hippocampal integrity.",
        "norms": {"mean": 11.0, "sd": 3.0, "direction": "higher_better"},
    },
    {
        "id": 7,
        "metric": "Recognition Hits",
        "abbreviation": "RH",
        "description": "Correct identifications of List A words from a 50-word list (0–15).",
        "measures": "Discrimination ability; encoding vs retrieval dissociation.",
        "norms": {"mean": 14.0, "sd": 1.5, "direction": "higher_better"},
    },
    {
        "id": 8,
        "metric": "Learning Slope",
        "abbreviation": "LS",
        "description": "(T5 − T1) / 4 — average words gained per trial.",
        "measures": "Rate of verbal learning across repeated exposures.",
        "norms": {"mean": 1.75, "sd": 0.5, "direction": "higher_better"},
    },
    {
        "id": 9,
        "metric": "Forgetting Rate",
        "abbreviation": "FR%",
        "description": "(T5 − DR) / T5 × 100 — percentage of learned material lost.",
        "measures": "Retention efficiency; consolidation failure indicator.",
        "norms": {"mean": 15.0, "sd": 10.0, "direction": "lower_better"},
    },
]

# Severity bands keyed on Delayed Recall z-score — Lezak 2012 / Mitrushina 2005
SEVERITY_BANDS = [
    {
        "range": [-1.0, 999.0],
        "label": "Normal",
        "color": "green",
        "description": (
            "Delayed recall within expected range (z ≥ −1.0); "
            "verbal memory consolidation intact."
        ),
    },
    {
        "range": [-1.5, -1.01],
        "label": "Low-normal",
        "color": "olive",
        "description": (
            "Mildly reduced delayed recall (−1.5 ≤ z < −1.0); "
            "subtle verbal memory concern, borderline findings."
        ),
    },
    {
        "range": [-2.0, -1.51],
        "label": "Borderline",
        "color": "orange",
        "description": (
            "Moderately impaired delayed recall (−2.0 ≤ z < −1.5); "
            "consistent with hippocampal dysfunction, monitor closely."
        ),
    },
    {
        "range": [-999.0, -2.01],
        "label": "Impaired",
        "color": "red",
        "description": (
            "Severely impaired delayed recall (z < −2.0); "
            "significant verbal memory deficit, comprehensive assessment indicated."
        ),
    },
]

# All recognised AEDs (same 25-agent set as WCST / Stroop modules)
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

# AEDs with documented high verbal-memory / hippocampal cognitive burden
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
    Fetch real patient data from clinical.db for RAVLT estimation.

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

            # Demographics — column is 'disease' in schema (some modules
            # use 'disease_type'; we try both for portability)
            try:
                cur.execute(
                    "SELECT patient_id, name, age, gender, disease "
                    "FROM patients WHERE patient_id = ?",
                    (patient_id,),
                )
            except Exception:
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

def _estimate_ravlt(data: dict) -> dict:
    """
    Estimate RAVLT metrics for a single patient using real clinical data.

    Methodology
    -----------
    Baseline norms (Lezak 2012 / Mitrushina 2005, healthy adults 20–49) are
    adjusted by deterministic clinical modifiers. A hash-seeded per-patient
    noise term ensures reproducibility across calls.

    Modifier logic
    --------------
    Delayed Recall (DR, base 11.0, lower = worse):
        age_delta   : −0.08/year after 50; additional −0.06/year after 65
        disease_delta: epilepsy −2.5 (TLE hippocampal); parkinson −1.5;
                       alzheimer −4.0; depression −1.0; adhd −0.6; default −0.5
        aed_delta   : count ≥3 → −2.5; 2 → −1.2; 1 → −0.4;
                      high-burden count * −0.8
        sz_delta    : sz >10 → −1.5; >5 → −0.8; >0 → −0.3
        func_delta  : barthel <60 → −1.0; <80 → −0.4

    Trial 5 (T5, base 13.0) and Total Learning (TL, base 52.0) follow
    similar modifier patterns but with different magnitudes. The critical
    clinical feature is that DR drops *more* than T5 in epilepsy (reflecting
    consolidation failure due to hippocampal dysfunction — the hallmark of TLE).
    """
    demo = data.get("demographics", {})
    pid = demo.get("patient_id", "unknown")
    age = demo.get("age") or 40
    disease = (demo.get("disease_type", "") or "").lower()
    barthel = data.get("barthel") or 100
    sz_count = data.get("seizure_count_30d") or 0
    aed_count = data.get("aed_count") or 0
    high_burden_count = len(data.get("high_burden_aeds") or [])

    # --- Deterministic per-patient seed ---
    seed = int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)

    # =====================================================================
    # Trial 1 (T1, base 6.0 — attention span, less disease-sensitive)
    # =====================================================================
    t1_base = 6.0

    # Age effect
    t1_age = 0.0
    if age > 65:
        t1_age = -(0.04 * (65 - 50)) - (0.06 * (age - 65))
    elif age > 50:
        t1_age = -0.04 * (age - 50)

    # Disease effect (smaller for T1 — it measures attention, not memory)
    t1_disease_map = {
        "epilepsy": -0.5,
        "parkinson": -0.4,
        "alzheimer": -1.0,
        "depression": -0.3,
        "adhd": -0.8,
    }
    t1_disease = next(
        (v for k, v in t1_disease_map.items() if k in disease), -0.2
    )

    # AED effect
    if aed_count >= 3:
        t1_aed = -0.8
    elif aed_count == 2:
        t1_aed = -0.4
    elif aed_count == 1:
        t1_aed = -0.1
    else:
        t1_aed = 0.0
    t1_aed -= high_burden_count * 0.3

    # Seizure effect
    if sz_count > 10:
        t1_sz = -0.5
    elif sz_count > 5:
        t1_sz = -0.2
    elif sz_count > 0:
        t1_sz = -0.1
    else:
        t1_sz = 0.0

    # Functional status
    if barthel < 60:
        t1_func = -0.4
    elif barthel < 80:
        t1_func = -0.2
    else:
        t1_func = 0.0

    t1_noise = ((seed % 41) - 20) / 40.0  # ±0.5
    t1_raw = t1_base + t1_age + t1_disease + t1_aed + t1_sz + t1_func + t1_noise
    trial_1 = max(1, min(15, round(t1_raw)))

    # =====================================================================
    # Trial 5 (T5, base 13.0 — learning ceiling)
    # =====================================================================
    t5_base = 13.0

    t5_age = 0.0
    if age > 65:
        t5_age = -(0.06 * (65 - 50)) - (0.08 * (age - 65))
    elif age > 50:
        t5_age = -0.06 * (age - 50)

    t5_disease_map = {
        "epilepsy": -1.5,
        "parkinson": -1.0,
        "alzheimer": -3.0,
        "depression": -0.8,
        "adhd": -0.5,
    }
    t5_disease = next(
        (v for k, v in t5_disease_map.items() if k in disease), -0.3
    )

    if aed_count >= 3:
        t5_aed = -1.8
    elif aed_count == 2:
        t5_aed = -0.8
    elif aed_count == 1:
        t5_aed = -0.3
    else:
        t5_aed = 0.0
    t5_aed -= high_burden_count * 0.5

    if sz_count > 10:
        t5_sz = -1.0
    elif sz_count > 5:
        t5_sz = -0.5
    elif sz_count > 0:
        t5_sz = -0.2
    else:
        t5_sz = 0.0

    if barthel < 60:
        t5_func = -0.6
    elif barthel < 80:
        t5_func = -0.3
    else:
        t5_func = 0.0

    t5_noise = ((seed >> 4) % 41 - 20) / 40.0
    t5_raw = t5_base + t5_age + t5_disease + t5_aed + t5_sz + t5_func + t5_noise
    trial_5 = max(trial_1, min(15, round(t5_raw)))  # T5 >= T1 always

    # =====================================================================
    # Total Learning (TL = sum of T1–T5, base 52.0)
    # =====================================================================
    # Interpolate intermediate trials: roughly linear from T1 to T5
    # T2 ≈ T1 + 0.25*(T5-T1), T3 ≈ T1 + 0.50*(T5-T1), T4 ≈ T1 + 0.75*(T5-T1)
    diff = trial_5 - trial_1
    t2_est = trial_1 + round(0.25 * diff)
    t3_est = trial_1 + round(0.50 * diff)
    t4_est = trial_1 + round(0.75 * diff)
    total_learning = trial_1 + t2_est + t3_est + t4_est + trial_5

    # =====================================================================
    # List B (base 6.0 — single-trial interference list)
    # =====================================================================
    lb_base = 6.0

    lb_age = 0.0
    if age > 65:
        lb_age = -(0.04 * (65 - 50)) - (0.05 * (age - 65))
    elif age > 50:
        lb_age = -0.04 * (age - 50)

    lb_disease_map = {
        "epilepsy": -0.5,
        "parkinson": -0.3,
        "alzheimer": -1.2,
        "depression": -0.3,
        "adhd": -0.6,
    }
    lb_disease = next(
        (v for k, v in lb_disease_map.items() if k in disease), -0.2
    )

    lb_aed = -0.2 * min(aed_count, 3) - high_burden_count * 0.2
    lb_noise = ((seed >> 8) % 31 - 15) / 30.0
    lb_raw = lb_base + lb_age + lb_disease + lb_aed + lb_noise
    list_b = max(1, min(15, round(lb_raw)))

    # =====================================================================
    # Trial 6 — post-interference recall (base 11.0)
    # =====================================================================
    # T6 typically falls between T5 and DR; it captures retroactive
    # interference. In epilepsy, T6 drops more than in healthy controls.
    t6_base = 11.0

    t6_age = 0.0
    if age > 65:
        t6_age = -(0.06 * (65 - 50)) - (0.08 * (age - 65))
    elif age > 50:
        t6_age = -0.06 * (age - 50)

    t6_disease_map = {
        "epilepsy": -1.8,
        "parkinson": -1.2,
        "alzheimer": -3.5,
        "depression": -0.8,
        "adhd": -0.4,
    }
    t6_disease = next(
        (v for k, v in t6_disease_map.items() if k in disease), -0.3
    )

    if aed_count >= 3:
        t6_aed = -2.0
    elif aed_count == 2:
        t6_aed = -1.0
    elif aed_count == 1:
        t6_aed = -0.3
    else:
        t6_aed = 0.0
    t6_aed -= high_burden_count * 0.6

    if sz_count > 10:
        t6_sz = -1.2
    elif sz_count > 5:
        t6_sz = -0.6
    elif sz_count > 0:
        t6_sz = -0.2
    else:
        t6_sz = 0.0

    if barthel < 60:
        t6_func = -0.8
    elif barthel < 80:
        t6_func = -0.3
    else:
        t6_func = 0.0

    t6_noise = ((seed >> 12) % 41 - 20) / 40.0
    t6_raw = t6_base + t6_age + t6_disease + t6_aed + t6_sz + t6_func + t6_noise
    trial_6 = max(0, min(trial_5, round(t6_raw)))  # T6 <= T5

    # =====================================================================
    # Delayed Recall (DR, base 11.0) — THE PRIMARY SEVERITY METRIC
    # =====================================================================
    # DR is the most clinically important RAVLT metric. In TLE, DR drops
    # substantially below T5 and T6, reflecting hippocampal consolidation
    # failure. This is the hallmark dissociation: learning (T5) may be
    # adequate, but retention (DR) is severely impaired.
    dr_base = 11.0

    dr_age = 0.0
    if age > 65:
        dr_age = -(0.08 * (65 - 50)) - (0.10 * (age - 65))
    elif age > 50:
        dr_age = -0.08 * (age - 50)

    # Disease delta — epilepsy hits DR the hardest (hippocampal consolidation)
    dr_disease_map = {
        "epilepsy": -2.5,
        "parkinson": -1.5,
        "alzheimer": -4.0,
        "depression": -1.0,
        "adhd": -0.6,
    }
    dr_disease = next(
        (v for k, v in dr_disease_map.items() if k in disease), -0.5
    )

    # AED delta
    if aed_count >= 3:
        dr_aed = -2.5
    elif aed_count == 2:
        dr_aed = -1.2
    elif aed_count == 1:
        dr_aed = -0.4
    else:
        dr_aed = 0.0
    dr_aed -= high_burden_count * 0.8

    # Seizure delta
    if sz_count > 10:
        dr_sz = -1.5
    elif sz_count > 5:
        dr_sz = -0.8
    elif sz_count > 0:
        dr_sz = -0.3
    else:
        dr_sz = 0.0

    # Functional status
    if barthel < 60:
        dr_func = -1.0
    elif barthel < 80:
        dr_func = -0.4
    else:
        dr_func = 0.0

    dr_noise = ((seed >> 16) % 41 - 20) / 40.0
    dr_raw = (
        dr_base + dr_age + dr_disease + dr_aed + dr_sz + dr_func + dr_noise
    )
    delayed_recall = max(0, min(trial_6, round(dr_raw)))  # DR <= T6

    # =====================================================================
    # Recognition Hits (base 14.0 — relatively preserved even in TLE)
    # =====================================================================
    rh_base = 14.0

    rh_disease_map = {
        "epilepsy": -0.6,
        "parkinson": -0.4,
        "alzheimer": -3.0,
        "depression": -0.3,
        "adhd": -0.2,
    }
    rh_disease = next(
        (v for k, v in rh_disease_map.items() if k in disease), -0.1
    )
    rh_age_pen = -0.3 if age > 65 else -0.1 if age > 50 else 0.0
    rh_aed_pen = -0.2 * min(high_burden_count, 3)
    rh_noise = ((seed >> 20) % 21 - 10) / 20.0
    rh_raw = rh_base + rh_disease + rh_age_pen + rh_aed_pen + rh_noise
    recognition_hits = max(0, min(15, round(rh_raw)))

    # =====================================================================
    # Derived metrics
    # =====================================================================
    learning_slope = round((trial_5 - trial_1) / 4.0, 2)

    if trial_5 > 0:
        forgetting_rate = round((trial_5 - delayed_recall) / trial_5 * 100, 1)
    else:
        forgetting_rate = 0.0
    forgetting_rate = max(0.0, min(100.0, forgetting_rate))

    # =====================================================================
    # Z-score and percentile (DR-based; norms: mean 11, SD 3)
    # =====================================================================
    dr_mean = 11.0
    dr_sd = 3.0
    dr_z = round((delayed_recall - dr_mean) / dr_sd, 2)

    # Percentile from z-score (higher DR = better; z>0 = above average)
    z_pct_lookup = {
        -3: 1, -2: 2, -1: 16, 0: 50, 1: 84, 2: 98, 3: 99,
    }
    z_floor = max(-3, min(3, int(dr_z) if dr_z >= 0 else int(dr_z) - 1))
    if dr_z < 0 and dr_z == int(dr_z):
        z_floor = int(dr_z)
    z_floor = max(-3, min(2, z_floor))
    z_ceil = min(3, z_floor + 1)
    frac = dr_z - z_floor
    pct_floor = z_pct_lookup.get(z_floor, 50)
    pct_ceil = z_pct_lookup.get(z_ceil, 50)
    dr_percentile = max(1, min(99, round(pct_floor + frac * (pct_ceil - pct_floor))))

    # =====================================================================
    # Severity band (DR z-score based)
    # =====================================================================
    severity_info = SEVERITY_BANDS[-1].copy()  # default to Impaired
    if dr_z >= -1.0:
        severity_info = SEVERITY_BANDS[0].copy()
    elif dr_z >= -1.5:
        severity_info = SEVERITY_BANDS[1].copy()
    elif dr_z >= -2.0:
        severity_info = SEVERITY_BANDS[2].copy()

    return {
        "trial_1": trial_1,
        "trial_5": trial_5,
        "total_learning": total_learning,
        "list_b": list_b,
        "trial_6": trial_6,
        "delayed_recall": delayed_recall,
        "recognition_hits": recognition_hits,
        "learning_slope": learning_slope,
        "forgetting_rate": forgetting_rate,
        "dr_z_score": dr_z,
        "dr_percentile": dr_percentile,
        "severity": severity_info["label"],
        "severity_color": severity_info["color"],
        "severity_description": severity_info["description"],
    }


# ---------------------------------------------------------------------------
# Public API — dashboard
# ---------------------------------------------------------------------------

def ravlt_dashboard(patient_id: str = None) -> dict:
    """
    Return RAVLT results for a single patient or all patients.

    Parameters
    ----------
    patient_id : str, optional
        If provided, return results for that patient only.
        If None (default), aggregate across all patients in the DB.

    Returns (single patient)
    -------------------------
    dict with: patient_id, patient_name, age, disease, data_sources,
               trial_1, trial_5, total_learning, list_b, trial_6,
               delayed_recall, recognition_hits, learning_slope,
               forgetting_rate, dr_z_score, dr_percentile,
               severity, severity_color, severity_description

    Returns (all patients)
    ----------------------
    dict with: scale_name, total_patients, patients (list),
               severity_distribution, mean_delayed_recall,
               mean_total_learning, impairment_rate_pct, norm_reference
    """
    if patient_id:
        data = _get_patient_data(patient_id)
        est = _estimate_ravlt(data)
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
        est = _estimate_ravlt(d)
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
            "scale_name": "Rey Auditory Verbal Learning Test (RAVLT)",
            "total_patients": 0,
            "patients": [],
            "severity_distribution": {},
            "mean_delayed_recall": None,
            "mean_total_learning": None,
            "impairment_rate_pct": None,
            "norm_reference": "Lezak 2012; Mitrushina 2005",
        }

    # Aggregates
    severity_dist: dict[str, int] = {}
    for p in patients:
        sev = p.get("severity", "Unknown")
        severity_dist[sev] = severity_dist.get(sev, 0) + 1

    mean_dr = round(
        sum(p["delayed_recall"] for p in patients) / len(patients), 1
    )
    mean_tl = round(
        sum(p["total_learning"] for p in patients) / len(patients), 1
    )
    impaired_count = sum(
        1 for p in patients
        if p.get("severity") in ("Borderline", "Impaired")
    )
    impairment_rate = round(impaired_count / len(patients) * 100, 1)

    return {
        "scale_name": "Rey Auditory Verbal Learning Test (RAVLT)",
        "total_patients": len(patients),
        "patients": patients,
        "severity_distribution": severity_dist,
        "mean_delayed_recall": mean_dr,
        "mean_total_learning": mean_tl,
        "impairment_rate_pct": impairment_rate,
        "norm_reference": "Lezak 2012; Mitrushina 2005",
    }


# ---------------------------------------------------------------------------
# Public API — detail
# ---------------------------------------------------------------------------

def ravlt_detail(patient_id: str) -> dict:
    """
    Return full RAVLT detail for one patient, including contributing factors,
    clinical interpretation, and recommendations.

    Extends ravlt_dashboard() with:
        contributing_factors : dict — quantified per-domain contributions
        clinical_interpretation : dict — encoding, consolidation, retrieval profile
        clinical_recommendations : list[str]
    """
    base = ravlt_dashboard(patient_id)
    data = _get_patient_data(patient_id)
    demo = data.get("demographics", {})

    age = demo.get("age", 40)
    disease = (demo.get("disease_type", "") or "").lower()
    barthel = data.get("barthel", 100)
    sz_count = data.get("seizure_count_30d", 0)
    aed_count = data.get("aed_count", 0)
    high_burden = data.get("high_burden_aeds", [])
    severity = base.get("severity", "Normal")
    t1 = base.get("trial_1", 6)
    t5 = base.get("trial_5", 13)
    dr = base.get("delayed_recall", 11)
    t6 = base.get("trial_6", 11)
    rh = base.get("recognition_hits", 14)
    fr = base.get("forgetting_rate", 15.0)

    # Contributing factors
    contributing_factors = {
        "age_effect": (
            "Minimal" if age <= 50
            else "Moderate" if age <= 65
            else "Significant"
        ),
        "disease_effect": (
            "High — hippocampal verbal memory impact" if "epilepsy" in disease
            else "High" if any(d in disease for d in ["alzheimer", "parkinson"])
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
                "high impact on memory consolidation" if sz_count > 10
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

    # Clinical interpretation: encoding vs consolidation vs retrieval profile
    encoding_intact = t1 >= 5
    learning_intact = t5 >= 10
    consolidation_intact = fr < 25.0
    retrieval_intact = rh >= 12

    clinical_interpretation = {
        "encoding": (
            "Intact — adequate initial registration"
            if encoding_intact
            else "Impaired — reduced attention span or initial registration"
        ),
        "learning_curve": (
            "Adequate learning slope across trials"
            if learning_intact
            else "Flat learning curve — limited benefit from repetition"
        ),
        "consolidation": (
            "Intact — material retained over delay"
            if consolidation_intact
            else "Impaired — significant forgetting over delay, "
            "consistent with hippocampal dysfunction"
        ),
        "retrieval": (
            "Intact — recognition at ceiling supports retrieval difficulty only"
            if retrieval_intact
            else "Impaired — low recognition suggests encoding/storage deficit, "
            "not merely retrieval failure"
        ),
        "pattern_summary": _classify_memory_pattern(
            encoding_intact, learning_intact, consolidation_intact, retrieval_intact, disease
        ),
    }

    # Clinical recommendations
    recs: list[str] = []

    if severity == "Impaired":
        recs.append(
            "RAVLT delayed recall indicates significant verbal memory impairment; "
            "comprehensive neuropsychological assessment and neuroimaging "
            "(MRI volumetry of hippocampal structures) recommended."
        )
    elif severity == "Borderline":
        recs.append(
            "Borderline verbal memory performance; serial RAVLT monitoring "
            "every 6 months and evaluation for memory rehabilitation strategies."
        )

    if high_burden:
        burden_list = ", ".join(high_burden)
        recs.append(
            f"High-burden AEDs identified ({burden_list}); review pharmacotherapy — "
            "topiramate and phenobarbital substantially impair verbal learning; "
            "switching to levetiracetam or lamotrigine may improve RAVLT scores."
        )

    if aed_count >= 3:
        recs.append(
            "Polypharmacy (≥3 AEDs) significantly burdens verbal memory consolidation; "
            "rationalisation of AED regimen to ≤2 agents should be evaluated."
        )

    if sz_count > 5:
        recs.append(
            f"{sz_count} seizures in the past 30 days contribute to post-ictal "
            "and inter-ictal memory disruption; optimise seizure control as the "
            "primary intervention for memory improvement."
        )

    if not consolidation_intact and "epilepsy" in disease:
        recs.append(
            "Consolidation failure pattern (T5 adequate, DR impaired) is the hallmark "
            "of temporal-lobe epilepsy with hippocampal dysfunction. Consider lateralised "
            "Wada testing or fMRI for language lateralisation if surgical evaluation is planned."
        )

    if not retrieval_intact:
        recs.append(
            "Low recognition hits suggest a true storage deficit rather than retrieval "
            "failure alone; consider differential diagnosis including neurodegenerative "
            "processes, particularly if progressive."
        )

    if fr > 40.0:
        recs.append(
            f"Forgetting rate of {fr}% over 20-minute delay is clinically significant; "
            "memory compensation strategies (external aids, spaced retrieval, "
            "errorless learning) should be considered."
        )

    if not recs:
        recs.append(
            "RAVLT performance within or near normal limits; "
            "continue routine annual verbal memory monitoring."
        )

    base["contributing_factors"] = contributing_factors
    base["clinical_interpretation"] = clinical_interpretation
    base["clinical_recommendations"] = recs
    return base


def _classify_memory_pattern(
    encoding: bool, learning: bool, consolidation: bool,
    retrieval: bool, disease: str,
) -> str:
    """Classify the overall memory pattern for clinical interpretation."""
    if encoding and learning and consolidation and retrieval:
        return "Normal verbal memory profile — all domains intact."

    if encoding and learning and not consolidation and retrieval:
        return (
            "Consolidation-deficit pattern: adequate encoding and learning, but "
            "accelerated forgetting over delay with preserved recognition. "
            "Characteristic of hippocampal/medial temporal lobe dysfunction "
            "(e.g., temporal-lobe epilepsy, early Alzheimer's disease)."
        )

    if encoding and learning and not consolidation and not retrieval:
        return (
            "Storage-deficit pattern: material is learned but neither retained "
            "nor recognised after delay. Suggests medial temporal lobe pathology "
            "affecting both consolidation and storage mechanisms."
        )

    if not encoding and not learning and not consolidation:
        return (
            "Global encoding-deficit pattern: poor initial registration and flat "
            "learning curve with subsequent failure of retention. Consider "
            "attentional, motivational, or widespread cortical factors."
        )

    if encoding and not learning:
        return (
            "Learning-plateau pattern: adequate initial span but failure to "
            "benefit from repetition. Suggests executive/strategic learning "
            "difficulties (frontal-subcortical dysfunction)."
        )

    return (
        "Mixed verbal memory pattern; detailed neuropsychological interpretation "
        "recommended to disentangle encoding, consolidation, and retrieval factors."
    )


# ---------------------------------------------------------------------------
# Public API — trend
# ---------------------------------------------------------------------------

def ravlt_trend(patient_id: str, months: int = 12) -> dict:
    """
    Project RAVLT trajectory over *months* for one patient.

    Trajectory logic
    ----------------
    Delayed Recall improves with:
        - AED optimisation  : starts month 2 if high_burden_count > 0
          DR +0.15/month, T5 +0.10/month
        - Seizure control   : starts month 1 if sz_count > 5
          DR +0.10/month, T5 +0.06/month
        - Age-related decline: DR −0.04/month, T5 −0.02/month if age > 50
          (gradual background worsening regardless of intervention)

    Returns
    -------
    dict with: patient_id, patient_name, baseline_delayed_recall,
               baseline_total_learning, baseline_severity,
               trajectory (list of month+1 points), assumptions
    """
    data = _get_patient_data(patient_id)
    demo = data.get("demographics", {})
    est = _estimate_ravlt(data)

    age = demo.get("age", 40)
    sz_count = data.get("seizure_count_30d", 0)
    high_burden_count = len(data.get("high_burden_aeds", []))

    baseline_dr = est["delayed_recall"]
    baseline_tl = est["total_learning"]
    baseline_t5 = est["trial_5"]
    baseline_severity = est["severity"]

    trajectory = []
    current_dr = float(baseline_dr)
    current_t5 = float(baseline_t5)

    for month in range(months + 1):  # 0 to months
        # AED optimisation effect (month 2+)
        if month >= 2 and high_burden_count > 0:
            current_dr += 0.15
            current_t5 += 0.10

        # Seizure control effect (month 1+)
        if month >= 1 and sz_count > 5:
            current_dr += 0.10
            current_t5 += 0.06

        # Background age-related decline
        if month > 0 and age > 50:
            current_dr -= 0.04
            current_t5 -= 0.02

        # Clamp to valid ranges
        dr_point = max(0, min(15, round(current_dr, 1)))
        t5_point = max(0, min(15, round(current_t5, 1)))

        # Severity at this time point (DR z-score)
        dr_z_now = round((dr_point - 11.0) / 3.0, 2)
        if dr_z_now >= -1.0:
            sev_label = "Normal"
        elif dr_z_now >= -1.5:
            sev_label = "Low-normal"
        elif dr_z_now >= -2.0:
            sev_label = "Borderline"
        else:
            sev_label = "Impaired"

        trajectory.append({
            "month": month,
            "projected_delayed_recall": dr_point,
            "projected_trial_5": t5_point,
            "dr_z_score": dr_z_now,
            "severity": sev_label,
            "label": f"Month {month}" if month > 0 else "Baseline",
        })

    # Build assumptions list
    assumptions = [
        f"Baseline: DR {baseline_dr}, T5 {baseline_t5}, TL {baseline_tl} ({baseline_severity})",
        "Lezak 2012 norms: mean DR 11 (SD 3), mean T5 13 (SD 2), mean TL 52 (SD 8)",
        "AED optimisation benefit applies from month 2 when high-burden AEDs present",
        "Seizure control benefit applies from month 1 when >5 seizures/30 days",
        f"Age-related decline (age {age}): " + (
            "−0.04 DR/month, −0.02 T5/month applied (age >50)" if age > 50
            else "not applied (age ≤50)"
        ),
        "Projections are model estimates only; clinical outcomes may differ",
    ]
    if high_burden_count > 0:
        assumptions.append(
            f"{high_burden_count} high-burden AED(s) identified; "
            "DR +0.15/month, T5 +0.10/month improvement modelled from month 2"
        )
    if sz_count > 5:
        assumptions.append(
            f"Frequent seizures ({sz_count}/30d); DR +0.10/month, T5 +0.06/month "
            "improvement modelled from month 1 with seizure control"
        )

    return {
        "patient_id": patient_id,
        "patient_name": demo.get("name", "Unknown"),
        "baseline_delayed_recall": baseline_dr,
        "baseline_total_learning": baseline_tl,
        "baseline_severity": baseline_severity,
        "trajectory": trajectory,
        "assumptions": assumptions,
    }


# ---------------------------------------------------------------------------
# Public API — scale definitions
# ---------------------------------------------------------------------------

def scale_definitions() -> dict:
    """
    Return metadata and reference information for the RAVLT.
    """
    return {
        "scale_name": "Rey Auditory Verbal Learning Test",
        "abbreviation": "RAVLT",
        "author": "Rey A (original 1941; Schmidt M 1996 standardisation)",
        "reference": (
            "Lezak MD, Howieson DB, Bigler ED, Tranel D. "
            "Neuropsychological Assessment. 5th ed. Oxford University Press; 2012. "
            "Mitrushina M, Boone KB, Razani J, D'Elia LF. "
            "Handbook of Normative Data for Neuropsychological Assessment. 2nd ed. "
            "Oxford University Press; 2005."
        ),
        "purpose": (
            "Assess verbal learning, short-term and long-term verbal memory, "
            "retroactive and proactive interference, and recognition memory. "
            "The RAVLT is the most widely used word-list learning test in "
            "clinical neuropsychology and epilepsy pre-surgical evaluation."
        ),
        "administration": {
            "stimuli": "List A: 15 unrelated words; List B: 15-word interference list",
            "trials": (
                "5 learning trials (List A read aloud, free recall each time), "
                "1 List B trial, immediate recall of List A (Trial 6), "
                "20-min delayed recall, recognition (50-word list: 15 A + 15 B + 20 distractors)"
            ),
            "duration_minutes": "30–35 (including delay period)",
            "version": "Manual (standard); multiple standardised word-list versions available",
        },
        "metrics": RAVLT_METRICS,
        "primary_metric": (
            "Delayed Recall (DR) — key indicator of hippocampal verbal memory "
            "consolidation, most sensitive to temporal-lobe epilepsy effects"
        ),
        "severity_bands": SEVERITY_BANDS,
        "normative_data": {
            "source": "Lezak 2012; Mitrushina 2005 — age-stratified norms (adults 20–89)",
            "healthy_adult_t1_mean": 6.0,
            "healthy_adult_t1_sd": 2.0,
            "healthy_adult_t5_mean": 13.0,
            "healthy_adult_t5_sd": 2.0,
            "healthy_adult_tl_mean": 52.0,
            "healthy_adult_tl_sd": 8.0,
            "healthy_adult_dr_mean": 11.0,
            "healthy_adult_dr_sd": 3.0,
            "healthy_adult_rh_mean": 14.0,
            "healthy_adult_rh_sd": 1.5,
            "impairment_threshold_dr": "DR z-score < −2.0",
            "note": (
                "Norms are stratified by age; scores decline significantly "
                "after age 50 with accelerated decline after 65. "
                "Education corrections also apply in comprehensive norms."
            ),
        },
        "psychometrics": {
            "test_retest_reliability": (
                "r = 0.60–0.77 (Total Learning most stable); "
                "practice effects expected on re-administration within 6 months"
            ),
            "validity": (
                "Highly sensitive to hippocampal/mesial temporal lobe dysfunction; "
                "validated in epilepsy, Alzheimer's, TBI, depression, and ageing studies"
            ),
            "construct": (
                "Verbal learning (T1–T5 curve), consolidation (DR vs T5), "
                "interference (T6 vs T5, List B vs T1), recognition memory (hits)"
            ),
            "sensitivity_temporal": (
                "High — DR is the most sensitive RAVLT metric for detecting "
                "left temporal-lobe epilepsy and hippocampal sclerosis"
            ),
            "ecological_validity": (
                "DR and Total Learning correlate with everyday memory complaints, "
                "medication management, and appointment adherence"
            ),
        },
        "epilepsy_context": {
            "applications": [
                "Pre-surgical neuropsychological evaluation (left vs right TLE lateralisation)",
                "AED cognitive monitoring — detecting verbal memory impairment",
                "Baseline and follow-up in clinical trials assessing cognitive AED effects",
                "Hippocampal sclerosis detection support (consolidation-failure pattern)",
                "Post-surgical memory outcome prediction (Wada concordance)",
            ],
            "aed_effects": {
                "high_burden_aeds": list(HIGH_COGNITIVE_BURDEN_AEDS),
                "effect_summary": (
                    "Topiramate and phenobarbital most strongly impair verbal learning "
                    "and delayed recall, reflecting disrupted hippocampal consolidation. "
                    "Lamotrigine and levetiracetam show minimal verbal memory burden. "
                    "Zonisamide and phenytoin produce intermediate effects on encoding "
                    "and consolidation."
                ),
                "temporal_lobe_note": (
                    "In temporal-lobe epilepsy, the RAVLT consolidation-failure pattern "
                    "(adequate T5 but impaired DR) reflects hippocampal dysfunction and "
                    "is the hallmark dissociation used in pre-surgical lateralisation. "
                    "Left TLE patients show significantly lower DR than right TLE patients. "
                    "Helmstaedter & Kurthen (2001) demonstrated that RAVLT DR is the "
                    "single strongest predictor of left-TLE verbal memory lateralisation."
                ),
            },
        },
        "data_derivation": (
            "Scores are clinically modelled from real patient data (demographics, "
            "AED regimen, seizure frequency, Barthel Index) using Lezak 2012 and "
            "Mitrushina 2005 normative baselines with deterministic hash-seeded "
            "per-patient noise. Epilepsy patients are specifically modelled to show "
            "the consolidation-failure pattern (DR impaired > T5). "
            "Intended for research, educational, and prototype purposes only. "
            "Replace with administered RAVLT data for clinical decision-making."
        ),
    }


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    d = ravlt_dashboard()
    print(f"RAVLT Dashboard: {d['total_patients']} patients")
    print(f"Mean delayed recall: {d['mean_delayed_recall']}")
    print(f"Mean total learning: {d['mean_total_learning']}")
    print(f"Impairment rate: {d['impairment_rate_pct']}%")
    print(f"Distribution: {d['severity_distribution']}")
