"""
Verbal Fluency (FAS + Category + Switching) — NeuroAI Clinical Dashboard Module
================================================================================
Measures phonemic (letter) fluency, semantic (category) fluency, and switching
fluency. Verbal fluency tests are among the most widely used neuropsychological
measures, sensitive to frontal and temporal lobe dysfunction.

Citation
--------
Tombaugh TN, Kozak J, Rees L. Normative data stratified by age and education
for two measures of verbal fluency: FAS and animal naming.
Archives of Clinical Neuropsychology. 1999;14(2):167-177.

Benton AL, Hamsher K, Sivan AB.
Multilingual Aphasia Examination. 3rd ed. AJA Associates; 1994.

Strauss E, Sherman EMS, Spreen O.
A Compendium of Neuropsychological Tests. 3rd ed.
Oxford University Press; 2006.

Primary Metrics
---------------
- FAS Total (F+A+S)         phonemic fluency (mean ~42, SD ~12)
- F score                   letter F words in 60s (mean ~14, SD ~4)
- A score                   letter A words in 60s (mean ~14, SD ~4)
- S score                   letter S words in 60s (mean ~14, SD ~4)
- Animals (category)        semantic fluency (mean ~20, SD ~5)
- Switching Total            alternating-category fluency (mean ~15, SD ~4)
- Switching Errors           incorrect category switches (mean ~2, SD ~1.5)
- Clustering Coefficient     related-word groups / total (mean ~0.35, SD ~0.15)
- Phonemic-Semantic Ratio    FAS / Animals — frontal vs temporal dissociation
                            (mean ~2.1, SD ~0.6)

Severity (FAS Total z-score based, Tombaugh 1999 norms)
-------------------------------------------------------
Normal       z >= -1.0        (green)
Low-normal   -1.5 <= z < -1.0 (olive)
Borderline   -2.0 <= z < -1.5 (orange)
Impaired     z < -2.0         (red)

Clinical context — epilepsy
----------------------------
Frontal-lobe epilepsy (FLE) specifically impairs phonemic fluency (FAS) more
than semantic fluency (Animals), reflecting executive/frontal dysfunction.
Temporal-lobe epilepsy (TLE) impairs semantic fluency (Animals) more than
phonemic (FAS), reflecting temporal/semantic store disruption. The FAS/Animals
dissociation ratio helps lateralise and localise seizure focus. AEDs such as
topiramate, phenobarbital, and phenytoin impair word-finding and verbal output.

Helmstaedter C, Kurthen M. Memory and temporal lobe epilepsy: a critical
review. Epilepsy & Behavior. 2001;2(3):126-150.
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

VF_METRICS = [
    {
        "id": 1,
        "metric": "FAS Total",
        "abbreviation": "FAS",
        "description": "Sum of words generated for letters F, A, and S (60 seconds each).",
        "measures": "Phonemic (letter) fluency; frontal executive function and lexical retrieval speed.",
        "norms": {"mean": 42.0, "sd": 12.0, "direction": "higher_better"},
    },
    {
        "id": 2,
        "metric": "F Score",
        "abbreviation": "F",
        "description": "Words beginning with F generated in 60 seconds.",
        "measures": "Single-letter phonemic fluency; lexical access speed.",
        "norms": {"mean": 14.0, "sd": 4.0, "direction": "higher_better"},
    },
    {
        "id": 3,
        "metric": "A Score",
        "abbreviation": "A",
        "description": "Words beginning with A generated in 60 seconds.",
        "measures": "Single-letter phonemic fluency; lexical access speed.",
        "norms": {"mean": 14.0, "sd": 4.0, "direction": "higher_better"},
    },
    {
        "id": 4,
        "metric": "S Score",
        "abbreviation": "S",
        "description": "Words beginning with S generated in 60 seconds.",
        "measures": "Single-letter phonemic fluency; lexical access speed.",
        "norms": {"mean": 14.0, "sd": 4.0, "direction": "higher_better"},
    },
    {
        "id": 5,
        "metric": "Animals (Category Fluency)",
        "abbreviation": "ANI",
        "description": "Animal names generated in 60 seconds.",
        "measures": "Semantic (category) fluency; temporal lobe / semantic store integrity.",
        "norms": {"mean": 20.0, "sd": 5.0, "direction": "higher_better"},
    },
    {
        "id": 6,
        "metric": "Switching Total",
        "abbreviation": "SW",
        "description": "Words generated while alternating between two categories (e.g., fruits/furniture) in 60 seconds.",
        "measures": "Executive set-shifting and cognitive flexibility.",
        "norms": {"mean": 15.0, "sd": 4.0, "direction": "higher_better"},
    },
    {
        "id": 7,
        "metric": "Switching Errors",
        "abbreviation": "SWE",
        "description": "Incorrect category switches or perseverative responses during alternating fluency.",
        "measures": "Error monitoring and inhibition during set-shifting.",
        "norms": {"mean": 2.0, "sd": 1.5, "direction": "lower_better"},
    },
    {
        "id": 8,
        "metric": "Clustering Coefficient",
        "abbreviation": "CC",
        "description": "Proportion of semantically or phonemically related word-groups relative to total output.",
        "measures": "Strategic word-generation approach; subcortical-frontal clustering efficiency.",
        "norms": {"mean": 0.35, "sd": 0.15, "direction": "higher_better"},
    },
    {
        "id": 9,
        "metric": "Phonemic-Semantic Ratio",
        "abbreviation": "PSR",
        "description": "FAS Total / Animals — ratio of phonemic to semantic fluency.",
        "measures": "Frontal vs temporal dissociation; ratio >2.5 suggests relative temporal impairment, <1.5 suggests relative frontal impairment.",
        "norms": {"mean": 2.1, "sd": 0.6, "direction": "context_dependent"},
    },
]

# Severity bands keyed on FAS Total z-score — Tombaugh 1999
SEVERITY_BANDS = [
    {
        "range": [-1.0, 999.0],
        "label": "Normal",
        "color": "green",
        "description": (
            "Phonemic fluency within expected range (z >= -1.0); "
            "frontal executive function and lexical retrieval intact."
        ),
    },
    {
        "range": [-1.5, -1.01],
        "label": "Low-normal",
        "color": "olive",
        "description": (
            "Mildly reduced phonemic fluency (-1.5 <= z < -1.0); "
            "subtle word-finding concern, borderline frontal findings."
        ),
    },
    {
        "range": [-2.0, -1.51],
        "label": "Borderline",
        "color": "orange",
        "description": (
            "Moderately impaired phonemic fluency (-2.0 <= z < -1.5); "
            "consistent with frontal executive dysfunction, monitor closely."
        ),
    },
    {
        "range": [-999.0, -2.01],
        "label": "Impaired",
        "color": "red",
        "description": (
            "Severely impaired phonemic fluency (z < -2.0); "
            "significant frontal/executive deficit, comprehensive assessment indicated."
        ),
    },
]

# All recognised AEDs (same 25-agent set as WCST / Stroop / RAVLT modules)
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

# AEDs with documented high verbal fluency / word-finding cognitive burden
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
    Fetch real patient data from clinical.db for verbal fluency estimation.

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

def _estimate_verbal_fluency(data: dict) -> dict:
    """
    Estimate Verbal Fluency metrics for a single patient using real clinical data.

    Methodology
    -----------
    Baseline norms (Tombaugh 1999, healthy adults 20-49) are adjusted by
    deterministic clinical modifiers. A hash-seeded per-patient noise term
    ensures reproducibility across calls.

    Modifier logic
    --------------
    FAS Total (base 42.0, higher = better):
        age_delta   : -0.3/year after 50; additional -0.4/year after 65
        disease_delta: frontal epilepsy -8.0 (phonemic > semantic);
                       temporal epilepsy -4.0; parkinson -5.0;
                       alzheimer -7.0; depression -3.0; adhd -2.0; default -1.5
        aed_delta   : count >=3 -> -6.0; 2 -> -3.0; 1 -> -1.0;
                      high-burden count * -2.0
        sz_delta    : sz >10 -> -4.0; >5 -> -2.0; >0 -> -0.8
        func_delta  : barthel <60 -> -3.0; <80 -> -1.2

    Animals (base 20.0) follows similar pattern but temporal epilepsy
    hits harder (-6.0) while frontal epilepsy hits less (-2.5), producing
    the classic frontal-temporal dissociation.
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

    # Detect frontal vs temporal localisation from disease string
    is_frontal = any(k in disease for k in ["frontal", "fle", "front"])
    is_temporal = any(k in disease for k in ["temporal", "tle", "hippocamp"])
    is_epilepsy = "epilepsy" in disease or is_frontal or is_temporal

    # =====================================================================
    # FAS Total (base 42.0 — phonemic fluency, frontal-executive)
    # =====================================================================
    fas_base = 42.0

    # Age effect
    fas_age = 0.0
    if age > 65:
        fas_age = -(0.3 * (65 - 50)) - (0.4 * (age - 65))
    elif age > 50:
        fas_age = -0.3 * (age - 50)

    # Disease effect — frontal epilepsy hits phonemic hardest
    if is_frontal:
        fas_disease = -8.0
    elif is_temporal:
        fas_disease = -4.0
    elif is_epilepsy:
        fas_disease = -5.5
    else:
        fas_disease_map = {
            "parkinson": -5.0,
            "alzheimer": -7.0,
            "depression": -3.0,
            "adhd": -2.0,
        }
        fas_disease = next(
            (v for k, v in fas_disease_map.items() if k in disease), -1.5
        )

    # AED effect
    if aed_count >= 3:
        fas_aed = -6.0
    elif aed_count == 2:
        fas_aed = -3.0
    elif aed_count == 1:
        fas_aed = -1.0
    else:
        fas_aed = 0.0
    fas_aed -= high_burden_count * 2.0

    # Seizure effect
    if sz_count > 10:
        fas_sz = -4.0
    elif sz_count > 5:
        fas_sz = -2.0
    elif sz_count > 0:
        fas_sz = -0.8
    else:
        fas_sz = 0.0

    # Functional status
    if barthel < 60:
        fas_func = -3.0
    elif barthel < 80:
        fas_func = -1.2
    else:
        fas_func = 0.0

    fas_noise = ((seed % 41) - 20) / 40.0 * 4.0  # +/-2.0
    fas_raw = fas_base + fas_age + fas_disease + fas_aed + fas_sz + fas_func + fas_noise
    fas_total = max(3, round(fas_raw))

    # Per-letter scores (split FAS total roughly equally with small variation)
    letter_base = fas_total / 3.0
    f_noise = ((seed >> 3) % 7 - 3) / 3.0  # +/-1.0
    a_noise = ((seed >> 6) % 7 - 3) / 3.0
    s_noise = ((seed >> 9) % 7 - 3) / 3.0

    f_score = max(1, round(letter_base + f_noise))
    a_score = max(1, round(letter_base + a_noise))
    s_score = max(1, fas_total - f_score - a_score)  # Ensure they sum to fas_total

    # =====================================================================
    # Animals — Category Fluency (base 20.0 — semantic/temporal)
    # =====================================================================
    ani_base = 20.0

    # Age effect
    ani_age = 0.0
    if age > 65:
        ani_age = -(0.2 * (65 - 50)) - (0.3 * (age - 65))
    elif age > 50:
        ani_age = -0.2 * (age - 50)

    # Disease effect — temporal epilepsy hits semantic hardest (dissociation)
    if is_temporal:
        ani_disease = -6.0
    elif is_frontal:
        ani_disease = -2.5
    elif is_epilepsy:
        ani_disease = -4.0
    else:
        ani_disease_map = {
            "alzheimer": -8.0,
            "parkinson": -4.0,
            "depression": -2.0,
            "adhd": -1.5,
        }
        ani_disease = next(
            (v for k, v in ani_disease_map.items() if k in disease), -1.0
        )

    # AED effect (less severe for semantic than phonemic)
    if aed_count >= 3:
        ani_aed = -4.0
    elif aed_count == 2:
        ani_aed = -2.0
    elif aed_count == 1:
        ani_aed = -0.6
    else:
        ani_aed = 0.0
    ani_aed -= high_burden_count * 1.2

    # Seizure effect
    if sz_count > 10:
        ani_sz = -3.0
    elif sz_count > 5:
        ani_sz = -1.5
    elif sz_count > 0:
        ani_sz = -0.5
    else:
        ani_sz = 0.0

    # Functional status
    if barthel < 60:
        ani_func = -2.0
    elif barthel < 80:
        ani_func = -0.8
    else:
        ani_func = 0.0

    ani_noise = ((seed >> 4) % 41 - 20) / 40.0 * 2.5  # +/-1.25
    ani_raw = ani_base + ani_age + ani_disease + ani_aed + ani_sz + ani_func + ani_noise
    animals = max(2, round(ani_raw))

    # =====================================================================
    # Switching Total (base 15.0 — executive set-shifting)
    # =====================================================================
    sw_base = 15.0

    # Age effect
    sw_age = 0.0
    if age > 65:
        sw_age = -(0.2 * (65 - 50)) - (0.3 * (age - 65))
    elif age > 50:
        sw_age = -0.2 * (age - 50)

    # Disease effect — frontal hits switching hardest
    if is_frontal:
        sw_disease = -5.0
    elif is_temporal:
        sw_disease = -2.5
    elif is_epilepsy:
        sw_disease = -3.5
    else:
        sw_disease_map = {
            "parkinson": -4.0,
            "alzheimer": -5.5,
            "depression": -2.0,
            "adhd": -3.0,
        }
        sw_disease = next(
            (v for k, v in sw_disease_map.items() if k in disease), -1.0
        )

    # AED effect
    if aed_count >= 3:
        sw_aed = -3.5
    elif aed_count == 2:
        sw_aed = -1.8
    elif aed_count == 1:
        sw_aed = -0.5
    else:
        sw_aed = 0.0
    sw_aed -= high_burden_count * 1.0

    # Seizure effect
    if sz_count > 10:
        sw_sz = -2.5
    elif sz_count > 5:
        sw_sz = -1.2
    elif sz_count > 0:
        sw_sz = -0.4
    else:
        sw_sz = 0.0

    if barthel < 60:
        sw_func = -1.5
    elif barthel < 80:
        sw_func = -0.6
    else:
        sw_func = 0.0

    sw_noise = ((seed >> 8) % 31 - 15) / 30.0 * 2.0  # +/-1.0
    sw_raw = sw_base + sw_age + sw_disease + sw_aed + sw_sz + sw_func + sw_noise
    switching_total = max(1, round(sw_raw))

    # =====================================================================
    # Switching Errors (base 2.0 — lower is better)
    # =====================================================================
    swe_base = 2.0

    # More errors with frontal/executive dysfunction
    if is_frontal:
        swe_disease = 2.5
    elif is_temporal:
        swe_disease = 1.0
    elif is_epilepsy:
        swe_disease = 1.5
    else:
        swe_disease_map = {
            "parkinson": 2.0,
            "alzheimer": 3.0,
            "adhd": 2.5,
            "depression": 0.8,
        }
        swe_disease = next(
            (v for k, v in swe_disease_map.items() if k in disease), 0.3
        )

    swe_age = 0.3 if age > 65 else 0.1 if age > 50 else 0.0
    swe_aed = 0.3 * min(high_burden_count, 3)
    swe_noise = ((seed >> 12) % 21 - 10) / 20.0
    swe_raw = swe_base + swe_disease + swe_age + swe_aed + swe_noise
    switching_errors = max(0, round(swe_raw))

    # =====================================================================
    # Clustering Coefficient (base 0.35 — proportion)
    # =====================================================================
    cc_base = 0.35

    # Better clustering with intact frontal-subcortical circuits
    if is_frontal:
        cc_disease = -0.12
    elif is_temporal:
        cc_disease = -0.06
    elif is_epilepsy:
        cc_disease = -0.08
    else:
        cc_disease_map = {
            "alzheimer": -0.15,
            "parkinson": -0.10,
            "adhd": -0.08,
            "depression": -0.04,
        }
        cc_disease = next(
            (v for k, v in cc_disease_map.items() if k in disease), -0.02
        )

    cc_age = -0.03 if age > 65 else -0.01 if age > 50 else 0.0
    cc_aed = -0.02 * min(high_burden_count, 3)
    cc_noise = ((seed >> 16) % 21 - 10) / 200.0  # +/-0.05
    cc_raw = cc_base + cc_disease + cc_age + cc_aed + cc_noise
    clustering_coeff = max(0.05, min(0.80, round(cc_raw, 2)))

    # =====================================================================
    # Phonemic-Semantic Ratio (FAS / Animals)
    # =====================================================================
    if animals > 0:
        ps_ratio = round(fas_total / animals, 2)
    else:
        ps_ratio = 0.0

    # =====================================================================
    # Z-score and percentile (FAS Total based; norms: mean 42, SD 12)
    # =====================================================================
    fas_mean = 42.0
    fas_sd = 12.0
    fas_z = round((fas_total - fas_mean) / fas_sd, 2)

    # Percentile from z-score (higher FAS = better; z>0 = above average)
    z_pct_lookup = {
        -3: 1, -2: 2, -1: 16, 0: 50, 1: 84, 2: 98, 3: 99,
    }
    z_floor = max(-3, min(2, int(fas_z) if fas_z >= 0 else int(fas_z) - 1))
    if fas_z < 0 and fas_z == int(fas_z):
        z_floor = int(fas_z)
    z_floor = max(-3, min(2, z_floor))
    z_ceil = min(3, z_floor + 1)
    frac = fas_z - z_floor
    pct_floor = z_pct_lookup.get(z_floor, 50)
    pct_ceil = z_pct_lookup.get(z_ceil, 50)
    fas_percentile = max(1, min(99, round(pct_floor + frac * (pct_ceil - pct_floor))))

    # =====================================================================
    # Severity band (FAS Total z-score based)
    # =====================================================================
    severity_info = SEVERITY_BANDS[-1].copy()  # default to Impaired
    if fas_z >= -1.0:
        severity_info = SEVERITY_BANDS[0].copy()
    elif fas_z >= -1.5:
        severity_info = SEVERITY_BANDS[1].copy()
    elif fas_z >= -2.0:
        severity_info = SEVERITY_BANDS[2].copy()

    return {
        "fas_total": fas_total,
        "f_score": f_score,
        "a_score": a_score,
        "s_score": s_score,
        "animals": animals,
        "switching_total": switching_total,
        "switching_errors": switching_errors,
        "clustering_coefficient": clustering_coeff,
        "phonemic_semantic_ratio": ps_ratio,
        "fas_z_score": fas_z,
        "fas_percentile": fas_percentile,
        "severity": severity_info["label"],
        "severity_color": severity_info["color"],
        "severity_description": severity_info["description"],
    }


# ---------------------------------------------------------------------------
# Public API — dashboard
# ---------------------------------------------------------------------------

def verbal_fluency_dashboard(patient_id: str = None) -> dict:
    """
    Return Verbal Fluency results for a single patient or all patients.

    Parameters
    ----------
    patient_id : str, optional
        If provided, return results for that patient only.
        If None (default), aggregate across all patients in the DB.

    Returns (single patient)
    -------------------------
    dict with: patient_id, patient_name, age, disease, data_sources,
               fas_total, f_score, a_score, s_score, animals,
               switching_total, switching_errors, clustering_coefficient,
               phonemic_semantic_ratio, fas_z_score, fas_percentile,
               severity, severity_color, severity_description

    Returns (all patients)
    ----------------------
    dict with: scale_name, total_patients, patients (list),
               severity_distribution, mean_fas_total,
               mean_animals, impairment_rate_pct, norm_reference
    """
    if patient_id:
        data = _get_patient_data(patient_id)
        est = _estimate_verbal_fluency(data)
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
        est = _estimate_verbal_fluency(d)
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
            "scale_name": "Verbal Fluency (FAS + Category + Switching)",
            "total_patients": 0,
            "patients": [],
            "severity_distribution": {},
            "mean_fas_total": None,
            "mean_animals": None,
            "impairment_rate_pct": None,
            "norm_reference": "Tombaugh 1999; Strauss 2006",
        }

    # Aggregates
    severity_dist: dict[str, int] = {}
    for p in patients:
        sev = p.get("severity", "Unknown")
        severity_dist[sev] = severity_dist.get(sev, 0) + 1

    mean_fas = round(
        sum(p["fas_total"] for p in patients) / len(patients), 1
    )
    mean_ani = round(
        sum(p["animals"] for p in patients) / len(patients), 1
    )
    impaired_count = sum(
        1 for p in patients
        if p.get("severity") in ("Borderline", "Impaired")
    )
    impairment_rate = round(impaired_count / len(patients) * 100, 1)

    return {
        "scale_name": "Verbal Fluency (FAS + Category + Switching)",
        "total_patients": len(patients),
        "patients": patients,
        "severity_distribution": severity_dist,
        "mean_fas_total": mean_fas,
        "mean_animals": mean_ani,
        "impairment_rate_pct": impairment_rate,
        "norm_reference": "Tombaugh 1999; Strauss 2006",
    }


# ---------------------------------------------------------------------------
# Public API — detail
# ---------------------------------------------------------------------------

def verbal_fluency_detail(patient_id: str) -> dict:
    """
    Return full Verbal Fluency detail for one patient, including contributing
    factors, clinical interpretation, and recommendations.

    Extends verbal_fluency_dashboard() with:
        contributing_factors : dict — quantified per-domain contributions
        clinical_interpretation : dict — phonemic, semantic, switching profile
        clinical_recommendations : list[str]
    """
    base = verbal_fluency_dashboard(patient_id)
    data = _get_patient_data(patient_id)
    demo = data.get("demographics", {})

    age = demo.get("age") or 40
    disease = (demo.get("disease_type", "") or "").lower()
    barthel = data.get("barthel") or 100
    sz_count = data.get("seizure_count_30d") or 0
    aed_count = data.get("aed_count") or 0
    high_burden = data.get("high_burden_aeds") or []
    severity = base.get("severity", "Normal")
    fas_total = base.get("fas_total", 42)
    animals = base.get("animals", 20)
    switching_total = base.get("switching_total", 15)
    switching_errors = base.get("switching_errors", 2)
    ps_ratio = base.get("phonemic_semantic_ratio", 2.1)
    clustering = base.get("clustering_coefficient", 0.35)

    is_frontal = any(k in disease for k in ["frontal", "fle", "front"])
    is_temporal = any(k in disease for k in ["temporal", "tle", "hippocamp"])

    # Contributing factors
    contributing_factors = {
        "age_effect": (
            "Minimal" if age <= 50
            else "Moderate" if age <= 65
            else "Significant"
        ),
        "disease_effect": (
            "High — frontal executive dysfunction impairs phonemic fluency"
            if is_frontal
            else "High — temporal/semantic store disruption impairs category fluency"
            if is_temporal
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
                "high impact on verbal output and fluency" if sz_count > 10
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

    # Clinical interpretation: phonemic vs semantic vs switching profile
    phonemic_intact = fas_total >= 30  # ~1 SD below mean
    semantic_intact = animals >= 15    # ~1 SD below mean
    switching_intact = switching_total >= 11  # ~1 SD below mean
    errors_elevated = switching_errors >= 4

    clinical_interpretation = {
        "phonemic_fluency": (
            "Intact — adequate letter-based word generation"
            if phonemic_intact
            else "Impaired — reduced phonemic output, consistent with "
            "frontal/executive dysfunction affecting lexical retrieval"
        ),
        "semantic_fluency": (
            "Intact — adequate category-based word generation"
            if semantic_intact
            else "Impaired — reduced category fluency, consistent with "
            "temporal lobe / semantic store disruption"
        ),
        "switching_fluency": (
            "Intact — adequate cognitive flexibility in alternating categories"
            if switching_intact
            else "Impaired — reduced set-shifting ability, consistent with "
            "executive dysfunction"
        ),
        "error_monitoring": (
            "Elevated switching errors — impaired inhibition and error monitoring"
            if errors_elevated
            else "Switching errors within normal limits"
        ),
        "dissociation_analysis": _classify_fluency_pattern(
            phonemic_intact, semantic_intact, switching_intact,
            ps_ratio, disease
        ),
        "clustering_strategy": (
            "Adequate clustering — strategic word-generation approach preserved"
            if clustering >= 0.25
            else "Reduced clustering — impoverished strategic search, "
            "suggesting subcortical-frontal dysfunction"
        ),
    }

    # Clinical recommendations
    recs: list[str] = []

    if severity == "Impaired":
        recs.append(
            "Verbal fluency indicates significant impairment (FAS z < -2.0); "
            "comprehensive neuropsychological assessment including Boston Naming "
            "Test, Trail Making, and Wisconsin Card Sorting recommended to "
            "characterise the full executive/language profile."
        )
    elif severity == "Borderline":
        recs.append(
            "Borderline verbal fluency performance; serial monitoring every "
            "6 months with repeat FAS + Animals to track trajectory."
        )

    if high_burden:
        burden_list = ", ".join(high_burden)
        recs.append(
            f"High-burden AEDs identified ({burden_list}); review pharmacotherapy — "
            "topiramate is the most strongly associated with word-finding difficulty; "
            "switching to levetiracetam or lamotrigine may improve verbal fluency."
        )

    if aed_count >= 3:
        recs.append(
            "Polypharmacy (>=3 AEDs) significantly impairs verbal output speed; "
            "rationalisation of AED regimen to <=2 agents should be evaluated."
        )

    if sz_count > 5:
        recs.append(
            f"{sz_count} seizures in the past 30 days contribute to post-ictal "
            "word-finding difficulty; optimise seizure control as the primary "
            "intervention for fluency improvement."
        )

    if not phonemic_intact and not semantic_intact:
        recs.append(
            "Both phonemic and semantic fluency impaired — global verbal output "
            "reduction. Evaluate for generalised cognitive decline, depression-related "
            "psychomotor slowing, or diffuse cerebral dysfunction."
        )
    elif not phonemic_intact and semantic_intact:
        recs.append(
            "Phonemic impaired but semantic preserved — frontal dissociation pattern. "
            "Consistent with frontal-lobe epilepsy or dorsolateral prefrontal "
            "dysfunction. Consider frontal-lobe-specific neuroimaging (fMRI/PET)."
        )
    elif phonemic_intact and not semantic_intact:
        recs.append(
            "Semantic impaired but phonemic preserved — temporal dissociation pattern. "
            "Consistent with temporal-lobe epilepsy or semantic memory degradation. "
            "Consider temporal-lobe MRI volumetry and semantic memory battery."
        )

    if errors_elevated:
        recs.append(
            f"Elevated switching errors ({switching_errors}) indicate impaired "
            "inhibition and error monitoring. Consider additional executive function "
            "testing (Stroop, Go/No-Go) to characterise inhibitory control."
        )

    if clustering < 0.20:
        recs.append(
            "Very low clustering coefficient suggests impoverished strategic search. "
            "Compensatory strategies (categorical prompting, semantic cuing) may "
            "improve functional word-finding in daily communication."
        )

    if not recs:
        recs.append(
            "Verbal fluency performance within or near normal limits; "
            "continue routine annual fluency monitoring."
        )

    base["contributing_factors"] = contributing_factors
    base["clinical_interpretation"] = clinical_interpretation
    base["clinical_recommendations"] = recs
    return base


def _classify_fluency_pattern(
    phonemic: bool, semantic: bool, switching: bool,
    ps_ratio: float, disease: str,
) -> str:
    """Classify the overall fluency pattern for clinical interpretation."""
    if phonemic and semantic and switching:
        return "Normal verbal fluency profile — all domains intact."

    if not phonemic and semantic and switching:
        return (
            "Phonemic-deficit pattern: impaired letter fluency with preserved "
            "category fluency and switching. Characteristic of dorsolateral "
            "prefrontal dysfunction (e.g., frontal-lobe epilepsy). "
            "FAS/Animals ratio likely <1.5."
        )

    if phonemic and not semantic and switching:
        return (
            "Semantic-deficit pattern: impaired category fluency with preserved "
            "letter fluency and switching. Characteristic of temporal lobe "
            "dysfunction or semantic memory degradation (e.g., temporal-lobe "
            "epilepsy, semantic dementia). FAS/Animals ratio likely >2.5."
        )

    if not phonemic and not semantic and not switching:
        return (
            "Global fluency deficit: all fluency domains impaired. Consider "
            "generalised cognitive decline, severe AED burden, depression-related "
            "psychomotor slowing, or diffuse cerebral pathology."
        )

    if not phonemic and not semantic and switching:
        return (
            "Combined phonemic and semantic deficit with preserved switching. "
            "Suggests dual frontal-temporal involvement or global word-retrieval "
            "difficulty with intact set-shifting."
        )

    if not switching and (phonemic or semantic):
        return (
            "Switching-deficit pattern: impaired alternating fluency with "
            "relatively preserved single-category output. Suggests specific "
            "executive set-shifting difficulty (frontal-subcortical dysfunction)."
        )

    return (
        "Mixed verbal fluency pattern; detailed neuropsychological interpretation "
        "recommended to disentangle phonemic, semantic, and executive factors."
    )


# ---------------------------------------------------------------------------
# Public API — trend
# ---------------------------------------------------------------------------

def verbal_fluency_trend(patient_id: str, months: int = 12) -> dict:
    """
    Project verbal fluency trajectory over *months* for one patient.

    Trajectory logic
    ----------------
    FAS Total improves with:
        - AED optimisation  : starts month 2 if high_burden_count > 0
          FAS +0.5/month, Animals +0.3/month
        - Seizure control   : starts month 1 if sz_count > 5
          FAS +0.3/month, Animals +0.2/month
        - Age-related decline: FAS -0.1/month, Animals -0.08/month if age > 50
          (gradual background worsening regardless of intervention)

    Returns
    -------
    dict with: patient_id, patient_name, baseline_fas_total,
               baseline_animals, baseline_severity,
               trajectory (list of month+1 points), assumptions
    """
    data = _get_patient_data(patient_id)
    demo = data.get("demographics", {})
    est = _estimate_verbal_fluency(data)

    age = demo.get("age") or 40
    sz_count = data.get("seizure_count_30d") or 0
    high_burden_count = len(data.get("high_burden_aeds") or [])

    baseline_fas = est["fas_total"]
    baseline_ani = est["animals"]
    baseline_severity = est["severity"]

    trajectory = []
    current_fas = float(baseline_fas)
    current_ani = float(baseline_ani)

    for month in range(months + 1):  # 0 to months
        # AED optimisation effect (month 2+)
        if month >= 2 and high_burden_count > 0:
            current_fas += 0.5
            current_ani += 0.3

        # Seizure control effect (month 1+)
        if month >= 1 and sz_count > 5:
            current_fas += 0.3
            current_ani += 0.2

        # Background age-related decline
        if month > 0 and age > 50:
            current_fas -= 0.1
            current_ani -= 0.08

        # Clamp to valid ranges
        fas_point = max(0, round(current_fas, 1))
        ani_point = max(0, round(current_ani, 1))

        # Severity at this time point (FAS z-score)
        fas_z_now = round((fas_point - 42.0) / 12.0, 2)
        if fas_z_now >= -1.0:
            sev_label = "Normal"
        elif fas_z_now >= -1.5:
            sev_label = "Low-normal"
        elif fas_z_now >= -2.0:
            sev_label = "Borderline"
        else:
            sev_label = "Impaired"

        trajectory.append({
            "month": month,
            "projected_fas_total": fas_point,
            "projected_animals": ani_point,
            "fas_z_score": fas_z_now,
            "severity": sev_label,
            "label": f"Month {month}" if month > 0 else "Baseline",
        })

    # Build assumptions list
    assumptions = [
        f"Baseline: FAS {baseline_fas}, Animals {baseline_ani} ({baseline_severity})",
        "Tombaugh 1999 norms: mean FAS 42 (SD 12), mean Animals 20 (SD 5)",
        "AED optimisation benefit applies from month 2 when high-burden AEDs present",
        "Seizure control benefit applies from month 1 when >5 seizures/30 days",
        f"Age-related decline (age {age}): " + (
            "-0.1 FAS/month, -0.08 Animals/month applied (age >50)" if age > 50
            else "not applied (age <=50)"
        ),
        "Projections are model estimates only; clinical outcomes may differ",
    ]
    if high_burden_count > 0:
        assumptions.append(
            f"{high_burden_count} high-burden AED(s) identified; "
            "FAS +0.5/month, Animals +0.3/month improvement modelled from month 2"
        )
    if sz_count > 5:
        assumptions.append(
            f"Frequent seizures ({sz_count}/30d); FAS +0.3/month, Animals +0.2/month "
            "improvement modelled from month 1 with seizure control"
        )

    return {
        "patient_id": patient_id,
        "patient_name": demo.get("name", "Unknown"),
        "baseline_fas_total": baseline_fas,
        "baseline_animals": baseline_ani,
        "baseline_severity": baseline_severity,
        "trajectory": trajectory,
        "assumptions": assumptions,
    }


# ---------------------------------------------------------------------------
# Public API — scale definitions
# ---------------------------------------------------------------------------

def scale_definitions() -> dict:
    """
    Return metadata and reference information for Verbal Fluency tests.
    """
    return {
        "scale_name": "Verbal Fluency (FAS + Category + Switching)",
        "abbreviation": "VF / FAS / CFL",
        "author": "Benton AL, Hamsher K (original Controlled Oral Word Association Test, 1976)",
        "reference": (
            "Tombaugh TN, Kozak J, Rees L. Normative data stratified by age "
            "and education for two measures of verbal fluency: FAS and animal naming. "
            "Archives of Clinical Neuropsychology. 1999;14(2):167-177. "
            "Benton AL, Hamsher K, Sivan AB. Multilingual Aphasia Examination. "
            "3rd ed. AJA Associates; 1994. "
            "Strauss E, Sherman EMS, Spreen O. A Compendium of Neuropsychological Tests. "
            "3rd ed. Oxford University Press; 2006."
        ),
        "purpose": (
            "Assess phonemic (letter) fluency, semantic (category) fluency, and "
            "switching fluency. Verbal fluency tests are among the most widely used "
            "neuropsychological measures, sensitive to frontal executive dysfunction "
            "(phonemic), temporal lobe / semantic store integrity (category), and "
            "cognitive flexibility (switching). The FAS/Animals dissociation ratio "
            "aids localisation and lateralisation in epilepsy."
        ),
        "administration": {
            "stimuli": (
                "Phonemic: letters F, A, S (or C, F, L in alternate form). "
                "Semantic: animal category (or supermarket items). "
                "Switching: alternating between two categories (e.g., fruits and furniture)."
            ),
            "trials": (
                "Phonemic: 3 trials (one per letter), 60 seconds each. "
                "Semantic: 1 trial, 60 seconds (animal naming). "
                "Switching: 1 trial, 60 seconds (alternating categories)."
            ),
            "duration_minutes": "10-15 (all three components)",
            "version": "Manual (standard); multiple letter-set versions available (FAS, CFL, PRW)",
        },
        "metrics": VF_METRICS,
        "primary_metric": (
            "FAS Total — key indicator of phonemic fluency and frontal executive "
            "function; most sensitive to frontal-lobe epilepsy effects"
        ),
        "severity_bands": SEVERITY_BANDS,
        "normative_data": {
            "source": "Tombaugh 1999 — age- and education-stratified norms (adults 16-95)",
            "healthy_adult_fas_mean": 42.0,
            "healthy_adult_fas_sd": 12.0,
            "healthy_adult_f_mean": 14.0,
            "healthy_adult_f_sd": 4.0,
            "healthy_adult_a_mean": 14.0,
            "healthy_adult_a_sd": 4.0,
            "healthy_adult_s_mean": 14.0,
            "healthy_adult_s_sd": 4.0,
            "healthy_adult_animals_mean": 20.0,
            "healthy_adult_animals_sd": 5.0,
            "healthy_adult_switching_mean": 15.0,
            "healthy_adult_switching_sd": 4.0,
            "impairment_threshold_fas": "FAS Total z-score < -2.0",
            "note": (
                "Norms are stratified by age and education; scores decline "
                "after age 50 with accelerated decline after 65. Education "
                "corrections significantly affect phonemic (FAS) norms; "
                "semantic (Animals) is less education-dependent."
            ),
        },
        "psychometrics": {
            "test_retest_reliability": (
                "r = 0.70-0.83 (FAS most stable); alternate-form reliability "
                "r = 0.85-0.92 for FAS/CFL equivalence"
            ),
            "validity": (
                "Highly sensitive to frontal-lobe dysfunction (phonemic) and "
                "temporal-lobe/semantic dysfunction (category). Validated in "
                "epilepsy, Alzheimer's, Parkinson's, TBI, depression, and ageing."
            ),
            "construct": (
                "Phonemic fluency (FAS): frontal executive, lexical retrieval. "
                "Semantic fluency (Animals): temporal lobe, semantic store. "
                "Switching: executive set-shifting, cognitive flexibility. "
                "FAS/Animals ratio: frontal vs temporal dissociation index."
            ),
            "sensitivity_frontal": (
                "High — FAS is the most sensitive verbal fluency metric for "
                "detecting frontal-lobe epilepsy and dorsolateral prefrontal "
                "dysfunction"
            ),
            "sensitivity_temporal": (
                "High — Animals is the most sensitive verbal fluency metric for "
                "detecting temporal-lobe epilepsy and semantic memory degradation"
            ),
            "ecological_validity": (
                "Verbal fluency correlates with everyday word-finding difficulty, "
                "conversational fluency, and functional communication in daily life"
            ),
        },
        "epilepsy_context": {
            "applications": [
                "Pre-surgical neuropsychological evaluation (frontal vs temporal localisation)",
                "AED cognitive monitoring — detecting word-finding impairment",
                "Baseline and follow-up in clinical trials assessing cognitive AED effects",
                "Frontal-temporal dissociation via FAS/Animals ratio",
                "Post-surgical verbal fluency outcome prediction",
            ],
            "aed_effects": {
                "high_burden_aeds": list(HIGH_COGNITIVE_BURDEN_AEDS),
                "effect_summary": (
                    "Topiramate most strongly impairs word-finding and verbal fluency, "
                    "affecting both phonemic and semantic output. Phenobarbital and "
                    "phenytoin produce moderate verbal output slowing. Lamotrigine and "
                    "levetiracetam show minimal verbal fluency burden. The word-finding "
                    "difficulty with topiramate is its most commonly reported cognitive "
                    "side effect and is dose-dependent."
                ),
                "frontal_temporal_note": (
                    "In epilepsy, the FAS/Animals dissociation ratio is clinically "
                    "informative: frontal-lobe epilepsy (FLE) patients show impaired "
                    "phonemic (FAS) with relatively preserved semantic (Animals) fluency "
                    "(ratio <1.5), while temporal-lobe epilepsy (TLE) patients show "
                    "impaired semantic with relatively preserved phonemic fluency "
                    "(ratio >2.5). This dissociation pattern assists in localising and "
                    "lateralising the seizure focus and predicting post-surgical "
                    "verbal outcome."
                ),
            },
        },
        "data_derivation": (
            "Scores are clinically modelled from real patient data (demographics, "
            "AED regimen, seizure frequency, Barthel Index) using Tombaugh 1999 "
            "normative baselines with deterministic hash-seeded per-patient noise. "
            "Epilepsy patients are specifically modelled to show the frontal-temporal "
            "dissociation pattern (FLE: impaired FAS > Animals; TLE: impaired Animals > FAS). "
            "Intended for research, educational, and prototype purposes only. "
            "Replace with administered verbal fluency data for clinical decision-making."
        ),
    }


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    d = verbal_fluency_dashboard()
    print(f"Verbal Fluency Dashboard: {d['total_patients']} patients")
    print(f"Mean FAS Total: {d['mean_fas_total']}")
    print(f"Mean Animals: {d['mean_animals']}")
    print(f"Impairment rate: {d['impairment_rate_pct']}%")
    print(f"Distribution: {d['severity_distribution']}")
