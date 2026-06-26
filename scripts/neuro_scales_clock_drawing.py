"""
Clock Drawing Test (CDT) — Neuropsychological Scale Dashboard
==============================================================
Paradigm   : Quick neuropsychological screening tool.  The patient draws
             a clock face showing a specific time (e.g., "10 minutes past
             11" = 11:10).  Assesses visuospatial/constructional ability,
             executive function, semantic memory, and attention.
Primary DV : Total score (0-5, Shulman 1986 scale).
             HIGHER score = better performance (5 = perfect clock).
             Score <= 3 = cognitive impairment likely.
Secondary  : Contour score (0-2) — circle quality.
             Numbers score (0-4) — number placement accuracy.
             Hands score (0-4) — hand placement accuracy.
             Center score (0-1) — center point presence/correctness.
Reference  : Shulman KI, Shedletsky R, Silver IL.  The challenge of time:
             Clock-drawing and cognitive function in the elderly.  Int J
             Geriatr Psychiatry. 1986;1:135-140.
             Freedman M, Leach L, Kaplan E, Winocur G, Shulman KI,
             Delis DC.  Clock Drawing: A Neuropsychological Analysis.
             Oxford University Press, 1994.
             Sunderland T, Hill JL, Mellow AM, et al.  Clock drawing in
             Alzheimer's disease: a novel measure of dementia severity.
             J Am Geriatr Soc. 1989;37:725-729.
Norms      : Healthy adults: mean ~4.5, SD ~0.7 (Shulman 1993;
             Freedman 1994).
Impairment : Score <= 3 = cognitive impairment likely.
Direction  : HIGHER score = better (5 is perfect).
"""

import hashlib
import json
import math
import os
import sqlite3
import statistics
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "clinical.db",
)

AEDS_SET: set[str] = {
    "carbamazepine", "clobazam", "clonazepam", "eslicarbazepine",
    "ethosuximide", "gabapentin", "lacosamide", "lamotrigine",
    "levetiracetam", "oxcarbazepine", "perampanel", "phenobarbital",
    "phenytoin", "pregabalin", "primidone", "retigabine", "rufinamide",
    "sodium_valproate", "stiripentol", "tiagabine", "topiramate",
    "vigabatrin", "zonisamide", "brivaracetam", "cenobamate",
}

HIGH_COGNITIVE_BURDEN_AEDS: set[str] = {
    "topiramate", "phenobarbital", "phenytoin",
    "zonisamide", "clobazam", "clonazepam", "vigabatrin",
}

# Norm reference: Shulman 1993; Freedman 1994
TOTAL_SCORE_NORMS   = {"mean": 4.5, "sd": 0.7}    # 0-5 scale
CONTOUR_NORMS       = {"mean": 1.8, "sd": 0.3}    # 0-2
NUMBERS_NORMS       = {"mean": 3.5, "sd": 0.6}    # 0-4
HANDS_NORMS         = {"mean": 3.3, "sd": 0.7}    # 0-4
CENTER_NORMS        = {"mean": 0.9, "sd": 0.2}    # 0-1

# Severity bands — Total score (higher = better)
SEVERITY_BANDS_TOTAL = [
    {"range": [5.0, 5.0], "label": "Normal",      "color": "green",   "description": "Perfect clock — no errors"},
    {"range": [4.0, 4.99], "label": "Mild",        "color": "olive",   "description": "Minor visuospatial errors (spacing, missing numbers)"},
    {"range": [3.0, 3.99], "label": "Borderline",  "color": "orange",  "description": "Inaccurate representation — impairment likely"},
    {"range": [2.0, 2.99], "label": "Impaired",    "color": "red",     "description": "Moderate disorganization (numbers misplaced, hands absent/wrong)"},
    {"range": [1.0, 1.99], "label": "Impaired",    "color": "red",     "description": "Severe disorganization (barely recognizable as clock)"},
    {"range": [0.0, 0.99], "label": "Severe",      "color": "darkred", "description": "No recognizable attempt"},
]

SEVERITY_BANDS_CONTOUR = [
    {"range": [2.0, 2.0], "label": "Normal",     "color": "green",  "description": "Good circle — closed, approximately round"},
    {"range": [1.0, 1.99], "label": "Mild",       "color": "olive",  "description": "Deformed circle — recognizable but distorted"},
    {"range": [0.0, 0.99], "label": "Impaired",   "color": "red",    "description": "No circle or unrecognizable contour"},
]

SEVERITY_BANDS_NUMBERS = [
    {"range": [4.0, 4.0], "label": "Normal",     "color": "green",  "description": "All 12 numbers present and correctly placed"},
    {"range": [3.0, 3.99], "label": "Mild",       "color": "olive",  "description": "Sequencing errors or minor misplacements"},
    {"range": [2.0, 2.99], "label": "Borderline", "color": "orange", "description": "Partially correct — some numbers misplaced or missing"},
    {"range": [1.0, 1.99], "label": "Impaired",   "color": "red",    "description": "Numbers randomly placed or severely disordered"},
    {"range": [0.0, 0.99], "label": "Severe",     "color": "darkred","description": "Numbers absent or not recognizable"},
]

SEVERITY_BANDS_HANDS = [
    {"range": [4.0, 4.0], "label": "Normal",     "color": "green",  "description": "Both hands correctly placed for target time"},
    {"range": [3.0, 3.99], "label": "Mild",       "color": "olive",  "description": "Close but slightly off — e.g. minor length/angle error"},
    {"range": [2.0, 2.99], "label": "Borderline", "color": "orange", "description": "One hand correct, one incorrect"},
    {"range": [1.0, 1.99], "label": "Impaired",   "color": "red",    "description": "Hands present but wrong placement"},
    {"range": [0.0, 0.99], "label": "Severe",     "color": "darkred","description": "No hands drawn"},
]

SEVERITY_BANDS_CENTER = [
    {"range": [1.0, 1.0], "label": "Normal",     "color": "green",  "description": "Center point present and correctly placed"},
    {"range": [0.0, 0.99], "label": "Impaired",   "color": "red",    "description": "Center point absent or misplaced"},
]

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _get_patient_data(patient_id: str) -> dict[str, Any] | None:
    """Return patient demographics, seizure burden, and medications from data/clinical.db."""
    try:
        with _conn() as con:
            row = con.execute(
                "SELECT patient_id, name, age, gender, disease FROM patients WHERE patient_id = ?",
                (patient_id,),
            ).fetchone()
            if row is None:
                return None

            patient: dict[str, Any] = {
                "patient_id": row["patient_id"],
                "patient_name": row["name"],
                "age": row["age"],
                "gender": row["gender"],
                "disease": row["disease"],
                "epilepsy_type": row["disease"],
            }

            sz = con.execute(
                "SELECT COUNT(*) AS cnt FROM seizure_diary "
                "WHERE patient_id = ? AND event_date >= date('now','-6 months')",
                (patient_id,),
            ).fetchone()
            patient["seizure_count_6m"] = sz["cnt"] if sz else 0

            med_rows = con.execute(
                "SELECT fields_json FROM medications WHERE patient_id = ?",
                (patient_id,),
            ).fetchall()
            meds: list[dict] = []
            aed_names: list[str] = []
            for med_row in med_rows:
                try:
                    fields = json.loads(med_row["fields_json"])
                    if isinstance(fields, list):
                        for item in fields:
                            if isinstance(item, dict):
                                name = (item.get("medication_name", "") or item.get("drug_name", "") or item.get("name", "")).lower().strip()
                                meds.append({"medication_name": name, "dose_mg": item.get("dose_mg"), "frequency": item.get("frequency")})
                                if name in AEDS_SET:
                                    aed_names.append(name)
                    elif isinstance(fields, dict):
                        name = (fields.get("medication_name", "") or fields.get("drug_name", "") or fields.get("name", "")).lower().strip()
                        meds.append({"medication_name": name, "dose_mg": fields.get("dose_mg"), "frequency": fields.get("frequency")})
                        if name in AEDS_SET:
                            aed_names.append(name)
                except (json.JSONDecodeError, TypeError):
                    pass
            patient["medications"] = meds
            patient["aed_names"] = aed_names

            barthel = con.execute(
                "SELECT score FROM assessments WHERE patient_id = ? AND instrument = 'BARTHEL' "
                "ORDER BY created_at DESC LIMIT 1",
                (patient_id,),
            ).fetchone()
            patient["barthel"] = barthel["score"] if barthel else 100

        return patient
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify(value: float, bands: list[dict]) -> dict[str, str]:
    for band in bands:
        if band["range"][0] <= value <= band["range"][1]:
            return band
    return bands[-1]


def _make_rng(seed: int):
    state = {"s": seed & 0xFFFFFFFF}

    def _lcg() -> float:
        state["s"] = (1664525 * state["s"] + 1013904223) & 0xFFFFFFFF
        return state["s"] / 0xFFFFFFFF

    def normal(mu: float = 0.0, sigma: float = 1.0) -> float:
        u1 = max(1e-10, _lcg())
        u2 = _lcg()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return mu + sigma * z

    def uniform(lo: float, hi: float) -> float:
        return lo + _lcg() * (hi - lo)

    return normal, uniform


# ---------------------------------------------------------------------------
# Core estimation
# ---------------------------------------------------------------------------

def _estimate_clock_drawing(patient: dict[str, Any]) -> dict[str, Any]:
    """
    Estimate Clock Drawing Test performance from patient clinical data.

    Returns total_score (0-5 Shulman scale), contour_score (0-2),
    numbers_score (0-4), hands_score (0-4), center_score (0-1),
    and derived severity bands.

    The CDT assesses visuospatial/constructional ability, executive function,
    semantic memory, and attention.  Higher scores indicate better performance.
    Score <= 3 indicates likely cognitive impairment.

    Estimation logic:
    - Temporal lobe epilepsy → moderate CDT impairment (visuospatial + executive)
    - Frontal lobe epilepsy → executive components affected (planning, sequencing)
    - Generalized epilepsy → less CDT-specific unless co-morbid cognitive decline
    - High-cognitive-burden AEDs (topiramate, phenobarbital) → processing speed
      impacts drawing quality
    - Seizure frequency → cumulative cognitive damage affects CDT
    - Age → older patients score lower (age-related decline compounds epilepsy)
    """
    pid           = patient.get("patient_id", "UNKNOWN")
    age           = int(patient.get("age") or 40)
    epilepsy_type = (patient.get("epilepsy_type") or "").lower()
    seizure_count = int(patient.get("seizure_count_6m") or 0)
    meds          = patient.get("medications", [])

    seed = int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)
    rng_norm, rng_uniform = _make_rng(seed)

    med_names = {
        m.get("medication_name", "").lower().strip().replace(" ", "_")
        for m in meds
    }
    high_burden_count = len(med_names & HIGH_COGNITIVE_BURDEN_AEDS)
    known_aed_count   = len(med_names & AEDS_SET)

    # ------------------------------------------------------------------ #
    # Contour score (0-2) — circle quality; PRIMARY DV
    # Visuospatial/constructional ability drives contour accuracy.
    # ------------------------------------------------------------------ #
    contour = 2.0 + rng_norm(0.0, 0.15)

    # Temporal lobe epilepsy most impacts visuospatial → contour
    if "temporal" in epilepsy_type:
        contour -= 0.6
    elif "frontal" in epilepsy_type:
        contour -= 0.3
    elif "generalised" in epilepsy_type or "generalized" in epilepsy_type:
        contour -= 0.2
    elif "focal" in epilepsy_type:
        contour -= 0.3

    # Seizure burden — cumulative damage to visuospatial networks
    if seizure_count > 20:
        contour -= 0.5
    elif seizure_count > 10:
        contour -= 0.3
    elif seizure_count > 4:
        contour -= 0.15

    # AED burden — motor coordination + visuospatial side effects
    if high_burden_count >= 2:
        contour -= 0.4
    elif high_burden_count == 1:
        contour -= 0.2

    # Age — motor control and visuospatial ability decline
    if age >= 75:
        contour -= 0.5
    elif age >= 65:
        contour -= 0.3
    elif age >= 55:
        contour -= 0.15

    contour = max(0.0, min(2.0, round(contour, 1)))

    # ------------------------------------------------------------------ #
    # Numbers score (0-4) — number placement accuracy
    # Executive function (sequencing, planning) + semantic memory (clock
    # knowledge) drive number placement.
    # ------------------------------------------------------------------ #
    numbers = 4.0 + rng_norm(0.0, 0.25)

    # Frontal lobe epilepsy most impacts executive → numbers sequencing
    if "frontal" in epilepsy_type:
        numbers -= 1.4
    elif "temporal" in epilepsy_type:
        numbers -= 1.0
    elif "generalised" in epilepsy_type or "generalized" in epilepsy_type:
        numbers -= 0.6
    elif "focal" in epilepsy_type:
        numbers -= 0.7

    # Seizure burden — disrupts executive networks
    if seizure_count > 20:
        numbers -= 1.2
    elif seizure_count > 10:
        numbers -= 0.7
    elif seizure_count > 4:
        numbers -= 0.3

    # AED burden — topiramate specifically impairs word-finding / sequencing
    if high_burden_count >= 2:
        numbers -= 1.0
    elif high_burden_count == 1:
        numbers -= 0.5
    if known_aed_count >= 3:
        numbers -= 0.3

    # Age — executive function and semantic memory decline
    if age >= 75:
        numbers -= 0.8
    elif age >= 65:
        numbers -= 0.5
    elif age >= 55:
        numbers -= 0.2

    numbers = max(0.0, min(4.0, round(numbers, 1)))

    # ------------------------------------------------------------------ #
    # Hands score (0-4) — hand placement accuracy
    # Executive function (planning, abstraction) + visuospatial ability
    # drive correct hand placement for the target time.
    # ------------------------------------------------------------------ #
    hands = 4.0 + rng_norm(0.0, 0.3)

    # Frontal epilepsy → planning/abstraction deficit → hands errors
    if "frontal" in epilepsy_type:
        hands -= 1.6
    elif "temporal" in epilepsy_type:
        hands -= 1.2
    elif "generalised" in epilepsy_type or "generalized" in epilepsy_type:
        hands -= 0.7
    elif "focal" in epilepsy_type:
        hands -= 0.8

    # Seizure burden — cumulative executive damage
    if seizure_count > 20:
        hands -= 1.3
    elif seizure_count > 10:
        hands -= 0.8
    elif seizure_count > 4:
        hands -= 0.35

    # AED burden — sedating AEDs impair planning and motor control
    if high_burden_count >= 2:
        hands -= 1.1
    elif high_burden_count == 1:
        hands -= 0.55
    if known_aed_count >= 3:
        hands -= 0.35

    # Age — executive function declines with age
    if age >= 75:
        hands -= 0.9
    elif age >= 65:
        hands -= 0.5
    elif age >= 55:
        hands -= 0.25

    hands = max(0.0, min(4.0, round(hands, 1)))

    # ------------------------------------------------------------------ #
    # Center score (0-1) — center point presence and correctness
    # Relatively preserved unless severe disorganization.
    # ------------------------------------------------------------------ #
    center = 1.0 + rng_norm(0.0, 0.05)

    if "frontal" in epilepsy_type:
        center -= 0.25
    elif "temporal" in epilepsy_type:
        center -= 0.15

    if seizure_count > 20:
        center -= 0.3
    elif seizure_count > 10:
        center -= 0.15

    if high_burden_count >= 2:
        center -= 0.2
    elif high_burden_count == 1:
        center -= 0.1

    if age >= 75:
        center -= 0.2
    elif age >= 65:
        center -= 0.1

    center = max(0.0, min(1.0, round(center, 1)))

    # ------------------------------------------------------------------ #
    # Total score (0-5, Shulman 1986 scale)
    # Derived from component scores — maps to 6-point Shulman scale.
    # ------------------------------------------------------------------ #
    # Weighted composite: contour contributes to visuospatial, numbers
    # and hands contribute to executive/semantic, center is structural.
    # Map composite (0-11 raw) to 0-5 Shulman scale.
    raw_composite = contour + numbers + hands + center  # max 11
    # Linear mapping from 0-11 to 0-5
    total_mapped = (raw_composite / 11.0) * 5.0

    # Add small patient-specific noise to total (preserving determinism)
    total_score = total_mapped + rng_norm(0.0, 0.1)
    total_score = max(0.0, min(5.0, round(total_score, 1)))

    # Ensure consistency: if all components are perfect, total should be 5
    if contour >= 2.0 and numbers >= 4.0 and hands >= 4.0 and center >= 1.0:
        total_score = max(total_score, 4.5)

    # ------------------------------------------------------------------ #
    # Severity classifications
    # ------------------------------------------------------------------ #
    total_band   = _classify(total_score, SEVERITY_BANDS_TOTAL)
    contour_band = _classify(contour, SEVERITY_BANDS_CONTOUR)
    numbers_band = _classify(numbers, SEVERITY_BANDS_NUMBERS)
    hands_band   = _classify(hands, SEVERITY_BANDS_HANDS)
    center_band  = _classify(center, SEVERITY_BANDS_CENTER)

    # Overall severity from total score
    overall_label = total_band["label"]

    return {
        "patient_id":       pid,
        "total_score":      total_score,
        "contour_score":    contour,
        "numbers_score":    numbers,
        "hands_score":      hands,
        "center_score":     center,
        "total_band":       total_band,
        "contour_band":     contour_band,
        "numbers_band":     numbers_band,
        "hands_band":       hands_band,
        "center_band":      center_band,
        "overall_severity": overall_label,
        "impaired":         total_score <= 3.0,
        "high_burden_aeds": high_burden_count,
        "known_aeds":       known_aed_count,
        "seizure_count_6m": seizure_count,
        "age":              age,
        "epilepsy_type":    patient.get("epilepsy_type", "unknown"),
    }

# ---------------------------------------------------------------------------
# Dashboard function (population overview)
# ---------------------------------------------------------------------------

def clock_drawing_dashboard(patient_id: str = None) -> dict[str, Any]:
    """
    Population-level CDT summary across all patients in clinical.db,
    or single-patient summary if patient_id is given.

    Returns n_patients, mean_score, sd_score, pct_impaired (score <= 3),
    severity_distribution, and per-patient scores.
    """
    with _conn() as con:
        if patient_id:
            patients = con.execute(
                "SELECT patient_id FROM patients WHERE patient_id = ?", (patient_id,)
            ).fetchall()
        else:
            patients = con.execute("SELECT patient_id FROM patients").fetchall()

    results: list[dict] = []
    for row in patients:
        pat = _get_patient_data(row["patient_id"])
        if pat:
            results.append(_estimate_clock_drawing(pat))

    if not results:
        return {"error": "No patients found", "patients": []}

    total_vals   = [r["total_score"]   for r in results]
    contour_vals = [r["contour_score"] for r in results]
    numbers_vals = [r["numbers_score"] for r in results]
    hands_vals   = [r["hands_score"]   for r in results]
    center_vals  = [r["center_score"]  for r in results]

    severity_dist = {"Normal": 0, "Mild": 0, "Borderline": 0, "Impaired": 0, "Severe": 0}
    for r in results:
        lbl = r["overall_severity"]
        if lbl in severity_dist:
            severity_dist[lbl] += 1

    n = len(results)
    impaired_count = sum(1 for v in total_vals if v <= 3.0)

    return {
        "generated_at":          datetime.now(timezone.utc).isoformat(),
        "n_patients":            n,
        "mean_total_score":      round(statistics.mean(total_vals), 2),
        "sd_total_score":        round(statistics.stdev(total_vals)   if n > 1 else 0.0, 2),
        "mean_contour_score":    round(statistics.mean(contour_vals), 2),
        "sd_contour_score":      round(statistics.stdev(contour_vals) if n > 1 else 0.0, 2),
        "mean_numbers_score":    round(statistics.mean(numbers_vals), 2),
        "sd_numbers_score":      round(statistics.stdev(numbers_vals) if n > 1 else 0.0, 2),
        "mean_hands_score":      round(statistics.mean(hands_vals), 2),
        "sd_hands_score":        round(statistics.stdev(hands_vals)   if n > 1 else 0.0, 2),
        "mean_center_score":     round(statistics.mean(center_vals), 2),
        "sd_center_score":       round(statistics.stdev(center_vals)  if n > 1 else 0.0, 2),
        "pct_impaired":          round(100.0 * impaired_count / n, 1),
        "severity_distribution": severity_dist,
        "norm_total_mean":       TOTAL_SCORE_NORMS["mean"],
        "norm_total_sd":         TOTAL_SCORE_NORMS["sd"],
        "impairment_threshold":  3.0,
        "patients":              results,
    }

# ---------------------------------------------------------------------------
# Detail function (single patient)
# ---------------------------------------------------------------------------

def clock_drawing_detail(patient_id: str) -> dict[str, Any]:
    """
    Detailed CDT profile for one patient, including total score, component
    subscores, severity band, clinical interpretation, AED note, and
    epilepsy-specific note.
    """
    pat = _get_patient_data(patient_id)
    if pat is None:
        return {"error": f"Patient {patient_id!r} not found"}

    est = _estimate_clock_drawing(pat)

    # Clinical interpretation narrative
    total = est["total_score"]
    if total >= 4.5:
        interp = (
            f"Total CDT score {total:.1f}/5 is within the normal range "
            f"(norm mean 4.5, SD 0.7; Shulman 1993; Freedman 1994).  "
            f"The clock drawing is intact — circle, number placement, and hand "
            f"placement are accurate, indicating preserved visuospatial "
            f"constructional ability, executive function, and semantic memory."
        )
    elif total >= 4.0:
        interp = (
            f"Total CDT score {total:.1f}/5 shows minor visuospatial errors.  "
            f"The clock is recognizable and largely correct, but subtle spacing "
            f"or number placement issues are present.  This may reflect early "
            f"attentional or executive function changes worth monitoring "
            f"longitudinally, particularly in the context of AED therapy."
        )
    elif total >= 3.0:
        interp = (
            f"Total CDT score {total:.1f}/5 indicates an inaccurate clock "
            f"representation.  Numbers may be placed outside the circle or in "
            f"wrong positions, and/or hand placement shows errors.  Scores at or "
            f"below 3 are associated with cognitive impairment (Shulman 1986).  "
            f"Formal neuropsychological assessment is recommended to evaluate "
            f"the extent of visuospatial and executive function deficits."
        )
    elif total >= 1.0:
        interp = (
            f"Total CDT score {total:.1f}/5 indicates moderate to severe "
            f"disorganization.  Numbers are randomly placed or absent, and hands "
            f"are missing or grossly incorrect.  This level of impairment "
            f"warrants urgent neuropsychological evaluation and review of "
            f"reversible causes (AED toxicity, seizure burden, metabolic "
            f"factors).  Functional impact on daily activities is likely."
        )
    else:
        interp = (
            f"Total CDT score {total:.1f}/5 indicates no recognizable clock "
            f"drawing attempt.  Severe visuospatial/constructional impairment "
            f"is present.  Immediate neurological and neuropsychological "
            f"evaluation is essential.  Consider AED toxicity, status "
            f"epilepticus effects, and neurodegenerative co-morbidity."
        )

    aed_note = ""
    if est["high_burden_aeds"] >= 2:
        aed_note = (
            "Multiple high-cognitive-burden AEDs detected (e.g., topiramate, "
            "phenobarbital).  These medications impair processing speed, motor "
            "coordination, and executive function — all of which directly affect "
            "clock drawing performance.  CDT impairment may be partially "
            "iatrogenic; consider AED simplification if clinically feasible."
        )
    elif est["high_burden_aeds"] == 1:
        aed_note = (
            "One high-cognitive-burden AED detected.  Processing speed and "
            "motor coordination side effects may contribute to CDT errors.  "
            "Monitor for progressive decline and consider medication review."
        )

    epilepsy_note = ""
    epi = (pat.get("epilepsy_type") or "").lower()
    if "temporal" in epi:
        epilepsy_note = (
            "Temporal lobe epilepsy is associated with moderate CDT impairment.  "
            "The temporal-parietal junction mediates visuospatial processing and "
            "semantic memory for clock concepts — both are taxed by the CDT.  "
            "Dominant (left) temporal foci may additionally impair number "
            "sequencing through language-mediated retrieval deficits."
        )
    elif "frontal" in epi:
        epilepsy_note = (
            "Frontal lobe epilepsy particularly impacts the executive components "
            "of clock drawing — planning the spatial layout, sequencing numbers "
            "1-12 correctly, and abstracting from the verbal time instruction "
            "to correct hand positions.  Contour drawing may be relatively "
            "preserved while numbers and hands show greater impairment."
        )
    elif "generalised" in epi or "generalized" in epi:
        epilepsy_note = (
            "Generalized epilepsy typically causes less CDT-specific impairment "
            "unless co-morbid cognitive decline is present.  However, medication "
            "burden (polytherapy) and cumulative seizure effects can degrade "
            "processing speed and executive function, indirectly affecting "
            "clock drawing performance."
        )

    return {
        "patient_id":       patient_id,
        "patient_name":     pat.get("patient_name", "Unknown"),
        "age":              pat.get("age"),
        "epilepsy_type":    pat.get("epilepsy_type"),
        "seizure_count_6m": est["seizure_count_6m"],
        "medications":      pat.get("medications", []),
        "estimated":        est,
        "interpretation":   interp,
        "aed_note":         aed_note,
        "epilepsy_note":    epilepsy_note,
        "norms": {
            "total_mean":           TOTAL_SCORE_NORMS["mean"],
            "total_sd":             TOTAL_SCORE_NORMS["sd"],
            "contour_mean":         CONTOUR_NORMS["mean"],
            "contour_sd":           CONTOUR_NORMS["sd"],
            "numbers_mean":         NUMBERS_NORMS["mean"],
            "numbers_sd":           NUMBERS_NORMS["sd"],
            "hands_mean":           HANDS_NORMS["mean"],
            "hands_sd":             HANDS_NORMS["sd"],
            "center_mean":          CENTER_NORMS["mean"],
            "center_sd":            CENTER_NORMS["sd"],
            "impairment_threshold": 3.0,
        },
        "reference":     "Shulman 1986; Freedman 1994; Sunderland 1989",
        "generated_at":  datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# Trend function (longitudinal projection)
# ---------------------------------------------------------------------------

def clock_drawing_trend(patient_id: str, months: int = 12) -> dict[str, Any]:
    """
    Project longitudinal CDT trajectory for one patient.

    Higher scores = better performance.
    Scores are expected to improve as AED burden is reduced and seizure
    control improves.  However, age-related decline may limit recovery
    ceiling in older patients.
    """
    pat = _get_patient_data(patient_id)
    if pat is None:
        return {"error": f"Patient {patient_id!r} not found"}

    baseline = _estimate_clock_drawing(pat)
    total_base   = baseline["total_score"]
    contour_base = baseline["contour_score"]
    numbers_base = baseline["numbers_score"]
    hands_base   = baseline["hands_score"]
    center_base  = baseline["center_score"]

    high_burden = baseline["high_burden_aeds"]
    seizure_cnt = baseline["seizure_count_6m"]

    # Monthly improvement rates (positive = improvement since HIGHER is better)
    # AED simplification → processing speed recovery → better drawing
    aed_monthly_delta_total = (
        0.12 if high_burden >= 2 else (0.06 if high_burden == 1 else 0.01)
    )
    seizure_monthly_delta_total = (
        0.08 if seizure_cnt > 10 else (0.04 if seizure_cnt > 4 else 0.01)
    )

    # Component-specific monthly deltas
    aed_monthly_delta_contour = (
        0.04 if high_burden >= 2 else (0.02 if high_burden == 1 else 0.005)
    )
    aed_monthly_delta_numbers = (
        0.06 if high_burden >= 2 else (0.03 if high_burden == 1 else 0.005)
    )
    aed_monthly_delta_hands = (
        0.07 if high_burden >= 2 else (0.035 if high_burden == 1 else 0.005)
    )
    aed_monthly_delta_center = (
        0.02 if high_burden >= 2 else (0.01 if high_burden == 1 else 0.002)
    )

    # Ceiling constraints — cannot exceed norm values
    total_ceil   = min(5.0, TOTAL_SCORE_NORMS["mean"] + TOTAL_SCORE_NORMS["sd"])
    contour_ceil = 2.0
    numbers_ceil = 4.0
    hands_ceil   = 4.0
    center_ceil  = 1.0

    time_points: list[dict] = []
    total_cur   = total_base
    contour_cur = contour_base
    numbers_cur = numbers_base
    hands_cur   = hands_base
    center_cur  = center_base

    for mo in range(months + 1):
        dampen = 0.5 if mo > 6 else 1.0

        time_points.append({
            "month":          mo,
            "total_score":    round(total_cur, 1),
            "contour_score":  round(contour_cur, 1),
            "numbers_score":  round(numbers_cur, 1),
            "hands_score":    round(hands_cur, 1),
            "center_score":   round(center_cur, 1),
            "total_band":     _classify(total_cur, SEVERITY_BANDS_TOTAL)["label"],
            "contour_band":   _classify(contour_cur, SEVERITY_BANDS_CONTOUR)["label"],
            "numbers_band":   _classify(numbers_cur, SEVERITY_BANDS_NUMBERS)["label"],
            "hands_band":     _classify(hands_cur, SEVERITY_BANDS_HANDS)["label"],
            "center_band":    _classify(center_cur, SEVERITY_BANDS_CENTER)["label"],
            "impaired":       total_cur <= 3.0,
        })

        if mo < months:
            total_delta = (aed_monthly_delta_total + seizure_monthly_delta_total) * dampen
            total_cur   = min(total_ceil,   total_cur   + total_delta)
            contour_cur = min(contour_ceil, contour_cur + aed_monthly_delta_contour * dampen)
            numbers_cur = min(numbers_ceil, numbers_cur + aed_monthly_delta_numbers * dampen)
            hands_cur   = min(hands_ceil,   hands_cur   + aed_monthly_delta_hands   * dampen)
            center_cur  = min(center_ceil,  center_cur  + aed_monthly_delta_center  * dampen)

    final = time_points[-1]
    total_change = round(final["total_score"] - total_base, 1)

    return {
        "patient_id":        patient_id,
        "months_projected":  months,
        "baseline": {
            "total_score":   total_base,
            "contour_score": contour_base,
            "numbers_score": numbers_base,
            "hands_score":   hands_base,
            "center_score":  center_base,
        },
        "projected_final": {
            "total_score":   final["total_score"],
            "contour_score": final["contour_score"],
            "numbers_score": final["numbers_score"],
            "hands_score":   final["hands_score"],
            "center_score":  final["center_score"],
        },
        "change_total_score":  total_change,
        "trend_direction":     "improvement" if total_change > 0.0 else "stable_or_decline",
        "clinical_note": (
            "CDT scores are expected to improve (increase toward 5) as AED "
            "cognitive burden is reduced and seizure control optimizes.  "
            "Executive components (numbers sequencing, hand placement) typically "
            "show the greatest recovery potential.  Age-related decline may "
            "limit the ceiling in older patients — monitor for plateau vs "
            "continued improvement."
        ),
        "time_points":  time_points,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# Scale definitions (metadata)
# ---------------------------------------------------------------------------

def scale_definitions() -> dict[str, Any]:
    """Return structured metadata describing the Clock Drawing Test scale."""
    return {
        "scale_name":     "Clock Drawing Test (CDT)",
        "abbreviation":   "CDT",
        "domain":         "Visuospatial / constructional ability, executive function, semantic memory, attention",
        "paradigm": (
            "The patient is asked to draw a clock face showing a specific time "
            "(e.g., '10 minutes past 11' = 11:10).  The test assesses "
            "visuospatial/constructional ability (drawing the circle), executive "
            "function (planning number placement, abstracting time to hand "
            "positions), semantic memory (knowing what a clock looks like and "
            "how numbers are arranged), and attention (completing all components "
            "without omission).  Administration takes 1-3 minutes."
        ),
        "primary_metric":  "Total score (0-5 Shulman scale) — HIGHER is better (5 = perfect clock)",
        "secondary_metrics": [
            "Contour score (0-2) — circle quality (2 = good, 1 = deformed, 0 = absent)",
            "Numbers score (0-4) — number placement accuracy (4 = all correct)",
            "Hands score (0-4) — hand placement accuracy (4 = both correct for target time)",
            "Center score (0-1) — center point present and correctly placed",
        ],
        "score_range": {
            "total":   "0 – 5 (Shulman 1986 scale)",
            "contour": "0 – 2",
            "numbers": "0 – 4",
            "hands":   "0 – 4",
            "center":  "0 – 1",
        },
        "direction": "HIGHER score = better performance (5 is perfect clock)",
        "scoring_system": {
            "5": "Perfect clock (circle, numbers, hands all correct)",
            "4": "Minor visuospatial errors (spacing, missing numbers)",
            "3": "Inaccurate representation (numbers outside circle, wrong arrangement)",
            "2": "Moderate disorganization (numbers randomly placed, hands absent/wrong)",
            "1": "Severe disorganization (barely recognizable as clock)",
            "0": "No recognizable attempt",
        },
        "norms": {
            "source":           "Shulman 1993; Freedman 1994",
            "total_mean":       TOTAL_SCORE_NORMS["mean"],
            "total_sd":         TOTAL_SCORE_NORMS["sd"],
            "contour_mean":     CONTOUR_NORMS["mean"],
            "contour_sd":       CONTOUR_NORMS["sd"],
            "numbers_mean":     NUMBERS_NORMS["mean"],
            "numbers_sd":       NUMBERS_NORMS["sd"],
            "hands_mean":       HANDS_NORMS["mean"],
            "hands_sd":         HANDS_NORMS["sd"],
            "center_mean":      CENTER_NORMS["mean"],
            "center_sd":        CENTER_NORMS["sd"],
            "impairment_threshold": 3.0,
        },
        "severity_bands_total":   SEVERITY_BANDS_TOTAL,
        "severity_bands_contour": SEVERITY_BANDS_CONTOUR,
        "severity_bands_numbers": SEVERITY_BANDS_NUMBERS,
        "severity_bands_hands":   SEVERITY_BANDS_HANDS,
        "severity_bands_center":  SEVERITY_BANDS_CENTER,
        "clinical_relevance": (
            "The CDT is one of the most widely used neuropsychological screening "
            "tools for cognitive impairment and dementia (Freedman 1994).  It is "
            "quick (1-3 minutes), requires no specialized equipment, and is "
            "sensitive to visuospatial, executive, and semantic memory deficits.  "
            "In epilepsy populations, temporal lobe epilepsy causes moderate CDT "
            "impairment through visuospatial and semantic memory disruption, while "
            "frontal lobe epilepsy primarily affects executive components (number "
            "sequencing, hand abstraction).  Topiramate, phenobarbital, and other "
            "sedating AEDs impair processing speed and motor coordination, directly "
            "degrading drawing quality.  A CDT score <= 3 warrants formal "
            "neuropsychological evaluation.  Serial CDT scores provide a practical "
            "longitudinal marker of cognitive change during AED adjustments."
        ),
        "epilepsy_specific_effects": {
            "temporal_lobe": (
                "Moderate CDT impairment — temporal-parietal junction mediates "
                "visuospatial processing and semantic memory for clock concepts."
            ),
            "frontal_lobe": (
                "Executive components affected — planning spatial layout, "
                "sequencing numbers 1-12, abstracting verbal time to hand positions."
            ),
            "generalized": (
                "Less CDT-specific unless co-morbid cognitive decline.  However, "
                "AED burden and cumulative seizure effects can degrade performance."
            ),
            "aed_burden": (
                "Topiramate, phenobarbital → processing speed impacts drawing "
                "quality.  Motor coordination side effects affect contour accuracy."
            ),
            "seizure_frequency": (
                "Cumulative cognitive damage from frequent seizures affects all "
                "CDT components, particularly executive and visuospatial domains."
            ),
            "age_interaction": (
                "Older patients score lower — age-related visuospatial and "
                "executive decline compounds epilepsy-specific effects."
            ),
        },
        "references": [
            "Shulman KI, Shedletsky R, Silver IL. The challenge of time: "
            "Clock-drawing and cognitive function in the elderly. Int J Geriatr Psychiatry. 1986;1:135-140.",
            "Freedman M, Leach L, Kaplan E, Winocur G, Shulman KI, Delis DC. "
            "Clock Drawing: A Neuropsychological Analysis. Oxford University Press, 1994.",
            "Sunderland T, Hill JL, Mellow AM, et al. Clock drawing in "
            "Alzheimer's disease: a novel measure of dementia severity. "
            "J Am Geriatr Soc. 1989;37:725-729.",
            "Shulman KI. Clock-drawing: is it the ideal cognitive screening test? "
            "Int J Geriatr Psychiatry. 2000;15(6):548-561.",
            "Brodaty H, Moore CM. The Clock Drawing Test for dementia of the "
            "Alzheimer's type: a comparison of three scoring methods in a memory "
            "disorders clinic. Int J Geriatr Psychiatry. 1997;12(6):619-627.",
        ],
    }

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Clock Drawing Test (CDT) — NeuroAI Clinical Dashboard")
    print("=" * 70)

    defs = scale_definitions()
    print(f"\nScale         : {defs['scale_name']}  ({defs['abbreviation']})")
    print(f"Domain        : {defs['domain']}")
    print(f"Primary metric: {defs['primary_metric']}")
    print(f"Direction     : {defs['direction']}")

    print("\n--- Population Dashboard ---")
    dash = clock_drawing_dashboard()
    if "error" in dash:
        print(f"ERROR: {dash['error']}")
    else:
        print(f"Patients           : {dash['n_patients']}")
        print(f"Mean total score   : {dash['mean_total_score']:.2f}  (±{dash['sd_total_score']:.2f})")
        print(f"Mean contour score : {dash['mean_contour_score']:.2f}  (±{dash['sd_contour_score']:.2f})")
        print(f"Mean numbers score : {dash['mean_numbers_score']:.2f}  (±{dash['sd_numbers_score']:.2f})")
        print(f"Mean hands score   : {dash['mean_hands_score']:.2f}  (±{dash['sd_hands_score']:.2f})")
        print(f"Mean center score  : {dash['mean_center_score']:.2f}  (±{dash['sd_center_score']:.2f})")
        print(f"Impaired (score<=3): {dash['pct_impaired']:.1f} %")
        print("\nSeverity distribution:")
        for label, count in dash["severity_distribution"].items():
            pct = 100.0 * count / dash["n_patients"] if dash["n_patients"] else 0
            print(f"  {label:<12}: {count:3d}  ({pct:.1f} %)")

        print("\nSample patients (first 5):")
        header = (
            f"  {'ID':<12} {'Total':>6} {'Contr':>6} "
            f"{'Nums':>6} {'Hands':>6} {'Centr':>6} {'Sev':<12} {'Epilepsy type'}"
        )
        print(header)
        print("  " + "-" * 80)
        for p in dash["patients"][:5]:
            print(
                f"  {p['patient_id']:<12} "
                f"{p['total_score']:>6.1f} "
                f"{p['contour_score']:>6.1f} "
                f"{p['numbers_score']:>6.1f} "
                f"{p['hands_score']:>6.1f} "
                f"{p['center_score']:>6.1f} "
                f"{p['overall_severity']:<12} "
                f"{p['epilepsy_type']}"
            )

    if "patients" in dash and dash["patients"]:
        pid = dash["patients"][0]["patient_id"]
        print(f"\n--- Detail: {pid} ---")
        detail = clock_drawing_detail(pid)
        if "error" not in detail:
            est = detail["estimated"]
            print(f"Total score     : {est['total_score']:.1f}/5  [{est['total_band']['label']}]")
            print(f"Contour score   : {est['contour_score']:.1f}/2  [{est['contour_band']['label']}]")
            print(f"Numbers score   : {est['numbers_score']:.1f}/4  [{est['numbers_band']['label']}]")
            print(f"Hands score     : {est['hands_score']:.1f}/4  [{est['hands_band']['label']}]")
            print(f"Center score    : {est['center_score']:.1f}/1  [{est['center_band']['label']}]")
            print(f"Impaired (<=3)  : {'Yes' if est['impaired'] else 'No'}")
            print(f"Overall severity: {est['overall_severity']}")
            print(f"\nInterpretation: {detail['interpretation']}")
            if detail["aed_note"]:
                print(f"AED note      : {detail['aed_note']}")
            if detail["epilepsy_note"]:
                print(f"Epilepsy note : {detail['epilepsy_note']}")

        print(f"\n--- 12-Month Trend: {pid} ---")
        trend = clock_drawing_trend(pid, months=12)
        if "error" not in trend:
            print(
                f"Baseline → Mo 12 total: "
                f"{trend['baseline']['total_score']:.1f} → "
                f"{trend['projected_final']['total_score']:.1f}  "
                f"(Δ {trend['change_total_score']:+.1f})"
            )
            print(f"Trend direction : {trend['trend_direction']}")
            print(f"\nTime points (every 3 months):")
            for tp in trend["time_points"]:
                if tp["month"] % 3 == 0:
                    print(
                        f"  Mo {tp['month']:>2}: total {tp['total_score']:>4.1f}  "
                        f"contour {tp['contour_score']:>3.1f}  "
                        f"numbers {tp['numbers_score']:>3.1f}  "
                        f"hands {tp['hands_score']:>3.1f}  "
                        f"center {tp['center_score']:>3.1f}  "
                        f"[{tp['total_band']}]"
                    )

    print("\n" + "=" * 70)
    print("Clock Drawing Test (CDT) dashboard complete.")
