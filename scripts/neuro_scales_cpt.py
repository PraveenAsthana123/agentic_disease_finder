"""
Continuous Performance Test (CPT) — Neuropsychological Scale Dashboard
=======================================================================
Paradigm   : Sustained-attention / vigilance task.  Subjects see a rapid
             stream of letters (e.g., 1 per 1-2 s, ~14 min) and press
             a button for EVERY letter EXCEPT a designated non-target
             (e.g., 'X').  Targets ≈ 90 %, non-targets ≈ 10 %.
Primary DV : Omissions — missed targets (inattention, %).
             LOWER omission rate = better sustained attention.
             Omission rate > 10 % = clinically significant inattention.
Secondary  : Commissions — false alarms to non-target (impulsivity, %).
             Hit RT — mean reaction time to correct target presses (ms).
             Hit RT SE — standard error of RT (attention consistency).
             d' (d-prime) — signal-detection discriminability index.
             β (beta) — response bias (conservative vs liberal).
Reference  : Conners CK.  Conners Continuous Performance Test 3rd Edition
             (CPT 3).  Multi-Health Systems (MHS), 2014.
             Riccio CA et al.  The continuous performance test: a window
             to the neural substrates for attention?  Arch Clin
             Neuropsychol. 2002;17(3):235-72.
             Seidel WT & Joschko M.  Evidence of difficulties in sustained
             attention in children with ADDH.  J Abnorm Child Psychol.
             1990;18(2):217-29.
Norms      : Healthy adults: omission rate < 3 %, commission rate < 8 %,
             hit RT 340-420 ms, d' > 2.5  (Conners 2014; Riccio 2002).
Impairment : Omissions > 10 % = clinically significant inattention.
             Commissions > 20 % = significant impulsivity.
             d' < 1.5 = poor discriminability.
Direction  : Omissions DECREASE with improvement (lower is better).
             d' INCREASES with improvement (higher is better).
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

# Norm reference: Conners CPT-3 (2014); Riccio 2002
OMISSION_NORMS   = {"mean": 1.8, "sd": 1.5}   # percent
COMMISSION_NORMS  = {"mean": 5.5, "sd": 3.5}   # percent
HIT_RT_NORMS      = {"mean": 380, "sd": 40}    # milliseconds
HIT_RT_SE_NORMS   = {"mean": 6.5, "sd": 2.5}   # milliseconds
D_PRIME_NORMS     = {"mean": 3.2, "sd": 0.7}   # dimensionless
BETA_NORMS        = {"mean": 1.1, "sd": 0.4}   # dimensionless

# Severity bands — Omission rate (lower = better sustained attention)
SEVERITY_BANDS_OMISSION = [
    {"range": [0.0,   2.99], "label": "Normal",     "color": "green",  "description": "Omission rate < 3 %"},
    {"range": [3.0,   5.99], "label": "Mild",       "color": "olive",  "description": "Omission rate 3-6 %"},
    {"range": [6.0,   9.99], "label": "Borderline", "color": "orange", "description": "Omission rate 6-10 %"},
    {"range": [10.0, 100.0], "label": "Impaired",   "color": "red",    "description": "Omission rate ≥ 10 % — clinically significant inattention"},
]

SEVERITY_BANDS_COMMISSION = [
    {"range": [0.0,   7.99], "label": "Normal",     "color": "green",  "description": "Commission rate < 8 %"},
    {"range": [8.0,  13.99], "label": "Mild",       "color": "olive",  "description": "Commission rate 8-14 %"},
    {"range": [14.0, 19.99], "label": "Borderline", "color": "orange", "description": "Commission rate 14-20 %"},
    {"range": [20.0, 100.0], "label": "Impaired",   "color": "red",    "description": "Commission rate ≥ 20 % — significant impulsivity"},
]

SEVERITY_BANDS_HIT_RT = [
    {"range": [0,   420], "label": "Normal",     "color": "green",  "description": "Hit RT ≤ 420 ms"},
    {"range": [421, 500], "label": "Slow",       "color": "olive",  "description": "Hit RT 421-500 ms"},
    {"range": [501, 600], "label": "Very slow",  "color": "orange", "description": "Hit RT 501-600 ms"},
    {"range": [601, 9999],"label": "Impaired",   "color": "red",    "description": "Hit RT > 600 ms"},
]

SEVERITY_BANDS_D_PRIME = [
    {"range": [2.5, 99.0], "label": "Normal",     "color": "green",  "description": "d' ≥ 2.5 — good discriminability"},
    {"range": [2.0,  2.49], "label": "Mild",       "color": "olive",  "description": "d' 2.0-2.5 — mildly reduced"},
    {"range": [1.5,  1.99], "label": "Borderline", "color": "orange", "description": "d' 1.5-2.0 — borderline"},
    {"range": [0.0,  1.49], "label": "Impaired",   "color": "red",    "description": "d' < 1.5 — poor discriminability"},
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


def _age_rt_penalty(age: int) -> float:
    """Older adults show slower RTs — sustained attention fatigues faster with age."""
    if age < 35:
        return 0.0
    if age < 55:
        return 22.0
    return 50.0


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


def _z_from_rate(rate_pct: float) -> float:
    """Convert a hit/FA rate (%) to a z-score via probit approximation (Abramowitz & Stegun)."""
    p = max(0.01, min(0.99, rate_pct / 100.0))
    if p <= 0.5:
        t = math.sqrt(-2.0 * math.log(p))
        z = -(t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
              (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t))
    else:
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / \
            (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)
    return z


# ---------------------------------------------------------------------------
# Core estimation
# ---------------------------------------------------------------------------

def _estimate_cpt(patient: dict[str, Any]) -> dict[str, Any]:
    """
    Estimate CPT performance from patient clinical data.

    Returns omission_rate, commission_rate, hit_rt_ms, hit_rt_se_ms, d_prime,
    beta, and derived severity bands.
    Omissions measure sustained attention; commissions measure impulsivity.
    d' measures discriminability (signal detection).
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
    # Omission rate (%) — sustained attention failures; PRIMARY DV
    # ------------------------------------------------------------------ #
    omission = 1.8 + rng_norm(0.0, 0.8)

    # Absence/generalised epilepsy most impacts sustained attention
    if "generalised" in epilepsy_type or "generalized" in epilepsy_type:
        omission += 6.5
    elif "absence" in epilepsy_type:
        omission += 8.0
    elif "frontal" in epilepsy_type:
        omission += 3.5
    elif "focal" in epilepsy_type and "temporal" in epilepsy_type:
        omission += 4.5
    elif "focal" in epilepsy_type:
        omission += 2.5

    # Seizure burden — recent seizures disrupt sustained networks
    if seizure_count > 20:
        omission += 6.0
    elif seizure_count > 10:
        omission += 3.5
    elif seizure_count > 4:
        omission += 1.5

    # AED burden — sedating AEDs cause vigilance lapses
    if high_burden_count >= 2:
        omission += 5.0
    elif high_burden_count == 1:
        omission += 2.5
    if known_aed_count >= 3:
        omission += 1.5
    elif known_aed_count == 2:
        omission += 0.5

    # Age effect — sustained attention declines with age
    if age >= 65:
        omission += 3.0
    elif age >= 50:
        omission += 1.2

    omission = max(0.2, min(50.0, round(omission, 1)))

    # ------------------------------------------------------------------ #
    # Commission rate (%) — false alarms to non-targets (impulsivity)
    # ------------------------------------------------------------------ #
    commission = 5.5 + rng_norm(0.0, 1.5)

    if "frontal" in epilepsy_type:
        commission += 8.0
    elif "generalised" in epilepsy_type or "generalized" in epilepsy_type:
        commission += 4.0
    elif "focal" in epilepsy_type:
        commission += 2.5

    if seizure_count > 20:
        commission += 5.0
    elif seizure_count > 10:
        commission += 3.0
    elif seizure_count > 4:
        commission += 1.5

    if high_burden_count >= 2:
        commission += 4.0
    elif high_burden_count == 1:
        commission += 2.0
    if known_aed_count >= 3:
        commission += 1.5

    if age >= 65:
        commission += 2.0
    elif age >= 50:
        commission += 0.8

    commission = max(0.3, min(55.0, round(commission, 1)))

    # ------------------------------------------------------------------ #
    # Hit RT (ms) — mean reaction time to correct target presses
    # ------------------------------------------------------------------ #
    rt_base = 380.0 + rng_norm(0.0, 20.0) + _age_rt_penalty(age)

    if high_burden_count >= 2:
        rt_base += 60.0
    elif high_burden_count == 1:
        rt_base += 30.0
    if known_aed_count >= 3:
        rt_base += 18.0
    elif known_aed_count == 2:
        rt_base += 8.0

    if seizure_count > 10:
        rt_base += 30.0
    elif seizure_count > 4:
        rt_base += 14.0

    if "generalised" in epilepsy_type or "generalized" in epilepsy_type:
        rt_base += 20.0
    elif "absence" in epilepsy_type:
        rt_base += 25.0

    hit_rt_ms = max(180.0, min(850.0, round(rt_base, 0)))

    # ------------------------------------------------------------------ #
    # Hit RT SE (ms) — intra-individual variability / consistency
    # ------------------------------------------------------------------ #
    rt_se = 6.5 + rng_norm(0.0, 1.5)
    if high_burden_count >= 1:
        rt_se += 3.0
    if seizure_count > 10:
        rt_se += 2.5
    if "absence" in epilepsy_type:
        rt_se += 4.0
    rt_se = max(2.0, min(25.0, round(rt_se, 1)))

    # ------------------------------------------------------------------ #
    # d-prime (signal-detection discriminability)
    # d' = z(hit rate) - z(false alarm rate)
    # ------------------------------------------------------------------ #
    hit_rate_pct = max(1.0, 100.0 - omission)
    fa_rate_pct  = max(0.5, commission)
    d_prime = _z_from_rate(hit_rate_pct) - _z_from_rate(fa_rate_pct)
    d_prime = max(0.0, min(5.0, round(d_prime, 2)))

    # ------------------------------------------------------------------ #
    # Beta (response criterion — conservative vs liberal)
    # β = exp(-0.5 * (z_hit² - z_fa²))
    # ------------------------------------------------------------------ #
    z_hit = _z_from_rate(hit_rate_pct)
    z_fa  = _z_from_rate(fa_rate_pct)
    beta = math.exp(-0.5 * (z_hit * z_hit - z_fa * z_fa))
    beta = max(0.1, min(5.0, round(beta, 2)))

    # ------------------------------------------------------------------ #
    # Severity classifications
    # ------------------------------------------------------------------ #
    omission_band   = _classify(omission, SEVERITY_BANDS_OMISSION)
    commission_band = _classify(commission, SEVERITY_BANDS_COMMISSION)
    rt_band         = _classify(hit_rt_ms, SEVERITY_BANDS_HIT_RT)
    d_prime_band    = _classify(d_prime, SEVERITY_BANDS_D_PRIME)

    severity_order = ["Normal", "Mild", "Borderline", "Impaired"]
    all_labels = [omission_band["label"], commission_band["label"], d_prime_band["label"]]
    worst_idx  = max(severity_order.index(lb) for lb in all_labels if lb in severity_order)
    overall_label = severity_order[worst_idx]

    return {
        "patient_id":       pid,
        "omission_rate":    omission,
        "commission_rate":  commission,
        "hit_rt_ms":        hit_rt_ms,
        "hit_rt_se_ms":     rt_se,
        "d_prime":          d_prime,
        "beta":             beta,
        "omission_band":    omission_band,
        "commission_band":  commission_band,
        "rt_band":          rt_band,
        "d_prime_band":     d_prime_band,
        "overall_severity": overall_label,
        "high_burden_aeds": high_burden_count,
        "known_aeds":       known_aed_count,
        "seizure_count_6m": seizure_count,
        "age":              age,
        "epilepsy_type":    patient.get("epilepsy_type", "unknown"),
    }

# ---------------------------------------------------------------------------
# Dashboard function (population overview)
# ---------------------------------------------------------------------------

def cpt_dashboard(patient_id: str = None) -> dict[str, Any]:
    """
    Population-level CPT summary across all patients in clinical.db,
    or single-patient summary if patient_id is given.
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
            results.append(_estimate_cpt(pat))

    if not results:
        return {"error": "No patients found", "patients": []}

    omission_vals   = [r["omission_rate"]   for r in results]
    commission_vals = [r["commission_rate"]  for r in results]
    rt_vals         = [r["hit_rt_ms"]       for r in results]
    d_prime_vals    = [r["d_prime"]          for r in results]

    severity_dist = {"Normal": 0, "Mild": 0, "Borderline": 0, "Impaired": 0}
    for r in results:
        lbl = r["overall_severity"]
        if lbl in severity_dist:
            severity_dist[lbl] += 1

    n = len(results)
    impaired_omission   = sum(1 for v in omission_vals   if v >= 10.0)
    impaired_commission = sum(1 for v in commission_vals  if v >= 20.0)
    impaired_d_prime    = sum(1 for v in d_prime_vals     if v < 1.5)

    return {
        "generated_at":          datetime.now(timezone.utc).isoformat(),
        "n_patients":            n,
        "mean_omission":         round(statistics.mean(omission_vals), 2),
        "sd_omission":           round(statistics.stdev(omission_vals)   if n > 1 else 0.0, 2),
        "mean_commission":       round(statistics.mean(commission_vals), 2),
        "sd_commission":         round(statistics.stdev(commission_vals) if n > 1 else 0.0, 2),
        "mean_hit_rt_ms":        round(statistics.mean(rt_vals), 1),
        "sd_hit_rt_ms":          round(statistics.stdev(rt_vals)         if n > 1 else 0.0, 1),
        "mean_d_prime":          round(statistics.mean(d_prime_vals), 2),
        "sd_d_prime":            round(statistics.stdev(d_prime_vals)    if n > 1 else 0.0, 2),
        "severity_distribution": severity_dist,
        "pct_omission_impaired":   round(100.0 * impaired_omission   / n, 1),
        "pct_commission_impaired": round(100.0 * impaired_commission / n, 1),
        "pct_d_prime_impaired":    round(100.0 * impaired_d_prime    / n, 1),
        "norm_omission_mean":      OMISSION_NORMS["mean"],
        "norm_d_prime_mean":       D_PRIME_NORMS["mean"],
        "norm_hit_rt_mean_ms":     HIT_RT_NORMS["mean"],
        "patients":              results,
    }

# ---------------------------------------------------------------------------
# Detail function (single patient)
# ---------------------------------------------------------------------------

def cpt_detail(patient_id: str) -> dict[str, Any]:
    """Detailed CPT profile for one patient, including clinical interpretation."""
    pat = _get_patient_data(patient_id)
    if pat is None:
        return {"error": f"Patient {patient_id!r} not found"}

    est = _estimate_cpt(pat)

    # Clinical interpretation narrative
    omission_pct = est["omission_rate"]
    d_prime_val  = est["d_prime"]
    if omission_pct < 3.0 and d_prime_val >= 2.5:
        interp = (
            f"Omission rate {omission_pct:.1f} % and d' {d_prime_val:.2f} are within the "
            f"normal range, indicating intact sustained attention and signal discriminability."
        )
    elif omission_pct < 10.0:
        interp = (
            f"Omission rate {omission_pct:.1f} % is mildly to moderately elevated above the "
            f"healthy adult norm (< 3 %), with d' {d_prime_val:.2f}.  "
            f"Vigilance lapses during the 14-minute task suggest emerging attentional fatigue.  "
            f"Monitor cognitive effects of AEDs and seizure control."
        )
    else:
        interp = (
            f"Omission rate {omission_pct:.1f} % exceeds the clinical threshold of 10 %, "
            f"with d' {d_prime_val:.2f}, indicating significant sustained attention impairment.  "
            f"The patient shows difficulty maintaining vigilance over the 14-minute CPT duration.  "
            f"Neuropsychological intervention, AED review, and seizure management "
            f"optimization are recommended."
        )

    aed_note = ""
    if est["high_burden_aeds"] >= 2:
        aed_note = (
            "Multiple high-cognitive-burden AEDs detected.  "
            "Omissions and hit RT slowing may be partially iatrogenic — sedation impairs vigilance."
        )
    elif est["high_burden_aeds"] == 1:
        aed_note = (
            "One high-cognitive-burden AED detected.  "
            "Monitor for vigilance decrement during extended cognitive tasks."
        )

    absence_note = ""
    epi = (pat.get("epilepsy_type") or "").lower()
    if "absence" in epi:
        absence_note = (
            "Absence epilepsy is associated with marked vigilance impairment on the CPT.  "
            "Brief absence seizures during the 14-minute task can be mistaken for "
            "omission errors and vice versa — video-EEG during CPT may help differentiate."
        )
    elif "generalised" in epi or "generalized" in epi:
        absence_note = (
            "Generalised epilepsy syndromes often show elevated omissions on sustained "
            "attention tasks, reflecting intermittent attentional lapses associated with "
            "subclinical generalised discharges."
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
        "absence_note":     absence_note,
        "norms": {
            "omission_mean_pct":   OMISSION_NORMS["mean"],
            "omission_sd":         OMISSION_NORMS["sd"],
            "commission_mean_pct": COMMISSION_NORMS["mean"],
            "commission_sd":       COMMISSION_NORMS["sd"],
            "hit_rt_mean_ms":      HIT_RT_NORMS["mean"],
            "hit_rt_sd_ms":        HIT_RT_NORMS["sd"],
            "d_prime_mean":        D_PRIME_NORMS["mean"],
            "d_prime_sd":          D_PRIME_NORMS["sd"],
            "impairment_threshold_omission_pct":   10.0,
            "impairment_threshold_commission_pct": 20.0,
            "impairment_threshold_d_prime":        1.5,
        },
        "reference":     "Conners CPT-3 (2014); Riccio 2002",
        "generated_at":  datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# Trend function (longitudinal projection)
# ---------------------------------------------------------------------------

def cpt_trend(patient_id: str, months: int = 12) -> dict[str, Any]:
    """
    Project longitudinal CPT trajectory for one patient.

    Omissions DECREASE with improvement (lower = better sustained attention).
    d' INCREASES with improvement (higher = better discriminability).
    """
    pat = _get_patient_data(patient_id)
    if pat is None:
        return {"error": f"Patient {patient_id!r} not found"}

    baseline = _estimate_cpt(pat)
    omission_base   = baseline["omission_rate"]
    commission_base = baseline["commission_rate"]
    rt_base         = baseline["hit_rt_ms"]
    d_prime_base    = baseline["d_prime"]

    high_burden = baseline["high_burden_aeds"]
    seizure_cnt = baseline["seizure_count_6m"]

    # Monthly improvement rates
    aed_monthly_delta_omission = (
        -0.80 if high_burden >= 2 else (-0.40 if high_burden == 1 else -0.10)
    )
    seizure_monthly_delta_omission = (
        -0.50 if seizure_cnt > 10 else (-0.25 if seizure_cnt > 4 else -0.05)
    )
    aed_monthly_delta_commission = (
        -0.60 if high_burden >= 2 else (-0.30 if high_burden == 1 else -0.08)
    )
    aed_monthly_delta_rt = (
        -4.0 if high_burden >= 2 else (-2.0 if high_burden == 1 else -0.5)
    )

    omission_floor   = max(0.5, OMISSION_NORMS["mean"] - OMISSION_NORMS["sd"])
    commission_floor = max(1.0, COMMISSION_NORMS["mean"] - COMMISSION_NORMS["sd"])
    rt_floor         = max(220.0, HIT_RT_NORMS["mean"] - HIT_RT_NORMS["sd"])

    time_points: list[dict] = []
    omission_cur   = omission_base
    commission_cur = commission_base
    rt_cur         = rt_base

    for mo in range(months + 1):
        dampen = 0.5 if mo > 6 else 1.0

        # Recompute d' from current omission/commission
        hit_rate = max(1.0, 100.0 - omission_cur)
        fa_rate  = max(0.5, commission_cur)
        d_prime_cur = max(0.0, round(_z_from_rate(hit_rate) - _z_from_rate(fa_rate), 2))

        time_points.append({
            "month":           mo,
            "omission_rate":   round(omission_cur, 1),
            "commission_rate": round(commission_cur, 1),
            "hit_rt_ms":       round(rt_cur, 1),
            "d_prime":         d_prime_cur,
            "omission_band":   _classify(omission_cur, SEVERITY_BANDS_OMISSION)["label"],
            "commission_band": _classify(commission_cur, SEVERITY_BANDS_COMMISSION)["label"],
            "rt_band":         _classify(rt_cur, SEVERITY_BANDS_HIT_RT)["label"],
            "d_prime_band":    _classify(d_prime_cur, SEVERITY_BANDS_D_PRIME)["label"],
        })

        if mo < months:
            omission_cur = max(
                omission_floor,
                omission_cur + (aed_monthly_delta_omission + seizure_monthly_delta_omission) * dampen
            )
            commission_cur = max(commission_floor, commission_cur + aed_monthly_delta_commission * dampen)
            rt_cur         = max(rt_floor,         rt_cur         + aed_monthly_delta_rt         * dampen)

    final = time_points[-1]
    omission_change = round(final["omission_rate"] - omission_base, 1)
    d_prime_change  = round(final["d_prime"] - d_prime_base, 2)

    return {
        "patient_id":        patient_id,
        "months_projected":  months,
        "baseline": {
            "omission_rate":   omission_base,
            "commission_rate": commission_base,
            "hit_rt_ms":       rt_base,
            "d_prime":         d_prime_base,
        },
        "projected_final": {
            "omission_rate":   final["omission_rate"],
            "commission_rate": final["commission_rate"],
            "hit_rt_ms":       final["hit_rt_ms"],
            "d_prime":         final["d_prime"],
        },
        "change_omission_pct": omission_change,
        "change_d_prime":      d_prime_change,
        "trend_direction":     "improvement" if omission_change < 0.0 else "stable_or_decline",
        "clinical_note": (
            "Omissions are expected to decrease as AED sedation is reduced "
            "and seizure control improves.  d' increases as discriminability "
            "recovers — better ability to distinguish targets from non-targets."
        ),
        "time_points":  time_points,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# Scale definitions (metadata)
# ---------------------------------------------------------------------------

def scale_definitions() -> dict[str, Any]:
    """Return structured metadata describing the CPT scale."""
    return {
        "scale_name":     "Continuous Performance Test (CPT)",
        "abbreviation":   "CPT",
        "domain":         "Sustained attention / vigilance / signal detection",
        "paradigm": (
            "Subjects view a rapid stream of letters (one per 1-2 seconds, ~14 minutes) "
            "and press a button for EVERY letter EXCEPT a designated non-target (e.g., 'X').  "
            "Targets ≈ 90 %, non-targets ≈ 10 %.  The long duration taxes sustained "
            "attention; omissions increase as vigilance wanes."
        ),
        "primary_metric":  "Omission rate (%) — LOWER is better (missed targets = inattention)",
        "secondary_metrics": [
            "Commission rate (%) — LOWER is better (false alarms = impulsivity)",
            "Hit RT (ms) — mean reaction time to correct target presses",
            "Hit RT SE (ms) — intra-individual RT standard error (consistency)",
            "d' (d-prime) — signal-detection discriminability (HIGHER is better)",
            "β (beta) — response bias (conservative > 1, liberal < 1)",
        ],
        "score_range": {
            "omission":   "0 – 100 %",
            "commission": "0 – 100 %",
            "hit_rt_ms":  "180 – 850 ms",
            "d_prime":    "0 – 5.0",
        },
        "direction": "Omissions DECREASE and d' INCREASES with improvement",
        "norms": {
            "source":              "Conners CPT-3 (2014); Riccio 2002",
            "omission_mean_pct":   OMISSION_NORMS["mean"],
            "omission_sd":         OMISSION_NORMS["sd"],
            "commission_mean_pct": COMMISSION_NORMS["mean"],
            "commission_sd":       COMMISSION_NORMS["sd"],
            "hit_rt_mean_ms":      HIT_RT_NORMS["mean"],
            "hit_rt_sd_ms":        HIT_RT_NORMS["sd"],
            "d_prime_mean":        D_PRIME_NORMS["mean"],
            "d_prime_sd":          D_PRIME_NORMS["sd"],
        },
        "severity_bands_omission":   SEVERITY_BANDS_OMISSION,
        "severity_bands_commission": SEVERITY_BANDS_COMMISSION,
        "severity_bands_hit_rt":     SEVERITY_BANDS_HIT_RT,
        "severity_bands_d_prime":    SEVERITY_BANDS_D_PRIME,
        "clinical_relevance": (
            "The CPT is a gold-standard measure of sustained attention and vigilance, "
            "widely used in ADHD diagnosis and anti-epileptic drug cognitive monitoring.  "
            "Absence epilepsy patients show marked omission increases during the 14-minute "
            "task — brief seizures mimic omission errors, requiring video-EEG differentiation.  "
            "Topiramate, phenobarbital, and other sedating AEDs increase omissions and slow RT "
            "independently of the underlying epilepsy.  d' < 1.5 warrants neuropsychological "
            "consultation."
        ),
        "references": [
            "Conners CK. Conners Continuous Performance Test 3rd Edition (CPT 3). MHS, 2014.",
            "Riccio CA et al. Arch Clin Neuropsychol. 2002;17(3):235-72.",
            "Seidel WT & Joschko M. J Abnorm Child Psychol. 1990;18(2):217-29.",
            "Aldenkamp AP et al. Epilepsy Behav. 2003;4(Suppl 2):S36-40.",
        ],
    }

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Continuous Performance Test (CPT) — NeuroAI Clinical Dashboard")
    print("=" * 70)

    defs = scale_definitions()
    print(f"\nScale         : {defs['scale_name']}  ({defs['abbreviation']})")
    print(f"Domain        : {defs['domain']}")
    print(f"Primary metric: {defs['primary_metric']}")
    print(f"Direction     : {defs['direction']}")

    print("\n--- Population Dashboard ---")
    dash = cpt_dashboard()
    if "error" in dash:
        print(f"ERROR: {dash['error']}")
    else:
        print(f"Patients         : {dash['n_patients']}")
        print(f"Mean omission    : {dash['mean_omission']:.1f} %  (±{dash['sd_omission']:.1f})")
        print(f"Mean commission  : {dash['mean_commission']:.1f} %  (±{dash['sd_commission']:.1f})")
        print(f"Mean hit RT      : {dash['mean_hit_rt_ms']:.0f} ms  (±{dash['sd_hit_rt_ms']:.0f})")
        print(f"Mean d'          : {dash['mean_d_prime']:.2f}  (±{dash['sd_d_prime']:.2f})")
        print(f"Omission impaired  (≥ 10 %): {dash['pct_omission_impaired']:.1f} %")
        print(f"Commission impaired (≥ 20 %): {dash['pct_commission_impaired']:.1f} %")
        print(f"d' impaired (< 1.5)        : {dash['pct_d_prime_impaired']:.1f} %")
        print("\nSeverity distribution:")
        for label, count in dash["severity_distribution"].items():
            pct = 100.0 * count / dash["n_patients"] if dash["n_patients"] else 0
            print(f"  {label:<12}: {count:3d}  ({pct:.1f} %)")

        print("\nSample patients (first 5):")
        header = (
            f"  {'ID':<12} {'Omit%':>7} {'Comm%':>7} "
            f"{'RT ms':>7} {'dprime':>6} {'Sev':<12} {'Epilepsy type'}"
        )
        print(header)
        print("  " + "-" * 74)
        for p in dash["patients"][:5]:
            print(
                f"  {p['patient_id']:<12} "
                f"{p['omission_rate']:>7.1f} "
                f"{p['commission_rate']:>7.1f} "
                f"{p['hit_rt_ms']:>7.0f} "
                f"{p['d_prime']:>6.2f} "
                f"{p['overall_severity']:<12} "
                f"{p['epilepsy_type']}"
            )

    if "patients" in dash and dash["patients"]:
        pid = dash["patients"][0]["patient_id"]
        print(f"\n--- Detail: {pid} ---")
        detail = cpt_detail(pid)
        if "error" not in detail:
            est = detail["estimated"]
            print(f"Omission rate   : {est['omission_rate']:.1f} %  [{est['omission_band']['label']}]")
            print(f"Commission rate : {est['commission_rate']:.1f} %  [{est['commission_band']['label']}]")
            print(f"Hit RT          : {est['hit_rt_ms']:.0f} ms  [{est['rt_band']['label']}]")
            print(f"d'              : {est['d_prime']:.2f}  [{est['d_prime_band']['label']}]")
            print(f"β               : {est['beta']:.2f}")
            print(f"Overall severity: {est['overall_severity']}")
            print(f"\nInterpretation: {detail['interpretation']}")
            if detail["aed_note"]:
                print(f"AED note      : {detail['aed_note']}")
            if detail["absence_note"]:
                print(f"Absence note  : {detail['absence_note']}")

        print(f"\n--- 12-Month Trend: {pid} ---")
        trend = cpt_trend(pid, months=12)
        if "error" not in trend:
            print(
                f"Baseline → Mo 12 omission: "
                f"{trend['baseline']['omission_rate']:.1f} % → "
                f"{trend['projected_final']['omission_rate']:.1f} %  "
                f"(Δ {trend['change_omission_pct']:+.1f} %)"
            )
            print(
                f"Baseline → Mo 12 d'      : "
                f"{trend['baseline']['d_prime']:.2f} → "
                f"{trend['projected_final']['d_prime']:.2f}  "
                f"(Δ {trend['change_d_prime']:+.2f})"
            )
            print(f"Trend direction : {trend['trend_direction']}")
            print(f"\nTime points (every 3 months):")
            for tp in trend["time_points"]:
                if tp["month"] % 3 == 0:
                    print(
                        f"  Mo {tp['month']:>2}: omission {tp['omission_rate']:>5.1f} %  "
                        f"d' {tp['d_prime']:>5.2f}  RT {tp['hit_rt_ms']:>5.0f} ms  "
                        f"[{tp['omission_band']}]"
                    )

    print("\n" + "=" * 70)
    print("Continuous Performance Test (CPT) dashboard complete.")
