"""
Go/No-Go Test — Neuropsychological Scale Dashboard
====================================================
Paradigm   : Response-inhibition task.  Subjects press a button for "Go"
             stimuli and WITHHOLD the response for "No-Go" stimuli presented
             in a continuous stream (typically 80 % Go, 20 % No-Go).
Primary DV : Commission error rate — false alarms on No-Go trials (%).
             A commission error = pressing the button when No-Go appeared.
             LOWER commission rate = better inhibitory control.
             Commission rate > 25 % = clinically significant disinhibition.
Secondary  : Omission error rate — missed Go responses (inattention, %).
             Mean RT for correct Go responses (ms).
Reference  : Bezdjian S et al.  Assessing inattention and impulsivity in
             children during the Go/No-go task.  Br J Dev Psychol. 2009;
             27(Pt 2):365-83.
             Garavan H et al.  Right hemispheric dominance of inhibitory
             control: an event-related functional MRI study.  Proc Natl
             Acad Sci. 1999;96(14):8301-6.
             Drane DL et al.  Cognitive effects of seizures.  Seizure. 2006;
             15(4):221-34.
Norms      : Healthy adults: commission rate < 10 %, omission rate < 5 %,
             mean Go RT 260-330 ms  (Bezdjian 2009; Garavan 1999).
Impairment : Commission rate > 25 % = clinically significant disinhibition.
             Commission rate 10-25 % = borderline.
Direction  : Commission errors DECREASE with improvement (lower is better).
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

# Norm reference: Bezdjian 2009 + Garavan 1999
COMMISSION_NORMS = {"mean": 7.5, "sd": 4.5}   # percent
OMISSION_NORMS   = {"mean": 3.0, "sd": 2.5}   # percent
RT_NORMS         = {"mean": 295, "sd": 35}     # milliseconds

# Severity bands — Commission error rate (lower = better)
SEVERITY_BANDS_COMMISSION = [
    {"range": [0.0,  9.99], "label": "Normal",     "color": "green",  "description": "Commission rate < 10 %"},
    {"range": [10.0, 17.49], "label": "Mild",       "color": "olive",  "description": "Commission rate 10-17.5 %"},
    {"range": [17.5, 24.99], "label": "Borderline", "color": "orange", "description": "Commission rate 17.5-25 %"},
    {"range": [25.0, 100.0], "label": "Impaired",   "color": "red",    "description": "Commission rate ≥ 25 % — clinically significant disinhibition"},
]

SEVERITY_BANDS_OMISSION = [
    {"range": [0.0,   4.99], "label": "Normal",     "color": "green",  "description": "Omission rate < 5 %"},
    {"range": [5.0,  11.99], "label": "Mild",       "color": "olive",  "description": "Omission rate 5-12 %"},
    {"range": [12.0, 19.99], "label": "Borderline", "color": "orange", "description": "Omission rate 12-20 %"},
    {"range": [20.0, 100.0], "label": "Impaired",   "color": "red",    "description": "Omission rate ≥ 20 % — clinically significant inattention"},
]

SEVERITY_BANDS_RT = [
    {"range": [0,   350], "label": "Normal",     "color": "green",  "description": "Mean Go RT ≤ 350 ms"},
    {"range": [351, 430], "label": "Slow",       "color": "olive",  "description": "Mean Go RT 351-430 ms"},
    {"range": [431, 530], "label": "Very slow",  "color": "orange", "description": "Mean Go RT 431-530 ms"},
    {"range": [531, 9999],"label": "Impaired",   "color": "red",    "description": "Mean Go RT > 530 ms"},
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
                "epilepsy_type": row["disease"],  # alias for downstream compatibility
            }

            # Seizure count in last 6 months
            sz = con.execute(
                "SELECT COUNT(*) AS cnt FROM seizure_diary "
                "WHERE patient_id = ? AND event_date >= date('now','-6 months')",
                (patient_id,),
            ).fetchone()
            patient["seizure_count_6m"] = sz["cnt"] if sz else 0

            # Medications (fields_json column)
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

            # Barthel score (latest)
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

def _classify_commission(rate: float) -> dict[str, str]:
    for band in SEVERITY_BANDS_COMMISSION:
        if band["range"][0] <= rate <= band["range"][1]:
            return band
    return SEVERITY_BANDS_COMMISSION[-1]


def _classify_omission(rate: float) -> dict[str, str]:
    for band in SEVERITY_BANDS_OMISSION:
        if band["range"][0] <= rate <= band["range"][1]:
            return band
    return SEVERITY_BANDS_OMISSION[-1]


def _classify_rt(rt_ms: float) -> dict[str, str]:
    for band in SEVERITY_BANDS_RT:
        if band["range"][0] <= rt_ms <= band["range"][1]:
            return band
    return SEVERITY_BANDS_RT[-1]


def _age_rt_penalty(age: int) -> float:
    """Older adults show slower RTs (ms added to mean RT)."""
    if age < 35:
        return 0.0
    if age < 55:
        return 18.0
    return 42.0


# Simple Box-Muller LCG for deterministic patient-specific noise
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

def _estimate_gonogo(patient: dict[str, Any]) -> dict[str, Any]:
    """
    Estimate Go/No-Go performance from patient clinical data.

    Returns commission_rate, omission_rate, go_rt_ms, and derived severity bands.
    Commission rate is in percent (lower = better inhibitory control).
    """
    pid            = patient.get("patient_id", "UNKNOWN")
    age            = int(patient.get("age") or 40)
    epilepsy_type  = (patient.get("epilepsy_type") or "").lower()
    seizure_count  = int(patient.get("seizure_count_6m") or 0)
    meds           = patient.get("medications", [])

    # Deterministic seed from patient_id
    seed = int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)
    rng_norm, rng_uniform = _make_rng(seed)

    # Classify AED burden
    med_names = {
        m.get("medication_name", "").lower().strip().replace(" ", "_")
        for m in meds
    }
    high_burden_count = len(med_names & HIGH_COGNITIVE_BURDEN_AEDS)
    known_aed_count   = len(med_names & AEDS_SET)

    # ------------------------------------------------------------------ #
    # Commission error rate (%) — base near healthy adult norm
    # ------------------------------------------------------------------ #
    commission = 7.5 + rng_norm(0.0, 2.0)   # ~ N(7.5, 2)

    # Epilepsy type modifiers — frontal most impactful for inhibition
    if "frontal" in epilepsy_type:
        commission += 12.0   # frontal lobe epilepsy markedly impairs inhibitory control
    elif "focal" in epilepsy_type and "temporal" in epilepsy_type:
        commission += 7.0
    elif "generalised" in epilepsy_type or "generalized" in epilepsy_type:
        commission += 5.0
    elif "focal" in epilepsy_type:
        commission += 4.0

    # Seizure burden (recent seizures disrupt frontal networks)
    if seizure_count > 20:
        commission += 9.0
    elif seizure_count > 10:
        commission += 5.5
    elif seizure_count > 4:
        commission += 2.5

    # AED burden — GABAergic / high-burden AEDs increase commission errors
    if high_burden_count >= 2:
        commission += 6.0
    elif high_burden_count == 1:
        commission += 3.0
    if known_aed_count >= 3:
        commission += 2.5
    elif known_aed_count == 2:
        commission += 1.0

    # Age effect — commission errors increase with age (disinhibition)
    if age >= 65:
        commission += 3.5
    elif age >= 50:
        commission += 1.5

    commission = max(0.5, min(60.0, round(commission, 1)))

    # ------------------------------------------------------------------ #
    # Omission error rate (%) — mostly inattention
    # ------------------------------------------------------------------ #
    omission = 3.0 + rng_norm(0.0, 1.2)

    if "generalised" in epilepsy_type or "generalized" in epilepsy_type:
        omission += 4.0
    elif "frontal" in epilepsy_type:
        omission += 2.5
    elif "focal" in epilepsy_type:
        omission += 1.5

    if seizure_count > 20:
        omission += 5.0
    elif seizure_count > 10:
        omission += 2.5
    elif seizure_count > 4:
        omission += 1.0

    if high_burden_count >= 2:
        omission += 3.0
    elif high_burden_count == 1:
        omission += 1.5

    omission = max(0.2, min(50.0, round(omission, 1)))

    # ------------------------------------------------------------------ #
    # Mean Go RT (ms) — slower with AEDs and age
    # ------------------------------------------------------------------ #
    rt_base = 295.0 + rng_norm(0.0, 20.0) + _age_rt_penalty(age)

    # Generalised slowing from AEDs
    if high_burden_count >= 2:
        rt_base += 55.0
    elif high_burden_count == 1:
        rt_base += 30.0
    if known_aed_count >= 3:
        rt_base += 20.0
    elif known_aed_count == 2:
        rt_base += 8.0

    # Seizure burden
    if seizure_count > 10:
        rt_base += 25.0
    elif seizure_count > 4:
        rt_base += 12.0

    # Epilepsy type RT modifiers (frontal paradoxically faster impulsive responses)
    if "frontal" in epilepsy_type:
        rt_base -= 10.0   # impulsive fast errors, but Go RT also slightly accelerated
    elif "generalised" in epilepsy_type or "generalized" in epilepsy_type:
        rt_base += 15.0

    go_rt_ms = max(160.0, min(800.0, round(rt_base, 0)))

    # ------------------------------------------------------------------ #
    # RT variability (intra-individual standard deviation)
    # ------------------------------------------------------------------ #
    rt_sd = 45.0 + rng_norm(0.0, 10.0)
    if high_burden_count >= 1:
        rt_sd += 15.0
    if seizure_count > 10:
        rt_sd += 12.0
    rt_sd = max(20.0, min(150.0, round(rt_sd, 1)))

    # ------------------------------------------------------------------ #
    # Severity classifications
    # ------------------------------------------------------------------ #
    commission_band = _classify_commission(commission)
    omission_band   = _classify_omission(omission)
    rt_band         = _classify_rt(go_rt_ms)

    # Overall severity: worst of commission / omission
    severity_order = ["Normal", "Mild", "Borderline", "Impaired"]
    all_labels = [commission_band["label"], omission_band["label"]]
    worst_idx  = max(severity_order.index(lb) for lb in all_labels if lb in severity_order)
    overall_label = severity_order[worst_idx]

    return {
        "patient_id":       pid,
        "commission_rate":  commission,
        "omission_rate":    omission,
        "go_rt_ms":         go_rt_ms,
        "rt_sd_ms":         rt_sd,
        "commission_band":  commission_band,
        "omission_band":    omission_band,
        "rt_band":          rt_band,
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

def gonogo_dashboard() -> dict[str, Any]:
    """
    Population-level Go/No-Go summary across all patients in clinical.db.

    Returns per-patient estimates, severity distribution, means, and flags.
    """
    with _conn() as con:
        patients = con.execute("SELECT patient_id FROM patients").fetchall()

    results: list[dict] = []
    for row in patients:
        pat = _get_patient_data(row["patient_id"])
        if pat:
            results.append(_estimate_gonogo(pat))

    if not results:
        return {"error": "No patients found", "patients": []}

    commission_vals = [r["commission_rate"] for r in results]
    omission_vals   = [r["omission_rate"]   for r in results]
    rt_vals         = [r["go_rt_ms"]         for r in results]

    severity_dist = {"Normal": 0, "Mild": 0, "Borderline": 0, "Impaired": 0}
    for r in results:
        lbl = r["overall_severity"]
        if lbl in severity_dist:
            severity_dist[lbl] += 1

    n = len(results)
    impaired_commission = sum(1 for v in commission_vals if v >= 25.0)
    impaired_omission   = sum(1 for v in omission_vals   if v >= 20.0)

    return {
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "n_patients":       n,
        "mean_commission":  round(statistics.mean(commission_vals), 2),
        "sd_commission":    round(statistics.stdev(commission_vals) if n > 1 else 0.0, 2),
        "mean_omission":    round(statistics.mean(omission_vals), 2),
        "sd_omission":      round(statistics.stdev(omission_vals)   if n > 1 else 0.0, 2),
        "mean_go_rt_ms":    round(statistics.mean(rt_vals), 1),
        "sd_go_rt_ms":      round(statistics.stdev(rt_vals)          if n > 1 else 0.0, 1),
        "severity_distribution": severity_dist,
        "pct_commission_impaired": round(100.0 * impaired_commission / n, 1),
        "pct_omission_impaired":   round(100.0 * impaired_omission   / n, 1),
        "norm_commission_mean":    COMMISSION_NORMS["mean"],
        "norm_rt_mean_ms":         RT_NORMS["mean"],
        "patients":         results,
    }

# ---------------------------------------------------------------------------
# Detail function (single patient)
# ---------------------------------------------------------------------------

def gonogo_detail(patient_id: str) -> dict[str, Any]:
    """
    Detailed Go/No-Go profile for one patient, including clinical interpretation.
    """
    pat = _get_patient_data(patient_id)
    if pat is None:
        return {"error": f"Patient {patient_id!r} not found"}

    est = _estimate_gonogo(pat)

    # Clinical interpretation narrative
    commission_pct = est["commission_rate"]
    if commission_pct < 10.0:
        interp = (
            f"Commission error rate {commission_pct:.1f} % is within the normal range "
            f"(< 10 %), indicating intact response inhibition."
        )
    elif commission_pct < 25.0:
        interp = (
            f"Commission error rate {commission_pct:.1f} % is elevated above the healthy "
            f"adult norm (< 10 %), suggesting borderline to mild disinhibition.  "
            f"Monitoring and targeted cognitive rehabilitation may be warranted."
        )
    else:
        interp = (
            f"Commission error rate {commission_pct:.1f} % exceeds the clinical threshold "
            f"of 25 %, indicating significant inhibitory control impairment consistent with "
            f"frontal executive dysfunction.  Neuropsychological intervention should be "
            f"considered.  Review AED cognitive burden and seizure frequency."
        )

    aed_note = ""
    if est["high_burden_aeds"] >= 2:
        aed_note = (
            "Multiple high-cognitive-burden AEDs detected.  "
            "Consider whether commission errors and RT slowing are partially iatrogenic."
        )
    elif est["high_burden_aeds"] == 1:
        aed_note = (
            "One high-cognitive-burden AED detected.  "
            "Monitor response inhibition alongside AED adjustments."
        )

    frontal_note = ""
    if "frontal" in (pat.get("epilepsy_type") or "").lower():
        frontal_note = (
            "Frontal lobe epilepsy is associated with disproportionate impairment of "
            "response inhibition; commission errors may reflect both AED burden and "
            "epileptogenic frontal network disruption."
        )

    return {
        "patient_id":      patient_id,
        "patient_name":    pat.get("patient_name", "Unknown"),
        "age":             pat.get("age"),
        "epilepsy_type":   pat.get("epilepsy_type"),
        "seizure_count_6m": est["seizure_count_6m"],
        "medications":     pat.get("medications", []),
        "estimated":       est,
        "interpretation":  interp,
        "aed_note":        aed_note,
        "frontal_note":    frontal_note,
        "norms": {
            "commission_mean_pct":  COMMISSION_NORMS["mean"],
            "commission_sd":        COMMISSION_NORMS["sd"],
            "omission_mean_pct":    OMISSION_NORMS["mean"],
            "omission_sd":          OMISSION_NORMS["sd"],
            "go_rt_mean_ms":        RT_NORMS["mean"],
            "go_rt_sd_ms":          RT_NORMS["sd"],
            "impairment_threshold_commission_pct": 25.0,
            "impairment_threshold_omission_pct":   20.0,
        },
        "reference": "Bezdjian 2009; Garavan 1999",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# Trend function (longitudinal projection)
# ---------------------------------------------------------------------------

def gonogo_trend(patient_id: str, months: int = 12) -> dict[str, Any]:
    """
    Project longitudinal Go/No-Go trajectory for one patient.

    Commission errors DECREASE with clinical improvement (lower = better).
    Improvement is driven by AED optimisation and seizure control.
    """
    pat = _get_patient_data(patient_id)
    if pat is None:
        return {"error": f"Patient {patient_id!r} not found"}

    baseline = _estimate_gonogo(pat)
    commission_base = baseline["commission_rate"]
    omission_base   = baseline["omission_rate"]
    rt_base         = baseline["go_rt_ms"]

    high_burden  = baseline["high_burden_aeds"]
    seizure_cnt  = baseline["seizure_count_6m"]

    # Monthly improvement rates (commission errors decrease toward norm)
    # AED optimisation has biggest impact on commission errors
    aed_monthly_delta_commission = (
        -1.20 if high_burden >= 2 else (-0.60 if high_burden == 1 else -0.15)
    )
    seizure_monthly_delta_commission = (
        -0.80 if seizure_cnt > 10 else (-0.40 if seizure_cnt > 4 else -0.10)
    )
    # Omission improvement rate (smaller effect)
    aed_monthly_delta_omission = (
        -0.40 if high_burden >= 2 else (-0.20 if high_burden == 1 else -0.05)
    )
    # RT improvement rate (ms/month, smaller effect)
    aed_monthly_delta_rt = (
        -3.5 if high_burden >= 2 else (-1.8 if high_burden == 1 else -0.4)
    )

    # Minimum achievable values (floor near healthy adult norms)
    commission_floor = max(3.0, COMMISSION_NORMS["mean"] - COMMISSION_NORMS["sd"])
    omission_floor   = max(1.0, OMISSION_NORMS["mean"] - OMISSION_NORMS["sd"])
    rt_floor         = max(200.0, RT_NORMS["mean"] - RT_NORMS["sd"])

    # Diminishing returns: rate halves after 6 months
    time_points: list[dict] = []
    commission_cur = commission_base
    omission_cur   = omission_base
    rt_cur         = rt_base

    for mo in range(months + 1):
        dampen = 0.5 if mo > 6 else 1.0

        time_points.append({
            "month":           mo,
            "commission_rate": round(commission_cur, 1),
            "omission_rate":   round(omission_cur, 1),
            "go_rt_ms":        round(rt_cur, 1),
            "commission_band": _classify_commission(commission_cur)["label"],
            "omission_band":   _classify_omission(omission_cur)["label"],
            "rt_band":         _classify_rt(rt_cur)["label"],
        })

        if mo < months:
            commission_cur = max(
                commission_floor,
                commission_cur
                + (aed_monthly_delta_commission + seizure_monthly_delta_commission) * dampen
            )
            omission_cur = max(omission_floor, omission_cur + aed_monthly_delta_omission * dampen)
            rt_cur       = max(rt_floor,       rt_cur       + aed_monthly_delta_rt       * dampen)

    final = time_points[-1]
    commission_change = round(final["commission_rate"] - commission_base, 1)
    rt_change         = round(final["go_rt_ms"]        - rt_base,        1)

    return {
        "patient_id":       patient_id,
        "months_projected": months,
        "baseline": {
            "commission_rate": commission_base,
            "omission_rate":   omission_base,
            "go_rt_ms":        rt_base,
        },
        "projected_final": {
            "commission_rate": final["commission_rate"],
            "omission_rate":   final["omission_rate"],
            "go_rt_ms":        final["go_rt_ms"],
        },
        "change_commission_pct": commission_change,
        "change_rt_ms":          rt_change,
        "trend_direction":       "improvement" if commission_change < 0.0 else "stable_or_decline",
        "clinical_note": (
            "Commission errors are expected to decrease as AED regimen is optimised "
            "and seizure burden reduces.  Lower commission rate = better inhibitory control."
        ),
        "time_points": time_points,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# Scale definitions (metadata)
# ---------------------------------------------------------------------------

def scale_definitions() -> dict[str, Any]:
    """Return structured metadata describing the Go/No-Go scale."""
    return {
        "scale_name":     "Go/No-Go Test",
        "abbreviation":   "GNG",
        "domain":         "Response inhibition / executive function",
        "paradigm": (
            "Subjects respond to frequent 'Go' stimuli (≈80 %) and "
            "WITHHOLD responses to rare 'No-Go' stimuli (≈20 %).  "
            "Commission errors reflect impulsive responding; omission errors reflect inattention."
        ),
        "primary_metric": "Commission error rate (%) — LOWER is better",
        "secondary_metrics": [
            "Omission error rate (%) — LOWER is better",
            "Mean Go RT (ms) — lower reflects faster inhibitory set",
            "RT intra-individual SD (ms)",
        ],
        "score_range":     {"commission": "0 – 100 %", "go_rt_ms": "160 – 800 ms"},
        "direction":       "Commission errors DECREASE with improvement",
        "norms": {
            "source":              "Bezdjian 2009; Garavan 1999",
            "commission_mean_pct": COMMISSION_NORMS["mean"],
            "commission_sd":       COMMISSION_NORMS["sd"],
            "omission_mean_pct":   OMISSION_NORMS["mean"],
            "omission_sd":         OMISSION_NORMS["sd"],
            "go_rt_mean_ms":       RT_NORMS["mean"],
            "go_rt_sd_ms":         RT_NORMS["sd"],
        },
        "severity_bands_commission": SEVERITY_BANDS_COMMISSION,
        "severity_bands_omission":   SEVERITY_BANDS_OMISSION,
        "severity_bands_rt":         SEVERITY_BANDS_RT,
        "clinical_relevance": (
            "Frontal lobe epilepsy disproportionately impairs No-Go suppression.  "
            "Topiramate, phenobarbital, and other high-burden AEDs increase commission "
            "errors and slow RT independently of the underlying epilepsy.  "
            "Commission rate > 25 % warrants neuropsychological consultation."
        ),
        "references": [
            "Bezdjian S et al. Br J Dev Psychol. 2009;27(Pt 2):365-83.",
            "Garavan H et al. Proc Natl Acad Sci. 1999;96(14):8301-6.",
            "Drane DL et al. Seizure. 2006;15(4):221-34.",
        ],
    }

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("Go/No-Go Test — NeuroAI Clinical Dashboard")
    print("=" * 70)

    # Scale metadata
    defs = scale_definitions()
    print(f"\nScale         : {defs['scale_name']}  ({defs['abbreviation']})")
    print(f"Domain        : {defs['domain']}")
    print(f"Primary metric: {defs['primary_metric']}")
    print(f"Direction     : {defs['direction']}")

    # Population dashboard
    print("\n--- Population Dashboard ---")
    dash = gonogo_dashboard()
    if "error" in dash:
        print(f"ERROR: {dash['error']}")
    else:
        print(f"Patients         : {dash['n_patients']}")
        print(f"Mean commission  : {dash['mean_commission']:.1f} %  (±{dash['sd_commission']:.1f})")
        print(f"Mean omission    : {dash['mean_omission']:.1f} %  (±{dash['sd_omission']:.1f})")
        print(f"Mean Go RT       : {dash['mean_go_rt_ms']:.0f} ms  (±{dash['sd_go_rt_ms']:.0f})")
        print(f"Commission impaired (≥ 25 %): {dash['pct_commission_impaired']:.1f} %")
        print(f"Omission impaired  (≥ 20 %): {dash['pct_omission_impaired']:.1f} %")
        print("\nSeverity distribution:")
        for label, count in dash["severity_distribution"].items():
            pct = 100.0 * count / dash["n_patients"] if dash["n_patients"] else 0
            print(f"  {label:<12}: {count:3d}  ({pct:.1f} %)")

        # Sample first 5 patients
        print("\nSample patients (first 5):")
        header = (
            f"  {'ID':<12} {'Commit%':>8} {'Omit%':>7} "
            f"{'RT ms':>7} {'Sev':<12} {'Epilepsy type'}"
        )
        print(header)
        print("  " + "-" * 68)
        for p in dash["patients"][:5]:
            print(
                f"  {p['patient_id']:<12} "
                f"{p['commission_rate']:>7.1f} "
                f"{p['omission_rate']:>7.1f} "
                f"{p['go_rt_ms']:>7.0f} "
                f"{p['overall_severity']:<12} "
                f"{p['epilepsy_type']}"
            )

    # Detail for first patient
    if "patients" in dash and dash["patients"]:
        pid = dash["patients"][0]["patient_id"]
        print(f"\n--- Detail: {pid} ---")
        detail = gonogo_detail(pid)
        if "error" not in detail:
            est = detail["estimated"]
            print(f"Commission rate : {est['commission_rate']:.1f} %  "
                  f"[{est['commission_band']['label']}]")
            print(f"Omission rate   : {est['omission_rate']:.1f} %  "
                  f"[{est['omission_band']['label']}]")
            print(f"Mean Go RT      : {est['go_rt_ms']:.0f} ms  [{est['rt_band']['label']}]")
            print(f"Overall severity: {est['overall_severity']}")
            print(f"\nInterpretation: {detail['interpretation']}")
            if detail["aed_note"]:
                print(f"AED note      : {detail['aed_note']}")
            if detail["frontal_note"]:
                print(f"Frontal note  : {detail['frontal_note']}")

        # Trend
        print(f"\n--- 12-Month Trend: {pid} ---")
        trend = gonogo_trend(pid, months=12)
        if "error" not in trend:
            print(
                f"Baseline  → Month 12 commission: "
                f"{trend['baseline']['commission_rate']:.1f} % → "
                f"{trend['projected_final']['commission_rate']:.1f} %  "
                f"(Δ {trend['change_commission_pct']:+.1f} %)"
            )
            print(
                f"Baseline  → Month 12 Go RT     : "
                f"{trend['baseline']['go_rt_ms']:.0f} ms → "
                f"{trend['projected_final']['go_rt_ms']:.0f} ms  "
                f"(Δ {trend['change_rt_ms']:+.0f} ms)"
            )
            print(f"Trend direction : {trend['trend_direction']}")
            print(f"\nTime points (every 3 months):")
            for tp in trend["time_points"]:
                if tp["month"] % 3 == 0:
                    print(
                        f"  Mo {tp['month']:>2}: commission {tp['commission_rate']:>5.1f} %  "
                        f"RT {tp['go_rt_ms']:>5.0f} ms  [{tp['commission_band']}]"
                    )

    print("\n" + "=" * 70)
    print("Go/No-Go Test dashboard complete.")
    print("=" * 70)
