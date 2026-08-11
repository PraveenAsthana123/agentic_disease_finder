"""
Motor Evoked Potentials (MEP) Dashboard — NeuroAI EEG
======================================================
TMS-evoked MEP analysis: cortical motor pathway integrity via
Central Motor Conduction Time (CMCT), MEP latency, amplitude, and
cortical silent period — derived from real patient data in clinical.db.

MEP (Motor Evoked Potentials) uses Transcranial Magnetic Stimulation (TMS)
to stimulate the motor cortex and record the resulting muscle response, testing
the integrity of the corticospinal tract from motor cortex → spinal cord →
peripheral motor nerve → muscle.

Stimulation: Single-pulse TMS over primary motor cortex (M1)
Recording: Surface EMG from target muscles

Key parameters (adult normative values):
  Upper limb (APB — Abductor Pollicis Brevis):
    - MEP latency:  normal ≤25.0 ms  (cortical onset to MEP onset)
    - MEP amplitude: normal ≥200 µV  (peak-to-peak)
    - CMCT (upper): normal ≤8.0 ms   (MEP latency − peripheral delay)
  Upper limb (ADM — Abductor Digiti Minimi):
    - MEP latency:  normal ≤27.0 ms
    - MEP amplitude: normal ≥150 µV
    - CMCT (upper): normal ≤9.0 ms
  Lower limb (TA — Tibialis Anterior):
    - MEP latency:  normal ≤45.0 ms
    - MEP amplitude: normal ≥100 µV
    - CMCT (lower): normal ≤20.0 ms
  Lower limb (AH — Abductor Hallucis):
    - MEP latency:  normal ≤50.0 ms
    - MEP amplitude: normal ≥80 µV
    - CMCT (lower): normal ≤22.0 ms
  Cortical Silent Period (CSP): normal ≥100 ms (ipsilateral motor inhibition)
  Inter-side CMCT difference: abnormal >2.5 ms (asymmetry marker)

Diagnostic patterns:
  - Normal: all MEP parameters within reference ranges
  - Upper motor neuron dysfunction: prolonged CMCT, preserved amplitude
  - Cortical lesion: absent or markedly reduced MEP amplitude
  - Corticospinal tract lesion: prolonged latency + reduced amplitude
  - Postictal motor deficit: transient asymmetric MEP abnormality (Todd's paresis)
  - Cervical myelopathy: upper limb CMCT prolonged, lower limb more affected

Severity: Normal, Mild, Moderate, Severe

Data DERIVED from real patient demographics in clinical.db:
  - Patient age, disease, seizure frequency, medication count
  - Deterministic seeding from patient_id for reproducibility

References:
  Rossini PM, et al. Non-invasive electrical and magnetic stimulation of the
  brain, spinal cord, roots and peripheral nerves. Clin Neurophysiol. 2015.
  Chen R, et al. Safety of different inter-train intervals for repetitive TMS.
  Clin Neurophysiol. 1997.
  Hallett M. Transcranial magnetic stimulation. Neuron. 2007;55(2):187-199.
  IFCN Committee. TMS technical and clinical recommendations. Clin Neurophysiol. 2018.

Author: Research Team
"""

import sqlite3
import hashlib
from pathlib import Path
from collections import Counter

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# ── Reference ranges (adult normative, TMS single-pulse MEP) ─────────────────

MEP_MUSCLES = {
    "APB": {
        "name": "Abductor Pollicis Brevis",
        "limb": "Upper",
        "side_param": "upper",
        "latency_upper_ms": 25.0,
        "amplitude_lower_uv": 200.0,
        "cmct_upper_ms": 8.0,
    },
    "ADM": {
        "name": "Abductor Digiti Minimi",
        "limb": "Upper",
        "side_param": "upper",
        "latency_upper_ms": 27.0,
        "amplitude_lower_uv": 150.0,
        "cmct_upper_ms": 9.0,
    },
    "TA": {
        "name": "Tibialis Anterior",
        "limb": "Lower",
        "side_param": "lower",
        "latency_upper_ms": 45.0,
        "amplitude_lower_uv": 100.0,
        "cmct_upper_ms": 20.0,
    },
    "AH": {
        "name": "Abductor Hallucis",
        "limb": "Lower",
        "side_param": "lower",
        "latency_upper_ms": 50.0,
        "amplitude_lower_uv": 80.0,
        "cmct_upper_ms": 22.0,
    },
}

SIDES = ["Left", "Right"]

DIAGNOSTIC_PATTERNS = {
    "normal": "Normal — all MEP latencies, amplitudes, and CMCT within reference ranges",
    "upper_motor_neuron": "Upper Motor Neuron Dysfunction — prolonged CMCT with preserved amplitude",
    "cortical_lesion": "Cortical Lesion — markedly reduced or absent MEP amplitude",
    "corticospinal_tract": "Corticospinal Tract Lesion — prolonged latency + reduced amplitude",
    "postictal_motor": "Postictal Motor Deficit (Todd's Paresis) — transient asymmetric MEP abnormality",
    "cervical_myelopathy": "Cervical Myelopathy — upper limb CMCT prolonged, asymmetric findings",
}

SEVERITY_LEVELS = ["Normal", "Mild", "Moderate", "Severe"]

CSP_LOWER_MS = 100.0          # Cortical silent period minimum (ms)
INTER_SIDE_CMCT_UPPER = 2.5   # Inter-hemispheric CMCT asymmetry (ms)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _seed(patient_id: str, side: str, muscle: str, param: str) -> float:
    """Deterministic pseudo-random value [0,1) from patient+side+muscle+param."""
    h = hashlib.md5(f"{patient_id}:{side}:{muscle}:{param}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _get_patients():
    """Fetch real patients from clinical.db."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT p.patient_id, p.name, p.age, p.disease,
               COUNT(DISTINCT s.id)  AS seizure_count,
               COUNT(DISTINCT m.id)  AS med_count
        FROM   patients p
        LEFT JOIN seizure_diary s ON p.patient_id = s.patient_id
        LEFT JOIN medications   m ON p.patient_id = m.patient_id
        GROUP  BY p.patient_id
        ORDER  BY p.patient_id
        LIMIT  30
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def _abnormal_probability(patient, muscle_key: str) -> float:
    """Compute the probability that this patient's MEP is abnormal for a given muscle."""
    age = patient.get("age", 40) or 40
    disease = (patient.get("disease") or "").lower()
    med_count = patient.get("med_count", 0) or 0
    seizure_count = patient.get("seizure_count", 0) or 0
    muscle = MEP_MUSCLES[muscle_key]

    base = 0.08

    # Age effect: older → slower motor conduction
    if age > 70:
        base += 0.18
    elif age > 55:
        base += 0.10
    elif age > 40:
        base += 0.05

    # Disease-specific risk factors
    if "multiple sclerosis" in disease or " ms " in disease:
        base += 0.45          # CST demyelination
    if "stroke" in disease or "infarct" in disease or "hemiplegia" in disease:
        base += 0.40          # Cortical/subcortical lesion
    if "tumor" in disease or "glioma" in disease or "meningioma" in disease:
        base += 0.35          # Mass lesion
    if "cervical" in disease or "myelopathy" in disease or "spondylosis" in disease:
        base += 0.30          # Cervical spinal cord
    if "motor neuron" in disease or "als" in disease or "amyotrophic" in disease:
        base += 0.50          # UMN/LMN combined
    if "parkinson" in disease:
        base += 0.15          # Mild CMCT prolongation
    if "epilepsy" in disease:
        # Postictal Todd's paresis risk — higher with frequent seizures
        if seizure_count > 20:
            base += 0.20
        elif seizure_count > 5:
            base += 0.10
        if med_count > 3:
            base += 0.05      # Polypharmacy slows motor cortex excitability
    if "migraine" in disease:
        base += 0.08
    if muscle["limb"] == "Lower" and ("back" in disease or "lumbar" in disease):
        base += 0.20          # Lumbosacral pathology more likely in lower limb

    return min(base, 0.90)


def _generate_muscle_result(patient, side: str, muscle_key: str) -> dict:
    """Generate MEP result for one muscle on one side."""
    pid = patient["patient_id"]
    refs = MEP_MUSCLES[muscle_key]
    prob = _abnormal_probability(patient, muscle_key)
    s_abn = _seed(pid, side, muscle_key, "is_abn")

    is_abnormal = s_abn < prob

    s_val = _seed(pid, side, muscle_key, "val")
    s_sev = _seed(pid, side, muscle_key, "sev")

    if is_abnormal:
        if s_sev < 0.40:
            severity = "Mild"
            lat_factor   = 1 + 0.06 + s_val * 0.08   # 6-14% prolonged
            amp_factor   = 0.55 + s_val * 0.25        # 55-80% of lower bound → slightly reduced
            cmct_factor  = 1 + 0.08 + s_val * 0.10
        elif s_sev < 0.75:
            severity = "Moderate"
            lat_factor   = 1 + 0.14 + s_val * 0.14
            amp_factor   = 0.25 + s_val * 0.25
            cmct_factor  = 1 + 0.18 + s_val * 0.20
        else:
            severity = "Severe"
            lat_factor   = 1 + 0.30 + s_val * 0.30
            amp_factor   = max(0.02, s_val * 0.15)    # Near-absent to absent
            cmct_factor  = 1 + 0.35 + s_val * 0.40
    else:
        severity = "Normal"
        lat_factor   = 0.70 + s_val * 0.25            # 70-95% of upper bound
        amp_factor   = 1.5  + s_val * 6.0             # 1.5-7.5× lower bound (healthy range)
        cmct_factor  = 0.60 + s_val * 0.30            # 60-90% of upper bound

    latency_ms  = round(refs["latency_upper_ms"] * lat_factor, 1)
    amplitude_uv = round(refs["amplitude_lower_uv"] * amp_factor, 0)
    cmct_ms     = round(refs["cmct_upper_ms"] * cmct_factor, 1)
    csp_ms      = round(CSP_LOWER_MS * (0.60 + _seed(pid, side, muscle_key, "csp") * 0.80), 1)

    return {
        "side": side,
        "muscle": muscle_key,
        "muscle_name": refs["name"],
        "limb": refs["limb"],
        "latency_ms": latency_ms,
        "latency_ref_ms": refs["latency_upper_ms"],
        "latency_abnormal": latency_ms > refs["latency_upper_ms"],
        "amplitude_uv": amplitude_uv,
        "amplitude_ref_uv": refs["amplitude_lower_uv"],
        "amplitude_abnormal": amplitude_uv < refs["amplitude_lower_uv"],
        "cmct_ms": cmct_ms,
        "cmct_ref_ms": refs["cmct_upper_ms"],
        "cmct_abnormal": cmct_ms > refs["cmct_upper_ms"],
        "csp_ms": csp_ms,
        "csp_abnormal": csp_ms < CSP_LOWER_MS,
        "severity": severity,
    }


def _classify_pattern(study: dict) -> str:
    """Classify diagnostic pattern from muscle results."""
    if study["overall_severity"] == "Normal":
        return "normal"

    results = study["muscle_results"]

    any_cmct_abn   = any(r["cmct_abnormal"] for r in results)
    any_amp_abn    = any(r["amplitude_abnormal"] for r in results)
    any_lat_abn    = any(r["latency_abnormal"] for r in results)
    upper_abn      = any(r["cmct_abnormal"] for r in results if r["limb"] == "Upper")
    lower_abn      = any(r["cmct_abnormal"] for r in results if r["limb"] == "Lower")
    severe_amp     = any(r["severity"] == "Severe" and r["amplitude_abnormal"] for r in results)

    # Asymmetry: left vs right CMCT difference for any muscle
    cmct_by_side = {}
    for r in results:
        key = (r["muscle"], r["side"])
        cmct_by_side[key] = r["cmct_ms"]
    asym = False
    for m in MEP_MUSCLES:
        l_cmct = cmct_by_side.get((m, "Left"), 0)
        r_cmct = cmct_by_side.get((m, "Right"), 0)
        if abs(l_cmct - r_cmct) > INTER_SIDE_CMCT_UPPER:
            asym = True

    pid = study["patient_id"]
    disease = (study.get("disease") or "").lower()

    # Epilepsy with asymmetric, transient
    if "epilepsy" in disease and asym and not upper_abn and not lower_abn:
        return "postictal_motor"
    # Cortical lesion: markedly reduced/absent amplitude
    if severe_amp and not any_cmct_abn:
        return "cortical_lesion"
    # Cervical myelopathy: upper limb CMCT prolonged disproportionately
    if upper_abn and not lower_abn:
        return "cervical_myelopathy"
    # Corticospinal: both lat and amp abnormal
    if any_lat_abn and any_amp_abn:
        return "corticospinal_tract"
    # UMN: CMCT prolonged, amplitude preserved
    if any_cmct_abn and not any_amp_abn:
        return "upper_motor_neuron"
    # Fallback
    return "corticospinal_tract"


def _generate_mep_study(patient: dict) -> dict:
    """Generate a deterministic MEP study for a patient based on their clinical profile."""
    pid = patient["patient_id"]
    age = patient.get("age", 40) or 40
    disease = patient.get("disease", "Unknown")

    results = []
    for side in SIDES:
        for muscle in MEP_MUSCLES:
            results.append(_generate_muscle_result(patient, side, muscle))

    # Overall severity
    sev_order = {"Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
    max_sev_idx = max(sev_order.get(r["severity"], 0) for r in results)
    overall_severity = SEVERITY_LEVELS[max_sev_idx]
    abnormal_results = sum(1 for r in results if r["severity"] != "Normal")

    # Per-limb summary
    limb_abn = {}
    for r in results:
        key = r["limb"]
        limb_abn.setdefault(key, {"total": 0, "abnormal": 0})
        limb_abn[key]["total"] += 1
        if r["severity"] != "Normal":
            limb_abn[key]["abnormal"] += 1

    study = {
        "patient_id": pid,
        "patient_name": patient.get("name", pid),
        "age": age,
        "disease": disease,
        "muscle_results": results,
        "overall_severity": overall_severity,
        "abnormal_results": abnormal_results,
        "total_results": len(results),
        "limb_abnormality": limb_abn,
    }

    study["diagnostic_pattern"] = _classify_pattern(study)
    return study


def _get_all_studies() -> list:
    patients = _get_patients()
    return [_generate_mep_study(p) for p in patients]


# ── Public API ─────────────────────────────────────────────────────────────────

def overview() -> dict:
    """KPIs, severity distribution, diagnostic pattern distribution,
    per-limb abnormality rates, per-muscle mean parameters, per-patient summary."""
    studies = _get_all_studies()
    total = len(studies)
    abnormal_count = sum(1 for s in studies if s["overall_severity"] != "Normal")

    sev_dist    = Counter(s["overall_severity"] for s in studies)
    pattern_dist = Counter(s["diagnostic_pattern"] for s in studies)

    # Mean CMCT per muscle across all studies
    cmct_by_muscle = {m: [] for m in MEP_MUSCLES}
    lat_by_muscle  = {m: [] for m in MEP_MUSCLES}
    amp_by_muscle  = {m: [] for m in MEP_MUSCLES}
    for s in studies:
        for r in s["muscle_results"]:
            m = r["muscle"]
            cmct_by_muscle[m].append(r["cmct_ms"])
            lat_by_muscle[m].append(r["latency_ms"])
            amp_by_muscle[m].append(r["amplitude_uv"])

    muscle_summary = []
    for m, refs in MEP_MUSCLES.items():
        cmct_list = cmct_by_muscle[m]
        lat_list  = lat_by_muscle[m]
        amp_list  = amp_by_muscle[m]
        n = len(cmct_list) or 1
        muscle_summary.append({
            "muscle": m,
            "muscle_name": refs["name"],
            "limb": refs["limb"],
            "mean_cmct_ms": round(sum(cmct_list) / n, 1),
            "cmct_ref_ms": refs["cmct_upper_ms"],
            "mean_latency_ms": round(sum(lat_list) / n, 1),
            "latency_ref_ms": refs["latency_upper_ms"],
            "mean_amplitude_uv": round(sum(amp_list) / n, 0),
            "amplitude_ref_uv": refs["amplitude_lower_uv"],
        })

    # Per-limb abnormality
    limb_data: dict[str, dict] = {}
    for s in studies:
        for limb, v in s["limb_abnormality"].items():
            limb_data.setdefault(limb, {"abnormal": 0, "total": 0})
            limb_data[limb]["abnormal"] += v["abnormal"]
            limb_data[limb]["total"]    += v["total"]

    limb_rates = [
        {"limb": limb, "abnormal": v["abnormal"], "total": v["total"],
         "rate_pct": round(100 * v["abnormal"] / v["total"], 1) if v["total"] else 0}
        for limb, v in limb_data.items()
    ]

    # Per-patient summary
    patient_summary = sorted([
        {
            "patient_id": s["patient_id"],
            "name": s["patient_name"],
            "age": s["age"],
            "disease": s["disease"],
            "overall_severity": s["overall_severity"],
            "diagnostic_pattern": s["diagnostic_pattern"],
            "abnormal_results": s["abnormal_results"],
            "total_results": s["total_results"],
        }
        for s in studies
    ], key=lambda x: SEVERITY_LEVELS.index(x["overall_severity"]) if x["overall_severity"] in SEVERITY_LEVELS else 0, reverse=True)

    return {
        "kpis": {
            "total_studies": total,
            "abnormal_count": abnormal_count,
            "abnormal_rate_pct": round(100 * abnormal_count / total, 1) if total else 0,
            "muscles_tested": len(MEP_MUSCLES),
            "sides_tested": len(SIDES),
            "total_muscle_recordings": total * len(MEP_MUSCLES) * len(SIDES),
        },
        "severity_distribution": [
            {"severity": sev, "count": sev_dist.get(sev, 0)}
            for sev in SEVERITY_LEVELS
        ],
        "pattern_distribution": [
            {
                "pattern": p,
                "label": DIAGNOSTIC_PATTERNS[p].split(" \u2014 ")[0],
                "count": pattern_dist.get(p, 0),
            }
            for p in DIAGNOSTIC_PATTERNS
        ],
        "limb_abnormality_rates": limb_rates,
        "muscle_summary": muscle_summary,
        "patient_summary": patient_summary,
    }


def breakdown() -> dict:
    """Per-muscle CMCT distributions, latency histograms, amplitude histograms,
    left vs right comparison, per-patient detailed results."""
    studies = _get_all_studies()

    # Histograms: CMCT, latency, amplitude per muscle
    cmct_hist:  dict[str, list] = {m: [] for m in MEP_MUSCLES}
    lat_hist:   dict[str, list] = {m: [] for m in MEP_MUSCLES}
    amp_hist:   dict[str, list] = {m: [] for m in MEP_MUSCLES}

    for s in studies:
        for r in s["muscle_results"]:
            m = r["muscle"]
            cmct_hist[m].append(round(r["cmct_ms"], 1))
            lat_hist[m].append(r["latency_ms"])
            amp_hist[m].append(r["amplitude_uv"])

    def _hist_bins(vals: list, n_bins: int = 10) -> list:
        if not vals:
            return []
        lo, hi = min(vals), max(vals)
        if lo == hi:
            return [{"bin": str(lo), "count": len(vals)}]
        step = (hi - lo) / n_bins
        bins: dict[str, int] = {}
        for v in vals:
            b = round(lo + step * int((v - lo) / step + 0.5 if (v - lo) / step < n_bins else n_bins - 0.5), 1)
            bins[str(b)] = bins.get(str(b), 0) + 1
        return [{"bin": k, "count": v} for k, v in sorted(bins.items(), key=lambda x: float(x[0]))]

    muscle_histograms = {}
    for m in MEP_MUSCLES:
        refs = MEP_MUSCLES[m]
        muscle_histograms[m] = {
            "muscle_name": refs["name"],
            "limb": refs["limb"],
            "cmct_histogram": _hist_bins(cmct_hist[m]),
            "cmct_ref": refs["cmct_upper_ms"],
            "latency_histogram": _hist_bins(lat_hist[m]),
            "latency_ref": refs["latency_upper_ms"],
            "amplitude_histogram": _hist_bins(amp_hist[m]),
            "amplitude_ref": refs["amplitude_lower_uv"],
        }

    # Left vs right comparison per muscle
    side_comparison = []
    for m, refs in MEP_MUSCLES.items():
        for side in SIDES:
            vals = [r for s in studies for r in s["muscle_results"] if r["muscle"] == m and r["side"] == side]
            n = len(vals) or 1
            side_comparison.append({
                "muscle": m,
                "side": side,
                "limb": refs["limb"],
                "mean_cmct_ms": round(sum(r["cmct_ms"] for r in vals) / n, 1),
                "mean_latency_ms": round(sum(r["latency_ms"] for r in vals) / n, 1),
                "mean_amplitude_uv": round(sum(r["amplitude_uv"] for r in vals) / n, 0),
                "abnormal_count": sum(1 for r in vals if r["severity"] != "Normal"),
                "total": n,
                "abnormal_rate_pct": round(100 * sum(1 for r in vals if r["severity"] != "Normal") / n, 1),
            })

    # Per-patient detailed breakdown
    per_patient = []
    for s in studies:
        # Group results by muscle across sides
        by_muscle = {}
        for r in s["muscle_results"]:
            by_muscle.setdefault(r["muscle"], []).append(r)
        muscles_detail = []
        for m, rs in by_muscle.items():
            muscles_detail.append({
                "muscle": m,
                "muscle_name": MEP_MUSCLES[m]["name"],
                "limb": MEP_MUSCLES[m]["limb"],
                "sides": [
                    {
                        "side": r["side"],
                        "latency_ms": r["latency_ms"],
                        "amplitude_uv": r["amplitude_uv"],
                        "cmct_ms": r["cmct_ms"],
                        "csp_ms": r["csp_ms"],
                        "severity": r["severity"],
                        "latency_abnormal": r["latency_abnormal"],
                        "amplitude_abnormal": r["amplitude_abnormal"],
                        "cmct_abnormal": r["cmct_abnormal"],
                    }
                    for r in sorted(rs, key=lambda x: x["side"])
                ],
            })
        per_patient.append({
            "patient_id": s["patient_id"],
            "name": s["patient_name"],
            "age": s["age"],
            "disease": s["disease"],
            "overall_severity": s["overall_severity"],
            "diagnostic_pattern": s["diagnostic_pattern"],
            "muscles": muscles_detail,
        })

    per_patient.sort(
        key=lambda x: SEVERITY_LEVELS.index(x["overall_severity"]) if x["overall_severity"] in SEVERITY_LEVELS else 0,
        reverse=True,
    )

    return {
        "muscle_histograms": muscle_histograms,
        "side_comparison": side_comparison,
        "per_patient": per_patient,
    }


def definitions() -> dict:
    """Clinical definitions, normative references, diagnostic criteria, terminology."""
    return {
        "title": "Motor Evoked Potentials (MEP) — Clinical Reference",
        "overview": (
            "MEP testing uses single-pulse Transcranial Magnetic Stimulation (TMS) "
            "over the primary motor cortex to activate the corticospinal tract and "
            "record the resulting compound muscle action potential (CMAP). "
            "The Central Motor Conduction Time (CMCT), calculated as MEP latency minus "
            "peripheral motor conduction time, is the primary diagnostic index."
        ),
        "technique": {
            "stimulation": "Figure-of-eight coil over M1 (hand area: 5cm lateral, 2cm anterior to vertex)",
            "recording_muscles": [
                {"muscle": "APB", "name": "Abductor Pollicis Brevis", "limb": "Upper limb",
                 "note": "Most reliable upper limb MEP target; C8/T1 innervation"},
                {"muscle": "ADM", "name": "Abductor Digiti Minimi", "limb": "Upper limb",
                 "note": "Alternative upper limb target; ulnar nerve (C8/T1)"},
                {"muscle": "TA", "name": "Tibialis Anterior", "limb": "Lower limb",
                 "note": "Primary lower limb MEP target; L4/L5 innervation"},
                {"muscle": "AH", "name": "Abductor Hallucis", "limb": "Lower limb",
                 "note": "Alternative lower limb target; S1 innervation"},
            ],
            "patient_prep": "Slight voluntary pre-activation (5-10% MVC) enhances MEP recruitment",
            "facilitation": "Jendrassik manoeuvre for lower limb MEPs",
        },
        "normative_values": [
            {
                "parameter": "APB MEP Latency", "upper_limit_ms": 25.0,
                "note": "Onset latency from TMS pulse to MEP; includes corticospinal + neuromuscular delay",
            },
            {
                "parameter": "ADM MEP Latency", "upper_limit_ms": 27.0,
                "note": "Slightly longer than APB due to ulnar nerve anatomy",
            },
            {
                "parameter": "TA MEP Latency", "upper_limit_ms": 45.0,
                "note": "Lower limb latencies are substantially longer due to cord length",
            },
            {
                "parameter": "AH MEP Latency", "upper_limit_ms": 50.0,
                "note": "Most distal lower limb target; S1 root exit adds delay",
            },
            {
                "parameter": "Upper Limb CMCT", "upper_limit_ms": 8.0,
                "note": "CMCT = MEP latency − F-wave half-latency correction; key CST index",
            },
            {
                "parameter": "Lower Limb CMCT", "upper_limit_ms": 20.0,
                "note": "Lower limb CMCT is less reliable; magnetic conus stimulation preferred",
            },
            {
                "parameter": "MEP Amplitude (APB)", "lower_limit_uv": 200.0,
                "note": "Peak-to-peak amplitude; amplitude ratio ≥20% of CMAP is an alternative norm",
            },
            {
                "parameter": "Cortical Silent Period (CSP)", "lower_limit_ms": 100.0,
                "note": "Duration of motor inhibition following suprathreshold TMS; GABA-B mediated",
            },
            {
                "parameter": "Inter-side CMCT asymmetry", "upper_limit_ms": 2.5,
                "note": "CMCT difference between hemispheres; >2.5 ms indicates asymmetric pathology",
            },
        ],
        "diagnostic_patterns": [
            {
                "pattern": "normal",
                "label": "Normal",
                "description": "All MEP latencies, CMCT, amplitude, and CSP within reference ranges bilaterally.",
                "clinical_note": "Normal MEP does not exclude mild corticospinal pathology.",
            },
            {
                "pattern": "upper_motor_neuron",
                "label": "Upper Motor Neuron Dysfunction",
                "description": "Prolonged CMCT with preserved or mildly reduced MEP amplitude.",
                "clinical_note": "Typical of early MS, ALS, or subcortical white matter lesions.",
            },
            {
                "pattern": "cortical_lesion",
                "label": "Cortical Lesion",
                "description": "Markedly reduced or absent MEP amplitude, CMCT may be normal.",
                "clinical_note": "Seen with focal cortical infarcts, tumors, or severe postictal depression.",
            },
            {
                "pattern": "corticospinal_tract",
                "label": "Corticospinal Tract Lesion",
                "description": "Both prolonged latency/CMCT and reduced amplitude — indicates axonal + myelin loss.",
                "clinical_note": "Progressive MS, ALS with mixed UMN/LMN, or hemispheric infarcts.",
            },
            {
                "pattern": "postictal_motor",
                "label": "Postictal Motor Deficit (Todd's Paresis)",
                "description": "Transient asymmetric MEP abnormality post-seizure; recovers within hours.",
                "clinical_note": "Asymmetric CMCT or amplitude reduction identifies focal seizure origin hemisphere.",
            },
            {
                "pattern": "cervical_myelopathy",
                "label": "Cervical Myelopathy",
                "description": "Upper limb CMCT disproportionately prolonged; lower limb less affected.",
                "clinical_note": "C3-C6 disc disease; correlate with cervical spine imaging.",
            },
        ],
        "epilepsy_relevance": [
            {
                "topic": "Postictal (Todd's) Paresis Mapping",
                "description": "MEP asymmetry after focal seizure identifies the seizure-onset hemisphere — useful when ictal EEG is non-localizing.",
                "citation": "Werhahn KJ, et al. Ann Neurol. 2000.",
            },
            {
                "topic": "Pre-Surgical Motor Cortex Mapping",
                "description": "Single-pulse TMS maps the motor cortex hotspot prior to epilepsy surgery near the rolandic cortex; identifies safe resection margins.",
                "citation": "Rossini PM, et al. Clin Neurophysiol. 2015.",
            },
            {
                "topic": "AED Effects on Motor Excitability",
                "description": "Carbamazepine, valproate, and levetiracetam modulate cortical excitability; serial MEP monitoring tracks pharmacodynamic effects.",
                "citation": "Manganotti P, et al. Epilepsia. 2004.",
            },
            {
                "topic": "Motor Threshold & Seizure Risk",
                "description": "Low resting motor threshold (high excitability) correlates with lower seizure threshold in idiopathic generalized epilepsy.",
                "citation": "Badawy RA, et al. Neurology. 2009.",
            },
        ],
        "safety_contraindications": [
            "Active cochlear implant or deep brain stimulator",
            "Cardiac pacemaker within 15 cm of coil",
            "Metal in skull/spine (clips, plates) within TMS field",
            "History of unprovoked seizures without AED coverage (relative CI — risk-benefit discussion)",
            "Pregnancy (first trimester — insufficient safety data)",
        ],
        "references": [
            "Rossini PM, et al. Non-invasive electrical and magnetic stimulation of the brain, spinal cord, roots and peripheral nerves. Clin Neurophysiol. 2015;126(6):1071-1107.",
            "Chen R, et al. Safety of different inter-train intervals for repetitive TMS and recommendations for safe ranges of stimulation parameters. Electroencephalogr Clin Neurophysiol. 1997.",
            "Hallett M. Transcranial magnetic stimulation: A primer. Neuron. 2007;55(2):187-199.",
            "Lefaucheur JP, et al. Evidence-based guidelines on the therapeutic use of repetitive TMS. Clin Neurophysiol. 2020.",
        ],
    }
