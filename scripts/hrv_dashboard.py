"""
HRV / RR Variation Dashboard
Heart Rate Variability time/frequency domain features + autonomic dysfunction scoring.

Registry item: RR_VAR
Advancement: HRV time/freq features + autonomic dysfunction score
Biomarkers: SDNN, RMSSD, LF/HF
AI Models: TS-Transformer, XGBoost
Standards: Task Force of ESC/NASPE, Eur Heart J 1996
"""

import hashlib
import json
import math
import os
import sqlite3

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB_PATH = os.path.join(_BASE_DIR, "data", "clinical.db")

# ---------------------------------------------------------------------------
# Reference ranges — Task Force of ESC/NASPE 1996 (Eur Heart J)
# ---------------------------------------------------------------------------
REFERENCE_RANGES = {
    "mean_rr_ms":      {"low": 700,   "high": 1100,  "unit": "ms",   "label": "Mean RR Interval"},
    "sdnn_ms":         {"low": 100,   "high": 200,   "unit": "ms",   "label": "SDNN"},
    "rmssd_ms":        {"low": 20,    "high": 75,    "unit": "ms",   "label": "RMSSD"},
    "pnn50_pct":       {"low": 5,     "high": 40,    "unit": "%",    "label": "pNN50"},
    "hr_mean_bpm":     {"low": 60,    "high": 85,    "unit": "bpm",  "label": "Mean Heart Rate"},
    "vlf_power_ms2":   {"low": 300,   "high": 1500,  "unit": "ms²",  "label": "VLF Power"},
    "lf_power_ms2":    {"low": 750,   "high": 2500,  "unit": "ms²",  "label": "LF Power"},
    "hf_power_ms2":    {"low": 250,   "high": 1500,  "unit": "ms²",  "label": "HF Power"},
    "lf_hf_ratio":     {"low": 1.0,   "high": 3.0,   "unit": "",     "label": "LF/HF Ratio"},
    "total_power_ms2": {"low": 1500,  "high": 5000,  "unit": "ms²",  "label": "Total Power"},
}

DIAGNOSTIC_PATTERNS = {
    "normal": {
        "label": "Normal Autonomic Function",
        "description": (
            "SDNN, RMSSD, and LF/HF ratio all within normal limits. "
            "Balanced sympathovagal tone with no evidence of autonomic dysfunction."
        ),
    },
    "sympathetic_dominance": {
        "label": "Sympathetic Dominance",
        "description": (
            "Elevated LF/HF ratio (>3.0) with reduced HF power. "
            "Indicates sympathetic overdrive — commonly seen in anxiety, hypertension, "
            "heart failure, and sleep disorders."
        ),
    },
    "parasympathetic_dominance": {
        "label": "Parasympathetic Dominance",
        "description": (
            "LF/HF ratio <0.5 with elevated HF power. "
            "Seen in highly trained athletes, vagotonia, or early diabetic autonomic neuropathy "
            "with selective sympathetic loss."
        ),
    },
    "autonomic_neuropathy": {
        "label": "Autonomic Neuropathy",
        "description": (
            "Globally reduced HRV (SDNN <50 ms, RMSSD <20 ms, Total power <1500 ms²). "
            "Loss of both sympathetic and parasympathetic modulation. "
            "Classic diabetic autonomic neuropathy or Parkinson's disease."
        ),
    },
    "cardiac_risk": {
        "label": "Elevated Cardiac Risk",
        "description": (
            "SDNN <100 ms with depressed total power — established predictor of "
            "all-cause and cardiovascular mortality (Task Force 1996). "
            "Associated with post-MI risk and SUDEP in epilepsy."
        ),
    },
    "chronotropic_incompetence": {
        "label": "Chronotropic Incompetence",
        "description": (
            "Blunted HR range (HR max − HR min <20 bpm) with reduced SDNN. "
            "Inability to appropriately modulate heart rate; seen in severe autonomic failure, "
            "beta-blocker therapy, and advanced cardiac disease."
        ),
    },
}

SEVERITY_LEVELS = {
    "Normal":   {"score_max": 20,  "color": "green"},
    "Mild":     {"score_max": 45,  "color": "yellow"},
    "Moderate": {"score_max": 70,  "color": "orange"},
    "Severe":   {"score_max": 100, "color": "red"},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_patients():
    """Read up to 30 patients from clinical.db — identical query to BERA dashboard."""
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            p.patient_id,
            p.name,
            p.age,
            p.disease,
            COUNT(DISTINCT s.id)  AS seizure_count,
            COUNT(DISTINCT m.id)  AS med_count
        FROM patients p
        LEFT JOIN seizure_diary s ON s.patient_id = p.patient_id
        LEFT JOIN medications m   ON m.patient_id = p.patient_id
        GROUP BY p.patient_id
        ORDER BY p.patient_id
        LIMIT 30
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def _seed(patient_id: int, domain: str, param: str) -> float:
    """Deterministic pseudo-random float in [0, 1) using MD5."""
    key = f"{patient_id}:{domain}:{param}"
    digest = hashlib.md5(key.encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * t


def _abnormality_probability(patient: dict) -> float:
    """Disease/age-based probability of HRV abnormality."""
    disease = (patient.get("disease") or "").lower()
    age = patient.get("age") or 0
    prob = 0.10  # baseline

    if any(k in disease for k in ("epilep", "seizure")):
        prob += 0.20   # anticonvulsants affect ANS; SUDEP risk
    if any(k in disease for k in ("diabet", "neuropath")):
        prob += 0.40   # diabetic autonomic neuropathy
    if any(k in disease for k in ("stroke", "cardiovasc", "cardiac", "heart")):
        prob += 0.35
    if any(k in disease for k in ("parkinson",)):
        prob += 0.30
    if any(k in disease for k in ("multiple sclerosis", " ms ")):
        prob += 0.25
    if any(k in disease for k in ("anxiety", "depression", "depress")):
        prob += 0.15   # sympathetic overdrive
    if age > 60:
        prob += 0.20
    elif age > 40:
        prob += 0.08

    return min(prob, 0.90)


def _classify_pattern(hrv: dict) -> str:
    """Classify autonomic pattern from HRV metrics."""
    sdnn       = hrv["sdnn_ms"]
    rmssd      = hrv["rmssd_ms"]
    lf_hf      = hrv["lf_hf_ratio"]
    total_pwr  = hrv["total_power_ms2"]
    hr_min     = hrv["hr_min_bpm"]
    hr_max     = hrv["hr_max_bpm"]

    # Globally depressed — autonomic neuropathy
    if sdnn < 50 and rmssd < 20 and total_pwr < 1500:
        return "autonomic_neuropathy"

    # Cardiac risk threshold (Task Force 1996: SDNN <100 ms)
    if sdnn < 100 and total_pwr < 1500:
        return "cardiac_risk"

    # Chronotropic incompetence
    if (hr_max - hr_min) < 20 and sdnn < 100:
        return "chronotropic_incompetence"

    # Sympathetic dominance
    if lf_hf > 3.0:
        return "sympathetic_dominance"

    # Parasympathetic dominance
    if lf_hf < 0.5:
        return "parasympathetic_dominance"

    return "normal"


def _severity_from_score(score: float) -> str:
    if score <= 20:
        return "Normal"
    elif score <= 45:
        return "Mild"
    elif score <= 70:
        return "Moderate"
    return "Severe"


def _compute_autonomic_score(hrv: dict) -> float:
    """
    Composite autonomic dysfunction score 0–100.
    Each parameter contributes proportionally to deviation from normal range.
    """
    penalties = []

    def _pen(val, lo, hi, weight=1.0):
        """Penalty in [0,1] for being outside [lo, hi]."""
        if val < lo:
            return weight * min(1.0, (lo - val) / lo)
        if val > hi:
            return weight * min(1.0, (val - hi) / hi)
        return 0.0

    penalties.append(_pen(hrv["sdnn_ms"],         100, 200, weight=2.0))
    penalties.append(_pen(hrv["rmssd_ms"],          20,  75, weight=1.5))
    penalties.append(_pen(hrv["pnn50_pct"],          5,  40, weight=1.0))
    penalties.append(_pen(hrv["mean_rr_ms"],        700, 1100, weight=1.0))
    penalties.append(_pen(hrv["lf_power_ms2"],      750, 2500, weight=1.0))
    penalties.append(_pen(hrv["hf_power_ms2"],      250, 1500, weight=1.0))
    penalties.append(_pen(hrv["lf_hf_ratio"],       1.0,  3.0, weight=1.5))
    penalties.append(_pen(hrv["total_power_ms2"],  1500, 5000, weight=2.0))

    total_weight = 2.0 + 1.5 + 1.0 + 1.0 + 1.0 + 1.0 + 1.5 + 2.0  # 11.0
    raw = sum(penalties)
    score = min(100.0, raw / total_weight * 100)
    return round(score, 1)


# ---------------------------------------------------------------------------
# HRV study generation
# ---------------------------------------------------------------------------

def _generate_hrv_study(patient: dict) -> dict:
    """Generate a deterministic HRV study for one patient."""
    pid = patient["patient_id"]
    prob = _abnormality_probability(patient)
    s = _seed(pid, "hrv", "is_abnormal")
    is_abnormal = s < prob

    def rnd(param, lo, hi):
        return _seed(pid, "hrv", param) * (hi - lo) + lo

    if is_abnormal:
        # Depressed / dysregulated HRV
        sdnn       = rnd("sdnn",   30,  99)
        rmssd      = rnd("rmssd",   8,  35)
        pnn50      = rnd("pnn50",   0,  10)
        mean_rr    = rnd("mean_rr", 600, 900)
        hr_mean    = round(60000 / mean_rr, 1)
        hr_min     = round(hr_mean - rnd("hr_range_lo", 2, 12), 1)
        hr_max     = round(hr_mean + rnd("hr_range_hi", 3, 18), 1)
        vlf_power  = rnd("vlf",  100,  700)
        lf_power   = rnd("lf",   150,  749)
        hf_power   = rnd("hf",    50,  249)
        # Bias toward sympathetic dominance for abnormal cases
        if _seed(pid, "hrv", "lf_bias") > 0.4:
            lf_power = rnd("lf2",  800, 3500)
            hf_power = rnd("hf2",   50,  300)
        lf_hf      = round(lf_power / max(hf_power, 1), 2)
        total_pwr  = vlf_power + lf_power + hf_power
    else:
        # Normal HRV
        sdnn       = rnd("sdnn",  100,  200)
        rmssd      = rnd("rmssd",  20,   75)
        pnn50      = rnd("pnn50",   5,   40)
        mean_rr    = rnd("mean_rr", 700, 1100)
        hr_mean    = round(60000 / mean_rr, 1)
        hr_min     = round(hr_mean - rnd("hr_range_lo", 8, 20), 1)
        hr_max     = round(hr_mean + rnd("hr_range_hi", 10, 30), 1)
        vlf_power  = rnd("vlf",   300, 1500)
        lf_power   = rnd("lf",    750, 2500)
        hf_power   = rnd("hf",    250, 1500)
        lf_hf      = round(lf_power / max(hf_power, 1), 2)
        total_pwr  = vlf_power + lf_power + hf_power

    hrv = {
        # Time domain
        "mean_rr_ms":       round(mean_rr, 1),
        "sdnn_ms":          round(sdnn, 1),
        "rmssd_ms":         round(rmssd, 1),
        "pnn50_pct":        round(pnn50, 1),
        "hr_mean_bpm":      hr_mean,
        "hr_min_bpm":       hr_min,
        "hr_max_bpm":       hr_max,
        # Frequency domain
        "vlf_power_ms2":    round(vlf_power, 1),
        "lf_power_ms2":     round(lf_power, 1),
        "hf_power_ms2":     round(hf_power, 1),
        "lf_hf_ratio":      lf_hf,
        "total_power_ms2":  round(total_pwr, 1),
    }

    hrv["autonomic_score"]   = _compute_autonomic_score(hrv)
    hrv["diagnostic_pattern"] = _classify_pattern(hrv)
    hrv["severity"]           = _severity_from_score(hrv["autonomic_score"])
    hrv["is_abnormal"]        = hrv["severity"] != "Normal"
    hrv["recording_type"]     = "5-min resting" if _seed(pid, "hrv", "rec_type") > 0.3 else "24h Holter"

    return hrv


def _flag(val, ref_key):
    """Return 'normal', 'low', or 'high' relative to reference range."""
    lo = REFERENCE_RANGES[ref_key]["low"]
    hi = REFERENCE_RANGES[ref_key]["high"]
    if val < lo:
        return "low"
    if val > hi:
        return "high"
    return "normal"


def _all_studies():
    """Return list of (patient, hrv_study) dicts for all patients."""
    patients = _get_patients()
    results = []
    for p in patients:
        hrv = _generate_hrv_study(p)
        results.append({"patient": p, "hrv": hrv})
    return results


# ---------------------------------------------------------------------------
# Public API — overview()
# ---------------------------------------------------------------------------

def overview() -> dict:
    """
    KPI cards, severity distribution, pattern distribution, patient summary table.
    Mirrors the BERA overview() structure.
    """
    studies = _all_studies()

    total      = len(studies)
    abnormals  = [s for s in studies if s["hrv"]["is_abnormal"]]
    abn_count  = len(abnormals)
    abn_rate   = round(abn_count / total * 100, 1) if total else 0

    sdnn_vals   = [s["hrv"]["sdnn_ms"]    for s in studies]
    rmssd_vals  = [s["hrv"]["rmssd_ms"]   for s in studies]
    lf_hf_vals  = [s["hrv"]["lf_hf_ratio"] for s in studies]

    mean_sdnn  = round(sum(sdnn_vals)  / len(sdnn_vals),  1)
    mean_rmssd = round(sum(rmssd_vals) / len(rmssd_vals), 1)
    mean_lf_hf = round(sum(lf_hf_vals) / len(lf_hf_vals), 2)

    # Severity distribution
    sev_dist = {"Normal": 0, "Mild": 0, "Moderate": 0, "Severe": 0}
    for s in studies:
        sev_dist[s["hrv"]["severity"]] += 1

    # Pattern distribution
    pattern_dist = {k: 0 for k in DIAGNOSTIC_PATTERNS}
    for s in studies:
        pat = s["hrv"]["diagnostic_pattern"]
        pattern_dist[pat] = pattern_dist.get(pat, 0) + 1

    # Patient summary table
    patient_summary = []
    for s in studies:
        p, hrv = s["patient"], s["hrv"]
        patient_summary.append({
            "patient_id":         p["patient_id"],
            "name":               p["name"],
            "age":                p["age"],
            "disease":            p["disease"],
            "severity":           hrv["severity"],
            "diagnostic_pattern": hrv["diagnostic_pattern"],
            "pattern_label":      DIAGNOSTIC_PATTERNS[hrv["diagnostic_pattern"]]["label"],
            "autonomic_score":    hrv["autonomic_score"],
            "sdnn_ms":            hrv["sdnn_ms"],
            "rmssd_ms":           hrv["rmssd_ms"],
            "lf_hf_ratio":        hrv["lf_hf_ratio"],
            "is_abnormal":        hrv["is_abnormal"],
            "recording_type":     hrv["recording_type"],
        })

    return {
        "kpis": {
            "total_studies":       total,
            "abnormal_count":      abn_count,
            "abnormal_rate_pct":   abn_rate,
            "mean_sdnn_ms":        mean_sdnn,
            "mean_rmssd_ms":       mean_rmssd,
            "mean_lf_hf_ratio":    mean_lf_hf,
        },
        "severity_distribution": sev_dist,
        "pattern_distribution":  pattern_dist,
        "patient_summary":       patient_summary,
    }


# ---------------------------------------------------------------------------
# Public API — breakdown()
# ---------------------------------------------------------------------------

def breakdown() -> dict:
    """
    Parameter summary tables, histograms, per-patient detail cards.
    Mirrors the BERA breakdown() structure.
    """
    studies = _all_studies()

    # ---- Time domain summary ----
    time_params = [
        ("mean_rr_ms",  "mean_rr_ms"),
        ("sdnn_ms",     "sdnn_ms"),
        ("rmssd_ms",    "rmssd_ms"),
        ("pnn50_pct",   "pnn50_pct"),
        ("hr_mean_bpm", "hr_mean_bpm"),
    ]
    time_domain_summary = []
    for param, ref_key in time_params:
        vals = [s["hrv"][param] for s in studies]
        ref  = REFERENCE_RANGES.get(ref_key, {})
        n_abn = sum(1 for v in vals if v < ref.get("low", -1e9) or v > ref.get("high", 1e9))
        time_domain_summary.append({
            "parameter":    ref.get("label", param),
            "param_key":    param,
            "mean":         round(sum(vals) / len(vals), 2),
            "min":          round(min(vals), 2),
            "max":          round(max(vals), 2),
            "unit":         ref.get("unit", ""),
            "ref_low":      ref.get("low"),
            "ref_high":     ref.get("high"),
            "abnormal_n":   n_abn,
            "abnormal_pct": round(n_abn / len(vals) * 100, 1),
        })

    # ---- Frequency domain summary ----
    freq_params = [
        ("vlf_power_ms2",  "vlf_power_ms2"),
        ("lf_power_ms2",   "lf_power_ms2"),
        ("hf_power_ms2",   "hf_power_ms2"),
        ("lf_hf_ratio",    "lf_hf_ratio"),
        ("total_power_ms2","total_power_ms2"),
    ]
    freq_domain_summary = []
    for param, ref_key in freq_params:
        vals = [s["hrv"][param] for s in studies]
        ref  = REFERENCE_RANGES.get(ref_key, {})
        n_abn = sum(1 for v in vals if v < ref.get("low", -1e9) or v > ref.get("high", 1e9))
        freq_domain_summary.append({
            "parameter":    ref.get("label", param),
            "param_key":    param,
            "mean":         round(sum(vals) / len(vals), 2),
            "min":          round(min(vals), 2),
            "max":          round(max(vals), 2),
            "unit":         ref.get("unit", ""),
            "ref_low":      ref.get("low"),
            "ref_high":     ref.get("high"),
            "abnormal_n":   n_abn,
            "abnormal_pct": round(n_abn / len(vals) * 100, 1),
        })

    # ---- Histograms ----
    def _histogram(values, n_bins=10, lo=None, hi=None):
        lo = lo if lo is not None else min(values)
        hi = hi if hi is not None else max(values)
        width = (hi - lo) / n_bins if hi > lo else 1
        bins = [{"bin_start": round(lo + i * width, 2),
                 "bin_end":   round(lo + (i + 1) * width, 2),
                 "count":     0} for i in range(n_bins)]
        for v in values:
            idx = min(int((v - lo) / width), n_bins - 1)
            if 0 <= idx < n_bins:
                bins[idx]["count"] += 1
        return bins

    sdnn_hist  = _histogram([s["hrv"]["sdnn_ms"]     for s in studies], lo=0,   hi=220)
    rmssd_hist = _histogram([s["hrv"]["rmssd_ms"]    for s in studies], lo=0,   hi=100)
    lf_hf_hist = _histogram([s["hrv"]["lf_hf_ratio"] for s in studies], lo=0.0, hi=6.0)
    score_hist = _histogram([s["hrv"]["autonomic_score"] for s in studies], lo=0, hi=100)

    # ---- Per-patient detail cards ----
    detail_cards = []
    for s in studies:
        p, hrv = s["patient"], s["hrv"]

        time_domain_detail = {
            k: {
                "value": hrv[k],
                "unit":  REFERENCE_RANGES.get(k, {}).get("unit", ""),
                "flag":  _flag(hrv[k], k) if k in REFERENCE_RANGES else "normal",
                "ref_low":  REFERENCE_RANGES.get(k, {}).get("low"),
                "ref_high": REFERENCE_RANGES.get(k, {}).get("high"),
            }
            for k in ["mean_rr_ms", "sdnn_ms", "rmssd_ms", "pnn50_pct", "hr_mean_bpm"]
        }
        # hr_min_bpm / hr_max_bpm are derived — no reference range key
        time_domain_detail["hr_min_bpm"] = {"value": hrv["hr_min_bpm"], "unit": "bpm", "flag": "info"}
        time_domain_detail["hr_max_bpm"] = {"value": hrv["hr_max_bpm"], "unit": "bpm", "flag": "info"}

        freq_domain_detail = {
            k: {
                "value": hrv[k],
                "unit":  REFERENCE_RANGES.get(k, {}).get("unit", ""),
                "flag":  _flag(hrv[k], k) if k in REFERENCE_RANGES else "normal",
                "ref_low":  REFERENCE_RANGES.get(k, {}).get("low"),
                "ref_high": REFERENCE_RANGES.get(k, {}).get("high"),
            }
            for k in ["vlf_power_ms2", "lf_power_ms2", "hf_power_ms2",
                      "lf_hf_ratio", "total_power_ms2"]
        }

        detail_cards.append({
            "patient_id":          p["patient_id"],
            "name":                p["name"],
            "age":                 p["age"],
            "disease":             p["disease"],
            "seizure_count":       p["seizure_count"],
            "med_count":           p["med_count"],
            "recording_type":      hrv["recording_type"],
            "severity":            hrv["severity"],
            "diagnostic_pattern":  hrv["diagnostic_pattern"],
            "pattern_label":       DIAGNOSTIC_PATTERNS[hrv["diagnostic_pattern"]]["label"],
            "autonomic_score":     hrv["autonomic_score"],
            "is_abnormal":         hrv["is_abnormal"],
            "time_domain":         time_domain_detail,
            "frequency_domain":    freq_domain_detail,
        })

    return {
        "time_domain_summary":   time_domain_summary,
        "freq_domain_summary":   freq_domain_summary,
        "sdnn_histogram":        sdnn_hist,
        "rmssd_histogram":       rmssd_hist,
        "lf_hf_ratio_histogram": lf_hf_hist,
        "autonomic_score_histogram": score_hist,
        "patient_detail_cards":  detail_cards,
    }


# ---------------------------------------------------------------------------
# Public API — definitions()
# ---------------------------------------------------------------------------

def definitions() -> dict:
    """
    Protocol, parameters, reference ranges, diagnostic patterns, severity levels,
    and clinical significance.  Mirrors the BERA definitions() structure.
    """
    return {
        "test_name": "Heart Rate Variability (HRV) / RR Variation Analysis",
        "code": "RR_VAR",
        "protocol": {
            "description": (
                "HRV analysis quantifies beat-to-beat fluctuations in the RR interval "
                "(time between successive R-peaks of the ECG). Two recording paradigms are used: "
                "(1) Short-term 5-minute resting ECG — standardized supine position, controlled breathing, "
                "sufficient for time and frequency domain analysis. "
                "(2) 24-hour ambulatory Holter ECG — captures circadian variation, total power, and VLF. "
                "Beat detection uses Pan-Tompkins QRS algorithm. Ectopic beats are removed by "
                "interpolation. Frequency domain analysis uses FFT or autoregressive modelling."
            ),
            "recording_types": [
                "5-min resting ECG (standard)",
                "24h Holter ECG (gold standard for total power and VLF)",
            ],
            "standard": "Task Force of the European Society of Cardiology and the North American "
                        "Society of Pacing and Electrophysiology. Heart rate variability. "
                        "Eur Heart J. 1996;17(3):354-381.",
            "ai_models": ["TS-Transformer", "XGBoost"],
            "biomarkers": ["SDNN", "RMSSD", "LF/HF ratio"],
        },
        "parameters": {
            "time_domain": [
                {
                    "key":         "mean_rr_ms",
                    "label":       "Mean RR Interval",
                    "unit":        "ms",
                    "description": "Average time between successive R-peaks. Reciprocal of mean heart rate.",
                    "ref_low":     700,
                    "ref_high":    1100,
                    "clinical":    "Mean RR <700 ms (HR >85 bpm) may indicate tachycardia or sympathetic activation.",
                },
                {
                    "key":         "sdnn_ms",
                    "label":       "SDNN",
                    "unit":        "ms",
                    "description": (
                        "Standard deviation of all NN intervals. Reflects overall HRV from "
                        "all cyclic components. Index of total autonomic modulation."
                    ),
                    "ref_low":     100,
                    "ref_high":    200,
                    "clinical":    (
                        "SDNN <100 ms indicates compromised autonomic regulation. "
                        "SDNN <50 ms is associated with significantly increased cardiovascular mortality risk "
                        "(Task Force 1996). Key predictor of SUDEP in epilepsy."
                    ),
                },
                {
                    "key":         "rmssd_ms",
                    "label":       "RMSSD",
                    "unit":        "ms",
                    "description": (
                        "Root mean square of successive RR interval differences. "
                        "Primary marker of parasympathetic (vagal) tone. "
                        "Reflects short-term beat-to-beat variability."
                    ),
                    "ref_low":     20,
                    "ref_high":    75,
                    "clinical":    (
                        "RMSSD <20 ms indicates severely reduced vagal tone. "
                        "Low RMSSD is associated with increased sympathetic dominance, "
                        "anxiety disorders, and poor cardiac prognosis."
                    ),
                },
                {
                    "key":         "pnn50_pct",
                    "label":       "pNN50",
                    "unit":        "%",
                    "description": (
                        "Percentage of successive RR interval differences >50 ms. "
                        "Parasympathetic index; highly correlated with RMSSD."
                    ),
                    "ref_low":     5,
                    "ref_high":    40,
                    "clinical":    "pNN50 <5% indicates severely reduced parasympathetic activity.",
                },
                {
                    "key":         "hr_mean_bpm",
                    "label":       "Mean Heart Rate",
                    "unit":        "bpm",
                    "description": "Average heart rate over the recording period.",
                    "ref_low":     60,
                    "ref_high":    85,
                    "clinical":    "Resting HR >85 bpm at rest warrants further autonomic evaluation.",
                },
            ],
            "frequency_domain": [
                {
                    "key":         "vlf_power_ms2",
                    "label":       "VLF Power",
                    "unit":        "ms²",
                    "description": (
                        "Very Low Frequency band power (0.003–0.04 Hz). "
                        "Associated with thermoregulation, renin-angiotensin system, "
                        "and peripheral vasomotion. Requires 24h recording for reliable estimation."
                    ),
                    "ref_low":     300,
                    "ref_high":    1500,
                    "clinical":    "Severely depressed VLF is a strong predictor of mortality in post-MI patients.",
                },
                {
                    "key":         "lf_power_ms2",
                    "label":       "LF Power",
                    "unit":        "ms²",
                    "description": (
                        "Low Frequency band power (0.04–0.15 Hz). "
                        "Reflects both sympathetic and parasympathetic activity, "
                        "with a dominant sympathetic component. Related to baroreflex activity."
                    ),
                    "ref_low":     750,
                    "ref_high":    2500,
                    "clinical":    "Elevated LF with reduced HF indicates sympathetic dominance.",
                },
                {
                    "key":         "hf_power_ms2",
                    "label":       "HF Power",
                    "unit":        "ms²",
                    "description": (
                        "High Frequency band power (0.15–0.40 Hz). "
                        "Corresponds to respiratory sinus arrhythmia (RSA). "
                        "Marker of vagal (parasympathetic) cardiac control."
                    ),
                    "ref_low":     250,
                    "ref_high":    1500,
                    "clinical":    "Reduced HF power indicates impaired vagal tone.",
                },
                {
                    "key":         "lf_hf_ratio",
                    "label":       "LF/HF Ratio",
                    "unit":        "",
                    "description": (
                        "Ratio of LF to HF power. Widely used index of sympathovagal balance. "
                        "Higher values indicate sympathetic dominance; lower values indicate "
                        "parasympathetic dominance."
                    ),
                    "ref_low":     1.0,
                    "ref_high":    3.0,
                    "clinical":    (
                        "LF/HF >3.0 = sympathetic dominance (anxiety, hypertension, heart failure). "
                        "LF/HF <0.5 = parasympathetic dominance or selective sympathetic loss. "
                        "Note: LF/HF ratio as sole sympathovagal balance index is debated in literature."
                    ),
                },
                {
                    "key":         "total_power_ms2",
                    "label":       "Total Power",
                    "unit":        "ms²",
                    "description": (
                        "Variance of all NN intervals over the recording (≤0.4 Hz). "
                        "Global measure of autonomic modulation. Roughly equivalent to SDNN² for 24h recordings."
                    ),
                    "ref_low":     1500,
                    "ref_high":    5000,
                    "clinical":    (
                        "Total power <1500 ms² indicates globally depressed HRV — "
                        "associated with autonomic neuropathy and high cardiac risk."
                    ),
                },
            ],
        },
        "reference_ranges": REFERENCE_RANGES,
        "diagnostic_patterns": DIAGNOSTIC_PATTERNS,
        "severity_levels": {
            "Normal":   {
                "autonomic_score_range": "0–20",
                "description": "All HRV parameters within normal limits. No autonomic dysfunction detected.",
            },
            "Mild":     {
                "autonomic_score_range": "21–45",
                "description": (
                    "One or two parameters mildly outside normal range. "
                    "Early autonomic dysfunction. Lifestyle intervention recommended."
                ),
            },
            "Moderate": {
                "autonomic_score_range": "46–70",
                "description": (
                    "Multiple parameters significantly outside normal range. "
                    "Clear autonomic dysfunction. Medical management warranted."
                ),
            },
            "Severe":   {
                "autonomic_score_range": "71–100",
                "description": (
                    "Globally depressed HRV with severe autonomic impairment. "
                    "High cardiovascular mortality risk. Urgent specialist referral."
                ),
            },
        },
        "clinical_significance": {
            "guidelines": (
                "Task Force of the European Society of Cardiology and the North American Society "
                "of Pacing and Electrophysiology. Heart rate variability: standards of measurement, "
                "physiological interpretation, and clinical use. "
                "Eur Heart J. 1996;17(3):354-381."
            ),
            "sdnn_mortality": (
                "SDNN <100 ms is an established independent predictor of all-cause and "
                "cardiovascular mortality (Task Force 1996). Patients with SDNN <50 ms have "
                "significantly reduced survival compared to those with SDNN >100 ms."
            ),
            "rmssd_vagal": (
                "RMSSD is the recommended time-domain index for vagal tone estimation. "
                "It is the most robust short-term HRV measure, with minimal sensitivity "
                "to recording duration. RMSSD <20 ms indicates severely impaired parasympathetic activity."
            ),
            "lf_hf_sympathovagal": (
                "The LF/HF ratio reflects the balance between sympathetic and parasympathetic "
                "nervous system activity. An elevated LF/HF ratio (>3.0) indicates sympathetic "
                "dominance, associated with stress, hypertension, and heart failure. "
                "A low LF/HF ratio (<0.5) suggests parasympathetic dominance or autonomic neuropathy "
                "with selective sympathetic failure."
            ),
            "hrv_epilepsy_sudep": (
                "HRV reduction is a recognised biomarker for Sudden Unexpected Death in Epilepsy (SUDEP) "
                "risk stratification. Periictal autonomic dysregulation, including sustained ictal "
                "tachycardia and postictal bradycardia, is associated with low interictal HRV. "
                "Patients with epilepsy show significantly lower SDNN and RMSSD compared to controls, "
                "partly attributable to antiepileptic drug effects on autonomic tone."
            ),
            "diabetic_autonomic_neuropathy": (
                "Diabetic autonomic neuropathy (DAN) is the most common cause of globally depressed HRV "
                "in clinical practice. Early DAN manifests as reduced RMSSD and HF power. "
                "Advanced DAN presents with nearly absent HRV across all frequency bands — "
                "the 'cardiac denervation' pattern."
            ),
            "ai_models": {
                "TS-Transformer": (
                    "Temporal Self-Attention Transformer trained on sequential RR interval time series. "
                    "Captures long-range dependencies in 24h Holter recordings for autonomic pattern classification."
                ),
                "XGBoost": (
                    "Gradient-boosted decision tree trained on extracted HRV feature vectors "
                    "(time + frequency domain). High interpretability for clinical use — "
                    "SHAP values identify dominant features driving autonomic dysfunction score."
                ),
            },
        },
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("HRV / RR Variation Dashboard — Standalone Test")
    print("=" * 60)

    print("\n--- overview() ---")
    ov = overview()
    print(json.dumps(ov["kpis"], indent=2))
    print("Severity distribution:", ov["severity_distribution"])
    print("Pattern distribution:", ov["pattern_distribution"])
    print(f"Patient summary: {len(ov['patient_summary'])} records")

    print("\n--- breakdown() ---")
    bk = breakdown()
    print("Time domain summary parameters:", len(bk["time_domain_summary"]))
    print("Frequency domain summary parameters:", len(bk["freq_domain_summary"]))
    print("SDNN histogram bins:", len(bk["sdnn_histogram"]))
    print("Patient detail cards:", len(bk["patient_detail_cards"]))

    print("\n--- definitions() ---")
    dfn = definitions()
    print("Test name:", dfn["test_name"])
    print("Time domain parameters defined:", len(dfn["parameters"]["time_domain"]))
    print("Frequency domain parameters defined:", len(dfn["parameters"]["frequency_domain"]))
    print("Diagnostic patterns:", list(dfn["diagnostic_patterns"].keys()))

    print("\n--- Sample patient card ---")
    card = bk["patient_detail_cards"][0]
    print(f"  Patient: {card['name']} (age {card['age']}) — {card['disease']}")
    print(f"  Severity: {card['severity']}  |  Pattern: {card['pattern_label']}")
    print(f"  Autonomic score: {card['autonomic_score']}")
    print(f"  SDNN: {card['time_domain']['sdnn_ms']['value']} ms  "
          f"({card['time_domain']['sdnn_ms']['flag']})")
    print(f"  RMSSD: {card['time_domain']['rmssd_ms']['value']} ms  "
          f"({card['time_domain']['rmssd_ms']['flag']})")
    print(f"  LF/HF: {card['frequency_domain']['lf_hf_ratio']['value']}  "
          f"({card['frequency_domain']['lf_hf_ratio']['flag']})")

    print("\nAll three API functions OK.")
