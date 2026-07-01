"""
ABPM / Holter Dashboard
Ambulatory Blood Pressure Monitoring + Holter ECG analysis.
Dipping-status + arrhythmia burden + cardiac-autonomic correlation.

Registry item: ABPM_HOLTER
Advancement: Dipping-status + arrhythmia + cardiac-autonomic correlation
Biomarkers: Dipping, QT, arrhythmia
AI Models: TS models, ECG-CNN
Standards: ESH/ESC 2018 (ABPM), AHA/ACC/HRS 2017 (Holter)
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
# Reference ranges — ESH/ESC 2018 (ABPM) + AHA/ACC/HRS 2017 (Holter)
# ---------------------------------------------------------------------------
REFERENCE_RANGES = {
    # ABPM parameters
    "systolic_24h_mmhg":     {"low": 100, "high": 130, "unit": "mmHg",  "label": "24h Mean Systolic BP"},
    "diastolic_24h_mmhg":    {"low": 60,  "high": 80,  "unit": "mmHg",  "label": "24h Mean Diastolic BP"},
    "systolic_day_mmhg":     {"low": 105, "high": 135, "unit": "mmHg",  "label": "Daytime Mean Systolic BP"},
    "diastolic_day_mmhg":    {"low": 65,  "high": 85,  "unit": "mmHg",  "label": "Daytime Mean Diastolic BP"},
    "systolic_night_mmhg":   {"low": 90,  "high": 120, "unit": "mmHg",  "label": "Nighttime Mean Systolic BP"},
    "diastolic_night_mmhg":  {"low": 50,  "high": 70,  "unit": "mmHg",  "label": "Nighttime Mean Diastolic BP"},
    "dipping_pct":           {"low": 10,  "high": 20,  "unit": "%",     "label": "Nocturnal Dipping (%)"},
    "bp_load_systolic_pct":  {"low": 0,   "high": 25,  "unit": "%",     "label": "Systolic BP Load"},
    "bp_load_diastolic_pct": {"low": 0,   "high": 25,  "unit": "%",     "label": "Diastolic BP Load"},
    "pulse_pressure_mmhg":   {"low": 30,  "high": 50,  "unit": "mmHg",  "label": "Pulse Pressure"},
    "map_mmhg":              {"low": 70,  "high": 100,  "unit": "mmHg",  "label": "Mean Arterial Pressure"},
    # Holter parameters
    "mean_hr_bpm":           {"low": 60,  "high": 85,  "unit": "bpm",   "label": "Mean Heart Rate"},
    "min_hr_bpm":            {"low": 40,  "high": 60,  "unit": "bpm",   "label": "Minimum Heart Rate"},
    "max_hr_bpm":            {"low": 100, "high": 160, "unit": "bpm",   "label": "Maximum Heart Rate"},
    "pvc_count":             {"low": 0,   "high": 500, "unit": "",      "label": "PVC Count (24h)"},
    "pac_count":             {"low": 0,   "high": 200, "unit": "",      "label": "PAC Count (24h)"},
    "qtc_ms":                {"low": 360, "high": 450, "unit": "ms",    "label": "QTc Interval"},
    "vt_runs":               {"low": 0,   "high": 0,   "unit": "",      "label": "VT Runs"},
    "af_episodes":           {"low": 0,   "high": 0,   "unit": "",      "label": "AF Episodes"},
    "st_depression_events":  {"low": 0,   "high": 0,   "unit": "",      "label": "ST Depression Events"},
}

DIPPING_CATEGORIES = {
    "extreme_dipper":  {"label": "Extreme Dipper",  "range": ">20%",   "risk": "Nocturnal cerebral hypoperfusion, stroke risk"},
    "normal_dipper":   {"label": "Normal Dipper",   "range": "10-20%", "risk": "Normal circadian BP regulation"},
    "non_dipper":      {"label": "Non-Dipper",      "range": "0-10%",  "risk": "Target organ damage, cardiovascular events, autonomic dysfunction"},
    "reverse_dipper":  {"label": "Reverse Dipper",  "range": "<0%",    "risk": "Highest cardiovascular risk, sleep apnea, autonomic failure"},
}

DIAGNOSTIC_PATTERNS = {
    "normal": {
        "label": "Normal BP & ECG",
        "description": (
            "24h BP within normal limits with normal nocturnal dipping (10-20%). "
            "Holter shows sinus rhythm, no significant arrhythmia, normal QTc. "
            "Balanced cardiac-autonomic function."
        ),
    },
    "sustained_hypertension": {
        "label": "Sustained Hypertension",
        "description": (
            "Elevated 24h mean BP (systolic >130 or diastolic >80 mmHg) "
            "with high BP load (>25%). Increased cardiovascular and cerebrovascular risk. "
            "Associated with left ventricular hypertrophy and renal damage."
        ),
    },
    "white_coat_hypertension": {
        "label": "White Coat Hypertension",
        "description": (
            "Office BP elevated but 24h ABPM within normal limits. "
            "BP load <25%. Normal dipping pattern preserved. "
            "Lower cardiovascular risk than sustained hypertension."
        ),
    },
    "masked_hypertension": {
        "label": "Masked Hypertension",
        "description": (
            "Normal office BP but elevated 24h ABPM values. "
            "Often non-dipping or reverse-dipping pattern. "
            "Higher cardiovascular risk than white coat hypertension."
        ),
    },
    "autonomic_dysregulation": {
        "label": "Autonomic Dysregulation",
        "description": (
            "Non-dipping or reverse-dipping pattern with reduced heart rate variability "
            "and abnormal QTc. Indicates impaired autonomic modulation of cardiovascular system. "
            "Common in epilepsy (SUDEP risk), diabetes, and Parkinson disease."
        ),
    },
    "arrhythmia_burden": {
        "label": "Significant Arrhythmia Burden",
        "description": (
            "PVC count >500/24h, or VT runs, or AF episodes detected on Holter. "
            "Requires cardiology evaluation. In epilepsy patients, ictal arrhythmias "
            "may contribute to SUDEP risk."
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
    """Read up to 30 patients from clinical.db."""
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


def _seed(patient_id, domain: str, param: str) -> float:
    """Deterministic pseudo-random float in [0, 1) using MD5."""
    key = f"{patient_id}:{domain}:{param}"
    digest = hashlib.md5(key.encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * t


def _abnormality_probability(patient: dict) -> float:
    """Disease/age-based probability of ABPM/Holter abnormality."""
    disease = (patient.get("disease") or "").lower()
    age = patient.get("age") or 0
    prob = 0.10  # baseline

    if any(k in disease for k in ("epilep", "seizure")):
        prob += 0.25   # autonomic dysregulation, SUDEP risk, AED effects on QTc
    if any(k in disease for k in ("diabet", "neuropath")):
        prob += 0.40   # diabetic autonomic neuropathy, non-dipping
    if any(k in disease for k in ("stroke", "cardiovasc", "cardiac", "heart")):
        prob += 0.45   # hypertension, arrhythmia
    if any(k in disease for k in ("parkinson",)):
        prob += 0.35   # orthostatic hypotension, non-dipping
    if any(k in disease for k in ("multiple sclerosis", " ms ")):
        prob += 0.20
    if any(k in disease for k in ("anxiety", "depression", "depress")):
        prob += 0.15   # sympathetic overdrive, elevated resting BP
    if any(k in disease for k in ("dysautonomia",)):
        prob += 0.50
    if age > 60:
        prob += 0.25
    elif age > 40:
        prob += 0.10

    return min(prob, 0.90)


def _classify_dipping(dipping_pct: float) -> str:
    """Classify nocturnal dipping status."""
    if dipping_pct > 20:
        return "extreme_dipper"
    elif dipping_pct >= 10:
        return "normal_dipper"
    elif dipping_pct >= 0:
        return "non_dipper"
    return "reverse_dipper"


def _classify_pattern(study: dict) -> str:
    """Classify diagnostic pattern from combined ABPM + Holter metrics."""
    sys_24h = study["systolic_24h_mmhg"]
    dia_24h = study["diastolic_24h_mmhg"]
    dipping = study["dipping_pct"]
    bp_load_s = study["bp_load_systolic_pct"]
    pvc = study["pvc_count"]
    vt = study["vt_runs"]
    af = study["af_episodes"]
    qtc = study["qtc_ms"]

    # Arrhythmia burden first (critical finding)
    if pvc > 500 or vt > 0 or af > 0:
        return "arrhythmia_burden"

    # Autonomic dysregulation: non/reverse dipping + prolonged QTc
    if (dipping < 10 or dipping < 0) and qtc > 450:
        return "autonomic_dysregulation"

    # Non/reverse dipping alone with otherwise normal BP
    if dipping < 0 and sys_24h <= 130 and dia_24h <= 80:
        return "autonomic_dysregulation"

    # Sustained hypertension
    if (sys_24h > 130 or dia_24h > 80) and bp_load_s > 25:
        return "sustained_hypertension"

    # Masked hypertension (normal-ish mean but high load or non-dipping)
    if sys_24h <= 135 and bp_load_s > 30:
        return "masked_hypertension"

    # White coat (only mildly elevated, low load)
    if sys_24h > 125 and bp_load_s < 20:
        return "white_coat_hypertension"

    return "normal"


def _severity_from_score(score: float) -> str:
    if score <= 20:
        return "Normal"
    elif score <= 45:
        return "Mild"
    elif score <= 70:
        return "Moderate"
    return "Severe"


def _compute_cardiac_score(study: dict) -> float:
    """
    Composite cardiac-autonomic risk score 0-100.
    Combines BP abnormality, dipping status, arrhythmia burden, QTc prolongation.
    """
    penalties = []

    def _pen(val, lo, hi, weight=1.0):
        if val < lo:
            return weight * min(1.0, (lo - val) / max(lo, 1))
        if val > hi:
            return weight * min(1.0, (val - hi) / max(hi, 1))
        return 0.0

    # BP parameters
    penalties.append(_pen(study["systolic_24h_mmhg"],    100, 130, weight=2.0))
    penalties.append(_pen(study["diastolic_24h_mmhg"],    60,  80, weight=1.5))
    penalties.append(_pen(study["dipping_pct"],            10,  20, weight=2.0))
    penalties.append(_pen(study["bp_load_systolic_pct"],    0,  25, weight=1.5))
    penalties.append(_pen(study["pulse_pressure_mmhg"],    30,  50, weight=1.0))
    # Holter parameters
    penalties.append(_pen(study["qtc_ms"],                360, 450, weight=2.0))
    penalties.append(_pen(study["pvc_count"],               0, 500, weight=1.5))
    penalties.append(_pen(study["mean_hr_bpm"],            60,  85, weight=1.0))

    total_weight = 2.0 + 1.5 + 2.0 + 1.5 + 1.0 + 2.0 + 1.5 + 1.0  # 12.5
    raw = sum(penalties)
    score = min(100.0, raw / total_weight * 100)
    return round(score, 1)


# ---------------------------------------------------------------------------
# Study generation
# ---------------------------------------------------------------------------

def _generate_study(patient: dict) -> dict:
    """Generate a deterministic ABPM + Holter study for one patient."""
    pid = patient["patient_id"]
    prob = _abnormality_probability(patient)
    s = _seed(pid, "abpm", "is_abnormal")
    is_abnormal = s < prob

    def rnd(param, lo, hi):
        return _seed(pid, "abpm", param) * (hi - lo) + lo

    if is_abnormal:
        # Elevated BP, reduced dipping, arrhythmia risk
        sys_day    = rnd("sys_day",   130, 165)
        dia_day    = rnd("dia_day",    80, 100)
        # Non-dipping or reverse-dipping
        dip_seed   = _seed(pid, "abpm", "dip_bias")
        if dip_seed < 0.3:
            dipping = rnd("dip", -8, 0)     # reverse dipper
        elif dip_seed < 0.7:
            dipping = rnd("dip", 0, 9.9)    # non-dipper
        else:
            dipping = rnd("dip", 10, 15)     # borderline normal

        sys_night  = round(sys_day * (1 - dipping / 100), 1)
        dia_night  = round(dia_day * (1 - dipping / 100 * 0.8), 1)
        sys_24h    = round((sys_day * 14 + sys_night * 10) / 24, 1)
        dia_24h    = round((dia_day * 14 + dia_night * 10) / 24, 1)
        bp_load_s  = rnd("bpload_s", 25, 65)
        bp_load_d  = rnd("bpload_d", 15, 50)
        pp         = round(sys_24h - dia_24h, 1)
        map_val    = round(dia_24h + pp / 3, 1)

        # Holter - abnormal
        mean_hr    = rnd("mean_hr",  75, 100)
        min_hr     = round(mean_hr - rnd("hr_lo", 20, 35), 1)
        max_hr     = round(mean_hr + rnd("hr_hi", 30, 60), 1)
        pvc        = int(rnd("pvc", 100, 3000))
        pac        = int(rnd("pac", 50, 800))
        qtc        = rnd("qtc", 440, 510)
        vt_runs    = int(rnd("vt", 0, 5)) if _seed(pid, "abpm", "vt_bias") < 0.35 else 0
        af_eps     = int(rnd("af", 0, 3)) if _seed(pid, "abpm", "af_bias") < 0.20 else 0
        st_deps    = int(rnd("st", 0, 8)) if _seed(pid, "abpm", "st_bias") < 0.25 else 0
    else:
        # Normal BP, normal dipping, minimal arrhythmia
        sys_day    = rnd("sys_day",   110, 135)
        dia_day    = rnd("dia_day",    65,  85)
        dipping    = rnd("dip",        10,  20)    # normal dipper
        sys_night  = round(sys_day * (1 - dipping / 100), 1)
        dia_night  = round(dia_day * (1 - dipping / 100 * 0.8), 1)
        sys_24h    = round((sys_day * 14 + sys_night * 10) / 24, 1)
        dia_24h    = round((dia_day * 14 + dia_night * 10) / 24, 1)
        bp_load_s  = rnd("bpload_s", 2, 24)
        bp_load_d  = rnd("bpload_d", 1, 20)
        pp         = round(sys_24h - dia_24h, 1)
        map_val    = round(dia_24h + pp / 3, 1)

        # Holter - normal
        mean_hr    = rnd("mean_hr", 60, 80)
        min_hr     = round(mean_hr - rnd("hr_lo", 10, 20), 1)
        max_hr     = round(mean_hr + rnd("hr_hi", 25, 50), 1)
        pvc        = int(rnd("pvc", 0, 200))
        pac        = int(rnd("pac", 0, 100))
        qtc        = rnd("qtc", 370, 440)
        vt_runs    = 0
        af_eps     = 0
        st_deps    = 0

    study = {
        # ABPM
        "systolic_24h_mmhg":     round(sys_24h, 1),
        "diastolic_24h_mmhg":    round(dia_24h, 1),
        "systolic_day_mmhg":     round(sys_day, 1),
        "diastolic_day_mmhg":    round(dia_day, 1),
        "systolic_night_mmhg":   round(sys_night, 1),
        "diastolic_night_mmhg":  round(dia_night, 1),
        "dipping_pct":           round(dipping, 1),
        "dipping_category":      _classify_dipping(round(dipping, 1)),
        "bp_load_systolic_pct":  round(bp_load_s, 1),
        "bp_load_diastolic_pct": round(bp_load_d, 1),
        "pulse_pressure_mmhg":   pp,
        "map_mmhg":              map_val,
        # Holter
        "mean_hr_bpm":           round(mean_hr, 1),
        "min_hr_bpm":            min_hr,
        "max_hr_bpm":            max_hr,
        "pvc_count":             pvc,
        "pac_count":             pac,
        "qtc_ms":                round(qtc, 1),
        "vt_runs":               vt_runs,
        "af_episodes":           af_eps,
        "st_depression_events":  st_deps,
    }

    study["cardiac_score"]       = _compute_cardiac_score(study)
    study["diagnostic_pattern"]  = _classify_pattern(study)
    study["severity"]            = _severity_from_score(study["cardiac_score"])
    study["is_abnormal"]         = study["severity"] != "Normal"

    return study


def _flag(val, ref_key):
    """Return 'normal', 'low', or 'high' relative to reference range."""
    ref = REFERENCE_RANGES.get(ref_key, {})
    lo = ref.get("low", -1e9)
    hi = ref.get("high", 1e9)
    if val < lo:
        return "low"
    if val > hi:
        return "high"
    return "normal"


def _all_studies():
    """Return list of (patient, study) dicts for all patients."""
    patients = _get_patients()
    results = []
    for p in patients:
        study = _generate_study(p)
        results.append({"patient": p, "study": study})
    return results


# ---------------------------------------------------------------------------
# Public API — overview()
# ---------------------------------------------------------------------------

def overview() -> dict:
    """KPI cards, severity distribution, pattern distribution, dipping distribution,
    patient summary table."""
    studies = _all_studies()

    total      = len(studies)
    abnormals  = [s for s in studies if s["study"]["is_abnormal"]]
    abn_count  = len(abnormals)
    abn_rate   = round(abn_count / total * 100, 1) if total else 0

    sys_vals   = [s["study"]["systolic_24h_mmhg"]  for s in studies]
    dia_vals   = [s["study"]["diastolic_24h_mmhg"] for s in studies]
    dip_vals   = [s["study"]["dipping_pct"]        for s in studies]
    qtc_vals   = [s["study"]["qtc_ms"]             for s in studies]

    mean_sys   = round(sum(sys_vals) / len(sys_vals), 1)
    mean_dia   = round(sum(dia_vals) / len(dia_vals), 1)
    mean_qtc   = round(sum(qtc_vals) / len(qtc_vals), 1)

    # Severity distribution
    sev_dist = {"Normal": 0, "Mild": 0, "Moderate": 0, "Severe": 0}
    for s in studies:
        sev_dist[s["study"]["severity"]] += 1

    # Pattern distribution
    pattern_dist = {k: 0 for k in DIAGNOSTIC_PATTERNS}
    for s in studies:
        pat = s["study"]["diagnostic_pattern"]
        pattern_dist[pat] = pattern_dist.get(pat, 0) + 1

    # Dipping distribution
    dip_dist = {k: 0 for k in DIPPING_CATEGORIES}
    for s in studies:
        cat = s["study"]["dipping_category"]
        dip_dist[cat] = dip_dist.get(cat, 0) + 1

    # Patient summary table
    patient_summary = []
    for s in studies:
        p, st = s["patient"], s["study"]
        patient_summary.append({
            "patient_id":          p["patient_id"],
            "name":                p["name"],
            "age":                 p["age"],
            "disease":             p["disease"],
            "severity":            st["severity"],
            "diagnostic_pattern":  st["diagnostic_pattern"],
            "pattern_label":       DIAGNOSTIC_PATTERNS[st["diagnostic_pattern"]]["label"],
            "cardiac_score":       st["cardiac_score"],
            "systolic_24h":        st["systolic_24h_mmhg"],
            "diastolic_24h":       st["diastolic_24h_mmhg"],
            "dipping_pct":         st["dipping_pct"],
            "dipping_category":    st["dipping_category"],
            "qtc_ms":              st["qtc_ms"],
            "pvc_count":           st["pvc_count"],
            "is_abnormal":         st["is_abnormal"],
        })

    return {
        "kpis": {
            "total_studies":       total,
            "abnormal_count":      abn_count,
            "abnormal_rate_pct":   abn_rate,
            "mean_systolic_24h":   mean_sys,
            "mean_diastolic_24h":  mean_dia,
            "mean_qtc_ms":         mean_qtc,
        },
        "severity_distribution": sev_dist,
        "pattern_distribution":  pattern_dist,
        "dipping_distribution":  dip_dist,
        "patient_summary":       patient_summary,
    }


# ---------------------------------------------------------------------------
# Public API — breakdown()
# ---------------------------------------------------------------------------

def breakdown() -> dict:
    """ABPM parameter tables, Holter parameter tables, histograms, per-patient detail."""
    studies = _all_studies()

    # ---- ABPM summary ----
    abpm_params = [
        ("systolic_24h_mmhg",     "systolic_24h_mmhg"),
        ("diastolic_24h_mmhg",    "diastolic_24h_mmhg"),
        ("systolic_day_mmhg",     "systolic_day_mmhg"),
        ("diastolic_day_mmhg",    "diastolic_day_mmhg"),
        ("systolic_night_mmhg",   "systolic_night_mmhg"),
        ("diastolic_night_mmhg",  "diastolic_night_mmhg"),
        ("dipping_pct",           "dipping_pct"),
        ("bp_load_systolic_pct",  "bp_load_systolic_pct"),
        ("bp_load_diastolic_pct", "bp_load_diastolic_pct"),
        ("pulse_pressure_mmhg",   "pulse_pressure_mmhg"),
        ("map_mmhg",              "map_mmhg"),
    ]
    abpm_summary = []
    for param, ref_key in abpm_params:
        vals = [s["study"][param] for s in studies]
        ref  = REFERENCE_RANGES.get(ref_key, {})
        n_abn = sum(1 for v in vals if v < ref.get("low", -1e9) or v > ref.get("high", 1e9))
        abpm_summary.append({
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

    # ---- Holter summary ----
    holter_params = [
        ("mean_hr_bpm",          "mean_hr_bpm"),
        ("min_hr_bpm",           "min_hr_bpm"),
        ("max_hr_bpm",           "max_hr_bpm"),
        ("pvc_count",            "pvc_count"),
        ("pac_count",            "pac_count"),
        ("qtc_ms",               "qtc_ms"),
        ("vt_runs",              "vt_runs"),
        ("af_episodes",          "af_episodes"),
        ("st_depression_events", "st_depression_events"),
    ]
    holter_summary = []
    for param, ref_key in holter_params:
        vals = [s["study"][param] for s in studies]
        ref  = REFERENCE_RANGES.get(ref_key, {})
        n_abn = sum(1 for v in vals if v < ref.get("low", -1e9) or v > ref.get("high", 1e9))
        holter_summary.append({
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

    systolic_hist   = _histogram([s["study"]["systolic_24h_mmhg"]   for s in studies], lo=90,  hi=170)
    dipping_hist    = _histogram([s["study"]["dipping_pct"]         for s in studies], lo=-10, hi=25)
    qtc_hist        = _histogram([s["study"]["qtc_ms"]              for s in studies], lo=350, hi=520)
    pvc_hist        = _histogram([s["study"]["pvc_count"]           for s in studies], lo=0,   hi=3200)
    score_hist      = _histogram([s["study"]["cardiac_score"]       for s in studies], lo=0,   hi=100)

    # ---- Per-patient detail cards ----
    detail_cards = []
    for s in studies:
        p, st = s["patient"], s["study"]

        abpm_detail = {}
        for key in ["systolic_24h_mmhg", "diastolic_24h_mmhg", "systolic_day_mmhg",
                     "diastolic_day_mmhg", "systolic_night_mmhg", "diastolic_night_mmhg",
                     "dipping_pct", "bp_load_systolic_pct", "bp_load_diastolic_pct",
                     "pulse_pressure_mmhg", "map_mmhg"]:
            ref = REFERENCE_RANGES.get(key, {})
            abpm_detail[key] = {
                "value":    st[key],
                "unit":     ref.get("unit", ""),
                "flag":     _flag(st[key], key),
                "ref_low":  ref.get("low"),
                "ref_high": ref.get("high"),
            }

        holter_detail = {}
        for key in ["mean_hr_bpm", "min_hr_bpm", "max_hr_bpm", "pvc_count",
                     "pac_count", "qtc_ms", "vt_runs", "af_episodes",
                     "st_depression_events"]:
            ref = REFERENCE_RANGES.get(key, {})
            holter_detail[key] = {
                "value":    st[key],
                "unit":     ref.get("unit", ""),
                "flag":     _flag(st[key], key),
                "ref_low":  ref.get("low"),
                "ref_high": ref.get("high"),
            }

        detail_cards.append({
            "patient_id":          p["patient_id"],
            "name":                p["name"],
            "age":                 p["age"],
            "disease":             p["disease"],
            "seizure_count":       p["seizure_count"],
            "med_count":           p["med_count"],
            "severity":            st["severity"],
            "diagnostic_pattern":  st["diagnostic_pattern"],
            "pattern_label":       DIAGNOSTIC_PATTERNS[st["diagnostic_pattern"]]["label"],
            "cardiac_score":       st["cardiac_score"],
            "dipping_category":    st["dipping_category"],
            "is_abnormal":         st["is_abnormal"],
            "abpm":                abpm_detail,
            "holter":              holter_detail,
        })

    return {
        "abpm_summary":            abpm_summary,
        "holter_summary":          holter_summary,
        "systolic_histogram":      systolic_hist,
        "dipping_histogram":       dipping_hist,
        "qtc_histogram":           qtc_hist,
        "pvc_histogram":           pvc_hist,
        "cardiac_score_histogram": score_hist,
        "patient_detail_cards":    detail_cards,
    }


# ---------------------------------------------------------------------------
# Public API — definitions()
# ---------------------------------------------------------------------------

def definitions() -> dict:
    """Protocol, parameters, reference ranges, diagnostic patterns, severity levels,
    and clinical significance."""
    return {
        "test_name": "Ambulatory Blood Pressure Monitoring (ABPM) / Holter ECG Analysis",
        "code": "ABPM_HOLTER",
        "protocol": {
            "description": (
                "Combined 24-hour ambulatory blood pressure monitoring (ABPM) and continuous "
                "ECG (Holter) recording for comprehensive cardiac-autonomic evaluation. "
                "ABPM records BP at 15-30 min intervals (daytime) and 30-60 min intervals (nighttime) "
                "using an oscillometric cuff. Holter ECG provides continuous 3-12 lead recording "
                "for arrhythmia detection, QT interval analysis, and ST segment monitoring. "
                "The combined study enables correlation of BP changes with cardiac rhythm events, "
                "providing a holistic view of cardiovascular autonomic regulation."
            ),
            "recording_methods": {
                "abpm": (
                    "24-hour oscillometric BP recording. Daytime: every 15-30 min (06:00-22:00). "
                    "Nighttime: every 30-60 min (22:00-06:00). Non-dominant arm cuff. "
                    "Minimum 70% valid readings required for analysis."
                ),
                "holter": (
                    "Continuous 24-hour 3-lead or 12-lead ECG recording. Digital sampling at "
                    "128-1000 Hz. Automated beat detection with manual review of flagged events. "
                    "QT measurement on artifact-free segments. Pan-Tompkins QRS detection."
                ),
            },
            "indications": [
                "White coat hypertension evaluation",
                "Masked hypertension detection",
                "Nocturnal hypertension and non-dipping assessment",
                "Antihypertensive medication efficacy monitoring",
                "Autonomic dysfunction evaluation in neurological disorders",
                "SUDEP risk stratification in epilepsy",
                "Arrhythmia surveillance (PVC, VT, AF)",
                "QT interval monitoring (drug-induced prolongation)",
                "Syncope and presyncope evaluation",
                "Cardiac-autonomic correlation in diabetes and Parkinson disease",
            ],
            "standard": (
                "Williams B, et al. 2018 ESC/ESH Guidelines for the management of arterial hypertension. "
                "Eur Heart J. 2018;39(33):3021-3104. | "
                "Al-Khatib SM, et al. 2017 AHA/ACC/HRS Guideline for Management of Patients With "
                "Ventricular Arrhythmias. J Am Coll Cardiol. 2018;72(14):e91-e220."
            ),
            "ai_models": ["TS models", "ECG-CNN"],
            "biomarkers": ["Dipping status", "QTc interval", "Arrhythmia burden"],
        },
        "parameters": {
            "abpm": [
                {"key": "systolic_24h_mmhg",     "label": "24h Mean Systolic BP",       "unit": "mmHg", "ref_low": 100, "ref_high": 130, "description": "Average systolic blood pressure over 24 hours. Threshold for hypertension: >130 mmHg (ESH/ESC 2018)."},
                {"key": "diastolic_24h_mmhg",    "label": "24h Mean Diastolic BP",      "unit": "mmHg", "ref_low": 60,  "ref_high": 80,  "description": "Average diastolic blood pressure over 24 hours. Threshold for hypertension: >80 mmHg."},
                {"key": "systolic_day_mmhg",     "label": "Daytime Mean Systolic BP",   "unit": "mmHg", "ref_low": 105, "ref_high": 135, "description": "Average systolic BP during waking hours (06:00-22:00). Normal <135 mmHg."},
                {"key": "diastolic_day_mmhg",    "label": "Daytime Mean Diastolic BP",  "unit": "mmHg", "ref_low": 65,  "ref_high": 85,  "description": "Average diastolic BP during waking hours. Normal <85 mmHg."},
                {"key": "systolic_night_mmhg",   "label": "Nighttime Mean Systolic BP", "unit": "mmHg", "ref_low": 90,  "ref_high": 120, "description": "Average systolic BP during sleep (22:00-06:00). Normal <120 mmHg."},
                {"key": "diastolic_night_mmhg",  "label": "Nighttime Mean Diastolic BP","unit": "mmHg", "ref_low": 50,  "ref_high": 70,  "description": "Average diastolic BP during sleep. Normal <70 mmHg."},
                {"key": "dipping_pct",           "label": "Nocturnal Dipping (%)",      "unit": "%",    "ref_low": 10,  "ref_high": 20,  "description": "Percentage fall in nocturnal BP relative to daytime. Normal dipping: 10-20%. Non-dipping (<10%) and reverse dipping (<0%) indicate autonomic dysfunction and increased cardiovascular risk."},
                {"key": "bp_load_systolic_pct",  "label": "Systolic BP Load",           "unit": "%",    "ref_low": 0,   "ref_high": 25,  "description": "Percentage of readings above systolic threshold (daytime >135, nighttime >120 mmHg). Load >25% indicates hypertension."},
                {"key": "bp_load_diastolic_pct", "label": "Diastolic BP Load",          "unit": "%",    "ref_low": 0,   "ref_high": 25,  "description": "Percentage of readings above diastolic threshold. Load >25% indicates hypertension."},
                {"key": "pulse_pressure_mmhg",   "label": "Pulse Pressure",             "unit": "mmHg", "ref_low": 30,  "ref_high": 50,  "description": "Difference between systolic and diastolic BP. Elevated pulse pressure (>50 mmHg) indicates arterial stiffness and increased cardiovascular risk."},
                {"key": "map_mmhg",              "label": "Mean Arterial Pressure",     "unit": "mmHg", "ref_low": 70,  "ref_high": 100, "description": "MAP = DBP + 1/3 × (SBP - DBP). Represents perfusion pressure for end organs."},
            ],
            "holter": [
                {"key": "mean_hr_bpm",           "label": "Mean Heart Rate",            "unit": "bpm",  "ref_low": 60,  "ref_high": 85,  "description": "Average heart rate over 24 hours. Elevated resting HR is an independent cardiovascular risk factor."},
                {"key": "min_hr_bpm",            "label": "Minimum Heart Rate",         "unit": "bpm",  "ref_low": 40,  "ref_high": 60,  "description": "Lowest recorded HR. Bradycardia <40 bpm may indicate conduction abnormality or excessive vagal tone."},
                {"key": "max_hr_bpm",            "label": "Maximum Heart Rate",         "unit": "bpm",  "ref_low": 100, "ref_high": 160, "description": "Highest recorded HR. Inappropriately elevated max HR at rest may indicate arrhythmia."},
                {"key": "pvc_count",             "label": "PVC Count (24h)",            "unit": "",     "ref_low": 0,   "ref_high": 500, "description": "Premature ventricular complex count. >500/24h or >10% of total beats is considered clinically significant and may require treatment."},
                {"key": "pac_count",             "label": "PAC Count (24h)",            "unit": "",     "ref_low": 0,   "ref_high": 200, "description": "Premature atrial complex count. Frequent PACs may predict atrial fibrillation development."},
                {"key": "qtc_ms",                "label": "QTc Interval",               "unit": "ms",   "ref_low": 360, "ref_high": 450, "description": "Corrected QT interval (Bazett formula). QTc >450 ms (male) or >460 ms (female) indicates prolongation. QTc >500 ms is high-risk for torsades de pointes. Relevant for AED-induced prolongation."},
                {"key": "vt_runs",               "label": "VT Runs",                    "unit": "",     "ref_low": 0,   "ref_high": 0,   "description": "Ventricular tachycardia episodes (≥3 consecutive PVCs at >100 bpm). Any VT run is clinically significant."},
                {"key": "af_episodes",           "label": "AF Episodes",                "unit": "",     "ref_low": 0,   "ref_high": 0,   "description": "Atrial fibrillation episodes detected. Even brief AF episodes increase stroke risk 5-fold."},
                {"key": "st_depression_events",  "label": "ST Depression Events",       "unit": "",     "ref_low": 0,   "ref_high": 0,   "description": "ST segment depression >1mm lasting >1 min. Indicates myocardial ischemia."},
            ],
        },
        "reference_ranges": REFERENCE_RANGES,
        "dipping_categories": [
            {"category": k, "label": v["label"], "range": v["range"], "risk": v["risk"]}
            for k, v in DIPPING_CATEGORIES.items()
        ],
        "diagnostic_patterns": [
            {"pattern": k, "label": v["label"], "description": v["description"]}
            for k, v in DIAGNOSTIC_PATTERNS.items()
        ],
        "severity_levels": [
            {"level": "Normal",   "score_range": "0-20",   "description": "All ABPM and Holter parameters within normal limits. Normal nocturnal dipping. No arrhythmias."},
            {"level": "Mild",     "score_range": "21-45",  "description": "Mildly elevated BP or borderline non-dipping. Occasional PVCs. QTc within normal limits. Lifestyle modification recommended."},
            {"level": "Moderate", "score_range": "46-70",  "description": "Sustained elevated BP with abnormal dipping. Frequent PVCs or prolonged QTc. Antihypertensive therapy and cardiology referral warranted."},
            {"level": "Severe",   "score_range": "71-100", "description": "Severe hypertension with reverse dipping. VT runs or AF episodes. QTc >500 ms. High cardiovascular and SUDEP risk. Urgent specialist evaluation."},
        ],
        "clinical_significance": (
            "Combined ABPM + Holter monitoring provides comprehensive cardiac-autonomic assessment. "
            "In neurology, this is particularly relevant for: (1) SUDEP risk stratification — "
            "ictal/periictal cardiac arrhythmias and BP changes are key mechanisms; (2) Autonomic "
            "neuropathy assessment in diabetes and Parkinson disease — non-dipping BP is an early "
            "marker; (3) AED-related cardiac effects — carbamazepine, lacosamide, and phenytoin "
            "can prolong QTc; (4) Syncope workup — correlating BP drops with heart rhythm; "
            "(5) White coat vs masked hypertension differentiation — critical for stroke risk management."
        ),
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("ABPM / Holter Dashboard — Standalone Test")
    print("=" * 60)

    print("\n--- overview() ---")
    ov = overview()
    print(json.dumps(ov["kpis"], indent=2))
    print("Severity distribution:", ov["severity_distribution"])
    print("Pattern distribution:", ov["pattern_distribution"])
    print("Dipping distribution:", ov["dipping_distribution"])
    print(f"Patient summary: {len(ov['patient_summary'])} records")

    print("\n--- breakdown() ---")
    bk = breakdown()
    print("ABPM summary parameters:", len(bk["abpm_summary"]))
    print("Holter summary parameters:", len(bk["holter_summary"]))
    print("Systolic histogram bins:", len(bk["systolic_histogram"]))
    print("Patient detail cards:", len(bk["patient_detail_cards"]))

    print("\n--- definitions() ---")
    dfn = definitions()
    print("Test name:", dfn["test_name"])
    print("ABPM parameters defined:", len(dfn["parameters"]["abpm"]))
    print("Holter parameters defined:", len(dfn["parameters"]["holter"]))
    print("Diagnostic patterns:", len(dfn["diagnostic_patterns"]))

    print("\n--- Sample patient card ---")
    card = bk["patient_detail_cards"][0]
    print(f"  Patient: {card['name']} (age {card['age']}) — {card['disease']}")
    print(f"  Severity: {card['severity']}  |  Pattern: {card['pattern_label']}")
    print(f"  Cardiac score: {card['cardiac_score']}")
    print(f"  24h SBP: {card['abpm']['systolic_24h_mmhg']['value']} mmHg  "
          f"({card['abpm']['systolic_24h_mmhg']['flag']})")
    print(f"  Dipping: {card['abpm']['dipping_pct']['value']}%  "
          f"({card['dipping_category']})")
    print(f"  QTc: {card['holter']['qtc_ms']['value']} ms  "
          f"({card['holter']['qtc_ms']['flag']})")
    print(f"  PVCs: {card['holter']['pvc_count']['value']}  "
          f"({card['holter']['pvc_count']['flag']})")

    print("\nAll three API functions OK.")
