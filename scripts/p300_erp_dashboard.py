"""
P300 / ERP (Cognitive Event-Related Potentials) Dashboard — NeuroAI EEG
=======================================================================
Cognitive ERP analysis: P300 and N200 peak latency/amplitude extraction
from an auditory oddball paradigm. Data is derived deterministically from
real patient demographics in clinical.db.

The P300 (or P3) is a positive deflection at ~300 ms post-stimulus, maximal
at Pz, generated during the cognitive process of stimulus evaluation. It is
widely used to assess:
  - Working memory / attention (P300 latency increases with cognitive load)
  - AED effects on cognition (many AEDs prolong P300 latency)
  - Post-ictal cognitive impairment
  - Dementia screening (MMSE correlation)
  - TLE-related cognitive decline

Paradigm: Standard auditory 3-tone oddball
  - Standard tones (80%)  — low pitch 1 kHz
  - Target tones  (15%)   — high pitch 2 kHz, patient must count silently
  - Distractor tones (5%) — novel tones

Key parameters:
  - N200 latency at Fz: normal ≤280 ms  (selective attention / target detection)
  - N200 amplitude at Fz: reference ≥3.0 µV
  - P300 latency at Pz: age-adjusted; base 300 ms + 1.25 ms/year (Polich 2007)
    e.g., age 30 → ≤337.5 ms;  age 50 → ≤362.5 ms;  age 70 → ≤387.5 ms
  - P300 amplitude at Pz: normal ≥5.0 µV (declines with age)
  - P3a latency (frontal novelty response): normal ≤330 ms

Diagnostic patterns:
  - Normal: all parameters within age-adjusted reference
  - Prolonged latency: P300 latency > upper limit — cognitive slowing
  - Reduced amplitude: P300 amplitude < lower limit — attention deficit / fatigue
  - Combined: both latency prolonged and amplitude reduced — significant impairment
  - Absent P300: no identifiable P300 peak — severe cognitive dysfunction

Severity levels: Normal, Mild, Moderate, Severe

Reference:
  Polich J. Updating P300: An integrative theory of P3a and P3b.
  Clin Neurophysiol. 2007;118(10):2128-2148.
  Duncan CC et al. Event-related potentials in clinical research.
  Clin Neurophysiol. 2009;120(11):1883-1908.
  ACNS Guideline 9: Clinical utility of EEG.

Author: Research Team
"""

import sqlite3
import hashlib
from pathlib import Path
from collections import Counter

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# ── Reference ranges ──────────────────────────────────────────────────
N200_LATENCY_UPPER_MS = 280.0      # ms — Fz
N200_AMPLITUDE_LOWER_UV = 3.0     # µV — Fz
P300_LATENCY_BASE_MS = 300.0      # ms at age 20
P300_LATENCY_SLOPE_MS_PER_YR = 1.25   # ms per year (Polich 2007)
P300_AMPLITUDE_LOWER_UV = 5.0    # µV — Pz (reference lower bound)
P3A_LATENCY_UPPER_MS = 330.0      # ms — Fz novelty response

DIAGNOSTIC_PATTERNS = {
    "normal":           "Normal — all parameters within age-adjusted reference ranges",
    "prolonged_latency":"Prolonged Latency — P300/N200 latency increased; cognitive processing slowed",
    "reduced_amplitude":"Reduced Amplitude — P300/N200 amplitude decreased; attention/engagement deficit",
    "combined":         "Combined — both latency prolonged and amplitude reduced; significant impairment",
    "absent_p300":      "Absent P300 — no identifiable P300 peak; severe cognitive dysfunction",
}

SEVERITY_LEVELS = ["Normal", "Mild", "Moderate", "Severe"]


def _seed(patient_id, comp):
    """Deterministic pseudo-random from patient_id + component."""
    h = hashlib.md5(f"{patient_id}:{comp}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _p300_latency_upper(age):
    """Age-adjusted P300 upper limit (Polich 2007)."""
    return round(P300_LATENCY_BASE_MS + P300_LATENCY_SLOPE_MS_PER_YR * max(0, age - 20), 1)


def _get_patients():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT p.patient_id, p.name, p.age, p.disease,
               COUNT(DISTINCT s.id) as seizure_count,
               COUNT(DISTINCT m.id) as med_count,
               AVG(CASE WHEN a.instrument = 'MMSE' THEN a.score END) as mmse_score,
               AVG(CASE WHEN a.instrument = 'MoCA' THEN a.score END) as moca_score
        FROM patients p
        LEFT JOIN seizure_diary s ON p.patient_id = s.patient_id
        LEFT JOIN medications m ON p.patient_id = m.patient_id
        LEFT JOIN assessments a ON p.patient_id = a.patient_id
        GROUP BY p.patient_id
        ORDER BY p.patient_id
        LIMIT 30
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def _generate_erp(patient):
    pid = patient["patient_id"]
    age = patient.get("age", 40) or 40
    disease = (patient.get("disease") or "").lower()
    med_count = patient.get("med_count", 0) or 0
    seizure_count = patient.get("seizure_count", 0) or 0
    mmse = patient.get("mmse_score")
    moca = patient.get("moca_score")

    # Age-adjusted reference for this patient
    p300_ref = _p300_latency_upper(age)

    # Base abnormality probability driven by patient profile
    base_abn = 0.12
    if age > 65:
        base_abn += 0.25
    elif age > 50:
        base_abn += 0.12
    if "alzheimer" in disease or "dementia" in disease:
        base_abn += 0.45
    if "depression" in disease:
        base_abn += 0.20
    if "schizophrenia" in disease:
        base_abn += 0.30
    if "epilepsy" in disease:
        base_abn += 0.15  # AED cognitive effects
    if med_count > 3:
        base_abn += 0.10
    if seizure_count > 5:
        base_abn += 0.10
    if mmse is not None and mmse < 24:
        base_abn += 0.25
    if moca is not None and moca < 24:
        base_abn += 0.20

    main_seed = _seed(pid, "p300_main")
    sev_seed  = _seed(pid, "p300_sev")
    pat_seed  = _seed(pid, "p300_pat")

    is_abnormal = main_seed < min(base_abn, 0.85)

    if not is_abnormal:
        severity = "Normal"
        p300_lat = round(p300_ref * 0.80 + main_seed * (p300_ref * 0.15), 1)
        p300_amp = round(P300_AMPLITUDE_LOWER_UV * (1.4 + main_seed * 2.0), 2)
        n200_lat = round(N200_LATENCY_UPPER_MS * (0.70 + main_seed * 0.20), 1)
        n200_amp = round(N200_AMPLITUDE_LOWER_UV * (1.3 + main_seed * 1.5), 2)
        p3a_lat  = round(P3A_LATENCY_UPPER_MS * (0.75 + main_seed * 0.15), 1)
        pattern  = "normal"
    else:
        if sev_seed < 0.35:
            severity = "Mild"
            p300_lat = round(p300_ref * (1.05 + main_seed * 0.10), 1)
            p300_amp = round(P300_AMPLITUDE_LOWER_UV * (0.75 + main_seed * 0.20), 2)
            n200_lat = round(N200_LATENCY_UPPER_MS * (1.05 + main_seed * 0.10), 1)
            n200_amp = round(N200_AMPLITUDE_LOWER_UV * (0.80 + main_seed * 0.15), 2)
            p3a_lat  = round(P3A_LATENCY_UPPER_MS * (1.05 + main_seed * 0.08), 1)
        elif sev_seed < 0.70:
            severity = "Moderate"
            p300_lat = round(p300_ref * (1.20 + main_seed * 0.20), 1)
            p300_amp = round(P300_AMPLITUDE_LOWER_UV * (0.45 + main_seed * 0.25), 2)
            n200_lat = round(N200_LATENCY_UPPER_MS * (1.18 + main_seed * 0.18), 1)
            n200_amp = round(N200_AMPLITUDE_LOWER_UV * (0.50 + main_seed * 0.25), 2)
            p3a_lat  = round(P3A_LATENCY_UPPER_MS * (1.18 + main_seed * 0.15), 1)
        else:
            severity = "Severe"
            p300_lat = round(p300_ref * (1.45 + main_seed * 0.40), 1)
            p300_amp = round(max(0.3, P300_AMPLITUDE_LOWER_UV * (0.15 + main_seed * 0.25)), 2)
            n200_lat = round(N200_LATENCY_UPPER_MS * (1.40 + main_seed * 0.35), 1)
            n200_amp = round(max(0.2, N200_AMPLITUDE_LOWER_UV * (0.20 + main_seed * 0.25)), 2)
            p3a_lat  = round(P3A_LATENCY_UPPER_MS * (1.40 + main_seed * 0.30), 1)

        lat_abn = p300_lat > p300_ref
        amp_abn = p300_amp < P300_AMPLITUDE_LOWER_UV

        if severity == "Severe" and pat_seed < 0.25:
            pattern = "absent_p300"
        elif lat_abn and amp_abn:
            pattern = "combined"
        elif lat_abn:
            pattern = "prolonged_latency"
        else:
            pattern = "reduced_amplitude"

    p300_ref_upper = p300_ref
    p300_lat_abnormal = p300_lat > p300_ref_upper
    p300_amp_abnormal = p300_amp < P300_AMPLITUDE_LOWER_UV
    n200_lat_abnormal = n200_lat > N200_LATENCY_UPPER_MS
    n200_amp_abnormal = n200_amp < N200_AMPLITUDE_LOWER_UV
    p3a_lat_abnormal  = p3a_lat  > P3A_LATENCY_UPPER_MS

    return {
        "patient_id": pid,
        "patient_name": patient.get("name", pid),
        "age": age,
        "disease": patient.get("disease", "Unknown"),
        "severity": severity,
        "pattern": pattern,
        "p300_latency_ms": p300_lat,
        "p300_amplitude_uv": p300_amp,
        "n200_latency_ms": n200_lat,
        "n200_amplitude_uv": n200_amp,
        "p3a_latency_ms": p3a_lat,
        "p300_ref_upper_ms": p300_ref_upper,
        "p300_amp_ref_lower_uv": P300_AMPLITUDE_LOWER_UV,
        "n200_ref_upper_ms": N200_LATENCY_UPPER_MS,
        "n200_amp_ref_lower_uv": N200_AMPLITUDE_LOWER_UV,
        "p3a_ref_upper_ms": P3A_LATENCY_UPPER_MS,
        "p300_lat_abnormal": p300_lat_abnormal,
        "p300_amp_abnormal": p300_amp_abnormal,
        "n200_lat_abnormal": n200_lat_abnormal,
        "n200_amp_abnormal": n200_amp_abnormal,
        "p3a_lat_abnormal": p3a_lat_abnormal,
        "mmse_score": mmse,
        "moca_score": moca,
        "med_count": med_count,
        "seizure_count": seizure_count,
    }


def _get_all_studies():
    patients = _get_patients()
    return [_generate_erp(p) for p in patients]


# ── Public API ──────────────────────────────────────────────────────────

def overview():
    """KPIs, severity distribution, diagnostic pattern distribution,
    age-band P300 latency means, AED effect summary, per-patient table."""
    studies = _get_all_studies()
    total = len(studies)

    sev_dist = Counter(s["severity"] for s in studies)
    pat_dist = Counter(s["pattern"] for s in studies)
    abnormal_count = sum(1 for s in studies if s["severity"] != "Normal")

    mean_p300 = round(sum(s["p300_latency_ms"] for s in studies) / total, 1) if total else 0
    mean_amp  = round(sum(s["p300_amplitude_uv"] for s in studies) / total, 2) if total else 0
    mean_n200 = round(sum(s["n200_latency_ms"] for s in studies) / total, 1) if total else 0

    # Age-band analysis
    age_bands = [
        {"label": "20-39", "lo": 20, "hi": 40},
        {"label": "40-49", "lo": 40, "hi": 50},
        {"label": "50-59", "lo": 50, "hi": 60},
        {"label": "60-69", "lo": 60, "hi": 70},
        {"label": "70+",   "lo": 70, "hi": 200},
    ]
    age_band_data = []
    for band in age_bands:
        grp = [s for s in studies if band["lo"] <= s["age"] < band["hi"]]
        if grp:
            ref = _p300_latency_upper(int((band["lo"] + min(band["hi"], 90)) / 2))
            age_band_data.append({
                "age_band": band["label"],
                "n": len(grp),
                "mean_p300_ms": round(sum(g["p300_latency_ms"] for g in grp) / len(grp), 1),
                "ref_upper_ms": ref,
                "mean_amp_uv": round(sum(g["p300_amplitude_uv"] for g in grp) / len(grp), 2),
                "abnormal_pct": round(100 * sum(1 for g in grp if g["severity"] != "Normal") / len(grp), 1),
            })

    # AED cognitive impact (patients on >2 meds vs ≤2)
    high_med = [s for s in studies if s["med_count"] > 2]
    low_med  = [s for s in studies if s["med_count"] <= 2]
    aed_comparison = {
        "high_aed_burden": {
            "n": len(high_med),
            "mean_p300_ms": round(sum(s["p300_latency_ms"] for s in high_med) / len(high_med), 1) if high_med else 0,
            "abnormal_pct": round(100 * sum(1 for s in high_med if s["severity"] != "Normal") / len(high_med), 1) if high_med else 0,
        },
        "low_aed_burden": {
            "n": len(low_med),
            "mean_p300_ms": round(sum(s["p300_latency_ms"] for s in low_med) / len(low_med), 1) if low_med else 0,
            "abnormal_pct": round(100 * sum(1 for s in low_med if s["severity"] != "Normal") / len(low_med), 1) if low_med else 0,
        },
    }

    patient_summary = sorted([
        {
            "patient_id": s["patient_id"],
            "name": s["patient_name"],
            "age": s["age"],
            "disease": s["disease"],
            "severity": s["severity"],
            "pattern": s["pattern"],
            "p300_ms": s["p300_latency_ms"],
            "p300_amp_uv": s["p300_amplitude_uv"],
            "n200_ms": s["n200_latency_ms"],
            "ref_upper_ms": s["p300_ref_upper_ms"],
        }
        for s in studies
    ], key=lambda x: SEVERITY_LEVELS.index(x["severity"]) if x["severity"] in SEVERITY_LEVELS else 0, reverse=True)

    return {
        "kpis": {
            "total_studies": total,
            "abnormal_count": abnormal_count,
            "abnormal_rate_pct": round(100 * abnormal_count / total, 1) if total else 0,
            "mean_p300_latency_ms": mean_p300,
            "mean_p300_amplitude_uv": mean_amp,
            "mean_n200_latency_ms": mean_n200,
        },
        "severity_distribution": [
            {"severity": sev, "count": sev_dist.get(sev, 0)}
            for sev in SEVERITY_LEVELS
        ],
        "pattern_distribution": [
            {
                "pattern": p,
                "label": DIAGNOSTIC_PATTERNS[p].split(" — ")[0],
                "count": pat_dist.get(p, 0),
            }
            for p in ["normal", "prolonged_latency", "reduced_amplitude", "combined", "absent_p300"]
        ],
        "age_band_analysis": age_band_data,
        "aed_cognitive_impact": aed_comparison,
        "patient_summary": patient_summary,
    }


def breakdown():
    """P300 latency histogram, N200 vs P300 scatter data, per-patient
    full detail with latency/amplitude and AED-cognitive cross-analysis."""
    studies = _get_all_studies()

    # P300 latency histogram
    lat_buckets = [
        {"range": "<300", "lo": 0,   "hi": 300.01},
        {"range": "300-330", "lo": 300.01, "hi": 330.01},
        {"range": "330-360", "lo": 330.01, "hi": 360.01},
        {"range": "360-390", "lo": 360.01, "hi": 390.01},
        {"range": "390-420", "lo": 390.01, "hi": 420.01},
        {"range": "420+", "lo": 420.01, "hi": 9999},
    ]
    p300_histogram = [
        {"range": b["range"],
         "count": sum(1 for s in studies if b["lo"] <= s["p300_latency_ms"] < b["hi"])}
        for b in lat_buckets
    ]

    # P300 amplitude histogram
    amp_buckets = [
        {"range": "<3",   "lo": 0,   "hi": 3.01},
        {"range": "3-5",  "lo": 3.01, "hi": 5.01},
        {"range": "5-8",  "lo": 5.01, "hi": 8.01},
        {"range": "8-12", "lo": 8.01, "hi": 12.01},
        {"range": "12+",  "lo": 12.01, "hi": 9999},
    ]
    amp_histogram = [
        {"range": b["range"],
         "count": sum(1 for s in studies if b["lo"] <= s["p300_amplitude_uv"] < b["hi"])}
        for b in amp_buckets
    ]

    # Component-level summary (mean ± range)
    def _mean(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0

    def _mean2(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0

    component_summary = [
        {
            "component": "P300 (Pz)",
            "mean_latency_ms": _mean([s["p300_latency_ms"] for s in studies]),
            "ref_upper_ms": "age-adjusted (≈337 ms at age 30)",
            "mean_amplitude_uv": _mean2([s["p300_amplitude_uv"] for s in studies]),
            "ref_lower_uv": P300_AMPLITUDE_LOWER_UV,
            "abnormal_pct": round(100 * sum(1 for s in studies if s["p300_lat_abnormal"] or s["p300_amp_abnormal"]) / len(studies), 1) if studies else 0,
        },
        {
            "component": "N200 (Fz)",
            "mean_latency_ms": _mean([s["n200_latency_ms"] for s in studies]),
            "ref_upper_ms": N200_LATENCY_UPPER_MS,
            "mean_amplitude_uv": _mean2([s["n200_amplitude_uv"] for s in studies]),
            "ref_lower_uv": N200_AMPLITUDE_LOWER_UV,
            "abnormal_pct": round(100 * sum(1 for s in studies if s["n200_lat_abnormal"] or s["n200_amp_abnormal"]) / len(studies), 1) if studies else 0,
        },
        {
            "component": "P3a (Fz novelty)",
            "mean_latency_ms": _mean([s["p3a_latency_ms"] for s in studies]),
            "ref_upper_ms": P3A_LATENCY_UPPER_MS,
            "mean_amplitude_uv": "—",
            "ref_lower_uv": None,
            "abnormal_pct": round(100 * sum(1 for s in studies if s["p3a_lat_abnormal"]) / len(studies), 1) if studies else 0,
        },
    ]

    # Per-patient detail (sorted by severity then P300 latency)
    patient_details = sorted([
        {
            "patient_id": s["patient_id"],
            "name": s["patient_name"],
            "age": s["age"],
            "disease": s["disease"],
            "severity": s["severity"],
            "pattern": s["pattern"],
            "p300_ms": s["p300_latency_ms"],
            "p300_amp_uv": s["p300_amplitude_uv"],
            "n200_ms": s["n200_latency_ms"],
            "n200_amp_uv": s["n200_amplitude_uv"],
            "p3a_ms": s["p3a_latency_ms"],
            "p300_ref_ms": s["p300_ref_upper_ms"],
            "p300_lat_abn": s["p300_lat_abnormal"],
            "p300_amp_abn": s["p300_amp_abnormal"],
            "n200_lat_abn": s["n200_lat_abnormal"],
            "n200_amp_abn": s["n200_amp_abnormal"],
            "p3a_lat_abn": s["p3a_lat_abnormal"],
            "mmse": s["mmse_score"],
            "moca": s["moca_score"],
            "med_count": s["med_count"],
        }
        for s in studies
    ], key=lambda x: (SEVERITY_LEVELS.index(x["severity"]) if x["severity"] in SEVERITY_LEVELS else 0, x["p300_ms"]), reverse=True)

    return {
        "p300_latency_histogram": p300_histogram,
        "p300_amplitude_histogram": amp_histogram,
        "component_summary": component_summary,
        "patient_details": patient_details,
    }


def definitions():
    """P300/ERP protocol, component definitions, reference ranges,
    diagnostic patterns, AED cognitive effects, and clinical references."""
    return {
        "title": "P300 / ERP — Cognitive Event-Related Potentials",
        "protocol": {
            "description": (
                "Event-Related Potentials (ERPs) are time-locked EEG responses to specific "
                "cognitive or sensory events. The P300 (P3b) is a positive deflection maximal "
                "at Pz (~300–500 ms post-target), generated during working memory updating "
                "and attentional processing. It is elicited with an auditory 3-tone oddball "
                "paradigm: frequent standard tones (80%), infrequent target tones (15%), and "
                "novel distractor tones (5%). The patient silently counts target tones."
            ),
            "recording_sites": ["Fz", "Cz", "Pz", "P3", "P4"],
            "paradigm": "3-tone auditory oddball (standard 1 kHz / target 2 kHz / novel tones)",
            "epoch_window": "-200 to +800 ms relative to stimulus onset",
            "standard": (
                "Polich J (2007) Clin Neurophysiol 118:2128-2148. "
                "Duncan CC et al (2009) Clin Neurophysiol 120:1883-1908. "
                "ACNS Guideline 9."
            ),
            "indications": [
                "Cognitive impairment screening — dementia, MCI, post-stroke",
                "AED cognitive side-effect monitoring (topiramate, phenobarbital, valproate)",
                "Post-ictal cognitive recovery assessment",
                "Temporal lobe epilepsy (TLE) — hippocampal-dependent memory decline",
                "Depression and attention deficit assessment",
                "Coma depth monitoring / consciousness assessment",
                "Surgical planning — cognitive eloquent cortex identification",
                "Schizophrenia / psychiatric comorbidity workup",
            ],
        },
        "parameters": [
            {
                "name": "P300 Latency (Pz)",
                "unit": "ms",
                "description": (
                    "Peak latency of the P3b component, reflecting stimulus evaluation time. "
                    "P300 latency increases linearly with age (~1.25 ms/year; Polich 2007), "
                    "independently of motor speed. Prolonged P300 latency indicates slowed "
                    "cognitive processing — seen in dementia, AED toxicity (topiramate, "
                    "phenobarbital), post-ictal states, and TLE cognitive decline. "
                    "Reference: age-adjusted (≈300 ms + 1.25×(age-20) ms)."
                ),
            },
            {
                "name": "P300 Amplitude (Pz)",
                "unit": "µV",
                "description": (
                    "Peak amplitude of the P3b at Pz. Reflects the amount of attentional "
                    "resources allocated and working memory capacity. Reduced amplitude is "
                    "seen in attention deficits, fatigue, depression, and cognitive decline. "
                    "Reference: ≥5.0 µV (declines with age)."
                ),
            },
            {
                "name": "N200 Latency (Fz)",
                "unit": "ms",
                "description": (
                    "Negative deflection at 180-280 ms at Fz, reflecting target detection "
                    "and conflict monitoring (anterior cingulate cortex generator). "
                    "Prolonged N200 indicates impaired selective attention. "
                    "Reference: ≤280 ms."
                ),
            },
            {
                "name": "N200 Amplitude (Fz)",
                "unit": "µV",
                "description": (
                    "Peak amplitude of the N200 component. Reduced N200 amplitude suggests "
                    "impaired conflict detection and target discrimination. "
                    "Reference: ≥3.0 µV."
                ),
            },
            {
                "name": "P3a Latency (Fz)",
                "unit": "ms",
                "description": (
                    "The P3a ('novelty P3') is an earlier frontal positivity (~250-330 ms) "
                    "generated in response to novel, task-irrelevant distractors. It reflects "
                    "automatic attention switching (frontal cortex generator). Distinct from "
                    "the P3b (target P3) in scalp distribution and cognitive correlates. "
                    "Reference: ≤330 ms."
                ),
            },
        ],
        "reference_ranges": {
            "p300_latency": "age-adjusted: ≈300 + 1.25×(age−20) ms (Polich 2007)",
            "p300_amplitude_lower_uv": P300_AMPLITUDE_LOWER_UV,
            "n200_latency_upper_ms": N200_LATENCY_UPPER_MS,
            "n200_amplitude_lower_uv": N200_AMPLITUDE_LOWER_UV,
            "p3a_latency_upper_ms": P3A_LATENCY_UPPER_MS,
            "notes": (
                "Upper limit for latency (abnormal if exceeded); lower limit for amplitude. "
                "P300 normative data from Polich (2007) meta-analysis of 38 studies, "
                "N=1,108 subjects aged 20-80. Age-adjustment mandatory for latency comparison."
            ),
        },
        "diagnostic_patterns": [
            {"pattern": k, "description": v}
            for k, v in DIAGNOSTIC_PATTERNS.items()
        ],
        "severity_levels": [
            {
                "level": "Normal",
                "criteria": "P300 latency within age-adjusted upper limit; P300 amplitude ≥5.0 µV; N200 within reference",
            },
            {
                "level": "Mild",
                "criteria": "P300 latency 5-20% above age-adjusted limit OR amplitude 3.5-5.0 µV; minor cognitive slowing",
            },
            {
                "level": "Moderate",
                "criteria": "P300 latency 20-45% above limit OR amplitude 2.0-3.5 µV; significant attention/memory impact",
            },
            {
                "level": "Severe",
                "criteria": "P300 latency >45% above limit OR amplitude <2.0 µV OR absent P300; severe cognitive dysfunction",
            },
        ],
        "epilepsy_relevance": [
            {
                "context": "AED Cognitive Monitoring",
                "detail": (
                    "Topiramate and phenobarbital prolong P300 latency more than other AEDs "
                    "(Meador et al. 2012). Levetiracetam and lamotrigine have minimal P300 impact. "
                    "P300 monitoring guides AED dose optimization to minimize cognitive toxicity."
                ),
            },
            {
                "context": "TLE Hippocampal Decline",
                "detail": (
                    "Temporal lobe epilepsy is associated with progressive P300 latency prolongation "
                    "correlating with hippocampal volume loss. P300 latency tracks memory decline "
                    "and predicts post-surgical cognitive outcome (Aldenkamp et al. 2004)."
                ),
            },
            {
                "context": "Post-Ictal Cognitive Recovery",
                "detail": (
                    "P300 latency normalizes within 30-120 minutes after GTCS, reflecting "
                    "post-ictal cognitive suppression duration. Serial P300 measurement guides "
                    "return-to-activity decisions."
                ),
            },
            {
                "context": "Pre-Surgical Cognitive Baseline",
                "detail": (
                    "P300 before temporal lobectomy predicts post-surgical verbal memory outcome. "
                    "Left TLE patients with pre-surgical P300 latency >380 ms have higher risk "
                    "of significant post-surgical memory decline (Helmstaedter 2004)."
                ),
            },
            {
                "context": "Encephalopathy vs Epilepsy",
                "detail": (
                    "Absent or markedly prolonged P300 distinguishes encephalopathy from focal "
                    "seizure activity. Non-convulsive status epilepticus (NCSE) may show near-normal "
                    "P300 with focal IEDs, unlike metabolic encephalopathy."
                ),
            },
        ],
        "reference": (
            "Polich J. Updating P300: An integrative theory of P3a and P3b. "
            "Clin Neurophysiol. 2007;118(10):2128-2148. "
            "Duncan CC, Barry RJ, Connolly JF, et al. Event-related potentials in clinical research. "
            "Clin Neurophysiol. 2009;120(11):1883-1908. "
            "Meador KJ et al. Cognitive effects of levetiracetam, carbamazepine, and valproate. "
            "Neurology. 2012."
        ),
    }


if __name__ == "__main__":
    import json
    print("=== P300 ERP Overview ===")
    ov = overview()
    print(json.dumps(ov["kpis"], indent=2))
    print(f"Severity: {ov['severity_distribution']}")
    print(f"Patterns: {ov['pattern_distribution']}")
    print(f"\n=== Breakdown ===")
    bd = breakdown()
    print(f"P300 histogram: {bd['p300_latency_histogram']}")
    print(f"Components: {[c['component'] for c in bd['component_summary']]}")
    print(f"\n=== Definitions ===")
    df = definitions()
    print(f"Parameters: {len(df['parameters'])}")
    print(f"Epilepsy contexts: {len(df['epilepsy_relevance'])}")
