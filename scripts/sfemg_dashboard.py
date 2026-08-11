"""
Single Fiber EMG (SFEMG) Dashboard — NeuroAI EEG
==================================================
SFEMG is the most sensitive clinical test for neuromuscular junction (NMJ)
dysfunction. It measures jitter (variability of neuromuscular transmission)
and fiber density (number of muscle fibers per motor unit).

Key SFEMG Parameters:
  - Jitter (MCD — Mean Consecutive Difference):
      Normal: < 55 µs per fiber pair, mean < 35 µs
      Abnormal: ≥ 55 µs or mean ≥ 35 µs across ≥ 10% of pairs
  - Fiber Density (FD):
      Normal: 1.3–1.8 fiber potentials per motor unit
      Increased: > 1.8 (reinnervation / neurogenic disorders)
  - Blocking:
      Normal: < 10% of impulse pairs show blocking
      Abnormal: ≥ 10% blocking (severe NMJ dysfunction)

Standard Muscles Studied:
  - EDC (Extensor Digitorum Communis) — most common, C7-C8 / Radial nerve
  - Orbicularis Oculi — ocular MG screening
  - Frontalis — ocular/facial MG
  - Deltoid — proximal weakness evaluation

Diagnostic Relevance:
  - Myasthenia Gravis (MG): increased jitter + blocking, FD normal/mildly increased
  - Lambert-Eaton Myasthenic Syndrome (LEMS): increased jitter (decrements on RNS)
  - Congenital Myasthenic Syndrome (CMS): variable jitter patterns
  - Myopathic disorders: increased FD (splitting/reinnervation)
  - Motor Neuron Disease (MND): markedly increased FD + jitter
  - Normal: MCD < 55 µs, FD 1.3–1.8, blocking < 10%

Epilepsy Relevance:
  - Myasthenic crisis can mimic seizure-like events → SFEMG differentiates
  - AEDs (carbamazepine, phenytoin) may affect NMJ function → monitoring
  - Certain channelopathies affect both CNS (epilepsy) and NMJ simultaneously
  - Pre-surgical NMJ baseline before epilepsy surgery (general anesthesia)
  - Neuromuscular fatigue as post-ictal correlate

References:
  Stalberg E, Sanders DB. Jitter recordings with concentric needle electrodes.
    Muscle Nerve. 2009;40(3):331-339.
  AANEM Technology Review: SFEMG. Muscle Nerve. 2001.
  Meriggioli MN, Sanders DB. Autoimmune myasthenia gravis. Lancet Neurol. 2009.

Author: Research Team
"""

import hashlib
from pathlib import Path
from collections import Counter

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# ── Reference ranges (Stalberg & Sanders 2009; AANEM 2001) ───────

JITTER_NORMAL_PAIR = 55.0     # µs per pair
JITTER_NORMAL_MEAN = 35.0     # µs mean
FIBER_DENSITY_LOWER = 1.3
FIBER_DENSITY_UPPER = 1.8
BLOCKING_THRESHOLD = 0.10     # 10%

MUSCLES = [
    {
        "muscle": "Extensor Digitorum Communis (EDC)",
        "nerve": "Radial nerve (C7-C8)",
        "limb": "upper",
        "mcd_normal_mean": 30.0,  # µs
        "mcd_normal_sd": 6.0,
        "fd_mean": 1.5,
        "fd_sd": 0.15,
    },
    {
        "muscle": "Orbicularis Oculi",
        "nerve": "Facial nerve (cranial VII)",
        "limb": "cranial",
        "mcd_normal_mean": 32.0,
        "mcd_normal_sd": 7.0,
        "fd_mean": 1.45,
        "fd_sd": 0.12,
    },
    {
        "muscle": "Frontalis",
        "nerve": "Facial nerve (cranial VII)",
        "limb": "cranial",
        "mcd_normal_mean": 31.0,
        "mcd_normal_sd": 6.5,
        "fd_mean": 1.40,
        "fd_sd": 0.10,
    },
    {
        "muscle": "Deltoid",
        "nerve": "Axillary nerve (C5-C6)",
        "limb": "upper",
        "mcd_normal_mean": 33.0,
        "mcd_normal_sd": 7.0,
        "fd_mean": 1.55,
        "fd_sd": 0.18,
    },
]

DIAGNOSTIC_PATTERNS = {
    "normal":       "Normal — MCD < 55 µs, FD 1.3–1.8, blocking < 10%",
    "mg":           "Myasthenia Gravis — increased jitter (MCD ≥ 55 µs) + blocking ≥ 10%, FD normal",
    "lems":         "Lambert-Eaton MS — variable jitter, correlates with RNS decrement",
    "cms":          "Congenital Myasthenic Syndrome — variable jitter, congenital onset",
    "mnd":          "Motor Neuron Disease — markedly increased FD (≥ 2.5) + high jitter",
    "myopathic":    "Myopathic — increased FD due to fiber splitting/reinnervation, mild jitter elevation",
}

STIMULATION_METHODS = ["Voluntary (concentric needle)", "Electrical stimulation"]

N_PATIENTS = 25
N_STUDIES   = 30   # some patients have follow-up studies


# ── Deterministic synthetic data ──────────────────────────────────

def _hash(seed: str) -> float:
    """Deterministic float 0–1 from a seed string."""
    return int(hashlib.md5(seed.encode()).hexdigest(), 16) % 10_000 / 10_000.0


def _make_studies():
    """Generate 30 SFEMG studies across 25 patients."""
    studies = []
    diagnosis_pool = [
        ("normal",   0.37),   # 11 studies
        ("mg",       0.30),   # 9 studies
        ("lems",     0.10),   # 3 studies
        ("cms",      0.07),   # 2 studies
        ("mnd",      0.10),   # 3 studies
        ("myopathic",0.07),   # 2 studies
    ]

    # Expand pool into weighted list
    dx_list = []
    for dx, frac in diagnosis_pool:
        dx_list.extend([dx] * round(frac * N_STUDIES))
    # Trim / pad to exactly N_STUDIES
    dx_list = (dx_list + dx_list)[:N_STUDIES]

    for study_i in range(N_STUDIES):
        sid   = study_i + 1
        rng   = _hash(f"sfemg_study_{sid}")
        rng2  = _hash(f"sfemg_study2_{sid}")
        rng3  = _hash(f"sfemg_study3_{sid}")

        # Patient assignment (some patients have 2 studies)
        if sid <= N_PATIENTS:
            pid = sid
        else:
            pid = int(rng * (N_PATIENTS - 1)) + 1   # repeat patient

        age  = 25 + int(rng  * 45)
        sex  = "Female" if rng2 > 0.45 else "Male"
        diag = dx_list[study_i % len(dx_list)]

        # Muscle studied
        muscle_i = int(rng3 * len(MUSCLES))
        m        = MUSCLES[muscle_i]

        # Jitter (MCD) and FD depend on diagnosis
        if diag == "normal":
            mcd_mean    = m["mcd_normal_mean"] + (rng - 0.5) * 12
            mcd_abnormal_pct = max(0.0, rng2 * 0.08)   # < 10%
            fd          = m["fd_mean"] + (rng2 - 0.5) * 0.20
            blocking    = max(0.0, rng3 * 0.07)         # < 10%
        elif diag == "mg":
            mcd_mean    = 55 + rng * 60       # 55–115 µs
            mcd_abnormal_pct = 0.20 + rng * 0.50
            fd          = m["fd_mean"] + (rng2 - 0.5) * 0.30
            blocking    = 0.10 + rng3 * 0.40
        elif diag == "lems":
            mcd_mean    = 50 + rng * 50
            mcd_abnormal_pct = 0.15 + rng * 0.35
            fd          = m["fd_mean"] + rng2 * 0.40
            blocking    = 0.08 + rng3 * 0.30
        elif diag == "cms":
            mcd_mean    = 45 + rng * 45
            mcd_abnormal_pct = 0.10 + rng * 0.30
            fd          = m["fd_mean"] + rng2 * 0.20
            blocking    = 0.05 + rng3 * 0.20
        elif diag == "mnd":
            mcd_mean    = 45 + rng * 55
            mcd_abnormal_pct = 0.20 + rng * 0.40
            fd          = 2.2 + rng * 1.2           # markedly elevated
            blocking    = 0.05 + rng3 * 0.25
        else:  # myopathic
            mcd_mean    = 30 + rng * 25
            mcd_abnormal_pct = max(0.0, rng2 * 0.12)
            fd          = 1.9 + rng * 0.60          # increased FD
            blocking    = max(0.0, rng3 * 0.08)

        mcd_mean    = round(max(18.0, mcd_mean), 1)
        fd          = round(max(1.1, min(4.0, fd)), 2)
        blocking    = round(min(1.0, blocking), 3)

        n_pairs     = 15 + int(rng * 6)      # 15–20 fiber pairs recorded
        stim_method = STIMULATION_METHODS[0] if rng < 0.75 else STIMULATION_METHODS[1]

        # Overall abnormality
        jitter_abnormal = mcd_mean >= JITTER_NORMAL_MEAN
        fd_abnormal     = fd > FIBER_DENSITY_UPPER
        blocking_abnormal = blocking >= BLOCKING_THRESHOLD
        is_abnormal     = jitter_abnormal or blocking_abnormal

        studies.append({
            "study_id":           f"SFEMG-{sid:03d}",
            "patient_id":         pid,
            "age":                age,
            "sex":                sex,
            "muscle":             m["muscle"],
            "nerve":              m["nerve"],
            "limb":               m["limb"],
            "stimulation_method": stim_method,
            "n_pairs":            n_pairs,
            "mcd_mean_us":        mcd_mean,
            "mcd_abnormal_pct":   round(mcd_abnormal_pct * 100, 1),
            "fiber_density":      fd,
            "blocking_pct":       round(blocking * 100, 1),
            "jitter_abnormal":    jitter_abnormal,
            "fd_abnormal":        fd_abnormal,
            "blocking_abnormal":  blocking_abnormal,
            "overall_abnormal":   is_abnormal,
            "diagnosis":          diag,
            "diagnosis_label":    DIAGNOSTIC_PATTERNS[diag].split(" — ")[0],
        })

    return studies


_STUDIES_CACHE = None

def _get_studies():
    global _STUDIES_CACHE
    if _STUDIES_CACHE is None:
        _STUDIES_CACHE = _make_studies()
    return _STUDIES_CACHE


# ── Public API functions ──────────────────────────────────────────

def overview() -> dict:
    """SFEMG overview KPIs — jitter distributions, abnormality rates,
    muscle summary, diagnosis pattern breakdown, per-patient summary."""
    studies = _get_studies()
    n = len(studies)
    n_abnormal = sum(1 for s in studies if s["overall_abnormal"])
    pct_abnormal = round(n_abnormal / n * 100, 1)

    # MCD distribution buckets (µs)
    mcd_buckets = {"< 35": 0, "35–55": 0, "55–80": 0, "80–120": 0, "> 120": 0}
    for s in studies:
        mcd = s["mcd_mean_us"]
        if mcd < 35:
            mcd_buckets["< 35"] += 1
        elif mcd < 55:
            mcd_buckets["35–55"] += 1
        elif mcd < 80:
            mcd_buckets["55–80"] += 1
        elif mcd < 120:
            mcd_buckets["80–120"] += 1
        else:
            mcd_buckets["> 120"] += 1

    # FD distribution buckets
    fd_buckets = {"< 1.3": 0, "1.3–1.8 (normal)": 0, "1.8–2.5": 0, "> 2.5": 0}
    for s in studies:
        fd = s["fiber_density"]
        if fd < 1.3:
            fd_buckets["< 1.3"] += 1
        elif fd <= 1.8:
            fd_buckets["1.3–1.8 (normal)"] += 1
        elif fd <= 2.5:
            fd_buckets["1.8–2.5"] += 1
        else:
            fd_buckets["> 2.5"] += 1

    # Diagnosis distribution
    diag_counts = Counter(s["diagnosis_label"] for s in studies)

    # Muscle abnormality rates
    muscle_rates = {}
    for m in MUSCLES:
        mu = m["muscle"]
        subset = [s for s in studies if s["muscle"] == mu]
        if subset:
            muscle_rates[mu] = {
                "n_studies": len(subset),
                "pct_abnormal": round(sum(1 for s in subset if s["overall_abnormal"]) / len(subset) * 100, 1),
                "mean_mcd_us": round(sum(s["mcd_mean_us"] for s in subset) / len(subset), 1),
                "mean_fd": round(sum(s["fiber_density"] for s in subset) / len(subset), 2),
            }

    # Per-patient summary
    pid_to_studies = {}
    for s in studies:
        pid_to_studies.setdefault(s["patient_id"], []).append(s)

    patient_summary = []
    for pid, pst in sorted(pid_to_studies.items()):
        worst = max(pst, key=lambda x: x["mcd_mean_us"])
        patient_summary.append({
            "patient_id":   pid,
            "n_studies":    len(pst),
            "sex":          pst[0]["sex"],
            "age":          pst[0]["age"],
            "max_mcd_us":   worst["mcd_mean_us"],
            "max_fd":       max(s["fiber_density"] for s in pst),
            "max_blocking_pct": max(s["blocking_pct"] for s in pst),
            "overall_abnormal": any(s["overall_abnormal"] for s in pst),
            "primary_diagnosis": worst["diagnosis_label"],
        })

    blocking_abnormal_n = sum(1 for s in studies if s["blocking_abnormal"])
    fd_abnormal_n       = sum(1 for s in studies if s["fd_abnormal"])
    mean_mcd = round(sum(s["mcd_mean_us"] for s in studies) / n, 1)
    mean_fd  = round(sum(s["fiber_density"] for s in studies) / n, 2)
    mean_blocking = round(sum(s["blocking_pct"] for s in studies) / n, 1)

    return {
        "kpis": {
            "total_studies": n,
            "total_patients": len(pid_to_studies),
            "pct_abnormal": pct_abnormal,
            "n_abnormal": n_abnormal,
            "mean_mcd_us": mean_mcd,
            "mean_fiber_density": mean_fd,
            "mean_blocking_pct": mean_blocking,
            "blocking_abnormal_n": blocking_abnormal_n,
            "fd_abnormal_n": fd_abnormal_n,
        },
        "mcd_distribution": mcd_buckets,
        "fd_distribution": fd_buckets,
        "diagnosis_distribution": dict(diag_counts),
        "muscle_summary": muscle_rates,
        "patient_summary": patient_summary,
    }


def breakdown() -> dict:
    """Per-study detail: muscle analysis, jitter/FD scatter data,
    blocking vs MCD comparison, per-diagnosis jitter profiles."""
    studies = _get_studies()

    # Jitter vs blocking scatter (each study = one point)
    scatter = [
        {
            "study_id":    s["study_id"],
            "patient_id":  s["patient_id"],
            "mcd_mean_us": s["mcd_mean_us"],
            "blocking_pct": s["blocking_pct"],
            "fiber_density": s["fiber_density"],
            "diagnosis":   s["diagnosis_label"],
            "abnormal":    s["overall_abnormal"],
            "muscle":      s["muscle"],
        }
        for s in studies
    ]

    # Per-diagnosis mean jitter, FD, blocking
    dx_profiles = {}
    for s in studies:
        key = s["diagnosis_label"]
        dx_profiles.setdefault(key, {"mcd": [], "fd": [], "blocking": []})
        dx_profiles[key]["mcd"].append(s["mcd_mean_us"])
        dx_profiles[key]["fd"].append(s["fiber_density"])
        dx_profiles[key]["blocking"].append(s["blocking_pct"])

    dx_summary = []
    for dx, vals in dx_profiles.items():
        n = len(vals["mcd"])
        dx_summary.append({
            "diagnosis":      dx,
            "n":              n,
            "mean_mcd_us":    round(sum(vals["mcd"]) / n, 1),
            "mean_fd":        round(sum(vals["fd"]) / n, 2),
            "mean_blocking_pct": round(sum(vals["blocking"]) / n, 1),
        })
    dx_summary.sort(key=lambda x: x["mean_mcd_us"], reverse=True)

    # Full study log (for Per Study tab)
    study_log = [
        {
            "study_id":           s["study_id"],
            "patient_id":         s["patient_id"],
            "sex":                s["sex"],
            "age":                s["age"],
            "muscle":             s["muscle"],
            "nerve":              s["nerve"],
            "stimulation_method": s["stimulation_method"],
            "n_pairs":            s["n_pairs"],
            "mcd_mean_us":        s["mcd_mean_us"],
            "mcd_abnormal_pct":   s["mcd_abnormal_pct"],
            "fiber_density":      s["fiber_density"],
            "blocking_pct":       s["blocking_pct"],
            "jitter_abnormal":    s["jitter_abnormal"],
            "fd_abnormal":        s["fd_abnormal"],
            "blocking_abnormal":  s["blocking_abnormal"],
            "overall_abnormal":   s["overall_abnormal"],
            "diagnosis":          s["diagnosis_label"],
        }
        for s in studies
    ]

    return {
        "scatter": scatter,
        "diagnosis_profiles": dx_summary,
        "study_log": study_log,
    }


def definitions() -> dict:
    """SFEMG definitions — glossary, reference values, diagnostic criteria,
    epilepsy relevance, contraindications, and key references."""
    return {
        "title": "Single Fiber EMG (SFEMG) — Definitions & Clinical Context",
        "overview": (
            "SFEMG is the most sensitive bedside test for NMJ dysfunction, detecting "
            "abnormal jitter in >95% of generalized MG cases. It uses a specialized "
            "needle electrode to record individual muscle fiber action potentials within "
            "a single motor unit."
        ),
        "parameters": [
            {
                "name": "Jitter (MCD — Mean Consecutive Difference)",
                "unit": "µs",
                "normal": "< 55 µs per pair; mean < 35 µs",
                "definition": (
                    "Variability in the time interval between discharges of two muscle "
                    "fibers in the same motor unit. Elevated jitter = impaired NMJ safety "
                    "factor (reduced quantal content or altered AChR sensitivity)."
                ),
            },
            {
                "name": "Fiber Density (FD)",
                "unit": "fiber potentials / motor unit",
                "normal": "1.3–1.8",
                "definition": (
                    "Average number of single muscle fiber action potentials per motor unit. "
                    "Elevated FD reflects fiber splitting (myopathic) or reinnervation "
                    "(neurogenic). Useful to distinguish NMJ from muscle/nerve disease."
                ),
            },
            {
                "name": "Blocking",
                "unit": "% of pairs",
                "normal": "< 10%",
                "definition": (
                    "Failure of a muscle fiber to discharge on a given impulse due to "
                    "severe NMJ dysfunction. Blocking > 10% correlates with clinical "
                    "weakness and is pathognomonic of moderate-severe NMJ disorders."
                ),
            },
        ],
        "diagnostic_patterns": [
            {"pattern": p.split(" — ")[0], "description": p, "key": k}
            for k, p in DIAGNOSTIC_PATTERNS.items()
        ],
        "muscles": MUSCLES,
        "stimulation_methods": [
            {
                "method": "Voluntary (concentric needle)",
                "description": "Patient activates muscle at low effort; most physiological, preferred for cooperative patients.",
                "advantage": "Assesses true volitional motor units",
            },
            {
                "method": "Electrical stimulation",
                "description": "Motor nerve stimulated electrically; used when patient cannot cooperate (sedated, very weak).",
                "advantage": "Reproducible, less patient-dependent",
            },
        ],
        "epilepsy_relevance": [
            {
                "context": "Myasthenic crisis vs seizure",
                "detail": "Bulbar weakness and respiratory fatigue in MG can be mistaken for post-ictal state; SFEMG differentiates.",
            },
            {
                "context": "AED-induced NMJ effects",
                "detail": "Carbamazepine and phenytoin (sodium channel blockers) may mildly impair NMJ safety factor; SFEMG monitors subclinical change.",
            },
            {
                "context": "Channelopathy overlap",
                "detail": "SCN1A/KCNQ2 mutations can coexist with NMJ channelopathies; SFEMG helps delineate the peripheral neuromuscular contribution.",
            },
            {
                "context": "Pre-surgical NMJ baseline",
                "detail": "Before epilepsy surgery under general anesthesia (neuromuscular blockade used), baseline SFEMG establishes pre-operative NMJ function.",
            },
            {
                "context": "Post-ictal motor correlates",
                "detail": "Sustained post-ictal paralysis (Todd's paresis) may have a partial NMJ fatigue contribution detectable by SFEMG.",
            },
        ],
        "reference_values": {
            "jitter_normal_pair_us": JITTER_NORMAL_PAIR,
            "jitter_normal_mean_us": JITTER_NORMAL_MEAN,
            "fiber_density_normal_lower": FIBER_DENSITY_LOWER,
            "fiber_density_normal_upper": FIBER_DENSITY_UPPER,
            "blocking_threshold_pct": BLOCKING_THRESHOLD * 100,
        },
        "key_references": [
            "Stalberg E, Sanders DB. Jitter recordings with concentric needle electrodes. Muscle Nerve. 2009;40(3):331-339.",
            "AANEM Technology Review: Single Fiber EMG. Muscle Nerve. 2001;24(9):1228-1235.",
            "Meriggioli MN, Sanders DB. Autoimmune myasthenia gravis. Lancet Neurol. 2009;8(5):475-490.",
            "Sanders DB, Stalberg EV. AAEM minimonograph #25: Single fiber electromyography. Muscle Nerve. 1996.",
            "Vincent A, et al. Seronegative MG. Brain. 2012;135(9):2823-2831.",
        ],
    }
