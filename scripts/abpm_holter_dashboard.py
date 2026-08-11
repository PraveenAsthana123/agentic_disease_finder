"""ABPM/Holter Cardiac Monitoring Dashboard — 24h ambulatory BP + Holter ECG data.

Provides 24h blood pressure profiles (systolic/diastolic day/night), dipping
category classification, BP pattern labels, cardiac arrhythmia burden (AF, VT,
PVC, SVT, bradycardia), QTc monitoring, and per-patient cardiac risk scores —
all drawn from the abpm_holter_studies table (23 records, 23 patients).

Clinical rationale:
- Cardiac arrhythmias (especially AF and VT) can mimic epileptic seizures or
  trigger syncope; ABPM/Holter is used to differentiate syncope from epilepsy
  (ILAE differential diagnosis workup).
- Many AEDs (carbamazepine, lamotrigine, phenytoin) prolong QTc; monitoring
  QTc > 450 ms (men) / > 470 ms (women) is mandatory for patient safety
  (ESC Guidelines 2022).
- Non-dipping and reverse-dipping patterns (>10% night/day SBP ratio) are
  associated with increased cardiovascular and cerebrovascular events in
  epilepsy patients (Tanabe et al., Epilepsia 2017).
- Nocturnal hypertension is an independent risk factor for SUDEP (Hermann 2019).

Sources:
  abpm_holter_studies table (clinical.db) — 23 rows, direct column layout
  patients table            (clinical.db) — demographics cross-reference
"""

import pathlib
import sqlite3
import statistics
from collections import Counter, defaultdict

DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    return con


def _load_rows():
    con = _conn()
    c = con.cursor()
    c.execute("SELECT * FROM abpm_holter_studies ORDER BY id")
    raw = c.fetchall()
    con.close()
    return [dict(r) for r in raw]


def _avg(vals):
    valid = [v for v in vals if v is not None]
    return round(statistics.mean(valid), 1) if valid else None


def _qtc_category(qtc):
    if qtc is None:
        return "Unknown"
    if qtc < 440:
        return "Normal (<440 ms)"
    if qtc < 470:
        return "Borderline (440-469 ms)"
    return "Prolonged (≥470 ms)"


def _dip_color(cat):
    return {
        "normal_dipper": "#22c55e",
        "extreme_dipper": "#f59e0b",
        "non_dipper": "#ef4444",
        "reverse_dipper": "#dc2626",
    }.get(cat, "#94a3b8")


def _severity_color(sev):
    return {
        "Normal": "#22c55e",
        "Mild": "#f59e0b",
        "Moderate": "#ef4444",
        "Severe": "#dc2626",
    }.get(sev, "#94a3b8")


def overview():
    data = _load_rows()
    n = len(data)

    # KPIs
    abnormal_count = sum(1 for d in data if d.get("is_abnormal"))
    af_patients = sum(1 for d in data if (d.get("af_episodes") or 0) > 0)
    vt_patients = sum(1 for d in data if (d.get("vt_runs") or 0) > 0)
    brady_patients = sum(1 for d in data if (d.get("bradycardia_episodes") or 0) > 0)
    st_patients = sum(1 for d in data if (d.get("st_depression_events") or 0) > 0)
    total_pvc = sum(d.get("pvc_count") or 0 for d in data)

    # Averages
    avg_sbp = _avg([d.get("systolic_24h") for d in data])
    avg_dbp = _avg([d.get("diastolic_24h") for d in data])
    avg_hr = _avg([d.get("heart_rate_24h") for d in data])
    avg_qtc = _avg([d.get("qtc_ms") for d in data])
    avg_cardiac_score = _avg([d.get("cardiac_score") for d in data])

    # Dipping distribution
    dipping_dist = Counter(d.get("dipping_category") for d in data if d.get("dipping_category"))
    dipping_list = [
        {
            "category": k,
            "label": k.replace("_", " ").title(),
            "count": v,
            "color": _dip_color(k),
        }
        for k, v in dipping_dist.most_common()
    ]

    # Severity distribution
    severity_dist = Counter(d.get("severity") for d in data if d.get("severity"))
    severity_list = [
        {
            "severity": k,
            "count": v,
            "color": _severity_color(k),
        }
        for k, v in severity_dist.most_common()
    ]

    # BP pattern distribution
    pattern_dist = Counter(d.get("pattern_label") for d in data if d.get("pattern_label"))
    pattern_list = [
        {"pattern": k.replace("_", " ").title(), "count": v}
        for k, v in pattern_dist.most_common()
    ]

    # QTc distribution
    qtc_dist = Counter(_qtc_category(d.get("qtc_ms")) for d in data)
    qtc_list = [
        {"bucket": k, "count": v}
        for k, v in sorted(qtc_dist.items())
    ]

    # Non-dipper rate (non_dipper + reverse_dipper)
    adverse_dipping = sum(1 for d in data
                          if d.get("dipping_category") in ("non_dipper", "reverse_dipper"))

    return {
        "total_studies": n,
        "total_patients": len({d["patient_id"] for d in data}),
        "abnormal_count": abnormal_count,
        "adverse_dipping_count": adverse_dipping,
        "adverse_dipping_pct": round(adverse_dipping / n * 100, 1) if n else 0,
        "af_patients": af_patients,
        "vt_patients": vt_patients,
        "brady_patients": brady_patients,
        "st_depression_patients": st_patients,
        "total_pvc": total_pvc,
        "avg_sbp_24h": avg_sbp,
        "avg_dbp_24h": avg_dbp,
        "avg_hr_24h": avg_hr,
        "avg_qtc_ms": avg_qtc,
        "avg_cardiac_score": avg_cardiac_score,
        "dipping_distribution": dipping_list,
        "severity_distribution": severity_list,
        "bp_pattern_distribution": pattern_list,
        "qtc_distribution": qtc_list,
    }


def breakdown():
    data = _load_rows()
    n = len(data)

    # Arrhythmia burden table
    arrhythmia_summary = [
        {"type": "Atrial Fibrillation (AF)", "key": "af_episodes",
         "total_events": sum(d.get("af_episodes") or 0 for d in data),
         "patients_affected": sum(1 for d in data if (d.get("af_episodes") or 0) > 0)},
        {"type": "Ventricular Tachycardia (VT)", "key": "vt_runs",
         "total_events": sum(d.get("vt_runs") or 0 for d in data),
         "patients_affected": sum(1 for d in data if (d.get("vt_runs") or 0) > 0)},
        {"type": "Premature Ventricular Contractions (PVC)", "key": "pvc_count",
         "total_events": sum(d.get("pvc_count") or 0 for d in data),
         "patients_affected": sum(1 for d in data if (d.get("pvc_count") or 0) > 0)},
        {"type": "Supraventricular Tachycardia (SVT)", "key": "svt_episodes",
         "total_events": sum(d.get("svt_episodes") or 0 for d in data),
         "patients_affected": sum(1 for d in data if (d.get("svt_episodes") or 0) > 0)},
        {"type": "Bradycardia Episodes", "key": "bradycardia_episodes",
         "total_events": sum(d.get("bradycardia_episodes") or 0 for d in data),
         "patients_affected": sum(1 for d in data if (d.get("bradycardia_episodes") or 0) > 0)},
        {"type": "ST Depression Events", "key": "st_depression_events",
         "total_events": sum(d.get("st_depression_events") or 0 for d in data),
         "patients_affected": sum(1 for d in data if (d.get("st_depression_events") or 0) > 0)},
    ]

    # Per-patient table sorted by cardiac_score descending
    patients = []
    for d in sorted(data, key=lambda x: -(x.get("cardiac_score") or 0)):
        patients.append({
            "patient_id": d.get("patient_id"),
            "study_date": d.get("study_date"),
            "systolic_24h": d.get("systolic_24h"),
            "diastolic_24h": d.get("diastolic_24h"),
            "heart_rate_24h": d.get("heart_rate_24h"),
            "qtc_ms": d.get("qtc_ms"),
            "dipping_category": (d.get("dipping_category") or "").replace("_", " ").title(),
            "pattern_label": (d.get("pattern_label") or "").replace("_", " ").title(),
            "severity": d.get("severity"),
            "cardiac_score": d.get("cardiac_score"),
            "is_abnormal": bool(d.get("is_abnormal")),
            "af_episodes": d.get("af_episodes"),
            "vt_runs": d.get("vt_runs"),
            "pvc_count": d.get("pvc_count"),
            "bradycardia_episodes": d.get("bradycardia_episodes"),
            "st_depression_events": d.get("st_depression_events"),
            "dipping_pct": d.get("dipping_pct"),
            "pulse_pressure": d.get("pulse_pressure"),
            "map_24h": d.get("map_24h"),
        })

    # Day vs night BP comparison by dipping category
    dipping_bp = defaultdict(lambda: {"day_sbp": [], "night_sbp": [], "count": 0})
    for d in data:
        cat = (d.get("dipping_category") or "unknown").replace("_", " ").title()
        dipping_bp[cat]["day_sbp"].append(d.get("systolic_day") or 0)
        dipping_bp[cat]["night_sbp"].append(d.get("systolic_night") or 0)
        dipping_bp[cat]["count"] += 1

    dipping_bp_comparison = []
    for cat, vals in dipping_bp.items():
        dipping_bp_comparison.append({
            "category": cat,
            "count": vals["count"],
            "avg_day_sbp": round(statistics.mean(vals["day_sbp"]), 1) if vals["day_sbp"] else None,
            "avg_night_sbp": round(statistics.mean(vals["night_sbp"]), 1) if vals["night_sbp"] else None,
        })

    return {
        "arrhythmia_summary": arrhythmia_summary,
        "dipping_bp_comparison": sorted(dipping_bp_comparison, key=lambda x: -x["count"]),
        "patients": patients,
    }


def definitions():
    return {
        "dashboard": "ABPM/Holter Cardiac Monitoring Dashboard",
        "data_source": "abpm_holter_studies table (clinical.db) — 23 records, 23 patients",
        "terms": [
            {
                "term": "ABPM (Ambulatory Blood Pressure Monitoring)",
                "definition": (
                    "Continuous non-invasive blood pressure recording over 24 hours using "
                    "an automated cuff. Captures day/night BP variability, dipping pattern, "
                    "and masked/white-coat hypertension not seen in clinic. Standard for "
                    "epilepsy patients on AEDs with cardiovascular risk (ESC 2018)."
                ),
            },
            {
                "term": "Holter Monitor",
                "definition": (
                    "Continuous 24-48h ECG recording for arrhythmia detection. In epilepsy, "
                    "used to differentiate cardiac syncope (AF, VT, bradycardia) from "
                    "epileptic seizures — a critical differential since 15-20% of patients "
                    "referred to epilepsy clinics have primary cardiac disease."
                ),
            },
            {
                "term": "Dipping Pattern",
                "definition": (
                    "Normal dipper: ≥10% night/day SBP drop (physiological); "
                    "Non-dipper: <10% drop (elevated cardiovascular risk); "
                    "Extreme dipper: >20% drop (risk of nocturnal cerebral hypoperfusion); "
                    "Reverse dipper: night SBP > day SBP (highest stroke/SUDEP risk)."
                ),
            },
            {
                "term": "QTc (Corrected QT Interval)",
                "definition": (
                    "Heart-rate-corrected QT interval (Bazett formula). Normal: <440 ms (men), "
                    "<460 ms (women). Borderline: 440-469 ms. Prolonged (≥470 ms): risk of "
                    "Torsades de Pointes. Carbamazepine, lamotrigine, and phenytoin can all "
                    "prolong QTc — mandatory monitoring per ESC 2022 guidelines."
                ),
            },
            {
                "term": "Cardiac Risk Score",
                "definition": (
                    "Composite score (0-100) derived from BP severity, arrhythmia burden, "
                    "QTc prolongation, dipping pattern, and ST depression events. Higher "
                    "scores indicate greater cardiovascular risk requiring cardiology referral."
                ),
            },
            {
                "term": "AF (Atrial Fibrillation)",
                "definition": (
                    "Irregular rapid atrial rhythm — most common sustained arrhythmia. "
                    "Can cause cardiac syncope mimicking seizure. Prevalence 3-5% in "
                    "epilepsy cohorts vs 1-2% general population. Anticoagulation "
                    "decision intersects with AED therapy."
                ),
            },
            {
                "term": "PVC (Premature Ventricular Contractions)",
                "definition": (
                    "Ectopic ventricular beats. Frequent PVCs (>10,000/24h or >10% of beats) "
                    "may cause PVC-induced cardiomyopathy. Isolated PVCs <500/24h are "
                    "generally benign. Phenytoin has historically been used as an "
                    "antiarrhythmic for PVCs."
                ),
            },
            {
                "term": "Masked Hypertension",
                "definition": (
                    "Normal clinic BP but elevated ambulatory BP (≥135/85 mmHg daytime). "
                    "Prevalence ~15% in epilepsy. Associated with target organ damage "
                    "comparable to sustained hypertension — only detectable by ABPM."
                ),
            },
            {
                "term": "SUDEP Cardiac Link",
                "definition": (
                    "Post-ictal cardiac arrhythmias (asystole, bradycardia, AF) are implicated "
                    "in SUDEP mechanism. Nocturnal hypertension and non-dipping increase "
                    "autonomic instability risk. Holter monitoring post-generalized seizure "
                    "is recommended by the ILAE SUDEP taskforce (Devinsky et al., 2016)."
                ),
            },
            {
                "term": "Standards",
                "definition": (
                    "ESC/ESH Arterial Hypertension Guidelines (2018 + 2022 update); "
                    "AHA/ACC Holter Monitoring Appropriate Use Criteria; "
                    "ILAE Commission on Therapeutic Strategies — cardiac comorbidity; "
                    "ACNS Clinical Guideline on Cardiac Monitoring in Epilepsy."
                ),
            },
        ],
        "abbreviations": {
            "ABPM": "Ambulatory Blood Pressure Monitoring",
            "AED": "Anti-Epileptic Drug",
            "AF": "Atrial Fibrillation",
            "Brady": "Bradycardia",
            "DBP": "Diastolic Blood Pressure",
            "ESC": "European Society of Cardiology",
            "ILAE": "International League Against Epilepsy",
            "MAP": "Mean Arterial Pressure",
            "PVC": "Premature Ventricular Contraction",
            "QTc": "Corrected QT Interval",
            "SBP": "Systolic Blood Pressure",
            "ST": "ST Segment (ECG)",
            "SUDEP": "Sudden Unexpected Death in Epilepsy",
            "SVT": "Supraventricular Tachycardia",
            "VT": "Ventricular Tachycardia",
        },
    }
