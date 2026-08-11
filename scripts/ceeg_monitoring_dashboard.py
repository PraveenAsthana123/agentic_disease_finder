"""Continuous EEG (cEEG) ICU Monitoring Dashboard — real clinical.db data.

Continuous EEG monitoring is the gold standard for detecting non-convulsive
seizures (NCS) and non-convulsive status epilepticus (NCSE) in ICU patients
with altered consciousness.  ACNS 2021 guidelines recommend cEEG for ≥ 24 h
in high-risk ICU patients (post-cardiac arrest, SAH, TBI, SE).

cEEG detects subclinical seizures in 8-48 % of ICU patients — most of which
would be missed without continuous monitoring (Claassen et al., NEJM 2004).
Critical patterns: Periodic Lateralized Discharges (PLDs/LPDs), GRDA,
Ictal-interictal continuum (IIC), NCSE.

Data sources (clinical.db):
  eeg_acquisition     (30 rows)  — recording_type (LTM/ambulatory/routine/video_eeg),
                                   duration_min, sampling_rate, montage, study_date
  hospitalization     (115 rows) — ward (ICU=15), admission_reason, LOS, complications
  seizure_metadata    (71 rows)  — eeg_pattern (PLDs, hypsarrhythmia, etc.), syndrome,
                                   etiology, drug_responsiveness
  artifact_annotations(169 rows) — artifact_type, severity, channel, duration_sec
  patients            (41 rows)  — age, gender

All computations: raw SQL + Python stdlib only (no pandas, no numpy).
"""

import json
import pathlib
import sqlite3
from collections import Counter, defaultdict
from typing import Dict, List

DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    return con


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_eeg_acquisitions() -> List[dict]:
    con = _conn()
    rows = con.execute(
        "SELECT patient_id, fields_json FROM eeg_acquisition ORDER BY id"
    ).fetchall()
    con.close()
    records = []
    for r in rows:
        try:
            d = json.loads(r["fields_json"])
            records.append({
                "patient_id": r["patient_id"],
                "recording_type": (d.get("recording_type") or "routine").lower(),
                "duration_min": float(d.get("duration_min") or 0),
                "sampling_rate": int(d.get("sampling_rate") or 256),
                "montage": (d.get("montage") or "referential").lower(),
                "electrode_system": d.get("electrode_system", "10-20"),
                "technician_notes": d.get("technician_notes", ""),
                "study_date": d.get("study_date", ""),
            })
        except Exception:
            pass
    return records


def _load_hospitalizations() -> List[dict]:
    con = _conn()
    rows = con.execute(
        "SELECT patient_id, fields_json FROM hospitalization ORDER BY id"
    ).fetchall()
    con.close()
    records = []
    for r in rows:
        try:
            d = json.loads(r["fields_json"])
            records.append({
                "patient_id": r["patient_id"],
                "ward": (d.get("ward") or "Unknown"),
                "admission_reason": (d.get("admission_reason") or "unknown").lower(),
                "admission_type": (d.get("admission_type") or "planned").lower(),
                "los_days": float(d.get("length_of_stay_days") or 1),
                "discharge_disposition": (d.get("discharge_disposition") or "home").lower(),
                "seizure_free_at_discharge": bool(d.get("seizure_free_at_discharge", False)),
                "readmission_30d": bool(d.get("readmission_within_30d", False)),
                "complications": d.get("complications"),
                "total_cost_usd": float(d.get("total_cost_usd") or 0),
            })
        except Exception:
            pass
    return records


def _load_seizure_meta() -> Dict[str, dict]:
    con = _conn()
    rows = con.execute(
        "SELECT patient_id, fields_json FROM seizure_metadata ORDER BY id"
    ).fetchall()
    con.close()
    result: Dict[str, dict] = {}
    for r in rows:
        try:
            d = json.loads(r["fields_json"])
            result[r["patient_id"]] = {
                "eeg_pattern": d.get("eeg_pattern", ""),
                "syndrome": d.get("syndrome", ""),
                "etiology": (d.get("etiology") or "unknown").lower(),
                "drug_resp": (d.get("drug_responsiveness") or "").lower(),
                "onset_zone": d.get("onset_zone", ""),
                "lateralization": d.get("lateralization", ""),
                "seizure_types": d.get("ilae_seizure_types", []),
                "seizure_freq": d.get("current_seizure_frequency", ""),
            }
        except Exception:
            pass
    return result


def _load_artifacts() -> List[dict]:
    con = _conn()
    rows = con.execute(
        "SELECT patient_id, fields_json FROM artifact_annotations ORDER BY id"
    ).fetchall()
    con.close()
    records = []
    for r in rows:
        try:
            d = json.loads(r["fields_json"])
            records.append({
                "patient_id": r["patient_id"],
                "artifact_type": d.get("artifact_type", "unknown"),
                "channel": d.get("channel", ""),
                "duration_sec": float(d.get("duration_sec") or 0),
                "severity": (d.get("severity") or "mild").lower(),
            })
        except Exception:
            pass
    return records


def _load_patients() -> Dict[str, dict]:
    con = _conn()
    rows = con.execute(
        "SELECT patient_id, age, sex FROM patient_demographics ORDER BY id"
    ).fetchall()
    con.close()
    result: Dict[str, dict] = {}
    for r in rows:
        try:
            raw = dict(r)
            result[raw["patient_id"]] = {
                "age": raw.get("age"),
                "gender": raw.get("sex", ""),
            }
        except Exception:
            pass
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

_CONTINUOUS_TYPES = {"ltm", "ambulatory"}  # long-term monitoring = continuous EEG

_CRITICAL_PATTERNS = {
    "Periodic Lateralized Discharges",
    "Generalized spike-and-wave <3 Hz",
    "Hypsarrhythmia",
}


def _is_critical_pattern(pattern: str) -> bool:
    return any(cp.lower() in pattern.lower() for cp in _CRITICAL_PATTERNS)


def _monitoring_tier(rec_type: str, duration_min: float) -> str:
    if rec_type == "ltm":
        return "Long-Term cEEG (≥8 h)"
    if rec_type == "ambulatory":
        return "Ambulatory cEEG"
    if duration_min >= 60:
        return "Extended Routine (≥1 h)"
    return "Routine EEG (<1 h)"


# ── Public API ────────────────────────────────────────────────────────────────

def overview() -> dict:
    """cEEG monitoring KPIs, recording-type distribution, ICU monitoring stats,
    EEG pattern landscape, and artifact burden summary."""
    acqs = _load_eeg_acquisitions()
    hosps = _load_hospitalizations()
    meta = _load_seizure_meta()
    artifacts = _load_artifacts()

    # Recording type distribution
    type_ctr: Counter = Counter(a["recording_type"] for a in acqs)
    continuous_recs = [a for a in acqs if a["recording_type"] in _CONTINUOUS_TYPES]
    total_recs = len(acqs)

    # Duration stats
    durations = [a["duration_min"] for a in acqs if a["duration_min"] > 0]
    avg_duration_h = round(sum(durations) / len(durations) / 60, 1) if durations else 0
    max_duration_h = round(max(durations) / 60, 1) if durations else 0
    total_monitoring_h = round(sum(durations) / 60, 1)

    # LTM-specific stats
    ltm_recs = [a for a in acqs if a["recording_type"] == "ltm"]
    ltm_avg_h = round(
        sum(a["duration_min"] for a in ltm_recs) / len(ltm_recs) / 60, 1
    ) if ltm_recs else 0

    # ICU admissions
    icu_hosps = [h for h in hosps if h["ward"] == "ICU"]
    icu_n = len(icu_hosps)
    icu_avg_los = round(
        sum(h["los_days"] for h in icu_hosps) / icu_n, 1
    ) if icu_hosps else 0
    icu_seizure_free = sum(1 for h in icu_hosps if h["seizure_free_at_discharge"])
    icu_readmit = sum(1 for h in icu_hosps if h["readmission_30d"])

    # EEG pattern distribution (all patients)
    pattern_ctr: Counter = Counter(
        m["eeg_pattern"] for m in meta.values() if m["eeg_pattern"]
    )
    critical_n = sum(
        1 for m in meta.values() if _is_critical_pattern(m.get("eeg_pattern", ""))
    )

    # Periodic Lateralized Discharges — key ICU cEEG finding
    pld_n = sum(
        1 for m in meta.values()
        if "periodic lateralized" in m.get("eeg_pattern", "").lower()
    )

    # Artifact burden by type
    artifact_type_ctr: Counter = Counter(a["artifact_type"] for a in artifacts)
    severe_artifacts = sum(1 for a in artifacts if a["severity"] == "severe")
    moderate_artifacts = sum(1 for a in artifacts if a["severity"] == "moderate")

    # Sampling rate distribution
    sr_ctr: Counter = Counter(a["sampling_rate"] for a in acqs)

    # Montage distribution
    montage_ctr: Counter = Counter(a["montage"] for a in acqs)

    return {
        "kpis": {
            "total_recordings": total_recs,
            "continuous_ltm_recordings": len(ltm_recs),
            "ambulatory_recordings": type_ctr.get("ambulatory", 0),
            "icu_admissions": icu_n,
            "avg_monitoring_hours": avg_duration_h,
            "ltm_avg_hours": ltm_avg_h,
            "max_session_hours": max_duration_h,
            "total_monitoring_hours": total_monitoring_h,
            "critical_eeg_patterns": critical_n,
            "pld_cases": pld_n,
            "icu_seizure_free_discharge": icu_seizure_free,
            "icu_readmission_30d": icu_readmit,
        },
        "recording_type_distribution": [
            {"type": t.upper() if len(t) <= 4 else t.title(), "count": c}
            for t, c in sorted(type_ctr.items(), key=lambda x: -x[1])
        ],
        "monitoring_tier_distribution": [
            {"tier": _monitoring_tier(a["recording_type"], a["duration_min"]), "count": 1}
            for a in acqs
        ],
        "icu_admission_breakdown": [
            {"reason": r.replace("_", " ").title(), "count": c}
            for r, c in sorted(
                Counter(h["admission_reason"] for h in icu_hosps).items(),
                key=lambda x: -x[1]
            )
        ],
        "eeg_pattern_landscape": [
            {
                "pattern": p,
                "count": c,
                "critical": _is_critical_pattern(p),
            }
            for p, c in sorted(pattern_ctr.items(), key=lambda x: -x[1])
        ],
        "artifact_burden": [
            {"artifact_type": t.replace("_", " ").title(), "count": c}
            for t, c in sorted(artifact_type_ctr.items(), key=lambda x: -x[1])
        ],
        "sampling_rate_distribution": [
            {"rate_hz": f"{sr} Hz", "count": c}
            for sr, c in sorted(sr_ctr.items(), key=lambda x: -x[1])
        ],
        "montage_distribution": [
            {"montage": m.title(), "count": c}
            for m, c in sorted(montage_ctr.items(), key=lambda x: -x[1])
        ],
        "severity_summary": {
            "total_artifacts": len(artifacts),
            "severe": severe_artifacts,
            "moderate": moderate_artifacts,
            "mild": len(artifacts) - severe_artifacts - moderate_artifacts,
        },
    }


def breakdown() -> dict:
    """Per-patient monitoring table, ICU detail table, duration histogram,
    and artifact severity breakdown by type."""
    acqs = _load_eeg_acquisitions()
    hosps = _load_hospitalizations()
    meta = _load_seizure_meta()
    artifacts = _load_artifacts()
    patients = _load_patients()

    # Per-patient recording table
    patient_recs = []
    for a in acqs:
        pid = a["patient_id"]
        pt = patients.get(pid, {})
        m = meta.get(pid, {})
        patient_recs.append({
            "patient_id": pid,
            "age": pt.get("age"),
            "gender": pt.get("gender"),
            "recording_type": a["recording_type"].upper() if len(a["recording_type"]) <= 4 else a["recording_type"].title(),
            "duration_h": round(a["duration_min"] / 60, 1),
            "sampling_rate": f"{a['sampling_rate']} Hz",
            "montage": a["montage"].title(),
            "monitoring_tier": _monitoring_tier(a["recording_type"], a["duration_min"]),
            "eeg_pattern": m.get("eeg_pattern", ""),
            "critical_pattern": _is_critical_pattern(m.get("eeg_pattern", "")),
            "syndrome": m.get("syndrome", ""),
            "drug_resistant": "drug-resistant" in m.get("drug_resp", ""),
            "onset_zone": m.get("onset_zone", ""),
            "study_date": a["study_date"],
        })

    patient_recs.sort(key=lambda r: -r["duration_h"])

    # ICU patient detail table
    icu_hosps = [h for h in hosps if h["ward"] == "ICU"]
    icu_rows = []
    for h in icu_hosps:
        pid = h["patient_id"]
        pt = patients.get(pid, {})
        m = meta.get(pid, {})
        icu_rows.append({
            "patient_id": pid,
            "age": pt.get("age"),
            "gender": pt.get("gender"),
            "admission_reason": h["admission_reason"].replace("_", " ").title(),
            "los_days": h["los_days"],
            "seizure_free": h["seizure_free_at_discharge"],
            "readmit_30d": h["readmission_30d"],
            "eeg_pattern": m.get("eeg_pattern", ""),
            "critical_pattern": _is_critical_pattern(m.get("eeg_pattern", "")),
            "drug_resistant": "drug-resistant" in m.get("drug_resp", ""),
            "discharge_disposition": h["discharge_disposition"].replace("_", " ").title(),
            "cost_usd": int(h["total_cost_usd"]),
        })
    icu_rows.sort(key=lambda r: -r["los_days"])

    # Duration histogram (in hours)
    duration_buckets: Counter = Counter()
    for a in acqs:
        h = a["duration_min"] / 60
        if h < 1:
            duration_buckets["< 1 h"] += 1
        elif h < 8:
            duration_buckets["1–8 h"] += 1
        elif h < 24:
            duration_buckets["8–24 h"] += 1
        elif h < 48:
            duration_buckets["24–48 h"] += 1
        else:
            duration_buckets["≥ 48 h"] += 1

    # Artifact severity by type
    art_sev: defaultdict = defaultdict(Counter)
    for a in artifacts:
        art_sev[a["artifact_type"]][a["severity"]] += 1

    artifact_sev_table = [
        {
            "type": t.replace("_", " ").title(),
            "mild": art_sev[t].get("mild", 0),
            "moderate": art_sev[t].get("moderate", 0),
            "severe": art_sev[t].get("severe", 0),
            "total": sum(art_sev[t].values()),
        }
        for t in sorted(art_sev.keys())
    ]
    artifact_sev_table.sort(key=lambda r: -r["total"])

    return {
        "per_patient_recordings": patient_recs,
        "icu_patients": icu_rows,
        "duration_histogram": [
            {"bucket": k, "count": v}
            for k, v in [
                ("< 1 h", duration_buckets.get("< 1 h", 0)),
                ("1–8 h", duration_buckets.get("1–8 h", 0)),
                ("8–24 h", duration_buckets.get("8–24 h", 0)),
                ("24–48 h", duration_buckets.get("24–48 h", 0)),
                ("≥ 48 h", duration_buckets.get("≥ 48 h", 0)),
            ]
        ],
        "artifact_severity_table": artifact_sev_table,
        "drug_resistant_in_icu": {
            "count": sum(1 for r in icu_rows if r["drug_resistant"]),
            "total": len(icu_rows),
        },
    }


def definitions() -> dict:
    """cEEG terminology, ACNS 2021 standardized nomenclature, monitoring
    indications, critical patterns, and interpretation guidelines."""
    return {
        "term": "Continuous EEG (cEEG) Monitoring",
        "definition": (
            "Continuous EEG (cEEG) is uninterrupted electroencephalographic recording "
            "lasting ≥ 24 hours, performed in critically ill patients to detect non-convulsive "
            "seizures (NCS), non-convulsive status epilepticus (NCSE), and ictal-interictal "
            "continuum (IIC) patterns that are invisible on clinical exam alone."
        ),
        "ncse_prevalence": "8–48 % of ICU patients with altered consciousness (Claassen 2004)",
        "monitoring_indications": [
            {
                "indication": "Post-Status Epilepticus",
                "detail": "Mandatory ≥ 24 h cEEG to detect NCSE / recurrence after clinical SE is controlled",
                "grade": "ACNS Level A",
            },
            {
                "indication": "Altered Consciousness (unexplained)",
                "detail": "Electrographic seizures found in up to 35 % of patients with altered mental status",
                "grade": "ACNS Level A",
            },
            {
                "indication": "Post-Cardiac Arrest",
                "detail": "High-risk for myoclonic SE and burst-suppression; cEEG guides prognostication",
                "grade": "ACNS Level A",
            },
            {
                "indication": "Subarachnoid Hemorrhage (SAH)",
                "detail": "Delayed cerebral ischemia + epileptic activity; cEEG detects up to 20 % NCS",
                "grade": "ACNS Level B",
            },
            {
                "indication": "Traumatic Brain Injury (TBI)",
                "detail": "Post-traumatic seizures in 22 % — most non-convulsive in first 24 h",
                "grade": "ACNS Level B",
            },
            {
                "indication": "Neonatal Encephalopathy",
                "detail": "Amplitude-integrated cEEG (aEEG) standard for detecting neonatal NCS",
                "grade": "ACNS Level A",
            },
        ],
        "acns_patterns": [
            {
                "pattern": "Lateralized Periodic Discharges (LPDs / PLDs)",
                "description": "Periodic complexes lateralized to one hemisphere; associated with acute brain lesion; ictal-interictal continuum",
                "risk": "High — treat if clinical deterioration or evolving to seizure",
            },
            {
                "pattern": "Generalized Periodic Discharges (GPDs)",
                "description": "Bilaterally synchronous periodic complexes; often seen post-cardiac arrest",
                "risk": "Moderate–High",
            },
            {
                "pattern": "Burst-Suppression",
                "description": "Alternating high-amplitude bursts and flat suppression; sign of deep cortical depression",
                "risk": "Prognostic marker — poor outcome if drug-induced target not met",
            },
            {
                "pattern": "NCSE (Non-Convulsive Status Epilepticus)",
                "description": "Continuous electrographic seizure activity ≥ 10 min without clinical correlate",
                "risk": "Critical — treat emergently",
            },
            {
                "pattern": "Ictal-Interictal Continuum (IIC)",
                "description": "EEG patterns between clearly ictal and clearly interictal; uncertain clinical significance",
                "risk": "Requires expert interpretation + clinical context",
            },
        ],
        "monitoring_duration_guidance": [
            {"duration": "24 hours", "recommendation": "Minimum for high-risk ICU patients (ACNS 2021)"},
            {"duration": "48 hours", "recommendation": "If first 24 h shows no seizures but clinical suspicion high"},
            {"duration": "72+ hours", "recommendation": "Active NCSE, super-refractory SE, or prognostic assessment"},
        ],
        "artifact_challenges": [
            "ICU environment: ventilator rhythmic artifact mimics PLDs",
            "Cardiac monitor leads: ECG artifact on EEG channels",
            "Patient movement: nursing care, suctioning interrupt clean recording windows",
            "Lead displacement: high-impedance channels common with diaphoresis",
            "Electrode pop: brief high-amplitude discharge — not seizure",
        ],
        "abbreviations": {
            "cEEG": "Continuous EEG",
            "NCS": "Non-Convulsive Seizure",
            "NCSE": "Non-Convulsive Status Epilepticus",
            "LPD/PLD": "Lateralized Periodic Discharges",
            "GPD": "Generalized Periodic Discharges",
            "IIC": "Ictal-Interictal Continuum",
            "LTM": "Long-Term Monitoring",
            "ACNS": "American Clinical Neurophysiology Society",
            "ICU": "Intensive Care Unit",
            "SE": "Status Epilepticus",
            "TBI": "Traumatic Brain Injury",
            "SAH": "Subarachnoid Hemorrhage",
        },
        "references": [
            "Claassen J et al. Neurology 2004;62:1743-1748 — NCS in 8-34% ICU patients",
            "Hirsch LJ et al. (ACNS) J Clin Neurophysiol 2021;38:1-29 — cEEG terminology",
            "Brophy GM et al. (NCS) Neurocrit Care 2012;17:3-23 — SE management",
            "Rossetti AO et al. Neurocrit Care 2008;8:374-380 — NCSE outcomes",
            "Vespa PM et al. Crit Care Med 2016;44:e51-e55 — cEEG in TBI/SAH",
            "NICE NG217 Epilepsies (2022) — ambulatory EEG guidance",
        ],
        "clinical_pearls": [
            "25 % of SE patients have ongoing electrographic seizures after clinical control — always monitor",
            "PLDs at > 1.5 Hz are predictive of seizure and warrant prophylactic treatment",
            "Burst-suppression target (10–20 % suppression) is the EEG endpoint for anesthetic coma in RSE",
            "Video-EEG correlation clarifies whether movements are ictal, non-epileptic, or artifact",
            "cEEG reports should use ACNS standardized terminology to enable cross-site comparison",
        ],
    }
