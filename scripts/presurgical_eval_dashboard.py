#!/usr/bin/env python3
"""Pre-Surgical Epilepsy Evaluation Dashboard
==============================================
Real data: seizure_metadata (71 patients) · mri_findings (40) ·
neuropsych (37) · medications (9) · patients (41) · analyses (133).

Surfaces surgical candidacy scoring, drug-resistant epilepsy (DRE) flags,
MRI lesion concordance, onset-zone lateralization, and neuropsychological
risk profiles for epilepsy surgery pre-evaluation.

All statistics derived from real clinical.db — never fabricated.
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "clinical.db")

# Surgical candidacy thresholds
MIN_ASM_FAILURES = 2          # DRE definition: ≥2 ASM trials failed
MOCA_RISK_THRESHOLD = 24      # Below → cognitive risk for surgery
PHQ9_RISK_THRESHOLD = 15      # Moderate–severe depression

# Onset zones that are potentially operable
FOCAL_ZONES = {
    "Temporal (mesial/lateral)", "Temporal (mesial)", "Temporal (lateral)",
    "Frontal (mesial/SMA)", "Frontal (lateral)", "Parietal",
    "Occipital", "Insula / Opercular", "Posterior cortex",
    "Fronto-parietal", "Fronto-temporal", "Temporal-parietal-occipital",
}

# Known anti-seizure medications
ASM_NAMES = {
    "levetiracetam", "lamotrigine", "valproate", "carbamazepine",
    "oxcarbazepine", "lacosamide", "topiramate", "phenytoin",
    "zonisamide", "perampanel", "brivaracetam", "clobazam",
    "clonazepam", "vigabatrin", "eslicarbazepine", "phenobarbital",
}


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _load_seizure_metadata() -> List[Dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT patient_id, fields_json FROM seizure_metadata ORDER BY patient_id"
    ).fetchall()
    conn.close()
    result = []
    for pid, fj in rows:
        try:
            d = json.loads(fj)
            d["patient_id"] = pid
            result.append(d)
        except Exception:
            pass
    return result


def _load_mri() -> Dict[str, Dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT patient_id, fields_json FROM mri_findings ORDER BY patient_id"
    ).fetchall()
    conn.close()
    by_patient: Dict[str, Dict] = {}
    for pid, fj in rows:
        try:
            d = json.loads(fj)
            by_patient[pid] = d
        except Exception:
            pass
    return by_patient


def _load_neuropsych() -> Dict[str, Dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT patient_id, fields_json FROM neuropsych ORDER BY patient_id"
    ).fetchall()
    conn.close()
    by_patient: Dict[str, Dict] = {}
    for pid, fj in rows:
        try:
            d = json.loads(fj)
            by_patient[pid] = d
        except Exception:
            pass
    return by_patient


def _load_medications() -> Dict[str, List[Dict]]:
    conn = _conn()
    rows = conn.execute(
        "SELECT patient_id, fields_json FROM medications ORDER BY patient_id"
    ).fetchall()
    conn.close()
    by_patient: Dict[str, List[Dict]] = defaultdict(list)
    for pid, fj in rows:
        try:
            d = json.loads(fj)
            by_patient[pid].append(d)
        except Exception:
            pass
    return dict(by_patient)


def _load_patients() -> Dict[str, Dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT patient_id, name, age, gender, disease FROM patients"
    ).fetchall()
    conn.close()
    return {
        r[0]: {"name": r[1], "age": r[2], "gender": r[3], "disease": r[4]}
        for r in rows
    }


def _is_focal(onset_zone: str) -> bool:
    if not onset_zone:
        return False
    return (
        onset_zone in FOCAL_ZONES
        or (
            "Generalized" not in onset_zone
            and "bilateral" not in onset_zone.lower()
            and "Non-lateralized" not in onset_zone
        )
    )


def _is_mri_lesional(mri: Optional[Dict]) -> bool:
    if not mri:
        return False
    lesion = mri.get("lesion_type", "")
    avail = mri.get("mri_available", "No")
    return avail == "Yes" and lesion and lesion not in ("None", "", "Normal")


def _count_asm_trials(meds: List[Dict]) -> int:
    """Count distinct ASM drug names trialled."""
    names = set()
    for m in meds:
        name = m.get("drug_name", "")
        if name and name.lower() in ASM_NAMES:
            names.add(name.lower())
        # also check 'aed' list field
        for aed in m.get("aed", []):
            if isinstance(aed, str) and aed.lower() in ASM_NAMES:
                names.add(aed.lower())
    return len(names)


def _candidacy_score(
    focal: bool,
    mri_lesional: bool,
    lateralized: bool,
    asm_count: int,
) -> int:
    """
    Simple 0–4 surgical candidacy score.
    +1 focal onset, +1 MRI lesional, +1 lateralized, +1 DRE (≥2 ASMs)
    """
    score = 0
    if focal:
        score += 1
    if mri_lesional:
        score += 1
    if lateralized:
        score += 1
    if asm_count >= MIN_ASM_FAILURES:
        score += 1
    return score


def overview() -> Dict[str, Any]:
    """
    KPIs + distribution summaries for the surgical evaluation panel.
    """
    sz_records = _load_seizure_metadata()
    mri_by_pt = _load_mri()
    np_by_pt = _load_neuropsych()
    meds_by_pt = _load_medications()
    patients = _load_patients()

    # De-duplicate by patient (take first record per patient)
    by_patient: Dict[str, Dict] = {}
    for rec in sz_records:
        pid = rec["patient_id"]
        if pid not in by_patient:
            by_patient[pid] = rec

    n_total = len(by_patient)

    focal_count = 0
    lesional_count = 0
    lateralized_count = 0
    dre_count = 0
    high_candidate_count = 0   # score ≥ 3
    perfect_candidate_count = 0  # score = 4

    onset_zone_dist: Dict[str, int] = defaultdict(int)
    laterality_dist: Dict[str, int] = defaultdict(int)
    lesion_type_dist: Dict[str, int] = defaultdict(int)
    asm_count_dist: Dict[int, int] = defaultdict(int)
    candidacy_dist: Dict[int, int] = defaultdict(int)

    for pid, rec in by_patient.items():
        onset_zone = rec.get("onset_zone", "Unknown")
        lateralization = rec.get("lateralization", "Unknown")
        mri = mri_by_pt.get(pid)
        meds = meds_by_pt.get(pid, [])
        asm_count = _count_asm_trials(meds)

        focal = _is_focal(onset_zone)
        mri_lesional = _is_mri_lesional(mri)
        lateralized = lateralization not in ("Non-lateralized", "Unknown", "Bilateral", "")

        score = _candidacy_score(focal, mri_lesional, lateralized, asm_count)

        if focal:
            focal_count += 1
        if mri_lesional:
            lesional_count += 1
        if lateralized:
            lateralized_count += 1
        if asm_count >= MIN_ASM_FAILURES:
            dre_count += 1
        if score >= 3:
            high_candidate_count += 1
        if score == 4:
            perfect_candidate_count += 1

        onset_zone_dist[onset_zone] += 1
        laterality_dist[lateralization] += 1
        asm_count_dist[asm_count] += 1
        candidacy_dist[score] += 1

        if mri and mri.get("lesion_type") not in (None, "", "None"):
            lesion_type_dist[mri["lesion_type"]] += 1

    # Neuropsychological risk
    np_risk_count = 0
    moca_scores = []
    for pid, npd in np_by_pt.items():
        moca = npd.get("moca")
        phq9 = npd.get("phq9")
        if moca is not None:
            moca_scores.append(moca)
        if (moca is not None and moca < MOCA_RISK_THRESHOLD) or (
            phq9 is not None and phq9 >= PHQ9_RISK_THRESHOLD
        ):
            np_risk_count += 1

    avg_moca = round(mean(moca_scores), 1) if moca_scores else None

    return {
        "kpis": {
            "total_patients": n_total,
            "focal_onset_pct": round(focal_count / n_total * 100, 1) if n_total else 0,
            "mri_lesional_pct": round(lesional_count / n_total * 100, 1) if n_total else 0,
            "lateralized_pct": round(lateralized_count / n_total * 100, 1) if n_total else 0,
            "dre_pct": round(dre_count / n_total * 100, 1) if n_total else 0,
            "high_candidates": high_candidate_count,
            "perfect_candidates": perfect_candidate_count,
            "np_risk_count": np_risk_count,
            "avg_moca": avg_moca,
        },
        "onset_zone_dist": [
            {"zone": k, "count": v}
            for k, v in sorted(onset_zone_dist.items(), key=lambda x: -x[1])[:12]
        ],
        "laterality_dist": [
            {"side": k, "count": v}
            for k, v in sorted(laterality_dist.items(), key=lambda x: -x[1])
        ],
        "lesion_type_dist": [
            {"lesion": k, "count": v}
            for k, v in sorted(lesion_type_dist.items(), key=lambda x: -x[1])
        ],
        "candidacy_dist": [
            {"score": k, "count": v, "label": ["Not candidate", "Poor", "Fair", "Good", "Excellent"][k]}
            for k, v in sorted(candidacy_dist.items())
        ],
        "asm_count_dist": [
            {"asm_trials": k, "count": v}
            for k, v in sorted(asm_count_dist.items())
        ],
        "data_sources": {
            "seizure_metadata_records": len(sz_records),
            "mri_records": len(mri_by_pt),
            "neuropsych_records": len(np_by_pt),
            "medication_records": sum(len(v) for v in meds_by_pt.values()),
        },
    }


def breakdown() -> Dict[str, Any]:
    """
    Per-patient surgical candidacy cards + MRI concordance table +
    neuropsychological risk matrix.
    """
    sz_records = _load_seizure_metadata()
    mri_by_pt = _load_mri()
    np_by_pt = _load_neuropsych()
    meds_by_pt = _load_medications()
    patients = _load_patients()

    by_patient: Dict[str, Dict] = {}
    for rec in sz_records:
        pid = rec["patient_id"]
        if pid not in by_patient:
            by_patient[pid] = rec

    patient_cards = []
    mri_concordance = []

    for pid, rec in sorted(by_patient.items()):
        onset_zone = rec.get("onset_zone", "Unknown")
        lateralization = rec.get("lateralization", "Unknown")
        mri = mri_by_pt.get(pid)
        npd = np_by_pt.get(pid, {})
        meds = meds_by_pt.get(pid, [])
        pat = patients.get(pid, {})
        asm_count = _count_asm_trials(meds)

        focal = _is_focal(onset_zone)
        mri_lesional = _is_mri_lesional(mri)
        lateralized = lateralization not in ("Non-lateralized", "Unknown", "Bilateral", "")
        score = _candidacy_score(focal, mri_lesional, lateralized, asm_count)

        moca = npd.get("moca")
        phq9 = npd.get("phq9")
        gad7 = npd.get("gad7")

        card = {
            "patient_id": pid,
            "age": pat.get("age"),
            "gender": pat.get("gender"),
            "disease": pat.get("disease"),
            "onset_zone": onset_zone,
            "lateralization": lateralization,
            "focal_onset": focal,
            "mri_lesional": mri_lesional,
            "lateralized": lateralized,
            "asm_trials": asm_count,
            "dre": asm_count >= MIN_ASM_FAILURES,
            "candidacy_score": score,
            "candidacy_label": ["Not candidate", "Poor", "Fair", "Good", "Excellent"][score],
            "moca": moca,
            "phq9": phq9,
            "gad7": gad7,
            "np_risk": (
                (moca is not None and moca < MOCA_RISK_THRESHOLD)
                or (phq9 is not None and phq9 >= PHQ9_RISK_THRESHOLD)
            ),
            "lesion_type": (mri or {}).get("lesion_type"),
            "lesion_label": (mri or {}).get("lesion_label"),
            "eeg_pattern": rec.get("eeg_pattern"),
            "ilae_types": rec.get("ilae_seizure_types", []),
        }
        patient_cards.append(card)

        # MRI concordance: does MRI lesion laterality match EEG laterality?
        if mri and mri.get("mri_available") == "Yes":
            mri_lat = mri.get("lesion_side") or mri.get("lateralization") or "Unknown"
            eeg_lat = lateralization
            concordant = (
                mri_lat != "Unknown"
                and eeg_lat != "Unknown"
                and mri_lat.lower() == eeg_lat.lower()
            )
            mri_concordance.append({
                "patient_id": pid,
                "eeg_lateralization": eeg_lat,
                "mri_lateralization": mri_lat,
                "lesion_type": mri.get("lesion_type", "—"),
                "lesion_label": mri.get("lesion_label", "—"),
                "concordant": concordant,
                "mri_quality": mri.get("quality", "Unknown"),
            })

    # Sort by candidacy score descending
    patient_cards.sort(key=lambda x: -x["candidacy_score"])

    # Neuropsych risk matrix
    np_matrix = []
    for pid, npd in sorted(np_by_pt.items()):
        moca = npd.get("moca")
        phq9 = npd.get("phq9")
        gad7 = npd.get("gad7")
        mmse = npd.get("mmse")
        np_matrix.append({
            "patient_id": pid,
            "battery": npd.get("battery_type", "Unknown"),
            "moca": moca,
            "mmse": mmse,
            "phq9": phq9,
            "gad7": gad7,
            "cognitive_risk": moca is not None and moca < MOCA_RISK_THRESHOLD,
            "mood_risk": phq9 is not None and phq9 >= PHQ9_RISK_THRESHOLD,
        })

    return {
        "patient_cards": patient_cards[:60],   # cap at 60 for payload
        "mri_concordance": mri_concordance,
        "neuropsych_matrix": np_matrix,
        "total_evaluated": len(patient_cards),
        "high_candidate_ids": [
            c["patient_id"] for c in patient_cards if c["candidacy_score"] >= 3
        ][:20],
    }


def definitions() -> Dict[str, Any]:
    return {
        "dashboard": "Pre-Surgical Epilepsy Evaluation Dashboard",
        "purpose": (
            "Screen epilepsy patients for surgical candidacy based on drug-resistant "
            "epilepsy (DRE) criteria, focal EEG onset, MRI lesion concordance, and "
            "neuropsychological risk profile."
        ),
        "surgical_candidacy_score": {
            "description": "0–4 composite score derived from 4 binary criteria",
            "criteria": {
                "focal_onset": "EEG seizure onset is focal (not generalized or non-lateralized)",
                "mri_lesional": "Structural MRI identifies a resectable lesion",
                "lateralized": "Clear lateralization to one hemisphere",
                "dre": f"Drug-resistant epilepsy — ≥{MIN_ASM_FAILURES} adequate ASM trials failed",
            },
            "thresholds": {
                "0": "Not a surgical candidate",
                "1": "Poor candidacy",
                "2": "Fair candidacy",
                "3": "Good candidacy — refer for SEEG/grid workup",
                "4": "Excellent candidacy — strong resection candidate",
            },
        },
        "dre_definition": {
            "standard": "ILAE 2010 (Kwan et al.)",
            "criteria": (
                f"Failure of ≥{MIN_ASM_FAILURES} adequately chosen, tolerated, and used "
                "anti-seizure medication (ASM) schedules to achieve sustained seizure freedom."
            ),
        },
        "mri_concordance": (
            "Concordance between EEG lateralization and MRI lesion side. "
            "High concordance improves surgical outcome prediction."
        ),
        "neuropsychological_risks": {
            "cognitive_risk": f"MoCA < {MOCA_RISK_THRESHOLD} suggests pre-operative cognitive impairment",
            "mood_risk": f"PHQ-9 ≥ {PHQ9_RISK_THRESHOLD} indicates moderate–severe depression requiring pre-surgical management",
        },
        "data_sources": {
            "seizure_metadata": "71 patient seizure characterization records (ILAE type, onset zone, lateralization)",
            "mri_findings": "40 structural MRI reports (lesion type, quality, concordance)",
            "neuropsych": "37 neuropsychological battery results (MoCA, MMSE, PHQ-9, GAD-7)",
            "medications": "9 ASM prescription records (drug name, dose, frequency)",
            "patients": "41 patient demographics",
        },
        "clinical_standards": [
            "ILAE 2010 DRE definition (Kwan et al., Epilepsia 2010)",
            "ILAE 2017 seizure type classification",
            "ACNS electrode placement standard (10-20 system)",
            "NAEC (National Association of Epilepsy Centers) surgical evaluation guidelines",
        ],
        "abbreviations": {
            "ASM": "Anti-Seizure Medication",
            "DRE": "Drug-Resistant Epilepsy",
            "ECoG": "Electrocorticography",
            "SEEG": "Stereo-EEG (depth electrode recording)",
            "HS": "Hippocampal Sclerosis",
            "MTS": "Mesial Temporal Sclerosis",
            "MoCA": "Montreal Cognitive Assessment",
            "MMSE": "Mini-Mental State Examination",
            "PHQ-9": "Patient Health Questionnaire-9 (depression)",
            "GAD-7": "Generalized Anxiety Disorder-7",
        },
    }
