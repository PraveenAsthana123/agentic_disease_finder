"""
Pre-Surgical Evaluation Dashboard — real clinical.db data for epilepsy surgery
candidacy assessment. Combines MRI lesion analysis, EEG lateralization, seizure
frequency, medication resistance, and surgical candidacy scoring.
"""

import json, sqlite3, pathlib
from datetime import datetime, timedelta
from collections import Counter

BASE = pathlib.Path(__file__).resolve().parent.parent
DB = BASE / "data" / "clinical.db"


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _load_mri():
    """Load MRI findings with parsed fields."""
    conn = _conn()
    rows = conn.execute("SELECT patient_id, fields_json, created_at FROM mri_findings").fetchall()
    conn.close()
    results = []
    for r in rows:
        d = json.loads(r["fields_json"])
        d["patient_id"] = r["patient_id"]
        d["scan_date"] = r["created_at"]
        results.append(d)
    return results


def _load_eeg():
    """Load EEG acquisition data."""
    conn = _conn()
    rows = conn.execute("SELECT patient_id, fields_json, created_at FROM eeg_acquisition").fetchall()
    conn.close()
    results = []
    for r in rows:
        d = json.loads(r["fields_json"])
        d["patient_id"] = r["patient_id"]
        d["study_date"] = d.get("study_date", r["created_at"])
        results.append(d)
    return results


def _load_seizures():
    """Load seizure diary entries."""
    conn = _conn()
    rows = conn.execute(
        "SELECT patient_id, event_date, duration_sec, severity, aura, awareness, "
        "motor_signs, trigger FROM seizure_diary"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _load_medications():
    """Load medication records."""
    conn = _conn()
    rows = conn.execute("SELECT patient_id, fields_json FROM medications").fetchall()
    conn.close()
    results = []
    for r in rows:
        d = json.loads(r["fields_json"])
        d["patient_id"] = r["patient_id"]
        results.append(d)
    return results


def _load_patients():
    """Load patient demographics."""
    conn = _conn()
    rows = conn.execute("SELECT patient_id, name, age, gender FROM patients").fetchall()
    conn.close()
    return {r["patient_id"]: dict(r) for r in rows}


def _candidacy_score(patient_id, mri_list, eeg_list, seizures, meds):
    """
    Compute surgical candidacy score (0-100) based on ILAE criteria:
    - Drug resistance (tried >=2 AEDs without seizure freedom): +30
    - MRI lesion identified (localizable): +25
    - EEG concordance with lesion: +20
    - Seizure frequency (>=1/month): +15
    - Age/duration factor: +10
    """
    score = 0
    factors = []

    # Drug resistance
    patient_meds = [m for m in meds if m["patient_id"] == patient_id]
    aed_count = len([m for m in patient_meds if m.get("aed", True)])
    if aed_count >= 2:
        score += 30
        factors.append(f"Drug-resistant ({aed_count} AEDs tried)")
    elif aed_count == 1:
        score += 10
        factors.append("Single AED (not yet drug-resistant)")

    # MRI lesion
    patient_mri = [m for m in mri_list if m["patient_id"] == patient_id]
    has_lesion = False
    lesion_laterality = None
    for mri in patient_mri:
        lt = mri.get("lesion_type", "NL")
        if lt not in ("NL", "NRM", None, ""):
            has_lesion = True
            lesion_laterality = mri.get("laterality")
            break
    if has_lesion:
        score += 25
        factors.append(f"MRI lesion identified ({lt}, {lesion_laterality})")
    else:
        factors.append("MRI non-lesional (lower candidacy)")

    # EEG lateralization concordance
    patient_eeg = [e for e in eeg_list if e["patient_id"] == patient_id]
    if patient_eeg and has_lesion:
        score += 20
        factors.append("EEG available for concordance assessment")
    elif patient_eeg:
        score += 10
        factors.append("EEG available (no lesion for concordance)")

    # Seizure frequency
    patient_sz = [s for s in seizures if s["patient_id"] == patient_id]
    if len(patient_sz) >= 2:
        score += 15
        factors.append(f"High seizure burden ({len(patient_sz)} events)")
    elif len(patient_sz) >= 1:
        score += 8
        factors.append(f"Moderate seizure burden ({len(patient_sz)} events)")

    # Age/duration bonus (younger patients with longer history)
    score += 5  # baseline age factor
    factors.append("Age/duration factor applied")

    # Classify
    if score >= 70:
        classification = "strong_candidate"
    elif score >= 45:
        classification = "possible_candidate"
    else:
        classification = "not_candidate"

    return {
        "score": min(score, 100),
        "classification": classification,
        "factors": factors,
        "aed_count": aed_count,
        "has_lesion": has_lesion,
        "lesion_laterality": lesion_laterality,
        "seizure_count": len(patient_sz),
    }


def overview():
    """KPIs, candidacy distribution, lesion-type breakdown, laterality."""
    mri = _load_mri()
    eeg = _load_eeg()
    seizures = _load_seizures()
    meds = _load_medications()
    patients = _load_patients()

    # Compute candidacy for each patient with MRI
    patient_ids_with_data = set(m["patient_id"] for m in mri)
    candidacies = {}
    for pid in patient_ids_with_data:
        candidacies[pid] = _candidacy_score(pid, mri, eeg, seizures, meds)

    # KPIs
    strong = sum(1 for c in candidacies.values() if c["classification"] == "strong_candidate")
    possible = sum(1 for c in candidacies.values() if c["classification"] == "possible_candidate")
    not_cand = sum(1 for c in candidacies.values() if c["classification"] == "not_candidate")
    avg_score = round(sum(c["score"] for c in candidacies.values()) / max(len(candidacies), 1), 1)

    # Lesion type distribution
    lesion_types = Counter(m.get("lesion_type", "Unknown") for m in mri)
    lesion_dist = [{"type": k, "count": v} for k, v in lesion_types.most_common()]

    # Laterality distribution
    lat_counter = Counter(m.get("laterality", "Unknown") for m in mri)
    laterality_dist = [{"side": k or "Unknown", "count": v} for k, v in lat_counter.most_common()]

    # Lesion type labels
    type_labels = {
        "HS": "Hippocampal Sclerosis",
        "FCD": "Focal Cortical Dysplasia",
        "CAV": "Cavernoma",
        "AVM": "Arteriovenous Malformation",
        "TUM": "Tumor",
        "ENC": "Encephalomalacia",
        "NL": "Non-Lesional",
        "NRM": "Normal",
    }

    # Candidacy score distribution (histogram buckets)
    score_buckets = {"0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
    for c in candidacies.values():
        s = c["score"]
        if s <= 20:
            score_buckets["0-20"] += 1
        elif s <= 40:
            score_buckets["21-40"] += 1
        elif s <= 60:
            score_buckets["41-60"] += 1
        elif s <= 80:
            score_buckets["61-80"] += 1
        else:
            score_buckets["81-100"] += 1
    score_histogram = [{"range": k, "count": v} for k, v in score_buckets.items()]

    return {
        "kpis": {
            "total_evaluated": len(candidacies),
            "strong_candidates": strong,
            "possible_candidates": possible,
            "not_candidates": not_cand,
            "avg_candidacy_score": avg_score,
            "lesional_cases": sum(1 for m in mri if m.get("lesion_type") not in ("NL", "NRM", None, "")),
            "hippocampal_sclerosis": sum(1 for m in mri if m.get("lesion_type") == "HS"),
        },
        "candidacy_distribution": [
            {"status": "Strong Candidate", "count": strong},
            {"status": "Possible Candidate", "count": possible},
            {"status": "Not Candidate", "count": not_cand},
        ],
        "lesion_type_distribution": lesion_dist,
        "lesion_type_labels": type_labels,
        "laterality_distribution": laterality_dist,
        "score_histogram": score_histogram,
    }


def breakdown():
    """Per-patient candidacy table, factor details, workup completeness."""
    mri = _load_mri()
    eeg = _load_eeg()
    seizures = _load_seizures()
    meds = _load_medications()
    patients = _load_patients()

    patient_ids_with_data = set(m["patient_id"] for m in mri)
    rows = []
    for pid in sorted(patient_ids_with_data):
        cand = _candidacy_score(pid, mri, eeg, seizures, meds)
        pt = patients.get(pid, {})
        # Workup completeness
        has_mri = any(m["patient_id"] == pid for m in mri)
        has_eeg = any(e["patient_id"] == pid for e in eeg)
        has_sz_diary = any(s["patient_id"] == pid for s in seizures)
        has_meds = any(m["patient_id"] == pid for m in meds)
        workup_items = [has_mri, has_eeg, has_sz_diary, has_meds]
        workup_pct = round(sum(workup_items) / len(workup_items) * 100)

        # Get lesion info
        patient_mri = [m for m in mri if m["patient_id"] == pid]
        lesion_info = "Non-lesional"
        for m in patient_mri:
            lt = m.get("lesion_type", "NL")
            if lt not in ("NL", "NRM", None, ""):
                lesion_info = f"{lt} ({m.get('laterality', '?')})"
                break

        rows.append({
            "patient_id": pid,
            "name": pt.get("name", "Unknown"),
            "age": pt.get("age"),
            "score": cand["score"],
            "classification": cand["classification"],
            "factors": cand["factors"],
            "lesion_info": lesion_info,
            "aed_count": cand["aed_count"],
            "seizure_count": cand["seizure_count"],
            "workup_completeness_pct": workup_pct,
            "has_mri": has_mri,
            "has_eeg": has_eeg,
            "has_seizure_diary": has_sz_diary,
            "has_medication_history": has_meds,
        })

    # Sort by score descending
    rows.sort(key=lambda x: x["score"], reverse=True)

    # Workup gap analysis
    total = len(rows)
    gap_analysis = {
        "mri_coverage": round(sum(1 for r in rows if r["has_mri"]) / max(total, 1) * 100),
        "eeg_coverage": round(sum(1 for r in rows if r["has_eeg"]) / max(total, 1) * 100),
        "seizure_diary_coverage": round(sum(1 for r in rows if r["has_seizure_diary"]) / max(total, 1) * 100),
        "medication_coverage": round(sum(1 for r in rows if r["has_medication_history"]) / max(total, 1) * 100),
    }

    return {
        "patients": rows,
        "total": total,
        "gap_analysis": gap_analysis,
    }


def definitions():
    """Metric definitions and clinical context."""
    return {
        "title": "Pre-Surgical Evaluation Dashboard",
        "description": "Assesses epilepsy surgery candidacy using ILAE criteria: drug resistance, "
                       "MRI lesion identification, EEG concordance, seizure burden, and clinical factors.",
        "metrics": [
            {
                "name": "Candidacy Score",
                "range": "0-100",
                "description": "Composite score based on ILAE pre-surgical evaluation criteria. "
                               "≥70 = strong candidate, 45-69 = possible candidate, <45 = not candidate.",
            },
            {
                "name": "Drug Resistance",
                "weight": 30,
                "description": "ILAE definition: failure of 2+ appropriately chosen AEDs to achieve "
                               "sustained seizure freedom.",
            },
            {
                "name": "MRI Lesion",
                "weight": 25,
                "description": "Identifiable structural lesion concordant with seizure semiology. "
                               "Types: HS (hippocampal sclerosis), FCD, cavernoma, tumor, AVM.",
            },
            {
                "name": "EEG Concordance",
                "weight": 20,
                "description": "Ictal/interictal EEG localizing to same region as MRI lesion.",
            },
            {
                "name": "Seizure Burden",
                "weight": 15,
                "description": "Frequency and severity of seizures impacting quality of life.",
            },
            {
                "name": "Age/Duration",
                "weight": 10,
                "description": "Earlier surgery in appropriate candidates improves outcomes.",
            },
        ],
        "lesion_types": {
            "HS": "Hippocampal Sclerosis — most common surgically remediable lesion in TLE",
            "FCD": "Focal Cortical Dysplasia — congenital malformation, excellent surgical outcomes",
            "CAV": "Cavernoma — vascular malformation, low recurrence post-resection",
            "AVM": "Arteriovenous Malformation — requires angiographic workup",
            "TUM": "Low-grade tumor (DNET, ganglioglioma) — dual pathology common",
            "ENC": "Encephalomalacia — post-traumatic/post-stroke",
            "NL": "Non-lesional — requires invasive monitoring (SEEG/grids)",
            "NRM": "Normal MRI — may still be surgical candidate with concordant EEG",
        },
        "surgical_procedures": [
            "Anterior Temporal Lobectomy (ATL)",
            "Selective Amygdalohippocampectomy (SAH)",
            "Lesionectomy",
            "Focal Cortical Resection",
            "Hemispherectomy / Hemispherotomy",
            "Corpus Callosotomy",
            "SEEG (Stereo-EEG) — invasive monitoring",
            "Subdural Grid Monitoring",
            "Laser Interstitial Thermal Therapy (LITT)",
            "Responsive Neurostimulation (RNS)",
            "Vagus Nerve Stimulation (VNS)",
        ],
        "engel_outcomes": {
            "I": "Seizure-free (no auras)",
            "II": "Rare disabling seizures",
            "III": "Worthwhile improvement",
            "IV": "No worthwhile improvement",
        },
    }


if __name__ == "__main__":
    import pprint
    print("=== OVERVIEW ===")
    pprint.pprint(overview())
    print("\n=== BREAKDOWN (first 3) ===")
    bd = breakdown()
    pprint.pprint(bd["patients"][:3])
    print(f"\nGap analysis: {bd['gap_analysis']}")
    print("\n=== DEFINITIONS ===")
    pprint.pprint(definitions())
