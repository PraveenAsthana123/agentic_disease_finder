"""
MRI Brain Review Dashboard
==========================
Structural MRI review for epilepsy pre-surgical evaluation.

Key findings tracked per patient:
  - MRI availability and quality
  - Hippocampal sclerosis (mesial temporal sclerosis)
  - Focal cortical dysplasia (FCD)
  - Tumours / cavernomas / vascular malformations
  - Lesion location (lobe, laterality)
  - Volumetric analysis (hippocampal volume asymmetry)
  - Overall MRI classification: Normal / Lesional / Non-lesional / Equivocal

Data seeded into mri_findings table in clinical.db from realistic distributions
based on published epilepsy surgery series (Téllez-Zenteno et al., Epilepsia 2010;
Bernasconi et al., Brain 2011).

Author: Research Team
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from typing import Optional

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# ── MRI finding categories ──────────────────────────────────────────

LESION_TYPES = [
    {"code": "HS", "label": "Hippocampal Sclerosis", "prevalence": 0.35,
     "description": "Mesial temporal sclerosis — hippocampal atrophy + T2/FLAIR signal increase"},
    {"code": "FCD", "label": "Focal Cortical Dysplasia", "prevalence": 0.20,
     "description": "Cortical malformation — blurring of grey-white junction, transmantle sign"},
    {"code": "TUM", "label": "Low-Grade Tumour", "prevalence": 0.10,
     "description": "DNET, ganglioglioma, or low-grade glioma — well-circumscribed, non-enhancing"},
    {"code": "CAV", "label": "Cavernoma", "prevalence": 0.05,
     "description": "Cavernous malformation — popcorn appearance with hemosiderin ring"},
    {"code": "AVM", "label": "Vascular Malformation", "prevalence": 0.03,
     "description": "Arteriovenous malformation — flow voids, feeding arteries"},
    {"code": "ENC", "label": "Encephalomalacia", "prevalence": 0.07,
     "description": "Post-injury/infection gliosis — CSF-signal cavity with surrounding gliosis"},
    {"code": "NL", "label": "Non-Lesional", "prevalence": 0.15,
     "description": "No structural abnormality detected on standard MRI protocol"},
    {"code": "NRM", "label": "Normal", "prevalence": 0.05,
     "description": "Completely normal study — no epilepsy-related abnormalities"},
]

LOBES = ["Temporal", "Frontal", "Parietal", "Occipital", "Insular", "Multifocal"]
LATERALITY = ["Left", "Right", "Bilateral"]
MRI_QUALITY = ["Diagnostic", "Adequate", "Suboptimal", "Non-diagnostic"]

MRI_CLASSIFICATION = [
    {"code": "LESIONAL", "label": "Lesional", "description": "Discrete structural abnormality concordant with seizure focus"},
    {"code": "NON_LESIONAL", "label": "Non-Lesional", "description": "No visible structural abnormality on standard MRI"},
    {"code": "EQUIVOCAL", "label": "Equivocal", "description": "Subtle findings — uncertain clinical significance"},
    {"code": "NORMAL", "label": "Normal", "description": "Completely normal brain MRI"},
]


def _conn():
    return sqlite3.connect(DB_PATH)


def _seed_mri_data():
    """Seed mri_findings table with realistic epilepsy MRI data for all patients."""
    conn = _conn()
    c = conn.cursor()

    # Check current count
    existing = c.execute("SELECT COUNT(*) FROM mri_findings").fetchone()[0]
    if existing >= 15:
        conn.close()
        return existing

    # Get all patients
    c.execute("SELECT patient_id, name, age, gender, disease FROM patients ORDER BY patient_id")
    patients = c.fetchall()

    seeded = 0
    for pid, name, age, gender, disease in patients:
        # Check if already has MRI
        has = c.execute("SELECT COUNT(*) FROM mri_findings WHERE patient_id = ?", (pid,)).fetchone()[0]
        if has > 0:
            continue

        # Deterministic seed from patient_id
        h = int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)

        # Not every patient gets MRI — about 60%
        if h % 100 > 60 and "epilep" not in (disease or "").lower():
            continue

        # Pick lesion type using weighted distribution matching published prevalence
        # Cumulative: HS 35%, FCD 20%, TUM 10%, CAV 5%, AVM 3%, ENC 7%, NL 15%, NRM 5%
        bucket = h % 100
        if bucket < 35:
            lesion = LESION_TYPES[0]   # HS
        elif bucket < 55:
            lesion = LESION_TYPES[1]   # FCD
        elif bucket < 65:
            lesion = LESION_TYPES[2]   # TUM
        elif bucket < 70:
            lesion = LESION_TYPES[3]   # CAV
        elif bucket < 73:
            lesion = LESION_TYPES[4]   # AVM
        elif bucket < 80:
            lesion = LESION_TYPES[5]   # ENC
        elif bucket < 95:
            lesion = LESION_TYPES[6]   # NL
        else:
            lesion = LESION_TYPES[7]   # NRM

        # Location
        lobe_idx = (h >> 4) % len(LOBES)
        lat_idx = (h >> 8) % len(LATERALITY)
        qual_idx = (h >> 12) % 100
        if qual_idx < 70:
            quality = "Diagnostic"
        elif qual_idx < 90:
            quality = "Adequate"
        elif qual_idx < 97:
            quality = "Suboptimal"
        else:
            quality = "Non-diagnostic"

        # Hippocampal volume asymmetry index (normal < 0.05, HS typically > 0.10)
        base_asym = 0.02
        if lesion["code"] == "HS":
            base_asym = 0.10 + (h % 15) / 100  # 0.10-0.24
        elif lesion["code"] in ("FCD", "ENC"):
            base_asym = 0.04 + (h % 8) / 100   # 0.04-0.11
        else:
            base_asym = 0.01 + (h % 5) / 100   # 0.01-0.05

        # MRI classification
        if lesion["code"] in ("NL", "NRM"):
            if lesion["code"] == "NRM":
                classification = "NORMAL"
            else:
                classification = "NON_LESIONAL"
            lobe = None
            laterality = None
        elif lesion["code"] in ("FCD",) and (h % 3 == 0):
            classification = "EQUIVOCAL"
            lobe = LOBES[lobe_idx]
            laterality = LATERALITY[lat_idx]
        else:
            classification = "LESIONAL"
            lobe = LOBES[lobe_idx]
            laterality = LATERALITY[lat_idx]

        # T2/FLAIR signal
        t2_signal = "Normal"
        if lesion["code"] in ("HS", "FCD", "ENC", "TUM"):
            t2_signal = "Increased"
        elif lesion["code"] == "CAV":
            t2_signal = "Mixed (hemosiderin ring)"

        # Enhancement
        enhancing = False
        if lesion["code"] in ("AVM",):
            enhancing = True
        elif lesion["code"] == "TUM" and h % 4 == 0:
            enhancing = True

        fields = {
            "mri_available": "Yes",
            "quality": quality,
            "lesion_type": lesion["code"],
            "lesion_label": lesion["label"],
            "lesion_description": lesion["description"],
            "lesion_location": lobe,
            "laterality": laterality,
            "hippocampal_sclerosis": "Yes" if lesion["code"] == "HS" else "No",
            "hippocampal_volume_asymmetry": round(base_asym, 3),
            "t2_flair_signal": t2_signal,
            "enhancing": enhancing,
            "classification": classification,
            "classification_label": next(
                (cl["label"] for cl in MRI_CLASSIFICATION if cl["code"] == classification),
                "Unknown"
            ),
            "protocol": "Epilepsy MRI protocol (3T, 3D-FLAIR, coronal T2, volumetric T1)",
            "radiologist_confidence": "High" if quality in ("Diagnostic", "Adequate") else "Low",
        }

        from datetime import datetime, timedelta
        scan_date = datetime(2026, 1, 1) + timedelta(days=h % 180)

        c.execute(
            "INSERT INTO mri_findings (patient_id, fields_json, created_at) VALUES (?, ?, ?)",
            (pid, json.dumps(fields), scan_date.strftime("%Y-%m-%dT%H:%M:%S-06:00"))
        )
        seeded += 1

    conn.commit()
    total = c.execute("SELECT COUNT(*) FROM mri_findings").fetchone()[0]
    conn.close()
    return total


def mri_overview(patient_id: str = None) -> dict:
    """MRI Brain Review dashboard — overview of all patients or single patient."""
    _seed_mri_data()
    conn = _conn()
    c = conn.cursor()

    if patient_id:
        c.execute("SELECT fields_json, created_at FROM mri_findings WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1",
                  (patient_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return {"error": f"No MRI data for patient {patient_id}"}
        fields = json.loads(row[0])
        # Get patient demographics
        c.execute("SELECT name, age, gender, disease FROM patients WHERE patient_id = ?", (patient_id,))
        pat = c.fetchone()
        conn.close()
        return {
            "patient_id": patient_id,
            "patient_name": pat[0] if pat else patient_id,
            "age": pat[1] if pat else None,
            "disease": pat[3] if pat else None,
            "scan_date": row[1],
            **fields,
        }

    # All patients with MRI
    c.execute("""
        SELECT m.patient_id, p.name, p.age, p.gender, p.disease,
               m.fields_json, m.created_at
        FROM mri_findings m
        JOIN patients p ON m.patient_id = p.patient_id
        ORDER BY m.created_at DESC
    """)
    rows = c.fetchall()
    conn.close()

    patients = []
    classification_dist = {}
    lesion_dist = {}
    lobe_dist = {}
    laterality_dist = {}
    hs_count = 0

    for pid, name, age, gender, disease, fj, created in rows:
        f = json.loads(fj)
        patients.append({
            "patient_id": pid,
            "name": name,
            "age": age,
            "disease": disease,
            "classification": f.get("classification_label", "Unknown"),
            "lesion_type": f.get("lesion_label", "Unknown"),
            "lesion_location": f.get("lesion_location"),
            "laterality": f.get("laterality"),
            "hippocampal_sclerosis": f.get("hippocampal_sclerosis", "No"),
            "quality": f.get("quality", "Unknown"),
            "scan_date": created,
        })

        cl = f.get("classification_label", "Unknown")
        classification_dist[cl] = classification_dist.get(cl, 0) + 1

        lt = f.get("lesion_label", "Unknown")
        lesion_dist[lt] = lesion_dist.get(lt, 0) + 1

        loc = f.get("lesion_location")
        if loc:
            lobe_dist[loc] = lobe_dist.get(loc, 0) + 1

        lat = f.get("laterality")
        if lat:
            laterality_dist[lat] = laterality_dist.get(lat, 0) + 1

        if f.get("hippocampal_sclerosis") == "Yes":
            hs_count += 1

    total = len(patients)
    lesional_count = classification_dist.get("Lesional", 0)
    lesional_rate = round(lesional_count / total * 100, 1) if total else 0

    return {
        "title": "MRI Brain Review — Epilepsy Pre-Surgical Evaluation",
        "total_patients": total,
        "lesional_rate": lesional_rate,
        "hippocampal_sclerosis_count": hs_count,
        "classification_distribution": classification_dist,
        "lesion_type_distribution": lesion_dist,
        "lobe_distribution": lobe_dist,
        "laterality_distribution": laterality_dist,
        "patients": patients,
    }


def mri_breakdown(patient_id: str = None) -> dict:
    """Detailed MRI breakdown per patient — all fields, volumetric data, concordance."""
    _seed_mri_data()
    conn = _conn()
    c = conn.cursor()

    if patient_id:
        c.execute("""
            SELECT m.patient_id, p.name, p.age, p.disease,
                   m.fields_json, m.created_at
            FROM mri_findings m
            JOIN patients p ON m.patient_id = p.patient_id
            WHERE m.patient_id = ?
            ORDER BY m.created_at DESC
        """, (patient_id,))
        rows = c.fetchall()
        if not rows:
            conn.close()
            return {"error": f"No MRI data for patient {patient_id}"}

        details = []
        for pid, name, age, disease, fj, created in rows:
            f = json.loads(fj)
            # Check concordance with seizure diary
            sz_count = c.execute("SELECT COUNT(*) FROM seizure_diary WHERE patient_id = ?", (pid,)).fetchone()[0]

            details.append({
                "patient_id": pid,
                "name": name,
                "age": age,
                "disease": disease,
                "scan_date": created,
                "findings": f,
                "seizure_diary_entries": sz_count,
                "concordant": _check_concordance(f, sz_count),
            })
        conn.close()
        return {"patient_id": patient_id, "scans": details}

    # All patients breakdown
    c.execute("""
        SELECT m.patient_id, p.name, p.age, p.disease,
               m.fields_json, m.created_at
        FROM mri_findings m
        JOIN patients p ON m.patient_id = p.patient_id
        ORDER BY m.patient_id, m.created_at DESC
    """)
    rows = c.fetchall()

    breakdown = []
    for pid, name, age, disease, fj, created in rows:
        f = json.loads(fj)
        sz_count = c.execute("SELECT COUNT(*) FROM seizure_diary WHERE patient_id = ?", (pid,)).fetchone()[0]
        breakdown.append({
            "patient_id": pid,
            "name": name,
            "age": age,
            "disease": disease,
            "scan_date": created,
            "lesion_type": f.get("lesion_label", "Unknown"),
            "lesion_code": f.get("lesion_type", "?"),
            "location": f.get("lesion_location"),
            "laterality": f.get("laterality"),
            "classification": f.get("classification_label", "Unknown"),
            "hippocampal_sclerosis": f.get("hippocampal_sclerosis", "No"),
            "hippocampal_volume_asymmetry": f.get("hippocampal_volume_asymmetry"),
            "t2_flair_signal": f.get("t2_flair_signal", "Normal"),
            "enhancing": f.get("enhancing", False),
            "quality": f.get("quality", "Unknown"),
            "protocol": f.get("protocol"),
            "confidence": f.get("radiologist_confidence", "Unknown"),
            "seizure_diary_entries": sz_count,
            "concordant": _check_concordance(f, sz_count),
        })

    conn.close()

    # Volume asymmetry stats
    asym_values = [b["hippocampal_volume_asymmetry"] for b in breakdown
                   if b["hippocampal_volume_asymmetry"] is not None]
    abnormal_asym = sum(1 for v in asym_values if v >= 0.08)

    return {
        "title": "MRI Findings Breakdown",
        "total_scans": len(breakdown),
        "scans": breakdown,
        "volume_asymmetry_stats": {
            "mean": round(sum(asym_values) / len(asym_values), 3) if asym_values else 0,
            "max": max(asym_values) if asym_values else 0,
            "abnormal_count": abnormal_asym,
            "abnormal_threshold": 0.08,
        },
    }


def _check_concordance(mri_fields: dict, sz_count: int) -> str:
    """Check if MRI findings are concordant with clinical seizure data."""
    classification = mri_fields.get("classification", "")
    if classification == "NORMAL" and sz_count == 0:
        return "concordant"
    elif classification == "LESIONAL" and sz_count > 0:
        return "concordant"
    elif classification == "NON_LESIONAL" and sz_count > 0:
        return "discordant — seizures without visible lesion"
    elif classification == "NORMAL" and sz_count > 0:
        return "discordant — seizures with normal MRI"
    elif classification == "LESIONAL" and sz_count == 0:
        return "incidental — lesion without documented seizures"
    return "indeterminate"


def mri_definitions() -> dict:
    """MRI Brain Review definitions — lesion types, classification criteria, protocol."""
    return {
        "name": "MRI Brain Review — Epilepsy Pre-Surgical Evaluation",
        "purpose": "Structural MRI assessment to identify epileptogenic lesions, "
                   "guide surgical planning, and assess candidacy for resective surgery.",
        "protocol": {
            "name": "Epilepsy MRI Protocol",
            "field_strength": "3 Tesla (minimum 1.5T)",
            "sequences": [
                "3D T1-weighted (1mm isotropic) — volumetric analysis",
                "3D FLAIR — cortical/subcortical signal abnormalities",
                "Coronal T2 (perpendicular to hippocampus) — hippocampal evaluation",
                "Coronal FLAIR — temporal lobe lesions",
                "SWI/GRE — hemosiderin (cavernomas, prior hemorrhage)",
                "DWI — acute pathology, restricted diffusion",
            ],
            "reference": "Bernasconi A et al. Recommendations for the use of structural "
                         "MRI in the care of patients with epilepsy. Epilepsia. 2019;60(6):1054-1068",
        },
        "lesion_types": LESION_TYPES,
        "classification": MRI_CLASSIFICATION,
        "volumetric_analysis": {
            "method": "Hippocampal volume asymmetry index = |Vleft - Vright| / (Vleft + Vright)",
            "normal_range": "< 0.05",
            "suspicious": "0.05 - 0.08",
            "abnormal": ">= 0.08 (suggests unilateral hippocampal atrophy)",
        },
        "clinical_significance": [
            "MRI-positive (lesional) epilepsy has ~70% seizure-free rate post-surgery",
            "MRI-negative (non-lesional) epilepsy has ~40-50% seizure-free rate",
            "Hippocampal sclerosis is the most common finding in temporal lobe epilepsy",
            "FCD is the most common finding in extratemporal epilepsy",
            "Concordance between MRI and EEG improves surgical outcome prediction",
        ],
        "references": [
            "Téllez-Zenteno JF et al. Surgical outcomes in lesional and non-lesional epilepsy. Epilepsia. 2010;51(5):899-908",
            "Bernasconi A et al. Recommendations for structural MRI in epilepsy. Epilepsia. 2019;60(6):1054-1068",
            "Blümcke I et al. The clinicopathologic spectrum of FCD. Epilepsia. 2011;52(1):158-174",
        ],
    }


if __name__ == "__main__":
    import sys
    pid = sys.argv[1] if len(sys.argv) > 1 else None
    result = mri_overview(pid)
    print(json.dumps(result, indent=2, default=str))
