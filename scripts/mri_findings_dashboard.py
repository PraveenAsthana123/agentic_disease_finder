"""
Neuro AI Ecosystem — MRI Findings Dashboard
=============================================
MRI findings analytics from the mri_findings table in clinical.db.

Lesion types: HS, FCD, NL, CAV, ENC, VM, NORM, LGT (short codes)
Classifications: LESIONAL, NON-LESIONAL, EQUIVOCAL, NORMAL
Qualities: Diagnostic, Adequate, Suboptimal, Non-diagnostic

Real data: mri_findings table in clinical.db (40 rows, 40 distinct patients).
"""

import sqlite3
from pathlib import Path
from collections import defaultdict

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def overview():
    """MRI findings overview — KPIs (total scans, lesional rate, HS rate,
    avg volume asymmetry, high confidence rate), distributions (lesion_type pie,
    location bar, laterality pie, classification pie, quality bar),
    monthly timeline (scans per month)."""
    conn = _conn()
    cur = conn.cursor()

    # Total scans
    cur.execute("SELECT COUNT(*) FROM mri_findings")
    total_scans = cur.fetchone()[0]

    # Lesional rate
    cur.execute("""
        SELECT COUNT(*) FROM mri_findings
        WHERE json_extract(fields_json, '$.classification') = 'LESIONAL'
    """)
    lesional_count = cur.fetchone()[0]
    lesional_rate_pct = round(lesional_count * 100.0 / total_scans, 1) if total_scans else 0.0

    # HS rate (hippocampal_sclerosis = "Yes")
    cur.execute("""
        SELECT COUNT(*) FROM mri_findings
        WHERE json_extract(fields_json, '$.hippocampal_sclerosis') = 'Yes'
    """)
    hs_count = cur.fetchone()[0]
    hs_rate_pct = round(hs_count * 100.0 / total_scans, 1) if total_scans else 0.0

    # Average hippocampal volume asymmetry
    cur.execute("""
        SELECT AVG(CAST(json_extract(fields_json, '$.hippocampal_volume_asymmetry') AS REAL))
        FROM mri_findings
        WHERE json_extract(fields_json, '$.hippocampal_volume_asymmetry') IS NOT NULL
    """)
    avg_vol_raw = cur.fetchone()[0]
    avg_volume_asymmetry = round(avg_vol_raw, 3) if avg_vol_raw is not None else 0.0

    # High confidence rate
    cur.execute("""
        SELECT COUNT(*) FROM mri_findings
        WHERE json_extract(fields_json, '$.radiologist_confidence') = 'High'
    """)
    high_conf_count = cur.fetchone()[0]
    high_confidence_rate_pct = round(high_conf_count * 100.0 / total_scans, 1) if total_scans else 0.0

    # Lesion type distribution
    cur.execute("""
        SELECT json_extract(fields_json, '$.lesion_label') lesion_label, COUNT(*) cnt
        FROM mri_findings
        GROUP BY lesion_label
        ORDER BY cnt DESC
    """)
    lesion_type_distribution = {r[0]: r[1] for r in cur.fetchall() if r[0] is not None}

    # Location distribution
    cur.execute("""
        SELECT COALESCE(json_extract(fields_json, '$.lesion_location'), 'Unknown') location, COUNT(*) cnt
        FROM mri_findings
        GROUP BY location
        ORDER BY cnt DESC
    """)
    location_distribution = {r[0]: r[1] for r in cur.fetchall()}

    # Laterality distribution
    cur.execute("""
        SELECT COALESCE(json_extract(fields_json, '$.laterality'), 'Unknown') laterality, COUNT(*) cnt
        FROM mri_findings
        GROUP BY laterality
        ORDER BY cnt DESC
    """)
    laterality_distribution = {r[0]: r[1] for r in cur.fetchall()}

    # Classification distribution
    cur.execute("""
        SELECT json_extract(fields_json, '$.classification_label') classification_label, COUNT(*) cnt
        FROM mri_findings
        GROUP BY classification_label
        ORDER BY cnt DESC
    """)
    classification_distribution = {r[0]: r[1] for r in cur.fetchall() if r[0] is not None}

    # Quality distribution
    cur.execute("""
        SELECT json_extract(fields_json, '$.quality') quality, COUNT(*) cnt
        FROM mri_findings
        GROUP BY quality
        ORDER BY cnt DESC
    """)
    quality_distribution = {r[0]: r[1] for r in cur.fetchall() if r[0] is not None}

    # Monthly timeline — scans per month (by created_at)
    cur.execute("""
        SELECT DISTINCT substr(created_at, 1, 7) month
        FROM mri_findings
        WHERE created_at IS NOT NULL
        ORDER BY month DESC
        LIMIT 12
    """)
    month_rows = cur.fetchall()
    months = sorted([r[0] for r in month_rows])

    monthly_timeline = []
    for m in months:
        cur.execute("""
            SELECT COUNT(*) FROM mri_findings
            WHERE substr(created_at, 1, 7) = ?
        """, (m,))
        scan_count = cur.fetchone()[0]

        monthly_timeline.append({
            "month": m,
            "scans": scan_count,
        })

    conn.close()
    return {
        "total_scans": total_scans,
        "lesional_count": lesional_count,
        "lesional_rate_pct": lesional_rate_pct,
        "hs_count": hs_count,
        "hs_rate_pct": hs_rate_pct,
        "avg_volume_asymmetry": avg_volume_asymmetry,
        "high_confidence_rate_pct": high_confidence_rate_pct,
        "lesion_type_distribution": lesion_type_distribution,
        "location_distribution": location_distribution,
        "laterality_distribution": laterality_distribution,
        "classification_distribution": classification_distribution,
        "quality_distribution": quality_distribution,
        "monthly_timeline": monthly_timeline,
    }


def breakdown():
    """MRI findings breakdown — per-patient table, HS patients list,
    low-confidence scans alert list, lesion_type x location cross-tab,
    lesion_type x laterality cross-tab."""
    conn = _conn()
    cur = conn.cursor()

    # -- Per-patient table --------------------------------------------------------
    cur.execute("""
        SELECT
            patient_id,
            json_extract(fields_json, '$.lesion_label') lesion_label,
            json_extract(fields_json, '$.lesion_location') lesion_location,
            json_extract(fields_json, '$.laterality') laterality,
            json_extract(fields_json, '$.classification_label') classification_label,
            json_extract(fields_json, '$.quality') quality,
            json_extract(fields_json, '$.hippocampal_sclerosis') hippocampal_sclerosis,
            json_extract(fields_json, '$.hippocampal_volume_asymmetry') volume_asymmetry,
            json_extract(fields_json, '$.radiologist_confidence') confidence,
            created_at
        FROM mri_findings
        ORDER BY created_at DESC
    """)
    per_patient = []
    for r in cur.fetchall():
        per_patient.append({
            "patient_id": r[0],
            "lesion_label": r[1],
            "location": r[2],
            "laterality": r[3],
            "classification": r[4],
            "quality": r[5],
            "hippocampal_sclerosis": r[6],
            "volume_asymmetry": round(float(r[7]), 3) if r[7] is not None else None,
            "confidence": r[8],
            "created_at": r[9],
        })

    # -- HS patients list ---------------------------------------------------------
    cur.execute("""
        SELECT
            patient_id,
            json_extract(fields_json, '$.lesion_location') lesion_location,
            json_extract(fields_json, '$.laterality') laterality,
            json_extract(fields_json, '$.hippocampal_volume_asymmetry') volume_asymmetry,
            json_extract(fields_json, '$.t2_flair_signal') t2_flair_signal,
            json_extract(fields_json, '$.radiologist_confidence') confidence,
            created_at
        FROM mri_findings
        WHERE json_extract(fields_json, '$.hippocampal_sclerosis') = 'Yes'
        ORDER BY CAST(json_extract(fields_json, '$.hippocampal_volume_asymmetry') AS REAL) DESC
    """)
    hs_patients = []
    for r in cur.fetchall():
        hs_patients.append({
            "patient_id": r[0],
            "location": r[1],
            "laterality": r[2],
            "volume_asymmetry": round(float(r[3]), 3) if r[3] is not None else None,
            "t2_flair_signal": r[4],
            "confidence": r[5],
            "created_at": r[6],
        })

    # -- Low-confidence scans alert list ------------------------------------------
    cur.execute("""
        SELECT
            patient_id,
            json_extract(fields_json, '$.lesion_label') lesion_label,
            json_extract(fields_json, '$.classification_label') classification_label,
            json_extract(fields_json, '$.quality') quality,
            json_extract(fields_json, '$.lesion_description') lesion_description,
            created_at
        FROM mri_findings
        WHERE json_extract(fields_json, '$.radiologist_confidence') = 'Low'
        ORDER BY created_at DESC
    """)
    low_confidence_scans = []
    for r in cur.fetchall():
        low_confidence_scans.append({
            "patient_id": r[0],
            "lesion_label": r[1],
            "classification": r[2],
            "quality": r[3],
            "lesion_description": r[4],
            "created_at": r[5],
        })

    # -- Lesion type x location cross-tab -----------------------------------------
    cur.execute("""
        SELECT
            json_extract(fields_json, '$.lesion_label') lesion_label,
            COALESCE(json_extract(fields_json, '$.lesion_location'), 'Unknown') location,
            COUNT(*) cnt
        FROM mri_findings
        GROUP BY lesion_label, location
        ORDER BY lesion_label, location
    """)
    lesion_location_crosstab = defaultdict(dict)
    for r in cur.fetchall():
        if r[0] is not None:
            lesion_location_crosstab[r[0]][r[1]] = r[2]
    lesion_location_crosstab = dict(lesion_location_crosstab)

    # -- Lesion type x laterality cross-tab ---------------------------------------
    cur.execute("""
        SELECT
            json_extract(fields_json, '$.lesion_label') lesion_label,
            COALESCE(json_extract(fields_json, '$.laterality'), 'Unknown') laterality,
            COUNT(*) cnt
        FROM mri_findings
        GROUP BY lesion_label, laterality
        ORDER BY lesion_label, laterality
    """)
    lesion_laterality_crosstab = defaultdict(dict)
    for r in cur.fetchall():
        if r[0] is not None:
            lesion_laterality_crosstab[r[0]][r[1]] = r[2]
    lesion_laterality_crosstab = dict(lesion_laterality_crosstab)

    conn.close()
    return {
        "per_patient": per_patient,
        "hs_patients": hs_patients,
        "low_confidence_scans": low_confidence_scans,
        "lesion_location_crosstab": lesion_location_crosstab,
        "lesion_laterality_crosstab": lesion_laterality_crosstab,
    }


def definitions():
    """MRI findings definitions — clinical glossary, lesion type descriptions,
    quality tier definitions, classification definitions, laterality definitions,
    and clinical notes about epilepsy MRI protocol."""
    return {
        "lesion_types": {
            "Hippocampal Sclerosis": "Neuronal loss and gliosis in the hippocampus, the most common pathological finding in temporal lobe epilepsy. Appears as hippocampal atrophy with increased T2/FLAIR signal.",
            "Focal Cortical Dysplasia": "A malformation of cortical development characterised by abnormal cortical lamination and dysmorphic neurons. A leading cause of drug-resistant focal epilepsy, often amenable to surgical resection.",
            "Non-Lesional": "No structural abnormality identified on MRI despite clinical seizures. Approximately 20-30% of epilepsy patients have non-lesional MRI, presenting a challenge for surgical planning.",
            "Cavernoma": "A cluster of abnormal, thinly-walled blood vessels (cavernous malformation) that can cause seizures through hemosiderin deposition on surrounding cortex.",
            "Encephalomalacia": "Softening or loss of brain tissue, typically resulting from prior infarction, trauma, or surgery. Appears as a well-defined area of cerebrospinal fluid signal on MRI.",
            "Vascular Malformation": "Abnormal vascular structures including arteriovenous malformations (AVMs) and developmental venous anomalies that may cause seizures through mass effect or haemorrhage.",
            "Normal": "MRI within normal limits — no structural abnormality, no signal change, no mass effect. Clinical correlation required as normal MRI does not exclude epilepsy.",
            "Low-Grade Tumour": "Slow-growing neoplasm (WHO Grade I-II) such as ganglioglioma or DNET (dysembryoplastic neuroepithelial tumour), commonly associated with chronic epilepsy and often surgically curable.",
        },
        "quality_tiers": {
            "Diagnostic": "High-quality study with optimal resolution, complete sequences, and no significant artefact. Fully adequate for clinical interpretation and surgical planning.",
            "Adequate": "Acceptable image quality with minor limitations (e.g., mild motion artefact) that do not significantly impair diagnostic interpretation.",
            "Suboptimal": "Reduced image quality due to motion artefact, incomplete sequences, or technical issues. Key findings may still be identifiable but with reduced confidence.",
            "Non-diagnostic": "Study severely compromised by artefact, motion, or incomplete acquisition. Unable to reliably exclude pathology; repeat imaging recommended.",
        },
        "classification_definitions": {
            "Lesional": "A definite structural abnormality is identified on MRI that is concordant with the electroclinical localisation and is a plausible cause of the patient's epilepsy.",
            "Non-Lesional": "No structural abnormality is identified on MRI despite adequate image quality and an epilepsy-specific protocol.",
            "Equivocal": "A possible abnormality is identified but its significance is uncertain — further imaging, post-processing, or clinical correlation is required.",
            "Normal": "MRI is entirely within normal limits with no structural or signal abnormality identified.",
        },
        "laterality_definitions": {
            "Left": "Lesion or abnormality localised to the left cerebral hemisphere.",
            "Right": "Lesion or abnormality localised to the right cerebral hemisphere.",
            "Bilateral": "Abnormality present in both cerebral hemispheres, which may indicate a diffuse or multifocal process.",
            "Unknown": "Laterality could not be determined — either the scan is non-lesional or the lesion does not lateralise clearly.",
        },
        "glossary": [
            {"term": "MRI", "definition": "Magnetic Resonance Imaging — a non-invasive imaging technique using strong magnetic fields and radio waves to produce detailed images of brain structure. The gold standard for structural evaluation in epilepsy."},
            {"term": "FLAIR", "definition": "Fluid-Attenuated Inversion Recovery — an MRI sequence that suppresses cerebrospinal fluid signal, making periventricular and cortical lesions more conspicuous."},
            {"term": "T2", "definition": "T2-weighted imaging — an MRI sequence sensitive to water content and oedema. Pathological tissue often appears hyperintense (bright) on T2."},
            {"term": "Hippocampal Sclerosis", "definition": "Neuronal loss and gliosis in the hippocampus, the most common structural finding in mesial temporal lobe epilepsy. Diagnosed by hippocampal atrophy and T2/FLAIR signal increase."},
            {"term": "FCD", "definition": "Focal Cortical Dysplasia — a malformation of cortical development with abnormal neuronal organisation. Classified into types I, II, and III (ILAE classification)."},
            {"term": "Cavernoma", "definition": "Cavernous malformation — a benign vascular lesion composed of sinusoidal vascular channels surrounded by a hemosiderin ring, visible on susceptibility-weighted imaging."},
            {"term": "Lesional vs Non-Lesional", "definition": "Lesional epilepsy has an identifiable structural cause on MRI; non-lesional epilepsy has no visible structural abnormality, making surgical planning more reliant on electrophysiology and functional imaging."},
            {"term": "Equivocal", "definition": "An MRI finding of uncertain clinical significance — the abnormality may or may not be the epileptogenic substrate. Requires further investigation or multidisciplinary review."},
            {"term": "Volume Asymmetry", "definition": "Hippocampal volume asymmetry index — a quantitative measure comparing left and right hippocampal volumes. Values > 0.1 suggest significant asymmetry consistent with unilateral hippocampal sclerosis."},
            {"term": "3T MRI", "definition": "MRI performed at 3 Tesla field strength, providing higher spatial resolution and signal-to-noise ratio compared to 1.5T. Recommended for epilepsy protocols to improve lesion detection."},
            {"term": "Hemosiderin", "definition": "An iron-storage complex deposited in tissue after haemorrhage. Appears as a dark ring on T2/susceptibility-weighted MRI sequences, characteristic of cavernomas and prior bleeds."},
            {"term": "Encephalomalacia", "definition": "Focal brain tissue loss and softening, usually secondary to prior ischaemic injury, trauma, or surgery. Appears as a well-defined CSF-signal cavity on MRI."},
        ],
        "clinical_notes": [
            "An epilepsy-specific MRI protocol (3T, thin-slice coronal T2/FLAIR, volumetric T1) significantly improves lesion detection compared to standard brain MRI.",
            "Up to 30% of patients with drug-resistant epilepsy have a non-lesional MRI, necessitating advanced post-processing (e.g., MAP, voxel-based morphometry) or repeat imaging at higher field strength.",
            "Hippocampal volume asymmetry > 0.1 combined with increased T2/FLAIR signal has high specificity for hippocampal sclerosis and is associated with favourable surgical outcomes.",
            "Dual pathology (e.g., HS + FCD) occurs in approximately 15% of surgical epilepsy cases and requires careful evaluation as it may affect surgical prognosis.",
            "Low radiologist confidence should trigger multidisciplinary review (MDT) including re-acquisition with optimised protocol, advanced post-processing, or complementary imaging (PET/SPECT).",
            "Cavernomas with a hemosiderin ring are epileptogenic due to iron deposition on surrounding cortex; lesionectomy with removal of the hemosiderin rim is associated with better seizure outcomes.",
        ],
    }


if __name__ == "__main__":
    import json

    print("=== OVERVIEW ===")
    ov = overview()
    print(json.dumps(ov, indent=2))

    print("\n=== BREAKDOWN ===")
    bk = breakdown()
    # Truncate per_patient for readability
    bk_display = dict(bk)
    bk_display["per_patient"] = bk["per_patient"][:5]
    print(json.dumps(bk_display, indent=2))

    print("\n=== DEFINITIONS (keys only) ===")
    df = definitions()
    print(list(df.keys()))
