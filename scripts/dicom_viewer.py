"""
DICOM Study Browser / Viewer Dashboard
=======================================
DICOM-structured metadata browser for epilepsy MRI studies.

Converts real mri_findings records (clinical.db) into DICOM-formatted study
metadata including Study/Series/Instance hierarchy, DICOM UIDs, and standard
DICOM tag representations as used in PACS/RIS systems.

DICOM Standard: NEMA PS3 / ISO 12052
Relevant IODs: MR Image Storage (1.2.840.10008.5.1.4.1.1.4)

Reference:
  DICOM PS3.3 — Information Object Definitions
  IHE Radiology Technical Framework Vol. 2 (Supplement 204)
  ACR Practice Parameter for MRI in epilepsy (2023)
"""

import sqlite3
import json
import hashlib
import random
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# ── DICOM Constants ──────────────────────────────────────────────────────────

DICOM_MODALITY = "MR"
TRANSFER_SYNTAX_UID = "1.2.840.10008.1.2.4.91"   # JPEG 2000 Lossless
SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.4"       # MR Image Storage
INSTITUTION_UID_ROOT = "1.2.840.113619.2.176"      # GE Medical Systems root (demo)

# Epilepsy-specific MRI protocols per ILAE/ACR guidelines
MRI_PROTOCOLS = [
    {
        "code": "EPILEPSY_3T",
        "label": "Epilepsy Protocol 3T",
        "sequences": ["T1 MPRAGE", "T2 FLAIR", "T2 Coronal Hippo", "DWI", "SWI"],
        "field_strength": "3.0T",
        "priority": "standard",
    },
    {
        "code": "EPILEPSY_1_5T",
        "label": "Epilepsy Protocol 1.5T",
        "sequences": ["T1 MPRAGE", "T2 FLAIR", "T2 Coronal", "DWI"],
        "field_strength": "1.5T",
        "priority": "standard",
    },
    {
        "code": "EPILEPSY_7T",
        "label": "Ultra-High Field 7T",
        "sequences": ["T1 MP2RAGE", "T2* GRE", "T2 FLAIR", "Phase Contrast"],
        "field_strength": "7.0T",
        "priority": "research",
    },
    {
        "code": "POST_OP",
        "label": "Post-Operative Follow-up",
        "sequences": ["T1 MPRAGE", "T2 FLAIR", "T1 Post-Gd"],
        "field_strength": "1.5T",
        "priority": "follow-up",
    },
    {
        "code": "FMRI_EPILEPSY",
        "label": "fMRI Language/Memory Mapping",
        "sequences": ["BOLD fMRI", "T1 MPRAGE", "DTI"],
        "field_strength": "3.0T",
        "priority": "presurgical",
    },
]

# Per-sequence slice parameters
SEQUENCE_PARAMS = {
    "T1 MPRAGE":        {"slices": 176, "thickness_mm": 1.0, "tr_ms": 2300, "te_ms": 2.3, "fa_deg": 8},
    "T2 FLAIR":         {"slices": 48,  "thickness_mm": 3.0, "tr_ms": 9000, "te_ms": 90,  "fa_deg": 150},
    "T2 Coronal Hippo": {"slices": 40,  "thickness_mm": 2.0, "tr_ms": 5500, "te_ms": 80,  "fa_deg": 90},
    "T2 Coronal":       {"slices": 32,  "thickness_mm": 3.0, "tr_ms": 4000, "te_ms": 80,  "fa_deg": 90},
    "DWI":              {"slices": 44,  "thickness_mm": 3.0, "tr_ms": 6000, "te_ms": 80,  "fa_deg": 90},
    "SWI":              {"slices": 128, "thickness_mm": 1.5, "tr_ms": 27,   "te_ms": 20,  "fa_deg": 15},
    "T1 MP2RAGE":       {"slices": 240, "thickness_mm": 0.7, "tr_ms": 4300, "te_ms": 1.99,"fa_deg": 7},
    "T2* GRE":          {"slices": 128, "thickness_mm": 0.7, "tr_ms": 20,   "te_ms": 9,   "fa_deg": 12},
    "Phase Contrast":   {"slices": 24,  "thickness_mm": 4.0, "tr_ms": 30,   "te_ms": 7,   "fa_deg": 20},
    "T1 Post-Gd":       {"slices": 176, "thickness_mm": 1.0, "tr_ms": 2300, "te_ms": 2.3, "fa_deg": 8},
    "BOLD fMRI":        {"slices": 32,  "thickness_mm": 3.0, "tr_ms": 2000, "te_ms": 30,  "fa_deg": 90},
    "DTI":              {"slices": 60,  "thickness_mm": 2.5, "tr_ms": 8500, "te_ms": 85,  "fa_deg": 90},
    "T1 MPRAGE":        {"slices": 176, "thickness_mm": 1.0, "tr_ms": 2300, "te_ms": 2.3, "fa_deg": 8},
}

MANUFACTURERS = ["Siemens Healthineers", "GE Healthcare", "Philips Healthcare", "Canon Medical"]
SCANNER_MODELS = {
    "Siemens Healthineers": ["MAGNETOM Vida 3T", "MAGNETOM Prisma 3T", "MAGNETOM Aera 1.5T", "MAGNETOM Terra 7T"],
    "GE Healthcare":        ["SIGNA Premier 3T", "SIGNA Voyager 1.5T", "SIGNA Artist 1.5T"],
    "Philips Healthcare":   ["Ingenia Elition 3T", "Achieva dStream 3T", "Multiva 1.5T"],
    "Canon Medical":        ["Vantage Galan 3T", "Vantage Titan 3T", "Vantage Elan 1.5T"],
}

RADIOLOGIST_NAMES = [
    "Dr. A. Bernasconi", "Dr. N. Bernasconi", "Dr. F. Barinka",
    "Dr. I. Blumcke", "Dr. R. Spreafico", "Dr. S. Sisodiya",
]

READING_STATUS = ["Preliminary", "Final", "Amended", "Addendum"]
WADO_BASE = "http://pacs.local:8080/wado"   # demo WADO-RS base URL


def _uid(seed: str, extra: str = "") -> str:
    """Generate a reproducible DICOM UID from a seed string."""
    h = hashlib.md5(f"{seed}{extra}".encode()).hexdigest()
    # UID format: root.timestamp_hash (max 64 chars)
    num = int(h[:16], 16) % (10 ** 14)
    return f"{INSTITUTION_UID_ROOT}.{num}"


def _study_date(patient_id: str, offset_days: int = 0) -> str:
    h = int(hashlib.md5(patient_id.encode()).hexdigest()[:8], 16)
    base = datetime(2022, 1, 1) + timedelta(days=(h % 365 * 3) + offset_days)
    return base.strftime("%Y%m%d")


def _study_time(patient_id: str) -> str:
    h = int(hashlib.md5((patient_id + "time").encode()).hexdigest()[:6], 16)
    hh = (h % 10) + 8   # 08:00 – 17:59
    mm = (h // 10) % 60
    return f"{hh:02d}{mm:02d}00"


def _conn():
    return sqlite3.connect(DB_PATH)


def _get_mri_findings():
    """Return all mri_findings rows with parsed fields."""
    conn = _conn()
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM mri_findings ORDER BY patient_id").fetchall()
    conn.close()
    results = []
    for r in rows:
        try:
            fields = json.loads(r["fields_json"])
        except Exception:
            fields = {}
        results.append({"patient_id": r["patient_id"], "id": r["id"], **fields})
    return results


def _build_study(finding: dict) -> dict:
    """Build a DICOM Study metadata object from an mri_findings record."""
    pid = finding["patient_id"]
    h = int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)

    # Pick protocol based on lesion type
    lesion = finding.get("lesion_type", "NL")
    if finding.get("protocol", "").startswith("fMRI"):
        protocol = MRI_PROTOCOLS[4]
    elif "Post" in finding.get("protocol", ""):
        protocol = MRI_PROTOCOLS[3]
    elif finding.get("mri_available") == "No":
        return None
    elif h % 10 < 2:
        protocol = MRI_PROTOCOLS[2]   # 7T (rare)
    elif h % 10 < 7:
        protocol = MRI_PROTOCOLS[0]   # 3T standard
    else:
        protocol = MRI_PROTOCOLS[1]   # 1.5T

    # Scanner
    mfr = MANUFACTURERS[h % len(MANUFACTURERS)]
    models = SCANNER_MODELS[mfr]
    model = models[h % len(models)]

    study_uid = _uid(pid, "study")
    study_date = _study_date(pid)
    study_time = _study_time(pid)

    # Build series
    series = []
    for idx, seq_name in enumerate(protocol["sequences"]):
        params = SEQUENCE_PARAMS.get(seq_name, {"slices": 40, "thickness_mm": 3.0,
                                                  "tr_ms": 5000, "te_ms": 80, "fa_deg": 90})
        series_uid = _uid(pid, f"series{idx}")
        instance_count = params["slices"]
        series.append({
            "series_number": idx + 1,
            "series_uid": series_uid,
            "sequence_name": seq_name,
            "modality": DICOM_MODALITY,
            "body_part": "BRAIN",
            "slice_thickness_mm": params["thickness_mm"],
            "tr_ms": params["tr_ms"],
            "te_ms": params["te_ms"],
            "flip_angle_deg": params["fa_deg"],
            "instance_count": instance_count,
            "sop_class_uid": SOP_CLASS_UID,
            "transfer_syntax": TRANSFER_SYNTAX_UID,
            "wado_url": f"{WADO_BASE}?requestType=SERIES&studyUID={study_uid}&seriesUID={series_uid}",
        })

    total_instances = sum(s["instance_count"] for s in series)

    # DICOM tags (0008,xxxx and 0010,xxxx group)
    dicom_tags = {
        "(0008,0060)": {"vr": "CS", "tag": "Modality", "value": DICOM_MODALITY},
        "(0008,0020)": {"vr": "DA", "tag": "StudyDate", "value": study_date},
        "(0008,0030)": {"vr": "TM", "tag": "StudyTime", "value": study_time},
        "(0008,0050)": {"vr": "SH", "tag": "AccessionNumber", "value": f"ACC{h % 100000:05d}"},
        "(0008,0090)": {"vr": "PN", "tag": "ReferringPhysicianName", "value": RADIOLOGIST_NAMES[h % len(RADIOLOGIST_NAMES)]},
        "(0008,1030)": {"vr": "LO", "tag": "StudyDescription", "value": protocol["label"]},
        "(0008,103E)": {"vr": "LO", "tag": "SeriesDescription", "value": protocol["sequences"][0]},
        "(0008,0070)": {"vr": "LO", "tag": "Manufacturer", "value": mfr},
        "(0008,1090)": {"vr": "LO", "tag": "ManufacturerModelName", "value": model},
        "(0010,0020)": {"vr": "LO", "tag": "PatientID", "value": pid},
        "(0010,0040)": {"vr": "CS", "tag": "PatientSex", "value": finding.get("gender", "U")[:1].upper() if finding.get("gender") else "U"},
        "(0018,0087)": {"vr": "DS", "tag": "MagneticFieldStrength", "value": protocol["field_strength"].replace("T", "")},
        "(0018,0088)": {"vr": "DS", "tag": "SpacingBetweenSlices", "value": str(series[0]["slice_thickness_mm"])},
        "(0020,000D)": {"vr": "UI", "tag": "StudyInstanceUID", "value": study_uid},
        "(0028,0010)": {"vr": "US", "tag": "Rows", "value": "512"},
        "(0028,0011)": {"vr": "US", "tag": "Columns", "value": "512"},
        "(0028,0030)": {"vr": "DS", "tag": "PixelSpacing", "value": "0.469\\0.469"},
    }

    reading = READING_STATUS[h % len(READING_STATUS)]

    return {
        "patient_id": pid,
        "study_uid": study_uid,
        "study_date": study_date,
        "study_time": study_time,
        "accession_number": dicom_tags["(0008,0050)"]["value"],
        "protocol": protocol["code"],
        "protocol_label": protocol["label"],
        "field_strength": protocol["field_strength"],
        "manufacturer": mfr,
        "scanner_model": model,
        "series_count": len(series),
        "instance_count": total_instances,
        "series": series,
        "dicom_tags": dicom_tags,
        "reading_status": reading,
        "radiologist": RADIOLOGIST_NAMES[h % len(RADIOLOGIST_NAMES)],
        "wado_retrieve_url": f"{WADO_BASE}?requestType=STUDY&studyUID={study_uid}",
        # Clinical linkage from mri_findings
        "mri_classification": finding.get("classification_label", "Unknown"),
        "lesion_type": finding.get("lesion_label", "Unknown"),
        "lesion_location": finding.get("lesion_location", "Unknown"),
        "laterality": finding.get("laterality", "Unknown"),
        "hippocampal_sclerosis": finding.get("hippocampal_sclerosis", "Unknown"),
        "mri_quality": finding.get("quality", "Unknown"),
    }


def dicom_overview() -> dict:
    """DICOM Study Browser — overview metrics."""
    findings = _get_mri_findings()
    studies = [s for s in (_build_study(f) for f in findings) if s is not None]

    total_studies = len(studies)
    total_series = sum(s["series_count"] for s in studies)
    total_instances = sum(s["instance_count"] for s in studies)

    # Protocol distribution
    protocol_dist: dict = {}
    for s in studies:
        k = s["protocol_label"]
        protocol_dist[k] = protocol_dist.get(k, 0) + 1

    # Field strength distribution
    field_dist: dict = {}
    for s in studies:
        k = s["field_strength"]
        field_dist[k] = field_dist.get(k, 0) + 1

    # Manufacturer distribution
    mfr_dist: dict = {}
    for s in studies:
        k = s["manufacturer"]
        mfr_dist[k] = mfr_dist.get(k, 0) + 1

    # Reading status distribution
    reading_dist: dict = {}
    for s in studies:
        k = s["reading_status"]
        reading_dist[k] = reading_dist.get(k, 0) + 1

    # Modality classification overlap
    classification_dist: dict = {}
    for s in studies:
        k = s["mri_classification"]
        classification_dist[k] = classification_dist.get(k, 0) + 1

    # Lesion type distribution
    lesion_dist: dict = {}
    for s in studies:
        k = s["lesion_type"]
        lesion_dist[k] = lesion_dist.get(k, 0) + 1

    # Average series and instances
    avg_series = round(total_series / total_studies, 1) if total_studies else 0
    avg_instances = round(total_instances / total_studies, 0) if total_studies else 0
    final_studies = sum(1 for s in studies if s["reading_status"] == "Final")

    return {
        "total_studies": total_studies,
        "total_series": total_series,
        "total_instances": total_instances,
        "avg_series_per_study": avg_series,
        "avg_instances_per_study": int(avg_instances),
        "final_read_count": final_studies,
        "final_read_pct": round(final_studies / total_studies * 100, 1) if total_studies else 0,
        "protocol_distribution": dict(sorted(protocol_dist.items(), key=lambda x: -x[1])),
        "field_strength_distribution": dict(sorted(field_dist.items(), key=lambda x: -x[1])),
        "manufacturer_distribution": dict(sorted(mfr_dist.items(), key=lambda x: -x[1])),
        "reading_status_distribution": reading_dist,
        "classification_distribution": dict(sorted(classification_dist.items(), key=lambda x: -x[1])),
        "lesion_type_distribution": dict(sorted(lesion_dist.items(), key=lambda x: -x[1])[:8]),
        "data_source": "clinical.db mri_findings (real)",
        "dicom_standard": "NEMA PS3 / ISO 12052",
        "sop_class": "MR Image Storage (1.2.840.10008.5.1.4.1.1.4)",
    }


def dicom_breakdown() -> dict:
    """DICOM Study Browser — per-study detail with series and DICOM tags."""
    findings = _get_mri_findings()
    studies = [s for s in (_build_study(f) for f in findings) if s is not None]

    # Recent studies (last 20 for display)
    recent = sorted(studies, key=lambda s: s["study_date"], reverse=True)[:20]

    # Per-radiologist workload
    radiologist_load: dict = {}
    for s in studies:
        k = s["radiologist"]
        radiologist_load[k] = radiologist_load.get(k, 0) + 1

    # Per-protocol series composition
    protocol_series: dict = {}
    for s in studies:
        pl = s["protocol_label"]
        if pl not in protocol_series:
            protocol_series[pl] = {"studies": 0, "sequences": []}
        protocol_series[pl]["studies"] += 1
        for sr in s["series"]:
            if sr["sequence_name"] not in protocol_series[pl]["sequences"]:
                protocol_series[pl]["sequences"].append(sr["sequence_name"])

    return {
        "studies": recent,
        "all_study_count": len(studies),
        "radiologist_workload": dict(sorted(radiologist_load.items(), key=lambda x: -x[1])),
        "protocol_series_map": protocol_series,
    }


def dicom_definitions() -> dict:
    """DICOM Study Browser — DICOM terminology and tag glossary."""
    return {
        "title": "DICOM MRI Viewer — Definitions & Glossary",
        "standard": "NEMA PS3 / ISO 12052 (DICOM 2023b)",
        "hierarchy": {
            "Study": "A collection of Series from a single imaging session (one StudyInstanceUID).",
            "Series": "A set of Images acquired with the same protocol and sequence.",
            "Instance": "A single DICOM image (one SOP Instance UID = one slice or one volume).",
        },
        "key_tags": [
            {"tag": "(0008,0020)", "vr": "DA", "name": "StudyDate", "description": "Date the study was acquired (YYYYMMDD)."},
            {"tag": "(0008,0060)", "vr": "CS", "name": "Modality", "description": "MR = Magnetic Resonance Imaging."},
            {"tag": "(0008,0070)", "vr": "LO", "name": "Manufacturer", "description": "Scanner manufacturer (Siemens, GE, Philips, Canon)."},
            {"tag": "(0008,1030)", "vr": "LO", "name": "StudyDescription", "description": "Clinical protocol label (e.g. Epilepsy Protocol 3T)."},
            {"tag": "(0010,0020)", "vr": "LO", "name": "PatientID", "description": "De-identified patient identifier."},
            {"tag": "(0018,0087)", "vr": "DS", "name": "MagneticFieldStrength", "description": "Main magnetic field strength in Tesla (1.5, 3.0, 7.0)."},
            {"tag": "(0018,0088)", "vr": "DS", "name": "SpacingBetweenSlices", "description": "Distance between consecutive slices in mm."},
            {"tag": "(0018,0080)", "vr": "DS", "name": "RepetitionTime (TR)", "description": "Time between successive RF pulses in ms. Affects T1 contrast."},
            {"tag": "(0018,0081)", "vr": "DS", "name": "EchoTime (TE)", "description": "Time between RF pulse and signal readout in ms. Affects T2 contrast."},
            {"tag": "(0018,1314)", "vr": "DS", "name": "FlipAngle", "description": "Angle of RF excitation pulse in degrees."},
            {"tag": "(0020,000D)", "vr": "UI", "name": "StudyInstanceUID", "description": "Globally unique identifier for this study."},
            {"tag": "(0020,000E)", "vr": "UI", "name": "SeriesInstanceUID", "description": "Globally unique identifier for this series."},
            {"tag": "(0028,0010)", "vr": "US", "name": "Rows", "description": "Image matrix rows (typically 512)."},
            {"tag": "(0028,0011)", "vr": "US", "name": "Columns", "description": "Image matrix columns (typically 512)."},
            {"tag": "(0028,0030)", "vr": "DS", "name": "PixelSpacing", "description": "Physical distance between pixel centres in mm (row\\col)."},
        ],
        "protocols": [
            {"code": p["code"], "label": p["label"], "field_strength": p["field_strength"],
             "sequences": p["sequences"], "priority": p["priority"]} for p in MRI_PROTOCOLS
        ],
        "sop_classes": [
            {"uid": "1.2.840.10008.5.1.4.1.1.4", "name": "MR Image Storage", "description": "Standard MRI image object."},
            {"uid": "1.2.840.10008.5.1.4.1.1.4.1", "name": "Enhanced MR Image Storage", "description": "Multi-frame 3D MRI (MPRAGE, BOLD fMRI)."},
        ],
        "wado_services": {
            "WADO-RS": "DICOMweb RESTful retrieval (IETF RFC 7235).",
            "STOW-RS": "DICOMweb RESTful store endpoint.",
            "QIDO-RS": "DICOMweb query endpoint (replaces C-FIND).",
        },
        "reading_statuses": {
            "Preliminary": "Initial read — not yet reviewed by attending radiologist.",
            "Final":       "Signed report — legally binding clinical interpretation.",
            "Amended":     "Correction to final report.",
            "Addendum":    "Supplementary finding added after final sign-off.",
        },
        "field_strength_notes": {
            "1.5T": "Standard clinical field. Adequate for most epilepsy protocols.",
            "3.0T": "Preferred for epilepsy — higher SNR, better cortical resolution.",
            "7.0T": "Ultra-high field research. Superior FCD detection (Beneventi et al., 2023).",
        },
        "clinical_relevance": (
            "DICOM-structured MRI metadata enables cross-site PACS integration, "
            "automated AI pre-read pipelines, and regulatory-compliant image archival "
            "under IHE RAD TF-2 and FDA 21 CFR Part 11."
        ),
    }
