"""Data Completeness Dashboard — Per-patient data field completeness across clinical categories.

All data derived from REAL clinical data in data/clinical.db.  No synthetic or
fabricated data is used.

Data completeness is a cornerstone of clinical data quality.  In EEG-based
neurological research and clinical decision support, incomplete records can
introduce bias, reduce model accuracy, and compromise patient safety.  The FDA
21 CFR Part 11 and ICH E6(R2) Good Clinical Practice guidelines mandate that
clinical data be "attributable, legible, contemporaneous, original, and accurate"
(ALCOA).  Missing data fields undermine each of these principles.

This module systematically evaluates 9 categories of clinical data requirements
against the actual records stored in clinical.db.  For each patient, the module
checks whether data exists in the relevant tables and fields, producing a
completeness percentage per category and overall.

Categories assessed:
  1. EEG Signal Data — Raw signal acquisition, sampling, event markers, annotations
  2. Clinical History — Diagnosis, seizure classification, comorbidities, imaging
  3. Medication — Current medications, AED drugs, dosage, refill history
  4. Imaging — MRI findings and reports
  5. Neuropsychological — MoCA, PHQ-9, GAD-7, cognitive tracking, QoL
  6. Outcomes — Hospitalization, seizure recurrence, follow-up visits
  7. Expert Review — Clinician review, expert agreement, audit trail, decision log
  8. Signal Quality — Artifact reports, SNR metrics, noise labels
  9. Demographics — Age, sex, occupation

Completeness scoring:
  - Each field is binary: present (1) or absent (0) for a given patient
  - Category completeness = fields_present / fields_total * 100
  - Overall completeness = sum of all fields_present / sum of all fields_total * 100

Quality tiers:
  - Excellent (>=90%): Data ready for regulatory submission
  - Good (75-89%): Minor gaps, suitable for analysis with caveats
  - Fair (50-74%): Significant gaps, requires remediation before analysis
  - Poor (<50%): Critical gaps, data not reliable for clinical decisions

References:
  Kahn MG et al. A Harmonized Data Quality Assessment Terminology and Framework
    for the Secondary Use of EHR Data. eGEMs 2016;4(1):18.
  Weiskopf NG, Weng C. Methods and dimensions of electronic health record data
    quality assessment. JAMIA 2013;20(1):144-151.
  OHDSI Data Quality Dashboard. https://ohdsi.github.io/DataQualityDashboard/

Author: Research Team
"""
import sqlite3
import json
from pathlib import Path
from collections import Counter, defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def _scalar(query, params=()):
    with _conn() as c:
        row = c.execute(query, params).fetchone()
        return row[0] if row else 0


def _patient_ids():
    """Return sorted list of all patient_ids from patients table."""
    return [r["patient_id"] for r in _rows("SELECT patient_id FROM patients ORDER BY patient_id")]


def _patients_with_data_in(table, extra_where="", params=()):
    """Return set of patient_ids that have at least one row in the given table."""
    q = f"SELECT DISTINCT patient_id FROM {table}"
    if extra_where:
        q += f" WHERE {extra_where}"
    return {r["patient_id"] for r in _rows(q, params)}


def _patients_with_field_in_json(table, json_col="fields_json", field_key=None):
    """Return set of patient_ids where the JSON column contains a non-empty field."""
    rows = _rows(f"SELECT patient_id, {json_col} FROM {table}")
    result = set()
    for r in rows:
        pid = r["patient_id"]
        raw = r.get(json_col, "{}")
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if field_key:
            val = data.get(field_key)
            if val is not None and val != "" and val != [] and val != {}:
                result.add(pid)
        else:
            # Just check JSON is non-empty
            if data:
                result.add(pid)
    return result


def _patients_with_column(table, column):
    """Return set of patient_ids where the given column is not null/empty."""
    q = f"SELECT DISTINCT patient_id FROM {table} WHERE {column} IS NOT NULL AND {column} != ''"
    return {r["patient_id"] for r in _rows(q)}


# ─────────────────────────────────────────────────────────────────────
# Field definitions: each field maps to a check function
# ─────────────────────────────────────────────────────────────────────

CATEGORIES = {
    "EEG Signal Data": {
        "fields": {
            "EDF/BDF raw signal": lambda: _patients_with_data_in("eeg_acquisition"),
            "Sampling frequency": lambda: _patients_with_field_in_json("eeg_acquisition", field_key="sampling_rate"),
            "Recording duration": lambda: _patients_with_field_in_json("eeg_acquisition", field_key="duration_min"),
            "Event markers": lambda: _patients_with_data_in("seizure_diary"),
            "Annotation file": lambda: _patients_with_data_in("artifact_annotations"),
            "Channel coordinates": lambda: _patients_with_data_in("channel_quality"),
            "Artifact labels": lambda: _patients_with_field_in_json("artifact_annotations"),
        },
        "description": "Raw EEG signal acquisition data including sampling parameters, event markers, and channel metadata.",
    },
    "Clinical History": {
        "fields": {
            "Diagnosis": lambda: _patients_with_column("patients", "disease"),
            "Seizure classification": lambda: _patients_with_data_in("analyses"),
            "Comorbidities": lambda: _patients_with_data_in("comorbidities"),
            "Neurological findings": lambda: _patients_with_data_in("mri_findings"),
        },
        "description": "Clinical history including primary diagnosis, seizure classification, comorbidities, and neurological findings from imaging.",
    },
    "Medication": {
        "fields": {
            "Current medication": lambda: _patients_with_data_in("medications"),
            "AED drugs": lambda: _patients_with_data_in("medication_adherence"),
            "Dosage": lambda: _patients_with_column("medication_adherence", "dose_mg"),
            "Medication history": lambda: _patients_with_data_in("medication_refills"),
        },
        "description": "Medication records including current prescriptions, AED adherence tracking, dosage data, and refill history.",
    },
    "Imaging": {
        "fields": {
            "MRI report": lambda: _patients_with_data_in("mri_findings"),
        },
        "description": "Neuroimaging data including MRI reports and structural findings.",
    },
    "Neuropsychological": {
        "fields": {
            "MoCA": lambda: _patients_with_data_in("assessments", "instrument='MOCA'"),
            "Depression score (PHQ-9)": lambda: _patients_with_data_in("assessments", "instrument='PHQ9'"),
            "Anxiety score (GAD-7)": lambda: _patients_with_data_in("assessments", "instrument='GAD7'"),
            "Cognitive assessment": lambda: _patients_with_data_in("cognitive_decline_tracking"),
            "Quality of life": lambda: _patients_with_data_in("pro_outcomes"),
        },
        "description": "Neuropsychological assessments including MoCA, PHQ-9, GAD-7, cognitive tracking, and patient-reported quality of life.",
    },
    "Outcomes": {
        "fields": {
            "Hospitalization": lambda: _patients_with_data_in("appointments", "appt_type='Emergency'"),
            "Seizure recurrence": lambda: _patients_with_data_in("seizure_diary"),
            "Follow-up": lambda: _patients_with_data_in("patient_appointments"),
        },
        "description": "Clinical outcome data including emergency visits, seizure recurrence events, and follow-up appointment records.",
    },
    "Expert Review": {
        "fields": {
            "Clinician review": lambda: _patients_with_data_in("expert_reviews"),
            "Expert agreement": lambda: _patients_with_column("expert_reviews", "agree_with_ai"),
            "Audit trail": lambda: _patients_with_data_in("transaction_log"),
            "Decision log": lambda: _patients_with_data_in("clinical_decisions"),
        },
        "description": "Expert review records including clinician reviews, AI agreement assessments, audit trail entries, and clinical decision logs.",
    },
    "Signal Quality": {
        "fields": {
            "Artifact report": lambda: _patients_with_data_in("artifact_annotations"),
            "Signal quality (SNR)": lambda: _patients_with_data_in("channel_quality"),
            "Noise labels": lambda: _patients_with_field_in_json("artifact_annotations"),
        },
        "description": "Signal quality metrics including artifact annotations, SNR measurements, and noise classification labels.",
    },
    "Demographics": {
        "fields": {
            "Age": lambda: _patients_with_column("patients", "age"),
            "Sex": lambda: _patients_with_column("patients", "gender"),
            "Occupation": lambda: _patients_with_column("patient_demographics", "occupation"),
        },
        "description": "Demographic data including age, sex, and occupational information.",
    },
}

TOTAL_FIELDS = sum(len(cat["fields"]) for cat in CATEGORIES.values())


def _compute_completeness():
    """Compute per-patient, per-category completeness matrix.

    Returns:
        all_pids: sorted list of patient IDs
        matrix: dict[patient_id][category_name] = {fields_present, fields_total, present_fields, missing_fields}
        field_presence: dict[field_name] = set of patient_ids that have it
    """
    all_pids = _patient_ids()

    # Pre-compute field presence sets
    field_presence = {}
    for cat_name, cat_info in CATEGORIES.items():
        for field_name, check_fn in cat_info["fields"].items():
            field_presence[field_name] = check_fn()

    # Build per-patient matrix
    matrix = {}
    for pid in all_pids:
        matrix[pid] = {}
        for cat_name, cat_info in CATEGORIES.items():
            present = []
            missing = []
            for field_name in cat_info["fields"]:
                if pid in field_presence[field_name]:
                    present.append(field_name)
                else:
                    missing.append(field_name)
            matrix[pid][cat_name] = {
                "fields_present": len(present),
                "fields_total": len(cat_info["fields"]),
                "present_fields": present,
                "missing_fields": missing,
            }

    return all_pids, matrix, field_presence


# ─────────────────────────────────────────────────────────────────────
# 1. overview()
# ─────────────────────────────────────────────────────────────────────

def overview():
    """KPIs and aggregate statistics for data completeness.

    Returns a dict with:
      - total_patients (int)
      - total_fields (int)
      - overall_completeness_pct (float)
      - per_category (list of {category, fields_total, fields_present_avg, completeness_pct})
      - completeness_distribution (list of {range, count})
      - top_missing_fields (list of {field, missing_count, missing_pct})
    """
    all_pids, matrix, field_presence = _compute_completeness()
    total_patients = len(all_pids)

    if total_patients == 0:
        return {
            "available": False,
            "reason": "No patient data found",
            "kpis": {"total_patients": 0, "total_fields": 0, "overall_completeness_pct": 0.0},
            "per_category": [],
            "completeness_distribution": [],
            "top_missing_fields": [],
        }

    # Per-patient overall completeness
    patient_completeness = {}
    for pid in all_pids:
        total_present = sum(matrix[pid][cat]["fields_present"] for cat in CATEGORIES)
        patient_completeness[pid] = round(total_present / TOTAL_FIELDS * 100, 1)

    overall_pct = round(sum(patient_completeness.values()) / total_patients, 1)

    # Per-category aggregate
    per_category = []
    for cat_name, cat_info in CATEGORIES.items():
        fields_total = len(cat_info["fields"])
        # Average fields present across patients
        avg_present = sum(matrix[pid][cat_name]["fields_present"] for pid in all_pids) / total_patients
        cat_pct = round(avg_present / fields_total * 100, 1) if fields_total > 0 else 0.0
        per_category.append({
            "category": cat_name,
            "fields_total": fields_total,
            "fields_present_avg": round(avg_present, 2),
            "completeness_pct": cat_pct,
        })

    # Sort by completeness ascending (worst first for attention)
    per_category.sort(key=lambda x: x["completeness_pct"])

    # Completeness distribution buckets
    dist_buckets = {"0-25%": 0, "25-50%": 0, "50-75%": 0, "75-100%": 0}
    for pid, pct in patient_completeness.items():
        if pct < 25:
            dist_buckets["0-25%"] += 1
        elif pct < 50:
            dist_buckets["25-50%"] += 1
        elif pct < 75:
            dist_buckets["50-75%"] += 1
        else:
            dist_buckets["75-100%"] += 1

    completeness_distribution = [
        {"range": k, "count": v} for k, v in dist_buckets.items()
    ]

    # Top missing fields
    top_missing = []
    for cat_name, cat_info in CATEGORIES.items():
        for field_name in cat_info["fields"]:
            present_count = len(field_presence[field_name])
            missing_count = total_patients - present_count
            if missing_count > 0:
                top_missing.append({
                    "field": field_name,
                    "category": cat_name,
                    "missing_count": missing_count,
                    "missing_pct": round(missing_count / total_patients * 100, 1),
                    "present_count": present_count,
                })

    top_missing.sort(key=lambda x: x["missing_count"], reverse=True)

    return {
        "available": True,
        "kpis": {
            "total_patients": total_patients,
            "total_fields": TOTAL_FIELDS,
            "overall_completeness_pct": overall_pct,
        },
        "per_category": per_category,
        "completeness_distribution": completeness_distribution,
        "top_missing_fields": top_missing[:20],
    }


# ─────────────────────────────────────────────────────────────────────
# 2. breakdown()
# ─────────────────────────────────────────────────────────────────────

def breakdown():
    """Detailed per-patient and per-category completeness analytics.

    Returns a dict with:
      - per_patient (list of {patient_id, name, fields_present, fields_total,
                              completeness_pct, missing_fields, missing_count})
      - category_matrix (list of {patient_id, name, ...category_pcts})
      - category_rankings (list of {category, completeness_pct, rank})
    """
    all_pids, matrix, field_presence = _compute_completeness()
    total_patients = len(all_pids)

    if total_patients == 0:
        return {
            "available": False,
            "reason": "No patient data found",
            "per_patient": [],
            "category_matrix": [],
            "category_rankings": [],
        }

    # Patient name lookup
    patient_names = {}
    for r in _rows("SELECT patient_id, name FROM patients"):
        patient_names[r["patient_id"]] = r["name"]

    # Per-patient completeness
    per_patient = []
    for pid in all_pids:
        total_present = 0
        all_missing = []
        for cat_name in CATEGORIES:
            total_present += matrix[pid][cat_name]["fields_present"]
            all_missing.extend(matrix[pid][cat_name]["missing_fields"])

        pct = round(total_present / TOTAL_FIELDS * 100, 1)
        per_patient.append({
            "patient_id": pid,
            "name": patient_names.get(pid, pid),
            "fields_present": total_present,
            "fields_total": TOTAL_FIELDS,
            "completeness_pct": pct,
            "missing_fields": all_missing,
            "missing_count": len(all_missing),
        })

    per_patient.sort(key=lambda x: x["completeness_pct"])

    # Category matrix: patient x category completeness percentages
    category_matrix = []
    for pid in all_pids:
        row = {
            "patient_id": pid,
            "name": patient_names.get(pid, pid),
        }
        for cat_name in CATEGORIES:
            m = matrix[pid][cat_name]
            cat_pct = round(m["fields_present"] / m["fields_total"] * 100, 1) if m["fields_total"] > 0 else 0.0
            row[cat_name] = cat_pct
        category_matrix.append(row)

    # Category rankings
    cat_totals = {}
    for cat_name, cat_info in CATEGORIES.items():
        fields_total = len(cat_info["fields"])
        avg_present = sum(matrix[pid][cat_name]["fields_present"] for pid in all_pids) / total_patients
        cat_pct = round(avg_present / fields_total * 100, 1) if fields_total > 0 else 0.0
        cat_totals[cat_name] = cat_pct

    sorted_cats = sorted(cat_totals.items(), key=lambda x: x[1], reverse=True)
    category_rankings = [
        {"category": cat, "completeness_pct": pct, "rank": i + 1}
        for i, (cat, pct) in enumerate(sorted_cats)
    ]

    return {
        "available": True,
        "per_patient": per_patient,
        "category_matrix": category_matrix,
        "category_rankings": category_rankings,
    }


# ─────────────────────────────────────────────────────────────────────
# 3. definitions()
# ─────────────────────────────────────────────────────────────────────

def definitions():
    """Category definitions, field mappings, and quality level descriptions.

    Returns a dict with:
      - categories (list of {name, description, fields})
      - data_quality_levels (list of {level, range, description})
      - methodology (str)
    """
    categories = []
    for cat_name, cat_info in CATEGORIES.items():
        categories.append({
            "name": cat_name,
            "description": cat_info["description"],
            "fields": list(cat_info["fields"].keys()),
        })

    data_quality_levels = [
        {
            "level": "Excellent",
            "range": "90-100%",
            "color": "#10b981",
            "description": "Data ready for regulatory submission and clinical decision support. Meets ALCOA+ standards.",
        },
        {
            "level": "Good",
            "range": "75-89%",
            "color": "#3b82f6",
            "description": "Minor gaps present. Suitable for analysis with documented caveats. Remediation recommended before submission.",
        },
        {
            "level": "Fair",
            "range": "50-74%",
            "color": "#f59e0b",
            "description": "Significant gaps detected. Requires data remediation before use in analysis or clinical decisions.",
        },
        {
            "level": "Poor",
            "range": "0-49%",
            "color": "#ef4444",
            "description": "Critical gaps. Data not reliable for clinical decisions. Immediate data collection effort required.",
        },
    ]

    methodology = (
        "Completeness is assessed by checking for the existence of records in the "
        "corresponding database tables for each patient. A field is marked 'present' "
        "if at least one non-null, non-empty record exists for that patient in the "
        "relevant table/column. For JSON-based tables (eeg_acquisition, artifact_annotations, "
        "etc.), specific keys within the fields_json column are checked. "
        "Overall completeness is the mean of per-patient completeness percentages. "
        "Categories are independently scored so that targeted remediation can be prioritized."
    )

    return {
        "categories": categories,
        "data_quality_levels": data_quality_levels,
        "methodology": methodology,
    }


if __name__ == "__main__":
    import pprint
    print("=== OVERVIEW ===")
    pprint.pprint(overview())
    print("\n=== BREAKDOWN (first 3 patients) ===")
    bd = breakdown()
    for p in bd.get("per_patient", [])[:3]:
        pprint.pprint(p)
    print("\n=== CATEGORY RANKINGS ===")
    pprint.pprint(bd.get("category_rankings", []))
