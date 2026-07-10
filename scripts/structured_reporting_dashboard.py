"""Structured Reporting Dashboard — clinical report template analytics from clinical.db.

Generates structured clinical report templates and tracks finding capture
for EEG, MRI, and neuropsychological (neuropsych) reports in an ILAE-aligned
epilepsy monitoring platform.

Clinical relevance:
  Structured reporting replaces free-text narratives with standardised templates
  that enforce completeness, reduce inter-reader variability, and enable
  machine-readable extraction of key findings. The ILAE Commission on
  Diagnostic Methods recommends structured reports for presurgical evaluation
  to ensure all critical data points (lesion localisation, EEG lateralisation,
  cognitive lateralisation) are captured consistently.

Data sources (all from data/clinical.db):
  - eeg_acquisition  (30 rows) — fields_json contains recording_type,
    duration_min, sampling_rate, montage, electrode_system, technician_notes,
    study_date
  - mri_findings     (40 rows) — fields_json contains mri_available, quality,
    lesion_type, lesion_label, lesion_description, lesion_location, laterality,
    hippocampal_sclerosis, hippocampal_volume_asymmetry, t2_flair_signal,
    enhancing, classification, classification_label, protocol,
    radiologist_confidence
  - neuropsych        (37 rows) — fields_json contains battery_type, phq9,
    gad7, moca, mmse, memory_index, attention_index, executive_index,
    language_index, processing_speed_index, verbal_memory_raw,
    visual_memory_raw, digit_span_forward, digit_span_backward,
    trail_a_seconds, trail_b_seconds, impairment_flag,
    lateralization_hypothesis, referral_reason, assessor
  - component_findings (doctor review / AI agreement tracking)
  - patients           (demographics cross-reference)

Author: Research Team
"""

import json
import sqlite3
import pathlib
from collections import defaultdict

DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "clinical.db"

# ---------------------------------------------------------------------------
# Template registry — defines expected fields per report type
# ---------------------------------------------------------------------------

REPORT_TEMPLATES = {
    "EEG Report": {
        "modality": "eeg_acquisition",
        "ilae_sections": [
            "Patient Demographics",
            "Recording Parameters",
            "Background Activity",
            "Epileptiform Abnormalities",
            "Technician Notes",
        ],
        "expected_fields": [
            "recording_type",
            "duration_min",
            "sampling_rate",
            "montage",
            "electrode_system",
            "technician_notes",
            "study_date",
        ],
        "mandatory_fields": [
            "recording_type",
            "duration_min",
            "sampling_rate",
            "montage",
            "electrode_system",
        ],
    },
    "MRI Report": {
        "modality": "mri_findings",
        "ilae_sections": [
            "Patient Demographics",
            "Scan Parameters",
            "Lesion Identification",
            "Hippocampal Assessment",
            "Classification",
            "Radiologist Impression",
        ],
        "expected_fields": [
            "mri_available",
            "quality",
            "lesion_type",
            "lesion_label",
            "lesion_description",
            "lesion_location",
            "laterality",
            "hippocampal_sclerosis",
            "hippocampal_volume_asymmetry",
            "t2_flair_signal",
            "enhancing",
            "classification",
            "classification_label",
            "protocol",
            "radiologist_confidence",
        ],
        "mandatory_fields": [
            "mri_available",
            "quality",
            "lesion_type",
            "lesion_location",
            "laterality",
            "classification",
            "protocol",
        ],
    },
    "Neuropsych Report": {
        "modality": "neuropsych",
        "ilae_sections": [
            "Patient Demographics",
            "Battery Selection",
            "Mood Screening",
            "Cognitive Indices",
            "Memory Assessment",
            "Attention / Executive",
            "Lateralisation Hypothesis",
        ],
        "expected_fields": [
            "battery_type",
            "phq9",
            "gad7",
            "moca",
            "mmse",
            "memory_index",
            "attention_index",
            "executive_index",
            "language_index",
            "processing_speed_index",
            "verbal_memory_raw",
            "visual_memory_raw",
            "digit_span_forward",
            "digit_span_backward",
            "trail_a_seconds",
            "trail_b_seconds",
            "impairment_flag",
            "lateralization_hypothesis",
            "referral_reason",
            "assessor",
        ],
        "mandatory_fields": [
            "battery_type",
            "moca",
            "mmse",
            "memory_index",
            "attention_index",
            "impairment_flag",
            "assessor",
        ],
    },
    "Comprehensive Epilepsy Report": {
        "modality": "comprehensive",
        "ilae_sections": [
            "Patient Demographics",
            "EEG Summary",
            "MRI Summary",
            "Neuropsych Summary",
            "Seizure Classification",
            "Presurgical Concordance",
            "Management Recommendation",
        ],
        "expected_fields": [],  # composite of above three
        "mandatory_fields": [],
    },
}

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _conn():
    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    return con


def _safe(cur, sql, params=(), default=0):
    try:
        cur.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else default
    except Exception:
        return default


def _safe_rows(cur, sql, params=()):
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    except Exception:
        return []


def _parse_json(raw):
    """Safely parse a JSON string, returning empty dict on failure."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _field_filled(value):
    """Return True if the value is present and non-empty."""
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


def _completeness_for_rows(rows, expected_fields):
    """Compute per-row completeness against expected_fields.

    Returns list of dicts with patient_id, filled, total, pct.
    """
    results = []
    for r in rows:
        data = _parse_json(r["fields_json"])
        total = len(expected_fields)
        filled = sum(1 for f in expected_fields if _field_filled(data.get(f)))
        pct = round(filled / total * 100, 1) if total else 0.0
        results.append({
            "id": r["id"],
            "patient_id": r["patient_id"],
            "filled_fields": filled,
            "total_fields": total,
            "completeness_pct": pct,
            "created_at": r["created_at"],
        })
    return results


def _mandatory_capture_rate(rows, mandatory_fields):
    """What fraction of rows have ALL mandatory fields filled."""
    if not rows or not mandatory_fields:
        return 0.0
    fully_captured = 0
    for r in rows:
        data = _parse_json(r["fields_json"])
        if all(_field_filled(data.get(f)) for f in mandatory_fields):
            fully_captured += 1
    return round(fully_captured / len(rows) * 100, 1)


def _monthly_trends(rows):
    """Group rows by YYYY-MM from created_at and count."""
    months = defaultdict(int)
    for r in rows:
        ca = r["created_at"] or ""
        month_key = ca[:7] if len(ca) >= 7 else "unknown"
        months[month_key] += 1
    return dict(sorted(months.items()))


def _field_coverage_heatmap(rows, expected_fields):
    """For each expected field, count how many rows have it filled."""
    counts = {f: 0 for f in expected_fields}
    total = len(rows)
    for r in rows:
        data = _parse_json(r["fields_json"])
        for f in expected_fields:
            if _field_filled(data.get(f)):
                counts[f] += 1
    return {
        f: {
            "filled_count": c,
            "total_rows": total,
            "fill_rate_pct": round(c / total * 100, 1) if total else 0.0,
        }
        for f, c in counts.items()
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def overview():
    """Dashboard overview — template registry, report generation stats,
    completeness metrics, structured finding capture rates, monthly trends,
    and AI-assisted finding capture statistics.

    Returns a dict suitable for JSON serialisation.
    """
    con = _conn()
    cur = con.cursor()

    # ----- row counts per modality -----
    eeg_count = _safe(cur, "SELECT COUNT(*) FROM eeg_acquisition")
    mri_count = _safe(cur, "SELECT COUNT(*) FROM mri_findings")
    neuro_count = _safe(cur, "SELECT COUNT(*) FROM neuropsych")
    total_reports = eeg_count + mri_count + neuro_count

    # ----- per-patient coverage (how many modalities each patient has) -----
    eeg_patients = set()
    for r in _safe_rows(cur, "SELECT DISTINCT patient_id FROM eeg_acquisition"):
        eeg_patients.add(r["patient_id"])
    mri_patients = set()
    for r in _safe_rows(cur, "SELECT DISTINCT patient_id FROM mri_findings"):
        mri_patients.add(r["patient_id"])
    neuro_patients = set()
    for r in _safe_rows(cur, "SELECT DISTINCT patient_id FROM neuropsych"):
        neuro_patients.add(r["patient_id"])
    all_patients = eeg_patients | mri_patients | neuro_patients
    total_patients = _safe(cur, "SELECT COUNT(*) FROM patients")

    coverage = {
        "patients_with_any_report": len(all_patients),
        "total_patients": total_patients,
        "coverage_pct": round(len(all_patients) / total_patients * 100, 1) if total_patients else 0.0,
        "eeg_patients": len(eeg_patients),
        "mri_patients": len(mri_patients),
        "neuropsych_patients": len(neuro_patients),
    }

    # ----- template completeness per modality -----
    eeg_rows = _safe_rows(cur, "SELECT id, patient_id, fields_json, created_at FROM eeg_acquisition")
    mri_rows = _safe_rows(cur, "SELECT id, patient_id, fields_json, created_at FROM mri_findings")
    neuro_rows = _safe_rows(cur, "SELECT id, patient_id, fields_json, created_at FROM neuropsych")

    eeg_template = REPORT_TEMPLATES["EEG Report"]
    mri_template = REPORT_TEMPLATES["MRI Report"]
    neuro_template = REPORT_TEMPLATES["Neuropsych Report"]

    eeg_completeness = _completeness_for_rows(eeg_rows, eeg_template["expected_fields"])
    mri_completeness = _completeness_for_rows(mri_rows, mri_template["expected_fields"])
    neuro_completeness = _completeness_for_rows(neuro_rows, neuro_template["expected_fields"])

    def _avg_completeness(comps):
        if not comps:
            return 0.0
        return round(sum(c["completeness_pct"] for c in comps) / len(comps), 1)

    template_completeness = {
        "EEG Report": {
            "row_count": len(eeg_completeness),
            "avg_completeness_pct": _avg_completeness(eeg_completeness),
            "mandatory_capture_rate_pct": _mandatory_capture_rate(
                eeg_rows, eeg_template["mandatory_fields"]
            ),
        },
        "MRI Report": {
            "row_count": len(mri_completeness),
            "avg_completeness_pct": _avg_completeness(mri_completeness),
            "mandatory_capture_rate_pct": _mandatory_capture_rate(
                mri_rows, mri_template["mandatory_fields"]
            ),
        },
        "Neuropsych Report": {
            "row_count": len(neuro_completeness),
            "avg_completeness_pct": _avg_completeness(neuro_completeness),
            "mandatory_capture_rate_pct": _mandatory_capture_rate(
                neuro_rows, neuro_template["mandatory_fields"]
            ),
        },
    }

    # ----- monthly report generation trends -----
    monthly_eeg = _monthly_trends(eeg_rows)
    monthly_mri = _monthly_trends(mri_rows)
    monthly_neuro = _monthly_trends(neuro_rows)

    # Merge all months into a combined view
    all_months = sorted(set(list(monthly_eeg) + list(monthly_mri) + list(monthly_neuro)))
    monthly_combined = []
    for m in all_months:
        monthly_combined.append({
            "month": m,
            "eeg": monthly_eeg.get(m, 0),
            "mri": monthly_mri.get(m, 0),
            "neuropsych": monthly_neuro.get(m, 0),
            "total": monthly_eeg.get(m, 0) + monthly_mri.get(m, 0) + monthly_neuro.get(m, 0),
        })

    # ----- AI-assisted finding capture stats -----
    ai_rows = _safe_rows(
        cur,
        "SELECT patient_id, component, agree_with_ai, doctor_finding, doctor "
        "FROM component_findings",
    )
    ai_stats = {
        "total_findings": len(ai_rows),
        "agree_count": sum(1 for r in ai_rows if (r["agree_with_ai"] or "").lower() == "agree"),
        "disagree_count": sum(1 for r in ai_rows if (r["agree_with_ai"] or "").lower() == "disagree"),
        "agreement_rate_pct": 0.0,
    }
    if ai_stats["total_findings"]:
        ai_stats["agreement_rate_pct"] = round(
            ai_stats["agree_count"] / ai_stats["total_findings"] * 100, 1
        )

    con.close()

    return {
        "report_template_registry": {
            name: {
                "fields_count": len(t["expected_fields"]),
                "mandatory_fields_count": len(t["mandatory_fields"]),
                "ilae_sections": t["ilae_sections"],
            }
            for name, t in REPORT_TEMPLATES.items()
        },
        "report_generation_stats": {
            "total_reports_generated": total_reports,
            "eeg_reports": eeg_count,
            "mri_reports": mri_count,
            "neuropsych_reports": neuro_count,
        },
        "patient_coverage": coverage,
        "template_completeness": template_completeness,
        "monthly_trends": monthly_combined,
        "ai_assisted_finding_capture": ai_stats,
    }


def breakdown():
    """Detailed per-patient and per-field breakdown.

    Returns:
        dict with per_patient_inventory, field_coverage_heatmap,
        report_quality_scores, turnaround_analysis, cross_modality_concordance.
    """
    con = _conn()
    cur = con.cursor()

    # ----- per-patient report inventory -----
    eeg_rows = _safe_rows(cur, "SELECT id, patient_id, fields_json, created_at FROM eeg_acquisition")
    mri_rows = _safe_rows(cur, "SELECT id, patient_id, fields_json, created_at FROM mri_findings")
    neuro_rows = _safe_rows(cur, "SELECT id, patient_id, fields_json, created_at FROM neuropsych")

    # Build patient -> modalities map
    patient_modalities = defaultdict(lambda: {"eeg": 0, "mri": 0, "neuropsych": 0})
    for r in eeg_rows:
        patient_modalities[r["patient_id"]]["eeg"] += 1
    for r in mri_rows:
        patient_modalities[r["patient_id"]]["mri"] += 1
    for r in neuro_rows:
        patient_modalities[r["patient_id"]]["neuropsych"] += 1

    per_patient_inventory = []
    for pid in sorted(patient_modalities):
        m = patient_modalities[pid]
        modalities_present = [k for k, v in m.items() if v > 0]
        per_patient_inventory.append({
            "patient_id": pid,
            "eeg_count": m["eeg"],
            "mri_count": m["mri"],
            "neuropsych_count": m["neuropsych"],
            "total_reports": m["eeg"] + m["mri"] + m["neuropsych"],
            "modalities_present": modalities_present,
            "modality_count": len(modalities_present),
        })

    # ----- field coverage heatmap per template -----
    eeg_template = REPORT_TEMPLATES["EEG Report"]
    mri_template = REPORT_TEMPLATES["MRI Report"]
    neuro_template = REPORT_TEMPLATES["Neuropsych Report"]

    field_coverage = {
        "EEG Report": _field_coverage_heatmap(eeg_rows, eeg_template["expected_fields"]),
        "MRI Report": _field_coverage_heatmap(mri_rows, mri_template["expected_fields"]),
        "Neuropsych Report": _field_coverage_heatmap(neuro_rows, neuro_template["expected_fields"]),
    }

    # ----- report quality scores per patient -----
    eeg_comp = _completeness_for_rows(eeg_rows, eeg_template["expected_fields"])
    mri_comp = _completeness_for_rows(mri_rows, mri_template["expected_fields"])
    neuro_comp = _completeness_for_rows(neuro_rows, neuro_template["expected_fields"])

    patient_quality = defaultdict(list)
    for c in eeg_comp:
        patient_quality[c["patient_id"]].append(c["completeness_pct"])
    for c in mri_comp:
        patient_quality[c["patient_id"]].append(c["completeness_pct"])
    for c in neuro_comp:
        patient_quality[c["patient_id"]].append(c["completeness_pct"])

    quality_scores = []
    for pid in sorted(patient_quality):
        scores = patient_quality[pid]
        quality_scores.append({
            "patient_id": pid,
            "report_count": len(scores),
            "avg_quality_pct": round(sum(scores) / len(scores), 1),
            "min_quality_pct": round(min(scores), 1),
            "max_quality_pct": round(max(scores), 1),
        })

    # ----- turnaround time analysis -----
    # Use created_at timestamps to compute inter-report intervals per patient
    patient_timestamps = defaultdict(list)
    for r in eeg_rows + mri_rows + neuro_rows:
        ca = r["created_at"] or ""
        if len(ca) >= 10:
            patient_timestamps[r["patient_id"]].append(ca)

    turnaround = []
    for pid, ts_list in sorted(patient_timestamps.items()):
        ts_sorted = sorted(ts_list)
        turnaround.append({
            "patient_id": pid,
            "first_report": ts_sorted[0] if ts_sorted else None,
            "last_report": ts_sorted[-1] if ts_sorted else None,
            "total_reports": len(ts_sorted),
        })

    # ----- cross-modality concordance -----
    eeg_pids = {r["patient_id"] for r in eeg_rows}
    mri_pids = {r["patient_id"] for r in mri_rows}
    neuro_pids = {r["patient_id"] for r in neuro_rows}

    eeg_and_mri = sorted(eeg_pids & mri_pids)
    eeg_and_neuro = sorted(eeg_pids & neuro_pids)
    mri_and_neuro = sorted(mri_pids & neuro_pids)
    all_three = sorted(eeg_pids & mri_pids & neuro_pids)

    cross_modality = {
        "eeg_and_mri": {"count": len(eeg_and_mri), "patients": eeg_and_mri},
        "eeg_and_neuropsych": {"count": len(eeg_and_neuro), "patients": eeg_and_neuro},
        "mri_and_neuropsych": {"count": len(mri_and_neuro), "patients": mri_and_neuro},
        "all_three_modalities": {"count": len(all_three), "patients": all_three},
    }

    con.close()

    return {
        "per_patient_inventory": per_patient_inventory,
        "field_coverage_heatmap": field_coverage,
        "report_quality_scores": quality_scores,
        "turnaround_analysis": turnaround,
        "cross_modality_concordance": cross_modality,
    }


def definitions():
    """Clinical term definitions relevant to structured reporting.

    Returns a dict mapping term name to its definition string.
    """
    return {
        "Structured Reporting": (
            "A reporting methodology that replaces free-text narratives with "
            "standardised templates containing predefined fields. Ensures "
            "completeness, reduces variability between readers, and enables "
            "machine-readable data extraction."
        ),
        "ILAE Report Template": (
            "Report templates aligned with International League Against Epilepsy "
            "(ILAE) guidelines for presurgical evaluation documentation, covering "
            "EEG, MRI, neuropsychology, and comprehensive epilepsy assessments."
        ),
        "EEG Montage": (
            "The arrangement of electrode pairs used to display EEG signals. "
            "Common montages include referential (each electrode vs a common "
            "reference), bipolar (sequential electrode pairs), and average "
            "reference. The 10-20 system defines standard electrode placement."
        ),
        "MRI Lesion Classification": (
            "Categorisation of brain lesions identified on MRI, including "
            "hippocampal sclerosis (HS), focal cortical dysplasia (FCD), "
            "cavernoma (CAV), tumour, vascular malformation, and non-lesional. "
            "The ILAE classification distinguishes lesional from non-lesional "
            "epilepsy for surgical candidacy."
        ),
        "Neuropsychological Battery": (
            "A standardised set of cognitive tests measuring memory, attention, "
            "executive function, language, and processing speed. Common "
            "instruments include MoCA, MMSE, digit span, trail-making (A and B), "
            "and domain-specific indices. Used to lateralise cognitive deficits "
            "and predict post-surgical outcomes."
        ),
        "Report Completeness Score": (
            "The percentage of expected template fields that are filled with "
            "non-null, non-empty values in a given report. A score of 100% "
            "indicates all expected data points were captured."
        ),
        "Finding Capture": (
            "The process of recording discrete clinical findings in structured "
            "fields rather than embedding them in free text. Higher capture "
            "rates improve data quality for analytics, research, and clinical "
            "decision support."
        ),
        "AI-Assisted Reporting": (
            "Use of artificial intelligence to pre-populate report fields, "
            "suggest findings, or flag discrepancies. Clinician agreement or "
            "disagreement with AI suggestions is tracked to measure AI "
            "reliability and identify areas needing model improvement."
        ),
        "Cross-Modal Concordance": (
            "Agreement between findings across different diagnostic modalities "
            "(EEG, MRI, neuropsychology). High concordance (e.g., EEG "
            "lateralisation matching MRI lesion side) strengthens the case "
            "for surgical intervention in epilepsy."
        ),
        "Template Compliance": (
            "The rate at which all mandatory fields in a report template are "
            "completed. Non-compliant reports may lack critical data points "
            "required for clinical decision-making or quality audits."
        ),
        "Mandatory Fields": (
            "Template fields that must be filled for a report to be considered "
            "complete. These represent the minimum dataset required by clinical "
            "guidelines (e.g., recording type and montage for EEG, lesion "
            "location and classification for MRI)."
        ),
        "Field Coverage Heatmap": (
            "A visual or tabular representation of which template fields are "
            "commonly filled versus frequently missing across all reports. "
            "Highlights systematic documentation gaps that may require "
            "workflow changes or additional training."
        ),
    }


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    print("=== OVERVIEW ===")
    pprint.pprint(overview())
    print("\n=== BREAKDOWN (summary) ===")
    bd = breakdown()
    print(f"  Patients with reports: {len(bd['per_patient_inventory'])}")
    print(f"  Quality scores: {len(bd['report_quality_scores'])} patients")
    cm = bd["cross_modality_concordance"]
    print(f"  EEG+MRI concordance: {cm['eeg_and_mri']['count']} patients")
    print(f"  All 3 modalities: {cm['all_three_modalities']['count']} patients")
    print("\n=== DEFINITIONS ===")
    for term in definitions():
        print(f"  - {term}")
