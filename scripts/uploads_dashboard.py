"""File Uploads Dashboard — uploads table analytics.

Real data: uploads (141 rows, 49 patients, 5 diseases, 7 departments,
5 file formats) — EEG recording file upload tracking with disease/department
breakdown, format distribution, and upload trend analysis.
"""

import sqlite3
import os
from collections import Counter

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def overview():
    conn = _conn()
    rows = conn.execute(
        "SELECT id, patient_id, file_name, disease, department, created_at FROM uploads"
    ).fetchall()
    conn.close()

    total = len(rows)
    patients = set()
    diseases = Counter()
    departments = Counter()
    formats = Counter()
    monthly = Counter()

    for _, pid, fname, disease, dept, created in rows:
        patients.add(pid)
        diseases[disease or 'unknown'] += 1
        departments[dept if dept else 'unassigned'] += 1
        ext = fname.rsplit('.', 1)[-1].lower() if '.' in fname else 'unknown'
        formats[ext] += 1
        month = (created or '')[:7]
        if month:
            monthly[month] += 1

    disease_pie = [{"disease": d, "count": c} for d, c in diseases.most_common()]
    dept_bar = [{"department": d, "count": c} for d, c in departments.most_common()]
    format_pie = [{"format": f".{f}", "count": c} for f, c in formats.most_common()]
    monthly_trend = [{"month": m, "uploads": c} for m, c in sorted(monthly.items())]

    # Top uploaders (patients with most uploads)
    patient_counts = Counter()
    for _, pid, *_ in rows:
        patient_counts[pid] += 1
    top_uploaders = [{"patient_id": p, "count": c} for p, c in patient_counts.most_common(10)]

    return {
        "total_uploads": total,
        "total_patients": len(patients),
        "total_diseases": len(diseases),
        "total_departments": len([d for d in departments if d != 'unassigned']),
        "total_formats": len(formats),
        "disease_distribution": disease_pie,
        "department_distribution": dept_bar,
        "format_distribution": format_pie,
        "monthly_trend": monthly_trend,
        "top_uploaders": top_uploaders,
    }


def breakdown():
    conn = _conn()
    rows = conn.execute(
        "SELECT id, patient_id, file_name, disease, department, created_at "
        "FROM uploads ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    all_uploads = []
    by_disease = {}
    by_department = {}

    for uid, pid, fname, disease, dept, created in rows:
        ext = fname.rsplit('.', 1)[-1].lower() if '.' in fname else 'unknown'
        disease = disease or 'unknown'
        dept = dept if dept else 'unassigned'

        rec = {
            "id": uid,
            "patient_id": pid,
            "file_name": fname,
            "disease": disease,
            "department": dept,
            "format": f".{ext}",
            "created_at": (created or '')[:10],
        }
        all_uploads.append(rec)

        # By disease
        if disease not in by_disease:
            by_disease[disease] = {"disease": disease, "count": 0, "patients": set(), "formats": Counter()}
        by_disease[disease]["count"] += 1
        by_disease[disease]["patients"].add(pid)
        by_disease[disease]["formats"][ext] += 1

        # By department
        if dept not in by_department:
            by_department[dept] = {"department": dept, "count": 0, "patients": set(), "diseases": Counter()}
        by_department[dept]["count"] += 1
        by_department[dept]["patients"].add(pid)
        by_department[dept]["diseases"][disease] += 1

    disease_summary = []
    for d in sorted(by_disease.values(), key=lambda x: -x["count"]):
        disease_summary.append({
            "disease": d["disease"],
            "count": d["count"],
            "patients": len(d["patients"]),
            "top_format": d["formats"].most_common(1)[0][0] if d["formats"] else "—",
        })

    dept_summary = []
    for d in sorted(by_department.values(), key=lambda x: -x["count"]):
        dept_summary.append({
            "department": d["department"],
            "count": d["count"],
            "patients": len(d["patients"]),
            "top_disease": d["diseases"].most_common(1)[0][0] if d["diseases"] else "—",
        })

    # Disease × format cross-tab
    diseases_list = sorted(by_disease.keys())
    all_fmts = sorted({r["format"] for r in all_uploads})
    cross_tab = []
    for disease in diseases_list:
        row = {"disease": disease}
        for fmt in all_fmts:
            row[fmt] = by_disease[disease]["formats"].get(fmt.lstrip('.'), 0)
        cross_tab.append(row)

    return {
        "all_uploads": all_uploads,
        "disease_summary": disease_summary,
        "department_summary": dept_summary,
        "disease_format_cross": cross_tab,
        "formats": all_fmts,
    }


def definitions():
    return {
        "fields": {
            "id": "Unique upload identifier",
            "patient_id": "Patient who uploaded the file",
            "file_name": "Original filename of the uploaded recording",
            "disease": "Primary diagnosis associated with the upload (epilepsy, parkinsons, etc.)",
            "department": "Clinical department that submitted the upload",
            "created_at": "Timestamp when the file was uploaded",
        },
        "file_formats": {
            ".edf": "European Data Format — standard EEG interchange format (16-bit)",
            ".bdf": "BioSemi Data Format — 24-bit variant of EDF for high-resolution EEG",
            ".fif": "Elekta/Neuromag FIF — MEG/EEG format used by MNE-Python",
            ".csv": "Comma-Separated Values — tabular export of processed EEG features",
            ".set": "EEGLAB SET — MATLAB-based EEG analysis format",
        },
        "departments": {
            "neurology": "General neurology — primary EEG referral department",
            "neurosurgery": "Pre-surgical epilepsy evaluation and monitoring",
            "psychiatry": "Psychiatric EEG for depression, anxiety, PNES screening",
            "sleep_lab": "Polysomnography and sleep-disorder EEG recordings",
            "geriatrics": "Alzheimer's and age-related neurological assessments",
        },
        "clinical_notes": [
            "EDF is the most widely supported format across EEG systems worldwide.",
            "BDF provides 24-bit resolution, preferred for research-grade recordings.",
            "FIF files may include MEG channels alongside EEG — verify channel selection.",
            "CSV uploads are typically feature-extracted data, not raw signals.",
            "All uploads are virus-scanned and validated for format integrity before storage.",
        ],
    }
