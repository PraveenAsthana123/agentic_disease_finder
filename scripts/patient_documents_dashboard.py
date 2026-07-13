"""
Neuro AI Ecosystem — Patient Document Management Dashboard
============================================================
Tracks clinical, administrative, and educational documents
shared with epilepsy patients — upload trends, sharing analytics,
document type distribution, and per-patient document inventory.

Document Types:
  Seizure Action Plan, MRI Report, EEG Report, Lab Results,
  Referral Letter, Discharge Summary, Consent Form,
  Medication List, Insurance Auth, Education Material

Categories:
  clinical        — Clinical reports and medical records
  administrative  — Insurance, consent, referral documents
  educational     — Patient education materials

All records from real patients in clinical.db patient_documents table.

Author: Research Team
"""

import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def overview(patient_id: Optional[str] = None):
    """Document management overview — KPIs, type distribution, category breakdown, upload timeline."""
    conn = _conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    where = ""
    params = []
    if patient_id:
        where = "WHERE patient_id = ?"
        params = [patient_id]

    # KPIs
    cur.execute(f"SELECT COUNT(*) FROM patient_documents {where}", params)
    total = cur.fetchone()[0]

    cur.execute(f"SELECT COUNT(DISTINCT patient_id) FROM patient_documents {where}", params)
    total_patients = cur.fetchone()[0]

    cur.execute(f"SELECT SUM(shared_with_patient) FROM patient_documents {where}", params)
    shared = cur.fetchone()[0] or 0

    cur.execute(f"SELECT SUM(downloaded_by_patient) FROM patient_documents {where}", params)
    downloaded = cur.fetchone()[0] or 0

    cur.execute(f"SELECT ROUND(AVG(file_size_kb), 0) FROM patient_documents {where}", params)
    avg_size = cur.fetchone()[0] or 0

    cur.execute(f"SELECT SUM(file_size_kb) FROM patient_documents {where}", params)
    total_size_kb = cur.fetchone()[0] or 0

    cur.execute(f"SELECT COUNT(DISTINCT document_type) FROM patient_documents {where}", params)
    type_count = cur.fetchone()[0]

    share_rate = round(shared / total * 100, 1) if total else 0
    download_rate = round(downloaded / total * 100, 1) if total else 0

    kpis = [
        {"label": "Total Documents", "value": total},
        {"label": "Patients", "value": total_patients},
        {"label": "Document Types", "value": type_count},
        {"label": "Shared with Patient", "value": shared, "sub": f"{share_rate}% share rate"},
        {"label": "Downloaded", "value": downloaded, "sub": f"{download_rate}% download rate", "color": "#3b82f6"},
        {"label": "Avg File Size", "value": f"{int(avg_size)} KB"},
        {"label": "Total Storage", "value": f"{round(total_size_kb / 1024, 1)} MB"},
    ]

    # Type distribution (for pie chart)
    cur.execute(
        f"SELECT document_type, COUNT(*) as cnt FROM patient_documents {where} GROUP BY document_type ORDER BY cnt DESC",
        params,
    )
    type_distribution = [{"name": r["document_type"], "value": r["cnt"]} for r in cur.fetchall()]

    # Category distribution (for pie chart)
    cur.execute(
        f"SELECT category, COUNT(*) as cnt FROM patient_documents {where} GROUP BY category ORDER BY cnt DESC",
        params,
    )
    category_distribution = [{"name": r["category"], "value": r["cnt"]} for r in cur.fetchall()]

    # Monthly upload trend
    cur.execute(
        f"""SELECT substr(upload_date, 1, 7) as month, COUNT(*) as cnt
            FROM patient_documents {where}
            GROUP BY month ORDER BY month""",
        params,
    )
    monthly_trend = [{"month": r["month"], "uploads": r["cnt"]} for r in cur.fetchall()]

    # Sharing status breakdown
    cur.execute(
        f"""SELECT
              SUM(CASE WHEN shared_with_patient = 1 AND downloaded_by_patient = 1 THEN 1 ELSE 0 END) as shared_downloaded,
              SUM(CASE WHEN shared_with_patient = 1 AND downloaded_by_patient = 0 THEN 1 ELSE 0 END) as shared_not_downloaded,
              SUM(CASE WHEN shared_with_patient = 0 THEN 1 ELSE 0 END) as not_shared
            FROM patient_documents {where}""",
        params,
    )
    r = cur.fetchone()
    sharing_status = [
        {"name": "Shared & Downloaded", "value": r["shared_downloaded"] or 0, "color": "#22c55e"},
        {"name": "Shared, Not Downloaded", "value": r["shared_not_downloaded"] or 0, "color": "#eab308"},
        {"name": "Not Shared", "value": r["not_shared"] or 0, "color": "#94a3b8"},
    ]

    conn.close()
    return {
        "kpis": kpis,
        "type_distribution": type_distribution,
        "category_distribution": category_distribution,
        "monthly_trend": monthly_trend,
        "sharing_status": sharing_status,
    }


def breakdown(patient_id: Optional[str] = None):
    """Per-patient breakdown, document detail table, type-by-category matrix."""
    conn = _conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    where = ""
    params = []
    if patient_id:
        where = "WHERE patient_id = ?"
        params = [patient_id]

    # Per-patient summary
    cur.execute(
        f"""SELECT patient_id,
                   COUNT(*) as doc_count,
                   SUM(file_size_kb) as total_kb,
                   SUM(shared_with_patient) as shared,
                   SUM(downloaded_by_patient) as downloaded,
                   MIN(upload_date) as first_upload,
                   MAX(upload_date) as last_upload,
                   COUNT(DISTINCT document_type) as type_count
            FROM patient_documents {where}
            GROUP BY patient_id
            ORDER BY patient_id""",
        params,
    )
    per_patient = []
    for r in cur.fetchall():
        per_patient.append({
            "patient_id": r["patient_id"],
            "doc_count": r["doc_count"],
            "total_size_mb": round(r["total_kb"] / 1024, 1),
            "shared": r["shared"],
            "downloaded": r["downloaded"],
            "share_rate": round(r["shared"] / r["doc_count"] * 100, 1) if r["doc_count"] else 0,
            "first_upload": r["first_upload"],
            "last_upload": r["last_upload"],
            "type_count": r["type_count"],
        })

    # Recent documents table
    cur.execute(
        f"""SELECT patient_id, document_type, document_name, upload_date,
                   file_size_kb, shared_with_patient, downloaded_by_patient, category
            FROM patient_documents {where}
            ORDER BY upload_date DESC
            LIMIT 50""",
        params,
    )
    recent_documents = []
    for r in cur.fetchall():
        recent_documents.append({
            "patient_id": r["patient_id"],
            "document_type": r["document_type"],
            "document_name": r["document_name"],
            "upload_date": r["upload_date"],
            "file_size_kb": r["file_size_kb"],
            "shared": bool(r["shared_with_patient"]),
            "downloaded": bool(r["downloaded_by_patient"]),
            "category": r["category"],
        })

    # Type × category matrix
    cur.execute(
        f"""SELECT document_type, category, COUNT(*) as cnt
            FROM patient_documents {where}
            GROUP BY document_type, category
            ORDER BY document_type""",
        params,
    )
    type_category = []
    for r in cur.fetchall():
        type_category.append({
            "document_type": r["document_type"],
            "category": r["category"],
            "count": r["cnt"],
        })

    # File size distribution by type
    cur.execute(
        f"""SELECT document_type,
                   ROUND(AVG(file_size_kb), 0) as avg_kb,
                   MIN(file_size_kb) as min_kb,
                   MAX(file_size_kb) as max_kb,
                   SUM(file_size_kb) as total_kb,
                   COUNT(*) as cnt
            FROM patient_documents {where}
            GROUP BY document_type
            ORDER BY total_kb DESC""",
        params,
    )
    size_by_type = []
    for r in cur.fetchall():
        size_by_type.append({
            "document_type": r["document_type"],
            "avg_kb": int(r["avg_kb"] or 0),
            "min_kb": r["min_kb"],
            "max_kb": r["max_kb"],
            "total_mb": round(r["total_kb"] / 1024, 1),
            "count": r["cnt"],
        })

    conn.close()
    return {
        "per_patient": per_patient,
        "recent_documents": recent_documents,
        "type_category": type_category,
        "size_by_type": size_by_type,
    }


def definitions():
    """Document type definitions, categories, sharing workflow, glossary."""
    return {
        "document_types": [
            {"type": "Seizure Action Plan", "description": "Emergency protocol for seizure management — distributed to caregivers, schools, workplaces"},
            {"type": "MRI Report", "description": "Brain MRI findings — lesion identification, structural abnormalities, surgical candidacy"},
            {"type": "EEG Report", "description": "Electroencephalogram interpretation — seizure localization, background activity, epileptiform discharges"},
            {"type": "Lab Results", "description": "Blood work, AED drug levels, metabolic panels — medication monitoring"},
            {"type": "Referral Letter", "description": "Specialist referral documentation — reason for referral, clinical summary, urgency"},
            {"type": "Discharge Summary", "description": "Hospital discharge documentation — admission diagnosis, treatment, follow-up plan"},
            {"type": "Consent Form", "description": "Signed informed consent — treatment, research participation, data sharing"},
            {"type": "Medication List", "description": "Current medication inventory — AEDs, doses, schedules, interactions"},
            {"type": "Insurance Auth", "description": "Insurance pre-authorization — procedure approval, coverage confirmation"},
            {"type": "Education Material", "description": "Patient education — seizure first aid, medication guides, lifestyle guidance"},
        ],
        "categories": [
            {"name": "clinical", "description": "Clinical reports and medical records (EEG, MRI, labs, discharge summaries)", "color": "#3b82f6"},
            {"name": "administrative", "description": "Insurance, consent, referral, and administrative documents", "color": "#f59e0b"},
            {"name": "educational", "description": "Patient education materials and self-management resources", "color": "#22c55e"},
        ],
        "sharing_workflow": {
            "description": "Documents are uploaded by clinicians, optionally shared with patients via the patient portal, and tracked for download/acknowledgment.",
            "statuses": [
                {"status": "Not Shared", "description": "Document uploaded but not yet shared with patient"},
                {"status": "Shared, Not Downloaded", "description": "Document shared via portal but patient has not accessed it"},
                {"status": "Shared & Downloaded", "description": "Document shared and confirmed accessed by patient"},
            ],
        },
        "glossary": [
            {"term": "AED", "definition": "Anti-Epileptic Drug — first-line pharmacological treatment for seizure control"},
            {"term": "Share Rate", "definition": "Percentage of documents shared with patients via the portal"},
            {"term": "Download Rate", "definition": "Percentage of documents downloaded/accessed by patients"},
            {"term": "Document Type", "definition": "Classification of the document by its clinical or administrative purpose"},
            {"term": "Category", "definition": "Broad grouping: clinical, administrative, or educational"},
        ],
        "clinical_note": "Comprehensive document management in epilepsy care ensures patients and caregivers have timely access to seizure action plans, medication lists, and test results — critical for emergency situations and care coordination across providers.",
    }
