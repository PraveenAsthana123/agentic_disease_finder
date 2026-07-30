"""Patient Onboarding Dashboard — epilepsy clinical portal.

Goal: Reduce patient intake from 2-3 hours → 8-10 minutes via required-first
capture, multi-format document extraction, and field deferral.

Process:
  Step 1 — Demographics  : patient_demographics (core identity fields)
  Step 2 — Clinical Core : emergency_contacts + medications + comorbidities
  Step 3 — Documents     : patient_documents (upload + extraction)

Output per patient: ~80 required intake fields captured; ~1170 deferred.

Exports:
  overview()     — KPIs + completion distribution + monthly trend
  breakdown()    — Per-patient detail across 5 onboarding sections
  definitions()  — Static spec: sections, required fields, deferred fields,
                   process steps, glossary
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _conn():
    return sqlite3.connect(DB_PATH)


def _fetch(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _scalar(conn, sql, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    row = cur.fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Field completeness per patient
# ---------------------------------------------------------------------------

# Required demographics fields (20 core intake fields out of 32 total columns)
_DEMO_REQUIRED = [
    "full_name", "date_of_birth", "age", "sex", "blood_type",
    "primary_language", "insurance_type", "emergency_contact_name",
    "emergency_contact_phone", "address_city", "address_state",
    "address_zip", "referral_source", "epilepsy_type",
    "epilepsy_onset_age", "years_with_epilepsy", "primary_neurologist",
    "enrollment_date", "height_cm", "weight_kg",
]
_DEMO_DEFERRED = [
    "gender_identity", "bmi", "ethnicity", "race",
    "interpreter_needed", "education_level", "occupation",
    "employment_status", "marital_status",
]

# Total intake field counts (spec: ~80 required, ~1170 deferred)
_TOTAL_REQUIRED_FIELDS = 80
_TOTAL_DEFERRED_FIELDS = 1170


def _count_demo_fields(row: dict) -> int:
    """Count non-null required demographics fields present for a patient."""
    return sum(1 for f in _DEMO_REQUIRED if row.get(f) not in (None, "", 0))


def _section_completeness(conn) -> dict:
    """
    Returns dict keyed by patient_id with completeness booleans/counts
    for each of the 5 onboarding sections.
    """
    patients = _fetch(conn, "SELECT patient_id FROM patients")
    pids = [r["patient_id"] for r in patients]

    # --- Demographics ---
    demo_rows = _fetch(
        conn,
        "SELECT patient_id, " + ", ".join(_DEMO_REQUIRED) + " FROM patient_demographics",
    )
    demo_by_pid = {r["patient_id"]: r for r in demo_rows}

    # --- Emergency contacts ---
    ec_pids = set(
        r["patient_id"]
        for r in _fetch(conn, "SELECT DISTINCT patient_id FROM emergency_contacts")
    )

    # --- Medications ---
    med_pids = set(
        r["patient_id"]
        for r in _fetch(conn, "SELECT DISTINCT patient_id FROM medications")
    )
    med_count_rows = _fetch(
        conn,
        "SELECT patient_id, COUNT(*) as cnt FROM medications GROUP BY patient_id",
    )
    med_count_by_pid = {r["patient_id"]: r["cnt"] for r in med_count_rows}

    # --- Documents ---
    doc_rows = _fetch(
        conn,
        "SELECT patient_id, COUNT(*) as cnt FROM patient_documents GROUP BY patient_id",
    )
    doc_count_by_pid = {r["patient_id"]: r["cnt"] for r in doc_rows}
    doc_pids = set(doc_count_by_pid.keys())

    # --- Comorbidities ---
    comorbidity_rows = _fetch(
        conn,
        "SELECT patient_id, fields_json FROM comorbidities",
    )
    comorbidity_pids = set()
    for r in comorbidity_rows:
        try:
            f = json.loads(r["fields_json"] or "{}")
            if f.get("screened"):
                comorbidity_pids.add(r["patient_id"])
        except Exception:
            pass

    result = {}
    for pid in pids:
        demo_row = demo_by_pid.get(pid, {})
        demo_field_count = _count_demo_fields(demo_row)
        has_demographics = pid in demo_by_pid
        has_emergency_contact = pid in ec_pids
        has_medications = pid in med_pids
        has_documents = pid in doc_pids
        has_comorbidities = pid in comorbidity_pids

        # Completion percentage: weight each section
        # Demographics=35, Emergency=20, Medications=20, Documents=15, Comorbidities=10
        pct = 0
        if has_demographics:
            # Partial credit based on required field fill rate
            pct += round(35 * (demo_field_count / max(len(_DEMO_REQUIRED), 1)), 1)
        if has_emergency_contact:
            pct += 20
        if has_medications:
            pct += 20
        if has_documents:
            doc_cnt = doc_count_by_pid.get(pid, 0)
            pct += min(15, round(15 * min(doc_cnt / 5, 1), 1))  # full at 5+ docs
        if has_comorbidities:
            pct += 10

        result[pid] = {
            "has_demographics": has_demographics,
            "has_emergency_contact": has_emergency_contact,
            "has_medications": has_medications,
            "has_documents": has_documents,
            "has_comorbidities": has_comorbidities,
            "demo_fields_completed": demo_field_count,
            "demo_fields_required": len(_DEMO_REQUIRED),
            "medication_count": med_count_by_pid.get(pid, 0),
            "document_count": doc_count_by_pid.get(pid, 0),
            "completion_pct": round(min(pct, 100), 1),
        }
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def overview():
    """
    Return top-level KPIs + completion distribution + onboarding by month.

    Returns:
        {
            kpis: {
                total_patients, onboarded_count, onboarding_completion_pct,
                patients_with_demographics, patients_with_emergency_contacts,
                patients_with_medications, patients_with_documents_uploaded,
                avg_documents_per_patient, avg_fields_completed_per_patient
            },
            completion_distribution: {
                "0-24": int, "25-49": int, "50-74": int, "75-99": int, "100": int
            },
            onboarding_by_month: [
                { month: "YYYY-MM", new_patients: int, demographics_enrolled: int }
            ]
        }
    """
    conn = _conn()
    try:
        completeness = _section_completeness(conn)
        total = len(completeness)

        if total == 0:
            return {"available": False, "reason": "No patients found"}

        # A patient is "onboarded" when they have all 4 core sections
        def _is_onboarded(s):
            return (
                s["has_demographics"]
                and s["has_emergency_contact"]
                and s["has_medications"]
                and s["has_documents"]
            )

        onboarded_count = sum(1 for s in completeness.values() if _is_onboarded(s))
        onboarding_completion_pct = round(100 * onboarded_count / total, 1)

        patients_with_demographics = sum(
            1 for s in completeness.values() if s["has_demographics"]
        )
        patients_with_emergency_contacts = sum(
            1 for s in completeness.values() if s["has_emergency_contact"]
        )
        patients_with_medications = sum(
            1 for s in completeness.values() if s["has_medications"]
        )
        patients_with_documents = sum(
            1 for s in completeness.values() if s["has_documents"]
        )

        avg_docs = _scalar(
            conn,
            "SELECT ROUND(AVG(cnt), 2) FROM "
            "(SELECT COUNT(*) as cnt FROM patient_documents GROUP BY patient_id)",
        ) or 0.0

        avg_fields = round(
            sum(s["demo_fields_completed"] for s in completeness.values()) / total, 1
        )

        # Completion distribution buckets
        dist = {"0-24": 0, "25-49": 0, "50-74": 0, "75-99": 0, "100": 0}
        for s in completeness.values():
            p = s["completion_pct"]
            if p < 25:
                dist["0-24"] += 1
            elif p < 50:
                dist["25-49"] += 1
            elif p < 75:
                dist["50-74"] += 1
            elif p < 100:
                dist["75-99"] += 1
            else:
                dist["100"] += 1

        # Onboarding by month (from patients.created_at + demographics.enrollment_date)
        patients_by_month = _fetch(
            conn,
            "SELECT strftime('%Y-%m', created_at) as month, COUNT(*) as new_patients "
            "FROM patients "
            "WHERE created_at IS NOT NULL "
            "GROUP BY month ORDER BY month",
        )
        demo_by_month = _fetch(
            conn,
            "SELECT strftime('%Y-%m', enrollment_date) as month, COUNT(*) as enrolled "
            "FROM patient_demographics "
            "WHERE enrollment_date IS NOT NULL "
            "GROUP BY month ORDER BY month",
        )
        demo_month_map = {r["month"]: r["enrolled"] for r in demo_by_month}
        onboarding_by_month = [
            {
                "month": r["month"],
                "new_patients": r["new_patients"],
                "demographics_enrolled": demo_month_map.get(r["month"], 0),
            }
            for r in patients_by_month
        ]

        return {
            "kpis": {
                "total_patients": total,
                "onboarded_count": onboarded_count,
                "onboarding_completion_pct": onboarding_completion_pct,
                "patients_with_demographics": patients_with_demographics,
                "patients_with_emergency_contacts": patients_with_emergency_contacts,
                "patients_with_medications": patients_with_medications,
                "patients_with_documents_uploaded": patients_with_documents,
                "avg_documents_per_patient": round(float(avg_docs), 2),
                "avg_fields_completed_per_patient": avg_fields,
            },
            "completion_distribution": dist,
            "onboarding_by_month": onboarding_by_month,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    finally:
        conn.close()


def breakdown():
    """
    Return per-patient onboarding detail across 5 sections + section-level stats.

    Returns:
        {
            patients: [
                {
                    patient_id, name, age, gender, disease, department,
                    sections: {
                        demographics: { complete, fields_done, fields_required },
                        emergency_contact: { complete, contacts_count, primary_set, notify_on_seizure },
                        medications: { complete, med_count, aed_listed },
                        documents: { complete, doc_count, categories_present },
                        comorbidities: { complete, screened, comorbidity_count, risk_severity }
                    },
                    completion_pct, onboarded
                }
            ],
            section_stats: {
                demographics: { complete_count, pct_complete },
                emergency_contact: { complete_count, pct_complete },
                medications: { complete_count, pct_complete },
                documents: { complete_count, pct_complete },
                comorbidities: { complete_count, pct_complete }
            }
        }
    """
    conn = _conn()
    try:
        completeness = _section_completeness(conn)
        patients = _fetch(
            conn,
            "SELECT patient_id, name, age, gender, disease, department, created_at "
            "FROM patients ORDER BY created_at",
        )

        # Pre-fetch emergency contact detail
        ec_rows = _fetch(
            conn,
            "SELECT patient_id, is_primary, notify_on_seizure, COUNT(*) as cnt "
            "FROM emergency_contacts GROUP BY patient_id",
        )
        ec_detail = {}
        for r in ec_rows:
            ec_detail[r["patient_id"]] = r
        # For accurate primary/notify per patient (re-aggregate)
        ec_all = _fetch(
            conn,
            "SELECT patient_id, is_primary, notify_on_seizure FROM emergency_contacts",
        )
        ec_agg = {}
        for r in ec_all:
            pid = r["patient_id"]
            if pid not in ec_agg:
                ec_agg[pid] = {"cnt": 0, "primary_set": False, "notify_on_seizure": False}
            ec_agg[pid]["cnt"] += 1
            if r["is_primary"]:
                ec_agg[pid]["primary_set"] = True
            if r["notify_on_seizure"]:
                ec_agg[pid]["notify_on_seizure"] = True

        # Pre-fetch medication detail
        med_rows = _fetch(conn, "SELECT patient_id, fields_json FROM medications")
        med_agg = {}
        for r in med_rows:
            pid = r["patient_id"]
            try:
                f = json.loads(r["fields_json"] or "{}")
            except Exception:
                f = {}
            if pid not in med_agg:
                med_agg[pid] = {"med_count": 0, "aed_listed": False}
            med_agg[pid]["med_count"] += 1
            if f.get("aed") or f.get("drug_name"):
                med_agg[pid]["aed_listed"] = True

        # Pre-fetch document categories
        doc_cat_rows = _fetch(
            conn,
            "SELECT patient_id, GROUP_CONCAT(DISTINCT category) as categories, "
            "COUNT(*) as doc_count FROM patient_documents GROUP BY patient_id",
        )
        doc_agg = {r["patient_id"]: r for r in doc_cat_rows}

        # Pre-fetch comorbidity screening detail
        comorbidity_rows = _fetch(
            conn,
            "SELECT patient_id, fields_json FROM comorbidities",
        )
        comorbidity_agg = {}
        for r in comorbidity_rows:
            pid = r["patient_id"]
            try:
                f = json.loads(r["fields_json"] or "{}")
            except Exception:
                f = {}
            comorbidity_agg[pid] = {
                "screened": bool(f.get("screened")),
                "comorbidity_count": f.get("comorbidity_count", 0),
                "risk_severity": f.get("risk_severity", "unknown"),
            }

        # Build per-patient list
        result_patients = []
        for p in patients:
            pid = p["patient_id"]
            s = completeness.get(pid, {})

            ec_info = ec_agg.get(pid, {"cnt": 0, "primary_set": False, "notify_on_seizure": False})
            med_info = med_agg.get(pid, {"med_count": 0, "aed_listed": False})
            doc_info = doc_agg.get(pid, {"doc_count": 0, "categories": ""})
            comorbidity_info = comorbidity_agg.get(
                pid, {"screened": False, "comorbidity_count": 0, "risk_severity": "not_screened"}
            )

            onboarded = (
                s.get("has_demographics", False)
                and s.get("has_emergency_contact", False)
                and s.get("has_medications", False)
                and s.get("has_documents", False)
            )

            result_patients.append({
                "patient_id": pid,
                "name": p["name"],
                "age": p["age"],
                "gender": p["gender"],
                "disease": p["disease"],
                "department": p["department"],
                "sections": {
                    "demographics": {
                        "complete": s.get("has_demographics", False),
                        "fields_done": s.get("demo_fields_completed", 0),
                        "fields_required": s.get("demo_fields_required", len(_DEMO_REQUIRED)),
                    },
                    "emergency_contact": {
                        "complete": s.get("has_emergency_contact", False),
                        "contacts_count": ec_info["cnt"],
                        "primary_set": ec_info["primary_set"],
                        "notify_on_seizure": ec_info["notify_on_seizure"],
                    },
                    "medications": {
                        "complete": s.get("has_medications", False),
                        "med_count": med_info["med_count"],
                        "aed_listed": med_info["aed_listed"],
                    },
                    "documents": {
                        "complete": s.get("has_documents", False),
                        "doc_count": doc_info.get("doc_count", 0),
                        "categories_present": (doc_info.get("categories") or "").split(","),
                    },
                    "comorbidities": {
                        "complete": s.get("has_comorbidities", False),
                        "screened": comorbidity_info["screened"],
                        "comorbidity_count": comorbidity_info["comorbidity_count"],
                        "risk_severity": comorbidity_info["risk_severity"],
                    },
                },
                "completion_pct": s.get("completion_pct", 0),
                "onboarded": onboarded,
            })

        total = len(result_patients)

        def _sect_stats(key):
            complete = sum(
                1 for p in result_patients if p["sections"][key]["complete"]
            )
            return {
                "complete_count": complete,
                "pct_complete": round(100 * complete / total, 1) if total else 0,
            }

        section_stats = {
            "demographics": _sect_stats("demographics"),
            "emergency_contact": _sect_stats("emergency_contact"),
            "medications": _sect_stats("medications"),
            "documents": _sect_stats("documents"),
            "comorbidities": _sect_stats("comorbidities"),
        }

        # Sort: incomplete patients first, then by completion_pct desc
        result_patients.sort(key=lambda p: (p["onboarded"], -p["completion_pct"]))

        return {
            "patients": result_patients,
            "section_stats": section_stats,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    finally:
        conn.close()


def definitions():
    """
    Return static onboarding process definitions: sections, field counts,
    process steps, and glossary.

    Returns:
        {
            sections: [...],
            required_fields: int,
            deferred_fields: int,
            process_steps: [...],
            glossary: [...]
        }
    """
    return {
        "sections": [
            {
                "step": 1,
                "name": "Demographics",
                "table": "patient_demographics",
                "description": (
                    "Core patient identity capture: name, date of birth, sex, blood type, "
                    "primary language, insurance type, address, referral source, and "
                    "epilepsy-specific intake fields (type, onset age, years with epilepsy, "
                    "primary neurologist, enrollment date)."
                ),
                "required_fields": 20,
                "deferred_fields": 9,
                "weight_pct": 35,
                "target_time_min": 3,
                "required_examples": [
                    "full_name", "date_of_birth", "age", "sex", "blood_type",
                    "primary_language", "insurance_type", "address_city",
                    "epilepsy_type", "epilepsy_onset_age",
                ],
                "deferred_examples": [
                    "gender_identity", "ethnicity", "race",
                    "interpreter_needed", "education_level",
                    "occupation", "employment_status", "marital_status", "bmi",
                ],
            },
            {
                "step": 2,
                "name": "Emergency Contact",
                "table": "emergency_contacts",
                "description": (
                    "Capture at least one emergency contact with name, phone, relationship, "
                    "and seizure notification preference. Designate a primary contact. "
                    "Seizure notification flag is required — it drives alert routing."
                ),
                "required_fields": 5,
                "deferred_fields": 2,
                "weight_pct": 20,
                "target_time_min": 1,
                "required_examples": [
                    "contact_name", "phone", "relationship", "is_primary", "notify_on_seizure"
                ],
                "deferred_examples": ["email", "last_verified"],
            },
            {
                "step": 2,
                "name": "Medications",
                "table": "medications",
                "description": (
                    "List current anti-epileptic drugs (AEDs) and any co-medications. "
                    "Minimum: drug name, dose (mg), frequency. Structured via fields_json. "
                    "Drives pharmacogenomics screening and drug-interaction checks."
                ),
                "required_fields": 3,
                "deferred_fields": 5,
                "weight_pct": 20,
                "target_time_min": 2,
                "required_examples": ["drug_name", "dose_mg", "frequency"],
                "deferred_examples": ["aed", "start_date", "end_date", "prescriber", "notes"],
            },
            {
                "step": 3,
                "name": "Documents",
                "table": "patient_documents",
                "description": (
                    "Upload and classify intake documents. Key types: Consent Form, "
                    "Referral Letter, Insurance Auth, Medication List, Seizure Action Plan. "
                    "System extracts structured data from uploaded files; non-critical fields "
                    "are deferred to the background extraction queue."
                ),
                "required_fields": 4,
                "deferred_fields": 1100,
                "weight_pct": 15,
                "target_time_min": 2,
                "required_examples": [
                    "document_type", "document_name", "upload_date", "category"
                ],
                "deferred_examples": [
                    "extracted_fields (structured data from OCR/NLP pipeline)"
                ],
                "document_types": [
                    "Lab Results", "Education Material", "MRI Report",
                    "Referral Letter", "Insurance Auth", "Medication List",
                    "Consent Form", "Discharge Summary", "Seizure Action Plan", "EEG Report",
                ],
            },
            {
                "step": 3,
                "name": "Comorbidities",
                "table": "comorbidities",
                "description": (
                    "Psychiatric comorbidity screening using validated instruments "
                    "(PHQ-9, GAD-7, C-SSRS, NDDI-E, PCL-5). Record screened/not-screened, "
                    "condition list, behavioral risk score, and risk severity. "
                    "Detailed clinical scoring is deferred."
                ),
                "required_fields": 3,
                "deferred_fields": 54,
                "weight_pct": 10,
                "target_time_min": 1,
                "required_examples": ["screened", "comorbidity_count", "risk_severity"],
                "deferred_examples": [
                    "screening_instruments detail", "functional_impact",
                    "treatment_status", "individual_scores", "follow_up_plan",
                ],
            },
        ],
        "required_fields": _TOTAL_REQUIRED_FIELDS,
        "deferred_fields": _TOTAL_DEFERRED_FIELDS,
        "process_steps": [
            {
                "step": 1,
                "label": "Demographics",
                "description": "Required-first capture of patient identity and epilepsy intake fields.",
                "inputs": [
                    "Full name, DOB, sex, blood type",
                    "Primary language, insurance type",
                    "Address (city, state, ZIP)",
                    "Epilepsy type, onset age, years with condition",
                    "Primary neurologist, enrollment date",
                ],
                "duration_min": 3,
                "tables": ["patient_demographics"],
            },
            {
                "step": 2,
                "label": "Clinical Core",
                "description": "Emergency contact, current medications, comorbidity screening.",
                "inputs": [
                    "Emergency contact name, phone, relationship, seizure-notify flag",
                    "AED names, doses, frequencies",
                    "Comorbidity screening (PHQ-9, GAD-7 at minimum)",
                ],
                "duration_min": 3,
                "tables": ["emergency_contacts", "medications", "comorbidities"],
            },
            {
                "step": 3,
                "label": "Documents",
                "description": "Upload intake documents; extraction pipeline defers ~1170 fields.",
                "inputs": [
                    "Consent Form",
                    "Referral Letter",
                    "Insurance Authorization",
                    "Medication List (if not entered manually)",
                    "Seizure Action Plan (if available)",
                ],
                "duration_min": 2,
                "tables": ["patient_documents"],
            },
        ],
        "glossary": [
            {
                "term": "Onboarded",
                "definition": (
                    "A patient is considered fully onboarded when all 4 core sections are "
                    "present: patient_demographics row, at least one emergency_contact, "
                    "at least one medication, and at least one patient_document."
                ),
            },
            {
                "term": "Required Fields (~80)",
                "definition": (
                    "Fields captured during the 8-10 minute intake session. Prioritized by "
                    "clinical urgency: identity, seizure history, emergency response, active "
                    "medications, and document classification."
                ),
            },
            {
                "term": "Deferred Fields (~1170)",
                "definition": (
                    "Non-critical fields extracted asynchronously by the document extraction "
                    "pipeline, filled in by the patient via portal, or completed during "
                    "follow-up appointments. Never block initial onboarding."
                ),
            },
            {
                "term": "AED",
                "definition": "Anti-Epileptic Drug — medications used to control seizures.",
            },
            {
                "term": "Field Deferral",
                "definition": (
                    "Strategy of skipping low-urgency intake fields to minimize session time. "
                    "Deferred fields are queued for later completion without blocking care."
                ),
            },
            {
                "term": "Completion %",
                "definition": (
                    "Weighted score per patient: Demographics 35%, Emergency Contact 20%, "
                    "Medications 20%, Documents 15%, Comorbidities 10%. "
                    "Demographics weight is further adjusted by field fill rate."
                ),
            },
            {
                "term": "Multi-format Document Extraction",
                "definition": (
                    "OCR + NLP pipeline that reads uploaded PDFs, images, and structured "
                    "files to auto-populate deferred fields — reducing manual data entry."
                ),
            },
            {
                "term": "Notify on Seizure",
                "definition": (
                    "emergency_contacts.notify_on_seizure flag. When true, this contact "
                    "receives automated alert messages when a seizure event is detected "
                    "via wearable or manually logged."
                ),
            },
        ],
    }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "overview"

    if cmd == "overview":
        result = overview()
    elif cmd == "breakdown":
        result = breakdown()
    elif cmd == "definitions":
        result = definitions()
    else:
        print(f"Unknown command: {cmd}. Use: overview | breakdown | definitions")
        sys.exit(1)

    print(json.dumps(result, indent=2, default=str))
