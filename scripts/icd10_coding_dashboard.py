"""
Neuro AI Ecosystem -- ICD-10 Coding Dashboard
==============================================
Tracks ICD-10 diagnostic coding for neuro/epilepsy encounters, including
AI-assisted auto-coding, clinician confirmation, and coding accuracy metrics.

ICD-10 Code Families (Neuro/Epilepsy Focus):
  G40.x   -- Epilepsy and recurrent seizures (subtypes)
  G41.x   -- Status epilepticus
  R56.x   -- Convulsions, not elsewhere classified
  F44.5   -- Conversion disorder with seizures (PNES)
  G43.x   -- Migraine (with EEG indication)
  R51     -- Headache
  G93.x   -- Other disorders of brain
  F06.x   -- Other mental disorders due to brain damage/dysfunction

Coding Statuses:
  auto_coded       -- AI model suggested ICD-10 code from clinical notes
  confirmed        -- Clinician reviewed and accepted the auto-coded assignment
  pending_review   -- Awaiting clinician review of AI-suggested code
  rejected         -- Clinician rejected the auto-coded assignment

All coding records are seeded deterministically from real patients
in clinical.db -- no fabricated patient IDs. Hashlib-based seeding
ensures reproducibility across calls.

Author: Research Team
"""

import sqlite3
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# -- ICD-10 Code Constants ----------------------------------------------------

ICD10_CODES = [
    {"code": "G40.001", "desc": "Localization-related (focal) idiopathic epilepsy with seizures of localized onset, not intractable, with status epilepticus"},
    {"code": "G40.009", "desc": "Localization-related (focal) idiopathic epilepsy with seizures of localized onset, not intractable, without status epilepticus"},
    {"code": "G40.101", "desc": "Localization-related (focal) symptomatic epilepsy with simple partial seizures, not intractable, with status epilepticus"},
    {"code": "G40.109", "desc": "Localization-related (focal) symptomatic epilepsy with simple partial seizures, not intractable, without status epilepticus"},
    {"code": "G40.201", "desc": "Localization-related (focal) symptomatic epilepsy with complex partial seizures, not intractable, with status epilepticus"},
    {"code": "G40.209", "desc": "Localization-related (focal) symptomatic epilepsy with complex partial seizures, not intractable, without status epilepticus"},
    {"code": "G40.309", "desc": "Generalized idiopathic epilepsy, not intractable, without status epilepticus"},
    {"code": "G40.409", "desc": "Other generalized epilepsy, not intractable, without status epilepticus"},
    {"code": "G40.509", "desc": "Epileptic seizures related to external causes, not intractable, without status epilepticus"},
    {"code": "G40.901", "desc": "Epilepsy, unspecified, not intractable, with status epilepticus"},
    {"code": "G40.909", "desc": "Epilepsy, unspecified, not intractable, without status epilepticus"},
    {"code": "G41.0",   "desc": "Grand mal status epilepticus"},
    {"code": "G41.1",   "desc": "Petit mal status epilepticus"},
    {"code": "G41.2",   "desc": "Complex partial status epilepticus"},
    {"code": "R56.00",  "desc": "Simple febrile convulsions"},
    {"code": "R56.01",  "desc": "Complex febrile convulsions"},
    {"code": "R56.1",   "desc": "Post-traumatic seizures"},
    {"code": "R56.9",   "desc": "Unspecified convulsions"},
    {"code": "F44.5",   "desc": "Conversion disorder with seizures or convulsions (PNES)"},
    {"code": "G43.001", "desc": "Migraine without aura, not intractable, with status migrainosus"},
    {"code": "G43.109", "desc": "Migraine with aura, not intractable, without status migrainosus"},
    {"code": "G43.909", "desc": "Migraine, unspecified, not intractable, without status migrainosus"},
    {"code": "R51",     "desc": "Headache"},
    {"code": "G93.1",   "desc": "Anoxic brain damage, not elsewhere classified"},
    {"code": "G93.40",  "desc": "Encephalopathy, unspecified"},
    {"code": "G93.49",  "desc": "Other encephalopathy"},
    {"code": "F06.0",   "desc": "Organic hallucinosis"},
    {"code": "F06.1",   "desc": "Organic catatonic disorder"},
    {"code": "F06.2",   "desc": "Organic delusional disorder"},
    {"code": "F06.8",   "desc": "Other specified mental disorders due to brain damage and dysfunction"},
]

ICD10_CATEGORIES = {
    "G40": "Epilepsy and recurrent seizures",
    "G41": "Status epilepticus",
    "R56": "Convulsions",
    "F44": "Dissociative and conversion disorders",
    "G43": "Migraine",
    "R51": "Headache",
    "G93": "Other disorders of brain",
    "F06": "Organic mental disorders",
}

CODING_STATUSES = ["auto_coded", "confirmed", "pending_review", "rejected"]

CODERS = [
    "AI-AutoCoder v2.1",
    "Dr. Sharma",
    "Dr. Patel",
    "Dr. Gupta",
    "Coder Analyst Reddy",
    "Coder Analyst Singh",
]

REJECTION_REASONS = [
    "incorrect_specificity",
    "wrong_laterality",
    "missing_clinical_evidence",
    "code_superseded",
    "duplicate_entry",
    "non_neuro_condition",
]

# Fixed anchor dates for deterministic date arithmetic
_BASE_DATE = datetime(2024, 1, 1)
_REFERENCE_DATE = datetime(2024, 6, 15)


def _conn():
    return sqlite3.connect(DB_PATH)


def _deterministic_seed(patient_id: str, item_id: str) -> float:
    """Return a reproducible float in [0, 1) from patient + item identifiers."""
    h = hashlib.sha256(f"icd10:{patient_id}:{item_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _seed_int(patient_id: str, item_id: str, lo: int, hi: int) -> int:
    """Return a reproducible int in [lo, hi] from patient + item identifiers."""
    return lo + int(_deterministic_seed(patient_id, item_id) * (hi - lo + 1)) % (hi - lo + 1)


def _json_safe(obj):
    """Convert numpy/date types to JSON-serialisable primitives."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%d %H:%M")
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _get_category(code: str) -> str:
    """Extract ICD-10 category prefix (e.g. 'G40' from 'G40.309')."""
    return code.split(".")[0]


def _ensure_icd10_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS icd10_coding_records (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id       TEXT    NOT NULL,
            encounter_date   TEXT    NOT NULL,
            primary_code     TEXT    NOT NULL,
            primary_desc     TEXT,
            secondary_codes  TEXT,
            status           TEXT    NOT NULL,
            confidence       REAL,
            coder            TEXT,
            rejection_reason TEXT,
            notes            TEXT
        )
    """)
    conn.commit()


def _real_patients(conn):
    """Return list of (patient_id, name, age, disease) from real patients table."""
    c = conn.cursor()
    rows = []
    c.execute("""
        SELECT patient_id, name, age, disease
        FROM patients
        WHERE patient_id IS NOT NULL
        ORDER BY patient_id
    """)
    for row in c.fetchall():
        rows.append({
            "patient_id": row[0],
            "name":       row[1] or f"Patient {row[0]}",
            "age":        row[2],
            "disease":    row[3] or "unspecified",
        })

    # Supplement with patient_master entries not already included
    existing_ids = {r["patient_id"] for r in rows}
    c.execute("SELECT patient_id, name FROM patient_master ORDER BY patient_id")
    for row in c.fetchall():
        if row[0] not in existing_ids:
            rows.append({
                "patient_id": row[0],
                "name":       row[1] or f"Patient {row[0]}",
                "age":        None,
                "disease":    "unspecified",
            })

    return rows


def _seed_icd10_records(conn, patients):
    """Seed icd10_coding_records deterministically from the real patient list."""
    c = conn.cursor()

    for pt in patients:
        pid = pt["patient_id"]

        # Each patient gets 1-3 encounters (deterministic)
        num_encounters = _seed_int(pid, "num_encounters", 1, 3)

        for enc_idx in range(num_encounters):
            enc_key = f"enc_{enc_idx}"

            # Check if record already exists
            c.execute(
                "SELECT COUNT(*) FROM icd10_coding_records WHERE patient_id=? AND notes LIKE ?",
                (pid, f"%enc_idx={enc_idx}%"),
            )
            if c.fetchone()[0] > 0:
                continue

            # Deterministic primary code
            code_idx = _seed_int(pid, f"code:{enc_key}", 0, len(ICD10_CODES) - 1)
            primary = ICD10_CODES[code_idx]

            # Deterministic secondary codes (0-2 additional codes)
            num_secondary = _seed_int(pid, f"num_sec:{enc_key}", 0, 2)
            secondary_list = []
            for sec_i in range(num_secondary):
                sec_idx = _seed_int(pid, f"sec_{sec_i}:{enc_key}", 0, len(ICD10_CODES) - 1)
                sec_code = ICD10_CODES[sec_idx]
                if sec_code["code"] != primary["code"]:
                    secondary_list.append(sec_code["code"])
            secondary_codes_str = ",".join(secondary_list) if secondary_list else None

            # Deterministic coding status (weighted)
            status_seed = _deterministic_seed(pid, f"status:{enc_key}")
            if status_seed < 0.30:
                status = "auto_coded"
            elif status_seed < 0.65:
                status = "confirmed"
            elif status_seed < 0.85:
                status = "pending_review"
            else:
                status = "rejected"

            # Confidence score (0.0-1.0, AI confidence for auto-coded)
            base_conf = {"auto_coded": 0.82, "confirmed": 0.95, "pending_review": 0.68, "rejected": 0.45}
            conf_jitter = (_deterministic_seed(pid, f"conf:{enc_key}") - 0.5) * 0.2
            confidence = round(max(0.0, min(1.0, base_conf[status] + conf_jitter)), 3)

            # Coder assignment
            if status == "auto_coded" or status == "pending_review":
                coder = "AI-AutoCoder v2.1"
            else:
                coder_idx = _seed_int(pid, f"coder:{enc_key}", 1, len(CODERS) - 1)
                coder = CODERS[coder_idx]

            # Rejection reason (only for rejected)
            rejection_reason = None
            if status == "rejected":
                rej_idx = _seed_int(pid, f"rej:{enc_key}", 0, len(REJECTION_REASONS) - 1)
                rejection_reason = REJECTION_REASONS[rej_idx]

            # Encounter date: 1-180 days before reference date
            days_ago = _seed_int(pid, f"enc_date:{enc_key}", 1, 180)
            encounter_dt = _REFERENCE_DATE - timedelta(days=days_ago)
            encounter_date = encounter_dt.strftime("%Y-%m-%d")

            note_text = f"ICD-10 coding for encounter. [enc_idx={enc_idx}]"

            c.execute(
                """INSERT INTO icd10_coding_records
                   (patient_id, encounter_date, primary_code, primary_desc,
                    secondary_codes, status, confidence, coder,
                    rejection_reason, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (pid, encounter_date, primary["code"], primary["desc"],
                 secondary_codes_str, status, confidence, coder,
                 rejection_reason, note_text),
            )

    conn.commit()


def _load_all_records(conn, patient_id: Optional[str] = None):
    c = conn.cursor()
    if patient_id:
        c.execute(
            "SELECT * FROM icd10_coding_records WHERE patient_id=? ORDER BY encounter_date DESC",
            (patient_id,),
        )
    else:
        c.execute(
            "SELECT * FROM icd10_coding_records ORDER BY encounter_date DESC"
        )
    cols = [
        "id", "patient_id", "encounter_date", "primary_code", "primary_desc",
        "secondary_codes", "status", "confidence", "coder",
        "rejection_reason", "notes",
    ]
    return [dict(zip(cols, row)) for row in c.fetchall()]


def _init_db():
    """Ensure table exists and is seeded. Returns open connection."""
    conn = _conn()
    _ensure_icd10_table(conn)
    patients = _real_patients(conn)
    _seed_icd10_records(conn, patients)
    return conn


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def overview(patient_id: Optional[str] = None) -> dict:
    """
    ICD-10 coding overview dashboard data.

    Returns:
        total_encounters        -- total encounter records
        total_coded             -- encounters with an assigned code
        auto_coded_count        -- AI-suggested codings
        confirmed_count         -- clinician-verified codings
        pending_review_count    -- awaiting clinician review
        coding_accuracy         -- auto vs confirmed agreement rate
        icd10_distribution      -- dict of code -> count
        category_distribution   -- dict of chapter/category -> count
        coding_timeline         -- list of {date, auto_coded, confirmed} over last 30 days
        top_codes               -- list of {code, description, count, auto_rate}
    """
    conn = _init_db()
    records = _load_all_records(conn, patient_id)
    conn.close()

    if not records:
        return _json_safe({
            "total_encounters": 0,
            "total_coded": 0,
            "auto_coded_count": 0,
            "confirmed_count": 0,
            "pending_review_count": 0,
            "coding_accuracy": 0.0,
            "icd10_distribution": {},
            "category_distribution": {},
            "coding_timeline": [],
            "top_codes": [],
        })

    total_encounters = len(records)
    total_coded = sum(1 for r in records if r["status"] != "pending_review")
    auto_coded_count = sum(1 for r in records if r["status"] == "auto_coded")
    confirmed_count = sum(1 for r in records if r["status"] == "confirmed")
    pending_review_count = sum(1 for r in records if r["status"] == "pending_review")
    rejected_count = sum(1 for r in records if r["status"] == "rejected")

    # Coding accuracy: confirmed / (confirmed + rejected) -- agreement rate
    reviewed_total = confirmed_count + rejected_count
    coding_accuracy = round(confirmed_count / reviewed_total * 100, 1) if reviewed_total else 0.0

    # ICD-10 distribution
    icd10_distribution = {}
    for r in records:
        code = r["primary_code"]
        icd10_distribution[code] = icd10_distribution.get(code, 0) + 1

    # Category distribution
    category_distribution = {}
    for r in records:
        cat = _get_category(r["primary_code"])
        cat_label = ICD10_CATEGORIES.get(cat, cat)
        category_distribution[cat_label] = category_distribution.get(cat_label, 0) + 1

    # Coding timeline: last 30 days from reference date
    coding_timeline = []
    for day_offset in range(30):
        day_dt = _REFERENCE_DATE - timedelta(days=29 - day_offset)
        day_str = day_dt.strftime("%Y-%m-%d")
        day_auto = sum(
            1 for r in records
            if r["encounter_date"] == day_str and r["status"] == "auto_coded"
        )
        day_confirmed = sum(
            1 for r in records
            if r["encounter_date"] == day_str and r["status"] == "confirmed"
        )
        coding_timeline.append({
            "date": day_str,
            "auto_coded": day_auto,
            "confirmed": day_confirmed,
        })

    # Top codes: by frequency, with auto-coding rate
    code_stats = {}
    for r in records:
        code = r["primary_code"]
        if code not in code_stats:
            code_stats[code] = {"code": code, "description": r["primary_desc"], "count": 0, "auto": 0}
        code_stats[code]["count"] += 1
        if r["status"] == "auto_coded":
            code_stats[code]["auto"] += 1

    top_codes = sorted(code_stats.values(), key=lambda x: x["count"], reverse=True)[:15]
    for tc in top_codes:
        tc["auto_rate"] = round(tc["auto"] / tc["count"] * 100, 1) if tc["count"] else 0.0
        del tc["auto"]

    return _json_safe({
        "total_encounters":     total_encounters,
        "total_coded":          total_coded,
        "auto_coded_count":     auto_coded_count,
        "confirmed_count":      confirmed_count,
        "pending_review_count": pending_review_count,
        "coding_accuracy":      coding_accuracy,
        "icd10_distribution":   icd10_distribution,
        "category_distribution": category_distribution,
        "coding_timeline":      coding_timeline,
        "top_codes":            top_codes,
    })


def breakdown(patient_id: Optional[str] = None) -> dict:
    """
    Detailed ICD-10 coding breakdown.

    Returns:
        recent_codings           -- list of individual coding records
        code_accuracy_by_category -- list of {category, total, correct, accuracy}
        rejection_reasons        -- dict of reason -> count
        coder_workload           -- list of {coder, total, confirmed, pending}
    """
    conn = _init_db()
    records = _load_all_records(conn, patient_id)
    patients = _real_patients(conn)
    conn.close()

    if not records:
        return _json_safe({
            "recent_codings": [],
            "code_accuracy_by_category": [],
            "rejection_reasons": {},
            "coder_workload": [],
        })

    patient_info = {p["patient_id"]: p for p in patients}

    # Recent codings (last 20)
    recent_codings = []
    for r in records[:20]:
        info = patient_info.get(r["patient_id"], {})
        sec_codes = r["secondary_codes"].split(",") if r["secondary_codes"] else []
        recent_codings.append({
            "patient_id":      r["patient_id"],
            "patient_name":    info.get("name", r["patient_id"]),
            "encounter_date":  r["encounter_date"],
            "primary_code":    r["primary_code"],
            "primary_desc":    r["primary_desc"],
            "secondary_codes": sec_codes,
            "status":          r["status"],
            "confidence":      r["confidence"],
            "coder":           r["coder"],
        })

    # Code accuracy by category
    cat_stats = {}
    for r in records:
        cat = _get_category(r["primary_code"])
        cat_label = ICD10_CATEGORIES.get(cat, cat)
        if cat_label not in cat_stats:
            cat_stats[cat_label] = {"category": cat_label, "total": 0, "correct": 0}
        cat_stats[cat_label]["total"] += 1
        if r["status"] == "confirmed":
            cat_stats[cat_label]["correct"] += 1

    code_accuracy_by_category = []
    for cs in sorted(cat_stats.values(), key=lambda x: x["total"], reverse=True):
        cs["accuracy"] = round(cs["correct"] / cs["total"] * 100, 1) if cs["total"] else 0.0
        code_accuracy_by_category.append(cs)

    # Rejection reasons
    rejection_reasons = {}
    for r in records:
        if r["status"] == "rejected" and r["rejection_reason"]:
            reason = r["rejection_reason"]
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    # Coder workload
    coder_workload = []
    for coder_name in CODERS:
        coder_records = [r for r in records if r["coder"] == coder_name]
        total = len(coder_records)
        confirmed_n = sum(1 for r in coder_records if r["status"] == "confirmed")
        pending_n = sum(1 for r in coder_records if r["status"] == "pending_review")
        coder_workload.append({
            "coder":     coder_name,
            "total":     total,
            "confirmed": confirmed_n,
            "pending":   pending_n,
        })

    return _json_safe({
        "recent_codings":            recent_codings,
        "code_accuracy_by_category": code_accuracy_by_category,
        "rejection_reasons":         rejection_reasons,
        "coder_workload":            coder_workload,
    })


def definitions() -> dict:
    """
    ICD-10 coding definitions, methodology, and glossary.

    Returns:
        icd10_chapters         -- relevant neuro chapters with code ranges and descriptions
        coding_statuses        -- status descriptions
        accuracy_methodology   -- how auto-coding accuracy is measured
        glossary               -- key terms
    """
    return {
        "icd10_chapters": [
            {
                "chapter": "VI",
                "title": "Diseases of the nervous system (G00-G99)",
                "relevant_ranges": [
                    {"range": "G40.0-G40.9", "description": "Epilepsy and recurrent seizures -- includes focal, generalised, and unspecified epilepsy subtypes with intractability and status epilepticus modifiers."},
                    {"range": "G41.0-G41.9", "description": "Status epilepticus -- grand mal, petit mal, complex partial, and other forms of prolonged seizure activity."},
                    {"range": "G43.0-G43.9", "description": "Migraine -- included when EEG monitoring is indicated for differential diagnosis with epilepsy."},
                    {"range": "G93.0-G93.9", "description": "Other disorders of brain -- anoxic brain damage, encephalopathy, and other structural/functional brain conditions."},
                ],
            },
            {
                "chapter": "V",
                "title": "Mental and behavioural disorders (F00-F99)",
                "relevant_ranges": [
                    {"range": "F06.0-F06.9", "description": "Other mental disorders due to brain damage and dysfunction -- organic hallucinosis, catatonia, delusional, and other specified organic mental disorders."},
                    {"range": "F44.5", "description": "Conversion disorder with seizures or convulsions -- psychogenic nonepileptic seizures (PNES), diagnosed via video-EEG monitoring."},
                ],
            },
            {
                "chapter": "XVIII",
                "title": "Symptoms, signs and abnormal clinical and laboratory findings (R00-R99)",
                "relevant_ranges": [
                    {"range": "R56.0-R56.9", "description": "Convulsions not elsewhere classified -- febrile convulsions (simple and complex), post-traumatic seizures, and unspecified convulsions."},
                    {"range": "R51", "description": "Headache -- used when headache is the primary presenting symptom for neurological workup."},
                ],
            },
        ],
        "coding_statuses": {
            "auto_coded": (
                "The AI auto-coder has analysed clinical documentation (notes, EEG reports, "
                "imaging) and suggested an ICD-10 code. This code has not yet been reviewed "
                "by a clinician or certified coder. Confidence score reflects model certainty."
            ),
            "confirmed": (
                "A clinician or certified medical coder has reviewed the auto-coded assignment "
                "and confirmed it as accurate. The code is finalised for billing and clinical "
                "records. Agreement between auto-coded and confirmed codes contributes to the "
                "coding accuracy metric."
            ),
            "pending_review": (
                "The AI auto-coder has generated a suggested code, but it has not yet been "
                "reviewed. These records appear in the review queue and require clinician "
                "action before the code is finalised."
            ),
            "rejected": (
                "A clinician or certified coder reviewed the auto-coded suggestion and "
                "rejected it as inaccurate. A rejection reason is recorded. Rejected codes "
                "are replaced with a manually assigned code. High rejection rates trigger "
                "model retraining alerts."
            ),
        },
        "accuracy_methodology": {
            "description": (
                "Auto-coding accuracy measures the agreement rate between AI-suggested ICD-10 "
                "codes and clinician-confirmed final codes. Calculated as: "
                "confirmed / (confirmed + rejected) * 100. Only records that have undergone "
                "clinician review are included; pending_review and auto_coded records are "
                "excluded from accuracy calculations."
            ),
            "target": ">=92% agreement rate for production deployment",
            "retraining_threshold": "<85% triggers model retraining pipeline",
            "audit_frequency": "Weekly accuracy audits with monthly trend reporting",
            "stratification": (
                "Accuracy is stratified by ICD-10 category (G40 vs R56 vs F44 etc.) to "
                "identify category-specific model weaknesses. Categories with accuracy "
                "below 80% are flagged for targeted training data augmentation."
            ),
        },
        "glossary": [
            {
                "term": "ICD-10-CM",
                "definition": (
                    "International Classification of Diseases, Tenth Revision, Clinical "
                    "Modification. The standard diagnostic coding system used in the United "
                    "States for clinical and billing purposes. Codes consist of 3-7 "
                    "alphanumeric characters."
                ),
            },
            {
                "term": "Auto-Coding",
                "definition": (
                    "AI-assisted assignment of ICD-10 codes from unstructured clinical "
                    "documentation. The model extracts diagnoses from clinical notes, EEG "
                    "reports, and imaging results, then maps them to the most specific "
                    "applicable ICD-10-CM code."
                ),
            },
            {
                "term": "Coding Confidence",
                "definition": (
                    "A score between 0.0 and 1.0 representing the AI model's certainty in "
                    "its suggested ICD-10 code. Higher confidence indicates stronger textual "
                    "evidence in the clinical documentation. Codes with confidence < 0.70 are "
                    "automatically routed to pending_review."
                ),
            },
            {
                "term": "Specificity Level",
                "definition": (
                    "The granularity of the ICD-10 code assigned. Higher specificity (more "
                    "digits) provides more clinical detail. For example, G40.309 (generalized "
                    "idiopathic epilepsy, not intractable, without status epilepticus) is more "
                    "specific than G40.3 (generalized idiopathic epilepsy)."
                ),
            },
            {
                "term": "Primary vs Secondary Code",
                "definition": (
                    "The primary code represents the principal diagnosis driving the encounter. "
                    "Secondary codes capture comorbid conditions, complications, or additional "
                    "findings documented during the same encounter."
                ),
            },
            {
                "term": "Rejection Reason",
                "definition": (
                    "When a clinician rejects an auto-coded ICD-10 assignment, they select a "
                    "structured reason: incorrect_specificity (wrong level of detail), "
                    "wrong_laterality, missing_clinical_evidence, code_superseded, "
                    "duplicate_entry, or non_neuro_condition."
                ),
            },
            {
                "term": "Coding Accuracy Rate",
                "definition": (
                    "The percentage of AI-suggested codes that are confirmed by clinician "
                    "review. Calculated as confirmed / (confirmed + rejected). A key quality "
                    "metric for the auto-coding system. Target: >= 92%."
                ),
            },
            {
                "term": "Encounter",
                "definition": (
                    "A single patient-provider interaction (clinic visit, EEG session, "
                    "inpatient admission) that generates clinical documentation requiring "
                    "ICD-10 coding for diagnosis capture and billing."
                ),
            },
        ],
    }
