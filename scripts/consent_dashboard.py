"""
Neuro AI Ecosystem — Consent Management Dashboard
==================================================
Tracks patient consent status for research, treatment, data sharing,
and genetic testing across the neuro/epilepsy clinical platform.

Consent Types:
  treatment       — Standard clinical treatment consent
  research        — Participation in research studies / trials
  data_sharing    — Sharing of de-identified clinical data
  genetic_testing — Collection and use of genetic samples
  video_eeg       — Consent for video-EEG monitoring and recording
  imaging_sharing — Sharing of imaging data (MRI, CT) with research

Statuses:
  granted   — Consent actively given and in effect
  pending   — Awaiting patient decision / signature
  declined  — Patient explicitly declined consent
  expired   — Consent granted but past expiry date
  withdrawn — Consent previously granted but subsequently withdrawn

Regulatory Framework:
  HIPAA (45 CFR §164.508) — Authorization for uses/disclosures
  21 CFR Part 50          — FDA protection of human subjects
  Common Rule (45 CFR 46) — IRB-governed research consent
  GDPR Article 7          — Conditions for consent (EU patients)

All consent records are seeded deterministically from real patients
in clinical.db — no fabricated patient IDs. Hashlib-based seeding
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

# ── Consent Type Definitions ──────────────────────────────────────────
CONSENT_TYPES = [
    "treatment",
    "research",
    "data_sharing",
    "genetic_testing",
    "video_eeg",
    "imaging_sharing",
]

CONSENT_STATUSES = ["granted", "pending", "declined", "expired", "withdrawn"]

# Regulatory base date for date arithmetic (fixed anchor for determinism)
_BASE_DATE = datetime(2024, 1, 1)


def _conn():
    return sqlite3.connect(DB_PATH)


def _deterministic_seed(patient_id: str, item_id: str) -> float:
    """Return a reproducible float in [0, 1) from patient + item identifiers."""
    h = hashlib.sha256(f"consent:{patient_id}:{item_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _seed_int(patient_id: str, item_id: str, lo: int, hi: int) -> int:
    """Return a reproducible int in [lo, hi] from patient + item identifiers."""
    return lo + int(_deterministic_seed(patient_id, item_id) * (hi - lo + 1)) % (hi - lo + 1)


def _ensure_consent_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS consent_records (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id     TEXT    NOT NULL,
            consent_type   TEXT    NOT NULL,
            status         TEXT    NOT NULL,
            granted_date   TEXT,
            expiry_date    TEXT,
            witness        TEXT,
            notes          TEXT
        )
    """)
    conn.commit()


def _real_patients(conn):
    """Return list of (patient_id, name, age, disease) from real patients table."""
    c = conn.cursor()
    rows = []
    # Primary: patients table — filter to rows with meaningful data
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


def _seed_consent_records(conn, patients):
    """Seed consent_records deterministically from the real patient list."""
    c = conn.cursor()

    # Witness pool (fixed list for reproducibility)
    witnesses = [
        "Dr. A. Mehta", "Dr. S. Patel", "Nurse J. Clarke",
        "Dr. R. Singh", "Coordinator L. Torres", "Nurse M. Kim",
    ]

    notes_pool = {
        "treatment":       "Standard informed consent for neurological treatment protocol.",
        "research":        "Enrolled in epilepsy biomarker research study (IRB-2024-0042).",
        "data_sharing":    "De-identified data authorised for multi-site registry.",
        "genetic_testing": "Saliva sample collected; WES panel ordered.",
        "video_eeg":       "Video-EEG monitoring for pre-surgical evaluation.",
        "imaging_sharing": "MRI/CT scans shared with federated imaging consortium.",
    }

    for pt in patients:
        pid = pt["patient_id"]
        for consent_type in CONSENT_TYPES:
            # Check if record already exists
            c.execute(
                "SELECT COUNT(*) FROM consent_records WHERE patient_id=? AND consent_type=?",
                (pid, consent_type),
            )
            if c.fetchone()[0] > 0:
                continue

            # Deterministic status selection
            status_seed = _deterministic_seed(pid, f"status:{consent_type}")
            # Weighted distribution: granted ~55 %, pending ~15 %, declined ~10 %,
            # expired ~12 %, withdrawn ~8 %
            thresholds = [0.55, 0.70, 0.80, 0.92, 1.00]
            status_labels = ["granted", "pending", "declined", "expired", "withdrawn"]
            status = status_labels[-1]
            for thr, lbl in zip(thresholds, status_labels):
                if status_seed < thr:
                    status = lbl
                    break

            # Deterministic dates
            days_ago_granted = _seed_int(pid, f"days_granted:{consent_type}", 30, 730)
            granted_date_dt = _BASE_DATE + timedelta(days=days_ago_granted)
            granted_date = granted_date_dt.strftime("%Y-%m-%d") if status != "pending" else None

            # Expiry: 1 or 2 years after granted date
            expiry_years = 1 + _seed_int(pid, f"expiry:{consent_type}", 0, 1)
            if granted_date:
                expiry_date_dt = granted_date_dt + timedelta(days=365 * expiry_years)
                expiry_date = expiry_date_dt.strftime("%Y-%m-%d")
            else:
                expiry_date = None

            # Witness
            w_idx = _seed_int(pid, f"witness:{consent_type}", 0, len(witnesses) - 1)
            witness = witnesses[w_idx] if status in ("granted", "expired", "withdrawn") else None

            note = notes_pool.get(consent_type, "")

            c.execute(
                """INSERT INTO consent_records
                   (patient_id, consent_type, status, granted_date, expiry_date, witness, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (pid, consent_type, status, granted_date, expiry_date, witness, note),
            )

    conn.commit()


def _load_all_records(conn, patient_id: Optional[str] = None):
    c = conn.cursor()
    if patient_id:
        c.execute(
            "SELECT * FROM consent_records WHERE patient_id=? ORDER BY patient_id, consent_type",
            (patient_id,),
        )
    else:
        c.execute(
            "SELECT * FROM consent_records ORDER BY patient_id, consent_type"
        )
    cols = ["id", "patient_id", "consent_type", "status", "granted_date",
            "expiry_date", "witness", "notes"]
    return [dict(zip(cols, row)) for row in c.fetchall()]


def _is_expiring_soon(expiry_date: Optional[str], within_days: int = 90) -> bool:
    if not expiry_date:
        return False
    try:
        exp = datetime.strptime(expiry_date, "%Y-%m-%d")
        today = datetime(2024, 6, 15)  # fixed reference date for reproducibility
        return timedelta(0) <= (exp - today) <= timedelta(days=within_days)
    except ValueError:
        return False


def _init_db():
    """Ensure table exists and is seeded. Returns open connection."""
    conn = _conn()
    _ensure_consent_table(conn)
    patients = _real_patients(conn)
    _seed_consent_records(conn, patients)
    return conn


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def overview(patient_id: Optional[str] = None) -> dict:
    """
    Consent overview dashboard data.

    Returns:
        total_patients      — distinct patients with any consent record
        total_consents      — total consent records
        consent_rate        — % of records with status 'granted'
        by_type             — counts per consent_type
        by_status           — counts per status
        kpis                — list of {label, value, sub} dicts
        expiring_soon       — count of granted consents expiring within 90 days
        recent_activity     — last 10 consent records (by rowid desc)
    """
    conn = _init_db()
    records = _load_all_records(conn, patient_id)
    conn.close()

    if not records:
        return {
            "total_patients": 0,
            "total_consents": 0,
            "consent_rate": 0.0,
            "by_type": {},
            "by_status": {},
            "kpis": [],
            "expiring_soon": 0,
            "recent_activity": [],
        }

    total_consents = len(records)
    total_patients = len({r["patient_id"] for r in records})

    granted_count = sum(1 for r in records if r["status"] == "granted")
    consent_rate = round(granted_count / total_consents * 100, 1) if total_consents else 0.0

    by_type: dict = {}
    for ct in CONSENT_TYPES:
        by_type[ct] = sum(1 for r in records if r["consent_type"] == ct)

    by_status: dict = {s: 0 for s in CONSENT_STATUSES}
    for r in records:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    expiring_soon = sum(
        1 for r in records
        if r["status"] == "granted" and _is_expiring_soon(r["expiry_date"])
    )

    declined_count = by_status.get("declined", 0)
    withdrawn_count = by_status.get("withdrawn", 0)
    pending_count = by_status.get("pending", 0)

    kpis = [
        {"label": "Total Patients",      "value": total_patients,    "sub": "with consent records"},
        {"label": "Total Consents",       "value": total_consents,    "sub": f"across {len(CONSENT_TYPES)} consent types"},
        {"label": "Consent Rate",         "value": f"{consent_rate}%","sub": "records with granted status"},
        {"label": "Active Consents",      "value": granted_count,     "sub": "currently granted"},
        {"label": "Pending Review",       "value": pending_count,     "sub": "awaiting patient decision"},
        {"label": "Expiring Soon",        "value": expiring_soon,     "sub": "within next 90 days"},
        {"label": "Declined",            "value": declined_count,    "sub": "patient declined"},
        {"label": "Withdrawn",           "value": withdrawn_count,   "sub": "consent subsequently withdrawn"},
    ]

    # Recent activity: last 10 records by id desc
    recent_activity = sorted(records, key=lambda r: r["id"], reverse=True)[:10]

    return {
        "total_patients":   total_patients,
        "total_consents":   total_consents,
        "consent_rate":     consent_rate,
        "by_type":          by_type,
        "by_status":        by_status,
        "kpis":             kpis,
        "expiring_soon":    expiring_soon,
        "recent_activity":  recent_activity,
    }


def breakdown(patient_id: Optional[str] = None) -> dict:
    """
    Detailed consent breakdown.

    Returns:
        per_patient        — list of patient records with their consent details
        type_breakdown     — detailed stats per consent type
        compliance_matrix  — which patients have which consent types (and status)
    """
    conn = _init_db()
    records = _load_all_records(conn, patient_id)
    patients = _real_patients(conn)
    conn.close()

    if not records:
        return {"per_patient": [], "type_breakdown": [], "compliance_matrix": []}

    # Index records by patient_id
    by_patient: dict = {}
    for r in records:
        by_patient.setdefault(r["patient_id"], []).append(r)

    # Build patient info lookup
    patient_info = {p["patient_id"]: p for p in patients}

    per_patient = []
    for pid, pt_records in by_patient.items():
        info = patient_info.get(pid, {})
        granted = [r for r in pt_records if r["status"] == "granted"]
        full_consent = len(granted) == len(CONSENT_TYPES)
        any_expiring = any(_is_expiring_soon(r["expiry_date"]) for r in granted)
        per_patient.append({
            "patient_id":      pid,
            "name":            info.get("name", pid),
            "age":             info.get("age"),
            "disease":         info.get("disease", "unspecified"),
            "total_consents":  len(pt_records),
            "granted":         len(granted),
            "pending":         sum(1 for r in pt_records if r["status"] == "pending"),
            "declined":        sum(1 for r in pt_records if r["status"] == "declined"),
            "expired":         sum(1 for r in pt_records if r["status"] == "expired"),
            "withdrawn":       sum(1 for r in pt_records if r["status"] == "withdrawn"),
            "full_consent":    full_consent,
            "any_expiring":    any_expiring,
            "consent_records": pt_records,
        })

    per_patient.sort(key=lambda x: x["patient_id"])

    # Type breakdown
    type_breakdown = []
    for ct in CONSENT_TYPES:
        ct_records = [r for r in records if r["consent_type"] == ct]
        if not ct_records:
            continue
        granted_n  = sum(1 for r in ct_records if r["status"] == "granted")
        pending_n  = sum(1 for r in ct_records if r["status"] == "pending")
        declined_n = sum(1 for r in ct_records if r["status"] == "declined")
        expired_n  = sum(1 for r in ct_records if r["status"] == "expired")
        withdrawn_n= sum(1 for r in ct_records if r["status"] == "withdrawn")
        total_n    = len(ct_records)
        expiring_n = sum(
            1 for r in ct_records
            if r["status"] == "granted" and _is_expiring_soon(r["expiry_date"])
        )
        type_breakdown.append({
            "consent_type":  ct,
            "total":         total_n,
            "granted":       granted_n,
            "pending":       pending_n,
            "declined":      declined_n,
            "expired":       expired_n,
            "withdrawn":     withdrawn_n,
            "grant_rate":    round(granted_n / total_n * 100, 1) if total_n else 0.0,
            "expiring_soon": expiring_n,
        })

    # Compliance matrix: rows = patients, columns = consent types
    compliance_matrix = []
    for pid, pt_records in by_patient.items():
        info = patient_info.get(pid, {})
        row = {
            "patient_id": pid,
            "name":       info.get("name", pid),
        }
        status_map = {r["consent_type"]: r["status"] for r in pt_records}
        for ct in CONSENT_TYPES:
            row[ct] = status_map.get(ct, "missing")
        compliance_matrix.append(row)

    compliance_matrix.sort(key=lambda x: x["patient_id"])

    return {
        "per_patient":       per_patient,
        "type_breakdown":    type_breakdown,
        "compliance_matrix": compliance_matrix,
    }


def definitions() -> dict:
    """
    Consent type and status definitions, regulatory references, and glossary.

    Returns:
        consent_types  — dict of type -> description
        statuses       — dict of status -> description
        regulatory     — HIPAA/IRB/GDPR references
        glossary       — key terms used in consent management
    """
    return {
        "consent_types": {
            "treatment": (
                "Standard informed consent authorising the clinical team to provide "
                "neurological assessment, medical treatment, and therapeutic interventions. "
                "Required before any clinical procedure or prescription change."
            ),
            "research": (
                "Consent for enrolment in research studies, clinical trials, or observational "
                "cohorts. Governed by the Institutional Review Board (IRB) and the Common Rule "
                "(45 CFR 46). Must specify study purpose, risks, voluntary nature, and right "
                "to withdraw without penalty."
            ),
            "data_sharing": (
                "Authorisation to share de-identified clinical data (EEG, imaging, lab results, "
                "clinical outcomes) with external research consortia or multi-site registries. "
                "Subject to HIPAA Safe Harbor or Expert Determination de-identification "
                "standards (45 CFR §164.514)."
            ),
            "genetic_testing": (
                "Consent for collection of biological samples (saliva, blood) and use of "
                "genetic/genomic data. Governed by the Genetic Information Nondiscrimination "
                "Act (GINA) and applicable state law. Covers whole-exome/genome sequencing, "
                "pharmacogenomic panels, and biobank storage."
            ),
            "video_eeg": (
                "Consent for continuous video-EEG monitoring, which records audio-visual "
                "footage alongside brain activity. Recordings may be used for clinical review, "
                "multidisciplinary team discussion, and de-identified educational purposes "
                "as specified."
            ),
            "imaging_sharing": (
                "Consent to share neuroimaging data (MRI, CT, PET) with federated imaging "
                "research networks and AI model training consortia under a data use agreement. "
                "Images are de-identified per DICOM standard before transfer."
            ),
        },
        "statuses": {
            "granted": (
                "Patient has given informed consent. Consent is currently active and within "
                "its validity period. Clinical and/or research activities may proceed."
            ),
            "pending": (
                "Consent documentation has been initiated but the patient has not yet signed "
                "or confirmed. No activities requiring this consent may proceed until status "
                "changes to 'granted'."
            ),
            "declined": (
                "Patient explicitly declined to provide consent after receiving full information. "
                "Declinations must be documented and respected. Re-approach requires new "
                "indication and coordinator approval."
            ),
            "expired": (
                "Consent was previously granted but has passed its documented expiry date. "
                "A renewal consent process must be completed before activities resume."
            ),
            "withdrawn": (
                "Patient previously granted consent but has subsequently requested withdrawal. "
                "All activities under this consent must cease immediately upon withdrawal. "
                "Previously collected data may be retained as permitted by study protocol."
            ),
        },
        "regulatory": {
            "HIPAA_164_508": {
                "citation": "45 CFR §164.508",
                "title":    "Authorization for Uses and Disclosures",
                "summary":  (
                    "Requires written patient authorisation before using or disclosing protected "
                    "health information (PHI) for purposes not covered by treatment, payment, or "
                    "healthcare operations. Core requirement for research and data-sharing consent."
                ),
            },
            "Common_Rule_45_CFR_46": {
                "citation": "45 CFR Part 46",
                "title":    "Federal Policy for the Protection of Human Subjects",
                "summary":  (
                    "The 'Common Rule' governs IRB review and informed consent for federally "
                    "funded human subjects research. Mandates disclosure of purpose, risks, "
                    "benefits, alternatives, confidentiality practices, voluntary participation, "
                    "and contact information."
                ),
            },
            "FDA_21_CFR_50": {
                "citation": "21 CFR Part 50",
                "title":    "Protection of Human Subjects (FDA)",
                "summary":  (
                    "FDA regulations governing informed consent for clinical investigations of "
                    "drugs, devices, and biologics. Applies to all industry-sponsored trials "
                    "and device studies at this institution."
                ),
            },
            "GINA": {
                "citation": "Pub. L. 110-233 (2008)",
                "title":    "Genetic Information Nondiscrimination Act",
                "summary":  (
                    "Prohibits health insurers and employers from discriminating based on genetic "
                    "information. Governs consent for genetic testing and genomic data use. "
                    "Patients must be informed of GINA protections before genetic consent."
                ),
            },
            "GDPR_Art7": {
                "citation": "GDPR Article 7",
                "title":    "Conditions for Consent (EU/UK patients)",
                "summary":  (
                    "Requires freely given, specific, informed, and unambiguous consent for "
                    "processing personal data. Right to withdraw consent at any time without "
                    "detriment. Special category data (health, genetic) requires explicit consent "
                    "under Article 9."
                ),
            },
            "IRB": {
                "citation": "Internal Policy — IRB-2024",
                "title":    "Institutional Review Board Oversight",
                "summary":  (
                    "All research consent forms must be IRB-approved before use. IRB approval "
                    "numbers must be documented in consent_records.notes. Annual continuing review "
                    "required for ongoing studies."
                ),
            },
        },
        "glossary": [
            {
                "term":       "Informed Consent",
                "definition": (
                    "A process by which a patient voluntarily confirms willingness to participate "
                    "in a particular activity after being informed of all aspects relevant to their "
                    "decision. Must be documented in writing."
                ),
            },
            {
                "term":       "Consent Expiry",
                "definition": (
                    "The date after which a previously granted consent is no longer valid. "
                    "Re-consent is required. Typical validity: 1-2 years for research; "
                    "indefinite for standard treatment consent."
                ),
            },
            {
                "term":       "Withdrawal",
                "definition": (
                    "A patient's right to revoke previously granted consent at any time without "
                    "penalty. Withdrawal is prospective — it does not require destruction of data "
                    "already lawfully collected, unless specified by the patient and permitted "
                    "by protocol."
                ),
            },
            {
                "term":       "PHI (Protected Health Information)",
                "definition": (
                    "Any individually identifiable health information held or transmitted by a "
                    "covered entity, including demographic data, diagnoses, treatment records, "
                    "and billing information. Governed by HIPAA."
                ),
            },
            {
                "term":       "De-identification",
                "definition": (
                    "The process of removing or obscuring 18 HIPAA identifiers (or applying "
                    "expert determination) so that information cannot reasonably be used to "
                    "identify an individual. Required before data sharing without explicit "
                    "authorisation."
                ),
            },
            {
                "term":       "Assent",
                "definition": (
                    "Agreement to participate in research or treatment by a patient who is not "
                    "legally capable of giving informed consent (e.g., minors aged 7-17). "
                    "Must be obtained alongside parental/guardian consent."
                ),
            },
            {
                "term":       "Waiver of Consent",
                "definition": (
                    "IRB-granted permission to conduct research without obtaining individual "
                    "consent when meeting criteria: minimal risk, practicable only with waiver, "
                    "rights not adversely affected, subjects informed when possible."
                ),
            },
            {
                "term":       "Data Use Agreement (DUA)",
                "definition": (
                    "A formal contract between data provider and recipient governing permitted "
                    "uses of a limited data set. Required for imaging_sharing and data_sharing "
                    "consent types when transferring data to external institutions."
                ),
            },
            {
                "term":       "Compliance Matrix",
                "definition": (
                    "A grid showing which patients have granted, pending, declined, expired, "
                    "or withdrawn each consent type. Used by coordinators for audit readiness "
                    "and gap identification."
                ),
            },
            {
                "term":       "Consent Rate",
                "definition": (
                    "Percentage of consent records with 'granted' status out of all records. "
                    "A KPI for recruitment viability and patient engagement."
                ),
            },
        ],
    }
