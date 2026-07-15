"""Consent Dashboard — consent record analytics from clinical.db.

Tracks patient consent lifecycle: type distribution, status breakdown,
compliance rates, expiration tracking, witness coverage, and monthly volume.

Sources:
- consent_records table (patient_id, consent_type, status, dates, witness, notes)
"""

import sqlite3
import json
import datetime
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def overview():
    """Return high-level consent metrics: totals, distributions, compliance, expiry."""
    conn = _conn()
    cur = conn.cursor()
    today = datetime.date.today()
    ninety_days = today + datetime.timedelta(days=90)
    today_str = today.isoformat()
    ninety_str = ninety_days.isoformat()

    # total records & patients
    cur.execute("SELECT COUNT(*) FROM consent_records")
    total_records = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM consent_records")
    total_patients = cur.fetchone()[0]

    # consent_type distribution
    cur.execute(
        "SELECT consent_type, COUNT(*) FROM consent_records GROUP BY consent_type ORDER BY COUNT(*) DESC"
    )
    consent_type_distribution = dict(cur.fetchall())

    # status distribution
    cur.execute(
        "SELECT status, COUNT(*) FROM consent_records GROUP BY status ORDER BY COUNT(*) DESC"
    )
    status_distribution = dict(cur.fetchall())

    # compliance rate
    granted_count = status_distribution.get("granted", 0)
    compliance_rate = round((granted_count / total_records) * 100, 1) if total_records else 0.0

    # expiring soon (within 90 days from today)
    cur.execute(
        "SELECT COUNT(*) FROM consent_records "
        "WHERE expiry_date IS NOT NULL AND expiry_date >= ? AND expiry_date <= ? AND status != 'expired'",
        (today_str, ninety_str),
    )
    expiring_soon = cur.fetchone()[0]

    # expired count
    expired_count = status_distribution.get("expired", 0)

    # witness distribution
    cur.execute(
        "SELECT witness, COUNT(*) FROM consent_records GROUP BY witness ORDER BY COUNT(*) DESC"
    )
    witness_distribution = {}
    for witness, count in cur.fetchall():
        key = witness if witness else "Unwitnessed"
        witness_distribution[key] = count

    # monthly volume (last 12 months by granted_date)
    twelve_months_ago = (today.replace(day=1) - datetime.timedelta(days=365)).replace(day=1)
    cur.execute(
        "SELECT strftime('%Y-%m', granted_date) AS month, COUNT(*) FROM consent_records "
        "WHERE granted_date IS NOT NULL AND granted_date >= ? "
        "GROUP BY month ORDER BY month",
        (twelve_months_ago.isoformat(),),
    )
    monthly_volume = dict(cur.fetchall())

    # type-status matrix
    cur.execute(
        "SELECT consent_type, status, COUNT(*) FROM consent_records GROUP BY consent_type, status"
    )
    matrix_raw = cur.fetchall()
    matrix_map = {}
    for ctype, status, count in matrix_raw:
        if ctype not in matrix_map:
            matrix_map[ctype] = {
                "consent_type": ctype,
                "granted": 0,
                "pending": 0,
                "withdrawn": 0,
                "declined": 0,
                "expired": 0,
            }
        matrix_map[ctype][status] = count
    type_status_matrix = list(matrix_map.values())

    conn.close()

    return {
        "total_records": total_records,
        "total_patients": total_patients,
        "consent_type_distribution": consent_type_distribution,
        "status_distribution": status_distribution,
        "compliance_rate": compliance_rate,
        "expiring_soon": expiring_soon,
        "expired_count": expired_count,
        "witness_distribution": witness_distribution,
        "monthly_volume": monthly_volume,
        "type_status_matrix": type_status_matrix,
    }


def breakdown():
    """Return per-patient, recent, expiring, withdrawn, and per-type detail."""
    conn = _conn()
    cur = conn.cursor()
    today = datetime.date.today()
    ninety_days = today + datetime.timedelta(days=90)
    today_str = today.isoformat()
    ninety_str = ninety_days.isoformat()

    # per_patient
    cur.execute(
        """
        SELECT
            patient_id,
            COUNT(*) AS total,
            SUM(CASE WHEN status='granted' THEN 1 ELSE 0 END) AS granted,
            SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending,
            SUM(CASE WHEN status='withdrawn' THEN 1 ELSE 0 END) AS withdrawn,
            SUM(CASE WHEN status='declined' THEN 1 ELSE 0 END) AS declined,
            SUM(CASE WHEN status='expired' THEN 1 ELSE 0 END) AS expired,
            MAX(granted_date) AS latest_date
        FROM consent_records
        GROUP BY patient_id
        ORDER BY total DESC
        """
    )
    per_patient = _dict_rows(cur)

    # recent_consents (last 20 by granted_date DESC)
    cur.execute(
        """
        SELECT * FROM consent_records
        WHERE granted_date IS NOT NULL
        ORDER BY granted_date DESC
        LIMIT 20
        """
    )
    recent_consents = _dict_rows(cur)

    # expiring_soon_list
    cur.execute(
        """
        SELECT * FROM consent_records
        WHERE expiry_date IS NOT NULL
          AND expiry_date >= ?
          AND expiry_date <= ?
          AND status != 'expired'
        ORDER BY expiry_date ASC
        """,
        (today_str, ninety_str),
    )
    expiring_soon_list = _dict_rows(cur)

    # withdrawn_list
    cur.execute(
        "SELECT * FROM consent_records WHERE status='withdrawn' ORDER BY granted_date DESC"
    )
    withdrawn_list = _dict_rows(cur)

    # type_detail
    cur.execute(
        """
        SELECT
            consent_type AS type,
            COUNT(*) AS total,
            ROUND(SUM(CASE WHEN status='granted' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) AS granted_pct,
            SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending_count,
            SUM(CASE WHEN status='withdrawn' THEN 1 ELSE 0 END) AS withdrawn_count,
            ROUND(AVG(
                CASE WHEN expiry_date IS NOT NULL AND granted_date IS NOT NULL
                     THEN julianday(expiry_date) - julianday(granted_date)
                     ELSE NULL
                END
            ), 1) AS avg_validity_days
        FROM consent_records
        GROUP BY consent_type
        ORDER BY total DESC
        """
    )
    type_detail = _dict_rows(cur)

    conn.close()

    return {
        "per_patient": per_patient,
        "recent_consents": recent_consents,
        "expiring_soon_list": expiring_soon_list,
        "withdrawn_list": withdrawn_list,
        "type_detail": type_detail,
    }


def definitions():
    """Return consent type descriptions, status definitions, glossary, and compliance notes."""
    return {
        "consent_types": {
            "treatment": "Authorization for medical treatment procedures including medication administration, surgical interventions, and therapeutic protocols.",
            "research": "Informed consent for participation in clinical research studies, trials, or observational research protocols.",
            "data_sharing": "Permission to share patient health data with external entities such as research institutions, registries, or partner healthcare organizations.",
            "genetic_testing": "Consent for genomic or genetic testing, including whole-genome sequencing, targeted panels, and pharmacogenomic analysis.",
            "video_eeg": "Authorization for continuous video-EEG monitoring, including recording, storage, and clinical use of audiovisual data.",
            "imaging_sharing": "Permission to share diagnostic imaging (MRI, CT, PET) with external specialists, tumor boards, or research databases.",
        },
        "statuses": {
            "granted": "Patient has provided informed consent; the authorization is active and valid.",
            "pending": "Consent form has been presented but the patient has not yet signed or provided a decision.",
            "withdrawn": "Patient has revoked previously granted consent; all related activities must cease.",
            "declined": "Patient has reviewed the consent form and actively refused to provide authorization.",
            "expired": "Consent was previously granted but has passed its expiry date and is no longer valid.",
        },
        "glossary": [
            {"term": "Informed Consent", "definition": "The process of providing a patient with sufficient information about risks, benefits, and alternatives to make a voluntary decision about treatment or research participation."},
            {"term": "Capacity", "definition": "A patient's cognitive and legal ability to understand information, appreciate consequences, and make an autonomous decision about consent."},
            {"term": "Assent", "definition": "Agreement from a minor or individual who lacks full decision-making capacity, obtained alongside consent from a legal guardian."},
            {"term": "Surrogate Decision-Maker", "definition": "A legally authorized individual who provides consent on behalf of a patient who lacks decision-making capacity."},
            {"term": "IRB (Institutional Review Board)", "definition": "An ethics committee that reviews and approves research involving human subjects to ensure participant protection and regulatory compliance."},
            {"term": "Waiver of Consent", "definition": "An IRB-approved exemption from the standard consent requirement, typically for minimal-risk research or emergency situations."},
            {"term": "Re-consent", "definition": "The process of obtaining updated consent when study protocols change, new risks emerge, or the original consent period expires."},
            {"term": "Consent Validity Period", "definition": "The time window during which a signed consent remains legally and clinically valid before requiring renewal."},
            {"term": "Witness Attestation", "definition": "A third-party signature confirming that the consent process was conducted properly and the patient signed voluntarily."},
            {"term": "Right to Withdraw", "definition": "A patient's right to revoke consent at any time without penalty or impact on their standard clinical care."},
            {"term": "HIPAA Authorization", "definition": "A specific form of consent required under HIPAA for the use or disclosure of protected health information beyond treatment, payment, or operations."},
            {"term": "Therapeutic Misconception", "definition": "A participant's mistaken belief that research procedures are designed primarily for their personal therapeutic benefit rather than to generate scientific knowledge."},
        ],
        "compliance_notes": [
            "All consent processes must comply with 45 CFR 46 (Common Rule) requirements for federally funded research.",
            "HIPAA-covered entities require separate authorization for use/disclosure of PHI beyond treatment, payment, and healthcare operations.",
            "IRB approval is mandatory before enrolling any participant in a research protocol; consent documents must use IRB-approved language.",
            "Withdrawn consent must be honored immediately; ongoing data collection and specimen retention must cease unless a waiver applies.",
            "Expired consents require re-consent before resuming any activities covered by the original authorization.",
            "Electronic consent (e-consent) is permissible under 21 CFR Part 11 provided audit trails, authentication, and tamper-evidence are maintained.",
            "Consent documents must be provided in the patient's preferred language and at an appropriate literacy level (recommended: 6th-8th grade reading level).",
            "Pediatric research requires both parental/guardian consent and, where appropriate, child assent (typically age 7+).",
        ],
    }


if __name__ == "__main__":
    print("=== Consent Dashboard ===\n")
    print("--- Overview ---")
    print(json.dumps(overview(), indent=2, default=str))
    print("\n--- Breakdown ---")
    print(json.dumps(breakdown(), indent=2, default=str))
    print("\n--- Definitions ---")
    print(json.dumps(definitions(), indent=2, default=str))
