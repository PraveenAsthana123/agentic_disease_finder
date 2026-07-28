"""
Neuro AI Ecosystem — Emergency Contacts Dashboard
===================================================
Emergency contact registry analytics from emergency_contacts table.

Relationship types: spouse, parent, sibling, child, partner, friend,
    neighbor, grandparent, professional_caregiver
Flags: is_primary, notify_on_seizure
Verification freshness tracked via last_verified date.

Real data: emergency_contacts (30 rows, 30 patients) in clinical.db.
"""

import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def overview():
    """Emergency contacts overview — total contacts, relationship distribution,
    seizure notification rate, verification freshness, coverage metrics."""
    conn = _conn()
    cur = conn.cursor()

    # Total contacts
    cur.execute("SELECT COUNT(*) FROM emergency_contacts")
    total_contacts = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM emergency_contacts")
    patients_with_contacts = cur.fetchone()[0]

    # Primary contact rate
    cur.execute("""
        SELECT ROUND(AVG(CASE WHEN is_primary = 1 THEN 1.0 ELSE 0.0 END) * 100, 1)
        FROM emergency_contacts
    """)
    primary_pct = cur.fetchone()[0]

    # Seizure notification rate
    cur.execute("""
        SELECT ROUND(AVG(CASE WHEN notify_on_seizure = 1 THEN 1.0 ELSE 0.0 END) * 100, 1)
        FROM emergency_contacts
    """)
    seizure_notify_pct = cur.fetchone()[0]

    # Relationship distribution
    cur.execute("""
        SELECT relationship, COUNT(*) cnt
        FROM emergency_contacts
        GROUP BY relationship
        ORDER BY cnt DESC
    """)
    relationship_dist = _dict_rows(cur)

    # Notification by relationship
    cur.execute("""
        SELECT relationship,
               COUNT(*) total,
               SUM(CASE WHEN notify_on_seizure = 1 THEN 1 ELSE 0 END) notify_enabled
        FROM emergency_contacts
        GROUP BY relationship
        ORDER BY total DESC
    """)
    notify_by_relationship = _dict_rows(cur)

    # Verification freshness
    cur.execute("""
        SELECT CASE
                 WHEN JULIANDAY('now') - JULIANDAY(last_verified) <= 90 THEN 'current'
                 WHEN JULIANDAY('now') - JULIANDAY(last_verified) <= 180 THEN 'due-soon'
                 ELSE 'overdue'
               END AS freshness,
               COUNT(*) cnt
        FROM emergency_contacts
        GROUP BY freshness
        ORDER BY cnt DESC
    """)
    freshness_dist = _dict_rows(cur)

    # Average days since verification
    cur.execute("""
        SELECT ROUND(AVG(JULIANDAY('now') - JULIANDAY(last_verified)), 0)
        FROM emergency_contacts
    """)
    avg_days_since_verified = cur.fetchone()[0]

    # Contacts per patient distribution
    cur.execute("""
        SELECT patient_id, COUNT(*) cnt
        FROM emergency_contacts
        GROUP BY patient_id
    """)
    contacts_per_patient = _dict_rows(cur)
    count_dist = {}
    for row in contacts_per_patient:
        c = str(row["cnt"])
        count_dist[c] = count_dist.get(c, 0) + 1

    # Monthly creation trend
    cur.execute("""
        SELECT SUBSTR(created_at, 1, 7) month, COUNT(*) cnt
        FROM emergency_contacts
        GROUP BY month
        ORDER BY month
    """)
    monthly_trend = _dict_rows(cur)

    conn.close()
    return {
        "kpis": {
            "total_contacts": total_contacts,
            "patients_covered": patients_with_contacts,
            "primary_pct": primary_pct,
            "seizure_notify_pct": seizure_notify_pct,
            "avg_days_since_verified": avg_days_since_verified,
        },
        "relationship_distribution": relationship_dist,
        "notify_by_relationship": notify_by_relationship,
        "verification_freshness": freshness_dist,
        "contacts_per_patient_dist": count_dist,
        "monthly_trend": monthly_trend,
    }


def breakdown():
    """Emergency contacts breakdown — all contacts table, by relationship,
    by patient, stale contacts needing re-verification."""
    conn = _conn()
    cur = conn.cursor()

    # All contacts
    cur.execute("""
        SELECT id, patient_id, contact_name, phone, email, relationship,
               is_primary, notify_on_seizure, last_verified,
               CAST(JULIANDAY('now') - JULIANDAY(last_verified) AS INTEGER) days_since_verified,
               created_at
        FROM emergency_contacts
        ORDER BY patient_id
    """)
    all_contacts = _dict_rows(cur)

    # By relationship summary
    cur.execute("""
        SELECT relationship,
               COUNT(*) total,
               SUM(CASE WHEN is_primary = 1 THEN 1 ELSE 0 END) primary_count,
               SUM(CASE WHEN notify_on_seizure = 1 THEN 1 ELSE 0 END) notify_count,
               ROUND(AVG(JULIANDAY('now') - JULIANDAY(last_verified)), 0) avg_days_since_verified
        FROM emergency_contacts
        GROUP BY relationship
        ORDER BY total DESC
    """)
    by_relationship = _dict_rows(cur)

    # By patient
    cur.execute("""
        SELECT patient_id,
               COUNT(*) contact_count,
               GROUP_CONCAT(relationship, ', ') relationships,
               SUM(CASE WHEN notify_on_seizure = 1 THEN 1 ELSE 0 END) notify_enabled,
               MIN(CAST(JULIANDAY('now') - JULIANDAY(last_verified) AS INTEGER)) freshest_days,
               MAX(CAST(JULIANDAY('now') - JULIANDAY(last_verified) AS INTEGER)) stalest_days
        FROM emergency_contacts
        GROUP BY patient_id
        ORDER BY patient_id
    """)
    by_patient = _dict_rows(cur)

    # Stale contacts (verified > 6 months ago)
    cur.execute("""
        SELECT patient_id, contact_name, relationship, phone, email,
               last_verified,
               CAST(JULIANDAY('now') - JULIANDAY(last_verified) AS INTEGER) days_since_verified
        FROM emergency_contacts
        WHERE JULIANDAY('now') - JULIANDAY(last_verified) > 180
        ORDER BY days_since_verified DESC
    """)
    stale_contacts = _dict_rows(cur)

    conn.close()
    return {
        "all_contacts": all_contacts,
        "by_relationship": by_relationship,
        "by_patient": by_patient,
        "stale_contacts": stale_contacts,
    }


def definitions():
    """Emergency contacts definitions — field glossary, relationship types,
    verification standards, and emergency preparedness references."""
    return {
        "glossary": [
            {"term": "Emergency Contact", "definition": "Designated person to be notified in case of a seizure, fall, or other medical emergency"},
            {"term": "Primary Contact", "definition": "The first person called during an emergency; each patient should have exactly one primary contact"},
            {"term": "Seizure Notification", "definition": "Automatic alert sent to contact when a seizure event is detected by wearable or reported by patient"},
            {"term": "Contact Verification", "definition": "Periodic confirmation that phone number, email, and relationship details are current and reachable"},
            {"term": "Verification Freshness", "definition": "Time since last contact verification — current (<=90 days), due-soon (91-180 days), overdue (>180 days)"},
            {"term": "Seizure Action Plan", "definition": "Written instructions provided to emergency contacts on what to do during and after a seizure"},
            {"term": "SUDEP Prevention", "definition": "Sudden Unexpected Death in Epilepsy — primary motivation for maintaining reliable emergency contact coverage"},
            {"term": "Caregiver Readiness", "definition": "Assessment of whether emergency contacts have received seizure first-aid training"},
        ],
        "relationship_types": [
            {"type": "spouse", "description": "Married or domestic partner living with the patient"},
            {"type": "parent", "description": "Biological or adoptive parent of the patient"},
            {"type": "sibling", "description": "Brother or sister of the patient"},
            {"type": "child", "description": "Son or daughter of the patient (adult child for adult patients)"},
            {"type": "partner", "description": "Unmarried romantic partner"},
            {"type": "friend", "description": "Close friend designated as emergency contact"},
            {"type": "neighbor", "description": "Nearby resident available for rapid in-person response"},
            {"type": "grandparent", "description": "Grandparent of the patient"},
            {"type": "professional_caregiver", "description": "Paid caregiver, nurse, or aide assigned to the patient"},
        ],
        "verification_standards": [
            {"standard": "Verification Interval", "target": "Every 90 days for primary contacts, 180 days for secondary"},
            {"standard": "Contact Methods", "target": "Both phone and email must be verified and reachable"},
            {"standard": "Coverage", "target": "100% of epilepsy patients must have at least 1 verified emergency contact"},
            {"standard": "Seizure Training", "target": "Primary contacts should complete seizure first-aid training annually"},
        ],
    }
