"""Referral Triage Dashboard — referral intake & triage analytics from clinical.db.

Tracks patient referrals, urgency classification, triage workflow, provider
assignment, and turnaround times.

Sources:
- referral_records table (id, patient_id, referral_source, referral_reason,
  urgency, triage_status, triage_score, assigned_to, referral_date, triage_date, notes)
"""

import sqlite3
from pathlib import Path
from collections import defaultdict

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


# ──────────────────────────────────────────────────────────────
#  /api/referral-triage/overview
# ──────────────────────────────────────────────────────────────

def overview():
    """High-level referral and triage KPIs, distributions, timeline."""
    conn = _conn()
    cur = conn.cursor()

    # KPIs
    cur.execute("SELECT COUNT(*) FROM referral_records")
    total_referrals = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM referral_records WHERE triage_status = 'pending_triage'")
    pending_triage = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM referral_records WHERE urgency IN ('urgent', 'emergent') AND triage_status = 'pending_triage'")
    urgent_pending = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM referral_records WHERE triage_status = 'completed'")
    completed = cur.fetchone()[0] or 0
    completion_rate_pct = round(completed / total_referrals * 100, 1) if total_referrals else 0.0

    # Average triage time (hours) — difference between triage_date and referral_date
    cur.execute("""
        SELECT AVG(
            (julianday(triage_date) - julianday(referral_date)) * 24
        ) FROM referral_records
        WHERE triage_date IS NOT NULL AND referral_date IS NOT NULL
    """)
    avg_triage_time_hrs = round(cur.fetchone()[0] or 0, 1)

    # Urgency distribution
    cur.execute("SELECT urgency, COUNT(*) FROM referral_records GROUP BY urgency")
    urgency_distribution = dict(cur.fetchall())

    # Source distribution
    cur.execute("SELECT referral_source, COUNT(*) FROM referral_records GROUP BY referral_source ORDER BY COUNT(*) DESC")
    source_distribution = dict(cur.fetchall())

    # Triage timeline — monthly
    cur.execute("""
        SELECT strftime('%Y-%m', referral_date) AS month, COUNT(*) AS referrals
        FROM referral_records
        WHERE referral_date IS NOT NULL
        GROUP BY month ORDER BY month
    """)
    monthly_referrals = dict(cur.fetchall())

    cur.execute("""
        SELECT strftime('%Y-%m', triage_date) AS month, COUNT(*) AS triaged
        FROM referral_records
        WHERE triage_date IS NOT NULL
        GROUP BY month ORDER BY month
    """)
    monthly_triaged = dict(cur.fetchall())

    all_months = sorted(set(list(monthly_referrals.keys()) + list(monthly_triaged.keys())))
    triage_timeline = [
        {"date": m, "referrals": monthly_referrals.get(m, 0), "triaged": monthly_triaged.get(m, 0)}
        for m in all_months
    ]

    conn.close()
    return {
        "total_referrals": total_referrals,
        "pending_triage": pending_triage,
        "avg_triage_time_hrs": avg_triage_time_hrs,
        "urgent_pending": urgent_pending,
        "completion_rate_pct": completion_rate_pct,
        "urgency_distribution": urgency_distribution,
        "source_distribution": source_distribution,
        "triage_timeline": triage_timeline,
    }


# ──────────────────────────────────────────────────────────────
#  /api/referral-triage/breakdown
# ──────────────────────────────────────────────────────────────

def breakdown():
    """Detailed referral breakdown — reasons, urgency-by-source, provider workload, recent list."""
    conn = _conn()
    cur = conn.cursor()

    # Referral KPIs for the Referrals tab
    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM referral_records")
    distinct_patients = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(DISTINCT referral_source) FROM referral_records")
    distinct_sources = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(DISTINCT referral_reason) FROM referral_records")
    distinct_reasons = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(DISTINCT assigned_to) FROM referral_records WHERE assigned_to IS NOT NULL")
    distinct_providers = cur.fetchone()[0] or 0

    referral_kpis = [
        {"label": "Patients Referred", "value": distinct_patients, "color": "#3b82f6"},
        {"label": "Referral Sources", "value": distinct_sources, "color": "#8b5cf6"},
        {"label": "Referral Reasons", "value": distinct_reasons, "color": "#06b6d4"},
        {"label": "Active Providers", "value": distinct_providers, "color": "#22c55e"},
    ]

    # Reason distribution
    cur.execute("SELECT referral_reason, COUNT(*) FROM referral_records GROUP BY referral_reason ORDER BY COUNT(*) DESC")
    reason_distribution = dict(cur.fetchall())

    # Urgency by source (for stacked bar)
    cur.execute("SELECT referral_source, urgency, COUNT(*) FROM referral_records GROUP BY referral_source, urgency")
    urgency_by_source = defaultdict(dict)
    for source, urgency, count in cur.fetchall():
        urgency_by_source[source][urgency] = count
    urgency_by_source = dict(urgency_by_source)

    # Status summary
    cur.execute("SELECT triage_status, COUNT(*) FROM referral_records GROUP BY triage_status")
    status_summary = dict(cur.fetchall())

    # Provider workload
    cur.execute("""
        SELECT assigned_to, COUNT(*) AS total,
               SUM(CASE WHEN triage_status = 'completed' THEN 1 ELSE 0 END) AS completed,
               SUM(CASE WHEN triage_status IN ('pending_triage', 'in_progress') THEN 1 ELSE 0 END) AS active
        FROM referral_records
        WHERE assigned_to IS NOT NULL
        GROUP BY assigned_to ORDER BY total DESC
    """)
    provider_workload = _dict_rows(cur)

    # Recent referrals (last 30)
    cur.execute("""
        SELECT patient_id, referral_source AS source, referral_reason AS reason,
               urgency, triage_status AS status, assigned_to, referral_date AS referred_date,
               triage_date AS triaged_date, triage_score, notes
        FROM referral_records
        ORDER BY referral_date DESC LIMIT 30
    """)
    recent_referrals = _dict_rows(cur)

    # Analytics: avg triage score by urgency
    cur.execute("""
        SELECT urgency, ROUND(AVG(triage_score), 1) AS avg_score, COUNT(*) AS count
        FROM referral_records
        GROUP BY urgency ORDER BY avg_score DESC
    """)
    score_by_urgency = _dict_rows(cur)

    # Analytics: avg triage score by source
    cur.execute("""
        SELECT referral_source AS source, ROUND(AVG(triage_score), 1) AS avg_score, COUNT(*) AS count
        FROM referral_records
        GROUP BY referral_source ORDER BY avg_score DESC
    """)
    score_by_source = _dict_rows(cur)

    # Analytics: turnaround by urgency
    cur.execute("""
        SELECT urgency,
               ROUND(AVG((julianday(triage_date) - julianday(referral_date)) * 24), 1) AS avg_hours,
               COUNT(*) AS count
        FROM referral_records
        WHERE triage_date IS NOT NULL AND referral_date IS NOT NULL
        GROUP BY urgency ORDER BY avg_hours
    """)
    turnaround_by_urgency = _dict_rows(cur)

    conn.close()
    return {
        "referral_kpis": referral_kpis,
        "reason_distribution": reason_distribution,
        "urgency_by_source": urgency_by_source,
        "status_summary": status_summary,
        "provider_workload": provider_workload,
        "recent_referrals": recent_referrals,
        "score_by_urgency": score_by_urgency,
        "score_by_source": score_by_source,
        "turnaround_by_urgency": turnaround_by_urgency,
    }


# ──────────────────────────────────────────────────────────────
#  /api/referral-triage/definitions
# ──────────────────────────────────────────────────────────────

def definitions():
    """Definitions and glossary for referral triage domain."""
    return {
        "urgency_levels": [
            {"level": "Emergent", "description": "Life-threatening or acute seizure emergency requiring immediate attention (< 1 hour)"},
            {"level": "Urgent", "description": "Significant clinical concern requiring attention within 24-48 hours (new onset seizures, status epilepticus history)"},
            {"level": "Routine", "description": "Standard referral for evaluation within 1-2 weeks (medication review, follow-up EEG)"},
            {"level": "Elective", "description": "Non-urgent consultation, can be scheduled within 4-6 weeks (second opinion, routine monitoring)"},
        ],
        "triage_statuses": [
            {"status": "Pending Triage", "description": "Referral received but not yet reviewed by triage coordinator"},
            {"status": "Triaged", "description": "Reviewed and scored; awaiting provider assignment or scheduling"},
            {"status": "Scheduled", "description": "Appointment scheduled with assigned provider"},
            {"status": "In Progress", "description": "Patient currently being evaluated or treated"},
            {"status": "Completed", "description": "Evaluation and initial management plan completed"},
            {"status": "Cancelled", "description": "Referral cancelled (patient declined, resolved, or redirected)"},
        ],
        "referral_sources": [
            {"source": "Primary Care", "description": "General practitioner or family medicine referral"},
            {"source": "Emergency", "description": "Emergency department referral following acute presentation"},
            {"source": "Neurology Clinic", "description": "Inter-specialty referral from neurology"},
            {"source": "Pediatrics", "description": "Pediatric referral for childhood-onset epilepsy"},
            {"source": "Psychiatry", "description": "Psychiatry referral for PNES evaluation or comorbid conditions"},
            {"source": "Other Specialist", "description": "Referral from other medical specialties"},
            {"source": "Self-Referral", "description": "Patient-initiated referral or walk-in"},
        ],
        "referral_reasons": [
            {"reason": "New Onset Seizure", "description": "First unprovoked seizure requiring workup (EEG, MRI, labs)"},
            {"reason": "Refractory Epilepsy", "description": "Drug-resistant epilepsy failing >= 2 appropriately chosen AEDs"},
            {"reason": "Status Epilepticus", "description": "Prolonged or repeated seizures without recovery — emergency evaluation"},
            {"reason": "Medication Review", "description": "AED side effects, drug interactions, or dose optimization"},
            {"reason": "Pre-Surgical Evaluation", "description": "Candidate for epilepsy surgery workup (video-EEG, fMRI, Wada)"},
            {"reason": "Psychogenic Nonepileptic", "description": "Suspected PNES requiring video-EEG and psychiatric evaluation"},
            {"reason": "EEG Abnormality", "description": "Incidental EEG finding requiring specialist interpretation"},
            {"reason": "Cognitive Decline", "description": "Progressive cognitive issues in epilepsy patient"},
            {"reason": "Headache Workup", "description": "Evaluation for seizure vs migraine differential"},
        ],
        "field_descriptions": [
            {"field": "Triage Score", "description": "Numeric priority score (0-100) calculated from urgency, reason, and clinical factors. Higher = more urgent."},
            {"field": "Turnaround Time", "description": "Hours from referral receipt to triage completion (provider assignment + scheduling)"},
            {"field": "Completion Rate", "description": "Percentage of referrals reaching 'completed' status out of total received"},
        ],
        "clinical_notes": [
            "Emergent referrals should be triaged within 1 hour of receipt",
            "All urgent referrals require provider contact within 24 hours",
            "Triage scores above 70 indicate high clinical priority",
            "PNES referrals require both neurology and psychiatry coordination",
            "Refractory epilepsy referrals should be routed to comprehensive epilepsy centers",
        ],
        "glossary": [
            {"term": "AED", "definition": "Anti-Epileptic Drug — medication used to control seizures"},
            {"term": "PNES", "definition": "Psychogenic Non-Epileptic Seizures — events resembling seizures without epileptic EEG correlate"},
            {"term": "Video-EEG", "definition": "Simultaneous video and EEG recording for seizure classification"},
            {"term": "Triage", "definition": "Process of determining priority and routing of patient referrals"},
            {"term": "Refractory", "definition": "Drug-resistant; not adequately controlled by standard treatments"},
            {"term": "Status Epilepticus", "definition": "Seizure lasting >5 minutes or repeated seizures without recovery"},
            {"term": "Wada Test", "definition": "Intracarotid amobarbital test for pre-surgical language/memory lateralization"},
            {"term": "fMRI", "definition": "Functional MRI — brain mapping for pre-surgical planning"},
            {"term": "Epileptologist", "definition": "Neurologist subspecialized in epilepsy diagnosis and management"},
            {"term": "Comorbidity", "definition": "Co-existing medical condition alongside epilepsy"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2, default=str))
    print("\n=== BREAKDOWN ===")
    print(json.dumps(breakdown(), indent=2, default=str))
    print("\n=== DEFINITIONS ===")
    print(json.dumps(definitions(), indent=2, default=str))
