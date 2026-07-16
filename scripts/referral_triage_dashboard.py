"""
Neuro AI Ecosystem — Referral Triage Dashboard
================================================
Referral triage analytics from referral_records table.

Urgency levels: emergent, urgent, routine, elective
Sources: emergency, primary_care, pediatrics, neurology_clinic
Statuses: pending_triage, triaged, scheduled, in_progress, completed, cancelled

Real data: referral_records table in clinical.db (84 rows).
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


def overview():
    """Referral triage overview — totals, urgency distribution, source breakdown,
    timeline, avg triage time, completion rate."""
    conn = _conn()
    cur = conn.cursor()

    # Total referrals
    cur.execute("SELECT COUNT(*) FROM referral_records")
    total_referrals = cur.fetchone()[0]

    # Pending triage count
    cur.execute("SELECT COUNT(*) FROM referral_records WHERE triage_status = 'pending_triage'")
    pending_triage = cur.fetchone()[0]

    # Urgent pending (emergent or urgent AND pending_triage)
    cur.execute("""
        SELECT COUNT(*) FROM referral_records
        WHERE urgency IN ('emergent', 'urgent')
          AND triage_status = 'pending_triage'
    """)
    urgent_pending = cur.fetchone()[0]

    # Average triage time in hours (rows where both dates are non-null)
    cur.execute("""
        SELECT AVG(
            (julianday(triage_date) - julianday(referral_date)) * 24.0
        )
        FROM referral_records
        WHERE referral_date IS NOT NULL AND triage_date IS NOT NULL
    """)
    avg_time = cur.fetchone()[0]
    avg_triage_time_hrs = round(avg_time, 1) if avg_time is not None else 0.0

    # Completion rate
    cur.execute("SELECT COUNT(*) FROM referral_records WHERE triage_status = 'completed'")
    completed = cur.fetchone()[0]
    completion_rate_pct = round(completed * 100.0 / total_referrals, 1) if total_referrals else 0.0

    # Urgency distribution
    cur.execute("""
        SELECT urgency, COUNT(*) cnt
        FROM referral_records
        GROUP BY urgency
    """)
    urgency_distribution = {r[0]: r[1] for r in cur.fetchall()}

    # Source distribution
    cur.execute("""
        SELECT referral_source, COUNT(*) cnt
        FROM referral_records
        GROUP BY referral_source
    """)
    source_distribution = {r[0]: r[1] for r in cur.fetchall()}

    # Triage timeline — daily aggregates for last 30 unique referral dates
    cur.execute("""
        SELECT DISTINCT DATE(referral_date) d
        FROM referral_records
        WHERE referral_date IS NOT NULL
        ORDER BY d DESC
        LIMIT 30
    """)
    date_rows = cur.fetchall()
    dates = sorted([r[0] for r in date_rows])

    timeline = []
    for d in dates:
        cur.execute("""
            SELECT COUNT(*) FROM referral_records
            WHERE DATE(referral_date) = ?
        """, (d,))
        ref_count = cur.fetchone()[0]
        cur.execute("""
            SELECT COUNT(*) FROM referral_records
            WHERE DATE(triage_date) = ?
        """, (d,))
        tri_count = cur.fetchone()[0]
        timeline.append({"date": d, "referrals": ref_count, "triaged": tri_count})

    conn.close()
    return {
        "total_referrals": total_referrals,
        "pending_triage": pending_triage,
        "avg_triage_time_hrs": avg_triage_time_hrs,
        "urgent_pending": urgent_pending,
        "completion_rate_pct": completion_rate_pct,
        "urgency_distribution": urgency_distribution,
        "source_distribution": source_distribution,
        "triage_timeline": timeline,
    }


def breakdown():
    """Referral triage breakdown — KPIs, reason distribution, urgency by source,
    recent referrals, status summary, provider workload."""
    conn = _conn()
    cur = conn.cursor()

    # -- Referral KPIs ---------------------------------------------------------
    cur.execute("SELECT COUNT(*) FROM referral_records")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM referral_records WHERE triage_status = 'pending_triage'")
    pending = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM referral_records WHERE triage_status = 'completed'")
    completed = cur.fetchone()[0]

    cur.execute("SELECT AVG(triage_score) FROM referral_records WHERE triage_score IS NOT NULL")
    avg_score_raw = cur.fetchone()[0]
    avg_score = round(avg_score_raw, 1) if avg_score_raw is not None else 0.0

    cur.execute("""
        SELECT COUNT(*) FROM referral_records
        WHERE urgency IN ('emergent', 'urgent')
    """)
    urgent_total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM referral_records WHERE triage_status = 'cancelled'")
    cancelled = cur.fetchone()[0]

    referral_kpis = [
        {"label": "Total Referrals", "value": total, "sub": None, "color": "#3b82f6"},
        {"label": "Pending Triage", "value": pending,
         "sub": "{}% of total".format(round(pending * 100.0 / total, 1)) if total else None,
         "color": "#f59e0b"},
        {"label": "Completed", "value": completed,
         "sub": "{}% completion".format(round(completed * 100.0 / total, 1)) if total else None,
         "color": "#10b981"},
        {"label": "Avg Triage Score", "value": avg_score, "sub": "out of 100", "color": "#6366f1"},
        {"label": "Urgent/Emergent", "value": urgent_total,
         "sub": "{}% of total".format(round(urgent_total * 100.0 / total, 1)) if total else None,
         "color": "#ef4444"},
        {"label": "Cancelled", "value": cancelled,
         "sub": "{}% cancellation".format(round(cancelled * 100.0 / total, 1)) if total else None,
         "color": "#94a3b8"},
    ]

    # -- Reason distribution ---------------------------------------------------
    cur.execute("""
        SELECT referral_reason, COUNT(*) cnt
        FROM referral_records
        GROUP BY referral_reason
        ORDER BY cnt DESC
    """)
    reason_distribution = {r[0]: r[1] for r in cur.fetchall()}

    # -- Urgency by source -----------------------------------------------------
    cur.execute("""
        SELECT referral_source, urgency, COUNT(*) cnt
        FROM referral_records
        GROUP BY referral_source, urgency
    """)
    urgency_by_source = defaultdict(dict)
    for r in cur.fetchall():
        urgency_by_source[r[0]][r[1]] = r[2]
    urgency_by_source = dict(urgency_by_source)

    # -- Recent referrals (most recent 50) -------------------------------------
    cur.execute("""
        SELECT patient_id, referral_source, referral_reason, urgency,
               triage_status, assigned_to, referral_date, triage_date
        FROM referral_records
        ORDER BY referral_date DESC
        LIMIT 50
    """)
    recent_referrals = []
    for r in cur.fetchall():
        recent_referrals.append({
            "patient_id": r[0],
            "patient_name": None,
            "source": r[1],
            "reason": r[2],
            "urgency": r[3],
            "status": r[4],
            "assigned_to": r[5],
            "referred_date": r[6],
            "triaged_date": r[7],
        })

    # -- Status summary --------------------------------------------------------
    cur.execute("""
        SELECT triage_status, COUNT(*) cnt
        FROM referral_records
        GROUP BY triage_status
    """)
    status_summary = {r[0]: r[1] for r in cur.fetchall()}

    # -- Provider workload -----------------------------------------------------
    cur.execute("""
        SELECT assigned_to,
               COUNT(*) assigned,
               SUM(CASE WHEN triage_status = 'completed' THEN 1 ELSE 0 END) completed,
               SUM(CASE WHEN triage_status IN ('pending_triage', 'triaged', 'scheduled', 'in_progress')
                   THEN 1 ELSE 0 END) pending
        FROM referral_records
        WHERE assigned_to IS NOT NULL
        GROUP BY assigned_to
        ORDER BY assigned DESC
    """)
    provider_workload = []
    for r in cur.fetchall():
        provider_workload.append({
            "provider": r[0],
            "assigned": r[1],
            "completed": r[2],
            "pending": r[3],
        })

    conn.close()
    return {
        "referral_kpis": referral_kpis,
        "reason_distribution": reason_distribution,
        "urgency_by_source": urgency_by_source,
        "recent_referrals": recent_referrals,
        "status_summary": status_summary,
        "provider_workload": provider_workload,
    }


def definitions():
    """Referral triage definitions — metric definitions, urgency criteria,
    triage scoring methodology, clinical glossary, references."""
    return {
        "metric_definitions": [
            {"metric": "Total Referrals", "definition": "Total number of referral records received by the epilepsy centre."},
            {"metric": "Pending Triage", "definition": "Referrals awaiting initial triage assessment by a clinician."},
            {"metric": "Avg Triage Time (hrs)", "definition": "Mean elapsed time in hours between referral receipt (referral_date) and triage completion (triage_date)."},
            {"metric": "Urgent Pending", "definition": "Count of emergent or urgent referrals still in pending_triage status."},
            {"metric": "Completion Rate (%)", "definition": "Percentage of all referrals that have reached 'completed' status."},
            {"metric": "Triage Score", "definition": "Numeric score (0-100) assigned during triage reflecting clinical priority, seizure severity, and resource needs."},
        ],
        "urgency_criteria": [
            {
                "level": "emergent",
                "description": "Immediate clinical attention required — active seizures, status epilepticus, or acute neurological deterioration.",
                "criteria": [
                    "Status epilepticus or prolonged seizure activity",
                    "Acute post-ictal neurological deficit",
                    "New-onset seizure with altered consciousness",
                    "Suspected brain lesion with seizure",
                ],
                "response_time": "< 1 hour",
            },
            {
                "level": "urgent",
                "description": "Prompt evaluation needed — uncontrolled seizures, medication failure, or safety concerns.",
                "criteria": [
                    "Breakthrough seizures despite medication",
                    "Adverse drug reaction to AED",
                    "Seizure-related injury",
                    "Driving/occupational safety concern",
                ],
                "response_time": "< 24 hours",
            },
            {
                "level": "routine",
                "description": "Standard scheduling — stable epilepsy requiring follow-up, EEG review, or medication adjustment.",
                "criteria": [
                    "Stable seizure control, routine follow-up",
                    "EEG abnormality requiring interpretation",
                    "Medication review or dose adjustment",
                    "Transition of care from paediatrics",
                ],
                "response_time": "1-2 weeks",
            },
            {
                "level": "elective",
                "description": "Non-urgent — second opinions, surgical candidacy evaluation, or wellness checks.",
                "criteria": [
                    "Second opinion on diagnosis",
                    "Pre-surgical evaluation candidacy",
                    "Psychogenic non-epileptic event evaluation",
                    "Genetic counselling referral",
                ],
                "response_time": "2-6 weeks",
            },
        ],
        "triage_scoring": {
            "description": "The triage score (0-100) is a composite measure reflecting clinical urgency, seizure burden, comorbidity complexity, and resource availability. Higher scores indicate greater priority.",
            "factors": [
                {"factor": "Seizure Frequency", "weight": "30%", "description": "Daily/weekly seizures score higher than monthly/rare."},
                {"factor": "Seizure Severity", "weight": "25%", "description": "Generalised tonic-clonic and status epilepticus score highest."},
                {"factor": "Comorbidities", "weight": "15%", "description": "Cognitive decline, psychiatric comorbidity, or structural lesion."},
                {"factor": "Medication Failures", "weight": "15%", "description": "Number of failed AEDs; drug-resistant epilepsy scores highest."},
                {"factor": "Safety Risk", "weight": "10%", "description": "Fall risk, SUDEP risk factors, driving status."},
                {"factor": "Social Factors", "weight": "5%", "description": "Caregiver availability, distance to centre, employment impact."},
            ],
        },
        "glossary": [
            {"term": "Triage", "definition": "The process of prioritising referrals based on clinical urgency to allocate appropriate resources and scheduling."},
            {"term": "AED", "definition": "Anti-Epileptic Drug — medication used to prevent or reduce seizure frequency."},
            {"term": "Status Epilepticus", "definition": "A prolonged seizure lasting > 5 minutes or recurrent seizures without recovery, constituting a medical emergency."},
            {"term": "SUDEP", "definition": "Sudden Unexpected Death in Epilepsy — sudden, unexpected death in a person with epilepsy with no other identifiable cause."},
            {"term": "Drug-Resistant Epilepsy", "definition": "Failure to achieve sustained seizure freedom after adequate trials of two tolerated, appropriately chosen AEDs."},
            {"term": "Post-ictal", "definition": "The recovery period following a seizure, often characterised by confusion, fatigue, or focal neurological deficits."},
            {"term": "EEG Abnormality", "definition": "Atypical electrical activity detected on electroencephalogram, such as epileptiform discharges or focal slowing."},
            {"term": "Psychogenic Non-Epileptic Seizure (PNES)", "definition": "Events resembling epileptic seizures but without corresponding EEG changes, often related to psychological factors."},
            {"term": "Referral Source", "definition": "The clinical setting or provider originating the referral (emergency, primary care, paediatrics, neurology clinic)."},
            {"term": "Triage Score", "definition": "A numeric priority score (0-100) assigned during triage to quantify clinical urgency and resource needs."},
        ],
        "references": [
            "NICE Guideline CG137: Epilepsies — diagnosis and management (2022 update)",
            "ILAE Guidelines for Epilepsy Centre Referral (2019)",
            "Kwan P, Brodie MJ. Definition of drug-resistant epilepsy. Epilepsia 2010;51:1069-77",
            "Harden CL et al. AAN Practice Guideline: Sudden unexpected death in epilepsy. Neurology 2017;88:1674-80",
            "NHS England. Referral to Treatment (RTT) waiting times standards.",
        ],
    }
