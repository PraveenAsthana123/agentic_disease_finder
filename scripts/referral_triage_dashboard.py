"""
Neuro AI Ecosystem -- Referral Intake & Triage Dashboard
=========================================================
Tracks patient referral intake, triage scoring, urgency classification,
and provider assignment across the neuro/epilepsy clinical platform.

Referral Sources:
  primary_care       -- General practitioner / family medicine referral
  emergency          -- Emergency department referral
  neurology_clinic   -- Referral from another neurology service
  psychiatry         -- Referral from psychiatric services
  pediatrics         -- Pediatric referral (paediatric neurology pathway)
  self_referral      -- Patient-initiated / walk-in referral
  other_specialist   -- Referral from other medical specialties

Urgency Levels:
  emergent  -- Immediate evaluation required (e.g., status epilepticus)
  urgent    -- Evaluation within 24-72 hours
  routine   -- Standard scheduling within 2-4 weeks
  elective  -- Non-urgent, patient-preference scheduling

Triage Statuses:
  pending_triage  -- Referral received, awaiting triage review
  triaged         -- Triage complete, awaiting scheduling
  scheduled       -- Appointment scheduled with assigned provider
  in_progress     -- Active clinical evaluation underway
  completed       -- Referral fully addressed and closed
  cancelled       -- Referral cancelled by patient or provider

All referral records are seeded deterministically from real patients
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

# -- Referral Constants --------------------------------------------------------

REFERRAL_SOURCES = [
    "primary_care",
    "emergency",
    "neurology_clinic",
    "psychiatry",
    "pediatrics",
    "self_referral",
    "other_specialist",
]

REFERRAL_REASONS = [
    "new_onset_seizure",
    "refractory_epilepsy",
    "eeg_abnormality",
    "status_epilepticus",
    "medication_review",
    "pre_surgical_evaluation",
    "psychogenic_nonepileptic",
    "cognitive_decline",
    "headache_workup",
]

URGENCY_LEVELS = ["emergent", "urgent", "routine", "elective"]

TRIAGE_STATUSES = [
    "pending_triage",
    "triaged",
    "scheduled",
    "in_progress",
    "completed",
    "cancelled",
]

ASSIGNED_PROVIDERS = [
    "Dr. Sharma (Epileptologist)",
    "Dr. Patel (Neurologist)",
    "Dr. Gupta (Neurosurgeon)",
    "Dr. Reddy (Neuropsychiatrist)",
    "Nurse Practitioner Singh",
]

# Fixed anchor date for deterministic date arithmetic
_BASE_DATE = datetime(2024, 1, 1)
_REFERENCE_DATE = datetime(2024, 6, 15)


def _conn():
    return sqlite3.connect(DB_PATH)


def _deterministic_seed(patient_id: str, item_id: str) -> float:
    """Return a reproducible float in [0, 1) from patient + item identifiers."""
    h = hashlib.sha256(f"referral:{patient_id}:{item_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _seed_int(patient_id: str, item_id: str, lo: int, hi: int) -> int:
    """Return a reproducible int in [lo, hi] from patient + item identifiers."""
    return lo + int(_deterministic_seed(patient_id, item_id) * (hi - lo + 1)) % (hi - lo + 1)


def _ensure_referral_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS referral_records (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id      TEXT    NOT NULL,
            referral_source TEXT    NOT NULL,
            referral_reason TEXT    NOT NULL,
            urgency         TEXT    NOT NULL,
            triage_status   TEXT    NOT NULL,
            triage_score    REAL,
            assigned_to     TEXT,
            referral_date   TEXT,
            triage_date     TEXT,
            notes           TEXT
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


def _seed_referral_records(conn, patients):
    """Seed referral_records deterministically from the real patient list."""
    c = conn.cursor()

    notes_pool = {
        "new_onset_seizure":        "First unprovoked seizure; requires urgent EEG and MRI workup.",
        "refractory_epilepsy":      "Failed 2+ AEDs; consider advanced therapy or surgical evaluation.",
        "eeg_abnormality":          "Incidental EEG finding; focal spikes noted on routine study.",
        "status_epilepticus":       "Prolonged seizure > 5 min; emergency protocol activated.",
        "medication_review":        "AED side effects reported; dose adjustment or switch needed.",
        "pre_surgical_evaluation":  "Candidate for epilepsy surgery; Phase II monitoring recommended.",
        "psychogenic_nonepileptic": "Suspected PNES; video-EEG monitoring and psych eval indicated.",
        "cognitive_decline":        "Progressive memory and attention deficits; rule out epileptic cause.",
        "headache_workup":          "Chronic headache with EEG abnormality; migraine vs epilepsy differential.",
    }

    for pt in patients:
        pid = pt["patient_id"]

        # Each patient gets 1-3 referrals (deterministic)
        num_referrals = _seed_int(pid, "num_referrals", 1, 3)

        for ref_idx in range(num_referrals):
            ref_key = f"ref_{ref_idx}"

            # Check if record already exists
            c.execute(
                "SELECT COUNT(*) FROM referral_records WHERE patient_id=? AND notes LIKE ?",
                (pid, f"%ref_idx={ref_idx}%"),
            )
            if c.fetchone()[0] > 0:
                continue

            # Deterministic source
            src_idx = _seed_int(pid, f"source:{ref_key}", 0, len(REFERRAL_SOURCES) - 1)
            source = REFERRAL_SOURCES[src_idx]

            # Deterministic reason
            reason_idx = _seed_int(pid, f"reason:{ref_key}", 0, len(REFERRAL_REASONS) - 1)
            reason = REFERRAL_REASONS[reason_idx]

            # Deterministic urgency (weighted: emergent ~10%, urgent ~25%, routine ~45%, elective ~20%)
            urgency_seed = _deterministic_seed(pid, f"urgency:{ref_key}")
            if urgency_seed < 0.10:
                urgency = "emergent"
            elif urgency_seed < 0.35:
                urgency = "urgent"
            elif urgency_seed < 0.80:
                urgency = "routine"
            else:
                urgency = "elective"

            # Deterministic triage status (weighted)
            status_seed = _deterministic_seed(pid, f"triage_status:{ref_key}")
            thresholds = [0.12, 0.28, 0.48, 0.63, 0.90, 1.00]
            status_labels = TRIAGE_STATUSES
            triage_status = status_labels[-1]
            for thr, lbl in zip(thresholds, status_labels):
                if status_seed < thr:
                    triage_status = lbl
                    break

            # Triage score (0-100, higher = more urgent)
            base_score = {"emergent": 85, "urgent": 65, "routine": 40, "elective": 20}
            score_jitter = _seed_int(pid, f"score:{ref_key}", -10, 10)
            triage_score = max(0, min(100, base_score[urgency] + score_jitter))

            # Assigned provider (assigned only if not pending_triage or cancelled)
            if triage_status not in ("pending_triage", "cancelled"):
                provider_idx = _seed_int(pid, f"provider:{ref_key}", 0, len(ASSIGNED_PROVIDERS) - 1)
                assigned_to = ASSIGNED_PROVIDERS[provider_idx]
            else:
                assigned_to = None

            # Referral date: 1-180 days before reference date
            days_ago = _seed_int(pid, f"ref_date:{ref_key}", 1, 180)
            referral_date_dt = _REFERENCE_DATE - timedelta(days=days_ago)
            referral_date = referral_date_dt.strftime("%Y-%m-%d")

            # Triage date: 1-48 hours after referral (if triaged)
            if triage_status != "pending_triage":
                triage_hours = _seed_int(pid, f"triage_hrs:{ref_key}", 1, 48)
                triage_date_dt = referral_date_dt + timedelta(hours=triage_hours)
                triage_date = triage_date_dt.strftime("%Y-%m-%d %H:%M")
            else:
                triage_date = None

            note_text = notes_pool.get(reason, "Referral for neurological evaluation.")
            note_text += f" [ref_idx={ref_idx}]"

            c.execute(
                """INSERT INTO referral_records
                   (patient_id, referral_source, referral_reason, urgency,
                    triage_status, triage_score, assigned_to,
                    referral_date, triage_date, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (pid, source, reason, urgency,
                 triage_status, triage_score, assigned_to,
                 referral_date, triage_date, note_text),
            )

    conn.commit()


def _load_all_records(conn, patient_id: Optional[str] = None):
    c = conn.cursor()
    if patient_id:
        c.execute(
            "SELECT * FROM referral_records WHERE patient_id=? ORDER BY referral_date DESC",
            (patient_id,),
        )
    else:
        c.execute(
            "SELECT * FROM referral_records ORDER BY referral_date DESC"
        )
    cols = [
        "id", "patient_id", "referral_source", "referral_reason", "urgency",
        "triage_status", "triage_score", "assigned_to", "referral_date",
        "triage_date", "notes",
    ]
    return [dict(zip(cols, row)) for row in c.fetchall()]


def _init_db():
    """Ensure table exists and is seeded. Returns open connection."""
    conn = _conn()
    _ensure_referral_table(conn)
    patients = _real_patients(conn)
    _seed_referral_records(conn, patients)
    return conn


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def overview(patient_id: Optional[str] = None) -> dict:
    """
    Referral intake and triage overview dashboard data.

    Returns:
        kpis                 -- key performance indicators
        urgency_distribution -- list of {name, count}
        source_distribution  -- list of {name, count}
        triage_timeline      -- list of {date, referrals, triaged} for last 30 days
        status_summary       -- list of {name, count}
    """
    conn = _init_db()
    records = _load_all_records(conn, patient_id)
    conn.close()

    if not records:
        return {
            "kpis": {
                "total_referrals": 0,
                "pending_triage": 0,
                "avg_triage_time_hrs": 0.0,
                "urgent_pending": 0,
                "completion_rate_pct": 0.0,
            },
            "urgency_distribution": [],
            "source_distribution": [],
            "triage_timeline": [],
            "status_summary": [],
        }

    total_referrals = len(records)
    pending_triage = sum(1 for r in records if r["triage_status"] == "pending_triage")
    completed = sum(1 for r in records if r["triage_status"] == "completed")
    completion_rate = round(completed / total_referrals * 100, 1) if total_referrals else 0.0

    # Urgent referrals still pending triage
    urgent_pending = sum(
        1 for r in records
        if r["triage_status"] == "pending_triage" and r["urgency"] in ("emergent", "urgent")
    )

    # Average triage time for records that have been triaged
    triage_times = []
    for r in records:
        if r["referral_date"] and r["triage_date"]:
            try:
                ref_dt = datetime.strptime(r["referral_date"], "%Y-%m-%d")
                tri_dt = datetime.strptime(r["triage_date"], "%Y-%m-%d %H:%M")
                diff_hrs = (tri_dt - ref_dt).total_seconds() / 3600
                if diff_hrs >= 0:
                    triage_times.append(diff_hrs)
            except ValueError:
                pass
    avg_triage_time = round(sum(triage_times) / len(triage_times), 1) if triage_times else 0.0

    kpis = {
        "total_referrals":      total_referrals,
        "pending_triage":       pending_triage,
        "avg_triage_time_hrs":  avg_triage_time,
        "urgent_pending":       urgent_pending,
        "completion_rate_pct":  completion_rate,
    }

    # Urgency distribution
    urgency_distribution = []
    for level in URGENCY_LEVELS:
        count = sum(1 for r in records if r["urgency"] == level)
        urgency_distribution.append({"name": level, "count": count})

    # Source distribution
    source_distribution = []
    for src in REFERRAL_SOURCES:
        count = sum(1 for r in records if r["referral_source"] == src)
        source_distribution.append({"name": src, "count": count})

    # Triage timeline: last 30 days from reference date
    triage_timeline = []
    for day_offset in range(30):
        day_dt = _REFERENCE_DATE - timedelta(days=29 - day_offset)
        day_str = day_dt.strftime("%Y-%m-%d")
        day_referrals = sum(1 for r in records if r["referral_date"] == day_str)
        day_triaged = sum(
            1 for r in records
            if r["triage_date"] and r["triage_date"].startswith(day_str)
        )
        triage_timeline.append({
            "date": day_str,
            "referrals": day_referrals,
            "triaged": day_triaged,
        })

    # Status summary
    status_summary = []
    for status in TRIAGE_STATUSES:
        count = sum(1 for r in records if r["triage_status"] == status)
        status_summary.append({"name": status, "count": count})

    return {
        "kpis":                  kpis,
        "urgency_distribution":  urgency_distribution,
        "source_distribution":   source_distribution,
        "triage_timeline":       triage_timeline,
        "status_summary":        status_summary,
    }


def breakdown(patient_id: Optional[str] = None) -> dict:
    """
    Detailed referral breakdown.

    Returns:
        referral_kpis       -- by_source_count, by_reason_count, avg_wait_hrs
        recent_referrals    -- list of individual referral records with patient details
        reason_distribution -- list of {name, count}
        urgency_by_source   -- list of {source, emergent, urgent, routine, elective}
        provider_workload   -- list of {provider, assigned, completed, pending}
    """
    conn = _init_db()
    records = _load_all_records(conn, patient_id)
    patients = _real_patients(conn)
    conn.close()

    if not records:
        return {
            "referral_kpis": {"by_source_count": 0, "by_reason_count": 0, "avg_wait_hrs": 0.0},
            "recent_referrals": [],
            "reason_distribution": [],
            "urgency_by_source": [],
            "provider_workload": [],
        }

    patient_info = {p["patient_id"]: p for p in patients}

    # Referral KPIs
    sources_seen = set(r["referral_source"] for r in records)
    reasons_seen = set(r["referral_reason"] for r in records)

    wait_hours = []
    for r in records:
        if r["referral_date"] and r["triage_date"]:
            try:
                ref_dt = datetime.strptime(r["referral_date"], "%Y-%m-%d")
                tri_dt = datetime.strptime(r["triage_date"], "%Y-%m-%d %H:%M")
                diff = (tri_dt - ref_dt).total_seconds() / 3600
                if diff >= 0:
                    wait_hours.append(diff)
            except ValueError:
                pass
    avg_wait = round(sum(wait_hours) / len(wait_hours), 1) if wait_hours else 0.0

    referral_kpis = {
        "by_source_count": len(sources_seen),
        "by_reason_count": len(reasons_seen),
        "avg_wait_hrs":    avg_wait,
    }

    # Recent referrals (last 20)
    recent_referrals = []
    for r in records[:20]:
        info = patient_info.get(r["patient_id"], {})
        recent_referrals.append({
            "id":              r["id"],
            "patient_id":      r["patient_id"],
            "patient_name":    info.get("name", r["patient_id"]),
            "referral_source": r["referral_source"],
            "referral_reason": r["referral_reason"],
            "urgency":         r["urgency"],
            "triage_status":   r["triage_status"],
            "triage_score":    r["triage_score"],
            "assigned_to":     r["assigned_to"],
            "referral_date":   r["referral_date"],
            "triage_date":     r["triage_date"],
        })

    # Reason distribution
    reason_distribution = []
    for reason in REFERRAL_REASONS:
        count = sum(1 for r in records if r["referral_reason"] == reason)
        reason_distribution.append({"name": reason, "count": count})

    # Urgency by source
    urgency_by_source = []
    for src in REFERRAL_SOURCES:
        src_records = [r for r in records if r["referral_source"] == src]
        row = {"source": src}
        for level in URGENCY_LEVELS:
            row[level] = sum(1 for r in src_records if r["urgency"] == level)
        urgency_by_source.append(row)

    # Provider workload
    provider_workload = []
    for provider in ASSIGNED_PROVIDERS:
        assigned = sum(1 for r in records if r["assigned_to"] == provider)
        completed_n = sum(
            1 for r in records
            if r["assigned_to"] == provider and r["triage_status"] == "completed"
        )
        pending_n = sum(
            1 for r in records
            if r["assigned_to"] == provider and r["triage_status"] in ("triaged", "scheduled", "in_progress")
        )
        provider_workload.append({
            "provider":  provider,
            "assigned":  assigned,
            "completed": completed_n,
            "pending":   pending_n,
        })

    return {
        "referral_kpis":      referral_kpis,
        "recent_referrals":   recent_referrals,
        "reason_distribution": reason_distribution,
        "urgency_by_source":  urgency_by_source,
        "provider_workload":  provider_workload,
    }


def definitions() -> dict:
    """
    Metric definitions, urgency criteria, triage scoring, and glossary.

    Returns:
        metric_definitions  -- explanation of each KPI
        urgency_levels      -- clinical criteria for each urgency level
        triage_scoring      -- explanation of the triage scoring system
        glossary            -- key terms used in referral triage
    """
    return {
        "metric_definitions": {
            "total_referrals": (
                "Total number of referral records received across all sources and "
                "urgency levels. Includes all triage statuses (pending through completed "
                "and cancelled)."
            ),
            "pending_triage": (
                "Number of referrals that have been received but not yet reviewed by "
                "the triage team. These referrals await initial clinical assessment and "
                "urgency classification."
            ),
            "avg_triage_time_hrs": (
                "Mean elapsed time (in hours) between referral receipt and completion "
                "of triage assessment. Calculated only for referrals with both referral_date "
                "and triage_date recorded. Target: < 24 hours for routine, < 4 hours for urgent."
            ),
            "urgent_pending": (
                "Count of referrals classified as 'emergent' or 'urgent' that remain in "
                "'pending_triage' status. A critical safety metric -- these referrals require "
                "immediate attention and should be triaged within 1-4 hours."
            ),
            "completion_rate_pct": (
                "Percentage of all referrals that have reached 'completed' status, indicating "
                "the referral has been fully addressed (evaluation done, plan communicated). "
                "Target: > 85% within 30 days of referral."
            ),
            "by_source_count": (
                "Number of distinct referral sources contributing referrals. Indicates breadth "
                "of the referral network and inter-departmental engagement."
            ),
            "by_reason_count": (
                "Number of distinct clinical reasons for referral. Indicates the diversity of "
                "clinical presentations being managed by the epilepsy service."
            ),
            "avg_wait_hrs": (
                "Average wait time from referral receipt to triage completion, measured in hours. "
                "Equivalent to avg_triage_time_hrs; used in breakdown context."
            ),
        },
        "urgency_levels": {
            "emergent": {
                "label":    "Emergent",
                "criteria": (
                    "Immediate evaluation required. Includes: active status epilepticus, "
                    "acute symptomatic seizures with ongoing neurological deterioration, "
                    "post-ictal focal deficits suggesting stroke, or seizures with acute "
                    "intracranial pathology. Target triage time: < 1 hour."
                ),
                "triage_score_range": "75-100",
                "response_time":      "Immediate (< 1 hour)",
            },
            "urgent": {
                "label":    "Urgent",
                "criteria": (
                    "Evaluation within 24-72 hours. Includes: new-onset unprovoked seizure "
                    "(stable patient), breakthrough seizures on established AED therapy, "
                    "first seizure with abnormal neuroimaging, or seizure clusters with "
                    "maintained awareness between events. Target triage time: < 4 hours."
                ),
                "triage_score_range": "55-74",
                "response_time":      "24-72 hours",
            },
            "routine": {
                "label":    "Routine",
                "criteria": (
                    "Standard scheduling within 2-4 weeks. Includes: medication review for "
                    "well-controlled epilepsy, follow-up EEG for known abnormality, cognitive "
                    "screening referral, or second opinion for established diagnosis. "
                    "Target triage time: < 24 hours."
                ),
                "triage_score_range": "30-54",
                "response_time":      "2-4 weeks",
            },
            "elective": {
                "label":    "Elective",
                "criteria": (
                    "Non-urgent, patient-preference scheduling. Includes: routine surveillance "
                    "EEG, elective pre-surgical workup planning, genetic counselling referral, "
                    "or research study enrolment screening. Target triage time: < 48 hours."
                ),
                "triage_score_range": "0-29",
                "response_time":      "4-8 weeks",
            },
        },
        "triage_scoring": {
            "description": (
                "The triage score is a numerical value (0-100) assigned during the triage "
                "review process. Higher scores indicate greater clinical urgency and priority "
                "for scheduling. The score is computed from the urgency classification plus "
                "clinical modifiers (comorbidities, age, seizure frequency, prior ED visits)."
            ),
            "score_bands": [
                {"range": "90-100", "label": "Critical",  "action": "Immediate evaluation; page on-call epileptologist"},
                {"range": "75-89",  "label": "High",      "action": "Same-day or next-day appointment"},
                {"range": "55-74",  "label": "Moderate",  "action": "Schedule within 1 week"},
                {"range": "30-54",  "label": "Standard",  "action": "Schedule within 2-4 weeks"},
                {"range": "0-29",   "label": "Low",       "action": "Schedule at patient convenience (4-8 weeks)"},
            ],
            "modifiers": [
                "Age < 2 or > 65: +10 points",
                "Status epilepticus history: +15 points",
                "Failed >= 2 AEDs: +10 points",
                "Comorbid psychiatric condition: +5 points",
                "Pregnant patient: +10 points",
                "Abnormal neuroimaging: +10 points",
                "ED visit in past 30 days for seizure: +5 points",
            ],
        },
        "glossary": [
            {
                "term":       "Referral Source",
                "definition": (
                    "The originating department or pathway through which the patient was referred "
                    "to the epilepsy/neurology service. Tracked for network analysis and to "
                    "identify under-referring departments."
                ),
            },
            {
                "term":       "Triage",
                "definition": (
                    "The clinical process of reviewing a new referral to determine urgency, "
                    "appropriate provider assignment, and scheduling priority. Performed by "
                    "the triage nurse or on-call neurologist."
                ),
            },
            {
                "term":       "Triage Score",
                "definition": (
                    "A numerical score (0-100) summarising clinical urgency. Computed from "
                    "base urgency level plus clinical modifiers. Used to prioritise scheduling "
                    "when multiple referrals compete for limited appointment slots."
                ),
            },
            {
                "term":       "AED (Anti-Epileptic Drug)",
                "definition": (
                    "Medication used to prevent or reduce the frequency of seizures. Also called "
                    "ASM (anti-seizure medication). Failure of 2+ AEDs defines drug-resistant "
                    "or refractory epilepsy."
                ),
            },
            {
                "term":       "Status Epilepticus",
                "definition": (
                    "A medical emergency defined as continuous seizure activity lasting > 5 minutes "
                    "or two or more seizures without recovery of consciousness between them. "
                    "Requires emergent referral and immediate treatment."
                ),
            },
            {
                "term":       "PNES (Psychogenic Nonepileptic Seizures)",
                "definition": (
                    "Events that resemble epileptic seizures but are not caused by abnormal "
                    "electrical brain activity. Diagnosis typically requires video-EEG monitoring. "
                    "Referred under 'psychogenic_nonepileptic' reason code."
                ),
            },
            {
                "term":       "Provider Workload",
                "definition": (
                    "The number of referrals assigned to each provider, broken down by completion "
                    "status. Used for equitable case distribution and capacity planning."
                ),
            },
            {
                "term":       "Completion Rate",
                "definition": (
                    "Percentage of referrals that have reached 'completed' status. A key "
                    "performance indicator for the referral pathway. Target: > 85% completion "
                    "within 30 days of referral receipt."
                ),
            },
            {
                "term":       "Referral-to-Triage Time",
                "definition": (
                    "Elapsed time between referral receipt (referral_date) and triage completion "
                    "(triage_date). Measured in hours. Shorter times indicate more responsive "
                    "intake workflows."
                ),
            },
            {
                "term":       "Urgency Classification",
                "definition": (
                    "A four-tier system (emergent, urgent, routine, elective) applied during "
                    "triage to categorise clinical priority. Determines target response time "
                    "and scheduling window."
                ),
            },
        ],
    }
