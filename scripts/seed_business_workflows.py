#!/usr/bin/env python3
"""Seed business_workflows table in clinical.db with realistic orchestration data.

Paperclip Business Orchestration OS — clinical business workflow tracking:
patient intake, referral processing, report generation, appointment scheduling,
compliance review, data export, EEG upload pipeline, medication reconciliation,
insurance pre-auth, discharge planning.
"""
import sqlite3
import random
import pathlib
from datetime import datetime, timedelta

DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def seed():
    conn = sqlite3.connect(str(DB))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS business_workflows (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workflow_id TEXT UNIQUE NOT NULL,
        workflow_name TEXT NOT NULL,
        category TEXT NOT NULL,
        status TEXT NOT NULL,
        priority TEXT NOT NULL,
        trigger_type TEXT NOT NULL,
        steps_total INTEGER NOT NULL,
        steps_completed INTEGER NOT NULL,
        owner TEXT NOT NULL,
        patient_id TEXT,
        created_at TEXT NOT NULL,
        completed_at TEXT,
        duration_seconds INTEGER,
        error_message TEXT,
        retry_count INTEGER DEFAULT 0
    )''')

    # Check if already seeded
    existing = c.execute("SELECT COUNT(*) FROM business_workflows").fetchone()[0]
    if existing >= 50:
        print(f"business_workflows already has {existing} rows, skipping seed.")
        conn.close()
        return

    workflow_types = [
        {"name": "Patient Intake", "category": "Clinical", "steps": 8, "owner_pool": ["clinician", "admin"]},
        {"name": "Referral Processing", "category": "Administrative", "steps": 6, "owner_pool": ["admin", "clinician"]},
        {"name": "Report Generation", "category": "Technical", "steps": 5, "owner_pool": ["system", "clinician"]},
        {"name": "Appointment Scheduling", "category": "Administrative", "steps": 4, "owner_pool": ["scheduler", "admin"]},
        {"name": "Compliance Review", "category": "Compliance", "steps": 10, "owner_pool": ["admin", "clinician"]},
        {"name": "Data Export", "category": "Technical", "steps": 3, "owner_pool": ["system"]},
        {"name": "EEG Upload Pipeline", "category": "Technical", "steps": 7, "owner_pool": ["system", "clinician"]},
        {"name": "Medication Reconciliation", "category": "Clinical", "steps": 6, "owner_pool": ["clinician"]},
        {"name": "Insurance Pre-Auth", "category": "Administrative", "steps": 9, "owner_pool": ["admin"]},
        {"name": "Discharge Planning", "category": "Clinical", "steps": 8, "owner_pool": ["clinician", "admin"]},
    ]

    statuses = ["active", "paused", "completed", "failed", "pending"]
    status_weights = [0.30, 0.08, 0.40, 0.10, 0.12]
    priorities = ["critical", "high", "medium", "low"]
    priority_weights = [0.10, 0.25, 0.40, 0.25]
    triggers = ["manual", "scheduled", "event-driven", "api"]
    trigger_weights = [0.25, 0.30, 0.25, 0.20]

    patients = [f"P{i:03d}" for i in range(1, 31)]
    error_messages = [
        "Timeout: upstream service did not respond within 30s",
        "Validation failed: missing required field 'insurance_id'",
        "EEG file corrupt — CRC mismatch",
        "Referral target clinic unavailable",
        "HIPAA compliance check failed — unsigned consent",
        "Database lock timeout exceeded",
        None,
    ]

    now = datetime.utcnow()
    random.seed(42)
    rows = []

    for i in range(1, 51):
        wf_type = random.choice(workflow_types)
        status = random.choices(statuses, weights=status_weights, k=1)[0]
        priority = random.choices(priorities, weights=priority_weights, k=1)[0]
        trigger = random.choices(triggers, weights=trigger_weights, k=1)[0]
        owner = random.choice(wf_type["owner_pool"])
        steps_total = wf_type["steps"]

        # Steps completed depends on status
        if status == "completed":
            steps_completed = steps_total
        elif status == "failed":
            steps_completed = random.randint(1, steps_total - 1)
        elif status == "pending":
            steps_completed = 0
        elif status == "paused":
            steps_completed = random.randint(1, steps_total - 1)
        else:  # active
            steps_completed = random.randint(0, steps_total - 1)

        created_at = now - timedelta(
            days=random.randint(0, 29),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )

        # Patient ID — system-level workflows (Data Export, Report Generation) sometimes have no patient
        if wf_type["name"] in ("Data Export", "Report Generation") and random.random() < 0.4:
            patient_id = None
        else:
            patient_id = random.choice(patients)

        # Completed at and duration
        completed_at = None
        duration_seconds = None
        if status == "completed":
            dur = random.randint(120, 86400)  # 2 min to 24 hours
            duration_seconds = dur
            completed_at = (created_at + timedelta(seconds=dur)).isoformat()
        elif status == "failed":
            dur = random.randint(30, 7200)
            duration_seconds = dur
            completed_at = (created_at + timedelta(seconds=dur)).isoformat()

        # Error message only for failed
        error_msg = None
        if status == "failed":
            error_msg = random.choice([e for e in error_messages if e is not None])

        retry_count = 0
        if status == "failed":
            retry_count = random.randint(1, 3)
        elif status == "active" and random.random() < 0.15:
            retry_count = 1

        rows.append((
            f"WF-{i:03d}",
            wf_type["name"],
            wf_type["category"],
            status,
            priority,
            trigger,
            steps_total,
            steps_completed,
            owner,
            patient_id,
            created_at.isoformat(),
            completed_at,
            duration_seconds,
            error_msg,
            retry_count,
        ))

    c.executemany(
        "INSERT OR IGNORE INTO business_workflows "
        "(workflow_id, workflow_name, category, status, priority, trigger_type, "
        "steps_total, steps_completed, owner, patient_id, created_at, completed_at, "
        "duration_seconds, error_message, retry_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    print(f"Seeded {len(rows)} rows into business_workflows.")
    conn.close()


if __name__ == "__main__":
    seed()
