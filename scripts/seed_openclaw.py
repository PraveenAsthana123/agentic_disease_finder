#!/usr/bin/env python3
"""Seed openclaw_executions table with ~200 realistic execution rows."""

import sqlite3
import json
import random
import uuid
from datetime import datetime, timedelta

DB_PATH = "data/clinical.db"
AGENT_TASKS_PATH = "config/agent_tasks.json"

# Load agent registry
with open(AGENT_TASKS_PATH) as f:
    registry = json.load(f)

agents = [(a["id"], a.get("task", "Unknown task")) for a in registry["agents"]]

# Schema
CREATE_SQL = """
CREATE TABLE IF NOT EXISTS openclaw_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    task_description TEXT,
    execution_mode TEXT NOT NULL,
    status TEXT NOT NULL,
    priority TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    duration_seconds REAL,
    steps_total INTEGER,
    steps_completed INTEGER,
    parent_execution_id TEXT,
    triggered_by TEXT,
    patient_id TEXT,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    metadata_json TEXT
);
"""

# Distributions
STATUS_WEIGHTS = [("completed", 60), ("running", 10), ("failed", 15), ("queued", 10), ("cancelled", 5)]
MODE_WEIGHTS = [("autonomous", 40), ("supervised", 35), ("manual", 25)]
TRIGGER_WEIGHTS = [("cron", 30), ("api", 25), ("user", 20), ("event", 15), ("chain", 10)]
PRIORITY_WEIGHTS = [("critical", 5), ("high", 20), ("medium", 50), ("low", 25)]

ERROR_MESSAGES = [
    "Timeout exceeded: agent did not respond within 120s",
    "OOM: memory limit exceeded during feature extraction",
    "Model file not found: models/epilepsy_v3.pkl",
    "Database connection pool exhausted",
    "Invalid EDF file format: header checksum mismatch",
    "Patient record locked by concurrent process",
    "API rate limit exceeded (429)",
    "CUDA out of memory during inference",
    "SSL certificate verification failed for upstream service",
    "Schema validation error: missing required field 'channel_names'",
]

def weighted_choice(items):
    population, weights = zip(*items)
    return random.choices(population, weights=weights, k=1)[0]

def generate_rows(n=200):
    rows = []
    now = datetime.utcnow()
    execution_ids = []

    for i in range(n):
        agent_id, task = random.choice(agents)
        agent_name = agent_id.replace("_", " ").title()
        exec_id = f"exec-{uuid.uuid4().hex[:12]}"
        execution_ids.append(exec_id)

        status = weighted_choice(STATUS_WEIGHTS)
        mode = weighted_choice(MODE_WEIGHTS)
        trigger = weighted_choice(TRIGGER_WEIGHTS)
        priority = weighted_choice(PRIORITY_WEIGHTS)

        # Time: spread over last 30 days
        created = now - timedelta(
            days=random.uniform(0, 30),
            hours=random.uniform(0, 24),
            minutes=random.uniform(0, 60),
        )
        created_str = created.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Steps
        steps_total = random.randint(3, 12)
        if status == "completed":
            steps_completed = steps_total
        elif status == "running":
            steps_completed = random.randint(1, steps_total - 1)
        elif status == "failed":
            steps_completed = random.randint(0, steps_total - 1)
        elif status == "cancelled":
            steps_completed = random.randint(0, steps_total - 1)
        else:  # queued
            steps_completed = 0

        # Duration
        if status == "queued":
            duration = None
            completed_at = None
        elif status == "running":
            duration = round(random.uniform(2, 120), 2)
            completed_at = None
        elif status == "completed":
            duration = round(random.uniform(2, 600), 2)
            completed_at = (created + timedelta(seconds=duration)).strftime("%Y-%m-%dT%H:%M:%SZ")
        elif status == "failed":
            duration = round(random.uniform(2, 300), 2)
            completed_at = (created + timedelta(seconds=duration)).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:  # cancelled
            duration = round(random.uniform(2, 60), 2)
            completed_at = (created + timedelta(seconds=duration)).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Tokens
        if status == "queued":
            input_tokens = None
            output_tokens = None
        else:
            input_tokens = random.randint(500, 5000)
            output_tokens = random.randint(200, 3000)

        # Patient ID (some executions are not patient-specific)
        patient_id = f"EPAT{random.randint(1, 30):03d}" if random.random() < 0.75 else None

        # Parent execution (chain triggers get a parent)
        parent_execution_id = None
        if trigger == "chain" and len(execution_ids) > 5:
            parent_execution_id = random.choice(execution_ids[:-1])

        # Error message for failed
        error_message = random.choice(ERROR_MESSAGES) if status == "failed" else None

        # Retry count
        retry_count = 0
        if status == "failed":
            retry_count = random.randint(0, 3)
        elif status == "completed" and random.random() < 0.1:
            retry_count = random.randint(1, 2)

        # Metadata
        metadata = json.dumps({
            "version": f"v{random.randint(1,4)}.{random.randint(0,9)}",
            "environment": random.choice(["production", "staging", "development"]),
            "resource_group": random.choice(["gpu-pool-1", "cpu-pool-1", "cpu-pool-2"]),
        })

        rows.append((
            exec_id, agent_id, agent_name, task, mode, status, priority,
            input_tokens, output_tokens, duration, steps_total, steps_completed,
            parent_execution_id, trigger, patient_id, error_message, retry_count,
            created_str, completed_at, metadata,
        ))

    return rows


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Drop and recreate for clean seed
    cur.execute("DROP TABLE IF EXISTS openclaw_executions")
    cur.execute(CREATE_SQL)

    rows = generate_rows(200)

    cur.executemany("""
        INSERT INTO openclaw_executions (
            execution_id, agent_id, agent_name, task_description,
            execution_mode, status, priority,
            input_tokens, output_tokens, duration_seconds,
            steps_total, steps_completed,
            parent_execution_id, triggered_by, patient_id,
            error_message, retry_count, created_at, completed_at, metadata_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)

    conn.commit()

    # Verify
    count = cur.execute("SELECT COUNT(*) FROM openclaw_executions").fetchone()[0]
    print(f"Seeded openclaw_executions: {count} rows")

    # Status distribution
    for row in cur.execute("SELECT status, COUNT(*) FROM openclaw_executions GROUP BY status ORDER BY COUNT(*) DESC"):
        print(f"  {row[0]}: {row[1]}")

    # Mode distribution
    print("Execution modes:")
    for row in cur.execute("SELECT execution_mode, COUNT(*) FROM openclaw_executions GROUP BY execution_mode ORDER BY COUNT(*) DESC"):
        print(f"  {row[0]}: {row[1]}")

    # Trigger distribution
    print("Triggers:")
    for row in cur.execute("SELECT triggered_by, COUNT(*) FROM openclaw_executions GROUP BY triggered_by ORDER BY COUNT(*) DESC"):
        print(f"  {row[0]}: {row[1]}")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
