"""Cognitive Tests Dashboard — digital cognitive test results from clinical.db.

Tracks patient cognitive testing: Stroop, Trail Making A/B, Digit Span,
Wisconsin Card Sorting, N-Back, Go/No-Go, Continuous Performance Test,
Clock Drawing, RAVLT, Verbal Fluency.

Sources:
- cognitive_tests table (patient_id, test_name, domain, score, max_score,
  reaction_time_ms, accuracy_pct, administered_at, administered_by, notes)
"""

import sqlite3
import datetime
import random
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

TESTS = [
    {"name": "Stroop Test", "domain": "Executive function", "max_score": 112, "typical_rt": 800},
    {"name": "Trail Making A", "domain": "Attention", "max_score": 100, "typical_rt": 35000},
    {"name": "Trail Making B", "domain": "Processing speed", "max_score": 100, "typical_rt": 75000},
    {"name": "Digit Span", "domain": "Working memory", "max_score": 30, "typical_rt": None},
    {"name": "Wisconsin Card Sorting", "domain": "Executive function", "max_score": 128, "typical_rt": None},
    {"name": "N-Back", "domain": "Working memory", "max_score": 100, "typical_rt": 550},
    {"name": "Go/No-Go", "domain": "Impulse control", "max_score": 100, "typical_rt": 420},
    {"name": "Continuous Performance Test", "domain": "Sustained attention", "max_score": 100, "typical_rt": 380},
    {"name": "Clock Drawing Test", "domain": "Visuospatial", "max_score": 10, "typical_rt": None},
    {"name": "RAVLT", "domain": "Memory", "max_score": 75, "typical_rt": None},
    {"name": "Verbal Fluency", "domain": "Language", "max_score": 60, "typical_rt": None},
]

ADMINISTRATORS = ["Dr. Patel", "Dr. Singh", "Dr. Mehta", "Neuropsych Tech", "Research Assistant"]

def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def ensure_table():
    """Create cognitive_tests table and seed data if not exists."""
    conn = _conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cognitive_tests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            test_name TEXT NOT NULL,
            domain TEXT NOT NULL,
            score REAL,
            max_score REAL,
            accuracy_pct REAL,
            reaction_time_ms REAL,
            administered_at TEXT NOT NULL,
            administered_by TEXT,
            notes TEXT
        )
    """)
    cur.execute("SELECT COUNT(*) FROM cognitive_tests")
    if cur.fetchone()[0] == 0:
        _seed(conn)
    conn.commit()
    conn.close()


def _seed(conn):
    """Seed realistic cognitive test data for 25 patients over 6 months."""
    cur = conn.cursor()
    rng = random.Random(42)
    patients = [f"EPAT{str(i).zfill(3)}" for i in range(1, 26)]
    base_date = datetime.date(2026, 1, 15)
    rows = []
    for pid in patients:
        # Each patient gets 3-6 test sessions over 6 months
        n_sessions = rng.randint(3, 6)
        session_days = sorted(rng.sample(range(0, 180), n_sessions))
        for day_offset in session_days:
            test_date = base_date + datetime.timedelta(days=day_offset)
            # Each session administers 3-6 tests
            tests_administered = rng.sample(TESTS, rng.randint(3, 6))
            admin = rng.choice(ADMINISTRATORS)
            for test in tests_administered:
                # Generate realistic scores
                max_s = test["max_score"]
                # Epilepsy patients may have lower cognitive scores
                base_pct = rng.gauss(0.68, 0.15)
                base_pct = max(0.15, min(0.98, base_pct))
                score = round(max_s * base_pct, 1)
                accuracy = round(base_pct * 100, 1)
                rt = None
                if test["typical_rt"]:
                    rt_factor = rng.gauss(1.0, 0.2)
                    rt = round(test["typical_rt"] * max(0.5, rt_factor))
                notes_options = [
                    None, None, None,  # most have no notes
                    "Patient fatigued", "Post-medication", "Pre-seizure concern",
                    "Baseline assessment", "Follow-up", "AED dose change",
                    "Good engagement", "Mild distraction noted",
                ]
                note = rng.choice(notes_options)
                rows.append((
                    pid, test["name"], test["domain"],
                    score, max_s, accuracy, rt,
                    test_date.isoformat(), admin, note
                ))
    cur.executemany(
        "INSERT INTO cognitive_tests "
        "(patient_id, test_name, domain, score, max_score, accuracy_pct, "
        "reaction_time_ms, administered_at, administered_by, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows
    )
    conn.commit()


def overview():
    """High-level cognitive test metrics."""
    ensure_table()
    conn = _conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM cognitive_tests")
    total_tests = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM cognitive_tests")
    total_patients = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT test_name) FROM cognitive_tests")
    total_test_types = cur.fetchone()[0]

    # Average accuracy
    cur.execute("SELECT ROUND(AVG(accuracy_pct), 1) FROM cognitive_tests")
    avg_accuracy = cur.fetchone()[0]

    # Test name distribution
    cur.execute(
        "SELECT test_name, COUNT(*) FROM cognitive_tests GROUP BY test_name ORDER BY COUNT(*) DESC"
    )
    test_distribution = dict(cur.fetchall())

    # Domain distribution
    cur.execute(
        "SELECT domain, COUNT(*) FROM cognitive_tests GROUP BY domain ORDER BY COUNT(*) DESC"
    )
    domain_distribution = dict(cur.fetchall())

    # Average score per test
    cur.execute(
        "SELECT test_name, ROUND(AVG(score), 1), ROUND(AVG(max_score), 0), "
        "ROUND(AVG(accuracy_pct), 1) "
        "FROM cognitive_tests GROUP BY test_name ORDER BY AVG(accuracy_pct) DESC"
    )
    test_performance = [
        {"test_name": r[0], "avg_score": r[1], "max_score": r[2], "avg_accuracy": r[3]}
        for r in cur.fetchall()
    ]

    # Monthly volume
    cur.execute(
        "SELECT SUBSTR(administered_at, 1, 7) AS month, COUNT(*) "
        "FROM cognitive_tests GROUP BY month ORDER BY month"
    )
    monthly_volume = [{"month": r[0], "count": r[1]} for r in cur.fetchall()]

    # Domain average accuracy
    cur.execute(
        "SELECT domain, ROUND(AVG(accuracy_pct), 1) "
        "FROM cognitive_tests GROUP BY domain ORDER BY AVG(accuracy_pct) DESC"
    )
    domain_accuracy = [{"domain": r[0], "avg_accuracy": r[1]} for r in cur.fetchall()]

    # Administrator distribution
    cur.execute(
        "SELECT administered_by, COUNT(*) FROM cognitive_tests "
        "WHERE administered_by IS NOT NULL GROUP BY administered_by ORDER BY COUNT(*) DESC"
    )
    admin_distribution = dict(cur.fetchall())

    conn.close()
    return {
        "total_tests": total_tests,
        "total_patients": total_patients,
        "total_test_types": total_test_types,
        "avg_accuracy": avg_accuracy,
        "test_distribution": test_distribution,
        "domain_distribution": domain_distribution,
        "test_performance": test_performance,
        "monthly_volume": monthly_volume,
        "domain_accuracy": domain_accuracy,
        "admin_distribution": admin_distribution,
    }


def breakdown():
    """Per-patient and per-test detail."""
    ensure_table()
    conn = _conn()
    cur = conn.cursor()

    # Per-patient summary
    cur.execute(
        "SELECT patient_id, COUNT(*) as total, "
        "COUNT(DISTINCT test_name) as tests_taken, "
        "ROUND(AVG(accuracy_pct), 1) as avg_accuracy, "
        "ROUND(AVG(score), 1) as avg_score, "
        "MIN(administered_at) as first_test, MAX(administered_at) as last_test "
        "FROM cognitive_tests GROUP BY patient_id ORDER BY patient_id"
    )
    per_patient = _dict_rows(cur)

    # Recent tests (last 25)
    cur.execute(
        "SELECT id, patient_id, test_name, domain, score, max_score, accuracy_pct, "
        "reaction_time_ms, administered_at, administered_by, notes "
        "FROM cognitive_tests ORDER BY administered_at DESC LIMIT 25"
    )
    recent_tests = _dict_rows(cur)

    # Low performers (avg accuracy < 55%)
    cur.execute(
        "SELECT patient_id, ROUND(AVG(accuracy_pct), 1) as avg_accuracy, COUNT(*) as total "
        "FROM cognitive_tests GROUP BY patient_id HAVING AVG(accuracy_pct) < 55 "
        "ORDER BY AVG(accuracy_pct) ASC"
    )
    low_performers = _dict_rows(cur)

    # High performers (avg accuracy >= 80%)
    cur.execute(
        "SELECT patient_id, ROUND(AVG(accuracy_pct), 1) as avg_accuracy, COUNT(*) as total "
        "FROM cognitive_tests GROUP BY patient_id HAVING AVG(accuracy_pct) >= 80 "
        "ORDER BY AVG(accuracy_pct) DESC"
    )
    high_performers = _dict_rows(cur)

    # Per-test detail with reaction times
    cur.execute(
        "SELECT test_name, domain, COUNT(*) as administrations, "
        "ROUND(AVG(accuracy_pct), 1) as avg_accuracy, "
        "ROUND(MIN(accuracy_pct), 1) as min_accuracy, "
        "ROUND(MAX(accuracy_pct), 1) as max_accuracy, "
        "ROUND(AVG(reaction_time_ms), 0) as avg_rt_ms "
        "FROM cognitive_tests GROUP BY test_name ORDER BY test_name"
    )
    test_detail = _dict_rows(cur)

    # Tests with notes (clinical observations)
    cur.execute(
        "SELECT patient_id, test_name, administered_at, notes "
        "FROM cognitive_tests WHERE notes IS NOT NULL "
        "ORDER BY administered_at DESC LIMIT 20"
    )
    clinical_notes = _dict_rows(cur)

    conn.close()
    return {
        "per_patient": per_patient,
        "recent_tests": recent_tests,
        "low_performers": low_performers,
        "high_performers": high_performers,
        "test_detail": test_detail,
        "clinical_notes": clinical_notes,
    }


def definitions():
    """Cognitive test definitions, domains, scoring, glossary."""
    return {
        "tests": [
            {
                "name": "Stroop Test",
                "domain": "Executive function",
                "description": "Measures selective attention and cognitive flexibility by naming ink colors of color-words.",
                "scoring": "Count of correct responses in 45 seconds (max ~112).",
                "clinical_relevance": "Frontal lobe dysfunction; AED cognitive side effects."
            },
            {
                "name": "Trail Making A",
                "domain": "Attention",
                "description": "Connect numbered circles in order (1-2-3...). Measures visual scanning and processing speed.",
                "scoring": "Time to completion in seconds (lower = better, max score 100).",
                "clinical_relevance": "Baseline processing speed; post-ictal cognitive assessment."
            },
            {
                "name": "Trail Making B",
                "domain": "Processing speed",
                "description": "Alternate between numbers and letters (1-A-2-B...). Measures cognitive flexibility.",
                "scoring": "Time to completion in seconds (lower = better, max score 100).",
                "clinical_relevance": "Executive function impairment; frontal/temporal lobe epilepsy."
            },
            {
                "name": "Digit Span",
                "domain": "Working memory",
                "description": "Repeat number sequences forward, backward, and in order. Measures working memory capacity.",
                "scoring": "Total correct sequences (max 30: 10 forward + 10 backward + 10 sequencing).",
                "clinical_relevance": "Working memory deficits common in temporal lobe epilepsy."
            },
            {
                "name": "Wisconsin Card Sorting",
                "domain": "Executive function",
                "description": "Match cards by color/shape/number with shifting rules. Measures set-shifting and perseveration.",
                "scoring": "Categories completed + perseverative errors (max 128 trials).",
                "clinical_relevance": "Frontal lobe function; perseveration in epilepsy."
            },
            {
                "name": "N-Back",
                "domain": "Working memory",
                "description": "Identify when current stimulus matches one N steps back. Measures working memory updating.",
                "scoring": "Percentage correct (0-100), reaction time for hits.",
                "clinical_relevance": "Working memory load; medication cognitive impact."
            },
            {
                "name": "Go/No-Go",
                "domain": "Impulse control",
                "description": "Press for 'Go' stimuli, withhold for 'No-Go'. Measures response inhibition.",
                "scoring": "Accuracy percentage (0-100), commission/omission errors.",
                "clinical_relevance": "Impulse control; frontal lobe epilepsy; AED side effects."
            },
            {
                "name": "Continuous Performance Test",
                "domain": "Sustained attention",
                "description": "Maintain vigilance over 10-15 minutes detecting infrequent targets.",
                "scoring": "Hit rate, false alarm rate, d-prime (0-100 scaled).",
                "clinical_relevance": "Attention deficits; ADHD comorbidity; medication monitoring."
            },
            {
                "name": "Clock Drawing Test",
                "domain": "Visuospatial",
                "description": "Draw a clock face showing a specific time. Quick screen for cognitive impairment.",
                "scoring": "0-10 scale (contour, numbers, hands placement).",
                "clinical_relevance": "Visuospatial/constructional ability; dementia screening."
            },
            {
                "name": "RAVLT",
                "domain": "Memory",
                "description": "Rey Auditory Verbal Learning Test — learn and recall a 15-word list over 5 trials.",
                "scoring": "Total words recalled across trials (max 75).",
                "clinical_relevance": "Verbal memory; temporal lobe epilepsy lateralization."
            },
            {
                "name": "Verbal Fluency",
                "domain": "Language",
                "description": "Generate words starting with a letter (FAS) or category (animals) in 60 seconds.",
                "scoring": "Total valid words produced (typical max ~60).",
                "clinical_relevance": "Left hemisphere function; language network integrity."
            }
        ],
        "domains": {
            "Executive function": "Planning, cognitive flexibility, inhibition, set-shifting — frontal lobe dependent.",
            "Working memory": "Holding and manipulating information — frontoparietal network.",
            "Attention": "Selective, sustained, and divided attention — frontoparietal and cingulate.",
            "Processing speed": "Speed of cognitive operations — white matter integrity marker.",
            "Impulse control": "Response inhibition and behavioral regulation — prefrontal cortex.",
            "Sustained attention": "Maintaining focus over extended periods — right hemisphere and brainstem.",
            "Visuospatial": "Spatial perception, construction, and mental rotation — parietal lobe.",
            "Memory": "Encoding, storage, and retrieval — hippocampal and temporal lobe circuits.",
            "Language": "Word retrieval, fluency, comprehension — left temporal and frontal regions."
        },
        "glossary": [
            {"term": "AED", "definition": "Anti-Epileptic Drug — medications that may affect cognitive performance."},
            {"term": "Reaction Time (RT)", "definition": "Time between stimulus onset and response (milliseconds)."},
            {"term": "Accuracy", "definition": "Percentage of correct responses out of total trials."},
            {"term": "d-prime (d')", "definition": "Signal detection sensitivity — ability to distinguish targets from non-targets."},
            {"term": "Perseveration", "definition": "Continued repetition of a response despite rule changes (WCST)."},
            {"term": "Commission Error", "definition": "False alarm — responding when no response is required."},
            {"term": "Omission Error", "definition": "Miss — failing to respond to a target stimulus."},
            {"term": "Cognitive Reserve", "definition": "Brain resilience that buffers against clinical expression of pathology."},
            {"term": "Practice Effect", "definition": "Score improvement on repeat testing due to familiarity, not true improvement."},
            {"term": "Normative Data", "definition": "Reference scores from healthy age/education-matched populations."},
            {"term": "Z-score", "definition": "Standard deviation units from the normative mean — indicates impairment severity."},
            {"term": "Post-ictal State", "definition": "Period after a seizure when cognitive function is transiently impaired."},
        ],
        "clinical_notes": {
            "aed_effects": "Topiramate, phenobarbital, and zonisamide are most associated with cognitive side effects. Lamotrigine and levetiracetam are generally better tolerated cognitively.",
            "testing_timing": "Avoid testing within 24h of a seizure (post-ictal effects). Morning testing preferred for consistency. Note time since last AED dose.",
            "epilepsy_patterns": "Temporal lobe epilepsy: memory deficits (especially verbal for left TLE). Frontal lobe epilepsy: executive function deficits. Generalized epilepsy: processing speed and attention.",
        }
    }


if __name__ == "__main__":
    import json
    ensure_table()
    print("=== Overview ===")
    print(json.dumps(overview(), indent=2))
    print("\n=== Breakdown (summary) ===")
    b = breakdown()
    print(f"Patients: {len(b['per_patient'])}, Recent: {len(b['recent_tests'])}")
    print(f"Low performers: {len(b['low_performers'])}, High: {len(b['high_performers'])}")
    print("\n=== Definitions ===")
    d = definitions()
    print(f"Tests defined: {len(d['tests'])}, Domains: {len(d['domains'])}, Glossary: {len(d['glossary'])}")
