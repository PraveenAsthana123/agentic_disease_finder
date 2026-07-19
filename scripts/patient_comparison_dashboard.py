"""Patient Comparison Dashboard — side-by-side comparison of two patients."""
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    return sqlite3.connect(str(DB))


def _row_dicts(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def patient_list():
    """Return list of patients available for comparison (those with demographics)."""
    conn = _conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT patient_id, full_name, age, sex, epilepsy_type
        FROM patient_demographics
        ORDER BY patient_id
    """)
    rows = _row_dicts(cur)
    conn.close()
    return {"patients": rows, "total": len(rows)}


def _get_demographics(cur, pid):
    cur.execute("SELECT * FROM patient_demographics WHERE patient_id = ?", (pid,))
    rows = _row_dicts(cur)
    return rows[0] if rows else {}


def _get_seizure_summary(cur, pid):
    cur.execute("""
        SELECT COUNT(*) as total_events,
               AVG(duration_sec) as avg_duration_sec,
               MAX(severity) as max_severity,
               SUM(CASE WHEN er_visit = 1 THEN 1 ELSE 0 END) as er_visits,
               SUM(CASE WHEN injury = 1 THEN 1 ELSE 0 END) as injuries
        FROM seizure_diary WHERE patient_id = ?
    """, (pid,))
    rows = _row_dicts(cur)
    return rows[0] if rows else {}


def _get_seizure_triggers(cur, pid):
    cur.execute("""
        SELECT trigger, COUNT(*) as count
        FROM seizure_diary
        WHERE patient_id = ? AND trigger IS NOT NULL AND trigger != ''
        GROUP BY trigger ORDER BY count DESC LIMIT 5
    """, (pid,))
    return _row_dicts(cur)


def _get_assessment_summary(cur, pid):
    cur.execute("""
        SELECT instrument, score, max_score, interpretation, level
        FROM assessments
        WHERE patient_id = ?
        ORDER BY created_at DESC
    """, (pid,))
    return _row_dicts(cur)


def _get_cognitive_summary(cur, pid):
    cur.execute("""
        SELECT domain,
               AVG(accuracy_pct) as avg_accuracy,
               AVG(reaction_time_ms) as avg_reaction_ms,
               COUNT(*) as test_count
        FROM cognitive_tests
        WHERE patient_id = ?
        GROUP BY domain
        ORDER BY domain
    """, (pid,))
    return _row_dicts(cur)


def _get_medication_adherence(cur, pid):
    cur.execute("""
        SELECT ROUND(AVG(CASE WHEN taken IN ('yes','late') THEN 100.0 ELSE 0.0 END), 1) as avg_adherence,
               COUNT(*) as total_records,
               COUNT(DISTINCT drug_name) as medications,
               SUM(CASE WHEN taken = 'yes' THEN 1 ELSE 0 END) as doses_taken,
               SUM(CASE WHEN taken = 'late' THEN 1 ELSE 0 END) as doses_late,
               SUM(CASE WHEN taken = 'no' THEN 1 ELSE 0 END) as doses_missed
        FROM medication_adherence
        WHERE patient_id = ?
    """, (pid,))
    rows = _row_dicts(cur)
    return rows[0] if rows else {}


def _get_analysis_summary(cur, pid):
    cur.execute("""
        SELECT disease, predicted_label, AVG(confidence) as avg_confidence,
               COUNT(*) as total_analyses
        FROM analyses
        WHERE patient_id = ?
        GROUP BY disease, predicted_label
    """, (pid,))
    return _row_dicts(cur)


def overview():
    """Overview — patient list + global stats for comparison context."""
    conn = _conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM patient_demographics")
    total_patients = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM seizure_diary")
    total_seizures = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM assessments")
    total_assessments = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM cognitive_tests")
    total_cognitive = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM medication_adherence")
    total_med_records = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM analyses")
    total_analyses = cur.fetchone()[0]

    # Patient list for selection
    plist = patient_list()

    conn.close()
    return {
        "kpis": {
            "total_patients": total_patients,
            "total_seizure_events": total_seizures,
            "total_assessments": total_assessments,
            "total_cognitive_tests": total_cognitive,
            "total_med_records": total_med_records,
            "total_analyses": total_analyses,
        },
        "patients": plist["patients"],
    }


def compare(patient_a, patient_b):
    """Side-by-side comparison of two patients across all domains."""
    conn = _conn()
    cur = conn.cursor()

    result = {}
    for label, pid in [("patient_a", patient_a), ("patient_b", patient_b)]:
        demo = _get_demographics(cur, pid)
        seizure = _get_seizure_summary(cur, pid)
        triggers = _get_seizure_triggers(cur, pid)
        assessments = _get_assessment_summary(cur, pid)
        cognitive = _get_cognitive_summary(cur, pid)
        medication = _get_medication_adherence(cur, pid)
        analyses = _get_analysis_summary(cur, pid)

        result[label] = {
            "patient_id": pid,
            "demographics": demo,
            "seizure_summary": seizure,
            "seizure_triggers": triggers,
            "assessments": assessments,
            "cognitive_domains": cognitive,
            "medication_adherence": medication,
            "analyses": analyses,
        }

    # Build comparison radar data (normalized 0-100)
    radar_dims = []
    for dim_name, getter_a, getter_b in [
        ("Seizure Frequency", result["patient_a"]["seizure_summary"].get("total_events", 0),
         result["patient_b"]["seizure_summary"].get("total_events", 0)),
        ("Medication Adherence", result["patient_a"]["medication_adherence"].get("avg_adherence", 0),
         result["patient_b"]["medication_adherence"].get("avg_adherence", 0)),
        ("Assessment Count", len(result["patient_a"]["assessments"]),
         len(result["patient_b"]["assessments"])),
        ("Cognitive Tests", sum(d.get("test_count", 0) for d in result["patient_a"]["cognitive_domains"]),
         sum(d.get("test_count", 0) for d in result["patient_b"]["cognitive_domains"])),
        ("EEG Analyses", sum(d.get("total_analyses", 0) for d in result["patient_a"]["analyses"]),
         sum(d.get("total_analyses", 0) for d in result["patient_b"]["analyses"])),
    ]:
        max_val = max(getter_a or 0, getter_b or 0, 1)
        radar_dims.append({
            "dimension": dim_name,
            "patient_a": round(((getter_a or 0) / max_val) * 100, 1),
            "patient_b": round(((getter_b or 0) / max_val) * 100, 1),
        })

    result["radar_comparison"] = radar_dims
    conn.close()
    return result


def definitions():
    """Definitions — glossary, comparison dimensions, clinical notes."""
    return {
        "comparison_dimensions": [
            {"dimension": "Demographics", "description": "Age, sex, epilepsy type, onset age, years with epilepsy"},
            {"dimension": "Seizure Profile", "description": "Total events, average duration, ER visits, injuries, triggers"},
            {"dimension": "Assessments", "description": "Clinical instruments (PHQ-9, GAD-7, QOLIE-31, etc.) with scores and interpretations"},
            {"dimension": "Cognitive Function", "description": "Domain-level accuracy and reaction time from neuropsych testing"},
            {"dimension": "Medication Adherence", "description": "Average adherence percentage, min/max range"},
            {"dimension": "EEG Analysis", "description": "AI predictions, confidence scores, disease classification"},
        ],
        "glossary": [
            {"term": "LOSO", "definition": "Leave-One-Subject-Out — cross-validation where each fold excludes one patient"},
            {"term": "PHQ-9", "definition": "Patient Health Questionnaire-9 — depression screening (0-27)"},
            {"term": "GAD-7", "definition": "Generalized Anxiety Disorder-7 — anxiety screening (0-21)"},
            {"term": "QOLIE-31", "definition": "Quality of Life in Epilepsy — 31-item instrument"},
            {"term": "Adherence %", "definition": "Percentage of prescribed medication doses taken as scheduled"},
            {"term": "Confidence", "definition": "Model's estimated probability for the predicted class (0.0-1.0)"},
            {"term": "Cognitive Domain", "definition": "Category of cognitive function (memory, attention, executive, language, visuospatial)"},
            {"term": "Trigger", "definition": "Factor identified by patient as preceding seizure (stress, sleep deprivation, etc.)"},
        ],
        "clinical_notes": [
            "Patient comparison is for clinical decision support only — not diagnostic",
            "Radar chart normalizes to the higher of the two patients per dimension",
            "Assessment scores are latest per instrument — historical trends require longitudinal view",
            "Medication adherence is averaged across all recorded days",
            "EEG confidence reflects model certainty, not clinical ground truth",
        ],
    }
