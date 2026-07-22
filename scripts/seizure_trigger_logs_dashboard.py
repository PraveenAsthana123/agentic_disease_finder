"""Seizure Trigger Logs Dashboard — backend analytics for seizure_trigger_logs table."""
import sqlite3, os

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')

def _conn():
    return sqlite3.connect(DB)

def overview():
    conn = _conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    total = c.execute("SELECT COUNT(*) FROM seizure_trigger_logs").fetchone()[0]
    patients = c.execute("SELECT COUNT(DISTINCT patient_id) FROM seizure_trigger_logs").fetchone()[0]
    seizures = c.execute("SELECT COUNT(*) FROM seizure_trigger_logs WHERE seizure_occurred=1").fetchone()[0]
    seizure_rate = round(seizures / total * 100, 1) if total else 0
    avg_sleep = c.execute("SELECT ROUND(AVG(sleep_hours),1) FROM seizure_trigger_logs").fetchone()[0]
    avg_stress = c.execute("SELECT ROUND(AVG(stress_level),1) FROM seizure_trigger_logs").fetchone()[0]
    avg_caffeine = c.execute("SELECT ROUND(AVG(caffeine_mg),0) FROM seizure_trigger_logs").fetchone()[0]
    avg_alcohol = c.execute("SELECT ROUND(AVG(alcohol_units),1) FROM seizure_trigger_logs").fetchone()[0]
    adherence_rate = c.execute("SELECT ROUND(AVG(medication_adherence)*100,1) FROM seizure_trigger_logs").fetchone()[0]
    avg_duration = c.execute("SELECT ROUND(AVG(seizure_duration_sec),0) FROM seizure_trigger_logs WHERE seizure_occurred=1").fetchone()[0]

    trigger_dist = [dict(r) for r in c.execute(
        "SELECT primary_trigger AS trigger, COUNT(*) AS count FROM seizure_trigger_logs GROUP BY primary_trigger ORDER BY count DESC")]

    sleep_quality_dist = [dict(r) for r in c.execute(
        "SELECT sleep_quality AS quality, COUNT(*) AS count FROM seizure_trigger_logs GROUP BY sleep_quality ORDER BY count DESC")]

    seizure_type_dist = [dict(r) for r in c.execute(
        "SELECT seizure_type AS type, COUNT(*) AS count FROM seizure_trigger_logs WHERE seizure_type IS NOT NULL GROUP BY seizure_type ORDER BY count DESC")]

    # Seizure rate by trigger
    trigger_seizure_rate = [dict(r) for r in c.execute("""
        SELECT primary_trigger AS trigger,
               COUNT(*) AS total,
               SUM(seizure_occurred) AS seizures,
               ROUND(SUM(seizure_occurred)*100.0/COUNT(*),1) AS rate
        FROM seizure_trigger_logs GROUP BY primary_trigger ORDER BY rate DESC
    """)]

    # Seizure rate by sleep quality
    sleep_vs_seizure = [dict(r) for r in c.execute("""
        SELECT sleep_quality AS quality,
               COUNT(*) AS total,
               SUM(seizure_occurred) AS seizures,
               ROUND(SUM(seizure_occurred)*100.0/COUNT(*),1) AS rate
        FROM seizure_trigger_logs GROUP BY sleep_quality
        ORDER BY CASE sleep_quality WHEN 'very_poor' THEN 1 WHEN 'poor' THEN 2 WHEN 'fair' THEN 3 WHEN 'good' THEN 4 END
    """)]

    # Monthly trend
    monthly_trend = [dict(r) for r in c.execute("""
        SELECT SUBSTR(log_date,1,7) AS month,
               COUNT(*) AS total_logs,
               SUM(seizure_occurred) AS seizures,
               ROUND(AVG(sleep_hours),1) AS avg_sleep,
               ROUND(AVG(stress_level),1) AS avg_stress
        FROM seizure_trigger_logs GROUP BY month ORDER BY month
    """)]

    # Risk factor averages: seizure vs no-seizure
    risk_comparison = []
    for col, label in [('sleep_hours','Sleep Hours'),('stress_level','Stress Level'),
                       ('fatigue_level','Fatigue Level'),('caffeine_mg','Caffeine (mg)'),
                       ('alcohol_units','Alcohol Units'),('screen_time_hours','Screen Time (h)')]:
        row_sz = c.execute(f"SELECT ROUND(AVG({col}),1) FROM seizure_trigger_logs WHERE seizure_occurred=1").fetchone()[0]
        row_no = c.execute(f"SELECT ROUND(AVG({col}),1) FROM seizure_trigger_logs WHERE seizure_occurred=0").fetchone()[0]
        risk_comparison.append({"factor": label, "with_seizure": row_sz, "without_seizure": row_no})

    conn.close()
    return {
        "total_logs": total,
        "total_patients": patients,
        "seizure_count": seizures,
        "seizure_rate": seizure_rate,
        "avg_sleep_hours": avg_sleep,
        "avg_stress_level": avg_stress,
        "avg_caffeine_mg": avg_caffeine,
        "avg_alcohol_units": avg_alcohol,
        "medication_adherence_rate": adherence_rate,
        "avg_seizure_duration_sec": avg_duration,
        "trigger_distribution": trigger_dist,
        "sleep_quality_distribution": sleep_quality_dist,
        "seizure_type_distribution": seizure_type_dist,
        "trigger_seizure_rate": trigger_seizure_rate,
        "sleep_vs_seizure": sleep_vs_seizure,
        "monthly_trend": monthly_trend,
        "risk_comparison": risk_comparison,
    }

def breakdown():
    conn = _conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    logs = [dict(r) for r in c.execute(
        "SELECT * FROM seizure_trigger_logs ORDER BY log_date DESC")]

    # Per-patient summary
    patient_summary = [dict(r) for r in c.execute("""
        SELECT patient_id,
               COUNT(*) AS total_logs,
               SUM(seizure_occurred) AS seizures,
               ROUND(AVG(sleep_hours),1) AS avg_sleep,
               ROUND(AVG(stress_level),1) AS avg_stress,
               ROUND(AVG(medication_adherence)*100,1) AS adherence_pct,
               MAX(log_date) AS latest_log
        FROM seizure_trigger_logs GROUP BY patient_id ORDER BY seizures DESC
    """)]

    # Top triggers per patient (patients with most seizures)
    top_patient_triggers = [dict(r) for r in c.execute("""
        SELECT patient_id, primary_trigger, COUNT(*) AS count
        FROM seizure_trigger_logs WHERE seizure_occurred=1
        GROUP BY patient_id, primary_trigger
        ORDER BY count DESC LIMIT 20
    """)]

    conn.close()
    return {
        "logs": logs,
        "patient_summary": patient_summary,
        "top_patient_triggers": top_patient_triggers,
    }

def definitions():
    return {
        "title": "Seizure Trigger Logs Dashboard — Definitions",
        "concepts": [
            {"name": "Primary Trigger", "description": "The main identified factor that may have contributed to or precipitated a seizure event. Common triggers include sleep deprivation, stress, photosensitivity, missed medication, alcohol, hormonal changes, illness, fatigue, and dehydration. Trigger identification is essential for seizure self-management and personalized action plans."},
            {"name": "Seizure Rate", "description": "Percentage of logged days on which a seizure event was recorded. Helps track seizure control over time and in response to lifestyle modifications."},
            {"name": "Sleep Quality", "description": "Self-reported sleep quality categorized as good, fair, poor, or very poor. Poor sleep quality is one of the strongest modifiable seizure triggers in epilepsy."},
            {"name": "Stress Level", "description": "Self-reported stress on a 1-10 scale. Chronic and acute stress are recognized seizure precipitants, mediated by cortisol and hypothalamic-pituitary-adrenal axis activation."},
            {"name": "Medication Adherence", "description": "Whether the patient took all prescribed antiepileptic medications (1=yes, 0=no). Non-adherence is the most common preventable cause of breakthrough seizures."},
            {"name": "Caffeine (mg)", "description": "Daily caffeine intake in milligrams. High caffeine intake (>300mg) may lower seizure threshold in some patients, though evidence is mixed."},
            {"name": "Alcohol Units", "description": "Daily alcohol consumption in standard units. Alcohol is a well-established seizure trigger, particularly during withdrawal. Even moderate intake can increase seizure risk."},
            {"name": "Screen Time", "description": "Daily screen exposure in hours. Relevant for photosensitive epilepsy patients, where flickering screens or specific visual patterns may provoke seizures."},
            {"name": "Risk Factor Comparison", "description": "Compares average lifestyle metrics (sleep, stress, caffeine, etc.) between days with and without seizures, helping identify which factors most strongly associate with seizure occurrence."},
        ],
        "seizure_types": [
            {"type": "focal aware", "description": "Seizure originating in one brain area with preserved consciousness. May involve motor, sensory, or autonomic symptoms."},
            {"type": "focal impaired awareness", "description": "Seizure originating in one brain area with impaired consciousness. Previously called complex partial seizure."},
            {"type": "focal to bilateral tonic-clonic", "description": "Seizure starting focally then spreading to both hemispheres, causing a generalized tonic-clonic convulsion."},
            {"type": "generalized tonic-clonic", "description": "Seizure involving both hemispheres from onset with tonic stiffening followed by clonic jerking."},
            {"type": "absence", "description": "Brief generalized seizure characterized by staring and unresponsiveness, typically lasting 5-30 seconds."},
            {"type": "myoclonic", "description": "Brief, shock-like involuntary jerks of a muscle or group of muscles, typically lasting 1-2 seconds."},
        ],
        "data_sources": [
            "seizure_trigger_logs table — 203 daily patient logs, 40 patients",
            "Self-reported data collected via seizure diary module",
            "Medication adherence from automated pill-tracking or self-report",
        ],
    }

if __name__ == "__main__":
    import json
    print(json.dumps(overview(), indent=2))
