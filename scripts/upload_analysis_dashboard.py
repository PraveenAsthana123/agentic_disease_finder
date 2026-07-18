"""Upload & Analysis Tracker Dashboard — EEG upload→analysis pipeline analytics.

Tracks patient EEG file uploads, AI predictions, confidence scores, signal quality,
and analysis completion rates from the uploads and analyses tables in clinical.db.

Sources:
- uploads table (patient_id, file_name, disease, department, created_at)
- analyses table (upload_id, patient_id, disease, predicted_label, confidence,
  signal_quality, report_path, result_json, created_at)
"""

import sqlite3
import datetime
import random
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

DISEASES = ["epilepsy", "depression", "sleep_disorder", "parkinsons", "alzheimers"]
DEPARTMENTS = ["neurology", "psychiatry", "sleep_lab", "neurosurgery", "geriatrics"]
LABELS_BY_DISEASE = {
    "epilepsy": ["Epilepsy", "Normal", "Abnormal (non-epileptic)"],
    "depression": ["Depression", "Normal", "Borderline"],
    "sleep_disorder": ["Sleep Disorder", "Normal", "Insomnia"],
    "parkinsons": ["Parkinsons", "Normal", "Tremor (non-PD)"],
    "alzheimers": ["Alzheimers", "Normal", "Mild Cognitive Impairment"],
}
SIGNAL_QUALITIES = ["Excellent", "Good", "Fair", "Poor"]
FILE_TYPES = [".edf", ".csv", ".bdf", ".set", ".fif"]


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def ensure_data():
    """Seed uploads + analyses with realistic multi-disease data if count < 50."""
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM uploads")
    if cur.fetchone()[0] < 50:
        _seed(conn)
    conn.commit()
    conn.close()


def _seed(conn):
    """Seed 120 uploads + matching analyses across 5 diseases, 40 patients, 6 months."""
    cur = conn.cursor()
    rng = random.Random(77)
    patients = [f"PAT-{str(i).zfill(3)}" for i in range(1, 41)]
    base_date = datetime.datetime(2026, 1, 15, 8, 0, 0)

    upload_id_start = cur.execute("SELECT COALESCE(MAX(id),0) FROM uploads").fetchone()[0] + 1

    uploads = []
    analyses = []
    for i in range(120):
        pid = rng.choice(patients)
        disease = rng.choices(DISEASES, weights=[40, 25, 20, 10, 5], k=1)[0]
        dept = rng.choice(DEPARTMENTS)
        ext = rng.choice(FILE_TYPES)
        fname = f"{pid.lower()}_{disease}_{rng.randint(1,999):03d}{ext}"
        offset_hours = rng.randint(0, 4320)  # ~6 months in hours
        ts = base_date + datetime.timedelta(hours=offset_hours, minutes=rng.randint(0, 59))
        created = ts.strftime("%Y-%m-%dT%H:%M:%S-06:00")

        uploads.append((pid, fname, disease, dept, created))

        # 92% of uploads get an analysis (8% still processing/failed)
        if rng.random() < 0.92:
            labels = LABELS_BY_DISEASE[disease]
            label = rng.choices(labels, weights=[60, 30, 10], k=1)[0]
            confidence = round(rng.uniform(0.45, 0.98), 3)
            sq = rng.choices(SIGNAL_QUALITIES, weights=[15, 45, 25, 15], k=1)[0]
            analysis_ts = ts + datetime.timedelta(seconds=rng.randint(30, 600))
            a_created = analysis_ts.strftime("%Y-%m-%dT%H:%M:%S-06:00")
            analyses.append((i, pid, disease, label, confidence, sq, a_created))

    cur.executemany(
        "INSERT INTO uploads (patient_id, file_name, disease, department, created_at) VALUES (?,?,?,?,?)",
        uploads,
    )
    # Re-fetch actual IDs
    new_uploads = cur.execute(
        f"SELECT id FROM uploads WHERE id >= {upload_id_start} ORDER BY id"
    ).fetchall()
    for idx, (analysis_tuple) in enumerate(analyses):
        orig_idx, pid, disease, label, confidence, sq, a_created = analysis_tuple
        upload_id = new_uploads[orig_idx][0]
        cur.execute(
            "INSERT INTO analyses (upload_id, patient_id, disease, predicted_label, confidence, signal_quality, report_path, result_json, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (upload_id, pid, disease, label, confidence, sq, "", "{}", a_created),
        )


def overview():
    """Upload & analysis overview — KPIs, disease distribution, daily upload trend, signal quality."""
    ensure_data()
    conn = _conn()
    cur = conn.cursor()

    total_uploads = cur.execute("SELECT COUNT(*) FROM uploads").fetchone()[0]
    total_analyses = cur.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    distinct_patients = cur.execute("SELECT COUNT(DISTINCT patient_id) FROM uploads").fetchone()[0]
    completion_rate = round(total_analyses / total_uploads * 100, 1) if total_uploads else 0
    avg_confidence = cur.execute("SELECT AVG(confidence) FROM analyses").fetchone()[0] or 0
    avg_confidence = round(avg_confidence, 3)
    poor_signal = cur.execute("SELECT COUNT(*) FROM analyses WHERE signal_quality='Poor'").fetchone()[0]
    poor_rate = round(poor_signal / total_analyses * 100, 1) if total_analyses else 0

    # Disease distribution
    disease_dist = _dict_rows(cur.execute(
        "SELECT disease, COUNT(*) as count FROM uploads GROUP BY disease ORDER BY count DESC"
    ))

    # Daily upload trend (last 30 days of data)
    daily_trend = _dict_rows(cur.execute("""
        SELECT DATE(created_at) as day, COUNT(*) as uploads
        FROM uploads
        GROUP BY DATE(created_at)
        ORDER BY day DESC
        LIMIT 30
    """))
    daily_trend.reverse()

    # Signal quality breakdown
    sq_dist = _dict_rows(cur.execute(
        "SELECT signal_quality, COUNT(*) as count FROM analyses GROUP BY signal_quality ORDER BY count DESC"
    ))

    # Confidence distribution (buckets)
    conf_buckets = _dict_rows(cur.execute("""
        SELECT
            CASE
                WHEN confidence >= 0.9 THEN '0.9-1.0'
                WHEN confidence >= 0.8 THEN '0.8-0.9'
                WHEN confidence >= 0.7 THEN '0.7-0.8'
                WHEN confidence >= 0.6 THEN '0.6-0.7'
                ELSE '<0.6'
            END as bucket,
            COUNT(*) as count
        FROM analyses
        GROUP BY bucket
        ORDER BY bucket
    """))

    # Prediction label distribution
    label_dist = _dict_rows(cur.execute(
        "SELECT predicted_label, COUNT(*) as count FROM analyses GROUP BY predicted_label ORDER BY count DESC"
    ))

    conn.close()
    return {
        "kpis": {
            "total_uploads": total_uploads,
            "total_analyses": total_analyses,
            "distinct_patients": distinct_patients,
            "completion_rate": completion_rate,
            "avg_confidence": avg_confidence,
            "poor_signal_rate": poor_rate,
        },
        "disease_distribution": disease_dist,
        "daily_upload_trend": daily_trend,
        "signal_quality_distribution": sq_dist,
        "confidence_buckets": conf_buckets,
        "label_distribution": label_dist,
    }


def breakdown():
    """Upload & analysis breakdown — per-patient summary, recent uploads, low-confidence, pending."""
    ensure_data()
    conn = _conn()
    cur = conn.cursor()

    # Per-patient summary
    patient_summary = _dict_rows(cur.execute("""
        SELECT u.patient_id,
               COUNT(u.id) as total_uploads,
               COUNT(a.id) as completed_analyses,
               ROUND(AVG(a.confidence), 3) as avg_confidence,
               GROUP_CONCAT(DISTINCT u.disease) as diseases
        FROM uploads u
        LEFT JOIN analyses a ON a.upload_id = u.id
        GROUP BY u.patient_id
        ORDER BY total_uploads DESC
        LIMIT 20
    """))

    # Recent uploads (last 15)
    recent = _dict_rows(cur.execute("""
        SELECT u.id, u.patient_id, u.file_name, u.disease, u.department, u.created_at,
               a.predicted_label, a.confidence, a.signal_quality
        FROM uploads u
        LEFT JOIN analyses a ON a.upload_id = u.id
        ORDER BY u.created_at DESC
        LIMIT 15
    """))

    # Low confidence analyses (< 0.6)
    low_conf = _dict_rows(cur.execute("""
        SELECT a.patient_id, a.disease, a.predicted_label, a.confidence, a.signal_quality, a.created_at
        FROM analyses a
        WHERE a.confidence < 0.6
        ORDER BY a.confidence ASC
        LIMIT 15
    """))

    # Pending (uploads without analyses)
    pending = _dict_rows(cur.execute("""
        SELECT u.id, u.patient_id, u.file_name, u.disease, u.department, u.created_at
        FROM uploads u
        LEFT JOIN analyses a ON a.upload_id = u.id
        WHERE a.id IS NULL
        ORDER BY u.created_at DESC
    """))

    # Department workload
    dept_workload = _dict_rows(cur.execute("""
        SELECT department, COUNT(*) as uploads, COUNT(DISTINCT patient_id) as patients
        FROM uploads
        WHERE department != ''
        GROUP BY department
        ORDER BY uploads DESC
    """))

    conn.close()
    return {
        "patient_summary": patient_summary,
        "recent_uploads": recent,
        "low_confidence_analyses": low_conf,
        "pending_analyses": pending,
        "department_workload": dept_workload,
    }


def definitions():
    """Upload & analysis definitions — field descriptions, quality criteria, glossary."""
    return {
        "signal_quality_criteria": [
            {"level": "Excellent", "description": "SNR > 20 dB, no artifacts, all channels clean"},
            {"level": "Good", "description": "SNR 10-20 dB, minor artifacts, >90% channels usable"},
            {"level": "Fair", "description": "SNR 5-10 dB, moderate artifacts, 70-90% channels usable"},
            {"level": "Poor", "description": "SNR < 5 dB, significant artifacts, <70% channels usable"},
        ],
        "confidence_interpretation": [
            {"range": "0.9-1.0", "interpretation": "High confidence — strong biomarker signature"},
            {"range": "0.8-0.9", "interpretation": "Good confidence — clear pattern match"},
            {"range": "0.7-0.8", "interpretation": "Moderate — pattern present but ambiguous features"},
            {"range": "0.6-0.7", "interpretation": "Low-moderate — borderline, recommend review"},
            {"range": "<0.6", "interpretation": "Low confidence — insufficient evidence, manual review required"},
        ],
        "file_types": [
            {"ext": ".edf", "format": "European Data Format — standard EEG interchange"},
            {"ext": ".bdf", "format": "BioSemi Data Format — 24-bit EDF variant"},
            {"ext": ".csv", "format": "Comma-Separated Values — exported channel data"},
            {"ext": ".set", "format": "EEGLAB dataset — MATLAB-based EEG toolbox"},
            {"ext": ".fif", "format": "MNE-Python / Elekta — MEG/EEG raw data"},
        ],
        "pipeline_stages": [
            {"stage": "Upload", "description": "File received and validated (format, size, channels)"},
            {"stage": "Preprocessing", "description": "Artifact removal, filtering, re-referencing"},
            {"stage": "Feature Extraction", "description": "Spectral, temporal, connectivity features computed"},
            {"stage": "Classification", "description": "AI model predicts label + confidence score"},
            {"stage": "Report Generation", "description": "Clinical report assembled with findings"},
        ],
        "glossary": [
            {"term": "Upload", "definition": "An EEG recording file submitted for AI analysis"},
            {"term": "Analysis", "definition": "AI classification result for an uploaded EEG file"},
            {"term": "Confidence", "definition": "Model certainty in prediction (0.0-1.0 probability)"},
            {"term": "Signal Quality", "definition": "Assessment of recording clarity and artifact level"},
            {"term": "Predicted Label", "definition": "AI-assigned diagnostic classification"},
            {"term": "Completion Rate", "definition": "Percentage of uploads with finished analyses"},
            {"term": "SNR", "definition": "Signal-to-Noise Ratio — higher is cleaner signal"},
            {"term": "Artifact", "definition": "Non-brain electrical activity contaminating the EEG"},
            {"term": "Pending", "definition": "Upload awaiting analysis (processing or queued)"},
            {"term": "Department", "definition": "Clinical department that ordered the EEG study"},
        ],
    }
