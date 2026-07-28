"""
Neuro AI Ecosystem — Analyses Dashboard
========================================
EEG analysis results analytics from analyses table.

Fields: id, upload_id, patient_id, disease, predicted_label,
        confidence, signal_quality, report_path, result_json, created_at

Real data: analyses (129 rows, 5 diseases, 49 patients) in clinical.db.
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
    """Analyses overview — KPIs, disease/label/quality distributions, confidence
    stats by disease, daily trend, high-confidence rate, quality cross-tab."""
    conn = _conn()
    cur = conn.cursor()

    # Total analyses
    cur.execute("SELECT COUNT(*) FROM analyses")
    total_analyses = cur.fetchone()[0]

    # Unique patients
    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM analyses")
    unique_patients = cur.fetchone()[0]

    # Average confidence
    cur.execute("SELECT ROUND(AVG(confidence), 3) FROM analyses")
    avg_confidence = cur.fetchone()[0]

    # Disease count
    cur.execute("SELECT COUNT(DISTINCT disease) FROM analyses")
    disease_count = cur.fetchone()[0]

    # Date range
    cur.execute("""
        SELECT MIN(date(created_at)), MAX(date(created_at)),
               COUNT(DISTINCT date(created_at))
        FROM analyses
    """)
    row = cur.fetchone()
    date_range = {"first_date": row[0], "last_date": row[1], "active_days": row[2]}

    # Disease distribution
    cur.execute("""
        SELECT disease, COUNT(*) cnt
        FROM analyses
        GROUP BY disease
        ORDER BY cnt DESC
    """)
    disease_distribution = _dict_rows(cur)

    # Label distribution
    cur.execute("""
        SELECT predicted_label, COUNT(*) cnt
        FROM analyses
        GROUP BY predicted_label
        ORDER BY cnt DESC
    """)
    label_distribution = _dict_rows(cur)

    # Quality distribution
    cur.execute("""
        SELECT signal_quality, COUNT(*) cnt
        FROM analyses
        GROUP BY signal_quality
        ORDER BY cnt DESC
    """)
    quality_distribution = _dict_rows(cur)

    # Confidence by disease
    cur.execute("""
        SELECT disease,
               ROUND(AVG(confidence), 3) avg_confidence,
               ROUND(MIN(confidence), 3) min_confidence,
               ROUND(MAX(confidence), 3) max_confidence,
               COUNT(*) cnt
        FROM analyses
        GROUP BY disease
        ORDER BY avg_confidence DESC
    """)
    confidence_by_disease = _dict_rows(cur)

    # Daily trend
    cur.execute("""
        SELECT date(created_at) day, COUNT(*) total
        FROM analyses
        GROUP BY day
        ORDER BY day
    """)
    daily_trend = _dict_rows(cur)

    # High confidence rate (>= 0.7)
    cur.execute("""
        SELECT ROUND(100.0 * SUM(CASE WHEN confidence >= 0.7 THEN 1 ELSE 0 END) / COUNT(*), 1)
        FROM analyses
    """)
    high_confidence_rate = cur.fetchone()[0]

    # Quality by disease cross-tab
    cur.execute("""
        SELECT disease, signal_quality, COUNT(*) cnt
        FROM analyses
        GROUP BY disease, signal_quality
        ORDER BY disease, cnt DESC
    """)
    quality_by_disease = _dict_rows(cur)

    conn.close()
    return {
        "total_analyses": total_analyses,
        "unique_patients": unique_patients,
        "avg_confidence": avg_confidence,
        "disease_count": disease_count,
        "date_range": date_range,
        "disease_distribution": disease_distribution,
        "label_distribution": label_distribution,
        "quality_distribution": quality_distribution,
        "confidence_by_disease": confidence_by_disease,
        "daily_trend": daily_trend,
        "high_confidence_rate": high_confidence_rate,
        "quality_by_disease": quality_by_disease,
    }


def breakdown():
    """Analyses breakdown — all rows, per-patient summary, per-disease summary,
    confidence distribution buckets."""
    conn = _conn()
    cur = conn.cursor()

    # All analyses ordered by created_at DESC
    cur.execute("""
        SELECT id, patient_id, disease, predicted_label, confidence,
               signal_quality, created_at
        FROM analyses
        ORDER BY created_at DESC
    """)
    all_analyses = _dict_rows(cur)

    # By patient
    cur.execute("""
        SELECT patient_id,
               COUNT(*) total_analyses,
               GROUP_CONCAT(DISTINCT disease) diseases,
               ROUND(AVG(confidence), 3) avg_confidence,
               MAX(date(created_at)) latest_date
        FROM analyses
        GROUP BY patient_id
        ORDER BY total_analyses DESC
    """)
    by_patient = _dict_rows(cur)

    # By disease — build quality_breakdown dict per disease in Python
    cur.execute("""
        SELECT disease,
               COUNT(*) total,
               GROUP_CONCAT(DISTINCT predicted_label) labels,
               ROUND(AVG(confidence), 3) avg_confidence
        FROM analyses
        GROUP BY disease
        ORDER BY total DESC
    """)
    disease_rows = _dict_rows(cur)

    # Quality breakdown per disease
    cur.execute("""
        SELECT disease, signal_quality, COUNT(*) cnt
        FROM analyses
        GROUP BY disease, signal_quality
    """)
    qb_rows = _dict_rows(cur)
    qb_map = {}
    for r in qb_rows:
        qb_map.setdefault(r["disease"], {})[r["signal_quality"]] = r["cnt"]

    by_disease = []
    for d in disease_rows:
        by_disease.append({
            "disease": d["disease"],
            "total": d["total"],
            "labels": d["labels"],
            "avg_confidence": d["avg_confidence"],
            "quality_breakdown": qb_map.get(d["disease"], {}),
        })

    # Confidence distribution buckets
    cur.execute("""
        SELECT CASE
                 WHEN confidence >= 0.4 AND confidence < 0.5 THEN '0.4-0.5'
                 WHEN confidence >= 0.5 AND confidence < 0.6 THEN '0.5-0.6'
                 WHEN confidence >= 0.6 AND confidence < 0.7 THEN '0.6-0.7'
                 WHEN confidence >= 0.7 AND confidence < 0.8 THEN '0.7-0.8'
                 WHEN confidence >= 0.8 AND confidence < 0.9 THEN '0.8-0.9'
                 WHEN confidence >= 0.9               THEN '0.9-1.0'
               END bucket,
               COUNT(*) cnt
        FROM analyses
        GROUP BY bucket
        ORDER BY MIN(confidence)
    """)
    confidence_distribution = _dict_rows(cur)

    conn.close()
    return {
        "all_analyses": all_analyses,
        "by_patient": by_patient,
        "by_disease": by_disease,
        "confidence_distribution": confidence_distribution,
    }


def definitions():
    """Analyses definitions — field glossary, disease descriptions,
    signal quality levels, prediction labels, data source."""
    conn = _conn()
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT predicted_label FROM analyses ORDER BY predicted_label")
    prediction_labels = [r[0] for r in cur.fetchall()]

    conn.close()
    return {
        "field_glossary": [
            {"field": "id", "description": "Auto-increment primary key for each analysis record"},
            {"field": "upload_id", "description": "Foreign key referencing the EEG upload that triggered this analysis"},
            {"field": "patient_id", "description": "Unique patient identifier (e.g. P0001, TESTUI01)"},
            {"field": "disease", "description": "Disease category the EEG was analysed for (epilepsy, depression, alzheimers, parkinsons, sleep_disorder)"},
            {"field": "predicted_label", "description": "Model output class label (e.g. Epilepsy, Normal, Control)"},
            {"field": "confidence", "description": "Model prediction confidence score in range 0.0–1.0; higher = more certain"},
            {"field": "signal_quality", "description": "EEG signal quality assessment: Excellent / Good / Fair / Poor"},
            {"field": "report_path", "description": "Filesystem path to the generated Markdown clinical report"},
            {"field": "result_json", "description": "Full JSON payload of analysis results including features and band-power metrics"},
            {"field": "created_at", "description": "ISO-8601 timestamp when the analysis was completed"},
        ],
        "diseases": {
            "epilepsy": "Seizure disorder analysis using interictal spike and band-power features",
            "depression": "Major depressive disorder screening via EEG alpha asymmetry and theta signatures",
            "alzheimers": "Alzheimer's disease detection using slow-wave activity and connectivity markers",
            "parkinsons": "Parkinson's disease classification via beta-band power and coherence features",
            "sleep_disorder": "Sleep disorder profiling including insomnia and hypersomnia detection",
        },
        "signal_quality_levels": {
            "Excellent": "Minimal artefacts, high SNR — results most reliable",
            "Good": "Minor artefacts, acceptable SNR — results reliable for clinical use",
            "Fair": "Moderate artefacts, reduced SNR — interpret results with caution",
            "Poor": "Significant artefacts or low SNR — results may be unreliable; re-acquisition recommended",
        },
        "prediction_labels": prediction_labels,
        "data_source": "analyses table (129 EEG analysis results across 5 diseases, 49 patients)",
    }
