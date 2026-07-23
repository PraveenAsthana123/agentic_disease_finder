"""EEG Analysis Results Dashboard — backend analytics for analyses table."""
import sqlite3, os

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')

def _conn():
    return sqlite3.connect(DB)

def overview():
    conn = _conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    total_analyses = c.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    total_patients = c.execute("SELECT COUNT(DISTINCT patient_id) FROM analyses").fetchone()[0]
    diseases_covered = c.execute("SELECT COUNT(DISTINCT disease) FROM analyses").fetchone()[0]
    avg_confidence = c.execute("SELECT ROUND(AVG(confidence),2) FROM analyses").fetchone()[0]
    high_confidence_count = c.execute("SELECT COUNT(*) FROM analyses WHERE confidence >= 0.8").fetchone()[0]
    low_confidence_count = c.execute("SELECT COUNT(*) FROM analyses WHERE confidence < 0.5").fetchone()[0]

    excellent_count = c.execute("SELECT COUNT(*) FROM analyses WHERE signal_quality='Excellent'").fetchone()[0]
    signal_quality_excellent_pct = round(excellent_count / total_analyses * 100, 1) if total_analyses else 0

    poor_count = c.execute("SELECT COUNT(*) FROM analyses WHERE signal_quality='Poor'").fetchone()[0]
    signal_quality_poor_pct = round(poor_count / total_analyses * 100, 1) if total_analyses else 0

    disease_dist = [dict(r) for r in c.execute(
        "SELECT disease, COUNT(*) AS count FROM analyses GROUP BY disease ORDER BY count DESC")]

    label_dist = [dict(r) for r in c.execute(
        "SELECT predicted_label AS label, COUNT(*) AS count FROM analyses GROUP BY predicted_label ORDER BY count DESC")]

    quality_dist = [dict(r) for r in c.execute(
        "SELECT signal_quality AS quality, COUNT(*) AS count FROM analyses GROUP BY signal_quality ORDER BY count DESC")]

    confidence_tiers = [dict(r) for r in c.execute("""
        SELECT CASE
            WHEN confidence >= 0.8 THEN 'High'
            WHEN confidence >= 0.6 THEN 'Medium'
            ELSE 'Low'
        END AS tier,
        COUNT(*) AS count
        FROM analyses GROUP BY tier ORDER BY count DESC
    """)]

    monthly_trend = [dict(r) for r in c.execute("""
        SELECT SUBSTR(created_at,1,7) AS month,
               COUNT(*) AS analyses,
               ROUND(AVG(confidence),2) AS avg_confidence
        FROM analyses GROUP BY month ORDER BY month
    """)]

    disease_confidence = [dict(r) for r in c.execute("""
        SELECT disease,
               ROUND(AVG(confidence),2) AS avg_confidence,
               ROUND(MIN(confidence),2) AS min_confidence,
               ROUND(MAX(confidence),2) AS max_confidence,
               COUNT(*) AS count
        FROM analyses GROUP BY disease ORDER BY avg_confidence DESC
    """)]

    conn.close()
    return {
        "kpis": {
            "total_analyses": total_analyses,
            "total_patients": total_patients,
            "diseases_covered": diseases_covered,
            "avg_confidence": avg_confidence,
            "high_confidence_count": high_confidence_count,
            "low_confidence_count": low_confidence_count,
            "signal_quality_excellent_pct": signal_quality_excellent_pct,
            "signal_quality_poor_pct": signal_quality_poor_pct,
        },
        "disease_dist": disease_dist,
        "label_dist": label_dist,
        "quality_dist": quality_dist,
        "confidence_tiers": confidence_tiers,
        "monthly_trend": monthly_trend,
        "disease_confidence": disease_confidence,
    }

def breakdown():
    conn = _conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    analyses = [dict(r) for r in c.execute(
        "SELECT id, upload_id, patient_id, disease, predicted_label, confidence, signal_quality, created_at "
        "FROM analyses ORDER BY created_at DESC LIMIT 200")]

    by_patient = [dict(r) for r in c.execute("""
        SELECT patient_id,
               COUNT(*) AS analyses,
               COUNT(DISTINCT disease) AS diseases,
               ROUND(AVG(confidence),2) AS avg_confidence,
               MAX(created_at) AS last_analysis
        FROM analyses GROUP BY patient_id ORDER BY analyses DESC
    """)]

    by_disease = [dict(r) for r in c.execute("""
        SELECT disease,
               COUNT(*) AS total,
               GROUP_CONCAT(DISTINCT predicted_label) AS labels,
               ROUND(AVG(confidence),2) AS avg_confidence,
               ROUND(AVG(CASE signal_quality
                   WHEN 'Excellent' THEN 4
                   WHEN 'Good' THEN 3
                   WHEN 'Fair' THEN 2
                   WHEN 'Poor' THEN 1
                   ELSE 0 END),2) AS avg_quality_score,
               MAX(created_at) AS latest
        FROM analyses GROUP BY disease ORDER BY total DESC
    """)]

    conn.close()
    return {
        "analyses": analyses,
        "by_patient": by_patient,
        "by_disease": by_disease,
    }

def definitions():
    return {
        "fields": {
            "id": "Unique analysis record identifier (auto-increment).",
            "upload_id": "Reference to the uploaded EEG file that was analysed.",
            "patient_id": "Anonymised patient identifier linking to the patient registry.",
            "disease": "Target disease the model was predicting (e.g. epilepsy, depression).",
            "predicted_label": "Classification label output by the ML model for this EEG recording.",
            "confidence": "Model confidence score between 0 and 1 for the predicted label.",
            "signal_quality": "Quality grade of the raw EEG signal: Excellent, Good, Fair, or Poor.",
            "report_path": "File-system path to the generated PDF/HTML analysis report.",
            "result_json": "Full JSON payload of model outputs including per-channel metrics.",
            "created_at": "Timestamp when the analysis was executed and stored.",
        },
        "diseases": [
            {"name": "epilepsy", "description": "Seizure detection and classification from EEG patterns."},
            {"name": "sleep_disorder", "description": "Sleep-stage anomalies and disorder markers from overnight EEG."},
            {"name": "depression", "description": "Frontal-alpha asymmetry and biomarkers associated with depressive disorders."},
            {"name": "parkinsons", "description": "Beta-band and motor-cortex signatures linked to Parkinson's disease."},
            {"name": "alzheimers", "description": "Theta/delta slowing and coherence loss indicative of Alzheimer's disease."},
        ],
        "signal_quality_levels": {
            "Excellent": "Minimal artefacts, high SNR — ideal for clinical-grade analysis.",
            "Good": "Acceptable artefact level, reliable predictions expected.",
            "Fair": "Moderate artefacts present; predictions usable but should be reviewed.",
            "Poor": "Significant artefacts or low SNR; predictions may be unreliable.",
        },
        "confidence_interpretation": {
            "High (>= 0.8)": "Strong model certainty — prediction is highly reliable.",
            "Medium (0.6 - 0.8)": "Moderate certainty — clinician review recommended.",
            "Low (< 0.6)": "Low certainty — treat as provisional; manual review required.",
        },
    }


if __name__ == "__main__":
    import json
    print("=== Overview ===")
    print(json.dumps(overview(), indent=2, default=str))
    print("\n=== Breakdown ===")
    b = breakdown()
    print(f"  analyses rows: {len(b['analyses'])}")
    print(f"  by_patient rows: {len(b['by_patient'])}")
    print(f"  by_disease rows: {len(b['by_disease'])}")
    print("\n=== Definitions ===")
    print(json.dumps(definitions(), indent=2))
