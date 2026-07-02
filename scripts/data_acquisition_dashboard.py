"""Data Acquisition Dashboard — ingestion analytics from clinical.db.

Tracks file uploads, EEG acquisition pipeline, format coverage, patient
ingestion coverage, signal quality distribution, and analysis completion
rates across the clinical EEG/epilepsy platform.

Sources:
- uploads table (21+ files across multiple patients)
- analyses table (21+ analyses with signal quality + confidence)
- transaction_log table (eeg_upload + ingest actions)
- patients table (40 patients, coverage tracking)
"""

import sqlite3
import os
from collections import defaultdict

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql, params=(), default=0):
    try:
        cur.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else default
    except Exception:
        return default


def _safe_rows(cur, sql, params=()):
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    except Exception:
        return []


# ── File format classification ──
FORMAT_MAP = {
    '.edf': 'EDF (European Data Format)',
    '.csv': 'CSV (Comma-Separated Values)',
    '.set': 'EEGLAB .set',
    '.fif': 'MNE-Python .fif',
    '.bdf': 'BDF (BioSemi)',
    '.gdf': 'GDF (General Data Format)',
    '.mat': 'MATLAB .mat',
}


def _classify_format(filename):
    if not filename:
        return 'Unknown'
    fn = filename.lower()
    for ext, label in FORMAT_MAP.items():
        if fn.endswith(ext):
            return label
    return 'Other'


def data_acquisition_overview():
    """Overview KPIs + trends for the data acquisition pipeline."""
    conn = _conn()
    cur = conn.cursor()
    try:
        # ── KPIs ──
        total_uploads = _safe(cur, "SELECT COUNT(*) FROM uploads")
        total_patients_uploaded = _safe(cur, "SELECT COUNT(DISTINCT patient_id) FROM uploads")
        total_patients = _safe(cur, "SELECT COUNT(*) FROM patients")
        total_analyses = _safe(cur, "SELECT COUNT(*) FROM analyses")
        avg_confidence = _safe(cur, "SELECT AVG(confidence) FROM analyses", default=0.0)
        good_quality = _safe(cur, "SELECT COUNT(*) FROM analyses WHERE signal_quality='Good'")
        quality_rate = round(good_quality / total_analyses * 100, 1) if total_analyses else 0
        eeg_upload_events = _safe(cur, "SELECT COUNT(*) FROM transaction_log WHERE component='eeg_upload'")
        ingest_events = _safe(cur, "SELECT COUNT(*) FROM transaction_log WHERE action='ingest'")
        unique_files = _safe(cur, "SELECT COUNT(DISTINCT file_name) FROM uploads")
        coverage_pct = round(total_patients_uploaded / total_patients * 100, 1) if total_patients else 0

        # ── Format distribution ──
        rows = _safe_rows(cur, "SELECT file_name FROM uploads")
        fmt_counts = defaultdict(int)
        for (fn,) in rows:
            fmt_counts[_classify_format(fn)] += 1
        format_dist = [{'format': k, 'count': v} for k, v in sorted(fmt_counts.items(), key=lambda x: -x[1])]

        # ── Signal quality distribution ──
        sq_rows = _safe_rows(cur, "SELECT signal_quality, COUNT(*) FROM analyses GROUP BY signal_quality ORDER BY COUNT(*) DESC")
        quality_dist = [{'quality': r[0] or 'Unknown', 'count': r[1]} for r in sq_rows]

        # ── Daily upload trend ──
        daily_rows = _safe_rows(cur, """
            SELECT SUBSTR(created_at, 1, 10) AS day, COUNT(*)
            FROM uploads GROUP BY day ORDER BY day
        """)
        daily_trend = [{'date': r[0], 'uploads': r[1]} for r in daily_rows]

        # ── Confidence distribution (buckets) ──
        conf_buckets = [
            ('<50%', 0, 0.5),
            ('50-60%', 0.5, 0.6),
            ('60-70%', 0.6, 0.7),
            ('70-80%', 0.7, 0.8),
            ('80-90%', 0.8, 0.9),
            ('90-100%', 0.9, 1.01),
        ]
        conf_dist = []
        for label, lo, hi in conf_buckets:
            cnt = _safe(cur, "SELECT COUNT(*) FROM analyses WHERE confidence >= ? AND confidence < ?", (lo, hi))
            conf_dist.append({'bucket': label, 'count': cnt})

        # ── Hourly activity pattern ──
        hourly_rows = _safe_rows(cur, """
            SELECT CAST(SUBSTR(created_at, 12, 2) AS INTEGER) AS hour, COUNT(*)
            FROM uploads GROUP BY hour ORDER BY hour
        """)
        hourly = [{'hour': r[0], 'count': r[1]} for r in hourly_rows]

        return {
            'available': True,
            'kpis': {
                'total_uploads': total_uploads,
                'unique_files': unique_files,
                'patients_with_data': total_patients_uploaded,
                'total_patients': total_patients,
                'coverage_pct': coverage_pct,
                'total_analyses': total_analyses,
                'avg_confidence': round(avg_confidence, 3),
                'signal_quality_rate': quality_rate,
                'eeg_upload_events': eeg_upload_events,
                'ingest_events': ingest_events,
            },
            'format_distribution': format_dist,
            'quality_distribution': quality_dist,
            'daily_trend': daily_trend,
            'confidence_distribution': conf_dist,
            'hourly_pattern': hourly,
        }
    finally:
        conn.close()


def data_acquisition_breakdown():
    """Detailed breakdown — per-patient uploads, analysis results, recent activity."""
    conn = _conn()
    cur = conn.cursor()
    try:
        # ── Per-patient upload profiles ──
        patient_rows = _safe_rows(cur, """
            SELECT u.patient_id,
                   COUNT(DISTINCT u.id) AS upload_count,
                   COUNT(DISTINCT u.file_name) AS unique_files,
                   GROUP_CONCAT(DISTINCT u.file_name) AS files,
                   MAX(u.created_at) AS last_upload
            FROM uploads u
            GROUP BY u.patient_id
            ORDER BY upload_count DESC
        """)
        patient_profiles = []
        for r in patient_rows:
            pid = r[0]
            # Get analysis info for this patient
            analysis_rows = _safe_rows(cur, """
                SELECT predicted_label, confidence, signal_quality
                FROM analyses WHERE patient_id = ? ORDER BY id DESC LIMIT 1
            """, (pid,))
            analysis = None
            if analysis_rows:
                a = analysis_rows[0]
                analysis = {'prediction': a[0], 'confidence': a[1], 'quality': a[2]}
            patient_profiles.append({
                'patient_id': pid,
                'uploads': r[1],
                'unique_files': r[2],
                'files': (r[3] or '').split(',')[:5],
                'last_upload': r[4],
                'latest_analysis': analysis,
            })

        # ── Analysis results table (recent 20) ──
        analysis_rows = _safe_rows(cur, """
            SELECT a.patient_id, a.disease, a.predicted_label, a.confidence,
                   a.signal_quality, u.file_name, a.created_at
            FROM analyses a
            LEFT JOIN uploads u ON a.upload_id = u.id
            ORDER BY a.id DESC LIMIT 20
        """)
        recent_analyses = [{
            'patient_id': r[0], 'disease': r[1], 'prediction': r[2],
            'confidence': r[3], 'quality': r[4], 'file': r[5], 'timestamp': r[6]
        } for r in analysis_rows]

        # ── File type breakdown with avg confidence ──
        file_rows = _safe_rows(cur, """
            SELECT u.file_name, AVG(a.confidence), COUNT(a.id)
            FROM uploads u
            LEFT JOIN analyses a ON a.upload_id = u.id
            GROUP BY u.file_name
            ORDER BY COUNT(a.id) DESC
        """)
        file_analysis = []
        for r in file_rows:
            file_analysis.append({
                'file': r[0],
                'format': _classify_format(r[0]),
                'avg_confidence': round(r[1], 3) if r[1] else None,
                'analysis_count': r[2],
            })

        # ── Transaction log — recent acquisition events ──
        tx_rows = _safe_rows(cur, """
            SELECT patient_id, component, action, actor, detail,
                   ts_local
            FROM transaction_log
            WHERE component = 'eeg_upload' OR action = 'ingest'
            ORDER BY id DESC LIMIT 30
        """)
        recent_events = [{
            'patient_id': r[0], 'component': r[1], 'action': r[2],
            'actor': r[3], 'detail': r[4], 'timestamp': r[5]
        } for r in tx_rows]

        # ── Department distribution ──
        dept_rows = _safe_rows(cur, """
            SELECT CASE WHEN department = '' THEN 'Unspecified' ELSE department END,
                   COUNT(*)
            FROM uploads GROUP BY department ORDER BY COUNT(*) DESC
        """)
        dept_dist = [{'department': r[0], 'count': r[1]} for r in dept_rows]

        # ── Disease coverage ──
        disease_rows = _safe_rows(cur, """
            SELECT disease, COUNT(*) FROM uploads GROUP BY disease ORDER BY COUNT(*) DESC
        """)
        disease_dist = [{'disease': r[0], 'count': r[1]} for r in disease_rows]

        return {
            'available': True,
            'patient_profiles': patient_profiles,
            'recent_analyses': recent_analyses,
            'file_analysis': file_analysis,
            'recent_events': recent_events,
            'department_distribution': dept_dist,
            'disease_distribution': disease_dist,
        }
    finally:
        conn.close()


def data_acquisition_definitions():
    """Data acquisition metric definitions for tooltip overlays."""
    return {
        'available': True,
        'definitions': {
            'acquisition_concepts': {
                'Data Ingestion': 'The process of importing raw EEG recordings (EDF, CSV, or other formats) into the clinical platform — each upload creates a record in the uploads table and triggers automated analysis.',
                'Signal Quality': 'Assessment of EEG recording quality — Good indicates clean signal with minimal artifacts, while Poor/Fair signals may have excessive noise, electrode disconnections, or movement artifacts.',
                'Format Support': 'The platform ingests multiple EEG data formats: EDF (European Data Format, the clinical standard), CSV (for tabular/numeric exports), and plans for EEGLAB .set, MNE .fif, BDF, GDF, and MATLAB .mat.',
                'Patient Coverage': 'Percentage of registered patients who have at least one uploaded EEG recording — higher coverage indicates more complete data acquisition across the patient population.',
                'Analysis Pipeline': 'Automated processing chain triggered on upload: signal validation → feature extraction → AI classification → report generation. Each step logs to the transaction_log for audit trail.',
                'Confidence Score': 'AI model output probability for the predicted class (e.g., Epilepsy vs Control) — ranges from 0 to 1, where higher values indicate stronger classification certainty.',
            },
            'quality_metrics': {
                'Upload Rate': 'Count of files ingested over time — daily upload trend reveals acquisition velocity and identifies periods of high or low clinical activity.',
                'Coverage Rate': 'Patients with data / total registered patients × 100 — tracks completeness of the clinical dataset for downstream AI training and validation.',
                'Signal Quality Rate': 'Proportion of analyses with Good signal quality — indicates recording equipment condition, technician skill, and patient cooperation.',
                'Average Confidence': 'Mean AI classification confidence across all analyses — higher values indicate the model is making strong predictions on the ingested data.',
                'Format Diversity': 'Count of distinct file formats ingested — broader format support reduces data acquisition barriers across clinical sites.',
            },
            'clinical_relevance': {
                'ILAE Data Standards': 'International League Against Epilepsy recommends standardized EEG recording protocols and data formats for multi-center studies — consistent data acquisition enables cross-site AI model validation.',
                'IEC 62304 §5.3': 'Medical device software architecture must specify data inputs and their validation — the ingestion pipeline validates format, channel count, sampling rate, and signal quality before analysis.',
                'FDA AI/ML SaMD': 'FDA guidance for AI/ML-based Software as a Medical Device requires documentation of training data provenance, collection methodology, and quality controls — tracked via upload metadata and signal quality records.',
                'HIPAA §164.312(c)': 'Integrity controls for electronic PHI — the acquisition pipeline must ensure EEG data is not improperly altered or destroyed during ingestion, with audit logs for all data handling events.',
                'EU AI Act Article 10': 'High-risk AI systems must use training and validation data that is relevant, representative, and free of errors — data acquisition metrics (coverage, quality, diversity) directly support this requirement.',
            },
            'remediation': {
                'Low patient coverage': 'If coverage is below 50%, prioritize outreach to departments with registered but data-less patients — check for workflow barriers preventing upload (format incompatibility, network issues, training gaps).',
                'Poor signal quality': 'If more than 20% of recordings are Fair/Poor, investigate recording equipment calibration, electrode placement protocols, and patient preparation procedures — retrain technicians if needed.',
                'Low confidence scores': 'If average confidence is below 0.6, review the model training data for representativeness — may indicate domain shift between training and incoming clinical data.',
                'Stale acquisition': 'If no uploads in 7+ days, verify the ingestion pipeline is operational — check for API errors, storage capacity, or clinical workflow disruptions.',
            },
        },
    }
