"""Data Quality Dashboard — data completeness, dedup, outlier, and profiling analytics.

Monitors data quality across the clinical EEG/epilepsy platform: field completeness
rates, duplicate detection, signal quality distribution, confidence outliers,
file format profiling, patient coverage gaps, and daily upload quality trends.

Sources:
- patients table (40 records): field completeness (name, age, gender, disease, department)
- uploads table (21 records): file format distribution, duplicate detection
- analyses table (21 records): signal quality, confidence scores, outlier detection
- transaction_log table: data pipeline event quality
"""

import sqlite3
import os
import json
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


def data_quality_overview():
    """KPIs, completeness rates, signal quality distribution, confidence stats,
    duplicate counts, daily quality trend, format distribution."""
    conn = _conn()
    cur = conn.cursor()

    # ── Field completeness across patients ──
    total_patients = _safe(cur, 'SELECT COUNT(*) FROM patients')
    fields = ['name', 'age', 'gender', 'disease', 'department']
    completeness = {}
    for f in fields:
        if f == 'age':
            filled = _safe(cur, f'SELECT COUNT(*) FROM patients WHERE {f} IS NOT NULL')
        else:
            filled = _safe(cur, f'SELECT COUNT(*) FROM patients WHERE {f} IS NOT NULL AND {f} != ""')
        completeness[f] = {
            'filled': filled,
            'total': total_patients,
            'rate': round(filled / total_patients * 100, 1) if total_patients else 0
        }

    overall_completeness = round(
        sum(c['rate'] for c in completeness.values()) / len(fields), 1
    ) if fields else 0

    # ── Signal quality distribution ──
    quality_rows = _safe_rows(cur, 'SELECT signal_quality, COUNT(*) FROM analyses GROUP BY signal_quality')
    signal_quality_dist = {row[0]: row[1] for row in quality_rows}
    total_analyses = sum(signal_quality_dist.values())
    good_pct = round(signal_quality_dist.get('Good', 0) / total_analyses * 100, 1) if total_analyses else 0

    # ── Confidence statistics ──
    avg_conf = _safe(cur, 'SELECT AVG(confidence) FROM analyses', default=0)
    min_conf = _safe(cur, 'SELECT MIN(confidence) FROM analyses', default=0)
    max_conf = _safe(cur, 'SELECT MAX(confidence) FROM analyses', default=0)
    std_conf_row = _safe_rows(cur, '''
        SELECT AVG((confidence - sub.avg_c) * (confidence - sub.avg_c))
        FROM analyses, (SELECT AVG(confidence) as avg_c FROM analyses) sub
    ''')
    variance = std_conf_row[0][0] if std_conf_row and std_conf_row[0][0] else 0
    std_conf = round(variance ** 0.5, 4)

    # ── Confidence buckets ──
    buckets = [
        ('0.50-0.55', 0.50, 0.55),
        ('0.55-0.60', 0.55, 0.60),
        ('0.60-0.65', 0.60, 0.65),
        ('0.65-0.70', 0.65, 0.70),
        ('0.70-0.80', 0.70, 0.80),
        ('0.80-1.00', 0.80, 1.01),
    ]
    conf_distribution = []
    for label, lo, hi in buckets:
        cnt = _safe(cur, 'SELECT COUNT(*) FROM analyses WHERE confidence >= ? AND confidence < ?', (lo, hi))
        conf_distribution.append({'bucket': label, 'count': cnt})

    # ── Confidence outliers (below 0.55 or above 0.85) ──
    low_conf = _safe(cur, 'SELECT COUNT(*) FROM analyses WHERE confidence < 0.55')
    high_conf = _safe(cur, 'SELECT COUNT(*) FROM analyses WHERE confidence > 0.85')

    # ── Duplicate detection in uploads ──
    dup_rows = _safe_rows(cur, '''
        SELECT file_name, COUNT(*) as cnt
        FROM uploads GROUP BY file_name HAVING cnt > 1
    ''')
    total_uploads = _safe(cur, 'SELECT COUNT(*) FROM uploads')
    duplicate_files = len(dup_rows)
    duplicate_uploads = sum(r[1] for r in dup_rows)
    unique_files = _safe(cur, 'SELECT COUNT(DISTINCT file_name) FROM uploads')
    dedup_rate = round((1 - duplicate_files / unique_files) * 100, 1) if unique_files else 100

    # ── File format distribution ──
    format_rows = _safe_rows(cur, '''
        SELECT CASE
            WHEN file_name LIKE '%.edf' THEN 'EDF'
            WHEN file_name LIKE '%.csv' THEN 'CSV'
            WHEN file_name LIKE '%.bdf' THEN 'BDF'
            ELSE 'Other'
        END as fmt, COUNT(*)
        FROM uploads GROUP BY fmt ORDER BY COUNT(*) DESC
    ''')
    format_dist = [{'format': r[0], 'count': r[1]} for r in format_rows]

    # ── Patient coverage ──
    patients_with_uploads = _safe(cur, 'SELECT COUNT(DISTINCT patient_id) FROM uploads')
    patients_with_analyses = _safe(cur, 'SELECT COUNT(DISTINCT patient_id) FROM analyses')
    upload_coverage = round(patients_with_uploads / total_patients * 100, 1) if total_patients else 0
    analysis_coverage = round(patients_with_analyses / total_patients * 100, 1) if total_patients else 0

    # ── Daily upload quality trend ──
    daily_rows = _safe_rows(cur, '''
        SELECT DATE(u.created_at) as day,
               COUNT(*) as uploads,
               SUM(CASE WHEN a.signal_quality = 'Good' THEN 1 ELSE 0 END) as good,
               SUM(CASE WHEN a.signal_quality = 'Poor' THEN 1 ELSE 0 END) as poor,
               ROUND(AVG(a.confidence), 3) as avg_conf
        FROM uploads u
        LEFT JOIN analyses a ON u.id = a.upload_id
        GROUP BY day ORDER BY day
    ''')
    daily_trend = [{
        'date': r[0],
        'uploads': r[1],
        'good': r[2] or 0,
        'poor': r[3] or 0,
        'avg_confidence': r[4] or 0
    } for r in daily_rows]

    # ── Null field distribution (which fields are most commonly missing) ──
    null_fields = []
    for f in fields:
        if f == 'age':
            missing = _safe(cur, f'SELECT COUNT(*) FROM patients WHERE {f} IS NULL')
        else:
            missing = _safe(cur, f'SELECT COUNT(*) FROM patients WHERE {f} IS NULL OR {f} = ""')
        null_fields.append({'field': f, 'missing': missing, 'present': total_patients - missing})

    # ── Disease distribution quality ──
    disease_rows = _safe_rows(cur, 'SELECT disease, COUNT(*) FROM patients GROUP BY disease ORDER BY COUNT(*) DESC')
    disease_dist = [{'disease': r[0] or 'Unknown', 'count': r[1]} for r in disease_rows]

    conn.close()

    return {
        'kpis': {
            'total_patients': total_patients,
            'total_uploads': total_uploads,
            'total_analyses': total_analyses,
            'overall_completeness_pct': overall_completeness,
            'good_signal_pct': good_pct,
            'avg_confidence': round(avg_conf, 3) if avg_conf else 0,
            'duplicate_files': duplicate_files,
            'upload_coverage_pct': upload_coverage,
        },
        'field_completeness': completeness,
        'signal_quality_distribution': signal_quality_dist,
        'confidence_stats': {
            'mean': round(avg_conf, 4) if avg_conf else 0,
            'min': round(min_conf, 4) if min_conf else 0,
            'max': round(max_conf, 4) if max_conf else 0,
            'std': std_conf,
            'low_outliers': low_conf,
            'high_outliers': high_conf,
        },
        'confidence_distribution': conf_distribution,
        'format_distribution': format_dist,
        'daily_trend': daily_trend,
        'null_field_distribution': null_fields,
        'disease_distribution': disease_dist,
        'coverage': {
            'patients_with_uploads': patients_with_uploads,
            'patients_with_analyses': patients_with_analyses,
            'upload_coverage_pct': upload_coverage,
            'analysis_coverage_pct': analysis_coverage,
        },
    }


def data_quality_breakdown():
    """Per-patient profiles, duplicate file log, outlier analyses,
    per-upload quality detail, recent quality events."""
    conn = _conn()
    cur = conn.cursor()

    # ── Per-patient data quality profiles ──
    patients = _safe_rows(cur, '''
        SELECT patient_id, name, age, gender, disease, department FROM patients
        ORDER BY patient_id
    ''')
    patient_profiles = []
    for p in patients:
        pid = p[0]
        # Count filled fields
        filled = sum(1 for v in p[1:] if v is not None and str(v).strip() != '')
        total_fields = 5
        completeness_pct = round(filled / total_fields * 100, 1)

        # Upload count
        upload_cnt = _safe(cur, 'SELECT COUNT(*) FROM uploads WHERE patient_id = ?', (pid,))
        # Analysis count + avg confidence
        analysis_cnt = _safe(cur, 'SELECT COUNT(*) FROM analyses WHERE patient_id = ?', (pid,))
        avg_conf = _safe(cur, 'SELECT AVG(confidence) FROM analyses WHERE patient_id = ?', (pid,), default=None)
        signal = _safe(cur, 'SELECT signal_quality FROM analyses WHERE patient_id = ? ORDER BY created_at DESC LIMIT 1', (pid,), default=None)

        missing_fields = []
        field_names = ['name', 'age', 'gender', 'disease', 'department']
        for i, fn in enumerate(field_names):
            val = p[i + 1]
            if val is None or str(val).strip() == '':
                missing_fields.append(fn)

        patient_profiles.append({
            'patient_id': pid,
            'completeness_pct': completeness_pct,
            'filled_fields': filled,
            'total_fields': total_fields,
            'missing_fields': missing_fields,
            'uploads': upload_cnt,
            'analyses': analysis_cnt,
            'avg_confidence': round(avg_conf, 3) if avg_conf else None,
            'latest_signal_quality': signal,
        })

    # ── Duplicate file log ──
    dup_detail = _safe_rows(cur, '''
        SELECT u.file_name, u.patient_id, u.created_at
        FROM uploads u
        WHERE u.file_name IN (
            SELECT file_name FROM uploads GROUP BY file_name HAVING COUNT(*) > 1
        )
        ORDER BY u.file_name, u.created_at
    ''')
    duplicate_log = [{'file_name': r[0], 'patient_id': r[1], 'uploaded_at': r[2]} for r in dup_detail]

    # ── Per-upload quality detail ──
    upload_detail = _safe_rows(cur, '''
        SELECT u.id, u.patient_id, u.file_name, u.disease, u.created_at,
               a.signal_quality, a.confidence, a.predicted_label
        FROM uploads u
        LEFT JOIN analyses a ON u.id = a.upload_id
        ORDER BY u.created_at DESC
    ''')
    upload_quality = [{
        'upload_id': r[0],
        'patient_id': r[1],
        'file_name': r[2],
        'disease': r[3],
        'uploaded_at': r[4],
        'signal_quality': r[5],
        'confidence': round(r[6], 3) if r[6] else None,
        'predicted_label': r[7],
    } for r in upload_detail]

    # ── Outlier analyses (confidence < 0.55 or signal = Poor) ──
    outlier_rows = _safe_rows(cur, '''
        SELECT a.id, a.patient_id, a.confidence, a.signal_quality,
               a.predicted_label, a.created_at, u.file_name
        FROM analyses a
        LEFT JOIN uploads u ON a.upload_id = u.id
        WHERE a.confidence < 0.55 OR a.signal_quality = 'Poor'
        ORDER BY a.confidence ASC
    ''')
    outliers = [{
        'analysis_id': r[0],
        'patient_id': r[1],
        'confidence': round(r[2], 3) if r[2] else None,
        'signal_quality': r[3],
        'predicted_label': r[4],
        'analyzed_at': r[5],
        'file_name': r[6],
    } for r in outlier_rows]

    # ── Quality events from transaction log ──
    quality_events = _safe_rows(cur, '''
        SELECT timestamp, actor, component, action, patient_id, detail
        FROM transaction_log
        WHERE component IN ('eeg_upload', 'eeg_analysis', 'validation', 'data_quality',
                            'signal_processing', 'artifact_detection')
        ORDER BY timestamp DESC LIMIT 30
    ''')
    event_log = [{
        'timestamp': r[0],
        'actor': r[1],
        'component': r[2],
        'action': r[3],
        'patient_id': r[4],
        'detail': r[5],
    } for r in quality_events]

    conn.close()

    return {
        'patient_profiles': patient_profiles,
        'duplicate_log': duplicate_log,
        'upload_quality': upload_quality,
        'outliers': outliers,
        'quality_event_log': event_log,
    }


def data_quality_definitions():
    """Data quality concepts, metrics, clinical relevance, and remediation strategies."""
    return {
        'sections': [
            {
                'title': 'Data Quality Concepts',
                'items': [
                    {'term': 'Completeness', 'definition': 'Percentage of required fields that contain non-null, non-empty values across patient records. Missing demographics reduce clinical context for AI predictions.'},
                    {'term': 'Deduplication', 'definition': 'Detection and flagging of duplicate file uploads (same file_name uploaded multiple times). Duplicates waste storage and can bias model training if not removed.'},
                    {'term': 'Signal Quality', 'definition': 'Classification of EEG recording quality as Good or Poor based on artifact levels, flat channels, and signal-to-noise ratio. Poor signals produce unreliable predictions.'},
                    {'term': 'Outlier Detection', 'definition': 'Identification of analyses with unusually low confidence scores (<0.55) or poor signal quality that may indicate data collection issues or edge cases.'},
                    {'term': 'Data Profiling', 'definition': 'Statistical summary of data distributions — field types, value ranges, format consistency, and coverage rates across the dataset.'},
                    {'term': 'Coverage', 'definition': 'Ratio of patients who have at least one upload/analysis vs total registered patients. Low coverage indicates gaps in the data pipeline.'},
                ]
            },
            {
                'title': 'Quality Metrics',
                'items': [
                    {'term': 'Field Completeness Rate', 'definition': 'Per-field percentage: (non-null records / total records) × 100. Target: >95% for critical fields (name, disease).'},
                    {'term': 'Overall Completeness', 'definition': 'Average completeness across all tracked fields. Weighted equally. Threshold: ≥80% acceptable, ≥95% excellent.'},
                    {'term': 'Good Signal Rate', 'definition': 'Percentage of analyses classified as Good signal quality. Target: ≥80% for clinical-grade data.'},
                    {'term': 'Confidence Mean/Std', 'definition': 'Average and standard deviation of prediction confidence scores. Low mean (<0.6) or high std (>0.1) suggests inconsistent data quality.'},
                    {'term': 'Duplicate Rate', 'definition': 'Fraction of unique filenames that appear more than once. Target: 0% (no duplicates).'},
                    {'term': 'Upload Coverage', 'definition': 'Percentage of registered patients with ≥1 uploaded file. Low coverage means many patients lack data for analysis.'},
                ]
            },
            {
                'title': 'Clinical Relevance',
                'items': [
                    {'term': 'ILAE Guidelines', 'definition': 'International League Against Epilepsy recommends standardized EEG recording protocols. Poor signal quality violates minimum recording standards.'},
                    {'term': 'IEC 62304 (Medical Device Software)', 'definition': 'Requires input data validation and traceability. Data quality monitoring satisfies verification requirements for SaMD.'},
                    {'term': 'FDA AI/ML SaMD', 'definition': 'FDA guidance on AI-based Software as Medical Device requires data quality assurance including completeness, representativeness, and absence of bias.'},
                    {'term': 'HIPAA Data Integrity', 'definition': 'HIPAA Security Rule §164.312(c) requires integrity controls ensuring ePHI is not improperly altered. Data quality checks support this requirement.'},
                    {'term': 'EU AI Act (High-Risk)', 'definition': 'Article 10 mandates training data quality criteria including completeness, representativeness, and freedom from errors for high-risk AI systems.'},
                ]
            },
            {
                'title': 'Remediation Strategies',
                'items': [
                    {'term': 'Missing Fields', 'definition': 'Flag patients with incomplete demographics for data entry review. Prioritize name and disease fields for clinical decision support.'},
                    {'term': 'Poor Signal', 'definition': 'Re-record EEG sessions with signal quality = Poor. Check electrode impedance, patient movement artifacts, and environmental noise.'},
                    {'term': 'Low Confidence', 'definition': 'Review analyses with confidence <0.55 for potential re-analysis with artifact removal or alternative model ensemble.'},
                    {'term': 'Duplicates', 'definition': 'Implement upload deduplication by file hash (SHA-256). Archive duplicate entries and retain the earliest upload timestamp.'},
                    {'term': 'Coverage Gaps', 'definition': 'Identify patients without uploads and coordinate with clinical staff to schedule data acquisition sessions.'},
                ]
            }
        ]
    }
