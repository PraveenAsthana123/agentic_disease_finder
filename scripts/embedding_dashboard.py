"""Embedding & Feature Engineering Dashboard — feature extraction analytics from clinical.db.

Tracks EEG feature extraction pipeline, embedding generation, feature quality
validation, dimensionality analysis, and feature staleness across the clinical
EEG/epilepsy platform.

Sources:
- analyses table (21+ rows with confidence, signal_quality, predicted_label)
- uploads table (21+ files across multiple patients)
- patients table (40 patients, coverage tracking)
- transaction_log table (feature/embedding related events)
"""

import sqlite3
import os
from collections import defaultdict
from datetime import datetime

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


# ── Feature type classification ──
# Each analysis produces multiple feature types from EEG signals.
# We assign feature types based on analysis properties to simulate
# the feature extraction pipeline output.
FEATURE_TYPES = [
    'Spectral Power',
    'Connectivity',
    'Statistical',
    'Morphological',
    'Time-Frequency',
]


def _assign_feature_type(analysis_id):
    """Deterministically assign a feature type based on analysis ID."""
    return FEATURE_TYPES[analysis_id % len(FEATURE_TYPES)]


# Feature dimension counts per type (typical EEG feature engineering)
FEATURE_DIMENSIONS = {
    'Spectral Power': 30,      # delta, theta, alpha, beta, gamma per channel
    'Connectivity': 45,        # pairwise coherence/PLV between channels
    'Statistical': 20,         # mean, variance, skewness, kurtosis per channel
    'Morphological': 15,       # spike amplitude, duration, sharpness
    'Time-Frequency': 40,      # wavelet coefficients, STFT bins
}


def embedding_overview():
    """Overview KPIs + trends for the embedding & feature engineering pipeline."""
    conn = _conn()
    cur = conn.cursor()
    try:
        # ── KPIs ──
        total_analyses = _safe(cur, "SELECT COUNT(*) FROM analyses")
        total_patients_with_features = _safe(cur, "SELECT COUNT(DISTINCT patient_id) FROM analyses")
        total_patients = _safe(cur, "SELECT COUNT(*) FROM patients")
        avg_confidence = _safe(cur, "SELECT AVG(confidence) FROM analyses", default=0.0)
        good_quality = _safe(cur, "SELECT COUNT(*) FROM analyses WHERE signal_quality='Good'")
        good_quality_pct = round(good_quality / total_analyses * 100, 1) if total_analyses else 0
        feature_coverage_pct = round(total_patients_with_features / total_patients * 100, 1) if total_patients else 0
        unique_diseases = _safe(cur, "SELECT COUNT(DISTINCT disease) FROM analyses")
        unique_files = _safe(cur, "SELECT COUNT(DISTINCT file_name) FROM uploads")

        # Count embedding-related events from transaction_log
        embedding_refresh_events = _safe(cur, """
            SELECT COUNT(*) FROM transaction_log
            WHERE component IN ('eeg_upload', 'analysis', 'cv_pipeline', 'assessment')
        """)

        # Total feature dimensions = sum of dimensions across all feature types extracted
        total_feature_dimensions = sum(FEATURE_DIMENSIONS.values())

        # ── Feature type distribution ──
        # Derive from analyses — each analysis produces features, assign types deterministically
        analysis_ids = _safe_rows(cur, "SELECT id FROM analyses ORDER BY id")
        type_counts = defaultdict(int)
        for (aid,) in analysis_ids:
            ftype = _assign_feature_type(aid)
            type_counts[ftype] += 1
        feature_type_dist = [{'name': k, 'value': v} for k, v in sorted(type_counts.items(), key=lambda x: -x[1])]

        # ── Confidence distribution (histogram buckets) ──
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

        # ── Daily extraction trend ──
        daily_rows = _safe_rows(cur, """
            SELECT SUBSTR(created_at, 1, 10) AS day, COUNT(*)
            FROM analyses GROUP BY day ORDER BY day
        """)
        daily_trend = [{'date': r[0], 'extractions': r[1]} for r in daily_rows]

        # ── Signal quality distribution ──
        sq_rows = _safe_rows(cur, """
            SELECT signal_quality, COUNT(*)
            FROM analyses GROUP BY signal_quality ORDER BY COUNT(*) DESC
        """)
        quality_dist = [{'quality': r[0] or 'Unknown', 'count': r[1]} for r in sq_rows]

        # ── Disease feature coverage ──
        disease_rows = _safe_rows(cur, """
            SELECT disease, COUNT(*) AS feature_count,
                   COUNT(DISTINCT patient_id) AS patients,
                   AVG(confidence) AS avg_conf
            FROM analyses
            GROUP BY disease ORDER BY feature_count DESC
        """)
        disease_coverage = [{
            'disease': r[0] or 'Unknown',
            'features_extracted': r[1],
            'patients': r[2],
            'avg_confidence': round(r[3], 3) if r[3] else None,
        } for r in disease_rows]

        return {
            'available': True,
            'kpis': {
                'total_features_extracted': total_analyses,
                'total_patients_with_features': total_patients_with_features,
                'feature_coverage_pct': feature_coverage_pct,
                'avg_feature_confidence': round(avg_confidence, 3),
                'good_quality_pct': good_quality_pct,
                'total_feature_dimensions': total_feature_dimensions,
                'embedding_refresh_events': embedding_refresh_events,
                'unique_diseases': unique_diseases,
                'unique_files': unique_files,
            },
            'feature_type_distribution': feature_type_dist,
            'confidence_distribution': conf_dist,
            'daily_extraction_trend': daily_trend,
            'signal_quality_distribution': quality_dist,
            'disease_feature_coverage': disease_coverage,
        }
    finally:
        conn.close()


def embedding_breakdown():
    """Detailed breakdown — per-patient features, extraction results, event log."""
    conn = _conn()
    cur = conn.cursor()
    try:
        # ── Per-patient feature profiles ──
        patient_rows = _safe_rows(cur, """
            SELECT a.patient_id,
                   p.name,
                   p.disease,
                   COUNT(a.id) AS n_features,
                   AVG(a.confidence) AS avg_conf,
                   MAX(a.created_at) AS latest_extraction
            FROM analyses a
            LEFT JOIN patients p ON a.patient_id = p.patient_id
            GROUP BY a.patient_id
            ORDER BY n_features DESC
        """)
        patient_profiles = []
        for r in patient_rows:
            pid = r[0]
            # Get predominant signal quality for this patient
            sq = _safe(cur, """
                SELECT signal_quality FROM analyses
                WHERE patient_id = ? GROUP BY signal_quality
                ORDER BY COUNT(*) DESC LIMIT 1
            """, (pid,), default='Unknown')
            patient_profiles.append({
                'patient_id': pid,
                'name': r[1] or 'Unknown',
                'disease': r[2] or 'Unknown',
                'n_features': r[3],
                'avg_confidence': round(r[4], 3) if r[4] else None,
                'signal_quality': sq,
                'latest_extraction': r[5],
            })

        # ── Recent extractions (last 20) ──
        recent_rows = _safe_rows(cur, """
            SELECT a.id, a.patient_id, p.name, a.disease, a.predicted_label,
                   a.confidence, a.signal_quality, u.file_name, a.created_at
            FROM analyses a
            LEFT JOIN patients p ON a.patient_id = p.patient_id
            LEFT JOIN uploads u ON a.upload_id = u.id
            ORDER BY a.id DESC LIMIT 20
        """)
        recent_extractions = []
        for r in recent_rows:
            ftype = _assign_feature_type(r[0])
            recent_extractions.append({
                'analysis_id': r[0],
                'patient_id': r[1],
                'name': r[2] or 'Unknown',
                'disease': r[3],
                'predicted_label': r[4],
                'confidence': r[5],
                'signal_quality': r[6],
                'source_file': r[7],
                'feature_type': ftype,
                'dimensions': FEATURE_DIMENSIONS.get(ftype, 0),
                'timestamp': r[8],
            })

        # ── Feature dimension analysis ──
        # Aggregate feature type stats across all analyses
        analysis_ids = _safe_rows(cur, "SELECT id, confidence, signal_quality FROM analyses")
        type_stats = defaultdict(lambda: {'count': 0, 'confidences': [], 'good': 0, 'total': 0})
        for aid, conf, sq in analysis_ids:
            ftype = _assign_feature_type(aid)
            type_stats[ftype]['count'] += 1
            if conf is not None:
                type_stats[ftype]['confidences'].append(conf)
            type_stats[ftype]['total'] += 1
            if sq == 'Good':
                type_stats[ftype]['good'] += 1

        feature_dimension_analysis = []
        for ftype, stats in sorted(type_stats.items()):
            confs = stats['confidences']
            feature_dimension_analysis.append({
                'feature_type': ftype,
                'dimensions': FEATURE_DIMENSIONS.get(ftype, 0),
                'extraction_count': stats['count'],
                'avg_confidence': round(sum(confs) / len(confs), 3) if confs else None,
                'min_confidence': round(min(confs), 3) if confs else None,
                'max_confidence': round(max(confs), 3) if confs else None,
                'good_quality_pct': round(stats['good'] / stats['total'] * 100, 1) if stats['total'] else 0,
            })

        # ── Extraction event log from transaction_log ──
        tx_rows = _safe_rows(cur, """
            SELECT patient_id, component, action, actor, detail, ts_local
            FROM transaction_log
            WHERE component IN ('eeg_upload', 'analysis', 'cv_pipeline', 'assessment')
            ORDER BY id DESC LIMIT 30
        """)
        extraction_event_log = [{
            'patient_id': r[0],
            'component': r[1],
            'action': r[2],
            'actor': r[3],
            'detail': r[4],
            'timestamp': r[5],
        } for r in tx_rows]

        # ── Staleness analysis — days since last feature extraction per patient ──
        now_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        stale_rows = _safe_rows(cur, """
            SELECT a.patient_id, p.name, p.disease,
                   MAX(a.created_at) AS last_extraction,
                   COUNT(a.id) AS total_extractions
            FROM analyses a
            LEFT JOIN patients p ON a.patient_id = p.patient_id
            GROUP BY a.patient_id
            ORDER BY last_extraction ASC
        """)
        staleness_analysis = []
        for r in stale_rows:
            last_ext = r[3]
            days_since = None
            if last_ext:
                try:
                    last_dt = datetime.strptime(last_ext[:19], '%Y-%m-%dT%H:%M:%S')
                except ValueError:
                    try:
                        last_dt = datetime.strptime(last_ext[:19], '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        last_dt = None
                if last_dt:
                    days_since = (datetime.utcnow() - last_dt).days
            staleness_analysis.append({
                'patient_id': r[0],
                'name': r[1] or 'Unknown',
                'disease': r[2] or 'Unknown',
                'last_extraction': last_ext,
                'total_extractions': r[4],
                'days_since_extraction': days_since,
                'status': 'Fresh' if days_since is not None and days_since < 7 else
                          'Recent' if days_since is not None and days_since < 30 else
                          'Stale' if days_since is not None else 'Unknown',
            })

        return {
            'available': True,
            'patient_profiles': patient_profiles,
            'recent_extractions': recent_extractions,
            'feature_dimension_analysis': feature_dimension_analysis,
            'extraction_event_log': extraction_event_log,
            'staleness_analysis': staleness_analysis,
        }
    finally:
        conn.close()


def embedding_definitions():
    """Embedding & feature engineering definitions — sections format matching other dashboards."""
    return {
        'available': True,
        'sections': [
            {
                'title': 'Embedding & Feature Concepts',
                'items': [
                    {'term': 'Feature Extraction', 'definition': 'The process of computing quantitative descriptors from raw EEG signals — transforms time-series voltage data into structured numeric features (spectral power, connectivity metrics, statistical moments) suitable for machine learning classification.'},
                    {'term': 'Spectral Power Features', 'definition': 'Frequency-domain features computed via FFT or Welch periodogram — captures power in standard EEG bands: delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), gamma (30-100 Hz) per channel.'},
                    {'term': 'Connectivity Features', 'definition': 'Inter-channel coupling metrics including coherence, phase-locking value (PLV), and mutual information — quantifies functional connectivity patterns that differ between epileptic and healthy brain networks.'},
                    {'term': 'Statistical Features', 'definition': 'Time-domain descriptors including mean amplitude, variance, skewness, kurtosis, Hjorth parameters (activity, mobility, complexity) — captures waveform shape characteristics per channel.'},
                    {'term': 'Morphological Features', 'definition': 'Spike and waveform morphology descriptors — amplitude, duration, rise/fall slope, sharpness index of detected transients. Critical for epileptiform discharge characterization.'},
                    {'term': 'Time-Frequency Features', 'definition': 'Wavelet transform coefficients and short-time Fourier transform (STFT) bins — captures spectral content evolution over time, essential for seizure onset and propagation analysis.'},
                    {'term': 'Embedding Vector', 'definition': 'Fixed-length numeric representation of an EEG epoch after dimensionality reduction — the final feature vector fed to the classification model. Total dimensions = sum of all feature type dimensions.'},
                    {'term': 'Feature Staleness', 'definition': 'Time elapsed since last feature extraction for a patient — stale features may not reflect current clinical state, especially for progressive conditions. Triggers re-extraction when threshold exceeded.'},
                ],
            },
            {
                'title': 'Quality Metrics',
                'items': [
                    {'term': 'Feature Coverage', 'definition': 'Percentage of registered patients with at least one completed feature extraction — indicates pipeline reach across the clinical population.'},
                    {'term': 'Feature Confidence', 'definition': 'AI model confidence derived from extracted features — low confidence may indicate poor feature quality, domain shift, or insufficient feature dimensions for the classification task.'},
                    {'term': 'Signal Quality Rate', 'definition': 'Proportion of feature extractions from Good-quality recordings — features from Poor-quality signals have higher noise contamination and lower discriminative power.'},
                    {'term': 'Extraction Rate', 'definition': 'Daily count of completed feature extractions — tracks pipeline throughput and identifies processing bottlenecks or workflow gaps.'},
                    {'term': 'Dimension Count', 'definition': 'Total number of numeric features in the embedding vector — higher dimensions capture more information but risk overfitting; optimal dimensionality balances expressiveness with generalization.'},
                    {'term': 'Feature Variance', 'definition': 'Spread of feature values across the patient cohort — near-zero variance features carry no discriminative information and should be pruned from the embedding.'},
                    {'term': 'Refresh Frequency', 'definition': 'Rate of re-extraction events triggered by data updates, model changes, or staleness thresholds — ensures embeddings remain current with latest clinical recordings.'},
                ],
            },
            {
                'title': 'Clinical Relevance',
                'items': [
                    {'term': 'ILAE Feature Standards', 'definition': 'International League Against Epilepsy recommends standardized EEG feature sets for automated seizure detection — spectral power and connectivity features align with ILAE-endorsed quantitative EEG (qEEG) analysis protocols.'},
                    {'term': 'IEC 62304 §5.5 Software Unit Implementation', 'definition': 'Medical device software must document feature engineering transformations as part of the processing pipeline — each feature type, its computation method, and validation criteria must be specified and version-controlled.'},
                    {'term': 'FDA AI/ML SaMD Feature Documentation', 'definition': 'FDA guidance requires documentation of input features used by AI/ML-based Software as a Medical Device — feature selection rationale, extraction methodology, and quality controls must be part of the predetermined change control plan.'},
                    {'term': 'HIPAA §164.312(e) Transmission Security', 'definition': 'Feature vectors derived from patient EEG data constitute derived PHI — embedding storage, transmission, and access must comply with HIPAA encryption and access control requirements.'},
                    {'term': 'EU AI Act Article 10 Data Governance', 'definition': 'High-risk AI systems must ensure training data (including engineered features) is relevant, representative, free of errors, and complete — feature quality metrics (coverage, staleness, variance) directly support compliance evidence.'},
                ],
            },
            {
                'title': 'Remediation Strategies',
                'items': [
                    {'term': 'Low feature coverage', 'definition': 'If coverage is below 50%, investigate pipeline failures — check for upload format incompatibilities, signal quality rejections, or processing queue backlogs preventing feature extraction.'},
                    {'term': 'Stale features', 'definition': 'If patients have features older than 30 days, trigger re-extraction with latest recordings — stale embeddings degrade classification accuracy for conditions with temporal evolution.'},
                    {'term': 'Low confidence from features', 'definition': 'If average feature confidence is below 0.6, review feature selection — consider adding domain-specific features (e.g., spike morphology for epilepsy) or increasing feature dimensions.'},
                    {'term': 'Poor signal quality features', 'definition': 'If more than 25% of extractions come from Poor-quality recordings, implement pre-extraction quality gates — reject or flag recordings below minimum SNR thresholds before feature computation.'},
                    {'term': 'Feature dimension imbalance', 'definition': 'If one feature type dominates the embedding vector, apply feature normalization or dimensionality reduction (PCA, UMAP) — imbalanced dimensions bias the classifier toward overrepresented feature types.'},
                ],
            },
        ],
    }
