"""Continuous Learning Dashboard — feedback collection, error analysis,
retrain triggers, and redeployment tracking across the clinical AI platform.

Aggregates data from:
- data/clinical.db feedback table (clinician feedback on AI outputs)
- data/clinical.db transaction_log (training, drift, consistency, fairness events)
- data/clinical.db analyses table (21 AI analyses with confidence/signal quality)
- data/clinical.db hitl_reviews table (human-in-the-loop override/accept decisions)
- data/clinical.db expert_reviews table (expert agreement with AI)
- jobs/reports/ (drift, consistency, fairness JSON reports)
"""

import sqlite3
import json
import os
import math
import glob
from datetime import datetime

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')
REPORTS = os.path.join(BASE, 'jobs', 'reports')


# ── Helpers ──────────────────────────────────────────────────────────────────

def _connect():
    if not os.path.exists(DB):
        return None
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn, name):
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row[0] > 0


def _safe(val):
    if val is None:
        return None
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
    return val


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ── Overview ─────────────────────────────────────────────────────────────────

def cl_overview():
    """KPI-level summary: feedback volume, training runs, drift events,
    retrain triggers, HITL overrides, prediction confidence trends."""
    conn = _connect()
    if conn is None:
        return {'available': False, 'message': 'Database not found.'}

    try:
        # ── Feedback KPIs ──────────────────────────────────────────────
        total_feedback = 0
        avg_rating = None
        feedback_by_role = []
        if _table_exists(conn, 'feedback'):
            total_feedback = conn.execute(
                "SELECT COUNT(*) FROM feedback"
            ).fetchone()[0]
            row = conn.execute(
                "SELECT AVG(rating) FROM feedback WHERE rating IS NOT NULL"
            ).fetchone()
            avg_rating = _safe(row[0]) if row else None
            feedback_by_role = [
                dict(r) for r in conn.execute(
                    "SELECT role, COUNT(*) as count, AVG(rating) as avg_rating "
                    "FROM feedback GROUP BY role ORDER BY count DESC"
                ).fetchall()
            ]

        # ── Training KPIs ─────────────────────────────────────────────
        total_training_runs = 0
        successful_runs = 0
        failed_runs = 0
        training_timeline = []
        if _table_exists(conn, 'transaction_log'):
            rows = conn.execute(
                "SELECT * FROM transaction_log WHERE component='training' "
                "ORDER BY ts_local"
            ).fetchall()
            total_training_runs = len(rows)
            for r in rows:
                detail = r['detail'] or ''
                success = 'succeeded' in detail
                if success:
                    successful_runs += 1
                else:
                    failed_runs += 1
                training_timeline.append({
                    'date': (r['ts_local'] or '')[:10],
                    'detail': detail,
                    'success': success,
                })

        # ── Drift / Consistency / Fairness event counts ───────────────
        drift_events = 0
        consistency_events = 0
        fairness_events = 0
        if _table_exists(conn, 'transaction_log'):
            drift_events = conn.execute(
                "SELECT COUNT(*) FROM transaction_log WHERE component='drift'"
            ).fetchone()[0]
            consistency_events = conn.execute(
                "SELECT COUNT(*) FROM transaction_log WHERE component='consistency'"
            ).fetchone()[0]
            fairness_events = conn.execute(
                "SELECT COUNT(*) FROM transaction_log WHERE component='fairness'"
            ).fetchone()[0]

        # ── HITL reviews ──────────────────────────────────────────────
        total_hitl = 0
        overrides = 0
        accepts = 0
        if _table_exists(conn, 'hitl_reviews'):
            total_hitl = conn.execute(
                "SELECT COUNT(*) FROM hitl_reviews"
            ).fetchone()[0]
            rows = conn.execute("SELECT fields_json FROM hitl_reviews").fetchall()
            for r in rows:
                try:
                    fields = json.loads(r['fields_json'] or '{}')
                    if fields.get('decision') == 'override':
                        overrides += 1
                    elif fields.get('decision') == 'accept':
                        accepts += 1
                except Exception:
                    pass

        # ── Prediction confidence distribution ────────────────────────
        total_analyses = 0
        avg_confidence = None
        confidence_buckets = []
        if _table_exists(conn, 'analyses'):
            total_analyses = conn.execute(
                "SELECT COUNT(*) FROM analyses"
            ).fetchone()[0]
            row = conn.execute(
                "SELECT AVG(confidence) FROM analyses WHERE confidence IS NOT NULL"
            ).fetchone()
            avg_confidence = _safe(round(row[0], 3)) if row and row[0] else None
            buckets = {'<0.5': 0, '0.5-0.6': 0, '0.6-0.7': 0, '0.7-0.8': 0,
                       '0.8-0.9': 0, '0.9-1.0': 0}
            for r in conn.execute("SELECT confidence FROM analyses WHERE confidence IS NOT NULL"):
                c = r[0]
                if c < 0.5:
                    buckets['<0.5'] += 1
                elif c < 0.6:
                    buckets['0.5-0.6'] += 1
                elif c < 0.7:
                    buckets['0.6-0.7'] += 1
                elif c < 0.8:
                    buckets['0.7-0.8'] += 1
                elif c < 0.9:
                    buckets['0.8-0.9'] += 1
                else:
                    buckets['0.9-1.0'] += 1
            confidence_buckets = [{'range': k, 'count': v} for k, v in buckets.items()]

        # ── Latest drift report ───────────────────────────────────────
        drift_report = _load_json(os.path.join(REPORTS, 'drift_latest.json'))
        drift_verdict = None
        if drift_report and isinstance(drift_report, dict):
            drift_verdict = drift_report.get('verdict') or drift_report.get('overall_verdict')

        # ── Latest consistency report ─────────────────────────────────
        consistency_report = _load_json(os.path.join(REPORTS, 'consistency_latest.json'))
        consistency_verdict = None
        if consistency_report and isinstance(consistency_report, dict):
            consistency_verdict = consistency_report.get('verdict') or consistency_report.get('overall_verdict')

        # ── Training success rate over time ───────────────────────────
        training_success_rate = []
        if training_timeline:
            by_date = {}
            for t in training_timeline:
                d = t['date']
                if d not in by_date:
                    by_date[d] = {'total': 0, 'success': 0}
                by_date[d]['total'] += 1
                if t['success']:
                    by_date[d]['success'] += 1
            for d in sorted(by_date):
                rate = round(by_date[d]['success'] / by_date[d]['total'] * 100, 1) if by_date[d]['total'] else 0
                training_success_rate.append({
                    'date': d,
                    'total': by_date[d]['total'],
                    'success': by_date[d]['success'],
                    'rate_pct': rate,
                })

        # ── ML event timeline (all ML-related events) ─────────────────
        ml_timeline = []
        if _table_exists(conn, 'transaction_log'):
            rows = conn.execute(
                "SELECT component, action, detail, ts_local FROM transaction_log "
                "WHERE component IN ('training','drift','consistency','fairness') "
                "ORDER BY ts_local"
            ).fetchall()
            for r in rows:
                ml_timeline.append({
                    'date': (r['ts_local'] or '')[:10],
                    'component': r['component'],
                    'action': r['action'],
                    'detail': (r['detail'] or '')[:120],
                })

        # ── Retrain trigger analysis ──────────────────────────────────
        retrain_triggers = {
            'scheduled': total_training_runs,
            'drift_detected': drift_events,
            'hitl_overrides': overrides,
            'feedback_driven': total_feedback,
        }

        return {
            'available': True,
            'kpis': {
                'total_feedback': total_feedback,
                'avg_feedback_rating': avg_rating,
                'total_training_runs': total_training_runs,
                'successful_runs': successful_runs,
                'failed_runs': failed_runs,
                'training_success_pct': round(successful_runs / total_training_runs * 100, 1) if total_training_runs else 0,
                'drift_events': drift_events,
                'consistency_events': consistency_events,
                'fairness_events': fairness_events,
                'total_hitl_reviews': total_hitl,
                'hitl_overrides': overrides,
                'hitl_accepts': accepts,
                'total_analyses': total_analyses,
                'avg_confidence': avg_confidence,
                'drift_verdict': drift_verdict,
                'consistency_verdict': consistency_verdict,
            },
            'feedback_by_role': feedback_by_role,
            'confidence_buckets': confidence_buckets,
            'training_timeline': training_timeline,
            'training_success_rate': training_success_rate,
            'ml_timeline': ml_timeline,
            'retrain_triggers': retrain_triggers,
        }
    finally:
        conn.close()


# ── Breakdown ────────────────────────────────────────────────────────────────

def cl_breakdown():
    """Detailed breakdowns: per-patient error analysis, feedback log,
    training run detail, drift feature analysis, expert review concordance."""
    conn = _connect()
    if conn is None:
        return {'available': False, 'message': 'Database not found.'}

    try:
        # ── Feedback detail ───────────────────────────────────────────
        feedback_log = []
        if _table_exists(conn, 'feedback'):
            rows = conn.execute(
                "SELECT * FROM feedback ORDER BY created_at DESC"
            ).fetchall()
            for r in rows:
                feedback_log.append({
                    'patient_id': r['patient_id'],
                    'role': r['role'],
                    'ai_output': (r['ai_output'] or '')[:100],
                    'rating': r['rating'],
                    'correction': r['correction'],
                    'reason': r['reason'],
                    'created_at': r['created_at'],
                })

        # ── HITL review detail ────────────────────────────────────────
        hitl_log = []
        if _table_exists(conn, 'hitl_reviews'):
            rows = conn.execute(
                "SELECT * FROM hitl_reviews ORDER BY created_at DESC"
            ).fetchall()
            for r in rows:
                fields = {}
                try:
                    fields = json.loads(r['fields_json'] or '{}')
                except Exception:
                    pass
                hitl_log.append({
                    'patient_id': r['patient_id'],
                    'analysis_id': r['analysis_id'],
                    'decision': fields.get('decision', 'unknown'),
                    'ai_prediction': fields.get('ai_prediction'),
                    'human_decision': fields.get('human_decision'),
                    'reason_code': fields.get('reason_code'),
                    'reviewer_id': fields.get('reviewer_id'),
                    'created_at': r['created_at'],
                })

        # ── Expert review concordance ─────────────────────────────────
        expert_reviews = []
        expert_agreement_rate = None
        if _table_exists(conn, 'expert_reviews'):
            rows = conn.execute(
                "SELECT * FROM expert_reviews ORDER BY created_at DESC"
            ).fetchall()
            agree_count = 0
            total = 0
            for r in rows:
                expert_reviews.append({
                    'patient_id': r['patient_id'],
                    'analysis_id': r['analysis_id'],
                    'role': r['role'],
                    'expert': r['expert'],
                    'finding': (r['finding'] or '')[:120],
                    'agree_with_ai': r['agree_with_ai'],
                    'note': r['note'],
                    'created_at': r['created_at'],
                })
                total += 1
                if r['agree_with_ai'] == 'agree':
                    agree_count += 1
            if total > 0:
                expert_agreement_rate = round(agree_count / total * 100, 1)

        # ── Per-patient error analysis ────────────────────────────────
        patient_errors = []
        if _table_exists(conn, 'analyses') and _table_exists(conn, 'hitl_reviews'):
            patients = conn.execute(
                "SELECT DISTINCT patient_id FROM analyses"
            ).fetchall()
            for p in patients:
                pid = p['patient_id']
                analyses = conn.execute(
                    "SELECT predicted_label, confidence, signal_quality FROM analyses "
                    "WHERE patient_id=? ORDER BY created_at DESC",
                    (pid,)
                ).fetchall()
                hitl = conn.execute(
                    "SELECT fields_json FROM hitl_reviews WHERE patient_id=?",
                    (pid,)
                ).fetchall()
                overrides_for_patient = 0
                for h in hitl:
                    try:
                        f = json.loads(h['fields_json'] or '{}')
                        if f.get('decision') == 'override':
                            overrides_for_patient += 1
                    except Exception:
                        pass
                patient_errors.append({
                    'patient_id': pid,
                    'total_analyses': len(analyses),
                    'latest_prediction': analyses[0]['predicted_label'] if analyses else None,
                    'latest_confidence': _safe(analyses[0]['confidence']) if analyses else None,
                    'latest_quality': analyses[0]['signal_quality'] if analyses else None,
                    'hitl_overrides': overrides_for_patient,
                    'has_errors': overrides_for_patient > 0,
                })

        # ── Training run detail ───────────────────────────────────────
        training_runs = []
        if _table_exists(conn, 'transaction_log'):
            rows = conn.execute(
                "SELECT * FROM transaction_log WHERE component='training' "
                "ORDER BY ts_local DESC"
            ).fetchall()
            for r in rows:
                detail = r['detail'] or ''
                training_runs.append({
                    'date': r['ts_local'],
                    'action': r['action'],
                    'detail': detail,
                    'success': 'succeeded' in detail,
                    'actor': r['actor'],
                })

        # ── Drift feature breakdown (from report) ─────────────────────
        drift_features = []
        drift_report = _load_json(os.path.join(REPORTS, 'drift_latest.json'))
        if drift_report and isinstance(drift_report, dict):
            features = drift_report.get('features') or drift_report.get('feature_drifts') or []
            if isinstance(features, list):
                for f in features[:30]:
                    if isinstance(f, dict):
                        drift_features.append({
                            'feature': f.get('feature') or f.get('name', ''),
                            'p_value': _safe(f.get('p_value')),
                            'drifted': f.get('drifted', False),
                            'statistic': _safe(f.get('statistic') or f.get('ks_stat')),
                        })

        # ── Daily ML event counts ─────────────────────────────────────
        daily_ml_events = []
        if _table_exists(conn, 'transaction_log'):
            rows = conn.execute(
                "SELECT SUBSTR(ts_local,1,10) as day, component, COUNT(*) as cnt "
                "FROM transaction_log "
                "WHERE component IN ('training','drift','consistency','fairness') "
                "GROUP BY day, component ORDER BY day"
            ).fetchall()
            day_map = {}
            for r in rows:
                d = r['day']
                if d not in day_map:
                    day_map[d] = {'date': d, 'training': 0, 'drift': 0, 'consistency': 0, 'fairness': 0}
                day_map[d][r['component']] = r['cnt']
            daily_ml_events = list(day_map.values())

        return {
            'available': True,
            'feedback_log': feedback_log,
            'hitl_log': hitl_log,
            'expert_reviews': expert_reviews,
            'expert_agreement_rate': expert_agreement_rate,
            'patient_errors': patient_errors,
            'training_runs': training_runs,
            'drift_features': drift_features,
            'daily_ml_events': daily_ml_events,
        }
    finally:
        conn.close()


# ── Definitions ──────────────────────────────────────────────────────────────

def definitions():
    """Continuous Learning concepts, metrics, clinical relevance, remediation."""
    return {
        'available': True,
        'sections': [
            {
                'title': 'Continuous Learning Concepts',
                'items': [
                    {'term': 'Feedback Loop', 'definition': 'Clinician ratings and corrections on AI outputs are collected and used to identify systematic errors for model improvement.'},
                    {'term': 'Retrain Trigger', 'definition': 'An event (data drift, HITL override, low confidence, scheduled) that initiates a model retraining cycle.'},
                    {'term': 'Error Analysis', 'definition': 'Per-patient and per-prediction review of AI errors, overrides, and low-confidence outputs to identify failure modes.'},
                    {'term': 'Drift Monitoring', 'definition': 'Statistical comparison of incoming data distributions against training data to detect when model assumptions no longer hold.'},
                    {'term': 'Consistency Check', 'definition': 'Repeated inference on the same input to verify prediction stability and identify non-deterministic behavior.'},
                    {'term': 'Human-in-the-Loop (HITL)', 'definition': 'Expert clinician review of AI predictions with accept/override decisions that feed back into the learning cycle.'},
                    {'term': 'Redeployment', 'definition': 'Process of replacing a live model with a newly trained version after validation passes quality and safety gates.'},
                ],
            },
            {
                'title': 'Quality Metrics',
                'items': [
                    {'term': 'Training Success Rate', 'definition': 'Percentage of scheduled training runs that complete successfully (target: >90%).'},
                    {'term': 'HITL Override Rate', 'definition': 'Proportion of AI predictions overridden by clinicians, indicating systematic model error patterns.'},
                    {'term': 'Expert Agreement Rate', 'definition': 'Percentage of expert reviews that agree with AI predictions (target: >80% for clinical deployment).'},
                    {'term': 'Feedback Rating', 'definition': 'Clinician satisfaction score (1-5) for AI output quality, averaged across all feedback submissions.'},
                    {'term': 'Confidence Drift', 'definition': 'Change in average prediction confidence over time; declining confidence may indicate distribution shift.'},
                    {'term': 'Drift Detection Rate', 'definition': 'Number of features flagged as drifted per monitoring cycle; high rates trigger retrain evaluation.'},
                ],
            },
            {
                'title': 'Clinical Relevance',
                'items': [
                    {'term': 'FDA AI/ML SaMD Framework', 'definition': 'Requires predetermined change control plans for continuously learning AI/ML-based Software as a Medical Device, including retrain triggers and performance monitoring.'},
                    {'term': 'ILAE Guidelines', 'definition': 'International League Against Epilepsy recommends continuous validation of EEG interpretation tools against expert consensus.'},
                    {'term': 'IEC 62304', 'definition': 'Medical device software lifecycle standard requiring documented change management, verification, and validation for each software update.'},
                    {'term': 'EU AI Act (Article 9)', 'definition': 'High-risk AI systems must implement continuous monitoring and post-market surveillance with feedback mechanisms.'},
                    {'term': 'HIPAA', 'definition': 'Continuous learning systems must maintain PHI protections during retraining; training data must be de-identified or properly consented.'},
                ],
            },
            {
                'title': 'Remediation Strategies',
                'items': [
                    {'term': 'Retrain on Override Data', 'definition': 'Incorporate HITL-corrected labels into training set to address systematic prediction errors.'},
                    {'term': 'Confidence Calibration', 'definition': 'Apply Platt scaling or temperature scaling when confidence drift is detected to maintain calibrated uncertainty estimates.'},
                    {'term': 'Active Learning', 'definition': 'Prioritize low-confidence predictions for expert review, maximizing learning signal per annotation.'},
                    {'term': 'Staged Rollout', 'definition': 'Deploy retrained models to a canary population first, comparing performance against the incumbent before full deployment.'},
                    {'term': 'Rollback Protocol', 'definition': 'Automated reversion to previous model version if post-deployment metrics fall below safety thresholds.'},
                ],
            },
        ],
    }
