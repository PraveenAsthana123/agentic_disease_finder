"""Decision AI Dashboard — decision routing, HITL overrides, confidence calibration,
and audit trail from real clinical.db data.

Aggregates data from:
- data/clinical.db analyses (21 analyses with confidence scores + predictions)
- data/clinical.db clinical_decisions (neurologist sign-off records)
- data/clinical.db hitl_reviews (human-in-the-loop override decisions)
- data/clinical.db transaction_log (635+ auditable events)
"""

import sqlite3
import json
import os
import math
from datetime import datetime, timezone
from collections import defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# Decision routing thresholds (matching clinical_db.decision_route)
THRESHOLDS = {
    'auto_approve': 0.85,
    'review': 0.60,
    'escalate': 0.0,  # anything below review
}


def _connect():
    if not os.path.exists(DB):
        return None
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def _safe_float(val):
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def _route_label(confidence):
    c = _safe_float(confidence)
    if c is None:
        return 'unknown'
    if c >= THRESHOLDS['auto_approve']:
        return 'auto_approve'
    if c >= THRESHOLDS['review']:
        return 'review'
    return 'escalate'


def _load_analyses(conn):
    rows = conn.execute(
        'SELECT id, patient_id, disease, predicted_label, confidence, '
        'signal_quality, result_json, created_at FROM analyses ORDER BY id'
    ).fetchall()
    results = []
    for r in rows:
        rj = {}
        if r['result_json']:
            try:
                rj = json.loads(r['result_json'])
            except (json.JSONDecodeError, TypeError):
                pass
        pred = rj.get('prediction', {})
        conf = _safe_float(r['confidence']) or _safe_float(pred.get('confidence'))
        results.append({
            'id': r['id'],
            'patient_id': r['patient_id'],
            'disease': r['disease'],
            'predicted_label': r['predicted_label'] or pred.get('predicted_label', 'Unknown'),
            'confidence': conf,
            'signal_quality': r['signal_quality'] or rj.get('analysis', {}).get('signal_quality', 'Unknown'),
            'route': _route_label(conf),
            'class_probs': pred.get('class_probabilities', {}),
            'created_at': r['created_at'],
        })
    return results


def _load_hitl_reviews(conn):
    rows = conn.execute(
        'SELECT id, patient_id, analysis_id, fields_json, created_at '
        'FROM hitl_reviews ORDER BY id'
    ).fetchall()
    results = []
    for r in rows:
        fields = {}
        if r['fields_json']:
            try:
                fields = json.loads(r['fields_json'])
            except (json.JSONDecodeError, TypeError):
                pass
        results.append({
            'id': r['id'],
            'patient_id': r['patient_id'],
            'analysis_id': r['analysis_id'],
            'decision': fields.get('decision', 'unknown'),
            'ai_prediction': fields.get('ai_prediction', ''),
            'human_decision': fields.get('human_decision', ''),
            'reason_code': fields.get('reason_code', ''),
            'created_at': r['created_at'],
        })
    return results


def _load_clinical_decisions(conn):
    rows = conn.execute(
        'SELECT * FROM clinical_decisions ORDER BY id'
    ).fetchall()
    results = []
    for r in rows:
        results.append({
            'id': r['id'],
            'patient_id': r['patient_id'],
            'analysis_id': r['analysis_id'],
            'ai_prediction': r['ai_prediction'],
            'ai_confidence': _safe_float(r['ai_confidence']),
            'top_channels': r['top_channels'],
            'artifact_risk': r['artifact_risk'],
            'neurologist_agreement': r['neurologist_agreement'],
            'final_decision': r['final_decision'],
            'reviewer': r['reviewer'],
            'note': r['note'],
            'created_at': r['created_at'],
        })
    return results


def _load_transaction_log(conn):
    rows = conn.execute(
        'SELECT id, patient_id, component, action, actor, detail, ts_utc, ts_local '
        'FROM transaction_log ORDER BY id'
    ).fetchall()
    results = []
    for r in rows:
        results.append({
            'id': r['id'],
            'patient_id': r['patient_id'],
            'component': r['component'],
            'action': r['action'],
            'actor': r['actor'],
            'detail': (r['detail'] or '')[:200],
            'ts_utc': r['ts_utc'],
            'ts_local': r['ts_local'],
        })
    return results


# ── Overview ────────────────────────────────────────────────────────────────

def decision_overview():
    conn = _connect()
    if not conn:
        return {'error': 'Database not available'}

    analyses = _load_analyses(conn)
    hitl = _load_hitl_reviews(conn)
    decisions = _load_clinical_decisions(conn)
    txlog = _load_transaction_log(conn)
    conn.close()

    # Route distribution
    route_counts = defaultdict(int)
    for a in analyses:
        route_counts[a['route']] += 1

    # Confidence distribution (histogram buckets)
    conf_buckets = {'0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0,
                    '0.6-0.8': 0, '0.8-1.0': 0}
    conf_values = []
    for a in analyses:
        c = a['confidence']
        if c is not None:
            conf_values.append(c)
            if c < 0.2:
                conf_buckets['0.0-0.2'] += 1
            elif c < 0.4:
                conf_buckets['0.2-0.4'] += 1
            elif c < 0.6:
                conf_buckets['0.4-0.6'] += 1
            elif c < 0.8:
                conf_buckets['0.6-0.8'] += 1
            else:
                conf_buckets['0.8-1.0'] += 1

    avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0

    # HITL stats
    override_count = sum(1 for h in hitl if h['decision'] == 'override')
    confirm_count = sum(1 for h in hitl if h['decision'] != 'override')

    # Transaction log stats
    components = defaultdict(int)
    actors = defaultdict(int)
    actions = defaultdict(int)
    for t in txlog:
        components[t['component']] += 1
        actors[t['actor']] += 1
        actions[t['action']] += 1

    # Disease breakdown
    disease_stats = defaultdict(lambda: {'count': 0, 'confidences': [], 'routes': defaultdict(int)})
    for a in analyses:
        d = a['disease'] or 'unknown'
        disease_stats[d]['count'] += 1
        if a['confidence'] is not None:
            disease_stats[d]['confidences'].append(a['confidence'])
        disease_stats[d]['routes'][a['route']] += 1

    disease_summary = []
    for d, s in sorted(disease_stats.items()):
        avg_c = sum(s['confidences']) / len(s['confidences']) if s['confidences'] else 0
        disease_summary.append({
            'disease': d,
            'count': s['count'],
            'avg_confidence': round(avg_c, 3),
            'routes': dict(s['routes']),
        })

    return _json_safe({
        'kpis': {
            'total_analyses': len(analyses),
            'avg_confidence': round(avg_conf, 3),
            'auto_approve_count': route_counts.get('auto_approve', 0),
            'review_count': route_counts.get('review', 0),
            'escalate_count': route_counts.get('escalate', 0),
            'hitl_overrides': override_count,
            'hitl_confirms': confirm_count,
            'audit_events': len(txlog),
        },
        'route_distribution': [
            {'route': k, 'count': v}
            for k, v in sorted(route_counts.items())
        ],
        'confidence_histogram': [
            {'bucket': k, 'count': v}
            for k, v in conf_buckets.items()
        ],
        'thresholds': THRESHOLDS,
        'disease_summary': disease_summary,
        'audit_summary': {
            'total_events': len(txlog),
            'components': len(components),
            'top_components': sorted(
                [{'name': k, 'count': v} for k, v in components.items()],
                key=lambda x: -x['count']
            )[:10],
            'top_actors': sorted(
                [{'name': k, 'count': v} for k, v in actors.items()],
                key=lambda x: -x['count']
            )[:5],
            'top_actions': sorted(
                [{'name': k, 'count': v} for k, v in actions.items()],
                key=lambda x: -x['count']
            )[:10],
        },
        'clinical_decisions': decisions,
    })


# ── Breakdown ───────────────────────────────────────────────────────────────

def decision_breakdown():
    conn = _connect()
    if not conn:
        return {'error': 'Database not available'}

    analyses = _load_analyses(conn)
    hitl = _load_hitl_reviews(conn)
    txlog = _load_transaction_log(conn)
    conn.close()

    # Per-patient decision summary
    patient_map = defaultdict(lambda: {
        'analyses': [], 'hitl_reviews': [], 'audit_events': 0
    })
    for a in analyses:
        patient_map[a['patient_id']]['analyses'].append({
            'id': a['id'],
            'disease': a['disease'],
            'predicted_label': a['predicted_label'],
            'confidence': a['confidence'],
            'route': a['route'],
            'signal_quality': a['signal_quality'],
        })
    for h in hitl:
        patient_map[h['patient_id']]['hitl_reviews'].append(h)
    for t in txlog:
        patient_map[t['patient_id']]['audit_events'] += 1

    patient_summaries = []
    for pid, data in sorted(patient_map.items()):
        confs = [a['confidence'] for a in data['analyses'] if a['confidence'] is not None]
        routes = defaultdict(int)
        for a in data['analyses']:
            routes[a['route']] += 1
        patient_summaries.append({
            'patient_id': pid,
            'analysis_count': len(data['analyses']),
            'avg_confidence': round(sum(confs) / len(confs), 3) if confs else None,
            'routes': dict(routes),
            'hitl_count': len(data['hitl_reviews']),
            'audit_events': data['audit_events'],
            'overrides': sum(1 for h in data['hitl_reviews'] if h['decision'] == 'override'),
        })

    # Confidence calibration — group predictions by confidence bucket,
    # check if HITL agrees
    calibration = []
    for bucket_label, lo, hi in [
        ('0.5-0.6', 0.5, 0.6), ('0.6-0.7', 0.6, 0.7),
        ('0.7-0.8', 0.7, 0.8), ('0.8-0.9', 0.8, 0.9), ('0.9-1.0', 0.9, 1.01),
    ]:
        in_bucket = [a for a in analyses if a['confidence'] is not None and lo <= a['confidence'] < hi]
        hitl_ids = {h['analysis_id'] for h in hitl}
        reviewed = [a for a in in_bucket if a['id'] in hitl_ids]
        override_ids = {h['analysis_id'] for h in hitl if h['decision'] == 'override'}
        overridden = [a for a in in_bucket if a['id'] in override_ids]
        calibration.append({
            'bucket': bucket_label,
            'total': len(in_bucket),
            'reviewed': len(reviewed),
            'overridden': len(overridden),
            'agreement_rate': round(1 - len(overridden) / len(reviewed), 3) if reviewed else None,
        })

    # Timeline of audit events by component
    monthly = defaultdict(lambda: defaultdict(int))
    for t in txlog:
        ts = t['ts_local'] or t['ts_utc'] or ''
        month = ts[:7] if len(ts) >= 7 else 'unknown'
        monthly[month][t['component']] += 1

    timeline = []
    for month in sorted(monthly.keys()):
        entry = {'month': month}
        entry.update(monthly[month])
        timeline.append(entry)

    return _json_safe({
        'patient_summaries': patient_summaries,
        'per_analysis': [{
            'id': a['id'],
            'patient_id': a['patient_id'],
            'disease': a['disease'],
            'predicted_label': a['predicted_label'],
            'confidence': a['confidence'],
            'route': a['route'],
            'signal_quality': a['signal_quality'],
            'class_probs': a['class_probs'],
        } for a in analyses],
        'hitl_reviews': hitl,
        'calibration': calibration,
        'audit_timeline': timeline,
    })


# ── Definitions ─────────────────────────────────────────────────────────────

def decision_definitions():
    return {
        'title': 'Decision AI — Definitions & Clinical Relevance',
        'sections': [
            {
                'heading': 'Decision AI Concepts',
                'items': [
                    {'term': 'Decision Routing', 'definition': 'Confidence-based triage of AI predictions into auto-approve (≥0.85), review (0.60–0.85), or escalate (<0.60) pathways. Ensures high-confidence predictions proceed efficiently while uncertain ones receive human oversight.'},
                    {'term': 'HITL (Human-in-the-Loop)', 'definition': 'A clinician reviews, confirms, or overrides the AI prediction. Override decisions are logged with reason codes for audit and model improvement.'},
                    {'term': 'Confidence Calibration', 'definition': 'Measures whether predicted confidence scores match actual accuracy. A well-calibrated model with 70% confidence should be correct ~70% of the time.'},
                    {'term': 'Decision Audit Trail', 'definition': 'Complete log of every action (ingest, predict, review, override, export) with actor, timestamp, and detail — required for regulatory compliance.'},
                    {'term': 'Override Rate', 'definition': 'Fraction of AI decisions that a human clinician changed. High override rates signal model drift or systematic bias.'},
                ]
            },
            {
                'heading': 'Routing Thresholds',
                'items': [
                    {'term': 'Auto-Approve (≥0.85)', 'definition': 'AI prediction confidence high enough for automated acceptance. Still logged for audit but does not require manual review.'},
                    {'term': 'Review (0.60–0.85)', 'definition': 'Moderate confidence — a clinician must review and confirm or override before the prediction is finalized.'},
                    {'term': 'Escalate (<0.60)', 'definition': 'Low confidence — routed to a senior specialist for expert assessment. May indicate poor signal quality, unusual presentation, or model uncertainty.'},
                ]
            },
            {
                'heading': 'Quality Metrics',
                'items': [
                    {'term': 'Agreement Rate', 'definition': 'Percentage of reviewed predictions where the clinician confirmed the AI decision (1 − override_rate).'},
                    {'term': 'Decision Latency', 'definition': 'Time from AI prediction to final human decision. Shorter latency improves patient throughput.'},
                    {'term': 'Audit Coverage', 'definition': 'Fraction of predictions with a complete audit trail (prediction + review + final decision). 100% coverage required for regulatory compliance.'},
                ]
            },
            {
                'heading': 'Clinical Relevance & Regulatory',
                'items': [
                    {'term': 'IEC 62304 (Medical Device Software)', 'definition': 'Decision routing implements risk-based classification: Class A (auto-approve for high confidence), Class B/C (human review for moderate/low confidence). Audit trail satisfies traceability requirements.'},
                    {'term': 'FDA AI/ML PCCP', 'definition': 'Predetermined Change Control Plan — confidence thresholds and override tracking support continuous learning with locked decision boundaries.'},
                    {'term': 'ILAE Classification', 'definition': 'Decision AI routes seizure type predictions through clinician review to ensure alignment with ILAE 2017 classification standards.'},
                    {'term': 'ISO 14971 (Risk Management)', 'definition': 'Escalation pathway for low-confidence predictions is a risk control measure. Override logging provides post-market surveillance data.'},
                    {'term': 'EU AI Act (High-Risk)', 'definition': 'Clinical decision support is high-risk AI. Decision routing with human oversight, audit trails, and override capability satisfies Articles 14 (human oversight) and 12 (record-keeping).'},
                ]
            },
            {
                'heading': 'Remediation Strategies',
                'items': [
                    {'term': 'High Override Rate', 'definition': 'If >20% of reviewed predictions are overridden: investigate model drift, retrain on recent data, review feature quality, adjust confidence thresholds.'},
                    {'term': 'Calibration Drift', 'definition': 'If confidence no longer matches accuracy: apply Platt scaling or isotonic regression recalibration. Monitor with reliability diagrams.'},
                    {'term': 'Audit Gaps', 'definition': 'If predictions lack complete audit trails: enforce logging middleware, add transaction_log writes to all decision endpoints, alert on missing records.'},
                    {'term': 'Threshold Adjustment', 'definition': 'Thresholds should be reviewed quarterly. Lower auto-approve threshold only if override rate is <5% and calibration is verified.'},
                ]
            },
        ],
    }


if __name__ == '__main__':
    import json as j
    o = decision_overview()
    print(j.dumps(o, indent=2, default=str)[:3000])
