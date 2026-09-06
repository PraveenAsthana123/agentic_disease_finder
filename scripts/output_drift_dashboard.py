"""Output / RAG Drift Dashboard — monitors drift in model prediction outputs
and RAG pipeline responses over time.

Aggregates data from:
- data/clinical.db analyses (prediction outputs: confidence, labels over time)
- data/clinical.db transaction_log (genai_bot + patient_chat + council events)
- data/clinical.db conversation_log (RAG conversations)
- jobs/reports/drift_latest.json (input drift context for correlation)
"""

import sqlite3
import json
import os
import math
from datetime import datetime

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')
REPORTS_DIR = os.path.join(BASE, 'jobs', 'reports')

# Thresholds
CONF_DRIFT_WARN = 0.03   # confidence mean shift > 3% = warning
CONF_DRIFT_HIGH = 0.06   # confidence mean shift > 6% = high
LABEL_DRIFT_WARN = 0.15  # label distribution shift > 15% = warning
LABEL_DRIFT_HIGH = 0.30  # label distribution shift > 30% = high
RAG_QUALITY_WARN = 0.7   # RAG quality below 70% = warning


def _db_query(sql, params=()):
    if not os.path.exists(DB):
        return []
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _load_analyses():
    """Load all prediction outputs."""
    return _db_query(
        "SELECT id, patient_id, disease, predicted_label, confidence, "
        "created_at FROM analyses ORDER BY created_at"
    )


def _load_rag_events():
    """Load RAG pipeline events (genai_bot, patient_chat, council)."""
    return _db_query(
        "SELECT id, component, action, actor, detail, ts_utc, ts_local "
        "FROM transaction_log "
        "WHERE component IN ('genai_bot', 'patient_chat', 'council') "
        "ORDER BY ts_utc"
    )


def _load_input_drift():
    """Load latest input drift for correlation context."""
    path = os.path.join(REPORTS_DIR, 'drift_latest.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data if data.get('available') else None
    return None


def _split_temporal(analyses):
    """Split analyses into early (reference) and late (live) halves for drift."""
    if len(analyses) < 4:
        return analyses, []
    mid = len(analyses) // 2
    return analyses[:mid], analyses[mid:]


def _confidence_stats(rows):
    """Compute confidence statistics for a set of analyses."""
    if not rows:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'n': 0}
    confs = [r['confidence'] for r in rows if r.get('confidence') is not None]
    if not confs:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'n': 0}
    mean = sum(confs) / len(confs)
    variance = sum((c - mean) ** 2 for c in confs) / len(confs) if len(confs) > 1 else 0
    return {
        'mean': round(mean, 4),
        'std': round(math.sqrt(variance), 4),
        'min': round(min(confs), 4),
        'max': round(max(confs), 4),
        'n': len(confs),
    }


def _label_distribution(rows):
    """Compute label distribution."""
    counts = {}
    for r in rows:
        label = r.get('predicted_label', 'unknown')
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values())
    return {k: round(v / total, 4) if total else 0 for k, v in counts.items()}


def _compute_jsd(p_dist, q_dist):
    """Jensen-Shannon divergence between two label distributions."""
    all_labels = set(list(p_dist.keys()) + list(q_dist.keys()))
    if not all_labels:
        return 0.0
    p = [p_dist.get(l, 0.0001) for l in all_labels]
    q = [q_dist.get(l, 0.0001) for l in all_labels]
    # Normalise
    sp, sq = sum(p), sum(q)
    p = [x / sp for x in p]
    q = [x / sq for x in q]
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]

    def kl(a, b):
        return sum(ai * math.log(ai / bi) for ai, bi in zip(a, b) if ai > 0 and bi > 0)

    return round((kl(p, m) + kl(q, m)) / 2, 4)


def _severity(value, warn_thresh, high_thresh):
    if value >= high_thresh:
        return 'high'
    if value >= warn_thresh:
        return 'moderate'
    return 'low'


def _severity_color(severity):
    return {'high': '#ef4444', 'moderate': '#f59e0b', 'low': '#10b981'}.get(severity, '#64748b')


def _rag_event_summary(events):
    """Summarise RAG events by component and action."""
    summary = {}
    for ev in events:
        comp = ev.get('component', 'unknown')
        action = ev.get('action', 'unknown')
        key = f"{comp}:{action}"
        summary[key] = summary.get(key, 0) + 1
    return summary


def overview():
    """KPI-level summary for the Output/RAG Drift dashboard."""
    analyses = _load_analyses()
    rag_events = _load_rag_events()
    input_drift = _load_input_drift()

    if not analyses and not rag_events:
        return {
            'available': False,
            'message': 'No output or RAG data available. Run analyses or use the patient chat first.'
        }

    ref, live = _split_temporal(analyses)
    ref_stats = _confidence_stats(ref)
    live_stats = _confidence_stats(live)

    # Confidence drift
    conf_shift = abs(live_stats['mean'] - ref_stats['mean']) if ref_stats['n'] and live_stats['n'] else 0
    conf_severity = _severity(conf_shift, CONF_DRIFT_WARN, CONF_DRIFT_HIGH)

    # Label distribution drift (JSD)
    ref_labels = _label_distribution(ref)
    live_labels = _label_distribution(live)
    label_jsd = _compute_jsd(ref_labels, live_labels) if ref and live else 0
    label_severity = _severity(label_jsd, 0.05, 0.15)

    # RAG pipeline stats
    rag_summary = _rag_event_summary(rag_events)
    n_queries = sum(1 for e in rag_events if e['action'] in ('query', 'ask'))
    n_blocked = sum(1 for e in rag_events if e['action'] == 'blocked')
    n_answered = sum(1 for e in rag_events if e['action'] in ('answer', 'query', 'ask'))
    rag_success_rate = round(n_answered / max(n_queries + n_blocked, 1), 4)
    rag_severity = _severity(1 - rag_success_rate, 0.1, 0.3)

    # Overall output drift score (composite)
    # Weight: confidence drift 40%, label drift 30%, RAG quality 30%
    conf_score = max(0, 100 - (conf_shift / CONF_DRIFT_HIGH * 100))
    label_score = max(0, 100 - (label_jsd / 0.15 * 100))
    rag_score = rag_success_rate * 100
    overall_score = round(conf_score * 0.4 + label_score * 0.3 + rag_score * 0.3, 1)

    if overall_score >= 80:
        verdict = 'STABLE'
    elif overall_score >= 50:
        verdict = 'MODERATE DRIFT'
    else:
        verdict = 'SEVERE DRIFT'

    # Input drift correlation
    input_frac = input_drift.get('frac_drifted', 0) if input_drift else None

    # Confidence trend (per-analysis over time)
    conf_trend = []
    for a in analyses:
        conf_trend.append({
            'ts': a['created_at'],
            'confidence': a['confidence'],
            'patient': a.get('patient_id', ''),
        })

    return {
        'available': True,
        'overall_score': overall_score,
        'verdict': verdict,
        'n_analyses': len(analyses),
        'n_rag_events': len(rag_events),
        'confidence_drift': {
            'shift': round(conf_shift, 4),
            'severity': conf_severity,
            'ref_mean': ref_stats['mean'],
            'live_mean': live_stats['mean'],
            'ref_n': ref_stats['n'],
            'live_n': live_stats['n'],
        },
        'label_drift': {
            'jsd': label_jsd,
            'severity': label_severity,
            'ref_distribution': ref_labels,
            'live_distribution': live_labels,
        },
        'rag_quality': {
            'success_rate': rag_success_rate,
            'n_queries': n_queries,
            'n_blocked': n_blocked,
            'severity': rag_severity,
        },
        'input_drift_correlation': {
            'input_frac_drifted': input_frac,
            'note': 'High input drift can cause downstream output drift' if input_frac and input_frac > 0.5 else 'Input drift within acceptable bounds' if input_frac is not None else 'No input drift data available',
        },
        'confidence_trend': conf_trend,
        'kpis': [
            {'label': 'Output Drift Score', 'value': f'{overall_score}%',
             'color': '#10b981' if overall_score >= 80 else '#f59e0b' if overall_score >= 50 else '#ef4444'},
            {'label': 'Verdict', 'value': verdict,
             'color': '#10b981' if verdict == 'STABLE' else '#f59e0b' if 'MODERATE' in verdict else '#ef4444'},
            {'label': 'Analyses Monitored', 'value': str(len(analyses))},
            {'label': 'Confidence Shift', 'value': f'{conf_shift:.4f}',
             'color': _severity_color(conf_severity)},
            {'label': 'Label JSD', 'value': f'{label_jsd:.4f}',
             'color': _severity_color(label_severity)},
            {'label': 'RAG Success Rate', 'value': f'{rag_success_rate * 100:.1f}%',
             'color': _severity_color(rag_severity)},
            {'label': 'RAG Events', 'value': str(len(rag_events))},
            {'label': 'Blocked Queries', 'value': str(n_blocked),
             'color': '#ef4444' if n_blocked > 0 else '#10b981'},
        ],
    }


def breakdown():
    """Detailed output drift breakdown — confidence timeline, label distribution,
    RAG event log, input-output drift correlation."""
    analyses = _load_analyses()
    rag_events = _load_rag_events()
    input_drift = _load_input_drift()

    if not analyses and not rag_events:
        return {'available': False}

    ref, live = _split_temporal(analyses)
    ref_stats = _confidence_stats(ref)
    live_stats = _confidence_stats(live)

    # Confidence distribution buckets
    conf_buckets = {}
    for a in analyses:
        c = a.get('confidence', 0)
        bucket = f'{c:.2f}'
        conf_buckets[bucket] = conf_buckets.get(bucket, 0) + 1
    conf_histogram = [{'bucket': k, 'count': v} for k, v in sorted(conf_buckets.items())]

    # Per-patient confidence
    patient_conf = {}
    for a in analyses:
        pid = a.get('patient_id', 'unknown')
        if pid not in patient_conf:
            patient_conf[pid] = []
        patient_conf[pid].append(a['confidence'])
    patient_summary = [
        {'patient': pid, 'mean': round(sum(cs) / len(cs), 4), 'n': len(cs),
         'min': round(min(cs), 4), 'max': round(max(cs), 4)}
        for pid, cs in patient_conf.items()
    ]

    # Confidence timeline chart data
    conf_timeline = [
        {'ts': a['created_at'], 'confidence': round(a['confidence'], 4),
         'patient': a.get('patient_id', '')}
        for a in analyses
    ]

    # Reference vs live comparison for bar chart
    period_comparison = [
        {'period': 'Reference (early)', 'mean_confidence': ref_stats['mean'],
         'n': ref_stats['n'], 'std': ref_stats['std']},
        {'period': 'Live (recent)', 'mean_confidence': live_stats['mean'],
         'n': live_stats['n'], 'std': live_stats['std']},
    ]

    # RAG event breakdown
    rag_by_component = {}
    for ev in rag_events:
        comp = ev['component']
        if comp not in rag_by_component:
            rag_by_component[comp] = {'total': 0, 'actions': {}}
        rag_by_component[comp]['total'] += 1
        action = ev['action']
        rag_by_component[comp]['actions'][action] = rag_by_component[comp]['actions'].get(action, 0) + 1

    rag_component_chart = [
        {'component': comp, 'total': info['total'],
         **{f'action_{k}': v for k, v in info['actions'].items()}}
        for comp, info in rag_by_component.items()
    ]

    # RAG event timeline
    rag_timeline = [
        {'ts': ev.get('ts_local', ev.get('ts_utc', '')),
         'component': ev['component'], 'action': ev['action'],
         'detail': ev.get('detail', ''), 'actor': ev.get('actor', '')}
        for ev in rag_events
    ]

    # Input-output drift correlation data
    correlation = None
    if input_drift and input_drift.get('available'):
        input_frac = input_drift.get('frac_drifted', 0)
        conf_shift = abs(live_stats['mean'] - ref_stats['mean'])
        correlation = {
            'input_drift_frac': input_frac,
            'output_conf_shift': round(conf_shift, 4),
            'correlation_note': 'Input features show {} drift; output confidence shift is {}'.format(
                'severe' if input_frac > 0.5 else 'moderate' if input_frac > 0.2 else 'low',
                'significant' if conf_shift > CONF_DRIFT_WARN else 'minimal'
            ),
        }

    return {
        'available': True,
        'conf_histogram': conf_histogram,
        'conf_timeline': conf_timeline,
        'patient_summary': patient_summary,
        'period_comparison': period_comparison,
        'rag_component_chart': rag_component_chart,
        'rag_timeline': rag_timeline,
        'input_output_correlation': correlation,
        'ref_stats': ref_stats,
        'live_stats': live_stats,
        'n_analyses': len(analyses),
        'n_rag_events': len(rag_events),
    }


def definitions():
    """Definitions tab for Output/RAG Drift dashboard."""
    return {
        'sections': [
            {
                'title': 'Output Drift',
                'items': [
                    {'term': 'Output Drift', 'definition': 'Change in model prediction outputs (confidence scores, label distributions) over time. Unlike input/feature drift which monitors inputs, output drift directly measures whether the model\'s decisions are shifting.'},
                    {'term': 'Confidence Shift', 'definition': 'Absolute difference in mean confidence between early (reference) and recent (live) predictions. A shift > 3% warrants monitoring; > 6% indicates significant drift.'},
                    {'term': 'Label Distribution Shift', 'definition': 'Change in the proportion of predicted labels between reference and live periods, measured by Jensen-Shannon Divergence.'},
                ]
            },
            {
                'title': 'Jensen-Shannon Divergence (JSD)',
                'items': [
                    {'term': 'JSD', 'definition': 'Symmetric measure of similarity between two probability distributions. Ranges from 0 (identical) to ln(2) (~0.693). Based on KL divergence but symmetric and bounded.'},
                    {'term': 'JSD < 0.05', 'definition': 'Distributions are very similar. No label drift detected.'},
                    {'term': '0.05 <= JSD < 0.15', 'definition': 'Moderate divergence. Label distribution is shifting — monitor and investigate.'},
                    {'term': 'JSD >= 0.15', 'definition': 'Significant label drift. The model is making materially different predictions than during the reference period.'},
                ]
            },
            {
                'title': 'RAG Pipeline Drift',
                'items': [
                    {'term': 'RAG Pipeline', 'definition': 'Retrieval-Augmented Generation — the system that retrieves relevant clinical data and generates natural-language responses for patient queries.'},
                    {'term': 'RAG Success Rate', 'definition': 'Percentage of RAG queries that produced an answer (not blocked or failed). Tracks pipeline reliability over time.'},
                    {'term': 'Blocked Queries', 'definition': 'Queries stopped by security/compliance guardrails (e.g., safety agent blocking PII requests). A rising block rate may indicate prompt injection attempts or policy drift.'},
                    {'term': 'Query Components', 'definition': 'patient_chat (patient-facing Q&A), genai_bot (clinical AI assistant), council (multi-agent decision council with compliance/security review).'},
                ]
            },
            {
                'title': 'Input-Output Drift Correlation',
                'items': [
                    {'term': 'Correlation', 'definition': 'When input features drift (detected by Data Drift monitor), outputs may also drift. This section links the two to identify causal chains.'},
                    {'term': 'Causal chain', 'definition': 'Input drift → feature distribution change → model confidence shift → potential label drift. Monitoring both ends validates the full pipeline.'},
                    {'term': 'Decoupled drift', 'definition': 'Output drift without input drift may indicate model degradation (concept drift) rather than data distribution change.'},
                ]
            },
            {
                'title': 'Monitoring Methodology',
                'items': [
                    {'term': 'Temporal split', 'definition': 'Analyses are split chronologically into reference (first half) and live (second half) windows to detect drift over the observation period.'},
                    {'term': 'Composite score', 'definition': 'Weighted combination: confidence drift (40%) + label drift (30%) + RAG quality (30%). Score > 80% = stable, 50-80% = moderate, < 50% = severe.'},
                    {'term': 'Thresholds', 'definition': 'Confidence shift: warn at 3%, high at 6%. Label JSD: warn at 0.05, high at 0.15. RAG success: warn below 90%, high below 70%.'},
                ]
            },
            {
                'title': 'Clinical Relevance',
                'items': [
                    {'term': 'IEC 62304', 'definition': 'Medical device software lifecycle standard. Output drift monitoring is part of post-market surveillance — verifying the device continues to perform as intended.'},
                    {'term': 'FDA AI/ML Framework', 'definition': 'The Predetermined Change Control Plan (PCCP) requires monitoring output behavior, not just input data, to ensure model decisions remain clinically valid.'},
                    {'term': 'EU AI Act', 'definition': 'High-risk AI systems must track output quality and consistency. Output drift monitoring satisfies Article 9 (risk management) and Article 72 (post-market monitoring).'},
                    {'term': 'Clinical impact', 'definition': 'Output drift can cause changed diagnosis rates, altered treatment recommendations, or inconsistent patient risk scores — even when the model code is unchanged.'},
                ]
            },
        ]
    }
