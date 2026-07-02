"""Data Lineage Dashboard — transaction-level data flow analytics from clinical.db.

Tracks how patient data moves through system components: ingestion → processing
→ analysis → reporting. Provides audit trail visibility, actor attribution,
component dependency mapping, and temporal activity patterns.

Sources:
- transaction_log table (645+ events across 25+ components)
- Derived lineage chains: patient data touchpoints across components
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


# ── Pipeline stage classification ──
STAGE_MAP = {
    'ingest': 'Ingestion',
    'create': 'Ingestion',
    'extract': 'Processing',
    'process': 'Processing',
    'analyze': 'Analysis',
    'query': 'Analysis',
    'build': 'Analysis',
    'check': 'Validation',
    'monitor': 'Monitoring',
    'scheduled_train': 'Training',
    'log': 'Logging',
    'message': 'Communication',
    'answer': 'Communication',
    'ask': 'Communication',
    'save': 'Storage',
    'update': 'Storage',
    'submit': 'Storage',
    'sign-off': 'Governance',
    'human_decision': 'Governance',
    'blocked': 'Governance',
    'assign': 'Governance',
    'rating': 'Feedback',
    'add': 'Ingestion',
    'delete': 'Storage',
}

COMPONENT_DOMAIN = {
    'patient_master': 'Clinical',
    'medications': 'Clinical',
    'patient_chat': 'Clinical',
    'seizure_diary': 'Clinical',
    'assessment': 'Clinical',
    'eeg_upload': 'Signal',
    'video_frames': 'Signal',
    'cv_pipeline': 'Signal',
    'training': 'ML',
    'drift': 'ML',
    'consistency': 'ML',
    'fairness': 'ML',
    'council': 'Governance',
    'expert_review': 'Governance',
    'feedback': 'Governance',
    'clinical_trust': 'Governance',
    'graph_db': 'Infrastructure',
    'team_chat': 'Communication',
    'chat_group': 'Communication',
    'genai_bot': 'AI',
    'consultant:neurologist': 'Clinical',
    'form': 'Clinical',
    'component_finding': 'QA',
    'finops': 'Infrastructure',
    'appointments': 'Clinical',
}


def _classify_stage(action):
    return STAGE_MAP.get((action or '').lower().strip(), 'Other')


def _classify_domain(component):
    return COMPONENT_DOMAIN.get((component or '').lower().strip(), 'Other')


def data_lineage_overview():
    """Full lineage dashboard KPIs, pipeline flow, and component graph."""
    conn = _conn()
    cur = conn.cursor()

    # ── Core KPIs ──
    total_events = _safe(cur, 'SELECT COUNT(*) FROM transaction_log')
    unique_components = _safe(cur, 'SELECT COUNT(DISTINCT component) FROM transaction_log')
    unique_actions = _safe(cur, 'SELECT COUNT(DISTINCT action) FROM transaction_log')
    unique_actors = _safe(cur, 'SELECT COUNT(DISTINCT actor) FROM transaction_log')
    unique_patients = _safe(cur, "SELECT COUNT(DISTINCT patient_id) FROM transaction_log WHERE patient_id IS NOT NULL AND patient_id != ''")
    date_range = _safe_rows(cur, 'SELECT MIN(DATE(ts_local)), MAX(DATE(ts_local)) FROM transaction_log')
    first_date = date_range[0][0] if date_range else None
    last_date = date_range[0][1] if date_range else None
    span_days = 0
    if first_date and last_date:
        from datetime import datetime
        d1 = datetime.strptime(first_date, '%Y-%m-%d')
        d2 = datetime.strptime(last_date, '%Y-%m-%d')
        span_days = (d2 - d1).days + 1

    avg_daily = round(total_events / span_days, 1) if span_days > 0 else 0

    # ── Pipeline stage distribution ──
    action_rows = _safe_rows(cur, 'SELECT action, COUNT(*) FROM transaction_log GROUP BY action')
    stage_counts = defaultdict(int)
    for action, cnt in action_rows:
        stage = _classify_stage(action)
        stage_counts[stage] += cnt
    pipeline_stages = [{'stage': s, 'count': c} for s, c in sorted(stage_counts.items(), key=lambda x: -x[1])]

    # ── Component domain distribution ──
    comp_rows = _safe_rows(cur, 'SELECT component, COUNT(*) FROM transaction_log GROUP BY component ORDER BY COUNT(*) DESC')
    domain_counts = defaultdict(int)
    component_detail = []
    for comp, cnt in comp_rows:
        domain = _classify_domain(comp)
        domain_counts[domain] += cnt
        component_detail.append({'component': comp, 'domain': domain, 'events': cnt})
    domain_dist = [{'domain': d, 'count': c} for d, c in sorted(domain_counts.items(), key=lambda x: -x[1])]

    # ── Actor distribution ──
    actor_rows = _safe_rows(cur, 'SELECT actor, COUNT(*) FROM transaction_log GROUP BY actor ORDER BY COUNT(*) DESC')
    actor_dist = [{'actor': r[0], 'events': r[1]} for r in actor_rows]

    # ── Component-to-component flow (lineage edges) ──
    # For each patient, find sequential component transitions
    patient_flows = _safe_rows(cur, '''
        SELECT patient_id, component, action, ts_utc
        FROM transaction_log
        WHERE patient_id IS NOT NULL AND patient_id != ''
        ORDER BY patient_id, ts_utc
    ''')
    edge_counts = defaultdict(int)
    prev_patient = None
    prev_component = None
    for pid, comp, action, ts in patient_flows:
        if pid == prev_patient and comp != prev_component and prev_component:
            edge = f'{prev_component} → {comp}'
            edge_counts[edge] += 1
        prev_patient = pid
        prev_component = comp
    lineage_edges = [{'from': e.split(' → ')[0], 'to': e.split(' → ')[1], 'weight': w}
                     for e, w in sorted(edge_counts.items(), key=lambda x: -x[1])[:20]]

    # ── Daily activity trend ──
    daily_rows = _safe_rows(cur, '''
        SELECT DATE(ts_local) as day, COUNT(*) as cnt
        FROM transaction_log
        WHERE ts_local IS NOT NULL
        GROUP BY DATE(ts_local)
        ORDER BY day
    ''')
    daily_trend = [{'date': r[0], 'events': r[1]} for r in daily_rows]

    # ── Hourly pattern ──
    hourly_rows = _safe_rows(cur, '''
        SELECT CAST(STRFTIME('%H', ts_local) AS INTEGER) as hr, COUNT(*) as cnt
        FROM transaction_log
        WHERE ts_local IS NOT NULL
        GROUP BY hr ORDER BY hr
    ''')
    hourly_pattern = [{'hour': r[0], 'events': r[1]} for r in hourly_rows]

    conn.close()

    return {
        'available': True,
        'kpis': {
            'total_events': total_events,
            'unique_components': unique_components,
            'unique_actions': unique_actions,
            'unique_actors': unique_actors,
            'patients_tracked': unique_patients,
            'span_days': span_days,
            'avg_daily_events': avg_daily,
            'lineage_edges': len(lineage_edges),
        },
        'pipeline_stages': pipeline_stages,
        'domain_distribution': domain_dist,
        'component_detail': component_detail,
        'actor_distribution': actor_dist,
        'lineage_edges': lineage_edges,
        'daily_trend': daily_trend,
        'hourly_pattern': hourly_pattern,
    }


def data_lineage_breakdown():
    """Per-patient lineage chains, per-component action breakdown, and audit trail."""
    conn = _conn()
    cur = conn.cursor()

    # ── Per-patient data touchpoints ──
    patient_rows = _safe_rows(cur, '''
        SELECT patient_id, COUNT(*) as events,
               COUNT(DISTINCT component) as components,
               COUNT(DISTINCT action) as actions,
               MIN(ts_local) as first_seen,
               MAX(ts_local) as last_seen
        FROM transaction_log
        WHERE patient_id IS NOT NULL AND patient_id != ''
        GROUP BY patient_id
        ORDER BY events DESC
    ''')
    patient_lineage = []
    for r in patient_rows:
        # Get component chain for this patient
        chain_rows = _safe_rows(cur, '''
            SELECT DISTINCT component FROM transaction_log
            WHERE patient_id = ? ORDER BY ts_utc
        ''', (r[0],))
        chain = [c[0] for c in chain_rows]
        patient_lineage.append({
            'patient_id': r[0],
            'total_events': r[1],
            'components_touched': r[2],
            'action_types': r[3],
            'first_seen': r[4],
            'last_seen': r[5],
            'component_chain': chain,
        })

    # ── Per-component action breakdown ──
    cross_rows = _safe_rows(cur, '''
        SELECT component, action, COUNT(*) as cnt
        FROM transaction_log
        GROUP BY component, action
        ORDER BY component, cnt DESC
    ''')
    comp_actions = defaultdict(list)
    for comp, action, cnt in cross_rows:
        comp_actions[comp].append({'action': action, 'count': cnt})
    component_breakdown = [{'component': c, 'actions': acts, 'total': sum(a['count'] for a in acts)}
                           for c, acts in sorted(comp_actions.items(), key=lambda x: -sum(a['count'] for a in x[1]))]

    # ── Recent audit trail (last 30 events) ──
    recent_rows = _safe_rows(cur, '''
        SELECT id, patient_id, component, action, actor, detail, ts_local
        FROM transaction_log
        ORDER BY ts_utc DESC LIMIT 30
    ''')
    audit_trail = []
    for r in recent_rows:
        audit_trail.append({
            'id': r[0],
            'patient_id': r[1] or '--',
            'component': r[2],
            'action': r[3],
            'actor': r[4],
            'detail': (r[5] or '')[:100],
            'timestamp': r[6],
        })

    # ── Action-stage mapping summary ──
    action_rows = _safe_rows(cur, 'SELECT action, COUNT(*) FROM transaction_log GROUP BY action ORDER BY COUNT(*) DESC')
    action_stages = [{'action': r[0], 'stage': _classify_stage(r[0]), 'count': r[1]} for r in action_rows]

    conn.close()

    return {
        'available': True,
        'patient_lineage': patient_lineage,
        'component_breakdown': component_breakdown,
        'audit_trail': audit_trail,
        'action_stages': action_stages,
    }


def data_lineage_definitions():
    """Data lineage metric definitions for tooltip overlays."""
    return {
        'available': True,
        'definitions': {
            'lineage_concepts': {
                'Data Lineage': 'The end-to-end path that patient data follows through the system — from ingestion through processing, analysis, and reporting.',
                'Pipeline Stage': 'A phase in the data processing pipeline: Ingestion → Processing → Analysis → Validation → Monitoring → Storage.',
                'Component': 'A system module that touches patient data — e.g., eeg_upload, assessment, drift, council.',
                'Lineage Edge': 'A directional connection between two components, indicating data flow from one to the next for a given patient.',
                'Audit Trail': 'A chronological record of every data operation, including who acted, what they did, and when.',
                'Actor': 'The entity that performed a data operation — system (automated), neurologist (human), compliance_agent (AI agent).',
            },
            'quality_metrics': {
                'Events per Day': 'Average number of transaction log events per day — indicates system throughput.',
                'Component Coverage': 'Number of distinct components that have logged at least one event.',
                'Patient Tracking Depth': 'Average number of components touched per patient — higher means more comprehensive data lineage.',
                'Edge Density': 'Number of unique component-to-component transitions observed — indicates data flow complexity.',
            },
            'clinical_relevance': {
                'ILAE': 'Data lineage supports ILAE documentation standards by tracking how seizure data moves from recording to classification.',
                'IEC 62304': 'Transaction logging provides software lifecycle traceability required by IEC 62304 §8 (software maintenance).',
                'FDA AI/ML': 'End-to-end data lineage supports FDA AI/ML framework requirements for data provenance and model reproducibility.',
                'HIPAA': 'Audit trail of all patient data access and processing supports HIPAA §164.312 (audit controls) and §164.528 (accounting of disclosures).',
                'EU AI Act': 'Data lineage documentation supports EU AI Act Article 12 (record-keeping) for high-risk AI systems.',
            },
            'remediation': {
                'Missing lineage edges': 'Components that process patient data but are not connected in the lineage graph may indicate logging gaps.',
                'Low patient coverage': 'Patients with few component touchpoints may be missing assessments or follow-up processing.',
                'Actor imbalance': 'Heavy system-actor dominance is expected for automation, but zero human-actor events may indicate missing clinical review.',
                'Temporal gaps': 'Days with zero events may indicate system downtime or logging failures.',
            },
        },
    }
