"""AI Observability Dashboard — transaction logging, cost tracking, conversation
monitoring, and analysis auditing across the clinical AI platform.

Aggregates data from:
- data/clinical.db transaction_log table (619 transactions, 20 components, 20 actions)
- data/clinical.db analyses table (21 AI analyses with confidence and signal quality)
- data/clinical.db conversation_log table (314 conversation entries)
- data/clinical.db finops_costs table (978 cost records with token/GPU usage)
"""

import sqlite3
import json
import os
import math
from datetime import datetime, timezone

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')


# ── Helpers ──────────────────────────────────────────────────────────────────

def _connect():
    """Return a DB connection with Row factory, or None if DB missing."""
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
    """Return JSON-safe numeric value (no NaN/Inf)."""
    if val is None:
        return None
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
    return val


# ── Public API ───────────────────────────────────────────────────────────────

def observability_overview():
    """KPI-level summary: transaction volumes, costs, component and actor
    distributions, daily/hourly patterns, conversation roles."""
    conn = _connect()
    if conn is None:
        return {'available': False, 'message': 'Database not found.'}

    try:
        # ── KPIs ────────────────────────────────────────────────────────
        total_transactions = 0
        total_components = 0
        total_actors = 0
        if _table_exists(conn, 'transaction_log'):
            total_transactions = conn.execute(
                "SELECT COUNT(*) FROM transaction_log"
            ).fetchone()[0]
            total_components = conn.execute(
                "SELECT COUNT(DISTINCT component) FROM transaction_log"
            ).fetchone()[0]
            total_actors = conn.execute(
                "SELECT COUNT(DISTINCT actor) FROM transaction_log"
            ).fetchone()[0]

        total_analyses = 0
        avg_confidence = None
        if _table_exists(conn, 'analyses'):
            total_analyses = conn.execute(
                "SELECT COUNT(*) FROM analyses"
            ).fetchone()[0]
            row = conn.execute(
                "SELECT AVG(confidence) FROM analyses WHERE confidence IS NOT NULL"
            ).fetchone()
            avg_confidence = _safe(round(row[0], 3)) if row[0] is not None else None

        total_conversations = 0
        if _table_exists(conn, 'conversation_log'):
            total_conversations = conn.execute(
                "SELECT COUNT(*) FROM conversation_log"
            ).fetchone()[0]

        total_cost_records = 0
        total_cost_usd = 0.0
        if _table_exists(conn, 'finops_costs'):
            total_cost_records = conn.execute(
                "SELECT COUNT(*) FROM finops_costs"
            ).fetchone()[0]
            row = conn.execute(
                "SELECT SUM(cost_usd) FROM finops_costs"
            ).fetchone()
            total_cost_usd = _safe(round(row[0], 2)) if row[0] is not None else 0.0

        kpis = {
            'total_transactions': total_transactions,
            'total_components': total_components,
            'total_actors': total_actors,
            'total_analyses': total_analyses,
            'total_conversations': total_conversations,
            'total_cost_records': total_cost_records,
            'avg_confidence': avg_confidence,
            'total_cost_usd': total_cost_usd,
        }

        # ── Component distribution ──────────────────────────────────────
        component_distribution = []
        if _table_exists(conn, 'transaction_log'):
            rows = conn.execute(
                "SELECT component, COUNT(*) as cnt FROM transaction_log "
                "GROUP BY component ORDER BY cnt DESC"
            ).fetchall()
            component_distribution = [
                {'component': r['component'], 'count': r['cnt']}
                for r in rows
            ]

        # ── Action distribution ─────────────────────────────────────────
        action_distribution = []
        if _table_exists(conn, 'transaction_log'):
            rows = conn.execute(
                "SELECT action, COUNT(*) as cnt FROM transaction_log "
                "GROUP BY action ORDER BY cnt DESC"
            ).fetchall()
            action_distribution = [
                {'action': r['action'], 'count': r['cnt']}
                for r in rows
            ]

        # ── Actor activity ──────────────────────────────────────────────
        actor_activity = []
        if _table_exists(conn, 'transaction_log'):
            rows = conn.execute(
                "SELECT actor, COUNT(*) as cnt FROM transaction_log "
                "GROUP BY actor ORDER BY cnt DESC"
            ).fetchall()
            actor_activity = [
                {'actor': r['actor'], 'count': r['cnt']}
                for r in rows
            ]

        # ── Daily transaction volume ────────────────────────────────────
        daily_transaction_volume = []
        if _table_exists(conn, 'transaction_log'):
            rows = conn.execute(
                "SELECT DATE(ts_utc) as dt, COUNT(*) as cnt FROM transaction_log "
                "WHERE ts_utc IS NOT NULL "
                "GROUP BY dt ORDER BY dt ASC"
            ).fetchall()
            daily_transaction_volume = [
                {'date': r['dt'], 'count': r['cnt']}
                for r in rows if r['dt'] is not None
            ]

        # ── Hourly heatmap ──────────────────────────────────────────────
        hourly_heatmap = []
        if _table_exists(conn, 'transaction_log'):
            rows = conn.execute(
                "SELECT CAST(strftime('%H', ts_utc) AS INTEGER) as hr, COUNT(*) as cnt "
                "FROM transaction_log WHERE ts_utc IS NOT NULL "
                "GROUP BY hr ORDER BY hr ASC"
            ).fetchall()
            hourly_heatmap = [
                {'hour': r['hr'], 'count': r['cnt']}
                for r in rows if r['hr'] is not None
            ]

        # ── Cost by category ────────────────────────────────────────────
        cost_by_category = []
        if _table_exists(conn, 'finops_costs'):
            rows = conn.execute(
                "SELECT category, SUM(cost_usd) as total_cost, SUM(requests) as requests "
                "FROM finops_costs GROUP BY category ORDER BY total_cost DESC"
            ).fetchall()
            cost_by_category = [
                {
                    'category': r['category'],
                    'total_cost': _safe(round(r['total_cost'], 4)) if r['total_cost'] else 0.0,
                    'requests': r['requests'] or 0,
                }
                for r in rows
            ]

        # ── Conversation role distribution ──────────────────────────────
        conversation_role_distribution = []
        if _table_exists(conn, 'conversation_log'):
            rows = conn.execute(
                "SELECT role, COUNT(*) as cnt FROM conversation_log "
                "GROUP BY role ORDER BY cnt DESC"
            ).fetchall()
            conversation_role_distribution = [
                {'role': r['role'], 'count': r['cnt']}
                for r in rows
            ]

    finally:
        conn.close()

    return {
        'available': True,
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'kpis': kpis,
        'component_distribution': component_distribution,
        'action_distribution': action_distribution,
        'actor_activity': actor_activity,
        'daily_transaction_volume': daily_transaction_volume,
        'hourly_heatmap': hourly_heatmap,
        'cost_by_category': cost_by_category,
        'conversation_role_distribution': conversation_role_distribution,
    }


def observability_breakdown():
    """Detailed drill-downs: per-component actions, per-actor components,
    transaction timelines, cost analysis, patient profiles, error actions,
    conversation timeline."""
    conn = _connect()
    if conn is None:
        return {'available': False, 'message': 'Database not found.'}

    try:
        # ── Per-component actions ───────────────────────────────────────
        per_component_actions = []
        if _table_exists(conn, 'transaction_log'):
            components = conn.execute(
                "SELECT DISTINCT component FROM transaction_log ORDER BY component"
            ).fetchall()
            for comp_row in components:
                comp = comp_row['component']
                actions = conn.execute(
                    "SELECT action, COUNT(*) as cnt FROM transaction_log "
                    "WHERE component = ? GROUP BY action ORDER BY cnt DESC",
                    (comp,),
                ).fetchall()
                per_component_actions.append({
                    'component': comp,
                    'actions': [{'action': a['action'], 'count': a['cnt']} for a in actions],
                })

        # ── Per-actor components ────────────────────────────────────────
        per_actor_components = []
        if _table_exists(conn, 'transaction_log'):
            actors = conn.execute(
                "SELECT DISTINCT actor FROM transaction_log ORDER BY actor"
            ).fetchall()
            for actor_row in actors:
                actor = actor_row['actor']
                comps = conn.execute(
                    "SELECT component, COUNT(*) as cnt FROM transaction_log "
                    "WHERE actor = ? GROUP BY component ORDER BY cnt DESC",
                    (actor,),
                ).fetchall()
                per_actor_components.append({
                    'actor': actor,
                    'components': [{'component': c['component'], 'count': c['cnt']} for c in comps],
                })

        # ── Transaction timeline (top 5 components by volume) ──────────
        transaction_timeline = []
        if _table_exists(conn, 'transaction_log'):
            top5 = conn.execute(
                "SELECT component FROM transaction_log "
                "GROUP BY component ORDER BY COUNT(*) DESC LIMIT 5"
            ).fetchall()
            top5_names = [r['component'] for r in top5]
            if top5_names:
                placeholders = ','.join('?' for _ in top5_names)
                rows = conn.execute(
                    f"SELECT DATE(ts_utc) as dt, component, COUNT(*) as cnt "
                    f"FROM transaction_log "
                    f"WHERE component IN ({placeholders}) AND ts_utc IS NOT NULL "
                    f"GROUP BY dt, component ORDER BY dt ASC",
                    top5_names,
                ).fetchall()
                transaction_timeline = [
                    {'date': r['dt'], 'component': r['component'], 'count': r['cnt']}
                    for r in rows if r['dt'] is not None
                ]

        # ── Cost timeline ───────────────────────────────────────────────
        cost_timeline = []
        if _table_exists(conn, 'finops_costs'):
            rows = conn.execute(
                "SELECT cost_date, SUM(cost_usd) as total_cost, "
                "SUM(requests) as requests, SUM(tokens_in) as tokens_in, "
                "SUM(tokens_out) as tokens_out "
                "FROM finops_costs GROUP BY cost_date ORDER BY cost_date ASC"
            ).fetchall()
            cost_timeline = [
                {
                    'date': r['cost_date'],
                    'total_cost': _safe(round(r['total_cost'], 4)) if r['total_cost'] else 0.0,
                    'requests': r['requests'] or 0,
                    'tokens_in': r['tokens_in'] or 0,
                    'tokens_out': r['tokens_out'] or 0,
                }
                for r in rows
            ]

        # ── Cost by service ─────────────────────────────────────────────
        cost_by_service = []
        if _table_exists(conn, 'finops_costs'):
            rows = conn.execute(
                "SELECT model_or_service, SUM(cost_usd) as total_cost, "
                "SUM(requests) as requests, "
                "CASE WHEN SUM(requests) > 0 THEN SUM(cost_usd) / SUM(requests) ELSE 0 END as avg_cost "
                "FROM finops_costs GROUP BY model_or_service ORDER BY total_cost DESC"
            ).fetchall()
            cost_by_service = [
                {
                    'model_or_service': r['model_or_service'],
                    'total_cost': _safe(round(r['total_cost'], 4)) if r['total_cost'] else 0.0,
                    'requests': r['requests'] or 0,
                    'avg_cost': _safe(round(r['avg_cost'], 6)) if r['avg_cost'] else 0.0,
                }
                for r in rows
            ]

        # ── Analysis summary ────────────────────────────────────────────
        analysis_summary = []
        if _table_exists(conn, 'analyses'):
            rows = conn.execute(
                "SELECT patient_id, disease, confidence, signal_quality, created_at "
                "FROM analyses ORDER BY created_at ASC"
            ).fetchall()
            analysis_summary = [
                {
                    'patient_id': r['patient_id'],
                    'disease': r['disease'],
                    'confidence': _safe(round(r['confidence'], 3)) if r['confidence'] is not None else None,
                    'signal_quality': r['signal_quality'],
                    'created_at': r['created_at'],
                }
                for r in rows
            ]

        # ── Patient transaction profiles ────────────────────────────────
        patient_transaction_profiles = []
        if _table_exists(conn, 'transaction_log'):
            rows = conn.execute(
                "SELECT patient_id, COUNT(*) as cnt, "
                "GROUP_CONCAT(DISTINCT component) as components, "
                "MAX(ts_utc) as latest_ts "
                "FROM transaction_log "
                "WHERE patient_id IS NOT NULL AND patient_id != '' "
                "GROUP BY patient_id ORDER BY cnt DESC"
            ).fetchall()
            patient_transaction_profiles = [
                {
                    'patient_id': r['patient_id'],
                    'transaction_count': r['cnt'],
                    'components': r['components'].split(',') if r['components'] else [],
                    'latest_ts': r['latest_ts'],
                }
                for r in rows
            ]

        # ── Error actions ───────────────────────────────────────────────
        error_actions = []
        if _table_exists(conn, 'transaction_log'):
            total_tx = conn.execute(
                "SELECT COUNT(*) FROM transaction_log"
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT action, COUNT(*) as cnt FROM transaction_log "
                "WHERE action IN ('blocked', 'delete') "
                "GROUP BY action ORDER BY cnt DESC"
            ).fetchall()
            error_actions = [
                {
                    'action': r['action'],
                    'count': r['cnt'],
                    'pct': round(r['cnt'] / max(total_tx, 1) * 100, 2),
                }
                for r in rows
            ]

        # ── Conversation timeline ───────────────────────────────────────
        conversation_timeline = []
        if _table_exists(conn, 'conversation_log'):
            rows = conn.execute(
                "SELECT DATE(ts_utc) as dt, COUNT(*) as cnt "
                "FROM conversation_log WHERE ts_utc IS NOT NULL "
                "GROUP BY dt ORDER BY dt ASC"
            ).fetchall()
            conversation_timeline = [
                {'date': r['dt'], 'count': r['cnt']}
                for r in rows if r['dt'] is not None
            ]

    finally:
        conn.close()

    return {
        'available': True,
        'per_component_actions': per_component_actions,
        'per_actor_components': per_actor_components,
        'transaction_timeline': transaction_timeline,
        'cost_timeline': cost_timeline,
        'cost_by_service': cost_by_service,
        'analysis_summary': analysis_summary,
        'patient_transaction_profiles': patient_transaction_profiles,
        'error_actions': error_actions,
        'conversation_timeline': conversation_timeline,
    }


def observability_definitions():
    """Definitions tab for the AI Observability dashboard."""
    return {
        'sections': [
            {
                'title': 'AI Observability Methods',
                'items': [
                    {
                        'term': 'Transaction Logging',
                        'definition': (
                            'Every interaction with the clinical AI platform is recorded '
                            'in the transaction_log table with component, action, actor, '
                            'patient reference, and UTC timestamp. Provides a complete '
                            'audit trail for regulatory compliance and system behaviour '
                            'analysis.'
                        ),
                    },
                    {
                        'term': 'Cost Tracking',
                        'definition': (
                            'The finops_costs table records per-request costs including '
                            'token usage (input/output), GPU minutes, and USD cost by '
                            'model/service and component. Enables cost attribution, '
                            'budget forecasting, and efficiency optimisation across AI '
                            'inference and training workloads.'
                        ),
                    },
                    {
                        'term': 'Conversation Monitoring',
                        'definition': (
                            'The conversation_log captures all human-AI dialogue with '
                            'role attribution (user, assistant, system) and timestamps. '
                            'Supports quality assurance, hallucination detection, and '
                            'clinical decision audit trails.'
                        ),
                    },
                    {
                        'term': 'Analysis Auditing',
                        'definition': (
                            'Each AI analysis (EEG classification, disease prediction) '
                            'is recorded with confidence score, signal quality, disease '
                            'label, and full result JSON. Enables post-market performance '
                            'monitoring and model drift detection.'
                        ),
                    },
                ],
            },
            {
                'title': 'System Components',
                'items': [
                    {
                        'term': 'Clinical Components',
                        'definition': (
                            'patient_master (demographics, registration), medications '
                            '(prescriptions, dosing), seizure_diary (event logging), '
                            'assessment (clinical instruments), form (data capture), '
                            'consultant:neurologist (specialist interactions).'
                        ),
                    },
                    {
                        'term': 'AI/ML Components',
                        'definition': (
                            'eeg_upload (raw signal ingestion), training (model training '
                            'runs), cv_pipeline (computer vision), genai_bot (generative '
                            'AI assistant), fairness (bias monitoring), graph_db '
                            '(knowledge graph).'
                        ),
                    },
                    {
                        'term': 'Collaboration Components',
                        'definition': (
                            'patient_chat (patient messaging), team_chat (clinical team), '
                            'chat_group (group discussions), council (multidisciplinary '
                            'review), expert_review (specialist consultation), feedback '
                            '(ratings and comments).'
                        ),
                    },
                    {
                        'term': 'Governance Components',
                        'definition': (
                            'component_finding (audit findings), video_frames (video '
                            'analysis artifacts), and cross-cutting actors including '
                            'neurologist, Clinical Advisor, system, compliance_agent, '
                            'and security_agent.'
                        ),
                    },
                ],
            },
            {
                'title': 'Metrics & KPIs',
                'items': [
                    {
                        'term': 'Transaction Volume',
                        'definition': (
                            'Total number of logged interactions across all components. '
                            'Daily and hourly breakdowns reveal usage patterns, peak '
                            'hours, and capacity requirements. Sudden drops may indicate '
                            'system outages; spikes may indicate batch processing or '
                            'incident response.'
                        ),
                    },
                    {
                        'term': 'Cost per Request',
                        'definition': (
                            'Average USD cost per AI inference or training request, '
                            'computed from finops_costs. Tracked by model/service to '
                            'identify expensive operations and optimisation targets. '
                            'Includes token costs (input + output) and GPU compute.'
                        ),
                    },
                    {
                        'term': 'Confidence Scores',
                        'definition': (
                            'Average AI prediction confidence across analyses. Low '
                            'confidence may indicate model uncertainty, out-of-distribution '
                            'inputs, or degraded signal quality. Tracked over time for '
                            'drift detection.'
                        ),
                    },
                    {
                        'term': 'Actor Coverage',
                        'definition': (
                            'Distribution of actions across actors (human clinicians, '
                            'AI agents, system processes). Ensures appropriate human '
                            'oversight of AI-driven decisions and identifies over-reliance '
                            'on automated actors.'
                        ),
                    },
                ],
            },
            {
                'title': 'Clinical Relevance',
                'items': [
                    {
                        'term': 'IEC 62304',
                        'definition': (
                            'Medical device software lifecycle standard requires '
                            'traceability of all software activities. Transaction logging '
                            'provides the audit trail evidence for software change control, '
                            'problem resolution, and post-market surveillance activities '
                            'mandated by IEC 62304 Section 6 (Software Maintenance).'
                        ),
                    },
                    {
                        'term': 'FDA AI/ML PCCP',
                        'definition': (
                            'Predetermined Change Control Plan requires continuous '
                            'performance monitoring of AI/ML-based medical devices. '
                            'Observability data (confidence trends, error rates, cost '
                            'anomalies) provides the real-world performance evidence '
                            'required for PCCP submissions and ongoing compliance.'
                        ),
                    },
                    {
                        'term': 'ILAE Guidelines',
                        'definition': (
                            'International League Against Epilepsy classification and '
                            'management guidelines require documented clinical decision '
                            'support rationale. Conversation logs and analysis audit '
                            'trails demonstrate that AI recommendations align with '
                            'ILAE-endorsed classification frameworks.'
                        ),
                    },
                ],
            },
            {
                'title': 'Remediation Strategies',
                'items': [
                    {
                        'term': 'Transaction Anomaly Response',
                        'definition': (
                            'When blocked actions or deletes exceed baseline rates, '
                            'investigate root causes: permission misconfigurations, '
                            'data validation failures, or adversarial access attempts. '
                            'Cross-reference with actor and component to isolate the '
                            'affected subsystem.'
                        ),
                    },
                    {
                        'term': 'Cost Spike Investigation',
                        'definition': (
                            'When daily cost exceeds 2x the rolling average, review '
                            'the cost_by_service breakdown for runaway inference loops, '
                            'unoptimised prompts, or unexpected training jobs. Check '
                            'token counts for prompt injection or context window abuse.'
                        ),
                    },
                    {
                        'term': 'Confidence Degradation Protocol',
                        'definition': (
                            'When average confidence drops below 0.5 or trends downward '
                            'over 7 days, trigger model performance review: check for '
                            'data distribution shift, signal quality degradation, or '
                            'concept drift. Escalate to human-in-the-loop review for '
                            'low-confidence predictions.'
                        ),
                    },
                    {
                        'term': 'Conversation Quality Audit',
                        'definition': (
                            'Periodically sample conversation logs for hallucination '
                            'patterns, inappropriate clinical advice, or failure to '
                            'escalate. Flag conversations where the AI provided definitive '
                            'diagnoses without appropriate caveats or clinician referral.'
                        ),
                    },
                    {
                        'term': 'Actor Coverage Rebalancing',
                        'definition': (
                            'When system or AI actors dominate transaction volume with '
                            'insufficient human clinician oversight, review the '
                            'human-in-the-loop policies. Ensure critical clinical '
                            'decisions have documented clinician sign-off in the '
                            'transaction log.'
                        ),
                    },
                ],
            },
        ],
    }


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import pprint
    print('=== OVERVIEW ===')
    ov = observability_overview()
    pprint.pprint(ov)
    print(f"\n=== BREAKDOWN ===")
    bd = observability_breakdown()
    if bd.get('available'):
        print(f"Per-component actions: {len(bd.get('per_component_actions', []))}")
        print(f"Per-actor components: {len(bd.get('per_actor_components', []))}")
        print(f"Transaction timeline entries: {len(bd.get('transaction_timeline', []))}")
        print(f"Cost timeline entries: {len(bd.get('cost_timeline', []))}")
        print(f"Cost by service entries: {len(bd.get('cost_by_service', []))}")
        print(f"Analysis summary entries: {len(bd.get('analysis_summary', []))}")
        print(f"Patient profiles: {len(bd.get('patient_transaction_profiles', []))}")
        print(f"Error actions: {len(bd.get('error_actions', []))}")
        print(f"Conversation timeline entries: {len(bd.get('conversation_timeline', []))}")
        print('\nTop 5 components by action variety:')
        for c in bd['per_component_actions'][:5]:
            print(f"  {c['component']}: {len(c['actions'])} actions")
        print('\nError actions:')
        for e in bd['error_actions']:
            print(f"  {e['action']}: {e['count']} ({e['pct']}%)")
    else:
        pprint.pprint(bd)
    print('\n=== DEFINITIONS ===')
    defs = observability_definitions()
    for sec in defs['sections']:
        print(f"  {sec['title']}: {len(sec['items'])} items")
