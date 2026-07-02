"""AI Security Dashboard — security posture analytics from clinical.db.

Monitors access control, PHI sensitivity, risk classification, human oversight,
and audit trail for the clinical EEG/epilepsy platform. Provides actor-component
access matrix, risk-level distribution, anomaly detection, and governance event
tracking.

Sources:
- transaction_log table (647+ events across 25+ components, 7+ actors)
- hitl_reviews table (human-in-the-loop review records)
- Derived metrics: risk classification, PHI access, anomaly indicators
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


# ── Risk classification of actions ──
RISK_MAP = {
    'delete': 'HIGH',
    'human_decision': 'HIGH',
    'sign-off': 'HIGH',
    'update': 'MEDIUM',
    'submit': 'MEDIUM',
    'ingest': 'MEDIUM',
    'create': 'MEDIUM',
    'add': 'MEDIUM',
    'query': 'LOW',
    'check': 'LOW',
    'monitor': 'LOW',
    'log': 'LOW',
    'analyze': 'LOW',
    'extract': 'LOW',
    'process': 'LOW',
    'build': 'LOW',
    'message': 'LOW',
    'answer': 'LOW',
    'ask': 'LOW',
    'save': 'MEDIUM',
    'assign': 'MEDIUM',
    'blocked': 'HIGH',
    'rating': 'LOW',
    'scheduled_train': 'MEDIUM',
}

# ── PHI-sensitive components ──
PHI_COMPONENTS = {'patient_master', 'medications', 'patient_chat', 'seizure_diary'}

# ── Security-relevant actors ──
SECURITY_ACTORS = {'security_agent', 'compliance_agent'}


def _classify_risk(action):
    return RISK_MAP.get((action or '').lower().strip(), 'LOW')


def ai_security_overview():
    """Full security dashboard KPIs, risk distribution, and access matrix."""
    conn = _conn()
    cur = conn.cursor()

    # ── Core KPIs ──
    total_events = _safe(cur, 'SELECT COUNT(*) FROM transaction_log')
    security_events = _safe(cur, '''
        SELECT COUNT(*) FROM transaction_log
        WHERE actor IN ('security_agent', 'compliance_agent')
    ''')
    distinct_actors = _safe(cur, 'SELECT COUNT(DISTINCT actor) FROM transaction_log')
    distinct_components = _safe(cur, 'SELECT COUNT(DISTINCT component) FROM transaction_log')
    hitl_count = _safe(cur, 'SELECT COUNT(*) FROM hitl_reviews')

    # ── PHI access metrics ──
    phi_access_count = _safe(cur, '''
        SELECT COUNT(*) FROM transaction_log
        WHERE component IN ('patient_master', 'medications', 'patient_chat', 'seizure_diary')
    ''')
    phi_access_rate = round(phi_access_count / total_events * 100, 1) if total_events > 0 else 0

    # ── Human oversight rate ──
    human_oversight_actions = _safe(cur, '''
        SELECT COUNT(*) FROM transaction_log
        WHERE action IN ('human_decision', 'sign-off')
    ''')
    human_oversight_rate = round(human_oversight_actions / total_events * 100, 2) if total_events > 0 else 0

    # ── Components accessed by each actor type ──
    actor_component_rows = _safe_rows(cur, '''
        SELECT actor, COUNT(DISTINCT component) as comp_count
        FROM transaction_log
        GROUP BY actor
        ORDER BY comp_count DESC
    ''')
    actor_component_coverage = [{'actor': r[0], 'components_accessed': r[1]} for r in actor_component_rows]

    # ── Risk level distribution ──
    action_rows = _safe_rows(cur, 'SELECT action, COUNT(*) FROM transaction_log GROUP BY action')
    risk_counts = defaultdict(int)
    for action, cnt in action_rows:
        risk = _classify_risk(action)
        risk_counts[risk] += cnt
    risk_distribution = [{'risk_level': lvl, 'count': c}
                         for lvl, c in sorted(risk_counts.items(), key=lambda x: -x[1])]

    # ── Daily security event trend ──
    daily_rows = _safe_rows(cur, '''
        SELECT DATE(ts_local) as day, COUNT(*) as cnt
        FROM transaction_log
        WHERE ts_local IS NOT NULL
        GROUP BY DATE(ts_local)
        ORDER BY day
    ''')
    daily_trend = [{'date': r[0], 'events': r[1]} for r in daily_rows]

    # ── Actor-component access matrix ──
    matrix_rows = _safe_rows(cur, '''
        SELECT actor, component, COUNT(*) as cnt
        FROM transaction_log
        GROUP BY actor, component
        ORDER BY actor, cnt DESC
    ''')
    access_matrix = defaultdict(list)
    for actor, comp, cnt in matrix_rows:
        access_matrix[actor].append({'component': comp, 'events': cnt})
    actor_component_matrix = [{'actor': a, 'components': comps}
                              for a, comps in sorted(access_matrix.items())]

    # ── Hourly access pattern ──
    hourly_rows = _safe_rows(cur, '''
        SELECT CAST(STRFTIME('%H', ts_local) AS INTEGER) as hr, COUNT(*) as cnt
        FROM transaction_log
        WHERE ts_local IS NOT NULL
        GROUP BY hr ORDER BY hr
    ''')
    hourly_pattern = [{'hour': r[0], 'events': r[1]} for r in hourly_rows]

    # ── Sensitive component access by actor ──
    phi_actor_rows = _safe_rows(cur, '''
        SELECT actor, component, COUNT(*) as cnt
        FROM transaction_log
        WHERE component IN ('patient_master', 'medications', 'patient_chat', 'seizure_diary')
        GROUP BY actor, component
        ORDER BY cnt DESC
    ''')
    phi_access_by_actor = [{'actor': r[0], 'component': r[1], 'events': r[2]} for r in phi_actor_rows]

    conn.close()

    return {
        'available': True,
        'kpis': {
            'total_events': total_events,
            'security_events': security_events,
            'distinct_actors': distinct_actors,
            'distinct_components': distinct_components,
            'hitl_review_count': hitl_count,
            'phi_access_count': phi_access_count,
            'phi_access_rate_pct': phi_access_rate,
            'human_oversight_rate_pct': human_oversight_rate,
        },
        'actor_component_coverage': actor_component_coverage,
        'risk_distribution': risk_distribution,
        'daily_trend': daily_trend,
        'actor_component_matrix': actor_component_matrix,
        'hourly_pattern': hourly_pattern,
        'phi_access_by_actor': phi_access_by_actor,
    }


def ai_security_breakdown():
    """PHI access log, per-actor profiles, high-risk actions, anomaly indicators, governance events."""
    conn = _conn()
    cur = conn.cursor()

    # ── PHI access log (last 30 events on sensitive components) ──
    phi_rows = _safe_rows(cur, '''
        SELECT id, patient_id, component, action, actor, detail, ts_local
        FROM transaction_log
        WHERE component IN ('patient_master', 'medications', 'patient_chat', 'seizure_diary')
        ORDER BY ts_utc DESC LIMIT 30
    ''')
    phi_access_log = []
    for r in phi_rows:
        phi_access_log.append({
            'id': r[0],
            'patient_id': r[1] or '--',
            'component': r[2],
            'action': r[3],
            'actor': r[4],
            'detail': (r[5] or '')[:100],
            'timestamp': r[6],
        })

    # ── Per-actor access profiles ──
    actor_rows = _safe_rows(cur, '''
        SELECT actor, COUNT(*) as events,
               COUNT(DISTINCT component) as components,
               COUNT(DISTINCT action) as actions,
               MIN(ts_local) as first_seen,
               MAX(ts_local) as last_seen
        FROM transaction_log
        GROUP BY actor
        ORDER BY events DESC
    ''')
    actor_profiles = []
    for r in actor_rows:
        # Get top components for this actor
        top_comps = _safe_rows(cur, '''
            SELECT component, COUNT(*) as cnt
            FROM transaction_log WHERE actor = ?
            GROUP BY component ORDER BY cnt DESC LIMIT 5
        ''', (r[0],))
        # Get top actions for this actor
        top_actions = _safe_rows(cur, '''
            SELECT action, COUNT(*) as cnt
            FROM transaction_log WHERE actor = ?
            GROUP BY action ORDER BY cnt DESC LIMIT 5
        ''', (r[0],))
        actor_profiles.append({
            'actor': r[0],
            'total_events': r[1],
            'components_accessed': r[2],
            'action_types': r[3],
            'first_seen': r[4],
            'last_seen': r[5],
            'top_components': [{'component': c[0], 'count': c[1]} for c in top_comps],
            'top_actions': [{'action': a[0], 'count': a[1]} for a in top_actions],
        })

    # ── High-risk action log (delete, human_decision, sign-off, blocked) ──
    high_risk_rows = _safe_rows(cur, '''
        SELECT id, patient_id, component, action, actor, detail, ts_local
        FROM transaction_log
        WHERE action IN ('delete', 'human_decision', 'sign-off', 'blocked')
        ORDER BY ts_utc DESC
    ''')
    high_risk_log = []
    for r in high_risk_rows:
        high_risk_log.append({
            'id': r[0],
            'patient_id': r[1] or '--',
            'component': r[2],
            'action': r[3],
            'actor': r[4],
            'detail': (r[5] or '')[:100],
            'timestamp': r[6],
            'risk_level': 'HIGH',
        })

    # ── Anomaly indicators ──
    # Off-hours activity (before 7 AM or after 8 PM)
    off_hours_count = _safe(cur, '''
        SELECT COUNT(*) FROM transaction_log
        WHERE ts_local IS NOT NULL
        AND (CAST(STRFTIME('%H', ts_local) AS INTEGER) < 7
             OR CAST(STRFTIME('%H', ts_local) AS INTEGER) >= 20)
    ''')
    off_hours_rate = round(off_hours_count / max(_safe(cur, 'SELECT COUNT(*) FROM transaction_log WHERE ts_local IS NOT NULL'), 1) * 100, 1)

    # Actors with unusual breadth (accessing many distinct components)
    breadth_rows = _safe_rows(cur, '''
        SELECT actor, COUNT(DISTINCT component) as comp_count, COUNT(*) as events
        FROM transaction_log
        GROUP BY actor
        HAVING comp_count > 5
        ORDER BY comp_count DESC
    ''')
    wide_access_actors = [{'actor': r[0], 'components_accessed': r[1], 'total_events': r[2]} for r in breadth_rows]

    # Actors accessing PHI components
    phi_actors = _safe_rows(cur, '''
        SELECT actor, COUNT(*) as phi_events, COUNT(DISTINCT component) as phi_components
        FROM transaction_log
        WHERE component IN ('patient_master', 'medications', 'patient_chat', 'seizure_diary')
        GROUP BY actor
        ORDER BY phi_events DESC
    ''')
    phi_access_actors = [{'actor': r[0], 'phi_events': r[1], 'phi_components': r[2]} for r in phi_actors]

    anomaly_indicators = {
        'off_hours_events': off_hours_count,
        'off_hours_rate_pct': off_hours_rate,
        'wide_access_actors': wide_access_actors,
        'phi_access_actors': phi_access_actors,
    }

    # ── Governance events (blocked, sign-off, human_decision) ──
    gov_rows = _safe_rows(cur, '''
        SELECT id, patient_id, component, action, actor, detail, ts_local
        FROM transaction_log
        WHERE action IN ('blocked', 'sign-off', 'human_decision')
        ORDER BY ts_utc DESC
    ''')
    governance_events = []
    for r in gov_rows:
        governance_events.append({
            'id': r[0],
            'patient_id': r[1] or '--',
            'component': r[2],
            'action': r[3],
            'actor': r[4],
            'detail': (r[5] or '')[:100],
            'timestamp': r[6],
        })

    conn.close()

    return {
        'available': True,
        'phi_access_log': phi_access_log,
        'actor_profiles': actor_profiles,
        'high_risk_log': high_risk_log,
        'anomaly_indicators': anomaly_indicators,
        'governance_events': governance_events,
    }


def ai_security_definitions():
    """AI security metric definitions for tooltip overlays."""
    return {
        'available': True,
        'definitions': {
            'security_concepts': {
                'Access Control': 'Mechanisms that restrict who can read, write, or modify patient data and system components — tracked via actor identity in the transaction log.',
                'PHI Sensitivity': 'Protected Health Information (PHI) components — patient_master, medications, patient_chat, seizure_diary — require heightened access controls and audit scrutiny.',
                'Risk Classification': 'Categorization of actions by potential impact: HIGH (delete, human_decision, sign-off, blocked), MEDIUM (update, submit, ingest, create, add, save, assign, scheduled_train), LOW (query, check, monitor, log, analyze, extract, process, build, message, answer, ask, rating).',
                'Audit Trail': 'A tamper-evident chronological record of every system operation — who acted, what component was touched, what action was performed, and when.',
                'Human-in-the-Loop': 'A governance mechanism requiring human review and approval before AI-generated decisions are finalized — tracked via human_decision and sign-off actions and hitl_reviews table.',
                'Threat Surface': 'The set of components and access paths through which unauthorized access or data breach could occur — measured by actor breadth, PHI exposure, and off-hours activity.',
            },
            'quality_metrics': {
                'Security Event Rate': 'Proportion of total events attributed to security_agent and compliance_agent actors — indicates active security monitoring coverage.',
                'PHI Access Rate': 'Percentage of all transaction log events that touch PHI-sensitive components — lower is better for least-privilege compliance.',
                'Human Oversight Ratio': 'Percentage of events that are human_decision or sign-off actions — indicates clinical governance engagement.',
                'Actor Diversity Index': 'Count of distinct actors in the system — higher diversity with proper role assignment indicates mature access control.',
                'Risk Score': 'Weighted aggregate of HIGH (×3), MEDIUM (×2), and LOW (×1) risk events — provides a single security posture indicator.',
            },
            'clinical_relevance': {
                'HIPAA §164.312': 'Technical safeguards requiring access controls (§164.312(a)), audit controls (§164.312(b)), integrity controls (§164.312(c)), and transmission security (§164.312(e)) for electronic PHI.',
                'FDA AI/ML': 'FDA guidance on AI/ML-based Software as a Medical Device (SaMD) requires cybersecurity documentation, access control validation, and audit logging for clinical AI systems.',
                'IEC 62304 §5.7': 'Medical device software risk management — requires identification of hazardous situations from software failures, including unauthorized access and data corruption.',
                'EU AI Act Article 15': 'High-risk AI systems must achieve appropriate levels of accuracy, robustness, and cybersecurity — including resilience against unauthorized third-party attempts to alter use or performance.',
                'NIST AI RMF': 'NIST AI Risk Management Framework — Manage function requires ongoing monitoring of AI system risks including security, privacy, and adversarial threats across the AI lifecycle.',
            },
            'remediation': {
                'Unauthorized PHI access': 'If an actor accesses PHI-sensitive components without a documented clinical role, investigate immediately — review actor permissions, restrict access, and log the incident per HIPAA breach notification rules.',
                'Low human oversight': 'If human_decision and sign-off events are below 2% of total events, increase clinical review touchpoints — add mandatory sign-off gates before AI-generated recommendations reach patients.',
                'Off-hours access spikes': 'Unusual activity outside business hours (before 7 AM or after 8 PM) may indicate automated processes or unauthorized access — correlate with known cron jobs and flag unexplained spikes.',
                'Missing audit records': 'Gaps in the transaction log timeline may indicate logging failures or tampering — verify log completeness, check system uptime records, and ensure all components write to the audit trail.',
            },
        },
    }
