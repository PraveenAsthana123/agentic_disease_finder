"""Council of Agents Dashboard — multi-agent consensus system with author/reviewer/chair roles.

Sources:
- config/agent_tasks.json (60 agents: id, task, module, status)
- data/clinical.db (clinical_decisions, expert_reviews, hitl_reviews, transaction_log)
"""

import sqlite3
import os
import json
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BASE, 'data', 'clinical.db')
AGENT_TASKS = os.path.join(BASE, 'config', 'agent_tasks.json')


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _conn():
    return sqlite3.connect(DB)


def _safe_count(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchone()[0]
    except Exception:
        return 0


def _assign_role(task_text):
    """Assign council role based on agent task description."""
    t = (task_text or '').lower()
    if any(kw in t for kw in ('review', 'audit', 'validat', 'check', 'verify', 'inspect')):
        return 'reviewer'
    if any(kw in t for kw in ('plan', 'orchestrat', 'schedul', 'coordinat', 'manag', 'supervis')):
        return 'chair'
    return 'author'


def council_overview():
    """Aggregate council posture: role distribution, consensus metrics,
    decision quality from real clinical.db + agent registry."""

    # ── Agent inventory with role assignment ──────────────────────────
    agents_data = _load_json(AGENT_TASKS)
    agents = agents_data.get('agents', []) if isinstance(agents_data, dict) else []
    total_agents = len(agents)

    role_counts = {'author': 0, 'reviewer': 0, 'chair': 0}
    for a in agents:
        role = _assign_role(a.get('task', ''))
        role_counts[role] = role_counts.get(role, 0) + 1

    # ── Consensus metrics from clinical.db ────────────────────────────
    n_decisions = 0
    n_expert_reviews = 0
    n_hitl_reviews = 0
    consensus_rate = 0.0
    override_rate = 0.0
    daily_quality = []
    review_status = []
    council_events = []

    if os.path.exists(DB):
        conn = _conn()
        cur = conn.cursor()

        n_decisions = _safe_count(cur, "SELECT count(*) FROM clinical_decisions")
        n_expert_reviews = _safe_count(cur, "SELECT count(*) FROM expert_reviews")
        n_hitl_reviews = _safe_count(cur, "SELECT count(*) FROM hitl_reviews")

        # Consensus rate from expert_reviews (agree_with_ai)
        try:
            cur.execute("SELECT count(*) FROM expert_reviews WHERE agree_with_ai = 'agree'")
            n_agree = cur.fetchone()[0]
            consensus_rate = round(n_agree / n_expert_reviews * 100, 1) if n_expert_reviews else 0.0
        except Exception:
            pass

        # Override rate from hitl_reviews (where fields_json indicates override)
        try:
            cur.execute("SELECT count(*) FROM hitl_reviews")
            total_hitl = cur.fetchone()[0]
            # Check for overrides in clinical_decisions where neurologist disagrees
            cur.execute(
                "SELECT count(*) FROM clinical_decisions WHERE neurologist_agreement = 'disagree'"
            )
            n_overrides = cur.fetchone()[0]
            total_reviewed = n_decisions + total_hitl
            override_rate = round(n_overrides / total_reviewed * 100, 1) if total_reviewed else 0.0
        except Exception:
            pass

        # Decision quality trend from transaction_log by day
        try:
            cur.execute("""
                SELECT date(ts_utc) as d, count(*) as cnt
                FROM transaction_log
                WHERE action IN ('sign-off', 'human_decision', 'submit', 'rating', 'assign')
                GROUP BY d ORDER BY d DESC LIMIT 14
            """)
            daily_quality = [{'date': r[0], 'decisions': r[1]}
                             for r in cur.fetchall()][::-1]
        except Exception:
            pass

        # Review status breakdown from expert_reviews
        try:
            cur.execute("""
                SELECT agree_with_ai, count(*) as cnt
                FROM expert_reviews
                GROUP BY agree_with_ai
            """)
            review_status = [{'status': r[0] or 'unknown', 'count': r[1]}
                             for r in cur.fetchall()]
        except Exception:
            pass

        # Council-related events from transaction_log
        try:
            cur.execute("""
                SELECT action, count(*) as cnt
                FROM transaction_log
                WHERE action IN ('sign-off', 'human_decision', 'submit',
                                 'rating', 'assign', 'blocked', 'update',
                                 'analyze', 'check', 'monitor')
                GROUP BY action ORDER BY cnt DESC
            """)
            council_events = [{'action': r[0], 'count': r[1]}
                              for r in cur.fetchall()]
        except Exception:
            pass

        conn.close()

    # Coverage: how many agents have reviewer or chair roles
    coverage_pct = round(
        (role_counts['reviewer'] + role_counts['chair']) / total_agents * 100
    ) if total_agents else 0

    return {
        'available': True,
        'summary': {
            'total_agents': total_agents,
            'authors': role_counts['author'],
            'reviewers': role_counts['reviewer'],
            'chairs': role_counts['chair'],
            'decisions_reviewed': n_decisions + n_expert_reviews + n_hitl_reviews,
            'consensus_rate': consensus_rate,
            'override_rate': override_rate,
            'coverage_pct': coverage_pct,
        },
        'role_distribution': [
            {'role': k, 'count': v}
            for k, v in sorted(role_counts.items(), key=lambda x: -x[1])
        ],
        'decision_quality_trend': daily_quality,
        'review_status_breakdown': review_status,
        'council_events': council_events,
    }


def council_breakdown():
    """Per-agent detail with roles, council sessions, voting history."""

    # ── Per-agent roster ──────────────────────────────────────────────
    agents_data = _load_json(AGENT_TASKS) or {}
    agents = agents_data.get('agents', []) if isinstance(agents_data, dict) else []

    agent_roster = []
    for a in agents:
        role = _assign_role(a.get('task', ''))
        agent_roster.append({
            'id': a.get('id', ''),
            'task': a.get('task', ''),
            'module': a.get('module', ''),
            'role': role,
            'status': a.get('status', 'unknown'),
        })

    # ── Council sessions + voting from clinical.db ────────────────────
    recent_sessions = []
    voting_history = []
    review_assignments = []

    if os.path.exists(DB):
        conn = _conn()
        cur = conn.cursor()

        # Recent council sessions from transaction_log grouped by date
        try:
            cur.execute("""
                SELECT date(ts_utc) as session_date,
                       count(*) as total_events,
                       count(DISTINCT component) as components,
                       count(DISTINCT action) as action_types,
                       count(DISTINCT actor) as participants
                FROM transaction_log
                GROUP BY session_date
                ORDER BY session_date DESC LIMIT 14
            """)
            recent_sessions = [{
                'date': r[0],
                'total_events': r[1],
                'components': r[2],
                'action_types': r[3],
                'participants': r[4],
            } for r in cur.fetchall()]
        except Exception:
            pass

        # Voting history from clinical_decisions + expert_reviews
        try:
            cur.execute("""
                SELECT patient_id, ai_prediction, ai_confidence,
                       neurologist_agreement, final_decision, reviewer, created_at
                FROM clinical_decisions
                ORDER BY created_at DESC LIMIT 20
            """)
            for r in cur.fetchall():
                voting_history.append({
                    'patient_id': r[0],
                    'ai_prediction': r[1],
                    'confidence': r[2],
                    'agreement': r[3],
                    'final_decision': r[4],
                    'reviewer': r[5],
                    'date': r[6],
                    'source': 'clinical_decision',
                })
        except Exception:
            pass

        try:
            cur.execute("""
                SELECT patient_id, role, expert, finding,
                       agree_with_ai, note, created_at
                FROM expert_reviews
                ORDER BY created_at DESC LIMIT 20
            """)
            for r in cur.fetchall():
                voting_history.append({
                    'patient_id': r[0],
                    'role': r[1],
                    'expert': r[2],
                    'finding': r[3],
                    'agreement': r[4],
                    'note': r[5],
                    'date': r[6],
                    'source': 'expert_review',
                })
        except Exception:
            pass

        # Review assignments: which components have review-type actions
        try:
            cur.execute("""
                SELECT component, count(*) as review_count,
                       count(DISTINCT actor) as reviewers
                FROM transaction_log
                WHERE action IN ('sign-off', 'human_decision', 'rating', 'check', 'monitor')
                GROUP BY component
                ORDER BY review_count DESC LIMIT 20
            """)
            review_assignments = [{
                'component': r[0],
                'review_count': r[1],
                'reviewers': r[2],
            } for r in cur.fetchall()]
        except Exception:
            pass

        conn.close()

    return {
        'available': True,
        'agent_roster': agent_roster,
        'recent_sessions': recent_sessions,
        'voting_history': voting_history,
        'review_assignments': review_assignments,
    }


def council_definitions():
    """Definitions for council roles, consensus types, clinical compliance."""
    return {
        'available': True,
        'definitions': [
            {
                'term': 'Author (Council Role)',
                'definition': 'An agent that creates artifacts — predictions, analyses, reports, or data transformations. Authors produce work products that are then reviewed by Reviewers and arbitrated by Chairs.',
            },
            {
                'term': 'Reviewer (Council Role)',
                'definition': 'An agent that validates, audits, or verifies work products from Authors. Reviewers check correctness, compliance, safety, and quality before artifacts are approved.',
            },
            {
                'term': 'Chair (Council Role)',
                'definition': 'An agent that orchestrates, plans, or coordinates council activities. Chairs arbitrate disagreements between Authors and Reviewers, manage scheduling, and ensure quorum.',
            },
            {
                'term': 'Consensus (Majority)',
                'definition': 'Decision-making requiring 2/3 majority approval from council members. Used for standard operational decisions, classification outcomes, and routine approvals.',
            },
            {
                'term': 'Consensus (Unanimous)',
                'definition': 'Decision-making requiring 100% agreement from all council members. Required for security-critical findings, safety overrides, and clinical risk escalations.',
            },
            {
                'term': 'Quorum',
                'definition': 'Minimum number of council members required for a valid decision. Typically requires at least one Chair, one Reviewer, and one Author to be present.',
            },
            {
                'term': 'Override',
                'definition': 'When a human clinician or Chair agent overrules the AI prediction or majority consensus. Overrides are logged for audit and safety tracking.',
            },
            {
                'term': 'Council Session',
                'definition': 'A time-bounded period of agent activity where multiple council members review, vote on, and decide outcomes for clinical or operational tasks.',
            },
            {
                'term': 'Agreement Score',
                'definition': 'Percentage of expert reviewers who agree with the AI prediction. High agreement indicates strong consensus; low agreement triggers escalation.',
            },
            {
                'term': 'Decision Quality',
                'definition': 'Composite metric reflecting consensus rate, override frequency, expert agreement, and HITL review coverage across council sessions.',
            },
        ],
        'compliance_refs': [
            'EU AI Act Art. 14 — Human oversight requirements for high-risk AI',
            'FDA AI/ML SaMD — Software as Medical Device multi-reviewer guidance',
            'IEC 62304 — Medical device software lifecycle with peer review gates',
            'ISO 14971 — Risk management requiring multi-stakeholder review',
            'ILAE Guidelines — Epilepsy classification consensus requirements',
            'HIPAA — Audit trail requirements for clinical decision-making',
        ],
        'remediation': [
            'Ensure every Author agent has at least one designated Reviewer',
            'Implement quorum checks before finalizing clinical decisions',
            'Log all overrides with rationale for regulatory audit compliance',
            'Monitor consensus rate trends — declining rates signal model drift',
            'Rotate Chair assignments to prevent single-point-of-failure governance',
            'Require unanimous consensus for safety-critical classification changes',
        ],
    }


if __name__ == '__main__':
    import json
    print('=== OVERVIEW ===')
    ov = council_overview()
    print(json.dumps(ov.get('summary', {}), indent=2))
    print(f"\nRole distribution: {ov.get('role_distribution', [])}")
    print(f"Decision quality points: {len(ov.get('decision_quality_trend', []))}")
    print(f"Council events: {len(ov.get('council_events', []))}")

    print('\n=== BREAKDOWN ===')
    br = council_breakdown()
    print(f"Agent roster: {len(br.get('agent_roster', []))}")
    print(f"Recent sessions: {len(br.get('recent_sessions', []))}")
    print(f"Voting history: {len(br.get('voting_history', []))}")
    print(f"Review assignments: {len(br.get('review_assignments', []))}")

    print('\n=== DEFINITIONS ===')
    df = council_definitions()
    print(f"Terms: {len(df.get('definitions', []))}")
    print(f"Compliance refs: {len(df.get('compliance_refs', []))}")
