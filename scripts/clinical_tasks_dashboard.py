"""Clinical Tasks Dashboard — role-operational task analytics from clinical.db.

Tracks clinical workflow tasks, request resolution, form assignments, and
transaction activity across the system.

Sources:
- operator_requests table (255 entries with status tracking)
- form_assignments table (assessment task assignments)
- transaction_log table (641 workflow events across 25 components)
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


# --- Status classification ---
DONE_STATUSES = {'addressed', 'completed', 'done', 'resolved'}
OPEN_STATUSES = {'open', 'pending', 'logged'}
REJECTED_STATUSES = {'rejected', 'not-implemented', 'wont-fix'}


def _classify_status(status):
    s = (status or '').lower().strip()
    if s in DONE_STATUSES:
        return 'done'
    if s in REJECTED_STATUSES:
        return 'rejected'
    return 'open'


def clinical_tasks_overview():
    """Full tasks dashboard KPIs and summary stats."""
    conn = _conn()
    cur = conn.cursor()

    # --- Operator Requests ---
    total_requests = _safe(cur, 'SELECT COUNT(*) FROM operator_requests')
    status_rows = _safe_rows(cur, 'SELECT status, COUNT(*) FROM operator_requests GROUP BY status ORDER BY COUNT(*) DESC')
    status_dist = {r[0]: r[1] for r in status_rows}

    # Classify into done/open/rejected
    done_count = sum(v for k, v in status_dist.items() if _classify_status(k) == 'done')
    open_count = sum(v for k, v in status_dist.items() if _classify_status(k) == 'open')
    rejected_count = sum(v for k, v in status_dist.items() if _classify_status(k) == 'rejected')

    completion_rate = round(done_count / total_requests * 100, 1) if total_requests else 0

    # Category breakdown
    cat_rows = _safe_rows(cur, 'SELECT category, COUNT(*) FROM operator_requests GROUP BY category ORDER BY COUNT(*) DESC')
    category_dist = {r[0]: r[1] for r in cat_rows}

    # --- Form Assignments ---
    total_forms = _safe(cur, 'SELECT COUNT(*) FROM form_assignments')
    completed_forms = _safe(cur, "SELECT COUNT(*) FROM form_assignments WHERE status='completed'")
    pending_forms = total_forms - completed_forms
    form_completion_rate = round(completed_forms / total_forms * 100, 1) if total_forms else 0

    form_instruments = _safe_rows(cur, 'SELECT instrument, COUNT(*) FROM form_assignments GROUP BY instrument ORDER BY COUNT(*) DESC')
    form_by_instrument = {r[0]: r[1] for r in form_instruments}

    # --- Transaction Log (workflow events) ---
    total_events = _safe(cur, 'SELECT COUNT(*) FROM transaction_log')
    unique_components = _safe(cur, 'SELECT COUNT(DISTINCT component) FROM transaction_log')
    unique_actors = _safe(cur, 'SELECT COUNT(DISTINCT actor) FROM transaction_log')

    comp_rows = _safe_rows(cur, 'SELECT component, COUNT(*) FROM transaction_log GROUP BY component ORDER BY COUNT(*) DESC LIMIT 10')
    top_components = [{'component': r[0], 'count': r[1]} for r in comp_rows]

    action_rows = _safe_rows(cur, 'SELECT action, COUNT(*) FROM transaction_log GROUP BY action ORDER BY COUNT(*) DESC LIMIT 10')
    top_actions = [{'action': r[0], 'count': r[1]} for r in action_rows]

    actor_rows = _safe_rows(cur, 'SELECT actor, COUNT(*) FROM transaction_log GROUP BY actor ORDER BY COUNT(*) DESC LIMIT 10')
    top_actors = [{'actor': r[0], 'count': r[1]} for r in actor_rows]

    # --- Recent requests (last 20) ---
    recent_rows = _safe_rows(cur, '''
        SELECT id, request_text, category, status, source, ts_local, impl_module, tested
        FROM operator_requests ORDER BY id DESC LIMIT 20
    ''')
    recent_requests = []
    for r in recent_rows:
        recent_requests.append({
            'id': r[0],
            'request': (r[1] or '')[:120],
            'category': r[2],
            'status': r[3],
            'source': r[4],
            'timestamp': r[5],
            'module': r[6],
            'tested': r[7],
        })

    conn.close()

    return {
        'available': True,
        'kpis': {
            'total_requests': total_requests,
            'done': done_count,
            'open': open_count,
            'rejected': rejected_count,
            'completion_rate': completion_rate,
            'total_form_assignments': total_forms,
            'form_completion_rate': form_completion_rate,
            'total_workflow_events': total_events,
        },
        'status_distribution': status_dist,
        'category_distribution': category_dist,
        'form_by_instrument': form_by_instrument,
        'top_components': top_components,
        'top_actions': top_actions,
        'top_actors': top_actors,
        'recent_requests': recent_requests,
    }


def clinical_tasks_breakdown():
    """Per-category task breakdown with completion metrics and request detail."""
    conn = _conn()
    cur = conn.cursor()

    # --- Per-category breakdown ---
    cat_rows = _safe_rows(cur, 'SELECT category, status, COUNT(*) FROM operator_requests GROUP BY category, status')
    by_category = defaultdict(lambda: {'total': 0, 'statuses': {}})
    for cat, status, cnt in cat_rows:
        by_category[cat]['total'] += cnt
        by_category[cat]['statuses'][status] = cnt

    category_details = []
    for cat, info in sorted(by_category.items(), key=lambda x: -x[1]['total']):
        done = sum(v for k, v in info['statuses'].items() if _classify_status(k) == 'done')
        rate = round(done / info['total'] * 100, 1) if info['total'] else 0
        category_details.append({
            'category': cat,
            'total': info['total'],
            'done': done,
            'open': sum(v for k, v in info['statuses'].items() if _classify_status(k) == 'open'),
            'rejected': sum(v for k, v in info['statuses'].items() if _classify_status(k) == 'rejected'),
            'completion_rate': rate,
            'statuses': info['statuses'],
        })

    # --- Implemented vs not (requests with impl_module set) ---
    implemented = _safe(cur, "SELECT COUNT(*) FROM operator_requests WHERE impl_module IS NOT NULL AND impl_module != ''")
    tested_count = _safe(cur, "SELECT COUNT(*) FROM operator_requests WHERE tested IS NOT NULL AND tested != ''")

    # --- Form assignment details ---
    form_rows = _safe_rows(cur, '''
        SELECT id, patient_id, instrument, assigned_by, status, message, created_at, completed_at
        FROM form_assignments ORDER BY id DESC
    ''')
    form_details = []
    for r in form_rows:
        form_details.append({
            'id': r[0],
            'patient_id': r[1],
            'instrument': r[2],
            'assigned_by': r[3],
            'status': r[4],
            'message': r[5],
            'created_at': r[6],
            'completed_at': r[7],
        })

    # --- Daily transaction activity ---
    daily_rows = _safe_rows(cur, '''
        SELECT DATE(ts_local) as day, COUNT(*) as cnt
        FROM transaction_log
        WHERE ts_local IS NOT NULL
        GROUP BY DATE(ts_local)
        ORDER BY day DESC LIMIT 30
    ''')
    daily_activity = [{'date': r[0], 'events': r[1]} for r in daily_rows]

    # --- Component x Action cross-tab (top 8 components) ---
    cross_rows = _safe_rows(cur, '''
        SELECT component, action, COUNT(*) as cnt
        FROM transaction_log
        GROUP BY component, action
        ORDER BY cnt DESC LIMIT 30
    ''')
    component_actions = [{'component': r[0], 'action': r[1], 'count': r[2]} for r in cross_rows]

    conn.close()

    return {
        'available': True,
        'category_details': category_details,
        'implementation_stats': {
            'implemented': implemented,
            'tested': tested_count,
        },
        'form_assignments': form_details,
        'daily_activity': daily_activity,
        'component_actions': component_actions,
    }


def clinical_tasks_definitions():
    """Task metric definitions for tooltip overlays."""
    return {
        'available': True,
        'definitions': {
            'task_concepts': {
                'Operator Request': 'A clinical workflow request logged by the operator — tracked from submission through resolution.',
                'Form Assignment': 'A clinical assessment form assigned to a patient by a provider — e.g., GAD-7, PHQ-9.',
                'Workflow Event': 'An automated transaction logged by the system — e.g., assessment creation, EEG upload, drift monitoring.',
                'Completion Rate': 'Percentage of tasks in a done/addressed/completed status vs total tasks.',
            },
            'status_classifications': {
                'Done': 'Addressed, completed, resolved — task has been fulfilled.',
                'Open': 'Open, pending, logged — task awaits action.',
                'Rejected': 'Rejected, not-implemented, wont-fix — task was reviewed and declined.',
            },
            'quality_metrics': {
                'Implementation Coverage': 'Percentage of requests that have an impl_module (backend module) assigned.',
                'Testing Coverage': 'Percentage of requests marked as tested.',
                'Category Distribution': 'Breakdown of requests by category — session, general, history, meta.',
            },
            'clinical_relevance': {
                'ILAE': 'Task tracking supports ILAE clinical workflow documentation standards.',
                'IEC 62304': 'Workflow event logging provides software lifecycle traceability (IEC 62304 §8).',
                'FDA AI/ML': 'Audit trail of clinical task decisions supports FDA AI/ML transparency requirements.',
                'HIPAA': 'Task assignment and completion tracking supports HIPAA administrative safeguards.',
            },
            'remediation': {
                'Low completion rate': 'Review open tasks for blockers — reassign, break into subtasks, or escalate.',
                'Unimplemented requests': 'Prioritize open requests with no impl_module — may indicate gaps in system coverage.',
                'Low form assignment rate': 'Ensure clinical assessments are assigned to all patients per protocol.',
            },
        },
    }
