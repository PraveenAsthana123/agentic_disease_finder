"""AI Lifecycle Management Dashboard -- ideation -> dev -> validate -> deploy
-> monitor -> retire lifecycle tracking from real agent_tasks.json,
enterprise_pipelines.json, deployed .joblib models, and clinical.db data."""

import sqlite3
import json
import os
import glob
import math
from collections import defaultdict, Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe_mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(sum(vals) / len(vals), 2) if vals else 0


# -- Status-to-lifecycle-stage mapping --------------------------------------

_STATUS_TO_STAGE = {
    'planned': 'ideation',
    'scaffold': 'development',
    'partial': 'development',
    'built': 'deployed',
}


def _map_stage(status):
    return _STATUS_TO_STAGE.get(status, 'ideation')


# -- Data loaders -----------------------------------------------------------

def _load_agents():
    path = os.path.join(BASE, 'config', 'agent_tasks.json')
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get('agents', [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _load_pipelines():
    path = os.path.join(BASE, 'config', 'enterprise_pipelines.json')
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        pipelines = []
        for group in data.get('groups', []):
            group_name = group.get('group', 'Unknown')
            for p in group.get('pipelines', []):
                pipelines.append({
                    'name': p.get('name', ''),
                    'status': p.get('status', 'planned'),
                    'group': group_name,
                    'stages': p.get('stages', []),
                    'maps_to': p.get('maps_to', ''),
                })
        return pipelines
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _load_models():
    pattern = os.path.join(BASE, 'models', '*.joblib')
    files = glob.glob(pattern)
    models = []
    for f in sorted(files):
        stat = os.stat(f)
        name = os.path.basename(f).replace('.joblib', '')
        models.append({
            'name': name,
            'file_path': f,
            'file_size_kb': round(stat.st_size / 1024, 1),
            'last_modified': os.path.getmtime(f),
            'lifecycle_stage': 'deployed',
        })
    return models


def _load_transaction_log(cur):
    try:
        rows = cur.execute(
            'SELECT id, patient_id, component, action, actor, ref_id, '
            'detail, ts_utc, ts_local FROM transaction_log '
            'ORDER BY ts_utc'
        ).fetchall()
        return [
            {
                'id': r[0], 'patient_id': r[1], 'component': r[2],
                'action': r[3], 'actor': r[4], 'ref_id': r[5],
                'detail': r[6], 'ts_utc': r[7], 'ts_local': r[8],
            }
            for r in rows
        ]
    except Exception:
        return []


def _load_hitl_reviews(cur):
    try:
        rows = cur.execute(
            'SELECT id, patient_id, analysis_id, fields_json, created_at '
            'FROM hitl_reviews'
        ).fetchall()
        reviews = []
        for r in rows:
            try:
                fields = json.loads(r[3]) if r[3] else {}
            except (json.JSONDecodeError, TypeError):
                fields = {}
            reviews.append({
                'id': r[0], 'patient_id': r[1], 'analysis_id': r[2],
                'decision': fields.get('decision', 'unknown'),
                'created_at': r[4],
            })
        return reviews
    except Exception:
        return []


def _load_expert_reviews(cur):
    try:
        rows = cur.execute(
            'SELECT id, patient_id, analysis_id, role, expert, finding, '
            'agree_with_ai, note, created_at FROM expert_reviews'
        ).fetchall()
        return [
            {
                'id': r[0], 'patient_id': r[1], 'analysis_id': r[2],
                'role': r[3], 'expert': r[4], 'finding': r[5],
                'agree_with_ai': r[6], 'note': r[7], 'created_at': r[8],
            }
            for r in rows
        ]
    except Exception:
        return []


def _load_clinical_decisions(cur):
    try:
        rows = cur.execute(
            'SELECT id, patient_id, analysis_id, ai_prediction, ai_confidence, '
            'neurologist_agreement, final_decision, reviewer, created_at '
            'FROM clinical_decisions'
        ).fetchall()
        return [
            {
                'id': r[0], 'patient_id': r[1], 'analysis_id': r[2],
                'ai_prediction': r[3], 'ai_confidence': r[4],
                'neurologist_agreement': r[5], 'final_decision': r[6],
                'reviewer': r[7], 'created_at': r[8],
            }
            for r in rows
        ]
    except Exception:
        return []


def _load_analyses(cur):
    try:
        rows = cur.execute(
            'SELECT id, patient_id, disease, confidence, signal_quality, '
            'created_at FROM analyses'
        ).fetchall()
        return [
            {
                'id': r[0], 'patient_id': r[1], 'disease': r[2],
                'confidence': r[3], 'signal_quality': r[4], 'created_at': r[5],
            }
            for r in rows
        ]
    except Exception:
        return []


# -- Overview ---------------------------------------------------------------

def lifecycle_overview():
    """KPI-level summary: lifecycle stage distribution, asset inventory,
    monitoring coverage, validation rates, training runs, health assessment."""
    agents = _load_agents()
    pipelines = _load_pipelines()
    models = _load_models()

    con = _conn()
    cur = con.cursor()
    tx_log = _load_transaction_log(cur)
    hitl = _load_hitl_reviews(cur)
    experts = _load_expert_reviews(cur)
    decisions = _load_clinical_decisions(cur)
    analyses = _load_analyses(cur)
    con.close()

    # -- Status counts for agents --
    agent_status = Counter(a.get('status', 'planned') for a in agents)
    # -- Status counts for pipelines --
    pipeline_status = Counter(p['status'] for p in pipelines)

    agents_operational = agent_status.get('built', 0)
    pipelines_active = pipeline_status.get('built', 0)
    models_deployed = len(models)

    total_ai_assets = len(agents) + len(pipelines) + models_deployed

    # Lifecycle coverage: % with at least partial implementation (not planned)
    agents_implemented = sum(1 for a in agents if a.get('status') in ('built', 'scaffold'))
    pipelines_implemented = sum(1 for p in pipelines if p['status'] in ('built', 'partial'))
    implemented = agents_implemented + pipelines_implemented + models_deployed
    lifecycle_coverage = round(100 * implemented / total_ai_assets, 1) if total_ai_assets else 0

    # Validation events: HITL + expert + clinical decisions
    validation_events = len(hitl) + len(experts) + len(decisions)

    # Monitoring events from transaction_log
    monitoring_actions = {'monitor', 'check', 'analyze'}
    monitoring_components = {'drift', 'consistency', 'fairness'}
    monitoring_events = sum(
        1 for t in tx_log
        if t['action'] in monitoring_actions and t['component'] in monitoring_components
    )

    # Training runs from transaction_log
    training_runs = sum(
        1 for t in tx_log
        if t['action'] == 'scheduled_train'
    )

    kpis = {
        'total_ai_assets': total_ai_assets,
        'lifecycle_coverage': lifecycle_coverage,
        'models_deployed': models_deployed,
        'agents_operational': agents_operational,
        'pipelines_active': pipelines_active,
        'validation_events': validation_events,
        'monitoring_events': monitoring_events,
        'training_runs': training_runs,
    }

    # -- Lifecycle stage distribution --
    stage_counts = Counter()
    for a in agents:
        stage_counts[_map_stage(a.get('status', 'planned'))] += 1
    for p in pipelines:
        stage_counts[_map_stage(p['status'])] += 1
    for _m in models:
        stage_counts['deployed'] += 1

    # Validation stage: items that have been reviewed (HITL or expert)
    reviewed_patients = set(h['patient_id'] for h in hitl) | set(e['patient_id'] for e in experts)
    stage_counts['validation'] = len(reviewed_patients)

    # Monitoring: items being actively monitored (drift + consistency + fairness tx events)
    monitored_components = set(
        t['component'] for t in tx_log
        if t['action'] in monitoring_actions and t['component'] in monitoring_components
    )
    stage_counts['monitoring'] = len(monitored_components)

    stage_counts['retired'] = 0

    lifecycle_stage_distribution = [
        {'stage': stage, 'count': count}
        for stage, count in sorted(stage_counts.items())
    ]

    # -- Asset type distribution --
    asset_type_distribution = [
        {'type': 'agents', 'total': len(agents),
         'built': agent_status.get('built', 0),
         'scaffold': agent_status.get('scaffold', 0),
         'planned': agent_status.get('planned', 0)},
        {'type': 'pipelines', 'total': len(pipelines),
         'built': pipeline_status.get('built', 0),
         'partial': pipeline_status.get('partial', 0),
         'planned': pipeline_status.get('planned', 0)},
        {'type': 'models', 'total': models_deployed,
         'deployed': models_deployed},
    ]

    # -- Daily lifecycle events --
    daily_events = defaultdict(lambda: Counter())
    for t in tx_log:
        date = (t['ts_utc'] or '')[:10]
        if date:
            daily_events[date][t['action']] += 1
    daily_lifecycle_events = [
        {'date': date, 'events': sum(actions.values()), 'breakdown': dict(actions)}
        for date, actions in sorted(daily_events.items())
    ]

    # -- Lifecycle health assessment --
    # Monitoring coverage: how many of the deployed models/agents have monitoring
    deployed_count = agents_operational + pipelines_active + models_deployed
    monitoring_coverage = round(100 * monitoring_events / max(deployed_count, 1), 1)

    # Validation rate: ratio of validation events to analyses
    total_analyses = len(analyses)
    validation_rate = round(100 * validation_events / max(total_analyses, 1), 1)

    avg_confidence = _safe_mean([a['confidence'] for a in analyses if a['confidence']])

    lifecycle_health = [
        {'dimension': 'Monitoring', 'score': min(monitoring_coverage, 100)},
        {'dimension': 'Validation', 'score': min(validation_rate, 100)},
        {'dimension': 'Confidence', 'score': round(avg_confidence * 100, 1)},
        {'dimension': 'Deployment', 'score': round(100 * deployed_count / max(total_ai_assets, 1), 1)},
        {'dimension': 'Training', 'score': min(round(100 * training_runs / max(models_deployed, 1), 1), 100)},
    ]

    return {
        'kpis': kpis,
        'lifecycle_stage_distribution': lifecycle_stage_distribution,
        'asset_type_distribution': asset_type_distribution,
        'daily_lifecycle_events': daily_lifecycle_events,
        'lifecycle_health': lifecycle_health,
    }


# -- Breakdown --------------------------------------------------------------

def lifecycle_breakdown():
    """Detailed per-item data: agent lifecycle, pipeline lifecycle, model
    inventory, validation log, monitoring log, training history,
    lifecycle transitions."""
    agents = _load_agents()
    pipelines = _load_pipelines()
    models = _load_models()

    con = _conn()
    cur = con.cursor()
    tx_log = _load_transaction_log(cur)
    hitl = _load_hitl_reviews(cur)
    experts = _load_expert_reviews(cur)
    decisions = _load_clinical_decisions(cur)
    con.close()

    # -- Agent lifecycle --
    agent_lifecycle = []
    for a in agents:
        status = a.get('status', 'planned')
        agent_lifecycle.append({
            'id': a.get('id', ''),
            'name': a.get('task', a.get('name', a.get('id', ''))),
            'status': status,
            'lifecycle_stage': _map_stage(status),
            'module': a.get('module', ''),
        })

    # -- Pipeline lifecycle --
    pipeline_lifecycle = []
    for p in pipelines:
        pipeline_lifecycle.append({
            'name': p['name'],
            'status': p['status'],
            'group': p['group'],
            'stages': p['stages'],
            'lifecycle_stage': _map_stage(p['status']),
            'maps_to': p['maps_to'],
        })

    # -- Model inventory --
    import datetime
    model_inventory = []
    for m in models:
        model_inventory.append({
            'name': m['name'],
            'file_size_kb': m['file_size_kb'],
            'last_modified': datetime.datetime.fromtimestamp(
                m['last_modified']
            ).strftime('%Y-%m-%d %H:%M:%S'),
            'lifecycle_stage': 'deployed',
        })

    # -- Validation log: combined HITL + expert reviews --
    validation_log = []
    for h in hitl:
        validation_log.append({
            'type': 'hitl_review',
            'id': h['id'],
            'patient_id': h['patient_id'],
            'decision': h['decision'],
            'created_at': h['created_at'],
        })
    for e in experts:
        validation_log.append({
            'type': 'expert_review',
            'id': e['id'],
            'patient_id': e['patient_id'],
            'role': e['role'],
            'expert': e['expert'],
            'agree_with_ai': e['agree_with_ai'],
            'created_at': e['created_at'],
        })
    for d in decisions:
        validation_log.append({
            'type': 'clinical_decision',
            'id': d['id'],
            'patient_id': d['patient_id'],
            'neurologist_agreement': d['neurologist_agreement'],
            'final_decision': d['final_decision'],
            'created_at': d['created_at'],
        })
    validation_log.sort(key=lambda x: x.get('created_at') or '')

    # -- Monitoring log: drift + consistency + fairness from transaction_log --
    monitoring_actions = {'monitor', 'check', 'analyze'}
    monitoring_components = {'drift', 'consistency', 'fairness'}
    monitoring_log = [
        {
            'id': t['id'],
            'patient_id': t['patient_id'],
            'component': t['component'],
            'action': t['action'],
            'actor': t['actor'],
            'detail': t['detail'],
            'ts_utc': t['ts_utc'],
        }
        for t in tx_log
        if t['action'] in monitoring_actions and t['component'] in monitoring_components
    ]

    # -- Training history: scheduled_train events --
    training_history = [
        {
            'id': t['id'],
            'patient_id': t['patient_id'],
            'component': t['component'],
            'action': t['action'],
            'actor': t['actor'],
            'detail': t['detail'],
            'ts_utc': t['ts_utc'],
        }
        for t in tx_log
        if t['action'] == 'scheduled_train'
    ]

    # -- Lifecycle transitions: recent events showing movement between stages --
    transition_actions = {
        'scheduled_train': 'development -> validation',
        'monitor': 'deployed -> monitoring',
        'check': 'deployed -> monitoring',
        'analyze': 'validation / monitoring',
        'build': 'development -> deployed',
        'add': 'ideation -> development',
        'human_decision': 'validation -> deployed',
    }
    lifecycle_transitions = []
    for t in tx_log:
        if t['action'] in transition_actions:
            lifecycle_transitions.append({
                'id': t['id'],
                'component': t['component'],
                'action': t['action'],
                'transition': transition_actions[t['action']],
                'actor': t['actor'],
                'detail': t['detail'],
                'ts_utc': t['ts_utc'],
            })

    return {
        'agent_lifecycle': agent_lifecycle,
        'pipeline_lifecycle': pipeline_lifecycle,
        'model_inventory': model_inventory,
        'validation_log': validation_log,
        'monitoring_log': monitoring_log,
        'training_history': training_history,
        'lifecycle_transitions': lifecycle_transitions,
    }


# -- Definitions ------------------------------------------------------------

def lifecycle_definitions():
    """AI Lifecycle Management definitions -- lifecycle concepts, metrics,
    clinical relevance, remediation strategies."""
    return {
        'sections': [
            {
                'title': 'AI Lifecycle Concepts',
                'items': [
                    {'term': 'Ideation', 'definition': 'The initial stage where an AI use case is identified, scoped, and prioritized. Includes requirement gathering, feasibility analysis, and ethical review. Assets in this stage have status "planned" in the registry.'},
                    {'term': 'Development', 'definition': 'Active building of the AI component: data preparation, feature engineering, model training, and code implementation. Assets in this stage have status "scaffold" or "partial" -- code exists but is not production-ready.'},
                    {'term': 'Validation', 'definition': 'Rigorous testing and review of the AI component: expert reviews, HITL assessments, clinical decisions, accuracy benchmarks, bias testing, and safety evaluation before deployment approval.'},
                    {'term': 'Deployment', 'definition': 'The AI component is live in production, serving predictions or performing its designated function. Assets with status "built" and deployed .joblib model files are in this stage.'},
                    {'term': 'Monitoring', 'definition': 'Continuous surveillance of deployed AI assets: data drift detection, model performance tracking, consistency checks, fairness audits, and alert generation for degradation.'},
                    {'term': 'Retirement', 'definition': 'Planned decommissioning of an AI asset that is no longer needed, has been superseded, or has degraded beyond remediation. Includes knowledge transfer, archival, and audit closure.'},
                ],
            },
            {
                'title': 'Lifecycle Metrics',
                'items': [
                    {'term': 'Lifecycle Coverage', 'definition': 'Percentage of total AI assets (agents + pipelines + models) that have progressed beyond the ideation stage to at least partial implementation. Target: >60% for production readiness.'},
                    {'term': 'Health Score', 'definition': 'Composite assessment of lifecycle health based on monitoring coverage (% of deployed assets with active monitoring) and validation rate (% of analyses with human oversight). Verdicts: HEALTHY, NEEDS_ATTENTION, AT_RISK.'},
                    {'term': 'Transition Velocity', 'definition': 'Rate at which AI assets move between lifecycle stages, measured by transaction log events per day. Higher velocity indicates active development and iteration.'},
                    {'term': 'Stage Duration', 'definition': 'Average time an AI asset spends in each lifecycle stage before transitioning. Long durations in development or validation may indicate blockers or resource constraints.'},
                    {'term': 'Monitoring Coverage', 'definition': 'Percentage of deployed AI assets that have active drift, consistency, or fairness monitoring. Target: 100% for production clinical systems.'},
                    {'term': 'Validation Rate', 'definition': 'Ratio of validation events (HITL reviews + expert reviews + clinical decisions) to total inference runs. Higher rates indicate stronger governance oversight.'},
                ],
            },
            {
                'title': 'Clinical Relevance & Regulatory Standards',
                'items': [
                    {'term': 'IEC 62304', 'definition': 'International standard for medical device software lifecycle processes. Requires documented lifecycle management including development planning, verification, validation, and maintenance for all AI/ML components in clinical systems.'},
                    {'term': 'FDA AI/ML PCCP', 'definition': 'FDA Predetermined Change Control Plan for AI/ML-based Software as a Medical Device (SaMD). Mandates lifecycle tracking of model changes, retraining events, and performance monitoring with pre-specified update protocols.'},
                    {'term': 'EU AI Act', 'definition': 'European regulation requiring high-risk AI systems (including clinical AI) to maintain complete lifecycle documentation, implement risk management at every stage, and ensure human oversight throughout the AI lifecycle.'},
                    {'term': 'ILAE', 'definition': 'International League Against Epilepsy guidelines for EEG interpretation and epilepsy diagnosis. AI models in this domain must demonstrate lifecycle compliance including validation against expert consensus and ongoing performance monitoring.'},
                    {'term': 'ISO 14971', 'definition': 'Risk management standard for medical devices applied throughout the AI lifecycle. Requires risk identification at ideation, mitigation during development, verification during validation, and residual risk monitoring post-deployment.'},
                ],
            },
            {
                'title': 'Remediation Strategies',
                'items': [
                    {'term': 'Stale Model Trigger', 'definition': 'If a deployed model has not been retrained or revalidated within 90 days, trigger a lifecycle review: check drift metrics, re-run validation benchmarks, and schedule retraining if performance has degraded.'},
                    {'term': 'Low Validation Trigger', 'definition': 'If fewer than 20% of inference runs have human oversight (HITL, expert review, or clinical decision), escalate to governance board and increase review sampling rate until coverage target is met.'},
                    {'term': 'Monitoring Gap Alert', 'definition': 'If any deployed AI asset lacks active drift, consistency, or fairness monitoring for more than 7 days, generate an alert and assign monitoring setup as a priority task.'},
                    {'term': 'Retirement Criteria', 'definition': 'An AI asset should be considered for retirement when: accuracy drops below acceptable threshold for 3 consecutive evaluations, a superior replacement is validated, or the clinical use case is discontinued.'},
                ],
            },
        ],
    }


if __name__ == '__main__':
    import pprint
    print('=== LIFECYCLE OVERVIEW ===')
    pprint.pprint(lifecycle_overview())
    print('\n=== LIFECYCLE BREAKDOWN ===')
    bd = lifecycle_breakdown()
    # Print summary counts instead of full lists for readability
    for key, val in bd.items():
        if isinstance(val, list):
            print(f'  {key}: {len(val)} items')
            if val:
                pprint.pprint(val[0])
        else:
            pprint.pprint({key: val})
    print('\n=== LIFECYCLE DEFINITIONS ===')
    pprint.pprint(lifecycle_definitions())
