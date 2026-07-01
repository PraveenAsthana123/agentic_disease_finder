"""Model Governance AI Dashboard -- consultant oversight matrix, HITL sign-off
chain, approval workflows, model lifecycle tracking, and compliance audit
from real clinical.db data."""

import sqlite3
import json
import os
import math
from collections import defaultdict, Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe_mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(sum(vals) / len(vals), 2) if vals else 0


# -- Data loaders ----------------------------------------------------------

def _load_hitl_reviews(cur):
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
            'fields': fields,
            'decision': fields.get('decision', 'unknown'),
            'ai_prediction': fields.get('ai_prediction'),
            'human_decision': fields.get('human_decision'),
            'reason_code': fields.get('reason_code'),
            'reviewer_id': fields.get('reviewer_id'),
            'created_at': r[4],
        })
    return reviews


def _load_expert_reviews(cur):
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


def _load_clinical_decisions(cur):
    rows = cur.execute(
        'SELECT id, patient_id, analysis_id, ai_prediction, ai_confidence, '
        'top_channels, artifact_risk, time_window, neurologist_agreement, '
        'final_decision, reviewer, note, created_at FROM clinical_decisions'
    ).fetchall()
    return [
        {
            'id': r[0], 'patient_id': r[1], 'analysis_id': r[2],
            'ai_prediction': r[3], 'ai_confidence': r[4],
            'top_channels': r[5], 'artifact_risk': r[6],
            'time_window': r[7], 'neurologist_agreement': r[8],
            'final_decision': r[9], 'reviewer': r[10],
            'note': r[11], 'created_at': r[12],
        }
        for r in rows
    ]


def _load_component_findings(cur):
    rows = cur.execute(
        'SELECT id, patient_id, component, doctor_finding, doctor, '
        'agree_with_ai, created_at, updated_at FROM component_findings'
    ).fetchall()
    return [
        {
            'id': r[0], 'patient_id': r[1], 'component': r[2],
            'doctor_finding': r[3], 'doctor': r[4],
            'agree_with_ai': r[5], 'created_at': r[6], 'updated_at': r[7],
        }
        for r in rows
    ]


def _load_feedback(cur):
    rows = cur.execute(
        'SELECT id, patient_id, role, ai_output, rating, correction, '
        'reason, created_at FROM feedback'
    ).fetchall()
    return [
        {
            'id': r[0], 'patient_id': r[1], 'role': r[2],
            'ai_output': r[3], 'rating': r[4], 'correction': r[5],
            'reason': r[6], 'created_at': r[7],
        }
        for r in rows
    ]


def _load_analyses(cur):
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


# -- Overview ---------------------------------------------------------------

def governance_overview():
    """KPI-level summary: consultant matrix, sign-off rates, approval chain,
    model lifecycle, compliance status."""
    con = _conn()
    cur = con.cursor()

    hitl = _load_hitl_reviews(cur)
    experts = _load_expert_reviews(cur)
    decisions = _load_clinical_decisions(cur)
    findings = _load_component_findings(cur)
    fb = _load_feedback(cur)
    analyses = _load_analyses(cur)
    con.close()

    # -- Consultant matrix: role -> expert -> agreement stats --
    consultant_matrix = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'agree': 0, 'disagree': 0}))
    for e in experts:
        role = e['role'] or 'Unknown'
        expert = e['expert'] or 'Unknown'
        consultant_matrix[role][expert]['total'] += 1
        if e['agree_with_ai'] == 'agree':
            consultant_matrix[role][expert]['agree'] += 1
        else:
            consultant_matrix[role][expert]['disagree'] += 1

    matrix_rows = []
    for role, experts_map in consultant_matrix.items():
        for expert, stats in experts_map.items():
            rate = round(100 * stats['agree'] / stats['total'], 1) if stats['total'] else 0
            matrix_rows.append({
                'role': role, 'expert': expert,
                'total_reviews': stats['total'], 'agree': stats['agree'],
                'disagree': stats['disagree'], 'agreement_rate': rate,
            })

    # -- Sign-off chain: HITL decision distribution --
    decision_counts = Counter(h['decision'] for h in hitl)
    sign_off_chain = [{'decision': d, 'count': c} for d, c in decision_counts.most_common()]

    total_sign_offs = len(hitl)
    accept_count = sum(1 for h in hitl if h['decision'] == 'accept')
    override_count = sum(1 for h in hitl if h['decision'] == 'override')
    sign_off_rate = round(100 * accept_count / total_sign_offs, 1) if total_sign_offs else 0
    override_rate = round(100 * override_count / total_sign_offs, 1) if total_sign_offs else 0

    # -- Model lifecycle: analyses per disease --
    disease_counts = Counter(a['disease'] for a in analyses if a['disease'])
    model_lifecycle = [{'disease': d, 'analyses': c} for d, c in disease_counts.most_common()]

    avg_confidence = _safe_mean([a['confidence'] for a in analyses if a['confidence']])

    # -- Approval workflow timeline --
    timeline = []
    for h in sorted(hitl, key=lambda x: x['created_at'] or ''):
        timeline.append({
            'date': (h['created_at'] or '')[:10],
            'type': 'hitl_review',
            'patient_id': h['patient_id'],
            'decision': h['decision'],
        })
    for d in sorted(decisions, key=lambda x: x['created_at'] or ''):
        timeline.append({
            'date': (d['created_at'] or '')[:10],
            'type': 'clinical_decision',
            'patient_id': d['patient_id'],
            'final_decision': d['final_decision'],
            'neurologist_agreement': d['neurologist_agreement'],
        })
    for e in sorted(experts, key=lambda x: x['created_at'] or ''):
        timeline.append({
            'date': (e['created_at'] or '')[:10],
            'type': 'expert_review',
            'patient_id': e['patient_id'],
            'role': e['role'],
            'agree_with_ai': e['agree_with_ai'],
        })
    timeline.sort(key=lambda x: x['date'])

    # -- Compliance status --
    total_expert_reviews = len(experts)
    total_agree = sum(1 for e in experts if e['agree_with_ai'] == 'agree')
    expert_agreement_rate = round(100 * total_agree / total_expert_reviews, 1) if total_expert_reviews else 0

    neuro_agreement = sum(1 for d in decisions if d['neurologist_agreement'] == 'Yes')
    neuro_total = len(decisions)
    neuro_agreement_rate = round(100 * neuro_agreement / neuro_total, 1) if neuro_total else 0

    avg_rating = _safe_mean([f['rating'] for f in fb if f['rating']])

    # -- Unique consultants --
    unique_experts = set()
    for e in experts:
        if e['expert']:
            unique_experts.add(e['expert'])
    unique_roles = set(e['role'] for e in experts if e['role'])

    kpis = {
        'total_analyses': len(analyses),
        'total_hitl_reviews': total_sign_offs,
        'total_expert_reviews': total_expert_reviews,
        'total_clinical_decisions': neuro_total,
        'sign_off_rate': sign_off_rate,
        'override_rate': override_rate,
        'expert_agreement_rate': expert_agreement_rate,
        'neuro_agreement_rate': neuro_agreement_rate,
        'avg_confidence': avg_confidence,
        'avg_feedback_rating': avg_rating,
        'unique_consultants': len(unique_experts),
        'unique_roles': len(unique_roles),
        'total_component_findings': len(findings),
        'total_feedback': len(fb),
        'diseases_modelled': len(disease_counts),
    }

    return {
        'kpis': kpis,
        'consultant_matrix': matrix_rows,
        'sign_off_chain': sign_off_chain,
        'model_lifecycle': model_lifecycle,
        'governance_timeline': timeline,
    }


# -- Breakdown --------------------------------------------------------------

def governance_breakdown():
    """Detailed breakdown: per-expert reviews, HITL detail, clinical decision
    chain, component agreement, feedback log, per-patient governance profile."""
    con = _conn()
    cur = con.cursor()

    hitl = _load_hitl_reviews(cur)
    experts = _load_expert_reviews(cur)
    decisions = _load_clinical_decisions(cur)
    findings = _load_component_findings(cur)
    fb = _load_feedback(cur)
    analyses = _load_analyses(cur)
    con.close()

    # -- Per-expert review detail --
    expert_detail = []
    for e in experts:
        expert_detail.append({
            'id': e['id'], 'patient_id': e['patient_id'],
            'analysis_id': e['analysis_id'], 'role': e['role'],
            'expert': e['expert'], 'finding': e['finding'],
            'agree_with_ai': e['agree_with_ai'], 'note': e['note'],
            'created_at': e['created_at'],
        })

    # -- HITL review detail --
    hitl_detail = []
    for h in hitl:
        hitl_detail.append({
            'id': h['id'], 'patient_id': h['patient_id'],
            'analysis_id': h['analysis_id'], 'decision': h['decision'],
            'ai_prediction': h['ai_prediction'],
            'human_decision': h['human_decision'],
            'reason_code': h['reason_code'],
            'reviewer_id': h['reviewer_id'],
            'created_at': h['created_at'],
        })

    # -- Clinical decision chain --
    decision_chain = []
    for d in decisions:
        decision_chain.append({
            'id': d['id'], 'patient_id': d['patient_id'],
            'ai_prediction': d['ai_prediction'],
            'ai_confidence': d['ai_confidence'],
            'neurologist_agreement': d['neurologist_agreement'],
            'final_decision': d['final_decision'],
            'reviewer': d['reviewer'], 'note': d['note'],
            'artifact_risk': d['artifact_risk'],
            'created_at': d['created_at'],
        })

    # -- Component findings --
    comp_findings = []
    for f in findings:
        comp_findings.append({
            'id': f['id'], 'patient_id': f['patient_id'],
            'component': f['component'],
            'doctor_finding': f['doctor_finding'],
            'doctor': f['doctor'],
            'agree_with_ai': f['agree_with_ai'],
            'created_at': f['created_at'],
        })

    # -- Feedback log --
    feedback_log = []
    for f in fb:
        feedback_log.append({
            'id': f['id'], 'patient_id': f['patient_id'],
            'role': f['role'], 'rating': f['rating'],
            'correction': f['correction'], 'reason': f['reason'],
            'created_at': f['created_at'],
        })

    # -- Per-patient governance profile --
    patient_ids = set()
    for a in analyses:
        patient_ids.add(a['patient_id'])
    for h in hitl:
        patient_ids.add(h['patient_id'])
    for e in experts:
        patient_ids.add(e['patient_id'])

    patient_profiles = []
    for pid in sorted(patient_ids):
        p_analyses = [a for a in analyses if a['patient_id'] == pid]
        p_hitl = [h for h in hitl if h['patient_id'] == pid]
        p_experts = [e for e in experts if e['patient_id'] == pid]
        p_decisions = [d for d in decisions if d['patient_id'] == pid]
        p_findings = [f for f in findings if f['patient_id'] == pid]

        patient_profiles.append({
            'patient_id': pid,
            'analyses': len(p_analyses),
            'hitl_reviews': len(p_hitl),
            'expert_reviews': len(p_experts),
            'clinical_decisions': len(p_decisions),
            'component_findings': len(p_findings),
            'diseases': list(set(a['disease'] for a in p_analyses if a['disease'])),
            'governance_complete': len(p_hitl) > 0 or len(p_experts) > 0 or len(p_decisions) > 0,
        })

    # -- Role-based agreement matrix --
    role_agreement = defaultdict(lambda: {'total': 0, 'agree': 0})
    for e in experts:
        role = e['role'] or 'Unknown'
        role_agreement[role]['total'] += 1
        if e['agree_with_ai'] == 'agree':
            role_agreement[role]['agree'] += 1
    role_matrix = []
    for role, stats in role_agreement.items():
        rate = round(100 * stats['agree'] / stats['total'], 1) if stats['total'] else 0
        role_matrix.append({'role': role, 'total': stats['total'],
                            'agree': stats['agree'], 'agreement_rate': rate})

    return {
        'expert_detail': expert_detail,
        'hitl_detail': hitl_detail,
        'decision_chain': decision_chain,
        'component_findings': comp_findings,
        'feedback_log': feedback_log,
        'patient_profiles': patient_profiles,
        'role_agreement_matrix': role_matrix,
    }


# -- Definitions ------------------------------------------------------------

def governance_definitions():
    """Model governance definitions — governance concepts, approval workflows,
    compliance frameworks, clinical relevance, remediation strategies."""
    return {
        'sections': [
            {
                'title': 'Model Governance Concepts',
                'items': [
                    {'term': 'Model Governance', 'definition': 'The framework of policies, procedures, and controls that ensure AI/ML models are developed, validated, deployed, and monitored in a responsible and compliant manner throughout their lifecycle.'},
                    {'term': 'Consultant Matrix', 'definition': 'A structured mapping of clinical experts (neurologists, EEG technicians, specialists) to their review responsibilities, agreement rates, and oversight authority for AI model outputs.'},
                    {'term': 'HITL Sign-Off', 'definition': 'Human-in-the-Loop sign-off requiring a qualified clinician to accept, override, or escalate AI predictions before they become actionable clinical decisions.'},
                    {'term': 'Approval Chain', 'definition': 'The sequential workflow of reviews and approvals required before an AI prediction is used in clinical practice — typically: AI prediction → expert review → HITL decision → clinical sign-off.'},
                    {'term': 'Model Lifecycle', 'definition': 'The complete journey of a model from development through validation, deployment, monitoring, and eventual retirement or retraining.'},
                ],
            },
            {
                'title': 'Approval Workflow Types',
                'items': [
                    {'term': 'Accept', 'definition': 'The reviewer agrees with the AI prediction and signs off without modification. The AI output proceeds as-is to clinical use.'},
                    {'term': 'Override', 'definition': 'The reviewer disagrees with the AI prediction and provides an alternative clinical decision. Requires a reason code for audit trail.'},
                    {'term': 'Escalate', 'definition': 'The reviewer is uncertain and escalates to a higher-authority clinician (e.g., senior neurologist) for final determination.'},
                    {'term': 'Agreement Rate', 'definition': 'Percentage of expert reviews where the clinician agrees with the AI prediction. A key governance metric — rates below 70% trigger model review.'},
                    {'term': 'Override Rate', 'definition': 'Percentage of HITL reviews resulting in an override decision. High override rates may indicate model drift or systematic prediction errors.'},
                ],
            },
            {
                'title': 'Governance Metrics & KPIs',
                'items': [
                    {'term': 'Sign-Off Rate', 'definition': 'Percentage of HITL reviews where the AI prediction was accepted without override. Target: >80% for production models.'},
                    {'term': 'Expert Agreement Rate', 'definition': 'Percentage of expert reviews with agree_with_ai = "agree". Measures alignment between AI and clinical expertise.'},
                    {'term': 'Neurologist Agreement Rate', 'definition': 'Percentage of clinical decisions where the neurologist agreed with the AI prediction. Critical for regulatory compliance.'},
                    {'term': 'Feedback Rating', 'definition': 'Average clinician satisfaction rating (1-5 scale) for AI outputs. Target: ≥4.0 for production readiness.'},
                    {'term': 'Governance Coverage', 'definition': 'Percentage of AI predictions that have received at least one form of human oversight (HITL review, expert review, or clinical decision).'},
                    {'term': 'Audit Completeness', 'definition': 'Whether all required governance steps (expert review → HITL sign-off → clinical decision) have been completed for each patient analysis.'},
                ],
            },
            {
                'title': 'Clinical Relevance & Regulatory Standards',
                'items': [
                    {'term': 'IEC 62304', 'definition': 'International standard for medical device software lifecycle. Requires documented governance of AI/ML model changes, validation, and deployment approvals.'},
                    {'term': 'FDA AI/ML PCCP', 'definition': 'FDA Predetermined Change Control Plan for AI/ML-based Software as a Medical Device (SaMD). Requires pre-specified governance for model updates.'},
                    {'term': 'ILAE Guidelines', 'definition': 'International League Against Epilepsy standards for EEG interpretation. AI models must demonstrate agreement with board-certified neurologists.'},
                    {'term': 'ISO 14971', 'definition': 'Risk management standard for medical devices. Model governance must include risk assessment for AI prediction errors and their clinical impact.'},
                    {'term': 'EU AI Act', 'definition': 'European regulation classifying AI systems by risk level. Clinical AI is high-risk, requiring documented governance, human oversight, and transparency.'},
                ],
            },
            {
                'title': 'Remediation Strategies',
                'items': [
                    {'term': 'Low Agreement Trigger', 'definition': 'If expert agreement rate drops below 70%, initiate model review: check training data recency, feature drift, and disease-specific accuracy.'},
                    {'term': 'High Override Trigger', 'definition': 'If override rate exceeds 30%, pause automated predictions for affected disease categories and schedule retraining with recent clinical data.'},
                    {'term': 'Governance Gap Alert', 'definition': 'If any patient analysis lacks human oversight for >48 hours, escalate to department head and flag in compliance dashboard.'},
                    {'term': 'Feedback Degradation', 'definition': 'If average feedback rating drops below 3.5, conduct root-cause analysis on recent AI outputs and convene governance review board.'},
                    {'term': 'Audit Trail Integrity', 'definition': 'All governance actions (reviews, sign-offs, overrides) must be immutably logged with timestamps, reviewer IDs, and decision rationale.'},
                ],
            },
        ],
    }


if __name__ == '__main__':
    import pprint
    print('=== OVERVIEW ===')
    pprint.pprint(governance_overview())
    print('\n=== BREAKDOWN ===')
    pprint.pprint(governance_breakdown())
    print('\n=== DEFINITIONS ===')
    pprint.pprint(governance_definitions())
