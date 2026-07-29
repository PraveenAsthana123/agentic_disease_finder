"""Component Findings Dashboard — Doctor-AI agreement per EEG report component.

Reads component_findings table (patient_id, component, doctor_finding, doctor,
agree_with_ai) and serves overview KPIs, breakdowns, and clinical definitions
for the 3-endpoint pattern (/overview, /breakdown, /definitions).
"""

import sqlite3
import os
import math
from collections import Counter, defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe_pct(num, den):
    return round(100 * num / den, 1) if den else 0


def _safe_mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else 0


def _load_findings():
    con = _conn()
    con.row_factory = sqlite3.Row
    rows = con.execute(
        'SELECT id, patient_id, component, doctor_finding, doctor, '
        'agree_with_ai, created_at, updated_at FROM component_findings '
        'ORDER BY created_at'
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


# ── Overview ──────────────────────────────────────────────────────────────

def component_findings_overview():
    rows = _load_findings()
    total = len(rows)
    patients = set(r['patient_id'] for r in rows)
    doctors = set(r['doctor'] for r in rows)
    components = set(r['component'] for r in rows)

    agree_counts = Counter(r['agree_with_ai'] for r in rows)
    agree_rate = _safe_pct(agree_counts.get('agree', 0), total)
    disagree_rate = _safe_pct(agree_counts.get('disagree', 0), total)
    partial_rate = _safe_pct(agree_counts.get('partial', 0), total)

    # KPIs
    kpis = {
        'total_findings': total,
        'total_patients': len(patients),
        'total_reviewers': len(doctors),
        'total_components': len(components),
        'agreement_rate': agree_rate,
        'disagreement_rate': disagree_rate,
    }

    # Agreement distribution (pie)
    agreement_distribution = [
        {'name': 'Agree', 'value': agree_counts.get('agree', 0)},
        {'name': 'Disagree', 'value': agree_counts.get('disagree', 0)},
        {'name': 'Partial', 'value': agree_counts.get('partial', 0)},
    ]

    # Per-component agreement (stacked bar)
    comp_agree = defaultdict(lambda: Counter())
    for r in rows:
        comp_agree[r['component']][r['agree_with_ai']] += 1
    component_agreement = []
    for comp in sorted(comp_agree):
        c = comp_agree[comp]
        t = sum(c.values())
        component_agreement.append({
            'component': comp,
            'agree': c.get('agree', 0),
            'disagree': c.get('disagree', 0),
            'partial': c.get('partial', 0),
            'total': t,
            'agree_pct': _safe_pct(c.get('agree', 0), t),
        })

    # Per-reviewer agreement (bar)
    doc_agree = defaultdict(lambda: Counter())
    for r in rows:
        doc_agree[r['doctor']][r['agree_with_ai']] += 1
    reviewer_agreement = []
    for doc in sorted(doc_agree):
        c = doc_agree[doc]
        t = sum(c.values())
        reviewer_agreement.append({
            'reviewer': doc,
            'agree': c.get('agree', 0),
            'disagree': c.get('disagree', 0),
            'partial': c.get('partial', 0),
            'total': t,
            'agree_pct': _safe_pct(c.get('agree', 0), t),
        })

    # Monthly trend
    monthly = defaultdict(lambda: Counter())
    for r in rows:
        month = r['created_at'][:7] if r['created_at'] else 'unknown'
        monthly[month][r['agree_with_ai']] += 1
    monthly_trend = []
    for m in sorted(monthly):
        c = monthly[m]
        monthly_trend.append({
            'month': m,
            'agree': c.get('agree', 0),
            'disagree': c.get('disagree', 0),
            'partial': c.get('partial', 0),
        })

    return {
        'kpis': kpis,
        'agreement_distribution': agreement_distribution,
        'component_agreement': component_agreement,
        'reviewer_agreement': reviewer_agreement,
        'monthly_trend': monthly_trend,
    }


# ── Breakdown ─────────────────────────────────────────────────────────────

def component_findings_breakdown():
    rows = _load_findings()

    # All findings table
    all_findings = []
    for r in rows:
        all_findings.append({
            'id': r['id'],
            'patient_id': r['patient_id'],
            'component': r['component'],
            'doctor_finding': r['doctor_finding'],
            'doctor': r['doctor'],
            'agree_with_ai': r['agree_with_ai'],
            'created_at': r['created_at'],
        })

    # Per-patient summary
    patient_map = defaultdict(list)
    for r in rows:
        patient_map[r['patient_id']].append(r)
    patient_summary = []
    for pid in sorted(patient_map):
        prows = patient_map[pid]
        t = len(prows)
        ag = sum(1 for r in prows if r['agree_with_ai'] == 'agree')
        comps = sorted(set(r['component'] for r in prows))
        flags = []
        if ag / t < 0.5:
            flags.append('low_agreement')
        if any(r['agree_with_ai'] == 'disagree' and r['component'] == 'epileptiform' for r in prows):
            flags.append('epileptiform_disagreement')
        patient_summary.append({
            'patient_id': pid,
            'total_reviews': t,
            'agree': ag,
            'disagree': sum(1 for r in prows if r['agree_with_ai'] == 'disagree'),
            'partial': sum(1 for r in prows if r['agree_with_ai'] == 'partial'),
            'agree_pct': _safe_pct(ag, t),
            'components_reviewed': comps,
            'reviewer': prows[0]['doctor'],
            'flags': flags,
        })

    # Per-component detail
    comp_map = defaultdict(list)
    for r in rows:
        comp_map[r['component']].append(r)
    component_detail = []
    for comp in sorted(comp_map):
        crows = comp_map[comp]
        t = len(crows)
        ag = sum(1 for r in crows if r['agree_with_ai'] == 'agree')
        findings_dist = Counter(r['doctor_finding'] for r in crows)
        top_findings = [{'finding': f, 'count': c} for f, c in findings_dist.most_common(5)]
        component_detail.append({
            'component': comp,
            'total': t,
            'agree': ag,
            'disagree': sum(1 for r in crows if r['agree_with_ai'] == 'disagree'),
            'partial': sum(1 for r in crows if r['agree_with_ai'] == 'partial'),
            'agree_pct': _safe_pct(ag, t),
            'top_findings': top_findings,
            'reviewers': sorted(set(r['doctor'] for r in crows)),
        })

    # Disagreement detail (only disagree + partial)
    disagreements = [r for r in rows if r['agree_with_ai'] in ('disagree', 'partial')]
    disagreement_detail = []
    for r in disagreements:
        disagreement_detail.append({
            'patient_id': r['patient_id'],
            'component': r['component'],
            'doctor_finding': r['doctor_finding'],
            'doctor': r['doctor'],
            'agree_with_ai': r['agree_with_ai'],
            'created_at': r['created_at'],
        })

    # Reviewer × Component heatmap
    reviewers = sorted(set(r['doctor'] for r in rows))
    comps = sorted(set(r['component'] for r in rows))
    heatmap = []
    for doc in reviewers:
        for comp in comps:
            subset = [r for r in rows if r['doctor'] == doc and r['component'] == comp]
            if subset:
                ag = sum(1 for r in subset if r['agree_with_ai'] == 'agree')
                heatmap.append({
                    'reviewer': doc,
                    'component': comp,
                    'total': len(subset),
                    'agree_pct': _safe_pct(ag, len(subset)),
                })

    return {
        'all_findings': all_findings,
        'patient_summary': patient_summary,
        'component_detail': component_detail,
        'disagreement_detail': disagreement_detail,
        'reviewer_component_heatmap': heatmap,
    }


# ── Definitions ───────────────────────────────────────────────────────────

def component_findings_definitions():
    return {
        'title': 'Component Findings — Doctor-AI Agreement',
        'description': (
            'Tracks per-component agreement between the AI EEG analysis and the '
            'reviewing clinician for each section of the structured EEG report. '
            'Components follow the standard EEG report layout: acquisition quality, '
            'artifact annotation, background activity, epileptiform/seizure activity, '
            'AI explainability (SHAP), and video correlation.'
        ),
        'components': {
            'acquisition': 'Signal quality, impedance, channel integrity assessment',
            'artifacts': 'Artifact detection and ICA cleaning annotation',
            'background': 'Background activity — dominant rhythm, symmetry, slowing',
            'epileptiform': 'Epileptiform discharges, seizure activity, spike-wave',
            'explainability': 'SHAP/XAI feature importance alignment with clinical read',
            'video': 'Video-EEG semiology-electrographic correlation (video-EEG only)',
        },
        'agreement_levels': {
            'agree': 'Doctor fully agrees with AI finding for this component',
            'partial': 'Doctor partially agrees — AI finding correct but incomplete or over-sensitive',
            'disagree': 'Doctor disagrees — AI finding incorrect or clinically misleading',
        },
        'metrics': {
            'agreement_rate': 'Percentage of component reviews where doctor agrees with AI',
            'disagreement_rate': 'Percentage where doctor disagrees (potential AI error)',
            'partial_rate': 'Percentage where doctor partially agrees (needs refinement)',
            'per_component_rate': 'Agreement rate broken down by EEG component type',
            'per_reviewer_rate': 'Agreement rate broken down by reviewing clinician',
        },
        'clinical_relevance': [
            'Epileptiform disagreements are the highest-priority flag — may indicate missed seizures or false positives',
            'Background disagreements may indicate AI over/under-reading diffuse slowing',
            'Acquisition disagreements help calibrate signal quality thresholds',
            'Explainability disagreements reveal when SHAP features misalign with clinical reasoning',
            'Low overall agreement for a reviewer may indicate need for calibration or training',
            'Tracking trends over time shows whether AI model updates improve clinical concordance',
        ],
        'data_source': 'component_findings table in clinical.db — populated by doctor review during EEG report sign-off',
        'related_dashboards': [
            'Human Evaluation Dashboard — broader HITL review context',
            'Model Governance Dashboard — approval chain and sign-off rates',
            'Structured Reporting Dashboard — EEG report templates and field completeness',
        ],
        'glossary': {
            'HITL': 'Human-in-the-Loop — clinician review of AI output before clinical action',
            'ICA': 'Independent Component Analysis — artifact removal technique for EEG',
            'SHAP': 'SHapley Additive exPlanations — model interpretability method',
            'PDR': 'Posterior Dominant Rhythm — alpha rhythm measured at O1/O2',
            'IED': 'Interictal Epileptiform Discharge — spike/sharp wave between seizures',
            'Semiology': 'Clinical manifestation (signs/symptoms) of a seizure',
        },
    }
