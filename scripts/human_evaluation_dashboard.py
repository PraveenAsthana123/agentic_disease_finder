"""Human Evaluation AI Dashboard -- HITL reviews, expert agreement,
clinical decisions, component findings, and clinician feedback
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
    """Load all HITL reviews with parsed fields_json."""
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
            'id': r[0],
            'patient_id': r[1],
            'analysis_id': r[2],
            'fields': fields,
            'decision': fields.get('decision', 'unknown'),
            'ai_prediction': fields.get('ai_prediction'),
            'human_decision': fields.get('human_decision'),
            'reason_code': fields.get('reason_code'),
            'created_at': r[4],
        })
    return reviews


def _load_expert_reviews(cur):
    """Load all expert reviews."""
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
    """Load all clinical decisions."""
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
            'final_decision': r[9], 'reviewer': r[10], 'note': r[11],
            'created_at': r[12],
        }
        for r in rows
    ]


def _load_component_findings(cur):
    """Load all component findings."""
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
    """Load all clinician feedback."""
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


def _load_analyses_for_eval(cur):
    """Load analyses with id, patient_id, confidence for cross-referencing."""
    rows = cur.execute(
        'SELECT id, patient_id, confidence FROM analyses'
    ).fetchall()
    return {r[0]: {'id': r[0], 'patient_id': r[1], 'confidence': r[2]}
            for r in rows}


# ==========================================================================
# API functions
# ==========================================================================

def human_eval_overview():
    """High-level KPIs, agreement breakdown, decision types, role distribution,
    feedback ratings, review timeline, and confidence vs agreement."""

    conn = _conn()
    cur = conn.cursor()

    hitl = _load_hitl_reviews(cur)
    experts = _load_expert_reviews(cur)
    decisions = _load_clinical_decisions(cur)
    findings = _load_component_findings(cur)
    fb = _load_feedback(cur)
    analyses = _load_analyses_for_eval(cur)
    conn.close()

    # -- KPIs --------------------------------------------------------------
    total_hitl = len(hitl)
    total_expert = len(experts)
    total_decisions = len(decisions)
    total_findings = len(findings)
    total_feedback = len(fb)

    # Agreement rate: % of expert_reviews where agree_with_ai == 'agree'
    agree_count = sum(1 for e in experts if (e['agree_with_ai'] or '').lower() == 'agree')
    disagree_count = total_expert - agree_count
    agreement_rate = round(agree_count / total_expert * 100, 1) if total_expert else 0

    # Override rate: % of hitl_reviews where decision == 'override'
    override_count = sum(1 for h in hitl if h['decision'] == 'override')
    accept_count = total_hitl - override_count
    override_rate = round(override_count / total_hitl * 100, 1) if total_hitl else 0

    # Avg feedback rating
    ratings = [f['rating'] for f in fb if f['rating'] is not None]
    avg_rating = _safe_mean(ratings)

    kpis = {
        'total_hitl_reviews': total_hitl,
        'total_expert_reviews': total_expert,
        'total_clinical_decisions': total_decisions,
        'total_component_findings': total_findings,
        'total_feedback': total_feedback,
        'agreement_rate': agreement_rate,
        'override_rate': override_rate,
        'avg_feedback_rating': avg_rating,
    }

    # -- Agreement breakdown -----------------------------------------------
    agreement_breakdown = [
        {'label': 'Agree', 'value': agree_count},
        {'label': 'Disagree', 'value': disagree_count},
    ]

    # -- Decision types ----------------------------------------------------
    decision_types = [
        {'label': 'Accept', 'value': accept_count},
        {'label': 'Override', 'value': override_count},
    ]

    # -- Role distribution -------------------------------------------------
    role_counts = Counter(e['role'] for e in experts if e['role'])
    role_distribution = [{'role': r, 'count': c}
                         for r, c in sorted(role_counts.items())]

    # -- Feedback ratings histogram ----------------------------------------
    rating_counts = Counter(f['rating'] for f in fb if f['rating'] is not None)
    feedback_ratings = [{'rating': r, 'count': rating_counts.get(r, 0)}
                        for r in sorted(rating_counts.keys())]

    # -- Review timeline ---------------------------------------------------
    date_counts = Counter()
    for e in experts:
        if e['created_at']:
            day = e['created_at'][:10]
            date_counts[day] += 1
    for h in hitl:
        if h['created_at']:
            day = h['created_at'][:10]
            date_counts[day] += 1
    review_timeline = [{'date': d, 'reviews': c}
                       for d, c in sorted(date_counts.items())]

    # -- Confidence vs agreement -------------------------------------------
    # For each analysis that has expert reviews, pair confidence with agreement
    expert_by_analysis = defaultdict(list)
    for e in experts:
        if e['analysis_id'] is not None:
            expert_by_analysis[e['analysis_id']].append(e)

    confidence_vs_agreement = []
    for aid, expert_list in expert_by_analysis.items():
        if aid in analyses:
            all_agree = all(
                (e['agree_with_ai'] or '').lower() == 'agree'
                for e in expert_list
            )
            confidence_vs_agreement.append({
                'analysis_id': aid,
                'confidence': analyses[aid]['confidence'],
                'agreed': all_agree,
            })

    return {
        'kpis': kpis,
        'agreement_breakdown': agreement_breakdown,
        'decision_types': decision_types,
        'role_distribution': role_distribution,
        'feedback_ratings': feedback_ratings,
        'review_timeline': review_timeline,
        'confidence_vs_agreement': confidence_vs_agreement,
    }


def human_eval_breakdown():
    """Detailed HITL reviews, expert reviews, clinical decisions,
    component findings, feedback, patient profiles, and role agreement."""

    conn = _conn()
    cur = conn.cursor()

    hitl = _load_hitl_reviews(cur)
    experts = _load_expert_reviews(cur)
    decisions = _load_clinical_decisions(cur)
    findings = _load_component_findings(cur)
    fb = _load_feedback(cur)
    conn.close()

    # -- Patient evaluation profiles ---------------------------------------
    all_pids = set()
    for h in hitl:
        if h['patient_id']:
            all_pids.add(h['patient_id'])
    for e in experts:
        if e['patient_id']:
            all_pids.add(e['patient_id'])
    for d in decisions:
        if d['patient_id']:
            all_pids.add(d['patient_id'])
    for f in findings:
        if f['patient_id']:
            all_pids.add(f['patient_id'])
    for f in fb:
        if f['patient_id']:
            all_pids.add(f['patient_id'])

    patient_profiles = []
    for pid in sorted(all_pids):
        p_hitl = [h for h in hitl if h['patient_id'] == pid]
        p_expert = [e for e in experts if e['patient_id'] == pid]
        p_dec = [d for d in decisions if d['patient_id'] == pid]
        p_find = [f for f in findings if f['patient_id'] == pid]
        p_fb = [f for f in fb if f['patient_id'] == pid]

        agree = sum(1 for e in p_expert
                    if (e['agree_with_ai'] or '').lower() == 'agree')
        total_e = len(p_expert)
        rate = round(agree / total_e * 100, 1) if total_e else 0

        patient_profiles.append({
            'patient_id': pid,
            'hitl_count': len(p_hitl),
            'expert_count': total_e,
            'decision_count': len(p_dec),
            'finding_count': len(p_find),
            'feedback_count': len(p_fb),
            'agreement_rate': rate,
        })

    # -- Role agreement matrix ---------------------------------------------
    role_groups = defaultdict(list)
    for e in experts:
        if e['role']:
            role_groups[e['role']].append(e)

    role_agreement_matrix = []
    for role, entries in sorted(role_groups.items()):
        agree_c = sum(1 for e in entries
                      if (e['agree_with_ai'] or '').lower() == 'agree')
        disagree_c = len(entries) - agree_c
        rate = round(agree_c / len(entries) * 100, 1) if entries else 0
        role_agreement_matrix.append({
            'role': role,
            'agree_count': agree_c,
            'disagree_count': disagree_c,
            'rate': rate,
        })

    return {
        'hitl_details': hitl,
        'expert_details': experts,
        'clinical_decision_details': decisions,
        'component_finding_details': findings,
        'feedback_details': fb,
        'patient_evaluation_profiles': patient_profiles,
        'role_agreement_matrix': role_agreement_matrix,
    }


def human_eval_definitions():
    """Static definitions, methodology, and compliance context for the
    Human Evaluation AI Dashboard."""

    return {
        'sections': [
            {
                'title': 'Human Evaluation Concept',
                'items': [
                    {
                        'term': 'Human-in-the-Loop (HITL) Evaluation',
                        'definition': (
                            'A quality assurance paradigm where human clinicians '
                            'review, validate, or override AI-generated predictions '
                            'before they influence clinical decisions. In epilepsy '
                            'EEG diagnostics, HITL ensures that AI classifications '
                            '(e.g., Epilepsy, Normal, Artifact) are verified by '
                            'domain experts before becoming part of the medical record.'
                        ),
                    },
                    {
                        'term': 'Clinical AI Oversight',
                        'definition': (
                            'The systematic process of monitoring AI system outputs '
                            'through expert review, feedback loops, and decision '
                            'auditing. Oversight ensures that automated EEG analysis '
                            'maintains clinical safety standards and that disagreements '
                            'between AI and human experts are documented, analysed, '
                            'and used for continuous model improvement.'
                        ),
                    },
                ],
            },
            {
                'title': 'Review Types',
                'items': [
                    {
                        'term': 'HITL Review',
                        'definition': (
                            'A structured accept-or-override decision by a clinician '
                            'on an AI prediction. Each review captures the AI '
                            'prediction, the human decision (accept or override), '
                            'the human-corrected label if overridden, and a reason '
                            'code (e.g., ART for artifact).'
                        ),
                    },
                    {
                        'term': 'Expert Review',
                        'definition': (
                            'A role-specific clinical assessment (Neurologist, EEG '
                            'Technician) that records detailed findings and an '
                            'explicit agree/disagree verdict against the AI output. '
                            'Multiple experts may review the same analysis.'
                        ),
                    },
                    {
                        'term': 'Clinical Decision',
                        'definition': (
                            'A formal diagnostic decision that combines AI prediction, '
                            'AI confidence, EEG channel analysis, artifact risk, and '
                            'neurologist agreement into a final clinical verdict. '
                            'Tracks the reviewing clinician and supporting notes.'
                        ),
                    },
                    {
                        'term': 'Component Finding',
                        'definition': (
                            'A doctor-level assessment of a specific system component '
                            '(e.g., signal acquisition, feature extraction) recording '
                            'clinical observations and agreement with AI analysis of '
                            'that component.'
                        ),
                    },
                    {
                        'term': 'Clinician Feedback',
                        'definition': (
                            'Structured feedback from clinicians on AI outputs, '
                            'including a numerical rating (1-5), optional correction '
                            'text, and reason for the rating. Used for continuous '
                            'quality monitoring and model retraining prioritisation.'
                        ),
                    },
                ],
            },
            {
                'title': 'Agreement Metrics',
                'items': [
                    {
                        'term': 'Agreement Rate',
                        'definition': (
                            'The percentage of expert reviews where the human expert '
                            'agreed with the AI prediction. Calculated as: '
                            '(agree count / total expert reviews) x 100. Higher rates '
                            'indicate better AI-human concordance.'
                        ),
                    },
                    {
                        'term': 'Override Rate',
                        'definition': (
                            'The percentage of HITL reviews where the clinician chose '
                            'to override the AI prediction. Calculated as: '
                            '(override count / total HITL reviews) x 100. High '
                            'override rates may indicate model calibration issues.'
                        ),
                    },
                    {
                        'term': 'Inter-Rater Reliability',
                        'definition': (
                            'The degree of agreement among multiple human reviewers '
                            'on the same analysis. Measured through role-level '
                            'agreement matrices comparing Neurologist vs EEG Technician '
                            'concordance. Essential for establishing ground truth '
                            'quality in training data curation.'
                        ),
                    },
                ],
            },
            {
                'title': 'Clinical Relevance',
                'items': [
                    {
                        'term': 'IEC 62304 Compliance',
                        'definition': (
                            'IEC 62304 requires documented verification and validation '
                            'of medical device software outputs. HITL evaluation '
                            'provides the verification layer: every AI prediction is '
                            'reviewed by a qualified clinician, and all decisions are '
                            'audit-logged with timestamps, roles, and rationale.'
                        ),
                    },
                    {
                        'term': 'FDA AI/ML PCCP',
                        'definition': (
                            'The FDA Predetermined Change Control Plan for AI/ML-based '
                            'SaMD requires real-world performance monitoring including '
                            'human expert agreement tracking. HITL metrics (agreement '
                            'rate, override rate, feedback scores) directly feed PCCP '
                            'performance monitoring reports.'
                        ),
                    },
                    {
                        'term': 'ILAE Guidelines',
                        'definition': (
                            'The International League Against Epilepsy emphasises that '
                            'EEG interpretation should involve trained epileptologists. '
                            'The HITL framework ensures AI-assisted EEG analysis is '
                            'always confirmed by qualified specialists before clinical '
                            'action, aligning with ILAE best practices.'
                        ),
                    },
                ],
            },
            {
                'title': 'Remediation Strategies',
                'items': [
                    {
                        'term': 'High Override Rate Trigger',
                        'definition': (
                            'When the override rate exceeds 20%, initiate a model '
                            'recalibration review: analyse overridden cases for '
                            'systematic error patterns, update training data with '
                            'corrected labels, and retrain affected model components.'
                        ),
                    },
                    {
                        'term': 'Low Agreement Rate Trigger',
                        'definition': (
                            'When the agreement rate drops below 70%, convene a '
                            'clinical review board to assess whether the disagreements '
                            'stem from model errors, ambiguous cases, or inter-rater '
                            'variability. Document findings and adjust decision '
                            'thresholds accordingly.'
                        ),
                    },
                    {
                        'term': 'Feedback Score Decline',
                        'definition': (
                            'When average clinician feedback rating drops below 3.0, '
                            'review recent model updates for regression, conduct '
                            'targeted evaluation on flagged outputs, and consider '
                            'rolling back to a previous model version.'
                        ),
                    },
                    {
                        'term': 'Role Discordance',
                        'definition': (
                            'When agreement rates differ significantly between roles '
                            '(e.g., Neurologists agree at 80% but Technicians at 40%), '
                            'investigate whether the AI output presentation needs '
                            'role-specific adaptation or whether additional training '
                            'materials are needed for certain reviewer roles.'
                        ),
                    },
                    {
                        'term': 'Continuous Improvement Loop',
                        'definition': (
                            'Establish a quarterly review cycle: aggregate all HITL '
                            'metrics, identify top-3 improvement areas, implement '
                            'targeted interventions, and measure impact in the '
                            'subsequent quarter. Feed corrected labels back into '
                            'model retraining pipelines.'
                        ),
                    },
                ],
            },
        ],
    }


# -- CLI quick-test --------------------------------------------------------

if __name__ == '__main__':
    import pprint
    print('=== Human Evaluation Overview ===')
    ov = human_eval_overview()
    pprint.pprint(ov)
    print()
    print('=== Human Evaluation Breakdown ===')
    bd = human_eval_breakdown()
    print(f"HITL details: {len(bd['hitl_details'])} reviews")
    print(f"Expert details: {len(bd['expert_details'])} reviews")
    print(f"Clinical decisions: {len(bd['clinical_decision_details'])}")
    print(f"Component findings: {len(bd['component_finding_details'])}")
    print(f"Feedback: {len(bd['feedback_details'])}")
    print(f"Patient profiles: {len(bd['patient_evaluation_profiles'])}")
    print(f"Role agreement matrix: {len(bd['role_agreement_matrix'])} roles")
    print()
    print('=== Human Evaluation Definitions ===')
    df = human_eval_definitions()
    pprint.pprint([s['title'] for s in df['sections']])
