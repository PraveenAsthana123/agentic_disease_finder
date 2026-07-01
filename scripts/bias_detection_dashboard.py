"""Bias Detection AI Dashboard — demographic fairness and bias analysis
for epilepsy patient data.  Identifies disparities across gender, age,
and intersectional groups in AI confidence, assessment coverage,
medication access, and seizure burden.

Aggregates real data from:
- data/clinical.db patients (40 patients)
- data/clinical.db analyses (21 EEG analyses)
- data/clinical.db assessments (423 assessment scores)
- data/clinical.db medications (9 medication records)
- data/clinical.db seizure_diary (25 seizure events)
- data/clinical.db mri_findings (40 MRI scans)
"""

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


def _safe_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(vals) < 2:
        return 0
    m = sum(vals) / len(vals)
    return round(math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)), 2)


# ── Age bracket helper ────────────────────────────────────────────────────

AGE_BRACKETS = [
    ('0-18', 0, 18),
    ('19-35', 19, 35),
    ('36-55', 36, 55),
    ('56+', 56, 999),
]


def _age_bracket(age):
    if age is None:
        return 'Unknown'
    for label, lo, hi in AGE_BRACKETS:
        if lo <= age <= hi:
            return label
    return 'Unknown'


def _gender_label(g):
    """Normalise gender to a display-safe label."""
    if not g or g.strip() == '':
        return 'Unspecified'
    return g.strip()


# ── Data loaders ──────────────────────────────────────────────────────────

def _load_patients(cur):
    rows = cur.execute(
        'SELECT patient_id, name, age, gender, disease FROM patients'
    ).fetchall()
    patients = {}
    for r in rows:
        patients[r[0]] = {
            'patient_id': r[0], 'name': r[1], 'age': r[2],
            'gender': _gender_label(r[3]), 'disease': r[4],
        }
    return patients


def _load_analyses(cur):
    rows = cur.execute(
        'SELECT patient_id, predicted_label, confidence, signal_quality '
        'FROM analyses'
    ).fetchall()
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r[0]].append({
            'predicted_label': r[1],
            'confidence': r[2],
            'signal_quality': r[3],
        })
    return by_patient


def _load_assessments(cur):
    rows = cur.execute(
        'SELECT patient_id, instrument, score, max_score, interpretation '
        'FROM assessments'
    ).fetchall()
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r[0]].append({
            'instrument': r[1], 'score': r[2],
            'max_score': r[3], 'interpretation': r[4],
        })
    return by_patient


def _load_medications(cur):
    rows = cur.execute(
        'SELECT patient_id, fields_json FROM medications'
    ).fetchall()
    by_patient = defaultdict(list)
    for r in rows:
        try:
            fj = json.loads(r[1]) if r[1] else {}
        except (json.JSONDecodeError, TypeError):
            fj = {}
        by_patient[r[0]].append(fj)
    return by_patient


def _load_seizures(cur):
    rows = cur.execute(
        'SELECT patient_id, event_date, duration_sec, severity, trigger '
        'FROM seizure_diary'
    ).fetchall()
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r[0]].append({
            'event_date': r[1], 'duration_sec': r[2],
            'severity': r[3], 'trigger': r[4],
        })
    return by_patient


def _load_mri(cur):
    rows = cur.execute(
        'SELECT patient_id, fields_json FROM mri_findings'
    ).fetchall()
    by_patient = defaultdict(list)
    for r in rows:
        try:
            fj = json.loads(r[1]) if r[1] else {}
        except (json.JSONDecodeError, TypeError):
            fj = {}
        by_patient[r[0]].append(fj)
    return by_patient


# ═══════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

def bias_detection_overview():
    conn = _conn()
    cur = conn.cursor()

    patients = _load_patients(cur)
    analyses = _load_analyses(cur)
    assessments = _load_assessments(cur)
    medications = _load_medications(cur)
    seizures = _load_seizures(cur)
    conn.close()

    total_patients = len(patients)
    total_analyses = sum(len(v) for v in analyses.values())
    total_assessments = sum(len(v) for v in assessments.values())

    # ── Gender distribution ──────────────────────────────────────────────
    gender_counts = Counter()
    for p in patients.values():
        gender_counts[p['gender']] += 1

    gender_distribution = []
    for g, cnt in gender_counts.most_common():
        gender_distribution.append({
            'gender': g,
            'count': cnt,
            'percentage': round(cnt / total_patients * 100, 1),
        })

    # Gender groups = distinct non-null/non-empty gender labels
    gender_groups = len(gender_counts)

    # ── Age distribution ─────────────────────────────────────────────────
    age_bracket_counts = Counter()
    for p in patients.values():
        age_bracket_counts[_age_bracket(p['age'])] += 1

    age_distribution = []
    for label, _, _ in AGE_BRACKETS:
        age_distribution.append({
            'age_group': label,
            'count': age_bracket_counts.get(label, 0),
        })
    if age_bracket_counts.get('Unknown', 0) > 0:
        age_distribution.append({
            'age_group': 'Unknown',
            'count': age_bracket_counts['Unknown'],
        })

    age_groups = len([a for a in age_distribution if a['count'] > 0])

    # ── Confidence by gender ─────────────────────────────────────────────
    gender_confidences = defaultdict(list)
    for pid, p in patients.items():
        for a in analyses.get(pid, []):
            if a.get('confidence') is not None:
                gender_confidences[p['gender']].append(a['confidence'])

    confidence_by_gender = []
    for g in sorted(gender_confidences.keys()):
        vals = gender_confidences[g]
        confidence_by_gender.append({
            'gender': g,
            'avg_confidence': _safe_mean(vals),
            'std_confidence': _safe_std(vals),
            'count': len(vals),
        })

    # Avg confidence gap
    avg_confs = [_safe_mean(v) for v in gender_confidences.values() if v]
    avg_confidence_gap = round(max(avg_confs) - min(avg_confs), 4) if len(avg_confs) >= 2 else 0

    # ── Confidence by age group ──────────────────────────────────────────
    age_confidences = defaultdict(list)
    for pid, p in patients.items():
        bracket = _age_bracket(p['age'])
        for a in analyses.get(pid, []):
            if a.get('confidence') is not None:
                age_confidences[bracket].append(a['confidence'])

    confidence_by_age_group = []
    for label, _, _ in AGE_BRACKETS:
        vals = age_confidences.get(label, [])
        confidence_by_age_group.append({
            'age_group': label,
            'avg_confidence': _safe_mean(vals),
            'std_confidence': _safe_std(vals),
            'count': len(vals),
        })

    # ── Assessment coverage by gender ────────────────────────────────────
    gender_assessment_counts = defaultdict(list)
    for pid, p in patients.items():
        gender_assessment_counts[p['gender']].append(len(assessments.get(pid, [])))

    assessment_coverage_by_gender = []
    for g in sorted(gender_assessment_counts.keys()):
        vals = gender_assessment_counts[g]
        assessment_coverage_by_gender.append({
            'gender': g,
            'avg_assessments_per_patient': _safe_mean(vals),
            'total_assessments': sum(vals),
            'patients': len(vals),
        })

    # ── Medication access by gender ──────────────────────────────────────
    gender_med_access = defaultdict(lambda: {'total': 0, 'with_meds': 0})
    for pid, p in patients.items():
        g = p['gender']
        gender_med_access[g]['total'] += 1
        if len(medications.get(pid, [])) > 0:
            gender_med_access[g]['with_meds'] += 1

    medication_access_by_gender = []
    for g in sorted(gender_med_access.keys()):
        info = gender_med_access[g]
        medication_access_by_gender.append({
            'gender': g,
            'patients_total': info['total'],
            'patients_with_medications': info['with_meds'],
            'fraction': round(info['with_meds'] / info['total'], 3) if info['total'] else 0,
        })

    # ── Seizure burden by gender ─────────────────────────────────────────
    gender_seizure_counts = defaultdict(list)
    for pid, p in patients.items():
        gender_seizure_counts[p['gender']].append(len(seizures.get(pid, [])))

    seizure_burden_by_gender = []
    for g in sorted(gender_seizure_counts.keys()):
        vals = gender_seizure_counts[g]
        seizure_burden_by_gender.append({
            'gender': g,
            'avg_seizures_per_patient': _safe_mean(vals),
            'total_seizures': sum(vals),
            'patients': len(vals),
        })

    # ── Representation gap ───────────────────────────────────────────────
    group_sizes = [cnt for cnt in gender_counts.values()]
    representation_gap = max(group_sizes) - min(group_sizes) if group_sizes else 0

    # ── Fairness index (1 - normalized max disparity) ────────────────────
    # Normalized disparity = confidence gap / max possible (1.0)
    fairness_index = round(1.0 - min(avg_confidence_gap, 1.0), 4)

    kpi_cards = {
        'total_patients': total_patients,
        'gender_groups': gender_groups,
        'age_groups': age_groups,
        'total_analyses': total_analyses,
        'total_assessments': total_assessments,
        'representation_gap': representation_gap,
        'avg_confidence_gap': avg_confidence_gap,
        'fairness_index': fairness_index,
    }

    return {
        'kpi_cards': kpi_cards,
        'gender_distribution': gender_distribution,
        'age_distribution': age_distribution,
        'confidence_by_gender': confidence_by_gender,
        'confidence_by_age_group': confidence_by_age_group,
        'assessment_coverage_by_gender': assessment_coverage_by_gender,
        'medication_access_by_gender': medication_access_by_gender,
        'seizure_burden_by_gender': seizure_burden_by_gender,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════

def bias_detection_breakdown():
    conn = _conn()
    cur = conn.cursor()

    patients = _load_patients(cur)
    analyses = _load_analyses(cur)
    assessments = _load_assessments(cur)
    medications = _load_medications(cur)
    seizures = _load_seizures(cur)
    mri = _load_mri(cur)
    conn.close()

    # ── Per-patient bias profile ─────────────────────────────────────────
    per_patient_bias_profile = []
    for pid, p in patients.items():
        p_analyses = analyses.get(pid, [])
        p_assess = assessments.get(pid, [])
        p_meds = medications.get(pid, [])
        p_seizures = seizures.get(pid, [])

        confidences = [a['confidence'] for a in p_analyses if a.get('confidence') is not None]
        instruments = sorted(set(a['instrument'] for a in p_assess if a.get('instrument')))

        per_patient_bias_profile.append({
            'patient_id': pid,
            'name': p.get('name', ''),
            'age': p.get('age'),
            'gender': p.get('gender', 'Unspecified'),
            'num_analyses': len(p_analyses),
            'avg_confidence': _safe_mean(confidences),
            'num_assessments': len(p_assess),
            'num_medications': len(p_meds),
            'num_seizures': len(p_seizures),
            'instruments_administered': instruments,
        })

    per_patient_bias_profile.sort(key=lambda x: x['patient_id'])

    # ── Instrument by gender ─────────────────────────────────────────────
    # For each instrument, per-gender avg score and count
    inst_gender_data = defaultdict(lambda: defaultdict(list))
    for pid, p in patients.items():
        g = p['gender']
        for a in assessments.get(pid, []):
            inst = a.get('instrument', '')
            score = a.get('score')
            if inst and score is not None:
                inst_gender_data[inst][g].append(score)

    instrument_by_gender = []
    for inst in sorted(inst_gender_data.keys()):
        entry = {'instrument': inst, 'by_gender': []}
        for g in sorted(inst_gender_data[inst].keys()):
            vals = inst_gender_data[inst][g]
            entry['by_gender'].append({
                'gender': g,
                'avg_score': _safe_mean(vals),
                'std_score': _safe_std(vals),
                'count': len(vals),
            })
        instrument_by_gender.append(entry)

    # ── Confidence histogram by gender ───────────────────────────────────
    bin_edges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']

    gender_conf_bins = defaultdict(lambda: {bl: 0 for bl in bin_labels})
    for pid, p in patients.items():
        g = p['gender']
        for a in analyses.get(pid, []):
            conf = a.get('confidence')
            if conf is None:
                continue
            for (lo, hi), bl in zip(bin_edges, bin_labels):
                if lo <= conf < hi or (hi == 1.0 and conf == 1.0):
                    gender_conf_bins[g][bl] += 1
                    break

    confidence_histogram = []
    for bl in bin_labels:
        entry = {'bin': bl}
        for g in sorted(gender_conf_bins.keys()):
            entry[g] = gender_conf_bins[g][bl]
        confidence_histogram.append(entry)

    # ── MRI coverage by gender ───────────────────────────────────────────
    gender_mri_coverage = defaultdict(lambda: {'total': 0, 'with_mri': 0})
    for pid, p in patients.items():
        g = p['gender']
        gender_mri_coverage[g]['total'] += 1
        if len(mri.get(pid, [])) > 0:
            gender_mri_coverage[g]['with_mri'] += 1

    mri_coverage_by_gender = []
    for g in sorted(gender_mri_coverage.keys()):
        info = gender_mri_coverage[g]
        mri_coverage_by_gender.append({
            'gender': g,
            'patients_total': info['total'],
            'patients_with_mri': info['with_mri'],
            'fraction': round(info['with_mri'] / info['total'], 3) if info['total'] else 0,
        })

    # ── Disparity metrics ────────────────────────────────────────────────
    # Statistical parity difference: max(P(positive|G=g)) - min(P(positive|G=g))
    # Here "positive" = has at least one analysis with confidence > 0.5
    gender_positive_rates = {}
    for pid, p in patients.items():
        g = p['gender']
        if g not in gender_positive_rates:
            gender_positive_rates[g] = {'total': 0, 'positive': 0}
        gender_positive_rates[g]['total'] += 1
        confs = [a['confidence'] for a in analyses.get(pid, []) if a.get('confidence') is not None]
        if any(c > 0.5 for c in confs):
            gender_positive_rates[g]['positive'] += 1

    rates = []
    for g, info in gender_positive_rates.items():
        r = info['positive'] / info['total'] if info['total'] else 0
        rates.append(r)
    statistical_parity_diff = round(max(rates) - min(rates), 4) if len(rates) >= 2 else 0

    # Equal opportunity difference: difference in avg confidence among patients
    # who have analyses, across gender groups
    gender_mean_confs = {}
    for pid, p in patients.items():
        g = p['gender']
        confs = [a['confidence'] for a in analyses.get(pid, []) if a.get('confidence') is not None]
        if confs:
            if g not in gender_mean_confs:
                gender_mean_confs[g] = []
            gender_mean_confs[g].append(_safe_mean(confs))

    eq_rates = [_safe_mean(v) for v in gender_mean_confs.values() if v]
    equal_opportunity_diff = round(max(eq_rates) - min(eq_rates), 4) if len(eq_rates) >= 2 else 0

    # Disparate impact ratio: min(rate) / max(rate)
    disparate_impact_ratio = round(min(rates) / max(rates), 4) if rates and max(rates) > 0 else 0

    disparity_metrics = {
        'statistical_parity_difference': statistical_parity_diff,
        'equal_opportunity_difference': equal_opportunity_diff,
        'disparate_impact_ratio': disparate_impact_ratio,
        'gender_positive_rates': [
            {'gender': g, 'rate': round(info['positive'] / info['total'], 4) if info['total'] else 0,
             'positive': info['positive'], 'total': info['total']}
            for g, info in sorted(gender_positive_rates.items())
        ],
    }

    # ── Intersectional analysis ──────────────────────────────────────────
    # Cross age-group x gender matrix: avg confidence and count
    intersect_data = defaultdict(lambda: {'confidences': [], 'count': 0})
    for pid, p in patients.items():
        g = p['gender']
        bracket = _age_bracket(p['age'])
        confs = [a['confidence'] for a in analyses.get(pid, []) if a.get('confidence') is not None]
        key = (bracket, g)
        intersect_data[key]['count'] += 1
        intersect_data[key]['confidences'].extend(confs)

    intersectional_analysis = []
    for (bracket, g), info in sorted(intersect_data.items()):
        intersectional_analysis.append({
            'age_group': bracket,
            'gender': g,
            'patient_count': info['count'],
            'analyses_count': len(info['confidences']),
            'avg_confidence': _safe_mean(info['confidences']),
        })

    return {
        'per_patient_bias_profile': per_patient_bias_profile,
        'instrument_by_gender': instrument_by_gender,
        'confidence_histogram': confidence_histogram,
        'mri_coverage_by_gender': mri_coverage_by_gender,
        'disparity_metrics': disparity_metrics,
        'intersectional_analysis': intersectional_analysis,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

def bias_detection_definitions():
    return {
        'sections': [
            {
                'title': 'Bias Detection Methods',
                'items': [
                    {'term': 'Demographic Parity',
                     'definition': 'A fairness criterion requiring that the probability of a positive outcome (e.g. high-confidence AI prediction) is equal across all demographic groups. Violations indicate that the model favours certain populations.'},
                    {'term': 'Equal Opportunity',
                     'definition': 'Requires that the true positive rate is equal across protected groups. In clinical AI, this means patients with the same condition should receive equally confident predictions regardless of gender or age.'},
                    {'term': 'Disparate Impact',
                     'definition': 'Measures the ratio of positive-outcome rates between the least and most favoured group. The four-fifths rule (ratio < 0.8) is a common threshold indicating potential discrimination.'},
                    {'term': 'Intersectional Analysis',
                     'definition': 'Examines bias at the intersection of multiple protected attributes (e.g. age group x gender). Intersectional disparities can be invisible when examining each attribute independently.'},
                ],
            },
            {
                'title': 'Protected Attributes',
                'items': [
                    {'term': 'Gender',
                     'definition': 'Patient-reported gender (Female, Male, Unspecified). Epilepsy prevalence and medication pharmacokinetics may vary by sex, but AI confidence should not systematically differ unless clinically justified.'},
                    {'term': 'Age Group',
                     'definition': 'Bracketed as 0-18 (paediatric), 19-35 (young adult), 36-55 (middle-aged), 56+ (older adult). Age-specific epilepsy syndromes exist, but assessment access and AI prediction quality should be equitable across brackets.'},
                ],
            },
            {
                'title': 'Fairness Metrics',
                'items': [
                    {'term': 'Statistical Parity Difference',
                     'definition': 'The difference between the highest and lowest positive-outcome rates across demographic groups. A value of 0 indicates perfect parity; values > 0.1 warrant investigation.'},
                    {'term': 'Disparate Impact Ratio',
                     'definition': 'min(rate_group) / max(rate_group). Values below 0.8 indicate potential disparate impact under the four-fifths rule. Values of 1.0 indicate perfect parity.'},
                    {'term': 'Equalized Odds',
                     'definition': 'Both true positive and false positive rates should be equal across groups. In the clinical setting, this means diagnostic accuracy should not vary with patient demographics.'},
                ],
            },
            {
                'title': 'Clinical Relevance',
                'items': [
                    {'term': 'IEC 62304',
                     'definition': 'Medical device software lifecycle standard. Bias detection is part of risk management (ISO 14971) and must be documented for Class B/C software that influences clinical decisions.'},
                    {'term': 'FDA AI/ML PCCP',
                     'definition': 'Pre-determined Change Control Plan for AI/ML-based SaMD. Bias monitoring is a continuous post-market requirement; fairness metric thresholds must be defined in the PCCP.'},
                    {'term': 'ILAE Guidelines',
                     'definition': 'International League Against Epilepsy guidelines emphasise equitable access to diagnosis and treatment. AI tools must not introduce or amplify disparities in epilepsy care.'},
                    {'term': 'Bias in Clinical AI',
                     'definition': 'Clinical AI systems trained on unbalanced cohorts may underperform for underrepresented groups. Continuous bias auditing across demographic strata is required for responsible deployment.'},
                ],
            },
            {
                'title': 'Remediation Strategies',
                'items': [
                    {'term': 'Re-sampling',
                     'definition': 'Oversampling underrepresented groups or undersampling overrepresented groups in the training data to achieve demographic balance. Includes SMOTE and adaptive synthetic sampling.'},
                    {'term': 'Re-weighting',
                     'definition': 'Assigning higher loss weights to underrepresented demographic groups during model training, so errors on minority groups contribute more to the optimisation objective.'},
                    {'term': 'Adversarial Debiasing',
                     'definition': 'Training an adversarial network that tries to predict the protected attribute from model outputs. The primary model is penalised for leaking demographic information into predictions.'},
                    {'term': 'Fairness Constraints',
                     'definition': 'Adding explicit fairness constraints (e.g. demographic parity, equalized odds) to the optimisation objective. Ensures that the trained model satisfies fairness criteria by construction.'},
                ],
            },
        ],
    }


if __name__ == '__main__':
    import json as _json
    print('=== OVERVIEW ===')
    ov = bias_detection_overview()
    print(_json.dumps(ov, indent=2, default=str)[:3000])
    print('\n=== BREAKDOWN (summary) ===')
    bd = bias_detection_breakdown()
    print(f"Patient profiles: {len(bd['per_patient_bias_profile'])}")
    print(f"Instruments: {len(bd['instrument_by_gender'])}")
    print(f"Confidence histogram bins: {len(bd['confidence_histogram'])}")
    print(f"MRI coverage groups: {len(bd['mri_coverage_by_gender'])}")
    print(f"Intersectional cells: {len(bd['intersectional_analysis'])}")
    print(f"Disparity metrics: {_json.dumps(bd['disparity_metrics'], indent=2)}")
    print('\n=== DEFINITIONS ===')
    defs = bias_detection_definitions()
    print(f"Sections: {len(defs['sections'])}")
    for s in defs['sections']:
        print(f"  - {s['title']}: {len(s['items'])} items")
