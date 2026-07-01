"""Digital Twin AI Dashboard -- patient digital twin completeness,
cross-domain correlation, and per-patient twin profiles from real
clinical.db data.

Aggregates real data from:
- data/clinical.db patients (40 patients)
- data/clinical.db analyses (21 EEG analyses)
- data/clinical.db assessments (423 assessment scores)
- data/clinical.db medications (9 medication records)
- data/clinical.db seizure_diary (25 seizure events)
- data/clinical.db mri_findings (40 MRI scans)
- data/clinical.db neuropsych (1 neuropsych record)
"""

import sqlite3
import json
import os
import math
from collections import defaultdict, Counter
from itertools import combinations

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe_mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(sum(vals) / len(vals), 2) if vals else 0


# -- Age bracket helper ---------------------------------------------------

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


# -- Data loaders ----------------------------------------------------------

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
    """Return dict patient_id -> list of analysis dicts."""
    rows = cur.execute(
        'SELECT patient_id, predicted_label, confidence, signal_quality '
        'FROM analyses'
    ).fetchall()
    out = defaultdict(list)
    for r in rows:
        out[r[0]].append({
            'predicted_label': r[1],
            'confidence': r[2],
            'signal_quality': r[3],
        })
    return out


def _load_seizures(cur):
    """Return dict patient_id -> list of seizure dicts."""
    rows = cur.execute(
        'SELECT patient_id, duration_sec, severity, trigger '
        'FROM seizure_diary'
    ).fetchall()
    out = defaultdict(list)
    for r in rows:
        out[r[0]].append({
            'duration_sec': r[1],
            'severity': r[2],
            'trigger': r[3],
        })
    return out


def _load_medications(cur):
    """Return dict patient_id -> list of medication dicts."""
    rows = cur.execute(
        'SELECT patient_id, fields_json FROM medications'
    ).fetchall()
    out = defaultdict(list)
    for r in rows:
        try:
            fj = json.loads(r[1]) if r[1] else {}
        except (json.JSONDecodeError, TypeError):
            fj = {}
        drug = {
            'name': fj.get('drug_name', 'Unknown'),
            'dose_mg': fj.get('dose_mg'),
            'frequency': fj.get('frequency'),
        }
        out[r[0]].append(drug)
    return out


def _load_mri(cur):
    """Return dict patient_id -> mri summary dict."""
    rows = cur.execute(
        'SELECT patient_id, fields_json FROM mri_findings'
    ).fetchall()
    out = {}
    for r in rows:
        try:
            fj = json.loads(r[1]) if r[1] else {}
        except (json.JSONDecodeError, TypeError):
            fj = {}
        out[r[0]] = {
            'lesion_type': fj.get('lesion_type'),
            'lesion_location': fj.get('lesion_location'),
            'laterality': fj.get('laterality'),
            'classification': fj.get('classification'),
            'hippocampal_sclerosis': fj.get('hippocampal_sclerosis'),
        }
    return out


def _load_assessments(cur):
    """Return dict patient_id -> list of assessment dicts."""
    rows = cur.execute(
        'SELECT patient_id, instrument, score, max_score, interpretation '
        'FROM assessments'
    ).fetchall()
    out = defaultdict(list)
    for r in rows:
        out[r[0]].append({
            'instrument': r[1],
            'score': r[2],
            'max_score': r[3],
            'interpretation': r[4],
        })
    return out


# -- Completeness calculation ----------------------------------------------

DOMAINS = [
    'demographics', 'eeg_analysis', 'seizure_diary',
    'medications', 'mri', 'assessments',
]


def _compute_domains(patient, analyses, seizures, medications, mri, assessments):
    """Return a dict of domain -> bool for a single patient."""
    pid = patient['patient_id']
    has_demographics = (patient.get('age') is not None and
                        patient.get('gender') not in (None, '', 'Unspecified'))
    return {
        'demographics': has_demographics,
        'eeg_analysis': pid in analyses and len(analyses[pid]) > 0,
        'seizure_diary': pid in seizures and len(seizures[pid]) > 0,
        'medications': pid in medications and len(medications[pid]) > 0,
        'mri': pid in mri,
        'assessments': pid in assessments and len(assessments[pid]) > 0,
    }


def _completeness_score(domains_dict):
    """Fraction of 6 domains that are populated."""
    filled = sum(1 for v in domains_dict.values() if v)
    return round(filled / 6, 4)


# -- Readiness label -------------------------------------------------------

def _twin_readiness_level(score):
    n = round(score * 6)
    if n >= 5:
        return 'Full Twin'
    if n >= 3:
        return 'Partial Twin'
    if n >= 1:
        return 'Minimal Twin'
    return 'No Data'


# ==========================================================================
# API functions
# ==========================================================================

def digital_twin_overview():
    """High-level KPIs, completeness distribution, domain coverage, and
    demographic distributions for the digital-twin concept."""

    conn = _conn()
    cur = conn.cursor()

    patients = _load_patients(cur)
    analyses = _load_analyses(cur)
    seizures = _load_seizures(cur)
    medications = _load_medications(cur)
    mri = _load_mri(cur)
    assessments = _load_assessments(cur)
    conn.close()

    # -- Per-patient completeness ------------------------------------------
    patient_scores = {}
    patient_domains = {}
    for pid, pat in patients.items():
        dom = _compute_domains(pat, analyses, seizures, medications, mri, assessments)
        patient_domains[pid] = dom
        patient_scores[pid] = _completeness_score(dom)

    # -- KPIs --------------------------------------------------------------
    total_patients = len(patients)
    patients_with_analyses = len([p for p in patients if p in analyses])
    patients_with_seizures = len([p for p in patients if p in seizures])
    patients_with_medications = len([p for p in patients if p in medications])
    patients_with_mri = len([p for p in patients if p in mri])
    patients_with_assessments = len([p for p in patients if p in assessments])

    total_analyses = sum(len(v) for v in analyses.values())
    total_seizures = sum(len(v) for v in seizures.values())
    total_meds = sum(len(v) for v in medications.values())
    total_mri = len(mri)
    total_assess = sum(len(v) for v in assessments.values())
    total_data_points = (total_patients + total_analyses + total_seizures +
                         total_meds + total_mri + total_assess)

    avg_completeness = _safe_mean(list(patient_scores.values()))

    # -- Completeness histogram (6 bins = 1..6 domains) --------------------
    bin_labels = ['1 domain', '2 domains', '3 domains',
                  '4 domains', '5 domains', '6 domains']
    bin_ranges = [
        (0.001, 0.17), (0.17, 0.34), (0.34, 0.51),
        (0.51, 0.68), (0.68, 0.84), (0.84, 1.01),
    ]
    completeness_distribution = []
    for idx, (lo, hi) in enumerate(bin_ranges):
        cnt = sum(1 for s in patient_scores.values() if lo <= s <= hi)
        completeness_distribution.append({
            'bin': bin_labels[idx], 'range': f'{lo:.2f}-{hi:.2f}', 'count': cnt,
        })
    # patients with 0 domains
    zero_count = sum(1 for s in patient_scores.values() if s == 0)
    if zero_count:
        completeness_distribution.insert(0, {
            'bin': '0 domains', 'range': '0.00', 'count': zero_count,
        })

    # -- Domain coverage ---------------------------------------------------
    domain_coverage = []
    for d in DOMAINS:
        covered = sum(1 for pid in patients if patient_domains[pid][d])
        domain_coverage.append({
            'domain': d,
            'patients_covered': covered,
            'coverage_pct': round(covered / total_patients * 100, 1) if total_patients else 0,
        })

    # -- Gender distribution -----------------------------------------------
    gender_counts = Counter(p['gender'] for p in patients.values())
    gender_distribution = [{'gender': g, 'count': c}
                           for g, c in sorted(gender_counts.items())]

    # -- Age distribution --------------------------------------------------
    age_counts = Counter(_age_bracket(p['age']) for p in patients.values())
    age_distribution = [{'bracket': b, 'count': age_counts.get(b, 0)}
                        for b, _, _ in AGE_BRACKETS]
    if age_counts.get('Unknown', 0):
        age_distribution.append({'bracket': 'Unknown', 'count': age_counts['Unknown']})

    # -- Twin readiness ----------------------------------------------------
    readiness_counts = Counter(_twin_readiness_level(s)
                               for s in patient_scores.values())
    readiness_order = ['Full Twin', 'Partial Twin', 'Minimal Twin', 'No Data']
    twin_readiness = [{'level': lvl, 'count': readiness_counts.get(lvl, 0)}
                      for lvl in readiness_order]

    return {
        'total_patients': total_patients,
        'patients_with_analyses': patients_with_analyses,
        'patients_with_seizures': patients_with_seizures,
        'patients_with_medications': patients_with_medications,
        'patients_with_mri': patients_with_mri,
        'patients_with_assessments': patients_with_assessments,
        'total_data_points': total_data_points,
        'avg_completeness_score': avg_completeness,
        'completeness_distribution': completeness_distribution,
        'domain_coverage': domain_coverage,
        'gender_distribution': gender_distribution,
        'age_distribution': age_distribution,
        'twin_readiness': twin_readiness,
    }


def digital_twin_breakdown():
    """Per-patient digital twin profiles, cross-domain correlation,
    and medication-EEG cross-analysis."""

    conn = _conn()
    cur = conn.cursor()

    patients = _load_patients(cur)
    analyses = _load_analyses(cur)
    seizures = _load_seizures(cur)
    medications = _load_medications(cur)
    mri = _load_mri(cur)
    assessments = _load_assessments(cur)
    conn.close()

    # -- Build per-patient twin profiles -----------------------------------
    patient_twins = []
    for pid, pat in sorted(patients.items()):
        dom = _compute_domains(pat, analyses, seizures, medications, mri, assessments)
        score = _completeness_score(dom)

        # EEG summary
        eeg_summary = None
        if pid in analyses and analyses[pid]:
            confs = [a['confidence'] for a in analyses[pid] if a['confidence'] is not None]
            labels = [a['predicted_label'] for a in analyses[pid] if a['predicted_label']]
            eeg_summary = {
                'count': len(analyses[pid]),
                'avg_confidence': _safe_mean(confs),
                'predicted_label': labels[0] if labels else None,
            }

        # Seizure summary
        seizure_summary = None
        if pid in seizures and seizures[pid]:
            durs = [s['duration_sec'] for s in seizures[pid]
                    if s['duration_sec'] is not None]
            sev_counts = Counter(s['severity'] for s in seizures[pid]
                                 if s['severity'])
            triggers = list(set(s['trigger'] for s in seizures[pid]
                                if s['trigger']))
            seizure_summary = {
                'count': len(seizures[pid]),
                'avg_duration_sec': _safe_mean(durs),
                'severities': dict(sev_counts),
                'triggers': triggers,
            }

        # Medication summary
        medication_summary = None
        if pid in medications and medications[pid]:
            medication_summary = {'drugs': medications[pid]}

        # MRI summary
        mri_summary = None
        if pid in mri:
            mri_summary = mri[pid]

        # Assessment summary
        assessment_summary = None
        if pid in assessments and assessments[pid]:
            instruments = list(set(a['instrument'] for a in assessments[pid]
                                   if a['instrument']))
            score_pcts = []
            for a in assessments[pid]:
                if a['score'] is not None and a['max_score'] and a['max_score'] > 0:
                    score_pcts.append(a['score'] / a['max_score'])
            assessment_summary = {
                'instruments': sorted(instruments),
                'count': len(assessments[pid]),
                'avg_score_pct': round(_safe_mean(score_pcts), 4) if score_pcts else 0,
            }

        patient_twins.append({
            'patient_id': pid,
            'name': pat['name'],
            'age': pat['age'],
            'gender': pat['gender'],
            'completeness_score': score,
            'domains': dom,
            'eeg_summary': eeg_summary,
            'seizure_summary': seizure_summary,
            'medication_summary': medication_summary,
            'mri_summary': mri_summary,
            'assessment_summary': assessment_summary,
        })

    # -- Domain correlation (pairwise) -------------------------------------
    # For each pair of domains count patients having both
    domain_sets = {d: set() for d in DOMAINS}
    for tw in patient_twins:
        for d in DOMAINS:
            if tw['domains'][d]:
                domain_sets[d].add(tw['patient_id'])

    total_patients = len(patients)
    domain_correlation = []
    for da, db in combinations(DOMAINS, 2):
        overlap = domain_sets[da] & domain_sets[db]
        domain_correlation.append({
            'domain_a': da,
            'domain_b': db,
            'overlap_count': len(overlap),
            'overlap_pct': round(len(overlap) / total_patients * 100, 1) if total_patients else 0,
        })

    # -- Top complete patients ---------------------------------------------
    sorted_twins = sorted(patient_twins, key=lambda t: t['completeness_score'], reverse=True)
    top_complete_patients = sorted_twins[:10]

    # -- Medication-EEG cross-analysis -------------------------------------
    medication_eeg_cross = []
    for tw in patient_twins:
        if tw['medication_summary'] and tw['eeg_summary']:
            for drug in tw['medication_summary']['drugs']:
                medication_eeg_cross.append({
                    'patient_id': tw['patient_id'],
                    'drug': drug['name'],
                    'confidence': tw['eeg_summary']['avg_confidence'],
                })

    return {
        'patient_twins': patient_twins,
        'domain_correlation': domain_correlation,
        'top_complete_patients': [
            {'patient_id': t['patient_id'], 'name': t['name'],
             'completeness_score': t['completeness_score'],
             'domains_filled': sum(1 for v in t['domains'].values() if v)}
            for t in top_complete_patients
        ],
        'medication_eeg_cross': medication_eeg_cross,
    }


def digital_twin_definitions():
    """Static definitions, methodology, and compliance context for the
    Digital Twin AI Dashboard."""

    return {
        'digital_twin_concept': (
            'A patient digital twin is a comprehensive, continuously updated '
            'virtual representation of an individual epilepsy patient. It '
            'integrates multi-modal clinical data -- demographics, EEG analyses, '
            'seizure diary events, medications, MRI findings, and neuropsychological '
            'assessments -- into a unified profile that enables personalised '
            'treatment planning, outcome prediction, and longitudinal monitoring.'
        ),
        'data_domains': [
            {
                'domain': 'demographics',
                'description': (
                    'Patient age, gender, and disease classification. Forms the '
                    'foundational identity layer of the digital twin.'
                ),
            },
            {
                'domain': 'eeg_analysis',
                'description': (
                    'EEG-derived features including predicted epilepsy labels, '
                    'AI confidence scores, and signal quality metrics from '
                    'automated analysis pipelines.'
                ),
            },
            {
                'domain': 'seizure_diary',
                'description': (
                    'Patient-reported and clinician-recorded seizure events '
                    'including duration, severity, triggers, awareness level, '
                    'motor signs, aura, and post-ictal recovery.'
                ),
            },
            {
                'domain': 'medications',
                'description': (
                    'Anti-epileptic drug (AED) regimens including drug name, '
                    'dosage in milligrams, and administration frequency. '
                    'Critical for treatment response modelling.'
                ),
            },
            {
                'domain': 'mri',
                'description': (
                    'Structural MRI findings including lesion type and location, '
                    'laterality, hippocampal sclerosis status, T2/FLAIR signals, '
                    'and radiologist classification.'
                ),
            },
            {
                'domain': 'assessments',
                'description': (
                    'Neuropsychological and clinical assessment instruments '
                    '(e.g. QOLIE-31, BDI-II, MoCA) with raw scores, maximum '
                    'scores, and clinical interpretations.'
                ),
            },
        ],
        'completeness_methodology': (
            'Each patient is evaluated across 6 data domains. A domain is '
            'considered populated if at least one record exists for that '
            'patient in the corresponding table (for demographics, both age '
            'and a specified gender must be present). The completeness score '
            'is the fraction of populated domains (0 to 1). Patients are then '
            'classified into readiness tiers: Full Twin (5-6 domains), '
            'Partial Twin (3-4 domains), Minimal Twin (1-2 domains), and '
            'No Data (0 domains).'
        ),
        'clinical_relevance': {
            'IEC_62304': (
                'Digital twin completeness tracking supports IEC 62304 software '
                'lifecycle requirements by ensuring data inputs to the AI/ML '
                'system are fully characterised, traceable, and auditable '
                'across all clinical data domains.'
            ),
            'FDA_AI_ML_PCCP': (
                'The FDA Predetermined Change Control Plan for AI/ML-based SaMD '
                'requires documentation of training data representativeness. '
                'Twin completeness metrics quantify data coverage gaps that may '
                'affect model performance and safety.'
            ),
            'ILAE': (
                'The International League Against Epilepsy recommends multi-modal '
                'data integration for epilepsy diagnosis and management. Digital '
                'twin profiles align with ILAE guidelines by combining EEG, '
                'imaging, seizure diaries, and neuropsychological data into '
                'unified patient representations.'
            ),
        },
        'remediation': [
            {
                'strategy': 'Targeted data collection',
                'description': (
                    'Identify patients with low completeness scores and '
                    'prioritise collection of missing domain data during '
                    'scheduled clinical visits.'
                ),
            },
            {
                'strategy': 'Automated data ingestion',
                'description': (
                    'Implement HL7 FHIR-based integration pipelines to '
                    'automatically pull EHR data (medications, assessments, '
                    'imaging reports) into the digital twin platform.'
                ),
            },
            {
                'strategy': 'Patient self-reporting',
                'description': (
                    'Deploy patient-facing mobile applications for seizure '
                    'diary capture and medication adherence tracking to '
                    'improve temporal resolution of twin data.'
                ),
            },
        ],
    }


# -- CLI quick-test --------------------------------------------------------

if __name__ == '__main__':
    import pprint
    print('=== Digital Twin Overview ===')
    ov = digital_twin_overview()
    pprint.pprint(ov)
    print()
    print('=== Digital Twin Breakdown (first 2 twins) ===')
    bd = digital_twin_breakdown()
    for tw in bd['patient_twins'][:2]:
        pprint.pprint(tw)
    print(f"\nDomain correlations: {len(bd['domain_correlation'])} pairs")
    print(f"Top complete: {len(bd['top_complete_patients'])} patients")
    print(f"Med-EEG cross: {len(bd['medication_eeg_cross'])} entries")
    print()
    print('=== Digital Twin Definitions ===')
    df = digital_twin_definitions()
    pprint.pprint(list(df.keys()))
