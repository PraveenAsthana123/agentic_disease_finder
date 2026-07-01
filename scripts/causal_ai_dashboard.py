"""Causal AI Dashboard — causal inference and relationship analysis for
epilepsy patient data.  Identifies medication→seizure-frequency causal
pathways, age/gender→outcome associations, MRI-finding→disease-severity
links, and trigger→seizure causal chains.

Aggregates real data from:
- data/clinical.db patients (40 patients)
- data/clinical.db seizure_diary (25 seizure events)
- data/clinical.db medications (9 medication records)
- data/clinical.db analyses (21 EEG analyses)
- data/clinical.db mri_findings (40 MRI scans)
- data/clinical.db assessments (423 assessment scores)
"""

import sqlite3
import json
import os
import math
from collections import defaultdict, Counter
from datetime import datetime

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


# ── Causal factor categories ────────────────────────────────────────────────
CAUSAL_CATEGORIES = {
    'Medication': {'color': '#3b82f6', 'icon': 'pill'},
    'Demographics': {'color': '#8b5cf6', 'icon': 'user'},
    'Neuroimaging': {'color': '#f59e0b', 'icon': 'brain'},
    'Triggers': {'color': '#ef4444', 'icon': 'zap'},
    'Clinical': {'color': '#10b981', 'icon': 'activity'},
    'EEG': {'color': '#06b6d4', 'icon': 'signal'},
}


def _load_patients(cur):
    rows = cur.execute(
        'SELECT patient_id, name, age, gender, disease FROM patients'
    ).fetchall()
    patients = {}
    for r in rows:
        patients[r[0]] = {
            'patient_id': r[0], 'name': r[1], 'age': r[2],
            'gender': r[3], 'disease': r[4],
        }
    return patients


def _load_seizures(cur):
    rows = cur.execute(
        'SELECT patient_id, event_date, duration_sec, severity, trigger, '
        'aura, awareness, injury FROM seizure_diary'
    ).fetchall()
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r[0]].append({
            'event_date': r[1], 'duration_sec': r[2], 'severity': r[3],
            'trigger': r[4], 'aura': r[5], 'awareness': r[6], 'injury': r[7],
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


# ═══════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

def causal_overview():
    conn = _conn()
    cur = conn.cursor()

    patients = _load_patients(cur)
    seizures = _load_seizures(cur)
    medications = _load_medications(cur)
    analyses = _load_analyses(cur)
    mri = _load_mri(cur)
    assessments = _load_assessments(cur)
    conn.close()

    total_patients = len(patients)
    total_seizures = sum(len(v) for v in seizures.values())
    patients_with_seizures = len(seizures)
    total_medications = sum(len(v) for v in medications.values())
    total_analyses = sum(len(v) for v in analyses.values())
    total_mri = sum(len(v) for v in mri.values())

    # ── Medication → Seizure causal pathways ──────────────────────────────
    med_seizure_links = []
    drug_seizure_counts = defaultdict(lambda: {'patients': set(), 'seizures': 0, 'doses': []})

    for pid, med_list in medications.items():
        patient_seizures = seizures.get(pid, [])
        for med in med_list:
            drug = med.get('drug_name', 'Unknown')
            dose = med.get('dose_mg', 0)
            drug_seizure_counts[drug]['patients'].add(pid)
            drug_seizure_counts[drug]['seizures'] += len(patient_seizures)
            if dose:
                drug_seizure_counts[drug]['doses'].append(dose)
            # Also handle AED lists
            for aed in med.get('aed', []):
                if aed != drug:
                    drug_seizure_counts[aed]['patients'].add(pid)
                    drug_seizure_counts[aed]['seizures'] += len(patient_seizures)

    for drug, info in drug_seizure_counts.items():
        n_patients = len(info['patients'])
        avg_seizures = round(info['seizures'] / n_patients, 1) if n_patients else 0
        avg_dose = _safe_mean(info['doses']) if info['doses'] else None
        strength = 'strong' if n_patients >= 3 else ('moderate' if n_patients >= 2 else 'weak')
        med_seizure_links.append({
            'drug': drug, 'patients': n_patients,
            'total_seizures': info['seizures'],
            'avg_seizures_per_patient': avg_seizures,
            'avg_dose_mg': avg_dose,
            'causal_strength': strength,
        })
    med_seizure_links.sort(key=lambda x: x['patients'], reverse=True)

    # ── Trigger → Seizure causal chains ───────────────────────────────────
    trigger_counts = Counter()
    trigger_severity = defaultdict(list)
    trigger_duration = defaultdict(list)
    for pid, evts in seizures.items():
        for e in evts:
            t = e.get('trigger') or 'Unknown'
            trigger_counts[t] += 1
            if e.get('severity'):
                trigger_severity[t].append(e['severity'])
            if e.get('duration_sec') and e['duration_sec'] > 0:
                trigger_duration[t].append(e['duration_sec'])

    trigger_chains = []
    for t, cnt in trigger_counts.most_common():
        sev_dist = dict(Counter(trigger_severity.get(t, [])))
        avg_dur = _safe_mean(trigger_duration.get(t, []))
        trigger_chains.append({
            'trigger': t, 'count': cnt,
            'severity_distribution': sev_dist,
            'avg_duration_sec': avg_dur,
            'risk_level': 'high' if cnt >= 3 else ('medium' if cnt >= 2 else 'low'),
        })

    # ── Age → Seizure severity association ────────────────────────────────
    age_groups = {'<30': [], '30-50': [], '50-70': [], '70+': []}
    for pid, evts in seizures.items():
        p = patients.get(pid, {})
        age = p.get('age')
        if age is None:
            continue
        if age < 30:
            grp = '<30'
        elif age < 50:
            grp = '30-50'
        elif age < 70:
            grp = '50-70'
        else:
            grp = '70+'
        for e in evts:
            age_groups[grp].append(e)

    age_severity = []
    for grp, evts in age_groups.items():
        if not evts:
            age_severity.append({'group': grp, 'seizures': 0, 'avg_duration': 0, 'severity_dist': {}})
            continue
        durations = [e['duration_sec'] for e in evts if e.get('duration_sec') and e['duration_sec'] > 0]
        sev = dict(Counter(e.get('severity', 'Unknown') for e in evts))
        age_severity.append({
            'group': grp, 'seizures': len(evts),
            'avg_duration': _safe_mean(durations), 'severity_dist': sev,
        })

    # ── Gender distribution ───────────────────────────────────────────────
    gender_dist = Counter()
    gender_seizures = defaultdict(int)
    for pid, p in patients.items():
        g = p.get('gender') or 'Unknown'
        if g in ('', None):
            g = 'Unknown'
        gender_dist[g] += 1
        gender_seizures[g] += len(seizures.get(pid, []))
    gender_analysis = [
        {'gender': g, 'patients': cnt, 'seizures': gender_seizures[g],
         'seizure_rate': round(gender_seizures[g] / cnt, 2) if cnt else 0}
        for g, cnt in gender_dist.most_common()
    ]

    # ── Causal graph summary ──────────────────────────────────────────────
    nodes = []
    edges = []
    # Nodes: medications, triggers, demographics, outcomes
    node_id = 0
    node_map = {}
    for drug_info in med_seizure_links:
        nid = f'med_{node_id}'
        nodes.append({'id': nid, 'label': drug_info['drug'], 'type': 'Medication',
                       'size': drug_info['patients'] * 10})
        node_map[('med', drug_info['drug'])] = nid
        node_id += 1

    for trig in trigger_chains:
        nid = f'trig_{node_id}'
        nodes.append({'id': nid, 'label': trig['trigger'], 'type': 'Trigger',
                       'size': trig['count'] * 8})
        node_map[('trig', trig['trigger'])] = nid
        node_id += 1

    # outcome node
    seizure_node = f'outcome_{node_id}'
    nodes.append({'id': seizure_node, 'label': 'Seizure Events', 'type': 'Outcome', 'size': 30})
    node_id += 1

    # Edges: med → seizure
    for drug_info in med_seizure_links:
        src = node_map.get(('med', drug_info['drug']))
        if src:
            edges.append({'source': src, 'target': seizure_node,
                          'strength': drug_info['causal_strength'],
                          'relation': 'modulates', 'weight': drug_info['patients']})
    # Edges: trigger → seizure
    for trig in trigger_chains:
        src = node_map.get(('trig', trig['trigger']))
        if src:
            edges.append({'source': src, 'target': seizure_node,
                          'strength': trig['risk_level'],
                          'relation': 'causes', 'weight': trig['count']})

    # ── Causal pathway count ──────────────────────────────────────────────
    total_pathways = len(med_seizure_links) + len(trigger_chains)
    strong_pathways = sum(1 for m in med_seizure_links if m['causal_strength'] == 'strong')
    strong_pathways += sum(1 for t in trigger_chains if t['risk_level'] == 'high')

    kpis = {
        'total_patients': total_patients,
        'patients_with_seizures': patients_with_seizures,
        'total_seizure_events': total_seizures,
        'total_causal_pathways': total_pathways,
        'strong_pathways': strong_pathways,
        'medications_analysed': len(med_seizure_links),
        'triggers_identified': len([t for t in trigger_chains if t['trigger'] != 'Unknown']),
        'causal_graph_nodes': len(nodes),
    }

    return {
        'kpis': kpis,
        'medication_seizure_links': med_seizure_links,
        'trigger_chains': trigger_chains,
        'age_severity': age_severity,
        'gender_analysis': gender_analysis,
        'causal_graph': {'nodes': nodes, 'edges': edges},
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════

def causal_breakdown():
    conn = _conn()
    cur = conn.cursor()

    patients = _load_patients(cur)
    seizures = _load_seizures(cur)
    medications = _load_medications(cur)
    analyses = _load_analyses(cur)
    mri = _load_mri(cur)
    assessments = _load_assessments(cur)
    conn.close()

    # ── Per-patient causal profile ────────────────────────────────────────
    patient_profiles = []
    for pid, p in patients.items():
        p_seizures = seizures.get(pid, [])
        p_meds = medications.get(pid, [])
        p_analyses = analyses.get(pid, [])
        p_mri = mri.get(pid, [])
        p_assess = assessments.get(pid, [])

        drugs = []
        for m in p_meds:
            dn = m.get('drug_name')
            if dn:
                drugs.append(dn)
            for aed in m.get('aed', []):
                if aed not in drugs:
                    drugs.append(aed)

        triggers = [s.get('trigger') for s in p_seizures if s.get('trigger')]
        severities = [s.get('severity') for s in p_seizures if s.get('severity')]
        durations = [s['duration_sec'] for s in p_seizures if s.get('duration_sec') and s['duration_sec'] > 0]

        mri_findings_list = []
        for m in p_mri:
            finding = m.get('finding') or m.get('interpretation') or m.get('result', '')
            if finding:
                mri_findings_list.append(finding)

        avg_confidence = _safe_mean([a['confidence'] for a in p_analyses if a.get('confidence')])

        causal_factors = len(drugs) + len(set(triggers))
        risk = 'high' if len(p_seizures) >= 3 else ('medium' if len(p_seizures) >= 1 else 'low')

        patient_profiles.append({
            'patient_id': pid,
            'name': p.get('name', ''),
            'age': p.get('age'),
            'gender': p.get('gender', ''),
            'seizure_count': len(p_seizures),
            'medications': drugs,
            'triggers': list(set(triggers)),
            'severity_dist': dict(Counter(severities)),
            'avg_duration_sec': _safe_mean(durations),
            'mri_findings': mri_findings_list[:3],
            'assessment_count': len(p_assess),
            'avg_ai_confidence': avg_confidence,
            'causal_factors': causal_factors,
            'risk_level': risk,
        })

    patient_profiles.sort(key=lambda x: x['seizure_count'], reverse=True)

    # ── MRI finding → seizure correlation ─────────────────────────────────
    mri_finding_types = defaultdict(lambda: {'patients': set(), 'seizures': 0})
    for pid, mri_list in mri.items():
        for m in mri_list:
            finding = m.get('finding') or m.get('interpretation') or m.get('result', 'Normal')
            if not finding:
                finding = 'Normal'
            # Normalize common findings
            flow = finding.lower()
            if 'normal' in flow or 'unremarkable' in flow:
                cat = 'Normal'
            elif 'sclero' in flow or 'mts' in flow or 'mesial' in flow:
                cat = 'Mesial Temporal Sclerosis'
            elif 'atrophy' in flow:
                cat = 'Cortical Atrophy'
            elif 'lesion' in flow or 'focal' in flow:
                cat = 'Focal Lesion'
            elif 'glio' in flow or 'tumor' in flow:
                cat = 'Tumor/Gliosis'
            elif 'dysplasia' in flow or 'fcd' in flow:
                cat = 'Cortical Dysplasia'
            else:
                cat = finding[:40]
            mri_finding_types[cat]['patients'].add(pid)
            mri_finding_types[cat]['seizures'] += len(seizures.get(pid, []))

    mri_correlations = []
    for cat, info in sorted(mri_finding_types.items(), key=lambda x: len(x[1]['patients']), reverse=True):
        n = len(info['patients'])
        mri_correlations.append({
            'finding': cat, 'patients': n, 'seizures': info['seizures'],
            'avg_seizures': round(info['seizures'] / n, 1) if n else 0,
            'association': 'positive' if info['seizures'] > 0 else 'neutral',
        })

    # ── Assessment score → seizure frequency association ──────────────────
    instrument_seizure_corr = defaultdict(lambda: {'scores': [], 'seizures': []})
    for pid, assess_list in assessments.items():
        n_seizures = len(seizures.get(pid, []))
        for a in assess_list:
            inst = a.get('instrument', '')
            score = a.get('score')
            if score is not None and inst:
                instrument_seizure_corr[inst]['scores'].append(score)
                instrument_seizure_corr[inst]['seizures'].append(n_seizures)

    assessment_links = []
    for inst, data in instrument_seizure_corr.items():
        n = len(data['scores'])
        if n < 2:
            continue
        avg_score = _safe_mean(data['scores'])
        avg_seizures = _safe_mean(data['seizures'])
        # Simple correlation direction
        high_score_seizures = [data['seizures'][i] for i in range(n) if data['scores'][i] > avg_score]
        low_score_seizures = [data['seizures'][i] for i in range(n) if data['scores'][i] <= avg_score]
        h = _safe_mean(high_score_seizures) if high_score_seizures else 0
        lo = _safe_mean(low_score_seizures) if low_score_seizures else 0
        direction = 'positive' if h > lo else ('negative' if h < lo else 'neutral')
        assessment_links.append({
            'instrument': inst, 'samples': n,
            'avg_score': avg_score, 'avg_seizures': avg_seizures,
            'direction': direction,
        })
    assessment_links.sort(key=lambda x: x['samples'], reverse=True)

    # ── Temporal causal analysis (seizure timeline) ───────────────────────
    timeline = []
    for pid, evts in seizures.items():
        p = patients.get(pid, {})
        for e in evts:
            timeline.append({
                'patient_id': pid,
                'date': e.get('event_date', ''),
                'severity': e.get('severity', 'Unknown'),
                'duration_sec': e.get('duration_sec', 0),
                'trigger': e.get('trigger', 'Unknown'),
                'aura': e.get('aura'),
                'injury': e.get('injury'),
            })
    timeline.sort(key=lambda x: x.get('date', ''))

    # ── Intervention effectiveness (medication comparison) ────────────────
    drug_outcomes = defaultdict(lambda: {'patients': set(), 'seizure_counts': [],
                                          'severities': [], 'durations': []})
    for pid, med_list in medications.items():
        p_seizures = seizures.get(pid, [])
        for m in med_list:
            drug = m.get('drug_name', 'Unknown')
            drug_outcomes[drug]['patients'].add(pid)
            drug_outcomes[drug]['seizure_counts'].append(len(p_seizures))
            for s in p_seizures:
                if s.get('severity'):
                    drug_outcomes[drug]['severities'].append(s['severity'])
                if s.get('duration_sec') and s['duration_sec'] > 0:
                    drug_outcomes[drug]['durations'].append(s['duration_sec'])

    interventions = []
    for drug, data in drug_outcomes.items():
        n = len(data['patients'])
        interventions.append({
            'drug': drug, 'patients': n,
            'avg_seizures': _safe_mean(data['seizure_counts']),
            'severity_dist': dict(Counter(data['severities'])),
            'avg_duration_sec': _safe_mean(data['durations']),
            'effectiveness': 'good' if _safe_mean(data['seizure_counts']) < 2 else (
                'moderate' if _safe_mean(data['seizure_counts']) < 4 else 'poor'),
        })
    interventions.sort(key=lambda x: x['patients'], reverse=True)

    return {
        'patient_profiles': patient_profiles[:20],
        'mri_correlations': mri_correlations,
        'assessment_links': assessment_links[:10],
        'seizure_timeline': timeline,
        'interventions': interventions,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

def definitions():
    return {
        'sections': [
            {
                'title': 'Causal Inference Methods',
                'items': [
                    {'term': 'Association Analysis',
                     'definition': 'Statistical co-occurrence between exposure (e.g. medication, trigger) and outcome (seizure events). Identifies factors that occur together more frequently than expected by chance.'},
                    {'term': 'Intervention Analysis',
                     'definition': 'Comparison of seizure outcomes across different medication regimens. Evaluates whether changing the intervention (drug, dose) alters the outcome distribution.'},
                    {'term': 'Temporal Precedence',
                     'definition': 'A cause must precede its effect in time. Seizure diary timeline analysis identifies temporal ordering between triggers, medications, and seizure events.'},
                    {'term': 'Confounding Control',
                     'definition': 'Adjusting for variables (age, gender, comorbidities) that may independently affect both the exposure and the outcome, to isolate the true causal effect.'},
                    {'term': 'Causal Pathway Strength',
                     'definition': 'Classified as Strong (3+ patients), Moderate (2 patients), or Weak (1 patient) based on the number of independent observations supporting the association.'},
                ],
            },
            {
                'title': 'Causal Factor Categories',
                'items': [
                    {'term': 'Medication Factors',
                     'definition': 'Antiepileptic drugs (AEDs) including Levetiracetam, Lamotrigine, Valproate, Carbamazepine, Topiramate. Dose, frequency, and polytherapy regimens are analysed for seizure-frequency modulation.'},
                    {'term': 'Demographic Factors',
                     'definition': 'Age groups (<30, 30-50, 50-70, 70+) and gender. Age-related epilepsy syndromes and sex-specific pharmacokinetics may act as effect modifiers.'},
                    {'term': 'Neuroimaging Factors',
                     'definition': 'MRI findings including mesial temporal sclerosis, cortical atrophy, focal lesions, cortical dysplasia, and tumors. Structural abnormalities are linked to seizure localisation and prognosis.'},
                    {'term': 'Seizure Triggers',
                     'definition': 'Identified precipitating factors: sleep deprivation, stress, alcohol, missed medication, photosensitivity, hormonal changes. Each trigger\'s causal contribution is quantified.'},
                    {'term': 'EEG Biomarkers',
                     'definition': 'AI-derived EEG features (spectral power, complexity measures, connectivity) that may serve as causal mediators between structural pathology and clinical seizure expression.'},
                ],
            },
            {
                'title': 'Causal Graph Notation',
                'items': [
                    {'term': 'Nodes',
                     'definition': 'Represent causal variables: medications, triggers, demographics, outcomes. Node size reflects the number of patients involved.'},
                    {'term': 'Edges (Directed)',
                     'definition': 'Arrows from cause to effect. "Modulates" edges (medication → seizure) indicate regulatory relationships; "causes" edges (trigger → seizure) indicate precipitating relationships.'},
                    {'term': 'Edge Strength',
                     'definition': 'Line weight and colour encode causal strength: strong (3+ patients, solid), moderate (2 patients, dashed), weak (1 patient, dotted).'},
                ],
            },
            {
                'title': 'Clinical Relevance',
                'items': [
                    {'term': 'IEC 62304',
                     'definition': 'Medical device software lifecycle standard. Causal AI outputs are classified as clinical decision support (Class B) requiring documented validation and risk analysis.'},
                    {'term': 'FDA AI/ML PCCP',
                     'definition': 'Pre-determined Change Control Plan for AI/ML-based SaMD. Causal model updates (new pathways, updated strengths) must follow the locked PCCP protocol.'},
                    {'term': 'ILAE Classification',
                     'definition': 'International League Against Epilepsy classification of seizure types and epilepsy syndromes. Causal analysis respects the ILAE 2017 operational classification framework.'},
                    {'term': 'Evidence Grading',
                     'definition': 'Causal pathways are graded: Level I (strong, multi-patient, temporally ordered), Level II (moderate, 2+ patients), Level III (weak, single observation). Only Level I/II should inform clinical action.'},
                ],
            },
            {
                'title': 'Remediation Strategies',
                'items': [
                    {'term': 'Medication Optimisation',
                     'definition': 'When causal analysis identifies a strong medication→high-seizure-rate pathway, recommend dose adjustment, drug substitution, or add-on therapy review.'},
                    {'term': 'Trigger Avoidance',
                     'definition': 'When a trigger→seizure chain is confirmed at Level I/II, encode trigger-specific avoidance guidance in the patient care plan.'},
                    {'term': 'Prospective Validation',
                     'definition': 'All causal hypotheses generated from observational data require prospective validation before clinical action. Flag as "hypothesis" until validated.'},
                ],
            },
        ],
    }


if __name__ == '__main__':
    import json as _json
    ov = causal_overview()
    print(_json.dumps(ov, indent=2, default=str)[:3000])
