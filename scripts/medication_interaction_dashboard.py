"""Medication Interaction Checker Dashboard — AED polytherapy risk,
drug-drug interaction screening, seizure threshold analysis, and
psychiatric comorbidity cross-reference from real clinical.db data.

Maps clinical.db tables to medication interaction concepts:
- medications        -> AED prescriptions (drug, dose, frequency)
- assessments (PHQ9) -> depression screening (seizure threshold impact)
- assessments (GAD7) -> anxiety screening (AED side-effect overlap)
- assessments (CSSRS)-> suicidality risk (AED psychiatric ADR flag)
- seizure_diary      -> seizure frequency (efficacy context)
- patients           -> per-patient medication profiles
- transaction_log    -> prescription pipeline events
"""

import sqlite3
import os
import json
from collections import Counter, defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# ── Known AED drug-drug interactions (evidence-based, published literature) ──
# Source: Patsalos PN et al., Epilepsia 2018; Zaccara G & Perucca E, CNS Drugs 2014
AED_INTERACTIONS = [
    {
        'drug_a': 'Valproate',
        'drug_b': 'Lamotrigine',
        'severity': 'major',
        'mechanism': 'Valproate inhibits lamotrigine glucuronidation (UGT1A4), '
                     'doubling lamotrigine levels. Risk of Stevens-Johnson syndrome.',
        'action': 'Reduce lamotrigine dose by 50% when co-prescribed with valproate.',
        'reference': 'Patsalos PN, Epilepsia 2018',
    },
    {
        'drug_a': 'Carbamazepine',
        'drug_b': 'Lamotrigine',
        'severity': 'moderate',
        'mechanism': 'Carbamazepine induces lamotrigine metabolism (CYP3A4/UGT), '
                     'reducing lamotrigine levels by ~40%.',
        'action': 'Increase lamotrigine dose; monitor serum levels.',
        'reference': 'Zaccara G, CNS Drugs 2014',
    },
    {
        'drug_a': 'Carbamazepine',
        'drug_b': 'Valproate',
        'severity': 'major',
        'mechanism': 'Carbamazepine induces valproate metabolism; valproate inhibits '
                     'carbamazepine-10,11-epoxide hydrolase, increasing toxic metabolite.',
        'action': 'Monitor carbamazepine-epoxide and valproate levels closely.',
        'reference': 'Patsalos PN, Epilepsia 2018',
    },
    {
        'drug_a': 'Valproate',
        'drug_b': 'Topiramate',
        'severity': 'moderate',
        'mechanism': 'Co-administration increases risk of hyperammonemia '
                     'and encephalopathy (valproate + topiramate synergy).',
        'action': 'Monitor ammonia levels; watch for lethargy/confusion.',
        'reference': 'Deutsch SI, Clin Neuropharmacol 2009',
    },
    {
        'drug_a': 'Carbamazepine',
        'drug_b': 'Topiramate',
        'severity': 'moderate',
        'mechanism': 'Carbamazepine induces topiramate clearance, reducing '
                     'topiramate levels by ~40%.',
        'action': 'May need higher topiramate dose; monitor efficacy.',
        'reference': 'Sachdeo RC, Epilepsia 2002',
    },
    {
        'drug_a': 'Levetiracetam',
        'drug_b': 'Lamotrigine',
        'severity': 'none',
        'mechanism': 'No clinically significant pharmacokinetic interaction. '
                     'Levetiracetam is renally excreted, not hepatically metabolised.',
        'action': 'Safe combination — no dose adjustment needed.',
        'reference': 'Patsalos PN, Epilepsia 2018',
    },
]

# AEDs that lower seizure threshold (psychiatric ADR concern)
AED_PSYCHIATRIC_ADRS = {
    'Levetiracetam': {
        'risk': 'behavioral',
        'detail': 'Irritability, aggression, depression (5-10% incidence). '
                  'Monitor PHQ-9/GAD-7 at follow-up.',
        'frequency': '5-10%',
    },
    'Topiramate': {
        'risk': 'cognitive',
        'detail': 'Word-finding difficulty, cognitive slowing, weight loss. '
                  'May worsen depression at high doses.',
        'frequency': '10-20%',
    },
    'Valproate': {
        'risk': 'metabolic',
        'detail': 'Weight gain, tremor, hair loss. Teratogenic — contraindicated '
                  'in women of childbearing age without contraception.',
        'frequency': '15-30%',
    },
    'Carbamazepine': {
        'risk': 'hematologic',
        'detail': 'Hyponatremia, aplastic anemia (rare). CYP3A4 inducer — '
                  'interacts with oral contraceptives, warfarin, many drugs.',
        'frequency': '5-15%',
    },
    'Lamotrigine': {
        'risk': 'dermatologic',
        'detail': 'Rash (5-10%), Stevens-Johnson syndrome (0.1-0.3%). '
                  'Slow titration mandatory. Generally mood-stabilising.',
        'frequency': '5-10%',
    },
}

SEVERITY_COLORS = {
    'major': '#ef4444',
    'moderate': '#f97316',
    'minor': '#f59e0b',
    'none': '#10b981',
}

PSYCH_SEVERITY_COLORS = {
    'normal': '#10b981',
    'mild': '#f59e0b',
    'moderate': '#f97316',
    'severe': '#ef4444',
}


def _db_query(sql, params=()):
    if not os.path.exists(DB):
        return []
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _parse_med(row):
    """Parse a medication row with fields_json."""
    fields = json.loads(row['fields_json']) if row.get('fields_json') else {}
    return {
        'id': row['id'],
        'patient_id': row['patient_id'],
        'drug_name': fields.get('drug_name', 'Unknown'),
        'dose_mg': fields.get('dose_mg', 0),
        'frequency': fields.get('frequency', 'N/A'),
        'aed_list': fields.get('aed', []),
        'created_at': row.get('created_at', ''),
    }


def _load_medications():
    rows = _db_query(
        "SELECT id, patient_id, fields_json, created_at "
        "FROM medications ORDER BY patient_id, created_at"
    )
    return [_parse_med(r) for r in rows]


def _load_psych_assessments():
    return _db_query(
        "SELECT id, patient_id, instrument, score, max_score, "
        "interpretation, level, alert, created_at "
        "FROM assessments WHERE instrument IN ('PHQ9','GAD7','CSSRS') "
        "ORDER BY patient_id, instrument, created_at"
    )


def _load_seizure_diary():
    return _db_query(
        "SELECT id, patient_id, event_date, severity, trigger, rescue_med, created_at "
        "FROM seizure_diary ORDER BY patient_id, created_at"
    )


def _load_patients():
    return _db_query(
        "SELECT patient_id, name, age, gender, disease, department, created_at "
        "FROM patients ORDER BY patient_id"
    )


def _load_pipeline_events():
    return _db_query(
        "SELECT id, component, action, actor, detail, ts_utc, ts_local "
        "FROM transaction_log "
        "WHERE component IN ('medication', 'pharmacy', 'prescription', 'patient_chat') "
        "OR action LIKE '%medication%' OR action LIKE '%prescri%' "
        "ORDER BY ts_utc"
    )


def _check_interactions(patient_meds):
    """Check known drug-drug interactions for a patient's medication list."""
    drug_names = [m['drug_name'] for m in patient_meds]
    found = []
    for ix in AED_INTERACTIONS:
        a, b = ix['drug_a'], ix['drug_b']
        if a in drug_names and b in drug_names:
            found.append(ix)
    return found


def _polytherapy_tier(count):
    if count <= 1:
        return 'monotherapy'
    elif count == 2:
        return 'duotherapy'
    elif count == 3:
        return 'tritherapy'
    else:
        return 'polytherapy (>3)'


# ---------------------------------------------------------------------------
# overview()
# ---------------------------------------------------------------------------

def overview():
    """KPI-level summary for the Medication Interaction Checker dashboard."""
    meds = _load_medications()
    psych = _load_psych_assessments()
    seizures = _load_seizure_diary()
    patients = _load_patients()
    pipeline = _load_pipeline_events()

    if not meds:
        return {
            'available': False,
            'message': 'No medication data available. Add prescriptions first.',
        }

    total_prescriptions = len(meds)
    unique_drugs = list(set(m['drug_name'] for m in meds))
    unique_patients = list(set(m['patient_id'] for m in meds))

    # Group meds by patient
    patient_meds = defaultdict(list)
    for m in meds:
        patient_meds[m['patient_id']].append(m)

    # Interaction screening
    all_interactions = []
    patients_with_interactions = set()
    for pid, pm in patient_meds.items():
        ix = _check_interactions(pm)
        if ix:
            patients_with_interactions.add(pid)
            for i in ix:
                all_interactions.append({**i, 'patient_id': pid})

    total_interactions = len(all_interactions)
    major_count = sum(1 for i in all_interactions if i['severity'] == 'major')
    moderate_count = sum(1 for i in all_interactions if i['severity'] == 'moderate')

    # Polytherapy distribution
    poly_dist = Counter(_polytherapy_tier(len(pm)) for pm in patient_meds.values())

    # Drug frequency
    drug_counts = Counter(m['drug_name'] for m in meds)

    # Psychiatric risk: patients with severe PHQ9/GAD7 who might need psychotropics
    psych_risk_patients = set()
    for a in psych:
        if a['level'] in ('severe', 'moderate') and a['instrument'] in ('PHQ9', 'GAD7'):
            psych_risk_patients.add(a['patient_id'])

    # AED psychiatric ADR flags
    adr_flags = []
    for pid, pm in patient_meds.items():
        for m in pm:
            if m['drug_name'] in AED_PSYCHIATRIC_ADRS:
                adr_flags.append({
                    'patient_id': pid,
                    'drug': m['drug_name'],
                    **AED_PSYCHIATRIC_ADRS[m['drug_name']],
                })

    # Severity distribution of interactions
    severity_dist = [
        {'severity': 'major', 'count': major_count, 'color': SEVERITY_COLORS['major']},
        {'severity': 'moderate', 'count': moderate_count, 'color': SEVERITY_COLORS['moderate']},
        {'severity': 'none', 'count': sum(1 for i in all_interactions if i['severity'] == 'none'),
         'color': SEVERITY_COLORS['none']},
    ]

    # Daily activity
    from collections import OrderedDict
    daily = OrderedDict()
    for m in meds:
        day = (m.get('created_at') or '')[:10]
        if day:
            daily[day] = daily.get(day, 0) + 1
    daily_activity = [{'date': d, 'count': c} for d, c in daily.items()]

    return {
        'available': True,
        'kpis': {
            'total_prescriptions': total_prescriptions,
            'unique_drugs': len(unique_drugs),
            'patients_on_meds': len(unique_patients),
            'interactions_detected': total_interactions,
            'major_interactions': major_count,
            'patients_at_risk': len(patients_with_interactions),
            'polytherapy_patients': sum(1 for pm in patient_meds.values() if len(pm) >= 2),
            'adr_flags': len(adr_flags),
        },
        'drug_frequency': [{'drug': d, 'count': c} for d, c in
                           drug_counts.most_common()],
        'polytherapy_distribution': [{'tier': t, 'count': c} for t, c in
                                     sorted(poly_dist.items())],
        'severity_distribution': severity_dist,
        'daily_activity': daily_activity,
        'psych_risk_patient_count': len(psych_risk_patients),
        'pipeline_event_count': len(pipeline),
    }


# ---------------------------------------------------------------------------
# breakdown()
# ---------------------------------------------------------------------------

def breakdown():
    """Detailed breakdown: medication inventory, interaction checks,
    patient profiles, ADR flags, psychiatric cross-reference."""
    meds = _load_medications()
    psych = _load_psych_assessments()
    seizures = _load_seizure_diary()
    patients = _load_patients()
    pipeline = _load_pipeline_events()

    # Medication inventory
    med_inventory = []
    for m in meds:
        med_inventory.append({
            'id': m['id'],
            'patient_id': m['patient_id'],
            'drug_name': m['drug_name'],
            'dose_mg': m['dose_mg'],
            'frequency': m['frequency'],
            'has_adr_risk': m['drug_name'] in AED_PSYCHIATRIC_ADRS,
            'adr_type': AED_PSYCHIATRIC_ADRS.get(m['drug_name'], {}).get('risk', ''),
            'created_at': m['created_at'],
        })

    # Group meds by patient
    patient_meds = defaultdict(list)
    for m in meds:
        patient_meds[m['patient_id']].append(m)

    # Interaction results per patient
    interaction_results = []
    for pid, pm in patient_meds.items():
        ix = _check_interactions(pm)
        drugs = [m['drug_name'] for m in pm]
        tier = _polytherapy_tier(len(pm))
        for i in ix:
            interaction_results.append({
                'patient_id': pid,
                'drug_a': i['drug_a'],
                'drug_b': i['drug_b'],
                'severity': i['severity'],
                'mechanism': i['mechanism'],
                'action': i['action'],
                'reference': i['reference'],
            })

    # Patient profiles with medication + psych summary
    psych_by_patient = defaultdict(list)
    for a in psych:
        psych_by_patient[a['patient_id']].append({
            'instrument': a['instrument'],
            'score': a['score'],
            'max_score': a['max_score'],
            'level': a['level'],
            'alert': a['alert'],
        })

    seizure_by_patient = defaultdict(int)
    for s in seizures:
        seizure_by_patient[s['patient_id']] += 1

    patient_profiles = []
    for pid, pm in sorted(patient_meds.items()):
        drugs = [m['drug_name'] for m in pm]
        ix = _check_interactions(pm)
        worst_severity = 'none'
        for i in ix:
            if i['severity'] == 'major':
                worst_severity = 'major'
            elif i['severity'] == 'moderate' and worst_severity != 'major':
                worst_severity = 'moderate'

        adr_risks = [AED_PSYCHIATRIC_ADRS[d]['risk'] for d in drugs
                     if d in AED_PSYCHIATRIC_ADRS]

        patient_profiles.append({
            'patient_id': pid,
            'drugs': drugs,
            'dose_details': [{'drug': m['drug_name'], 'dose_mg': m['dose_mg'],
                              'frequency': m['frequency']} for m in pm],
            'therapy_tier': _polytherapy_tier(len(pm)),
            'interaction_count': len(ix),
            'worst_interaction': worst_severity,
            'adr_risks': adr_risks,
            'seizure_entries': seizure_by_patient.get(pid, 0),
            'psych_assessments': psych_by_patient.get(pid, []),
        })

    # ADR detail table
    adr_detail = []
    for pid, pm in patient_meds.items():
        for m in pm:
            if m['drug_name'] in AED_PSYCHIATRIC_ADRS:
                info = AED_PSYCHIATRIC_ADRS[m['drug_name']]
                adr_detail.append({
                    'patient_id': pid,
                    'drug': m['drug_name'],
                    'dose_mg': m['dose_mg'],
                    'risk_type': info['risk'],
                    'detail': info['detail'],
                    'frequency': info['frequency'],
                })

    # Psychiatric cross-reference: patients with severe mood symptoms
    psych_alerts = []
    for a in psych:
        if a['level'] in ('severe', 'moderate') or a.get('alert'):
            psych_alerts.append({
                'patient_id': a['patient_id'],
                'instrument': a['instrument'],
                'score': a['score'],
                'max_score': a['max_score'],
                'level': a['level'],
                'alert': a['alert'] or '',
                'on_aeds': a['patient_id'] in patient_meds,
            })

    # Pipeline events
    pipeline_events = pipeline[:100]

    # Full interaction knowledge base
    interaction_kb = AED_INTERACTIONS

    return {
        'medication_inventory': med_inventory,
        'interaction_results': interaction_results,
        'patient_profiles': patient_profiles,
        'adr_detail': adr_detail,
        'psych_alerts': psych_alerts,
        'pipeline_events': pipeline_events,
        'interaction_knowledge_base': interaction_kb,
    }


# ---------------------------------------------------------------------------
# definitions()
# ---------------------------------------------------------------------------

def definitions():
    """Definitions, clinical concepts, compliance references, and remediation."""
    return {
        'concepts': [
            {
                'term': 'Anti-Epileptic Drug (AED)',
                'definition': 'Medications used to prevent or reduce seizure frequency. '
                              'Also called anti-seizure medications (ASMs). Common AEDs include '
                              'levetiracetam, lamotrigine, valproate, carbamazepine, topiramate.',
            },
            {
                'term': 'Polytherapy',
                'definition': 'Use of two or more AEDs simultaneously. Monotherapy is preferred '
                              'when effective (ILAE recommendation). Polytherapy increases interaction '
                              'risk and side-effect burden.',
            },
            {
                'term': 'Drug-Drug Interaction (DDI)',
                'definition': 'Pharmacokinetic or pharmacodynamic alteration when two drugs are '
                              'co-administered. AED interactions commonly involve hepatic enzyme '
                              'induction (CYP3A4) or inhibition (UGT1A4).',
            },
            {
                'term': 'CYP450 Enzyme Induction',
                'definition': 'Upregulation of cytochrome P450 enzymes (e.g., CYP3A4) by drugs like '
                              'carbamazepine and phenytoin, accelerating metabolism of co-administered drugs.',
            },
            {
                'term': 'UGT Glucuronidation Inhibition',
                'definition': 'Valproate inhibits UDP-glucuronosyltransferase (UGT1A4), the primary '
                              'metabolic pathway for lamotrigine, causing dangerous level elevation.',
            },
            {
                'term': 'Stevens-Johnson Syndrome (SJS)',
                'definition': 'Severe, potentially fatal skin reaction associated with lamotrigine '
                              '(0.1-0.3% incidence). Risk increases with rapid titration or '
                              'valproate co-administration.',
            },
            {
                'term': 'Adverse Drug Reaction (ADR)',
                'definition': 'Unintended harmful effect of a medication at normal doses. '
                              'AED ADRs include cognitive impairment, mood changes, rash, '
                              'weight changes, and hematologic effects.',
            },
            {
                'term': 'Seizure Threshold',
                'definition': 'The level of neuronal excitability required to trigger a seizure. '
                              'Some psychotropic medications (e.g., bupropion, clozapine) lower '
                              'seizure threshold and require careful use in epilepsy patients.',
            },
        ],
        'quality_metrics': [
            {'metric': 'Interaction Detection Rate', 'target': '100%',
             'description': 'All known AED-AED pairs screened against reference database.'},
            {'metric': 'Polytherapy Alert Coverage', 'target': '100%',
             'description': 'All patients on >=2 AEDs flagged for interaction review.'},
            {'metric': 'ADR Monitoring Compliance', 'target': '>90%',
             'description': 'Patients on high-ADR-risk AEDs have follow-up assessments scheduled.'},
            {'metric': 'Psychiatric Cross-Reference', 'target': '>85%',
             'description': 'Patients with severe PHQ-9/GAD-7 scores reviewed for AED-psychotropic DDI.'},
        ],
        'drug_classes': [
            {'class': 'Sodium Channel Blockers', 'examples': 'Carbamazepine, Lamotrigine, Lacosamide'},
            {'class': 'SV2A Ligands', 'examples': 'Levetiracetam, Brivaracetam'},
            {'class': 'GABA Modulators', 'examples': 'Valproate, Benzodiazepines, Vigabatrin'},
            {'class': 'Carbonic Anhydrase Inhibitors', 'examples': 'Topiramate, Zonisamide'},
            {'class': 'Broad Spectrum', 'examples': 'Valproate, Levetiracetam, Lamotrigine'},
        ],
        'compliance_references': [
            {'ref': 'FDA Drug Interaction Guidance', 'url': 'https://www.fda.gov/drugs/drug-interactions-labeling',
             'note': 'Clinical pharmacology DDI study requirements for new drugs.'},
            {'ref': 'ILAE Treatment Guidelines', 'url': 'https://www.ilae.org',
             'note': 'International League Against Epilepsy monotherapy-first recommendation.'},
            {'ref': 'EU AI Act Art. 6', 'url': '',
             'note': 'High-risk AI classification for medical decision support systems.'},
            {'ref': 'ISO 14971', 'url': '',
             'note': 'Medical device risk management — applies to CDSS interaction checkers.'},
            {'ref': 'HIPAA', 'url': '',
             'note': 'Patient medication data requires PHI-level protection.'},
        ],
        'remediation_strategies': [
            {'issue': 'Major interaction detected',
             'remedy': 'Flag for pharmacist review; suggest dose adjustment per published guidelines.'},
            {'issue': 'Polytherapy without documented rationale',
             'remedy': 'Prompt clinician to document why monotherapy is insufficient.'},
            {'issue': 'AED with psychiatric ADR + high PHQ-9',
             'remedy': 'Cross-refer to psychiatry; consider AED switch to mood-neutral alternative.'},
            {'issue': 'Missing serum level monitoring',
             'remedy': 'Schedule therapeutic drug monitoring for narrow-index AEDs (carbamazepine, valproate).'},
        ],
    }
