"""Neurosurgeon / Epilepsy Surgery Dashboard — pre-surgical evaluation,
MRI lesion classification, seizure diary analysis, surgical candidacy
assessment from clinical evaluations.

Maps clinical.db tables to neurosurgical concepts:
- mri_findings          -> lesion classification, laterality, hippocampal sclerosis
- seizure_diary         -> seizure frequency, severity, semiology
- analyses              -> EEG analysis results, predictions, confidence
- clinical_decisions    -> neurologist agreement, final surgical decisions
- patients              -> demographics
- transaction_log       -> pipeline events
"""

import json
import sqlite3
import os
from collections import Counter
from datetime import datetime, timedelta

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

LESION_TYPE_LABELS = {
    'HS': 'Hippocampal Sclerosis',
    'CAV': 'Cavernoma',
    'NL': 'Non-Lesional',
    'AVM': 'Arteriovenous Malformation',
    'FCD': 'Focal Cortical Dysplasia',
    'ENC': 'Encephalomalacia',
    'NRM': 'Normal',
    'TUM': 'Tumor',
}

SEVERITY_SCORE_MAP = {
    'Mild': 1,
    'Moderate': 2,
    'Severe': 3,
}

# Lesion types that indicate surgical candidacy (when combined with other criteria)
SURGICAL_LESION_TYPES = {'FCD', 'CAV', 'TUM', 'AVM'}


def _db_query(sql, params=()):
    if not os.path.exists(DB):
        return []
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


def _avg(values):
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _safe_json(raw):
    """Parse fields_json, returning empty dict on failure."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _load_mri_findings():
    rows = _db_query(
        "SELECT id, patient_id, fields_json, created_at "
        "FROM mri_findings ORDER BY created_at"
    )
    results = []
    for r in rows:
        fields = _safe_json(r.get('fields_json'))
        results.append({
            'id': r['id'],
            'patient_id': r['patient_id'],
            'created_at': r.get('created_at'),
            'lesion_type': fields.get('lesion_type'),
            'lesion_label': fields.get('lesion_label'),
            'lesion_description': fields.get('lesion_description'),
            'lesion_location': fields.get('lesion_location'),
            'laterality': fields.get('laterality'),
            'hippocampal_sclerosis': fields.get('hippocampal_sclerosis'),
            'hippocampal_volume_asymmetry': fields.get('hippocampal_volume_asymmetry'),
            't2_flair_signal': fields.get('t2_flair_signal'),
            'enhancing': fields.get('enhancing'),
            'classification': fields.get('classification'),
            'classification_label': fields.get('classification_label'),
            'protocol': fields.get('protocol'),
            'radiologist_confidence': fields.get('radiologist_confidence'),
            'mri_available': fields.get('mri_available'),
            'quality': fields.get('quality'),
        })
    return results


def _load_seizure_diary():
    return _db_query(
        "SELECT id, patient_id, event_date, event_time, duration_sec, "
        "location, witnessed, aura, awareness, motor_signs, injury, "
        "post_ictal, recovery_min, er_visit, rescue_med, severity, "
        "trigger, notes, created_at "
        "FROM seizure_diary ORDER BY event_date"
    )


def _load_analyses():
    return _db_query(
        "SELECT id, upload_id, patient_id, disease, predicted_label, "
        "confidence, signal_quality, report_path, result_json, created_at "
        "FROM analyses ORDER BY created_at"
    )


def _load_pipeline_events():
    return _db_query(
        "SELECT id, component, action, actor, detail, ts_utc, ts_local "
        "FROM transaction_log ORDER BY ts_utc DESC LIMIT 20"
    )


def _load_daily_activity():
    """Daily pipeline event counts for the last 30 days."""
    rows = _db_query(
        "SELECT DATE(ts_utc) AS day, COUNT(*) AS cnt "
        "FROM transaction_log "
        "WHERE ts_utc >= DATE('now', '-30 days') "
        "GROUP BY DATE(ts_utc) ORDER BY day"
    )
    return [{'date': r['day'], 'count': r['cnt']} for r in rows]


def _is_surgical_candidate(mri_record):
    """Determine surgical candidacy from a parsed MRI record."""
    hs = mri_record.get('hippocampal_sclerosis')
    lt = mri_record.get('lesion_type')
    classification = mri_record.get('classification')
    if classification not in ('LESIONAL', None):
        if hs != 'Yes' and lt not in SURGICAL_LESION_TYPES:
            return False
    if hs == 'Yes':
        return True
    if lt in SURGICAL_LESION_TYPES:
        return True
    return False


# ---------------------------------------------------------------------------
# overview()
# ---------------------------------------------------------------------------

def overview():
    """KPI-level summary for the Neurosurgeon / Epilepsy Surgery dashboard."""
    mri = _load_mri_findings()
    seizures = _load_seizure_diary()
    analyses = _load_analyses()
    pipeline = _load_pipeline_events()
    daily = _load_daily_activity()

    if not mri and not seizures:
        return {
            'available': False,
            'message': 'No MRI findings or seizure diary data available. '
                       'Import MRI reports or seizure logs first.',
        }

    # --- KPIs ---
    mri_patient_ids = set(m['patient_id'] for m in mri if m.get('patient_id'))
    total_patients = len(mri_patient_ids)
    total_mri_scans = len(mri)

    lesional_count = sum(1 for m in mri if m.get('classification') == 'LESIONAL')
    non_lesional_count = sum(
        1 for m in mri if m.get('classification') in ('NON_LESIONAL', 'NORMAL')
    )

    total_seizure_events = len(seizures)

    surgical_candidates = len(set(
        m['patient_id'] for m in mri
        if m.get('patient_id') and _is_surgical_candidate(m)
    ))

    eeg_analyses_count = len(analyses)

    # Average seizure severity score
    severity_scores = []
    for s in seizures:
        sc = SEVERITY_SCORE_MAP.get(s.get('severity'))
        if sc is not None:
            severity_scores.append(sc)
    avg_seizure_severity_score = _avg(severity_scores)

    # --- Chart data ---

    # Lesion type distribution
    lesion_type_counts = Counter(
        m.get('lesion_type') for m in mri if m.get('lesion_type')
    )
    lesion_type_distribution = [
        {
            'lesion_type': lt,
            'label': LESION_TYPE_LABELS.get(lt, lt),
            'count': cnt,
        }
        for lt, cnt in sorted(lesion_type_counts.items(), key=lambda x: -x[1])
    ]

    # Laterality distribution
    laterality_counts = Counter(
        m.get('laterality') for m in mri if m.get('laterality')
    )
    laterality_distribution = [
        {'laterality': lat, 'count': cnt}
        for lat, cnt in sorted(laterality_counts.items(), key=lambda x: -x[1])
    ]

    # Lesion location distribution
    location_counts = Counter(
        m.get('lesion_location') for m in mri if m.get('lesion_location')
    )
    lesion_location_distribution = [
        {'location': loc, 'count': cnt}
        for loc, cnt in sorted(location_counts.items(), key=lambda x: -x[1])
    ]

    # Seizure severity distribution
    sev_counts = Counter(
        s.get('severity') for s in seizures if s.get('severity')
    )
    seizure_severity_distribution = [
        {'severity': sev, 'count': cnt}
        for sev, cnt in sorted(sev_counts.items(), key=lambda x: -x[1])
    ]

    # MRI classification distribution
    class_counts = Counter(
        m.get('classification') for m in mri if m.get('classification')
    )
    mri_classification_distribution = [
        {'classification': cls, 'count': cnt}
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])
    ]

    # Daily activity
    daily_activity = daily

    # Pipeline events
    pipeline_events = [
        {
            'id': ev['id'],
            'component': ev.get('component'),
            'action': ev.get('action'),
            'actor': ev.get('actor'),
            'detail': ev.get('detail'),
            'ts_utc': ev.get('ts_utc'),
            'ts_local': ev.get('ts_local'),
        }
        for ev in pipeline
    ]

    return {
        'available': True,
        'total_patients': total_patients,
        'total_mri_scans': total_mri_scans,
        'lesional_count': lesional_count,
        'non_lesional_count': non_lesional_count,
        'total_seizure_events': total_seizure_events,
        'surgical_candidates': surgical_candidates,
        'eeg_analyses_count': eeg_analyses_count,
        'avg_seizure_severity_score': avg_seizure_severity_score,
        'lesion_type_distribution': lesion_type_distribution,
        'laterality_distribution': laterality_distribution,
        'lesion_location_distribution': lesion_location_distribution,
        'seizure_severity_distribution': seizure_severity_distribution,
        'mri_classification_distribution': mri_classification_distribution,
        'daily_activity': daily_activity,
        'pipeline_events': pipeline_events,
        'kpis': [
            {'label': 'Patients (MRI)', 'value': str(total_patients)},
            {'label': 'MRI Scans', 'value': str(total_mri_scans)},
            {'label': 'Lesional', 'value': str(lesional_count),
             'color': '#f59e0b' if lesional_count > 0 else '#10b981'},
            {'label': 'Non-Lesional', 'value': str(non_lesional_count)},
            {'label': 'Seizure Events', 'value': str(total_seizure_events),
             'color': '#ef4444' if total_seizure_events > 20 else '#f59e0b' if total_seizure_events > 10 else '#10b981'},
            {'label': 'Surgical Candidates', 'value': str(surgical_candidates),
             'color': '#ef4444' if surgical_candidates > 10 else '#f59e0b' if surgical_candidates > 0 else '#10b981'},
            {'label': 'EEG Analyses', 'value': str(eeg_analyses_count)},
            {'label': 'Avg Seizure Severity', 'value': f'{avg_seizure_severity_score:.2f}',
             'color': '#ef4444' if avg_seizure_severity_score >= 2.5 else '#f59e0b' if avg_seizure_severity_score >= 1.5 else '#10b981'},
        ],
    }


# ---------------------------------------------------------------------------
# breakdown()
# ---------------------------------------------------------------------------

def breakdown():
    """Detailed breakdown — MRI inventory, patient surgical profiles,
    seizure log, EEG summary."""
    mri = _load_mri_findings()
    seizures = _load_seizure_diary()
    analyses = _load_analyses()

    if not mri and not seizures:
        return {'available': False}

    # --- MRI inventory ---
    mri_inventory = [
        {
            'patient_id': m['patient_id'],
            'lesion_type': m.get('lesion_type'),
            'lesion_label': m.get('lesion_label'),
            'lesion_location': m.get('lesion_location'),
            'laterality': m.get('laterality'),
            'classification': m.get('classification'),
            'hippocampal_sclerosis': m.get('hippocampal_sclerosis'),
            'hippocampal_volume_asymmetry': m.get('hippocampal_volume_asymmetry'),
            'radiologist_confidence': m.get('radiologist_confidence'),
            'created_at': m.get('created_at'),
        }
        for m in mri
    ]

    # --- Patient surgical profiles ---
    patient_mri = {}
    for m in mri:
        pid = m.get('patient_id')
        if pid:
            patient_mri.setdefault(pid, []).append(m)

    patient_seizures = {}
    for s in seizures:
        pid = s.get('patient_id')
        if pid:
            patient_seizures.setdefault(pid, []).append(s)

    all_patient_ids = sorted(set(patient_mri.keys()) | set(patient_seizures.keys()))

    patient_surgical_profiles = []
    for pid in all_patient_ids:
        p_mri = patient_mri.get(pid, [])
        p_sz = patient_seizures.get(pid, [])

        lesion_types = sorted(set(
            m.get('lesion_type') for m in p_mri if m.get('lesion_type')
        ))
        locations = sorted(set(
            m.get('lesion_location') for m in p_mri if m.get('lesion_location')
        ))
        lateralities = sorted(set(
            m.get('laterality') for m in p_mri if m.get('laterality')
        ))
        has_hs = any(m.get('hippocampal_sclerosis') == 'Yes' for m in p_mri)
        is_candidate = any(_is_surgical_candidate(m) for m in p_mri)

        seizure_severities = sorted(set(
            s.get('severity') for s in p_sz if s.get('severity')
        ))

        patient_surgical_profiles.append({
            'patient_id': pid,
            'mri_count': len(p_mri),
            'lesion_types': lesion_types,
            'locations': locations,
            'lateralities': lateralities,
            'has_hippocampal_sclerosis': has_hs,
            'surgical_candidate': is_candidate,
            'seizure_count': len(p_sz),
            'seizure_severities': seizure_severities,
        })

    # --- Seizure log ---
    seizure_log = [
        {
            'patient_id': s.get('patient_id'),
            'event_date': s.get('event_date'),
            'duration_sec': s.get('duration_sec'),
            'severity': s.get('severity'),
            'location': s.get('location'),
            'motor_signs': s.get('motor_signs'),
            'injury': s.get('injury'),
            'aura': s.get('aura'),
            'trigger': s.get('trigger'),
        }
        for s in seizures
    ]

    # --- EEG summary ---
    eeg_summary = [
        {
            'patient_id': a.get('patient_id'),
            'disease': a.get('disease'),
            'predicted_label': a.get('predicted_label'),
            'confidence': a.get('confidence'),
            'signal_quality': a.get('signal_quality'),
            'created_at': a.get('created_at'),
        }
        for a in analyses
    ]

    return {
        'available': True,
        'mri_inventory': mri_inventory,
        'patient_surgical_profiles': patient_surgical_profiles,
        'seizure_log': seizure_log,
        'eeg_summary': eeg_summary,
    }


# ---------------------------------------------------------------------------
# definitions()
# ---------------------------------------------------------------------------

def definitions():
    """Definitions tab for the Neurosurgeon / Epilepsy Surgery dashboard."""
    return {
        'concepts': [
            {
                'name': 'Epilepsy Surgery Candidacy',
                'description': 'Systematic evaluation to determine whether a patient with '
                               'drug-resistant epilepsy (failure of two or more appropriately '
                               'chosen AEDs) may benefit from surgical intervention. Requires '
                               'concordance of seizure semiology, EEG localization, and '
                               'neuroimaging findings to identify a resectable epileptogenic zone.',
            },
            {
                'name': 'Pre-surgical Evaluation',
                'description': 'Comprehensive workup including video-EEG monitoring, '
                               'high-resolution MRI (epilepsy protocol), neuropsychological '
                               'testing, PET/SPECT imaging, and Wada testing. Goal is to '
                               'localize the seizure onset zone and map eloquent cortex to '
                               'minimize post-operative deficits.',
            },
            {
                'name': 'MRI Lesion Classification',
                'description': 'Categorization of structural brain abnormalities identified on '
                               'MRI: LESIONAL (definite structural abnormality concordant with '
                               'seizure focus), NON_LESIONAL (no visible lesion despite clinical '
                               'epilepsy), EQUIVOCAL (subtle or uncertain findings requiring '
                               'advanced post-processing), NORMAL (no abnormality detected).',
            },
            {
                'name': 'Hippocampal Sclerosis',
                'description': 'Most common pathological substrate of temporal lobe epilepsy, '
                               'characterized by neuronal loss and gliosis in the hippocampus. '
                               'MRI shows hippocampal atrophy, increased T2/FLAIR signal, and '
                               'volume asymmetry. Anterior temporal lobectomy yields 60-80% '
                               'seizure freedom (Engel I) in mesial temporal sclerosis.',
            },
            {
                'name': 'EEG-MRI Concordance',
                'description': 'Agreement between the seizure onset zone identified by EEG and '
                               'the structural lesion seen on MRI. Concordance is the strongest '
                               'predictor of favorable surgical outcome. Discordance may require '
                               'invasive monitoring (SEEG or subdural grids) before resection.',
            },
            {
                'name': 'Surgical Resection Types',
                'description': 'Standard epilepsy surgery procedures: anterior temporal '
                               'lobectomy (ATL) for mesial temporal sclerosis, lesionectomy for '
                               'focal lesions (FCD, tumors, cavernomas), hemispherectomy for '
                               'catastrophic hemispheric epilepsy, corpus callosotomy for drop '
                               'attacks, and thermoablation (LITT) for deep-seated lesions.',
            },
            {
                'name': 'Engel Outcome Classification',
                'description': 'Standard classification of surgical outcome: Class I (seizure '
                               'free or only auras), Class II (rare disabling seizures), '
                               'Class III (worthwhile improvement >90% reduction), Class IV '
                               '(no worthwhile improvement). Reported at 1-year and 2-year '
                               'follow-up.',
            },
            {
                'name': 'SEEG / Invasive Monitoring',
                'description': 'Stereoelectroencephalography (SEEG) uses depth electrodes '
                               'implanted stereotactically to record from deep cortical '
                               'structures and precisely delineate the epileptogenic zone. '
                               'Indicated when non-invasive evaluation is discordant or when '
                               'the suspected focus involves eloquent cortex.',
            },
        ],
        'quality_metrics': [
            {
                'name': 'MRI Sensitivity',
                'description': 'Proportion of epilepsy surgery candidates with detectable '
                               'MRI lesions. High-resolution 3T epilepsy protocols detect '
                               'lesions in 75-85% of surgical candidates. MRI-negative cases '
                               'have lower seizure freedom rates post-surgery (40-50% vs 70-80%).',
            },
            {
                'name': 'EEG-MRI Concordance Rate',
                'description': 'Percentage of patients where scalp EEG seizure onset zone '
                               'is concordant with the MRI lesion location. Concordance rates '
                               'above 80% are associated with Engel I outcomes in >70% of cases.',
            },
            {
                'name': 'Seizure Freedom Rate',
                'description': 'Proportion of patients achieving Engel Class I outcome at '
                               '1-year post-surgery. Benchmark: 60-70% for temporal lobe '
                               'epilepsy surgery, 50-60% for extratemporal resections, '
                               '40-50% for MRI-negative cases.',
            },
            {
                'name': 'Surgical Complication Rate',
                'description': 'Rate of major surgical complications including infection, '
                               'hemorrhage, neurological deficit, and CSF leak. Acceptable '
                               'threshold: <5% major complications, <2% permanent neurological '
                               'deficit for standard temporal lobe procedures.',
            },
        ],
        'surgical_procedures': [
            {
                'name': 'Anterior Temporal Lobectomy',
                'description': 'Standard procedure for mesial temporal lobe epilepsy with '
                               'hippocampal sclerosis. Involves resection of the anterior '
                               'temporal neocortex, amygdala, and hippocampus. Seizure freedom '
                               'rate: 60-80%. Risk of verbal memory decline with dominant '
                               'hemisphere resection.',
            },
            {
                'name': 'Lesionectomy',
                'description': 'Targeted resection of a discrete epileptogenic lesion (FCD, '
                               'cavernoma, low-grade tumor, AVM) with or without surrounding '
                               'epileptogenic cortex. Outcome depends on completeness of '
                               'resection and lesion type. Best results with complete removal '
                               'and concordant EEG.',
            },
            {
                'name': 'Hemispherectomy',
                'description': 'Disconnection or removal of one cerebral hemisphere for '
                               'catastrophic unilateral epilepsy (Rasmussen encephalitis, '
                               'hemimegalencephaly, Sturge-Weber). Seizure freedom rate >80%. '
                               'Results in contralateral hemiparesis and hemianopia.',
            },
            {
                'name': 'Corpus Callosotomy',
                'description': 'Disconnection of the corpus callosum to prevent bilateral '
                               'seizure spread. Primarily indicated for atonic seizures (drop '
                               'attacks) in Lennox-Gastaut syndrome. Palliative procedure that '
                               'reduces seizure severity rather than achieving seizure freedom.',
            },
            {
                'name': 'Laser Ablation (LITT)',
                'description': 'MRI-guided laser interstitial thermal therapy for minimally '
                               'invasive ablation of epileptogenic foci. Particularly suited '
                               'for mesial temporal sclerosis, hypothalamic hamartomas, and '
                               'periventricular nodular heterotopia. Shorter hospital stay but '
                               'slightly lower seizure freedom rates than open resection.',
            },
        ],
        'compliance_refs': [
            {
                'ref': 'FDA 510(k) for Neurosurgical Planning AI',
                'note': 'AI systems used for neurosurgical planning, lesion detection, or '
                        'surgical candidacy assessment require 510(k) clearance. Predicate '
                        'devices include computer-aided detection systems for neuroimaging. '
                        'Clinical validation must demonstrate non-inferiority to expert '
                        'neuroradiologist interpretation.',
            },
            {
                'ref': 'AAN Practice Guidelines',
                'note': 'American Academy of Neurology guidelines for the evaluation of '
                        'drug-resistant epilepsy recommend referral to a comprehensive '
                        'epilepsy center after failure of two AEDs. MRI, video-EEG, and '
                        'neuropsychological testing are Level A recommendations.',
            },
            {
                'ref': 'ILAE Surgical Commission',
                'note': 'International League Against Epilepsy recommendations for '
                        'pre-surgical evaluation protocols, minimum dataset requirements, '
                        'and standardized outcome reporting (Engel and ILAE classification). '
                        'Emphasizes multidisciplinary surgical conference for candidacy decisions.',
            },
            {
                'ref': 'NICE Epilepsy Surgery Pathway',
                'note': 'National Institute for Health and Care Excellence guidelines mandate '
                        'referral for surgical evaluation within 2 years of drug resistance '
                        'diagnosis. Specifies minimum requirements for epilepsy surgery centers '
                        'including volume thresholds and multidisciplinary team composition.',
            },
            {
                'ref': 'Joint Commission Neurosurgery Standards',
                'note': 'Joint Commission accreditation standards for neurosurgical programs '
                        'require documented surgical planning protocols, intraoperative '
                        'monitoring standards, complication tracking, and outcome reporting. '
                        'AI-assisted planning must be documented in the surgical record.',
            },
        ],
        'remediation_strategies': [
            {
                'strategy': 'MRI Protocol Optimization',
                'description': 'Implement standardized 3T epilepsy MRI protocols with thin-cut '
                               'coronal FLAIR, volumetric T1, and hippocampal volumetry. Add '
                               'post-processing techniques (MAP, morphometric analysis) to '
                               'improve detection of subtle FCD in MRI-negative cases.',
            },
            {
                'strategy': 'Surgical Conference Workflow',
                'description': 'Establish weekly multidisciplinary surgical conferences with '
                               'structured case presentation including EEG-MRI concordance '
                               'matrix, neuropsychological risk profile, and AI-assisted '
                               'lesion probability maps. Document consensus decisions.',
            },
            {
                'strategy': 'Outcome Tracking Program',
                'description': 'Implement prospective Engel classification at 3, 6, 12, and '
                               '24 months post-surgery. Track complication rates, AED reduction '
                               'status, neuropsychological change scores, and quality of life '
                               '(QOLIE-31) to benchmark against published outcomes.',
            },
            {
                'strategy': 'Invasive Monitoring Decision Protocol',
                'description': 'Define criteria for SEEG recommendation: MRI-EEG discordance, '
                               'MRI-negative with localizable scalp EEG, proximity to eloquent '
                               'cortex, or bilateral seizure onset. Standardize electrode '
                               'implantation planning with 3D reconstruction software.',
            },
        ],
    }
