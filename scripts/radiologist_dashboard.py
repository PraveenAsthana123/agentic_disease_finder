"""Radiologist Dashboard — MRI findings analysis, lesion classification,
hippocampal sclerosis detection, structural neuroimaging quality metrics,
and neuroradiology reporting for epilepsy pre-surgical evaluation.

Maps clinical.db tables to neuroradiology concepts:
- mri_findings      -> Lesion type/location, laterality, classification,
                       hippocampal sclerosis, volume asymmetry, T2/FLAIR,
                       radiologist confidence, protocol, quality
- patients          -> Demographics, disease, department
- analyses          -> EEG analysis results (for EEG-MRI correlation)
"""

import json
import sqlite3
import os
from collections import Counter, defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# Lesion type labels
LESION_TYPE_LABELS = {
    'HS':  'Hippocampal Sclerosis',
    'FCD': 'Focal Cortical Dysplasia',
    'TUM': 'Tumor',
    'CAV': 'Cavernoma',
    'AVM': 'Arteriovenous Malformation',
    'ENC': 'Encephalomalacia',
    'NRM': 'Normal/Non-lesional',
}

# Confidence level thresholds
CONFIDENCE_LEVELS = {
    'High':     lambda c: c >= 0.8,
    'Moderate': lambda c: 0.5 <= c < 0.8,
    'Low':      lambda c: c < 0.5,
}


def _db_query(sql, params=()):
    if not os.path.exists(DB):
        return []
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
    finally:
        conn.close()


def _safe_json(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _avg(values):
    return round(sum(values) / len(values), 2) if values else None


def _classify_confidence(score):
    """Classify a 0-1 confidence score into High/Moderate/Low."""
    if score is None:
        return 'Unknown'
    if score >= 0.8:
        return 'High'
    if score >= 0.5:
        return 'Moderate'
    return 'Low'


def _load_mri_findings():
    """Load all MRI findings with parsed JSON fields."""
    rows = _db_query(
        "SELECT id, patient_id, fields_json, created_at FROM mri_findings "
        "ORDER BY created_at DESC"
    )
    results = []
    for r in rows:
        fields = _safe_json(r.get('fields_json'))
        results.append({
            'id': r['id'],
            'patient_id': r['patient_id'],
            'created_at': r.get('created_at'),
            **fields,
        })
    return results


def _load_patients():
    """Load all patients."""
    return _db_query(
        "SELECT patient_id, name, age, gender, disease, department, created_at "
        "FROM patients"
    )


def _load_latest_analyses():
    """Load the latest EEG analysis per patient."""
    rows = _db_query(
        "SELECT patient_id, predicted_label, confidence, signal_quality "
        "FROM analyses ORDER BY created_at DESC"
    )
    latest = {}
    for r in rows:
        pid = r.get('patient_id')
        if pid and pid not in latest:
            latest[pid] = {
                'predicted_label': r.get('predicted_label'),
                'confidence': r.get('confidence'),
                'signal_quality': r.get('signal_quality'),
            }
    return latest


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------

def overview():
    """KPI-level summary for the Radiologist dashboard."""
    mri_list = _load_mri_findings()
    patients = _load_patients()

    if not mri_list:
        return {
            'available': False,
            'message': 'No MRI findings data available. '
                       'Import MRI findings into mri_findings table first.',
        }

    total_mri_scans = len(mri_list)
    total_patients = len(patients)
    patient_ids_with_mri = set(m['patient_id'] for m in mri_list if m.get('patient_id'))
    mri_coverage = round(len(patient_ids_with_mri) / total_patients * 100, 1) if total_patients else 0

    # Classification counts
    lesion_positive_count = sum(
        1 for m in mri_list if m.get('classification') == 'LESIONAL'
    )
    non_lesional_count = total_mri_scans - lesion_positive_count
    lesion_positive_rate = round(lesion_positive_count / total_mri_scans * 100, 1) if total_mri_scans else 0

    # Radiologist confidence
    confidence_values = [
        m['radiologist_confidence'] for m in mri_list
        if m.get('radiologist_confidence') is not None
    ]
    # Handle both numeric and string confidence
    numeric_confidence = []
    for c in confidence_values:
        try:
            numeric_confidence.append(float(c))
        except (ValueError, TypeError):
            pass
    avg_radiologist_confidence = _avg(numeric_confidence)

    # Hippocampal sclerosis
    hippocampal_sclerosis_count = sum(
        1 for m in mri_list
        if m.get('hippocampal_sclerosis') in (True, 'true', 'True', 1, 'yes', 'Yes')
    )

    # Quality distribution
    quality_counts = Counter(
        m.get('quality', 'Unknown') for m in mri_list
    )
    quality_distribution = [
        {'quality': q, 'count': c}
        for q, c in quality_counts.most_common()
    ]

    # Lesion type distribution
    lesion_type_counts = Counter(
        m.get('lesion_type', 'Unknown') for m in mri_list
    )
    lesion_type_distribution = [
        {
            'lesion_type': lt,
            'label': LESION_TYPE_LABELS.get(lt, lt),
            'count': c,
        }
        for lt, c in lesion_type_counts.most_common()
    ]

    # Location distribution
    location_counts = Counter(
        m.get('lesion_location', 'Unknown') for m in mri_list
        if m.get('lesion_location')
    )
    location_distribution = [
        {'location': loc, 'count': c}
        for loc, c in location_counts.most_common()
    ]

    # Laterality distribution
    laterality_counts = Counter(
        m.get('laterality', 'Unknown') for m in mri_list
        if m.get('laterality')
    )
    laterality_distribution = [
        {'laterality': lat, 'count': c}
        for lat, c in laterality_counts.most_common()
    ]

    # Classification distribution
    classification_counts = Counter(
        m.get('classification', 'Unknown') for m in mri_list
    )
    classification_distribution = [
        {'classification': cl, 'count': c}
        for cl, c in classification_counts.most_common()
    ]

    # Confidence distribution (High / Moderate / Low)
    conf_level_counts = Counter(
        _classify_confidence(v) for v in numeric_confidence
    )
    confidence_distribution = [
        {'level': level, 'count': conf_level_counts.get(level, 0)}
        for level in ('High', 'Moderate', 'Low')
        if conf_level_counts.get(level, 0) > 0
    ]

    # Hippocampal volume asymmetry stats
    asymmetry_values = []
    for m in mri_list:
        val = m.get('hippocampal_volume_asymmetry')
        if val is not None:
            try:
                asymmetry_values.append(float(val))
            except (ValueError, TypeError):
                pass
    hippocampal_volume_asymmetry_stats = {
        'mean': _avg(asymmetry_values),
        'max': round(max(asymmetry_values), 2) if asymmetry_values else None,
        'min': round(min(asymmetry_values), 2) if asymmetry_values else None,
    }

    # T2/FLAIR signal distribution
    t2_flair_counts = Counter(
        m.get('t2_flair_signal', 'Unknown') for m in mri_list
        if m.get('t2_flair_signal')
    )
    t2_flair_distribution = [
        {'signal': sig, 'count': c}
        for sig, c in t2_flair_counts.most_common()
    ]

    # Protocol distribution
    protocol_counts = Counter(
        m.get('protocol', 'Unknown') for m in mri_list
        if m.get('protocol')
    )
    protocol_distribution = [
        {'protocol': p, 'count': c}
        for p, c in protocol_counts.most_common()
    ]

    # Enhancing vs non-enhancing
    enhancing_count = sum(
        1 for m in mri_list
        if m.get('enhancing') in (True, 'true', 'True', 1, 'yes', 'Yes')
    )
    non_enhancing_count = total_mri_scans - enhancing_count

    return {
        'available': True,
        'total_mri_scans': total_mri_scans,
        'lesion_positive_rate': lesion_positive_rate,
        'lesion_positive_count': lesion_positive_count,
        'non_lesional_count': non_lesional_count,
        'total_patients': total_patients,
        'mri_coverage': mri_coverage,
        'avg_radiologist_confidence': avg_radiologist_confidence,
        'hippocampal_sclerosis_count': hippocampal_sclerosis_count,
        'quality_distribution': quality_distribution,
        'lesion_type_distribution': lesion_type_distribution,
        'location_distribution': location_distribution,
        'laterality_distribution': laterality_distribution,
        'classification_distribution': classification_distribution,
        'confidence_distribution': confidence_distribution,
        'hippocampal_volume_asymmetry_stats': hippocampal_volume_asymmetry_stats,
        't2_flair_distribution': t2_flair_distribution,
        'protocol_distribution': protocol_distribution,
        'enhancing_count': enhancing_count,
        'non_enhancing_count': non_enhancing_count,
    }


# ---------------------------------------------------------------------------
# breakdown
# ---------------------------------------------------------------------------

def breakdown():
    """Detailed per-patient MRI findings with linked EEG analysis."""
    mri_list = _load_mri_findings()
    patients = _load_patients()
    latest_eeg = _load_latest_analyses()

    patient_map = {}
    for p in patients:
        pid = p.get('patient_id')
        if pid:
            patient_map[pid] = p

    # Group MRI findings by patient (use latest per patient)
    mri_by_patient = {}
    for m in mri_list:
        pid = m.get('patient_id')
        if pid and pid not in mri_by_patient:
            mri_by_patient[pid] = m

    patient_list = []
    for pid in sorted(mri_by_patient.keys()):
        pinfo = patient_map.get(pid, {})
        m = mri_by_patient[pid]

        mri_detail = {
            'mri_available': m.get('mri_available'),
            'quality': m.get('quality'),
            'lesion_type': m.get('lesion_type'),
            'lesion_label': m.get('lesion_label', LESION_TYPE_LABELS.get(m.get('lesion_type'), '')),
            'lesion_description': m.get('lesion_description'),
            'lesion_location': m.get('lesion_location'),
            'laterality': m.get('laterality'),
            'hippocampal_sclerosis': m.get('hippocampal_sclerosis'),
            'hippocampal_volume_asymmetry': m.get('hippocampal_volume_asymmetry'),
            't2_flair_signal': m.get('t2_flair_signal'),
            'enhancing': m.get('enhancing'),
            'classification': m.get('classification'),
            'classification_label': m.get('classification_label'),
            'radiologist_confidence': m.get('radiologist_confidence'),
            'protocol': m.get('protocol'),
        }

        eeg = latest_eeg.get(pid, {
            'predicted_label': None,
            'confidence': None,
            'signal_quality': None,
        })

        patient_list.append({
            'patient_id': pid,
            'name': pinfo.get('name'),
            'age': pinfo.get('age'),
            'gender': pinfo.get('gender'),
            'disease': pinfo.get('disease'),
            'mri': mri_detail,
            'eeg_analysis': eeg,
        })

    return {'patients': patient_list}


# ---------------------------------------------------------------------------
# definitions
# ---------------------------------------------------------------------------

def definitions():
    """Definitions tab for the Radiologist dashboard — neuroradiology
    concepts, MRI protocols, and ILAE references for epilepsy imaging."""
    return {
        'concepts': [
            {
                'name': 'MRI Protocol for Epilepsy',
                'description': 'Dedicated epilepsy MRI protocol includes 3T field strength, '
                               '3D T1-weighted volumetric (1 mm isotropic), coronal T2-weighted '
                               'perpendicular to hippocampal axis (2-3 mm), coronal and axial FLAIR, '
                               'and T2* or SWI for hemosiderin. Thin-cut coronal sequences are '
                               'essential for detecting hippocampal sclerosis and focal cortical '
                               'dysplasia. ILAE recommends the HARNESS-MRI protocol for standardized '
                               'epilepsy imaging (Bernasconi et al., 2019).',
            },
            {
                'name': 'Hippocampal Sclerosis (MTS)',
                'description': 'Mesial temporal sclerosis is the most common structural lesion '
                               'in temporal lobe epilepsy (60-70% of surgical TLE cases). MRI '
                               'features: hippocampal atrophy (volume loss), increased T2/FLAIR '
                               'signal, loss of internal architecture, and ipsilateral temporal '
                               'horn dilation. Volume asymmetry > 10-15% is considered significant. '
                               'Bilateral HS occurs in 10-20% of cases and complicates surgical '
                               'planning.',
            },
            {
                'name': 'Focal Cortical Dysplasia (FCD)',
                'description': 'Malformation of cortical development classified by ILAE (Blumcke '
                               'et al., 2011). FCD Type I: abnormal cortical layering. FCD Type II: '
                               'dysmorphic neurons (IIa) or balloon cells (IIb). MRI features: '
                               'cortical thickening, blurring of grey-white junction, T2/FLAIR '
                               'hyperintensity, and transmantle sign (FCD IIb). FCD II is more '
                               'readily detected on MRI; FCD I may be MRI-negative.',
            },
            {
                'name': 'Cavernoma (Cavernous Malformation)',
                'description': 'Low-flow vascular malformation with characteristic MRI appearance: '
                               '"popcorn" mixed-signal core on T1/T2 with hemosiderin rim (dark on '
                               'T2*/SWI). Epileptogenic due to hemosiderin deposition in surrounding '
                               'cortex. Surgical resection with surrounding hemosiderin rim yields '
                               '70-80% seizure freedom (Engel class I).',
            },
            {
                'name': 'Arteriovenous Malformation (AVM)',
                'description': 'High-flow vascular lesion with tangle of abnormal arteries and '
                               'veins without intervening capillary bed. MRI shows flow voids on '
                               'T1/T2, surrounding gliosis/hemosiderin. Epilepsy occurs in 20-40% '
                               'of AVM patients. Spetzler-Martin grading guides surgical risk. '
                               'MR angiography or DSA required for complete characterization.',
            },
            {
                'name': 'Encephalomalacia',
                'description': 'Post-injury brain softening/gliosis from prior insult (stroke, '
                               'trauma, infection, surgery). MRI shows CSF-signal cavity with '
                               'surrounding T2/FLAIR gliosis. Epileptogenic zone often at the '
                               'gliotic margin rather than the cavity itself. Common cause of '
                               'acquired focal epilepsy.',
            },
            {
                'name': 'Lesional vs Non-lesional Classification',
                'description': 'Lesional epilepsy has an identifiable structural abnormality on MRI '
                               '(HS, FCD, tumor, vascular malformation, etc.). Non-lesional (MRI-'
                               'negative) epilepsy has no visible structural cause on standard MRI. '
                               'Non-lesional cases have lower surgical seizure-freedom rates (40-50% '
                               'vs 60-80% for lesional). Advanced techniques (7T MRI, post-processing, '
                               'MAP/MELD) may reveal subtle lesions in previously MRI-negative cases.',
            },
            {
                'name': 'Hippocampal Volume Asymmetry',
                'description': 'Quantitative measure comparing left and right hippocampal volumes, '
                               'calculated as percentage difference or asymmetry index. Asymmetry '
                               '> 10-15% suggests unilateral hippocampal pathology. Volumetric '
                               'analysis using FreeSurfer or similar tools provides objective '
                               'measurement complementing visual assessment. Correlates with '
                               'histopathological severity of neuronal loss.',
            },
            {
                'name': 'T2/FLAIR Signal Changes',
                'description': 'Increased T2 and FLAIR signal indicates increased tissue water '
                               'content from gliosis, edema, or demyelination. In epilepsy context: '
                               'hippocampal T2 hyperintensity suggests sclerosis; cortical FLAIR '
                               'hyperintensity suggests FCD or low-grade tumor; perilesional signal '
                               'changes indicate gliotic epileptogenic zone. Signal quantification '
                               '(T2 relaxometry) improves detection sensitivity.',
            },
            {
                'name': 'Radiologist Confidence Levels',
                'description': 'Structured reporting confidence scale: High (>= 0.80) indicates '
                               'definite lesion identification with clear imaging features; Moderate '
                               '(0.50-0.79) indicates probable lesion with some ambiguity requiring '
                               'clinical correlation; Low (< 0.50) indicates equivocal findings '
                               'needing additional imaging (higher field strength, advanced sequences) '
                               'or second opinion. Inter-reader agreement improves with epilepsy-'
                               'specific MRI training.',
            },
            {
                'name': 'ILAE Recommendations for MRI in Epilepsy',
                'description': 'The International League Against Epilepsy recommends: (1) MRI for all '
                               'patients with new-onset epilepsy except confirmed idiopathic generalized '
                               'epilepsy; (2) Dedicated epilepsy protocol (not standard brain MRI); '
                               '(3) 3T preferred over 1.5T for lesion detection; (4) Expert '
                               'neuroradiologist review increases lesion detection by 20-30%; '
                               '(5) Repeat MRI if initial scan is negative and epilepsy is drug-'
                               'resistant. Key references: Bernasconi et al., Epilepsia 2019 '
                               '(HARNESS-MRI); Von Oertzen et al., JNNP 2002.',
            },
        ],
    }
