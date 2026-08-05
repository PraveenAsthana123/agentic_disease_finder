"""Phase 2 Multimodal Coverage Dashboard — AI Module.

Tracks cross-modality data coverage for the four Phase 2 modalities:
  - Patient Video   → camera_monitoring_sessions (78 rows, 27 patients)
  - Video-EEG       → eeg_acquisition (30 rows, 30 patients; recording_type∈video_eeg/LTM/ambulatory/routine)
  - MRI             → mri_findings (40 rows, 40 patients)
  - Neuropsych      → neuropsych (37 rows, 30 patients)

Reads from clinical.db EPAT001-EPAT040 cohort.
Phase 2 completes dataset_coverage.json phases[1] partial→built.
"""

import json
import os
import sqlite3
from collections import Counter, defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')


def _conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def _q(sql, params=()):
    if not os.path.exists(DB):
        return []
    conn = _conn()
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


def _pct(n, total):
    return round(n / total * 100, 1) if total else 0.0


def _avg(vals):
    return round(sum(vals) / len(vals), 2) if vals else 0.0


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _patient_sets():
    """Return a dict of patient_id sets per modality."""
    cam = set(r['patient_id'] for r in _q('SELECT patient_id FROM camera_monitoring_sessions'))
    eeg = set(r['patient_id'] for r in _q('SELECT patient_id FROM eeg_acquisition'))
    mri = set(r['patient_id'] for r in _q('SELECT patient_id FROM mri_findings'))
    nps = set(r['patient_id'] for r in _q('SELECT patient_id FROM neuropsych'))
    all_pts = set(r['patient_id'] for r in _q('SELECT patient_id FROM patients'))
    return {'cam': cam, 'eeg': eeg, 'mri': mri, 'nps': nps, 'all': all_pts}


# ---------------------------------------------------------------------------
# overview()
# ---------------------------------------------------------------------------

def overview():
    """KPI cards + coverage bar + co-occurrence summary."""
    sets = _patient_sets()
    cam, eeg, mri, nps, all_pts = sets['cam'], sets['eeg'], sets['mri'], sets['nps'], sets['all']
    total = len(all_pts)

    all4 = cam & eeg & mri & nps
    any3_sets = [
        cam & eeg & mri,
        cam & eeg & nps,
        cam & mri & nps,
        eeg & mri & nps,
    ]
    exactly3 = set().union(*any3_sets) - all4
    any2 = (cam & eeg) | (cam & mri) | (cam & nps) | (eeg & mri) | (eeg & nps) | (mri & nps)
    exactly2 = any2 - set().union(*any3_sets)
    exactly1 = (cam | eeg | mri | nps) - any2
    none_pts = all_pts - (cam | eeg | mri | nps)

    # Camera session stats
    cam_rows = _q('SELECT duration_hours, seizure_events, recording_quality FROM camera_monitoring_sessions')
    avg_dur = _avg([r['duration_hours'] for r in cam_rows if r['duration_hours']])
    total_seizure_ev = sum(r['seizure_events'] or 0 for r in cam_rows)
    quality_dist = Counter(r['recording_quality'] for r in cam_rows if r['recording_quality'])

    # EEG type distribution
    eeg_rows = _q('SELECT fields_json FROM eeg_acquisition')
    eeg_types = Counter()
    for r in eeg_rows:
        d = json.loads(r['fields_json']) if r['fields_json'] else {}
        eeg_types[d.get('recording_type', 'unknown')] += 1

    # Neuropsych scores
    np_rows = _q('SELECT fields_json FROM neuropsych')
    moca_vals, mmse_vals = [], []
    for r in np_rows:
        d = json.loads(r['fields_json']) if r['fields_json'] else {}
        if d.get('moca') is not None:
            moca_vals.append(d['moca'])
        if d.get('mmse') is not None:
            mmse_vals.append(d['mmse'])

    # MRI lesion stats
    mri_rows = _q('SELECT fields_json FROM mri_findings')
    lesion_types = Counter()
    for r in mri_rows:
        d = json.loads(r['fields_json']) if r['fields_json'] else {}
        lt = d.get('lesion_type', 'unknown')
        lesion_types[lt] += 1

    return {
        'available': True,
        'phase': 2,
        'phase_name': 'Multimodal (patient video + video-EEG + MRI + neuropsych)',
        'total_patients': total,
        # Per-modality coverage
        'modality_coverage': [
            {'modality': 'Patient Video',  'code': 'CAM', 'count': len(cam), 'pct': _pct(len(cam), total), 'source': 'camera_monitoring_sessions'},
            {'modality': 'Video-EEG',      'code': 'EEG', 'count': len(eeg), 'pct': _pct(len(eeg), total), 'source': 'eeg_acquisition'},
            {'modality': 'MRI',            'code': 'MRI', 'count': len(mri), 'pct': _pct(len(mri), total), 'source': 'mri_findings'},
            {'modality': 'Neuropsychology','code': 'NPS', 'count': len(nps), 'pct': _pct(len(nps), total), 'source': 'neuropsych'},
        ],
        # Cross-modality completeness
        'complete_phase2': len(all4),
        'complete_phase2_pct': _pct(len(all4), total),
        'three_modalities': len(exactly3),
        'two_modalities': len(exactly2),
        'one_modality': len(exactly1),
        'no_phase2_data': len(none_pts),
        # Video stats
        'video_sessions_total': len(_q('SELECT id FROM camera_monitoring_sessions')),
        'video_avg_duration_hours': avg_dur,
        'video_total_seizure_events': total_seizure_ev,
        'video_quality_distribution': dict(quality_dist.most_common()),
        # EEG stats
        'eeg_total_studies': len(eeg_rows),
        'eeg_type_distribution': dict(eeg_types.most_common()),
        # MRI stats
        'mri_total_studies': len(mri_rows),
        'mri_lesion_type_distribution': dict(lesion_types.most_common(8)),
        # Neuropsych stats
        'neuropsych_total_assessments': len(np_rows),
        'neuropsych_avg_moca': _avg(moca_vals),
        'neuropsych_avg_mmse': _avg(mmse_vals),
    }


# ---------------------------------------------------------------------------
# breakdown()
# ---------------------------------------------------------------------------

def breakdown():
    """Per-patient modality coverage matrix + co-occurrence table."""
    sets = _patient_sets()
    cam, eeg, mri, nps, all_pts = sets['cam'], sets['eeg'], sets['mri'], sets['nps'], sets['all']

    # Co-occurrence 2×2 pair table
    pairs = [
        ('CAM', 'EEG', len(cam & eeg)),
        ('CAM', 'MRI', len(cam & mri)),
        ('CAM', 'NPS', len(cam & nps)),
        ('EEG', 'MRI', len(eeg & mri)),
        ('EEG', 'NPS', len(eeg & nps)),
        ('MRI', 'NPS', len(mri & nps)),
    ]

    # Per-patient table (sorted EPAT* first)
    pat_list = sorted(all_pts, key=lambda p: (0 if p.startswith('EPAT') else 1, p))
    rows = []
    for pid in pat_list:
        has_cam = pid in cam
        has_eeg = pid in eeg
        has_mri = pid in mri
        has_nps = pid in nps
        count = sum([has_cam, has_eeg, has_mri, has_nps])
        rows.append({
            'patient_id': pid,
            'video': has_cam,
            'eeg': has_eeg,
            'mri': has_mri,
            'neuropsych': has_nps,
            'modality_count': count,
            'complete': count == 4,
            'tier': 'Complete' if count == 4 else ('3 of 4' if count == 3 else ('2 of 4' if count == 2 else ('1 of 4' if count == 1 else 'None'))),
        })

    # Monthly trend of video sessions (camera)
    cam_rows = _q("SELECT session_date, seizure_events, duration_hours FROM camera_monitoring_sessions ORDER BY session_date")
    monthly = defaultdict(lambda: {'sessions': 0, 'seizure_events': 0, 'total_hours': 0.0})
    for r in cam_rows:
        mo = (r['session_date'] or '')[:7]
        monthly[mo]['sessions'] += 1
        monthly[mo]['seizure_events'] += r['seizure_events'] or 0
        monthly[mo]['total_hours'] += r['duration_hours'] or 0.0
    trend = [{'month': k, **v} for k, v in sorted(monthly.items()) if k]

    # EEG type+duration breakdown
    eeg_rows = _q('SELECT patient_id, fields_json FROM eeg_acquisition')
    eeg_detail = []
    for r in eeg_rows:
        d = json.loads(r['fields_json']) if r['fields_json'] else {}
        eeg_detail.append({
            'patient_id': r['patient_id'],
            'recording_type': d.get('recording_type', 'unknown'),
            'duration_min': d.get('duration_min'),
            'sampling_rate': d.get('sampling_rate'),
            'montage': d.get('montage'),
        })

    # Neuropsych domain comparison
    np_rows = _q('SELECT patient_id, fields_json FROM neuropsych')
    np_summary = []
    for r in np_rows:
        d = json.loads(r['fields_json']) if r['fields_json'] else {}
        np_summary.append({
            'patient_id': r['patient_id'],
            'moca': d.get('moca'),
            'mmse': d.get('mmse'),
            'phq9': d.get('phq9'),
            'gad7': d.get('gad7'),
            'memory_index': d.get('memory_index'),
            'attention_index': d.get('attention_index'),
            'executive_index': d.get('executive_index'),
        })

    return {
        'available': True,
        'per_patient': rows,
        'cooccurrence_pairs': [{'a': a, 'b': b, 'count': c} for a, b, c in pairs],
        'completeness_distribution': {
            'Complete (4/4)': sum(1 for r in rows if r['modality_count'] == 4),
            '3 of 4': sum(1 for r in rows if r['modality_count'] == 3),
            '2 of 4': sum(1 for r in rows if r['modality_count'] == 2),
            '1 of 4': sum(1 for r in rows if r['modality_count'] == 1),
            'None': sum(1 for r in rows if r['modality_count'] == 0),
        },
        'video_monthly_trend': trend,
        'eeg_detail': eeg_detail,
        'neuropsych_summary': np_summary,
    }


# ---------------------------------------------------------------------------
# definitions()
# ---------------------------------------------------------------------------

def definitions():
    return {
        'available': True,
        'phase': 2,
        'modalities': [
            {
                'code': 'CAM',
                'name': 'Patient Video',
                'description': 'Camera-monitored sessions capturing clinical behavior, seizure events, movement, and response patterns during in-hospital or home monitoring.',
                'source_table': 'camera_monitoring_sessions',
                'key_fields': ['session_type', 'duration_hours', 'seizure_events', 'recording_quality'],
                'ai_potential': 'Computer vision, behavioral analysis, seizure detection from motion',
            },
            {
                'code': 'EEG',
                'name': 'Video-EEG / EEG Acquisition',
                'description': 'EEG study acquisitions including routine, ambulatory, long-term monitoring (LTM), and synchronized video-EEG studies.',
                'source_table': 'eeg_acquisition',
                'key_fields': ['recording_type', 'duration_min', 'sampling_rate', 'montage'],
                'ai_potential': 'Seizure detection, band-power analysis, spike detection, classification',
            },
            {
                'code': 'MRI',
                'name': 'MRI Findings',
                'description': 'Structural MRI findings including lesion type, location, laterality, hippocampal sclerosis, and T2/FLAIR signal abnormalities.',
                'source_table': 'mri_findings',
                'key_fields': ['lesion_type', 'lesion_location', 'laterality', 'hippocampal_sclerosis'],
                'ai_potential': 'Multimodal fusion, lesion classification, surgical candidacy scoring',
            },
            {
                'code': 'NPS',
                'name': 'Neuropsychology',
                'description': 'Neuropsychological battery results including MoCA, MMSE, PHQ-9, GAD-7, and domain indices (memory, attention, executive, language, processing speed).',
                'source_table': 'neuropsych',
                'key_fields': ['moca', 'mmse', 'phq9', 'gad7', 'memory_index', 'attention_index', 'executive_index'],
                'ai_potential': 'Cognitive decline tracking, comorbidity profiling, QoL prediction',
            },
        ],
        'phase_context': {
            'phase': 2,
            'name': 'Multimodal (patient video + video-EEG + MRI + neuropsych)',
            'predecessor': 'Phase 1 — Core EEG + diagnosis + demographics + medication + outcome',
            'successor': 'Phase 3 — Autonomic (Holter, RR variation, SSR, ABPM)',
            'goal': 'Establish cross-modality concordance and enable multimodal AI fusion for improved seizure prediction and subtype classification.',
        },
        'completeness_tiers': [
            {'tier': 'Complete (4/4)', 'description': 'Patient has all four Phase 2 modalities recorded'},
            {'tier': '3 of 4', 'description': 'One modality missing — partial Phase 2 coverage'},
            {'tier': '2 of 4', 'description': 'Two modalities missing — limited multimodal value'},
            {'tier': '1 of 4', 'description': 'Only one modality — insufficient for fusion'},
            {'tier': 'None', 'description': 'No Phase 2 data recorded for this patient'},
        ],
    }
