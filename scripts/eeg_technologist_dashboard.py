"""EEG Technologist Dashboard — recording quality metrics, channel impedance/SNR
assessment, artifact burden analysis, activation procedure coverage, and recording
conditions. Sourced from eeg_acquisition, channel_quality, artifact_annotations,
and recording_conditions tables in clinical.db.

Consultant role: EEG Acquisition Advisor
Objective: Validate technical quality of EEG recordings
"""

import json
import sqlite3
import os
import statistics
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')


def _db_query(sql, params=()):
    if not os.path.exists(DB):
        return []
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    finally:
        con.close()


def _safe_json(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _avg(values):
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _pct(part, total):
    if not total:
        return 0.0
    return round(100 * part / total, 1)


# ── Data loaders ────────────────────────────────────────────────────────────

def _load_acquisitions():
    rows = _db_query(
        "SELECT id, patient_id, fields_json, created_at FROM eeg_acquisition ORDER BY id"
    )
    results = []
    for r in rows:
        f = _safe_json(r.get('fields_json'))
        results.append({
            'id': r['id'],
            'patient_id': r['patient_id'],
            'recording_type': f.get('recording_type', 'unknown'),
            'duration_min': f.get('duration_min', 0),
            'sampling_rate': f.get('sampling_rate', 256),
            'montage': f.get('montage', 'unknown'),
            'electrode_system': f.get('electrode_system', '10-20'),
            'technician_notes': f.get('technician_notes', ''),
            'study_date': f.get('study_date', ''),
            'created_at': r.get('created_at', ''),
        })
    return results


def _load_channel_quality():
    rows = _db_query(
        "SELECT id, patient_id, fields_json, created_at FROM channel_quality ORDER BY id"
    )
    results = []
    for r in rows:
        f = _safe_json(r.get('fields_json'))
        channels = f.get('channels', [])
        results.append({
            'id': r['id'],
            'patient_id': r['patient_id'],
            'channels': channels,
            'created_at': r.get('created_at', ''),
        })
    return results


def _load_artifacts():
    rows = _db_query(
        "SELECT id, patient_id, fields_json, created_at FROM artifact_annotations ORDER BY id"
    )
    results = []
    for r in rows:
        f = _safe_json(r.get('fields_json'))
        results.append({
            'id': r['id'],
            'patient_id': r['patient_id'],
            'artifact_type': f.get('artifact_type', 'unknown'),
            'channel': f.get('channel', ''),
            'start_time_min': f.get('start_time_min', 0),
            'duration_sec': f.get('duration_sec', 0),
            'severity': f.get('severity', 'unknown'),
            'created_at': r.get('created_at', ''),
        })
    return results


def _load_recording_conditions():
    rows = _db_query(
        "SELECT id, patient_id, fields_json, created_at FROM recording_conditions ORDER BY id"
    )
    results = []
    for r in rows:
        f = _safe_json(r.get('fields_json'))
        results.append({
            'id': r['id'],
            'patient_id': r['patient_id'],
            'eyes_open': f.get('eyes_open', False),
            'hyperventilation': f.get('hyperventilation', False),
            'photic_stimulation': f.get('photic_stimulation', False),
            'sleep_recorded': f.get('sleep_recorded', False),
            'patient_state': f.get('patient_state', 'unknown'),
            'cooperation': f.get('cooperation', 'unknown'),
            'created_at': r.get('created_at', ''),
        })
    return results


# ── Endpoints ────────────────────────────────────────────────────────────────

def overview():
    """KPI summary + recording quality + artifact burden overview."""
    acqs = _load_acquisitions()
    cqs = _load_channel_quality()
    arts = _load_artifacts()
    rcs = _load_recording_conditions()

    # Basic KPIs
    n_recordings = len(acqs)
    n_patients = len(set(a['patient_id'] for a in acqs))
    n_artifact_events = len(arts)

    # Channel quality aggregate
    all_channels = []
    for cq in cqs:
        all_channels.extend(cq['channels'])
    n_channels_total = len(all_channels)
    n_good = sum(1 for c in all_channels if c.get('quality_grade') == 'Good')
    channel_usability_pct = _pct(n_good, n_channels_total)

    # SNR stats
    snrs = [c.get('snr_db', 0) for c in all_channels if c.get('snr_db')]
    avg_snr = _avg(snrs)

    # Impedance good rate
    n_imp_good = sum(1 for c in all_channels if c.get('impedance_grade') == 'Good')
    imp_good_pct = _pct(n_imp_good, n_channels_total)

    # Duration stats
    durations = [a['duration_min'] for a in acqs if a['duration_min'] > 0]
    avg_duration = _avg(durations)

    # Recording type distribution
    rec_type_dist = dict(Counter(a['recording_type'] for a in acqs))

    # Montage distribution
    montage_dist = dict(Counter(a['montage'] for a in acqs))

    # Sampling rate distribution
    sr_dist = dict(Counter(a['sampling_rate'] for a in acqs))

    # Channel quality grade distribution
    quality_grade_dist = dict(Counter(c.get('quality_grade', 'Unknown') for c in all_channels))

    # Impedance grade distribution
    imp_grade_dist = dict(Counter(c.get('impedance_grade', 'Unknown') for c in all_channels))

    # Artifact type distribution
    artifact_type_dist = dict(Counter(a['artifact_type'] for a in arts))

    # Artifact severity distribution
    artifact_severity_dist = dict(Counter(a['severity'] for a in arts))

    # Activation procedures (recording conditions)
    hv_count = sum(1 for r in rcs if r['hyperventilation'])
    photic_count = sum(1 for r in rcs if r['photic_stimulation'])
    sleep_count = sum(1 for r in rcs if r['sleep_recorded'])
    eyes_open_count = sum(1 for r in rcs if r['eyes_open'])
    n_rc = len(rcs) or 1

    activation_rates = {
        'Eyes Open': round(100 * eyes_open_count / n_rc, 1),
        'Hyperventilation': round(100 * hv_count / n_rc, 1),
        'Photic Stimulation': round(100 * photic_count / n_rc, 1),
        'Sleep Recorded': round(100 * sleep_count / n_rc, 1),
    }

    # Patient cooperation distribution
    cooperation_dist = dict(Counter(r['cooperation'] for r in rcs))

    # Patient state distribution
    state_dist = dict(Counter(r['patient_state'] for r in rcs))

    # Per-recording summary (last 10)
    recording_log = [
        {
            'patient_id': a['patient_id'],
            'recording_type': a['recording_type'],
            'duration_min': a['duration_min'],
            'sampling_rate': a['sampling_rate'],
            'montage': a['montage'],
            'study_date': a['study_date'],
        }
        for a in acqs[-10:]
    ]

    return {
        'kpis': [
            {'label': 'Total Recordings', 'value': n_recordings, 'unit': 'studies'},
            {'label': 'Patients', 'value': n_patients, 'unit': 'patients'},
            {'label': 'Channel Usability', 'value': channel_usability_pct, 'unit': '%'},
            {'label': 'Avg SNR', 'value': avg_snr, 'unit': 'dB'},
            {'label': 'Impedance Pass Rate', 'value': imp_good_pct, 'unit': '%'},
            {'label': 'Artifact Events', 'value': n_artifact_events, 'unit': 'annotations'},
            {'label': 'Avg Duration', 'value': avg_duration, 'unit': 'min'},
        ],
        'recording_type_distribution': rec_type_dist,
        'montage_distribution': montage_dist,
        'sampling_rate_distribution': {str(k) + ' Hz': v for k, v in sr_dist.items()},
        'channel_quality_grade_distribution': quality_grade_dist,
        'impedance_grade_distribution': imp_grade_dist,
        'artifact_type_distribution': artifact_type_dist,
        'artifact_severity_distribution': artifact_severity_dist,
        'activation_procedure_rates': activation_rates,
        'cooperation_distribution': cooperation_dist,
        'patient_state_distribution': state_dist,
        'recent_recordings': recording_log,
    }


def breakdown():
    """Per-patient channel quality + artifact burden detail."""
    acqs = _load_acquisitions()
    cqs = _load_channel_quality()
    arts = _load_artifacts()
    rcs = _load_recording_conditions()

    # Index by patient
    acq_by_patient = {}
    for a in acqs:
        acq_by_patient.setdefault(a['patient_id'], []).append(a)

    cq_by_patient = {}
    for cq in cqs:
        cq_by_patient.setdefault(cq['patient_id'], []).append(cq)

    art_by_patient = {}
    for art in arts:
        art_by_patient.setdefault(art['patient_id'], []).append(art)

    rc_by_patient = {}
    for rc in rcs:
        rc_by_patient.setdefault(rc['patient_id'], []).append(rc)

    # All patients with any EEG data
    all_patients = sorted(set(
        list(acq_by_patient.keys()) + list(cq_by_patient.keys())
    ))

    per_patient = []
    for pid in all_patients:
        patient_acqs = acq_by_patient.get(pid, [])
        patient_cqs = cq_by_patient.get(pid, [])
        patient_arts = art_by_patient.get(pid, [])
        patient_rcs = rc_by_patient.get(pid, [])

        # Channel usability for this patient
        ch_flat = []
        for cq in patient_cqs:
            ch_flat.extend(cq['channels'])
        n_ch = len(ch_flat)
        n_good = sum(1 for c in ch_flat if c.get('quality_grade') == 'Good')
        usability_pct = _pct(n_good, n_ch) if n_ch else None

        # Avg SNR
        snrs = [c.get('snr_db', 0) for c in ch_flat if c.get('snr_db')]
        avg_snr = _avg(snrs)

        # Impedance pass rate
        n_imp_good = sum(1 for c in ch_flat if c.get('impedance_grade') == 'Good')
        imp_pct = _pct(n_imp_good, n_ch) if n_ch else None

        # Artifact count
        n_arts = len(patient_arts)
        art_types = list(set(a['artifact_type'] for a in patient_arts))

        # Recording type
        rec_type = patient_acqs[0]['recording_type'] if patient_acqs else '—'
        duration = patient_acqs[0]['duration_min'] if patient_acqs else None

        # HV / photic
        hv = any(r['hyperventilation'] for r in patient_rcs)
        photic = any(r['photic_stimulation'] for r in patient_rcs)

        per_patient.append({
            'patient_id': pid,
            'recording_type': rec_type,
            'duration_min': duration,
            'channel_usability_pct': usability_pct,
            'avg_snr_db': avg_snr if snrs else None,
            'impedance_pass_pct': imp_pct,
            'artifact_count': n_arts,
            'artifact_types': ', '.join(art_types) if art_types else '—',
            'hyperventilation': hv,
            'photic_stimulation': photic,
        })

    # Artifact channel heat map (top 10 channels by count)
    art_channel_counts = Counter(a['channel'] for a in arts if a.get('channel'))
    top_artifact_channels = [
        {'channel': ch, 'count': cnt}
        for ch, cnt in art_channel_counts.most_common(10)
    ]

    # Artifact type x severity matrix
    art_matrix = {}
    for a in arts:
        at = a['artifact_type']
        sv = a['severity']
        art_matrix.setdefault(at, {})
        art_matrix[at][sv] = art_matrix[at].get(sv, 0) + 1

    # SNR distribution histogram
    all_channels = []
    for cq in cqs:
        all_channels.extend(cq['channels'])
    snrs_all = [c.get('snr_db', 0) for c in all_channels if c.get('snr_db')]
    snr_hist = {
        '<10 dB': sum(1 for s in snrs_all if s < 10),
        '10-20 dB': sum(1 for s in snrs_all if 10 <= s < 20),
        '20-30 dB': sum(1 for s in snrs_all if 20 <= s < 30),
        '≥30 dB': sum(1 for s in snrs_all if s >= 30),
    }

    return {
        'per_patient': per_patient,
        'top_artifact_channels': top_artifact_channels,
        'artifact_type_severity_matrix': art_matrix,
        'snr_histogram': snr_hist,
        'total_patients': len(all_patients),
        'total_artifact_annotations': len(arts),
    }


def definitions():
    """EEG Technologist key concepts, quality standards, and AI workflow context."""
    return {
        'role': 'EEG Technologist / EEG Acquisition Advisor',
        'objective': 'Validate technical quality of EEG recordings and ensure acquisition standards meet clinical and AI training requirements.',
        'concepts': [
            {
                'term': 'Impedance',
                'definition': 'Resistance to AC current at electrode-scalp interface. Clinically acceptable: <5 kΩ (Good), 5-10 kΩ (Fair), >10 kΩ (Poor). High impedance increases artifact susceptibility.',
                'standard': 'ACNS Guideline 1 (2016)',
            },
            {
                'term': 'Signal-to-Noise Ratio (SNR)',
                'definition': 'Ratio of desired EEG signal power to background noise. Higher SNR (>20 dB) indicates cleaner recordings suitable for both clinical interpretation and AI feature extraction.',
                'standard': 'IEEE 802.11 / clinical EEG lab SOP',
            },
            {
                'term': 'Montage',
                'definition': 'The arrangement of electrode pairs for differential amplification. Bipolar (chain), Referential (common reference), and Average reference montages affect waveform morphology and epileptiform detection sensitivity.',
                'standard': 'IFCN 10-20 electrode system',
            },
            {
                'term': 'Hyperventilation (HV)',
                'definition': '3-minute over-breathing activation procedure. Induces slow-wave activity and may provoke absence seizures or focal slowing. Standard activation in routine and video-EEG protocols.',
                'standard': 'ACNS Guideline 5 (2016); ILAE recommendation',
            },
            {
                'term': 'Photic Stimulation',
                'definition': 'Intermittent photic stimulation (IPS) at 1–60 Hz to elicit photoparoxysmal responses. Essential for detecting photosensitive epilepsy.',
                'standard': 'ACNS Guideline 5; EEG frequency range 1-60 Hz',
            },
            {
                'term': 'Artifact Annotation',
                'definition': 'Marking of non-cerebral signals: eye-blink (delta wave at Fp1/Fp2), muscle (high-frequency EMG), movement, electrode pop, ECG, and sweat artifacts. Critical for clean AI training data.',
                'standard': 'ACNS Artifact Classification; Tatum et al. 2011',
            },
            {
                'term': 'Channel Usability',
                'definition': 'Percentage of EEG channels meeting quality thresholds (Good grade). Target >80% usability for diagnostic-grade recordings; <60% may require re-recording.',
                'standard': 'Lab SOP; ACNS Quality Standards',
            },
            {
                'term': 'Recording Type',
                'definition': 'Routine EEG (20-40 min awake), Video-EEG (inpatient telemetry with camera), Ambulatory EEG (24-72h outpatient Holter-style), Long-Term Monitoring (LTM, days to weeks for surgical evaluation).',
                'standard': 'ACNS Guideline 6 (2016)',
            },
            {
                'term': 'Electrode System',
                'definition': '10-20 international system: 19 standard scalp electrodes placed at 10% and 20% intervals between skull landmarks (nasion, inion, preauricular points). Extended systems (10-10, 10-5) for dense arrays.',
                'standard': 'Jasper 1958; IFCN 1999 revised system',
            },
            {
                'term': 'AI Data Quality Impact',
                'definition': 'Poor-quality EEG recordings degrade AI model training. Artifact-contaminated channels reduce feature validity; inconsistent montages require normalization. EEG Technologist sign-off is a prerequisite for including recordings in training datasets.',
                'standard': 'FDA AI/ML SaMD Action Plan 2021; EU MDR 2017/745',
            },
        ],
        'quality_thresholds': {
            'impedance_good_kohm': '<5',
            'impedance_fair_kohm': '5-10',
            'impedance_poor_kohm': '>10',
            'snr_excellent_db': '>30',
            'snr_good_db': '20-30',
            'snr_fair_db': '10-20',
            'snr_poor_db': '<10',
            'channel_usability_target': '>80%',
            'artifact_burden_acceptable': '<20% of recording',
        },
        'activation_protocols': [
            {'procedure': 'Eyes Open / Eyes Closed', 'purpose': 'Alpha rhythm reactivity', 'duration': '30 sec each'},
            {'procedure': 'Hyperventilation', 'purpose': 'Provoke absence / focal slowing', 'duration': '3 min'},
            {'procedure': 'Photic Stimulation', 'purpose': 'Photosensitivity detection', 'duration': '1-60 Hz sweep'},
            {'procedure': 'Sleep Recording', 'purpose': 'Sleep-activated epileptiform activity', 'duration': 'NREM stage 2+'},
        ],
        'references': [
            'ACNS (2016). Guideline 1: Minimum Technical Requirements for EEG.',
            'ACNS (2016). Guideline 5: Standard and Minimum Technical Specifications for EEG Recording.',
            'Tatum WO et al. (2011). Artifact: Recorded with Ambulatory EEG. Neurology 75(11).',
            'Jasper HH (1958). The ten-twenty electrode system of the International Federation. EEG Clin Neurophysiol 10:371-375.',
            'IFCN (1999). A proposal for an EEG terminology by the Terminology and Nosology Subcommittee. Clin Neurophysiol 110(9).',
        ],
    }
