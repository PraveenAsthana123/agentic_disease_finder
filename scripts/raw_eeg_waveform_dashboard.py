"""Raw EEG Waveform Dashboard — real data from clinical.db.

Sources:
- eeg_acquisition (30 rows): recording type, duration, sampling rate, montage,
  electrode system, technician notes, study date
- channel_quality (30 rows): per-channel impedance (kOhm) and SNR (dB) + quality grades
- artifact_annotations (169 rows): artifact type, channel, start time (min), duration (sec),
  severity (mild/moderate/severe)
- recording_conditions (30 rows): eyes-open/HV/photic/sleep, patient state, cooperation
- seizure_metadata (71 rows): EEG pattern, lateralization, onset zone

Clinical context:
- Raw EEG waveform quality is gated by electrode impedance (target <5 kΩ) and SNR (>20 dB)
- Artifact burden directly impacts AI model accuracy; high-artifact recordings must be flagged
  for re-recording or manual review before automated seizure detection
- ACNS standards require 10-20 system, ≥19 channels, ≥256 Hz sampling
- Hyperventilation (HV) and photic stimulation (PS) are activation procedures that unmask
  absence seizures and photoparoxysmal responses
"""

import sqlite3
import json
import os
import math
from collections import Counter, defaultdict

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')

# Standard 10-20 channel order for display
CHANNEL_ORDER = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2',
]


def _conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def _load_acquisitions():
    conn = _conn()
    rows = [dict(r) for r in conn.execute(
        'SELECT patient_id, fields_json, created_at FROM eeg_acquisition'
    ).fetchall()]
    conn.close()
    parsed = []
    for r in rows:
        try:
            fields = json.loads(r['fields_json'])
        except Exception:
            fields = {}
        fields['patient_id'] = r['patient_id']
        parsed.append(fields)
    return parsed


def _load_channel_quality():
    """Returns dict: patient_id → list of channel dicts."""
    conn = _conn()
    rows = [dict(r) for r in conn.execute(
        'SELECT patient_id, fields_json FROM channel_quality'
    ).fetchall()]
    conn.close()
    result = {}
    for r in rows:
        try:
            fields = json.loads(r['fields_json'])
        except Exception:
            fields = {}
        channels = fields.get('channels', [])
        result[r['patient_id']] = channels
    return result


def _load_artifacts():
    """Returns dict: patient_id → list of artifact dicts."""
    conn = _conn()
    rows = [dict(r) for r in conn.execute(
        'SELECT patient_id, fields_json FROM artifact_annotations'
    ).fetchall()]
    conn.close()
    result = defaultdict(list)
    for r in rows:
        try:
            fields = json.loads(r['fields_json'])
        except Exception:
            fields = {}
        result[r['patient_id']].append(fields)
    return dict(result)


def _load_recording_conditions():
    """Returns dict: patient_id → condition dict."""
    conn = _conn()
    rows = [dict(r) for r in conn.execute(
        'SELECT patient_id, fields_json FROM recording_conditions'
    ).fetchall()]
    conn.close()
    result = {}
    for r in rows:
        try:
            fields = json.loads(r['fields_json'])
        except Exception:
            fields = {}
        result[r['patient_id']] = fields
    return result


def _load_seizure_metadata():
    """Returns dict: patient_id → first seizure metadata dict."""
    conn = _conn()
    rows = [dict(r) for r in conn.execute(
        'SELECT patient_id, fields_json FROM seizure_metadata'
    ).fetchall()]
    conn.close()
    result = {}
    for r in rows:
        pid = r['patient_id']
        if pid not in result:
            try:
                result[pid] = json.loads(r['fields_json'])
            except Exception:
                result[pid] = {}
    return result


def _channel_grade_color(grade):
    return {'Good': 'success', 'Fair': 'warning', 'Poor': 'danger'}.get(grade, 'secondary')


# ── Overview ──────────────────────────────────────────────────────────────────

def raw_eeg_overview():
    acquisitions = _load_acquisitions()
    channel_quality = _load_channel_quality()
    artifacts = _load_artifacts()
    conditions = _load_recording_conditions()

    total_recordings = len(acquisitions)
    unique_patients = len(set(a['patient_id'] for a in acquisitions))

    # Duration stats
    durations = [a.get('duration_min') for a in acquisitions if a.get('duration_min')]
    avg_duration = round(sum(durations) / len(durations)) if durations else None
    total_hours = round(sum(durations) / 60, 1) if durations else None

    # Sampling rate distribution
    sr_dist = Counter(a.get('sampling_rate', 'unknown') for a in acquisitions)
    sr_rows = [{'sampling_rate': k, 'count': v} for k, v in sorted(sr_dist.items(), key=lambda x: -x[1])]

    # Recording type distribution
    rt_dist = Counter(a.get('recording_type', 'unknown') for a in acquisitions)
    rt_rows = [{'type': k, 'count': v} for k, v in sorted(rt_dist.items(), key=lambda x: -x[1])]

    # Montage distribution
    mt_dist = Counter(a.get('montage', 'unknown') for a in acquisitions)
    mt_rows = [{'montage': k, 'count': v} for k, v in sorted(mt_dist.items(), key=lambda x: -x[1])]

    # Channel quality — aggregate across all patients
    all_channels = []
    for pid, chs in channel_quality.items():
        all_channels.extend(chs)

    total_channels = len(all_channels)
    quality_dist = Counter(c.get('quality_grade', 'unknown') for c in all_channels)
    quality_rows = [
        {'grade': g, 'count': quality_dist.get(g, 0),
         'pct': round(quality_dist.get(g, 0) / total_channels * 100, 1) if total_channels else 0,
         'color': _channel_grade_color(g)}
        for g in ['Good', 'Fair', 'Poor']
    ]

    impedance_dist = Counter(c.get('impedance_grade', 'unknown') for c in all_channels)
    impedance_rows = [
        {'grade': g, 'count': impedance_dist.get(g, 0),
         'pct': round(impedance_dist.get(g, 0) / total_channels * 100, 1) if total_channels else 0,
         'color': _channel_grade_color(g)}
        for g in ['Good', 'Fair', 'Poor']
    ]

    # Average SNR and impedance across all channels
    snrs = [c.get('snr_db') for c in all_channels if c.get('snr_db') is not None]
    imps = [c.get('impedance_kohm') for c in all_channels if c.get('impedance_kohm') is not None]
    avg_snr = round(sum(snrs) / len(snrs), 1) if snrs else None
    avg_impedance = round(sum(imps) / len(imps), 1) if imps else None

    # SNR by channel name (mean across patients)
    snr_by_channel = defaultdict(list)
    for c in all_channels:
        if c.get('channel') and c.get('snr_db') is not None:
            snr_by_channel[c['channel']].append(c['snr_db'])
    snr_channel_rows = [
        {
            'channel': ch,
            'avg_snr': round(sum(v) / len(v), 1),
            'n': len(v),
        }
        for ch, v in snr_by_channel.items()
    ]
    # Sort by canonical order
    snr_channel_rows.sort(
        key=lambda x: CHANNEL_ORDER.index(x['channel']) if x['channel'] in CHANNEL_ORDER else 99
    )

    # Artifact burden summary
    all_artifacts = []
    for pid_arts in artifacts.values():
        all_artifacts.extend(pid_arts)
    total_artifacts = len(all_artifacts)
    artifact_type_dist = Counter(a.get('artifact_type', 'unknown') for a in all_artifacts)
    artifact_type_rows = [{'type': k, 'count': v} for k, v in sorted(artifact_type_dist.items(), key=lambda x: -x[1])]

    artifact_severity_dist = Counter(a.get('severity', 'unknown') for a in all_artifacts)
    artifact_severity_rows = [
        {'severity': k, 'count': v,
         'color': {'mild': 'warning', 'moderate': 'orange', 'severe': 'danger'}.get(k, 'secondary')}
        for k, v in sorted(artifact_severity_dist.items(), key=lambda x: -x[1])
    ]

    # Total artifact duration
    total_artifact_sec = sum(a.get('duration_sec', 0) for a in all_artifacts)
    avg_artifacts_per_recording = round(total_artifacts / total_recordings, 1) if total_recordings else 0

    # Activation procedure coverage
    hv_count = sum(1 for c in conditions.values() if c.get('hyperventilation'))
    ps_count = sum(1 for c in conditions.values() if c.get('photic_stimulation'))
    sleep_count = sum(1 for c in conditions.values() if c.get('sleep_recorded'))
    eyes_open_count = sum(1 for c in conditions.values() if c.get('eyes_open'))
    n_cond = len(conditions)

    patient_state_dist = Counter(c.get('patient_state', 'unknown') for c in conditions.values())
    patient_state_rows = [{'state': k, 'count': v} for k, v in sorted(patient_state_dist.items(), key=lambda x: -x[1])]

    # Artifact-by-channel distribution
    artifact_channel_dist = Counter(a.get('channel') for a in all_artifacts if a.get('channel'))
    artifact_channel_rows = [
        {'channel': ch, 'count': cnt}
        for ch, cnt in sorted(artifact_channel_dist.items(), key=lambda x: -x[1])
    ][:15]

    return {
        'kpis': {
            'total_recordings': total_recordings,
            'unique_patients': unique_patients,
            'avg_duration_min': avg_duration,
            'total_eeg_hours': total_hours,
            'avg_channel_snr_db': avg_snr,
            'avg_impedance_kohm': avg_impedance,
            'total_artifacts': total_artifacts,
            'avg_artifacts_per_recording': avg_artifacts_per_recording,
            'total_artifact_sec': round(total_artifact_sec),
        },
        'recording_type_distribution': rt_rows,
        'montage_distribution': mt_rows,
        'sampling_rate_distribution': sr_rows,
        'channel_quality_distribution': quality_rows,
        'channel_impedance_distribution': impedance_rows,
        'snr_by_channel': snr_channel_rows,
        'artifact_type_distribution': artifact_type_rows,
        'artifact_severity_distribution': artifact_severity_rows,
        'artifact_channel_distribution': artifact_channel_rows,
        'activation_procedures': {
            'n_recordings': n_cond,
            'hyperventilation': {'count': hv_count, 'pct': round(hv_count / n_cond * 100) if n_cond else 0},
            'photic_stimulation': {'count': ps_count, 'pct': round(ps_count / n_cond * 100) if n_cond else 0},
            'sleep_recorded': {'count': sleep_count, 'pct': round(sleep_count / n_cond * 100) if n_cond else 0},
            'eyes_open': {'count': eyes_open_count, 'pct': round(eyes_open_count / n_cond * 100) if n_cond else 0},
        },
        'patient_state_distribution': patient_state_rows,
    }


# ── Breakdown ─────────────────────────────────────────────────────────────────

def raw_eeg_breakdown():
    acquisitions = _load_acquisitions()
    channel_quality = _load_channel_quality()
    artifacts = _load_artifacts()
    conditions = _load_recording_conditions()
    seizure_meta = _load_seizure_metadata()

    # Per-patient recording profiles
    patient_profiles = []
    for acq in sorted(acquisitions, key=lambda x: x['patient_id']):
        pid = acq['patient_id']
        chs = channel_quality.get(pid, [])
        arts = artifacts.get(pid, [])
        cond = conditions.get(pid, {})
        meta = seizure_meta.get(pid, {})

        # Channel quality summary
        good_n = sum(1 for c in chs if c.get('quality_grade') == 'Good')
        fair_n = sum(1 for c in chs if c.get('quality_grade') == 'Fair')
        poor_n = sum(1 for c in chs if c.get('quality_grade') == 'Poor')
        total_ch = len(chs)

        # Mean SNR and impedance
        snrs = [c['snr_db'] for c in chs if c.get('snr_db') is not None]
        imps = [c['impedance_kohm'] for c in chs if c.get('impedance_kohm') is not None]
        mean_snr = round(sum(snrs) / len(snrs), 1) if snrs else None
        mean_imp = round(sum(imps) / len(imps), 1) if imps else None

        # Poor channels list
        poor_channels = [c['channel'] for c in chs if c.get('quality_grade') == 'Poor']

        # Artifact summary
        artifact_count = len(arts)
        artifact_types = list(set(a.get('artifact_type', '') for a in arts if a.get('artifact_type')))
        severe_arts = [a for a in arts if a.get('severity') == 'severe']

        # Artifact burden in seconds
        artifact_sec = sum(a.get('duration_sec', 0) for a in arts)

        # EEG pattern from seizure metadata
        eeg_pattern = meta.get('eeg_pattern', '')
        onset_zone = meta.get('onset_zone', '')

        # Recording quality score (0–100)
        quality_score = 0
        if total_ch > 0:
            quality_score += round(good_n / total_ch * 40)  # 40 pts for channel quality
        if mean_snr is not None:
            quality_score += min(30, round(mean_snr / 35 * 30))  # 30 pts for SNR
        if mean_imp is not None:
            quality_score += max(0, min(20, round((20 - mean_imp) / 20 * 20)))  # 20 pts for impedance
        artifact_burden_pct = artifact_sec / (acq.get('duration_min', 1) * 60) * 100 if acq.get('duration_min') else 0
        quality_score += max(0, min(10, round((100 - artifact_burden_pct) / 100 * 10)))  # 10 pts for low artifacts
        quality_score = min(100, quality_score)

        tier = 'Excellent' if quality_score >= 85 else 'Good' if quality_score >= 70 else 'Fair' if quality_score >= 50 else 'Poor'

        patient_profiles.append({
            'patient_id': pid,
            'study_date': acq.get('study_date', ''),
            'recording_type': acq.get('recording_type', ''),
            'duration_min': acq.get('duration_min'),
            'sampling_rate': acq.get('sampling_rate'),
            'montage': acq.get('montage', ''),
            'electrode_system': acq.get('electrode_system', ''),
            'total_channels': total_ch,
            'good_channels': good_n,
            'fair_channels': fair_n,
            'poor_channels': poor_n,
            'poor_channel_names': poor_channels,
            'mean_snr_db': mean_snr,
            'mean_impedance_kohm': mean_imp,
            'artifact_count': artifact_count,
            'artifact_sec': round(artifact_sec),
            'artifact_types': artifact_types,
            'severe_artifacts': len(severe_arts),
            'eeg_pattern': eeg_pattern,
            'onset_zone': onset_zone,
            'hyperventilation': cond.get('hyperventilation', False),
            'photic_stimulation': cond.get('photic_stimulation', False),
            'sleep_recorded': cond.get('sleep_recorded', False),
            'patient_state': cond.get('patient_state', ''),
            'technician_notes': acq.get('technician_notes', ''),
            'quality_score': quality_score,
            'quality_tier': tier,
        })

    # Sort by quality score ascending (worst first)
    patient_profiles.sort(key=lambda x: x['quality_score'])

    # Per-patient channel quality detail (first patient only for deep drill)
    # Return all channel data per patient for drill-down
    channel_details = {}
    for pid, chs in channel_quality.items():
        sorted_chs = sorted(
            chs,
            key=lambda c: CHANNEL_ORDER.index(c.get('channel', '')) if c.get('channel') in CHANNEL_ORDER else 99
        )
        channel_details[pid] = sorted_chs

    # Artifact timeline per patient (first 10 patients, sorted by time)
    artifact_timelines = {}
    for pid, arts in artifacts.items():
        timeline = sorted(arts, key=lambda a: a.get('start_time_min', 0))
        artifact_timelines[pid] = timeline

    return {
        'patient_profiles': patient_profiles,
        'channel_details': channel_details,
        'artifact_timelines': artifact_timelines,
    }


# ── Definitions ───────────────────────────────────────────────────────────────

def raw_eeg_definitions():
    return {
        'title': 'Raw EEG Waveform Dashboard — Definitions & Clinical Reference',
        'description': (
            'The Raw EEG Waveform Dashboard aggregates EEG acquisition metadata, channel-level '
            'quality metrics, artifact annotations, and recording conditions across 30 recordings '
            '(30 patients) to support AI training data curation and clinical quality assurance. '
            'All data derives from the clinical.db eeg_acquisition, channel_quality, '
            'artifact_annotations, and recording_conditions tables.'
        ),
        'quality_tiers': [
            {
                'tier': 'Excellent',
                'score_range': '≥ 85',
                'color': 'success',
                'description': 'High SNR (>25 dB avg), low impedance (<5 kΩ avg), minimal artifacts. '
                               'Suitable for AI training without preprocessing.',
            },
            {
                'tier': 'Good',
                'score_range': '70–84',
                'color': 'primary',
                'description': 'Acceptable SNR (20–25 dB avg), moderate impedance, few artifacts. '
                               'Suitable after standard artifact rejection.',
            },
            {
                'tier': 'Fair',
                'score_range': '50–69',
                'color': 'warning',
                'description': 'Borderline SNR or impedance. Some channels require rejection. '
                               'Requires manual review before inclusion in AI datasets.',
            },
            {
                'tier': 'Poor',
                'score_range': '< 50',
                'color': 'danger',
                'description': 'High artifact burden, high impedance, or multiple poor-quality channels. '
                               'Consider re-recording. Flag before AI model inference.',
            },
        ],
        'quality_score_components': [
            {'component': 'Channel quality (Good/Fair/Poor ratio)', 'weight': 40,
             'rationale': 'Proportion of Good-grade channels drives usable EEG bandwidth.'},
            {'component': 'Mean SNR (dB)', 'weight': 30,
             'rationale': 'Signal-to-noise ratio <20 dB degrades automated feature extraction (ACNS 2021).'},
            {'component': 'Mean impedance (kΩ)', 'weight': 20,
             'rationale': 'Electrode impedance >5 kΩ introduces 50/60 Hz noise and skin-contact drift.'},
            {'component': 'Artifact burden (%)', 'weight': 10,
             'rationale': 'Minutes of artifact-contaminated EEG as % of total recording duration.'},
        ],
        'artifact_types': {
            'muscle': 'EMG contamination from scalp/neck muscles — high-frequency (>30 Hz), '
                      'irregular. Common in temporal channels (T3/T4, T5/T6).',
            'eye_blink': 'Fp1/Fp2 deflections from blink artifact — ΔV ~100–300 µV, duration 200–400 ms. '
                         'Removed by ICA or EOG regression.',
            'movement': 'Large-amplitude, low-frequency artefact from patient motion. '
                        'Diffuse channel involvement. Segment flagged for exclusion.',
            'sweat': 'Very slow baseline drift (<0.5 Hz) from galvanic skin response. '
                     'High-pass filter (0.5 Hz) typically removes.',
            'electrode_pop': 'Single-channel spike artifact from loose electrode contact. '
                             'Resolved by re-application of conductive gel.',
            '60Hz': 'Power-line interference. Indicates impedance >5 kΩ or environmental EMF. '
                    'Removed by notch filter (60 Hz ± 2 Hz).',
        },
        'channel_grades': {
            'Good': 'Impedance <5 kΩ OR SNR >20 dB — suitable for analysis.',
            'Fair': 'Impedance 5–10 kΩ or SNR 15–20 dB — usable with caution; flag in reports.',
            'Poor': 'Impedance >10 kΩ or SNR <15 dB — exclude from automated analysis; consider re-recording.',
        },
        'impedance_grades': {
            'Good': '< 5 kΩ — ACNS recommended target for clinical EEG.',
            'Fair': '5–10 kΩ — marginal; acceptable for routine EEG.',
            'Poor': '> 10 kΩ — indicates poor electrode contact; 50/60 Hz and electrode-pop artifacts expected.',
        },
        'activation_procedures': {
            'hyperventilation': 'HV — 3 minutes of forced overbreathing. Unmasks absence seizures via alkalosis-induced '
                                'cerebral vasoconstriction. Contraindicated in recent stroke/cardiovascular disease.',
            'photic_stimulation': 'PS — strobe light 1–30 Hz. Elicits photoparoxysmal response (PPR) in '
                                  'photosensitive epilepsy. ILAE PPR grade I–IV.',
            'sleep_recorded': 'Captures sleep-activated discharges (NREM spike-and-wave, centrotemporal spikes '
                              'in BECTS, frontal lobe epilepsy nocturnal onset).',
            'eyes_open': 'Eyes-open state to assess alpha rhythm (8–12 Hz occipital, posterior).'
                         'Alpha attenuation confirms wakefulness.',
        },
        'standards': [
            'ACNS Guideline 5 (2021) — Minimum standards for EEG recording (electrode placement, '
            'sampling rate ≥256 Hz, impedance <5 kΩ)',
            'IFCN Standards for Digital EEG (2017) — 10-20 system, common reference, 0.5–70 Hz bandwidth',
            'ILAE Commission on Diagnostic Methods — EEG reporting and interpretation standards',
            'IEEE 11073-20601 — Medical device data model for neurophysiology waveforms',
            'ISO 60601-2-26 — Safety requirements for EEG equipment',
        ],
        'data_sources': [
            'eeg_acquisition — 30 recordings, 30 patients, video-EEG and routine EEG, '
            '256–1024 Hz sampling, 10-20 electrode system',
            'channel_quality — 30 patients × 19 channels = 570 channel-quality assessments '
            '(impedance kΩ, SNR dB, Good/Fair/Poor grade)',
            'artifact_annotations — 169 artifact events across all recordings '
            '(type, channel, start time, duration, severity)',
            'recording_conditions — 30 recordings with activation procedure flags '
            '(HV, photic, sleep, eyes-open, patient state)',
            'seizure_metadata — 71 records with EEG pattern, onset zone, lateralization',
        ],
    }


if __name__ == '__main__':
    import json
    print(json.dumps(raw_eeg_overview(), indent=2))
