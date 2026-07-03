"""EEG Technician Dashboard — acquisition quality, signal quality, artifact
analysis, impedance checks, and daily recording logs for the EEG technician's
quality control hub.

Populates and reads from:
  - channel_quality       (per-channel impedance, SNR, quality grade)
  - artifact_annotations  (artifact type, channel, severity, duration)
  - eeg_acquisition       (recording type, duration, sampling rate, montage)
  - recording_conditions  (eyes open, HV, photic, sleep, patient state)

Uses real patient_ids from the patients table.
ACNS minimum technical standards applied throughout.
"""

import json
import os
import random
import sqlite3
from collections import Counter, defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# 10-20 International System channels (19 standard)
CHANNELS_1020 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4',
    'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
    'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz',
]

ARTIFACT_TYPES = [
    'eye_blink', 'muscle', 'movement', 'electrode_pop',
    'sweat', 'ECG',
]

RECORDING_TYPES = ['routine', 'ambulatory', 'video_eeg', 'LTM']
MONTAGES = ['bipolar', 'referential', 'average']
PATIENT_STATES = ['awake', 'drowsy', 'asleep']
COOPERATION_LEVELS = ['excellent', 'good', 'fair', 'poor']
QUALITY_GRADES = ['Good', 'Fair', 'Poor']
OVERALL_GRADES = ['Excellent', 'Good', 'Acceptable', 'Poor']

# Impedance thresholds (kΩ) — ACNS guidelines
IMPEDANCE_GOOD_MAX = 5.0    # < 5 kΩ = Good
IMPEDANCE_FAIR_MAX = 10.0   # 5–10 kΩ = Fair
# > 10 kΩ = Poor

# Typical recording durations by type (minutes)
DURATION_BY_TYPE = {
    'routine':    (20, 40),
    'ambulatory': (1440, 4320),   # 24–72 h
    'video_eeg':  (60, 480),
    'LTM':        (360, 2880),    # 6–48 h
}

# Sampling rates (Hz)
SAMPLING_RATES = [256, 512, 1024]

# Technician notes pool
NOTES_POOL = [
    'Patient cooperative throughout. Good electrode contact.',
    'Mild muscle artifact during first 5 min, resolved after repositioning.',
    'Eye-blink artifact prominent in Fp1/Fp2, patient instructed to minimise blinking.',
    'Recording interrupted briefly for bathroom break. Electrodes reapplied.',
    'Patient fell asleep during recording — sleep capture obtained.',
    'Photic stimulation completed at 1, 2, 4, 8, 10, 12, 15, 18, 20 Hz.',
    'Hyperventilation performed for 3 minutes. Good effort.',
    'High 60 Hz noise noted on T3/T4. Electrode impedance rechecked and improved.',
    'Electrode pop artifact on Fp2 at ~12 min, replaced electrode.',
    'ECG artifact seen in temporal channels, cardiac overlay noted in report.',
    'Excellent recording with minimal artifact burden.',
    'Sweat artifact noted in parietal channels during hyperventilation.',
    'Patient drowsy at onset, activated by verbal stimuli.',
    'Movement artifact during tonic episode captured on video.',
    'All impedances < 5 kΩ at start. Excellent electrode application.',
]

random.seed(42)


def _db_conn():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def _db_query(sql, params=()):
    if not os.path.exists(DB):
        return []
    conn = _db_conn()
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
    return round(sum(values) / len(values), 2) if values else 0


def _impedance_grade(kohm):
    if kohm < IMPEDANCE_GOOD_MAX:
        return 'Good'
    elif kohm < IMPEDANCE_FAIR_MAX:
        return 'Fair'
    return 'Poor'


def _channel_snr_grade(snr_db):
    """SNR (dB): > 20 = Good, 10–20 = Fair, < 10 = Poor."""
    if snr_db >= 20:
        return 'Good'
    elif snr_db >= 10:
        return 'Fair'
    return 'Poor'


def _overall_quality(good_pct, artifact_count, impedance_fail):
    """Derive overall quality grade from signal metrics."""
    if good_pct >= 85 and artifact_count <= 3 and not impedance_fail:
        return 'Excellent'
    elif good_pct >= 70 and artifact_count <= 8:
        return 'Good'
    elif good_pct >= 50:
        return 'Acceptable'
    return 'Poor'


def _generate_channel_data(patient_id, rng):
    """Generate realistic per-channel quality data for one patient."""
    channels = []
    for ch in CHANNELS_1020:
        impedance_kohm = round(rng.uniform(0.5, 18.0), 1)
        snr_db = round(rng.gauss(22, 7), 1)
        snr_db = max(2.0, snr_db)
        grade = _channel_snr_grade(snr_db)
        imp_grade = _impedance_grade(impedance_kohm)
        channels.append({
            'channel': ch,
            'impedance_kohm': impedance_kohm,
            'impedance_grade': imp_grade,
            'snr_db': snr_db,
            'quality_grade': grade,
        })
    return channels


def _generate_artifact_data(patient_id, recording_type, rng):
    """Generate realistic artifact annotations for one patient."""
    n_artifacts = rng.randint(1, 12)
    artifacts = []
    for _ in range(n_artifacts):
        art_type = rng.choice(ARTIFACT_TYPES)
        ch = rng.choice(CHANNELS_1020)
        start_time = round(rng.uniform(0.5, 25.0), 1)
        duration_sec = round(rng.uniform(0.5, 8.0), 1)
        severity = rng.choices(
            ['mild', 'moderate', 'severe'],
            weights=[0.5, 0.35, 0.15]
        )[0]
        artifacts.append({
            'artifact_type': art_type,
            'channel': ch,
            'start_time_min': start_time,
            'duration_sec': duration_sec,
            'severity': severity,
        })
    return artifacts


def _generate_acquisition_data(patient_id, rng):
    """Generate realistic EEG acquisition parameters for one patient."""
    rec_type = rng.choice(RECORDING_TYPES)
    dur_range = DURATION_BY_TYPE[rec_type]
    duration_min = round(rng.uniform(*dur_range), 0)
    sampling_rate = rng.choice(SAMPLING_RATES)
    montage = rng.choice(MONTAGES)
    study_date = f'2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}'
    note = rng.choice(NOTES_POOL)
    return {
        'recording_type': rec_type,
        'duration_min': int(duration_min),
        'sampling_rate': sampling_rate,
        'montage': montage,
        'electrode_system': '10-20',
        'technician_notes': note,
        'study_date': study_date,
    }


def _generate_conditions_data(patient_id, rec_type, rng):
    """Generate realistic recording conditions for one patient."""
    photic = rng.random() < 0.75
    hv = rng.random() < 0.70
    sleep = rec_type in ('LTM', 'ambulatory') or rng.random() < 0.35
    eyes_open = rng.random() < 0.80
    state = rng.choices(
        PATIENT_STATES,
        weights=[0.55, 0.3, 0.15]
    )[0]
    cooperation = rng.choices(
        COOPERATION_LEVELS,
        weights=[0.4, 0.35, 0.18, 0.07]
    )[0]
    return {
        'eyes_open': eyes_open,
        'hyperventilation': hv,
        'photic_stimulation': photic,
        'sleep_recorded': sleep,
        'patient_state': state,
        'cooperation': cooperation,
    }


def _populate_if_empty():
    """Populate the four EEG technician tables with realistic data if empty."""
    if not os.path.exists(DB):
        return

    conn = _db_conn()
    try:
        count = conn.execute('SELECT COUNT(*) FROM eeg_acquisition').fetchone()[0]
        if count > 0:
            return  # already populated

        # Get real patient IDs with demographics
        patients = conn.execute(
            'SELECT patient_id, name, age, gender FROM patients'
        ).fetchall()
        patients = [dict(p) for p in patients]

        # Use EPAT001–EPAT022 first (have real demographics), pad with others
        epat = [p for p in patients if p['patient_id'].startswith('EPAT')]
        others = [p for p in patients if not p['patient_id'].startswith('EPAT')]
        ordered = epat + others
        # Use up to 30 patients
        target_patients = ordered[:30]

        rng = random.Random(7)  # deterministic seed

        for pt in target_patients:
            pid = pt['patient_id']

            # 1. eeg_acquisition
            acq = _generate_acquisition_data(pid, rng)
            conn.execute(
                'INSERT INTO eeg_acquisition (patient_id, fields_json, created_at) VALUES (?, ?, datetime("now"))',
                (pid, json.dumps(acq))
            )

            # 2. recording_conditions
            cond = _generate_conditions_data(pid, acq['recording_type'], rng)
            conn.execute(
                'INSERT INTO recording_conditions (patient_id, fields_json, created_at) VALUES (?, ?, datetime("now"))',
                (pid, json.dumps(cond))
            )

            # 3. channel_quality
            channels = _generate_channel_data(pid, rng)
            conn.execute(
                'INSERT INTO channel_quality (patient_id, fields_json, created_at) VALUES (?, ?, datetime("now"))',
                (pid, json.dumps({'channels': channels}))
            )

            # 4. artifact_annotations
            artifacts = _generate_artifact_data(pid, acq['recording_type'], rng)
            for art in artifacts:
                conn.execute(
                    'INSERT INTO artifact_annotations (patient_id, fields_json, created_at) VALUES (?, ?, datetime("now"))',
                    (pid, json.dumps(art))
                )

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()


def _load_all_data():
    """Load all four tables and return indexed by patient_id."""
    _populate_if_empty()

    acquisitions = {}
    for r in _db_query('SELECT patient_id, fields_json FROM eeg_acquisition'):
        acquisitions[r['patient_id']] = _safe_json(r['fields_json'])

    conditions = {}
    for r in _db_query('SELECT patient_id, fields_json FROM recording_conditions'):
        conditions[r['patient_id']] = _safe_json(r['fields_json'])

    channel_quality = {}
    for r in _db_query('SELECT patient_id, fields_json FROM channel_quality'):
        channel_quality[r['patient_id']] = _safe_json(r['fields_json']).get('channels', [])

    artifact_annotations = defaultdict(list)
    for r in _db_query('SELECT patient_id, fields_json FROM artifact_annotations'):
        artifact_annotations[r['patient_id']].append(_safe_json(r['fields_json']))

    patients = {r['patient_id']: r for r in _db_query('SELECT * FROM patients')}

    return acquisitions, conditions, channel_quality, dict(artifact_annotations), patients


def overview():
    """Return KPI cards + chart data for the EEG Technician overview tab."""
    acquisitions, conditions, channel_quality, artifacts, patients = _load_all_data()

    all_pids = set(acquisitions.keys())

    # ── Recording type distribution
    type_counts = Counter(acq.get('recording_type', 'routine') for acq in acquisitions.values())
    recording_type_dist = [{'type': t, 'count': type_counts.get(t, 0)} for t in RECORDING_TYPES]

    # ── Artifact type distribution
    all_artifacts = []
    for arts in artifacts.values():
        all_artifacts.extend(arts)
    art_type_counts = Counter(a.get('artifact_type', '') for a in all_artifacts)
    artifact_type_dist = [{'type': t, 'count': art_type_counts.get(t, 0)} for t in ARTIFACT_TYPES]

    # ── Channel quality aggregation
    total_channels = 0
    good_channels = 0
    fair_channels = 0
    poor_channels = 0
    all_impedances = []
    all_snrs = []
    impedance_failures = 0  # patients with >= 1 Poor impedance channel

    for pid, chs in channel_quality.items():
        has_imp_fail = False
        for ch in chs:
            total_channels += 1
            grade = ch.get('quality_grade', 'Good')
            imp = ch.get('impedance_kohm', 5.0)
            snr = ch.get('snr_db', 20.0)
            all_impedances.append(imp)
            all_snrs.append(snr)
            if grade == 'Good':
                good_channels += 1
            elif grade == 'Fair':
                fair_channels += 1
            else:
                poor_channels += 1
            if imp >= IMPEDANCE_FAIR_MAX:
                has_imp_fail = True
        if has_imp_fail:
            impedance_failures += 1

    signal_quality_pass = (good_channels + fair_channels)
    signal_quality_pass_rate = round(100 * signal_quality_pass / total_channels, 1) if total_channels else 0
    impedance_failure_rate = round(100 * impedance_failures / len(all_pids), 1) if all_pids else 0

    # ── Recording stats
    durations = [acq.get('duration_min', 0) for acq in acquisitions.values()]
    mean_duration = round(sum(durations) / len(durations), 1) if durations else 0

    total_artifacts = len(all_artifacts)
    artifact_rate = round(total_artifacts / len(all_pids), 1) if all_pids else 0

    # ── Photic / sleep / HV rates
    photic_done = sum(1 for c in conditions.values() if c.get('photic_stimulation'))
    sleep_recorded = sum(1 for c in conditions.values() if c.get('sleep_recorded'))
    hv_done = sum(1 for c in conditions.values() if c.get('hyperventilation'))
    n_cond = len(conditions) or 1
    photic_rate = round(100 * photic_done / n_cond, 1)
    sleep_rate = round(100 * sleep_recorded / n_cond, 1)
    hv_rate = round(100 * hv_done / n_cond, 1)

    avg_channels = round(len(CHANNELS_1020), 1)  # standard 10-20 = 19 channels

    # ── Quality grade severity distribution (patient-level)
    sev_counts = Counter()
    for pid in all_pids:
        chs = channel_quality.get(pid, [])
        arts = artifacts.get(pid, [])
        imp_fail = any(ch.get('impedance_kohm', 0) >= IMPEDANCE_FAIR_MAX for ch in chs)
        good = sum(1 for ch in chs if ch.get('quality_grade') == 'Good')
        total = len(chs) or 1
        good_pct = 100 * good / total
        grade = _overall_quality(good_pct, len(arts), imp_fail)
        sev_counts[grade] += 1
    severity_distribution = [
        {'grade': g, 'count': sev_counts.get(g, 0)} for g in OVERALL_GRADES
    ]

    # ── Quality score histogram (% good channels per patient)
    quality_scores = []
    for pid, chs in channel_quality.items():
        if chs:
            good = sum(1 for ch in chs if ch.get('quality_grade') == 'Good')
            quality_scores.append(round(100 * good / len(chs), 0))
    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    quality_histogram = []
    for lo, hi in bins:
        cnt = sum(1 for s in quality_scores if lo <= s < hi or (hi == 100 and s == 100))
        quality_histogram.append({'range': f'{lo}-{hi}%', 'count': cnt})

    # ── Impedance histogram (kΩ)
    imp_bins = [(0, 2), (2, 5), (5, 8), (8, 12), (12, 18)]
    impedance_histogram = []
    for lo, hi in imp_bins:
        cnt = sum(1 for v in all_impedances if lo <= v < hi)
        impedance_histogram.append({'range': f'{lo}-{hi} kΩ', 'count': cnt})

    return {
        'kpis': {
            'total_recordings': len(all_pids),
            'signal_quality_pass_rate': signal_quality_pass_rate,
            'artifact_rate': artifact_rate,
            'mean_recording_duration_min': mean_duration,
            'impedance_failure_rate': impedance_failure_rate,
            'photic_stim_rate': photic_rate,
            'sleep_capture_rate': sleep_rate,
            'hv_completion_rate': hv_rate,
            'avg_channels_per_study': avg_channels,
            'total_artifacts_logged': total_artifacts,
        },
        'severity_distribution': severity_distribution,
        'recording_type_distribution': recording_type_dist,
        'artifact_type_distribution': artifact_type_dist,
        'quality_histogram': quality_histogram,
        'impedance_histogram': impedance_histogram,
    }


def breakdown():
    """Return per-patient EEG acquisition detail for the Patient Detail tab."""
    acquisitions, conditions, channel_quality, artifacts, patients = _load_all_data()

    all_pids = sorted(set(acquisitions.keys()))
    profiles = []

    for pid in all_pids:
        pt = patients.get(pid, {})
        acq = acquisitions.get(pid, {})
        cond = conditions.get(pid, {})
        chs = channel_quality.get(pid, [])
        arts = artifacts.get(pid, [])

        # Channel quality counts
        good_ch = sum(1 for ch in chs if ch.get('quality_grade') == 'Good')
        fair_ch = sum(1 for ch in chs if ch.get('quality_grade') == 'Fair')
        poor_ch = sum(1 for ch in chs if ch.get('quality_grade') == 'Poor')
        total_ch = len(chs) or 1
        good_pct = 100 * good_ch / total_ch

        # Impedance pass
        imp_fail = any(ch.get('impedance_kohm', 0) >= IMPEDANCE_FAIR_MAX for ch in chs)
        impedance_pass = not imp_fail

        # Average impedance and SNR
        avg_imp = _avg([ch.get('impedance_kohm', 0) for ch in chs])
        avg_snr = _avg([ch.get('snr_db', 0) for ch in chs])

        # Dominant artifact
        art_types = [a.get('artifact_type', '') for a in arts]
        dom_art = Counter(art_types).most_common(1)[0][0] if art_types else None

        # Overall quality
        overall_grade = _overall_quality(good_pct, len(arts), imp_fail)

        # Per-channel detail (for expanded view)
        channel_detail = sorted(chs, key=lambda c: c.get('impedance_kohm', 0), reverse=True)

        # Artifact detail (for expanded view)
        artifact_detail = sorted(arts, key=lambda a: a.get('start_time_min', 0))

        profiles.append({
            'patient_id': pid,
            'name': pt.get('name') or pid,
            'age': pt.get('age'),
            'gender': pt.get('gender'),
            'disease': pt.get('disease', 'epilepsy'),
            # Acquisition
            'recording_type': acq.get('recording_type', ''),
            'duration_min': acq.get('duration_min', 0),
            'sampling_rate': acq.get('sampling_rate', 256),
            'montage': acq.get('montage', ''),
            'electrode_system': acq.get('electrode_system', '10-20'),
            'study_date': acq.get('study_date', ''),
            'technician_notes': acq.get('technician_notes', ''),
            # Conditions
            'eyes_open': cond.get('eyes_open', False),
            'hyperventilation_done': cond.get('hyperventilation', False),
            'photic_done': cond.get('photic_stimulation', False),
            'sleep_recorded': cond.get('sleep_recorded', False),
            'patient_state': cond.get('patient_state', 'awake'),
            'cooperation': cond.get('cooperation', 'good'),
            # Channel quality summary
            'channel_count': len(chs),
            'good_channels': good_ch,
            'fair_channels': fair_ch,
            'poor_channels': poor_ch,
            'avg_impedance_kohm': avg_imp,
            'avg_snr_db': avg_snr,
            'impedance_pass': impedance_pass,
            # Artifacts
            'artifact_count': len(arts),
            'dominant_artifact': dom_art,
            # Overall
            'overall_quality_grade': overall_grade,
            # Detail lists (expanded)
            'channel_detail': channel_detail,
            'artifact_detail': artifact_detail,
        })

    return {'patients': profiles}


def definitions():
    """Return EEG technician terminology, ACNS standards, and quality metrics."""
    return {
        'title': 'EEG Acquisition — Definitions & ACNS Standards',
        'sections': [
            {
                'heading': '10-20 International Electrode System',
                'items': [
                    {'term': '10-20 System', 'detail': 'Standardized placement of 19 scalp electrodes based on 10% and 20% distances between nasion-inion and pre-auricular points. Ensures reproducible recordings across centers. Standard positions: Fp1/Fp2, F3/F4, F7/F8, C3/C4, T3/T4, P3/P4, T5/T6, O1/O2, Fz, Cz, Pz.'},
                    {'term': 'Impedance Standard', 'detail': 'ACNS mandates electrode-scalp impedance < 5 kΩ (Good), 5–10 kΩ (Fair/acceptable), > 10 kΩ (Poor — must be corrected). High impedance degrades SNR and introduces artifact.'},
                    {'term': 'SNR (Signal-to-Noise Ratio)', 'detail': 'Ratio of neural EEG amplitude to background noise, expressed in dB. > 20 dB = Good, 10–20 dB = Fair, < 10 dB = Poor. Affected by impedance, electrode contact, and environmental EMI.'},
                    {'term': 'Reference Electrode', 'detail': 'Common reference (linked mastoids A1+A2, Cz, average reference) determines signal polarity and amplitude. Choice of reference affects the apparent lateralization of abnormalities.'},
                ],
            },
            {
                'heading': 'Recording Types',
                'items': [
                    {'term': 'Routine EEG (20–40 min)', 'detail': 'Standard outpatient recording with eyes open/closed, hyperventilation, and photic stimulation. Diagnostic yield ~30–50% for epileptiform discharges. ACNS minimum: 20 min of artifact-free recording.'},
                    {'term': 'Ambulatory EEG (24–72 h)', 'detail': 'Portable continuous recording with reduced electrode set. Used to capture intermittent events. Artifact burden is higher; technician must review electrode application and patient diary.'},
                    {'term': 'Video-EEG (1–8 h)', 'detail': 'Synchronized video and EEG recording for ictal semiology correlation. Used for pre-surgical evaluation and PNES differentiation. Requires careful electrode fixation and cable management.'},
                    {'term': 'Long-Term Monitoring (LTM)', 'detail': 'Inpatient continuous monitoring (6–48+ h) with or without video. Used for seizure quantification, medication wean, and surgical candidacy. Daily impedance checks and electrode re-gelling required.'},
                ],
            },
            {
                'heading': 'Montages',
                'items': [
                    {'term': 'Bipolar Montage', 'detail': 'Each channel is the difference between two adjacent electrodes (e.g., Fp1-F3). Excellent for localizing focal discharges (phase reversal). Less affected by common-mode noise.'},
                    {'term': 'Referential Montage', 'detail': 'Each electrode referenced to a common point (A1+A2, Cz). Better for detecting diffuse changes and amplitude asymmetry. Reference contamination can mimic bilateral abnormalities.'},
                    {'term': 'Average Reference Montage', 'detail': 'Each electrode referenced to the mean of all electrodes. Minimizes reference contamination for focal discharges. Requires all electrodes to be recording well; poor-quality channels should be excluded.'},
                ],
            },
            {
                'heading': 'Artifact Types & Recognition',
                'items': [
                    {'term': 'Eye Blink (eye_blink)', 'detail': 'Frontopolar (Fp1/Fp2) slow wave (~200–400 ms) from corneoretinal potential. Seen in awake patients. Distinguish from frontal delta by distribution limited to Fp channels.'},
                    {'term': 'Muscle Artifact (muscle)', 'detail': 'High-frequency (> 30 Hz) bursts, maximal in temporal/frontal electrodes. Most common artifact. Exacerbated by patient anxiety, teeth clenching, or poor scalp preparation.'},
                    {'term': 'Movement Artifact (movement)', 'detail': 'Irregular slow drifts and sharp transients from patient movement or electrode displacement. Often widespread. Critically important to capture on video during ictal events.'},
                    {'term': 'Electrode Pop (electrode_pop)', 'detail': 'Sudden large transient limited to a single electrode, often with a characteristic triangular/spike morphology. Caused by poor electrode-skin contact or trapped air bubble. Requires immediate electrode replacement.'},
                    {'term': 'Sweat Artifact (sweat)', 'detail': 'Very slow (< 0.5 Hz) sinusoidal drifts from skin conductance changes. Maximal during hyperventilation, particularly in parieto-occipital regions. Reduce with HF filter adjustment (LFF 0.3 Hz).'},
                    {'term': 'ECG Artifact (ECG)', 'detail': 'QRS-synchronous potentials (especially in temporal and vertex electrodes) from cardiac electrical field. Confirm with simultaneous ECG channel. Does not indicate interictal discharges.'},
                ],
            },
            {
                'heading': 'Activation Procedures',
                'items': [
                    {'term': 'Hyperventilation (HV)', 'detail': 'Patient breathes deeply for 3 minutes. Induces hypocapnia and cerebral vasoconstriction, enhancing generalized epileptiform activity (especially absence epilepsy). Record 1 min post-HV for persistent changes.'},
                    {'term': 'Photic Stimulation (IPS)', 'detail': 'Stroboscopic light at 1–30 Hz. Stimulates occipital cortex and detects photoparoxysmal responses (PPR). Record bilateral occipital channels. ILAE IPS protocol: 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 60 Hz (1-s ON / 7-s OFF per frequency).'},
                    {'term': 'Sleep Capture', 'detail': 'Sleep EEG enhances yield of temporal lobe epileptiform discharges by 25–40%. Deepening of drowsiness and Stage N1/N2 sleep also distinguishes benign variants (POSTS, wicket spikes) from pathological discharges.'},
                    {'term': 'Eyes Open / Closed', 'detail': 'Alpha rhythm (8–13 Hz) attenuates with eye opening (Berger effect). Used to confirm alpha reactivity and baseline wakefulness. Persistent alpha with eyes open is abnormal.'},
                ],
            },
            {
                'heading': 'ACNS Minimum Technical Standards',
                'items': [
                    {'term': 'Minimum recording duration', 'detail': 'Routine EEG: ≥ 20 min artifact-free recording (ACNS 2016). Activation procedures are additional.'},
                    {'term': 'Electrode impedance', 'detail': '< 5 kΩ for all electrodes before starting. Must be documented. Recheck every 30–60 min for LTM.'},
                    {'term': 'Sampling rate', 'detail': 'Minimum 256 Hz; preferred ≥ 512 Hz. High-frequency oscillation (HFO) analysis requires ≥ 1000 Hz.'},
                    {'term': 'Filter settings', 'detail': 'Low-frequency filter (LFF): 0.3–1 Hz. High-frequency filter (HFF): 70–100 Hz. Notch filter (60 Hz) only if EMI present; avoid routinely.'},
                    {'term': 'Calibration', 'detail': 'Biocalibration (patient resting, eyes closed, 10 s) and electronic calibration must be performed at study start and end.'},
                    {'term': 'Documentation', 'detail': 'Technician must annotate: electrode application time, impedances, activation procedures performed, patient state, behavioral events, and recording end time.'},
                ],
            },
            {
                'heading': 'Quality Metrics & Targets',
                'items': [
                    {'term': 'Signal Quality Pass Rate', 'detail': 'Target: ≥ 90% of channels Good or Fair (impedance < 10 kΩ, SNR ≥ 10 dB). Poor quality channels must be documented and re-gelled where possible.'},
                    {'term': 'Impedance Failure Rate', 'detail': 'Target: < 10% of studies with any electrode > 10 kΩ at study start. Persistent failures require electrode replacement or scalp preparation review.'},
                    {'term': 'Artifact Burden', 'detail': 'Target: ≤ 5 artifact epochs per 20-min recording. High artifact burden degrades diagnostic yield and reviewer efficiency.'},
                    {'term': 'Activation Completion', 'detail': 'Photic stimulation target ≥ 90% of eligible studies. HV target ≥ 80% (contraindicated in recent stroke/TIA, sickle cell disease, severe COPD).'},
                    {'term': 'Sleep Capture Rate', 'detail': 'Target ≥ 40% for routine EEG with sleep deprivation protocol; ≥ 95% for LTM recordings.'},
                ],
            },
        ],
    }


if __name__ == '__main__':
    import pprint
    print('=== OVERVIEW ===')
    pprint.pprint(overview())
    print('\n=== BREAKDOWN (first 2 patients) ===')
    bd = breakdown()
    for p in bd['patients'][:2]:
        p_display = {k: v for k, v in p.items() if k not in ('channel_detail', 'artifact_detail')}
        pprint.pprint(p_display)
    print(f'\nTotal patients: {len(bd["patients"])}')
    print('\n=== DEFINITIONS ===')
    df = definitions()
    print(f'Sections: {len(df["sections"])}')
    for s in df['sections']:
        print(f'  {s["heading"]}: {len(s["items"])} items')
