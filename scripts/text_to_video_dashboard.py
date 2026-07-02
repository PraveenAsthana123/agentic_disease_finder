"""Text-to-Video AI Dashboard — clinical data-to-video synthesis pipeline
monitoring, EEG recording visualization, seizure event video timelines,
MRI volumetric video rendering, clinical report video summaries.

Maps clinical.db tables to text-to-video AI concepts:
- uploads            -> source EEG recordings for video rendering
- analyses           -> AI-annotated video overlays (predictions, confidence)
- mri_findings       -> 3D brain volume video renders
- seizure_diary      -> seizure event timeline videos
- transaction_log    -> video pipeline events (cv_pipeline, eeg_upload)
- patients           -> per-patient video synthesis profiles
- finops_costs       -> GPU rendering cost tracking
"""

import sqlite3
import os
import json
from collections import Counter
from datetime import datetime

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# Video pipeline components in transaction_log
VIDEO_COMPONENTS = ['cv_pipeline', 'eeg_upload', 'genai_bot']

# Video output categories
VIDEO_CATEGORIES = {
    'eeg_timelapse': 'EEG Timelapse Video',
    'seizure_event': 'Seizure Event Clip',
    'mri_flythrough': 'MRI 3D Flythrough',
    'clinical_summary': 'Clinical Summary Video',
    'patient_education': 'Patient Education Video',
}

# Video duration tiers (seconds)
DURATION_TIERS = [
    ('short', 0, 30, '#10b981'),
    ('medium', 30, 120, '#3b82f6'),
    ('long', 120, 600, '#f59e0b'),
    ('extended', 600, 999999, '#ef4444'),
]

# Video resolution presets
RESOLUTIONS = {
    'SD': '640x480',
    'HD': '1280x720',
    'FHD': '1920x1080',
    '4K': '3840x2160',
}


def _db_query(sql, params=()):
    if not os.path.exists(DB):
        return []
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _avg(values):
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _duration_tier(dur_sec):
    for label, lo, hi, color in DURATION_TIERS:
        if lo <= dur_sec < hi:
            return label
    return 'extended'


def _estimate_render_time(n_frames, complexity=1.0):
    """Estimate GPU render time in seconds (~30 fps, ~0.5s/frame for medical video)."""
    return round(n_frames * 0.5 * complexity, 1)


def _load_uploads():
    return _db_query(
        "SELECT id, patient_id, file_name, disease, department, created_at "
        "FROM uploads ORDER BY created_at"
    )


def _load_analyses():
    return _db_query(
        "SELECT id, upload_id, patient_id, disease, predicted_label, "
        "confidence, signal_quality, report_path, created_at "
        "FROM analyses ORDER BY created_at"
    )


def _load_mri_findings():
    return _db_query(
        "SELECT id, patient_id, fields_json, created_at "
        "FROM mri_findings ORDER BY created_at"
    )


def _load_seizure_diary():
    return _db_query(
        "SELECT id, patient_id, event_date, event_time, duration_sec, "
        "location, witnessed, aura, awareness, motor_signs, severity, "
        "trigger, notes, created_at "
        "FROM seizure_diary ORDER BY event_date"
    )


def _load_pipeline_events():
    return _db_query(
        "SELECT id, patient_id, component, action, actor, detail, ts_utc, ts_local "
        "FROM transaction_log "
        "WHERE component IN ('cv_pipeline', 'eeg_upload', 'genai_bot') "
        "ORDER BY ts_utc"
    )


def _load_patients():
    return _db_query(
        "SELECT patient_id, name, age, gender, disease, department, created_at "
        "FROM patients ORDER BY patient_id"
    )


def _load_finops():
    return _db_query(
        "SELECT id, cost_date, category, sub_category, model_or_service, "
        "component, requests, tokens_in, tokens_out, gpu_minutes, cost_usd "
        "FROM finops_costs WHERE category = 'inference' "
        "ORDER BY cost_date"
    )


def _parse_mri_fields(row):
    """Parse fields_json from mri_findings into usable dict."""
    fj = row.get('fields_json', '{}')
    try:
        return json.loads(fj) if fj else {}
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# overview()
# ---------------------------------------------------------------------------

def overview():
    """KPI-level summary for the Text-to-Video AI dashboard."""
    uploads = _load_uploads()
    analyses = _load_analyses()
    mri = _load_mri_findings()
    seizures = _load_seizure_diary()
    pipeline = _load_pipeline_events()
    patients = _load_patients()
    finops = _load_finops()

    if not uploads and not mri and not seizures:
        return {
            'available': False,
            'message': 'No video source data available. '
                       'Add EEG uploads, MRI findings, or seizure diary entries first.',
        }

    # --- Source data counts ---
    total_uploads = len(uploads)
    total_analyses = len(analyses)
    total_mri = len(mri)
    total_seizures = len(seizures)

    # --- Estimated video outputs ---
    # Each upload can produce an EEG timelapse video
    # Each seizure event can produce an event clip
    # Each MRI finding can produce a 3D flythrough
    total_potential_videos = total_uploads + total_seizures + total_mri
    est_video_minutes = round(
        (total_uploads * 2.0) +         # ~2 min per EEG timelapse
        (total_seizures * 0.5) +         # ~30 sec per seizure clip
        (total_mri * 1.5),               # ~1.5 min per MRI flythrough
        1
    )

    # --- Parse MRI findings for video rendering stats ---
    mri_parsed = [_parse_mri_fields(m) for m in mri]
    mri_regions = Counter()
    mri_classes = Counter()
    for mp in mri_parsed:
        region = mp.get('location') or mp.get('region') or 'unknown'
        mri_regions[region] += 1
        cls = mp.get('classification') or mp.get('finding_class') or 'unknown'
        mri_classes[cls] += 1

    # --- Seizure severity distribution ---
    severity_counts = Counter(s.get('severity', 'unknown') for s in seizures)
    severity_distribution = [
        {'severity': sev, 'count': cnt}
        for sev, cnt in sorted(severity_counts.items(), key=lambda x: -x[1])
    ]

    # --- Video type distribution ---
    video_types = [
        {'type': 'EEG Timelapse', 'count': total_uploads, 'color': '#3b82f6'},
        {'type': 'Seizure Event Clip', 'count': total_seizures, 'color': '#ef4444'},
        {'type': 'MRI 3D Flythrough', 'count': total_mri, 'color': '#8b5cf6'},
        {'type': 'Clinical Summary', 'count': total_analyses, 'color': '#10b981'},
    ]

    # --- Daily activity ---
    daily_counts = Counter()
    for u in uploads:
        day = (u.get('created_at') or '')[:10]
        if day:
            daily_counts[day] += 1
    for s in seizures:
        day = (s.get('event_date') or '')[:10]
        if day:
            daily_counts[day] += 1
    daily_activity = [
        {'date': day, 'count': cnt}
        for day, cnt in sorted(daily_counts.items())
    ]

    # --- Confidence distribution from analyses ---
    confidence_tiers = Counter()
    for a in analyses:
        c = a.get('confidence', 0) or 0
        if c >= 0.9:
            confidence_tiers['high (≥0.9)'] += 1
        elif c >= 0.7:
            confidence_tiers['medium (0.7-0.9)'] += 1
        elif c >= 0.5:
            confidence_tiers['low (0.5-0.7)'] += 1
        else:
            confidence_tiers['very low (<0.5)'] += 1

    confidence_distribution = [
        {'tier': t, 'count': cnt}
        for t, cnt in sorted(confidence_tiers.items(), key=lambda x: -x[1])
    ]

    # Unique patients
    patient_ids = set()
    for u in uploads:
        if u.get('patient_id'):
            patient_ids.add(u['patient_id'])
    for s in seizures:
        if s.get('patient_id'):
            patient_ids.add(s['patient_id'])
    for m in mri:
        if m.get('patient_id'):
            patient_ids.add(m['patient_id'])
    patients_covered = len(patient_ids)

    # Pipeline events
    pipeline_count = len(pipeline)

    # FinOps — GPU rendering costs
    gpu_costs = [f for f in finops if (f.get('gpu_minutes') or 0) > 0
                 or 'video' in (f.get('sub_category') or '').lower()
                 or 'render' in (f.get('sub_category') or '').lower()]
    total_gpu_cost = round(sum(f.get('cost_usd', 0) for f in gpu_costs), 2)
    total_gpu_minutes = round(sum(f.get('gpu_minutes', 0) or 0 for f in gpu_costs), 1)
    if not gpu_costs:
        total_gpu_cost = round(sum(f.get('cost_usd', 0) for f in finops) * 0.08, 2)
        total_gpu_minutes = round(total_gpu_cost / 0.50 * 60, 1)  # est $0.50/hr

    # Mean confidence
    confidences = [a.get('confidence', 0) or 0 for a in analyses if a.get('confidence')]
    mean_confidence = round(_avg(confidences), 2)

    return {
        'available': True,
        'total_uploads': total_uploads,
        'total_mri_findings': total_mri,
        'total_seizure_events': total_seizures,
        'total_analyses': total_analyses,
        'total_potential_videos': total_potential_videos,
        'est_video_minutes': est_video_minutes,
        'patients_covered': patients_covered,
        'pipeline_events': pipeline_count,
        'gpu_cost_usd': total_gpu_cost,
        'gpu_minutes': total_gpu_minutes,
        'mean_confidence': mean_confidence,
        'video_type_distribution': video_types,
        'severity_distribution': severity_distribution,
        'confidence_distribution': confidence_distribution,
        'daily_activity': daily_activity,
        'kpis': [
            {'label': 'Potential Videos', 'value': str(total_potential_videos)},
            {'label': 'EEG Uploads', 'value': str(total_uploads)},
            {'label': 'Seizure Events', 'value': str(total_seizures)},
            {'label': 'MRI Findings', 'value': str(total_mri)},
            {'label': 'Est. Video (min)', 'value': str(est_video_minutes),
             'color': '#3b82f6'},
            {'label': 'Patients Covered', 'value': str(patients_covered)},
            {'label': 'Mean Confidence', 'value': str(mean_confidence),
             'color': '#10b981' if mean_confidence >= 0.7 else '#f59e0b'},
            {'label': 'GPU Cost (USD)', 'value': f'${total_gpu_cost:.2f}',
             'color': '#f59e0b' if total_gpu_cost > 100 else '#10b981'},
        ],
    }


# ---------------------------------------------------------------------------
# breakdown()
# ---------------------------------------------------------------------------

def breakdown():
    """Detailed text-to-video breakdown — source inventory, seizure events,
    MRI renders, per-patient profiles, pipeline events."""
    uploads = _load_uploads()
    analyses = _load_analyses()
    mri = _load_mri_findings()
    seizures = _load_seizure_diary()
    patients = _load_patients()
    pipeline = _load_pipeline_events()

    if not uploads and not mri and not seizures:
        return {'available': False}

    patient_map = {p['patient_id']: p for p in patients}
    analysis_by_upload = {}
    for a in analyses:
        analysis_by_upload[a.get('upload_id')] = a

    # --- EEG source inventory (uploads -> timelapse videos) ---
    source_inventory = []
    for u in uploads:
        a = analysis_by_upload.get(u['id'], {})
        est_dur = 120  # ~2 min per EEG timelapse
        source_inventory.append({
            'id': u['id'],
            'patient_id': u['patient_id'],
            'file_name': u.get('file_name', ''),
            'disease': u.get('disease', ''),
            'department': u.get('department', ''),
            'predicted_label': a.get('predicted_label', ''),
            'confidence': a.get('confidence'),
            'signal_quality': a.get('signal_quality'),
            'video_type': 'EEG Timelapse',
            'est_duration_sec': est_dur,
            'duration_tier': _duration_tier(est_dur),
            'created_at': u.get('created_at'),
        })

    # --- Seizure event clips ---
    seizure_clips = []
    for s in seizures:
        dur = s.get('duration_sec') or 30
        clip_dur = min(dur + 30, 300)  # event + 30s context, cap 5 min
        seizure_clips.append({
            'id': s['id'],
            'patient_id': s['patient_id'],
            'event_date': s.get('event_date'),
            'event_time': s.get('event_time'),
            'duration_sec': dur,
            'clip_duration_sec': clip_dur,
            'duration_tier': _duration_tier(clip_dur),
            'severity': s.get('severity', 'unknown'),
            'location': s.get('location', ''),
            'awareness': s.get('awareness', ''),
            'motor_signs': s.get('motor_signs', ''),
            'aura': s.get('aura'),
            'trigger': s.get('trigger', ''),
            'video_type': 'Seizure Event Clip',
        })

    # --- MRI 3D flythrough renders ---
    mri_renders = []
    for m in mri:
        mp = _parse_mri_fields(m)
        est_dur = 90  # ~1.5 min per flythrough
        mri_renders.append({
            'id': m['id'],
            'patient_id': m['patient_id'],
            'region': mp.get('location') or mp.get('region') or 'unknown',
            'classification': mp.get('classification') or mp.get('finding_class') or 'unknown',
            'confidence': mp.get('confidence'),
            'iou': mp.get('iou'),
            'video_type': 'MRI 3D Flythrough',
            'est_duration_sec': est_dur,
            'duration_tier': _duration_tier(est_dur),
            'created_at': m.get('created_at'),
        })

    # --- Patient video profiles ---
    patient_uploads = {}
    for u in uploads:
        pid = u.get('patient_id')
        if pid:
            patient_uploads.setdefault(pid, []).append(u)

    patient_seizures = {}
    for s in seizures:
        pid = s.get('patient_id')
        if pid:
            patient_seizures.setdefault(pid, []).append(s)

    patient_mri = {}
    for m in mri:
        pid = m.get('patient_id')
        if pid:
            patient_mri.setdefault(pid, []).append(m)

    all_pids = set(patient_uploads.keys()) | set(patient_seizures.keys()) | set(patient_mri.keys())
    patient_profiles = []
    for pid in sorted(all_pids):
        pu = patient_uploads.get(pid, [])
        ps = patient_seizures.get(pid, [])
        pm = patient_mri.get(pid, [])
        pinfo = patient_map.get(pid, {})

        total_videos = len(pu) + len(ps) + len(pm)
        est_total_dur = (len(pu) * 120) + sum((s.get('duration_sec') or 30) + 30 for s in ps) + (len(pm) * 90)

        video_types = []
        if pu:
            video_types.append('EEG Timelapse')
        if ps:
            video_types.append('Seizure Clip')
        if pm:
            video_types.append('MRI Flythrough')

        worst_severity = 'none'
        for s in ps:
            sev = s.get('severity', '')
            if sev in ('severe', 'status_epilepticus'):
                worst_severity = sev
                break
            elif sev == 'moderate' and worst_severity not in ('severe',):
                worst_severity = sev
            elif sev == 'mild' and worst_severity == 'none':
                worst_severity = sev

        patient_profiles.append({
            'patient_id': pid,
            'name': pinfo.get('name', ''),
            'age': pinfo.get('age'),
            'disease': pinfo.get('disease'),
            'n_uploads': len(pu),
            'n_seizure_events': len(ps),
            'n_mri_findings': len(pm),
            'total_videos': total_videos,
            'est_total_duration_sec': est_total_dur,
            'video_types': video_types,
            'worst_severity': worst_severity,
        })

    # --- Pipeline events ---
    pipeline_events = [
        {
            'id': ev['id'],
            'patient_id': ev.get('patient_id'),
            'component': ev.get('component'),
            'action': ev.get('action'),
            'actor': ev.get('actor'),
            'detail': (ev.get('detail') or '')[:120],
            'ts_utc': ev.get('ts_utc'),
        }
        for ev in pipeline
    ]

    # --- Action distribution ---
    action_counts = Counter(e.get('action', 'unknown') for e in pipeline)
    action_distribution = [
        {'action': act, 'count': cnt}
        for act, cnt in sorted(action_counts.items(), key=lambda x: -x[1])
    ]

    return {
        'available': True,
        'source_inventory': source_inventory,
        'seizure_clips': seizure_clips,
        'mri_renders': mri_renders,
        'patient_profiles': patient_profiles,
        'pipeline_events': pipeline_events[:200],
        'action_distribution': action_distribution,
    }


# ---------------------------------------------------------------------------
# definitions()
# ---------------------------------------------------------------------------

def definitions():
    """Definitions tab for the Text-to-Video AI dashboard."""
    return {
        'concepts': [
            {
                'name': 'Text-to-Video Synthesis',
                'description': 'AI technology that generates video content from textual '
                               'descriptions or structured clinical data. In epilepsy care, '
                               'this includes rendering EEG signal visualizations, seizure '
                               'event timelines, and MRI volumetric flythroughs from patient '
                               'records and diagnostic reports.',
            },
            {
                'name': 'EEG Timelapse Rendering',
                'description': 'Automated generation of time-compressed video showing EEG '
                               'signal evolution over recording sessions. Highlights ictal '
                               'events, artifact zones, and channel-specific activity with '
                               'color-coded overlays for rapid clinical review.',
            },
            {
                'name': 'Seizure Event Visualization',
                'description': 'Video clip generation centered on seizure diary entries, '
                               'showing pre-ictal baseline, ictal onset, seizure propagation '
                               'patterns, and post-ictal recovery. Includes annotations for '
                               'duration, severity, awareness, and motor manifestations.',
            },
            {
                'name': 'MRI Volumetric Flythrough',
                'description': '3D rendering of brain MRI data as navigable video tours. '
                               'Highlights lesion locations (hippocampal sclerosis, FCD, '
                               'cavernomas), generates rotation views, and provides '
                               'volumetric measurement overlays for surgical planning.',
            },
            {
                'name': 'Clinical Summary Video',
                'description': 'AI-generated video summarizing patient clinical data: '
                               'diagnosis timeline, medication history, seizure frequency '
                               'trends, and treatment response. Designed for multi-disciplinary '
                               'team meetings and patient education sessions.',
            },
            {
                'name': 'Temporal Encoding',
                'description': 'Method of representing time-series clinical data (EEG, '
                               'seizure frequency, medication changes) as sequential video '
                               'frames. Uses learned temporal embeddings to create smooth '
                               'transitions and maintain clinical accuracy across frames.',
            },
            {
                'name': 'Neural Video Generation',
                'description': 'Deep learning models (diffusion-based, GAN-based, or '
                               'transformer-based) that synthesize video frames from latent '
                               'representations. Medical applications require frame-level '
                               'accuracy and cannot tolerate hallucinated visual artifacts.',
            },
            {
                'name': 'Video Annotation Pipeline',
                'description': 'Automated overlay of clinical metadata onto generated videos: '
                               'patient ID, timestamp, channel labels, prediction confidence, '
                               'alert badges. Ensures every frame carries provenance and '
                               'clinical context for downstream review.',
            },
        ],
        'quality_metrics': [
            {
                'name': 'Frame Accuracy',
                'description': 'Percentage of generated video frames that faithfully represent '
                               'the underlying clinical data without visual artifacts or '
                               'hallucinated features. Medical video requires ≥ 99% accuracy.',
            },
            {
                'name': 'Temporal Consistency',
                'description': 'Measure of smooth transitions between consecutive frames '
                               'without flickering, discontinuities, or temporal aliasing. '
                               'Assessed via perceptual similarity (LPIPS) between frames.',
            },
            {
                'name': 'Render Latency',
                'description': 'Time from data submission to video availability. Real-time '
                               'rendering targets < 1x real-time factor; batch rendering '
                               'targets completion within clinical workflow timelines.',
            },
            {
                'name': 'Clinical Fidelity Score',
                'description': 'Expert-assessed rating (1-5) of how accurately the generated '
                               'video represents the clinical data. Covers signal accuracy, '
                               'annotation correctness, and diagnostic relevance.',
            },
        ],
        'video_categories': [
            {'category': k, 'label': v}
            for k, v in VIDEO_CATEGORIES.items()
        ],
        'compliance': [
            {
                'ref': 'FDA AI/ML Framework',
                'note': 'Video synthesis from clinical data must preserve diagnostic '
                        'accuracy. Generated visualizations used in clinical decision-making '
                        'require validation against ground truth data. AI-generated video '
                        'must not introduce misleading visual artifacts.',
            },
            {
                'ref': 'EU AI Act Art. 6',
                'note': 'AI-generated clinical videos must be clearly labeled as '
                        'machine-generated content. Clinicians must be informed when '
                        'viewing synthesized vs. recorded video. Transparency requirements '
                        'apply to all AI-generated visual content in healthcare.',
            },
            {
                'ref': 'ISO 14971',
                'note': 'Risk analysis must address video generation failure modes: '
                        'incorrect temporal alignment of events, misleading color mappings, '
                        'hallucinated lesions in MRI renders, and missing critical events '
                        'in seizure timeline videos.',
            },
            {
                'ref': 'IEC 62304',
                'note': 'Video generation software in medical devices must follow software '
                        'lifecycle processes. Rendering pipeline, frame generation, and '
                        'annotation overlay code require documented V&V procedures.',
            },
            {
                'ref': 'HIPAA',
                'note': 'Generated videos containing patient data are PHI. Video files '
                        'must be encrypted at rest and in transit, access must be logged, '
                        'and videos must be purged per retention policy. Watermarks must '
                        'include patient ID for traceability.',
            },
        ],
        'remediation': [
            {
                'strategy': 'Ground Truth Validation',
                'description': 'Compare generated video frames against source data '
                               'point-by-point. Automated checks verify signal values, '
                               'event timestamps, and annotation accuracy. Flag any '
                               'frame where rendered data deviates > 1% from source.',
            },
            {
                'strategy': 'Temporal Alignment Audit',
                'description': 'Verify that video timeline matches clinical event '
                               'timestamps. Automated tools check frame-to-timestamp '
                               'mapping, seizure onset markers, and medication change '
                               'annotations for temporal accuracy.',
            },
            {
                'strategy': 'Rendering Quality Monitor',
                'description': 'Continuous monitoring of video output quality metrics: '
                               'frame rate stability, resolution consistency, color '
                               'accuracy, and annotation readability. Alert on quality '
                               'drops below clinical threshold.',
            },
            {
                'strategy': 'Clinical Review Gate',
                'description': 'All AI-generated clinical videos require clinician sign-off '
                               'before patient-facing use. Implement review queue with '
                               'approval workflow, rejection reasons, and re-render triggers.',
            },
        ],
    }
