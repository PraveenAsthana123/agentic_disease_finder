"""Image Segmentation AI Dashboard — monitors EEG trace digitization and
image segmentation tasks applied to EEG recordings.

Maps clinical.db tables to segmentation concepts:
- uploads        → EEG images submitted for segmentation (input images)
- mri_findings   → imaging segmentation results (lesion_type = segment class,
                   quality = segmentation quality score)
- analyses       → segmentation-based prediction outputs (signal quality,
                   confidence, channel/waveform data)
- transaction_log (eeg_upload, cv_pipeline, genai_bot) → pipeline events
"""

import sqlite3
import json
import os

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# Quality tier mapping: MRI/imaging quality → segmentation quality score
QUALITY_SCORE_MAP = {
    'Diagnostic': 1.0,
    'Adequate': 0.75,
    'Suboptimal': 0.4,
    'Non-diagnostic': 0.1,
}

# Lesion type → human-readable segment class label
SEGMENT_CLASS_LABELS = {
    'HS': 'Hippocampal Sclerosis',
    'FCD': 'Focal Cortical Dysplasia',
    'NL': 'Non-Lesional',
    'CAV': 'Cavernoma',
    'AVM': 'Arteriovenous Malformation',
    'ENC': 'Encephalitis',
    'NRM': 'Normal',
    'TUM': 'Tumour',
}

# Signal quality → segmentation output quality mapping
SIGNAL_QUALITY_SCORE_MAP = {
    'Good': 1.0,
    'Adequate': 0.75,
    'Poor': 0.35,
}


def _db_query(sql, params=()):
    if not os.path.exists(DB):
        return []
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _load_uploads():
    """Load all uploads — treated as EEG images submitted for segmentation."""
    return _db_query(
        "SELECT id, patient_id, file_name, disease, department, created_at "
        "FROM uploads ORDER BY created_at"
    )


def _load_mri_findings():
    """Load MRI findings — treated as imaging segmentation results."""
    rows = _db_query(
        "SELECT id, patient_id, fields_json, created_at FROM mri_findings ORDER BY created_at"
    )
    parsed = []
    for r in rows:
        try:
            fields = json.loads(r['fields_json']) if r.get('fields_json') else {}
        except (json.JSONDecodeError, TypeError):
            fields = {}
        parsed.append({
            'id': r['id'],
            'patient_id': r['patient_id'],
            'created_at': r['created_at'],
            'quality': fields.get('quality', 'unknown'),
            'lesion_type': fields.get('lesion_type', 'unknown'),
            'lesion_label': fields.get('lesion_label', 'Unknown'),
            'lesion_location': fields.get('lesion_location', 'unknown'),
            'laterality': fields.get('laterality', 'unknown'),
            'classification': fields.get('classification', 'unknown'),
            'classification_label': fields.get('classification_label', 'Unknown'),
            'radiologist_confidence': fields.get('radiologist_confidence', 'unknown'),
            'mri_available': fields.get('mri_available', 'unknown'),
        })
    return parsed


def _load_analyses():
    """Load analyses — treated as segmentation-based prediction outputs."""
    rows = _db_query(
        "SELECT id, upload_id, patient_id, disease, predicted_label, confidence, "
        "signal_quality, result_json, created_at FROM analyses ORDER BY created_at"
    )
    parsed = []
    for r in rows:
        try:
            rj = json.loads(r['result_json']) if r.get('result_json') else {}
        except (json.JSONDecodeError, TypeError):
            rj = {}
        analysis_block = rj.get('analysis', {})
        parsed.append({
            'id': r['id'],
            'upload_id': r['upload_id'],
            'patient_id': r['patient_id'],
            'disease': r['disease'],
            'predicted_label': r['predicted_label'],
            'confidence': r['confidence'],
            'signal_quality': r['signal_quality'],
            'created_at': r['created_at'],
            'n_channels': analysis_block.get('n_channels', 0),
            'sampling_rate': analysis_block.get('sampling_rate', 0),
            'duration_seconds': analysis_block.get('duration_seconds', 0),
            'channels': rj.get('channels', []),
        })
    return parsed


def _load_pipeline_events():
    """Load segmentation pipeline events from transaction_log."""
    return _db_query(
        "SELECT id, component, action, actor, detail, ts_utc, ts_local "
        "FROM transaction_log "
        "WHERE component IN ('eeg_upload', 'cv_pipeline', 'genai_bot') "
        "ORDER BY ts_utc"
    )


def _quality_score(quality_str):
    """Convert quality label to numeric segmentation quality score 0-1."""
    return QUALITY_SCORE_MAP.get(quality_str, 0.5)


def _signal_quality_score(sq):
    return SIGNAL_QUALITY_SCORE_MAP.get(sq, 0.5)


def _avg(values):
    return round(sum(values) / len(values), 4) if values else 0.0


def overview():
    """KPI-level summary for the Image Segmentation AI dashboard."""
    uploads = _load_uploads()
    findings = _load_mri_findings()
    analyses = _load_analyses()
    pipeline_events = _load_pipeline_events()

    if not uploads and not findings and not analyses:
        return {
            'available': False,
            'message': 'No image segmentation data available. Submit EEG uploads or imaging findings first.'
        }

    # --- KPI 1: Total Images Processed (uploads = input EEG image files) ---
    total_images = len(uploads)

    # --- KPI 2: Segmentation Tasks (mri_findings = imaging segmentation runs) ---
    total_seg_tasks = len(findings)

    # --- KPI 3: Avg Quality Score (from mri_findings quality field) ---
    quality_scores = [_quality_score(f['quality']) for f in findings]
    avg_quality = _avg(quality_scores)

    # --- KPI 4: Segment Classes Found (distinct lesion_type values) ---
    segment_classes = {f['lesion_type'] for f in findings if f['lesion_type'] != 'unknown'}
    n_segment_classes = len(segment_classes)

    # --- KPI 5: Patients Covered (distinct across uploads + findings) ---
    all_patient_ids = {u['patient_id'] for u in uploads} | {f['patient_id'] for f in findings}
    n_patients = len(all_patient_ids)

    # --- KPI 6: Lesional Detection Rate (LESIONAL / total classifications) ---
    lesional = sum(1 for f in findings if f['classification'] == 'LESIONAL')
    lesional_rate = round(lesional / max(total_seg_tasks, 1), 4)

    # --- KPI 7: Mean Confidence (from analyses predictions) ---
    conf_values = [a['confidence'] for a in analyses if a.get('confidence') is not None]
    mean_confidence = _avg(conf_values)

    # --- KPI 8: Pipeline Events (eeg_upload + cv_pipeline + genai_bot) ---
    n_pipeline_events = len(pipeline_events)

    # --- Segment Class Distribution ---
    seg_class_dist = {}
    for f in findings:
        lt = f['lesion_type']
        label = SEGMENT_CLASS_LABELS.get(lt, lt)
        seg_class_dist[label] = seg_class_dist.get(label, 0) + 1

    # --- Quality Distribution ---
    quality_dist = {}
    for f in findings:
        q = f['quality']
        quality_dist[q] = quality_dist.get(q, 0) + 1

    # --- Patient Coverage ---
    upload_patients = {u['patient_id'] for u in uploads}
    finding_patients = {f['patient_id'] for f in findings}
    analysis_patients = {a['patient_id'] for a in analyses}

    # --- Signal quality distribution from analyses ---
    sig_quality_dist = {}
    for a in analyses:
        sq = a.get('signal_quality') or 'unknown'
        sig_quality_dist[sq] = sig_quality_dist.get(sq, 0) + 1

    # --- Overall segmentation health score ---
    # Weights: avg_quality (40%) + lesional_rate (30%) + mean_confidence (30%)
    health_score = round(avg_quality * 0.4 + lesional_rate * 0.3 + mean_confidence * 0.3, 4)
    health_pct = round(health_score * 100, 1)

    if health_pct >= 75:
        verdict = 'HIGH QUALITY'
        health_color = '#10b981'
    elif health_pct >= 50:
        verdict = 'MODERATE QUALITY'
        health_color = '#f59e0b'
    else:
        verdict = 'REVIEW NEEDED'
        health_color = '#ef4444'

    return {
        'available': True,
        'verdict': verdict,
        'health_score': health_pct,
        'total_images': total_images,
        'total_seg_tasks': total_seg_tasks,
        'avg_quality': avg_quality,
        'n_segment_classes': n_segment_classes,
        'n_patients': n_patients,
        'lesional_rate': lesional_rate,
        'mean_confidence': mean_confidence,
        'n_pipeline_events': n_pipeline_events,
        'seg_class_distribution': seg_class_dist,
        'quality_distribution': quality_dist,
        'signal_quality_distribution': sig_quality_dist,
        'patient_coverage': {
            'total': n_patients,
            'from_uploads': len(upload_patients),
            'from_findings': len(finding_patients),
            'from_analyses': len(analysis_patients),
        },
        'kpis': [
            {
                'label': 'Total Images Processed',
                'value': str(total_images),
                'color': '#6366f1',
            },
            {
                'label': 'Segmentation Tasks',
                'value': str(total_seg_tasks),
                'color': '#8b5cf6',
            },
            {
                'label': 'Avg Quality Score',
                'value': f'{avg_quality:.2f}',
                'color': '#10b981' if avg_quality >= 0.75 else '#f59e0b' if avg_quality >= 0.4 else '#ef4444',
            },
            {
                'label': 'Segment Classes Found',
                'value': str(n_segment_classes),
                'color': '#3b82f6',
            },
            {
                'label': 'Patients Covered',
                'value': str(n_patients),
                'color': '#0ea5e9',
            },
            {
                'label': 'Lesional Detection Rate',
                'value': f'{lesional_rate * 100:.1f}%',
                'color': '#10b981' if lesional_rate >= 0.5 else '#f59e0b',
            },
            {
                'label': 'Mean Confidence',
                'value': f'{mean_confidence:.4f}',
                'color': '#10b981' if mean_confidence >= 0.65 else '#f59e0b' if mean_confidence >= 0.45 else '#ef4444',
            },
            {
                'label': 'Pipeline Events',
                'value': str(n_pipeline_events),
                'color': '#64748b',
            },
        ],
    }


def breakdown():
    """Detailed breakdown — per-patient segmentation, segment type distribution,
    timeline, quality matrix, file inventory."""
    uploads = _load_uploads()
    findings = _load_mri_findings()
    analyses = _load_analyses()
    pipeline_events = _load_pipeline_events()

    if not uploads and not findings and not analyses:
        return {'available': False}

    # --- Per-patient segmentation details ---
    patient_seg = {}
    for f in findings:
        pid = f['patient_id']
        if pid not in patient_seg:
            patient_seg[pid] = {
                'patient_id': pid,
                'seg_tasks': 0,
                'quality_scores': [],
                'lesion_types': [],
                'classifications': [],
                'lesional': 0,
            }
        qs = _quality_score(f['quality'])
        patient_seg[pid]['seg_tasks'] += 1
        patient_seg[pid]['quality_scores'].append(qs)
        patient_seg[pid]['lesion_types'].append(
            SEGMENT_CLASS_LABELS.get(f['lesion_type'], f['lesion_type'])
        )
        patient_seg[pid]['classifications'].append(f['classification'])
        if f['classification'] == 'LESIONAL':
            patient_seg[pid]['lesional'] += 1

    per_patient = []
    for pid, info in sorted(patient_seg.items()):
        qs = info['quality_scores']
        per_patient.append({
            'patient_id': pid,
            'seg_tasks': info['seg_tasks'],
            'avg_quality': _avg(qs),
            'lesional': info['lesional'],
            'lesional_rate': round(info['lesional'] / max(info['seg_tasks'], 1), 4),
            'primary_class': max(set(info['lesion_types']), key=info['lesion_types'].count) if info['lesion_types'] else 'N/A',
        })

    # --- Segment type distribution (with counts and percentages) ---
    seg_type_counts = {}
    total_findings = len(findings)
    for f in findings:
        lt = f['lesion_type']
        label = SEGMENT_CLASS_LABELS.get(lt, lt)
        if label not in seg_type_counts:
            seg_type_counts[label] = {'short': lt, 'count': 0}
        seg_type_counts[label]['count'] += 1
    seg_type_distribution = [
        {
            'class': label,
            'short': info['short'],
            'count': info['count'],
            'pct': round(info['count'] / max(total_findings, 1) * 100, 1),
        }
        for label, info in sorted(seg_type_counts.items(), key=lambda x: -x[1]['count'])
    ]

    # --- Timeline: uploads + findings + analyses ordered by date ---
    timeline_events = []
    for u in uploads:
        timeline_events.append({
            'ts': u['created_at'],
            'type': 'image_upload',
            'patient_id': u['patient_id'],
            'detail': f"EEG image uploaded: {u['file_name']} ({u['disease']})",
        })
    for f in findings:
        timeline_events.append({
            'ts': f['created_at'],
            'type': 'segmentation_task',
            'patient_id': f['patient_id'],
            'detail': f"Segment class: {SEGMENT_CLASS_LABELS.get(f['lesion_type'], f['lesion_type'])} | Quality: {f['quality']} | Class: {f['classification_label']}",
        })
    for a in analyses:
        timeline_events.append({
            'ts': a['created_at'],
            'type': 'seg_prediction',
            'patient_id': a['patient_id'],
            'detail': f"Prediction: {a['predicted_label']} | Confidence: {a['confidence']:.3f} | Signal: {a['signal_quality']}",
        })
    timeline_events.sort(key=lambda e: e['ts'])

    # --- Quality matrix: quality label × classification cross-tab ---
    quality_matrix = {}
    for f in findings:
        q = f['quality']
        cl = f['classification']
        if q not in quality_matrix:
            quality_matrix[q] = {}
        quality_matrix[q][cl] = quality_matrix[q].get(cl, 0) + 1
    quality_matrix_rows = [
        {'quality': q, **cls_counts}
        for q, cls_counts in sorted(quality_matrix.items(), key=lambda x: -QUALITY_SCORE_MAP.get(x[0], 0))
    ]

    # --- File inventory (uploads) ---
    file_inventory = [
        {
            'id': u['id'],
            'patient_id': u['patient_id'],
            'file_name': u['file_name'],
            'disease': u['disease'],
            'department': u['department'],
            'created_at': u['created_at'],
        }
        for u in uploads
    ]

    # --- Per-analysis segmentation output table ---
    analysis_table = [
        {
            'id': a['id'],
            'patient_id': a['patient_id'],
            'disease': a['disease'],
            'predicted_label': a['predicted_label'],
            'confidence': a['confidence'],
            'signal_quality': a['signal_quality'],
            'n_channels': a['n_channels'],
            'duration_seconds': a['duration_seconds'],
            'sampling_rate': a['sampling_rate'],
            'created_at': a['created_at'],
        }
        for a in analyses
    ]

    # --- Pipeline event log ---
    pipeline_log = [
        {
            'ts': ev.get('ts_local') or ev.get('ts_utc', ''),
            'component': ev['component'],
            'action': ev['action'],
            'actor': ev.get('actor', ''),
            'detail': ev.get('detail', ''),
        }
        for ev in pipeline_events
    ]

    # --- Pipeline event breakdown by component ---
    pipeline_by_component = {}
    for ev in pipeline_events:
        comp = ev['component']
        action = ev['action']
        if comp not in pipeline_by_component:
            pipeline_by_component[comp] = {'total': 0, 'actions': {}}
        pipeline_by_component[comp]['total'] += 1
        pipeline_by_component[comp]['actions'][action] = pipeline_by_component[comp]['actions'].get(action, 0) + 1
    pipeline_component_chart = [
        {'component': comp, 'total': info['total'], **{f'action_{k}': v for k, v in info['actions'].items()}}
        for comp, info in pipeline_by_component.items()
    ]

    # --- Channel stats from analyses (waveform digitization output) ---
    total_channels = sum(a['n_channels'] for a in analyses)
    avg_channels = round(total_channels / max(len(analyses), 1), 1)
    avg_duration = round(
        sum(a['duration_seconds'] for a in analyses) / max(len(analyses), 1), 1
    )

    return {
        'available': True,
        'per_patient': per_patient,
        'seg_type_distribution': seg_type_distribution,
        'timeline': timeline_events,
        'quality_matrix': quality_matrix_rows,
        'file_inventory': file_inventory,
        'analysis_table': analysis_table,
        'pipeline_log': pipeline_log,
        'pipeline_component_chart': pipeline_component_chart,
        'waveform_stats': {
            'total_channels_digitized': total_channels,
            'avg_channels_per_image': avg_channels,
            'avg_duration_seconds': avg_duration,
            'n_analyses': len(analyses),
        },
        'n_uploads': len(uploads),
        'n_findings': len(findings),
        'n_analyses': len(analyses),
        'n_pipeline_events': len(pipeline_events),
    }


def definitions():
    """Definitions tab for the Image Segmentation AI dashboard."""
    return {
        'sections': [
            {
                'title': 'Image Segmentation Concepts',
                'items': [
                    {
                        'term': 'Image Segmentation',
                        'definition': (
                            'The process of partitioning a digital image into multiple segments '
                            '(regions of interest) to simplify representation. In EEG digitization, '
                            'segmentation isolates individual trace channels, artifact regions, and '
                            'annotation fields from scanned paper EEG recordings.'
                        ),
                    },
                    {
                        'term': 'ROI Extraction (Region of Interest)',
                        'definition': (
                            'Automated detection and extraction of clinically relevant sub-regions '
                            'within an EEG scan image — e.g., isolating a single EEG channel strip, '
                            'a seizure discharge region, or a scale/calibration marker. ROI bounding '
                            'boxes are the primary output of the segmentation stage.'
                        ),
                    },
                    {
                        'term': 'Trace Digitization',
                        'definition': (
                            'Converting an EEG waveform from pixel coordinates (raster image) into '
                            'a numerical time-series. After segmentation identifies the trace region, '
                            'digitization maps pixel row positions to amplitude values and assigns '
                            'timestamps based on the detected time-axis calibration.'
                        ),
                    },
                    {
                        'term': 'U-Net',
                        'definition': (
                            'A fully convolutional encoder-decoder neural network originally designed '
                            'for biomedical image segmentation. U-Net uses skip connections between '
                            'encoder and decoder layers to preserve spatial detail, making it effective '
                            'for segmenting thin EEG trace lines against noisy paper backgrounds.'
                        ),
                    },
                    {
                        'term': 'Mask R-CNN',
                        'definition': (
                            'An instance segmentation framework that extends Faster R-CNN with a '
                            'parallel branch for predicting per-instance pixel masks. Used to '
                            'independently segment overlapping EEG channels and distinguish them '
                            'from artifact bleed-through or grid lines on scanned records.'
                        ),
                    },
                    {
                        'term': 'Intersection over Union (IoU)',
                        'definition': (
                            'The ratio of the overlap area to the union area between a predicted '
                            'segment mask and the ground-truth mask. IoU >= 0.5 is conventionally '
                            'the threshold for a correct detection. Higher IoU indicates the model '
                            'is accurately localising the EEG trace region.'
                        ),
                    },
                    {
                        'term': 'Dice Coefficient',
                        'definition': (
                            'A spatial overlap metric computed as 2 * |A ∩ B| / (|A| + |B|). '
                            'Preferred over IoU for thin or elongated structures like EEG trace lines '
                            'because it is more sensitive to small overlap differences. Dice = 0 means '
                            'no overlap; Dice = 1 means perfect segmentation.'
                        ),
                    },
                    {
                        'term': 'Boundary Detection',
                        'definition': (
                            'A post-segmentation step that locates the precise pixel edges of each '
                            'EEG channel region. Accurate boundary detection is required for correct '
                            'amplitude extraction — a boundary error of 1 mm at standard EEG paper '
                            'speed translates directly to a waveform amplitude error.'
                        ),
                    },
                ],
            },
            {
                'title': 'Segmentation Quality Metrics',
                'items': [
                    {
                        'term': 'Diagnostic Quality',
                        'definition': (
                            'Segmentation output meets full clinical interpretation standards. '
                            'Trace lines are cleanly separated, calibration markers are detected, '
                            'and waveform amplitudes are within expected physiological ranges. '
                            'Quality score: 1.0.'
                        ),
                    },
                    {
                        'term': 'Adequate Quality',
                        'definition': (
                            'Segmentation output is usable for clinical review with minor reservations. '
                            'Some trace segments may have reduced boundary precision or minor artifact '
                            'overlap. Quality score: 0.75.'
                        ),
                    },
                    {
                        'term': 'Suboptimal Quality',
                        'definition': (
                            'Segmentation output shows notable degradation — possibly due to scan '
                            'quality, pen bleed-through, or paper fold artefacts. Manual review '
                            'is recommended before clinical use. Quality score: 0.4.'
                        ),
                    },
                    {
                        'term': 'Non-Diagnostic',
                        'definition': (
                            'Segmentation failed to produce clinically usable output. Image quality '
                            'is insufficient for reliable trace extraction. Re-scan or manual '
                            'digitization is required. Quality score: 0.1.'
                        ),
                    },
                    {
                        'term': 'Lesional Detection Rate',
                        'definition': (
                            'Proportion of segmentation tasks that identified a lesional pattern '
                            '(LESIONAL classification). A high rate reflects accurate identification '
                            'of structurally abnormal regions in the imaging data supporting EEG '
                            'source localization.'
                        ),
                    },
                ],
            },
            {
                'title': 'Segment Class Types',
                'items': [
                    {'term': 'HS — Hippocampal Sclerosis', 'definition': 'Mesial temporal sclerosis with hippocampal atrophy and T2/FLAIR signal increase. Most common lesional substrate in temporal lobe epilepsy.'},
                    {'term': 'FCD — Focal Cortical Dysplasia', 'definition': 'Malformation of cortical development causing architectural disruption. Frequent cause of drug-resistant focal epilepsy.'},
                    {'term': 'CAV — Cavernoma', 'definition': 'Cavernous malformation with characteristic popcorn MRI appearance and hemosiderin ring. Associated with focal seizures and cortical irritation.'},
                    {'term': 'AVM — Arteriovenous Malformation', 'definition': 'Tangle of abnormal blood vessels causing cortical irritation. Segmentation identifies the flow-void nidus region on imaging.'},
                    {'term': 'ENC — Encephalitis', 'definition': 'Inflammatory lesion pattern associated with autoimmune or infectious encephalitis. Detected via T2/FLAIR signal abnormality on imaging.'},
                    {'term': 'NL — Non-Lesional', 'definition': 'No structural lesion identified on imaging. EEG segmentation focuses on functional patterns rather than structural abnormalities.'},
                    {'term': 'NRM — Normal', 'definition': 'Imaging within normal limits. Segmentation result confirms absence of pathological regions.'},
                    {'term': 'TUM — Tumour', 'definition': 'Primary or metastatic brain tumour causing perilesional cortical irritation and epileptiform activity.'},
                ],
            },
            {
                'title': 'Compliance References',
                'items': [
                    {
                        'term': 'FDA AI/ML Action Plan',
                        'definition': (
                            'The FDA\'s framework for AI/ML-based software as a medical device (SaMD) '
                            'requires ongoing performance monitoring of imaging AI outputs. Image '
                            'segmentation dashboards satisfy the Real-World Performance (RWP) '
                            'monitoring requirement under the Predetermined Change Control Plan (PCCP).'
                        ),
                    },
                    {
                        'term': 'EU AI Act — Article 10',
                        'definition': (
                            'High-risk AI systems (including medical imaging AI) must use training, '
                            'validation, and test datasets that are relevant, representative, and '
                            'free of errors. Segmentation quality tracking validates that the AI '
                            'output meets Article 10 data governance requirements.'
                        ),
                    },
                    {
                        'term': 'ISO 14971 — Medical Device Risk Management',
                        'definition': (
                            'Requires identification and control of risks throughout the device lifecycle. '
                            'Suboptimal or non-diagnostic segmentation output is a residual risk that '
                            'must be tracked, trended, and reported as part of post-market surveillance.'
                        ),
                    },
                    {
                        'term': 'IEC 62304 — Medical Device Software Lifecycle',
                        'definition': (
                            'Defines software development and maintenance processes for medical devices. '
                            'Image segmentation pipelines must maintain audit logs, version tracking, '
                            'and anomaly detection as part of IEC 62304 Class B/C software obligations.'
                        ),
                    },
                    {
                        'term': 'HIPAA — Protected Health Information',
                        'definition': (
                            'EEG scan images containing patient identifiers (name labels, DOB markers, '
                            'hospital IDs) are ePHI. Segmentation pipelines must redact or de-identify '
                            'annotation regions prior to storage and downstream model training to comply '
                            'with HIPAA Privacy and Security Rules.'
                        ),
                    },
                ],
            },
            {
                'title': 'Remediation Strategies',
                'items': [
                    {
                        'term': 'Suboptimal Scan Resolution',
                        'definition': (
                            'If quality scores are consistently Suboptimal or Non-diagnostic, rescan '
                            'at higher DPI (>=300 dpi recommended), apply adaptive histogram '
                            'equalisation before segmentation, and flag originals for manual review.'
                        ),
                    },
                    {
                        'term': 'Channel Boundary Overlap',
                        'definition': (
                            'When Dice scores fall below threshold due to adjacent channel bleed, '
                            'apply morphological erosion post-mask to separate touching regions, '
                            'or switch from semantic to instance segmentation (Mask R-CNN).'
                        ),
                    },
                    {
                        'term': 'Low Lesional Detection Rate',
                        'definition': (
                            'If lesional detection rate drops unexpectedly, audit the image '
                            'pre-processing pipeline for contrast normalisation drift, retrain '
                            'segment classifiers on updated lesion exemplars, and cross-validate '
                            'with radiologist-confirmed ground truth.'
                        ),
                    },
                    {
                        'term': 'Pipeline Event Failures',
                        'definition': (
                            'Monitor the cv_pipeline:process event log for error spikes. Elevated '
                            'failure rates may indicate upstream format changes (DICOM → JPEG '
                            'conversion artefacts), model version mismatches, or GPU memory '
                            'pressure causing partial inference. Scale compute resources accordingly.'
                        ),
                    },
                ],
            },
        ]
    }
