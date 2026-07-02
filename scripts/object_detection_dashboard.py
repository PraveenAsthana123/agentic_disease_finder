"""Object Detection AI Dashboard — monitors body-movement and lesion
detection applied to video-EEG and MRI imaging data.

Maps clinical.db tables to object detection concepts:
- uploads        → video/EEG frames submitted for detection
- mri_findings   → detected objects (lesion_type = detection class,
                   lesion_location = bounding box region,
                   radiologist_confidence = detection confidence)
- analyses       → detection-based prediction outputs (confidence,
                   predicted_label, signal quality)
- transaction_log (cv_pipeline) → detection pipeline events
"""

import sqlite3
import json
import os

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')

# Detection class mapping from lesion_type
DETECTION_CLASS_LABELS = {
    'HS': 'Hippocampal Sclerosis',
    'FCD': 'Focal Cortical Dysplasia',
    'NL': 'Non-Lesional',
    'CAV': 'Cavernoma',
    'AVM': 'Arteriovenous Malformation',
    'ENC': 'Encephalitis',
    'NRM': 'Normal',
    'TUM': 'Tumour',
}

# Confidence tiers for detection quality
CONFIDENCE_TIERS = {
    'high': (0.75, 1.0),
    'medium': (0.5, 0.75),
    'low': (0.25, 0.5),
    'very_low': (0.0, 0.25),
}

# Radiologist confidence → detection confidence score
RADIOLOGIST_CONF_MAP = {
    'High': 0.9,
    'Moderate': 0.65,
    'Low': 0.35,
}

# Quality → IoU proxy score
QUALITY_IOU_MAP = {
    'Diagnostic': 0.92,
    'Adequate': 0.75,
    'Suboptimal': 0.45,
    'Non-diagnostic': 0.15,
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
    """Load all uploads — treated as frames/images submitted for detection."""
    return _db_query(
        "SELECT id, patient_id, file_name, disease, department, created_at "
        "FROM uploads ORDER BY created_at"
    )


def _load_mri_findings():
    """Load MRI findings — treated as detected objects with bounding regions."""
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
    """Load analyses — treated as detection model prediction outputs."""
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
        })
    return parsed


def _load_pipeline_events():
    """Load detection pipeline events from transaction_log."""
    return _db_query(
        "SELECT id, component, action, actor, detail, ts_utc, ts_local "
        "FROM transaction_log "
        "WHERE component = 'cv_pipeline' "
        "ORDER BY ts_utc"
    )


def _avg(values):
    return round(sum(values) / len(values), 4) if values else 0.0


def _confidence_tier(conf):
    """Map numeric confidence to tier label."""
    if conf >= 0.75:
        return 'high'
    elif conf >= 0.5:
        return 'medium'
    elif conf >= 0.25:
        return 'low'
    return 'very_low'


def overview():
    """KPI-level summary for the Object Detection AI dashboard."""
    uploads = _load_uploads()
    findings = _load_mri_findings()
    analyses = _load_analyses()
    pipeline_events = _load_pipeline_events()

    if not uploads and not findings and not analyses:
        return {
            'available': False,
            'message': 'No object detection data available. Submit video-EEG or imaging data first.'
        }

    # KPI 1: Total Frames Processed
    total_frames = len(uploads)

    # KPI 2: Objects Detected (mri_findings with lesion_type != NL/NRM)
    detected_objects = [f for f in findings if f['lesion_type'] not in ('NL', 'NRM', 'unknown')]
    n_detections = len(detected_objects)

    # KPI 3: Detection Classes (distinct lesion types)
    det_classes = {f['lesion_type'] for f in findings if f['lesion_type'] != 'unknown'}
    n_classes = len(det_classes)

    # KPI 4: Mean Detection Confidence (from radiologist_confidence)
    conf_values = [RADIOLOGIST_CONF_MAP.get(f['radiologist_confidence'], 0.5) for f in findings]
    mean_det_conf = _avg(conf_values)

    # KPI 5: Mean IoU Proxy (from quality → IoU map)
    iou_values = [QUALITY_IOU_MAP.get(f['quality'], 0.5) for f in findings]
    mean_iou = _avg(iou_values)

    # KPI 6: Patients Covered
    all_patients = {u['patient_id'] for u in uploads} | {f['patient_id'] for f in findings}
    n_patients = len(all_patients)

    # KPI 7: Model Confidence (from analyses)
    model_conf = [a['confidence'] for a in analyses if a.get('confidence') is not None]
    mean_model_conf = _avg(model_conf)

    # KPI 8: Pipeline Events
    n_pipeline = len(pipeline_events)

    # Detection Rate (objects found / total findings)
    detection_rate = round(n_detections / max(len(findings), 1), 4)

    # Detection class distribution
    class_dist = {}
    for f in findings:
        lt = f['lesion_type']
        label = DETECTION_CLASS_LABELS.get(lt, lt)
        class_dist[label] = class_dist.get(label, 0) + 1

    # Confidence distribution (from analyses)
    conf_dist = {'high': 0, 'medium': 0, 'low': 0, 'very_low': 0}
    for a in analyses:
        if a.get('confidence') is not None:
            tier = _confidence_tier(a['confidence'])
            conf_dist[tier] += 1

    # Quality (IoU proxy) distribution
    iou_dist = {}
    for f in findings:
        q = f['quality']
        iou_dist[q] = iou_dist.get(q, 0) + 1

    # Location distribution (bounding box region)
    location_dist = {}
    for f in findings:
        loc = f['lesion_location']
        if loc and loc != 'unknown':
            location_dist[loc] = location_dist.get(loc, 0) + 1

    # Health score: mean_det_conf (30%) + mean_iou (40%) + detection_rate (30%)
    health_score = round(mean_det_conf * 0.3 + mean_iou * 0.4 + detection_rate * 0.3, 4)
    health_pct = round(health_score * 100, 1)

    if health_pct >= 75:
        verdict = 'HIGH ACCURACY'
        health_color = '#10b981'
    elif health_pct >= 50:
        verdict = 'MODERATE ACCURACY'
        health_color = '#f59e0b'
    else:
        verdict = 'REVIEW NEEDED'
        health_color = '#ef4444'

    return {
        'available': True,
        'verdict': verdict,
        'health_score': health_pct,
        'total_frames': total_frames,
        'n_detections': n_detections,
        'n_classes': n_classes,
        'mean_det_confidence': mean_det_conf,
        'mean_iou': mean_iou,
        'n_patients': n_patients,
        'mean_model_confidence': mean_model_conf,
        'n_pipeline_events': n_pipeline,
        'detection_rate': detection_rate,
        'class_distribution': class_dist,
        'confidence_distribution': conf_dist,
        'iou_distribution': iou_dist,
        'location_distribution': location_dist,
        'kpis': [
            {
                'label': 'Frames Processed',
                'value': str(total_frames),
                'color': '#6366f1',
            },
            {
                'label': 'Objects Detected',
                'value': str(n_detections),
                'color': '#ef4444',
            },
            {
                'label': 'Detection Classes',
                'value': str(n_classes),
                'color': '#3b82f6',
            },
            {
                'label': 'Mean Det. Confidence',
                'value': f'{mean_det_conf:.2f}',
                'color': '#10b981' if mean_det_conf >= 0.7 else '#f59e0b' if mean_det_conf >= 0.5 else '#ef4444',
            },
            {
                'label': 'Mean IoU Score',
                'value': f'{mean_iou:.2f}',
                'color': '#10b981' if mean_iou >= 0.7 else '#f59e0b' if mean_iou >= 0.5 else '#ef4444',
            },
            {
                'label': 'Patients Covered',
                'value': str(n_patients),
                'color': '#0ea5e9',
            },
            {
                'label': 'Model Confidence',
                'value': f'{mean_model_conf:.4f}',
                'color': '#10b981' if mean_model_conf >= 0.65 else '#f59e0b' if mean_model_conf >= 0.45 else '#ef4444',
            },
            {
                'label': 'Pipeline Events',
                'value': str(n_pipeline),
                'color': '#64748b',
            },
        ],
    }


def breakdown():
    """Detailed breakdown — detection inventory, per-patient, location map,
    confidence matrix, pipeline events."""
    uploads = _load_uploads()
    findings = _load_mri_findings()
    analyses = _load_analyses()
    pipeline_events = _load_pipeline_events()

    if not uploads and not findings and not analyses:
        return {'available': False}

    # Detection inventory table (from mri_findings)
    detection_inventory = []
    for f in findings:
        det_conf = RADIOLOGIST_CONF_MAP.get(f['radiologist_confidence'], 0.5)
        iou_proxy = QUALITY_IOU_MAP.get(f['quality'], 0.5)
        detection_inventory.append({
            'id': f['id'],
            'patient_id': f['patient_id'],
            'detection_class': DETECTION_CLASS_LABELS.get(f['lesion_type'], f['lesion_type']),
            'class_code': f['lesion_type'],
            'region': f['lesion_location'],
            'laterality': f['laterality'],
            'det_confidence': det_conf,
            'iou_score': iou_proxy,
            'classification': f['classification'],
            'quality': f['quality'],
            'created_at': f['created_at'],
        })

    # Per-patient detection summary
    patient_det = {}
    for f in findings:
        pid = f['patient_id']
        if pid not in patient_det:
            patient_det[pid] = {
                'patient_id': pid,
                'n_detections': 0,
                'classes': [],
                'regions': [],
                'confidences': [],
            }
        patient_det[pid]['n_detections'] += 1
        cls_label = DETECTION_CLASS_LABELS.get(f['lesion_type'], f['lesion_type'])
        if cls_label not in patient_det[pid]['classes']:
            patient_det[pid]['classes'].append(cls_label)
        region = f['lesion_location']
        if region and region != 'unknown' and region not in patient_det[pid]['regions']:
            patient_det[pid]['regions'].append(region)
        patient_det[pid]['confidences'].append(
            RADIOLOGIST_CONF_MAP.get(f['radiologist_confidence'], 0.5)
        )

    # Add analysis data to patient profiles
    for a in analyses:
        pid = a['patient_id']
        if pid not in patient_det:
            patient_det[pid] = {
                'patient_id': pid,
                'n_detections': 0,
                'classes': [],
                'regions': [],
                'confidences': [],
            }

    patient_uploads = {}
    for u in uploads:
        pid = u['patient_id']
        patient_uploads[pid] = patient_uploads.get(pid, 0) + 1

    patient_analyses = {}
    patient_conf = {}
    for a in analyses:
        pid = a['patient_id']
        patient_analyses[pid] = patient_analyses.get(pid, 0) + 1
        if a.get('confidence') is not None:
            if pid not in patient_conf:
                patient_conf[pid] = []
            patient_conf[pid].append(a['confidence'])

    per_patient = []
    for pid, info in sorted(patient_det.items()):
        per_patient.append({
            'patient_id': pid,
            'n_frames': patient_uploads.get(pid, 0),
            'n_detections': info['n_detections'],
            'n_analyses': patient_analyses.get(pid, 0),
            'classes': info['classes'],
            'regions': info['regions'],
            'mean_confidence': _avg(info['confidences']),
            'mean_model_conf': _avg(patient_conf.get(pid, [])),
        })

    # Detection class breakdown with stats
    class_stats = {}
    for f in findings:
        lt = f['lesion_type']
        label = DETECTION_CLASS_LABELS.get(lt, lt)
        if label not in class_stats:
            class_stats[label] = {'code': lt, 'count': 0, 'confidences': [], 'ious': []}
        class_stats[label]['count'] += 1
        class_stats[label]['confidences'].append(
            RADIOLOGIST_CONF_MAP.get(f['radiologist_confidence'], 0.5)
        )
        class_stats[label]['ious'].append(QUALITY_IOU_MAP.get(f['quality'], 0.5))

    class_breakdown = [
        {
            'class': label,
            'code': info['code'],
            'count': info['count'],
            'pct': round(info['count'] / max(len(findings), 1) * 100, 1),
            'mean_confidence': _avg(info['confidences']),
            'mean_iou': _avg(info['ious']),
        }
        for label, info in sorted(class_stats.items(), key=lambda x: -x[1]['count'])
    ]

    # Timeline of detection events
    timeline = []
    for u in uploads:
        timeline.append({
            'ts': u['created_at'],
            'type': 'frame_input',
            'patient_id': u['patient_id'],
            'detail': f"Frame submitted: {u['file_name']} ({u['disease']})",
        })
    for f in findings:
        timeline.append({
            'ts': f['created_at'],
            'type': 'detection',
            'patient_id': f['patient_id'],
            'detail': f"Detected: {DETECTION_CLASS_LABELS.get(f['lesion_type'], f['lesion_type'])} at {f['lesion_location']} | Conf: {f['radiologist_confidence']}",
        })
    for a in analyses:
        timeline.append({
            'ts': a['created_at'],
            'type': 'prediction',
            'patient_id': a['patient_id'],
            'detail': f"Prediction: {a['predicted_label']} | Conf: {a['confidence']:.3f} | Signal: {a['signal_quality']}",
        })
    timeline.sort(key=lambda e: e['ts'])

    # Analysis outputs table
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
            'created_at': a['created_at'],
        }
        for a in analyses
    ]

    # Pipeline event log (cv_pipeline only — computer vision = object detection)
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

    # Pipeline action breakdown
    action_counts = {}
    for ev in pipeline_events:
        act = ev['action']
        action_counts[act] = action_counts.get(act, 0) + 1
    pipeline_action_chart = [
        {'action': act, 'count': cnt}
        for act, cnt in sorted(action_counts.items(), key=lambda x: -x[1])
    ]

    # Location heatmap data (region → count)
    loc_counts = {}
    for f in findings:
        loc = f['lesion_location']
        if loc and loc != 'unknown':
            loc_counts[loc] = loc_counts.get(loc, 0) + 1
    location_heatmap = [
        {'region': loc, 'count': cnt}
        for loc, cnt in sorted(loc_counts.items(), key=lambda x: -x[1])
    ]

    return {
        'available': True,
        'detection_inventory': detection_inventory,
        'per_patient': per_patient,
        'class_breakdown': class_breakdown,
        'timeline': timeline,
        'analysis_table': analysis_table,
        'pipeline_log': pipeline_log,
        'pipeline_action_chart': pipeline_action_chart,
        'location_heatmap': location_heatmap,
        'n_uploads': len(uploads),
        'n_findings': len(findings),
        'n_analyses': len(analyses),
        'n_pipeline_events': len(pipeline_events),
    }


def definitions():
    """Definitions tab for the Object Detection AI dashboard."""
    return {
        'sections': [
            {
                'title': 'Object Detection Concepts',
                'items': [
                    {
                        'term': 'Object Detection',
                        'definition': (
                            'A computer vision task that identifies and localises objects within '
                            'images or video frames by predicting bounding boxes and class labels. '
                            'In epilepsy monitoring, object detection identifies seizure-related '
                            'body movements, patient limb positions, and lesion regions in MRI scans.'
                        ),
                    },
                    {
                        'term': 'Bounding Box',
                        'definition': (
                            'A rectangular region defined by (x, y, width, height) that encloses '
                            'a detected object. In body-movement detection, bounding boxes track '
                            'limb positions frame-by-frame. In MRI analysis, they delineate the '
                            'spatial extent of a detected lesion.'
                        ),
                    },
                    {
                        'term': 'YOLO (You Only Look Once)',
                        'definition': (
                            'A real-time object detection architecture that processes entire images '
                            'in a single forward pass, predicting bounding boxes and class '
                            'probabilities simultaneously. YOLOv8/v9 variants are used for '
                            'seizure movement detection due to their low latency (< 30ms/frame).'
                        ),
                    },
                    {
                        'term': 'Intersection over Union (IoU)',
                        'definition': (
                            'The ratio of overlap area to union area between predicted and ground-truth '
                            'bounding boxes. IoU >= 0.5 is the standard threshold for a true positive '
                            'detection. Higher IoU indicates more precise object localisation.'
                        ),
                    },
                    {
                        'term': 'Non-Maximum Suppression (NMS)',
                        'definition': (
                            'A post-processing step that removes redundant overlapping bounding boxes '
                            'by keeping only the highest-confidence detection for each object. Critical '
                            'for avoiding duplicate lesion detections when multiple anchors fire on '
                            'the same region.'
                        ),
                    },
                    {
                        'term': 'Anchor Boxes',
                        'definition': (
                            'Predefined bounding box templates of various aspect ratios and scales '
                            'that the detector refines to fit actual objects. Anchor shapes are tuned '
                            'to typical lesion morphologies (round for cavernoma, elongated for FCD).'
                        ),
                    },
                    {
                        'term': 'Feature Pyramid Network (FPN)',
                        'definition': (
                            'A multi-scale feature extraction architecture that detects objects at '
                            'different sizes by combining low-resolution semantically strong features '
                            'with high-resolution spatially precise features. Essential for detecting '
                            'both large (tumour) and small (cavernoma) lesions in the same scan.'
                        ),
                    },
                    {
                        'term': 'Mean Average Precision (mAP)',
                        'definition': (
                            'The primary metric for object detection performance, computed as the '
                            'mean of per-class average precision values. mAP@0.5 uses IoU >= 0.5; '
                            'mAP@0.5:0.95 averages across IoU thresholds for stricter evaluation. '
                            'Clinical detection models target mAP@0.5 >= 0.80.'
                        ),
                    },
                ],
            },
            {
                'title': 'Detection Quality Metrics',
                'items': [
                    {
                        'term': 'Detection Confidence',
                        'definition': (
                            'The model\'s probability estimate that a detected region contains an '
                            'object of the predicted class. High confidence (>= 0.75) indicates '
                            'reliable detection. Values below 0.5 trigger manual radiologist review.'
                        ),
                    },
                    {
                        'term': 'True Positive Rate (Recall)',
                        'definition': (
                            'The proportion of actual objects correctly detected by the model. '
                            'In clinical use, high recall is critical — a missed lesion (false '
                            'negative) has more severe consequences than a false alarm.'
                        ),
                    },
                    {
                        'term': 'Precision',
                        'definition': (
                            'The proportion of detected objects that are true positives. Low '
                            'precision means the model generates many false alarms, increasing '
                            'radiologist workload for review.'
                        ),
                    },
                    {
                        'term': 'Detection Rate',
                        'definition': (
                            'Proportion of all imaging assessments where an object (lesion, '
                            'movement, abnormality) was positively identified. Tracks the '
                            'overall sensitivity of the detection pipeline.'
                        ),
                    },
                    {
                        'term': 'Localisation Accuracy',
                        'definition': (
                            'How precisely the bounding box or region label matches the true '
                            'anatomical location. Measured via IoU for bounding boxes or region '
                            'concordance for label-based localisation.'
                        ),
                    },
                ],
            },
            {
                'title': 'Detection Class Types',
                'items': [
                    {'term': 'HS — Hippocampal Sclerosis', 'definition': 'Mesial temporal sclerosis with hippocampal atrophy. Detection focuses on volume asymmetry and T2/FLAIR signal changes.'},
                    {'term': 'FCD — Focal Cortical Dysplasia', 'definition': 'Cortical malformation causing architectural disruption. Detected via blurring of grey-white junction and cortical thickening.'},
                    {'term': 'CAV — Cavernoma', 'definition': 'Cavernous malformation with popcorn-appearance MRI pattern. Detected by hemosiderin ring and mixed-signal core.'},
                    {'term': 'AVM — Arteriovenous Malformation', 'definition': 'Abnormal vessel tangle detected via flow-void signals. Bounding box captures the nidus region.'},
                    {'term': 'ENC — Encephalitis', 'definition': 'Inflammatory lesion detected via diffuse T2/FLAIR signal increase in temporal or limbic regions.'},
                    {'term': 'TUM — Tumour', 'definition': 'Primary or metastatic brain tumour detected via mass effect, enhancement patterns, and perilesional oedema.'},
                    {'term': 'NL — Non-Lesional', 'definition': 'No structural lesion detected. Detection model outputs empty bounding box set for the scan.'},
                    {'term': 'NRM — Normal', 'definition': 'Imaging within normal limits. All detection scores below threshold — no abnormalities flagged.'},
                ],
            },
            {
                'title': 'Compliance References',
                'items': [
                    {
                        'term': 'FDA AI/ML Action Plan',
                        'definition': (
                            'The FDA\'s framework for AI/ML-based SaMD requires ongoing real-world '
                            'performance monitoring. Object detection models used for lesion detection '
                            'fall under Class II medical device regulations and require 510(k) or '
                            'De Novo pathway clearance with predetermined change control plans.'
                        ),
                    },
                    {
                        'term': 'EU AI Act — Article 6 (High-Risk)',
                        'definition': (
                            'Medical imaging AI systems are classified as high-risk under Annex III. '
                            'Object detection models must meet transparency, accuracy, and robustness '
                            'requirements including documented performance on diverse populations.'
                        ),
                    },
                    {
                        'term': 'ISO 14971 — Risk Management',
                        'definition': (
                            'Missed detections (false negatives) and false alarms (false positives) '
                            'are residual risks requiring documented risk-benefit analysis and '
                            'post-market surveillance tracking via the detection dashboard.'
                        ),
                    },
                    {
                        'term': 'IEC 62304 — Software Lifecycle',
                        'definition': (
                            'Detection model updates (retraining, anchor tuning, threshold changes) '
                            'require change control documentation, regression testing, and version '
                            'tracking as part of Class B/C software maintenance.'
                        ),
                    },
                    {
                        'term': 'HIPAA — Protected Health Information',
                        'definition': (
                            'MRI scans and video-EEG recordings containing patient identifiers are '
                            'ePHI. Detection pipelines must de-identify images before model training '
                            'and restrict access to detection results per HIPAA Security Rule.'
                        ),
                    },
                ],
            },
            {
                'title': 'Remediation Strategies',
                'items': [
                    {
                        'term': 'Low Detection Rate',
                        'definition': (
                            'If detection rate drops below 50%, review model threshold settings, '
                            'retrain on updated exemplar sets, and verify input image preprocessing '
                            '(contrast normalisation, resolution) has not drifted from training specs.'
                        ),
                    },
                    {
                        'term': 'Poor IoU Scores',
                        'definition': (
                            'Low IoU indicates bounding box regression is inaccurate. Recalibrate '
                            'anchor box aspect ratios for the specific lesion morphologies in the '
                            'dataset, increase training data diversity, or apply test-time augmentation.'
                        ),
                    },
                    {
                        'term': 'High False Positive Rate',
                        'definition': (
                            'Excessive false alarms increase radiologist review burden. Raise the '
                            'detection confidence threshold, apply stricter NMS, or add a '
                            'second-stage classifier to filter candidate detections.'
                        ),
                    },
                    {
                        'term': 'Pipeline Failures',
                        'definition': (
                            'Monitor cv_pipeline events for error spikes. Common causes: DICOM '
                            'format changes, GPU memory exhaustion during batch inference, or '
                            'model version mismatches after deployment. Implement health checks '
                            'and automatic fallback to CPU inference.'
                        ),
                    },
                ],
            },
        ],
    }
