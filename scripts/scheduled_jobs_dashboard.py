"""Scheduled Jobs Dashboard — cron/background job registry visualization.

Reads config/jobs.json and checks report files on disk to provide
overview KPIs, per-job breakdowns, and definitions for the frontend.
"""

import json
import os

BASE = os.path.join(os.path.dirname(__file__), '..')
CONFIG = os.path.join(BASE, 'config', 'jobs.json')


def _load():
    with open(CONFIG) as f:
        return json.load(f)


def _schedule_type(schedule):
    s = (schedule or '').lower()
    if 'hourly' in s:
        return 'Hourly'
    if 'daily' in s or any(c.isdigit() for c in s):
        return 'Daily'
    return 'Other'


def _report_info(report_path):
    if not report_path:
        return False, 0
    full = os.path.join(BASE, report_path)
    if os.path.isfile(full):
        return True, os.path.getsize(full)
    return False, 0


def overview():
    data = _load()
    jobs = data.get('jobs', [])

    schedule_counts = {}
    jobs_summary = []
    cron_tags = set()
    scripts = set()
    reports_ok = 0

    for j in jobs:
        stype = _schedule_type(j.get('schedule', ''))
        schedule_counts[stype] = schedule_counts.get(stype, 0) + 1
        exists, size = _report_info(j.get('report'))
        if exists:
            reports_ok += 1
        if j.get('cron_tag'):
            cron_tags.add(j['cron_tag'])
        if j.get('script'):
            scripts.add(j['script'])
        jobs_summary.append({
            'id': j['id'],
            'label': j.get('label', j['id']),
            'schedule': j.get('schedule', ''),
            'has_report': exists,
            'purpose': j.get('purpose', ''),
        })

    schedule_dist = [{'name': k, 'value': v} for k, v in sorted(schedule_counts.items())]

    return {
        'available': True,
        'title': data.get('title', 'Scheduled Jobs Registry'),
        'kpis': {
            'total_jobs': len(jobs),
            'daily_jobs': schedule_counts.get('Daily', 0),
            'hourly_jobs': schedule_counts.get('Hourly', 0),
            'jobs_with_reports': reports_ok,
            'unique_cron_tags': len(cron_tags),
            'unique_scripts': len(scripts),
        },
        'schedule_distribution': schedule_dist,
        'jobs_summary': jobs_summary,
    }


def breakdown():
    data = _load()
    jobs = data.get('jobs', [])
    result = []

    for j in jobs:
        exists, size = _report_info(j.get('report'))
        result.append({
            'id': j['id'],
            'label': j.get('label', j['id']),
            'schedule': j.get('schedule', ''),
            'script': j.get('script', ''),
            'cron_tag': j.get('cron_tag', ''),
            'report': j.get('report', ''),
            'report_exists': exists,
            'report_size_bytes': size,
            'purpose': j.get('purpose', ''),
        })

    return {'jobs': result}


def definitions():
    return {
        'schedule_legend': [
            {'label': 'Daily', 'color': '#22c55e', 'description': 'Runs once per day at a fixed time (e.g., 02:30, 07:00)'},
            {'label': 'Hourly', 'color': '#3b82f6', 'description': 'Runs every hour (continuous pipelines)'},
            {'label': 'Other', 'color': '#f97316', 'description': 'Custom or event-driven schedule'},
        ],
        'glossary': [
            {'term': 'Cron Tag', 'definition': 'Unique identifier used in crontab comments to track and manage scheduled jobs (e.g., AGENTICFINDER-TRAIN).'},
            {'term': 'CHB-MIT', 'definition': 'Children\'s Hospital Boston EEG dataset from PhysioNet — benchmark for seizure detection models.'},
            {'term': 'ChromaDB', 'definition': 'Open-source vector database used for RAG (Retrieval-Augmented Generation) embedding storage and similarity search.'},
            {'term': 'PSI', 'definition': 'Population Stability Index — measures distribution shift between training and serving feature distributions.'},
            {'term': 'KS Test', 'definition': 'Kolmogorov-Smirnov test — non-parametric test for detecting drift in continuous feature distributions.'},
            {'term': 'Fairlearn', 'definition': 'Microsoft toolkit for assessing and improving fairness of machine learning models across demographic groups.'},
            {'term': 'RDF', 'definition': 'Resource Description Framework — W3C standard for representing knowledge graphs as subject-predicate-object triples.'},
            {'term': 'SHAP', 'definition': 'SHapley Additive exPlanations — game-theoretic approach to explain individual model predictions.'},
            {'term': 'CDM', 'definition': 'Clinical Data Manager — responsible for data intake, validation, quality, and governance.'},
            {'term': 'Bonn Dataset', 'definition': 'University of Bonn EEG dataset — 5-class benchmark (A-E) used for external validation of seizure classifiers.'},
            {'term': 'YOLO', 'definition': 'You Only Look Once — real-time object detection model used in the CV pipeline for EEG image analysis.'},
            {'term': 'Bootstrap CI', 'definition': 'Confidence intervals computed via resampling — provides uncertainty estimates for model accuracy metrics.'},
        ],
        'clinical_notes': [
            'Scheduled training ensures the model stays current with new CHB-MIT data and prevents accuracy decay.',
            'Drift monitoring (PSI + KS) guards against silent model degradation from train/serve feature skew.',
            'Fairness audits run daily to detect and flag demographic bias before clinical deployment.',
            'Consistency checks verify that explainability (SHAP) and classification use the same model bundle.',
        ],
        'references': [
            {'ref': 'config/jobs.json', 'detail': 'Canonical job registry — all cron schedules, scripts, report paths, and tags.'},
            {'ref': 'IEC 62304', 'detail': 'Medical device software lifecycle — requires traceability of automated data processing.'},
            {'ref': 'FDA AI/ML SaMD', 'detail': 'Guidance on continuous learning and monitoring for AI-based Software as a Medical Device.'},
            {'ref': 'FAIR Principles', 'detail': 'Findable, Accessible, Interoperable, Reusable — data governance standards for clinical research.'},
        ],
    }
