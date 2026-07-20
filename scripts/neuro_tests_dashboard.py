"""Neuro Tests Dashboard — neurophysiology / electrodiagnostic test catalog
visualization from config/neuro_tests.json.
13 tests, 5 EEG linkage categories, all built status."""

import json
import os
from collections import Counter

_DIR = os.path.dirname(__file__)
_CFG = os.path.join(_DIR, '..', 'config')


def _load(fname):
    path = os.path.join(_CFG, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _categorize_eeg_link(eeg_link):
    """Categorize a test's eeg_link field into one of 5 groups."""
    link = (eeg_link or '').lower()
    if 'core' in link:
        return 'core'
    if 'evoked' in link:
        return 'evoked-potential'
    if 'autonomic' in link or 'sudep' in link or 'cardiac' in link:
        return 'autonomic'
    if 'motor' in link or 'myoclonus' in link:
        return 'motor'
    if 'independent' in link or 'peripheral' in link:
        return 'peripheral/independent'
    return 'other'


def overview():
    """Summary KPIs: test counts, status distribution, EEG linkage breakdown,
    role distribution, tests table."""
    cfg = _load('neuro_tests.json')
    if not cfg:
        return {"available": False, "note": "neuro_tests.json missing"}

    tests = cfg.get('tests', [])
    summary = cfg.get('summary', {})
    eeg_linkage_summary = cfg.get('eeg_linkage_summary', [])

    # Status distribution
    status_counts = Counter(t.get('status', 'unknown') for t in tests)
    status_distribution = [
        {"name": s, "value": c}
        for s, c in sorted(status_counts.items(), key=lambda x: -x[1])
    ]

    # Unique roles and output types
    unique_roles = sorted(set(t.get('role', '') for t in tests if t.get('role')))
    unique_outputs = sorted(set(t.get('output', '') for t in tests if t.get('output')))

    # EEG link categorization
    eeg_link_cats = Counter(_categorize_eeg_link(t.get('eeg_link', '')) for t in tests)
    eeg_link_distribution = [
        {"name": cat, "value": c}
        for cat, c in sorted(eeg_link_cats.items(), key=lambda x: -x[1])
    ]

    # Role distribution
    role_counts = Counter(t.get('role', 'unspecified') for t in tests)
    role_distribution = [
        {"name": r, "value": c}
        for r, c in sorted(role_counts.items(), key=lambda x: -x[1])
    ]

    # Tests table
    tests_table = [
        {
            "id": t.get('id', ''),
            "name": t.get('name', ''),
            "purpose": t.get('purpose', ''),
            "role": t.get('role', ''),
            "output": t.get('output', ''),
            "eeg_link": t.get('eeg_link', ''),
            "status": t.get('status', ''),
            "has_case_data": 'case_data' in t,
        }
        for t in tests
    ]

    return {
        "available": True,
        "title": cfg.get('title', ''),
        "summary": {
            "total_tests": len(tests),
            "built": status_counts.get('built', 0),
            "partial": status_counts.get('partial', 0),
            "cataloged": status_counts.get('cataloged', 0),
            "eeg_linkage_categories": len(eeg_linkage_summary),
            "unique_roles": len(unique_roles),
            "unique_output_types": len(unique_outputs),
        },
        "status_distribution": status_distribution,
        "eeg_link_categories": dict(eeg_link_cats),
        "eeg_link_distribution": eeg_link_distribution,
        "role_distribution": role_distribution,
        "tests_table": tests_table,
    }


def breakdown():
    """Per-EEG-link-category test details, per-role grouping, linkage summary,
    tests with case data."""
    cfg = _load('neuro_tests.json')
    if not cfg:
        return {"available": False, "note": "neuro_tests.json missing"}

    tests = cfg.get('tests', [])
    eeg_linkage_summary = cfg.get('eeg_linkage_summary', [])

    # Group tests by EEG link category
    by_eeg_link = {}
    for t in tests:
        cat = _categorize_eeg_link(t.get('eeg_link', ''))
        if cat not in by_eeg_link:
            by_eeg_link[cat] = []
        by_eeg_link[cat].append({
            "id": t.get('id', ''),
            "name": t.get('name', ''),
            "purpose": t.get('purpose', ''),
            "role": t.get('role', ''),
            "output": t.get('output', ''),
            "eeg_link": t.get('eeg_link', ''),
            "status": t.get('status', ''),
            "note": t.get('note', ''),
            "has_case_data": 'case_data' in t,
        })

    # Group tests by role
    by_role = {}
    for t in tests:
        role = t.get('role', 'unspecified')
        if role not in by_role:
            by_role[role] = []
        by_role[role].append({
            "id": t.get('id', ''),
            "name": t.get('name', ''),
            "purpose": t.get('purpose', ''),
            "output": t.get('output', ''),
            "eeg_link": t.get('eeg_link', ''),
            "status": t.get('status', ''),
        })

    # Tests with case_data
    tests_with_case_data = [
        {
            "id": t.get('id', ''),
            "name": t.get('name', ''),
            "purpose": t.get('purpose', ''),
            "case_data": t.get('case_data'),
        }
        for t in tests
        if 'case_data' in t
    ]

    return {
        "available": True,
        "by_eeg_link": by_eeg_link,
        "by_role": by_role,
        "eeg_linkage_summary": eeg_linkage_summary,
        "tests_with_case_data": tests_with_case_data,
    }


def definitions():
    """Status legend, EEG link type descriptions, glossary, clinical notes, references."""
    return {
        "available": True,
        "status_legend": [
            {"status": "built", "description": "Live in this platform — endpoint verified, dashboard functional, real clinical logic"},
            {"status": "partial", "description": "Core logic exists but not all sub-features are wired or validated"},
            {"status": "cataloged", "description": "Test type documented and structured — implementation pending"},
            {"status": "planned", "description": "Identified as clinically relevant — not yet cataloged or built"},
        ],
        "eeg_link_types": [
            {"type": "core", "description": "Directly part of or derived from standard EEG recording — shares electrodes, montages, or raw signal pipeline"},
            {"type": "evoked-potential", "description": "Stimulus-locked averaged EEG responses — SSEP, VEP, BERA use EEG amplifiers with specialized paradigms"},
            {"type": "autonomic", "description": "Autonomic nervous system tests — HRV, SSR, tilt-table — linked to SUDEP risk and seizure-related cardiac monitoring"},
            {"type": "motor", "description": "Motor pathway and neuromuscular junction tests — EMG, NCV, RNST — assess peripheral motor function relevant to myoclonus and seizure semiology"},
            {"type": "peripheral/independent", "description": "Tests that run independently of EEG hardware — ABPM, blink reflex — but provide complementary neurophysiology data"},
        ],
        "glossary": [
            {"term": "EEG", "definition": "Electroencephalography — scalp recording of cortical electrical activity, gold standard for seizure detection and epilepsy diagnosis"},
            {"term": "EMG", "definition": "Electromyography — needle or surface recording of muscle electrical activity, used to assess motor unit pathology and myoclonus"},
            {"term": "NCV", "definition": "Nerve Conduction Velocity — measures speed of electrical signal propagation along peripheral nerves, detects neuropathy"},
            {"term": "SSEP", "definition": "Somatosensory Evoked Potential — cortical response to peripheral nerve stimulation, assesses sensory pathway integrity"},
            {"term": "VEP", "definition": "Visual Evoked Potential — occipital cortex response to visual stimuli, detects optic pathway lesions and demyelination"},
            {"term": "BERA", "definition": "Brainstem Evoked Response Audiometry — auditory brainstem responses to click stimuli, assesses hearing and brainstem pathway integrity"},
            {"term": "RNST", "definition": "Repetitive Nerve Stimulation Test — repeated motor nerve stimulation to detect neuromuscular junction disorders (myasthenia gravis)"},
            {"term": "HRV", "definition": "Heart Rate Variability — beat-to-beat interval analysis reflecting autonomic nervous system function, relevant to SUDEP risk stratification"},
            {"term": "SUDEP", "definition": "Sudden Unexpected Death in Epilepsy — leading cause of epilepsy-related mortality, linked to autonomic dysfunction and seizure severity"},
            {"term": "ABPM", "definition": "Ambulatory Blood Pressure Monitoring — 24-hour BP recording detecting circadian patterns and autonomic dysregulation"},
            {"term": "SSR", "definition": "Sympathetic Skin Response — galvanic skin response to stimuli, measures sympathetic sudomotor pathway integrity"},
            {"term": "Blink Reflex", "definition": "Electrically elicited blink response testing trigeminal-facial nerve arc, assesses brainstem reflex pathways"},
        ],
        "clinical_notes": [
            "All 13 tests in the catalog are electrodiagnostic procedures performed in neurophysiology labs — they complement but do not replace clinical EEG",
            "EEG linkage classification helps the AI pipeline decide which tests share signal preprocessing steps and which require independent pipelines",
            "Evoked potential tests (SSEP, VEP, BERA) use EEG amplifiers but require stimulus-locked averaging rather than continuous monitoring",
            "Autonomic tests (HRV, SSR, tilt-table) are increasingly prioritized in epilepsy centers for SUDEP risk assessment programs",
        ],
        "references": [
            "ACNS: American Clinical Neurophysiology Society — Guidelines for Standard Electrode Position Nomenclature (2006, updated 2016)",
            "AANEM: American Association of Neuromuscular & Electrodiagnostic Medicine — Practice Guidelines for EMG and NCV Studies",
            "ILAE: International League Against Epilepsy — Operational Classification of Seizure Types (2017) and SUDEP risk factors",
            "IFCN: International Federation of Clinical Neurophysiology — Standards for EEG, Evoked Potentials, and EMG recording",
            "Devinsky et al., 2016 — Sudden unexpected death in epilepsy: epidemiology, mechanisms, and prevention (Lancet Neurology)",
        ],
    }
