"""Epilepsy Challenges Dashboard — 30 clinical/AI challenges by difficulty
level (basic/intermediate/high) with STAR justification, from
config/epilepsy_challenges.json."""

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


def overview():
    """Summary KPIs: total challenges, level distribution, AI help coverage."""
    cfg = _load('epilepsy_challenges.json')
    if not cfg:
        return {"available": False, "note": "epilepsy_challenges.json missing"}

    challenges = cfg.get('challenges', [])
    total = len(challenges)

    level_counts = Counter(c.get('level', 'unknown') for c in challenges)

    has_ai_help = sum(1 for c in challenges if c.get('ai_help'))
    has_star = sum(1 for c in challenges if c.get('star'))

    level_distribution = [
        {"name": lvl, "value": level_counts.get(lvl, 0)}
        for lvl in ['basic', 'intermediate', 'high']
    ]

    ai_coverage = round(has_ai_help / total * 100, 1) if total else 0

    return {
        "available": True,
        "summary": {
            "total_challenges": total,
            "basic": level_counts.get('basic', 0),
            "intermediate": level_counts.get('intermediate', 0),
            "high": level_counts.get('high', 0),
            "ai_help_coverage_pct": ai_coverage,
            "star_documented": has_star,
        },
        "level_distribution": level_distribution,
        "challenge_table": [
            {
                "n": c.get('n'),
                "level": c.get('level', ''),
                "challenge": c.get('challenge', ''),
                "ai_help": c.get('ai_help', ''),
            }
            for c in challenges
        ],
    }


def breakdown():
    """Per-challenge detail including STAR justification."""
    cfg = _load('epilepsy_challenges.json')
    if not cfg:
        return {"available": False, "note": "epilepsy_challenges.json missing"}

    challenges = cfg.get('challenges', [])

    by_level = {}
    for c in challenges:
        lvl = c.get('level', 'unknown')
        if lvl not in by_level:
            by_level[lvl] = []
        star = c.get('star', {})
        by_level[lvl].append({
            "n": c.get('n'),
            "challenge": c.get('challenge', ''),
            "ai_help": c.get('ai_help', ''),
            "situation": star.get('s', ''),
            "task": star.get('t', ''),
            "action": star.get('a', ''),
            "result": star.get('r', ''),
        })

    return {
        "available": True,
        "by_level": by_level,
        "all_challenges": [
            {
                "n": c.get('n'),
                "level": c.get('level', ''),
                "challenge": c.get('challenge', ''),
                "ai_help": c.get('ai_help', ''),
                "situation": c.get('star', {}).get('s', ''),
                "task": c.get('star', {}).get('t', ''),
                "action": c.get('star', {}).get('a', ''),
                "result": c.get('star', {}).get('r', ''),
            }
            for c in challenges
        ],
    }


def definitions():
    """Glossary, level descriptions, clinical notes, and references."""
    return {
        "available": True,
        "levels": [
            {"name": "basic", "description": "Foundational challenges in EEG acquisition, metadata, and patient compliance that every epilepsy monitoring unit faces."},
            {"name": "intermediate", "description": "Analytical challenges requiring signal processing, machine learning, and clinical correlation expertise."},
            {"name": "high", "description": "Advanced research challenges involving prediction, multimodal fusion, federated learning, and regulatory governance."},
        ],
        "star_method": {
            "s": "Situation — the clinical/technical problem context",
            "t": "Task — what needs to be achieved",
            "a": "Action — the AI/engineering solution approach",
            "r": "Result — the expected clinical or technical outcome",
        },
        "glossary": [
            {"term": "IED", "definition": "Interictal Epileptiform Discharge — abnormal spike/sharp wave between seizures, key diagnostic marker."},
            {"term": "ICA", "definition": "Independent Component Analysis — blind source separation to remove artifacts from EEG."},
            {"term": "STAR", "definition": "Situation-Task-Action-Result — structured justification framework for each challenge."},
            {"term": "DRE", "definition": "Drug-Resistant Epilepsy — failure of 2+ appropriate anti-seizure medications."},
            {"term": "ASM", "definition": "Anti-Seizure Medication — drugs used to control epileptic seizures."},
            {"term": "SHAP", "definition": "SHapley Additive exPlanations — model interpretability via feature contribution scores."},
            {"term": "GroupKFold", "definition": "Cross-validation strategy that keeps all data from one subject in the same fold, preventing leakage."},
            {"term": "VEEG", "definition": "Video EEG — continuous EEG with synchronized video for seizure characterization."},
            {"term": "HITL", "definition": "Human-In-The-Loop — clinical expert oversight gate for AI decisions."},
            {"term": "Federated Learning", "definition": "Training across institutions without sharing raw patient data; only model updates are exchanged."},
        ],
        "clinical_notes": [
            "Challenges are ordered by difficulty: basic (acquisition/data quality), intermediate (analysis/modeling), high (prediction/governance).",
            "Each challenge includes an AI solution approach — this is the thesis contribution mapping.",
            "STAR justification ensures every challenge is grounded in a real clinical scenario, not hypothetical.",
            "The 30 challenges map to the DBA epilepsy governance thesis problem space.",
        ],
        "references": [
            "ILAE Classification of Epilepsies (2017) — Fisher et al.",
            "ACNS Guidelines for EEG Recording — American Clinical Neurophysiology Society",
            "EU AI Act — High-Risk AI System Requirements (2024)",
            "STAR Interview Method — structured behavioral justification framework",
        ],
    }
