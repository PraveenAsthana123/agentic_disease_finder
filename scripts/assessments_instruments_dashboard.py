"""Assessment Instruments Dashboard — 10 validated clinical instruments catalog
from config/assessments.json.
10 instruments, 6 roles, scoring methods, interpretation bands."""

import json
from pathlib import Path
from collections import Counter

_CFG = Path(__file__).resolve().parent.parent / "config" / "assessments.json"


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        return json.load(f)


# ── overview ────────────────────────────────────────────────────────────
def overview():
    """Summary KPIs: instrument count, role distribution, scoring types,
    band counts, max-score ranges, direction distribution."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "assessments.json missing"}

    instruments = cfg.get("instruments", [])
    total = len(instruments)

    role_counter = Counter()
    scoring_counter = Counter()
    direction_counter = Counter()
    total_bands = 0
    total_domains = 0
    max_scores = []

    instrument_summary = []
    role_distribution = []
    scoring_distribution = []
    direction_distribution = []
    max_score_chart = []

    for inst in instruments:
        role = inst.get("role", "Unknown")
        scoring = inst.get("scoring", "unknown")
        direction = inst.get("direction", "unknown")
        bands = inst.get("bands", [])
        domains = inst.get("domains", [])
        items = inst.get("items", [])
        mx = inst.get("max", 0)

        role_counter[role] += 1
        scoring_counter[scoring] += 1
        direction_counter[direction] += 1
        total_bands += len(bands)
        total_domains += len(domains)
        max_scores.append(mx)

        instrument_summary.append({
            "id": inst.get("id", ""),
            "name": inst.get("name", ""),
            "role": role,
            "icon": inst.get("icon", ""),
            "max": mx,
            "scoring": scoring,
            "direction": direction,
            "band_count": len(bands),
            "domain_count": len(domains),
            "item_count": len(items),
        })

        max_score_chart.append({
            "name": inst.get("id", ""),
            "value": mx,
        })

    for role, cnt in sorted(role_counter.items(), key=lambda x: -x[1]):
        role_distribution.append({"name": role, "value": cnt})

    for sc, cnt in sorted(scoring_counter.items(), key=lambda x: -x[1]):
        scoring_distribution.append({"name": sc, "value": cnt})

    for dr, cnt in sorted(direction_counter.items(), key=lambda x: -x[1]):
        direction_distribution.append({"name": dr.replace("_", " ").title(), "value": cnt})

    unique_roles = len(role_counter)

    return {
        "available": True,
        "title": cfg.get("title", "Assessment Instruments"),
        "updated_at": cfg.get("updated_at", ""),
        "kpis": {
            "total_instruments": total,
            "unique_roles": unique_roles,
            "total_bands": total_bands,
            "total_domains": total_domains,
            "scoring_types": len(scoring_counter),
            "avg_max_score": round(sum(max_scores) / len(max_scores), 1) if max_scores else 0,
        },
        "instruments": instrument_summary,
        "role_distribution": role_distribution,
        "scoring_distribution": scoring_distribution,
        "direction_distribution": direction_distribution,
        "max_score_chart": max_score_chart,
    }


# ── breakdown ───────────────────────────────────────────────────────────
def breakdown():
    """Per-instrument detail: bands, domains, items, notes, alerts."""
    cfg = _load()
    if not cfg:
        return {"available": False}

    instruments = cfg.get("instruments", [])
    details = []

    for inst in instruments:
        detail = {
            "id": inst.get("id", ""),
            "name": inst.get("name", ""),
            "role": inst.get("role", ""),
            "icon": inst.get("icon", ""),
            "max": inst.get("max", 0),
            "scoring": inst.get("scoring", ""),
            "direction": inst.get("direction", ""),
            "note": inst.get("note", ""),
            "alert": inst.get("alert", ""),
            "bands": inst.get("bands", []),
            "domains": inst.get("domains", []),
            "items": inst.get("items", []),
        }
        details.append(detail)

    return {
        "available": True,
        "instruments": details,
    }


# ── definitions ─────────────────────────────────────────────────────────
def definitions():
    """Status legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "scoring_legend": [
            {"key": "sum", "label": "Sum scoring — total of all item scores"},
            {"key": "mean", "label": "Mean scoring — average across items/domains"},
        ],
        "direction_legend": [
            {"key": "higher_better", "label": "Higher is Better — higher scores indicate better function"},
            {"key": "lower_better", "label": "Lower is Better — lower scores indicate better function"},
        ],
        "severity_colors": [
            {"level": "normal", "color": "#22c55e", "label": "Normal / Good"},
            {"level": "mild", "color": "#eab308", "label": "Mild impairment"},
            {"level": "moderate", "color": "#f97316", "label": "Moderate impairment"},
            {"level": "severe", "color": "#ef4444", "label": "Severe impairment"},
            {"level": "critical", "color": "#991b1b", "label": "Critical / Very severe"},
        ],
        "glossary": [
            {"term": "MoCA", "definition": "Montreal Cognitive Assessment — screens for mild cognitive impairment (visuospatial, naming, attention, language, abstraction, recall, orientation)"},
            {"term": "PHQ-9", "definition": "Patient Health Questionnaire-9 — validated 9-item depression severity measure"},
            {"term": "GAD-7", "definition": "Generalized Anxiety Disorder-7 — 7-item anxiety severity scale"},
            {"term": "NDDI-E", "definition": "Neurological Disorders Depression Inventory for Epilepsy — epilepsy-specific depression screen"},
            {"term": "COPM", "definition": "Canadian Occupational Performance Measure — client-centred occupational performance and satisfaction"},
            {"term": "MMSE", "definition": "Mini-Mental State Examination — brief cognitive screening (orientation, registration, attention, recall, language)"},
            {"term": "QOLIE-31", "definition": "Quality of Life in Epilepsy — 31-item epilepsy-specific quality of life measure"},
            {"term": "Barthel Index", "definition": "Activities of Daily Living (ADL) independence measure — 10 functional areas"},
            {"term": "ESS", "definition": "Epworth Sleepiness Scale — 8-item daytime sleepiness measure"},
            {"term": "LSSS", "definition": "Liverpool Seizure Severity Scale — 20-item self-report seizure severity measure"},
            {"term": "Band", "definition": "Score interpretation range mapping raw scores to clinical severity levels"},
            {"term": "Domain", "definition": "Cognitive or functional sub-scale within an instrument (e.g. visuospatial, orientation)"},
        ],
        "clinical_notes": [
            "All instruments use published, validated scoring with peer-reviewed cutoff values.",
            "PHQ-9 item 9 and NDDI-E item 4 (suicidality) require immediate clinical escalation regardless of total score.",
            "MoCA adds +1 point for patients with <=12 years education.",
            "COPM change of >=2 points is considered clinically significant improvement.",
        ],
        "references": [
            {"name": "Nasreddine et al. 2005", "desc": "MoCA: a brief screening tool for mild cognitive impairment (JAGS)"},
            {"name": "Kroenke et al. 2001", "desc": "PHQ-9: validity of a brief depression severity measure (J Gen Intern Med)"},
            {"name": "Spitzer et al. 2006", "desc": "GAD-7: a brief measure for assessing generalized anxiety (Arch Intern Med)"},
            {"name": "Baker et al. 1998", "desc": "Liverpool Seizure Severity Scale — development and psychometric evaluation (Epilepsy & Behavior)"},
        ],
    }
