"""WAIS Dashboard — Wechsler Adult Intelligence Scale analytics.

All data from REAL WAIS assessments in data/clinical.db (assessments table, instrument='WAIS').

The WAIS-IV is the gold-standard measure of adult intellectual functioning,
used in neuropsychological evaluations for epilepsy surgery workups,
cognitive decline monitoring, and disability assessment.

Structure (10 core subtests, 4 index scores + FSIQ):
  Verbal Comprehension Index (VCI):
    - Similarities
    - Vocabulary
    - Information
  Perceptual Reasoning Index (PRI):
    - Block Design
    - Matrix Reasoning
    - Visual Puzzles
  Working Memory Index (WMI):
    - Digit Span
    - Arithmetic
  Processing Speed Index (PSI):
    - Symbol Search
    - Coding

  Index scores: mean=100, SD=15  (range ~40-160)
  FSIQ: weighted composite of all 4 indices, mean=100, SD=15
  Subtest scaled scores: mean=10, SD=3  (range 1-19)

  IQ Classification (Wechsler):
    130+      Extremely High
    120-129   Superior
    110-119   High Average
    90-109    Average
    80-89     Low Average
    70-79     Borderline
    <70       Extremely Low

Reference:
  Wechsler D. Wechsler Adult Intelligence Scale — Fourth Edition (WAIS-IV).
  San Antonio, TX: Pearson; 2008.

Author: Research Team
"""
import sqlite3
import json
from pathlib import Path
from collections import Counter, defaultdict

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"

SUBTESTS = [
    {"id": "similarities", "label": "Similarities", "index": "VCI", "description": "Verbal abstract reasoning — identify how two concepts are alike"},
    {"id": "vocabulary", "label": "Vocabulary", "index": "VCI", "description": "Word knowledge — define words of increasing difficulty"},
    {"id": "information", "label": "Information", "index": "VCI", "description": "General knowledge — factual questions across domains"},
    {"id": "block_design", "label": "Block Design", "index": "PRI", "description": "Visuospatial construction — replicate geometric patterns with blocks"},
    {"id": "matrix_reasoning", "label": "Matrix Reasoning", "index": "PRI", "description": "Nonverbal abstract reasoning — complete visual pattern matrices"},
    {"id": "visual_puzzles", "label": "Visual Puzzles", "index": "PRI", "description": "Visual-spatial analysis — select pieces that reconstruct a puzzle"},
    {"id": "digit_span", "label": "Digit Span", "index": "WMI", "description": "Working memory — repeat digit sequences forward, backward, and sequencing"},
    {"id": "arithmetic", "label": "Arithmetic", "index": "WMI", "description": "Mental math — solve arithmetic problems without paper"},
    {"id": "symbol_search", "label": "Symbol Search", "index": "PSI", "description": "Processing speed — scan and match target symbols in rows"},
    {"id": "coding", "label": "Coding", "index": "PSI", "description": "Processing speed — transcribe digit-symbol pairs under time pressure"},
]

INDEX_DEFS = [
    {"id": "VCI", "label": "Verbal Comprehension", "color": "#3b82f6", "subtests": ["similarities", "vocabulary", "information"]},
    {"id": "PRI", "label": "Perceptual Reasoning", "color": "#22c55e", "subtests": ["block_design", "matrix_reasoning", "visual_puzzles"]},
    {"id": "WMI", "label": "Working Memory", "color": "#f97316", "subtests": ["digit_span", "arithmetic"]},
    {"id": "PSI", "label": "Processing Speed", "color": "#8b5cf6", "subtests": ["symbol_search", "coding"]},
]

IQ_CLASSES = [
    {"range": [130, 160], "label": "Extremely High", "color": "#16a34a"},
    {"range": [120, 129], "label": "Superior", "color": "#22c55e"},
    {"range": [110, 119], "label": "High Average", "color": "#3b82f6"},
    {"range": [90, 109], "label": "Average", "color": "#64748b"},
    {"range": [80, 89], "label": "Low Average", "color": "#eab308"},
    {"range": [70, 79], "label": "Borderline", "color": "#f97316"},
    {"range": [40, 69], "label": "Extremely Low", "color": "#ef4444"},
]


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def _iq_class(fsiq):
    for cls in IQ_CLASSES:
        if cls["range"][0] <= fsiq <= cls["range"][1]:
            return cls["label"]
    if fsiq > 160:
        return "Extremely High"
    return "Extremely Low"


def _iq_color(fsiq):
    for cls in IQ_CLASSES:
        if cls["range"][0] <= fsiq <= cls["range"][1]:
            return cls["color"]
    return "#94a3b8"


def overview():
    """Summary KPIs + IQ distribution + per-patient latest scores."""
    rows = _rows("SELECT * FROM assessments WHERE instrument='WAIS' ORDER BY created_at DESC")
    if not rows:
        return {"total_assessments": 0, "patients": 0, "message": "No WAIS data yet"}

    patients = set()
    latest_per_patient = {}
    all_fsiq = []
    iq_dist = Counter()

    for r in rows:
        patients.add(r["patient_id"])
        all_fsiq.append(r["score"])
        iq_dist[r["level"]] += 1
        if r["patient_id"] not in latest_per_patient:
            latest_per_patient[r["patient_id"]] = r

    avg_fsiq = round(sum(all_fsiq) / len(all_fsiq), 1)
    min_fsiq = min(all_fsiq)
    max_fsiq = max(all_fsiq)

    patient_summary = []
    for pid, r in sorted(latest_per_patient.items()):
        answers = json.loads(r["answers_json"]) if r["answers_json"] else {}
        patient_summary.append({
            "patient_id": pid,
            "fsiq": r["score"],
            "vci": answers.get("VCI", 0),
            "pri": answers.get("PRI", 0),
            "wmi": answers.get("WMI", 0),
            "psi": answers.get("PSI", 0),
            "classification": r["level"],
            "interpretation": r["interpretation"],
            "assessed_at": r["created_at"],
        })

    # Sort by FSIQ ascending (lowest first — clinical priority)
    patient_summary.sort(key=lambda x: x["fsiq"])

    return {
        "total_assessments": len(rows),
        "unique_patients": len(patients),
        "avg_fsiq": avg_fsiq,
        "min_fsiq": min_fsiq,
        "max_fsiq": max_fsiq,
        "iq_distribution": dict(iq_dist),
        "patient_summary": patient_summary,
    }


def breakdown():
    """Index profiles, subtest analysis, per-patient history."""
    rows = _rows("SELECT * FROM assessments WHERE instrument='WAIS' ORDER BY created_at ASC")
    if not rows:
        return {"message": "No WAIS data"}

    # Aggregate subtest scores across all assessments
    subtest_totals = {s["id"]: [] for s in SUBTESTS}
    index_totals = {idx["id"]: [] for idx in INDEX_DEFS}
    fsiq_list = []

    patient_history = defaultdict(list)

    for r in rows:
        answers = json.loads(r["answers_json"]) if r["answers_json"] else {}
        fsiq_list.append(r["score"])

        for s in SUBTESTS:
            val = answers.get(s["id"], 0)
            if val > 0:
                subtest_totals[s["id"]].append(val)

        for idx in INDEX_DEFS:
            val = answers.get(idx["id"], 0)
            if val > 0:
                index_totals[idx["id"]].append(val)

        patient_history[r["patient_id"]].append({
            "fsiq": r["score"],
            "vci": answers.get("VCI", 0),
            "pri": answers.get("PRI", 0),
            "wmi": answers.get("WMI", 0),
            "psi": answers.get("PSI", 0),
            "classification": r["level"],
            "date": r["created_at"],
        })

    # Index profile (group average)
    index_profile = []
    for idx in INDEX_DEFS:
        vals = index_totals[idx["id"]]
        index_profile.append({
            "id": idx["id"],
            "label": idx["label"],
            "color": idx["color"],
            "avg": round(sum(vals) / len(vals), 1) if vals else 0,
            "min": min(vals) if vals else 0,
            "max": max(vals) if vals else 0,
            "n": len(vals),
        })

    # Subtest scaled scores (group average)
    subtest_summary = []
    for s in SUBTESTS:
        vals = subtest_totals[s["id"]]
        subtest_summary.append({
            "id": s["id"],
            "label": s["label"],
            "index": s["index"],
            "avg_scaled": round(sum(vals) / len(vals), 1) if vals else 0,
            "min": min(vals) if vals else 0,
            "max": max(vals) if vals else 0,
            "n": len(vals),
        })

    # Sort subtests by avg_scaled ascending (weakest first)
    subtest_summary.sort(key=lambda x: x["avg_scaled"])

    # Per-patient index comparison (for radar/bar chart)
    patient_index_comparison = []
    for pid, hist in patient_history.items():
        latest = hist[-1]
        patient_index_comparison.append({
            "patient_id": pid,
            "VCI": latest["vci"],
            "PRI": latest["pri"],
            "WMI": latest["wmi"],
            "PSI": latest["psi"],
            "FSIQ": latest["fsiq"],
        })

    # Monthly trend
    date_scores = defaultdict(list)
    for r in rows:
        month = (r["created_at"] or "")[:7]
        if month:
            date_scores[month].append(r["score"])
    trend = [{"month": m, "avg_fsiq": round(sum(v) / len(v), 1), "count": len(v)}
             for m, v in sorted(date_scores.items())]

    return {
        "index_profile": index_profile,
        "subtest_summary": subtest_summary,
        "patient_index_comparison": patient_index_comparison,
        "patient_history": dict(patient_history),
        "trend": trend,
    }


def definitions():
    """Metric definitions for the WAIS dashboard."""
    return {
        "title": "Wechsler Adult Intelligence Scale (WAIS-IV) — Metric Definitions",
        "definitions": [
            {"term": "WAIS-IV", "definition": "Wechsler Adult Intelligence Scale, Fourth Edition — the gold-standard individually administered test of adult intelligence. Published by Pearson, normed on 2,200 adults aged 16-90."},
            {"term": "Full Scale IQ (FSIQ)", "definition": "Composite score from all 4 index scores. Mean=100, SD=15. Represents overall cognitive ability."},
            {"term": "Verbal Comprehension Index (VCI)", "definition": "Measures verbal concept formation, reasoning, and acquired knowledge. Subtests: Similarities, Vocabulary, Information."},
            {"term": "Perceptual Reasoning Index (PRI)", "definition": "Measures nonverbal and fluid reasoning, visuospatial processing. Subtests: Block Design, Matrix Reasoning, Visual Puzzles."},
            {"term": "Working Memory Index (WMI)", "definition": "Measures ability to hold and manipulate information in short-term memory. Subtests: Digit Span, Arithmetic."},
            {"term": "Processing Speed Index (PSI)", "definition": "Measures speed of mental and graphomotor processing. Subtests: Symbol Search, Coding."},
            {"term": "Scaled Score", "definition": "Subtest-level score. Mean=10, SD=3. Range 1-19. Allows comparison across subtests."},
            {"term": "Index Score", "definition": "Composite score for each cognitive domain. Mean=100, SD=15. Derived from component subtest scaled scores."},
            {"term": "Extremely High (130+)", "definition": "Top ~2% of the population. Exceptional cognitive abilities."},
            {"term": "Superior (120-129)", "definition": "Top ~8%. Well above average cognitive functioning."},
            {"term": "High Average (110-119)", "definition": "Top ~25%. Above average cognitive abilities."},
            {"term": "Average (90-109)", "definition": "Middle 50% of the population. Typical cognitive functioning."},
            {"term": "Low Average (80-89)", "definition": "Bottom ~25%. Below average but within normal limits."},
            {"term": "Borderline (70-79)", "definition": "Bottom ~8%. Borderline intellectual functioning; may need support."},
            {"term": "Extremely Low (<70)", "definition": "Bottom ~2%. May indicate intellectual disability; requires comprehensive evaluation."},
            {"term": "Epilepsy Context", "definition": "Pre-surgical neuropsychological testing with WAIS is standard of care for epilepsy surgery candidates. Lateralized cognitive deficits help localize seizure focus and predict post-surgical outcomes (Jones-Gotman et al., 2010)."},
            {"term": "Index Discrepancy", "definition": "Significant difference (15+ points) between index scores suggests uneven cognitive profile, common in epilepsy patients with focal lesions."},
        ],
    }
