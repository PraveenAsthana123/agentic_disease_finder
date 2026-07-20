"""Role Tests Dashboard — per-role testing matrix across API/Data/Model/Accuracy/Process/Frontend/Manual
dimensions with pass/partial/planned status, from config/role_tests.json."""

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
    """Summary KPIs: total roles, total test cases, status distribution, per-role counts, per-dimension counts."""
    cfg = _load('role_tests.json')
    if not cfg:
        return {"available": False, "note": "role_tests.json missing"}

    roles = cfg.get('roles', [])
    total_roles = len(roles)

    all_tests = [t for role in roles for t in role.get('tests', [])]
    total_tests = len(all_tests)

    status_counts = Counter(t.get('status', 'unknown') for t in all_tests)
    passed = status_counts.get('pass', 0) + status_counts.get('built', 0)
    partial = status_counts.get('partial', 0)
    planned = status_counts.get('planned', 0)
    pass_pct = round(passed / total_tests * 100, 1) if total_tests else 0

    dim_counts = Counter(t.get('dim', 'Unknown') for t in all_tests)
    all_dims = sorted(dim_counts.keys())

    roles_all_pass = sum(
        1 for role in roles
        if all(t.get('status', 'unknown') in ('pass', 'built') for t in role.get('tests', []))
    )

    status_distribution = [
        {"name": s, "value": c}
        for s, c in [('pass', passed), ('partial', partial), ('planned', planned)]
        if c > 0
    ]

    tests_per_role = []
    for role in roles:
        tests = role.get('tests', [])
        rs = Counter(t.get('status', 'unknown') for t in tests)
        tests_per_role.append({
            "name": role.get('role', ''),
            "value": len(tests),
            "pass": rs.get('pass', 0) + rs.get('built', 0),
            "partial": rs.get('partial', 0),
            "planned": rs.get('planned', 0),
        })

    tests_per_dim = []
    for dim in all_dims:
        dim_tests = [t for t in all_tests if t.get('dim') == dim]
        ds = Counter(t.get('status', 'unknown') for t in dim_tests)
        tests_per_dim.append({
            "name": dim,
            "value": len(dim_tests),
            "pass": ds.get('pass', 0) + ds.get('built', 0),
            "partial": ds.get('partial', 0),
            "planned": ds.get('planned', 0),
        })

    role_summary = []
    for role in roles:
        tests = role.get('tests', [])
        rs = Counter(t.get('status', 'unknown') for t in tests)
        role_summary.append({
            "role": role.get('role', ''),
            "total": len(tests),
            "pass": rs.get('pass', 0) + rs.get('built', 0),
            "partial": rs.get('partial', 0),
            "planned": rs.get('planned', 0),
        })

    return {
        "available": True,
        "summary": {
            "total_roles": total_roles,
            "total_tests": total_tests,
            "passed": passed,
            "partial": partial,
            "planned": planned,
            "pass_pct": pass_pct,
            "roles_all_pass": roles_all_pass,
            "total_dims": len(all_dims),
        },
        "status_distribution": status_distribution,
        "tests_per_role": tests_per_role,
        "tests_per_dim": tests_per_dim,
        "role_summary": role_summary,
    }


def breakdown():
    """Per-role full test list with dimension, case, status, and maps_to."""
    cfg = _load('role_tests.json')
    if not cfg:
        return {"available": False, "note": "role_tests.json missing"}

    roles = cfg.get('roles', [])

    roles_out = []
    for role in roles:
        tests = []
        for t in role.get('tests', []):
            tests.append({
                "dim": t.get('dim', ''),
                "case": t.get('case', ''),
                "status": t.get('status', ''),
                "maps_to": t.get('maps_to', ''),
            })
        roles_out.append({
            "role": role.get('role', ''),
            "tests": tests,
        })

    # Also build a flat matrix: every role x dim combination
    all_dims = sorted(set(t.get('dim', '') for role in roles for t in role.get('tests', [])))
    matrix = []
    for role in roles:
        row = {"role": role.get('role', '')}
        tests_by_dim = {}
        for t in role.get('tests', []):
            tests_by_dim[t.get('dim', '')] = t.get('status', 'N/A')
        for dim in all_dims:
            row[dim] = tests_by_dim.get(dim, 'N/A')
        matrix.append(row)

    return {
        "available": True,
        "roles": roles_out,
        "matrix": matrix,
        "dimensions": all_dims,
    }


def definitions():
    """Glossary, dimension descriptions, status legend, clinical notes, and references."""
    return {
        "available": True,
        "dimension_descriptions": [
            {"dim": "API", "description": "Endpoint-level tests verifying HTTP status, request/response contracts, and error handling for role-specific routes."},
            {"dim": "Data", "description": "Data integrity checks: parsing, feature extraction, schema compliance, NaN/null handling, and persistence verification."},
            {"dim": "Model", "description": "ML model validation: correct predictions on held-out samples, model loading, inference outputs, and confidence scores."},
            {"dim": "Accuracy", "description": "Quantitative accuracy metrics: SHAP explanations, fairness gates (DI/EO), and performance thresholds on clinical data."},
            {"dim": "Process", "description": "Workflow and business-logic tests: state transitions, audit trails, HITL override recording, and pipeline orchestration."},
            {"dim": "Frontend", "description": "UI component tests: dashboard rendering, data binding, user interactions, and responsive layout verification."},
            {"dim": "Manual", "description": "Human-in-the-loop tests requiring manual verification: UI walkthroughs, clinician confirmations, and onboarding flows."},
        ],
        "status_legend": [
            {"status": "pass", "description": "Test case is verified — endpoint returns expected output, data flows correctly, and the feature works end-to-end."},
            {"status": "built", "description": "Feature is built and functional — equivalent to pass for implementation status."},
            {"status": "partial", "description": "Partially implemented — core logic exists but integration, edge cases, or full verification is incomplete."},
            {"status": "planned", "description": "Planned for a future sprint — test case is defined but implementation has not started."},
        ],
        "glossary": [
            {"term": "SHAP", "definition": "SHapley Additive exPlanations — assigns each input feature a contribution score to explain individual AI predictions."},
            {"term": "HITL", "definition": "Human-In-The-Loop — mandatory clinician oversight gate where the AI surfaces a decision and a human must confirm or override."},
            {"term": "DI", "definition": "Disparate Impact — fairness metric measuring outcome rate ratio across protected groups; gate threshold >= 0.80."},
            {"term": "EO", "definition": "Equal Opportunity — fairness metric checking true-positive-rate parity across demographic groups."},
            {"term": "EDF", "definition": "European Data Format — standard file format for multi-channel biosignal data (EEG, EMG, EOG)."},
            {"term": "PNES", "definition": "Psychogenic Non-Epileptic Seizures — seizure-like episodes without ictal EEG correlate; require psychiatric differential."},
            {"term": "ICA", "definition": "Independent Component Analysis — blind source separation technique used to isolate and remove EEG artifacts."},
            {"term": "RAG", "definition": "Retrieval-Augmented Generation — LLM answers grounded in retrieved documents to prevent hallucination."},
            {"term": "IRB", "definition": "Institutional Review Board — ethics committee that reviews and approves research involving human subjects."},
            {"term": "COPM", "definition": "Canadian Occupational Performance Measure — standardised self-report instrument for occupational performance."},
        ],
        "clinical_notes": [
            "The testing matrix covers 7 clinical roles with tests scoped to each role's specific workflow across up to 7 dimensions.",
            "Status 'pass' or 'built' means the test case has been verified with real data and live endpoints returning 200 OK.",
            "Partial tests primarily affect IoT Engineer (device streaming), Patient (consent flows), and manual verification steps.",
            "Each dimension maps to a layer of the clinical AI stack: API (transport), Data (integrity), Model (inference), Accuracy (metrics), Process (workflow), Frontend (UI), Manual (human).",
        ],
        "references": [
            "ILAE Classification of Epilepsies (2017) — Fisher et al., Epilepsia 58(4):512-521",
            "ACNS Guidelines for EEG Recording — American Clinical Neurophysiology Society (2016)",
            "ISO/IEC 25010:2011 — Systems and Software Quality Requirements and Evaluation (SQuaRE)",
            "IEEE 829 Standard for Software Test Documentation",
        ],
    }


if __name__ == '__main__':
    import sys
    fn = sys.argv[1] if len(sys.argv) > 1 else 'overview'
    dispatch = {'overview': overview, 'breakdown': breakdown, 'definitions': definitions}
    if fn not in dispatch:
        print(f"Unknown function '{fn}'. Choose: overview, breakdown, definitions")
        sys.exit(1)
    print(json.dumps(dispatch[fn](), indent=2))
