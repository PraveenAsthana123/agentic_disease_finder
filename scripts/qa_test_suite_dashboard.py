"""QA Test Suite Dashboard — role-level test coverage matrix, testing dimension
status, user stories mapping, and demo-story readiness from config registries."""

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
    """Summary KPIs: total tests, pass/partial/planned counts, coverage %,
    per-dimension breakdown, per-role pass rate, user story count, demo count."""
    rt = _load('role_tests.json')
    st = _load('stories_and_tests.json')

    if not rt or not st:
        return {"available": False, "note": "Config files missing"}

    # --- Role test matrix ---
    roles = rt.get('roles', [])
    all_tests = []
    role_summaries = []
    for role_obj in roles:
        rname = role_obj.get('role', '?')
        tests = role_obj.get('tests', [])
        counts = Counter(t.get('status', '?') for t in tests)
        total = len(tests)
        pass_n = counts.get('pass', 0)
        role_summaries.append({
            "role": rname,
            "total": total,
            "pass": pass_n,
            "partial": counts.get('partial', 0),
            "planned": counts.get('planned', 0),
            "pass_rate": round(pass_n / total * 100, 1) if total else 0,
        })
        all_tests.extend(tests)

    total_tests = len(all_tests)
    status_counts = Counter(t.get('status', '?') for t in all_tests)
    pass_count = status_counts.get('pass', 0)
    partial_count = status_counts.get('partial', 0)
    planned_count = status_counts.get('planned', 0)
    coverage_pct = round((pass_count + partial_count * 0.5) / total_tests * 100, 1) if total_tests else 0

    # --- Dimension breakdown ---
    dim_counts = {}
    for t in all_tests:
        dim = t.get('dim', 'Unknown')
        st_val = t.get('status', '?')
        if dim not in dim_counts:
            dim_counts[dim] = {"dimension": dim, "pass": 0, "partial": 0, "planned": 0, "total": 0}
        dim_counts[dim][st_val] = dim_counts[dim].get(st_val, 0) + 1
        dim_counts[dim]["total"] += 1
    dim_breakdown = sorted(dim_counts.values(), key=lambda d: d['total'], reverse=True)

    # --- 9-Dimension testing matrix ---
    testing = st.get('testing', [])
    dim_matrix = []
    for item in testing:
        dim_matrix.append({
            "dimension": item.get('dim', '?'),
            "tests": item.get('tests', ''),
            "how": item.get('how', ''),
            "status": item.get('status', '?'),
        })
    matrix_built = sum(1 for d in dim_matrix if d['status'] == 'built')
    matrix_partial = sum(1 for d in dim_matrix if d['status'] == 'partial')

    # --- User stories & demo stories ---
    user_stories = st.get('user_stories', [])
    demo_stories = st.get('demo_stories', [])

    return {
        "available": True,
        "summary": {
            "total_tests": total_tests,
            "pass": pass_count,
            "partial": partial_count,
            "planned": planned_count,
            "coverage_pct": coverage_pct,
            "total_roles": len(roles),
            "user_stories": len(user_stories),
            "demo_stories": len(demo_stories),
            "testing_dimensions": len(dim_matrix),
            "dimensions_built": matrix_built,
            "dimensions_partial": matrix_partial,
        },
        "role_summaries": sorted(role_summaries, key=lambda r: r['pass_rate'], reverse=True),
        "dimension_breakdown": dim_breakdown,
        "dimension_matrix": dim_matrix,
        "status_distribution": {
            "pass": pass_count,
            "partial": partial_count,
            "planned": planned_count,
        },
    }


def breakdown():
    """Per-role detail: every test case with dimension, description, and status.
    Plus user stories with persona and endpoint mapping, and demo stories."""
    rt = _load('role_tests.json')
    st = _load('stories_and_tests.json')

    if not rt or not st:
        return {"available": False, "roles": [], "user_stories": [], "demo_stories": []}

    roles = rt.get('roles', [])
    role_details = []
    for role_obj in roles:
        rname = role_obj.get('role', '?')
        tests = role_obj.get('tests', [])
        role_details.append({
            "role": rname,
            "tests": [
                {
                    "dimension": t.get('dim', '?'),
                    "case": t.get('case', ''),
                    "status": t.get('status', '?'),
                }
                for t in tests
            ],
        })

    user_stories = st.get('user_stories', [])
    demo_stories = st.get('demo_stories', [])

    return {
        "available": True,
        "roles": role_details,
        "user_stories": [
            {
                "persona": us.get('persona', '?'),
                "story": us.get('story', ''),
                "endpoint": us.get('endpoint', ''),
            }
            for us in user_stories
        ],
        "demo_stories": [
            {
                "title": ds.get('title', '?'),
                "script": ds.get('script', ''),
                "shows": ds.get('shows', ''),
            }
            for ds in demo_stories
        ],
    }


def definitions():
    """QA testing terminology and methodology definitions."""
    return {
        "available": True,
        "definitions": [
            {"term": "Test Coverage", "definition": "Percentage of test cases that are passing or partially implemented. Calculated as (pass + 0.5 * partial) / total * 100."},
            {"term": "Testing Dimension", "definition": "A category of testing (API, Data, Model, Accuracy, Process, Frontend, Backend, Pipeline, Manual) that verifies a specific aspect of the system."},
            {"term": "Pass", "definition": "Test case is fully implemented and verified — the feature works as specified end-to-end."},
            {"term": "Partial", "definition": "Test case is partially implemented — the core logic exists but edge cases, UI polish, or full verification are pending."},
            {"term": "Planned", "definition": "Test case is defined but not yet implemented — the requirement is documented and scoped for future work."},
            {"term": "Role Testing Matrix", "definition": "A cross-tabulation of clinical/technical roles against testing dimensions, showing which role-specific workflows have been verified."},
            {"term": "User Story", "definition": "A requirement expressed as 'As a [persona], I [action] so that [outcome]' — maps personas to API endpoints and expected behavior."},
            {"term": "Demo Story", "definition": "A scripted demonstration scenario (30-45 seconds) designed to showcase a complete governed workflow for stakeholders."},
            {"term": "9-Dimension Matrix", "definition": "The project's testing framework covering API, Process, Data, Model, Accuracy, Frontend, Backend, Pipeline, and Manual testing across all system components."},
            {"term": "Pass Rate", "definition": "Per-role percentage of test cases with 'pass' status, indicating how thoroughly that role's workflow has been verified."},
        ],
    }


if __name__ == "__main__":
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2, default=str))
    print("\n=== BREAKDOWN ===")
    print(json.dumps(breakdown(), indent=2, default=str))
