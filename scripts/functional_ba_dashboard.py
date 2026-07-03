"""Functional / BA Dashboard — requirements traceability, use-case mapping,
acceptance-criteria tracking, and UAT readiness across all clinical roles.

Reads: config/stories_and_tests.json, config/role_tests.json,
       config/neurolab_readiness.json, config/patient_module.json,
       config/admin_module.json
"""

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
    """Summary KPIs: requirements count, traceability %, acceptance coverage,
    UAT readiness by role, process maturity, functionality coverage."""
    st = _load('stories_and_tests.json')
    rt = _load('role_tests.json')
    nr = _load('neurolab_readiness.json')
    pm = _load('patient_module.json')
    am = _load('admin_module.json')

    if not st or not rt:
        return {"available": False, "note": "Config files missing"}

    # --- User stories as requirements ---
    user_stories = st.get('user_stories', [])
    demo_stories = st.get('demo_stories', [])
    testing = st.get('testing', [])

    # --- Role test matrix (acceptance criteria proxy) ---
    roles = rt.get('roles', [])
    all_tests = []
    role_uat = []
    for role_obj in roles:
        rname = role_obj.get('role', '?')
        tests = role_obj.get('tests', [])
        counts = Counter(t.get('status', '?') for t in tests)
        total = len(tests)
        pass_n = counts.get('pass', 0)
        partial_n = counts.get('partial', 0)
        planned_n = counts.get('planned', 0)
        uat_pct = round((pass_n + partial_n * 0.5) / total * 100, 1) if total else 0
        role_uat.append({
            "role": rname,
            "total_criteria": total,
            "accepted": pass_n,
            "in_progress": partial_n,
            "pending": planned_n,
            "uat_readiness_pct": uat_pct,
        })
        all_tests.extend(tests)

    total_criteria = len(all_tests)
    status_counts = Counter(t.get('status', '?') for t in all_tests)
    accepted = status_counts.get('pass', 0)
    in_progress = status_counts.get('partial', 0)
    pending = status_counts.get('planned', 0)
    overall_acceptance_pct = round(
        (accepted + in_progress * 0.5) / total_criteria * 100, 1
    ) if total_criteria else 0

    # --- Process maturity (from neurolab_readiness) ---
    processes = (nr or {}).get('processes', [])
    proc_counts = Counter(p.get('status', '?') for p in processes)
    proc_total = len(processes)
    proc_built = proc_counts.get('built', 0)
    proc_partial = proc_counts.get('partial', 0)
    proc_missing = proc_counts.get('missing', 0)
    proc_maturity_pct = round(
        (proc_built + proc_partial * 0.5) / proc_total * 100, 1
    ) if proc_total else 0

    # --- Functionality coverage ---
    funcs = (nr or {}).get('functionality', [])
    func_counts = Counter(f.get('status', '?') for f in funcs)
    func_total = len(funcs)
    func_built = func_counts.get('built', 0)
    func_partial = func_counts.get('partial', 0)
    func_missing = func_counts.get('missing', 0)

    # --- Patient module sections ---
    pm_sections = (pm or {}).get('sections', [])
    pm_built = sum(1 for s in pm_sections if s.get('status') == 'built')
    pm_total = len(pm_sections)

    # --- Admin module roles ---
    am_roles = (am or {}).get('team_roles', [])
    am_built = sum(1 for r in am_roles if r.get('status') == 'built')
    am_total = len(am_roles)

    # --- Testing dimensions ---
    dim_built = sum(1 for t in testing if t.get('status') == 'built')
    dim_partial = sum(1 for t in testing if t.get('status') == 'partial')
    dim_total = len(testing)

    # --- Requirements by persona ---
    persona_counts = Counter(us.get('persona', '?') for us in user_stories)
    requirements_by_persona = [
        {"persona": p, "count": c}
        for p, c in sorted(persona_counts.items(), key=lambda x: -x[1])
    ]

    # --- Acceptance status distribution ---
    acceptance_distribution = [
        {"status": "Accepted", "count": accepted},
        {"status": "In Progress", "count": in_progress},
        {"status": "Pending", "count": pending},
    ]

    # --- Process status distribution ---
    process_distribution = [
        {"status": "Built", "count": proc_built},
        {"status": "Partial", "count": proc_partial},
        {"status": "Missing", "count": proc_missing},
    ]

    # --- Functionality status distribution ---
    functionality_distribution = [
        {"status": "Built", "count": func_built},
        {"status": "Partial", "count": func_partial},
        {"status": "Missing", "count": func_missing},
    ]

    return {
        "available": True,
        "summary": {
            "total_requirements": len(user_stories),
            "total_acceptance_criteria": total_criteria,
            "accepted": accepted,
            "in_progress": in_progress,
            "pending": pending,
            "overall_acceptance_pct": overall_acceptance_pct,
            "process_maturity_pct": proc_maturity_pct,
            "total_processes": proc_total,
            "processes_built": proc_built,
            "functionality_built": func_built,
            "functionality_total": func_total,
            "testing_dimensions": dim_total,
            "dimensions_built": dim_built,
            "patient_sections_built": pm_built,
            "patient_sections_total": pm_total,
            "admin_roles_built": am_built,
            "admin_roles_total": am_total,
        },
        "role_uat_readiness": sorted(role_uat, key=lambda r: r['uat_readiness_pct'], reverse=True),
        "requirements_by_persona": requirements_by_persona,
        "acceptance_distribution": acceptance_distribution,
        "process_distribution": process_distribution,
        "functionality_distribution": functionality_distribution,
    }


def breakdown():
    """Detailed breakdown: requirements traceability matrix, process detail,
    functionality gaps, and per-role acceptance criteria."""
    st = _load('stories_and_tests.json')
    rt = _load('role_tests.json')
    nr = _load('neurolab_readiness.json')

    if not st or not rt:
        return {"available": False}

    # --- Requirements traceability: story → endpoint → test status ---
    user_stories = st.get('user_stories', [])
    roles = rt.get('roles', [])
    role_test_map = {}
    for role_obj in roles:
        rname = role_obj.get('role', '?')
        role_test_map[rname] = role_obj.get('tests', [])

    traceability = []
    for us in user_stories:
        persona = us.get('persona', '?')
        tests = role_test_map.get(persona, [])
        counts = Counter(t.get('status', '?') for t in tests)
        traceability.append({
            "persona": persona,
            "story": us.get('story', ''),
            "endpoint": us.get('endpoint', ''),
            "linked_tests": len(tests),
            "pass": counts.get('pass', 0),
            "partial": counts.get('partial', 0),
            "planned": counts.get('planned', 0),
        })

    # --- Demo stories ---
    demo_stories = [
        {
            "title": ds.get('title', '?'),
            "script": ds.get('script', ''),
            "shows": ds.get('shows', ''),
        }
        for ds in st.get('demo_stories', [])
    ]

    # --- Testing dimensions ---
    testing_dims = [
        {
            "dimension": t.get('dim', '?'),
            "tests": t.get('tests', ''),
            "method": t.get('how', ''),
            "status": t.get('status', '?'),
        }
        for t in st.get('testing', [])
    ]

    # --- Process detail ---
    processes = [
        {"name": p.get('name', '?'), "status": p.get('status', '?')}
        for p in (nr or {}).get('processes', [])
    ]

    # --- Functionality gaps ---
    funcs = [
        {"capability": f.get('capability', '?'), "status": f.get('status', '?')}
        for f in (nr or {}).get('functionality', [])
    ]

    # --- Stakeholder gap analysis ---
    stakeholders = (nr or {}).get('stakeholders', [])
    stakeholder_gaps = []
    for sh in stakeholders:
        built = sh.get('built', [])
        missing = sh.get('missing', [])
        total = len(built) + len(missing)
        built_pct = round(len(built) / total * 100, 1) if total else 0
        stakeholder_gaps.append({
            "role": sh.get('role', '?'),
            "built_count": len(built),
            "missing_count": len(missing),
            "coverage_pct": built_pct,
            "built_items": built,
            "missing_items": missing,
        })

    # --- Per-role acceptance criteria detail ---
    role_criteria = []
    for role_obj in roles:
        rname = role_obj.get('role', '?')
        tests = role_obj.get('tests', [])
        role_criteria.append({
            "role": rname,
            "criteria": [
                {
                    "dimension": t.get('dim', '?'),
                    "criterion": t.get('case', ''),
                    "status": t.get('status', '?'),
                }
                for t in tests
            ],
        })

    return {
        "available": True,
        "traceability": traceability,
        "demo_stories": demo_stories,
        "testing_dimensions": testing_dims,
        "processes": processes,
        "functionality": funcs,
        "stakeholder_gaps": stakeholder_gaps,
        "role_criteria": role_criteria,
    }


def definitions():
    """Functional analysis and business analysis terminology."""
    return {
        "available": True,
        "definitions": [
            {"term": "Requirement", "definition": "A user story capturing a persona's need, expressed as 'As a [role], I [action] so that [outcome]'. Each maps to one or more API endpoints and acceptance criteria."},
            {"term": "Acceptance Criteria", "definition": "A testable condition that must be satisfied for a requirement to be considered complete. Mapped to testing dimensions (API, Data, Model, Process, Manual)."},
            {"term": "UAT Readiness", "definition": "Percentage of acceptance criteria that are accepted (pass) or in progress (partial, weighted at 50%). Indicates how ready a role's workflow is for User Acceptance Testing."},
            {"term": "Requirements Traceability", "definition": "The mapping from user stories through API endpoints to test cases, ensuring every requirement has verifiable acceptance criteria."},
            {"term": "Process Maturity", "definition": "Percentage of clinical workflow processes (referral, scheduling, acquisition, reporting, etc.) that are built or partially implemented."},
            {"term": "Functionality Coverage", "definition": "Count of system capabilities (AI governance, EMR integration, scheduling, etc.) that are built vs. missing vs. partially implemented."},
            {"term": "Stakeholder Gap Analysis", "definition": "Per-role breakdown of what features are built vs. missing, identifying which clinical roles have the most unmet requirements."},
            {"term": "Testing Dimension", "definition": "One of 9 verification categories (API, Process, Data, Model, Accuracy, Frontend, Backend, Pipeline, Manual) that together cover the full system."},
            {"term": "Use Case", "definition": "A specific interaction scenario between a user role and the system, including preconditions, steps, and expected outcomes."},
            {"term": "Demo Story", "definition": "A scripted 30-45 second demonstration scenario designed to showcase a governed workflow to stakeholders, with explicit 'shows' outcome."},
        ],
    }


if __name__ == "__main__":
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2, default=str))
    print("\n=== BREAKDOWN ===")
    print(json.dumps(breakdown(), indent=2, default=str))
