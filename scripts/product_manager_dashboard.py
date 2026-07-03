"""Product Manager Dashboard — roadmap readiness, stakeholder coverage,
business-case KPIs, process maturity, and module completion from config registries."""

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
    """Summary KPIs: module completion, stakeholder readiness, process maturity,
    implementation phase status, business-case levers."""
    nr = _load('neurolab_readiness.json')
    pm = _load('patient_module.json')
    am = _load('admin_module.json')

    if not nr:
        return {"available": False, "note": "neurolab_readiness.json missing"}

    # --- Stakeholder readiness ---
    stakeholders = nr.get('stakeholders', [])
    stakeholder_summary = []
    total_built_caps = 0
    total_missing_caps = 0
    for sh in stakeholders:
        built = sh.get('built', [])
        missing = sh.get('missing', [])
        total = len(built) + len(missing)
        pct = round(len(built) / total * 100, 1) if total else 0
        total_built_caps += len(built)
        total_missing_caps += len(missing)
        stakeholder_summary.append({
            "role": sh.get('role', '?'),
            "icon": sh.get('icon', ''),
            "built": len(built),
            "missing": len(missing),
            "total": total,
            "readiness_pct": pct,
        })
    overall_readiness = round(total_built_caps / (total_built_caps + total_missing_caps) * 100, 1) if (total_built_caps + total_missing_caps) else 0

    # --- Process maturity ---
    processes = nr.get('processes', [])
    proc_status = Counter(p.get('status', '?') for p in processes)
    proc_built = proc_status.get('built', 0)
    proc_partial = proc_status.get('partial', 0)
    proc_missing = proc_status.get('missing', 0)
    proc_maturity = round((proc_built + proc_partial * 0.5) / len(processes) * 100, 1) if processes else 0

    # --- Functionality coverage ---
    funcs = nr.get('functionality', [])
    func_status = Counter(f.get('status', '?') for f in funcs)
    func_built = func_status.get('built', 0)
    func_partial = func_status.get('partial', 0)
    func_missing = func_status.get('missing', 0)

    # --- Implementation phases ---
    phases = nr.get('implementation_phases', [])
    phase_status = Counter(p.get('status', '?') for p in phases)

    # --- Module completion (patient + admin) ---
    patient_sections = pm.get('sections', []) if pm else []
    pat_built = sum(1 for s in patient_sections if s.get('status') == 'built')
    pat_total = len(patient_sections)

    ops_dashboards = am.get('ops_dashboards', []) if am else []
    ops_built = sum(1 for d in ops_dashboards if d.get('status') == 'built')
    ops_total = len(ops_dashboards)

    # --- Business case levers ---
    bc = nr.get('business_case', {})
    cost_levers = len(bc.get('cost_decrease', []))
    revenue_levers = len(bc.get('revenue_increase', []))
    productivity_levers = len(bc.get('productivity_increase', []))

    return {
        "available": True,
        "summary": {
            "overall_readiness_pct": overall_readiness,
            "total_stakeholder_roles": len(stakeholders),
            "total_built_capabilities": total_built_caps,
            "total_missing_capabilities": total_missing_caps,
            "process_maturity_pct": proc_maturity,
            "processes_built": proc_built,
            "processes_partial": proc_partial,
            "processes_missing": proc_missing,
            "functionality_built": func_built,
            "functionality_partial": func_partial,
            "functionality_missing": func_missing,
            "patient_module_built": pat_built,
            "patient_module_total": pat_total,
            "ops_dashboards_built": ops_built,
            "ops_dashboards_total": ops_total,
            "phases_built": phase_status.get('built', 0),
            "phases_missing": phase_status.get('missing', 0),
            "total_phases": len(phases),
            "business_case_levers": cost_levers + revenue_levers + productivity_levers,
        },
        "stakeholder_readiness": sorted(stakeholder_summary, key=lambda s: s['readiness_pct'], reverse=True),
        "process_status": [
            {"name": p.get('name', '?'), "status": p.get('status', '?')}
            for p in processes
        ],
        "functionality_status": [
            {"capability": f.get('capability', '?'), "status": f.get('status', '?')}
            for f in funcs
        ],
        "phase_distribution": {
            "built": phase_status.get('built', 0),
            "missing": phase_status.get('missing', 0),
        },
    }


def breakdown():
    """Detailed view: per-stakeholder built/missing lists, business-case levers,
    implementation phases, module section details."""
    nr = _load('neurolab_readiness.json')
    pm = _load('patient_module.json')
    am = _load('admin_module.json')

    if not nr:
        return {"available": False}

    # --- Stakeholder detail ---
    stakeholders = nr.get('stakeholders', [])
    stakeholder_detail = []
    for sh in stakeholders:
        stakeholder_detail.append({
            "role": sh.get('role', '?'),
            "icon": sh.get('icon', ''),
            "built": sh.get('built', []),
            "missing": sh.get('missing', []),
        })

    # --- Business case ---
    bc = nr.get('business_case', {})
    cost_levers = [{"lever": c.get('lever', '?'), "impact": c.get('impact', '')} for c in bc.get('cost_decrease', [])]
    revenue_levers = [{"lever": r.get('lever', '?'), "impact": r.get('impact', '')} for r in bc.get('revenue_increase', [])]
    productivity_levers = [{"lever": p.get('lever', '?'), "impact": p.get('impact', '')} for p in bc.get('productivity_increase', [])]

    # --- Implementation phases ---
    phases = [{"phase": p.get('phase', '?'), "scope": p.get('scope', ''), "status": p.get('status', '?')} for p in nr.get('implementation_phases', [])]

    # --- Patient module sections ---
    patient_sections = []
    if pm:
        for s in pm.get('sections', []):
            patient_sections.append({
                "section": s.get('section', '?'),
                "fields": s.get('fields', '?'),
                "status": s.get('status', '?'),
            })

    # --- Ops dashboards ---
    ops = []
    if am:
        for d in am.get('ops_dashboards', []):
            ops.append({
                "label": d.get('label', '?'),
                "purpose": d.get('purpose', ''),
                "status": d.get('status', '?'),
            })

    return {
        "available": True,
        "stakeholders": stakeholder_detail,
        "cost_levers": cost_levers,
        "revenue_levers": revenue_levers,
        "productivity_levers": productivity_levers,
        "phases": phases,
        "patient_sections": patient_sections,
        "ops_dashboards": ops,
        "strategy": nr.get('strategy', ''),
    }


def definitions():
    """Product management terminology and KPI definitions."""
    return {
        "available": True,
        "definitions": [
            {"term": "Overall Readiness", "definition": "Percentage of stakeholder capabilities that are built vs. total (built + missing). Measures how close the platform is to covering all clinical role needs."},
            {"term": "Process Maturity", "definition": "Weighted completion of clinical workflow processes: built counts 100%, partial counts 50%, missing counts 0%. Higher maturity = more end-to-end workflows operational."},
            {"term": "Stakeholder Readiness", "definition": "Per-role percentage of capabilities built vs. total needed. A role at 100% has all required features; lower percentages indicate integration or feature gaps."},
            {"term": "Implementation Phase", "definition": "A deployment stage (Pilot → Foundation → Integration → Clinical → Business → Regulatory). Each phase builds on the previous and unlocks new capabilities."},
            {"term": "Business Case Lever", "definition": "A specific mechanism (cost decrease, revenue increase, or productivity gain) through which the AI platform delivers measurable value."},
            {"term": "Module Completion", "definition": "Ratio of built vs. total sections in the Patient Module (8 sections, ~1,250 fields) and Admin Ops Dashboards (10 dashboards)."},
            {"term": "Functionality Coverage", "definition": "Status of core platform capabilities (AI governance, RAG, auth, EMR, streaming, billing, etc.) — built, partial, or missing."},
            {"term": "Cost Decrease Lever", "definition": "An operational efficiency that reduces direct costs — e.g., AI pre-read reducing neurologist review time by 30-50% per study."},
            {"term": "Revenue Increase Lever", "definition": "A capability that generates new or incremental revenue — e.g., tele-EEG reads serving remote referrals, or higher throughput per day."},
            {"term": "Productivity Lever", "definition": "A workflow improvement that increases output per person — e.g., auto-QC + worklist enabling more studies per technician shift."},
        ],
    }


if __name__ == "__main__":
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2, default=str))
    print("\n=== BREAKDOWN ===")
    print(json.dumps(breakdown(), indent=2, default=str))
