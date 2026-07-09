"""
NeuroLab Readiness Dashboard
Deployment readiness metrics for NeuroLab AI in a real neuro-lab setting.
Reads config/neurolab_readiness.json and computes stakeholder coverage,
process maturity, functionality gaps, business-case levers, and phase progress.
"""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE / "config" / "neurolab_readiness.json"


def _load_config() -> dict:
    """Load the neurolab readiness config, returning empty dict on missing file."""
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _count_by_status(items: list, status_key: str = "status") -> dict:
    """Count items by their status field."""
    counts = {"built": 0, "partial": 0, "missing": 0}
    for item in items:
        s = item.get(status_key, "missing")
        if s in counts:
            counts[s] += 1
    return counts


def _pct(num: int, den: int) -> float:
    """Safe percentage rounded to 1 decimal."""
    return round(num / den * 100, 1) if den > 0 else 0.0


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------

def overview() -> dict:
    cfg = _load_config()
    if not cfg:
        return {"error": "neurolab_readiness.json not found"}

    stakeholders = cfg.get("stakeholders", [])
    processes = cfg.get("processes", [])
    functionality = cfg.get("functionality", [])
    phases = cfg.get("implementation_phases", [])

    # --- stakeholder aggregates ---
    total_built_items = sum(len(s.get("built", [])) for s in stakeholders)
    total_missing_items = sum(len(s.get("missing", [])) for s in stakeholders)
    total_stakeholder_items = total_built_items + total_missing_items

    # --- process counts ---
    proc_counts = _count_by_status(processes)
    processes_built = proc_counts["built"]
    processes_partial = proc_counts["partial"]

    # --- functionality counts ---
    func_counts = _count_by_status(functionality)
    functionality_built = func_counts["built"]
    functionality_missing = func_counts["missing"]

    # --- overall readiness ---
    # built = stakeholder built items + processes built + functionality built
    # total = all stakeholder items + all processes + all functionality
    total_denom = total_stakeholder_items + len(processes) + len(functionality)
    total_num = total_built_items + processes_built + functionality_built
    readiness_pct = _pct(total_num, total_denom)

    # --- stakeholder readiness for radar ---
    stakeholder_readiness = []
    for s in stakeholders:
        b = len(s.get("built", []))
        m = len(s.get("missing", []))
        stakeholder_readiness.append({
            "role": s["role"],
            "icon": s.get("icon", ""),
            "built_count": b,
            "missing_count": m,
            "readiness_pct": _pct(b, b + m),
        })

    # --- phase counts ---
    phase_counts = _count_by_status(phases)

    # --- readiness radar across dimensions ---
    stakeholder_pct = _pct(total_built_items, total_stakeholder_items)
    process_pct = _pct(processes_built, len(processes)) if processes else 0.0
    func_pct = _pct(functionality_built, len(functionality)) if functionality else 0.0
    phase_pct = _pct(phase_counts["built"], len(phases)) if phases else 0.0

    return {
        "kpis": {
            "total_stakeholders": len(stakeholders),
            "total_processes": len(processes),
            "processes_built": processes_built,
            "processes_partial": processes_partial,
            "total_functionality": len(functionality),
            "functionality_built": functionality_built,
            "functionality_missing": functionality_missing,
            "total_built_items": total_built_items,
            "total_missing_items": total_missing_items,
            "readiness_pct": readiness_pct,
        },
        "stakeholder_readiness": stakeholder_readiness,
        "process_status": [
            {"name": p["name"], "status": p.get("status", "missing")}
            for p in processes
        ],
        "functionality_status": [
            {"capability": f["capability"], "status": f.get("status", "missing")}
            for f in functionality
        ],
        "phase_progress": [
            {"phase": ph["phase"], "scope": ph.get("scope", ""), "status": ph.get("status", "missing")}
            for ph in phases
        ],
        "readiness_radar": {
            "dimensions": ["Stakeholders", "Processes", "Functionality", "Phases"],
            "values": [stakeholder_pct, process_pct, func_pct, phase_pct],
        },
    }


# ---------------------------------------------------------------------------
# breakdown
# ---------------------------------------------------------------------------

_GAP_KEYWORDS = {
    "integration": ["integration", "HL7", "FHIR", "EMR", "HIS", "DICOM", "sync", "push", "streaming"],
    "device": ["amplifier", "impedance", "montage", "Natus", "Nihon Kohden", "Compumedics", "device", "acquisition"],
    "compliance": ["HIPAA", "consent", "audit", "FDA", "CE", "regulatory", "e-signature", "compliance"],
    "clinical_tooling": ["scales", "scoring", "battery", "PHQ", "GAD", "NDDI", "WAIS", "WMS", "normative", "validated", "COPM", "FIM"],
    "workflow": ["scheduling", "worklist", "billing", "claims", "referral", "triage", "pre-auth", "insurance"],
    "user_experience": ["mobile", "alerts", "portal", "player", "viewer", "side-by-side", "editor", "reports", "tele-rehab"],
}


def _classify_gap(item_text: str) -> str:
    """Classify a missing item into a gap category based on keyword matching."""
    lower = item_text.lower()
    best_cat = "other"
    best_score = 0
    for cat, keywords in _GAP_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in lower)
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat


def breakdown() -> dict:
    cfg = _load_config()
    if not cfg:
        return {"error": "neurolab_readiness.json not found"}

    stakeholders = cfg.get("stakeholders", [])
    processes = cfg.get("processes", [])
    functionality = cfg.get("functionality", [])
    business_case = cfg.get("business_case", {})
    phases = cfg.get("implementation_phases", [])

    # --- stakeholder detail ---
    stakeholder_detail = []
    for s in stakeholders:
        b = s.get("built", [])
        m = s.get("missing", [])
        stakeholder_detail.append({
            "role": s["role"],
            "icon": s.get("icon", ""),
            "built": b,
            "missing": m,
            "readiness_pct": _pct(len(b), len(b) + len(m)),
        })

    # --- process detail ---
    process_detail = [
        {"name": p["name"], "status": p.get("status", "missing"), "maps_to": p.get("maps_to", "")}
        for p in processes
    ]

    # --- functionality detail ---
    functionality_detail = [
        {"capability": f["capability"], "status": f.get("status", "missing")}
        for f in functionality
    ]

    # --- implementation roadmap (enriched) ---
    implementation_roadmap = []
    for i, ph in enumerate(phases):
        implementation_roadmap.append({
            "phase": ph["phase"],
            "scope": ph.get("scope", ""),
            "status": ph.get("status", "missing"),
            "order": i,
            "is_current": ph.get("status") == "built" and (
                i + 1 >= len(phases) or phases[i + 1].get("status") != "built"
            ),
        })

    # --- gap analysis per stakeholder ---
    gap_analysis = []
    for s in stakeholders:
        missing = s.get("missing", [])
        if not missing:
            continue
        cats: dict[str, list[str]] = {}
        for item in missing:
            cat = _classify_gap(item)
            cats.setdefault(cat, []).append(item)
        gap_analysis.append({
            "role": s["role"],
            "icon": s.get("icon", ""),
            "total_gaps": len(missing),
            "categories": cats,
        })

    return {
        "stakeholder_detail": stakeholder_detail,
        "process_detail": process_detail,
        "functionality_detail": functionality_detail,
        "business_case": {
            "cost_decrease": business_case.get("cost_decrease", []),
            "revenue_increase": business_case.get("revenue_increase", []),
            "productivity_increase": business_case.get("productivity_increase", []),
        },
        "implementation_roadmap": implementation_roadmap,
        "gap_analysis": gap_analysis,
    }


# ---------------------------------------------------------------------------
# definitions
# ---------------------------------------------------------------------------

def definitions() -> dict:
    return {
        "terms": [
            {
                "term": "NeuroLab",
                "definition": (
                    "A clinical neuro-diagnostics laboratory that performs EEG, video-EEG, "
                    "ambulatory EEG, and related neurophysiology studies for diagnosis and "
                    "monitoring of neurological conditions."
                ),
            },
            {
                "term": "Readiness Score",
                "definition": (
                    "Percentage of total items (stakeholder capabilities, processes, and "
                    "functionality) that are fully built, relative to the total needed for "
                    "production deployment."
                ),
            },
            {
                "term": "Stakeholder Coverage",
                "definition": (
                    "For each clinical role (neurologist, technician, etc.), the fraction of "
                    "required capabilities that have been built vs. those still missing."
                ),
            },
            {
                "term": "Process Maturity",
                "definition": (
                    "Status of each end-to-end clinical workflow: 'built' (fully functional), "
                    "'partial' (some steps automated), or 'missing' (not yet implemented)."
                ),
            },
            {
                "term": "Implementation Phase",
                "definition": (
                    "A stage in the deployment roadmap (Pilot, Foundation, Integration, "
                    "Clinical, Business, Regulatory), each gating the next."
                ),
            },
            {
                "term": "Gap Analysis",
                "definition": (
                    "Per-stakeholder breakdown of missing capabilities classified into gap "
                    "categories: integration, device, compliance, clinical tooling, workflow, "
                    "and user experience."
                ),
            },
            {
                "term": "Business Case Lever",
                "definition": (
                    "A specific mechanism by which the AI layer produces financial value: "
                    "cost decrease (efficiency), revenue increase (throughput/reach), or "
                    "productivity increase (staff output)."
                ),
            },
            {
                "term": "EMR/HIS Integration",
                "definition": (
                    "Connection to the hospital's Electronic Medical Record or Health "
                    "Information System via HL7 FHIR or similar standards, enabling "
                    "bidirectional data flow."
                ),
            },
            {
                "term": "Governed AI Layer",
                "definition": (
                    "The explainable, auditable AI decision system (SHAP, fairness gates, "
                    "human-override) that is the core IP — designed to plug into existing "
                    "clinical IT infrastructure rather than replace it."
                ),
            },
            {
                "term": "RBAC",
                "definition": (
                    "Role-Based Access Control — restricting system access based on the "
                    "user's clinical role (neurologist, technician, admin, etc.)."
                ),
            },
        ],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    fn_map = {"overview": overview, "breakdown": breakdown, "definitions": definitions}
    target = sys.argv[1] if len(sys.argv) > 1 else "overview"
    func = fn_map.get(target)
    if func is None:
        print(f"Unknown function: {target}. Choose from: {', '.join(fn_map)}")
        sys.exit(1)
    print(json.dumps(func(), indent=2))
