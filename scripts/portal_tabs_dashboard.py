"""Portal Tabs Dashboard — Patient Self-Service Portal tab layout
from config/portal_tabs.json.
11 tabs (forms, campaigns, notifications, alerts, inbox, medication,
therapy, assessments, reports, chat, history)."""

import json
from pathlib import Path

_CFG = Path(__file__).resolve().parent.parent / "config" / "portal_tabs.json"


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        raw = json.load(f)
    return raw[0] if isinstance(raw, list) else raw


# ── overview ────────────────────────────────────────────────────────────
def overview():
    """Summary KPIs: total tabs, status distribution, endpoint mapping."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "portal_tabs.json missing"}

    tabs = cfg.get("tabs", [])
    total = len(tabs)

    # Status counts
    status_counts = {}
    for t in tabs:
        s = t.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    built = status_counts.get("built", 0)
    planned = status_counts.get("planned", 0)
    partial = status_counts.get("partial", 0)

    # Count unique endpoints
    all_endpoints = set()
    for t in tabs:
        maps = t.get("maps_to", "")
        for ep in maps.split(","):
            ep = ep.strip()
            if ep:
                all_endpoints.add(ep.split(" ")[0] if " " in ep else ep)

    # Status distribution pie chart
    status_dist = [{"name": k.title(), "value": v} for k, v in status_counts.items() if v > 0]

    # Endpoints per tab bar chart
    endpoints_per_tab = []
    for t in tabs:
        maps = t.get("maps_to", "")
        ep_count = len([e.strip() for e in maps.split(",") if e.strip()])
        # Count + separated endpoints too
        for part in maps.split("+"):
            sub_parts = [s.strip() for s in part.split(",") if s.strip()]
            if len(sub_parts) > 1:
                ep_count = max(ep_count, len(sub_parts))
        # Recount properly: split by + and ,
        all_parts = []
        for seg in maps.replace(",", " + ").split("+"):
            seg = seg.strip()
            if seg.startswith("/api"):
                all_parts.append(seg)
            elif "/" in seg:
                all_parts.append(seg)
        real_count = len(all_parts) if all_parts else 1
        endpoints_per_tab.append({
            "name": t.get("label", t.get("id", "")),
            "value": real_count,
            "id": t.get("id", ""),
        })

    # Purpose length bar chart (content richness)
    purpose_lengths = [
        {
            "name": t.get("label", t.get("id", "")),
            "value": len(t.get("purpose", "")),
            "id": t.get("id", ""),
        }
        for t in tabs
    ]

    return {
        "available": True,
        "title": cfg.get("title", "Patient Self-Service Portal — Tabs"),
        "note": cfg.get("note", ""),
        "updated_at": cfg.get("updated_at", ""),
        "kpis": {
            "total_tabs": total,
            "built": built,
            "planned": planned,
            "partial": partial,
            "total_endpoints": len(all_endpoints),
        },
        "charts": {
            "status_distribution": status_dist,
            "endpoints_per_tab": endpoints_per_tab,
            "purpose_lengths": purpose_lengths,
        },
        "summary_table": [
            {
                "id": t.get("id", ""),
                "label": t.get("label", ""),
                "purpose": t.get("purpose", ""),
                "status": t.get("status", "unknown"),
                "maps_to": t.get("maps_to", ""),
            }
            for t in tabs
        ],
    }


# ── breakdown ───────────────────────────────────────────────────────────
def breakdown():
    """Per-tab details: purpose, endpoints, status."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "portal_tabs.json missing"}

    tabs = cfg.get("tabs", [])

    per_tab = []
    for t in tabs:
        maps_raw = t.get("maps_to", "")
        # Parse endpoints
        endpoints = []
        for seg in maps_raw.replace(",", " + ").split("+"):
            seg = seg.strip()
            if seg.startswith("/api") or seg.startswith("/{"):
                endpoints.append(seg)
            elif "/" in seg and not seg.startswith("http"):
                endpoints.append(seg)

        per_tab.append({
            "id": t.get("id", ""),
            "label": t.get("label", ""),
            "purpose": t.get("purpose", ""),
            "status": t.get("status", "unknown"),
            "maps_to": maps_raw,
            "endpoints": endpoints,
            "endpoint_count": len(endpoints),
        })

    # Group by status
    by_status = {}
    for t in per_tab:
        s = t["status"]
        if s not in by_status:
            by_status[s] = []
        by_status[s].append(t)

    return {
        "available": True,
        "tabs": per_tab,
        "by_status": by_status,
        "total": len(per_tab),
    }


# ── definitions ─────────────────────────────────────────────────────────
def definitions():
    """Legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "status_legend": [
            {"status": "built", "meaning": "Tab is live, endpoint returns real data"},
            {"status": "partial", "meaning": "Tab exists but some endpoints are stubs or incomplete"},
            {"status": "planned", "meaning": "Tab is designed but not yet implemented"},
        ],
        "glossary": [
            {"term": "Self-Service Portal", "definition": "Patient-facing web interface for managing their own health data and care interactions"},
            {"term": "MoCA", "definition": "Montreal Cognitive Assessment — brief screening for cognitive impairment"},
            {"term": "PHQ-9", "definition": "Patient Health Questionnaire-9 — depression screening instrument"},
            {"term": "RAG", "definition": "Retrieval-Augmented Generation — AI grounded in patient records"},
            {"term": "Auto-Scored", "definition": "Assessment automatically scored on submission using validated algorithms"},
            {"term": "SOS", "definition": "Emergency distress signal, triggers immediate caregiver notification"},
            {"term": "Secure Messaging", "definition": "HIPAA-compliant encrypted messaging between patient and care team"},
            {"term": "Seizure Diary", "definition": "Patient self-reported seizure event log with type, duration, triggers"},
            {"term": "PRO", "definition": "Patient-Reported Outcomes — validated self-assessment instruments"},
            {"term": "AED", "definition": "Antiepileptic Drug — medication for seizure control"},
            {"term": "Rehab Plan", "definition": "Structured physiotherapy/meditation/exercise program assigned by therapist"},
            {"term": "UTC Stamped", "definition": "All events stored with UTC timestamp plus local timezone display"},
        ],
        "clinical_notes": [
            "Patient portal tabs follow HIPAA minimum-necessary principle — patients see only their own data",
            "All assessment auto-scoring uses published scoring algorithms (MoCA, PHQ-9, GAD-7, QOLIE-31)",
            "Chat AI (RAG) is grounded in patient records only — no hallucinated medical advice",
            "Emergency SOS triggers are logged in transaction_log for audit trail compliance",
        ],
        "references": [
            {"label": "portal_tabs.json", "note": "Config source for self-service portal tab definitions"},
            {"label": "HIPAA §164.502(b)", "note": "Minimum necessary standard for patient data access"},
            {"label": "QOLIE-31", "note": "Quality of Life in Epilepsy Inventory — validated PRO instrument"},
            {"label": "MoCA (Nasreddine et al. 2005)", "note": "Montreal Cognitive Assessment validation study"},
        ],
    }
