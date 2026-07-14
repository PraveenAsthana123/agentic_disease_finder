"""Dataset Requirements Dashboard — completeness tracking, tier compliance,
and gap analysis from config/data_requirements.json.

Sources:
- config/data_requirements.json (9 categories, 3 tiers, control groups, artifact template)
"""

import json
from pathlib import Path
from datetime import datetime, timezone

CONFIG = Path(__file__).resolve().parent.parent / "config" / "data_requirements.json"

# Category priority for gap sorting (lower = higher priority)
_CATEGORY_PRIORITY = {
    "2. Clinical": 0,
    "1. EEG Signal (for AI model dev)": 1,
    "3. Medication": 2,
    "4. Imaging": 3,
    "5. Neuropsychological": 4,
    "6. Outcome": 5,
    "7. Governance (DBA-critical)": 6,
    "8. Data Quality": 7,
    "9. Demographics": 8,
}


def _load():
    if not CONFIG.exists():
        return None
    with open(CONFIG) as f:
        return json.load(f)


def _match_tier_item(tier_item, all_items):
    """Find the best-matching item from all_items for a tier requirement string.
    Returns (name, status) or (tier_item, 'missing') if no match.
    Prefers exact matches, then shortest substring match, then fuzzy."""
    lower = tier_item.lower()
    # Exact match first
    for item in all_items:
        if lower == item["name"].lower():
            return item["name"], item["status"]
    # Substring match — prefer shortest name (most specific match)
    substring_matches = []
    for item in all_items:
        name_lower = item["name"].lower()
        if lower in name_lower or name_lower in lower:
            substring_matches.append(item)
    if substring_matches:
        best = min(substring_matches, key=lambda x: len(x["name"]))
        return best["name"], best["status"]
    # Fuzzy: check if any significant word overlaps
    stop = {"data", "the", "a", "of", "and", "for"}
    tier_words = set(lower.split()) - stop
    for item in all_items:
        name_words = set(item["name"].lower().split()) - stop
        overlap = tier_words & name_words
        if len(overlap) >= 1 and len(overlap) / max(len(tier_words), 1) >= 0.5:
            return item["name"], item["status"]
    return tier_item, "missing"


def overview():
    """Total items, status distribution, per-category completeness,
    tier compliance, and top gaps."""
    data = _load()
    if not data:
        return {"available": False, "note": "config/data_requirements.json not found"}

    categories = data.get("categories", [])
    tiers = data.get("tiers", {})

    # Flatten all items
    all_items = []
    for cat in categories:
        for item in cat.get("items", []):
            all_items.append({**item, "_category": cat["category"]})

    total = len(all_items)
    present_count = sum(1 for i in all_items if i["status"] == "present")
    partial_count = sum(1 for i in all_items if i["status"] == "partial")
    missing_count = sum(1 for i in all_items if i["status"] == "missing")

    # Completeness: present=100%, partial=50%, missing=0%
    completeness_score = (present_count * 100 + partial_count * 50) / total if total else 0
    overall_completeness = round(completeness_score, 1)

    # Per-category summary
    category_summary = []
    for cat in categories:
        items = cat.get("items", [])
        c_total = len(items)
        c_present = sum(1 for i in items if i["status"] == "present")
        c_partial = sum(1 for i in items if i["status"] == "partial")
        c_missing = sum(1 for i in items if i["status"] == "missing")
        c_pct = round((c_present * 100 + c_partial * 50) / c_total, 1) if c_total else 0
        category_summary.append({
            "category": cat["category"],
            "total": c_total,
            "present": c_present,
            "partial": c_partial,
            "missing": c_missing,
            "completeness_pct": c_pct,
        })

    # Tier compliance
    tier_compliance = []
    for tier_key in ["tier1_mandatory", "tier2_recommended", "tier3_dba_excellent"]:
        tier_items = tiers.get(tier_key, [])
        t_present = 0
        t_partial = 0
        t_missing = 0
        for ti in tier_items:
            _, status = _match_tier_item(ti, all_items)
            if status == "present":
                t_present += 1
            elif status == "partial":
                t_partial += 1
            else:
                t_missing += 1
        t_required = len(tier_items)
        t_pct = round((t_present * 100 + t_partial * 50) / t_required, 1) if t_required else 0
        tier_compliance.append({
            "tier": tier_key,
            "items_required": t_required,
            "items_present": t_present,
            "items_partial": t_partial,
            "items_missing": t_missing,
            "compliance_pct": t_pct,
        })

    # Status distribution (for pie chart)
    status_distribution = {
        "present": present_count,
        "partial": partial_count,
        "missing": missing_count,
    }

    # Top gaps: missing items sorted by category priority (clinical before demographics)
    missing_items = [i for i in all_items if i["status"] == "missing"]
    missing_items.sort(key=lambda x: _CATEGORY_PRIORITY.get(x["_category"], 99))
    top_gaps = [
        {"name": i["name"], "category": i["_category"]}
        for i in missing_items
    ]

    return {
        "available": True,
        "title": data.get("title", ""),
        "updated_at": data.get("updated_at", ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_items": total,
        "present_count": present_count,
        "partial_count": partial_count,
        "missing_count": missing_count,
        "overall_completeness": overall_completeness,
        "category_summary": category_summary,
        "tier_compliance": tier_compliance,
        "status_distribution": status_distribution,
        "top_gaps": top_gaps,
    }


def breakdown():
    """Full detail per category, control groups, artifact template, tier detail."""
    data = _load()
    if not data:
        return {"available": False, "note": "config/data_requirements.json not found"}

    categories_out = []
    all_items = []
    for cat in data.get("categories", []):
        items = cat.get("items", [])
        categories_out.append({
            "category": cat["category"],
            "items": [{"name": i["name"], "status": i["status"],
                        "note": i.get("note", "")} for i in items],
        })
        all_items.extend(items)

    # Artifact template grouped by category
    artifact_raw = data.get("artifact_template", [])
    artifact_grouped = {}
    for art in artifact_raw:
        grp = art.get("category", "Other")
        artifact_grouped.setdefault(grp, []).append({
            "type": art["type"],
            "mandatory": art.get("mandatory", False),
        })

    # Tier detail: for each tier, list items with their resolved status
    tiers = data.get("tiers", {})
    tier_detail = []
    for tier_key in ["tier1_mandatory", "tier2_recommended", "tier3_dba_excellent"]:
        tier_items = tiers.get(tier_key, [])
        resolved = []
        for ti in tier_items:
            matched_name, status = _match_tier_item(ti, all_items)
            resolved.append({
                "requirement": ti,
                "matched_item": matched_name,
                "status": status,
            })
        tier_detail.append({"tier": tier_key, "items": resolved})

    return {
        "available": True,
        "categories": categories_out,
        "control_groups": data.get("control_groups", {}),
        "artifact_template": artifact_grouped,
        "tier_detail": tier_detail,
    }


def definitions():
    """Status definitions, tier meanings, completeness formula, data source."""
    return {
        "statuses": {
            "present": "Data field exists, is populated, and usable for analysis.",
            "partial": "Schema/table exists or loader supports it, but real clinical data is missing or incomplete.",
            "missing": "Not yet available in the dataset — a gap that needs to be filled.",
        },
        "tiers": {
            "tier1_mandatory": "Minimum viable dataset — without these, no credible AI model can be trained or validated.",
            "tier2_recommended": "Strongly recommended for clinical-grade results and regulatory submissions.",
            "tier3_dba_excellent": "Gold-standard governance artefacts for DBA-level explainability and auditability.",
        },
        "completeness_formula": (
            "completeness = (present_count * 100 + partial_count * 50) / total_items. "
            "Present counts as 100%, partial as 50%, missing as 0%."
        ),
        "data_source": "config/data_requirements.json",
    }


if __name__ == "__main__":
    import pprint
    print("=== OVERVIEW ===")
    pprint.pprint(overview())
    print("\n=== BREAKDOWN ===")
    pprint.pprint(breakdown())
    print("\n=== DEFINITIONS ===")
    pprint.pprint(definitions())
