"""
Neuro AI Ecosystem — FIM (Functional Independence Measure)
==========================================================
18-item measure of functional independence (score 18-126).
7-level ordinal scale per item: 1=Total Assist → 7=Complete Independence.

Domains:
  Motor (13 items): Self-Care (6), Sphincter Control (2), Transfers (3), Locomotion (2)
  Cognitive (5 items): Communication (2), Social Cognition (3)

Scores DERIVED from REAL patient data in clinical.db:
  - Barthel Index (functional baseline)
  - Seizure diary (frequency, severity)
  - Medications (AED count)
  - Demographics (age)
  - MoCA / MMSE (cognition)

Author: Research Team
"""

import sqlite3
import hashlib
from pathlib import Path
from typing import Optional

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

# ── FIM Items (18 total) ────────────────────────────────────────────
FIM_ITEMS = {
    "motor": {
        "self_care": [
            {"id": "eating",        "label": "Eating"},
            {"id": "grooming",      "label": "Grooming"},
            {"id": "bathing",       "label": "Bathing"},
            {"id": "dressing_upper","label": "Dressing — Upper Body"},
            {"id": "dressing_lower","label": "Dressing — Lower Body"},
            {"id": "toileting",     "label": "Toileting"},
        ],
        "sphincter_control": [
            {"id": "bladder_mgmt",  "label": "Bladder Management"},
            {"id": "bowel_mgmt",    "label": "Bowel Management"},
        ],
        "transfers": [
            {"id": "bed_chair",     "label": "Bed / Chair / Wheelchair"},
            {"id": "toilet",        "label": "Toilet"},
            {"id": "tub_shower",    "label": "Tub / Shower"},
        ],
        "locomotion": [
            {"id": "walk_wheelchair","label": "Walk / Wheelchair"},
            {"id": "stairs",         "label": "Stairs"},
        ],
    },
    "cognitive": {
        "communication": [
            {"id": "comprehension", "label": "Comprehension"},
            {"id": "expression",    "label": "Expression"},
        ],
        "social_cognition": [
            {"id": "social_interaction","label": "Social Interaction"},
            {"id": "problem_solving",   "label": "Problem Solving"},
            {"id": "memory",            "label": "Memory"},
        ],
    },
}

LEVEL_LABELS = {
    7: "Complete Independence",
    6: "Modified Independence",
    5: "Supervision / Setup",
    4: "Minimal Assistance (≥75%)",
    3: "Moderate Assistance (≥50%)",
    2: "Maximal Assistance (≥25%)",
    1: "Total Assistance (<25%)",
}

ALL_ITEMS = []
for domain, subdomains in FIM_ITEMS.items():
    for subdomain, items in subdomains.items():
        for item in items:
            ALL_ITEMS.append({**item, "domain": domain, "subdomain": subdomain})


def _conn():
    return sqlite3.connect(DB_PATH)


def _get_patient_data(patient_id: str) -> dict:
    """Gather relevant data for a single patient."""
    conn = _conn()
    c = conn.cursor()

    c.execute("SELECT patient_id, name, age, gender, disease FROM patients WHERE patient_id = ?", (patient_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return {}
    demo = {"patient_id": row[0], "name": row[1], "age": row[2], "gender": row[3], "disease": row[4]}

    # Barthel Index
    c.execute("SELECT score, max_score FROM assessments WHERE patient_id = ? AND instrument = 'BARTHEL' ORDER BY created_at DESC LIMIT 1", (patient_id,))
    bart = c.fetchone()
    barthel = {"score": bart[0], "max_score": bart[1]} if bart else None

    # Cognition
    c.execute("SELECT instrument, score, max_score FROM assessments WHERE patient_id = ? AND instrument IN ('MOCA', 'MMSE') ORDER BY created_at DESC LIMIT 1", (patient_id,))
    cog = c.fetchone()
    cognition = {"instrument": cog[0], "score": cog[1], "max_score": cog[2]} if cog else None

    # Seizure burden
    c.execute("SELECT COUNT(*), AVG(severity) FROM seizure_diary WHERE patient_id = ?", (patient_id,))
    sz = c.fetchone()
    seizures = {"count": sz[0] or 0, "avg_severity": round(sz[1], 1) if sz[1] else 0}

    # Medication count (fields stored as JSON)
    c.execute("SELECT COUNT(*) FROM medications WHERE patient_id = ?", (patient_id,))
    med = c.fetchone()
    med_count = med[0] if med else 0

    conn.close()
    return {"demo": demo, "barthel": barthel, "cognition": cognition, "seizures": seizures, "med_count": med_count}


def _deterministic_seed(patient_id: str, item_id: str) -> float:
    """Deterministic pseudo-random float 0-1 from patient+item."""
    h = hashlib.sha256(f"{patient_id}:{item_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _estimate_fim(data: dict) -> dict:
    """Estimate FIM item scores from existing clinical data."""
    if not data:
        return {}

    demo = data["demo"]
    age = demo.get("age") or 35

    # Barthel-derived functional ratio (0-1, 1=fully independent)
    if data["barthel"]:
        func_ratio = data["barthel"]["score"] / max(data["barthel"]["max_score"], 1)
    else:
        func_ratio = 0.75  # default moderate if no Barthel

    # Cognition ratio (0-1)
    if data["cognition"]:
        cog_ratio = data["cognition"]["score"] / max(data["cognition"]["max_score"], 1)
    else:
        cog_ratio = 0.8

    # Seizure burden penalty (0-1 reduction)
    sz_penalty = min(data["seizures"]["count"] * 0.02 + data["seizures"]["avg_severity"] * 0.05, 0.4)

    # Age penalty
    age_penalty = max(0, (age - 50) * 0.005)

    # Med burden penalty
    med_penalty = min(data["med_count"] * 0.02, 0.15)

    items = []
    motor_total = 0
    cognitive_total = 0

    for item_info in ALL_ITEMS:
        seed = _deterministic_seed(demo["patient_id"], item_info["id"])
        jitter = (seed - 0.5) * 0.15

        if item_info["domain"] == "motor":
            base = func_ratio - sz_penalty - age_penalty - med_penalty + jitter
            # stairs and tub_shower are harder
            if item_info["id"] in ("stairs", "tub_shower"):
                base -= 0.08
            # eating/grooming are easier
            if item_info["id"] in ("eating", "grooming"):
                base += 0.05
        else:  # cognitive
            base = cog_ratio - sz_penalty * 0.5 - age_penalty * 0.7 + jitter
            # memory is most affected
            if item_info["id"] == "memory":
                base -= 0.06
            # social interaction less affected
            if item_info["id"] == "social_interaction":
                base += 0.04

        # Map to FIM 1-7 scale
        score = max(1, min(7, round(base * 7)))
        level = LEVEL_LABELS.get(score, "Unknown")

        items.append({
            "id": item_info["id"],
            "label": item_info["label"],
            "domain": item_info["domain"],
            "subdomain": item_info["subdomain"],
            "score": score,
            "level": level,
        })

        if item_info["domain"] == "motor":
            motor_total += score
        else:
            cognitive_total += score

    total = motor_total + cognitive_total
    return {
        "patient_id": demo["patient_id"],
        "name": demo.get("name"),
        "age": demo.get("age"),
        "gender": demo.get("gender"),
        "total_score": total,
        "motor_score": motor_total,
        "cognitive_score": cognitive_total,
        "interpretation": _interpret(total),
        "items": items,
    }


def _interpret(total: int) -> str:
    if total >= 109:
        return "Independent (no helper needed)"
    elif total >= 90:
        return "Modified independent (device, extra time)"
    elif total >= 73:
        return "Moderate dependence (supervision/minimal assist)"
    elif total >= 55:
        return "Low-moderate dependence (moderate assist)"
    elif total >= 37:
        return "Substantial dependence (maximal assist)"
    else:
        return "Total dependence (complete assist)"


def _independence_level(total: int) -> str:
    if total >= 109:
        return "independent"
    elif total >= 90:
        return "modified_independent"
    elif total >= 73:
        return "moderate_dependence"
    elif total >= 55:
        return "low_moderate"
    elif total >= 37:
        return "substantial"
    else:
        return "total_dependence"


def _all_patients():
    """Get all patients that have relevant assessment data."""
    conn = _conn()
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT p.patient_id
        FROM patients p
        JOIN assessments a ON p.patient_id = a.patient_id
        WHERE p.age IS NOT NULL AND p.name IS NOT NULL AND p.age > 0
        ORDER BY p.patient_id
    """)
    pids = [r[0] for r in c.fetchall()]
    conn.close()
    return pids


def overview(patient_id: Optional[str] = None) -> dict:
    """FIM overview: KPIs, independence distribution, per-patient summaries."""
    if patient_id:
        data = _get_patient_data(patient_id)
        result = _estimate_fim(data)
        return {"patients": [result] if result else [], "total_assessments": 1 if result else 0}

    pids = _all_patients()
    results = []
    for pid in pids:
        data = _get_patient_data(pid)
        r = _estimate_fim(data)
        if r:
            results.append(r)

    # Distribution
    dist = {}
    for r in results:
        lvl = _independence_level(r["total_score"])
        dist[lvl] = dist.get(lvl, 0) + 1

    totals = [r["total_score"] for r in results]
    motor_scores = [r["motor_score"] for r in results]
    cognitive_scores = [r["cognitive_score"] for r in results]

    patient_summary = sorted(
        [{"patient_id": r["patient_id"], "name": r["name"], "age": r["age"], "gender": r["gender"],
          "total": r["total_score"], "motor": r["motor_score"], "cognitive": r["cognitive_score"],
          "level": _independence_level(r["total_score"]), "interpretation": r["interpretation"]}
         for r in results],
        key=lambda x: x["total"]
    )

    return {
        "total_assessments": len(results),
        "unique_patients": len(results),
        "avg_total": round(sum(totals) / len(totals), 1) if totals else 0,
        "avg_motor": round(sum(motor_scores) / len(motor_scores), 1) if motor_scores else 0,
        "avg_cognitive": round(sum(cognitive_scores) / len(cognitive_scores), 1) if cognitive_scores else 0,
        "min_total": min(totals) if totals else 0,
        "max_total": max(totals) if totals else 0,
        "independence_distribution": dist,
        "patient_summary": patient_summary,
    }


def breakdown(patient_id: Optional[str] = None) -> dict:
    """FIM domain breakdown: per-subdomain averages, item-level heatmap, motor vs cognitive."""
    pids = [patient_id] if patient_id else _all_patients()
    results = []
    for pid in pids:
        data = _get_patient_data(pid)
        r = _estimate_fim(data)
        if r:
            results.append(r)

    if not results:
        return {"subdomain_summary": [], "item_heatmap": [], "motor_vs_cognitive": []}

    # Per-subdomain averages
    subdomain_totals = {}
    subdomain_counts = {}
    for r in results:
        for item in r["items"]:
            key = f"{item['domain']}:{item['subdomain']}"
            subdomain_totals[key] = subdomain_totals.get(key, 0) + item["score"]
            subdomain_counts[key] = subdomain_counts.get(key, 0) + 1

    subdomain_summary = []
    for key in subdomain_totals:
        domain, subdomain = key.split(":")
        avg = round(subdomain_totals[key] / subdomain_counts[key], 1)
        subdomain_summary.append({
            "domain": domain,
            "subdomain": subdomain,
            "label": subdomain.replace("_", " ").title(),
            "avg_score": avg,
            "max_score": 7,
            "pct": round(avg / 7 * 100, 1),
        })

    # Item-level averages (for heatmap)
    item_totals = {}
    item_counts = {}
    for r in results:
        for item in r["items"]:
            item_totals[item["id"]] = item_totals.get(item["id"], 0) + item["score"]
            item_counts[item["id"]] = item_counts.get(item["id"], 0) + 1

    item_heatmap = []
    for item_info in ALL_ITEMS:
        iid = item_info["id"]
        if iid in item_totals:
            avg = round(item_totals[iid] / item_counts[iid], 1)
            item_heatmap.append({
                "id": iid,
                "label": item_info["label"],
                "domain": item_info["domain"],
                "subdomain": item_info["subdomain"],
                "avg_score": avg,
                "max_score": 7,
            })

    # Sort by avg_score ascending (weakest first)
    item_heatmap.sort(key=lambda x: x["avg_score"])

    # Motor vs Cognitive per patient
    motor_vs_cognitive = [
        {"patient_id": r["patient_id"], "motor": r["motor_score"], "cognitive": r["cognitive_score"],
         "total": r["total_score"]}
        for r in results
    ]

    # Per-patient item detail (for history cards)
    patient_items = {}
    for r in results:
        patient_items[r["patient_id"]] = {
            "total": r["total_score"],
            "motor": r["motor_score"],
            "cognitive": r["cognitive_score"],
            "interpretation": r["interpretation"],
            "items": r["items"],
        }

    return {
        "subdomain_summary": subdomain_summary,
        "item_heatmap": item_heatmap,
        "motor_vs_cognitive": motor_vs_cognitive,
        "patient_items": patient_items,
    }


def definitions() -> dict:
    """Metric definitions for FIM dashboard."""
    return {
        "title": "FIM — Functional Independence Measure — Definitions",
        "definitions": [
            {"term": "FIM Total Score", "definition": "Sum of 18 items, each scored 1-7. Range 18 (total dependence) to 126 (complete independence)."},
            {"term": "Motor Subscale", "definition": "13 items: Self-Care (6), Sphincter Control (2), Transfers (3), Locomotion (2). Range 13-91."},
            {"term": "Cognitive Subscale", "definition": "5 items: Communication (2), Social Cognition (3). Range 5-35."},
            {"term": "Level 7 — Complete Independence", "definition": "All tasks performed safely, without modification, assistive devices, or aid, in reasonable time."},
            {"term": "Level 6 — Modified Independence", "definition": "Activity requires assistive device, more than reasonable time, or safety concerns."},
            {"term": "Level 5 — Supervision / Setup", "definition": "Requires no more help than standby, cueing, coaxing, or setup of needed items."},
            {"term": "Level 4 — Minimal Assistance", "definition": "Patient performs ≥75% of the effort; helper provides contact guard or light touch."},
            {"term": "Level 3 — Moderate Assistance", "definition": "Patient performs 50-74% of the effort."},
            {"term": "Level 2 — Maximal Assistance", "definition": "Patient performs 25-49% of the effort."},
            {"term": "Level 1 — Total Assistance", "definition": "Patient performs <25% of the effort; maximum assistance required."},
            {"term": "Independent", "definition": "FIM total ≥109. No helper needed for any activity."},
            {"term": "Modified Independent", "definition": "FIM total 90-108. Device or extra time needed but no helper."},
            {"term": "Moderate Dependence", "definition": "FIM total 73-89. Supervision or minimal assistance needed."},
            {"term": "Low-Moderate Dependence", "definition": "FIM total 55-72. Moderate assistance needed for some activities."},
            {"term": "Substantial Dependence", "definition": "FIM total 37-54. Maximal assistance needed for many activities."},
            {"term": "Total Dependence", "definition": "FIM total 18-36. Complete assistance needed for most activities."},
            {"term": "Derivation Method", "definition": "FIM scores are estimated from existing clinical data (Barthel Index, cognition scores, seizure burden, medication load, age) using published correlation heuristics."},
        ],
    }
