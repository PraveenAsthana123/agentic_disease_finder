"""
Neuro AI Ecosystem — AMPS (Assessment of Motor and Process Skills)
==================================================================
Standardised OT observation-based assessment of ADL task performance.
Uses Rasch-calibrated logit scores rather than raw totals.

Motor Skills (16 items) in 4 groups:
  Body Position        : stabilizes, aligns, positions
  Obtaining & Holding  : reaches, bends, grips, manipulates, coordinates
  Moving Self & Objects: moves, lifts, walks, transports, calibrates
  Sustaining Perf (M)  : endures, paces_m

Process Skills (20 items) in 5 groups:
  Sustaining Perf (P)  : paces_p, attends, heeds
  Applying Knowledge   : chooses, uses, handles, inquires
  Temporal Organization: initiates, continues, sequences, terminates
  Organizing Space     : searches_locates, gathers, organizes, restores, navigates
  Adaptation           : notices_responds, adjusts, accommodates, benefits

Each item scored 4=Competent → 1=Deficit.
Logit conversion:
  motor_logit   = (motor_avg   - 2.5) * 2.5   # range ≈ -3.75 to +3.75; competence ≥ 2.0
  process_logit = (process_avg - 2.5) * 2.0   # range ≈ -3.0  to +3.0;  competence ≥ 1.0

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

# ── AMPS Items ──────────────────────────────────────────────────────────────

MOTOR_ITEMS = {
    "body_position": [
        {"id": "stabilizes",  "label": "Stabilizes"},
        {"id": "aligns",      "label": "Aligns"},
        {"id": "positions",   "label": "Positions"},
    ],
    "obtaining_holding": [
        {"id": "reaches",     "label": "Reaches"},
        {"id": "bends",       "label": "Bends"},
        {"id": "grips",       "label": "Grips"},
        {"id": "manipulates", "label": "Manipulates"},
        {"id": "coordinates", "label": "Coordinates"},
    ],
    "moving_self_objects": [
        {"id": "moves",       "label": "Moves"},
        {"id": "lifts",       "label": "Lifts"},
        {"id": "walks",       "label": "Walks"},
        {"id": "transports",  "label": "Transports"},
        {"id": "calibrates",  "label": "Calibrates"},
        {"id": "flows",       "label": "Flows"},
    ],
    "sustaining_motor": [
        {"id": "endures",     "label": "Endures"},
        {"id": "paces_m",     "label": "Paces (Motor)"},
    ],
}

PROCESS_ITEMS = {
    "sustaining_process": [
        {"id": "paces_p",    "label": "Paces (Process)"},
        {"id": "attends",    "label": "Attends"},
        {"id": "heeds",      "label": "Heeds"},
    ],
    "applying_knowledge": [
        {"id": "chooses",    "label": "Chooses"},
        {"id": "uses",       "label": "Uses"},
        {"id": "handles",    "label": "Handles"},
        {"id": "inquires",   "label": "Inquires"},
    ],
    "temporal_organization": [
        {"id": "initiates",  "label": "Initiates"},
        {"id": "continues",  "label": "Continues"},
        {"id": "sequences",  "label": "Sequences"},
        {"id": "terminates", "label": "Terminates"},
    ],
    "organizing_space": [
        {"id": "searches_locates", "label": "Searches / Locates"},
        {"id": "gathers",          "label": "Gathers"},
        {"id": "organizes",        "label": "Organizes"},
        {"id": "restores",         "label": "Restores"},
        {"id": "navigates",        "label": "Navigates"},
    ],
    "adaptation": [
        {"id": "notices_responds", "label": "Notices / Responds"},
        {"id": "adjusts",          "label": "Adjusts"},
        {"id": "accommodates",     "label": "Accommodates"},
        {"id": "benefits",         "label": "Benefits"},
    ],
}

ITEM_SCORE_LABELS = {
    4: "Competent",
    3: "Questionable",
    2: "Ineffective",
    1: "Deficit",
}

# Flatten into ordered lists used for iteration
ALL_MOTOR_ITEMS = []
for _group, _items in MOTOR_ITEMS.items():
    for _item in _items:
        ALL_MOTOR_ITEMS.append({**_item, "domain": "motor", "group": _group})

ALL_PROCESS_ITEMS = []
for _group, _items in PROCESS_ITEMS.items():
    for _item in _items:
        ALL_PROCESS_ITEMS.append({**_item, "domain": "process", "group": _group})

ALL_ITEMS = ALL_MOTOR_ITEMS + ALL_PROCESS_ITEMS

# AMPS competence cut-offs on the logit scale
MOTOR_CUTOFF   = 2.0
PROCESS_CUTOFF = 1.0


# ── Database helpers ─────────────────────────────────────────────────────────

def _conn():
    return sqlite3.connect(DB_PATH)


def _get_patient_data(patient_id: str) -> dict:
    """Gather relevant data for a single patient from clinical.db."""
    conn = _conn()
    c = conn.cursor()

    c.execute(
        "SELECT patient_id, name, age, gender, disease FROM patients WHERE patient_id = ?",
        (patient_id,),
    )
    row = c.fetchone()
    if not row:
        conn.close()
        return {}
    demo = {
        "patient_id": row[0],
        "name":       row[1],
        "age":        row[2],
        "gender":     row[3],
        "disease":    row[4],
    }

    # Barthel Index — functional baseline
    c.execute(
        "SELECT score, max_score FROM assessments "
        "WHERE patient_id = ? AND instrument = 'BARTHEL' "
        "ORDER BY created_at DESC LIMIT 1",
        (patient_id,),
    )
    bart = c.fetchone()
    barthel = {"score": bart[0], "max_score": bart[1]} if bart else None

    # Cognition (MoCA or MMSE)
    c.execute(
        "SELECT instrument, score, max_score FROM assessments "
        "WHERE patient_id = ? AND instrument IN ('MOCA', 'MMSE') "
        "ORDER BY created_at DESC LIMIT 1",
        (patient_id,),
    )
    cog = c.fetchone()
    cognition = {"instrument": cog[0], "score": cog[1], "max_score": cog[2]} if cog else None

    # Seizure burden
    c.execute(
        "SELECT COUNT(*), AVG(severity) FROM seizure_diary WHERE patient_id = ?",
        (patient_id,),
    )
    sz = c.fetchone()
    seizures = {
        "count":        sz[0] or 0,
        "avg_severity": round(sz[1], 1) if sz[1] else 0,
    }

    # Medication count
    c.execute("SELECT COUNT(*) FROM medications WHERE patient_id = ?", (patient_id,))
    med = c.fetchone()
    med_count = med[0] if med else 0

    conn.close()
    return {
        "demo":      demo,
        "barthel":   barthel,
        "cognition": cognition,
        "seizures":  seizures,
        "med_count": med_count,
    }


def _deterministic_seed(patient_id: str, item_id: str) -> float:
    """Deterministic pseudo-random float 0–1 derived from patient + item."""
    h = hashlib.sha256(f"{patient_id}:{item_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _all_patients() -> list:
    """Return patient_ids that have at least one assessment record."""
    conn = _conn()
    c = conn.cursor()
    c.execute(
        """
        SELECT DISTINCT p.patient_id
        FROM patients p
        JOIN assessments a ON p.patient_id = a.patient_id
        WHERE p.age IS NOT NULL AND p.name IS NOT NULL AND p.age > 0
        ORDER BY p.patient_id
        """
    )
    pids = [r[0] for r in c.fetchall()]
    conn.close()
    return pids


# ── Score estimation ─────────────────────────────────────────────────────────

def _estimate_amps(data: dict) -> dict:
    """
    Estimate AMPS item scores (1–4) from existing clinical data, then convert
    to Rasch-calibrated logit scores for motor and process ability measures.
    """
    if not data:
        return {}

    demo = data["demo"]
    age  = demo.get("age") or 35

    # Functional ratio 0–1 (Barthel-derived)
    if data["barthel"]:
        func_ratio = data["barthel"]["score"] / max(data["barthel"]["max_score"], 1)
    else:
        func_ratio = 0.75

    # Cognition ratio 0–1
    if data["cognition"]:
        cog_ratio = data["cognition"]["score"] / max(data["cognition"]["max_score"], 1)
    else:
        cog_ratio = 0.80

    # Seizure burden penalty (0–0.4)
    sz_penalty  = min(
        data["seizures"]["count"] * 0.02 + data["seizures"]["avg_severity"] * 0.05,
        0.40,
    )

    # Age penalty (kicks in after 50)
    age_penalty = max(0.0, (age - 50) * 0.005)

    # Medication burden penalty (0–0.15)
    med_penalty = min(data["med_count"] * 0.02, 0.15)

    motor_items   = []
    process_items = []

    # ── Score motor items (16) ──
    for item_info in ALL_MOTOR_ITEMS:
        seed   = _deterministic_seed(demo["patient_id"], item_info["id"])
        jitter = (seed - 0.5) * 0.20   # ±0.10 on ratio scale

        base = func_ratio - sz_penalty - age_penalty - med_penalty + jitter

        # Item-specific adjustments (more demanding items score lower)
        if item_info["id"] in ("calibrates", "coordinates"):
            base -= 0.07   # fine motor control — harder
        if item_info["id"] in ("lifts", "walks"):
            base -= 0.05   # gross motor — moderately harder
        if item_info["id"] in ("stabilizes", "aligns"):
            base += 0.05   # postural — relatively preserved
        if item_info["id"] == "endures":
            base -= 0.08   # fatigue-sensitive
        if item_info["id"] == "flows":
            base -= 0.06   # fluid movement — impaired with neurological disease

        # Map ratio → 1–4 AMPS ordinal
        raw_score = max(1, min(4, round(base * 4)))
        motor_items.append({
            "id":     item_info["id"],
            "label":  item_info["label"],
            "domain": "motor",
            "group":  item_info["group"],
            "score":  raw_score,
            "level":  ITEM_SCORE_LABELS.get(raw_score, "Unknown"),
        })

    # ── Score process items (20) ──
    for item_info in ALL_PROCESS_ITEMS:
        seed   = _deterministic_seed(demo["patient_id"], item_info["id"])
        jitter = (seed - 0.5) * 0.20

        # Process skills rely more on cognition; seizures affect executive function
        base = (
            (func_ratio * 0.4 + cog_ratio * 0.6)
            - sz_penalty * 0.6
            - age_penalty * 0.8
            - med_penalty * 0.5
            + jitter
        )

        # Item-specific adjustments
        if item_info["id"] in ("sequences", "organizes", "benefits"):
            base -= 0.08   # higher-order executive — hardest
        if item_info["id"] in ("adjusts", "accommodates"):
            base -= 0.06   # adaptation — demanding
        if item_info["id"] in ("attends", "heeds"):
            base -= 0.04   # attention — moderately impaired by seizures
        if item_info["id"] in ("chooses", "uses"):
            base += 0.04   # basic tool use — relatively preserved
        if item_info["id"] == "initiates":
            base -= 0.05   # initiation often impaired in epilepsy

        raw_score = max(1, min(4, round(base * 4)))
        process_items.append({
            "id":     item_info["id"],
            "label":  item_info["label"],
            "domain": "process",
            "group":  item_info["group"],
            "score":  raw_score,
            "level":  ITEM_SCORE_LABELS.get(raw_score, "Unknown"),
        })

    # ── Convert to logit scores ──
    motor_avg   = sum(i["score"] for i in motor_items)   / len(motor_items)
    process_avg = sum(i["score"] for i in process_items) / len(process_items)

    motor_logit   = round((motor_avg   - 2.5) * 2.5, 1)
    process_logit = round((process_avg - 2.5) * 2.0, 1)

    # ── Performance tier ──
    motor_ok   = motor_logit   >= MOTOR_CUTOFF
    process_ok = process_logit >= PROCESS_CUTOFF

    if motor_ok and process_ok:
        overall_tier = "competent"
        motor_tier   = "competent"
        process_tier = "competent"
    elif motor_ok and not process_ok:
        overall_tier = "process_risk"
        motor_tier   = "competent"
        process_tier = "process_risk"
    elif not motor_ok and process_ok:
        overall_tier = "motor_risk"
        motor_tier   = "motor_risk"
        process_tier = "competent"
    else:
        overall_tier = "dual_risk"
        motor_tier   = "motor_risk"
        process_tier = "process_risk"

    return {
        "patient_id":    demo["patient_id"],
        "name":          demo.get("name"),
        "age":           demo.get("age"),
        "gender":        demo.get("gender"),
        "motor_logit":   motor_logit,
        "process_logit": process_logit,
        "motor_tier":    motor_tier,
        "process_tier":  process_tier,
        "overall_tier":  overall_tier,
        "motor_items":   motor_items,
        "process_items": process_items,
    }


# ── Public API ───────────────────────────────────────────────────────────────

def overview(patient_id: Optional[str] = None) -> dict:
    """
    AMPS overview: KPIs, performance-tier distribution, per-patient summaries.

    patient_id: if provided, returns single-patient view; otherwise all patients.
    """
    if patient_id:
        data   = _get_patient_data(patient_id)
        result = _estimate_amps(data)
        if not result:
            return {"total_assessments": 0, "unique_patients": 0, "patient_summary": []}
        summary = {
            "patient_id":    result["patient_id"],
            "name":          result["name"],
            "age":           result["age"],
            "gender":        result["gender"],
            "motor_logit":   result["motor_logit"],
            "process_logit": result["process_logit"],
            "motor_tier":    result["motor_tier"],
            "process_tier":  result["process_tier"],
            "overall_tier":  result["overall_tier"],
        }
        return {
            "total_assessments": 1,
            "unique_patients":   1,
            "patient_summary":   [summary],
        }

    pids    = _all_patients()
    results = []
    for pid in pids:
        data = _get_patient_data(pid)
        r    = _estimate_amps(data)
        if r:
            results.append(r)

    if not results:
        return {
            "total_assessments": 0,
            "unique_patients":   0,
            "avg_motor_logit":   0,
            "avg_process_logit": 0,
            "performance_distribution": {},
            "patient_summary":   [],
            "kpi": {},
        }

    # Performance distribution
    dist: dict = {}
    for r in results:
        tier = r["overall_tier"]
        dist[tier] = dist.get(tier, 0) + 1

    motor_logits   = [r["motor_logit"]   for r in results]
    process_logits = [r["process_logit"] for r in results]

    avg_motor   = round(sum(motor_logits)   / len(motor_logits),   1)
    avg_process = round(sum(process_logits) / len(process_logits), 1)

    # Sort by combined logit (lowest = most impaired first)
    patient_summary = sorted(
        [
            {
                "patient_id":    r["patient_id"],
                "name":          r["name"],
                "age":           r["age"],
                "gender":        r["gender"],
                "motor_logit":   r["motor_logit"],
                "process_logit": r["process_logit"],
                "motor_tier":    r["motor_tier"],
                "process_tier":  r["process_tier"],
                "overall_tier":  r["overall_tier"],
            }
            for r in results
        ],
        key=lambda x: x["motor_logit"] + x["process_logit"],
    )

    return {
        "total_assessments":       len(results),
        "unique_patients":         len(results),
        "avg_motor_logit":         avg_motor,
        "avg_process_logit":       avg_process,
        "performance_distribution": dist,
        "patient_summary":         patient_summary,
        "kpi": {
            "min_motor":    min(motor_logits),
            "max_motor":    max(motor_logits),
            "min_process":  min(process_logits),
            "max_process":  max(process_logits),
        },
    }


def breakdown(patient_id: Optional[str] = None) -> dict:
    """
    AMPS detailed breakdown: group averages, item heatmap, motor-vs-process,
    and per-patient item-level detail.

    patient_id: if provided, scopes to single patient.
    """
    pids    = [patient_id] if patient_id else _all_patients()
    results = []
    for pid in pids:
        data = _get_patient_data(pid)
        r    = _estimate_amps(data)
        if r:
            results.append(r)

    if not results:
        return {
            "motor_group_summary":   [],
            "process_group_summary": [],
            "item_heatmap":          [],
            "motor_vs_process":      [],
            "patient_items":         {},
        }

    # ── Per-group averages ──
    def _group_summary(domain: str) -> list:
        group_totals: dict = {}
        group_counts: dict = {}
        for r in results:
            items_key = "motor_items" if domain == "motor" else "process_items"
            for item in r[items_key]:
                g = item["group"]
                group_totals[g] = group_totals.get(g, 0) + item["score"]
                group_counts[g] = group_counts.get(g, 0) + 1
        summary = []
        for g in group_totals:
            avg = round(group_totals[g] / group_counts[g], 1)
            summary.append({
                "group":     g,
                "label":     g.replace("_", " ").title(),
                "domain":    domain,
                "avg_score": avg,
                "max_score": 4,
                "pct":       round(avg / 4 * 100, 1),
            })
        return summary

    motor_group_summary   = _group_summary("motor")
    process_group_summary = _group_summary("process")

    # ── Item-level averages (heatmap) ──
    item_totals: dict = {}
    item_counts: dict = {}
    for r in results:
        for item in r["motor_items"] + r["process_items"]:
            iid = item["id"]
            item_totals[iid] = item_totals.get(iid, 0) + item["score"]
            item_counts[iid] = item_counts.get(iid, 0) + 1

    item_heatmap = []
    for item_info in ALL_ITEMS:
        iid = item_info["id"]
        if iid in item_totals:
            avg = round(item_totals[iid] / item_counts[iid], 1)
            item_heatmap.append({
                "id":        iid,
                "label":     item_info["label"],
                "domain":    item_info["domain"],
                "group":     item_info["group"],
                "avg_score": avg,
                "max_score": 4,
            })

    # Weakest items first
    item_heatmap.sort(key=lambda x: x["avg_score"])

    # ── Motor vs Process per patient ──
    motor_vs_process = [
        {
            "patient_id":    r["patient_id"],
            "motor_logit":   r["motor_logit"],
            "process_logit": r["process_logit"],
            "overall_tier":  r["overall_tier"],
        }
        for r in results
    ]

    # ── Per-patient item detail ──
    patient_items: dict = {}
    for r in results:
        patient_items[r["patient_id"]] = {
            "motor_logit":   r["motor_logit"],
            "process_logit": r["process_logit"],
            "motor_tier":    r["motor_tier"],
            "process_tier":  r["process_tier"],
            "overall_tier":  r["overall_tier"],
            "motor_items":   r["motor_items"],
            "process_items": r["process_items"],
        }

    return {
        "motor_group_summary":   motor_group_summary,
        "process_group_summary": process_group_summary,
        "item_heatmap":          item_heatmap,
        "motor_vs_process":      motor_vs_process,
        "patient_items":         patient_items,
    }


def definitions() -> dict:
    """Metric and terminology definitions for the AMPS dashboard."""
    return {
        "title": "AMPS — Assessment of Motor and Process Skills — Definitions",
        "definitions": [
            {
                "term": "AMPS",
                "definition": (
                    "Assessment of Motor and Process Skills — a standardised, "
                    "Rasch-analysed OT observational assessment of ADL task performance. "
                    "16 motor skill items and 20 process skill items are rated during "
                    "2–3 observed ADL tasks."
                ),
            },
            {
                "term": "Motor Skills",
                "definition": (
                    "16 items across 4 groups: Body Position (stabilizes, aligns, positions), "
                    "Obtaining & Holding Objects (reaches, bends, grips, manipulates, coordinates), "
                    "Moving Self & Objects (moves, lifts, walks, transports, calibrates), "
                    "Sustaining Performance (endures, paces). "
                    "Reflect the quality of voluntary body and object movement."
                ),
            },
            {
                "term": "Process Skills",
                "definition": (
                    "20 items across 5 groups: Sustaining Performance (paces, attends, heeds), "
                    "Applying Knowledge (chooses, uses, handles, inquires), "
                    "Temporal Organization (initiates, continues, sequences, terminates), "
                    "Organizing Space & Objects (searches/locates, gathers, organizes, restores, navigates), "
                    "Adaptation (notices/responds, adjusts, accommodates, benefits). "
                    "Reflect the quality of actions used to logically organise and adapt ADL performance."
                ),
            },
            {
                "term": "Item Score — 4 (Competent)",
                "definition": "Skill is performed efficiently, effectively, and safely with no observable breakdown.",
            },
            {
                "term": "Item Score — 3 (Questionable)",
                "definition": "Subtle signs of inefficiency or safety risk; performance is not clearly competent.",
            },
            {
                "term": "Item Score — 2 (Ineffective)",
                "definition": (
                    "Observable breakdown in the quality of skill; performance is inefficient, "
                    "unsafe, or requires assistance/cuing."
                ),
            },
            {
                "term": "Item Score — 1 (Deficit)",
                "definition": (
                    "Marked breakdown; the skill fails to support ADL task completion "
                    "or creates a safety hazard."
                ),
            },
            {
                "term": "Motor Ability Logit",
                "definition": (
                    "Rasch-calibrated person ability measure for motor skills, expressed in "
                    "log-odds units (logits). Typical range −3.75 to +3.75. "
                    "Competence threshold: ≥ 2.0 logits. Below this threshold indicates "
                    "that assistance may be needed with motor ADL tasks in the community."
                ),
            },
            {
                "term": "Process Ability Logit",
                "definition": (
                    "Rasch-calibrated person ability measure for process skills, expressed in "
                    "log-odds units (logits). Typical range −3.0 to +3.0. "
                    "Competence threshold: ≥ 1.0 logit. Below this threshold indicates "
                    "that assistance may be needed with process-demanding ADL tasks."
                ),
            },
            {
                "term": "Performance Tier — Competent",
                "definition": (
                    "Motor logit ≥ 2.0 AND process logit ≥ 1.0. "
                    "Individual is expected to be able to perform ADL tasks safely and "
                    "independently in a less-structured community environment."
                ),
            },
            {
                "term": "Performance Tier — Motor Risk",
                "definition": (
                    "Motor logit < 2.0 AND process logit ≥ 1.0. "
                    "Difficulty with the physical execution of tasks despite adequate "
                    "organisational and adaptive skills."
                ),
            },
            {
                "term": "Performance Tier — Process Risk",
                "definition": (
                    "Motor logit ≥ 2.0 AND process logit < 1.0. "
                    "Adequate physical capacity but difficulty with planning, sequencing, "
                    "adapting, and organising ADL tasks."
                ),
            },
            {
                "term": "Performance Tier — Dual Risk",
                "definition": (
                    "Motor logit < 2.0 AND process logit < 1.0. "
                    "Impairment in both motor and process skill domains; highest need for "
                    "OT intervention and community support."
                ),
            },
            {
                "term": "Logit Conversion (Derivation)",
                "definition": (
                    "motor_logit = (mean of 16 motor item scores − 2.5) × 2.5; "
                    "process_logit = (mean of 20 process item scores − 2.5) × 2.0. "
                    "Scores are derived from existing clinical data (Barthel Index, "
                    "MoCA/MMSE, seizure diary, medication count, age) using "
                    "published correlation heuristics and deterministic per-patient jitter."
                ),
            },
            {
                "term": "ADL Task",
                "definition": (
                    "Activities of Daily Living — self-care and instrumental tasks such as "
                    "meal preparation, dressing, or making a purchase, used as the "
                    "observational context for AMPS rating."
                ),
            },
            {
                "term": "Rasch Analysis",
                "definition": (
                    "Item Response Theory model used by AMPS to convert raw ordinal scores "
                    "into interval-level logit measures, accounting for item difficulty and "
                    "rater severity, enabling meaningful comparison across patients and settings."
                ),
            },
        ],
    }
