#!/usr/bin/env python3
"""Epilepsy Program Coordinator module — patient journey tracking, MDT coordination,
operational KPI dashboard, and resource/capacity planning.

ALL data is REAL (no synthetic) — aggregates existing clinical tables:
  patients (40) · uploads (21) · analyses (21) · assessments (259) ·
  seizure_diary (25) · expert_reviews (3) · hitl_reviews (2)

The Coordinator role is operational: it does not add a new clinical instrument,
it surfaces where each patient sits in the care pipeline and where the program's
workload + bottlenecks are. Mirrors the established expert-module pattern.
"""

import os
import sqlite3
from collections import Counter, defaultdict

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clinical.db")

# Care-pipeline stages in order — a patient "progresses" as each completes.
JOURNEY_STAGES = ["registered", "data_uploaded", "eeg_analyzed", "clinically_assessed", "expert_reviewed"]


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _rows(cur):
    return [dict(r) for r in cur.fetchall()]


def _coverage_sets(c):
    """Return dict stage -> set(patient_ids) present at that stage (real tables)."""
    def pids(sql):
        return {r[0] for r in c.execute(sql) if r[0]}
    return {
        "registered": pids("SELECT patient_id FROM patients"),
        "data_uploaded": pids("SELECT patient_id FROM uploads"),
        "eeg_analyzed": pids("SELECT patient_id FROM analyses"),
        "clinically_assessed": pids("SELECT patient_id FROM assessments"),
        "expert_reviewed": pids("SELECT patient_id FROM expert_reviews"),
    }


# ─── 1. Patient Journey / Pathway Tracking ───────────────────────────────

def patient_journey(patient_id=None):
    """Per-patient care-pipeline position: which of the 5 stages are complete,
    the current stage, and the next action."""
    c = _conn()
    cov = _coverage_sets(c)
    pts = _rows(c.execute(
        "SELECT patient_id, name, age, gender, department FROM patients"
        + (" WHERE patient_id = ?" if patient_id else ""),
        (patient_id,) if patient_id else ()))
    c.close()

    journeys = []
    for p in pts:
        pid = p["patient_id"]
        done = [s for s in JOURNEY_STAGES if pid in cov[s]]
        # current = last contiguous completed stage; next = first incomplete
        nxt = next((s for s in JOURNEY_STAGES if pid not in cov[s]), None)
        pct = round(100 * len(done) / len(JOURNEY_STAGES))
        journeys.append({
            "patient_id": pid, "name": p["name"], "age": p["age"], "gender": p["gender"],
            "department": p["department"] or "unassigned",
            "stages_complete": done, "current_stage": done[-1] if done else "registered",
            "next_action": nxt or "complete", "progress_pct": pct,
            "stalled": bool(nxt and len(done) >= 2 and nxt != "expert_reviewed"),
        })
    journeys.sort(key=lambda j: j["progress_pct"])
    stage_funnel = {s: sum(1 for j in journeys if s in j["stages_complete"]) for s in JOURNEY_STAGES}
    return {
        "available": True, "n_patients": len(journeys), "journeys": journeys,
        "stage_funnel": stage_funnel,
        "avg_progress_pct": round(sum(j["progress_pct"] for j in journeys) / max(1, len(journeys)), 1),
        "note": ("Pipeline stages derived from real table membership "
                 "(patients→uploads→analyses→assessments→expert_reviews)."),
    }


# ─── 2. MDT (Multi-Disciplinary Team) Coordination ───────────────────────

def mdt_coordination(patient_id=None):
    """Review status across the MDT — which analyses have expert/HITL sign-off,
    which are pending, and per-role review load."""
    c = _conn()
    where = "WHERE patient_id = ?" if patient_id else ""
    params = (patient_id,) if patient_id else ()
    analyses = _rows(c.execute(
        f"SELECT id, patient_id, predicted_label, confidence, signal_quality, created_at "
        f"FROM analyses {where} ORDER BY created_at DESC", params))
    reviews = _rows(c.execute(f"SELECT analysis_id, patient_id, role, expert, agree_with_ai FROM expert_reviews {where}", params))
    hitl = _rows(c.execute(f"SELECT id, patient_id FROM hitl_reviews {where}", params))
    c.close()

    reviewed_ids = {r["analysis_id"] for r in reviews if r["analysis_id"] is not None}
    pending = [a for a in analyses if a["id"] not in reviewed_ids]
    # low-confidence analyses most in need of review
    pending.sort(key=lambda a: (a["confidence"] or 0))
    role_load = dict(Counter(r["role"] for r in reviews if r["role"]))
    agree = sum(1 for r in reviews if r["agree_with_ai"])
    return {
        "available": True,
        "analyses_total": len(analyses), "reviewed": len(reviewed_ids),
        "pending_review": len(pending),
        "pending_queue": [{"analysis_id": a["id"], "patient_id": a["patient_id"],
                           "predicted_label": a["predicted_label"], "confidence": a["confidence"],
                           "priority": "high" if (a["confidence"] or 0) < 0.6 else "routine"}
                          for a in pending[:20]],
        "hitl_reviews": len(hitl),
        "mdt_role_load": role_load,
        "ai_agreement_rate": round(agree / len(reviews), 3) if reviews else None,
        "note": "Review status from real expert_reviews + hitl_reviews; low-confidence analyses prioritized.",
    }


# ─── 3. KPI / Operational Dashboard ──────────────────────────────────────

def kpi_dashboard():
    """Program-level operational KPIs over the real cohort."""
    c = _conn()
    n_patients = c.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    cov = _coverage_sets(c)
    n_analyses = c.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    avg_conf = c.execute("SELECT AVG(confidence) FROM analyses").fetchone()[0]
    low_conf = c.execute("SELECT COUNT(*) FROM analyses WHERE confidence < 0.6").fetchone()[0]
    n_assess = c.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
    n_seizure = c.execute("SELECT COUNT(*) FROM seizure_diary").fetchone()[0]
    n_reviews = c.execute("SELECT COUNT(*) FROM expert_reviews").fetchone()[0]
    dept = dict((r[0] or "unassigned", r[1]) for r in c.execute("SELECT department, COUNT(*) FROM patients GROUP BY department"))
    c.close()

    def rate(n):
        return round(100 * n / n_patients, 1) if n_patients else 0.0
    return {
        "available": True,
        "kpis": {
            "patients_enrolled": n_patients,
            "eeg_analyses_run": n_analyses,
            "avg_model_confidence": round(avg_conf, 3) if avg_conf else None,
            "low_confidence_analyses": low_conf,
            "assessments_recorded": n_assess,
            "seizure_diary_entries": n_seizure,
            "expert_reviews_done": n_reviews,
        },
        "coverage_rates_pct": {
            "data_uploaded": rate(len(cov["data_uploaded"])),
            "eeg_analyzed": rate(len(cov["eeg_analyzed"])),
            "clinically_assessed": rate(len(cov["clinically_assessed"])),
            "expert_reviewed": rate(len(cov["expert_reviewed"])),
        },
        "patients_by_department": dept,
        "flags": (["expert review coverage low"] if rate(len(cov["expert_reviewed"])) < 25 else [])
                 + (["avg model confidence < 0.65 — data-quality review"] if (avg_conf or 0) < 0.65 else []),
        "note": "All KPIs computed from live clinical tables (no synthetic).",
    }


# ─── 4. Resource / Capacity Planning ─────────────────────────────────────

def resource_planning():
    """Workload + backlog by stage — where the program needs capacity."""
    c = _conn()
    cov = _coverage_sets(c)
    c.close()
    registered = cov["registered"]
    backlog = {
        "awaiting_upload": len(registered - cov["data_uploaded"]),
        "awaiting_analysis": len(cov["data_uploaded"] - cov["eeg_analyzed"]),
        "awaiting_assessment": len(cov["eeg_analyzed"] - cov["clinically_assessed"]),
        "awaiting_expert_review": len(cov["clinically_assessed"] - cov["expert_reviewed"]),
    }
    bottleneck = max(backlog, key=backlog.get) if backlog else None
    return {
        "available": True, "backlog_by_stage": backlog,
        "primary_bottleneck": bottleneck, "bottleneck_count": backlog.get(bottleneck, 0),
        "recommendation": {
            "awaiting_upload": "chase data acquisition / device logistics",
            "awaiting_analysis": "scale EEG analysis throughput (compute/queue)",
            "awaiting_assessment": "schedule clinical assessment sessions",
            "awaiting_expert_review": "add MDT reviewer capacity / triage low-confidence first",
        }.get(bottleneck, "balanced — no single bottleneck"),
        "note": "Backlog = patients present at stage N but not yet at stage N+1 (real table deltas).",
    }


def full_dashboard(patient_id=None):
    """Combined Epilepsy Program Coordinator dashboard — all 4 modules."""
    return {
        "role": "Epilepsy Program Coordinator",
        "description": "Patient journey tracking, MDT coordination, operational KPIs, "
                       "and resource/capacity planning across the epilepsy program",
        "modules": {
            "patient_journey": patient_journey(patient_id),
            "mdt_coordination": mdt_coordination(patient_id),
            "kpi_dashboard": kpi_dashboard(),
            "resource_planning": resource_planning(),
        },
    }


if __name__ == "__main__":
    r = full_dashboard()
    print("Coordinator dashboard:")
    print("  journey funnel:", r["modules"]["patient_journey"]["stage_funnel"])
    print("  pending MDT review:", r["modules"]["mdt_coordination"]["pending_review"])
    print("  KPIs:", r["modules"]["kpi_dashboard"]["kpis"])
    print("  bottleneck:", r["modules"]["resource_planning"]["primary_bottleneck"],
          r["modules"]["resource_planning"]["bottleneck_count"])
